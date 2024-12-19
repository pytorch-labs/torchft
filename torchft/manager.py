# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Manager
=========

This module implements the Manager that manages the full fault tolerant training
loop.

The Manager is responsible for managing the
full training loop, communicating with the Lighthouse server to figure out
quorum, reconfiguring the ProcessGroups and restoring checkpoint state when
recovering.

This uses wrapper classes to wrap the standard PyTorch Optimizer and Module
classes to provide fault tolerance. These wrappers indented to add fault
tolerance with minimal changes to the users modeling code and training loop.

This is designed to work with the standard PyTorch DistributedDataParallel module
and Hybrid FSDP.

"""

import concurrent.futures
import logging
import os
import socket
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from enum import Enum
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, TypeVar, cast

import torch
from torch.distributed import ReduceOp, TCPStore

from torchft.checkpointing import CheckpointServer
from torchft.futures import future_timeout
from torchft.torchft import Manager as _Manager, ManagerClient

if TYPE_CHECKING:
    from torchft.process_group import ProcessGroup

MANAGER_ADDR_KEY: str = "manager_addr"
MANAGER_DEFAULT_PORT: int = int(os.environ.get("TORCHFT_MANAGER_PORT", 29511))
REPLICA_ID_KEY: str = "replica_id"

T = TypeVar("T")


class WorldSizeMode(Enum):
    """
    This controls the numerics for the job when doing allreduces across replicas
    when the world size is larger than ``min_replica_size``. The world size will
    never be smaller than ``min_replica_size``.

    DYNAMIC:
        The world size will dynamical increase to use all available
        replicas and normalize the gradient by the world size.
    FIXED_WITH_SPARES:
        The number of active replicas is ``min_replica_size`` and any spares
        will contribute zero gradients.
    """

    DYNAMIC = 0
    FIXED_WITH_SPARES = 1


class Manager:
    """
    Manager manages the full fault tolerant training loop.

    NOTE: when saving periodic checkpoints you must save and restore the
    Manager's state_dict as well to avoid synchronization issues.
    """

    def __init__(
        self,
        pg: "ProcessGroup",
        load_state_dict: Callable[[T], None],
        state_dict: Callable[[], T],
        min_replica_size: int,
        port: int = MANAGER_DEFAULT_PORT,
        use_async_quorum: bool = True,
        timeout: timedelta = timedelta(seconds=60),
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        world_size_mode: WorldSizeMode = WorldSizeMode.DYNAMIC,
        store_addr: Optional[str] = None,
        store_port: Optional[int] = None,
        lighthouse_addr: Optional[str] = None,
        replica_id: Optional[str] = None,
    ) -> None:
        """
        Args:
            load_state_dict: function to load the state dict when recovering
            state_dict: function to save the state dict with recovering
            min_replica_size: minimum number of replicas on each step
            port: if rank==0, the port to run the manager server on
            use_async_quorum: whether to run the quorum asynchronously during the forward pass
            timeout: timeout for all operations
            rank: the replica group local rank
            world_size: the replica group local world size
            store_addr: TCPStore address for this replica group
            store_port: TCPStore port for this replica group
            lighthouse_addr: if rank==0, the address of the lighthouse server
            replica_id: if rank==0, the replica_id for this group
        """
        self._load_state_dict = load_state_dict
        self._state_dict = state_dict
        self._pending_state_dict: Optional[Dict[str, object]] = None
        self._use_async_quorum = use_async_quorum
        self._timeout = timeout
        self._world_size_mode = world_size_mode

        store_addr = store_addr or os.environ["MASTER_ADDR"]
        store_port = store_port or int(os.environ["MASTER_PORT"])
        self._rank: int = rank if rank is not None else int(os.environ["RANK"])
        rank = self._rank
        world_size = world_size or int(os.environ["WORLD_SIZE"])
        self._min_replica_size = min_replica_size

        def _manager_state_dict() -> Dict[str, T]:
            return {
                "user": state_dict(),
                "torchft": cast(T, self.state_dict()),
            }

        self._ckpt_server = CheckpointServer[Dict[str, T]](_manager_state_dict)
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="async_quorum"
        )
        self._quorum_future: Optional[concurrent.futures.Future] = None

        self._store = TCPStore(
            host_name=store_addr,
            port=store_port,
            is_master=False,
            wait_for_workers=False,
        )
        self._pg = pg
        self._manager: Optional[_Manager] = None

        if rank == 0:
            hostname = socket.gethostname()
            addr = f"http://{hostname}:{port}"
            bind = f"[::]:{port}"
            lighthouse_addr = lighthouse_addr or os.environ["TORCHFT_LIGHTHOUSE"]

            if replica_id is None:
                replica_id = ""
            replica_id = replica_id + str(uuid.uuid4())
            self._manager = _Manager(
                replica_id=replica_id,
                lighthouse_addr=lighthouse_addr,
                address=addr,
                bind=bind,
                store_addr=f"{store_addr}:{store_port}",
                world_size=world_size,
            )

            self._store.set(MANAGER_ADDR_KEY, addr)
            self._store.set(REPLICA_ID_KEY, replica_id)

        addr = self._store.get(MANAGER_ADDR_KEY).decode("utf-8")
        self._client = ManagerClient(addr, timeout=timeout)

        replica_id = self._store.get(REPLICA_ID_KEY).decode("utf-8")
        self._logger = _ManagerLogger(
            manager=self, replica_id=replica_id or "", rank=rank
        )

        self._step = 0
        self._quorum_id = -1
        self._errored: Optional[Exception] = None
        self._healing = False
        self._pending_work: List[torch.futures.Future[object]] = []
        self._batches_committed = 0

        # first step is 1
        self._participating_rank: Optional[int] = None
        self._participating_world_size: int = 0

    def shutdown(self) -> None:
        """
        Shutdown the manager and checkpoint server.
        """
        self._ckpt_server.shutdown()
        if self._manager is not None:
            self._manager.shutdown()
        self._executor.shutdown()

    def allreduce(self, tensor: torch.Tensor) -> torch.futures.Future[torch.Tensor]:
        """
        Fault tolerant allreduce the tensor and return a Future that will be completed when
        the tensor is ready.

        This will automatically scale the tensor by 1 / world_size.

        If an error occurs during the allreduce:

        * The Future will be completed with no error and instead tracked asynchronously.
        * After the first error, all subsequent calls will be noops and immediately return.
        * The tensor must be zeroed before being used as it may be corrupted.

        Args:
            tensor: the tensor to allreduce
        Returns:
            a Future that will be completed with the allreduced tensor
        """
        if self.errored():
            fut = torch.futures.Future()  # pyre-fixme[29]: not a function
            fut.set_result(tensor)
            return fut

        self.wait_quorum()

        if not self.is_participating():
            tensor.zero_()

        # TODO: increase timeout when waiting when healing
        try:
            # Run the allreduce async and save the work object so we can wait on
            # it later.
            work = self._pg.allreduce([tensor], ReduceOp.SUM)
            fut = work.get_future()

            # schedule grad normalization as a continuation
            # on the Future
            def callback(
                fut: torch.futures.Future[List[torch.Tensor]],
            ) -> torch.Tensor:
                nonlocal tensor

                # check for exceptions
                fut.value()

                tensor /= self.num_participants()

                return tensor

            fut = fut.then(callback)
            fut = self.wrap_future(fut, tensor)
            return fut

        except Exception as e:
            self._logger.exception(
                f"got exception in all reduce -- skipping remaining: {e}"
            )
            self.report_error(e)

            fut = torch.futures.Future()  # pyre-fixme[29]: not a function
            fut.set_result(tensor)
            return fut

    def report_error(self, e: Exception) -> None:
        """
        Report an error to the manager.

        This will cause the manager to skip the current step and will be
        reconfigured on the next step.

        This should be called when an error occurs that leads to a corrupted
        gradient that needs to be discarded.
        """
        self._errored = e

    def errored(self) -> Optional[Exception]:
        """
        Get whether an error has occurred.

        Returns:
            The error or None if no error has occured.
        """
        return self._errored

    def wrap_future(
        self, fut: torch.futures.Future[T], default: T
    ) -> torch.futures.Future[T]:
        """
        Wrap a Future and swallow any errors that occur and report them to the manager.

        If an error occurs, the Future will be completed with the default value.

        Args:
            fut: the Future to wrap
            default: the default value to complete the Future with if an error occurs
        """

        # add a timeout to the future
        fut = future_timeout(fut, self._timeout)

        # schedule error handling as a continuation on the Future
        def callback(
            fut: torch.futures.Future[T],
        ) -> T:
            nonlocal default

            try:
                return fut.value()
            except Exception as e:
                self._logger.exception(
                    f"got exception in future -- skipping remaining: {e}"
                )
                self.report_error(e)
                return default

        fut = fut.then(callback)
        self._pending_work.append(cast(torch.futures.Future[object], fut))
        return fut

    def start_quorum(self, room_id: str = "default", allow_heal: bool = True) -> None:
        """
        .. note::
            We recommend using the :py:class:`torchft.optim.OptimizerWrapper` instead of calling this directly.

        Computes a new quorum (potentially asynchronously) and readies the
        manager for a new step.

        It's best practice to call this before the forwards pass of each step for
        performance as computing quorum may take some time.

        Args:
            allow_heal: (experimental) whether to allow healing at the beginning of the step
                If allow_heal is set, the manager will attempt to heal either
                synchronously before returning or asynchronously prior to any network
                calls. All replicas must pass the same value to allow_heal.
            room_id: (experimental) the room id to use for quorum, this allows
                for multiple quorums to be used within the same job.
        """

        # wait for previous quorum to complete
        if self._quorum_future is not None:
            self._quorum_future.result()

        self._errored = None
        self._healing = False
        self._ckpt_server.allow_checkpoint(self._step)

        # TODO: we should really be wrapping this whole section in a try-except
        # block to allow gracefully recovering from issues in PG setup and quorum.

        self._quorum_future = self._executor.submit(
            self._async_quorum, room_id=room_id, allow_heal=allow_heal
        )
        if not self._use_async_quorum:
            self.wait_quorum()

            if self._healing:
                # eagerly apply pending state_dict so we can run the forwards pass
                self._apply_pending_state_dict()

                # we are forcing healing at the beginning so we're in a good state
                # and don't need to zero_grad
                self._healing = False

    def wait_quorum(self) -> None:
        """
        Wait for the quorum to complete.

        ProcessGroup will be in a healthy state after this returns.
        """
        assert (
            self._quorum_future is not None
        ), "must call start_quorum before wait_quorum"
        self._quorum_future.result()

    def _async_quorum(self, room_id: str, allow_heal: bool) -> None:
        (
            quorum_id,
            replica_rank,
            replica_world_size,
            address,
            store_address,
            max_step,
            max_rank,
            max_world_size,
            heal,
        ) = self._client.quorum(
            room_id=room_id,
            rank=self._rank,
            step=self._step,
            checkpoint_server_addr=self._ckpt_server.address(),
        )

        # When using async quorum we need to take the recovered workers.
        # When not using async quorum we need to take the max world size as all
        # workers will be healthy.
        self._participating_rank, self._participating_world_size = (
            (max_rank, max_world_size)
            if self._use_async_quorum or not allow_heal
            else (replica_rank, replica_world_size)
        )

        # For fixed with spares we need to ensure that we don't have more
        # participating replicas than the min replica size.
        if self._world_size_mode == WorldSizeMode.FIXED_WITH_SPARES:
            self._participating_world_size = min(
                self._participating_world_size, self._min_replica_size
            )
            if (
                self._participating_rank is not None
                and self._participating_rank >= self._min_replica_size
            ):
                self._participating_rank = None

        if quorum_id != self._quorum_id:
            store_prefixed_addr = f"{store_address}/torchft/{quorum_id}/{self._rank}"

            self._logger.info(f"reconfiguring for {quorum_id=} {store_prefixed_addr=}")
            # We use the replica rank and world as we want all replicas in the PG.
            self._pg.configure(store_prefixed_addr, replica_rank, replica_world_size)
            self._quorum_id = quorum_id

        # See manager.rs for healing conditions
        if heal and allow_heal:
            self._healing = True
            self._logger.info(
                f"healing required, fetching checkpoint server address from {address=} {max_step=}"
            )
            primary_client = ManagerClient(address, timeout=self._timeout)
            checkpoint_server_address = primary_client.checkpoint_address(self._rank)

            self._logger.info(f"fetching checkpoint from {checkpoint_server_address=}")
            self._pending_state_dict = CheckpointServer.load_from_address(
                checkpoint_server_address
            )
            self.load_state_dict(self._pending_state_dict["torchft"])
            # we apply the user state dict only when safe from the main thread

            # This isn't strictly needed as loading the state_dict above should
            # restore the correct step but it makes writing tests simpler.
            self._step = max_step

    def _apply_pending_state_dict(self) -> None:
        assert self._healing, "must be in healing state"

        # synchronize on future
        assert self._quorum_future is not None, "must call step before should_commit"
        self._quorum_future.result()

        self._logger.info("applying pending state dict")

        assert self._pending_state_dict is not None, "checkpoint was not staged"
        self._load_state_dict(self._pending_state_dict["user"])
        self._pending_state_dict = None

    def should_commit(self) -> bool:
        """
        .. note::
            We recommend using the :py:class:`torchft.optim.OptimizerWrapper` instead of calling this directly.

        Must be called after the backwards pass completes but before stepping the optimizer.

        The optimizer must only be stepped if this returns True.

        This must be called on all workers within a replica group. This uses a
        collective to ensure all workers within a replica return the same value.
        If an error occurs on any worker, all workers will return False.
        Different replica groups may return different values.

        This should only be called once per step.

        Returns:
            True if the optimizer should be stepped, False otherwise
        """
        for work in self._pending_work:
            # check at the beginning of since .wait() may trigger errors
            if self._errored is not None:
                break

            # We swallow the error at in a future then callback so this will
            # never return an error.
            work.wait()

        self._pending_work = []

        # apply state_dict if healing
        if self._healing:
            self._apply_pending_state_dict()

        enough_replicas = self.num_participants() >= self._min_replica_size
        local_should_commit = enough_replicas and self._errored is None
        should_commit = self._client.should_commit(
            self._rank, self._step, local_should_commit
        )
        self._logger.info(
            f"should_commit={should_commit} enough_replicas={enough_replicas}, errored={self._errored}"
        )

        self._ckpt_server.disallow_checkpoint()

        # decide whether we're in a healthy state to increase the step count
        if should_commit:
            self._step += 1
            self._batches_committed += self.num_participants()

        return should_commit

    def load_state_dict(self, state_dict: Dict[str, int]) -> None:
        """
        Load the state dict from a previous checkpoint.

        This will restore the step count and internal metadata.

        Args:
            state_dict: the state dict to load
        """
        self._step = state_dict["step"]
        self._batches_committed = state_dict["batches_committed"]

    def state_dict(self) -> Dict[str, int]:
        """
        Get the state dict for this manager.

        This can be used to checkpoint the state of the manager to restore
        from a previous checkpoint.

        Returns:
            the state dict for this manager
        """
        return {"step": self._step, "batches_committed": self._batches_committed}

    def current_step(self) -> int:
        """
        Get the current step count.

        This number is incremented on .step()

        Returns:
            the current step count
        """
        return self._step

    def batches_committed(self) -> int:
        """
        Get the total number of batches committed across all steps and replicas.
        5 replicas participating in 2 steps is 10 batches but may be more than
        10 examples depending on batch size.

        This number is incremented on .step()

        Returns:
            the total number of batches committed
        """
        return self._batches_committed

    def num_participants(self) -> int:
        """
        Get the number of participants in the current quorum.

        This is the number of replicas participating in the current step.

        Returns:
            the number of participants in the current quorum
        """
        assert self._participating_world_size >= 0, "internal error"
        return self._participating_world_size

    def is_participating(self) -> bool:
        """
        Get whether this replica is participating in the current quorum.

        Returns:
            whether this replica is participating in the current quorum
        """
        if self._participating_rank is None:
            return False
        if self._healing:
            assert self._use_async_quorum
            return False
        return True


class _ManagerLogger:
    def __init__(self, manager: Manager, replica_id: str, rank: int) -> None:
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._replica_id = replica_id
        self._rank = rank
        self._manager = manager

    def prefix(self) -> str:
        return (
            f"[{self._replica_id}/{self._rank} - step {self._manager.current_step()}]"
        )

    def info(self, msg: str) -> None:
        self._logger.info(f"{self.prefix()} {msg}")

    def warn(self, msg: str) -> None:
        self._logger.warn(f"{self.prefix()} {msg}")

    def exception(self, msg: str) -> None:
        self._logger.exception(f"{self.prefix()} {msg}")
