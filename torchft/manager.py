# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import uuid
import socket
from typing import Dict, Optional, List
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import torch
from torch.distributed import TCPStore, PrefixStore, Work, ReduceOp
from torch.optim import Optimizer

# pyre-fixme[21]: can't find rust module
from torchft.torchft import Manager as _Manager, ManagerClient
from torchft.checkpointing import CheckpointServer

logger: logging.Logger = logging.getLogger(__name__)

MANAGER_ADDR_KEY: str = "manager_addr"
MANAGER_DEFAULT_PORT: int = int(os.environ.get("TORCHFT_MANAGER_PORT", 29511))


class Manager:
    """
    Manager manages the full fault tolerant training loop.

    NOTE: when saving periodic checkpoints you must save and restore the
    Manager's state_dict as well to avoid synchronization issues.
    """

    def __init__(
        self,
        pg,
        load_state_dict,
        state_dict,
        min_replica_size: int,
        port: int = MANAGER_DEFAULT_PORT,
        use_async_quorum: bool = True,
        timeout: timedelta = timedelta(seconds=60),
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
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
        self._use_async_quorum = use_async_quorum
        self._timeout = timeout

        store_addr = store_addr or os.environ["MASTER_ADDR"]
        store_port = store_port or int(os.environ["MASTER_PORT"])
        rank = rank or int(os.environ["RANK"])
        world_size = world_size or int(os.environ["WORLD_SIZE"])
        self._rank = rank
        self._min_replica_size = min_replica_size

        self._ckpt_server = CheckpointServer(
            lambda: {
                "user": state_dict(),
                "torchft": self.state_dict(),
            }
        )
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._quorum_future = None

        self._store = TCPStore(
            host_name=store_addr,
            port=store_port,
            is_master=False,
            wait_for_workers=False,
        )
        self._pg = pg

        if rank == 0:
            hostname = socket.gethostname()
            addr = f"http://{hostname}:{port}"
            bind = f"[::]:{port}"
            lighthouse_addr = lighthouse_addr or os.environ["TORCHFT_LIGHTHOUSE"]

            if replica_id is None:
                replica_id = str(uuid.uuid4())
            # pyre-fixme[16]: can't find rust module
            self._manager = _Manager(
                replica_id=replica_id,
                lighthouse_addr=lighthouse_addr,
                address=addr,
                bind=bind,
                store_addr=f"{store_addr}:{store_port}",
                world_size=world_size,
            )

            self._store.set(MANAGER_ADDR_KEY, addr)

        addr = self._store.get(MANAGER_ADDR_KEY).decode("utf-8")
        # pyre-fixme[16]: can't find rust module
        self._client = ManagerClient(addr, timeout=timeout)

        self._step = 0
        self._quorum_id = -1
        self._errored = False
        self._healing = False
        self._participating_replicas = 0
        self._pending_work: List[torch.futures.Future[torch.Tensor]] = []
        self._batches_committed = 0

        # first step is 1
        self._should_step = True

    def shutdown(self) -> None:
        self._ckpt_server.shutdown()

    def allreduce_grad(self, grad: torch.Tensor) -> torch.futures.Future[torch.Tensor]:
        if self._errored:
            fut = torch.futures.Future()
            fut.set_result(grad)
            return fut

        self._quorum_future.result()

        if self._healing:
            assert self._use_async_quorum
            grad.zero_()

        # TODO: increase timeout when waiting when healing
        try:
            # Run the allreduce async and save the work object so we can wait on
            # it later.
            work = self._pg.allreduce([grad], ReduceOp.SUM)
            fut = work.get_future()

            # schedule error handling and grad normalization as a continuation
            # on the Future
            def callback(
                fut: torch.futures.Future[List[torch.Tensor]],
            ) -> torch.futures.Future[torch.Tensor]:
                nonlocal grad

                try:
                    val = fut.value()
                except Exception:
                    logger.exception(
                        "got exception in all reduce future -- skipping remaining"
                    )
                    self._errored = True
                    return grad

                grad /= self._participating_replicas

                return grad

            fut = fut.then(callback)
            self._pending_work.append(fut)
            return fut

        except Exception as e:
            logger.exception("got exception in all reduce -- skipping remaining")
            self._errored = True

            fut = torch.futures.Future()
            fut.set_result(grad)
            return fut

    def step(self) -> None:
        if self._should_step:
            self._step += 1
            self._batches_committed += self._participating_replicas

        self._errored = False
        self._healing = False
        self._ckpt_server.allow_checkpoint(self._step)

        # TODO: we should really be wrapping this whole section in a try-except
        # block to allow gracefully recovering from issues in PG setup and quorum.

        self._quorum_future = self._executor.submit(self._async_quorum)
        if not self._use_async_quorum:
            self._quorum_future.result()

            # eagerly apply pending state_dict so we can run the forwards pass
            self._apply_pending_state_dict()

            # we are forcing healing at the beginning so we're in a good state
            # and don't need to zero_grad
            self._healing = False

    def _async_quorum(self) -> None:
        (
            quorum_id,
            replica_rank,
            replica_world,
            address,
            store_address,
            max_step,
            num_max,
            heal,
        ) = self._client.quorum(
            rank=self._rank,
            step=self._step,
            checkpoint_server_addr=self._ckpt_server.address(),
        )
        self._participating_replicas = (
            num_max if self._use_async_quorum else replica_world
        )

        if quorum_id != self._quorum_id:
            logger.info(f"reconfiguring for quorum_id {quorum_id}")
            store_prefixed_addr = f"{store_address}/torchft/{quorum_id}/{self._rank}"
            self._pg.configure(store_prefixed_addr, replica_rank, replica_world)
            self._quorum_id = quorum_id

        # See manager.rs for healing conditions
        if heal:
            self._healing = True
            logger.info("healing required")

            logger.info(f"fetching checkpoint server address from {address}")
            # pyre-fixme[16]: can't find rust module
            primary_client = ManagerClient(address, timeout=self._timeout)
            checkpoint_server_address = primary_client.checkpoint_address(self._rank)

            self._state_dict = CheckpointServer.load_from_address(
                checkpoint_server_address
            )
            self.load_state_dict(self._state_dict["torchft"])
            # we apply the user state dict only when safe from the main thread

            # This isn't strictly needed as loading the state_dict above should
            # restore the correct step but it makes writing tests simpler.
            self._step = max_step

    def _apply_pending_state_dict(self) -> None:
        assert self._healing, "must be in healing state"

        # synchronize on future
        self._quorum_future.result()

        assert self._state_dict is not None, "checkpoint was not staged"

        self._load_state_dict(self._state_dict["user"])
        self._state_dict = None

    def should_commit(self) -> bool:
        for work in self._pending_work:
            # check at the beginning of since .wait() may trigger errors
            if self._errored:
                break

            # We swallow the error at in a future then callback so this will
            # never return an error.
            work.wait()

        self._pending_work = []

        # apply state_dict if healing
        if self._healing:
            self._apply_pending_state_dict()

        enough_replicas = self._participating_replicas >= self._min_replica_size
        local_should_commit = enough_replicas and not self._errored
        should_commit = self._client.should_commit(
            self._rank, self._step, local_should_commit
        )
        logger.info(
            f"should_commit={should_commit} enough_replicas={enough_replicas}, errored={self._errored}"
        )

        self._ckpt_server.disallow_checkpoint()

        # decide whether we're in a healthy state to increase the step count
        self._should_step = should_commit

        return should_commit

    def load_state_dict(self, state_dict: Dict[str, int]) -> None:
        self._step = state_dict["step"]
        self._batches_committed = state_dict["batches_committed"]

    def state_dict(self) -> Dict[str, int]:
        return {"step": self._step, "batches_committed": self._batches_committed}

    def current_step(self) -> int:
        """
        Get the current step count.

        This number is incremented on .step()
        """
        return self._step

    def batches_committed(self) -> int:
        """
        Get the total number of batches committed across all steps and replicas.
        5 replicas participating in 2 steps is 10 batches but may be more than
        10 examples depending on batch size.

        This number is incremented on .step()
        """
        return self._batches_committed
