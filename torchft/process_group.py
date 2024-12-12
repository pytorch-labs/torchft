# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Process Groups
=========================

This module implements fault tolerant process groups that can be reconfigured
and resized at runtime.

These extend the standard PyTorch ProcessGroup API and can be used in most
places that would accept a standard process group. As these can change size at
runtime users need to take care to not assume a static rank or world size.
"""

import logging
import threading
from abc import ABC
from datetime import timedelta
from typing import TYPE_CHECKING, Dict, List, Optional, Type

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# pyre-fixme[21]: no attribute ProcessGroupNCCL
# pyre-fixme[21]: no attribute ProcessGroupGloo
from torch.distributed import (
    BroadcastOptions,
    DeviceMesh,
    PrefixStore,
    ProcessGroup as BaseProcessGroup,
    ProcessGroupGloo as BaseProcessGroupGloo,
    ProcessGroupNCCL as BaseProcessGroupNCCL,
    Store,
    TCPStore,
    get_rank,
)
from torch.distributed.distributed_c10d import Work, _world
from torch.futures import Future

if TYPE_CHECKING:
    from torchft.manager import Manager

logger: logging.Logger = logging.getLogger(__name__)

# TODO: use non strings which are cheaper
_QUEUE_CLOSE = "queue_close"
_FUTURE_RESULT = "fut_result"
_FUTURE_EXCEPTION = "fut_exception"


def _get(queue: mp.Queue, timeout: float) -> object:
    v = queue.get(timeout=timeout)
    if isinstance(v, Exception):
        raise v
    return v


def create_store_client(store_addr: str) -> Store:
    """
    Creates a PrefixStore(TCPStore(...)) client from an address in the format:

    host:port/prefix

    Ex: localhost:1234/my/prefix
    """
    host, _, rest = store_addr.partition(":")
    port, _, prefix = rest.partition("/")

    store = TCPStore(
        host_name=host,
        port=int(port),
        is_master=False,
        wait_for_workers=False,
    )
    store = PrefixStore(prefix, store)
    return store


class ProcessGroup(BaseProcessGroup):
    def __init__(self, *args: object, **kwargs: object) -> None:
        # pyre-fixme[6]: got object
        super().__init__(*args, **kwargs)

        self._group_name: Optional[str] = None

    def configure(self, store_addr: str, rank: int, world_size: int) -> None:
        """
        This reconfigures the ProcessGroup to use a new store, rank and world size.

        Every time this is called it must be provided with a unique prefixed
        store address. I.e. localhost:1234/my/prefix/1

        Args:
            store_addr: address of the store to use
            rank: rank of this process
            world_size: world size of this process group
        """
        raise NotImplementedError("not implemented")

    # pyre-fixme[14]: inconsistent override
    def allreduce(self, tensors: List[torch.Tensor], opts: object) -> Work:
        raise NotImplementedError("not implemented")

    # pyre-fixme[14]: inconsistent override
    def allgather(
        self,
        output_tensors: List[List[torch.Tensor]],
        input_tensor: List[torch.Tensor],
        opts: object,
    ) -> Work:
        raise NotImplementedError("not implemented")

    # pyre-fixme[14]: inconsistent override
    def broadcast(self, tensor_list: List[torch.Tensor], opts: object) -> Work:
        raise NotImplementedError("not implemented")

    def broadcast_one(self, tensor: torch.Tensor, root: int) -> Work:
        opts = BroadcastOptions()
        opts.rootRank = root
        return self.broadcast([tensor], opts)

    def size(self) -> int:
        raise NotImplementedError("not implemented")

    def getBackendName(self) -> str:
        raise NotImplementedError("not implemented")

    def register(self, name: str) -> "ProcessGroup":
        """
        Registers the process group with the global registry. This enables usage
        with things like functional_collectives which are compilable.

        This should only be called once.

        Args:
            name: name must be a unique name for this process group
        """

        group_name = f"{self.getBackendName()}:{name}"

        # This is needed for DeviceMesh and functional collectives to work.
        # Resizable worlds don't fit well into DeviceMesh so we register a world
        # size 1 PG.

        def create_pg(
            prefix_store: PrefixStore, rank: int, world_size: int, timeout: float
        ) -> ProcessGroup:
            return self

        if torch.cuda.is_available():
            devices = ["cuda", "cpu"]
        else:
            devices = ["cpu"]
        dist.Backend.register_backend(group_name, create_pg, devices=devices)

        return dist.new_group(
            ranks=[dist.get_rank()],
            backend=group_name,
            group_desc=group_name,
            timeout=timedelta(seconds=60.0),  # this timeout isn't used
        )

    @property
    def group_name(self) -> str:
        if self._group_name is None:
            raise ValueError("ProcessGroup name not set")
        return self._group_name

    def _set_group_name(self, name: str) -> None:
        self._group_name = name

    def unregister(self) -> None:
        """
        Unregisters the process group with the global registry.

        Must be registered first.
        """
        dist.destroy_process_group(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ProcessGroupWrapper(ProcessGroup):
    PG_CLASS: Type[BaseProcessGroup]  # pyre-fixme[13]: never initialized
    """
    This is a wrapper around any ProcessGroup with a reconfiguration method.
    """

    def __init__(self, pg: Optional[ProcessGroup] = None) -> None:
        super().__init__(0, 1)
        self._pg: Optional[BaseProcessGroup] = pg

    def configure(self, store_addr: str, rank: int, world_size: int) -> None:
        pg = self._pg
        if isinstance(pg, ProcessGroup):
            pg.configure(store_addr, rank, world_size)
            return

        if pg is not None:
            if hasattr(pg, "abort"):
                pg.abort()  # pyre-fixme[16]: no attribute abort
            self._pg = None

        store = create_store_client(store_addr)

        # TODO: set global timeout
        # pyre-fixme[20]: expects argument options
        self._pg = self.PG_CLASS(store, rank, world_size)

    def allreduce(self, tensors: List[torch.Tensor], opts: object) -> Work:
        return self.parent.allreduce(tensors, opts)

    def allgather(
        self,
        output_tensors: List[List[torch.Tensor]],
        input_tensor: List[torch.Tensor],
        opts: object,
    ) -> Work:
        return self.parent.allgather(output_tensors, input_tensor, opts)

    def broadcast(self, tensor_list: List[torch.Tensor], opts: object) -> Work:
        return self.parent.broadcast(tensor_list, opts)

    def size(self) -> int:
        return self.parent.size()

    @property
    def parent(self) -> BaseProcessGroup:
        assert self._pg is not None, "process group not initialized"
        return self._pg

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(pg={self._pg})"


class ProcessGroupGloo(ProcessGroupWrapper):
    """
    This is a reconfigurable version of ProcessGroupGloo.
    """

    PG_CLASS: Type[BaseProcessGroup] = (
        BaseProcessGroupGloo  # pyre-fixme[16]: no attribute ProcessGroupGloo
    )

    def getBackendName(self) -> str:
        return "torchft-gloo"


class ProcessGroupNCCL(ProcessGroupWrapper):
    """
    This is a reconfigurable version of ProcessGroupNCCL.

    WARNING: this may result in deadlocks due to NCCL error handling. This is
    provided for completeness but your mileage may vary.

    TODO: verify shutdown correctness with latest NCCL. This currently will call
    abort when reconfiguring, we need to ensure this is safe.
    """

    PG_CLASS: Type[BaseProcessGroup] = (
        BaseProcessGroupNCCL  # pyre-fixme[16]: no attribute ProcessGroupNCCL
    )

    def getBackendName(self) -> str:
        return "torchft-nccl"


class _DummyWork(dist._Work):
    def __init__(self, result: object) -> None:
        super().__init__()
        self.result_ = result
        # pyre-fixme[29]: Future is not a function
        self.future_: torch.futures.Future[object] = torch.futures.Future()
        self.future_.set_result(result)

    def wait(self, timeout: Optional[timedelta] = None) -> bool:
        return True

    def get_future(self) -> torch.futures.Future[object]:
        return self.future_


class ProcessGroupDummy(ProcessGroup):
    """
    This process group discards all data passed to it and returns success. This
    is intended for rare cases where we want to discard certain operations
    without modifying the underlying library.

    This PG only supports world_size of 1.
    """

    def __init__(self, rank: int, world: int) -> None:
        super().__init__(rank, world)
        assert rank == 0
        assert world == 1

        self._rank = rank
        self._world = world
        self.wait_count = 0
        self.get_future_count = 0
        self._work: List[Work] = []
        self.configure_count = 0

    def configure(self, store_addr: str, rank: int, world_size: int) -> None:
        self.configure_count += 1

    def broadcast(self, tensor_list: List[torch.Tensor], opts: object) -> Work:
        res = _DummyWork(tensor_list)
        self._work.append(res)
        return res

    def allgather(
        self,
        output_tensors: List[List[torch.Tensor]],
        input_tensor: List[torch.Tensor],
        opts: object,
    ) -> Work:
        for o, i in zip(output_tensors[0], input_tensor):
            o.copy_(i)

        res = _DummyWork(output_tensors)
        self._work.append(res)
        return res

    def allreduce(self, tensors: List[torch.Tensor], opts: object) -> Work:
        res = _DummyWork(tensors)
        self._work.append(res)
        return res

    def size(self) -> int:
        return self._world

    def getBackendName(self) -> str:
        return "torchft-dummy"


class _ErrorSwallowingWork(Work):
    def __init__(
        self,
        pg: "ErrorSwallowingProcessGroupWrapper",
        work: Work,
        default_result: object,
    ) -> None:
        super().__init__()

        self._pg = pg
        self._work = work
        self._default_result = default_result

    def wait(self, timeout: Optional[timedelta] = None) -> bool:
        try:
            self._work.wait()
        except Exception as e:
            self._pg.report_error(e)

        return True

    def get_future(self) -> Future[object]:
        fut = self._work.get_future()

        # schedule error handling as a continuation on the Future
        def callback(
            fut: torch.futures.Future[List[torch.Tensor]],
        ) -> object:
            try:
                return fut.value()
            except Exception as e:
                logger.exception(f"got exception in future -- skipping remaining: {e}")
                self._pg.report_error(e)
                return self._default_result

        fut = fut.then(callback)
        return fut


class ErrorSwallowingProcessGroupWrapper(ProcessGroupWrapper):
    """
    This is a wrapper around any ProcessGroup that will swallow errors and
    return dummy results on error.

    This is intended to allow handling errors outside of the training loop to
    avoid having to modify modeling code to support error handling.

    After an error occurs all future operations will be skipped until the
    process group is reconfigured via ``configure``.
    """

    def __init__(self, pg: ProcessGroup) -> None:
        super().__init__(pg)

        self._error: Optional[Exception] = None

    def configure(self, store_addr: str, rank: int, world_size: int) -> None:
        self._error = None

        super().configure(store_addr, rank, world_size)

    def report_error(self, e: Exception) -> None:
        """
        Report an error to this process group. This will cause all future
        operations to be skipped until the process group is reconfigured via
        ``configure``.

        Args:
            e: exception to report
        """
        self._error = e

    def error(self) -> Optional[Exception]:
        """
        Returns the error that was reported to this process group.

        Returns:
            exception that was reported
        """
        return self._error

    def allreduce(self, tensors: List[torch.Tensor], opts: object) -> Work:
        if self._error is not None:
            return _DummyWork(tensors)

        try:
            return _ErrorSwallowingWork(
                self,
                super().allreduce(tensors, opts),
                tensors,
            )
        except Exception as e:
            self.report_error(e)
            return _DummyWork(tensors)


class _ManagedWork(Work):
    def __init__(self, manager: "Manager", work: Work, default_result: object) -> None:
        super().__init__()

        self._manager = manager
        self._work = work
        self._default_result = default_result

    def wait(self, timeout: Optional[timedelta] = None) -> bool:
        try:
            if timeout is not None:
                self._work.wait(timeout)
            else:
                self._work.wait()
        except Exception as e:
            self._manager.report_error(e)

        return True

    def get_future(self) -> Future[object]:
        return self._manager.wrap_future(self._work.get_future(), self._default_result)


class ManagedProcessGroup(ProcessGroupWrapper):
    """
    This is a wrapper around any ProcessGroup that is managed by a torchft
    Manager.

    This uses the ProcessGroup that is configured in the Manager. The world size
    is dynamic and will report the number of active particpants in the quorum to
    the model.

    Any errors will be asynchronously reported to the manager and only successes
    will be returned to the caller.
    """

    def __init__(self, manager: "Manager") -> None:
        super().__init__(manager._pg)

        self._manager = manager

    def allreduce(self, tensors: List[torch.Tensor], opts: object) -> Work:
        if self._manager.errored() is not None:
            return _DummyWork(tensors)

        try:
            work = super().allreduce(tensors, opts)
        except Exception as e:
            self._manager.report_error(e)
            return _DummyWork(tensors)

        return _ManagedWork(
            self._manager,
            work,
            tensors,
        )

    def size(self) -> int:
        return self._manager.num_participants()


class _BabyWork(Work):
    def __init__(
        self,
        pg: "ProcessGroupBaby",
        tx: mp.Queue,
        rx: mp.Queue,
        op_id: int,
        timeout: float,
    ) -> None:
        super().__init__()

        self._pg = pg
        self._tx = tx
        self._rx = rx
        self._op_id = op_id
        self._timeout = timeout

    def wait(self, timeout: Optional[timedelta] = None) -> bool:
        self._tx.put(("wait", self._op_id), timeout=self._timeout)
        assert _get(self._rx, self._timeout) == self._op_id
        return True

    def get_future(self) -> Future[object]:
        return self._pg._get_future(self._op_id)


class _BabyWorkNCCL(_BabyWork):
    def wait(self, timeout: Optional[timedelta] = None) -> bool:
        self._tx.put(("synchronize", self._op_id), timeout=self._timeout)
        # pyre-fixme[23]: unable to unpack into 2 values
        op_id, event = _get(self._rx, self._timeout)
        assert op_id == self._op_id
        assert isinstance(event, torch.cuda.Event)

        # Wait on Event makes the stream wait but not the CPU thread.
        event.wait()

        return True


class ProcessGroupBaby(ProcessGroup):
    """
    This is a process group that runs the underlying process group in a
    subprocess. Since it's running in a subprocess all tensors need to be in
    shared memory or will be moved to shared memory. CUDA tensors are implicitly
    share able and don't need any changes.

    """

    PG_CLASS: Type[BaseProcessGroup]  # pyre-fixme[13]: never initialized
    WORK_CLASS: Type[_BabyWork] = _BabyWork

    def __init__(self, timeout: float = 60.0) -> None:
        super().__init__(0, 1)

        self._world_size = -1

        self._p: Optional[mp.Process] = None
        self._tx: Optional[mp.Queue] = None
        self._rx: Optional[mp.Queue] = None
        self._future_queue: Optional[mp.Queue] = None
        self._future_thread: Optional[threading.Thread] = None
        self._futures: Dict[int, Future[object]] = {}
        self._futures_lock = threading.Lock()

        self._timeout = timeout

    def configure(self, store_addr: str, rank: int, world_size: int) -> None:
        if self._p is not None:
            self._p.kill()

        self._world_size = world_size

        if self._tx is not None:
            self._tx.close()
        if self._rx is not None:
            self._rx.close()
        if self._future_queue is not None:
            self._future_queue.put(_QUEUE_CLOSE)
            assert self._future_queue is not None
            self._future_queue.close()

        ctx = mp.get_context("spawn")
        self._tx = ctx.Queue()
        self._rx = ctx.Queue()

        # futures need thread to fire callbacks
        self._future_queue = ctx.Queue()
        # this lock needs to be held when manipulating _futures
        self._futures_lock = threading.Lock()
        self._futures = {}
        self._future_thread = threading.Thread(
            target=self._future_handler,
            args=(self._future_queue,),
            daemon=True,
        )
        self._future_thread.start()

        self._p = ctx.Process(
            target=self._worker,
            args=(store_addr, rank, world_size, self._tx, self._rx, self._future_queue),
            daemon=True,
        )
        self._p.start()

    @classmethod
    def _worker(
        cls,
        store_addr: str,
        rank: int,
        world_size: int,
        rx: mp.Queue,
        tx: mp.Queue,
        future_queue: mp.Queue,
    ) -> None:
        try:
            store = create_store_client(store_addr)

            # pyre-fixme[20]: expects argument options
            pg = cls.PG_CLASS(store, rank, world_size)

            work = {}
            next_op_id: int = 0

            while True:
                op = rx.get()
                cmd = op[0]
                if cmd == "func":
                    func_name, args, kwargs = op[1:]
                    fn = getattr(pg, func_name)
                    work[next_op_id] = fn(*args, **kwargs)
                    tx.put(next_op_id)
                    next_op_id += 1
                elif cmd == "wait":
                    op_id: int = op[1]
                    work[op_id].wait()
                    del work[op_id]
                    tx.put(op_id)
                elif cmd == "future":
                    op_id: int = op[1]

                    def callback(fut: Future[object]) -> None:
                        try:
                            fut.wait()
                            future_queue.put((op_id, _FUTURE_RESULT, None))
                        except Exception as e:
                            future_queue.put((op_id, _FUTURE_EXCEPTION, e))

                    work[op_id].get_future().add_done_callback(callback)
                    tx.put(op_id)
                elif cmd == "synchronize":
                    # CUDA only, use events instead of waiting on CPU
                    op_id = op[1]

                    # With WorkNCCL this makes the stream wait not the CPU when
                    # no timeout is passed.
                    work[op_id].wait()

                    # Register event on the stream that we can pass to the main
                    # process.
                    event = torch.cuda.Event(interprocess=True)
                    event.record()

                    del work[op_id]
                    tx.put((op_id, event))
                else:
                    raise ValueError(f"unknown cmd: {cmd}")

        except Exception as e:
            logger.exception("worker errored")
            tx.put(e)

    def _future_handler(self, future_queue: mp.Queue) -> None:
        try:
            while True:
                cmd = future_queue.get()
                if cmd == _QUEUE_CLOSE:
                    break
                op_id, mode, data = cmd
                with self._futures_lock:
                    fut = self._futures[op_id]
                    del self._futures[op_id]
                if mode == _FUTURE_RESULT:
                    fut.set_result(data)
                elif mode == _FUTURE_EXCEPTION:
                    fut.set_exception(data)
                else:
                    raise ValueError(f"unknown mode {mode}")
        except Exception as e:
            logger.exception(f"got unexpected error in future handler: {e}")

    def _get_future(self, op_id: int) -> Future[object]:

        with self._futures_lock:
            fut = Future()  # pyre-fixme[29]: is not a function
            self._futures[op_id] = fut
            assert self._tx is not None
            self._tx.put(("future", op_id), timeout=self._timeout)

        assert self._rx is not None
        assert _get(self._rx, self._timeout) == op_id
        # TODO: return correct tensor instead of None
        return fut

    def _run_func(self, func: str, *args: object, **kwargs: object) -> Work:
        rx = self._rx
        tx = self._tx
        assert rx is not None
        assert tx is not None

        tx.put(("func", func, args, kwargs), timeout=self._timeout)

        op_id = _get(rx, self._timeout)
        assert isinstance(op_id, int), f"invalid return {op_id}"

        return self.WORK_CLASS(
            pg=self, tx=tx, rx=rx, op_id=op_id, timeout=self._timeout
        )

    def allreduce(self, tensors: List[torch.Tensor], opts: object) -> Work:
        assert isinstance(tensors, list), "input must be list"

        for tensor in tensors:
            if not tensor.is_shared():
                tensor.share_memory_()

        return self._run_func("allreduce", tensors, opts)

    def size(self) -> int:
        return self._world_size


class ProcessGroupBabyGloo(ProcessGroupBaby):
    """
    This is a ProcessGroup that runs Gloo in a subprocess.

    For most use cases you should prefer ProcessGroupGloo or
    ProcessGroupBabyNCCL.
    """

    PG_CLASS: Type[BaseProcessGroup] = (
        BaseProcessGroupGloo  # pyre-fixme[16]: no attribute ProcessGroupGloo
    )

    def getBackendName(self) -> str:
        return "torchft-baby-gloo"


class ProcessGroupBabyNCCL(ProcessGroupBaby):
    """
    This is a ProcessGroup that runs NCCL in a subprocess.

    For the NCCL backend, extra memory will be used by the subprocesses CUDA
    context compared to running NCCL in the main process. This is typically
    around ~1GB.

    The returned Work objects only synchronize on the cuda stream and not on the
    CPU side. This works by passing CUDA Events between the processes. To do a
    CPU synchronize, call torch.cuda.synchronize() after wait().

    WARNING: If the child process is killed while an operation is running, CUDA
    tensors may leak in the current PyTorch implementation. TODO fix
    """

    PG_CLASS: Type[BaseProcessGroup] = (
        BaseProcessGroupNCCL  # pyre-fixme[16]: no attribute ProcessGroupNCCL
    )
    WORK_CLASS = _BabyWorkNCCL

    def getBackendName(self) -> str:
        return "torchft-baby-nccl"


def extend_device_mesh(
    mesh: DeviceMesh, pg: ProcessGroup, name: str = "dp", dim: int = 0
) -> DeviceMesh:
    """
    This is a helper method to extend a traditional DeviceMesh with a torchft ProcessGroup for usage with DeviceMesh based APIs such as FSDPv2 with hybrid sharding.

    Resizable PGs aren't natively supported by DeviceMesh so we lie to
    DeviceMesh and say the PG is world size 1. This is fine as long as any
    numeric scaling is handled at the PG level.

    Args:
        mesh: The DeviceMesh to extend
        pg: The ProcessGroup to add to the mesh
        name: The name of the new dimension
        dim: The dimension to add the ProcessGroup to
    """
    groups = mesh.get_all_groups()
    groups.insert(dim, pg)
    mesh_dim_names = list(mesh.mesh_dim_names or [])
    mesh_dim_names.insert(dim, name)

    return DeviceMesh.from_group(
        group=groups,
        device_type=mesh.device_type,
        mesh=mesh.mesh.unsqueeze(dim),
        mesh_dim_names=tuple(mesh_dim_names),
    )
