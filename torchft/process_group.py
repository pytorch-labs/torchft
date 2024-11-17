# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import threading
from abc import ABC
from datetime import timedelta
from typing import Callable, List, Optional, Tuple, Type

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch._C._distributed_c10d import (
    _register_process_group,
    _unregister_process_group,
)
from torch.distributed import (
    BroadcastOptions,
    DeviceMesh,
    get_rank,
    PrefixStore,
    ProcessGroup as BaseProcessGroup,
    ProcessGroupGloo as BaseProcessGroupGloo,
    ProcessGroupNCCL as BaseProcessGroupNCCL,
    Store,
    TCPStore,
)
from torch.distributed.distributed_c10d import _world, Work

from torch.futures import Future

logger = logging.getLogger(__name__)

# TODO: use non strings which are cheaper
_QUEUE_CLOSE = "queue_close"
_FUTURE_RESULT = "fut_result"
_FUTURE_EXCEPTION = "fut_exception"


def _get(queue: mp.Queue, timeout) -> object:
    v = queue.get(timeout=timeout)
    if isinstance(v, Exception):
        raise v
    return v


def create_store(store_addr: str) -> Store:
    """
    Creates a PrefixStore(TCPStore(...)) from an address in the format:

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
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._group_name = None

    def configure(self, store_addr: str, rank: int, world_size: int) -> None:
        raise NotImplementedError("not implemented")

    def allreduce(self, tensors: List[torch.Tensor], opts: object) -> Work:
        raise NotImplementedError("not implemented")

    def allgather(
        self,
        output_tensors: List[List[torch.Tensor]],
        input_tensor: List[torch.Tensor],
        opts: object,
    ) -> Work:
        raise NotImplementedError("not implemented")

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

    def register(self, name: str) -> None:
        """
        Registers the process group with the global registry. This enables usage
        with things like functional_collectives which are compilable.

        This should only be called once.

        Args:
            name: name must be a unique name for this process group
        """

        self._group_name = f"{self.getBackendName()}:{name}"
        _register_process_group(self.group_name, self)

        # This is needed for DeviceMesh to work
        # Resizable worlds don't fit well into DeviceMesh so we register a world
        # size 1 PG.
        _world.pg_map[self] = (None, None)
        _world.pg_names[self] = self._group_name
        _world.pg_to_tag[self] = self._group_name
        _world.tags_to_pg.setdefault(self._group_name, []).append(self)
        # these PGs can be resized so we lie about the rank mapping
        _world.pg_group_ranks[self] = {get_rank(): 0}

    @property
    def group_name(self) -> str:
        if self._group_name is None:
            raise ValueError("ProcessGroup name not set")
        return self._group_name

    def unregister(self) -> None:
        """
        Unregisters the process group with the global registry.

        Must be registered first.
        """
        _unregister_process_group(self.group_name)


class ProcessGroupWrapper(ProcessGroup):
    PG_CLASS: Type[BaseProcessGroup]
    """
    This is a wrapper around any ProcessGroup with a reconfiguration method.
    """

    def __init__(self) -> None:
        super().__init__(0, 1)
        self._pg = None

    def configure(self, store_addr: str, rank: int, world_size: int) -> None:
        if self._pg is not None:
            if hasattr(self._pg, "abort"):
                self._pg.abort()
            self._pg = None

        store = create_store(store_addr)

        # TODO: set global timeout
        self._pg = self.PG_CLASS(store, rank, world_size)

    def allreduce(self, tensors: List[torch.Tensor], opts: object) -> Work:
        return self._pg.allreduce(tensors, opts)

    def allgather(
        self,
        output_tensors: List[List[torch.Tensor]],
        input_tensor: List[torch.Tensor],
        opts: object,
    ) -> Work:
        return self._pg.allgather(output_tensors, input_tensor, opts)

    def broadcast(self, tensor_list: List[torch.Tensor], opts: object) -> Work:
        return self._pg.broadcast(tensor_list, opts)

    def size(self) -> int:
        return self._pg.size()


class ProcessGroupGloo(ProcessGroupWrapper):
    """
    This is a reconfigurable version of ProcessGroupGloo.
    """

    PG_CLASS = BaseProcessGroupGloo

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

    PG_CLASS = BaseProcessGroupNCCL

    def getBackendName(self) -> str:
        return "torchft-nccl"


class DummyWork(dist._Work):
    def __init__(self, result):
        super().__init__()
        self.result_ = result
        self.future_ = torch.futures.Future()
        self.future_.set_result(result)

    def wait(self, timeout):
        return True

    def get_future(self):
        return self.future_


class ProcessGroupDummy(ProcessGroup):
    """
    This PG only supports world_size of 1
    """

    def __init__(self, rank, world):
        super().__init__(rank, world)
        assert rank == 0
        assert world == 1

        self._rank = rank
        self._world = world
        self.wait_count = 0
        self.get_future_count = 0
        self._work = []

    def broadcast(self, tensor_list, opts):
        res = DummyWork(tensor_list)
        self._work.append(res)
        return res

    def allgather(self, output_tensors, input_tensor, opts):
        for o, i in zip(output_tensors[0], input_tensor):
            o.copy_(i)

        res = DummyWork(output_tensors)
        self._work.append(res)
        return res

    def allreduce(self, tensors, opts):
        res = DummyWork(tensors)
        self._work.append(res)
        return res

    def size(self):
        return self._world

    def getBackendName(self):
        return "torchft-dummy"


class BabyWork(Work):
    def __init__(
        self,
        pg: "ProcessGroupBaby",
        tx: mp.Queue,
        rx: mp.Queue,
        op_id: int,
        timeout: float,
    ):
        super().__init__()

        self._pg = pg
        self._tx = tx
        self._rx = rx
        self._op_id = op_id
        self._timeout = timeout

    def wait(self) -> bool:
        self._tx.put(("wait", self._op_id), timeout=self._timeout)
        assert _get(self._rx, self._timeout) == self._op_id
        return True

    def get_future(self) -> Future:
        return self._pg._get_future(self._op_id)


class BabyWorkNCCL(BabyWork):
    def wait(self) -> bool:
        self._tx.put(("synchronize", self._op_id), timeout=self._timeout)
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

    PG_CLASS: Type[BaseProcessGroup]
    WORK_CLASS: Type[BabyWork] = BabyWork

    def __init__(self, timeout: float = 60.0) -> None:
        super().__init__(0, 1)

        self._world_size = -1

        self._p = None
        self._tx = None
        self._rx = None
        self._future_queue = None
        self._future_thread = None

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
            store = create_store(store_addr)

            pg = cls.PG_CLASS(store, rank, world_size)

            work = {}
            next_op_id = 0

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
                    op_id = op[1]
                    work[op_id].wait()
                    del work[op_id]
                    tx.put(op_id)
                elif cmd == "future":
                    op_id = op[1]

                    def callback(fut: Future):
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

    def _get_future(self, op_id: int) -> Future:
        with self._futures_lock:
            fut = Future()
            self._futures[op_id] = fut
            self._tx.put(("future", op_id), timeout=self._timeout)

        assert _get(self._rx, self._timeout) == op_id
        # TODO: return correct tensor instead of None
        return fut

    def _run_func(self, func: str, *args: object, **kwargs: object) -> Work:
        self._tx.put(("func", func, args, kwargs), timeout=self._timeout)
        op_id = _get(self._rx, self._timeout)
        assert isinstance(op_id, int), f"invalid return {op_id}"
        return self.WORK_CLASS(
            pg=self, tx=self._tx, rx=self._rx, op_id=op_id, timeout=self._timeout
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
    PG_CLASS = BaseProcessGroupGloo

    def getBackendName(self):
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

    PG_CLASS = BaseProcessGroupNCCL
    WORK_CLASS = BabyWorkNCCL

    def getBackendName(self):
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
    mesh_dim_names = list(mesh.mesh_dim_names)
    mesh_dim_names.insert(dim, name)

    return DeviceMesh.from_group(
        group=groups,
        device_type=mesh.device_type,
        mesh=mesh.mesh.unsqueeze(dim),
        mesh_dim_names=mesh_dim_names,
    )
