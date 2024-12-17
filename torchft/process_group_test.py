# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Tuple
from unittest import TestCase, skipUnless
from unittest.mock import Mock

import torch
import torch.distributed as dist
from torch import nn
from torch._C._distributed_c10d import (
    AllgatherOptions,
    AllreduceOptions,
    BroadcastOptions,
    ReduceOp,
    _resolve_process_group,
)
from torch.distributed import (
    ReduceOp,
    TCPStore,
    Work,
    _functional_collectives,
    get_world_size,
)
from torch.distributed.device_mesh import init_device_mesh

from torchft.manager import Manager
from torchft.process_group import (
    ErrorSwallowingProcessGroupWrapper,
    ManagedProcessGroup,
    ProcessGroup,
    ProcessGroupBabyGloo,
    ProcessGroupBabyNCCL,
    ProcessGroupDummy,
    ProcessGroupGloo,
    ProcessGroupNCCL,
    ProcessGroupWrapper,
    _DummyWork,
    _ErrorSwallowingWork,
    _ManagedWork,
    extend_device_mesh,
)


def dummy_init_pg() -> None:
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo", rank=0, world_size=1, store=dist.HashStore()
        )


def _test_pg(
    pg: ProcessGroup,
    example_tensor: torch.Tensor = torch.randn((2, 3), dtype=torch.float32),
) -> Dict[str, dist._Work]:
    """
    Helper function to test a set of collective operations on a given process group.
    """

    shape: torch.Size = example_tensor.shape
    dtype: torch.dtype = example_tensor.dtype

    # Create some dummy tensors for testing
    input_tensor = example_tensor.clone()
    output_tensors = [
        [torch.empty_like(input_tensor) for _ in range(get_world_size(pg))]
    ]
    tensor_list = [torch.empty_like(input_tensor)]

    def check_tensors(arg: Any) -> None:  # pyre-ignore[2]
        """Recursively check tensors for expected shape and dtype."""
        if isinstance(arg, torch.Tensor):
            assert arg.dtype == dtype, f"Output dtype mismatch: {arg.dtype} != {dtype}"
            assert arg.shape == shape, f"Output shape mismatch: {arg.shape} != {shape}"
        elif isinstance(arg, (list, tuple)):
            for item in arg:
                check_tensors(item)

    # Test collectives
    collectives = {
        "allreduce": ([input_tensor], AllreduceOptions()),
        "allgather": (output_tensors, [input_tensor], AllgatherOptions()),
        "broadcast": (tensor_list, BroadcastOptions()),
        "broadcast_one": (input_tensor, 0),
    }
    works: Dict[str, dist._Work] = {}
    for coll_str, args in collectives.items():
        coll = getattr(pg, coll_str)
        work = coll(*args)
        works[coll_str] = work
        work.wait()
        fut = work.get_future()
        fut.wait()

        # Check that all tensor arguments have the expected shapes and dtypes
        check_tensors(args)

    print(works)
    return works


class ProcessGroupTest(TestCase):
    def test_gloo(self) -> None:
        store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )

        store_addr = f"localhost:{store.port}/prefix"
        pg = ProcessGroupGloo()
        pg.configure(store_addr, 0, 1)

        self.assertEqual(pg.size(), 1)

        _test_pg(pg)

        m = nn.Linear(3, 4)
        m = torch.nn.parallel.DistributedDataParallel(m, process_group=pg)
        m(torch.rand(2, 3))

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_nccl(self) -> None:
        store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )
        device = "cuda"

        store_addr = f"localhost:{store.port}/prefix"
        pg = ProcessGroupNCCL()
        pg.configure(store_addr, 0, 1)

        self.assertEqual(pg.size(), 1)

        _test_pg(pg, torch.tensor([2], device=device))

        m = nn.Linear(3, 4).to(device)
        m = torch.nn.parallel.DistributedDataParallel(m, process_group=pg)
        m(torch.rand(2, 3, device=device))

        # reconfigure
        store_addr = f"localhost:{store.port}/prefix2"
        pg.configure(store_addr, 0, 1)

        _test_pg(pg, torch.tensor([2], device=device))

        torch.cuda.synchronize()

    def test_baby_gloo(self) -> None:
        store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )

        store_addr = f"localhost:{store.port}/prefix"

        a = ProcessGroupBabyGloo()
        b = ProcessGroupBabyGloo()

        a.configure(store_addr, 0, 2)
        b.configure(store_addr, 1, 2)

        self.assertEqual(a.size(), 2)

        at = torch.tensor([1])
        bt = torch.tensor([2])

        a_work = a.allreduce([at], ReduceOp.SUM)
        b_work = b.allreduce([bt], ReduceOp.SUM)

        a_work.wait()
        fut = b_work.get_future()

        fut.wait()

        torch.testing.assert_close(at, bt)

    def test_dummy(self) -> None:
        pg = ProcessGroupDummy(0, 1)
        m = nn.Linear(3, 4)
        m = torch.nn.parallel.DistributedDataParallel(m, process_group=pg)
        m(torch.rand(2, 3))

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @skipUnless(torch.cuda.device_count() >= 2, "need two CUDA devices")
    def test_baby_nccl_2gpu(self) -> None:
        store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )

        store_addr: str = f"localhost:{store.port}/prefix"

        def run(rank: int) -> Tuple[torch.Tensor, Work]:
            a = ProcessGroupBabyNCCL()
            a.configure(store_addr, rank, 2)

            self.assertEqual(a.size(), 2)

            at = torch.tensor([rank + 1], device=f"cuda:{rank}")

            a_work = a.allreduce([at], ReduceOp.SUM)
            return at, a_work

        with ThreadPoolExecutor(max_workers=2) as executor:
            a_fut = executor.submit(run, 0)
            b_fut = executor.submit(run, 1)

        at, a_work = a_fut.result()
        bt, b_work = b_fut.result()

        a_work.wait()
        b_work.get_future().wait()

        torch.testing.assert_close(at.cpu(), bt.cpu())

    def test_device_mesh(self) -> None:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(0)
        os.environ["RANK"] = str(0)
        os.environ["WORLD_SIZE"] = str(1)

        mesh_1d = init_device_mesh("cpu", mesh_shape=(1,), mesh_dim_names=("tp",))

        store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )
        store_addr = f"localhost:{store.port}/prefix"

        pg = ProcessGroupGloo()
        pg.register("test_device_mesh")
        pg.configure(store_addr, 0, 1)

        mesh_2d = extend_device_mesh(mesh_1d, pg)
        assert mesh_2d.ndim == 2

        pg.unregister()

    def test_functional_collectives(self) -> None:
        dummy_init_pg()

        store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )
        store_addr = f"localhost:{store.port}/prefix"

        pg = ProcessGroupGloo().register("test_func_col")
        pg.configure(store_addr, 0, 1)

        self.assertEqual(pg.group_name, str(dist.get_pg_count() - 1))

        self.assertIs(_resolve_process_group(pg.group_name), pg)

        try:
            t = torch.zeros(10)
            _functional_collectives.all_reduce(t, "sum", pg).wait()
        finally:
            pg.unregister()

    def test_process_group_wrapper(self) -> None:
        pg = ProcessGroupDummy(0, 1)
        wrapper = ProcessGroupWrapper(pg)
        self.assertIs(wrapper.parent, pg)

        wrapper.configure("addr", 0, 1)
        self.assertEqual(pg.configure_count, 1)

        self.assertEqual(repr(wrapper), "ProcessGroupWrapper(pg=ProcessGroupDummy())")

    def test_error_swallowing_process_group_wrapper(self) -> None:
        pg = ProcessGroupDummy(0, 1)
        wrapper = ErrorSwallowingProcessGroupWrapper(pg)
        self.assertIs(wrapper.parent, pg)

        works = _test_pg(wrapper)
        self.assertIsInstance(list(works.values())[0], _ErrorSwallowingWork)

        err = RuntimeError("test")
        wrapper.report_error(err)
        self.assertEqual(wrapper.error(), err)

        works = _test_pg(wrapper)
        for work in works.values():
            self.assertIsInstance(work, _DummyWork)

    def test_managed_process_group(self) -> None:
        manager = Mock(spec=Manager)
        manager.errored.return_value = None
        manager._pg = ProcessGroupDummy(0, 1)
        pg = ManagedProcessGroup(manager)
        manager.num_participants.return_value = 123

        self.assertEqual(pg.size(), 123)

        works = _test_pg(pg)
        self.assertIsInstance(list(works.values())[0], _ManagedWork)

        self.assertEqual(manager.report_error.call_count, 0)
        self.assertEqual(manager.wrap_future.call_count, 1)
