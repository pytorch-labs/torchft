# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from concurrent.futures import ThreadPoolExecutor
from unittest import skipUnless, TestCase

import torch
import torch.distributed as dist
from torch import nn
from torch._C._distributed_c10d import _resolve_process_group
from torch.distributed import _functional_collectives, ReduceOp, TCPStore
from torch.distributed.device_mesh import _mesh_resources, init_device_mesh

from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
)
from torch.testing._internal.common_utils import FILE_SCHEMA

from torchft.process_group import (
    extend_device_mesh,
    ProcessGroup,
    ProcessGroupBabyGloo,
    ProcessGroupBabyNCCL,
    ProcessGroupDummy,
    ProcessGroupGloo,
    ProcessGroupNCCL,
)


def dummy_init_pg() -> None:
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo", rank=0, world_size=1, store=dist.HashStore()
        )


class ProcessGroupTest(TestCase):
    def test_gloo(self) -> None:
        store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )

        store_addr = f"localhost:{store.port}/prefix"
        pg = ProcessGroupGloo()
        pg.configure(store_addr, 0, 1)

        self.assertEqual(pg.size(), 1)

        at = torch.tensor([2])

        a_work = pg.allreduce([at], ReduceOp.SUM)
        a_work.wait()
        a_work.get_future().wait()

        m = nn.Linear(3, 4)
        m = torch.nn.parallel.DistributedDataParallel(m, process_group=pg)
        m(torch.rand(2, 3))

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

        at = torch.tensor([2], device=device)
        a_work = pg.allreduce([at], ReduceOp.SUM)
        a_work.wait()
        a_work.get_future().wait()

        m = nn.Linear(3, 4).to(device)
        m = torch.nn.parallel.DistributedDataParallel(m, process_group=pg)
        m(torch.rand(2, 3, device=device))

        # reconfigure
        store_addr = f"localhost:{store.port}/prefix2"
        pg.configure(store_addr, 0, 1)

        at = torch.tensor([2], device=device)
        a_work = pg.allreduce([at], ReduceOp.SUM)
        a_work.wait()

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

    @skipUnless(torch.cuda.device_count() >= 2, "need two CUDA devices")
    def test_baby_nccl_2gpu(self) -> None:
        store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )

        store_addr = f"localhost:{store.port}/prefix"

        def run(rank: int) -> None:
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


class ProcessGroupMPTest(MultiProcessTestCase):
    @property
    def world_size(self):
        return 4

    def setUp(self):
        super().setUp()
        # Set TORCH_NCCL_DESYNC_DEBUG=0 to disable the NCCL `workCleanupLoop()`,
        # which can cause unit test flakiness:
        # https://github.com/pytorch/pytorch/issues/90848
        os.environ["TORCH_NCCL_DESYNC_DEBUG"] = "0"
        self._spawn_processes()

    def test_init_device_mesh(self) -> None:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(12345)
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(4)

        def ft_init_device_mesh(device, mesh_shape, mesh_dim_names, replicate_dim):
            if device == "cpu":
                pg = ProcessGroupGloo()
            elif device == "cuda":
                pg = ProcessGroupNCCL()
            else:
                raise ValueError()

            # We have to use MultiProcessTestCase, otherwise c10d will complain
            # the same backend has been registered.
            backend_name = pg._register(mesh_dim_names[replicate_dim])
            # This currently doesn't work with NCCL as DeviceMesh will ignore
            # `_set_mesh_dim_group_options()` and just use `split_group()`.
            # We will need to change DeviceMesh to use `new_group()` instead of
            # `split_group()` when backend is not None.
            _mesh_resources._set_mesh_dim_group_options(
                replicate_dim, backend_name, None
            )
            device_mesh = init_device_mesh(
                device, mesh_shape=mesh_shape, mesh_dim_names=mesh_dim_names
            )
            # We need an API to clear the mesh_dim_group_options because it will
            # affect the following flatten() API.
            return device_mesh

        device_mesh = ft_init_device_mesh(
            "cpu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"), replicate_dim=0
        )

        store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )
        store_addr = f"localhost:{store.port}/prefix"
        pg = device_mesh.get_group("dp")
        pg.configure(store_addr, 0, 1)
        pg.unregister()
