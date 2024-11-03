# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase, skipUnless
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.distributed import TCPStore, ReduceOp
import torch.distributed as dist
from torch import nn

from torchft.process_group import (
    ProcessGroupBabyGloo,
    ProcessGroupBabyNCCL,
    ProcessGroupGloo,
    ProcessGroupNCCL,
    ProcessGroupDummy,
    ProcessGroup,
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

    @skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_baby_nccl(self) -> None:
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
