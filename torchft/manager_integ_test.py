from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from typing import Set, Tuple
from unittest import TestCase

import torch
import torch.distributed as dist
from torch import nn, optim

from torchft.ddp import DistributedDataParallel
from torchft.manager import Manager
from torchft.optim import OptimizerWrapper
from torchft.process_group import ProcessGroupGloo
from torchft.torchft import Lighthouse


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class InjectedFailure(Exception):
    pass


class FailureInjector:
    def __init__(self) -> None:
        self._failures: Set[int] = set()
        self.count = 0

    def fail_at(self, step: int) -> "FailureInjector":
        self._failures.add(step)
        return self

    def check(self, step: int) -> None:
        if step in self._failures:
            self.count += 1
            self._failures.remove(step)
            print(f"injecting failure {step=}")
            raise InjectedFailure(f"injected failure {step=}")


def worker_manager(
    replica_id: int,
    lighthouse_address: str,
    failure_injector: FailureInjector,
    attempts: int = 3,
) -> None:
    for i in range(attempts):
        try:
            print(f"starting worker {replica_id} attempt {i}")
            return train_loop(
                replica_id, lighthouse_address, failure_injector=failure_injector
            )
        except InjectedFailure as e:
            print("got injected failure", i, e)
            if i == attempts - 1:
                raise
            continue


def train_loop(
    replica_id: int, lighthouse_address: str, failure_injector: FailureInjector
) -> None:
    with ExitStack() as stack:
        store = dist.TCPStore(
            host_name="localhost",
            port=0,
            is_master=True,
            wait_for_workers=False,
        )

        def load_state_dict(state_dict):
            m.load_state_dict(state_dict["model"])
            optimizer.load_state_dict(state_dict["optim"])

        def state_dict():
            return {
                "model": m.state_dict(),
                "optim": optimizer.state_dict(),
            }

        pg = ProcessGroupGloo()
        manager = Manager(
            pg=pg,
            min_replica_size=2,
            load_state_dict=load_state_dict,
            state_dict=state_dict,
            replica_id=str(replica_id),
            store_addr="localhost",
            store_port=store.port,
            rank=0,
            world_size=1,
            lighthouse_addr=lighthouse_address,
            port=19530 + replica_id,
        )
        stack.callback(manager.shutdown)

        m = DistributedDataParallel(manager, MyModel())
        optimizer = OptimizerWrapper(manager, optim.Adam(m.parameters()))
        criterion = nn.CrossEntropyLoss()

        while True:
            print(f"worker {replica_id} starting step {manager.current_step()}")
            inputs = torch.rand(2, 3)
            labels = torch.randint(4, (2,))

            optimizer.zero_grad()
            out = m(inputs)
            loss = criterion(out, labels)

            loss.backward()
            optimizer.step()

            if manager.current_step() >= 5:
                # return state_dict so we can check consistency
                return state_dict()

            failure_injector.check(manager.current_step())


class ManagerIntegTest(TestCase):
    def test_ddp_healthy(self):
        lighthouse = Lighthouse(
            bind="[::]:0",
            min_replicas=2,
        )
        num_replicas = 2
        futures = []

        with ThreadPoolExecutor(max_workers=num_replicas) as executor:
            for replica_id in range(num_replicas):
                failure_injector = FailureInjector()
                futures.append(
                    executor.submit(
                        worker_manager,
                        replica_id,
                        lighthouse.address(),
                        failure_injector=failure_injector,
                    )
                )

        state_dicts = []

        for fut in as_completed(futures):
            state_dicts.append(fut.result())

        lighthouse.shutdown()

        for state_dict in state_dicts:
            torch.testing.assert_close(state_dict, state_dicts[0])

    def test_ddp_recovery(self):
        lighthouse = Lighthouse(
            bind="[::]:0",
            min_replicas=2,
        )
        num_replicas = 2
        futures = []

        failure_injectors = [
            FailureInjector(),
            FailureInjector().fail_at(2),
        ]

        with ThreadPoolExecutor(max_workers=num_replicas) as executor:
            for replica_id, failure_injector in zip(
                range(num_replicas), failure_injectors
            ):
                futures.append(
                    executor.submit(
                        worker_manager,
                        replica_id,
                        lighthouse.address(),
                        failure_injector=failure_injector,
                    )
                )

        state_dicts = []

        for fut in as_completed(futures):
            state_dicts.append(fut.result())

        lighthouse.shutdown()

        for state_dict in state_dicts:
            torch.testing.assert_close(state_dict, state_dicts[0])

        self.assertEqual(failure_injectors[1].count, 1)
