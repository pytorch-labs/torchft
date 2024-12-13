import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, nullcontext
from typing import Dict, List, Set, Tuple
from unittest import TestCase

import torch
import torch.distributed as dist

# pyre-fixme[21]: missing module
from parameterized import parameterized
from torch import nn, optim

from torchft.ddp import DistributedDataParallel
from torchft.manager import Manager
from torchft.optim import OptimizerWrapper
from torchft.process_group import ProcessGroupGloo
from torchft.torchft import Lighthouse

logger: logging.Logger = logging.getLogger(__name__)


class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class InjectedFailure(Exception):
    pass


class FailureInjector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._failures: Set[Tuple[int, int]] = set()
        self.count = 0

    def fail_at(self, rank: int, step: int) -> "FailureInjector":
        with self._lock:
            self._failures.add((rank, step))
            return self

    def check(self, rank: int, step: int) -> None:
        with self._lock:
            key = (rank, step)
            if key in self._failures:
                self.count += 1
                self._failures.remove(key)
                print(f"injecting failure {rank=} {step=}")
                raise InjectedFailure(f"injected failure {rank=} {step=}")


def replica_main(
    replica_id: int,
    lighthouse_address: str,
    failure_injector: FailureInjector,
    world_size: int,
    manager_args: Dict[str, object],
) -> List[Dict[str, Dict[str, object]]]:
    store = dist.TCPStore(
        host_name="localhost",
        port=0,
        is_master=True,
        wait_for_workers=False,
    )

    with ThreadPoolExecutor(
        max_workers=world_size, thread_name_prefix=f"replica{replica_id}"
    ) as executor:
        futures = []
        for rank in range(world_size):
            futures.append(
                executor.submit(
                    train_loop,
                    replica_id,
                    lighthouse_address,
                    failure_injector=failure_injector,
                    rank=rank,
                    world_size=world_size,
                    store_port=store.port,
                    manager_args=manager_args,
                )
            )

        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                logger.exception(f"worker threw exception: {e}")
                raise

        return [fut.result() for fut in futures]


def worker_manager(
    replica_id: int,
    lighthouse_address: str,
    failure_injector: FailureInjector,
    manager_args: Dict[str, object],
    attempts: int = 3,
    world_size: int = 1,
) -> List[Dict[str, Dict[str, object]]]:

    for i in range(attempts):
        try:
            print(f"starting replica group {replica_id=} {world_size=} attempt {i}")
            return replica_main(
                replica_id,
                lighthouse_address,
                failure_injector=failure_injector,
                world_size=world_size,
                manager_args=manager_args,
            )
        except InjectedFailure as e:
            print("got injected failure", i, e)
            if i == attempts - 1:
                raise
            continue

    raise RuntimeError("ran out of attempts")


def train_loop(
    replica_id: int,
    lighthouse_address: str,
    failure_injector: FailureInjector,
    rank: int,
    world_size: int,
    store_port: int,
    manager_args: Dict[str, object],
) -> Dict[str, Dict[str, object]]:
    with ExitStack() as stack:

        def load_state_dict(state_dict: Dict[str, Dict[str, object]]) -> None:
            m.load_state_dict(state_dict["model"])
            optimizer.load_state_dict(state_dict["optim"])

        def state_dict() -> Dict[str, Dict[str, object]]:
            return {
                "model": m.state_dict(),
                "optim": optimizer.state_dict(),
            }

        print(f"worker {replica_id=} {rank=} {world_size=} starting")

        pg = ProcessGroupGloo()
        manager = Manager(
            pg=pg,
            min_replica_size=2,
            load_state_dict=load_state_dict,
            state_dict=state_dict,
            replica_id=str(replica_id),
            store_addr="localhost",
            store_port=store_port,
            rank=rank,
            world_size=world_size,
            lighthouse_addr=lighthouse_address,
            port=19530 + replica_id,
            # pyre-fixme[6]: Incompatible parameter type
            **manager_args,
        )
        stack.callback(manager.shutdown)

        m: nn.Module = DistributedDataParallel(manager, MyModel())
        optimizer: optim.Optimizer = OptimizerWrapper(
            manager, optim.Adam(m.parameters())
        )
        criterion = nn.CrossEntropyLoss()

        while True:
            inputs = torch.rand(2, 3)
            labels = torch.randint(4, (2,))

            optimizer.zero_grad()
            out = m(inputs)
            loss = criterion(out, labels)

            loss.backward()

            optimizer.step()

            if manager.current_step() >= 4:
                break

            failure_injector.check(rank, manager.current_step())

        # return state_dict so we can check consistency
        return state_dict()


class ManagerIntegTest(TestCase):
    def test_ddp_healthy(self) -> None:
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
                        manager_args={},
                    )
                )

        state_dicts = []

        for fut in as_completed(futures):
            state_dicts.append(fut.result())

        lighthouse.shutdown()

        for state_dict in state_dicts:
            torch.testing.assert_close(state_dict, state_dicts[0])

    # pyre-fixme[56]: couldn't infer type of decorator
    @parameterized.expand(
        [
            (
                "async_quorum",
                True,
            ),
            (
                "sync_quorum",
                False,
            ),
        ]
    )
    def test_ddp_recovery(self, name: str, use_async_quorum: bool) -> None:
        lighthouse = Lighthouse(
            bind="[::]:0",
            min_replicas=2,
        )
        num_replicas = 2
        futures = []

        failure_injectors = [
            FailureInjector(),
            FailureInjector().fail_at(0, 2),
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
                        manager_args={
                            "use_async_quorum": use_async_quorum,
                        },
                    )
                )

            state_dicts = []

            for fut in as_completed(futures):
                state_dicts.append(fut.result())

        lighthouse.shutdown()

        for state_dict in state_dicts:
            torch.testing.assert_close(state_dict, state_dicts[0])

        self.assertEqual(failure_injectors[1].count, 1)

    def test_ddp_recovery_multi_rank(self) -> None:
        lighthouse = Lighthouse(
            bind="[::]:0",
            min_replicas=2,
        )
        num_replicas = 2
        world_size = 2
        futures = []

        failure_injectors = [
            FailureInjector(),
            FailureInjector().fail_at(0, 2).fail_at(1, 2),
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
                        world_size=world_size,
                        manager_args={},
                    )
                )

            state_dicts = []

            for fut in as_completed(futures):
                try:
                    state_dicts.append(fut.result())
                except Exception as e:
                    print(e)
                    raise

        lighthouse.shutdown()

        for state_dict in state_dicts:
            torch.testing.assert_close(state_dict, state_dicts[0])
