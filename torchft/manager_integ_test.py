from concurrent.futures import ThreadPoolExecutor, as_completed
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


def train_loop(replica_id: int, lighthouse_address: str) -> None:
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
    m = DistributedDataParallel(manager, MyModel())
    optimizer = OptimizerWrapper(manager, optim.Adam(m.parameters()))
    criterion = nn.CrossEntropyLoss()

    while True:
        inputs = torch.rand(2, 3)
        labels = torch.randint(4, (2,))

        optimizer.zero_grad()
        out = m(inputs)
        loss = criterion(out, labels)

        loss.backward()
        optimizer.step()

        # TODO: assert weights are equal across replicas

        if manager.current_step() >= 5:
            break

    manager.shutdown()


class ManagerIntegTest(TestCase):
    def test_ddp(self):
        lighthouse = Lighthouse(
            bind="[::]:0",
            min_replicas=2,
        )
        num_replicas = 2
        futures = []

        with ThreadPoolExecutor(max_workers=num_replicas) as executor:
            for replica_id in range(num_replicas):
                futures.append(
                    executor.submit(train_loop, replica_id, lighthouse.address())
                )

        for fut in as_completed(futures):
            fut.result()

        lighthouse.shutdown()
