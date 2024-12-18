# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict
from unittest import TestCase
from unittest.mock import create_autospec

import torch
from torch import nn, optim

from torchft.local_sgd import LocalSGD
from torchft.manager import Manager


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(3, 4),
            nn.ReLU(),
            nn.Linear(4, 5),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def _params_dict(m: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: p.data for name, p in m.named_parameters()}


def _copy_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: value.clone().detach() for name, value in state_dict.items()}


class LocalSGDTest(TestCase):
    def test_local_sgd_healthy(self) -> None:
        base_m = SimpleModel()
        optimizer = optim.SGD(base_m.parameters())
        manager = create_autospec(Manager)

        m = LocalSGD(manager, base_m, optimizer, sync_every=2)
        self.assertEqual(m._local_step, 0)

        torch.testing.assert_close(m._backup_parameters, _params_dict(base_m))

        inp = torch.rand(2, 3)

        loss = m(inp).mean()
        loss.backward()
        optimizer.step()

        self.assertEqual(m._local_step, 1)
        self.assertEqual(manager.start_quorum.call_count, 1)

        loss = m(inp).mean()
        loss.backward()
        optimizer.step()

        manager.should_commit.return_value = True
        self.assertEqual(m._local_step, 0)

        torch.testing.assert_close(m._backup_parameters, _params_dict(base_m))
        self.assertEqual(manager.should_commit.call_count, 1)
        self.assertEqual(manager.allreduce.call_count, 4)

    def test_local_sgd_recovery(self) -> None:
        base_m = SimpleModel()
        optimizer = optim.SGD(base_m.parameters())
        manager = create_autospec(Manager)

        m = LocalSGD(manager, base_m, optimizer, sync_every=2)

        torch.testing.assert_close(m._backup_parameters, _params_dict(base_m))
        og_state_dict = _copy_state_dict(base_m.state_dict())

        inp = torch.rand(2, 3)

        loss = m(inp).mean()
        loss.backward()
        optimizer.step()

        self.assertEqual(m._local_step, 1)

        state_dict = m.state_dict()
        torch.testing.assert_close(state_dict, m._backup_parameters)
        torch.testing.assert_close(state_dict, og_state_dict)

        m.load_state_dict(state_dict)
        torch.testing.assert_close(_params_dict(base_m), state_dict)
        torch.testing.assert_close(m._backup_parameters, _params_dict(base_m))
