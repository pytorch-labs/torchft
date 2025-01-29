# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase
from unittest.mock import create_autospec, MagicMock

import torch
from torch.nn import Linear
from torch.optim import AdamW

from torchft.manager import Manager
from torchft.optim import OptimizerWrapper


class TestOptim(TestCase):
    def test_optimizer_wrapper(self) -> None:
        manager = create_autospec(Manager)

        m = Linear(3, 4)
        base_optim = AdamW(m.parameters())
        optim = OptimizerWrapper(manager, base_optim)
        optim.add_param_group(
            {
                "params": [],
                "lr": 1e-4,
            }
        )

        # test state_dict handling
        optim.load_state_dict(optim.state_dict())

        optim.zero_grad()
        self.assertEqual(manager.start_quorum.call_count, 1)

        b = torch.rand(3)
        m(b).sum().backward()

        manager.should_commit.return_value = True
        optim.step()
        manager.should_commit.return_value = False
        optim.step()
        self.assertEqual(len(optim.param_groups), 2)
        self.assertEqual(optim.param_groups[1]["lr"], 1e-4)
        self.assertEqual(optim.param_groups[1]["params"], [])
        self.assertEqual(len(optim.state), len(list(m.parameters())))

        self.assertEqual(manager.should_commit.call_count, 2)
