# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import urllib.error
from unittest import TestCase
from unittest.mock import MagicMock

from torchft.checkpointing import CheckpointServer


class TestCheckpointing(TestCase):
    def test_checkpoint_server(self) -> None:
        expected = {"state": "dict"}
        state_dict_fn = MagicMock()
        state_dict_fn.return_value = expected
        server = CheckpointServer(state_dict=state_dict_fn)

        server.disallow_checkpoint()
        server.allow_checkpoint(1234)

        addr = server.address()

        out = CheckpointServer.load_from_address(addr)
        self.assertEqual(out, expected)

        # test mismatch case
        server.allow_checkpoint(2345)

        with self.assertRaisesRegex(urllib.error.HTTPError, r"Error 400"):
            CheckpointServer.load_from_address(addr)

        server.shutdown()
