# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import urllib.error
from datetime import timedelta
from unittest import TestCase
from unittest.mock import MagicMock

from torchft.checkpointing import CheckpointServer


class TestCheckpointing(TestCase):
    def test_checkpoint_server(self) -> None:
        expected = {"state": "dict"}
        state_dict_fn = MagicMock()
        state_dict_fn.return_value = expected
        server = CheckpointServer(
            timeout=timedelta(seconds=10),
        )

        server.send_checkpoint(
            dst_ranks=[],
            step=1234,
            state_dict=expected,
            timeout=timedelta(seconds=10),
        )

        metadata = server.metadata()

        out = server.recv_checkpoint(
            src_rank=0, metadata=metadata, step=1234, timeout=timedelta(seconds=10)
        )
        self.assertEqual(out, expected)

        # test timeout
        with self.assertRaisesRegex(urllib.error.URLError, r"urlopen error"):
            server.recv_checkpoint(
                src_rank=0, metadata=metadata, step=1234, timeout=timedelta(seconds=0.0)
            )

        # test mismatch case
        server.send_checkpoint(
            dst_ranks=[],
            step=2345,
            state_dict=expected,
            timeout=timedelta(seconds=10),
        )

        with self.assertRaisesRegex(urllib.error.HTTPError, r"Error 400"):
            server.recv_checkpoint(
                src_rank=0, metadata=metadata, step=1234, timeout=timedelta(seconds=10)
            )

        server.shutdown()
