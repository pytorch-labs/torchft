# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import urllib.error
from unittest import TestCase
from unittest.mock import MagicMock
from io import BytesIO
import torch
from typing import Tuple
from checkpointing import CheckpointServer, TensorMetadata, write_state_dict, read_state_dict


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

    def setUp(self):
        self.file = BytesIO()

    def test_scalar_tensor(self):
        tensor = torch.tensor(42, dtype=torch.int32)
        state_dict = {'scalar': tensor}
        write_state_dict(state_dict, self.file)
        self.file.seek(0)

        result = read_state_dict(self.file)
        self.assertTrue(torch.equal(result['scalar'], tensor))

    def test_strided_tensor(self):
        base_tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        strided_tensor = base_tensor[::2, ::2]
        state_dict = {'strided': strided_tensor}
        write_state_dict(state_dict, self.file)
        self.file.seek(0)

        result = read_state_dict(self.file)
        self.assertTrue(torch.equal(result['strided'], strided_tensor))

    def test_tensor_with_offset(self):
        base_tensor = torch.arange(10, dtype=torch.float64)
        offset_tensor = base_tensor[2:]
        state_dict = {'offset': offset_tensor}
        write_state_dict(state_dict, self.file)
        self.file.seek(0)

        result = read_state_dict(self.file)
        self.assertTrue(torch.equal(result['offset'], offset_tensor))

    def test_nested_tensors(self):
        tensor1 = torch.tensor([1, 2, 3], dtype=torch.int32)
        tensor2 = torch.tensor([[1.5, 2.5], [3.5, 4.5]], dtype=torch.float64)
        state_dict = {'nested': {'tensor1': tensor1, 'tensor2': tensor2}}
        write_state_dict(state_dict, self.file)
        self.file.seek(0)

        result = read_state_dict(self.file)
        self.assertTrue(torch.equal(result['nested']['tensor1'], tensor1))
        self.assertTrue(torch.equal(result['nested']['tensor2'], tensor2))

    def test_various_data_types(self):
        tensor_float32 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        tensor_int16 = torch.tensor([1, 2, 3], dtype=torch.int16)
        tensor_bool = torch.tensor([True, False, True], dtype=torch.bool)
        state_dict = {
            'float32': tensor_float32,
            'int16': tensor_int16,
            'bool': tensor_bool,
        }
        write_state_dict(state_dict, self.file)
        self.file.seek(0)

        result = read_state_dict(self.file)
        self.assertTrue(torch.equal(result['float32'], tensor_float32))
        self.assertTrue(torch.equal(result['int16'], tensor_int16))
        self.assertTrue(torch.equal(result['bool'], tensor_bool))


if __name__ == '__main__':
    unittest.main()
