# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Checkpointing
==============

This module implements methods for checkpointing and resuming training from a checkpoint.
"""

import io
import logging
import socket
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler
from typing import Callable, Generic, TypeVar
from dataclasses import dataclass
import pickle
from io import BufferedIOBase
from typing import Tuple
import struct

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten
from hashlib import sha256
from torchft.http import _IPv6HTTPServer

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class TensorMetadata:
    nbytes: int
    dtype: torch.dtype
    storage_offset: int
    size: Tuple[int, ...]
    stride: Tuple[int, ...]


def write_state_dict(state_dict: object, f: BufferedIOBase) -> None:
    """
    Write the state_dict to the file-like object.
    """
    values, spec = tree_flatten(state_dict)

    storages = []
    non_tensor_values = []
    for value in values:
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            storages.append(storage)
            non_tensor_values.append(
                TensorMetadata(
                    nbytes=storage.nbytes(),
                    dtype=value.dtype,
                    storage_offset=value.storage_offset(),
                    size=value.size(),
                    stride=value.stride(),
                )
            )
        else:
            non_tensor_values.append(value)

    meta_buf = pickle.dumps((non_tensor_values, spec))
    checksum = sha256(meta_buf).hexdigest()
    total_length = len(meta_buf) + len(checksum)

    f.write(struct.pack("<q", total_length))
    f.write(meta_buf)
    f.write(checksum.encode("utf-8"))


    for storage in storages:
        storage._write_file(f, False, False, 1)


def read_state_dict(f: BufferedIOBase) -> object:
    """
    Read the state_dict from the file-like object.
    """

    total_length = struct.unpack("<q", f.read(8))[0]
    meta_buf = f.read(total_length - 64)
    checksum = f.read(64).decode("utf-8")

    # Verify checksum
    actual_checksum = sha256(meta_buf).hexdigest()
    if checksum != actual_checksum:
        raise ValueError("Checksum mismatch! Data may be corrupted.")
    non_tensor_values, spec = pickle.loads(meta_buf)
    values = []
    for value in non_tensor_values:
        if isinstance(value, TensorMetadata):
            data = f.read(value.nbytes)

            tensor = torch.as_strided(
                torch.frombuffer(data, dtype=value.dtype),
                size=value.size,
                stride=value.stride,
                storage_offset=value.storage_offset,
            )
            values.append(tensor)
        else:
            values.append(value)

    return tree_unflatten(values, spec)

class CheckpointServer(Generic[T]):
    """
    This is an HTTP server that can be used to transfer checkpoints
    between workers.

    This allows for fast recovery of workers by fetching the current weights
    from an existing worker.

    Args:
        state_dict: a callable that returns the state dict to be transferred
    """

    def __init__(self, state_dict: Callable[[], T]) -> None:
        self._checkpoint_lock = threading.Lock()
        self._disallowed = False
        self._step = -1

        ckpt_server = self

        class RequestHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                with ckpt_server._checkpoint_lock:
                    step = ckpt_server._step

                    if self.path != f"/checkpoint/{step}":
                        self.send_response(400)
                        self.send_header("Content-type", "text/plain")
                        self.end_headers()
                        self.err(
                            f"invalid checkpoint requested, serving {step} but got {self.path}"
                        )
                        return

                    self.send_response(200)
                    self.send_header(
                        "Content-type", "tensor"
                    )  # TODO: correct mime type
                    self.end_headers()

                    sd = state_dict()

                    write_state_dict(sd, self.wfile)

            def err(self, msg: str) -> None:
                logger.error(msg)
                self.wfile.write(msg.encode())

        server_address = ("", 0)
        self._server = _IPv6HTTPServer(server_address, RequestHandler)
        logger.info(f"Started CheckpointServer on {self.address()}...")

        self._thread = threading.Thread(
            target=self._serve,
            args=(),
            daemon=True,
        )
        self._thread.start()

    @classmethod
    def load_from_address(cls, address: str) -> T:
        """
        Loads a checkpoint from the given address.

        Args:
            address: the HTTP address to load the checkpoint from
        """
        logger.info(f"fetching checkpoint from {address}")

        with urllib.request.urlopen(address) as f:
            data = f.read()

        reader = io.BytesIO(data)
        state_dict = read_state_dict(reader)
        return state_dict

    def address(self) -> str:
        """
        Returns the HTTP address to fetch a checkpoint from this server at the current step.

        Format: http://host:port/checkpoint/1234

        Returns:
            an HTTP address
        """
        port = self._server.socket.getsockname()[1]
        return f"http://{socket.gethostname()}:{port}/checkpoint/{self._step}"

    def _serve(self) -> None:
        try:
            self._server.serve_forever()
        except Exception as e:
            logger.exception("got exception in checkpoint server")

    def disallow_checkpoint(self) -> None:
        """
        Disallows serving the checkpoint.

        All requests will block until allow_checkpoint is called.
        """
        if not self._disallowed:
            self._disallowed = True
            self._checkpoint_lock.acquire()

    def allow_checkpoint(self, step: int) -> None:
        """
        Allows serving the checkpoint with the specified step number.

        Args:
            step: the step number to serve
        """
        self._step = step

        if self._disallowed:
            self._disallowed = False
            self._checkpoint_lock.release()

    def shutdown(self) -> None:
        """
        Shutdown the server.
        """
        self._server.shutdown()
