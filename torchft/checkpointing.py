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

import torch

from torchft.http import _IPv6HTTPServer

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")


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

                    torch.save(sd, self.wfile)

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
        return torch.load(reader, weights_only=True)

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
