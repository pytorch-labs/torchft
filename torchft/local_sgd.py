# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
LocalSGD
=========

This module implements a fault tolerant version of LocalSGD and related methods.
"""

from typing import Any, Dict, List, Mapping, Optional

import torch
from torch import nn, optim

from torchft.manager import Manager


class LocalSGD(nn.Module):
    """
    LocalSGD is a model wrapper similar to DistributedDataParallel that
    implements the algorithm described in https://arxiv.org/pdf/1805.09767

    This will synchronize the model parameters periodically in a fault tolerant
    way using a torchft Manager. The allreduce on the parameters will happen
    every sync_every steps after the optimizer.step call.

    To implement safe and fault tolerant, this requires a backup copy of the
    weights. By default these are stored in CPU memory. If any error occurs
    during the LocalSGD step, the step will be discarded and the model
    parameters will reset back to the last time LocalSGD synchronized.

    The backup weights could be eliminated by relaxing the guarantee of exactly
    `sync_every` steps but that would diverge from the LocalSGD algorithm.
    DiLoCo also needs this backup copy to compute the delta.

    The torchft quorum is computed at the beginning of ``sync_every`` steps. If
    any error occurs, or a worker fails between syncs, ``sync_every`` steps will be
    discarded and a new quorum will be computed on the next step.

    If running in async mode, on a joining worker the first ``sync_every`` steps
    will discarded as the model will be recovering during that period. When
    using sync mode, the checkpoint will be restored prior to the first step.

    TODO: add a way via Manager to detect workers failing early for shrink only
    TODO: add DiLoCo support
    """

    def __init__(
        self,
        manager: Manager,
        model: nn.Module,
        optimizer: optim.Optimizer,
        sync_every: int,
        backup_device: Optional[torch.device] = None,
        pin_memory: bool = True,
    ) -> None:
        """
        Args:
            manager: The manager to use.
            model: The model to wrap.
            optimizer: The optimizer used by the model.
            sync_every: How often to sync the model weights.
            backup_device: The device to store the backup of the model parameters on. (default cpu)
            pin_memory: Whether to pin the memory used for the backup of the model parameters.
        """
        super().__init__()

        self._manager = manager
        self._model = model
        self._local_step = 0
        self._started_step = False
        self._sync_every = sync_every
        assert sync_every >= 1, "sync_every must be greater than or equal to 1"

        device = backup_device or torch.device("cpu")

        self._backup_parameters: Dict[str, torch.Tensor] = {}

        for name, p in self._model.named_parameters():
            t = torch.empty(*tuple(p.shape), dtype=p.dtype, device=device)
            if (
                pin_memory
                and t.device == torch.device("cpu")
                and torch.cuda.is_available()
            ):
                t = t.pin_memory()
            self._backup_parameters[name] = t

        # Need to copy the parameters to the host to be safe if we are on the first step.
        self._save_parameters()

        optimizer.register_step_post_hook(self._step_post_hook)

    def _save_parameters(self) -> None:
        # TODO: consider running copy on a separate stream
        for name, p in self._model.named_parameters():
            self._backup_parameters[name].copy_(p.data, non_blocking=True)

    def _restore_parameters(self) -> None:
        # TODO: consider running copy on a separate stream
        for name, p in self._model.named_parameters():
            p.data.copy_(self._backup_parameters[name], non_blocking=True)

    # pyre-fixme[14]: support state_dict args
    def state_dict(self) -> Dict[str, object]:
        """
        state_dict returns the state_dict from the last time LocalSGD
        synchronized and not the current weights.
        """
        state_dict = self._model.state_dict()
        for name, p in self._backup_parameters.items():
            assert name in state_dict
            state_dict[name] = p
        return state_dict

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> None:
        """
        Loads the state dict to the model and the backup parameters.

        This must be called while the model weights aren't being modified to
        avoid corrupting the backup weights.
        """
        self._model.load_state_dict(state_dict, strict=strict, assign=assign)
        self._save_parameters()

    def forward(self, *args: object, **kwargs: object) -> object:
        """
        Run the model parameters.

        This should be called before the optimizer step.

        This will start the quorum and save the parameters if this is the first step.
        """
        if self._local_step == 0:
            self._manager.start_quorum()

        self._started_step = True

        return self._model.forward(*args, **kwargs)

    def _step_post_hook(
        self, _optim: optim.Optimizer, _args: List[object], _kwargs: Dict[str, object]
    ) -> None:
        """
        This hook is registered on the optimizer and is called after the optimizer step.

        This will call the allreduce on the model weights every sync_every steps.
        If any errors occur it will restore to the weights from the previous sync.

        ``forward`` must be called before this function.
        """
        assert self._started_step, "forward must be called before step"
        self._started_step = False

        self._local_step += 1

        if self._local_step >= self._sync_every:
            self._local_step = 0
            self._average()

            if self._manager.should_commit():
                # save the parameters so we can restore from them later if necessary.
                self._save_parameters()
            else:
                # commit failed, restore from the backup parameters
                self._restore_parameters()

    def _average(self) -> None:
        # TODO: do we need to broadcast buffers like DDP does?

        works = []

        for p in self._model.parameters():
            # TODO: bucketize parameters
            works.append(self._manager.allreduce(p.data.detach()))

        for work in works:
            work.wait()
