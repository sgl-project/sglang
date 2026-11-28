# SPDX-License-Identifier: Apache-2.0
"""Validator: is a snapshot-loaded rank equal to the owner's raw load?

It shares no logic with load-diff and never imports it: a check made with
load-diff's own walk would only show that install repeated export.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def record_construction(model: nn.Module) -> Any:
    """What ``model`` holds right after construction, by the validator's own count."""
    raise NotImplementedError("the validator lands with sgl-project/sglang#50943 E1")


def attest(
    *, model: nn.Module, storages: dict[str, torch.Tensor], constructed: Any
) -> bytes:
    """Owner: describe the raw-loaded model and the storages it publishes."""
    raise NotImplementedError("the validator lands with sgl-project/sglang#50943 E1")


def validate(
    *,
    model: nn.Module,
    storages: dict[str, torch.Tensor],
    attestation: bytes,
    constructed: Any,
) -> None:
    """Reader: raise unless the installed model matches the owner's attestation."""
    raise NotImplementedError("the validator lands with sgl-project/sglang#50943 E1")
