# SPDX-License-Identifier: Apache-2.0
"""Load-diff: what loading produced on an owner, and installing exactly that.

It shares nothing with the validator but the schema.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from sglang.srt.model_loader.snapshot.schema import SnapshotManifest


def record_construction(model: nn.Module) -> Any:
    """Owner: what ``model`` holds right after construction, before weights load."""
    raise NotImplementedError("load-diff lands with sgl-project/sglang#50943 D1")


def export(
    *, model: nn.Module, constructed: Any
) -> tuple[dict[str, torch.Tensor], bytes]:
    """Owner: every storage the loaded model holds, and the manifest's ``load_diff``."""
    raise NotImplementedError("load-diff lands with sgl-project/sglang#50943 D1")


def install(
    *,
    skeleton: nn.Module,
    manifest: SnapshotManifest,
    storages: dict[str, torch.Tensor],
) -> nn.Module:
    """Reader: attach ``storages`` to the meta-built ``skeleton`` and set the values
    loading changed on the owner. Never post-processes."""
    raise NotImplementedError("load-diff lands with sgl-project/sglang#50943 D1")
