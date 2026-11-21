# SPDX-License-Identifier: Apache-2.0
"""SGLang's binding to the T-Head PPU monokernel plugin (sgl-t-head-mega). """

from __future__ import annotations

from typing import Any, Optional

import torch

from sglang.kernels.plugin_kernels.t_head.spec import BACKEND, FLAG

# same tag SAIL presence detection uses (platforms.sail, utils.is_sail)
_ZW_TAG = "ZW"


def selected_monokernel(cfg: Any) -> Optional[tuple[str, str]]:
    if getattr(cfg, "enable_t_head_monokernel", False):
        return BACKEND, FLAG
    return None


def device_isa() -> Optional[str]:
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name()
    return name if _ZW_TAG in name else None
