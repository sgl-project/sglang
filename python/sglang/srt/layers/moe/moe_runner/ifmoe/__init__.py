"""Register IFMoe kernel with sglang's FusedOpPool.

The kernel activates via: --moe-runner-backend ifmoe.
"""

import logging
import os

from sglang.srt.layers.moe.moe_runner.base import FusedOpPool
from sglang.srt.layers.moe.moe_runner.ifmoe.wrapper import fused_experts_ifmoe
from sglang.srt.layers.moe.utils import MoeRunnerBackend

logger = logging.getLogger(__name__)

_BACKEND_NAME = "ifmoe"

try:
    MoeRunnerBackend(_BACKEND_NAME)
except ValueError:
    logger.warning(
        f"MoeRunnerBackend.IFMOE not found in sglang. "
        f'Add `IFMOE = "{_BACKEND_NAME}"` to sglang/srt/layers/moe/utils.py.'
    )
else:
    FusedOpPool.register_fused_func("none", _BACKEND_NAME, fused_experts_ifmoe)
    logger.info(f"IFMoe kernel registered as moe-runner-backend='{_BACKEND_NAME}'")
