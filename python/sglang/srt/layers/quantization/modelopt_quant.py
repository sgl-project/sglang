# SPDX-License-Identifier: Apache-2.0
"""Deprecated compatibility shim for the ModelOpt quantization package.

Use ``sglang.srt.layers.quantization.modelopt`` instead.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

logger.warning(
    "sglang.srt.layers.quantization.modelopt_quant is deprecated; "
    "import from sglang.srt.layers.quantization.modelopt instead."
)

from sglang.srt.layers.quantization.modelopt import *  # noqa: F403
from sglang.srt.layers.quantization.modelopt import __all__ as __all__
