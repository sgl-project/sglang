# SPDX-License-Identifier: Apache-2.0
"""
Nunchaku quantization helpers for multimodal_gen.

This module currently exposes a single public helper:

- ``create_nunchaku_config_from_server_args``: construct :class:`NunchakuConfig`
  from ``ServerArgs`` so that layer-wise Nunchaku quantization can be enabled
  in the standard SGLang transformer loaders.

Historically this file also contained helpers for full-transformer replacement
(``load_nunchaku_model``, ``should_use_nunchaku``), but the multimodal_gen
pipeline now always uses **per-layer** quantization instead of swapping the
entire transformer implementation. Those legacy entry points have been removed
to avoid confusion.
"""

from sglang.multimodal_gen.runtime.layers.quantization.nunchaku_config import (
    NunchakuConfig,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def create_nunchaku_config_from_server_args(server_args) -> NunchakuConfig:
    """
    Create NunchakuConfig from server arguments.

    Args:
        server_args: Server configuration arguments

    Returns:
        NunchakuConfig instance
    """
    return NunchakuConfig(
        precision=getattr(server_args, "quantization_precision", "int4"),
        rank=getattr(server_args, "quantization_rank", 32),
        act_unsigned=getattr(server_args, "quantization_act_unsigned", False),
        quantized_model_path=getattr(server_args, "quantized_model_path", None),
    )
