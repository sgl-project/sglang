# SPDX-License-Identifier: Apache-2.0
"""Online FP8 installation for the H3 decoder's 144 block linears.

The opt-in deployment flag quantizes the four linears per block at load time.
Other weights keep their precision; request quality does not control FP8.
"""

import torch
from torch import nn

from sglang.multimodal_gen.runtime.layers.quantization.online_fp8_linear import (
    OnlineFP8Linear,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

from .fused_decoder import structure_reject_reason

logger = init_logger(__name__)

LINEAR_SHAPES = {
    "attn.to_qkv": (6144, 2048),
    "attn.to_out": (2048, 2048),
    "ff.w1": (16384, 2048),
    "ff.w2": (2048, 8192),
}


def inspect_fp8_scope(decoder):
    if len(decoder.transformer_blocks) != 36:
        raise ValueError("MiniMax-H3 online FP8 requires its 36-block decoder")
    paths = []
    for index, block in enumerate(decoder.transformer_blocks):
        for suffix, shape in LINEAR_SHAPES.items():
            name = f"transformer_blocks.{index}.{suffix}"
            module = block.get_submodule(suffix)
            if (
                not isinstance(module, (nn.Linear, OnlineFP8Linear))
                or tuple(module.weight.shape) != shape
            ):
                raise ValueError(f"Unexpected MiniMax-H3 FP8 linear scope: {name}")
            if module.bias is None or module.bias.shape != (module.out_features,):
                raise ValueError(f"Expected the released biased linear at {name}")
            paths.append(name)
    return paths


def install_fp8_block_linears(decoder) -> int:
    """Swap the block linears for FP8 modules; returns how many were replaced."""
    if decoder.fp8_installed:
        return 0
    # Validate every target before replacing any module; an offloaded or
    # unprepared decoder fails here with the original modules intact.
    paths = inspect_fp8_scope(decoder)
    decoder.prepare_autocast_linear_weights(torch.float16)
    for path in paths:
        source = decoder.get_submodule(path)
        if source.weight.dtype != torch.float16 or not source.weight.is_cuda:
            raise ValueError(
                "Online FP8 requires resident, prepared FP16 decoder weights"
            )
    for path in paths:
        source = decoder.get_submodule(path)
        decoder.set_submodule(path, OnlineFP8Linear.from_linear(source))
    decoder.fp8_installed = True
    decoder._autocast_linear_dtype = torch.float16
    reason = structure_reject_reason(decoder)
    decoder.fused_decoder_ready = reason is None
    if reason is not None:
        logger.info(
            "MiniMax-H3 VAE: FP8 decoder blocks keep the eager producers (%s)", reason
        )
    return len(paths)
