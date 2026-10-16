# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Binds the low-latency paths to a built DeepSeek-V4.1 target model. Each path
replaces one stock call and declines (falls back to it) outside its row bound."""

from __future__ import annotations

import logging

from torch import nn

from sglang.srt.models.dsv41_low_latency import indexer, mxfp8_linear, wo_b_publish

logger = logging.getLogger(__name__)


def install_decoder_layers(model: nn.Module) -> None:
    """Called at the end of the target model's construction, before weights load."""
    from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
        Mxfp4FlashinferTrtllmMoEMethod,
    )
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
    from sglang.srt.models.dsv41_low_latency.routed_expert_gemm import build

    for i in range(model.start_layer, model.end_layer):
        layer = model.layers[i]
        attn = layer.self_attn
        projections = (attn.wqkv_a,) if attn.fuse_wqa_wkv else ()
        mxfp8_linear.install(projections + (attn.wq_b, attn.wo_b))
        indexer.install_projections(attn)
        indexer.install_index_q_side_stream(attn)
        wo_b_publish.install(layer)
        if isinstance(layer.mlp, DeepseekV2MoE) and isinstance(
            layer.mlp.experts.quant_method, Mxfp4FlashinferTrtllmMoEMethod
        ):
            # Read by the MXFP4 method once the weights are shuffled.
            layer.mlp.experts.quant_method.low_batch_gemm_factory = build
    logger.info(
        "DSV4.1 low-latency decode installed on layers %d-%d.",
        model.start_layer,
        model.end_layer - 1,
    )


def install_lm_head(lm_head: nn.Module) -> None:
    from sglang.srt.models.dsv41_low_latency import lm_head as low_latency_lm_head

    low_latency_lm_head.install(lm_head)


def autotune(model: nn.Module, lm_head: nn.Module, *, dtype) -> None:
    """Inside the FlashInfer autotune context, which pins the MXFP8 tactics; the
    same calls on every rank."""
    from sglang.srt.models.dsv41_low_latency import lm_head as low_latency_lm_head

    mxfp8_linear.autotune(model, dtype=dtype)
    low_latency_lm_head.autotune(lm_head)
