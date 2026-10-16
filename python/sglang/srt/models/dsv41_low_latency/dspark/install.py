# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Binds the low-latency paths to a built DSpark V4 draft model."""

from __future__ import annotations

from functools import partial

from torch import nn

from sglang.srt.models.dsv41_low_latency import mxfp8_linear


def install_draft_model(model: nn.Module) -> None:
    """Called at the end of the draft model's construction, before weights load: the
    stages' attention projections on the low-latency MXFP8 GEMM and their MoE on the
    routed-expert GEMM pair. Draft stages do not defer the MoE finalize, so the pair
    also finalizes into the MoE output (finalize.cu)."""
    from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
        Mxfp4FlashinferTrtllmMoEMethod,
    )
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
    from sglang.srt.models.dsv41_low_latency.routed_expert_gemm import build

    for stage in model.stages:
        attn = stage.self_attn
        mxfp8_linear.install((attn.wq_b, attn.wo_b))
        if isinstance(stage.mlp, DeepseekV2MoE) and isinstance(
            stage.mlp.experts.quant_method, Mxfp4FlashinferTrtllmMoEMethod
        ):
            stage.mlp.experts.quant_method.low_batch_gemm_factory = partial(
                build, finalize=True
            )
