# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1-Flash monokernels for gfx950 (RFC #43949): K1 (``build_mono_k1``),
K2 (``build_mono_k2``) and the FFN launch (``build_mono_ffn``) in ``layer``, and
the host runner with its buffers in ``runner``, and the spec and its SGLang glue
in ``spec``.

Adapted from vLLM's DeepSeek-V4.1 mono decode, whose shared stages come from
ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6 (MIT License, Copyright (C)
2026, Advanced Micro Devices, Inc.); each file names its origin.
"""

# DeepSeek-V4.1-Flash's shapes, routing and activation, which the kernels build in
HF_CONFIG = dict(
    hidden_size=5120,
    moe_intermediate_size=2304,
    n_routed_experts=384,
    num_experts_per_tok=6,
    n_shared_experts=1,
    routed_scaling_factor=1.5,
    scoring_func="sqrtsoftplus",
    norm_topk_prob=True,
    topk_method="noaux_tc",
    hidden_act="silu",
    swiglu_limit=10.0,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    rms_norm_eps=1e-20,
    hc_pre_from_prev_sublayer=True,
    head_dim=512,
    q_lora_rank=1280,
    qk_rope_head_dim=64,
    sliding_window=128,
    index_topk=512,
)
