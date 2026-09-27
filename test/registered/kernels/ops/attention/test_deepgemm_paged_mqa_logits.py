# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsa import (
    deepgemm_paged_mqa_logits_native,
    deepgemm_paged_mqa_logits_split,
)
from sglang.srt.utils import is_sm90_supported, is_sm100_supported
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.paged_mqa import (
    BLOCK_KV,
    HEAD_DIM,
    assert_paged_mqa_matches_ref,
    generate_paged_mqa_test_data,
    ref_fp8_paged_mqa_logits,
)

register_cuda_ci(est_time=40, stage="nightly", runner_config="4-gpu-b200")


def _run_deepgemm_paged_mqa_logits(data, batch_size, next_n, num_heads, max_model_len):
    """Mirrors the DEEPGEMM dispatch in
    sglang.srt.layers.attention.dsa.dsa_indexer.Indexer._get_topk_paged:
    next_n>=2 (target-verify) goes through the native wrapper, everything
    else goes through the split wrapper."""
    import deep_gemm

    num_sms = torch.cuda.get_device_properties(0).multi_processor_count

    if next_n >= 2:
        ctx_lens_2d = (
            data["context_lens"].unsqueeze(-1)
            - next_n
            + torch.arange(
                1, next_n + 1, device=data["context_lens"].device, dtype=torch.int32
            )
        )
        schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
            ctx_lens_2d, BLOCK_KV, num_sms
        )
        block_tables_expanded = data["block_table"].repeat_interleave(next_n, dim=0)
        return deepgemm_paged_mqa_logits_native(
            deep_gemm.fp8_paged_mqa_logits,
            data["q_fp8"].view(batch_size * next_n, num_heads, HEAD_DIM),
            data["kv_fused"],
            data["weights"],
            ctx_lens_2d,
            block_tables_expanded,
            schedule_metadata,
            max_model_len,
            q_offset=batch_size * next_n,
            B=batch_size,
            next_n=next_n,
        )

    ctx_lens_2d = data["context_lens"].unsqueeze(-1)
    schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
        ctx_lens_2d, BLOCK_KV, num_sms
    )
    return deepgemm_paged_mqa_logits_split(
        deep_gemm.fp8_paged_mqa_logits,
        data["q_fp8"].squeeze(1),
        data["kv_fused"],
        data["weights"],
        ctx_lens_2d,
        data["block_table"],
        schedule_metadata,
        max_model_len,
        q_offset=batch_size,
    )


@pytest.mark.skipif(
    not (is_sm90_supported() or is_sm100_supported()),
    reason="DeepGEMM fp8_paged_mqa_logits requires SM90 (Hopper) or newer.",
)
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("next_n", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("num_heads", [32, 64])
@pytest.mark.parametrize("avg_ctx", [128, 1024, 4096, 16384])
def test_deepgemm_paged_mqa_logits(batch_size, next_n, num_heads, avg_ctx):
    max_model_len = max(avg_ctx * 2, 2048)
    data = generate_paged_mqa_test_data(
        batch_size, next_n, num_heads, avg_ctx, max_model_len
    )

    logits = _run_deepgemm_paged_mqa_logits(
        data, batch_size, next_n, num_heads, max_model_len
    )

    ref_logits = ref_fp8_paged_mqa_logits(
        data["q_fp8"],
        data["kv_fp8"],
        data["kv_scales"],
        data["weights"],
        data["context_lens"],
        data["block_table"],
        max_model_len,
        BLOCK_KV,
    )
    assert_paged_mqa_matches_ref(
        logits, ref_logits, data["context_lens"], batch_size, next_n, max_model_len
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
