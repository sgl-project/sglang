# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsa import cutedsl_paged_mqa_logits, pick_dsl_expand
from sglang.srt.layers.attention.dsa.utils import resolve_paged_mqa_logits_metadata_fn
from sglang.srt.utils import is_sm100_supported
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.paged_mqa import (
    BLOCK_KV,
    HEAD_DIM,
    assert_paged_mqa_matches_ref,
    generate_paged_mqa_test_data,
    ref_fp8_paged_mqa_logits,
)

register_cuda_ci(est_time=180, stage="nightly", runner_config="4-gpu-b200")


def _run_cutedsl_paged_mqa_logits(
    data,
    batch_size,
    next_n,
    num_heads,
    max_model_len,
    is_target_verify,
    draft_extend=False,
):
    """Mirrors the CUTEDSL dispatch in
    sglang.srt.layers.attention.dsa.dsa_indexer.Indexer._get_topk_paged."""
    get_metadata = resolve_paged_mqa_logits_metadata_fn(cutedsl=True)
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    if is_target_verify and next_n >= 2:
        dsl_expand_factor, dsl_atom = pick_dsl_expand(
            next_n,
            batch_size=batch_size,
            max_ctx=max_model_len,
            num_sms=num_sms,
            kernel_atoms=(1, 2, 3, 4),
            num_heads=num_heads,
        )
    else:
        dsl_expand_factor, dsl_atom = 1, 1

    context_lens = data["context_lens"]
    expanded_ctx = (
        context_lens.unsqueeze(-1)
        - next_n
        + torch.arange(1, next_n + 1, device=context_lens.device, dtype=torch.int32)
    ).flatten()
    block_tables_expanded = data["block_table"].repeat_interleave(next_n, dim=0)
    q_rows = data["q_fp8"].view(batch_size * next_n, num_heads, HEAD_DIM)
    if draft_extend:
        return cutedsl_paged_mqa_logits(
            q_rows,
            data["kv_fused"],
            data["weights"],
            expanded_ctx,
            block_tables_expanded,
            None,
            max_model_len,
            q_offset=batch_size * next_n,
            B=batch_size * next_n,
            next_n=1,
            is_target_verify=False,
            dsl_expand_factor=1,
            dsl_atom=1,
            blocksize=BLOCK_KV,
            sm_count=num_sms,
            get_paged_mqa_logits_metadata_fn=get_metadata,
        )
    schedule_metadata = get_metadata(expanded_ctx.unsqueeze(-1), BLOCK_KV, num_sms)

    return cutedsl_paged_mqa_logits(
        q_rows,
        data["kv_fused"],
        data["weights"],
        context_lens,
        block_tables_expanded,
        schedule_metadata,
        max_model_len,
        q_offset=batch_size * next_n,
        B=batch_size,
        next_n=next_n,
        is_target_verify=is_target_verify,
        dsl_expand_factor=dsl_expand_factor,
        dsl_atom=dsl_atom,
        blocksize=BLOCK_KV,
        sm_count=num_sms,
        get_paged_mqa_logits_metadata_fn=get_metadata,
    )


@pytest.mark.skipif(
    not is_sm100_supported(),
    reason="CuTe DSL FP8 Paged MQA Logits only supports SM 100 family.",
)
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("next_n", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("num_heads", [32, 64])
@pytest.mark.parametrize("avg_ctx", [128, 1024, 4096, 16384])
@pytest.mark.parametrize("draft_extend", [False, True])
def test_cutedsl_paged_mqa_logits(batch_size, next_n, num_heads, avg_ctx, draft_extend):
    if draft_extend and next_n == 1:
        pytest.skip("draft extend scores next_n >= 2 rows per request")
    max_model_len = max(avg_ctx * 2, 2048)
    data = generate_paged_mqa_test_data(
        batch_size, next_n, num_heads, avg_ctx, max_model_len
    )

    logits = _run_cutedsl_paged_mqa_logits(
        data,
        batch_size,
        next_n,
        num_heads,
        max_model_len,
        is_target_verify=next_n >= 2,
        draft_extend=draft_extend,
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
