# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest
import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsa import cutedsl_paged_mqa_logits, pick_dsl_expand
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


@triton.jit
def _late_copy_kernel(dst_a, src_a, n_a, dst_b, src_b, n_b, spin, BLOCK: tl.constexpr):
    # Lets the next (PDL) kernel launch at once, then writes its inputs late.
    tl.extra.cuda.gdc_launch_dependents()
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for _ in range(spin):
        acc = acc * 0.5 + 1.0
    late = tl.where(acc < 0.0, 1, 0)  # always 0; keeps the spin ahead of the stores
    offs = tl.arange(0, BLOCK)
    mask_a, mask_b = offs < n_a, offs < n_b
    tl.store(dst_a + offs, tl.load(src_a + offs, mask=mask_a) + late, mask=mask_a)
    tl.store(dst_b + offs, tl.load(src_b + offs, mask=mask_b) + late, mask=mask_b)


def _run_cutedsl_paged_mqa_logits(
    data, batch_size, next_n, num_heads, max_model_len, is_target_verify
):
    """Mirrors the CUTEDSL dispatch in
    sglang.srt.layers.attention.dsa.dsa_indexer.Indexer._get_topk_paged."""
    import deep_gemm

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
    schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
        expanded_ctx.unsqueeze(-1), BLOCK_KV, num_sms
    )
    block_tables_expanded = data["block_table"].repeat_interleave(next_n, dim=0)

    return cutedsl_paged_mqa_logits(
        data["q_fp8"].view(batch_size * next_n, num_heads, HEAD_DIM),
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
        get_paged_mqa_logits_metadata_fn=deep_gemm.get_paged_mqa_logits_metadata,
    )


@pytest.mark.skipif(
    not is_sm100_supported(),
    reason="CuTe DSL FP8 Paged MQA Logits only supports SM 100 family.",
)
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("next_n", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("num_heads", [32, 64])
@pytest.mark.parametrize("avg_ctx", [128, 1024, 4096, 16384])
def test_cutedsl_paged_mqa_logits(batch_size, next_n, num_heads, avg_ctx):
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


@pytest.mark.skipif(
    not is_sm100_supported(),
    reason="CuTe DSL FP8 Paged MQA Logits only supports SM 100 family.",
)
@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("num_heads", [32, 64])
def test_cutedsl_paged_mqa_logits_inputs_written_by_preceding_kernel(
    batch_size, num_heads
):
    """Under PDL the kernel can start while the preceding kernel still writes the
    schedule metadata and context lens, so it may read them only after
    griddepcontrol_wait."""
    import deep_gemm

    next_n, avg_ctx = 1, 4096
    max_model_len = avg_ctx * 2
    data = generate_paged_mqa_test_data(
        batch_size, next_n, num_heads, avg_ctx, max_model_len
    )
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    good_ctx = data["context_lens"]
    good_meta = deep_gemm.get_paged_mqa_logits_metadata(
        good_ctx.unsqueeze(-1), BLOCK_KV, num_sms
    )
    context_lens, schedule_metadata = good_ctx.clone(), good_meta.clone()
    weights = data["weights"]

    def run():
        # Decode dispatch: no kernel runs between the writer and this one.
        return cutedsl_paged_mqa_logits(
            data["q_fp8"].view(batch_size, num_heads, HEAD_DIM),
            data["kv_fused"],
            weights,
            context_lens,
            data["block_table"],
            schedule_metadata,
            max_model_len,
            q_offset=batch_size,
            B=batch_size,
            next_n=next_n,
            is_target_verify=False,
            dsl_expand_factor=1,
            dsl_atom=1,
            blocksize=BLOCK_KV,
            sm_count=num_sms,
            get_paged_mqa_logits_metadata_fn=deep_gemm.get_paged_mqa_logits_metadata,
        )

    # A config's first call compiles and loads the kernel on the host, long
    # enough for the writer to finish before the kernel starts.
    run()
    generator = torch.Generator(device="cuda").manual_seed(0)
    for _ in range(3):
        # New weights each round, so logits a CTA skipped cannot match from memory.
        weights.copy_(
            torch.randn(weights.shape, generator=generator, device=weights.device)
        )
        # Zero metadata gives every CTA an empty range; a context length of
        # max_model_len keeps a stale read in bounds.
        schedule_metadata.zero_()
        context_lens.fill_(max_model_len)
        _late_copy_kernel[(1,)](
            context_lens,
            good_ctx,
            context_lens.numel(),
            schedule_metadata,
            good_meta,
            schedule_metadata.numel(),
            200000,
            BLOCK=triton.next_power_of_2(schedule_metadata.numel()),
        )
        logits = run()

        ref_logits = ref_fp8_paged_mqa_logits(
            data["q_fp8"],
            data["kv_fp8"],
            data["kv_scales"],
            weights,
            good_ctx,
            data["block_table"],
            max_model_len,
            BLOCK_KV,
        )
        assert_paged_mqa_matches_ref(
            logits, ref_logits, good_ctx, batch_size, next_n, max_model_len
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
