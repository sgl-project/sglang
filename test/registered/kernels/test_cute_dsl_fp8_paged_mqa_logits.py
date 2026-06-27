# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Test the CuTe DSL fp8_paged_mqa_logits kernel against a PyTorch reference.

Ported from TensorRT-LLM PR #13219
(``tests/unittest/_torch/attention/sparse/test_cute_dsl_fp8_paged_mqa_logits.py``)
to SGLang.
"""

import pytest
import torch

from sglang.srt.utils import is_sm100_supported
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.cute_dsl_mqa_logits_utils import ceil_to_ue8m0, make_fused_kv

skip_not_sm100 = pytest.mark.skipif(
    not is_sm100_supported(),
    reason="CuTe DSL FP8 Paged MQA Logits only supports SM 100 family.",
)

register_cuda_ci(est_time=15, suite="nightly-4-gpu-b200", nightly=True)


def _ref_fp8_paged_mqa_logits(
    q_fp8,
    kv_fp8,
    kv_scales,
    weights,
    context_lens,
    block_table,
    max_model_len,
    block_kv,
    epi_dtype=torch.float32,
):
    B, next_n, H, D = q_fp8.shape
    device = q_fp8.device

    logits = torch.full(
        (B * next_n, max_model_len), float("-inf"), device=device, dtype=epi_dtype
    )
    q_f32 = q_fp8.float()

    for b in range(B):
        ctx_len = context_lens[b].item()
        q_positions = torch.arange(ctx_len - next_n, ctx_len, device=device)
        w = weights[b * next_n : (b + 1) * next_n, :].to(epi_dtype)

        for blk_idx in range((ctx_len + block_kv - 1) // block_kv):
            phys_blk = block_table[b, blk_idx].item()
            k_f32 = kv_fp8[phys_blk].float()
            scales = kv_scales[phys_blk].to(epi_dtype)

            k_positions = torch.arange(
                blk_idx * block_kv, (blk_idx + 1) * block_kv, device=device
            )
            mask = (k_positions[None, :] < ctx_len) & (
                k_positions[None, :] <= q_positions[:, None]
            )

            qk = torch.matmul(q_f32[b].permute(1, 0, 2), k_f32.T)
            qk = torch.where(mask[None, :, :], qk, torch.zeros(1, device=device))
            qk = torch.relu(qk)

            qk = qk.to(epi_dtype)
            weighted = (w.T[:, :, None] * qk).sum(dim=0)
            weighted = weighted * scales[None, :]

            start_pos = blk_idx * block_kv
            end_pos = start_pos + block_kv
            logits[b * next_n : (b + 1) * next_n, start_pos:end_pos] = torch.where(
                mask,
                weighted,
                torch.tensor(float("-inf"), device=device, dtype=epi_dtype),
            )

    return logits


def _generate_test_data(
    batch_size,
    next_n,
    num_heads,
    head_dim,
    block_kv,
    avg_context_len,
    max_model_len,
    device="cuda",
    use_int_data=False,
    fix_length=True,
):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    if fix_length:
        context_lens = torch.full(
            (batch_size,), max_model_len, dtype=torch.int32, device="cpu"
        )
    else:
        context_lens = torch.randint(
            max(block_kv, int(0.7 * avg_context_len)),
            int(1.3 * avg_context_len) + 1,
            (batch_size,),
            dtype=torch.int32,
            device="cpu",
        )
        context_lens = context_lens.clamp(max=max_model_len)

    max_blocks_per_seq = (max_model_len + block_kv - 1) // block_kv
    total_blocks = ((context_lens + block_kv - 1) // block_kv).sum().item()
    num_phys_blocks = total_blocks + batch_size * 2

    block_table = torch.full(
        (batch_size, max_blocks_per_seq), 0, dtype=torch.int32, device=device
    )
    blk_offset = 0
    for i in range(batch_size):
        n_blks = (context_lens[i].item() + block_kv - 1) // block_kv
        block_table[i, :n_blks] = torch.arange(
            blk_offset, blk_offset + n_blks, dtype=torch.int32, device=device
        )
        blk_offset += n_blks

    if use_int_data:
        q_fp8 = torch.randint(
            -3,
            4,
            (batch_size, next_n, num_heads, head_dim),
            device=device,
            dtype=torch.float32,
        ).to(torch.float8_e4m3fn)
        kv_fp8 = torch.randint(
            -3,
            4,
            (num_phys_blocks, block_kv, head_dim),
            device=device,
            dtype=torch.float32,
        ).to(torch.float8_e4m3fn)
        kv_scale = torch.ones(
            num_phys_blocks, block_kv, device=device, dtype=torch.float32
        )
        weights = torch.randint(
            -3,
            4,
            (batch_size * next_n, num_heads),
            device=device,
            dtype=torch.float32,
        )
    else:
        q_bf16 = torch.randn(batch_size, next_n, num_heads, head_dim, device=device)
        q_fp8 = q_bf16.to(torch.float8_e4m3fn)

        kv_bf16 = torch.randn(num_phys_blocks, block_kv, head_dim, device=device)
        kv_amax = kv_bf16.abs().float().amax(dim=-1, keepdim=True).clamp(1e-4)
        kv_scale = ceil_to_ue8m0(kv_amax / 448.0).squeeze(-1)
        kv_fp8 = (kv_bf16 / kv_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)

        weights = torch.randn(
            batch_size * next_n, num_heads, device=device, dtype=torch.float32
        )

    kv_fused = make_fused_kv(kv_fp8, kv_scale, block_kv, head_dim)

    return {
        "q_fp8": q_fp8,
        "kv_fp8": kv_fp8,
        "kv_scales": kv_scale,
        "kv_fused": kv_fused,
        "weights": weights,
        "context_lens": context_lens.to(device),
        "block_table": block_table,
        "max_model_len": max_model_len,
        "block_kv": block_kv,
        "num_phys_blocks": num_phys_blocks,
    }


@skip_not_sm100
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("next_n", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("num_heads", [64])
@pytest.mark.parametrize("avg_ctx", [256, 4096])
@pytest.mark.parametrize("output_dtype", [torch.float32, torch.float16])
def test_cute_dsl_fp8_paged_mqa_logits(
    batch_size, next_n, num_heads, avg_ctx, output_dtype
):
    """Compare the CuTe DSL kernel output against the PyTorch reference."""
    # Importing here so the module is only required when SM100 GPUs are available.
    import deep_gemm  # noqa: F401

    # Force op + runner registration.
    from sglang.srt.layers.attention.dsa import cute_dsl_paged_mqa_logits  # noqa: F401

    head_dim = 128
    block_kv = 128
    max_model_len = max(avg_ctx * 2, 2048)

    data = _generate_test_data(
        batch_size,
        next_n,
        num_heads,
        head_dim,
        block_kv,
        avg_ctx,
        max_model_len,
        use_int_data=(output_dtype == torch.float16),
        fix_length=True,
    )

    num_sms = torch.cuda.get_device_properties(0).multi_processor_count

    # See sglang/srt/layers/attention/dsa/cute_dsl_paged_mqa_logits.py for the
    # "block_kv = 64 == SPLIT_KV/4" alignment rationale (DG metadata is not
    # SM100-aware on this code path).
    DG_METADATA_BLOCK_KV = 64
    dsl_schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(
        data["context_lens"].unsqueeze(-1), DG_METADATA_BLOCK_KV, num_sms
    )

    ref_logits = _ref_fp8_paged_mqa_logits(
        data["q_fp8"],
        data["kv_fp8"],
        data["kv_scales"],
        data["weights"],
        data["context_lens"],
        data["block_table"],
        max_model_len,
        block_kv,
        epi_dtype=output_dtype,
    )

    dsl_logits = torch.ops.sglang.cute_dsl_fp8_paged_mqa_logits(
        data["q_fp8"],
        data["kv_fused"],
        data["weights"],
        data["context_lens"],
        data["block_table"],
        dsl_schedule_meta,
        max_model_len,
        epi_dtype=output_dtype,
        acc_dtype=output_dtype,
        output_dtype=output_dtype,
    )
    assert dsl_logits.dtype == output_dtype

    B = batch_size
    positions = torch.arange(max_model_len, device="cuda").unsqueeze(0)
    row_indices = torch.arange(B * next_n, device="cuda") // next_n
    next_n_offset = torch.arange(B * next_n, device="cuda") % next_n
    end_pos = data["context_lens"][row_indices] - next_n + next_n_offset
    mask = positions <= end_pos.unsqueeze(1)

    dsl_masked = dsl_logits.float().masked_fill(~mask, 0)
    ref_masked = ref_logits.float().masked_fill(~mask, 0)
    finite = torch.isfinite(dsl_masked) & torch.isfinite(ref_masked)
    dsl_clean = dsl_masked.masked_fill(~finite, 0)
    ref_clean = ref_masked.masked_fill(~finite, 0)

    elem_atol = 1e-3 if output_dtype == torch.float16 else 5e-5
    elem_rtol = 1e-3 if output_dtype == torch.float16 else 1e-5
    torch.testing.assert_close(dsl_clean, ref_clean, atol=elem_atol, rtol=elem_rtol)
