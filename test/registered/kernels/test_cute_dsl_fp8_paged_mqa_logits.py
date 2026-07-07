# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest
import torch

from sglang.jit_kernel.dsa import cutedsl_paged_mqa_logits, pick_dsl_expand
from sglang.srt.layers.attention.dsa.utils import (
    fp8_mqa_logits_ceil_to_ue8m0,
    fp8_mqa_logits_make_fused_kv,
)
from sglang.srt.utils import is_sm100_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, suite="nightly-4-gpu-b200", nightly=True)

BLOCK_KV = 64
HEAD_DIM = 128


def _ref_fp8_paged_mqa_logits(
    q_fp8,
    kv_fp8,
    kv_scales,
    weights,
    context_lens,
    block_table,
    max_model_len,
    block_kv,
):
    B, next_n, H, D = q_fp8.shape
    device = q_fp8.device

    logits = torch.full(
        (B * next_n, max_model_len), float("-inf"), device=device, dtype=torch.float32
    )
    q_f32 = q_fp8.float()

    for b in range(B):
        ctx_len = context_lens[b].item()
        q_positions = torch.arange(ctx_len - next_n, ctx_len, device=device)
        w = weights[b * next_n : (b + 1) * next_n, :]

        for blk_idx in range((ctx_len + block_kv - 1) // block_kv):
            phys_blk = block_table[b, blk_idx].item()
            k_f32 = kv_fp8[phys_blk].float()
            scales = kv_scales[phys_blk]

            k_positions = torch.arange(
                blk_idx * block_kv, (blk_idx + 1) * block_kv, device=device
            )
            mask = (k_positions[None, :] < ctx_len) & (
                k_positions[None, :] <= q_positions[:, None]
            )

            qk = torch.matmul(q_f32[b].permute(1, 0, 2), k_f32.T)
            qk = torch.where(mask[None, :, :], qk, torch.zeros(1, device=device))
            qk = torch.relu(qk)

            weighted = (w.T[:, :, None] * qk).sum(dim=0)
            weighted = weighted * scales[None, :]

            start_pos = blk_idx * block_kv
            end_pos = start_pos + block_kv
            logits[b * next_n : (b + 1) * next_n, start_pos:end_pos] = torch.where(
                mask,
                weighted,
                torch.tensor(float("-inf"), device=device, dtype=torch.float32),
            )

    return logits


def _generate_test_data(
    batch_size,
    next_n,
    num_heads,
    avg_context_len,
    max_model_len,
    device="cuda",
):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    context_lens = torch.randint(
        max(BLOCK_KV, int(0.7 * avg_context_len)),
        int(1.3 * avg_context_len) + 1,
        (batch_size,),
        dtype=torch.int32,
        device="cpu",
    ).clamp(max=max_model_len)

    max_blocks_per_seq = (max_model_len + BLOCK_KV - 1) // BLOCK_KV
    total_blocks = ((context_lens + BLOCK_KV - 1) // BLOCK_KV).sum().item()
    num_phys_blocks = total_blocks + batch_size * 2

    block_table = torch.full(
        (batch_size, max_blocks_per_seq), 0, dtype=torch.int32, device=device
    )
    blk_offset = 0
    for i in range(batch_size):
        n_blks = (context_lens[i].item() + BLOCK_KV - 1) // BLOCK_KV
        block_table[i, :n_blks] = torch.arange(
            blk_offset, blk_offset + n_blks, dtype=torch.int32, device=device
        )
        blk_offset += n_blks

    q_bf16 = torch.randn(batch_size, next_n, num_heads, HEAD_DIM, device=device)
    q_fp8 = q_bf16.to(torch.float8_e4m3fn)

    kv_bf16 = torch.randn(num_phys_blocks, BLOCK_KV, HEAD_DIM, device=device)
    kv_amax = kv_bf16.abs().float().amax(dim=-1, keepdim=True).clamp(1e-4)
    kv_scale = fp8_mqa_logits_ceil_to_ue8m0(kv_amax / 448.0).squeeze(-1)
    kv_fp8 = (kv_bf16 / kv_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)

    weights = torch.randn(
        batch_size * next_n, num_heads, device=device, dtype=torch.float32
    )
    kv_fused = fp8_mqa_logits_make_fused_kv(kv_fp8, kv_scale, BLOCK_KV, HEAD_DIM)

    return {
        "q_fp8": q_fp8,
        "kv_fp8": kv_fp8,
        "kv_scales": kv_scale,
        "kv_fused": kv_fused,
        "weights": weights,
        "context_lens": context_lens.to(device),
        "block_table": block_table,
    }


def _assert_matches_ref(logits, ref_logits, context_lens, B, next_n, max_model_len):
    device = logits.device
    positions = torch.arange(max_model_len, device=device).unsqueeze(0)
    row_indices = torch.arange(B * next_n, device=device) // next_n
    next_n_offset = torch.arange(B * next_n, device=device) % next_n
    end_pos = context_lens[row_indices] - next_n + next_n_offset
    mask = positions <= end_pos.unsqueeze(1)

    logits_masked = logits.float().masked_fill(~mask, 0)
    ref_masked = ref_logits.float().masked_fill(~mask, 0)
    torch.testing.assert_close(logits_masked, ref_masked, atol=5e-5, rtol=1e-5)


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
    dsl_atom_split = dsl_expand_factor > 1 and next_n == dsl_expand_factor * dsl_atom
    ctx_lens_1d = (
        context_lens.repeat_interleave(dsl_expand_factor)
        if dsl_atom_split
        else context_lens
    )
    schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
        ctx_lens_1d.unsqueeze(-1), BLOCK_KV, num_sms
    )
    block_tables_dsl = (
        data["block_table"].repeat_interleave(dsl_expand_factor, dim=0)
        if dsl_atom_split
        else data["block_table"].repeat_interleave(next_n, dim=0)
    )

    return cutedsl_paged_mqa_logits(
        data["q_fp8"].view(batch_size * next_n, num_heads, HEAD_DIM),
        data["kv_fused"],
        data["weights"],
        ctx_lens_1d,
        block_tables_dsl,
        schedule_metadata,
        max_model_len,
        q_offset=batch_size * next_n,
        B=batch_size,
        next_n=next_n,
        is_target_verify=is_target_verify,
        dsl_expand_factor=dsl_expand_factor,
        dsl_atom=dsl_atom,
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
    data = _generate_test_data(batch_size, next_n, num_heads, avg_ctx, max_model_len)

    logits = _run_cutedsl_paged_mqa_logits(
        data,
        batch_size,
        next_n,
        num_heads,
        max_model_len,
        is_target_verify=next_n >= 2,
    )

    ref_logits = _ref_fp8_paged_mqa_logits(
        data["q_fp8"],
        data["kv_fp8"],
        data["kv_scales"],
        data["weights"],
        data["context_lens"],
        data["block_table"],
        max_model_len,
        BLOCK_KV,
    )
    _assert_matches_ref(
        logits, ref_logits, data["context_lens"], batch_size, next_n, max_model_len
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
