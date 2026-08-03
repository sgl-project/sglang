from __future__ import annotations

import sys

import pytest
import torch

from sglang.kernels.ops.attention.deepseek_v4_rope import precompute_freqs_cis
from sglang.kernels.ops.attention.dsv4 import (
    CompressorDecodePlan,
    compress_norm_rope_store,
)
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

# ----------------------------------------------------------------------------
# FlashMLA fused-norm-rope-store geometry (head_dim=512, NOT the 128 indexer).
# ----------------------------------------------------------------------------
HEAD_DIM = 512
ROPE_DIM = 64  # RoPE applied to the TAIL 64 dims [448:512]
NOPE_DIM = HEAD_DIM - ROPE_DIM  # 448 dims [0:448] stored (post-norm) as-is
PAGE_SIZE = 64
COMPRESS_RATIO = 4
EPS = 1.0e-6

# FP8 store layout (bytes/token, per the kernel + PR draft):
#   448 fp8-e4m3 (nope) + 128 bf16 (=64 bf16 rope tail) + 8 scale bytes = 584,
#   the 576-byte payload (448 + 128) is page-strided, then an 8-byte scale group
#   per token (only 7 of the 8 bytes are used: one UE8M0 exp per nope warp-group).
FP8_NOPE_BYTES = NOPE_DIM  # 448 fp8 bytes
FP8_ROPE_BYTES = ROPE_DIM * 2  # 128 bytes (64 bf16 rope values)
FP8_PAYLOAD_BYTES = 576  # 448 + 128
FP8_SCALE_BYTES = 8  # 7 UE8M0 group bytes + 1 pad
FP8_NUM_NOPE_GROUPS = NOPE_DIM // 64  # 7 per-warp UE8M0 groups of 64 elems
FP8_PAGE_BYTES = ((584 * PAGE_SIZE + 575) // 576) * 576

# bf16 store layout: whole head_dim written as bf16, page-strided.
BF16_BYTES_PER_TOKEN = HEAD_DIM * 2
BF16_PAGE_BYTES = BF16_BYTES_PER_TOKEN * PAGE_SIZE

# FP8-e4m3 has 3 mantissa bits => round-to-nearest relative error <= 2^-4 = 1/16.
# That is exactly the per-value tolerance we allow on the dequantized nope dims.
FP8_RTOL = 1.0 / 16.0
FP8_ATOL = 0.03  # covers near-zero values (fp8 subnormal step * group scale)
# bf16 rounding tolerance (8 mantissa bits ~ 2^-8 relative); matches the parity
# tolerance used by the project's golden for this kernel.
BF16_RTOL = 2.0e-2
BF16_ATOL = 2.0e-2

# Shapes: 1/8/64/256 exercise the K=1 small-N launcher branch (num_tokens <
# cutoff 2048); 2048 hits the K=4 large-N branch (num_tokens >= 2048). Both
# branches run identical per-token math/store, so both must match the reference.
SHAPES = [1, 8, 64, 256, 2048]

# NOTE: decode-only. The prefill/extend path needs a CompressorPrefillPlan with
# extend_lens/num_q_tokens plumbing; the public math + both store paths are
# fully exercised by the decode plan here (same kernel, same stores), matching
# how the sibling test_fp4_indexer.py test also drives decode only.


def _build_decode_inputs(num_tokens: int, seed: int):
    """Common inputs for both store paths: random kv/weight, a decode plan whose
    seq_lens are all multiples of compress_ratio (=> every token is valid), and
    freqs_cis sized to cover the largest RoPE position."""
    torch.manual_seed(seed)
    dev = get_device()
    kv = torch.randn(num_tokens, HEAD_DIM, device=dev, dtype=torch.bfloat16)
    norm_weight = torch.randn(HEAD_DIM, device=dev, dtype=torch.bfloat16)
    # seq_lens are multiples of COMPRESS_RATIO -> plan.seq_len % ratio == 0 -> valid.
    seq_lens = (
        torch.arange(1, num_tokens + 1, device=dev, dtype=torch.int64) * COMPRESS_RATIO
    )
    req_pool_indices = torch.arange(num_tokens, device=dev, dtype=torch.int64)
    plan = CompressorDecodePlan.generate_legacy(
        COMPRESS_RATIO, req_pool_indices, seq_lens
    )
    out_loc = torch.arange(num_tokens, device=dev, dtype=torch.int64)
    freqs_cis = precompute_freqs_cis(
        ROPE_DIM, int(seq_lens.max().item()) + 1, 0, 10000, 1, 32, 1
    ).to(dev)
    return kv, norm_weight, seq_lens, plan, out_loc, freqs_cis


def _reference(kv, norm_weight, seq_lens, freqs_cis):
    """Independent torch reference: RMSNorm(512) -> per-dim weight -> RoPE on the
    TAIL 64 dims -> return fp32 [N, 512]. Mirrors the kernel's math exactly."""
    num_tokens = kv.shape[0]
    ref = kv.float()
    ref = ref * torch.rsqrt((ref * ref).mean(dim=-1, keepdim=True) + EPS)
    ref = ref * norm_weight.float()

    position = (seq_lens - COMPRESS_RATIO).long()
    freqs = torch.view_as_real(freqs_cis).flatten(-2)[position]
    rope = ref[:, NOPE_DIM:].reshape(num_tokens, ROPE_DIM // 2, 2)
    freqs = freqs.reshape(num_tokens, ROPE_DIM // 2, 2)
    rope_out = torch.empty_like(rope)
    rope_out[..., 0] = rope[..., 0] * freqs[..., 0] - rope[..., 1] * freqs[..., 1]
    rope_out[..., 1] = rope[..., 0] * freqs[..., 1] + rope[..., 1] * freqs[..., 0]
    ref[:, NOPE_DIM:] = rope_out.reshape(num_tokens, ROPE_DIM)
    return ref


@pytest.mark.parametrize("num_tokens", SHAPES)
def test_flashmla_norm_rope_bf16_store(num_tokens: int) -> None:
    dev = get_device()
    kv, norm_weight, seq_lens, plan, out_loc, freqs_cis = _build_decode_inputs(
        num_tokens, seed=num_tokens + 1000
    )
    num_pages = max(1, (num_tokens + PAGE_SIZE - 1) // PAGE_SIZE)
    cache = torch.zeros(num_pages, BF16_PAGE_BYTES, device=dev, dtype=torch.uint8)

    compress_norm_rope_store(
        kv.clone(),
        plan,
        norm_weight=norm_weight,
        norm_eps=EPS,
        freq_cis=freqs_cis,
        out_loc=out_loc,
        kvcache=cache,
        page_size=PAGE_SIZE,
        use_fp4=False,
        bf16_store=True,
    )
    torch.cuda.synchronize()

    ref = _reference(kv, norm_weight, seq_lens, freqs_cis)

    # Readback: page-strided [num_pages, PAGE_SIZE, HEAD_DIM] bf16 rows.
    kv_bf16 = cache.view(torch.bfloat16).view(num_pages, PAGE_SIZE, HEAD_DIM)
    slots = out_loc.long()
    rows = kv_bf16[slots >> 6, slots & (PAGE_SIZE - 1)].float()  # [N, 512]

    assert not torch.isnan(rows).any(), "bf16 store produced NaN"
    assert not torch.isinf(rows).any(), "bf16 store produced Inf"
    torch.testing.assert_close(rows, ref, rtol=BF16_RTOL, atol=BF16_ATOL)


@pytest.mark.parametrize("num_tokens", SHAPES)
def test_flashmla_norm_rope_fp8_store(num_tokens: int) -> None:
    dev = get_device()
    kv, norm_weight, seq_lens, plan, out_loc, freqs_cis = _build_decode_inputs(
        num_tokens, seed=num_tokens + 2000
    )
    num_pages = max(1, (num_tokens + PAGE_SIZE - 1) // PAGE_SIZE)
    cache = torch.zeros(num_pages, FP8_PAGE_BYTES, device=dev, dtype=torch.uint8)

    compress_norm_rope_store(
        kv.clone(),
        plan,
        norm_weight=norm_weight,
        norm_eps=EPS,
        freq_cis=freqs_cis,
        out_loc=out_loc,
        kvcache=cache,
        page_size=PAGE_SIZE,
        use_fp4=False,
        bf16_store=False,
    )
    torch.cuda.synchronize()

    ref = _reference(kv, norm_weight, seq_lens, freqs_cis)
    nope_ref = ref[:, :NOPE_DIM]  # [N, 448]
    rope_ref = ref[:, NOPE_DIM:]  # [N, 64]

    # ---- Readback (strategy A: dequantize the fp8 bytes + UE8M0 group scales).
    # Payload region: [num_pages, PAGE_SIZE, 576] (448 fp8 nope + 128 bf16 rope).
    payload = (
        cache[:, : FP8_PAYLOAD_BYTES * PAGE_SIZE]
        .contiguous()
        .view(num_pages * PAGE_SIZE, FP8_PAYLOAD_BYTES)
    )
    # Scale region: 8 bytes/token right after the payload region.
    scale_off = FP8_PAYLOAD_BYTES * PAGE_SIZE
    scales = (
        cache[:, scale_off : scale_off + FP8_SCALE_BYTES * PAGE_SIZE]
        .contiguous()
        .view(num_pages * PAGE_SIZE, FP8_SCALE_BYTES)
    )
    slots = out_loc.long()
    seg = payload[slots]  # [N, 576]
    scale_bytes = scales[slots]  # [N, 8]

    fp8_nope = seg[:, :FP8_NOPE_BYTES].contiguous().view(torch.float8_e4m3fn).float()
    rope_bf16 = (
        seg[:, FP8_NOPE_BYTES : FP8_NOPE_BYTES + FP8_ROPE_BYTES]
        .contiguous()
        .view(torch.bfloat16)
        .float()
    )  # [N, 64]

    # UE8M0 dequant: scale = 2^(exp - 127); one exp per 64-elem nope group.
    exps = scale_bytes[:, :FP8_NUM_NOPE_GROUPS].to(torch.int32).float()  # [N, 7]
    group_scale = torch.pow(2.0, exps - 127.0).unsqueeze(-1)  # [N, 7, 1]
    deq_nope = (fp8_nope.view(-1, FP8_NUM_NOPE_GROUPS, 64) * group_scale).reshape(
        -1, NOPE_DIM
    )  # [N, 448]

    assert not torch.isnan(deq_nope).any(), "fp8 nope dequant produced NaN"
    assert not torch.isinf(deq_nope).any(), "fp8 nope dequant produced Inf"
    assert not torch.isnan(rope_bf16).any(), "fp8 rope tail produced NaN"
    assert not torch.isinf(rope_bf16).any(), "fp8 rope tail produced Inf"

    # nope: fp8-e4m3 precision (~1/16 relative). rope tail: plain bf16.
    torch.testing.assert_close(deq_nope, nope_ref, rtol=FP8_RTOL, atol=FP8_ATOL)
    torch.testing.assert_close(rope_bf16, rope_ref, rtol=BF16_RTOL, atol=BF16_ATOL)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
