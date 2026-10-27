"""gfx950 NVFP4 sparse decode equals the bf16 kernel on the dequantized cache."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.minimax_sparse.common.nvfp4 import Nvfp4Cache
from sglang.kernels.ops.attention.minimax_sparse.decode import nvfp4_decode
from sglang.kernels.ops.attention.minimax_sparse.decode.topk_sparse import (
    flash_decode_with_gqa_share_sparse,
)
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.quant_ref_utils import nvfp4_kv_quantize

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

NQH, HD, BLK, TOPK = 16, 128, 128, 16
SEQ_LEN = 1024
# E2M1 code magnitudes; bit 3 is the sign.
E2M1_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _inputs(bs):
    slots = bs * SEQ_LEN
    q = torch.randn(bs, NQH, HD, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(2, slots, HD, dtype=torch.bfloat16, device="cuda")
    req_to_token = torch.empty(bs, SEQ_LEN, dtype=torch.int32, device="cuda")
    npages = SEQ_LEN // BLK
    for b in range(bs):
        pages = torch.randperm(npages, device="cuda") + b * npages
        page_slots = pages[:, None] * BLK + torch.arange(BLK, device="cuda")
        req_to_token[b] = page_slots.reshape(-1).to(torch.int32)
    seq_lens = SEQ_LEN - torch.randint(1, BLK, (bs,), dtype=torch.int32, device="cuda")
    for b in range(bs):
        # Unwritten slots hold garbage; NaN there makes any read of them show.
        kv[:, req_to_token[b, int(seq_lens[b]) :].long()] = float("nan")
    topk_idx = torch.full((1, bs, TOPK), -1, dtype=torch.int32, device="cuda")
    for b in range(bs):
        num_blocks = (int(seq_lens[b]) + BLK - 1) // BLK
        chosen = torch.randperm(num_blocks, device="cuda")[:TOPK]
        topk_idx[0, b, : chosen.numel()] = chosen.to(torch.int32)
    slot_ids = torch.arange(bs, dtype=torch.int64, device="cuda")
    return q, kv, req_to_token, seq_lens, slot_ids, topk_idx


# [slots, D] bf16 -> the pool-layout NVFP4 cache and its dequantized [slots, 1, D] values.
def _nvfp4(x):
    slots, dim = x.shape
    payload, scales = nvfp4_kv_quantize(x.contiguous(), torch.ones(1, device="cuda"))
    lut = torch.tensor(E2M1_MAG + [-m for m in E2M1_MAG], device="cuda")
    codes = torch.stack([payload & 0xF, payload >> 4], dim=-1).reshape(slots, dim)
    block_scales = scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
    dequant = (lut[codes.long()] * block_scales).to(torch.bfloat16)
    pages = slots // BLK
    cache = Nvfp4Cache(
        payload=payload.view(pages, 1, BLK, dim // 2),
        scales=scales.view(pages, 1, BLK, dim // 16),
    )
    return cache, dequant.view(slots, 1, dim)


@pytest.mark.skipif(not is_gfx95_supported(), reason="NVFP4 K/V decode is gfx950 only")
# Each request's top-k is 8 blocks: 16 requests get one block per chunk, 64 requests 4,
# and 512 requests run the whole top-k as one chunk.
@pytest.mark.parametrize("bs", [16, 64, 512])
def test_nvfp4_decode_matches_dequantized_cache(bs):
    torch.manual_seed(0)
    q, kv, req_to_token, seq_lens, slot_ids, topk_idx = _inputs(bs)
    (k_nvfp4, k_ref), (v_nvfp4, v_ref) = _nvfp4(kv[0]), _nvfp4(kv[1])
    assert nvfp4_decode._use_nvfp4_decode_gfx950(q, k_nvfp4, v_nvfp4, topk_idx, BLK)
    common = dict(
        q=q,
        sink=None,
        req_to_token=req_to_token,
        seq_lens=seq_lens,
        slot_ids=slot_ids,
        block_size=BLK,
        topk_idx=topk_idx,
    )
    out = flash_decode_with_gqa_share_sparse(
        k_cache=k_nvfp4, v_cache=v_nvfp4, page_size=BLK, **common
    )
    ref = flash_decode_with_gqa_share_sparse(
        k_cache=k_ref, v_cache=v_ref, page_size=0, **common
    )
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-3, rtol=2e-2)


@pytest.mark.skipif(not is_gfx95_supported(), reason="NVFP4 K/V decode is gfx950 only")
# 512 requests run the whole top-k as one chunk; 128 / 64 / 32 / 16 requests merge
# 2 / 4 / 8 / 16 chunks.
@pytest.mark.parametrize("bs", [16, 32, 64, 128, 512])
@pytest.mark.parametrize("out_quant", ["fp8", "fp8_per_token"])
def test_gluon_decode_hands_o_proj_the_separate_quant_bytes(bs, out_quant):
    """The o_proj input quantized in the Gluon epilogue or chunk merge must be the
    bytes and scale layout of the aiter quant the o_proj would otherwise run on the
    bf16 output."""
    import aiter

    from sglang.kernels.ops.attention.minimax_sparse.decode.nvfp4_decode_gfx950 import (
        flash_decode_nvfp4_gfx950,
    )
    from sglang.srt.layers.quantization.fp8_utils import (
        view_aiter_fused_rms_transposed_fp8_scale,
    )

    torch.manual_seed(0)
    q, kv, req_to_token, seq_lens, slot_ids, topk_idx = _inputs(bs)
    (k_nvfp4, _), (v_nvfp4, _) = _nvfp4(kv[0]), _nvfp4(kv[1])
    args = (q, k_nvfp4, v_nvfp4, req_to_token, seq_lens, slot_ids, BLK, topk_idx)
    o = flash_decode_nvfp4_gfx950(*args, sm_scale=HD**-0.5, v_scale=1.0)
    got = flash_decode_nvfp4_gfx950(
        *args, sm_scale=HD**-0.5, v_scale=1.0, out_quant=out_quant
    )
    if bs == 16:
        # Sixteen merged chunks: O stays bf16 and the o_proj quantizes it.
        assert torch.equal(got, o)
        return
    x = o.reshape(bs, -1)
    if out_quant == "fp8":
        quant = aiter.get_hip_quant(aiter.QuantType.per_1x128)
        ref_q, ref_s = quant(x, quant_dtype=aiter.dtypes.fp8, transpose_scale=True)
        ref_s = view_aiter_fused_rms_transposed_fp8_scale(ref_s)
    else:
        ref_q, ref_s = aiter.per_token_quant_hip(x, quant_dtype=aiter.dtypes.fp8)
    q8, scale = got
    assert torch.equal(q8.view(torch.uint8), ref_q.view(torch.uint8))
    assert scale.shape == ref_s.shape and scale.stride() == ref_s.stride()
    assert torch.equal(scale, ref_s)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
