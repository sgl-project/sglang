from __future__ import annotations

import sys

import pytest
import torch

from sglang.jit_kernel.dsv4 import (
    CompressorDecodePlan,
    compress_norm_rope_store,
    fused_q_indexer_rope_hadamard_fp4_quant,
)
from sglang.jit_kernel.dsv4.fused_norm_rope_v2_torch import _fwht128
from sglang.jit_kernel.hadamard import hadamard_transform
from sglang.kernels.ops.attention.deepseek_v4_rope import (
    apply_rotary_emb_triton,
    precompute_freqs_cis,
)
from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
    quantize_fp4_indexer_tensor,
    store_fp4_index_k_cache,
)
from sglang.srt.utils import get_device, is_cuda
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_is_cuda = is_cuda()

HEAD_DIM = 128
FP4_DIM = HEAD_DIM // 2
GROUP_SIZE = 32
SCALE_GROUPS = HEAD_DIM // GROUP_SIZE
SCALE_BYTES = 4
PAGE_SIZE = 64
E2M1_MAX = 6.0


def _hadamard_ref(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Device-agnostic Hadamard reference.

    On CUDA use the JIT kernel; elsewhere (e.g. XPU) use the torch-native 128-pt
    WHT the fallback itself uses, scaled to match the kernel's ``scale`` arg.
    """
    if _is_cuda:
        return hadamard_transform(x.contiguous(), scale=scale)
    return _fwht128(x.float().contiguous()) * scale


def _ceil_ue8m0_exp_ref(x: torch.Tensor) -> torch.Tensor:
    bits = x.to(torch.float32).contiguous().view(torch.int32)
    exp = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    exp = exp + (mantissa != 0).to(torch.int32)
    return exp.clamp(1, 254)


def _fp4_e2m1_code_ref(x: torch.Tensor) -> torch.Tensor:
    ax = torch.minimum(x.abs(), torch.tensor(E2M1_MAX, device=x.device))
    idx = torch.zeros_like(ax, dtype=torch.uint8)
    for threshold in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        idx += (ax > threshold).to(torch.uint8)
    sign = ((x < 0) & (idx != 0)).to(torch.uint8) * 8
    return idx | sign


def _ref_quantize_fp4_indexer(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.contiguous().view(-1, HEAD_DIM).float()
    groups = x.view(-1, SCALE_GROUPS, GROUP_SIZE)
    scale_raw = (groups.abs().amax(dim=-1) / E2M1_MAX).clamp_min(1.0e-4)
    scale_exp = _ceil_ue8m0_exp_ref(scale_raw)
    scale = (scale_exp << 23).contiguous().view(torch.float32)

    scaled = (groups / scale.unsqueeze(-1)).view(-1, HEAD_DIM)
    code = _fp4_e2m1_code_ref(scaled)
    packed = (code[:, 0::2].to(torch.int16) | (code[:, 1::2].to(torch.int16) << 4)).to(
        torch.uint8
    )

    packed_sf = scale_exp[:, 0].clone()
    for group_id in range(1, SCALE_GROUPS):
        packed_sf |= scale_exp[:, group_id] << (8 * group_id)
    return packed, packed_sf


def _ref_store_fp4_index_cache(
    x_fp4: torch.Tensor,
    x_sf: torch.Tensor,
    loc: torch.Tensor,
    num_pages: int,
) -> torch.Tensor:
    expected = torch.zeros(
        num_pages,
        PAGE_SIZE * (FP4_DIM + SCALE_BYTES),
        device=x_fp4.device,
        dtype=torch.uint8,
    )
    sf_shifts = torch.arange(0, 32, 8, device=x_fp4.device, dtype=torch.int32)
    for token_id in range(x_fp4.shape[0]):
        cache_loc = int(loc[token_id].item())
        page = cache_loc // PAGE_SIZE
        offset = cache_loc % PAGE_SIZE
        expected[page, offset * FP4_DIM : (offset + 1) * FP4_DIM] = x_fp4[token_id]
        sf_start = PAGE_SIZE * FP4_DIM + offset * SCALE_BYTES
        expected[page, sf_start : sf_start + SCALE_BYTES] = (
            (x_sf[token_id] >> sf_shifts) & 0xFF
        ).to(torch.uint8)
    return expected


@pytest.mark.parametrize("num_tokens", [1, 7, 96])
def test_quantize_fp4_indexer_tensor(num_tokens: int) -> None:
    torch.manual_seed(num_tokens)
    x = torch.randn(num_tokens, HEAD_DIM, device=get_device(), dtype=torch.bfloat16)
    x[0, :8] = torch.tensor(
        [-8.0, -6.0, -3.0, -1.5, 0.0, 0.5, 2.0, 8.0],
        device=get_device(),
        dtype=torch.bfloat16,
    )

    x_fp4, x_sf = quantize_fp4_indexer_tensor(x)
    ref_fp4, ref_sf = _ref_quantize_fp4_indexer(x)

    torch.testing.assert_close(x_fp4.view(torch.uint8), ref_fp4)
    torch.testing.assert_close(x_sf, ref_sf)


@pytest.mark.parametrize("num_tokens", [1, 16, 96])
def test_fp4_index_cache_store_layout(num_tokens: int) -> None:
    torch.manual_seed(num_tokens)
    num_pages = max(1, (num_tokens + PAGE_SIZE - 1) // PAGE_SIZE)
    x = torch.randn(num_tokens, HEAD_DIM, device=get_device(), dtype=torch.bfloat16)
    loc = torch.randperm(num_pages * PAGE_SIZE, device=get_device())[:num_tokens].to(
        torch.int64
    )
    cache = torch.zeros(
        num_pages,
        PAGE_SIZE * (FP4_DIM + SCALE_BYTES),
        device=get_device(),
        dtype=torch.uint8,
    )

    store_fp4_index_k_cache(x, cache, loc, page_size=PAGE_SIZE)

    ref_fp4, ref_sf = _ref_quantize_fp4_indexer(x)
    expected = _ref_store_fp4_index_cache(ref_fp4, ref_sf, loc, num_pages)
    torch.testing.assert_close(cache, expected)


@pytest.mark.parametrize("num_tokens", [1, 16, 96])
def test_fp4_fused_norm_rope_store_layout(num_tokens: int) -> None:
    torch.manual_seed(num_tokens + 100)
    num_pages = max(1, (num_tokens + PAGE_SIZE - 1) // PAGE_SIZE)
    compress_ratio = 4
    kv = torch.randn(num_tokens, HEAD_DIM, device=get_device(), dtype=torch.bfloat16)
    norm_weight = torch.randn(HEAD_DIM, device=get_device(), dtype=torch.bfloat16)
    seq_lens = (
        torch.arange(1, num_tokens + 1, device=get_device(), dtype=torch.int64)
        * compress_ratio
    )
    req_pool_indices = torch.arange(num_tokens, device=get_device(), dtype=torch.int64)
    plan = CompressorDecodePlan.generate_legacy(
        compress_ratio, req_pool_indices, seq_lens
    )
    loc = torch.arange(num_tokens, device=get_device(), dtype=torch.int64)
    freqs_cis = precompute_freqs_cis(
        64, int(seq_lens.max().item()) + 1, 0, 10000, 1, 32, 1
    ).to(get_device())
    cache = torch.zeros(
        num_pages,
        PAGE_SIZE * (FP4_DIM + SCALE_BYTES),
        device=get_device(),
        dtype=torch.uint8,
    )

    compress_norm_rope_store(
        kv.clone(),
        plan,
        norm_weight=norm_weight,
        norm_eps=1.0e-6,
        freq_cis=freqs_cis,
        out_loc=loc,
        kvcache=cache,
        page_size=PAGE_SIZE,
        use_fp4=True,
    )

    ref = kv.float()
    ref = ref * torch.rsqrt((ref * ref).sum(dim=-1, keepdim=True) / HEAD_DIM + 1.0e-6)
    ref = ref * norm_weight.float()
    freqs = torch.view_as_real(freqs_cis).flatten(-2)[
        (seq_lens - compress_ratio).long()
    ]
    rope = ref[:, 64:].reshape(num_tokens, 32, 2)
    freqs = freqs.reshape(num_tokens, 32, 2)
    rope_out = torch.empty_like(rope)
    rope_out[..., 0] = rope[..., 0] * freqs[..., 0] - rope[..., 1] * freqs[..., 1]
    rope_out[..., 1] = rope[..., 0] * freqs[..., 1] + rope[..., 1] * freqs[..., 0]
    ref[:, 64:] = rope_out.reshape(num_tokens, 64)
    ref = _hadamard_ref(ref, scale=HEAD_DIM**-0.5)
    ref_fp4, ref_sf = _ref_quantize_fp4_indexer(ref)

    expected = _ref_store_fp4_index_cache(
        ref_fp4,
        ref_sf,
        loc.to(torch.int64),
        num_pages,
    )
    torch.testing.assert_close(cache, expected)


@pytest.mark.skipif(
    not _is_cuda,
    reason="fused_q_indexer_rope_hadamard_fp4_quant is JIT-CUDA-only (no torch fallback)",
)
@pytest.mark.parametrize("batch_size", [1, 5, 17])
def test_fp4_fused_q_indexer_rope_hadamard_quant(batch_size: int) -> None:
    torch.manual_seed(batch_size + 200)
    num_heads = 8
    rope_dim = 64
    weight_scale = HEAD_DIM**-0.5 * num_heads**-0.5
    q = torch.randn(
        batch_size, num_heads, HEAD_DIM, device=get_device(), dtype=torch.bfloat16
    )
    weight = torch.randn(
        batch_size, num_heads, device=get_device(), dtype=torch.bfloat16
    )
    positions = (
        torch.arange(batch_size, device=get_device(), dtype=torch.int32) * 7
    ) % 63
    freqs_cis = precompute_freqs_cis(rope_dim, 64, 0, 10000, 1, 32, 1).to(get_device())

    (q_fp4, q_sf), weights_out = fused_q_indexer_rope_hadamard_fp4_quant(
        q, weight, weight_scale, freqs_cis, positions
    )

    ref = q.clone()
    apply_rotary_emb_triton(ref[..., -rope_dim:], freqs_cis, positions=positions)
    ref = _hadamard_ref(ref, scale=HEAD_DIM**-0.5)
    ref_fp4, ref_sf = _ref_quantize_fp4_indexer(ref.view(-1, HEAD_DIM))
    ref_fp4 = ref_fp4.view(batch_size, num_heads, FP4_DIM)
    ref_sf = ref_sf.view(batch_size, num_heads)

    torch.testing.assert_close(q_fp4.view(torch.uint8), ref_fp4)
    torch.testing.assert_close(q_sf, ref_sf)
    torch.testing.assert_close(weights_out.squeeze(-1), weight.float() * weight_scale)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
