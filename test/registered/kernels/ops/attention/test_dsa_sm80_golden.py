"""Cross-backend golden checks for the architecture-independent Torch DSA path.

These tests intentionally run on Hopper-or-newer CI because they use the existing
DeepGEMM/TileLang implementations only to produce goldens. The Torch functions
themselves are exercised on SM80 by ``test_dsa_sm80_reference.py`` and never
depend on either golden backend at runtime.
"""

from __future__ import annotations

import pytest
import torch

from sglang.kernels.ops.attention.dsa.paged_mqa_logits import (
    deepgemm_paged_mqa_logits_split,
)
from sglang.kernels.ops.attention.dsa.torch_sparse_mla import torch_sparse_mla
from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
from sglang.srt.layers.attention.dsa.torch_paged_mqa_logits import (
    torch_paged_mqa_logits,
)
from sglang.srt.utils import is_sm90_supported, is_sm100_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="nightly", runner_config="1-gpu-h100")

PAGE_SIZE = 64
INDEX_DIM = 128
VALUE_DIM = 512
ROPE_DIM = 64


def _pack(keys: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    pages = keys.shape[0]
    packed = torch.zeros(
        pages,
        PAGE_SIZE * (INDEX_DIM + 4),
        dtype=torch.uint8,
        device=keys.device,
    )
    split = PAGE_SIZE * INDEX_DIM
    packed[:, :split] = keys.view(torch.uint8).reshape(pages, -1)
    packed[:, split:] = scales.contiguous().view(torch.uint8).reshape(pages, -1)
    return packed.view(pages, PAGE_SIZE, 1, INDEX_DIM + 4)


@pytest.mark.skipif(
    not (is_sm90_supported() or is_sm100_supported()),
    reason="DeepGEMM paged-MQA golden generation requires SM90 or SM100",
)
def test_torch_paged_mqa_matches_deepgemm_golden_and_topk():
    import deep_gemm

    torch.manual_seed(20260818)
    batch_size, num_heads, max_seq_len = 2, 32, 192
    q = torch.randn(batch_size, num_heads, INDEX_DIM, device="cuda").to(
        torch.float8_e4m3fn
    )
    keys = torch.randn(4, PAGE_SIZE, INDEX_DIM, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.rand(4, PAGE_SIZE, device="cuda", dtype=torch.float32) + 0.25
    cache = _pack(keys, scales)
    weights = torch.randn(batch_size, num_heads, device="cuda", dtype=torch.float32)
    lengths = torch.tensor([129, 65], device="cuda", dtype=torch.int32)
    page_table = torch.tensor([[3, 0, 3], [2, 1, -1]], device="cuda", dtype=torch.int32)

    schedule = deep_gemm.get_paged_mqa_logits_metadata(
        lengths.unsqueeze(-1), PAGE_SIZE, deep_gemm.get_num_sms()
    )
    golden = deepgemm_paged_mqa_logits_split(
        deep_gemm.fp8_paged_mqa_logits,
        q,
        cache,
        weights,
        lengths.unsqueeze(-1),
        page_table,
        schedule,
        max_seq_len,
        q_offset=batch_size,
    )
    actual = torch_paged_mqa_logits(q, cache, weights, lengths, page_table, max_seq_len)

    positions = torch.arange(max_seq_len, device="cuda").unsqueeze(0)
    valid = positions < lengths.unsqueeze(1)
    torch.testing.assert_close(
        actual.masked_select(valid),
        golden.masked_select(valid),
        atol=5e-3,
        rtol=5e-4,
    )
    actual_topk = DSATopKBackend.TORCH.topk_func(actual, lengths, 32)
    golden_topk = DSATopKBackend.TORCH.topk_func(golden, lengths, 32)
    torch.testing.assert_close(
        torch.sort(actual_topk, dim=-1).values,
        torch.sort(golden_topk, dim=-1).values,
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(
    not is_sm90_supported(),
    reason="TileLang sparse-MLA golden is registered on the SM90 nightly runner",
)
def test_torch_sparse_mla_matches_tilelang_golden():
    from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd

    torch.manual_seed(20260819)
    num_queries, num_heads, topk = 2, 8, 2048
    q_nope = torch.randn(
        num_queries, num_heads, VALUE_DIM, device="cuda", dtype=torch.bfloat16
    )
    q_rope = torch.randn(
        num_queries, num_heads, ROPE_DIM, device="cuda", dtype=torch.bfloat16
    )
    kv = torch.randn(
        topk + 16,
        1,
        VALUE_DIM + ROPE_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    indices = torch.arange(topk, device="cuda", dtype=torch.int32).repeat(
        num_queries, 1
    )
    indices[1, -31:] = -1
    scale = 0.125

    golden = tilelang_sparse_fwd(
        q=torch.cat([q_nope, q_rope], dim=-1),
        kv=kv,
        indices=indices.unsqueeze(1),
        sm_scale=scale,
        d_v=VALUE_DIM,
    )
    actual = torch_sparse_mla(
        q_nope,
        q_rope,
        kv,
        indices,
        scale,
        query_chunk_size=1,
        topk_chunk_size=128,
    )

    torch.testing.assert_close(actual.float(), golden.float(), atol=3e-2, rtol=3e-2)
