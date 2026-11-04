"""Ownership, tail handling, graph reuse and attention for V32 token packing."""

import sys

import pytest
import torch

from sglang.srt.runtime_context import get_platform
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not get_platform().is_sm100,
    reason="token-packed FlashMLA requires Blackwell",
)


def _prepare(q, rope, indices):
    from sglang.kernels.ops.attention.dsa.flashmla_token_pack import (
        flashmla_token_pack_available,
    )

    if not flashmla_token_pack_available():
        pytest.skip("installed FlashMLA predates token packing")
    return torch.ops.sgl_kernel.flashmla_token_pack_prepare(q, rope, indices)


def _check_union(indices, packed_indices, lengths, tokens, group):
    rows = indices[:tokens].cpu().tolist()
    for first, actual, length in zip(
        range(0, tokens, group), packed_indices.cpu().tolist(), lengths.cpu().tolist()
    ):
        expected = {}
        for token, row in enumerate(rows[first : first + group]):
            for slot in row:
                if slot != -1:
                    expected[slot] = expected.get(slot, 0) | (1 << (26 + token))
        assert length == len(expected)
        assert sorted(actual[:length]) == sorted(k | v for k, v in expected.items())
        assert all(x == -1 for x in actual[length:])


@pytest.mark.parametrize(
    "heads,tokens,topk",
    [
        (1, 1, 1),
        (8, 13, 2048),
        (12, 0, 64),
        (12, 5, 2048),
        (12, 6, 2048),
        (16, 9, 64),
        (24, 5, 64),
        (32, 7, 2048),
    ],
)
@pytest.mark.parametrize("layout", ["combined", "split", "broadcast_rope"])
def test_prepare_ownership_and_q(heads, tokens, topk, layout):
    torch.manual_seed(41)
    q = torch.randn(tokens, heads, 576, device="cuda", dtype=torch.bfloat16)
    rope = None
    if layout != "combined":
        rope = q[:, :, 512:]
        q = q[:, :, :512]
        if layout == "broadcast_rope":
            rope = rope[:, :1].expand(-1, heads, -1)
    expected_q = q if rope is None else torch.cat((q, rope), dim=-1)
    # Strided rows, extra inactive rows, repeated slots and missing entries.
    backing = torch.randint(
        0, 3000, (tokens + 3, topk + 8), device="cuda", dtype=torch.int32
    )
    indices = backing[:, :topk]
    indices[:, ::7] = -1
    if tokens:
        indices[0, -1] = (1 << 26) - 1
    group = min(64 // heads, 5)
    packed_q, packed_indices, lengths = _prepare(q, rope, indices)
    unpacked_q = packed_q.view(-1, heads, 576)
    assert torch.equal(unpacked_q[:tokens], expected_q)
    assert torch.count_nonzero(unpacked_q[tokens:]).item() == 0
    if layout == "combined" and tokens % group == 0:
        assert packed_q.data_ptr() == q.data_ptr()
    _check_union(indices, packed_indices, lengths, tokens, group)


def test_collisions_and_graph_reuse():
    q = torch.randn(7, 12, 576, device="cuda", dtype=torch.bfloat16)
    # All slots collide at the initial bucket; token masks must still be exact.
    indices = (torch.arange(32, device="cuda", dtype=torch.int32) * 16384).repeat(7, 1)
    _prepare(q, None, indices)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        packed_q, packed_indices, lengths = _prepare(q, None, indices)
    for empty in (False, True, False):
        indices.copy_(
            (torch.arange(32, device="cuda", dtype=torch.int32) * 16384).repeat(7, 1)
        )
        indices[1::2] = -1
        if empty:
            indices.fill_(-1)
        q.normal_()
        graph.replay()
        torch.cuda.synchronize()
        _check_union(indices, packed_indices, lengths, 7, 5)
        assert torch.equal(packed_q.view(-1, 12, 576)[:7], q)


def test_device_guard():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two CUDA devices")
    torch.cuda.set_device(0)
    q = torch.randn(6, 12, 576, device="cuda:1", dtype=torch.bfloat16)
    rows = torch.arange(64, device="cuda:1", dtype=torch.int32).repeat(6, 1)
    packed_q, packed_indices, lengths = _prepare(q, None, rows)
    assert torch.cuda.current_device() == 0
    _check_union(rows, packed_indices, lengths, 6, 5)
    assert torch.equal(packed_q.view(-1, 12, 576)[:6], q)


@pytest.mark.parametrize(
    "heads,tokens,topk",
    [
        (8, 6, 2048),
        (12, 5, 2048),
        (12, 11, 2048),
        (16, 9, 256),
        (24, 7, 256),
        (32, 5, 2048),
    ],
)
@pytest.mark.parametrize("pattern", ["overlap", "disjoint", "empty"])
def test_attention(heads, tokens, topk, pattern):
    from sglang.kernels.ops.attention.dsa.flashmla_token_pack import (
        flashmla_kv_token_packed,
    )
    from sglang.kernels.ops.quantization.dsa_quant_k_cache import quantize_k_cache

    torch.manual_seed(42)
    slots = 16384
    q = torch.randn(tokens, heads, 576, device="cuda", dtype=torch.bfloat16)
    cache = quantize_k_cache(
        torch.randn(slots // 64, 64, 1, 576, device="cuda", dtype=torch.bfloat16),
        pow2_scale=True,
    )
    # Attention tests require distinct keys within each row; deduplication of
    # repeated valid keys would intentionally alter softmax multiplicity.
    rows = torch.arange(topk, device="cuda", dtype=torch.int32).repeat(tokens, 1)
    rows += torch.arange(tokens, device="cuda", dtype=torch.int32)[:, None] * (
        topk if pattern == "disjoint" else 17
    )
    rows.remainder_(slots)
    rows[1::3, topk // 2 :] = -1
    rows[0] = -1
    if pattern == "empty":
        rows.fill_(-1)
    packed_q, packed_indices, lengths = _prepare(q[:, :, :512], q[:, :, 512:], rows)

    def attend(query, idx, lens, group_heads):
        return torch.ops.sgl_kernel.sparse_decode_fwd.default(
            query,
            cache,
            idx.unsqueeze(1),
            lens,
            None,
            None,
            None,
            None,
            None,
            None,
            512,
            576**-0.5,
            "V32",
            group_heads,
        )[:2]

    reference, reference_lse = attend(
        q.unsqueeze(1), rows, (rows >= 0).sum(-1).int(), 0
    )
    actual, actual_lse = attend(packed_q, packed_indices, lengths, heads)
    actual = actual.view(-1, 1, heads, 512)[:tokens]
    actual_lse = actual_lse.reshape(-1, heads, 1)[:tokens]
    torch.testing.assert_close(actual, reference, atol=2e-3, rtol=1e-2)
    torch.testing.assert_close(actual_lse, reference_lse, atol=2e-4, rtol=2e-4)
    helper_out = flashmla_kv_token_packed(
        q[:, :, :512], q[:, :, 512:], cache, rows, 576**-0.5
    )
    torch.testing.assert_close(helper_out, reference, atol=2e-3, rtol=1e-2)


def test_packed_attention_graph():
    from sglang.kernels.ops.attention.dsa.flashmla_token_pack import (
        flashmla_kv_token_packed,
    )
    from sglang.kernels.ops.quantization.dsa_quant_k_cache import quantize_k_cache

    torch.manual_seed(43)
    q = torch.randn(7, 12, 576, device="cuda", dtype=torch.bfloat16)
    cache = quantize_k_cache(
        torch.randn(16, 64, 1, 576, device="cuda", dtype=torch.bfloat16)
    )
    rows = torch.arange(256, device="cuda", dtype=torch.int32).repeat(7, 1)
    _prepare(q, None, rows)
    flashmla_kv_token_packed(q, None, cache, rows, 576**-0.5)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = flashmla_kv_token_packed(q, None, cache, rows, 576**-0.5)
    for empty in (False, True, False):
        rows.copy_(torch.randperm(1024, device="cuda")[:256].int().expand(7, -1))
        rows[1::2, 32:] = -1
        if empty:
            rows.fill_(-1)
        q.normal_()
        graph.replay()
        reference = torch.ops.sgl_kernel.sparse_decode_fwd.default(
            q.unsqueeze(1),
            cache,
            rows.unsqueeze(1),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            512,
            576**-0.5,
            "V32",
            0,
        )[0]
        torch.testing.assert_close(actual, reference, atol=2e-3, rtol=1e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
