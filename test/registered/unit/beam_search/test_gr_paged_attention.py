"""Paged addressing, offset causality, ancestry isolation and local-head TP."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.beam_search.paged_kv import GRPagedForwardMetadata
from sglang.srt.layers.attention.gr_paged_backend import (
    GRPagedAttentionBackend,
    _torch_attention,
    torch_decode_attention,
    torch_prefill_attention,
    validate_gr_paged_configuration,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def dense_attention(q, k, v, scale, offset=None):
    """Independent FP64 oracle, with explicit per-query/per-head legal keys."""
    result = torch.empty((*q.shape[:2], v.shape[-1]), dtype=q.dtype)
    groups = q.shape[1] // k.shape[1]
    for row in range(q.shape[0]):
        end = k.shape[0] if offset is None else offset + row + 1
        for head in range(q.shape[1]):
            key = k[:end, head // groups].double()
            value = v[:end, head // groups].double()
            scores = torch.mv(key, q[row, head].double()) * scale
            result[row, head] = torch.softmax(scores, 0) @ value
    return result


def layout(context_k, context_v, beam_k, beam_v, page_size, ancestry):
    """Scatter known logical KV onto nonadjacent pages; poison padding."""
    context_len, kv_heads, dim = context_k.shape
    steps, width = beam_k.shape[:2]
    context_page_count = (context_len + page_size - 1) // page_size
    step_page_count = (width + page_size - 1) // page_size
    page_ids = torch.arange(1, 2 * (context_page_count + steps * step_page_count), 2)
    context_pages = page_ids[:context_page_count]
    beam_pages = page_ids[context_page_count:].reshape(steps, step_page_count)
    pool_shape = (int(page_ids[-1] + 1) * page_size, kv_heads, dim)
    key_pool = torch.full(pool_shape, torch.nan, dtype=context_k.dtype)
    value_pool = torch.full_like(key_pool, torch.nan)
    positions = torch.arange(context_len)
    context_locs = (
        context_pages[positions // page_size] * page_size + positions % page_size
    )
    key_pool[context_locs] = context_k
    value_pool[context_locs] = context_v
    columns = torch.arange(width)
    for step in range(steps):
        slots = beam_pages[step, columns // page_size] * page_size + columns % page_size
        key_pool[slots] = beam_k[step]
        value_pool[slots] = beam_v[step]
    metadata = GRPagedForwardMetadata(
        page_size=page_size,
        context_len=context_len,
        beam_width=width,
        step=steps - 1,
        context_locs=context_locs,
        context_pages=context_pages,
        beam_pages=beam_pages,
        topk_indices=ancestry,
        out_cache_loc=slots,
    )
    return key_pool, value_pool, metadata


class TestGRPagedAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_offset_causal_extension(self):
        # Suffix query 0 is absolute position 3, not position 0. The later
        # value 1000 must remain invisible to it, including on a partial page.
        q = torch.zeros(2, 4, 2)
        key_pool = torch.full((32, 1, 2), torch.nan)
        value_pool = torch.full_like(key_pool, torch.nan)
        pages = torch.tensor([3, 6])
        slots = torch.tensor([12, 13, 14, 15, 24])
        key_pool[slots] = 0
        values = torch.tensor([1, 2, 3, 4, 1000.0])
        value_pool[slots] = values[:, None, None]
        actual = torch_prefill_attention(q, key_pool, value_pool, pages, 5, 4, 3, 1.0)
        expected = torch.tensor([2.5, 202.0])[:, None, None].expand_as(actual)
        torch.testing.assert_close(actual, expected)

    def test_shared_parent_without_sibling_or_padding_visibility(self):
        context = torch.zeros(3, 1, 2)
        beam_k = torch.zeros(2, 2, 1, 2)
        beam_v = torch.tensor([[2.0, 1000.0], [4.0, 8.0]])[:, :, None, None].expand_as(
            beam_k
        )
        ancestry = torch.tensor([[0, 0], [0, 1]])
        key, value, meta = layout(context, context, beam_k, beam_v, 4, ancestry)
        actual = torch_decode_attention(torch.zeros(2, 4, 2), key, value, meta, 1.0)
        expected = torch.tensor([1.2, 2.0])[:, None, None].expand_as(actual)
        torch.testing.assert_close(actual, expected)

    def test_pages_and_tp_are_exact_with_fixed_arithmetic(self):
        for width in (2, 5, 17):
            with self.subTest(width=width):
                context_k = torch.randn(35, 4, 8).bfloat16()
                context_v = torch.randn_like(context_k)
                beam_k = torch.randn(3, width, 4, 8).bfloat16()
                beam_v = torch.randn_like(beam_k)
                q = torch.randn(width, 16, 8).bfloat16()
                ancestry = torch.stack(
                    (
                        torch.arange(width) % 2,
                        torch.zeros(width, dtype=torch.long),
                        torch.arange(width),
                    ),
                    1,
                )
                oracle = None
                for page_size in (1, 4, 16, 32):
                    key, value, meta = layout(
                        context_k, context_v, beam_k, beam_v, page_size, ancestry
                    )
                    actual = torch_decode_attention(q, key, value, meta, 8**-0.5)
                    if oracle is None:
                        oracle = actual
                    torch.testing.assert_close(actual, oracle, atol=0, rtol=0)
                    for tp in (2, 4):
                        outputs = [
                            torch_decode_attention(qpart, kpart, vpart, meta, 8**-0.5)
                            for qpart, kpart, vpart in zip(
                                q.chunk(tp, 1), key.chunk(tp, 1), value.chunk(tp, 1)
                            )
                        ]
                        torch.testing.assert_close(
                            torch.cat(outputs, 1), oracle, atol=0, rtol=0
                        )
                for beam in range(width):
                    k = torch.cat((context_k, beam_k[torch.arange(3), ancestry[beam]]))
                    v = torch.cat((context_v, beam_v[torch.arange(3), ancestry[beam]]))
                    expected = dense_attention(q[beam : beam + 1], k, v, 8**-0.5)
                    torch.testing.assert_close(
                        oracle[beam : beam + 1], expected, atol=0.002, rtol=0.008
                    )

    def test_prefill_matches_dense_oracle_across_query_blocks(self):
        q = torch.randn(37, 8, 8)
        k = torch.randn(42, 2, 8)
        v = torch.randn_like(k)
        expected = dense_attention(q, k, v, 8**-0.5, offset=5)
        actual = _torch_attention(q, k, v, 8**-0.5, query_offset=5)
        torch.testing.assert_close(actual, expected, atol=3e-7, rtol=3e-6)

    def test_prefill_and_decode_consume_previously_written_kv(self):
        context_k = torch.zeros(3, 1, 2)
        context_v = torch.tensor([2.0, 4.0, 6.0])[:, None, None].expand_as(context_k)
        beam_k = torch.zeros(1, 2, 1, 2)
        beam_v = torch.tensor([8.0, 10.0])[None, :, None, None].expand_as(beam_k)
        key, value, meta = layout(
            context_k, context_v, beam_k, beam_v, 16, torch.tensor([[0], [1]])
        )
        backend = GRPagedAttentionBackend.__new__(GRPagedAttentionBackend)
        backend.page_size = 16
        backend.token_to_kv_pool = SimpleNamespace(
            get_kv_buffer=Mock(return_value=(key, value)), set_kv_buffer=Mock()
        )
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=4,
            head_dim=2,
            scaling=1.0,
            is_cross_attention=False,
            logit_cap=0,
            sliding_window_size=-1,
        )
        for decode in (False, True):
            backend.forward_metadata = (
                meta
                if decode
                else SimpleNamespace(
                    context_pages=meta.context_pages, context_len=3, prefix_len=0
                )
            )
            expected = torch.tensor([5.0, 5.5] if decode else [2.0, 3.0, 4.0])
            q = torch.zeros(len(expected), 8)
            # Absent or stale input K/V must not replace a fused writer's data.
            for kv in (None, torch.full((len(expected), 1, 2), torch.nan)):
                with self.subTest(decode=decode, missing_kv=kv is None):
                    forward = (
                        backend.forward_decode if decode else backend.forward_extend
                    )
                    actual = forward(
                        q, kv, kv, layer, SimpleNamespace(), save_kv_cache=False
                    )
                    torch.testing.assert_close(actual, expected[:, None].expand_as(q))
        backend.token_to_kv_pool.set_kv_buffer.assert_not_called()

    def test_configuration_scope(self):
        base = dict(
            enable_gr=True,
            attention_backend="gr_paged",
            page_size=16,
            disable_overlap_schedule=True,
            chunked_prefill_size=-1,
            cuda_graph_backend_decode="disabled",
            cuda_graph_backend_prefill="disabled",
        )
        for tp in (1, 2, 4):
            validate_gr_paged_configuration({**base, "tp_size": tp})
        for key, value in dict(
            enable_gr=False,
            page_size=0,
            tp_size=8,
            pp_size=2,
            dcp_size=2,
            attn_cp_size=2,
            enable_hierarchical_cache=True,
            enable_lora=True,
            chunked_prefill_size=256,
            disable_overlap_schedule=False,
            decode_attention_backend="flashinfer",
            cuda_graph_backend_prefill="full",
            quantization="fp8",
        ).items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_gr_paged_configuration({**base, key: value})


if __name__ == "__main__":
    unittest.main()
