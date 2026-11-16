"""Tree cascade dispatch, suffix output/LSE layout, and CUDA Graph replay."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
from sglang.srt.layers.attention import flashattention_backend as fa_backend
from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")


class TestSuffixLayout(CustomTestCase):
    def test_layout_and_modifier_fallback(self):
        backend = FlashAttentionBackend.__new__(FlashAttentionBackend)
        backend.num_splits = 1
        metadata = FlashAttentionMetadata(
            page_table=torch.zeros(3, 4, dtype=torch.int32),
            cache_seqlens_int32=torch.ones(3, dtype=torch.int32),
            cu_seqlens_q=torch.arange(4, dtype=torch.int32),
            cu_seqlens_k=torch.arange(4, dtype=torch.int32),
        )
        backend.forward_metadata_spec_decode_expand = metadata
        q = torch.randn(3, 2, 64)
        k = v = torch.randn(4, 1, 1, 64)
        output, lse, scratch = torch.randn_like(q), torch.randn(2, 3), object()
        layer = SimpleNamespace(scaling=0.125, logit_cap=0.0)
        for version, kwargs in ((4, {}), (3, {}), (4, {"sinks": torch.zeros(2)})):
            with self.subTest(version=version, modifiers=tuple(kwargs)):
                fixed = version == 4 and not kwargs
                backend.flash_attn_with_kvcache = Mock(
                    return_value=(
                        output.unsqueeze(1) if fixed else output,
                        lse.T.unsqueeze(-1) if fixed else lse,
                        scratch,
                    )
                )
                actual, actual_lse, rest = backend._forward_suffix_attention(
                    q, k, v, layer, (-1, -1), kwargs, version
                )
                torch.testing.assert_close(actual, output)
                torch.testing.assert_close(actual_lse, lse)
                self.assertIs(rest, scratch)
                args = backend.flash_attn_with_kvcache.call_args.kwargs
                self.assertEqual(args["q"].shape, (3, 1, 2, 64) if fixed else q.shape)
                self.assertIs(
                    args["cu_seqlens_q"], None if fixed else metadata.cu_seqlens_q
                )
                for name, value in kwargs.items():
                    self.assertIs(args[name], value)


@unittest.skipUnless(
    torch.cuda.is_available() and get_device_sm() >= 90, "requires SM90+"
)
class TestTreeAttention(CustomTestCase):
    def _case(
        self,
        ancestors,
        version,
        *,
        mode="verify",
        page_size=1,
        dtype=torch.bfloat16,
        fused=False,
    ):
        torch.manual_seed(42)
        heads, kv_heads, dim = 4, 2, 64
        prefix_lengths = (0, 3, 65)
        batch, queries = len(prefix_lengths), len(ancestors)
        width = max(max(row) for row in ancestors) + 1
        depth = max(map(len, ancestors))
        pages_per_request = (max(prefix_lengths) + width + page_size - 1) // page_size
        pages = (
            torch.randperm(batch * pages_per_request, device="cuda")
            .int()
            .view(batch, -1)
        )
        k = torch.randn(
            batch * pages_per_request,
            page_size,
            kv_heads,
            dim,
            device="cuda",
            dtype=dtype,
        )
        v = torch.randn_like(k)
        # Match a query split from a fused QKV projection, including its row stride.
        q = torch.randn(
            batch * queries, heads + 2 * kv_heads, dim, device="cuda", dtype=dtype
        )[:, :heads]
        suffix_table = torch.full(
            (batch * queries, width), -1, device="cuda", dtype=torch.int32
        )
        suffix_lengths = torch.tensor(
            [len(row) for row in ancestors] * batch, device="cuda", dtype=torch.int32
        )
        visible_slots = []
        for b, prefix_length in enumerate(prefix_lengths):
            positions = torch.arange(prefix_length + width, device="cuda")
            slots = pages[b, positions // page_size] * page_size + positions % page_size
            for t, row_ancestors in enumerate(ancestors):
                suffix = slots[
                    prefix_length + torch.tensor(row_ancestors, device="cuda")
                ]
                suffix_table[b * queries + t, : len(suffix)] = suffix
                visible_slots.append(
                    suffix
                    if mode == "suffix"
                    else torch.cat((slots[:prefix_length], suffix))
                )

        def metadata(table, lengths, queries_per_request):
            return FlashAttentionMetadata(
                page_table=table,
                cache_seqlens_int32=lengths,
                cu_seqlens_q=torch.arange(
                    0,
                    batch * queries + 1,
                    queries_per_request,
                    device="cuda",
                    dtype=torch.int32,
                ),
                cu_seqlens_k=torch.cat((lengths.new_zeros(1), lengths.cumsum(0).int())),
                max_seq_len_q=queries_per_request,
            )

        backend = FlashAttentionBackend.__new__(FlashAttentionBackend)
        backend.fa_impl_ver = version
        backend.use_mla = backend.fa_skip_kv_cache = False
        backend.use_sliding_window_kv_pool = backend.is_prefill_aware_swa = False
        backend._decode_uses_static_max_seqlen_k = False
        backend.local_attn_builder = backend.full_cg_prefill_metadata = None
        backend.num_splits = backend.decode_num_splits = 1
        backend.topk = queries
        backend.flash_attn_with_kvcache = flash_attn_with_kvcache
        backend.get_paged_mha_kv_cache = lambda layer: (k, v)
        backend.prepare_paged_mha_query = lambda q, *args, **kwargs: (
            q,
            None,
            None,
            None,
            None,
        )
        backend._mxfp8_sf_kwargs = lambda *args: {}
        backend.forward_metadata = metadata(
            pages,
            torch.tensor(prefix_lengths, device="cuda", dtype=torch.int32),
            queries,
        )
        backend.forward_metadata_spec_decode_expand = metadata(
            suffix_table, suffix_lengths, 1
        )
        layer = RadixAttention(heads, dim, dim**-0.5, num_kv_heads=kv_heads, layer_id=0)
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE
            if mode == "draft"
            else ForwardMode.TARGET_VERIFY,
            batch_size=batch,
            spec_algorithm=SpeculativeAlgorithm.EAGLE3,
            spec_info=SimpleNamespace(max_tree_depth=depth),
        )

        def run():
            if mode == "suffix":
                return backend._forward_suffix_attention(
                    q, k, v, layer, (-1, -1), {}, version
                )
            forward = (
                backend.forward_decode if mode == "draft" else backend.forward_extend
            )
            return (forward(q, None, None, layer, forward_batch).view_as(q),)

        def check(result):
            outputs, lses = [], []
            head_indices = torch.arange(heads, device="cuda") // (heads // kv_heads)
            for row, slots in enumerate(visible_slots):
                keys = k.flatten(0, 1)[slots.long()][:, head_indices].float()
                values = v.flatten(0, 1)[slots.long()][:, head_indices].float()
                scores = (
                    torch.einsum("hd,lhd->hl", q[row].float(), keys) * layer.scaling
                )
                outputs.append(torch.einsum("hl,lhd->hd", scores.softmax(-1), values))
                lses.append(scores.logsumexp(-1))
            tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-3
            torch.testing.assert_close(
                result[0].float(), torch.stack(outputs), atol=tolerance, rtol=tolerance
            )
            if mode == "suffix":
                torch.testing.assert_close(
                    result[1], torch.stack(lses).T, atol=1e-4, rtol=1e-4
                )

        with patch.object(fa_backend, "is_cp_active", return_value=False):
            with patch.object(
                fa_backend,
                "merge_suffix_attention_in_place",
                wraps=fa_backend.merge_suffix_attention_in_place,
            ) as merge:
                check(run())
                self.assertEqual(merge.call_count, int(fused))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = run()
            q.copy_(torch.randn_like(q))
            k.normal_()
            v.normal_()
            # Replay with new KV contents and a different suffix for each query.
            for row, slots in enumerate(visible_slots):
                suffix_table[row, 0] = slots[-1]
                prefix_length = (
                    0 if mode == "suffix" else prefix_lengths[row // queries]
                )
                visible_slots[row] = torch.cat((slots[:prefix_length], slots[-1:]))
            suffix_lengths.fill_(1)
            backend.forward_metadata_spec_decode_expand.cu_seqlens_k.copy_(
                torch.arange(batch * queries + 1, device="cuda", dtype=torch.int32)
            )
            graph.replay()
            check(captured)

    def test_fa3_fusion_boundary(self):
        if get_device_sm() >= 100:
            self.skipTest("FA3 requires SM80-SM90")
        binary_tree = []
        for node in range(256):
            ancestors = [node]
            while ancestors[-1] > 0:
                ancestors.append((ancestors[-1] - 1) // 2)
            binary_tree.append(sorted(ancestors))
        for ancestors, fused, dtype in (
            (binary_tree, True, torch.float16),
            ([list(range(i + 1)) for i in range(64)], True, torch.bfloat16),
            ([list(range(i + 1)) for i in range(65)], False, torch.bfloat16),
        ):
            with self.subTest(width=len(ancestors), fused=fused):
                self._case(ancestors, 3, page_size=64, dtype=dtype, fused=fused)

    def test_fa4_verify_fused(self):
        self._case([[0], [0, 1], [0, 2], [0, 1, 3]], 4, fused=True)

    def test_fa4_verify_fallback(self):
        self._case([list(range(i + 1)) for i in range(65)], 4)

    def test_fa4_suffix_output_and_lse(self):
        self._case([list(range(length)) for length in (1, 5, 9)], 4, mode="suffix")

    def test_fa4_draft_decode(self):
        self._case([list(range(i * 3, (i + 1) * 3)) for i in range(4)], 4, mode="draft")


if __name__ == "__main__":
    unittest.main()
