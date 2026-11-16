import unittest

import torch

from sglang.kernels.ops.attention.extend_attention import (
    _get_block_sizes_for_extend_attention,
    extend_attention_fwd_unified,
)
from sglang.kernels.ops.attention.metadata import prepare_tree_suffix_indices
from sglang.kernels.ops.attention.score_mod import relative_bias_score_mod
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-small")
# backend-specific: HIP uses different dot tiles and reduction layouts for the
# same tree-attention contract, including the gfx950 unified-kernel launch.
register_amd_ci(est_time=45, suite="stage-b-test-1-gpu-small-amd")


class TestUnifiedTreeAttention(CustomTestCase):
    def setUp(self):
        self.device = get_device()

    def _tensor(self, values, dtype=torch.int32):
        return torch.tensor(values, dtype=dtype, device=self.device)

    def _tree_inputs(self, prefix_len, path, logical_k, logical_v, query):
        nodes = 12
        k = torch.randn(
            prefix_len + nodes,
            logical_k.shape[1],
            query.shape[-1],
            dtype=query.dtype,
            device=self.device,
        )
        v = torch.randn_like(k)
        visible = list(range(prefix_len)) + [prefix_len + node for node in path]
        k[visible], v[visible] = logical_k, logical_v
        q = query.expand(nodes, -1, -1).contiguous()
        mask = torch.zeros(
            nodes, prefix_len + nodes, dtype=torch.bool, device=self.device
        )
        mask[:, :prefix_len] = True
        parents = [-1] + [0] * (nodes - 1)
        for parent, child in zip(path, path[1:]):
            parents[child] = parent
        for node in range(nodes):
            ancestor = node
            while ancestor != -1:
                mask[node, prefix_len + ancestor] = True
                ancestor = parents[ancestor]
        qo_indptr = self._tensor([0, nodes])
        prefix_lens = self._tensor([prefix_len])
        mask_indptr = self._tensor([0, mask.numel()], dtype=torch.int64)
        tree_indices, tree_lengths = prepare_tree_suffix_indices(
            qo_indptr=qo_indptr,
            prefix_lens=prefix_lens,
            mask_indptr=mask_indptr,
            custom_mask=mask.flatten(),
            max_extend_len=nodes,
            num_queries=nodes,
        )
        return dict(
            q=q,
            o=torch.empty_like(q),
            k_buffer=k,
            v_buffer=v,
            k_scale=1.0,
            v_scale=1.0,
            qo_indptr=qo_indptr,
            kv_indptr=self._tensor([0, prefix_len + nodes]),
            kv_indices=torch.arange(
                prefix_len + nodes, dtype=torch.int64, device=self.device
            ),
            prefix_lens=prefix_lens,
            max_len_extend=nodes,
            custom_mask=mask.flatten(),
            mask_indptr=mask_indptr,
            tree_indices=tree_indices,
            tree_lengths=tree_lengths,
            tree_positions=prefix_len + tree_lengths.to(torch.int64) - 1,
        )

    def test_masked_sibling_layout_invariance(self):
        """The same ancestor path must not change when masked siblings move
        its keys across an attention tile boundary."""
        dim, heads, depth = 128, 4, 3
        block_n = _get_block_sizes_for_extend_attention(dim, dim)[4]
        for dtype in (torch.bfloat16, torch.float16):
            for prefix_len in (0, block_n - 4, block_n + 1, 2 * block_n):
                for seed in (5, 14, 19, 26):
                    with self.subTest(dtype=dtype, prefix_len=prefix_len, seed=seed):
                        torch.manual_seed(seed)
                        logical_k = torch.randn(
                            prefix_len + depth,
                            1,
                            dim,
                            dtype=dtype,
                            device=self.device,
                        )
                        logical_v = torch.randn_like(logical_k)
                        query = torch.randn(
                            1, heads, dim, dtype=dtype, device=self.device
                        )
                        outputs = []
                        for path in ([0, 1, 2], [0, 6, 11]):
                            inputs = self._tree_inputs(
                                prefix_len, path, logical_k, logical_v, query
                            )
                            extend_attention_fwd_unified(**inputs)
                            outputs.append(inputs["o"][path[-1]].clone())

                        # A dense mathematical oracle is independent of the
                        # production packing and floating-point reduction order.
                        scores = query[0].double() @ logical_k[:, 0].double().T
                        expected = (scores / dim**0.5).softmax(-1) @ logical_v[
                            :, 0
                        ].double()
                        tolerance = 0.02 if dtype == torch.bfloat16 else 0.003
                        for output in outputs:
                            torch.testing.assert_close(
                                output.double(),
                                expected,
                                atol=tolerance,
                                rtol=tolerance,
                            )
                        self.assertTrue(torch.equal(*outputs))

                        # Accepted draft tokens move into the cached prefix.
                        # Attention must preserve its reduction when this same
                        # logical path is processed as a normal causal extend.
                        causal_output = torch.empty_like(query)
                        extend_attention_fwd_unified(
                            q=query,
                            o=causal_output,
                            k_buffer=logical_k,
                            v_buffer=logical_v,
                            k_scale=1.0,
                            v_scale=1.0,
                            qo_indptr=self._tensor([0, 1]),
                            kv_indptr=self._tensor([0, prefix_len + depth]),
                            kv_indices=torch.arange(
                                prefix_len + depth,
                                dtype=torch.int64,
                                device=self.device,
                            ),
                            prefix_lens=self._tensor([prefix_len + depth - 1]),
                            max_len_extend=1,
                        )
                        self.assertTrue(torch.equal(outputs[0], causal_output[0]))

    def test_prefix_cutoff_invariance(self):
        """Accepting ancestors into the prefix must preserve the accumulator,
        including when acceptance moves a key across a complete prefix tile."""
        dim, heads, kv_heads = 128, 16, 8
        block_n = _get_block_sizes_for_extend_attention(dim, dim)[4]
        for dtype in (torch.bfloat16, torch.float16):
            for total, seed, scale in (
                (block_n + 3, 2, 1.0),
                (2 * block_n + 3, 4, 2.0),
            ):
                with self.subTest(dtype=dtype, total=total, seed=seed):
                    # Hq/Hkv=16/8, seed 2, total 131 exposed a rounding
                    # difference when prefix and tail used different M tiles.
                    torch.manual_seed(seed)
                    k = (
                        torch.randn(
                            total, kv_heads, dim, dtype=dtype, device=self.device
                        )
                        * scale
                    )
                    v = torch.randn_like(k)
                    q = (
                        torch.randn(1, heads, dim, dtype=dtype, device=self.device)
                        * scale
                    )
                    reference = torch.empty_like(q)
                    extend_attention_fwd_unified(
                        q=q,
                        o=reference,
                        k_buffer=k,
                        v_buffer=v,
                        k_scale=1.0,
                        v_scale=1.0,
                        qo_indptr=self._tensor([0, 1]),
                        kv_indptr=self._tensor([0, total]),
                        kv_indices=torch.arange(
                            total, dtype=torch.int64, device=self.device
                        ),
                        prefix_lens=self._tensor([total - 1]),
                        max_len_extend=1,
                    )
                    for depth in (12, 2):
                        prefix, path = total - depth, list(range(depth))
                        inputs = self._tree_inputs(prefix, path, k, v, q)
                        extend_attention_fwd_unified(**inputs)
                        self.assertTrue(
                            torch.equal(inputs["o"][path[-1]], reference[0])
                        )

    def test_batched_tree_matches_independent(self):
        """A batch must keep each request's prefix state and query/KV offsets
        separate when the requests need different numbers of prefix tiles."""
        torch.manual_seed(19)
        dim, heads = 128, 4
        block_n = _get_block_sizes_for_extend_attention(dim, dim)[4]
        batches, expected = [], []
        for prefix, path in ((block_n - 4, [0, 1, 2]), (2 * block_n + 5, [0, 6, 11])):
            k = torch.randn(
                prefix + len(path), 1, dim, dtype=torch.bfloat16, device=self.device
            )
            v = torch.randn_like(k)
            q = torch.randn(1, heads, dim, dtype=k.dtype, device=self.device)
            inputs = self._tree_inputs(prefix, path, k, v, q)
            extend_attention_fwd_unified(**inputs)
            expected.append(inputs["o"].clone())
            batches.append(inputs)
        first, second = batches
        combined = dict(first)
        for key in (
            "q",
            "o",
            "k_buffer",
            "v_buffer",
            "prefix_lens",
            "custom_mask",
            "tree_indices",
            "tree_lengths",
            "tree_positions",
        ):
            combined[key] = torch.cat([first[key], second[key]])
        first_kv = first["k_buffer"].shape[0]
        combined.update(
            qo_indptr=self._tensor([0, 12, 24]),
            kv_indptr=self._tensor([0, first_kv, combined["k_buffer"].shape[0]]),
            kv_indices=torch.cat(
                [first["kv_indices"], second["kv_indices"] + first_kv]
            ),
            mask_indptr=self._tensor(
                [0, first["custom_mask"].numel(), combined["custom_mask"].numel()],
                dtype=torch.int64,
            ),
        )
        extend_attention_fwd_unified(**combined)
        self.assertTrue(torch.equal(combined["o"], torch.cat(expected)))

    def test_ragged_metadata_graph_replay(self):
        """Graph replay must read the current tree, including ragged mask offsets,
        rather than retaining ancestor metadata from capture."""
        prefixes, nodes = [7, 67], [5, 12]
        masks = []
        for prefix, count in zip(prefixes, nodes):
            mask = torch.zeros(
                count, prefix + count, dtype=torch.bool, device=self.device
            )
            mask[:, :prefix] = True
            mask[:, prefix] = True
            mask[:, prefix:] |= torch.eye(count, dtype=torch.bool, device=self.device)
            masks.append(mask)
        masks[0][-1, prefixes[0] + 1] = True
        masks[1][-1, prefixes[1] + 6] = True
        custom_mask = torch.cat([mask.flatten() for mask in masks]).to(torch.uint8)
        # Graph buckets can contain a padded request without a matching mask.
        real_queries, padded_queries = sum(nodes), 4
        kwargs = dict(
            qo_indptr=self._tensor(
                [0, nodes[0], real_queries, real_queries + padded_queries]
            ),
            prefix_lens=self._tensor([*prefixes, 33]),
            mask_indptr=self._tensor(
                [0, masks[0].numel(), custom_mask.numel(), custom_mask.numel()],
                dtype=torch.int64,
            ),
            custom_mask=custom_mask,
            max_extend_len=max(nodes),
            num_queries=real_queries + padded_queries,
        )
        prepare_tree_suffix_indices(**kwargs)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            prepare_tree_suffix_indices(**kwargs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            indices, lengths = prepare_tree_suffix_indices(**kwargs)
        graph.replay()
        self.assertEqual(lengths[nodes[0] - 1].item(), 3)
        self.assertEqual(indices[nodes[0] - 1, :3].tolist(), [0, 1, 4])
        self.assertEqual(lengths[real_queries - 1].item(), 3)
        self.assertEqual(indices[real_queries - 1, :3].tolist(), [0, 6, 11])
        self.assertEqual(lengths[real_queries:].tolist(), [0] * padded_queries)
        # Change only the second request's final branch, with buffers unchanged.
        masks[1][-1, prefixes[1] + 6] = False
        custom_mask.copy_(torch.cat([mask.flatten() for mask in masks]))
        graph.replay()
        self.assertEqual(lengths[nodes[0] - 1].item(), 3)
        self.assertEqual(indices[nodes[0] - 1, :3].tolist(), [0, 1, 4])
        self.assertEqual(lengths[real_queries - 1].item(), 2)
        self.assertEqual(indices[real_queries - 1, :2].tolist(), [0, 11])
        self.assertEqual(lengths[real_queries:].tolist(), [0] * padded_queries)

    def test_windowed_tree_attention_features(self):
        """Logical positions must survive a trimmed prefix, paged/noncontiguous
        KV addresses, relative bias, sinks, and query-head grouping."""
        torch.manual_seed(8)
        dim, heads, kv_heads, depth = 64, 6, 2, 3
        block_n = _get_block_sizes_for_extend_attention(dim, dim)[4]
        prefix, window = block_n + 24, 8
        logical_k = torch.randn(
            prefix + depth,
            kv_heads,
            dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        logical_v = torch.randn_like(logical_k)
        query = torch.randn(1, heads, dim, dtype=logical_k.dtype, device=self.device)
        bias = torch.randn(heads, 32, dtype=torch.float32, device=self.device) * 0.1
        sinks = torch.linspace(-0.5, 0.5, heads, device=self.device)
        key_positions = torch.arange(prefix + depth, device=self.device)
        query_position = prefix + depth - 1
        distance = query_position - key_positions
        scores = torch.einsum(
            "hd,thd->ht",
            query[0].double(),
            logical_k.repeat_interleave(heads // kv_heads, dim=1).double(),
        ) * (0.75 / dim**0.5)
        scores = 2.0 * (scores / 2.0).tanh()
        scores += bias.double()[:, distance.clamp(0, 31)] * (distance < 32)
        scores = scores.masked_fill(
            key_positions < query_position - window, float("-inf")
        )
        weights = torch.cat([scores, sinks.double()[:, None]], dim=-1).softmax(-1)
        expected = (
            torch.einsum(
                "ht,thd->hd",
                weights[:, :-1],
                logical_v.repeat_interleave(heads // kv_heads, dim=1).double(),
            )
            * 1.25
        )
        outputs = []
        for page_size in (1, 16):
            for path in ([0, 1, 2], [0, 6, 11]):
                with self.subTest(page_size=page_size, path=path):
                    inputs = self._tree_inputs(
                        prefix, path, logical_k, logical_v, query
                    )
                    # Metadata still describes the full tree. Only layer-local
                    # KV indices and prefix length lose the earlier window.
                    indices = inputs["kv_indices"][block_n:]
                    inputs["kv_indptr"] = self._tensor([0, indices.numel()])
                    inputs["prefix_lens"] = self._tensor([prefix - block_n])
                    permutation = torch.randperm(
                        inputs["k_buffer"].shape[0], device=self.device
                    )
                    for name in ("k_buffer", "v_buffer"):
                        reordered = torch.empty_like(inputs[name])
                        reordered[permutation] = inputs[name]
                        inputs[name] = reordered
                    inputs.update(
                        kv_indices=permutation[indices],
                        page_size=page_size,
                        k_scale=0.75,
                        v_scale=1.25,
                        logit_cap=2.0,
                        sliding_window_size=window,
                        window_start_pos=self._tensor([block_n]),
                        sinks=sinks,
                        score_mod=relative_bias_score_mod,
                        aux_tensors=(bias[None].expand(12, -1, -1).contiguous(),),
                    )
                    extend_attention_fwd_unified(**inputs)
                    output = inputs["o"][path[-1]].clone()
                    torch.testing.assert_close(
                        output.double(), expected, atol=0.02, rtol=0.02
                    )
                    outputs.append(output)
        for output in outputs[1:]:
            self.assertTrue(torch.equal(outputs[0], output))

    def test_generic_bidirectional_mask(self):
        """Non-tree custom masks must retain visible future keys, including keys
        beyond a query tile; they do not opt into tree-path canonicalization."""
        torch.manual_seed(42)
        prefix, count, heads, dim = 7, 145, 4, 64
        q = torch.randn(count, heads, dim, dtype=torch.bfloat16, device=self.device)
        k = torch.randn(prefix + count, 1, dim, dtype=q.dtype, device=self.device)
        v = torch.randn_like(k)
        mask = torch.ones(count, prefix + count, dtype=torch.bool, device=self.device)
        mask[:, prefix:] = torch.ones(
            count, count, dtype=torch.bool, device=self.device
        ).tril()
        mask[:, 1] = False
        mask[:136, prefix : prefix + 136] = True
        scores = torch.einsum("qhd,kd->qhk", q.double(), k[:, 0].double())
        scores = (scores / dim**0.5).masked_fill(~mask[:, None, :], float("-inf"))
        expected = torch.einsum("qhk,kd->qhd", scores.softmax(-1), v[:, 0].double())
        for is_causal in (True, False):
            with self.subTest(is_causal=is_causal):
                out = torch.empty_like(q)
                extend_attention_fwd_unified(
                    q=q,
                    o=out,
                    k_buffer=k,
                    v_buffer=v,
                    k_scale=1.0,
                    v_scale=1.0,
                    qo_indptr=self._tensor([0, count]),
                    kv_indptr=self._tensor([0, prefix + count]),
                    kv_indices=torch.arange(
                        prefix + count, dtype=torch.int64, device=self.device
                    ),
                    prefix_lens=self._tensor([prefix]),
                    max_len_extend=count,
                    custom_mask=mask.flatten(),
                    mask_indptr=self._tensor([0, mask.numel()], dtype=torch.int64),
                    is_causal=is_causal,
                )
                torch.testing.assert_close(out.double(), expected, atol=0.02, rtol=0.02)


if __name__ == "__main__":
    unittest.main()
