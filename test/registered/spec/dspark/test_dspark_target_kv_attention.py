"""Logical-order KV attention with strided pools and changing graph inputs."""

import unittest
from itertools import pairwise

import torch
from sglang.kernels.ops.speculative.dspark.target_kv_attention import (
    target_kv_attention,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestTargetKVAttention(CustomTestCase):
    def make_inputs(
        self, query_lens, kv_lens, *, group=2, dim=128, dtype=torch.bfloat16
    ):
        torch.manual_seed(918)
        num_kv_heads = 2
        num_heads = num_kv_heads * group
        num_tokens = sum(query_lens)
        num_slots = max(1, sum(kv_lens) * 2 + 5)
        q = torch.randn(num_tokens, num_heads, dim + 8, device="cuda", dtype=dtype)
        k = torch.randn(num_slots, num_kv_heads, dim + 8, device="cuda", dtype=dtype)
        v = torch.randn_like(k)
        self.output_storage = torch.full_like(q, -123)
        self.dim = dim
        return {
            "q": q[..., :dim],
            "k": k[..., :dim],
            "v": v[..., :dim],
            "out": self.output_storage[..., :dim],
            "qo_indptr": torch.tensor(
                [0, *query_lens], device="cuda", dtype=torch.int32
            ).cumsum(0),
            "kv_indptr": torch.tensor(
                [0, *kv_lens], device="cuda", dtype=torch.int32
            ).cumsum(0),
            "kv_indices": torch.randperm(num_slots, device="cuda")[: sum(kv_lens)],
            "max_query": max(query_lens),
            "scale": dim**-0.5,
        }

    def assert_reference(self, inputs):
        q, k, v = (inputs[name] for name in ("q", "k", "v"))
        qo = inputs["qo_indptr"].tolist()
        ki = inputs["kv_indptr"].tolist()
        expected = torch.zeros_like(q)
        group = q.shape[1] // k.shape[1]
        for row in range(len(qo) - 1):
            if qo[row] == qo[row + 1] or ki[row] == ki[row + 1]:
                continue
            slots = inputs["kv_indices"][ki[row] : ki[row + 1]].long()
            query = q[qo[row] : qo[row + 1]].double().transpose(0, 1)
            key = k[slots].double().repeat_interleave(group, dim=1).transpose(0, 1)
            value = v[slots].double().repeat_interleave(group, dim=1).transpose(0, 1)
            probability = (query @ key.transpose(-1, -2) * inputs["scale"]).softmax(-1)
            expected[qo[row] : qo[row + 1]] = (
                (probability @ value).transpose(0, 1).to(q.dtype)
            )
        tolerance = (
            0.015
            if q.dtype == torch.bfloat16
            else 0.002
            if q.dtype == torch.float16
            else 1e-5
        )
        torch.testing.assert_close(
            inputs["out"], expected, rtol=tolerance, atol=tolerance
        )
        self.assertTrue(torch.all(self.output_storage[..., self.dim :] == -123).item())

    def test_ragged_and_tile_boundaries(self):
        cases = (
            ([3, 1, 0, 2], [162, 65, 17, 0], 2, 128, torch.bfloat16),
            ([1, 3, 7], [1, 63, 64], 1, 64, torch.float16),
            ([16, 2], [129, 1025], 4, 128, torch.bfloat16),
            ([64, 3], [65, 127], 8, 64, torch.float16),
            ([7, 1], [513, 0], 3, 80, torch.bfloat16),
            ([3], [8193], 2, 256, torch.bfloat16),
            ([3, 1], [65, 17], 2, 64, torch.float32),
        )
        for query_lens, kv_lens, group, dim, dtype in cases:
            with self.subTest(
                query_lens=query_lens,
                kv_lens=kv_lens,
                group=group,
                dim=dim,
                dtype=dtype,
            ):
                inputs = self.make_inputs(
                    query_lens, kv_lens, group=group, dim=dim, dtype=dtype
                )
                target_kv_attention(**inputs)
                self.assert_reference(inputs)

    def test_graph_replay_reads_new_lengths_slots_and_values(self):
        inputs = self.make_inputs([3, 3], [63, 65])
        inputs["max_query"] = 4
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            target_kv_attention(**inputs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            target_kv_attention(**inputs)
        graph.replay()
        self.assert_reference(inputs)
        before = inputs["out"].clone()
        inputs["qo_indptr"].copy_(torch.tensor([0, 2, 6], device="cuda"))
        inputs["kv_indptr"].copy_(torch.tensor([0, 65, 128], device="cuda"))
        inputs["kv_indices"].copy_(inputs["kv_indices"].flip(0))
        inputs["q"].neg_()
        inputs["v"].mul_(0.5)
        graph.replay()
        self.assert_reference(inputs)
        self.assertFalse(torch.equal(before, inputs["out"]))

    def test_prefix_and_block_indices_preserve_logical_tile_order(self):
        prefix_lens = [0, 63, 64, 65, 159, 160]
        inputs = self.make_inputs([3] * len(prefix_lens), [n + 3 for n in prefix_lens])
        target_kv_attention(**inputs)
        expected = inputs["out"].clone()
        indices = inputs["kv_indices"]
        indptr = inputs["kv_indptr"].tolist()
        inputs["extend_indices"] = torch.cat(
            [indices[end - 3 : end] for end in indptr[1:]]
        )
        inputs["kv_indices"] = torch.cat(
            [indices[start : end - 3] for start, end in pairwise(indptr)]
        )
        inputs["kv_indptr"] = torch.tensor([0, *prefix_lens], device="cuda").cumsum(0)
        inputs["out"].fill_(float("nan"))
        target_kv_attention(**inputs)
        torch.testing.assert_close(inputs["out"], expected, rtol=0, atol=0)

    def test_invalid_geometry_is_rejected_before_launch(self):
        inputs = self.make_inputs([3], [65])
        for replacement in (
            {"max_query": 65},
            {"k": inputs["k"].transpose(1, 2)},
            {"v": inputs["v"].float()},
            {"kv_indices": inputs["kv_indices"].float()},
            {"kv_indptr": inputs["kv_indptr"][:1]},
        ):
            with self.subTest(fields=tuple(replacement)), self.assertRaises(ValueError):
                target_kv_attention(**(inputs | replacement))


if __name__ == "__main__":
    unittest.main()
