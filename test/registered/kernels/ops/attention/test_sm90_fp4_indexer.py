# SPDX-License-Identifier: Apache-2.0
"""Hopper FP4 indexer scores across visible-length and graph replay boundaries."""

import unittest

import torch

from sglang.kernels.ops.attention.dsv4.sm90_fp4_indexer import (
    fp4_index_logits_decode,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

PAGE_SIZE = 64
HEAD_DIM = 128


def _make_inputs(num_heads, capacity, visible_lengths):
    torch.manual_seed(42 + num_heads + capacity)
    batch_size = len(visible_lengths)
    num_pages = max(1, (capacity + PAGE_SIZE - 1) // PAGE_SIZE)
    fp4_values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.bfloat16,
        device="cuda",
    )
    codes = torch.randint(0, 16, (num_pages * PAGE_SIZE, HEAD_DIM), device="cuda")
    scale_exponents = torch.randint(
        125, 129, (num_pages * PAGE_SIZE, HEAD_DIM // 32), device="cuda"
    )
    payload = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    table = torch.cat(
        [
            payload.reshape(num_pages, PAGE_SIZE * 64),
            scale_exponents.to(torch.uint8).reshape(num_pages, PAGE_SIZE * 4),
        ],
        dim=1,
    )
    keys = (
        fp4_values[codes].float()
        * torch.exp2(scale_exponents.float() - 127).repeat_interleave(32, dim=1)
    ).to(torch.bfloat16)
    query_values = torch.tensor([-0.5, 0, 0.5], dtype=torch.bfloat16, device="cuda")
    q = query_values[
        torch.randint(0, 3, (batch_size, num_heads, HEAD_DIM), device="cuda")
    ]
    weight_values = torch.tensor(
        [-0.5, 0.25, 0.5, 1], dtype=torch.bfloat16, device="cuda"
    )
    weights = weight_values[
        torch.randint(0, len(weight_values), (batch_size, num_heads), device="cuda")
    ]
    logical_positions = torch.arange(capacity, device="cuda")
    page_orders = torch.stack(
        [
            torch.arange(num_pages - 1, -1, -1, device="cuda").roll(row)
            for row in range(batch_size)
        ]
    )
    slots = (
        page_orders[:, logical_positions // PAGE_SIZE] * PAGE_SIZE
        + logical_positions % PAGE_SIZE
    )
    lens = torch.tensor(visible_lengths, dtype=torch.int64, device="cuda")
    return q, weights, slots, lens, table, keys


def _reference(q, weights, slots, lens, keys):
    # Small dyadic inputs keep both dot decompositions exact in FP32.
    scores = torch.einsum("bhd,bld->bhl", q.float(), keys[slots].float())
    scores = scores.to(torch.bfloat16).relu()
    scores = (scores * weights.unsqueeze(-1)).to(torch.bfloat16)
    logits = scores.sum(dim=1, dtype=torch.bfloat16).float()
    visible = torch.arange(slots.shape[1], device=slots.device) < lens[:, None]
    return logits.masked_fill(~visible, -torch.inf)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0),
    "requires SM90 (Hopper)",
)
class TestSM90FP4Indexer(CustomTestCase):
    def test_visible_length_boundaries(self):
        cases = (
            (0, [0]),
            (63, [0, 1, 62, 63]),
            (64, [0, 1, 63, 64]),
            (65, [0, 1, 63, 64, 65]),
            (129, [0, 1, 63, 64, 65, 129]),
        )
        for num_heads in (16, 32):
            for capacity, lengths in cases:
                with self.subTest(num_heads=num_heads, capacity=capacity):
                    q, weights, slots, lens, table, keys = _make_inputs(
                        num_heads, capacity, lengths
                    )
                    actual = fp4_index_logits_decode(
                        q, weights, slots, lens, table, PAGE_SIZE
                    )
                    expected = _reference(q, weights, slots, lens, keys)
                    self.assertEqual(actual.dtype, torch.float32)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    tail = torch.arange(capacity, device="cuda") >= lens[:, None]
                    self.assertTrue(torch.isneginf(actual[tail]).all().item())

    def test_device_lengths_change_during_graph_replay(self):
        capacity = 129
        for num_heads in (16, 32):
            with self.subTest(num_heads=num_heads):
                q, weights, slots, lens, table, keys = _make_inputs(
                    num_heads, capacity, [capacity] * 3
                )
                warmup = torch.cuda.Stream()
                warmup.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warmup):
                    for _ in range(3):
                        fp4_index_logits_decode(
                            q, weights, slots, lens, table, PAGE_SIZE
                        )
                torch.cuda.current_stream().wait_stream(warmup)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = fp4_index_logits_decode(
                        q, weights, slots, lens, table, PAGE_SIZE
                    )
                for visible_length in (capacity, 0, 65, 63, capacity):
                    with self.subTest(visible_length=visible_length):
                        lens.fill_(visible_length)
                        graph.replay()
                        expected = _reference(q, weights, slots, lens, keys)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        self.assertTrue(
                            torch.isneginf(actual[:, visible_length:]).all().item()
                        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
