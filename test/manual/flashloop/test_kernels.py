# SPDX-License-Identifier: MIT
# Copyright (c) 2026 FlashLoop contributors
"""GPU correctness gates against independent dense PyTorch calculations."""

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def tensors():
    torch.manual_seed(37)
    device = "cuda"
    # Shuffled physical slots, nonconsecutive request IDs, unequal lengths.
    mapping = torch.randperm(256, device=device).reshape(4, 64).int()
    ids = torch.tensor([3, 1], device=device, dtype=torch.int32)
    lengths = torch.tensor([49, 17], device=device, dtype=torch.int32)
    q = torch.randn(2, 3, 32, device=device, dtype=torch.bfloat16)
    k = torch.randn(256, 3, 32, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    return q, k, v, mapping, ids, lengths


def dense(q, k, v, mapping, ids, lengths):
    output = []
    for b in range(len(ids)):
        slots = mapping[ids[b].long(), : lengths[b]].long()
        kb, vb = k[slots].float().transpose(0, 1), v[slots].float().transpose(0, 1)
        p = torch.softmax(
            (kb * q[b].float()[:, None]).sum(-1) / math.sqrt(q.shape[-1]), -1
        )
        output.append((p[..., None] * vb).sum(1))
    return torch.stack(output)


@pytest.mark.parametrize("fraction", [0.01, 0.1, 0.5, 1.0])
def test_paged_cached_mass_matches_dense_formula(fraction):
    from sglang.srt.layers.attention.flashloop.kernels import (
        correct_attention,
        select_source,
    )

    q, k, v, mapping, ids, lengths = tensors()
    selection = select_source(q, k, v, mapping, ids, lengths, 64, fraction)
    selected, valid, mass, old_selected = selection
    source = dense(q, k, v, mapping, ids, lengths)
    q2, k2, v2 = q * 0.8, k * 1.1, v * 0.9
    actual = correct_attention(q2, k2, v2, mapping, ids, lengths, (*selection, source))
    expected = []
    for b in range(2):
        per_head = []
        for h in range(3):
            indices = selected[b, h, valid[b, h]]
            assert len(indices) == math.ceil(int(lengths[b]) * fraction)
            slots = mapping[ids[b].long(), indices].long()
            all_slots = mapping[ids[b].long(), : lengths[b]].long()
            old_p = torch.softmax(
                (k[all_slots, h].float() * q[b, h].float()).sum(-1) / math.sqrt(32), -1
            )
            old = (old_p[indices, None] * v[slots, h].float()).sum(0)
            new_p = torch.softmax(
                (k2[slots, h].float() * q2[b, h].float()).sum(-1) / math.sqrt(32), -1
            )
            new = (new_p[:, None] * v2[slots, h].float()).sum(0) * old_p[indices].sum()
            torch.testing.assert_close(old_selected[b, h], old, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(
                mass[b, h, 0], old_p[indices].sum(), atol=2e-5, rtol=2e-5
            )
            per_head.append(source[b, h] - old + new)
        expected.append(torch.stack(per_head))
    torch.testing.assert_close(
        actual.float(), torch.stack(expected), atol=0.008, rtol=0.008
    )


def test_source_attention_output_and_empty_graph_padding():
    from sglang.srt.layers.attention.flashloop.kernels import select_source

    q, k, v, mapping, ids, lengths = tensors()
    result = select_source(q, k, v, mapping, ids, lengths, 64, 0.1, return_output=True)
    torch.testing.assert_close(
        result[-1].float(),
        dense(q, k, v, mapping, ids, lengths),
        atol=0.008,
        rtol=0.008,
    )
    lengths[1] = 0
    result = select_source(q, k, v, mapping, ids, lengths, 64, 0.1, return_output=True)
    assert torch.isfinite(result[-1]).all()
    assert torch.count_nonzero(result[-1][1]) == 0


def test_cuda_graph_replay_tracks_new_lengths_and_request_slots():
    from sglang.srt.layers.attention.flashloop.kernels import (
        correct_attention,
        select_source,
    )

    q, k, v, mapping, ids, lengths = tensors()
    source = torch.zeros_like(q)

    def run():
        selected = select_source(q, k, v, mapping, ids, lengths, 64, 0.1)
        return correct_attention(q, k, v, mapping, ids, lengths, (*selected, source))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    lengths.copy_(torch.tensor([3, 63], device="cuda", dtype=torch.int32))
    ids.copy_(torch.tensor([0, 2], device="cuda", dtype=torch.int32))
    q.mul_(0.7)
    graph.replay()
    torch.testing.assert_close(output, run(), atol=0.008, rtol=0.008)


def test_sparse_prefill_uses_original_positions_and_separate_requests():
    from sglang.srt.layers.attention.flashloop.prefill import sparse_paged_prefill

    _, k, v, mapping, ids, lengths = tensors()
    positions = torch.tensor([0, 3, 12, 48, 5, 16], device="cuda")
    starts = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
    counts = torch.tensor([4, 2], dtype=torch.int32, device="cuda")
    q = torch.randn(6, 3, 32, dtype=torch.bfloat16, device="cuda")
    actual = sparse_paged_prefill(
        q, k, v, positions, mapping, ids, lengths, starts, counts, 4
    )
    expected = []
    for i, position in enumerate(positions):
        batch = 0 if i < 4 else 1
        slots = mapping[ids[batch].long(), : position + 1].long()
        keys, values = (
            k[slots].float().transpose(0, 1),
            v[slots].float().transpose(0, 1),
        )
        p = torch.softmax((q[i].float()[:, None] * keys).sum(-1) / math.sqrt(32), -1)
        expected.append((p[..., None] * values).sum(1))
    torch.testing.assert_close(
        actual.float(), torch.stack(expected), atol=0.012, rtol=0.012
    )


def test_nested_prefill_selection_keeps_last_token_per_request():
    from sglang.srt.layers.attention.flashloop.prefill import select_rows

    previous = torch.randn(30, 16, device="cuda")
    current = torch.randn_like(previous)
    first = select_rows(current, previous, [11, 19], 0.5)[0]
    second = select_rows(current * 1.1, current, [11, 19], 0.2, first)[0]
    assert set(second.tolist()) <= set(first.tolist())
    assert {10, 29} <= set(second.tolist())
    assert first.numel() == 6 + 10
    assert second.numel() == 3 + 4
