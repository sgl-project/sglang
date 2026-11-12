# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Shared-prefix beam attention kernels against an FP32 reference."""

import itertools
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention import (
    beam_shared_context_split_kv,
    beam_shared_context_stream_k,
)
from sglang.kernels.ops.attention.beam_shared_context import shared_context_attention
from sglang.kernels.ops.attention.beam_shared_context_split_kv import (
    shared_context_attention_split_kv,
)
from sglang.kernels.ops.attention.beam_shared_context_stream_k import (
    shared_context_attention_stream_k,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

KERNELS = {
    "default": shared_context_attention,
    "split_kv": shared_context_attention_split_kv,
    "stream_k": shared_context_attention_stream_k,
}
# Rows no kernel call may write: each bucket owns only its row_indices.
SENTINEL = 7.0


def _bucket(context_lens, decode_lens, *, beam_width, rows, slots, decode_capacity):
    """One width bucket at scattered K/V slots; `rows` [groups, beam_width] maps
    its beams to rows of Q, and decode capacity runs past the decode lengths."""
    device = rows.device
    num_context = sum(context_lens)
    context_slots = slots[:num_context]
    decode_slots = slots[
        num_context : num_context + len(context_lens) * decode_capacity * beam_width
    ].view(len(context_lens), decode_capacity * beam_width)
    return SimpleNamespace(
        row_indices=rows.to(torch.int32),
        context_slots=context_slots,
        context_cu_seqlens=torch.tensor(
            [0, *itertools.accumulate(context_lens)], dtype=torch.int32, device=device
        ),
        decode_slots=decode_slots,
        decode_lens=torch.tensor(decode_lens, dtype=torch.int32, device=device),
        beam_width=beam_width,
    )


def _reference(q, k_buffer, v_buffer, bucket, softmax_scale, output):
    """FP32 attention of every beam of `bucket`, written to its rows of `output`."""
    q_per_kv = q.shape[1] // k_buffer.shape[1]
    bounds = itertools.pairwise(bucket.context_cu_seqlens.tolist())
    for group, (start, end) in enumerate(bounds):
        rows = bucket.row_indices[group].long()
        decode_len = int(bucket.decode_lens[group])
        context = bucket.context_slots[start:end].long()
        decode = (
            bucket.decode_slots[group].view(-1, bucket.beam_width)[:decode_len].T.long()
        )
        query = q[rows].float()
        k_context = k_buffer[context].repeat_interleave(q_per_kv, dim=1).float()
        v_context = v_buffer[context].repeat_interleave(q_per_kv, dim=1).float()
        k_decode = k_buffer[decode].repeat_interleave(q_per_kv, dim=2).float()
        v_decode = v_buffer[decode].repeat_interleave(q_per_kv, dim=2).float()
        scores = torch.cat(
            (
                torch.einsum("bhd,lhd->bhl", query, k_context),
                torch.einsum("bhd,bthd->bht", query, k_decode),
            ),
            dim=-1,
        )
        probabilities = (scores * softmax_scale).softmax(dim=-1)
        num_context = context.numel()
        output[rows] = torch.einsum(
            "bhl,lhd->bhd", probabilities[..., :num_context], v_context
        ) + torch.einsum("bht,bthd->bhd", probabilities[..., num_context:], v_decode)


def _case(
    groups_per_width,
    *,
    q_heads=4,
    kv_heads=2,
    head_dim=64,
    decode_capacity=3,
    num_slots=1 << 14,
    unowned_rows=0,
    seed=0,
):
    """Width buckets whose beams sit at shuffled rows of one Q, and their reference.

    `groups_per_width` maps a beam width to (context_lens, decode_lens) per group;
    `unowned_rows` more rows of Q belong to no bucket and keep SENTINEL.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda")
    owned_rows = sum(w * len(c) for w, (c, _) in groups_per_width.items())
    num_rows = owned_rows + unowned_rows
    rows = torch.randperm(num_rows, device=device)[:owned_rows]
    slots = torch.randperm(num_slots, device=device).to(torch.int32)
    q = torch.randn(num_rows, q_heads, head_dim, dtype=torch.bfloat16, device=device)
    k_buffer = torch.randn(
        num_slots, kv_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    v_buffer = torch.randn_like(k_buffer)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    buckets = []
    reference = torch.full(q.shape, SENTINEL, dtype=torch.float32, device=device)
    for beam_width, (context_lens, decode_lens) in groups_per_width.items():
        bucket_rows = beam_width * len(context_lens)
        width_rows, rows = rows.split((bucket_rows, rows.numel() - bucket_rows))
        bucket_slots = (
            sum(context_lens) + len(context_lens) * decode_capacity * beam_width
        )
        bucket_slot_ids, slots = slots.split(
            (bucket_slots, slots.numel() - bucket_slots)
        )
        bucket = _bucket(
            context_lens,
            decode_lens,
            beam_width=beam_width,
            rows=width_rows.view(len(context_lens), beam_width),
            slots=bucket_slot_ids,
            decode_capacity=decode_capacity,
        )
        _reference(q, k_buffer, v_buffer, bucket, softmax_scale, reference)
        buckets.append(bucket)
    return SimpleNamespace(
        q=q,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        buckets=buckets,
        softmax_scale=softmax_scale,
        reference=reference,
    )


def _run(kernel, case, **kwargs):
    """Every bucket into one output that starts as SENTINEL everywhere."""
    output = torch.full_like(case.q, SENTINEL)
    for bucket in case.buckets:
        kernel(
            case.q,
            case.k_buffer,
            case.v_buffer,
            bucket.row_indices,
            bucket.context_slots,
            bucket.context_cu_seqlens,
            bucket.decode_slots,
            bucket.decode_lens,
            output=output,
            softmax_scale=case.softmax_scale,
            **kwargs,
        )
    return output


class TestBeamSharedContextAttention(CustomTestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_native_kernel_matches_fp32_reference(self):
        torch.manual_seed(0)
        beam_width, q_heads, kv_heads, head_dim = 8, 4, 2, 64
        context_len, decode_len = 5, 2
        num_slots = context_len + decode_len * beam_width
        with torch.device("cuda"):
            q = torch.randn((beam_width, q_heads, head_dim), dtype=torch.bfloat16)
            k_buffer = torch.randn(
                (num_slots, kv_heads, head_dim), dtype=torch.bfloat16
            )
            v_buffer = torch.randn_like(k_buffer)
            row_indices = torch.arange(beam_width, dtype=torch.int32).view(1, -1)
            context_slots = torch.arange(context_len, dtype=torch.int32)
            context_cu_seqlens = torch.tensor([0, context_len], dtype=torch.int32)
            decode_slots = torch.arange(context_len, num_slots, dtype=torch.int32).view(
                1, -1
            )
            decode_lens = torch.tensor([decode_len], dtype=torch.int32)
        softmax_scale = 1.0 / math.sqrt(head_dim)

        output = shared_context_attention(
            q,
            k_buffer,
            v_buffer,
            row_indices,
            context_slots,
            context_cu_seqlens,
            decode_slots,
            decode_lens,
            softmax_scale=softmax_scale,
        )

        slots = torch.cat(
            (
                context_slots.expand(beam_width, -1),
                decode_slots.view(decode_len, beam_width).T,
            ),
            dim=1,
        ).long()
        keys = k_buffer[slots].repeat_interleave(q_heads // kv_heads, dim=2).float()
        values = v_buffer[slots].repeat_interleave(q_heads // kv_heads, dim=2).float()
        scores = torch.einsum("bhd,bthd->bht", q.float(), keys) * softmax_scale
        reference = torch.einsum("bht,bthd->bhd", scores.softmax(dim=-1), values)

        torch.testing.assert_close(
            output.float(),
            reference.float(),
            rtol=1e-2,
            atol=1e-2,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_split_kv_matches_fp32_reference(self):
        """Every split count gives the unsplit result, also when a short prompt
        leaves splits empty; decode slots past a group's decode length are unread."""
        case = _case({4: ([1, 70, 300], [2, 1, 2])})
        for num_splits in (None, 1, 3, 8):
            with self.subTest(num_splits=num_splits):
                output = _run(
                    shared_context_attention_split_kv, case, num_splits=num_splits
                )
                torch.testing.assert_close(
                    output.float(), case.reference, rtol=1e-2, atol=1e-2
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_stream_k_matches_fp32_reference(self):
        """Whole tiles, tiles split across CTAs and CTAs without work give the same
        result; decode slots past a group's decode length are unread."""
        case = _case({4: ([1, 70, 300], [2, 1, 2])})
        # One CTA takes every tile whole; 200 CTAs outnumber the work units.
        for num_ctas in (None, 1, 3, 7, 40, 200):
            with self.subTest(num_ctas=num_ctas):
                output = _run(
                    shared_context_attention_stream_k, case, num_ctas=num_ctas
                )
                torch.testing.assert_close(
                    output.float(), case.reference, rtol=1e-2, atol=1e-2
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_kv_cache_past_2_31_elements(self):
        """KV offsets past 2^31 elements must not wrap: in 32 bits the split-KV
        kernel read outside the cache (an illegal memory access)."""
        if torch.cuda.mem_get_info()[0] < 6 << 30:
            self.skipTest("needs 6 GB of free GPU memory")
        device = torch.device("cuda")
        heads, kv_heads, head_dim, width, capacity = 4, 2, 64, 16, 3
        context_lens, decode_lens = [70, 300], [2, 1]
        # Every slot of the case sits past 2^31 elements of the cache.
        first_slot = (1 << 31) // (kv_heads * head_dim) + 16
        num_used = sum(context_lens) + len(context_lens) * capacity * width
        torch.manual_seed(0)
        # One tensor serves as both K and V, to halve the 4.3 GB it takes.
        kv = torch.empty(
            first_slot + num_used,
            kv_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        kv[first_slot:] = torch.randn(
            num_used, kv_heads, head_dim, dtype=torch.bfloat16, device=device
        )
        rows = torch.randperm(len(context_lens) * width, device=device)
        bucket = _bucket(
            context_lens,
            decode_lens,
            beam_width=width,
            rows=rows.view(len(context_lens), width),
            slots=torch.arange(
                first_slot, first_slot + num_used, dtype=torch.int32, device=device
            ),
            decode_capacity=capacity,
        )
        q = torch.randn(
            rows.numel(), heads, head_dim, dtype=torch.bfloat16, device=device
        )
        scale = 1.0 / math.sqrt(head_dim)
        reference = torch.empty(q.shape, dtype=torch.float32, device=device)
        _reference(q, kv, kv, bucket, scale, reference)
        for name in ("split_kv", "stream_k"):
            with self.subTest(kernel=name):
                output = KERNELS[name](
                    q,
                    kv,
                    kv,
                    bucket.row_indices,
                    bucket.context_slots,
                    bucket.context_cu_seqlens,
                    bucket.decode_slots,
                    bucket.decode_lens,
                    softmax_scale=scale,
                )
                torch.testing.assert_close(
                    output.float(), reference, rtol=1e-2, atol=1e-2
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_kernels_write_only_their_buckets_rows(self):
        """Width buckets share one output: each call writes exactly the rows its
        row_indices name, at the Qwen3-0.6B head layout and a 256-beam bucket."""
        case = _case(
            {
                256: ([700, 2048], [3, 1]),
                8: ([65, 1, 129], [1, 2, 3]),
                3: ([40], [2]),
            },
            q_heads=16,
            kv_heads=8,
            head_dim=128,
            unowned_rows=5,
        )
        for name, kernel in KERNELS.items():
            with self.subTest(kernel=name):
                output = _run(kernel, case)
                torch.testing.assert_close(
                    output.float(), case.reference, rtol=1e-2, atol=1e-2
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_graph_replay_refreshes_mixed_depths_and_row_mapping(self):
        """A graph captured once replays any staged lengths and row mapping: split
        counts, grids and workspaces depend only on the captured shapes."""
        torch.manual_seed(17)
        width, max_context, max_decode = 8, 80, 3
        with torch.device("cuda"):
            q = torch.randn((16, 4, 64), dtype=torch.bfloat16)
            k = torch.randn((256, 2, 64), dtype=torch.bfloat16)
            v = torch.randn_like(k)
            rows = torch.arange(16, dtype=torch.int32).view(2, width)
            slots = torch.arange(160, 208, dtype=torch.int32).view(2, 24)
            # row_indices, context_slots, context_cu_seqlens, decode_slots, decode_lens
            a = (
                rows,
                torch.arange(150, dtype=torch.int32),
                torch.tensor([0, 5, 150], dtype=torch.int32),
                slots[:, :16],
                torch.tensor([1, 2], dtype=torch.int32),
            )
            b = (
                rows.flip((0, 1)),
                torch.tensor([5, 4, 3], dtype=torch.int32),
                torch.tensor([0, 2, 3], dtype=torch.int32),
                slots.flip((1,)),
                torch.tensor([3, 1], dtype=torch.int32),
            )
            # Buffers of fixed capacity, as a decode graph captures them.
            staged = (
                torch.empty_like(rows),
                torch.empty(2 * max_context, dtype=torch.int32),
                torch.empty(3, dtype=torch.int32),
                torch.empty((2, width * max_decode), dtype=torch.int32),
                torch.empty(2, dtype=torch.int32),
            )

        def stage(source):
            row_indices, context_slots, cu_seqlens, decode_slots, decode_lens = source
            staged[1].fill_(255)
            staged[3].fill_(255)
            staged[0].copy_(row_indices)
            staged[1][: context_slots.numel()].copy_(context_slots)
            staged[2].copy_(cu_seqlens)
            staged[3][:, : decode_slots.shape[1]].copy_(decode_slots)
            staged[4].copy_(decode_lens)

        for name, kernel in KERNELS.items():
            with self.subTest(kernel=name):
                stage(a)
                kernel(q, k, v, *staged, softmax_scale=0.125)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = kernel(q, k, v, *staged, softmax_scale=0.125)
                for source in (a, b, a):
                    stage(source)
                    graph.replay()
                    expected = shared_context_attention(
                        q, k, v, *source, softmax_scale=0.125
                    )
                    torch.cuda.synchronize()
                    torch.testing.assert_close(
                        output.float(), expected.float(), rtol=1e-2, atol=1e-2
                    )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_forward_runs_the_selected_kernel(self):
        """SGLANG_BEAM_SHARED_CONTEXT_KERNEL picks the kernel every bucket runs."""
        from sglang.srt.beam_search.shared_context_attention import (
            BeamAttentionBucket,
            BeamAttentionMetadata,
            forward,
        )
        from sglang.srt.environ import envs

        case = _case({8: ([33, 70], [2, 1]), 4: ([5], [3])})
        num_rows, q_heads, head_dim = case.q.shape
        metadata = BeamAttentionMetadata(
            num_rows=num_rows,
            buckets=[
                BeamAttentionBucket(
                    row_indices=b.row_indices,
                    context_slots=b.context_slots,
                    context_cu_seqlens=b.context_cu_seqlens,
                    decode_slots=b.decode_slots,
                    beam_width=b.beam_width,
                    decode_lens=b.decode_lens,
                )
                for b in case.buckets
            ],
        )
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=q_heads,
            head_dim=head_dim,
            scaling=case.softmax_scale,
            logit_cap=0,
        )
        kv_layers = SimpleNamespace(
            read=lambda layer_id: SimpleNamespace(
                kv=lambda: (case.k_buffer, case.v_buffer)
            )
        )
        forward_batch = SimpleNamespace(beam_attention_metadata=metadata)
        for name in KERNELS:
            with (
                self.subTest(kernel=name),
                envs.SGLANG_BEAM_SHARED_CONTEXT_KERNEL.override(name),
                patch.object(
                    beam_shared_context_split_kv,
                    "shared_context_attention_split_kv",
                    wraps=shared_context_attention_split_kv,
                ) as split_kv,
                patch.object(
                    beam_shared_context_stream_k,
                    "shared_context_attention_stream_k",
                    wraps=shared_context_attention_stream_k,
                ) as stream_k,
            ):
                output = forward(
                    case.q.view(num_rows, q_heads * head_dim),
                    layer,
                    forward_batch,
                    kv_layers,
                )
                self.assertEqual(split_kv.call_count, 2 * (name == "split_kv"))
                self.assertEqual(stream_k.call_count, 2 * (name == "stream_k"))
                torch.testing.assert_close(
                    output.view(num_rows, q_heads, head_dim).float(),
                    case.reference,
                    rtol=1e-2,
                    atol=1e-2,
                )

    def test_unknown_kernel_is_rejected(self):
        from sglang.srt.beam_search.shared_context_attention import selected_kernel
        from sglang.srt.environ import envs

        with envs.SGLANG_BEAM_SHARED_CONTEXT_KERNEL.override("splitkv"):
            with self.assertRaisesRegex(
                ValueError, "SGLANG_BEAM_SHARED_CONTEXT_KERNEL"
            ):
                selected_kernel()


if __name__ == "__main__":
    unittest.main()
