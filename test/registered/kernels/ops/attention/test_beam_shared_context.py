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
import threading
import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention import beam_shared_context_split_kv
from sglang.kernels.ops.attention.beam_shared_context import shared_context_attention
from sglang.kernels.ops.attention.beam_shared_context_split_kv import (
    partial_buffers,
    shared_context_attention_split_kv,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-small")

KERNELS = {
    "default": shared_context_attention,
    "split_kv": shared_context_attention_split_kv,
}
# Rows no kernel call may write: each bucket owns only its row_indices.
SENTINEL = 7.0


def _bucket(context_lens, decode_lens, *, beam_width, rows, slots, decode_capacity):
    """One width bucket at scattered K/V slots; decode capacity exceeds the decode lengths."""
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
    """Width buckets at shuffled rows of one Q, and their reference.

    `unowned_rows` rows of Q belong to no bucket and keep SENTINEL.
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


def _spread(tensor, dim=0):
    """The same values and shape, with twice the stride along `dim`."""
    shape = list(tensor.shape)
    shape[dim] *= 2
    index = [slice(None)] * tensor.ndim
    index[dim] = slice(None, None, 2)
    view = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)[tuple(index)]
    view.copy_(tensor)
    return view


def _run(kernel, case, **kwargs):
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
    def test_split_kv_matches_fp32_reference(self):
        """Also with empty splits, and with the kept partial buffers poisoned: a
        launch must write every partial it reads."""
        case = _case({4: ([1, 70, 300], [2, 1, 2])})
        for num_splits in (None, 1, 8):
            with self.subTest(num_splits=num_splits):
                _run(shared_context_attention_split_kv, case, num_splits=num_splits)
                for buffer in partial_buffers(case.q.device, 1, 1):
                    buffer.fill_(float("nan"))
                output = _run(
                    shared_context_attention_split_kv, case, num_splits=num_splits
                )
                torch.testing.assert_close(
                    output.float(), case.reference, rtol=1e-2, atol=1e-2
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_kv_cache_past_2_31_elements(self):
        """KV offsets past 2^31 elements must not wrap (illegal memory access)."""
        if torch.cuda.mem_get_info()[0] < 6 << 30:
            self.skipTest("needs 6 GB of free GPU memory")
        device = torch.device("cuda")
        heads, kv_heads, head_dim, width, capacity = 4, 2, 64, 16, 3
        context_lens, decode_lens = [70, 300], [2, 1]
        first_slot = (1 << 31) // (kv_heads * head_dim) + 16
        num_used = sum(context_lens) + len(context_lens) * capacity * width
        torch.manual_seed(0)
        # One tensor serves as both K and V.
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
        for name, kernel in KERNELS.items():
            with self.subTest(kernel=name):
                output = kernel(
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
        torch.manual_seed(17)
        width, max_context, max_decode = 8, 80, 3
        with torch.device("cuda"):
            q = torch.randn((16, 4, 64), dtype=torch.bfloat16)
            k = torch.randn((256, 2, 64), dtype=torch.bfloat16)
            v = torch.randn_like(k)
            rows = torch.arange(16, dtype=torch.int32).view(2, width)
            slots = torch.arange(160, 208, dtype=torch.int32).view(2, 24)
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
    def test_plans_follow_the_strides(self):
        case = _case({8: ([33, 70], [2, 1])})
        bucket = case.buckets[0]
        tail = (
            bucket.context_slots,
            bucket.context_cu_seqlens,
            bucket.decode_slots,
            bucket.decode_lens,
        )
        beam_shared_context_split_kv._PLANS.clear()
        for _ in range(2):
            output = shared_context_attention_split_kv(
                case.q,
                case.k_buffer,
                case.v_buffer,
                bucket.row_indices,
                *tail,
                output=torch.full_like(case.q, SENTINEL),
                softmax_scale=case.softmax_scale,
            )
            torch.testing.assert_close(
                output.float(), case.reference, rtol=1e-2, atol=1e-2
            )
        self.assertEqual(len(beam_shared_context_split_kv._PLANS), 1)
        output = shared_context_attention_split_kv(
            _spread(case.q),
            _spread(case.k_buffer),
            _spread(case.v_buffer),
            _spread(bucket.row_indices, dim=1),
            *tail,
            output=_spread(torch.full_like(case.q, SENTINEL)),
            softmax_scale=case.softmax_scale,
        )
        torch.testing.assert_close(output.float(), case.reference, rtol=1e-2, atol=1e-2)
        self.assertEqual(len(beam_shared_context_split_kv._PLANS), 2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_partial_buffers_are_kept_per_thread_and_stream(self):
        device = torch.device("cuda", torch.cuda.current_device())
        eager = partial_buffers(device, 4096, 64)
        self.assertEqual(
            partial_buffers(device, 1024, 16)[0].data_ptr(), eager[0].data_ptr()
        )
        others = []
        thread = threading.Thread(
            target=lambda: others.append(partial_buffers(device, 4096, 64))
        )
        thread.start()
        thread.join()
        for graph in (torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()):
            with torch.cuda.graph(graph):
                others.append(partial_buffers(device, 4096, 64))
                others[-1][0].zero_()
        with torch.cuda.stream(torch.cuda.Stream()):
            others.append(partial_buffers(device, 4096, 64))
        pointers = {buffers[0].data_ptr() for buffers in (eager, *others)}
        self.assertEqual(len(pointers), 5)


if __name__ == "__main__":
    unittest.main()
