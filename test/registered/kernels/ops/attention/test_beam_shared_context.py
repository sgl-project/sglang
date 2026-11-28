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

"""Shared-prefix beam attention kernel against an FP32 reference."""

import itertools
import math
import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention.beam_shared_context import shared_context_attention
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")


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


def _check_against_reference(first_slot):
    """Run the kernel on one bucket of two groups of 16 beams whose K/V slots are
    the last ones of the cache, from `first_slot` on, and compare with FP32."""
    device = torch.device("cuda")
    heads, kv_heads, head_dim, width, capacity = 4, 2, 64, 16, 3
    context_lens, decode_lens = [70, 300], [2, 1]
    num_used = sum(context_lens) + len(context_lens) * capacity * width
    torch.manual_seed(0)
    # One tensor serves as both K and V, to halve the memory a large cache takes.
    kv = torch.empty(
        first_slot + num_used, kv_heads, head_dim, dtype=torch.bfloat16, device=device
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
    q = torch.randn(rows.numel(), heads, head_dim, dtype=torch.bfloat16, device=device)
    scale = 1.0 / math.sqrt(head_dim)
    reference = torch.empty(q.shape, dtype=torch.float32, device=device)
    _reference(q, kv, kv, bucket, scale, reference)
    output = shared_context_attention(
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
    torch.testing.assert_close(output.float(), reference, rtol=1e-2, atol=1e-2)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestSharedContextAttention(CustomTestCase):
    def test_small_kv_cache(self):
        """K/V offsets below 2^31 elements: the kernel keeps 32-bit address math."""
        _check_against_reference(first_slot=100)

    def test_kv_cache_past_2_31_elements(self):
        """KV offsets past 2^31 elements must not wrap: in 32 bits the kernel
        read outside the cache (an illegal memory access)."""
        if torch.cuda.mem_get_info()[0] < 6 << 30:
            self.skipTest("needs 6 GB of free GPU memory")
        # Every slot of the case sits past 2^31 elements of the 4.3 GB cache.
        _check_against_reference(first_slot=(1 << 31) // (2 * 64) + 16)


if __name__ == "__main__":
    unittest.main()
