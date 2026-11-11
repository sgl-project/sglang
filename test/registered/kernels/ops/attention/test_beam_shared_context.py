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

import math
import unittest

import torch

from sglang.kernels.ops.attention.beam_shared_context import shared_context_attention
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


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


if __name__ == "__main__":
    unittest.main()
