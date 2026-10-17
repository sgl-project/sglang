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

"""Shared-prefix attention correctness and mixed-width metadata contracts."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.beam_search.shared_context_attention import build_metadata, forward
from sglang.srt.beam_search.shared_context_attention_kernel import (
    shared_context_attention,
)
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestSharedContextAttentionMetadata(CustomTestCase):
    def setUp(self):
        self.enterContext(
            envs.SGLANG_ENABLE_BEAM_SHARED_CONTEXT_ATTENTION.override(True)
        )

    @staticmethod
    def model_runner(**overrides):
        options = dict(
            decode_attention_backend_str="flashinfer",
            sliding_window_size=None,
            model_config=SimpleNamespace(
                is_encoder_decoder=False, attention_arch=AttentionArch.MHA
            ),
            kv_cache_dtype=torch.bfloat16,
        )
        return SimpleNamespace(**(options | overrides))

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

    def test_mixed_width_metadata_and_dispatch(self):
        entries = [
            SimpleNamespace(
                leader_idx=leader,
                start=start,
                end=end,
                group=SimpleNamespace(prompt_len=1),
            )
            for leader, start, end in ((0, 0, 1), (1, 1, 2), (2, 2, 4))
        ]
        seq_lens_cpu = torch.tensor([3, 2, 2, 3, 2, 2, 2], dtype=torch.int32)
        batch = SimpleNamespace(
            beam_tail=SimpleNamespace(entries=entries, num_base_rows=3),
            seq_lens=seq_lens_cpu,
            seq_lens_cpu=seq_lens_cpu,
            req_pool_indices=torch.tensor([10, 11, 12, 13, 14, 15, 16]),
            device=torch.device("cpu"),
        )
        model_runner = self.model_runner(
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(60, dtype=torch.int32).view(20, 3)
            ),
        )

        metadata = build_metadata(batch, model_runner)

        self.assertEqual([bucket.beam_width for bucket in metadata.buckets], [2, 3])
        self.assertEqual(metadata.buckets[0].row_indices.tolist(), [[0, 3], [1, 4]])
        self.assertEqual(metadata.buckets[0].decode_lens.tolist(), [2, 1])
        self.assertEqual(metadata.buckets[0].decode_slots[1, 2:].tolist(), [0, 0])

        q = torch.randn(7, 8)[:, ::2]
        self.assertFalse(q.is_contiguous())
        with patch(
            "sglang.srt.beam_search.shared_context_attention_kernel."
            "shared_context_attention"
        ) as kernel:
            output = forward(
                q,
                SimpleNamespace(
                    layer_id=0,
                    tp_q_head_num=1,
                    head_dim=4,
                    scaling=0.5,
                    logit_cap=0,
                ),
                SimpleNamespace(beam_attention_metadata=metadata),
                SimpleNamespace(
                    get_kv_buffer=lambda _layer_id: (
                        torch.empty(1, 1, 4),
                        torch.empty(1, 1, 4),
                    )
                ),
            )

        self.assertEqual(output.shape, (7, 4))
        self.assertEqual(kernel.call_count, 2)
        self.assertTrue(
            all(call.args[0].is_contiguous() for call in kernel.call_args_list)
        )

    def test_standard_attention_fallbacks(self):
        self.assertIsNone(
            forward(
                torch.empty(0),
                SimpleNamespace(),
                SimpleNamespace(beam_attention_metadata=None),
                SimpleNamespace(),
            )
        )
        for unsupported in (
            {"decode_attention_backend_str": "triton"},
            {"sliding_window_size": 128},
            {"kv_cache_dtype": torch.float8_e4m3fn},
            {
                "model_config": SimpleNamespace(
                    is_encoder_decoder=True, attention_arch=AttentionArch.MHA
                )
            },
            {
                "model_config": SimpleNamespace(
                    is_encoder_decoder=False, attention_arch=AttentionArch.MLA
                )
            },
        ):
            with self.subTest(unsupported=unsupported):
                self.assertIsNone(
                    build_metadata(
                        SimpleNamespace(beam_tail=object()),
                        self.model_runner(**unsupported),
                    )
                )


if __name__ == "__main__":
    unittest.main()
