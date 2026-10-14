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
    def test_missing_metadata_preserves_normal_attention_fallback(self):
        self.assertIsNone(
            forward(
                torch.empty(0),
                SimpleNamespace(),
                SimpleNamespace(beam_attention_metadata=None),
                SimpleNamespace(),
            )
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_native_kernel_matches_fp32_reference(self):
        torch.manual_seed(0)
        device = torch.device("cuda")
        beam_width = 8
        q_heads = 4
        kv_heads = 2
        head_dim = 64
        context_len = 5
        decode_len = 2
        num_slots = context_len + decode_len * beam_width

        q = torch.randn(
            beam_width,
            q_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        k_buffer = torch.randn(
            num_slots,
            kv_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        v_buffer = torch.randn_like(k_buffer)
        row_indices = torch.arange(
            beam_width,
            dtype=torch.int32,
            device=device,
        ).view(1, beam_width)
        context_slots = torch.arange(
            context_len,
            dtype=torch.int32,
            device=device,
        )
        context_cu_seqlens = torch.tensor(
            [0, context_len],
            dtype=torch.int32,
            device=device,
        )
        decode_slots = torch.arange(
            context_len,
            num_slots,
            dtype=torch.int32,
            device=device,
        ).view(1, decode_len * beam_width)
        decode_lens = torch.tensor([decode_len], dtype=torch.int32, device=device)
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

        reference = torch.empty_like(output)
        for beam_index in range(beam_width):
            slot_ids = torch.cat(
                (
                    context_slots,
                    decode_slots[0, beam_index::beam_width],
                )
            ).long()
            for q_head in range(q_heads):
                kv_head = q_head // (q_heads // kv_heads)
                keys = k_buffer[slot_ids, kv_head].float()
                values = v_buffer[slot_ids, kv_head].float()
                scores = keys @ q[beam_index, q_head].float()
                probabilities = torch.softmax(scores * softmax_scale, dim=0)
                reference[beam_index, q_head] = (probabilities[:, None] * values).sum(
                    dim=0
                )

        torch.testing.assert_close(
            output.float(),
            reference.float(),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_mixed_width_metadata_and_dispatch(self):
        self.enterContext(
            envs.SGLANG_ENABLE_BEAM_SHARED_CONTEXT_ATTENTION.override(True)
        )
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
        model_runner = SimpleNamespace(
            decode_attention_backend_str="flashinfer",
            sliding_window_size=None,
            model_config=SimpleNamespace(
                is_encoder_decoder=False, attention_arch=AttentionArch.MHA
            ),
            kv_cache_dtype=torch.bfloat16,
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

    def test_unsupported_attention_preserves_standard_backend(self):
        options = dict(
            decode_attention_backend_str="flashinfer",
            sliding_window_size=None,
            model_config=SimpleNamespace(
                is_encoder_decoder=False, attention_arch=AttentionArch.MHA
            ),
            kv_cache_dtype=torch.bfloat16,
        )
        self.enterContext(
            envs.SGLANG_ENABLE_BEAM_SHARED_CONTEXT_ATTENTION.override(True)
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
                        SimpleNamespace(**(options | unsupported)),
                    )
                )


if __name__ == "__main__":
    unittest.main()
