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

"""Mixed-width shared-context metadata and standard-attention fallbacks."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.beam_search.shared_context_attention import build_metadata, forward
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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
            forward_mode=ForwardMode.DECODE,
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
            "sglang.kernels.ops.attention.beam_shared_context.shared_context_attention"
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
                        SimpleNamespace(
                            forward_mode=ForwardMode.DECODE, beam_tail=object()
                        ),
                        self.model_runner(**unsupported),
                    )
                )


if __name__ == "__main__":
    unittest.main()
