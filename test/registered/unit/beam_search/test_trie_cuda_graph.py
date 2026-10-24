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
"""Trie graph isolation and metadata freshness, with CPU-backed buffers."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.beam_search.shared_context_attention import build_metadata
from sglang.srt.beam_search.trie_cuda_graph import (
    BeamTrieShapeKey,
    TrieDecodeGraphState,
)
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestTrieDecodeGraphState(CustomTestCase):
    def setUp(self):
        for field, value in (
            (envs.SGLANG_ENABLE_BEAM_SHARED_CONTEXT_ATTENTION, True),
            (envs.SGLANG_OPT_BEAM_TRIE_TOPK_LOGPROB, False),
            (envs.SGLANG_BEAM_SHARED_CONTEXT_WIDTH, 2),
            (envs.SGLANG_BEAM_SHARED_CONTEXT_GRAPH_MAX_CONTEXT, 4),
            (envs.SGLANG_BEAM_SHARED_CONTEXT_GRAPH_MAX_DECODE, 3),
        ):
            self.enterContext(field.override(value))
        self.runner = SimpleNamespace(
            decode_attention_backend_str="flashinfer",
            sliding_window_size=None,
            model_config=SimpleNamespace(
                is_encoder_decoder=False, attention_arch=AttentionArch.MHA
            ),
            kv_cache_dtype=torch.bfloat16,
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(24, dtype=torch.int32).view(4, 6)
            ),
        )
        self.batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            beam_tail=SimpleNamespace(
                num_base_rows=2,
                entries=[
                    SimpleNamespace(
                        leader_idx=0,
                        start=0,
                        end=1,
                        group=SimpleNamespace(prompt_len=1),
                    ),
                    SimpleNamespace(
                        leader_idx=1,
                        start=1,
                        end=2,
                        group=SimpleNamespace(prompt_len=2),
                    ),
                ],
            ),
            seq_lens=torch.tensor([2, 4, 2, 4]),
            seq_lens_cpu=torch.tensor([2, 4, 2, 4]),
            req_pool_indices=torch.arange(4),
            device=torch.device("cpu"),
        )

    def test_ordinary_and_trie_outputs_cannot_share_keys(self):
        normal = ShapeKey(4)
        trie = BeamTrieShapeKey(4)
        self.assertNotEqual(normal, trie)
        self.assertEqual(len({normal: "dense", trie: "compact"}), 2)

    def test_staging_refreshes_all_request_data_and_clears_tail(self):
        state = TrieDecodeGraphState(self.runner, 4, "cpu")
        captured = SimpleNamespace()
        state.prepare_capture(captured, 4)
        source = build_metadata(self.batch, self.runner)
        forward = SimpleNamespace(
            beam_trie_levels=torch.tensor([1, 2, 1, 2]), beam_attention_metadata=source
        )
        original = source.buckets[0].context_slots.clone()
        for offset in (0, 100, 0):
            source.buckets[0].context_slots = original + offset
            target = captured.beam_attention_metadata.buckets[0]
            target.context_slots.fill_(999)
            target.decode_slots.fill_(999)
            state.stage(forward, 4, 4, 4)
            self.assertTrue(
                torch.equal(target.context_slots[: len(original)], original + offset)
            )
            self.assertEqual(
                torch.count_nonzero(target.context_slots[len(original) :]).item(), 0
            )
            columns = source.buckets[0].decode_slots.shape[1]
            self.assertEqual(
                torch.count_nonzero(target.decode_slots[:, columns:]).item(), 0
            )
            self.assertTrue(
                torch.equal(target.row_indices, source.buckets[0].row_indices)
            )
            self.assertTrue(
                torch.equal(target.decode_lens, source.buckets[0].decode_lens)
            )
            self.assertTrue(torch.equal(state.levels, forward.beam_trie_levels))

    def test_knob_changes_mixed_width_and_padding_reject_replay(self):
        state = TrieDecodeGraphState(self.runner, 4, "cpu")
        state.prepare_capture(SimpleNamespace(), 4)
        forward = SimpleNamespace(
            beam_trie_levels=torch.ones(4, dtype=torch.int64),
            beam_attention_metadata=build_metadata(self.batch, self.runner),
        )
        self.assertTrue(state.can_replay(forward, 4, 4))
        for field, value in (
            (envs.SGLANG_ENABLE_BEAM_SHARED_CONTEXT_ATTENTION, False),
            (envs.SGLANG_OPT_BEAM_TRIE_TOPK_LOGPROB, True),
            (envs.SGLANG_BEAM_SHARED_CONTEXT_WIDTH, 4),
            (envs.SGLANG_BEAM_SHARED_CONTEXT_GRAPH_MAX_CONTEXT, 8),
            (envs.SGLANG_BEAM_SHARED_CONTEXT_GRAPH_MAX_DECODE, 4),
        ):
            with self.subTest(field=field.name), field.override(value):
                self.assertFalse(state.can_replay(forward, 4, 4))
            self.assertTrue(state.can_replay(forward, 4, 4))
        self.assertFalse(state.can_replay(forward, 4, 8))
        forward.beam_attention_metadata.buckets.append(
            forward.beam_attention_metadata.buckets[0]
        )
        self.assertFalse(state.can_replay(forward, 4, 4))

    def test_invalid_staging_does_not_modify_prior_buffers(self):
        state = TrieDecodeGraphState(self.runner, 4, "cpu")
        state.prepare_capture(SimpleNamespace(), 4)
        snapshot = state.levels.clone()
        forward = SimpleNamespace(
            beam_trie_levels=torch.ones(3, dtype=torch.int64),
            beam_attention_metadata=build_metadata(self.batch, self.runner),
        )
        with self.assertRaisesRegex(ValueError, "captured configuration"):
            state.stage(forward, 4, 4, 4)
        self.assertTrue(torch.equal(state.levels, snapshot))


if __name__ == "__main__":
    unittest.main()
