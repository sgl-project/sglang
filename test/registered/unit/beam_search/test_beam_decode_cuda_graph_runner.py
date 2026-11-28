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
"""Beam decode graph inputs: loading a batch, and which batches replay."""

import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import torch

from sglang.srt.beam_search import beam_decode_cuda_graph_runner
from sglang.srt.beam_search.beam_decode_cuda_graph_runner import (
    BeamGraphInputs,
    FixedWidthGraphConfig,
    FixedWidthSharedContextInput,
    TrieLevelsInput,
)
from sglang.srt.beam_search.forward_inputs import BeamForwardInputs
from sglang.srt.beam_search.shared_context_attention import (
    BeamAttentionBucket,
    BeamAttentionMetadata,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

CONFIG = FixedWidthGraphConfig(beam_width=2, max_context_len=4, max_decode_len=3)


def _inputs(max_rows, *, trie_head=True) -> BeamGraphInputs:
    return BeamGraphInputs(
        trie_levels_input=(
            TrieLevelsInput(max_rows=max_rows, device="cpu") if trie_head else None
        ),
        shared_context_input=FixedWidthSharedContextInput(
            max_rows=max_rows, device="cpu", config=CONFIG
        ),
    )


def _metadata(prompt_lens, decode_lens, width=2) -> BeamAttentionMetadata:
    groups, steps = len(prompt_lens), max(decode_lens)
    return BeamAttentionMetadata(
        num_rows=groups * width,
        buckets=[
            BeamAttentionBucket(
                row_indices=torch.arange(groups * width, dtype=torch.int32)
                .view(groups, width)
                .flip(1),
                context_slots=100 + torch.arange(sum(prompt_lens), dtype=torch.int32),
                context_cu_seqlens=torch.tensor(
                    [0, *itertools.accumulate(prompt_lens)], dtype=torch.int32
                ),
                decode_slots=200
                + torch.arange(groups * width * steps, dtype=torch.int32).view(
                    groups, width * steps
                ),
                beam_width=width,
                decode_lens=torch.tensor(decode_lens, dtype=torch.int32),
            )
        ],
    )


class TestBeamGraphInputs(CustomTestCase):
    def test_loading_refreshes_every_tensor_the_graph_reads(self):
        for trie_head in (True, False):
            with self.subTest(trie_head=trie_head):
                inputs = _inputs(8, trie_head=trie_head)
                captured = inputs.prepare_capture_inputs(4)
                batch = BeamForwardInputs(
                    torch.tensor([1, 2, 1, 2]) if trie_head else None,
                    _metadata(prompt_lens=[1, 2], decode_lens=[1, 2]),
                )
                self.assertTrue(inputs.can_replay(batch, graph_size=4))
                inputs.load_replay_inputs(batch)

                if trie_head:
                    self.assertEqual(captured.trie_levels.tolist(), [1, 2, 1, 2])
                captured_bucket = captured.attention_metadata.buckets[0]
                batch_bucket = batch.attention_metadata.buckets[0]
                tensors = 0
                for field in msgspec.structs.fields(BeamAttentionBucket):
                    value = getattr(batch_bucket, field.name)
                    if not isinstance(value, torch.Tensor):
                        continue
                    tensors += 1
                    loaded = getattr(captured_bucket, field.name)[
                        tuple(map(slice, value.shape))
                    ]
                    self.assertEqual(loaded.tolist(), value.tolist(), field.name)
                self.assertEqual(tensors, 5)

    def test_only_a_batch_that_fits_the_graph_replays(self):
        inputs = _inputs(8)
        levels = torch.zeros(4, dtype=torch.int64)
        at_capacity = _metadata(
            prompt_lens=[CONFIG.max_context_len] * 2,
            decode_lens=[CONFIG.max_decode_len, 1],
        )
        self.assertTrue(
            inputs.can_replay(BeamForwardInputs(levels, at_capacity), graph_size=4)
        )

        two_widths = _metadata(prompt_lens=[1, 1], decode_lens=[1, 1])
        two_widths.buckets.append(two_widths.buckets[0])
        rejected = {
            "fewer rows than the graph": (
                levels,
                _metadata(prompt_lens=[1], decode_lens=[1]),
                4,
            ),
            "decode steps past the capacity": (
                levels,
                _metadata(
                    prompt_lens=[1, 1], decode_lens=[CONFIG.max_decode_len + 1, 1]
                ),
                4,
            ),
            "another beam width": (
                levels,
                _metadata(prompt_lens=[1], decode_lens=[1], width=4),
                4,
            ),
            "two widths in the batch": (levels, two_widths, 4),
        }
        for name, (trie_levels, metadata, graph_size) in rejected.items():
            with self.subTest(name):
                self.assertFalse(
                    inputs.can_replay(
                        BeamForwardInputs(trie_levels, metadata),
                        graph_size=graph_size,
                    )
                )


class TestFixedWidthSharedContextInput(CustomTestCase):
    def test_a_smaller_graph_captures_on_its_own_values(self):
        """A smaller graph reads the shared buffers at other offsets; what a larger
        graph's batch left there must not be captured."""
        shared_context_input = FixedWidthSharedContextInput(
            max_rows=8, device="cpu", config=CONFIG
        )
        shared_context_input.prepare_capture_inputs(8)
        shared_context_input.load_replay_inputs(
            _metadata(
                prompt_lens=[CONFIG.max_context_len] * 4,
                decode_lens=[CONFIG.max_decode_len] * 4,
            )
        )
        bucket = shared_context_input.prepare_capture_inputs(4).buckets[0]
        self.assertEqual(sorted(bucket.row_indices.flatten().tolist()), [0, 1, 2, 3])
        prompt_offsets = bucket.context_cu_seqlens.tolist()
        self.assertEqual(prompt_offsets[0], 0)
        self.assertEqual(prompt_offsets, sorted(prompt_offsets))
        self.assertLessEqual(prompt_offsets[-1], bucket.context_slots.numel())
        for decode_len in bucket.decode_lens.tolist():
            self.assertIn(decode_len, range(1, CONFIG.max_decode_len + 1))

    def test_graphs_need_a_width_and_a_token_budget(self):
        runner = SimpleNamespace(
            lora_manager=None,
            device="cpu",
            model_config=SimpleNamespace(context_len=16),
        )
        cases = (
            # (width, new token capacity, trie codebooks) -> max_decode_len or None
            ((2, None, None), None),
            ((2, 4, None), 3),
            ((2, None, 3), 2),
            ((2, 8, 3), 2),
        )
        for (width, new_token_capacity, codebooks), max_decode_len in cases:
            trie_config = (
                None if codebooks is None else SimpleNamespace(num_codebooks=codebooks)
            )
            with (
                self.subTest(
                    width=width,
                    new_token_capacity=new_token_capacity,
                    codebooks=codebooks,
                ),
                patch.object(
                    beam_decode_cuda_graph_runner,
                    "get_beam_trie_config",
                    return_value=trie_config,
                ),
                envs.SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH.override(width),
                envs.SGLANG_BEAM_DECODE_GRAPH_NEW_TOKEN_CAPACITY.override(
                    new_token_capacity
                ),
            ):
                attention = FixedWidthSharedContextInput.create(runner, max_rows=8)
                if max_decode_len is None:
                    self.assertIsNone(attention)
                else:
                    self.assertEqual(
                        attention.config,
                        FixedWidthGraphConfig(
                            beam_width=2,
                            max_context_len=16,
                            max_decode_len=max_decode_len,
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
