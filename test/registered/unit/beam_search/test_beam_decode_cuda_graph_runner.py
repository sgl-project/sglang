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
"""What a beam decode graph replays: the batch loaded into its buffers, and only
a batch that fits them."""

import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import torch

from sglang.srt.beam_search import beam_decode_cuda_graph_runner
from sglang.srt.beam_search.beam_decode_cuda_graph_runner import (
    BeamGraphInputs,
    FixedWidthAttentionInput,
    FixedWidthCapacity,
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

CAPACITY = FixedWidthCapacity(beam_width=2, max_context_len=4, max_decode_len=3)


def _inputs(max_rows, *, trie_head=True, shared_context=True) -> BeamGraphInputs:
    return BeamGraphInputs(
        trie_levels_input=(
            TrieLevelsInput(max_rows=max_rows, device="cpu") if trie_head else None
        ),
        attention_input=(
            FixedWidthAttentionInput(max_rows=max_rows, device="cpu", capacity=CAPACITY)
            if shared_context
            else None
        ),
    )


def _metadata(prompt_lens, decode_lens, width=2) -> BeamAttentionMetadata:
    """One bucket as build_metadata lays it out, with distinct slot numbers."""
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
        """A graph reads its inputs by address: a batch tensor that loading
        skips leaves the previous batch's values in the replay. Under the dense
        LM head a batch carries no trie levels and replays all the same."""
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
        """Replaying a beam graph for a batch that does not fit its buffers
        returns the attention of other slots; such a batch runs eagerly."""
        inputs = _inputs(8)
        inputs.prepare_capture_inputs(4)
        levels = torch.zeros(4, dtype=torch.int64)
        at_capacity = _metadata(
            prompt_lens=[CAPACITY.max_context_len] * 2,
            decode_lens=[CAPACITY.max_decode_len, 1],
        )
        self.assertTrue(
            inputs.can_replay(BeamForwardInputs(levels, at_capacity), graph_size=4)
        )

        two_widths = _metadata(prompt_lens=[1, 1], decode_lens=[1, 1])
        two_widths.buckets.append(two_widths.buckets[0])
        rejected = {
            "no graph of this size": (
                torch.zeros(8, dtype=torch.int64),
                _metadata(prompt_lens=[1] * 4, decode_lens=[1] * 4),
                8,
            ),
            "fewer rows than the graph": (
                levels,
                _metadata(prompt_lens=[1], decode_lens=[1]),
                4,
            ),
            "decode steps past the capacity": (
                levels,
                _metadata(
                    prompt_lens=[1, 1], decode_lens=[CAPACITY.max_decode_len + 1, 1]
                ),
                4,
            ),
            "another beam width": (
                levels,
                _metadata(prompt_lens=[1], decode_lens=[1], width=4),
                4,
            ),
            "two widths in the batch": (levels, two_widths, 4),
            "ordinary attention": (levels, None, 4),
        }
        for name, (trie_levels, metadata, graph_size) in rejected.items():
            with self.subTest(name):
                self.assertFalse(
                    inputs.can_replay(
                        BeamForwardInputs(trie_levels, metadata),
                        graph_size=graph_size,
                    )
                )

    def test_ordinary_attention_graph_takes_no_shared_context_batch(self):
        """Without shared-context attention a beam graph holds FlashInfer
        attention, which a batch with shared-context metadata does not run."""
        inputs = _inputs(4, shared_context=False)
        inputs.prepare_capture_inputs(4)
        levels = torch.zeros(3, dtype=torch.int64)
        self.assertTrue(
            inputs.can_replay(BeamForwardInputs(levels, None), graph_size=4)
        )
        shared = _metadata(prompt_lens=[1, 1], decode_lens=[1, 1])
        self.assertFalse(
            inputs.can_replay(BeamForwardInputs(levels, shared), graph_size=4)
        )


class TestFixedWidthAttentionInput(CustomTestCase):
    def test_a_smaller_graph_captures_on_its_own_values(self):
        """Graphs of every size share the buffers, and a graph of fewer groups
        reads them at other offsets. What a larger graph's batch left there is
        no valid input: a length past the capacity reads outside the buffers."""
        attention_input = FixedWidthAttentionInput(
            max_rows=8, device="cpu", capacity=CAPACITY
        )
        attention_input.prepare_capture_inputs(8)
        attention_input.load_replay_inputs(
            _metadata(
                prompt_lens=[CAPACITY.max_context_len] * 4,
                decode_lens=[CAPACITY.max_decode_len] * 4,
            )
        )
        bucket = attention_input.prepare_capture_inputs(4).buckets[0]
        self.assertEqual(sorted(bucket.row_indices.flatten().tolist()), [0, 1, 2, 3])
        prompt_offsets = bucket.context_cu_seqlens.tolist()
        self.assertEqual(prompt_offsets[0], 0)
        self.assertEqual(prompt_offsets, sorted(prompt_offsets))
        self.assertLessEqual(prompt_offsets[-1], bucket.context_slots.numel())
        for decode_len in bucket.decode_lens.tolist():
            self.assertIn(decode_len, range(1, CAPACITY.max_decode_len + 1))

    def test_graphs_need_a_width_and_a_token_budget(self):
        """The graph is captured for one beam width and a fixed number of decode
        steps. Nothing but the operator knows the width, and without a trie
        nothing bounds the steps, so a server that sets neither captures no graph.
        A step attends the tokens generated before it: one fewer than the budget."""
        runner = SimpleNamespace(
            lora_manager=None,
            device="cpu",
            model_config=SimpleNamespace(context_len=16),
        )
        cases = (
            # (width, max new tokens, trie codebooks) -> max_decode_len, or None for no graph
            ((None, 4, None), None),
            ((2, None, None), None),
            ((2, 4, None), 3),
            ((2, None, 3), 2),
            ((2, 8, 3), 2),
        )
        for (width, max_new_tokens, codebooks), max_decode_len in cases:
            trie_config = (
                None if codebooks is None else SimpleNamespace(num_codebooks=codebooks)
            )
            with (
                self.subTest(
                    width=width, max_new_tokens=max_new_tokens, codebooks=codebooks
                ),
                patch.object(
                    beam_decode_cuda_graph_runner,
                    "get_beam_trie_config",
                    return_value=trie_config,
                ),
                envs.SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH.override(width),
                envs.SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH_MAX_NEW_TOKENS.override(
                    max_new_tokens
                ),
            ):
                attention = FixedWidthAttentionInput.create(runner, max_rows=8)
                if max_decode_len is None:
                    self.assertIsNone(attention)
                else:
                    self.assertEqual(
                        attention.capacity,
                        FixedWidthCapacity(
                            beam_width=2,
                            max_context_len=16,
                            max_decode_len=max_decode_len,
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
