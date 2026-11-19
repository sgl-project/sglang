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
"""What a trie decode graph replays: the staged batch, and only a batch it holds."""

import itertools
import unittest

import msgspec
import torch

from sglang.srt.beam_search.forward_inputs import BeamForwardInputs
from sglang.srt.beam_search.shared_context_attention import (
    BeamAttentionBucket,
    BeamAttentionMetadata,
)
from sglang.srt.beam_search.trie_cuda_graph import (
    SharedContextCapacity,
    TrieGraphInputs,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

CAPACITY = SharedContextCapacity(beam_width=2, max_context=4, max_decode=3)


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


class TestTrieGraphInputs(CustomTestCase):
    def test_staging_refreshes_every_tensor_the_graph_reads(self):
        """A graph reads its inputs by address: a batch tensor that staging
        skips leaves the previous batch's values in the replay."""
        inputs = TrieGraphInputs(max_rows=8, device="cpu", shared_context=CAPACITY)
        captured = inputs.for_capture(4).attention_metadata.buckets[0]
        batch = BeamForwardInputs(
            torch.tensor([1, 2, 1, 2]),
            _metadata(prompt_lens=[1, 2], decode_lens=[1, 2]),
        )
        inputs.stage(batch)

        self.assertEqual(inputs.levels[:4].tolist(), [1, 2, 1, 2])
        source = batch.attention_metadata.buckets[0]
        tensors = 0
        for field in msgspec.structs.fields(BeamAttentionBucket):
            value = getattr(source, field.name)
            if not isinstance(value, torch.Tensor):
                continue
            tensors += 1
            staged = getattr(captured, field.name)[tuple(map(slice, value.shape))]
            self.assertEqual(staged.tolist(), value.tolist(), field.name)
        self.assertEqual(tensors, 5)

    def test_only_a_batch_the_graph_holds_replays(self):
        """Replaying a trie graph for a batch that does not fit its buffers
        returns the attention of other slots; such a batch runs eagerly."""
        inputs = TrieGraphInputs(max_rows=8, device="cpu", shared_context=CAPACITY)
        inputs.for_capture(4)
        levels = torch.zeros(4, dtype=torch.int64)
        at_capacity = _metadata(
            prompt_lens=[CAPACITY.max_context] * 2,
            decode_lens=[CAPACITY.max_decode, 1],
        )
        self.assertTrue(
            inputs.can_replay(BeamForwardInputs(levels, at_capacity), captured_rows=4)
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
                _metadata(prompt_lens=[1, 1], decode_lens=[CAPACITY.max_decode + 1, 1]),
                4,
            ),
            "another beam width": (
                levels,
                _metadata(prompt_lens=[1], decode_lens=[1], width=4),
                4,
            ),
            "two widths in the batch": (levels, two_widths, 4),
            "ordinary attention": (levels, None, 4),
            "dense head": (None, at_capacity, 4),
        }
        for name, (trie_levels, metadata, captured_rows) in rejected.items():
            with self.subTest(name):
                self.assertFalse(
                    inputs.can_replay(
                        BeamForwardInputs(trie_levels, metadata),
                        captured_rows=captured_rows,
                    )
                )

    def test_ordinary_attention_graph_takes_no_shared_context_batch(self):
        """Without shared-context attention a trie graph holds FlashInfer
        attention, which a batch with shared-context metadata does not run."""
        inputs = TrieGraphInputs(max_rows=4, device="cpu", shared_context=None)
        inputs.for_capture(4)
        levels = torch.zeros(3, dtype=torch.int64)
        self.assertTrue(
            inputs.can_replay(BeamForwardInputs(levels, None), captured_rows=4)
        )
        shared = _metadata(prompt_lens=[1, 1], decode_lens=[1, 1])
        self.assertFalse(
            inputs.can_replay(BeamForwardInputs(levels, shared), captured_rows=4)
        )


if __name__ == "__main__":
    unittest.main()
