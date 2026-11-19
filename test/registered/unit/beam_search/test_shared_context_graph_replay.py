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
"""Shared-context trie graphs replay the batch staged into their shared buffers."""

import unittest

import torch

from sglang.kernels.ops.attention.beam_shared_context import shared_context_attention
from sglang.srt.beam_search.forward_inputs import BeamForwardInputs
from sglang.srt.beam_search.shared_context_attention import (
    BeamAttentionBucket,
    BeamAttentionMetadata,
)
from sglang.srt.beam_search.trie_cuda_graph import (
    SharedContextCapacity,
    TrieGraphInputs,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

WIDTH = 8
CAPACITY = SharedContextCapacity(beam_width=WIDTH, max_context=8, max_decode=3)
# A KV slot no staged batch uses: reading it changes the result.
STALE_SLOT = 63


def _attention(q, k, v, bucket):
    return shared_context_attention(
        q,
        k,
        v,
        bucket.row_indices,
        bucket.context_slots,
        bucket.context_cu_seqlens,
        bucket.decode_slots,
        bucket.decode_lens,
        softmax_scale=0.125,
    )


def _batch(groups, context, prompt_offsets, decode_lens, *, flip):
    """A batch of `groups` groups whose decode slots start at KV slot 8."""
    rows = torch.arange(groups * WIDTH, dtype=torch.int32, device="cuda").view(
        groups, WIDTH
    )
    decode = 8 + torch.arange(
        groups * WIDTH * max(decode_lens), dtype=torch.int32, device="cuda"
    ).view(groups, -1)
    bucket = BeamAttentionBucket(
        row_indices=rows.flip((0, 1)) if flip else rows,
        context_slots=torch.tensor(context, dtype=torch.int32, device="cuda"),
        context_cu_seqlens=torch.tensor(
            prompt_offsets, dtype=torch.int32, device="cuda"
        ),
        decode_slots=decode.flip(1) if flip else decode,
        beam_width=WIDTH,
        decode_lens=torch.tensor(decode_lens, dtype=torch.int32, device="cuda"),
    )
    return BeamAttentionMetadata(buckets=[bucket], num_rows=groups * WIDTH)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestSharedContextGraphReplay(CustomTestCase):
    def test_graphs_of_two_sizes_replay_the_staged_batch(self):
        """Graphs of every size read one set of buffers. Each replay must follow
        the staged lengths, slots and row mapping of its own batch, and read no
        slot past them, where another batch's values remain."""
        torch.manual_seed(17)
        q = torch.randn((2 * WIDTH, 4, 64), dtype=torch.bfloat16, device="cuda")
        k = torch.randn((64, 2, 64), dtype=torch.bfloat16, device="cuda")
        v = torch.randn_like(k)
        inputs = TrieGraphInputs(
            max_rows=2 * WIDTH, device="cuda", shared_context=CAPACITY
        )

        graphs = {}
        for rows in (2 * WIDTH, WIDTH):
            static = inputs.for_capture(rows).attention_metadata.buckets[0]
            _attention(q[:rows], k, v, static)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = _attention(q[:rows], k, v, static)
            graphs[rows] = (graph, output)

        batches = (
            _batch(2, list(range(8)), [0, 5, 8], [1, 2], flip=False),
            _batch(1, [5, 4, 3], [0, 3], [3], flip=True),
            _batch(2, [7, 6, 5], [0, 2, 3], [2, 1], flip=True),
            _batch(1, [1, 2], [0, 2], [1], flip=False),
        )
        static = inputs.for_capture(2 * WIDTH).attention_metadata.buckets[0]
        for metadata in batches:
            static.context_slots.fill_(STALE_SLOT)
            static.decode_slots.fill_(STALE_SLOT)
            rows = metadata.num_rows
            levels = torch.zeros(rows, dtype=torch.int64, device="cuda")
            inputs.stage(BeamForwardInputs(levels, metadata))
            graph, output = graphs[rows]
            graph.replay()
            torch.testing.assert_close(
                output.float(),
                _attention(q[:rows], k, v, metadata.buckets[0]).float(),
                rtol=1e-2,
                atol=1e-2,
            )


if __name__ == "__main__":
    unittest.main()
