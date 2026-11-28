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
"""Shared-context beam graphs replay the batch loaded into their shared buffers."""

import unittest

import torch

from sglang.kernels.ops.attention.beam_shared_context import shared_context_attention
from sglang.srt.beam_search.beam_decode_cuda_graph_runner import (
    BeamGraphInputs,
    FixedWidthGraphConfig,
    FixedWidthSharedContextInput,
)
from sglang.srt.beam_search.forward_inputs import BeamForwardInputs
from sglang.srt.beam_search.shared_context_attention import (
    BeamAttentionBucket,
    BeamAttentionMetadata,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

WIDTH = 8
CONFIG = FixedWidthGraphConfig(beam_width=WIDTH, max_context_len=8, max_decode_len=3)
# A KV slot no loaded batch uses: reading it changes the result.
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
    def test_graphs_of_two_sizes_replay_the_loaded_batch(self):
        """Each replay follows the batch loaded for it and reads no stale slot."""
        torch.manual_seed(17)
        q = torch.randn((2 * WIDTH, 4, 64), dtype=torch.bfloat16, device="cuda")
        k = torch.randn((64, 2, 64), dtype=torch.bfloat16, device="cuda")
        v = torch.randn_like(k)
        inputs = BeamGraphInputs(
            trie_levels_input=None,
            shared_context_input=FixedWidthSharedContextInput(
                max_rows=2 * WIDTH, device="cuda", config=CONFIG
            ),
        )

        graphs = {}
        for rows in (2 * WIDTH, WIDTH):
            captured = inputs.prepare_capture_inputs(rows)
            graph_bucket = captured.attention_metadata.buckets[0]
            _attention(q[:rows], k, v, graph_bucket)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = _attention(q[:rows], k, v, graph_bucket)
            graphs[rows] = (graph, output, graph_bucket)

        batches = (
            _batch(2, list(range(8)), [0, 5, 8], [1, 2], flip=False),
            _batch(1, [5, 4, 3], [0, 3], [3], flip=True),
            _batch(2, [7, 6, 5], [0, 2, 3], [2, 1], flip=True),
            _batch(1, [1, 2], [0, 2], [1], flip=False),
        )
        for metadata in batches:
            rows = metadata.num_rows
            graph, output, graph_bucket = graphs[rows]
            graph_bucket.context_slots.fill_(STALE_SLOT)
            graph_bucket.decode_slots.fill_(STALE_SLOT)
            inputs.load_replay_inputs(BeamForwardInputs(None, metadata))
            graph.replay()
            torch.testing.assert_close(
                output.float(),
                _attention(q[:rows], k, v, metadata.buckets[0]).float(),
                rtol=1e-2,
                atol=1e-2,
            )


if __name__ == "__main__":
    unittest.main()
