# Copyright 2023-2026 SGLang Team
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

"""Keep LoRA lm_head segmentation aligned with logits-state pruning."""

import random
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.lora.utils import build_lm_head_pass_segments, get_lm_head_pruned_lens
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _reference_build_lm_head_pass_segments(
    weight_indices, pruned_lens, logprobs_chunk_size
):
    token_wi = []
    for wi, pl in zip(weight_indices, pruned_lens):
        token_wi.extend([wi] * pl)
    total = len(token_wi)
    num_passes = (total + logprobs_chunk_size - 1) // logprobs_chunk_size

    result = []
    for i in range(num_passes):
        start = i * logprobs_chunk_size
        end = min((i + 1) * logprobs_chunk_size, total)
        seg_wi = []
        seg_lens = []
        for t in range(start, end):
            if seg_wi and seg_wi[-1] == token_wi[t]:
                seg_lens[-1] += 1
            else:
                seg_wi.append(token_wi[t])
                seg_lens.append(1)
        result.append((seg_wi, seg_lens))
    return result


class TestLMHeadPruning(CustomTestCase):
    def test_lora_segments_match_pruned_rows_per_request(self):
        """Equal total rows must not hide routing one request through another LoRA."""
        cases = (
            {
                "name": "without_logprobs",
                "extend_seq_lens": [4, 5, 6],
                "return_logprob": False,
                "logprob_start_lens": None,
            },
            {
                "name": "with_logprobs",
                "extend_seq_lens": [4, 5, 6],
                "return_logprob": True,
                "logprob_start_lens": [0, 5, 3],
            },
        )

        for case in cases:
            with self.subTest(case=case["name"]):
                extend_seq_lens = case["extend_seq_lens"]
                sequence_ids = torch.repeat_interleave(
                    torch.arange(len(extend_seq_lens)),
                    torch.tensor(extend_seq_lens),
                )
                hidden_states = sequence_ids[:, None].expand(-1, 4).float()
                metadata = LogitsMetadata(
                    forward_mode=ForwardMode.EXTEND,
                    extend_return_logprob=case["return_logprob"],
                    extend_seq_lens=torch.tensor(extend_seq_lens),
                    extend_seq_lens_cpu=extend_seq_lens,
                    extend_logprob_start_lens_cpu=case["logprob_start_lens"],
                )
                pruned_states = LogitsProcessor._get_pruned_states(
                    None, hidden_states, None, None, metadata
                )[0]
                actual_lens = torch.bincount(
                    pruned_states[:, 0].long(), minlength=len(extend_seq_lens)
                ).tolist()

                forward_batch = SimpleNamespace(
                    forward_mode=ForwardMode.EXTEND,
                    batch_size=len(extend_seq_lens),
                    return_logprob=case["return_logprob"],
                    extend_seq_lens_cpu=extend_seq_lens,
                    extend_logprob_start_lens_cpu=case["logprob_start_lens"],
                )
                self.assertEqual(
                    get_lm_head_pruned_lens(forward_batch),
                    actual_lens,
                    "LoRA lm_head segment lengths must match the exact per-request "
                    "rows forwarded by logits pruning",
                )

    def test_build_lm_head_pass_segments_edge_cases_and_equivalence(self):
        self.assertEqual(build_lm_head_pass_segments([], [], 4), [])
        self.assertEqual(build_lm_head_pass_segments([0, 1], [0, 0], 4), [])
        self.assertEqual(
            build_lm_head_pass_segments([0], [8], 4),
            [([0], [4]), ([0], [4])],
        )
        self.assertEqual(
            build_lm_head_pass_segments([0, 1], [3, 3], 4),
            [([0, 1], [3, 1]), ([1], [2])],
        )
        # Zero-length sequence between two sequences with the same adapter
        self.assertEqual(
            build_lm_head_pass_segments([2, 9, 2], [2, 0, 3], 4),
            [([2], [4]), ([2], [1])],
        )
        with self.assertRaises(ValueError):
            build_lm_head_pass_segments([0], [4], 0)

        rng = random.Random(42)
        for _ in range(500):
            n = rng.randint(0, 10)
            weight_indices = [rng.randint(-1, 3) for _ in range(n)]
            pruned_lens = [rng.randint(0, 15) for _ in range(n)]
            chunk_size = rng.randint(1, 12)
            expected = _reference_build_lm_head_pass_segments(
                weight_indices, pruned_lens, chunk_size
            )
            actual = build_lm_head_pass_segments(
                weight_indices, pruned_lens, chunk_size
            )
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
