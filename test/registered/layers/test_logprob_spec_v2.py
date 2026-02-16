# Copyright 2023-2024 SGLang Team
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
"""Unit tests for add_output_logprobs_for_spec_v2 (overlap spec v2 logprob support)."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.utils.logprob import add_output_logprobs_for_spec_v2


def _make_logits_output(next_token_logits):
    """Minimal mock for LogitsProcessorOutput to avoid pulling heavy deps (e.g. sgl_kernel)."""
    out = SimpleNamespace()
    out.next_token_logits = next_token_logits
    return out


def _make_mock_req(return_logprob: bool = True, top_logprobs_num: int = 0):
    req = SimpleNamespace()
    req.return_logprob = return_logprob
    req.top_logprobs_num = top_logprobs_num
    req.output_token_logprobs_val = []
    req.output_token_logprobs_idx = []
    req.output_top_logprobs_val = []
    req.output_top_logprobs_idx = []
    return req


def _make_mock_batch(
    bs: int,
    stride: int,
    num_tokens_per_req: list,
    top_logprobs_nums: list = None,
    device: str = "cpu",
):
    reqs = [_make_mock_req(return_logprob=True, top_logprobs_num=top_logprobs_nums[i] if top_logprobs_nums else 0) for i in range(bs)]
    temperatures = torch.ones(bs, device=device)
    sampling_info = SimpleNamespace(temperatures=temperatures)
    batch = SimpleNamespace()
    batch.reqs = reqs
    batch.top_logprobs_nums = top_logprobs_nums or [0] * bs
    batch.sampling_info = sampling_info
    return batch


class TestAddOutputLogprobsForSpecV2(unittest.TestCase):
    """Test add_output_logprobs_for_spec_v2 with mock batch and logits."""

    def test_empty_accept_index_returns_early(self):
        """When no token is accepted (all -1), function returns without modifying reqs."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        bs, stride, vocab = 2, 4, 8
        batch = _make_mock_batch(bs, stride, [0, 0], device=device)
        logits_output = _make_logits_output(
            torch.randn(bs * stride, vocab, device=device)
        )
        predict = torch.zeros(bs * stride, dtype=torch.long, device=device)
        accept_length = torch.zeros(bs, dtype=torch.int32, device=device)
        accept_index = torch.full((bs, stride), -1, dtype=torch.int32, device=device)

        add_output_logprobs_for_spec_v2(
            batch, logits_output, predict, accept_length, accept_index, stride
        )

        for req in batch.reqs:
            self.assertEqual(len(req.output_token_logprobs_val), 0)
            self.assertEqual(len(req.output_token_logprobs_idx), 0)

    def test_single_token_per_req_appends_correctly(self):
        """Two reqs, each with 1 accepted token; logprobs and token ids appended."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        bs, stride, vocab = 2, 4, 8
        # accept_index: req0 accepts position 0, req1 accepts position stride (4).
        accept_index = torch.full((bs, stride), -1, dtype=torch.int32, device=device)
        accept_index[0, 0] = 0
        accept_index[1, 0] = stride
        accept_length = torch.tensor([1, 1], dtype=torch.int32, device=device)
        # Predict token ids at accepted positions: 2 and 5.
        predict = torch.zeros(bs * stride, dtype=torch.long, device=device)
        predict[0] = 2
        predict[stride] = 5
        logits = torch.randn(bs * stride, vocab, device=device)
        logits[0, 2] = 10.0
        logits[stride, 5] = 10.0
        logits_output = _make_logits_output(logits)
        batch = _make_mock_batch(bs, stride, [1, 1], device=device)

        add_output_logprobs_for_spec_v2(
            batch, logits_output, predict, accept_length, accept_index, stride
        )

        self.assertEqual(len(batch.reqs[0].output_token_logprobs_val), 1)
        self.assertEqual(len(batch.reqs[0].output_token_logprobs_idx), 1)
        self.assertEqual(batch.reqs[0].output_token_logprobs_idx[0], 2)
        self.assertIsInstance(batch.reqs[0].output_token_logprobs_val[0], (float,))

        self.assertEqual(len(batch.reqs[1].output_token_logprobs_val), 1)
        self.assertEqual(len(batch.reqs[1].output_token_logprobs_idx), 1)
        self.assertEqual(batch.reqs[1].output_token_logprobs_idx[0], 5)

    def test_multiple_tokens_per_req_order(self):
        """One req with 3 accepted tokens; order and count match accept_length."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        bs, stride, vocab = 1, 4, 16
        accept_index = torch.full((bs, stride), -1, dtype=torch.int32, device=device)
        accept_index[0, 0] = 0
        accept_index[0, 1] = 1
        accept_index[0, 2] = 2
        accept_length = torch.tensor([3], dtype=torch.int32, device=device)
        predict = torch.arange(stride, dtype=torch.long, device=device)
        for i in range(3):
            predict[i] = i + 1
        logits = torch.randn(bs * stride, vocab, device=device)
        for i in range(3):
            logits[i, i + 1] = 5.0
        logits_output = _make_logits_output(logits)
        batch = _make_mock_batch(bs, stride, [3], device=device)

        add_output_logprobs_for_spec_v2(
            batch, logits_output, predict, accept_length, accept_index, stride
        )

        self.assertEqual(len(batch.reqs[0].output_token_logprobs_val), 3)
        self.assertEqual(batch.reqs[0].output_token_logprobs_idx, [1, 2, 3])

    def test_top_logprobs_nums_none_uses_zeros(self):
        """batch.top_logprobs_nums is None => treated as [0]*bs, no top append."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        bs, stride, vocab = 2, 2, 4
        accept_index = torch.full((2, 2), -1, dtype=torch.int32, device=device)
        accept_index[0, 0], accept_index[0, 1] = 0, 1
        accept_length = torch.tensor([2, 0], dtype=torch.int32, device=device)
        predict = torch.tensor([1, 2, 0, 0], dtype=torch.long, device=device)
        logits = torch.randn(4, vocab, device=device)
        logits[0, 1] = 1.0
        logits[1, 2] = 1.0
        batch = _make_mock_batch(2, stride, [2, 0], device=device)
        batch.top_logprobs_nums = None
        logits_output = _make_logits_output(logits)

        add_output_logprobs_for_spec_v2(
            batch, logits_output, predict, accept_length, accept_index, stride
        )

        self.assertEqual(len(batch.reqs[0].output_token_logprobs_val), 2)
        self.assertEqual(len(batch.reqs[0].output_top_logprobs_val), 0)


if __name__ == "__main__":
    unittest.main()
