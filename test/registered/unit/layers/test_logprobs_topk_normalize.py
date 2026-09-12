import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.logprob_processor import (
    InputLogprobProcessor,
    OutputLogprobProcessor,
    compute_spec_logprobs,
    get_logprobs_topk_normalize,
    materialize_topk_normalized_logprobs,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestLogprobsTopKNormalize(unittest.TestCase):
    def test_normalizes_over_topk_and_floors_tail(self):
        logits = torch.tensor([[4.0, 2.0, 1.0, -1.0]])
        actual = materialize_topk_normalized_logprobs(logits, 2)
        expected_topk = torch.log_softmax(torch.tensor([[4.0, 2.0]]), dim=-1)
        expected = torch.tensor(
            [
                [
                    expected_topk[0, 0],
                    expected_topk[0, 1],
                    expected_topk[0, 1],
                    expected_topk[0, 1],
                ]
            ]
        )
        torch.testing.assert_close(actual, expected)

    def test_zero_preserves_full_vocab_normalization(self):
        logits = torch.tensor([[4.0, 2.0, 1.0, -1.0]])
        actual = materialize_topk_normalized_logprobs(logits, 0)
        torch.testing.assert_close(actual, torch.log_softmax(logits, dim=-1))

    def test_top50_normalizes_only_selected_support(self):
        logits = torch.arange(128, dtype=torch.float32).repeat(2, 1) / 16
        scores, indices = torch.topk(logits, 50, dim=-1)
        expected = torch.log_softmax(scores, dim=-1)
        actual = materialize_topk_normalized_logprobs(logits, 50)
        torch.testing.assert_close(actual.gather(-1, indices), expected)
        torch.testing.assert_close(actual[:, :78], expected[:, -1:].expand(2, 78))
        torch.testing.assert_close(expected.exp().sum(-1), torch.ones(2))

    def test_k_boundaries(self):
        logits = torch.tensor([[4.0, 2.0, 1.0, -1.0]])
        for k in (4, 5):
            torch.testing.assert_close(
                materialize_topk_normalized_logprobs(logits, k),
                torch.log_softmax(logits, dim=-1),
            )
        torch.testing.assert_close(
            materialize_topk_normalized_logprobs(logits, 1), torch.zeros_like(logits)
        )

    def test_requested_top_logprobs_cannot_exceed_support(self):
        with envs.SGLANG_LOGPROBS_TOPK_NORMALIZE.override(2):
            with self.assertRaisesRegex(ValueError, "exceeds"):
                OutputLogprobProcessor().compute_logprobs(
                    torch.log_softmax(torch.tensor([[4.0, 2.0, 1.0, -1.0]]), dim=-1),
                    top_logprobs_nums=[3],
                    token_ids_logprobs=[None],
                    batch_next_token_ids=torch.tensor([0]),
                )

    def test_prefill_topk_ids_across_chunks_and_tail_ties(self):
        # Floored tail tokens must not replace the top-k IDs at the boundary.
        logits = torch.tensor([[4.0, 2.0, 2.0, -1.0]]).repeat(5, 1)
        values, indices = torch.topk(logits, 2, dim=-1)
        values = torch.log_softmax(values, dim=-1)
        metadata = SimpleNamespace(
            sample_indices_cpu=[2, 4],
            input_logprob_indices_cpu=[0, 1, 2, 3, 4],
            extend_return_top_logprob=True,
            extend_token_ids_logprob=True,
            top_logprobs_nums=[2, 1],
            extend_logprob_pruned_lens_cpu=[3, 2],
            extend_input_logprob_token_ids_gpu=torch.tensor([3, 3, 3, 3, 3]),
            token_ids_logprobs=[[3], [3]],
        )
        for chunk_size in (1, 2, 5):
            for fast in (False, True):
                with self.subTest(chunk_size=chunk_size, fast=fast):
                    with envs.SGLANG_LOGPROBS_TOPK_NORMALIZE.override(2):
                        proc = InputLogprobProcessor(vocab_size=4)
                    proc.enable_logprobs_chunk = True
                    proc.logprobs_chunk_size = chunk_size
                    proc.enable_fast_input_logprobs = fast
                    result, sampled = proc.forward(
                        logits,
                        torch.tensor([2, 4]),
                        torch.tensor([0, 1, 2, 3, 4]),
                        [0, 0, 0, 1, 1],
                        None,
                        lambda states, *args, **kwargs: states,
                        metadata,
                    )
                    torch.testing.assert_close(sampled, logits[[2, 4]])
                    torch.testing.assert_close(result.token_logprobs, values[:, -1])
                    for seq, (rows, k) in enumerate(((3, 2), (2, 1))):
                        self.assertEqual(len(result.top_logprobs_idx[seq]), rows)
                        for row in range(rows):
                            torch.testing.assert_close(
                                torch.tensor(result.top_logprobs_idx[seq][row]),
                                indices[0, :k],
                            )
                            torch.testing.assert_close(
                                torch.tensor(result.top_logprobs_val[seq][row]),
                                values[0, :k],
                            )
                            torch.testing.assert_close(
                                torch.tensor(result.token_ids_logprobs_val[seq][row]),
                                values[0, -1:],
                            )

    def test_negative_setting_is_rejected(self):
        with patch.dict(os.environ, {"SGLANG_LOGPROBS_TOPK_NORMALIZE": "-1"}):
            with self.assertRaisesRegex(ValueError, "zero or a positive integer"):
                get_logprobs_topk_normalize()

    def test_output_processor_applies_flag_to_forced_tail_token(self):
        logits = torch.tensor([[4.0, 2.0, 1.0, -1.0]])
        full_vocab_logprobs = torch.log_softmax(logits, dim=-1)
        with envs.SGLANG_LOGPROBS_TOPK_NORMALIZE.override(2):
            result = OutputLogprobProcessor().compute_logprobs(
                full_vocab_logprobs,
                top_logprobs_nums=[2],
                token_ids_logprobs=[None],
                batch_next_token_ids=torch.tensor([3]),
            )

        expected_topk = torch.log_softmax(torch.tensor([[4.0, 2.0]]), dim=-1)
        torch.testing.assert_close(result.token_logprobs, expected_topk[:, 1])
        torch.testing.assert_close(result.top_logprobs_val[0], expected_topk.squeeze(0))
        torch.testing.assert_close(result.top_logprobs_idx[0], torch.tensor([0, 1]))

    def test_score_only_processor_applies_flag(self):
        logits = torch.tensor([[4.0, 2.0, 1.0, -1.0]])
        with envs.SGLANG_LOGPROBS_TOPK_NORMALIZE.override(2):
            result = OutputLogprobProcessor().compute_logprobs_only(
                logits,
                top_logprobs_nums=[2],
                token_ids_logprobs=[[3]],
                preprocess_fn=lambda value: value,
            )

        expected_topk = torch.log_softmax(torch.tensor([[4.0, 2.0]]), dim=-1)
        torch.testing.assert_close(
            result.token_ids_logprobs_val[0], expected_topk[:, 1]
        )
        torch.testing.assert_close(result.top_logprobs_idx[0], torch.tensor([0, 1]))

    def test_spec_processor_applies_flag(self):
        logits = torch.tensor([[4.0, 2.0, 1.0, -1.0]])
        batch = SimpleNamespace(
            seq_lens=[1],
            sampling_info=SimpleNamespace(is_all_greedy=True),
            top_logprobs_nums=[2],
            token_ids_logprobs=[],
        )
        output = SimpleNamespace(next_token_logits=logits)

        with envs.SGLANG_LOGPROBS_TOPK_NORMALIZE.override(2):
            compute_spec_logprobs(
                batch,
                output,
                predict=torch.tensor([3]),
                chain_stride=1,
            )

        expected_topk = torch.log_softmax(torch.tensor([[4.0, 2.0]]), dim=-1)
        torch.testing.assert_close(output.next_token_logprobs, expected_topk[:, 1:])
        torch.testing.assert_close(
            output.next_token_top_logprobs_idx[0], torch.tensor([0, 1])
        )


if __name__ == "__main__":
    unittest.main()
