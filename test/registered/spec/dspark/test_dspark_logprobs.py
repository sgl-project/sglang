import types
import unittest

import torch

from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDSparkLogprobs(CustomTestCase):
    def _worker(self, stride: int) -> DSparkWorkerV2:
        worker = object.__new__(DSparkWorkerV2)
        worker.verify_num_draft_tokens = stride
        worker._linear_accept_index_cache = None
        return worker

    def _batch(
        self,
        *,
        bs: int,
        return_logprob: bool = True,
        temperatures: torch.Tensor | None = None,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
    ):
        if temperatures is None:
            temperatures = torch.ones((bs, 1), dtype=torch.float32)
        return types.SimpleNamespace(
            return_logprob=return_logprob,
            seq_lens=torch.ones(bs, dtype=torch.int64),
            sampling_info=types.SimpleNamespace(
                is_all_greedy=bool(torch.all(temperatures == 1)),
                temperatures=temperatures,
            ),
            top_logprobs_nums=top_logprobs_nums,
            token_ids_logprobs=token_ids_logprobs,
        )

    @staticmethod
    def _logits(rows: int, vocab: int):
        values = torch.arange(rows * vocab, dtype=torch.float32)
        return types.SimpleNamespace(next_token_logits=values.view(rows, vocab) / 10)

    def test_greedy_logprobs_follow_strided_verify_rows(self):
        bs, stride, vocab = 2, 4, 7
        out_tokens = torch.tensor([[1, 3, 5, 0], [6, 4, 2, 1]], dtype=torch.int64)
        logits_output = self._logits(bs * stride, vocab)

        self._worker(stride)._compute_output_logprobs(
            batch=self._batch(bs=bs),
            logits_output=logits_output,
            out_tokens=out_tokens,
        )

        logprobs = torch.log_softmax(logits_output.next_token_logits, dim=-1)
        expected = logprobs[torch.arange(bs * stride), out_tokens.reshape(-1)].view(
            bs, stride
        )
        torch.testing.assert_close(logits_output.next_token_logprobs, expected)

    def test_sampling_logprobs_apply_per_request_temperature(self):
        bs, stride, vocab = 2, 3, 5
        temperatures = torch.tensor([[0.5], [2.0]], dtype=torch.float32)
        out_tokens = torch.tensor([[0, 1, 2], [2, 3, 4]], dtype=torch.int64)
        logits_output = self._logits(bs * stride, vocab)

        self._worker(stride)._compute_output_logprobs(
            batch=self._batch(bs=bs, temperatures=temperatures),
            logits_output=logits_output,
            out_tokens=out_tokens,
        )

        scaled = logits_output.next_token_logits / torch.repeat_interleave(
            temperatures, stride, dim=0
        )
        expected_all = torch.log_softmax(scaled, dim=-1)
        expected = expected_all[torch.arange(bs * stride), out_tokens.reshape(-1)].view(
            bs, stride
        )
        torch.testing.assert_close(logits_output.next_token_logprobs, expected)

    def test_optional_logprob_outputs_cover_every_verify_row(self):
        bs, stride, vocab = 2, 3, 6
        out_tokens = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int64)
        logits_output = self._logits(bs * stride, vocab)

        self._worker(stride)._compute_output_logprobs(
            batch=self._batch(
                bs=bs,
                top_logprobs_nums=[2, 1],
                token_ids_logprobs=[[0, 5], [1]],
            ),
            logits_output=logits_output,
            out_tokens=out_tokens,
        )

        self.assertEqual(len(logits_output.next_token_top_logprobs_val), bs * stride)
        self.assertEqual(len(logits_output.next_token_top_logprobs_idx), bs * stride)
        self.assertEqual(
            len(logits_output.next_token_token_ids_logprobs_val), bs * stride
        )
        self.assertEqual(
            len(logits_output.next_token_token_ids_logprobs_idx), bs * stride
        )

    def test_index_cache_reuses_and_resizes_storage(self):
        stride, vocab = 3, 5
        worker = self._worker(stride)

        worker._compute_output_logprobs(
            batch=self._batch(bs=2),
            logits_output=self._logits(2 * stride, vocab),
            out_tokens=torch.zeros((2, stride), dtype=torch.int64),
        )
        initial_cache = worker._linear_accept_index_cache

        worker._compute_output_logprobs(
            batch=self._batch(bs=1),
            logits_output=self._logits(stride, vocab),
            out_tokens=torch.zeros((1, stride), dtype=torch.int64),
        )
        self.assertIs(worker._linear_accept_index_cache, initial_cache)

        worker._compute_output_logprobs(
            batch=self._batch(bs=3),
            logits_output=self._logits(3 * stride, vocab),
            out_tokens=torch.zeros((3, stride), dtype=torch.int64),
        )
        self.assertIsNot(worker._linear_accept_index_cache, initial_cache)
        torch.testing.assert_close(
            worker._linear_accept_index_cache,
            torch.arange(3 * stride, dtype=torch.int64),
        )

    def test_disabled_logprobs_do_not_modify_logits_output(self):
        bs, stride, vocab = 2, 3, 5
        logits_output = self._logits(bs * stride, vocab)

        self._worker(stride)._compute_output_logprobs(
            batch=self._batch(bs=bs, return_logprob=False),
            logits_output=logits_output,
            out_tokens=torch.zeros((bs, stride), dtype=torch.int64),
        )

        self.assertFalse(hasattr(logits_output, "next_token_logprobs"))


if __name__ == "__main__":
    unittest.main()
