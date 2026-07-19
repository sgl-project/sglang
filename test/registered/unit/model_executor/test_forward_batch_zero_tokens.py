import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    compute_local_num_token_non_padded,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestForwardBatchZeroTokens(unittest.TestCase):
    def test_non_empty_sample_keeps_standard_processing(self):
        runner = object.__new__(ModelRunner)
        runner._preprocess_logits = MagicMock()
        expected_token_ids = torch.tensor([3], dtype=torch.int64)
        runner.sampler = MagicMock(return_value=expected_token_ids)
        runner.ngram_embedding_manager = MagicMock()
        logits_output = SimpleNamespace(
            next_token_logits=torch.ones((1, 7), dtype=torch.float32)
        )
        sampling_info = object()
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            sampling_info=sampling_info,
            return_logprob=False,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            positions=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([1], dtype=torch.int64),
        )

        next_token_ids = runner.sample(logits_output, forward_batch)

        self.assertIs(next_token_ids, expected_token_ids)
        runner._preprocess_logits.assert_called_once_with(logits_output, sampling_info)
        runner.sampler.assert_called_once()
        runner.ngram_embedding_manager.update_after_decode.assert_called_once_with(
            next_token_ids=expected_token_ids,
            forward_batch=forward_batch,
        )

    def test_idle_sample_skips_empty_logit_processing(self):
        runner = object.__new__(ModelRunner)
        runner._preprocess_logits = MagicMock()
        runner.sampler = MagicMock()
        runner.ngram_embedding_manager = MagicMock()
        logits = torch.empty((0, 7), dtype=torch.float32)

        next_token_ids = runner.sample(
            SimpleNamespace(next_token_logits=logits),
            SimpleNamespace(forward_mode=ForwardMode.IDLE),
        )

        self.assertEqual(next_token_ids.shape, (0,))
        self.assertEqual(next_token_ids.dtype, torch.int64)
        self.assertEqual(next_token_ids.device, logits.device)
        runner._preprocess_logits.assert_not_called()
        runner.sampler.assert_not_called()
        runner.ngram_embedding_manager.update_after_decode.assert_not_called()

    def test_non_idle_sample_rejects_empty_logits(self):
        runner = object.__new__(ModelRunner)
        runner._preprocess_logits = MagicMock()
        runner.sampler = MagicMock()
        runner.ngram_embedding_manager = MagicMock()

        with self.assertRaisesRegex(
            AssertionError, "empty next-token logits.*idle DP rank"
        ):
            runner.sample(
                SimpleNamespace(
                    next_token_logits=torch.empty((0, 7), dtype=torch.float32)
                ),
                SimpleNamespace(forward_mode=ForwardMode.DECODE),
            )

        runner._preprocess_logits.assert_not_called()
        runner.sampler.assert_not_called()
        runner.ngram_embedding_manager.update_after_decode.assert_not_called()

    def test_idle_dp_rank_avoids_device_arithmetic(self):
        count = torch.tensor(0, dtype=torch.int32)
        batch = object.__new__(ForwardBatch)
        batch.global_num_tokens_cpu = [7, 0]
        batch.num_token_non_padded = count
        batch.num_token_non_padded_cpu = 0

        with (
            get_parallel().override(attn_dp_rank=1),
            patch("sglang.srt.utils.common.require_mlp_tp_gather", return_value=True),
            patch(
                "sglang.srt.model_executor.forward_batch_info.compute_local_num_token_non_padded"
            ) as compute,
        ):
            batch.adjust_num_token_non_padded_for_attn_tp(object())

        compute.assert_not_called()
        self.assertIs(batch.num_token_non_padded, count)

    def test_idle_dp_rank_requires_zero_cpu_mirror(self):
        batch = object.__new__(ForwardBatch)
        batch.global_num_tokens_cpu = [7, 0]
        batch.num_token_non_padded = torch.tensor(1, dtype=torch.int32)
        batch.num_token_non_padded_cpu = 1

        with (
            get_parallel().override(attn_dp_rank=1),
            patch("sglang.srt.utils.common.require_mlp_tp_gather", return_value=True),
            self.assertRaises(AssertionError),
        ):
            batch.adjust_num_token_non_padded_for_attn_tp(object())

    def test_nonempty_rank_keeps_local_clamp(self):
        count = torch.tensor(7, dtype=torch.int32)

        with get_parallel().override(attn_tp_size=2, attn_tp_rank=1):
            local = compute_local_num_token_non_padded(
                global_num_token_non_padded=count,
                num_tokens_per_dp=8,
            )

        self.assertEqual(local.item(), 3)


if __name__ == "__main__":
    unittest.main()
