import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.layers.logits_processor as logits_processor
from sglang.srt.layers.logits_processor import LogitsMetadata
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLogitsDpAttentionMetadata(CustomTestCase):
    def _compute(self, metadata, dp_rank):
        parallel = SimpleNamespace(attn_dp_rank=dp_rank)
        with (
            patch.object(logits_processor, "get_parallel", return_value=parallel),
            patch.object(logits_processor, "get_dp_hidden_size", return_value=4),
            patch.object(logits_processor, "get_dp_dtype", return_value=torch.float32),
            patch.object(logits_processor, "get_dp_device", return_value="cpu"),
        ):
            metadata.compute_dp_attention_metadata()

    def test_cpu_metadata_avoids_device_cumsum_for_idle_rank(self):
        metadata = LogitsMetadata(
            forward_mode=ForwardMode.IDLE,
            global_num_tokens_for_logprob_cpu=[6, 0],
            global_num_tokens_for_logprob_gpu=torch.tensor([6, 0]),
            global_dp_buffer_len=6,
        )

        with patch.object(torch, "cumsum", side_effect=AssertionError("unexpected")):
            self._compute(metadata, dp_rank=1)

        self.assertEqual(metadata.dp_local_start_pos.item(), 6)
        self.assertEqual(metadata.dp_local_num_tokens.item(), 0)
        self.assertEqual(tuple(metadata.gathered_buffer.shape), (6, 4))

    def test_gpu_metadata_fallback_keeps_cumsum_path(self):
        metadata = LogitsMetadata(
            forward_mode=ForwardMode.IDLE,
            global_num_tokens_for_logprob_cpu=None,
            global_num_tokens_for_logprob_gpu=torch.tensor([3, 5]),
            global_dp_buffer_len=8,
        )

        with patch.object(torch, "cumsum", wraps=torch.cumsum) as cumsum:
            self._compute(metadata, dp_rank=1)

        cumsum.assert_called_once()
        self.assertEqual(metadata.dp_local_start_pos.item(), 3)
        self.assertEqual(metadata.dp_local_num_tokens.item(), 5)
        self.assertEqual(tuple(metadata.gathered_buffer.shape), (8, 4))


if __name__ == "__main__":
    unittest.main()
