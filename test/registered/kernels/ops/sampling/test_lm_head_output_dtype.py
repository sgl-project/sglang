"""Preserve close greedy scores without changing LM-head operand precision."""

import unittest

import torch
from torch import nn

from sglang.srt.configs.inkling import InklingMMConfig, InklingModelConfig
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.runtime_context import reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-small")


class TestLMHeadOutputDtype(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="test")

    def test_bf16_output_rounding_changes_greedy_winner(self):
        """BF16 output ties scores that are distinct with the same BF16 operands."""
        head = nn.Linear(2, 2, bias=False, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[1.0, 1 / 1024], [1.0, 1 / 512]]))
        hidden = torch.ones((1, 2), device="cuda", dtype=torch.bfloat16)
        processor = LogitsProcessor(
            InklingModelConfig(vocab_size=2), skip_all_gather=True
        )
        logits = processor._compute_lm_head(hidden, head)
        self.assertFalse(processor.use_fp32_lm_head)
        self.assertEqual(hidden.dtype, torch.bfloat16)
        self.assertEqual(head.weight.dtype, torch.bfloat16)
        self.assertEqual(logits.dtype, torch.float32)
        self.assertEqual(logits.argmax(dim=-1).item(), 1)
        torch.testing.assert_close(
            logits,
            torch.tensor([[1 + 1 / 1024, 1 + 1 / 512]], device="cuda"),
            rtol=0,
            atol=0,
        )

        config = InklingMMConfig(
            text_config={"vocab_size": 2, "enable_lm_head_output_fp32": False}
        )
        restored = InklingMMConfig.from_dict(config.to_dict())
        processor = LogitsProcessor(restored.text_config, skip_all_gather=True)
        rounded = processor._compute_lm_head(hidden, head)
        self.assertEqual(rounded.dtype, torch.bfloat16)
        self.assertEqual(rounded.argmax(dim=-1).item(), 0)


if __name__ == "__main__":
    unittest.main()
