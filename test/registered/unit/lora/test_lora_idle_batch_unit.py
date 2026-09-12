"""Idle DP-attention ranks (ForwardMode.IDLE, zero local tokens) must be accepted by the LoRA
segment helpers: MoE-LoRA batch prep calls them on every rank, and an idle rank used to raise
``ValueError: Unsupported forward mode: 4`` while capturing CUDA graphs (GLM-5.2, 32-GPU engine).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import types
import unittest

import torch

from sglang.srt.lora.utils import generate_sequence_lengths, get_batch_token_counts
from sglang.srt.model_executor.forward_batch_info import ForwardMode


def _batch(mode: ForwardMode, **kw) -> types.SimpleNamespace:
    return types.SimpleNamespace(forward_mode=mode, **kw)


class TestLoRAIdleBatch(unittest.TestCase):
    def test_idle_batch_has_zero_tokens(self):
        self.assertEqual(
            get_batch_token_counts(_batch(ForwardMode.IDLE, batch_size=0)), (0, 0)
        )

    def test_idle_batch_has_no_segments(self):
        seg_lens = generate_sequence_lengths(
            _batch(ForwardMode.IDLE, batch_size=0), device=torch.device("cpu")
        )
        self.assertEqual(seg_lens.numel(), 0)
        self.assertEqual(seg_lens.dtype, torch.int32)

    def test_decode_batch_unchanged(self):
        self.assertEqual(
            get_batch_token_counts(_batch(ForwardMode.DECODE, batch_size=3)), (3, 1)
        )
        seg_lens = generate_sequence_lengths(
            _batch(ForwardMode.DECODE, batch_size=3), device=torch.device("cpu")
        )
        self.assertEqual(seg_lens.tolist(), [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
