"""Regression: replay must read the same complete window as eager draft extend."""

import unittest

import torch

from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.dense_attention import (
    DenseAttentionCase,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_draft_window_runner import (
    run_dense_eagle_draft_extend_window_case,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=40, stage="base-b", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestFlashInferDraftWindowAttention(CustomTestCase):
    def test_draft_extend_rows_read_their_window(self):
        case = DenseAttentionCase(
            name="draft_extend_window_flashinfer",
            backend="flashinfer",
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            num_heads=4,
            num_kv_heads=2,
            page_size=16,
            prefix_lens=(2, 9, 20),
            extend_lens=(4, 4, 4),
            sliding_window_size=8,
        )
        run_dense_eagle_draft_extend_window_case(
            self, case, num_accept_tokens=(4, 1, 2), head_dim=64, hidden_size=256
        )


if __name__ == "__main__":
    unittest.main()
