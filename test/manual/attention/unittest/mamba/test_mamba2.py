import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.attention_methods.mamba2_attention import (
    DEFAULT_CONV_KERNEL,
    DEFAULT_HEAD_DIM,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_MAMBA_CHUNK_SIZE,
    DEFAULT_N_GROUPS,
    DEFAULT_NUM_HEADS,
    DEFAULT_STATE_SIZE,
    Mamba2AttentionCase,
    run_mamba2_attention_case,
)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestTritonMamba2BackendCorrectness(CustomTestCase):
    CASES = (
        Mamba2AttentionCase(
            name="mamba2_extend_zero_prefix_exact_page",
            backend="triton",
            forward_mode=ForwardMode.EXTEND,
            num_heads=DEFAULT_NUM_HEADS,
            head_dim=DEFAULT_HEAD_DIM,
            state_size=DEFAULT_STATE_SIZE,
            n_groups=DEFAULT_N_GROUPS,
            conv_kernel=DEFAULT_CONV_KERNEL,
            mamba_chunk_size=DEFAULT_MAMBA_CHUNK_SIZE,
            hidden_size=DEFAULT_HIDDEN_SIZE,
            page_size=16,
            prefix_lens=(0,),
            extend_lens=(16,),
        ),
    )

    def test_projected_mamba2_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_mamba2_attention_case(self, case)


if __name__ == "__main__":
    unittest.main()
