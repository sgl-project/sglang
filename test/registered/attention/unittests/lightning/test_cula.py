import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang.srt.layers.attention.linear.cula_entry import CULA_AVAILABLE
from sglang.test.kits.attention_unittest.attention_methods.lightning_attention import (
    LightningAttentionCase,
    run_lightning_attention_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_target_verify_runner import (
    run_lightning_eagle_verify_case,
)

register_cuda_ci(est_time=300, suite="base-b-test-4-gpu-b200")


@unittest.skipUnless(
    CULA_AVAILABLE, "cuLA not installed (pip install cuda-linear-attention)"
)
@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestCulaLightningBackendCorrectness(CustomTestCase):
    """Lightning attention backend correctness with linear_backend='cula'.

    Mirrors the triton test structure but exercises the cuLA kernel path.
    The reference computation is backend-agnostic (pure-Python seg_la recurrence),
    so any output mismatch means the cuLA adapter conventions are wrong.
    """

    CASES = (
        LightningAttentionCase(
            name="cula_extend_page_size_1",
            backend="triton",
            linear_backend="cula",
            forward_mode=ForwardMode.EXTEND,
            num_heads=2,
            page_size=1,
            prefix_lens=(2, 4),
            extend_lens=(3, 1),
        ),
        LightningAttentionCase(
            name="cula_extend_zero_prefix_exact_page",
            backend="triton",
            linear_backend="cula",
            forward_mode=ForwardMode.EXTEND,
            num_heads=2,
            page_size=16,
            prefix_lens=(0,),
            extend_lens=(16,),
        ),
        LightningAttentionCase(
            name="cula_extend_two_request_page16",
            backend="triton",
            linear_backend="cula",
            forward_mode=ForwardMode.EXTEND,
            num_heads=2,
            page_size=16,
            prefix_lens=(0, 0),
            extend_lens=(16, 16),
        ),
        LightningAttentionCase(
            name="cula_decode_bsz1_no_prefix",
            backend="triton",
            linear_backend="cula",
            forward_mode=ForwardMode.DECODE,
            num_heads=2,
            page_size=16,
            prefix_lens=(0,),
        ),
        LightningAttentionCase(
            name="cula_decode_bsz1_nonzero_prefix",
            backend="triton",
            linear_backend="cula",
            forward_mode=ForwardMode.DECODE,
            num_heads=2,
            page_size=16,
            prefix_lens=(7,),
        ),
    )

    EAGLE_VERIFY_CASES = (
        (
            LightningAttentionCase(
                name="cula_eagle_verify_chain",
                backend="triton",
                linear_backend="cula",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "eagle",
        ),
        (
            LightningAttentionCase(
                name="cula_frozen_kv_mtp_verify_chain",
                backend="triton",
                linear_backend="cula",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "frozen_kv_mtp",
        ),
    )

    def test_projected_lightning_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, linear_backend=case.linear_backend):
                try:
                    run_lightning_attention_case(self, case)
                except RuntimeError as e:
                    if "Only Blackwell GPUs" in str(e):
                        self.skipTest(f"cuLA prefill requires SM100+: {e}")
                    raise

    def test_runner_mode_eagle_verify_cases(self):
        for case, topk, spec_kind in self.EAGLE_VERIFY_CASES:
            with self.subTest(
                case=case.name,
                linear_backend=case.linear_backend,
                topk=topk,
                spec_kind=spec_kind,
            ):
                try:
                    run_lightning_eagle_verify_case(
                        self, case, topk=topk, spec_kind=spec_kind
                    )
                except RuntimeError as e:
                    if "Only Blackwell GPUs" in str(e):
                        self.skipTest(f"cuLA prefill requires SM100+: {e}")
                    raise


if __name__ == "__main__":
    unittest.main()
