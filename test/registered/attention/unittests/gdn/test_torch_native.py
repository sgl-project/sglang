import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_hip
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.gdn_attention import (
    GDNAttentionCase,
    make_gdn_cases,
    run_gdn_attention_case,
)
from sglang.test.kits.attention_unittest.runner_modes.split_op_runner import (
    run_gdn_split_op_extend_case,
)

register_cuda_ci(est_time=10, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-large-amd")


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestTorchNativeGDNBackendCorrectness(CustomTestCase):
    CASES = make_gdn_cases("torch_native")
    SPLIT_OP_CASES = (
        (
            GDNAttentionCase(
                name="runner_split_op_gdn_extend_ragged_page_boundary",
                backend="torch_native",
                forward_mode=ForwardMode.EXTEND,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(0, 8, 16),
                extend_lens=(15, 8, 1),
            ),
            32,
        ),
    )

    def test_projected_gdn_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_gdn_attention_case(self, case)

    # Layout-robustness. See dense/test_triton.py for the rationale.
    LAYOUT_ROBUSTNESS_CASES = (
        GDNAttentionCase(
            name="layout_gdn_extend_two_request",
            backend="torch_native",
            forward_mode=ForwardMode.EXTEND,
            num_k_heads=4,
            num_v_heads=4,
            page_size=16,
            prefix_lens=(0, 0),
            extend_lens=(16, 16),
        ),
        GDNAttentionCase(
            name="layout_gdn_decode_page_boundary",
            backend="torch_native",
            forward_mode=ForwardMode.DECODE,
            num_k_heads=4,
            num_v_heads=4,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
    )

    def test_layout_robustness_cases(self):
        for case in self.LAYOUT_ROBUSTNESS_CASES:
            for layout in ("interleaved_pages", "non_monotonic_extend"):
                if layout == "non_monotonic_extend" and case.forward_mode.is_decode():
                    continue
                with self.subTest(case=case.name, layout=layout):
                    run_gdn_attention_case(self, case, loc_layout=layout)

    @unittest.skipIf(
        is_hip(),
        "split-op extend runner exercises the piecewise-CUDA-graph path "
        "(TcPiecewiseForwardContext.num_tokens), which is not wired on ROCm.",
    )
    def test_runner_mode_split_op_extend_cases(self):
        for case, static_num_tokens in self.SPLIT_OP_CASES:
            for breakable in (False, True):
                runner = "bcg" if breakable else "pcg"
                with self.subTest(
                    case=case.name,
                    backend=case.backend,
                    runner=runner,
                ):
                    run_gdn_split_op_extend_case(
                        self,
                        case,
                        breakable=breakable,
                        static_num_tokens=static_num_tokens,
                    )


if __name__ == "__main__":
    unittest.main()
