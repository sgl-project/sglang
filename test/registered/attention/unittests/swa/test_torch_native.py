import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.dense_attention import (
    DenseAttentionCase,
    make_swa_no_prefix_input_config_cases,
    make_swa_prefix_input_config_cases,
    run_dense_attention_case,
)

register_cuda_ci(est_time=10, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-large-amd")


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestTorchNativeSWAAttentionBackendCorrectness(CustomTestCase):
    CASES = (
        make_swa_no_prefix_input_config_cases("torch_native")
        + make_swa_prefix_input_config_cases("torch_native")
        + (
            DenseAttentionCase(
                name="swa_decode_window_edges",
                backend="torch_native",
                forward_mode=ForwardMode.DECODE,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(3, 4, 5),
                sliding_window_size=4,
            ),
            DenseAttentionCase(
                name="swa_gqa_decode_window_edges",
                backend="torch_native",
                forward_mode=ForwardMode.DECODE,
                num_heads=4,
                num_kv_heads=2,
                page_size=16,
                prefix_lens=(3, 4, 5),
                sliding_window_size=4,
            ),
        )
    )
    # Eager runner-mode cases mirroring `dense/test_torch_native.py`.
    # `torch_native` is the only SWA backend with no CG / split-op
    # support (it raises `NotImplementedError` from
    # `BaseAttnBackend.init_*_cuda_graph`), so the eager path is the
    # only runner mode worth exercising. Cases pick up the SWA window
    # via `sliding_window_size`.
    RUNNER_EAGER_CASES = (
        DenseAttentionCase(
            name="runner_eager_swa_decode_window_edges",
            backend="torch_native",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(3, 4, 5),
            sliding_window_size=4,
        ),
        DenseAttentionCase(
            name="runner_eager_swa_extend_within_window",
            backend="torch_native",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(0,),
            extend_lens=(3,),
            sliding_window_size=4,
        ),
        DenseAttentionCase(
            name="runner_eager_swa_gqa_decode_window_edges",
            backend="torch_native",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=2,
            page_size=16,
            prefix_lens=(3, 4, 5),
            sliding_window_size=4,
        ),
    )

    def test_projected_swa_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_attention_case(self, case)

    def test_runner_mode_eager_cases(self):
        for case in self.RUNNER_EAGER_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_attention_case(self, case)

    # Layout-robustness. See dense/test_triton.py for the rationale.
    # torch_native SWA gathers K/V via cache locs without page-table
    # arithmetic, so it's robust to all non-tidy layouts.
    LAYOUT_ROBUSTNESS_CASES = (
        DenseAttentionCase(
            name="layout_swa_extend_within_window",
            backend="torch_native",
            forward_mode=ForwardMode.EXTEND,
            num_heads=8,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(8, 16),
            extend_lens=(8, 16),
            sliding_window_size=12,
        ),
        DenseAttentionCase(
            name="layout_swa_decode_page_boundary",
            backend="torch_native",
            forward_mode=ForwardMode.DECODE,
            num_heads=8,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(15, 16, 17),
            sliding_window_size=12,
        ),
    )

    def test_layout_robustness_cases(self):
        for case in self.LAYOUT_ROBUSTNESS_CASES:
            for layout in ("interleaved_pages", "non_monotonic_extend"):
                if layout == "non_monotonic_extend" and case.forward_mode.is_decode():
                    continue
                with self.subTest(case=case.name, layout=layout):
                    run_dense_attention_case(self, case, loc_layout=layout)


if __name__ == "__main__":
    unittest.main()
