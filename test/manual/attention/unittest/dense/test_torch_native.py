import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang.test.kits.attention_unittest.attention_methods.dense_attention import (
    DenseAttentionCase,
    make_dense_cases,
    run_dense_attention_case,
)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestTorchNativeDenseAttentionBackendCorrectness(CustomTestCase):
    CASES = make_dense_cases("torch_native")
    RUNNER_EAGER_CASES = (
        DenseAttentionCase(
            name="runner_eager_decode_page_boundary",
            backend="torch_native",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
        DenseAttentionCase(
            name="runner_eager_extend_ragged_page_boundary",
            backend="torch_native",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(0, 8, 16),
            extend_lens=(15, 8, 1),
        ),
        DenseAttentionCase(
            name="runner_eager_gqa_decode_page_boundary",
            backend="torch_native",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=2,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
        DenseAttentionCase(
            name="runner_eager_mqa_decode_bsz1",
            backend="torch_native",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=1,
            page_size=16,
            prefix_lens=(7,),
        ),
    )

    def test_projected_dense_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_attention_case(self, case)

    def test_runner_mode_eager_cases(self):
        for case in self.RUNNER_EAGER_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_attention_case(self, case)

    # Layout-robustness. See dense/test_triton.py for full rationale.
    # torch_native uses PyTorch SDPA on per-token-loc K/V gathered from
    # the cache, so all non-tidy layouts pass.
    LAYOUT_ROBUSTNESS_CASES = (
        DenseAttentionCase(
            name="layout_extend_two_request_ragged",
            backend="torch_native",
            forward_mode=ForwardMode.EXTEND,
            num_heads=12,
            num_kv_heads=12,
            page_size=16,
            prefix_lens=(8, 16),
            extend_lens=(8, 16),
        ),
        DenseAttentionCase(
            name="layout_decode_page_boundary",
            backend="torch_native",
            forward_mode=ForwardMode.DECODE,
            num_heads=12,
            num_kv_heads=12,
            page_size=16,
            prefix_lens=(15, 16, 17),
        ),
    )

    def test_layout_robustness_cases(self):
        for case in self.LAYOUT_ROBUSTNESS_CASES:
            # shuffled_pages is the default and already covered.
            for layout in (
                "interleaved_pages",
                "non_monotonic_extend",
            ):
                if (
                    layout == "non_monotonic_extend"
                    and case.forward_mode.is_decode()
                ):
                    continue
                with self.subTest(case=case.name, layout=layout):
                    run_dense_attention_case(self, case, loc_layout=layout)


if __name__ == "__main__":
    unittest.main()
