import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.attention_methods.dense_attention import (
    DenseAttentionCase,
    make_swa_no_prefix_input_config_cases,
    make_swa_prefix_input_config_cases,
    run_dense_attention_case,
)


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


if __name__ == "__main__":
    unittest.main()
