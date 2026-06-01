import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_flashinfer_available
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.dense_attention import (
    DenseAttentionCase,
    make_swa_no_prefix_input_config_cases,
    make_swa_prefix_input_config_cases,
    run_dense_attention_case,
)
from sglang.test.kits.attention_unittest.runner_modes.cuda_graph_decode_runner import (
    run_dense_cuda_graph_decode_case,
)
from sglang.test.kits.attention_unittest.runner_modes.split_op_runner import (
    run_dense_split_op_extend_case,
)

register_cuda_ci(est_time=14, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")


@unittest.skipIf(
    not torch.cuda.is_available() or not is_flashinfer_available(),
    "CUDA + flashinfer are required",
)
class TestFlashInferSWAAttentionBackendCorrectness(CustomTestCase):
    # FlashInfer SM90 prefill kernels require value head dim in {64, 128, 256}.
    HEAD_DIM = 64
    HIDDEN_SIZE = 256

    CASES = make_swa_no_prefix_input_config_cases(
        "flashinfer"
    ) + make_swa_prefix_input_config_cases("flashinfer")
    # Above-window decode case requires the `extend_window` reference rule
    # (window+1 keys), not the `min_seq_len_window` rule — FlashInfer's
    # decode metadata uses `clamp(seq_lens, max=window+1)` per
    # `flashinfer_backend.py:1031`. See `_SWA_DECODE_EXTEND_WINDOW` in
    # `common/attention_methods/dense_attention.py`.
    CUDA_GRAPH_CASES = (
        DenseAttentionCase(
            name="runner_cuda_graph_swa_decode_within_window",
            backend="flashinfer",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(1, 2, 3),
            sliding_window_size=4,
        ),
        DenseAttentionCase(
            name="runner_cuda_graph_swa_decode_above_window",
            backend="flashinfer",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(7, 8, 9),
            sliding_window_size=4,
        ),
    )
    # NOTE: a `runner_split_op_swa_extend_prefix_within_window` clone of the
    # triton SWA test fails on flashinfer (~0.21 max diff). FlashInfer's
    # prefill-split path does not handle SWA prefix the same way as triton;
    # the projected EXTEND covers the prefix path through the unsplit kernel
    # which does match the reference. Investigate before adding split_op
    # prefix to flashinfer SWA.
    SPLIT_OP_CASES = (
        (
            DenseAttentionCase(
                name="runner_split_op_swa_extend_no_prefix_window_edges",
                backend="flashinfer",
                forward_mode=ForwardMode.EXTEND,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(0, 0, 0),
                extend_lens=(3, 4, 5),
                sliding_window_size=4,
            ),
            16,
        ),
    )

    def test_projected_swa_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_attention_case(
                    self,
                    case,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                )

    # Layout-robustness. See dense/test_triton.py for the full rationale.
    # The default `shuffled_pages` is already exercised by
    # test_projected_swa_attention_cases on the existing case list.
    # This method opts into the more aggressive interleaved_pages +
    # non_monotonic_extend on within-window extend + decode.
    LAYOUT_ROBUSTNESS_CASES = (
        DenseAttentionCase(
            name="layout_swa_extend_below_window",
            backend="flashinfer",
            forward_mode=ForwardMode.EXTEND,
            num_heads=8,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(0,),
            extend_lens=(10,),
            sliding_window_size=12,
        ),
        DenseAttentionCase(
            name="layout_swa_decode_within_window",
            backend="flashinfer",
            forward_mode=ForwardMode.DECODE,
            num_heads=8,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(8, 10),
            sliding_window_size=12,
        ),
    )

    def test_layout_robustness_cases(self):
        for case in self.LAYOUT_ROBUSTNESS_CASES:
            for layout in ("interleaved_pages", "non_monotonic_extend"):
                if layout == "non_monotonic_extend" and case.forward_mode.is_decode():
                    continue
                with self.subTest(case=case.name, layout=layout):
                    run_dense_attention_case(
                        self,
                        case,
                        head_dim=self.HEAD_DIM,
                        hidden_size=self.HIDDEN_SIZE,
                        loc_layout=layout,
                    )

    def test_runner_mode_cuda_graph_decode_cases(self):
        for case in self.CUDA_GRAPH_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_cuda_graph_decode_case(
                    self,
                    case,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
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
                    run_dense_split_op_extend_case(
                        self,
                        case,
                        breakable=breakable,
                        static_num_tokens=static_num_tokens,
                        head_dim=self.HEAD_DIM,
                        hidden_size=self.HIDDEN_SIZE,
                    )


if __name__ == "__main__":
    unittest.main()
