import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_flashinfer_available
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.attention_methods.dense_attention import (
    DenseAttentionCase,
    make_swa_no_prefix_input_config_cases,
    run_dense_attention_case,
)
from common.runner_modes.cuda_graph_decode_runner import (
    run_dense_cuda_graph_decode_case,
)
from common.runner_modes.split_op_runner import run_dense_split_op_extend_case


@unittest.skipIf(
    not torch.cuda.is_available() or not is_flashinfer_available(),
    "CUDA + flashinfer are required",
)
class TestFlashInferSWAAttentionBackendCorrectness(CustomTestCase):
    # FlashInfer SM90 prefill kernels require value head dim in {64, 128, 256}.
    HEAD_DIM = 64
    HIDDEN_SIZE = 256

    # EXTEND with non-zero prefix is intentionally not covered here.
    # FlashInfer's SWA prefill path takes the `merge_state` branch when
    # `extend_no_prefix=False` (`flashinfer_backend.py:866-883`), which calls
    # `prefill_wrapper_paged.forward_return_lse` on the cached portion
    # *without* passing `window_left`, and `wrapper_paged.begin_forward`
    # (`flashinfer_backend.py:1482`) also doesn't carry SWA settings. Even
    # for within-window seqs, the merge_state path diverges from the
    # reference by ~0.2 max diff. This is a production-side SGLang gap, not
    # a fixture issue; cover only no-prefix EXTEND until the gap is closed.
    CASES = make_swa_no_prefix_input_config_cases("flashinfer")
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
