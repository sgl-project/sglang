import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.attention_methods.dual_chunk_attention import (
    DualChunkAttentionCase,
    make_dual_chunk_cases,
    make_dual_chunk_sparse_cases,
    make_dual_chunk_sparse_threshold_gated_cases,
    run_dual_chunk_attention_case,
    run_dual_chunk_sparse_attention_case,
    run_dual_chunk_sparse_threshold_gated_case,
)
from common.runner_modes.cuda_graph_decode_runner import (
    run_dual_chunk_cuda_graph_decode_case,
)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestDualChunkFlashAttentionBackendCorrectness(CustomTestCase):
    CASES = make_dual_chunk_cases("dual_chunk_flash_attn")
    SPARSE_CASES = make_dual_chunk_sparse_cases("dual_chunk_flash_attn")
    SPARSE_THRESHOLD_GATED_CASES = make_dual_chunk_sparse_threshold_gated_cases(
        "dual_chunk_flash_attn"
    )
    # Replay prefix_lens must each be >= capture_prefix_len (= fill-value - 1).
    # Dual-chunk's `get_cuda_graph_seq_len_fill_value()` returns 1, so capture
    # uses prefix=0. We pick a 3-request batch with varied lengths to exercise
    # both the page-boundary and within-page slots.
    CUDA_GRAPH_DECODE_CASES = (
        DualChunkAttentionCase(
            name="runner_cuda_graph_dual_chunk_decode_page_boundary",
            backend="dual_chunk_flash_attn",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
    )

    def test_projected_dual_chunk_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dual_chunk_attention_case(self, case)

    def test_sparse_dual_chunk_attention_cases(self):
        for case in self.SPARSE_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dual_chunk_sparse_attention_case(self, case)

    def test_sparse_dual_chunk_threshold_gated_cases(self):
        for case in self.SPARSE_THRESHOLD_GATED_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dual_chunk_sparse_threshold_gated_case(self, case)

    def test_runner_mode_cuda_graph_decode_cases(self):
        for case in self.CUDA_GRAPH_DECODE_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dual_chunk_cuda_graph_decode_case(self, case)


if __name__ == "__main__":
    unittest.main()
