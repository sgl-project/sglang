import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_sm90_supported
from sglang.srt.utils.hpc import has_hpc
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.dense_attention import (
    DenseAttentionCase,
    run_dense_attention_case,
)
from sglang.test.kits.attention_unittest.runner_modes.cuda_graph_decode_runner import (
    run_dense_cuda_graph_decode_case,
)

# HPC kernels require bfloat16
HPC_DTYPE = torch.bfloat16

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")


@unittest.skipIf(
    not torch.cuda.is_available() or not has_hpc() or not is_sm90_supported(),
    "CUDA + hpc-ops + SM90 are required",
)
class TestHpcDenseAttentionBackendCorrectness(CustomTestCase):
    # HPC attention backend requires head_dim=128
    HEAD_DIM = 128
    HIDDEN_SIZE = 512  # 4 heads * 128
    # Override default (64) to accommodate longer sequences with page_size=64
    MAX_CONTEXT_LEN = 512

    # Decode cases: GQA ratio 4 (4 Q heads / 1 KV head)
    DECODE_CASES = (
        DenseAttentionCase(
            name="hpc_gqa4_decode_page_boundary",
            backend="hpc",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=1,
            page_size=64,
            prefix_lens=(62, 63, 64),
        ),
        DenseAttentionCase(
            name="hpc_gqa4_decode_bsz1",
            backend="hpc",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=1,
            page_size=64,
            prefix_lens=(127,),
        ),
        DenseAttentionCase(
            name="hpc_gqa4_decode_multi_request",
            backend="hpc",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=1,
            page_size=64,
            prefix_lens=(10, 64, 128, 200),
        ),
    )

    # Decode cases: GQA ratio 8 (8 Q heads / 1 KV head)
    DECODE_GQA8_CASES = (
        DenseAttentionCase(
            name="hpc_gqa8_decode_page_boundary",
            backend="hpc",
            forward_mode=ForwardMode.DECODE,
            num_heads=8,
            num_kv_heads=1,
            page_size=64,
            prefix_lens=(62, 63, 64),
        ),
        DenseAttentionCase(
            name="hpc_gqa8_decode_bsz1",
            backend="hpc",
            forward_mode=ForwardMode.DECODE,
            num_heads=8,
            num_kv_heads=1,
            page_size=64,
            prefix_lens=(127,),
        ),
    )

    # Decode cases: GQA ratio 4 (8 Q heads / 2 KV heads)
    DECODE_GQA4_WIDE_CASES = (
        DenseAttentionCase(
            name="hpc_gqa4_wide_decode_page_boundary",
            backend="hpc",
            forward_mode=ForwardMode.DECODE,
            num_heads=8,
            num_kv_heads=2,
            page_size=64,
            prefix_lens=(62, 63, 64),
        ),
    )

    # Extend cases: GQA ratio 4
    EXTEND_CASES = (
        DenseAttentionCase(
            name="hpc_gqa4_extend_ragged",
            backend="hpc",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            num_kv_heads=1,
            page_size=64,
            prefix_lens=(0, 64),
            extend_lens=(32, 16),
        ),
        DenseAttentionCase(
            name="hpc_gqa4_extend_cross_page",
            backend="hpc",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            num_kv_heads=1,
            page_size=64,
            prefix_lens=(60, 128),
            extend_lens=(10, 20),
        ),
        DenseAttentionCase(
            name="hpc_gqa4_extend_bsz1",
            backend="hpc",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            num_kv_heads=1,
            page_size=64,
            prefix_lens=(100,),
            extend_lens=(50,),
        ),
    )

    # CUDA graph decode cases (GQA ratio 4 and 8)
    CUDA_GRAPH_DECODE_CASES = (
        DenseAttentionCase(
            name="runner_cuda_graph_hpc_gqa4_decode_page_boundary",
            backend="hpc",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=1,
            page_size=64,
            prefix_lens=(62, 63, 64),
        ),
        DenseAttentionCase(
            name="runner_cuda_graph_hpc_gqa8_decode_page_boundary",
            backend="hpc",
            forward_mode=ForwardMode.DECODE,
            num_heads=8,
            num_kv_heads=1,
            page_size=64,
            prefix_lens=(62, 63, 64),
        ),
        DenseAttentionCase(
            name="runner_cuda_graph_hpc_gqa4_wide_decode_bsz1",
            backend="hpc",
            forward_mode=ForwardMode.DECODE,
            num_heads=8,
            num_kv_heads=2,
            page_size=64,
            prefix_lens=(127,),
        ),
    )

    def test_decode_cases_gqa4(self):
        for case in self.DECODE_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_attention_case(
                    self,
                    case,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                    max_context_len=self.MAX_CONTEXT_LEN,
                    dtype=HPC_DTYPE,
                )

    def test_decode_cases_gqa8(self):
        for case in self.DECODE_GQA8_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_attention_case(
                    self,
                    case,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                    max_context_len=self.MAX_CONTEXT_LEN,
                    dtype=HPC_DTYPE,
                )

    def test_decode_cases_gqa4_wide(self):
        for case in self.DECODE_GQA4_WIDE_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_attention_case(
                    self,
                    case,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                    max_context_len=self.MAX_CONTEXT_LEN,
                    dtype=HPC_DTYPE,
                )

    def test_extend_cases(self):
        for case in self.EXTEND_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_attention_case(
                    self,
                    case,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                    max_context_len=self.MAX_CONTEXT_LEN,
                    dtype=HPC_DTYPE,
                )

    def test_runner_mode_cuda_graph_decode_cases(self):
        for case in self.CUDA_GRAPH_DECODE_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_cuda_graph_decode_case(
                    self,
                    case,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                    max_context_len=self.MAX_CONTEXT_LEN,
                    dtype=HPC_DTYPE,
                )


if __name__ == "__main__":
    unittest.main()
