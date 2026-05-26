import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.attention_methods.dense_attention import (
    DenseAttentionCase,
    make_dense_cases,
    run_dense_attention_case,
)
from common.runner_modes.cuda_graph_decode_runner import (
    run_dense_cuda_graph_decode_case,
)
from common.runner_modes.speculative_draft_extend_runner import (
    run_dense_draft_extend_v2_cuda_graph_case,
)
from common.runner_modes.speculative_target_verify_runner import (
    run_dense_spec_verify_case,
    run_dense_spec_verify_cuda_graph_case,
)
from common.runner_modes.split_op_runner import run_dense_split_op_extend_case


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestFA3DenseAttentionBackendCorrectness(CustomTestCase):
    # FlashAttention kernels are most stable in this harness with FA-friendly dims.
    HEAD_DIM = 64
    HIDDEN_SIZE = 256

    CASES = make_dense_cases("fa3")
    CUDA_GRAPH_CASES = (
        DenseAttentionCase(
            name="runner_cuda_graph_fa3_mha_decode_page_boundary",
            backend="fa3",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
    )
    DRAFT_EXTEND_V2_CUDA_GRAPH_CASES = (
        DenseAttentionCase(
            name="runner_cuda_graph_fa3_eagle_draft_extend_v2_fixed_tokens",
            backend="fa3",
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(4, 7),
            extend_lens=(3, 3),
        ),
    )
    # EAGLE chain verify (topk=1) — tree (topk=2) drifts ~0.16 vs the bf16
    # HF reference at the kernel level (not a CG mechanic) so it stays
    # deferred. See PLAN.md "Latest verification".
    SPEC_VERIFY_CHAIN_CASES = (
        DenseAttentionCase(
            name="runner_fa3_eagle_verify_chain",
            backend="fa3",
            forward_mode=ForwardMode.TARGET_VERIFY,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(4, 7),
            extend_lens=(3, 3),
        ),
    )
    SPEC_VERIFY_CHAIN_CUDA_GRAPH_CASES = (
        DenseAttentionCase(
            name="runner_cuda_graph_fa3_eagle_verify_chain",
            backend="fa3",
            forward_mode=ForwardMode.TARGET_VERIFY,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(4, 7),
            extend_lens=(3, 3),
        ),
    )
    SPLIT_OP_CASES = (
        (
            DenseAttentionCase(
                name="runner_split_op_mha_extend_ragged_page_boundary",
                backend="fa3",
                forward_mode=ForwardMode.EXTEND,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(0, 8, 16),
                extend_lens=(15, 8, 1),
            ),
            32,
        ),
        (
            DenseAttentionCase(
                name="runner_split_op_gqa_extend_cross_page_boundary",
                backend="fa3",
                forward_mode=ForwardMode.EXTEND,
                num_heads=4,
                num_kv_heads=2,
                page_size=16,
                prefix_lens=(15,),
                extend_lens=(2,),
            ),
            4,
        ),
    )

    def test_projected_dense_attention_cases(self):
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

    def test_runner_mode_eagle_draft_extend_v2_cuda_graph_cases(self):
        for case in self.DRAFT_EXTEND_V2_CUDA_GRAPH_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_draft_extend_v2_cuda_graph_case(
                    self,
                    case,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                )

    def test_runner_mode_spec_verify_cases(self):
        for case in self.SPEC_VERIFY_CHAIN_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_spec_verify_case(
                    self,
                    case,
                    topk=1,
                    spec_kind="eagle",
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                )

    def test_runner_mode_spec_verify_cuda_graph_cases(self):
        for case in self.SPEC_VERIFY_CHAIN_CUDA_GRAPH_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_spec_verify_cuda_graph_case(
                    self,
                    case,
                    topk=1,
                    spec_kind="eagle",
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                )


if __name__ == "__main__":
    unittest.main()
