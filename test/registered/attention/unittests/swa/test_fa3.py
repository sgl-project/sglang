import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import get_device_sm
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.dense_attention import (
    DenseAttentionCase,
    make_swa_no_prefix_input_config_cases,
    make_swa_prefix_input_config_cases,
    run_dense_attention_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_target_verify_runner import (
    run_dense_spec_verify_case,
    run_dense_spec_verify_cuda_graph_case,
)

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
@unittest.skipIf(
    get_device_sm() >= 100,
    "FA3 backend requires SM 80-90; skipping on Blackwell+ (B200/GB200/GB300)",
)
class TestFA3SWAAttentionBackendCorrectness(CustomTestCase):
    # FlashAttention kernels are most stable in this harness with FA-friendly dims.
    HEAD_DIM = 64
    HIDDEN_SIZE = 256

    CASES = make_swa_no_prefix_input_config_cases("fa3") + (
        make_swa_prefix_input_config_cases("fa3")
    )

    # topk > 1 tree verify + SWA drives `_init_sliding_window_attn_spec_metadata`
    # (page_size must be 1 on that path). Prefix lens straddle the window edge.
    SPEC_VERIFY_TREE_CASES = (
        (
            DenseAttentionCase(
                name="runner_eagle_verify_swa_tree",
                backend="fa3",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=1,
                prefix_lens=(3, 9),
                extend_lens=(3, 3),
                sliding_window_size=4,
            ),
            2,
            "eagle",
        ),
    )
    SPEC_VERIFY_TREE_CUDA_GRAPH_CASES = (
        (
            DenseAttentionCase(
                name="runner_cuda_graph_eagle_verify_swa_tree",
                backend="fa3",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=1,
                prefix_lens=(3, 9),
                extend_lens=(3, 3),
                sliding_window_size=4,
            ),
            2,
            "eagle",
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

    def test_runner_mode_spec_verify_cases(self):
        # Both seq-lens variants: mirrored (tight page-table sizing) and
        # gpu-only (static max_context_len bound, full-width concat table).
        for case, topk, spec_kind in self.SPEC_VERIFY_TREE_CASES:
            for force_gpu_only_seq_lens in (False, True):
                with self.subTest(
                    case=case.name,
                    backend=case.backend,
                    topk=topk,
                    spec_kind=spec_kind,
                    force_gpu_only_seq_lens=force_gpu_only_seq_lens,
                ):
                    run_dense_spec_verify_case(
                        self,
                        case,
                        topk=topk,
                        spec_kind=spec_kind,
                        head_dim=self.HEAD_DIM,
                        hidden_size=self.HIDDEN_SIZE,
                        force_gpu_only_seq_lens=force_gpu_only_seq_lens,
                    )

    def test_runner_mode_spec_verify_cuda_graph_cases(self):
        for case, topk, spec_kind in self.SPEC_VERIFY_TREE_CUDA_GRAPH_CASES:
            with self.subTest(
                case=case.name, backend=case.backend, topk=topk, spec_kind=spec_kind
            ):
                run_dense_spec_verify_cuda_graph_case(
                    self,
                    case,
                    topk=topk,
                    spec_kind=spec_kind,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                )


if __name__ == "__main__":
    unittest.main()
