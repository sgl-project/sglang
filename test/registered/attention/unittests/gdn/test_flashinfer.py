import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_flashinfer_available
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.gdn_attention import (
    GDNAttentionCase,
    make_gdn_cases,
    run_gdn_attention_case,
)
from sglang.test.kits.attention_unittest.runner_modes.cuda_graph_decode_runner import (
    run_gdn_cuda_graph_decode_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_target_verify_runner import (
    run_gdn_eagle_verify_case,
    run_gdn_eagle_verify_cuda_graph_case,
)
from sglang.test.kits.attention_unittest.runner_modes.split_op_runner import (
    run_gdn_split_op_extend_case,
)

register_cuda_ci(est_time=10, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=11, stage="base-b", runner_config="1-gpu-large")


@unittest.skipIf(
    not torch.cuda.is_available() or not is_flashinfer_available(),
    "CUDA + flashinfer are required",
)
class TestFlashInferGDNBackendCorrectness(CustomTestCase):
    # FlashInfer SM90 prefill kernels require value head dim in {64, 128, 256}.
    HEAD_K_DIM = 64
    HEAD_V_DIM = 64

    CASES = make_gdn_cases("flashinfer")
    CUDA_GRAPH_CASES = (
        GDNAttentionCase(
            name="runner_cuda_graph_gdn_decode_page_boundary",
            backend="flashinfer",
            forward_mode=ForwardMode.DECODE,
            num_k_heads=2,
            num_v_heads=2,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
    )
    SPLIT_OP_CASES = (
        (
            GDNAttentionCase(
                name="runner_split_op_gdn_extend_ragged_page_boundary",
                backend="flashinfer",
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
    EAGLE_VERIFY_CASES = (
        (
            GDNAttentionCase(
                name="runner_eagle_verify_gdn_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "eagle",
        ),
        (
            GDNAttentionCase(
                name="runner_eagle_verify_gdn_tree",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(5, 6),
                extend_lens=(3, 3),
            ),
            2,
            "eagle",
        ),
        (
            GDNAttentionCase(
                name="runner_frozen_kv_mtp_verify_gdn_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "frozen_kv_mtp",
        ),
        (
            GDNAttentionCase(
                name="runner_dflash_verify_gdn_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "dflash",
        ),
        (
            GDNAttentionCase(
                name="runner_ngram_verify_gdn_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "ngram",
        ),
    )
    EAGLE_VERIFY_CUDA_GRAPH_CASES = (
        (
            GDNAttentionCase(
                name="runner_cuda_graph_eagle_verify_gdn_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "eagle",
        ),
        (
            GDNAttentionCase(
                name="runner_cuda_graph_eagle_verify_gdn_tree",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(5, 6),
                extend_lens=(3, 3),
            ),
            2,
            "eagle",
        ),
        (
            GDNAttentionCase(
                name="runner_cuda_graph_frozen_kv_mtp_verify_gdn_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "frozen_kv_mtp",
        ),
        (
            GDNAttentionCase(
                name="runner_cuda_graph_dflash_verify_gdn_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "dflash",
        ),
        (
            GDNAttentionCase(
                name="runner_cuda_graph_ngram_verify_gdn_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "ngram",
        ),
    )

    def test_projected_gdn_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_gdn_attention_case(
                    self,
                    case,
                    head_k_dim=self.HEAD_K_DIM,
                    head_v_dim=self.HEAD_V_DIM,
                )

    # Layout-robustness. See dense/test_triton.py for the rationale.
    LAYOUT_ROBUSTNESS_CASES = (
        GDNAttentionCase(
            name="layout_gdn_extend_two_request",
            backend="flashinfer",
            forward_mode=ForwardMode.EXTEND,
            num_k_heads=4,
            num_v_heads=4,
            page_size=16,
            prefix_lens=(0, 0),
            extend_lens=(16, 16),
        ),
        GDNAttentionCase(
            name="layout_gdn_decode_page_boundary",
            backend="flashinfer",
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
                    run_gdn_attention_case(
                        self,
                        case,
                        head_k_dim=self.HEAD_K_DIM,
                        head_v_dim=self.HEAD_V_DIM,
                        loc_layout=layout,
                    )

    def test_runner_mode_cuda_graph_decode_cases(self):
        for case in self.CUDA_GRAPH_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_gdn_cuda_graph_decode_case(
                    self,
                    case,
                    head_k_dim=self.HEAD_K_DIM,
                    head_v_dim=self.HEAD_V_DIM,
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
                        head_k_dim=self.HEAD_K_DIM,
                        head_v_dim=self.HEAD_V_DIM,
                    )

    def test_runner_mode_eagle_verify_cases(self):
        for case, topk, spec_kind in self.EAGLE_VERIFY_CASES:
            with self.subTest(
                case=case.name,
                backend=case.backend,
                topk=topk,
                spec_kind=spec_kind,
            ):
                run_gdn_eagle_verify_case(
                    self,
                    case,
                    topk=topk,
                    spec_kind=spec_kind,
                    head_k_dim=self.HEAD_K_DIM,
                    head_v_dim=self.HEAD_V_DIM,
                )

    def test_runner_mode_eagle_verify_cuda_graph_cases(self):
        for case, topk, spec_kind in self.EAGLE_VERIFY_CUDA_GRAPH_CASES:
            with self.subTest(
                case=case.name,
                backend=case.backend,
                topk=topk,
                spec_kind=spec_kind,
            ):
                run_gdn_eagle_verify_cuda_graph_case(
                    self,
                    case,
                    topk=topk,
                    spec_kind=spec_kind,
                    head_k_dim=self.HEAD_K_DIM,
                    head_v_dim=self.HEAD_V_DIM,
                )


if __name__ == "__main__":
    unittest.main()
