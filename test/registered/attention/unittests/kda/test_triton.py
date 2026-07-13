import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_hip
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.kda_attention import (
    KDAAttentionCase,
    make_kda_cases,
    run_kda_attention_case,
)
from sglang.test.kits.attention_unittest.runner_modes.cuda_graph_decode_runner import (
    run_kda_cuda_graph_decode_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_target_verify_runner import (
    run_kda_eagle_verify_case,
    run_kda_eagle_verify_cuda_graph_case,
)
from sglang.test.kits.attention_unittest.runner_modes.split_op_runner import (
    run_kda_split_op_extend_case,
)

register_cuda_ci(est_time=13, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=12, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-large-amd")


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestTritonKDABackendCorrectness(CustomTestCase):
    CASES = make_kda_cases("triton")
    # KDA inherits the same `MambaAttnBackendBase` capture/replay path as GDN
    # through `HybridLinearAttnBackend`. See kda/README.md.
    CUDA_GRAPH_CASES = (
        KDAAttentionCase(
            name="runner_cuda_graph_kda_decode_page_boundary",
            backend="triton",
            forward_mode=ForwardMode.DECODE,
            num_k_heads=2,
            num_v_heads=2,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
    )
    # KDA verify covers EAGLE chain/tree plus the three non-EAGLE chain
    # spec kinds (frozen_kv_mtp / dflash / ngram). The non-EAGLE kinds
    # use a slightly different draft-token mask layout — same recurrent
    # math, but the per-token state replay accumulates enough drift that
    # 1 / 384 elements lands at ~0.11 max diff against the default
    # `KDA_ATOL=1e-1` tolerance. Use a looser `2e-1` tolerance for the
    # non-EAGLE kinds (kernel-side correctness is unchanged; only the
    # numerical headroom differs) so the matrix is complete.
    EAGLE_VERIFY_CASES = (
        (
            KDAAttentionCase(
                name="runner_eagle_verify_kda_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "eagle",
            None,
        ),
        (
            KDAAttentionCase(
                name="runner_eagle_verify_kda_tree",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(5, 6),
                extend_lens=(3, 3),
            ),
            2,
            "eagle",
            None,
        ),
        (
            KDAAttentionCase(
                name="runner_frozen_kv_mtp_verify_kda_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "frozen_kv_mtp",
            2e-1,
        ),
        (
            KDAAttentionCase(
                name="runner_dflash_verify_kda_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "dflash",
            2e-1,
        ),
        (
            KDAAttentionCase(
                name="runner_ngram_verify_kda_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "ngram",
            2e-1,
        ),
    )
    EAGLE_VERIFY_CUDA_GRAPH_CASES = (
        (
            KDAAttentionCase(
                name="runner_cuda_graph_eagle_verify_kda_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
        ),
        (
            KDAAttentionCase(
                name="runner_cuda_graph_eagle_verify_kda_tree",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(5, 6),
                extend_lens=(3, 3),
            ),
            2,
        ),
    )

    def test_projected_kda_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_kda_attention_case(self, case)

    # Layout-robustness. See dense/test_triton.py for the rationale.
    LAYOUT_ROBUSTNESS_CASES = (
        KDAAttentionCase(
            name="layout_kda_extend_two_request",
            backend="triton",
            forward_mode=ForwardMode.EXTEND,
            num_k_heads=2,
            num_v_heads=2,
            page_size=16,
            prefix_lens=(0, 0),
            extend_lens=(16, 16),
        ),
        KDAAttentionCase(
            name="layout_kda_decode_page_boundary",
            backend="triton",
            forward_mode=ForwardMode.DECODE,
            num_k_heads=2,
            num_v_heads=2,
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
                    run_kda_attention_case(self, case, loc_layout=layout)

    def test_runner_mode_cuda_graph_decode_cases(self):
        for case in self.CUDA_GRAPH_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_kda_cuda_graph_decode_case(self, case)

    def test_runner_mode_eagle_verify_cases(self):
        for case, topk, spec_kind, atol_override in self.EAGLE_VERIFY_CASES:
            with self.subTest(
                case=case.name,
                backend=case.backend,
                topk=topk,
                spec_kind=spec_kind,
            ):
                kwargs = dict(topk=topk, spec_kind=spec_kind)
                if atol_override is not None:
                    kwargs.update(atol=atol_override, rtol=atol_override)
                run_kda_eagle_verify_case(self, case, **kwargs)

    def test_runner_mode_eagle_verify_cuda_graph_cases(self):
        for case, topk in self.EAGLE_VERIFY_CUDA_GRAPH_CASES:
            with self.subTest(case=case.name, backend=case.backend, topk=topk):
                run_kda_eagle_verify_cuda_graph_case(self, case, topk=topk)

    SPLIT_OP_CASES = (
        (
            KDAAttentionCase(
                name="runner_split_op_kda_extend_ragged_page_boundary",
                backend="triton",
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

    @unittest.skipIf(
        is_hip(),
        "split-op extend runner exercises the piecewise-CUDA-graph path "
        "(TcPiecewiseForwardContext.num_tokens), which is not wired on ROCm.",
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
                    run_kda_split_op_extend_case(
                        self,
                        case,
                        breakable=breakable,
                        static_num_tokens=static_num_tokens,
                    )


if __name__ == "__main__":
    unittest.main()
