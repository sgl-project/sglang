import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_flashinfer_available
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.dense_attention import (
    DENSE_ATOL,
    DENSE_RTOL,
    DenseAttentionCase,
    build_dense_attention_fixture,
    expected_dense_fixture_output,
    make_dense_cases,
    replace_backend,
    run_dense_attention_case,
    run_dense_fixture_eager,
)
from sglang.test.kits.attention_unittest.runner_modes.cuda_graph_decode_runner import (
    run_dense_cuda_graph_decode_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_draft_runner import (
    run_dense_eagle_draft_cuda_graph_runner_case,
    run_dense_frozen_kv_mtp_cuda_graph_runner_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_target_verify_runner import (
    run_dense_spec_verify_case,
    run_dense_spec_verify_cuda_graph_case,
)
from sglang.test.kits.attention_unittest.runner_modes.split_op_runner import (
    run_dense_split_op_extend_case,
)

register_cuda_ci(est_time=25, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=25, stage="base-b", runner_config="1-gpu-large")


@unittest.skipIf(
    not torch.cuda.is_available() or not is_flashinfer_available(),
    "CUDA + flashinfer are required",
)
class TestFlashInferDenseAttentionBackendCorrectness(CustomTestCase):
    # FlashInfer SM90 prefill kernels require value head dim in {64, 128, 256}.
    HEAD_DIM = 64
    HIDDEN_SIZE = 256

    CASES = make_dense_cases("flashinfer")
    CUDA_GRAPH_CASES = (
        DenseAttentionCase(
            name="runner_cuda_graph_decode_page_boundary",
            backend="flashinfer",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
        DenseAttentionCase(
            name="runner_cuda_graph_gqa_decode_page_boundary",
            backend="flashinfer",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=2,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
        DenseAttentionCase(
            name="runner_cuda_graph_mqa_decode_bsz1",
            backend="flashinfer",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=1,
            page_size=16,
            prefix_lens=(7,),
        ),
    )
    SPLIT_OP_CASES = (
        (
            DenseAttentionCase(
                name="runner_split_op_mha_extend_ragged_page_boundary",
                backend="flashinfer",
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
                backend="flashinfer",
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
    SPEC_VERIFY_CASES = (
        (
            DenseAttentionCase(
                name="runner_eagle_verify_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "eagle",
        ),
        (
            DenseAttentionCase(
                name="runner_eagle_verify_tree",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(5, 6),
                extend_lens=(3, 3),
            ),
            2,
            "eagle",
        ),
        (
            DenseAttentionCase(
                name="runner_frozen_kv_mtp_verify_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "frozen_kv_mtp",
        ),
        (
            DenseAttentionCase(
                name="runner_dflash_verify_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "dflash",
        ),
        (
            DenseAttentionCase(
                name="runner_ngram_verify_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "ngram",
        ),
    )
    SPEC_VERIFY_CUDA_GRAPH_CASES = (
        (
            DenseAttentionCase(
                name="runner_cuda_graph_eagle_verify_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "eagle",
        ),
        (
            DenseAttentionCase(
                name="runner_cuda_graph_eagle_verify_tree",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            2,
            "eagle",
        ),
        (
            DenseAttentionCase(
                name="runner_cuda_graph_frozen_kv_mtp_verify_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "frozen_kv_mtp",
        ),
        (
            DenseAttentionCase(
                name="runner_cuda_graph_dflash_verify_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "dflash",
        ),
        (
            DenseAttentionCase(
                name="runner_cuda_graph_ngram_verify_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "ngram",
        ),
    )
    EAGLE_DRAFT_RUNNER_CASES = (
        (
            DenseAttentionCase(
                name="runner_eagle_draft_decode_cuda_graph_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.DECODE,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
            ),
            1,
            3,
        ),
        (
            DenseAttentionCase(
                name="runner_eagle_draft_decode_cuda_graph_tree",
                backend="flashinfer",
                forward_mode=ForwardMode.DECODE,
                num_heads=4,
                num_kv_heads=4,
                page_size=1,
                prefix_lens=(4, 7),
            ),
            2,
            4,
        ),
    )
    FROZEN_KV_MTP_RUNNER_CASES = (
        DenseAttentionCase(
            name="runner_frozen_kv_mtp_decode_cuda_graph_chain",
            backend="flashinfer",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(4, 7),
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

    # Layout-robustness: see dense/test_triton.py for full rationale.
    # Re-runs a representative extend + decode under non-tidy
    # (req_to_token, out_cache_loc) mappings to catch backend bugs in
    # page-table derivation that the default contiguous layout hides.
    LAYOUT_ROBUSTNESS_CASES = (
        DenseAttentionCase(
            name="layout_extend_two_request_ragged",
            backend="flashinfer",
            forward_mode=ForwardMode.EXTEND,
            num_heads=12,
            num_kv_heads=12,
            page_size=16,
            prefix_lens=(8, 16),
            extend_lens=(8, 16),
        ),
        DenseAttentionCase(
            name="layout_decode_page_boundary",
            backend="flashinfer",
            forward_mode=ForwardMode.DECODE,
            num_heads=12,
            num_kv_heads=12,
            page_size=16,
            prefix_lens=(15, 16, 17),
        ),
    )

    # Regression cases for #33915: FlashInfer treats logits_soft_cap (and
    # window_left on the ragged wrapper) as plan-time state that selects the
    # compiled kernel module, but the backend passed them only to the
    # deprecated per-call forward(), which silently ignores them — so capped
    # models (gemma-2, grok) ran uncapped and SWA layers ran full attention
    # within a ragged extend chunk. logit_cap=0.5 is far below the typical
    # logit magnitude of these fixtures so capping visibly changes the output.
    LOGIT_CAP_CASES = (
        DenseAttentionCase(
            name="logit_cap_extend_ragged_no_prefix",
            backend="flashinfer",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(0, 0, 0),
            extend_lens=(15, 8, 1),
            logit_cap=0.1,
        ),
        DenseAttentionCase(
            name="logit_cap_extend_ragged_paged_merge",
            backend="flashinfer",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(8, 16),
            extend_lens=(8, 15),
            logit_cap=0.1,
        ),
        DenseAttentionCase(
            name="logit_cap_decode",
            backend="flashinfer",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(14, 15, 16),
            logit_cap=0.1,
        ),
    )
    SWA_RAGGED_WINDOW_CASES = (
        DenseAttentionCase(
            name="swa_window_extend_ragged_no_prefix_beyond_window",
            backend="flashinfer",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(0, 0),
            extend_lens=(7, 9),
            sliding_window_size=4,
        ),
        DenseAttentionCase(
            name="swa_window_extend_ragged_merge_beyond_window",
            backend="flashinfer",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(6, 5),
            extend_lens=(6, 7),
            sliding_window_size=4,
        ),
    )

    def _run_case_with_model_wired_backend(self, case: DenseAttentionCase):
        fixture = build_dense_attention_fixture(
            self,
            case,
            head_dim=self.HEAD_DIM,
            hidden_size=self.HIDDEN_SIZE,
        )
        # The kit's MockModelRunner carries an empty nn.Module as the model;
        # the FlashInfer backend reads the plan-time logit cap off the model's
        # RadixAttention layers at construction, so rebuild the backend with
        # the real attention module wired into the runner.
        fixture.runner.model = fixture.actual_module
        replace_backend(fixture, ATTENTION_BACKENDS["flashinfer"](fixture.runner))
        actual = run_dense_fixture_eager(fixture)
        expected = expected_dense_fixture_output(fixture)
        torch.testing.assert_close(actual, expected, atol=DENSE_ATOL, rtol=DENSE_RTOL)

    def test_logit_cap_cases(self):
        """Bug regression (#33915): plan() never requested logits_soft_cap, so
        the kernels ran uncapped and the per-call forward() cap was dropped."""
        for case in self.LOGIT_CAP_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                self._run_case_with_model_wired_backend(case)

    def test_swa_ragged_window_cases(self):
        """Bug regression (#33915): the ragged prefill wrapper was planned
        without window_left, so SWA layers attended past the window whenever an
        extend chunk was longer than the window."""
        for case in self.SWA_RAGGED_WINDOW_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                self._run_case_with_model_wired_backend(case)

    def test_layout_robustness_cases(self):
        for case in self.LAYOUT_ROBUSTNESS_CASES:
            # shuffled_pages is the default and already covered.
            for layout in (
                "interleaved_pages",
                "non_monotonic_extend",
            ):
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

    def test_runner_mode_spec_verify_cases(self):
        for case, topk, spec_kind in self.SPEC_VERIFY_CASES:
            with self.subTest(
                case=case.name,
                backend=case.backend,
                topk=topk,
                spec_kind=spec_kind,
            ):
                run_dense_spec_verify_case(
                    self,
                    case,
                    topk=topk,
                    spec_kind=spec_kind,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                )

    def test_runner_mode_spec_verify_cuda_graph_cases(self):
        for case, topk, spec_kind in self.SPEC_VERIFY_CUDA_GRAPH_CASES:
            with self.subTest(
                case=case.name,
                backend=case.backend,
                topk=topk,
                spec_kind=spec_kind,
            ):
                run_dense_spec_verify_cuda_graph_case(
                    self,
                    case,
                    topk=topk,
                    spec_kind=spec_kind,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                )

    def test_runner_mode_eagle_draft_cuda_graph_runner_cases(self):
        for case, topk, num_draft_tokens in self.EAGLE_DRAFT_RUNNER_CASES:
            with self.subTest(case=case.name, backend=case.backend, topk=topk):
                run_dense_eagle_draft_cuda_graph_runner_case(
                    self,
                    case,
                    topk=topk,
                    speculative_num_draft_tokens=num_draft_tokens,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                )

    def test_runner_mode_frozen_kv_mtp_cuda_graph_runner_cases(self):
        for case in self.FROZEN_KV_MTP_RUNNER_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_frozen_kv_mtp_cuda_graph_runner_case(
                    self,
                    case,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                )


if __name__ == "__main__":
    unittest.main()
