import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.dense_attention import (
    DenseAttentionCase,
    make_dense_cases,
    run_dense_attention_case,
)
from sglang.test.kits.attention_unittest.runner_modes.cuda_graph_decode_runner import (
    run_dense_cuda_graph_decode_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_draft_extend_runner import (
    run_dense_draft_extend_v2_cuda_graph_case,
    run_dense_eagle_draft_extend_v2_cuda_graph_runner_case,
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

register_cuda_ci(est_time=32, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=26, stage="base-b", runner_config="1-gpu-large")


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestFA4DenseAttentionBackendCorrectness(CustomTestCase):
    # FlashAttention kernels are most stable in this harness with FA-friendly dims.
    HEAD_DIM = 64
    HIDDEN_SIZE = 256

    CASES = make_dense_cases("fa4")
    CUDA_GRAPH_CASES = (
        DenseAttentionCase(
            name="runner_cuda_graph_fa4_mha_decode_page_boundary",
            backend="fa4",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
    )
    DRAFT_EXTEND_V2_CUDA_GRAPH_CASES = (
        DenseAttentionCase(
            name="runner_cuda_graph_fa4_eagle_draft_extend_v2_fixed_tokens",
            backend="fa4",
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
    #
    # The non-EAGLE spec kinds (frozen_kv_mtp, dflash, ngram) are also
    # chain-only on FA; they pass the same shape through
    # `_make_spec_verify_input` with a different `spec_kind` tag.
    SPEC_VERIFY_CHAIN_CASES = (
        (
            DenseAttentionCase(
                name="runner_fa4_eagle_verify_chain",
                backend="fa4",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            "eagle",
        ),
        (
            DenseAttentionCase(
                name="runner_fa4_frozen_kv_mtp_verify_chain",
                backend="fa4",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            "frozen_kv_mtp",
        ),
        (
            DenseAttentionCase(
                name="runner_fa4_dflash_verify_chain",
                backend="fa4",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            "dflash",
        ),
        (
            DenseAttentionCase(
                name="runner_fa4_ngram_verify_chain",
                backend="fa4",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            "ngram",
        ),
    )
    SPEC_VERIFY_CHAIN_CUDA_GRAPH_CASES = (
        (
            DenseAttentionCase(
                name="runner_cuda_graph_fa4_eagle_verify_chain",
                backend="fa4",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            "eagle",
        ),
        (
            DenseAttentionCase(
                name="runner_cuda_graph_fa4_frozen_kv_mtp_verify_chain",
                backend="fa4",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            "frozen_kv_mtp",
        ),
        (
            DenseAttentionCase(
                name="runner_cuda_graph_fa4_dflash_verify_chain",
                backend="fa4",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            "dflash",
        ),
        (
            DenseAttentionCase(
                name="runner_cuda_graph_fa4_ngram_verify_chain",
                backend="fa4",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            "ngram",
        ),
    )
    EAGLE_DRAFT_EXTEND_V2_RUNNER_CASES = (
        DenseAttentionCase(
            name="runner_fa4_eagle_draft_extend_v2_cuda_graph_runner_fixed_tokens",
            backend="fa4",
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(4, 7),
            extend_lens=(3, 3),
        ),
    )
    EAGLE_DRAFT_RUNNER_CASES = (
        (
            DenseAttentionCase(
                name="runner_fa4_eagle_draft_decode_cuda_graph_chain",
                backend="fa4",
                forward_mode=ForwardMode.DECODE,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
            ),
            1,
            3,
        ),
    )
    FROZEN_KV_MTP_RUNNER_CASES = (
        DenseAttentionCase(
            name="runner_fa4_frozen_kv_mtp_decode_cuda_graph",
            backend="fa4",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(4, 7),
        ),
    )
    SPLIT_OP_CASES = (
        (
            DenseAttentionCase(
                name="runner_split_op_mha_extend_ragged_page_boundary",
                backend="fa4",
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
                backend="fa4",
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

    # Layout-robustness. See dense/test_triton.py for full rationale and
    # dense/test_fa3.py for the FA-family non_monotonic_extend known
    # failure. FA4 inherits FA3's prefill metadata convention and shows
    # the same divergence on scattered extend-token slots.
    LAYOUT_ROBUSTNESS_CASES = (
        DenseAttentionCase(
            name="layout_extend_two_request_ragged",
            backend="fa4",
            forward_mode=ForwardMode.EXTEND,
            num_heads=12,
            num_kv_heads=12,
            page_size=16,
            prefix_lens=(8, 16),
            extend_lens=(8, 16),
        ),
        DenseAttentionCase(
            name="layout_decode_page_boundary",
            backend="fa4",
            forward_mode=ForwardMode.DECODE,
            num_heads=12,
            num_kv_heads=12,
            page_size=16,
            prefix_lens=(15, 16, 17),
        ),
    )
    LAYOUT_KNOWN_FAILURES = {
        ("layout_extend_two_request_ragged", "non_monotonic_extend"): (
            "FA4 inherits FA3's prefill metadata assumption that "
            "out_cache_loc is monotonic within an extend."
        ),
    }

    def test_layout_robustness_cases(self):
        for case in self.LAYOUT_ROBUSTNESS_CASES:
            # shuffled_pages is the default and already covered.
            for layout in (
                "interleaved_pages",
                "non_monotonic_extend",
            ):
                if layout == "non_monotonic_extend" and case.forward_mode.is_decode():
                    continue
                reason = self.LAYOUT_KNOWN_FAILURES.get((case.name, layout))
                if reason is not None:
                    print(
                        f"[layout-known-failure] {case.name} x {layout}: {reason}",
                        flush=True,
                    )
                    continue
                with self.subTest(case=case.name, layout=layout):
                    run_dense_attention_case(
                        self,
                        case,
                        head_dim=self.HEAD_DIM,
                        hidden_size=self.HIDDEN_SIZE,
                        loc_layout=layout,
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
            for pad_style in ("small_real", "prod_fill"):
                for capture_bs in (
                    case.batch_size,
                    case.batch_size * 2,
                    case.batch_size * 4,
                ):
                    with self.subTest(
                        case=case.name,
                        backend=case.backend,
                        pad_style=pad_style,
                        capture_bs=capture_bs,
                    ):
                        run_dense_draft_extend_v2_cuda_graph_case(
                            self,
                            case,
                            head_dim=self.HEAD_DIM,
                            hidden_size=self.HIDDEN_SIZE,
                            cuda_graph_capture_batch_size=capture_bs,
                            pad_style=pad_style,
                        )

    def test_runner_mode_eagle_draft_extend_v2_cuda_graph_runner_cases(self):
        for case in self.EAGLE_DRAFT_EXTEND_V2_RUNNER_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_eagle_draft_extend_v2_cuda_graph_runner_case(
                    self,
                    case,
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

    def test_runner_mode_spec_verify_cases(self):
        for case, spec_kind in self.SPEC_VERIFY_CHAIN_CASES:
            with self.subTest(
                case=case.name, backend=case.backend, spec_kind=spec_kind
            ):
                run_dense_spec_verify_case(
                    self,
                    case,
                    topk=1,
                    spec_kind=spec_kind,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                )

    def test_runner_mode_spec_verify_cuda_graph_cases(self):
        for case, spec_kind in self.SPEC_VERIFY_CHAIN_CUDA_GRAPH_CASES:
            with self.subTest(
                case=case.name, backend=case.backend, spec_kind=spec_kind
            ):
                run_dense_spec_verify_cuda_graph_case(
                    self,
                    case,
                    topk=1,
                    spec_kind=spec_kind,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                )


if __name__ == "__main__":
    unittest.main()
