import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.mla_attention import (
    MLAAttentionCase,
    make_mla_cases,
    run_mla_attention_case,
)
from sglang.test.kits.attention_unittest.runner_modes.cuda_graph_decode_runner import (
    run_mla_cuda_graph_decode_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_draft_extend_runner import (
    run_mla_draft_extend_v2_cuda_graph_case,
    run_mla_eagle_draft_extend_v2_cuda_graph_runner_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_draft_runner import (
    run_mla_eagle_draft_cuda_graph_runner_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_target_verify_runner import (
    run_mla_eagle_verify_case,
    run_mla_eagle_verify_cuda_graph_case,
)
from sglang.test.kits.attention_unittest.runner_modes.split_op_runner import (
    run_mla_split_op_extend_case,
)

register_cuda_ci(est_time=15, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=24, stage="base-b", runner_config="1-gpu-large")


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestTritonMLAAttentionBackendCorrectness(CustomTestCase):
    CASES = make_mla_cases("triton")
    CUDA_GRAPH_CASES = (
        MLAAttentionCase(
            name="runner_cuda_graph_decode_page_boundary",
            backend="triton",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
    )
    SPLIT_OP_CASES = (
        (
            MLAAttentionCase(
                name="runner_split_op_mla_extend_ragged_page_boundary",
                backend="triton",
                forward_mode=ForwardMode.EXTEND,
                num_heads=4,
                page_size=16,
                prefix_lens=(0, 8, 16),
                extend_lens=(15, 8, 1),
            ),
            32,
        ),
    )
    # Spec verify covers EAGLE chain + tree plus the non-EAGLE chain
    # spec kinds (frozen_kv_mtp, dflash, ngram). FlashInfer MLA and
    # FlashMLA only support EAGLE — their forward_extend reads
    # EAGLE-specific spec_info attrs and trips a CUDA illegal-memory
    # access on the other kinds — so this matrix is Triton-only.
    EAGLE_VERIFY_CASES = (
        (
            MLAAttentionCase(
                name="runner_eagle_verify_mla_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "eagle",
        ),
        (
            MLAAttentionCase(
                name="runner_eagle_verify_mla_tree",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            2,
            "eagle",
        ),
        (
            MLAAttentionCase(
                name="runner_frozen_kv_mtp_verify_mla_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "frozen_kv_mtp",
        ),
        (
            MLAAttentionCase(
                name="runner_dflash_verify_mla_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "dflash",
        ),
        (
            MLAAttentionCase(
                name="runner_ngram_verify_mla_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
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
            MLAAttentionCase(
                name="runner_cuda_graph_eagle_verify_mla_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "eagle",
        ),
        (
            MLAAttentionCase(
                name="runner_cuda_graph_eagle_verify_mla_tree",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            2,
            "eagle",
        ),
        (
            MLAAttentionCase(
                name="runner_cuda_graph_frozen_kv_mtp_verify_mla_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "frozen_kv_mtp",
        ),
        (
            MLAAttentionCase(
                name="runner_cuda_graph_dflash_verify_mla_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "dflash",
        ),
        (
            MLAAttentionCase(
                name="runner_cuda_graph_ngram_verify_mla_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "ngram",
        ),
    )
    DRAFT_EXTEND_V2_CUDA_GRAPH_CASES = (
        MLAAttentionCase(
            name="runner_cuda_graph_eagle_draft_extend_v2_mla_fixed_tokens",
            backend="triton",
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            num_heads=4,
            page_size=16,
            prefix_lens=(4, 7),
            extend_lens=(3, 3),
        ),
    )
    EAGLE_DRAFT_EXTEND_V2_RUNNER_CASES = (
        MLAAttentionCase(
            name="runner_eagle_draft_extend_v2_mla_cuda_graph_runner_fixed_tokens",
            backend="triton",
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            num_heads=4,
            page_size=16,
            prefix_lens=(4, 7),
            extend_lens=(3, 3),
        ),
    )
    EAGLE_DRAFT_RUNNER_CASES = (
        (
            MLAAttentionCase(
                name="runner_eagle_draft_decode_mla_cuda_graph_chain",
                backend="triton",
                forward_mode=ForwardMode.DECODE,
                num_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
            ),
            1,
            3,
        ),
        (
            MLAAttentionCase(
                name="runner_eagle_draft_decode_mla_cuda_graph_tree",
                backend="triton",
                forward_mode=ForwardMode.DECODE,
                num_heads=4,
                page_size=1,
                prefix_lens=(4, 7),
            ),
            2,
            4,
        ),
    )

    def test_tiny_deepseek_mla_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_mla_attention_case(self, case)

    # Layout-robustness. See dense/test_triton.py for the full
    # rationale. shuffled_pages is the default for all tests via
    # build_mla_attention_fixture; this method opts into the more
    # aggressive interleaved_pages + non_monotonic_extend layouts on a
    # representative MLA extend + decode case. MLA Triton handles all
    # non-tidy layouts cleanly.
    LAYOUT_ROBUSTNESS_CASES = (
        MLAAttentionCase(
            name="layout_mla_extend_prefix_exact_page",
            backend="triton",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=16,
            prefix_lens=(16,),
            extend_lens=(2,),
        ),
        MLAAttentionCase(
            name="layout_mla_decode_page_boundary",
            backend="triton",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
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
                    run_mla_attention_case(self, case, loc_layout=layout)

    def test_runner_mode_cuda_graph_decode_cases(self):
        for case in self.CUDA_GRAPH_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_mla_cuda_graph_decode_case(self, case)

    def test_runner_mode_split_op_extend_cases(self):
        for case, static_num_tokens in self.SPLIT_OP_CASES:
            for breakable in (False, True):
                runner = "bcg" if breakable else "pcg"
                with self.subTest(
                    case=case.name,
                    backend=case.backend,
                    runner=runner,
                ):
                    run_mla_split_op_extend_case(
                        self,
                        case,
                        breakable=breakable,
                        static_num_tokens=static_num_tokens,
                    )

    def test_runner_mode_eagle_verify_cases(self):
        for case, topk, spec_kind in self.EAGLE_VERIFY_CASES:
            with self.subTest(
                case=case.name,
                backend=case.backend,
                topk=topk,
                spec_kind=spec_kind,
            ):
                run_mla_eagle_verify_case(self, case, topk=topk, spec_kind=spec_kind)

    def test_runner_mode_eagle_verify_cuda_graph_cases(self):
        for case, topk, spec_kind in self.EAGLE_VERIFY_CUDA_GRAPH_CASES:
            with self.subTest(
                case=case.name,
                backend=case.backend,
                topk=topk,
                spec_kind=spec_kind,
            ):
                run_mla_eagle_verify_cuda_graph_case(
                    self, case, topk=topk, spec_kind=spec_kind
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
                        run_mla_draft_extend_v2_cuda_graph_case(
                            self,
                            case,
                            cuda_graph_capture_batch_size=capture_bs,
                            pad_style=pad_style,
                        )

    def test_runner_mode_eagle_draft_extend_v2_cuda_graph_runner_cases(self):
        for case in self.EAGLE_DRAFT_EXTEND_V2_RUNNER_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_mla_eagle_draft_extend_v2_cuda_graph_runner_case(self, case)

    def test_runner_mode_eagle_draft_cuda_graph_runner_cases(self):
        for case, topk, num_draft_tokens in self.EAGLE_DRAFT_RUNNER_CASES:
            with self.subTest(case=case.name, backend=case.backend, topk=topk):
                run_mla_eagle_draft_cuda_graph_runner_case(
                    self,
                    case,
                    topk=topk,
                    speculative_num_draft_tokens=num_draft_tokens,
                )


if __name__ == "__main__":
    unittest.main()
