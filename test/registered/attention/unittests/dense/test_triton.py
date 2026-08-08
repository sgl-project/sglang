import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_hip
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
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
    run_dense_eagle_draft_extend_case,
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

register_cuda_ci(est_time=25, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=25, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=25, suite="stage-b-test-1-gpu-large-amd")


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestTritonDenseAttentionBackendCorrectness(CustomTestCase):
    CASES = make_dense_cases("triton")
    CUDA_GRAPH_CASES = (
        DenseAttentionCase(
            name="runner_cuda_graph_decode_page_boundary",
            backend="triton",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
        DenseAttentionCase(
            name="runner_cuda_graph_gqa_decode_page_boundary",
            backend="triton",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=2,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
        DenseAttentionCase(
            name="runner_cuda_graph_mqa_decode_bsz1",
            backend="triton",
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
                backend="triton",
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
                backend="triton",
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
                backend="triton",
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
                backend="triton",
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
                backend="triton",
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
                backend="triton",
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
                backend="triton",
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
                backend="triton",
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
                backend="triton",
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
                backend="triton",
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
                backend="triton",
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
                backend="triton",
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
    DRAFT_EXTEND_CASES = (
        (
            DenseAttentionCase(
                name="runner_eagle_draft_extend_ragged_accept",
                backend="triton",
                forward_mode=ForwardMode.DRAFT_EXTEND,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(2, 5),
                extend_lens=(1, 3),
            ),
            "eagle",
        ),
        (
            DenseAttentionCase(
                name="runner_frozen_kv_mtp_draft_extend_ragged_accept",
                backend="triton",
                forward_mode=ForwardMode.DRAFT_EXTEND,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(2, 5),
                extend_lens=(1, 3),
            ),
            "frozen_kv_mtp",
        ),
    )
    DRAFT_EXTEND_V2_CUDA_GRAPH_CASES = (
        DenseAttentionCase(
            name="runner_cuda_graph_eagle_draft_extend_v2_fixed_tokens",
            backend="triton",
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(4, 7),
            extend_lens=(3, 3),
        ),
    )
    EAGLE_DRAFT_EXTEND_V2_RUNNER_CASES = (
        DenseAttentionCase(
            name="runner_eagle_draft_extend_v2_cuda_graph_runner_fixed_tokens",
            backend="triton",
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
                name="runner_eagle_draft_decode_cuda_graph_chain",
                backend="triton",
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
                backend="triton",
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
            name="runner_frozen_kv_mtp_decode_cuda_graph",
            backend="triton",
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
                run_dense_attention_case(self, case)

    # Layout-robustness: re-run a representative extend + decode under
    # non-tidy `(req_to_token, out_cache_loc)` mappings. The fixture's
    # default contiguous layout uses
    # `_token_loc(req_idx, pos) = page_size + req_idx * max_ctx + pos`,
    # which is affine in `pos` — it hides any backend bug that assumes
    # `out_cache_loc` is monotonic within a request, or that a request's
    # pages occupy a contiguous physical range. Production allocators
    # routinely produce non-tidy `out_cache_loc` after fragmentation,
    # so these layouts catch a class of metadata-derivation bugs the
    # default layout doesn't exercise. The reference doesn't change —
    # it computes attention from projected Q/K/V directly without
    # reading the cache.
    LAYOUT_ROBUSTNESS_CASES = (
        DenseAttentionCase(
            name="layout_extend_two_request_ragged",
            backend="triton",
            forward_mode=ForwardMode.EXTEND,
            num_heads=12,
            num_kv_heads=12,
            page_size=16,
            prefix_lens=(8, 16),
            extend_lens=(8, 16),
        ),
        DenseAttentionCase(
            name="layout_decode_page_boundary",
            backend="triton",
            forward_mode=ForwardMode.DECODE,
            num_heads=12,
            num_kv_heads=12,
            page_size=16,
            prefix_lens=(15, 16, 17),
        ),
    )

    def test_layout_robustness_cases(self):
        for case in self.LAYOUT_ROBUSTNESS_CASES:
            # shuffled_pages is the default for all tests now, so it's
            # already covered by `test_projected_dense_attention_cases`.
            # The opt-in matrix here exercises the more aggressive
            # interleaved_pages + non_monotonic_extend layouts.
            for layout in (
                "interleaved_pages",
                "non_monotonic_extend",
            ):
                if layout == "non_monotonic_extend" and case.forward_mode.is_decode():
                    # decode has no extend tokens to scatter
                    continue
                with self.subTest(case=case.name, layout=layout):
                    run_dense_attention_case(self, case, loc_layout=layout)

    def test_runner_mode_cuda_graph_decode_cases(self):
        for case in self.CUDA_GRAPH_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_cuda_graph_decode_case(self, case)

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
                    run_dense_split_op_extend_case(
                        self,
                        case,
                        breakable=breakable,
                        static_num_tokens=static_num_tokens,
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
                )

    def test_runner_mode_eagle_draft_extend_cases(self):
        for case, spec_kind in self.DRAFT_EXTEND_CASES:
            with self.subTest(
                case=case.name,
                backend=case.backend,
                spec_kind=spec_kind,
            ):
                run_dense_eagle_draft_extend_case(
                    self,
                    case,
                    spec_kind=spec_kind,
                )

    def test_runner_mode_eagle_draft_extend_v2_cuda_graph_cases(self):
        # pad_ratio is expressed as the captured batch size relative to the
        # case's real batch size: 1.0x = no padding, 2.0x = 50% padded, etc.
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
                            cuda_graph_capture_batch_size=capture_bs,
                            pad_style=pad_style,
                        )

    def test_runner_mode_eagle_draft_extend_v2_cuda_graph_runner_cases(self):
        for case in self.EAGLE_DRAFT_EXTEND_V2_RUNNER_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_eagle_draft_extend_v2_cuda_graph_runner_case(self, case)

    def test_runner_mode_eagle_draft_cuda_graph_runner_cases(self):
        for case, topk, num_draft_tokens in self.EAGLE_DRAFT_RUNNER_CASES:
            with self.subTest(case=case.name, backend=case.backend, topk=topk):
                run_dense_eagle_draft_cuda_graph_runner_case(
                    self,
                    case,
                    topk=topk,
                    speculative_num_draft_tokens=num_draft_tokens,
                )

    def test_runner_mode_frozen_kv_mtp_cuda_graph_runner_cases(self):
        for case in self.FROZEN_KV_MTP_RUNNER_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_frozen_kv_mtp_cuda_graph_runner_case(self, case)


if __name__ == "__main__":
    unittest.main()
