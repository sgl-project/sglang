import unittest

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.dense_attention import (
    DenseAttentionCase,
    make_swa_no_prefix_input_config_cases,
    make_swa_prefix_input_config_cases,
    run_dense_attention_case,
)
from sglang.test.kits.attention_unittest.runner_modes.cuda_graph_decode_runner import (
    run_dense_cuda_graph_decode_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_target_verify_runner import (
    run_dense_spec_verify_case,
    run_dense_spec_verify_cuda_graph_case,
)
from sglang.test.kits.attention_unittest.runner_modes.split_op_runner import (
    run_dense_split_op_extend_case,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=12, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=14, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-large-amd")


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestTritonSWAAttentionBackendCorrectness(CustomTestCase):
    CASES = make_swa_no_prefix_input_config_cases(
        "triton"
    ) + make_swa_prefix_input_config_cases("triton")
    CUDA_GRAPH_CASES = (
        DenseAttentionCase(
            name="runner_cuda_graph_swa_decode_within_window",
            backend="triton",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(1, 2, 3),
            sliding_window_size=4,
        ),
        # Above-window decode exercises the `min(seq_lens, window)`
        # clipping in the replay metadata builder.
        DenseAttentionCase(
            name="runner_cuda_graph_swa_decode_above_window",
            backend="triton",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(7, 8, 9),
            sliding_window_size=4,
        ),
    )
    SPLIT_OP_CASES = (
        (
            DenseAttentionCase(
                name="runner_split_op_swa_extend_no_prefix_window_edges",
                backend="triton",
                forward_mode=ForwardMode.EXTEND,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(0, 0, 0),
                extend_lens=(3, 4, 5),
                sliding_window_size=4,
            ),
            16,
        ),
        (
            DenseAttentionCase(
                name="runner_split_op_swa_extend_prefix_within_window",
                backend="triton",
                forward_mode=ForwardMode.EXTEND,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(1, 2, 3),
                extend_lens=(1, 1, 1),
                sliding_window_size=4,
            ),
            4,
        ),
    )
    SPEC_VERIFY_CASES = (
        (
            DenseAttentionCase(
                name="runner_eagle_verify_swa_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(3, 5),
                extend_lens=(3, 3),
                sliding_window_size=4,
            ),
            1,
            "eagle",
        ),
        (
            DenseAttentionCase(
                name="runner_eagle_verify_swa_tree",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(3, 5),
                extend_lens=(3, 3),
                sliding_window_size=4,
            ),
            2,
            "eagle",
        ),
        # Non-EAGLE chain spec kinds. The verify-path math under a
        # sliding window is identical across kinds; only the draft
        # tag in `_make_spec_verify_input` differs.
        (
            DenseAttentionCase(
                name="runner_frozen_kv_mtp_verify_swa_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(3, 5),
                extend_lens=(3, 3),
                sliding_window_size=4,
            ),
            1,
            "frozen_kv_mtp",
        ),
        (
            DenseAttentionCase(
                name="runner_dflash_verify_swa_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(3, 5),
                extend_lens=(3, 3),
                sliding_window_size=4,
            ),
            1,
            "dflash",
        ),
        (
            DenseAttentionCase(
                name="runner_ngram_verify_swa_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(3, 5),
                extend_lens=(3, 3),
                sliding_window_size=4,
            ),
            1,
            "ngram",
        ),
    )
    SPEC_VERIFY_CUDA_GRAPH_CASES = (
        (
            DenseAttentionCase(
                name="runner_cuda_graph_eagle_verify_swa_tree",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(3, 5),
                extend_lens=(3, 3),
                sliding_window_size=4,
            ),
            2,
            "eagle",
        ),
        # Above-window verify exercises the `min(seq_lens, window)`
        # clipping in the verify-path replay metadata builder.
        (
            DenseAttentionCase(
                name="runner_cuda_graph_eagle_verify_swa_above_window",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(6, 8),
                extend_lens=(3, 3),
                sliding_window_size=4,
            ),
            1,
            "eagle",
        ),
        (
            DenseAttentionCase(
                name="runner_cuda_graph_frozen_kv_mtp_verify_swa_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(3, 5),
                extend_lens=(3, 3),
                sliding_window_size=4,
            ),
            1,
            "frozen_kv_mtp",
        ),
        (
            DenseAttentionCase(
                name="runner_cuda_graph_dflash_verify_swa_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(3, 5),
                extend_lens=(3, 3),
                sliding_window_size=4,
            ),
            1,
            "dflash",
        ),
        (
            DenseAttentionCase(
                name="runner_cuda_graph_ngram_verify_swa_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(3, 5),
                extend_lens=(3, 3),
                sliding_window_size=4,
            ),
            1,
            "ngram",
        ),
    )

    def test_projected_swa_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_attention_case(self, case)

    # Every other SWA case here keeps the extend length under BLOCK_M (64 or
    # 128 on the architectures we build for), so `cur_block_m` is always 0 and
    # the extend-KV loop always starts at tile 0. The kernel bounds that loop
    # below by the sliding-window floor, and that bound is only reachable once
    # a query tile starts past the window. These lengths are therefore load
    # bearing: they must stay above BLOCK_M with a window well below it, or
    # the bound goes unexercised and a wrong or missing one stays green.
    MULTI_TILE_SWA_CASES = (
        DenseAttentionCase(
            name="swa_extend_multi_tile_no_prefix",
            backend="triton",
            forward_mode=ForwardMode.EXTEND,
            num_heads=8,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(0, 0),
            extend_lens=(384, 321),
            sliding_window_size=24,
        ),
        # Non-empty prefix so the prefix sweep runs too, and a window that is
        # not a multiple of any BLOCK_N we use, which is where rounding the
        # floor down to a tile boundary would show up.
        DenseAttentionCase(
            name="swa_extend_multi_tile_with_prefix",
            backend="triton",
            forward_mode=ForwardMode.EXTEND,
            num_heads=8,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(48, 130),
            extend_lens=(384, 300),
            sliding_window_size=72,
        ),
        # Window exactly on a tile boundary: the floor lands on a tile start,
        # so an off-by-one in the bound drops a tile that still contributes.
        DenseAttentionCase(
            name="swa_extend_multi_tile_window_tile_aligned",
            backend="triton",
            forward_mode=ForwardMode.EXTEND,
            num_heads=8,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(0,),
            extend_lens=(448,),
            sliding_window_size=128,
        ),
    )

    def test_multi_tile_swa_extend_cases(self):
        for case in self.MULTI_TILE_SWA_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                # The kit defaults to 64, which these lengths exceed.
                run_dense_attention_case(self, case, max_context_len=512)

    # Layout-robustness. See dense/test_triton.py for full rationale.
    # The default `shuffled_pages` layout is already exercised by
    # test_projected_swa_attention_cases; this method opts into the
    # more aggressive interleaved_pages + non_monotonic_extend on a
    # representative SWA extend + decode case.
    LAYOUT_ROBUSTNESS_CASES = (
        DenseAttentionCase(
            name="layout_swa_extend_within_window",
            backend="triton",
            forward_mode=ForwardMode.EXTEND,
            num_heads=8,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(8, 16),
            extend_lens=(8, 16),
            sliding_window_size=12,
        ),
        DenseAttentionCase(
            name="layout_swa_decode_page_boundary",
            backend="triton",
            forward_mode=ForwardMode.DECODE,
            num_heads=8,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(15, 16, 17),
            sliding_window_size=12,
        ),
    )

    def test_layout_robustness_cases(self):
        for case in self.LAYOUT_ROBUSTNESS_CASES:
            for layout in ("interleaved_pages", "non_monotonic_extend"):
                if layout == "non_monotonic_extend" and case.forward_mode.is_decode():
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
                run_dense_spec_verify_case(self, case, topk=topk, spec_kind=spec_kind)

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


if __name__ == "__main__":
    unittest.main()
