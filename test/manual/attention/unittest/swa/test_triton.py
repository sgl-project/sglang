import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.attention_methods.dense_attention import (
    DenseAttentionCase,
    make_swa_no_prefix_input_config_cases,
    make_swa_prefix_input_config_cases,
    run_dense_attention_case,
)
from common.runner_modes.cuda_graph_decode_runner import (
    run_dense_cuda_graph_decode_case,
)
from common.runner_modes.speculative_target_verify_runner import (
    run_dense_spec_verify_case,
    run_dense_spec_verify_cuda_graph_case,
)
from common.runner_modes.split_op_runner import run_dense_split_op_extend_case


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
        # `seq_lens > sliding_window_size` so the metadata builder's
        # `window_kv_lens = min(seq_lens, sliding_window_size)` and
        # `window_kv_start_idx = seq_lens - window_kv_lens` actually clip.
        # A `sliding_window_size + 1` mutation in
        # `init_forward_metadata_replay_cuda_graph` (and its capture
        # twin) shifts every per-request window by one token; the
        # within-window case above cannot see that drift, but this case
        # does. The dense SWA reference also uses the matching
        # `min(seq_lens, window)` rule for decode-only triton/flashinfer
        # paths, so the assertion compares apples to apples.
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
        ),
        # `seq_lens (= prefix) > sliding_window_size` so the verify-path
        # SWA metadata builder's
        # `window_kv_lens = min(seq_lens, sliding_window_size)` clips.
        # An `sliding_window_size + 1` mutation in
        # `init_forward_metadata_replay_cuda_graph` for target_verify
        # changes the SWA prefix-only window length, which propagates
        # through the extend kernel's per-token mask. The within-window
        # case above never triggers the clip and so cannot detect M6.
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
        ),
    )

    def test_projected_swa_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_attention_case(self, case)

    def test_runner_mode_cuda_graph_decode_cases(self):
        for case in self.CUDA_GRAPH_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_cuda_graph_decode_case(self, case)

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
        for case, topk in self.SPEC_VERIFY_CASES:
            with self.subTest(case=case.name, backend=case.backend, topk=topk):
                run_dense_spec_verify_case(self, case, topk=topk, spec_kind="eagle")

    def test_runner_mode_spec_verify_cuda_graph_cases(self):
        for case, topk in self.SPEC_VERIFY_CUDA_GRAPH_CASES:
            with self.subTest(case=case.name, backend=case.backend, topk=topk):
                run_dense_spec_verify_cuda_graph_case(
                    self,
                    case,
                    topk=topk,
                    spec_kind="eagle",
                )


if __name__ == "__main__":
    unittest.main()
