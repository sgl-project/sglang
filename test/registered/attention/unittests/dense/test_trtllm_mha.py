import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_flashinfer_available
from sglang.srt.utils.common import is_sm90_supported, is_sm120_supported
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.dense_attention import (
    DenseAttentionCase,
    run_dense_attention_case,
)
from sglang.test.kits.attention_unittest.runner_modes.cuda_graph_decode_runner import (
    run_dense_cuda_graph_decode_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_draft_runner import (
    run_dense_eagle_draft_cuda_graph_runner_case,
)

register_cuda_ci(est_time=8, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=16, stage="base-b", runner_config="1-gpu-large")


@unittest.skipIf(
    not torch.cuda.is_available()
    or not is_flashinfer_available()
    or not (is_sm90_supported() or is_sm120_supported()),
    "CUDA + FlashInfer TRT-LLM MHA decode support are required",
)
class TestTRTLLMMHADenseAttentionBackendCorrectness(CustomTestCase):
    HEAD_DIM = 64
    HIDDEN_SIZE = 256

    DECODE_CASES = (
        DenseAttentionCase(
            name="trtllm_mha_decode_page_boundary",
            backend="trtllm_mha",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
        DenseAttentionCase(
            name="trtllm_mha_gqa_decode_page_boundary",
            backend="trtllm_mha",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=2,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
        DenseAttentionCase(
            name="trtllm_mha_mqa_decode_bsz1",
            backend="trtllm_mha",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=1,
            page_size=16,
            prefix_lens=(7,),
        ),
        DenseAttentionCase(
            name="trtllm_mha_decode_page32_boundary",
            backend="trtllm_mha",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=4,
            page_size=32,
            prefix_lens=(31, 32),
        ),
    )

    # CG decode replay across MHA/GQA/MQA layouts and a page-32 case.
    # Previously documented as "currently mismatches on replay"; the
    # FlashInfer TRT-LLM Gen FMHA decode backend has since stabilized
    # the capture/replay metadata path and all four shapes match the
    # HF-style dense reference.
    CUDA_GRAPH_DECODE_CASES = (
        DenseAttentionCase(
            name="runner_cuda_graph_trtllm_mha_decode_page_boundary",
            backend="trtllm_mha",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
        DenseAttentionCase(
            name="runner_cuda_graph_trtllm_mha_gqa_decode_page_boundary",
            backend="trtllm_mha",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=2,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
        DenseAttentionCase(
            name="runner_cuda_graph_trtllm_mha_mqa_decode_bsz1",
            backend="trtllm_mha",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=1,
            page_size=16,
            prefix_lens=(7,),
        ),
        DenseAttentionCase(
            name="runner_cuda_graph_trtllm_mha_decode_page32_boundary",
            backend="trtllm_mha",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=4,
            page_size=32,
            prefix_lens=(31, 32),
        ),
    )

    # EAGLE draft CG runner — chain only (topk=1). trtllm_mha is constrained
    # to topk=1 via `trtllm_mha_backend.py:459,492` so tree-mode tests don't
    # apply. This test exercises the draft-decode CG capture/replay path
    # (`init_forward_metadata_capture_cuda_graph` line 320 and
    # `init_forward_metadata_replay_cuda_graph` line 460) — the same path
    # patched by PR #26521 (capture-time NaN fix) and PR #26655 (replay-time
    # slice rebind).
    EAGLE_DRAFT_RUNNER_CASES = (
        (
            DenseAttentionCase(
                name="runner_eagle_draft_decode_trtllm_mha_cuda_graph_chain",
                backend="trtllm_mha",
                forward_mode=ForwardMode.DECODE,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(4, 7),
            ),
            1,  # topk
            3,  # num_draft_tokens
        ),
    )

    def test_projected_dense_decode_cases(self):
        for case in self.DECODE_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_attention_case(
                    self,
                    case,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                )

    def test_runner_mode_cuda_graph_decode_cases(self):
        for case in self.CUDA_GRAPH_DECODE_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_cuda_graph_decode_case(
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


if __name__ == "__main__":
    unittest.main()
