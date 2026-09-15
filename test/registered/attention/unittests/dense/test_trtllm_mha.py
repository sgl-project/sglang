import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_flashinfer_available
from sglang.srt.utils.common import (
    is_sm90_supported,
    is_sm100_supported,
    is_sm120_supported,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.dense_attention import (
    DENSE_ATOL,
    DENSE_RTOL,
    DenseAttentionCase,
    _make_forward_batch,
    build_dense_attention_fixture,
    expected_dense_fixture_output,
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
from sglang.test.kits.attention_unittest.runner_modes.split_op_runner import (
    run_dense_split_op_extend_case,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=17, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=16, stage="base-b", runner_config="1-gpu-large")


@unittest.skipIf(
    not torch.cuda.is_available()
    or not is_flashinfer_available()
    or not (is_sm90_supported() or is_sm100_supported() or is_sm120_supported()),
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
    SPLIT_OP_CASES = (
        (
            DenseAttentionCase(
                name="runner_split_op_trtllm_mha_extend_ragged",
                backend="trtllm_mha",
                forward_mode=ForwardMode.EXTEND,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(0, 8, 16),
                extend_lens=(15, 8, 1),
            ),
            32,
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

    # Frozen-KV MTP draft CG runner (chain, topk=1).
    FROZEN_KV_MTP_RUNNER_CASES = (
        DenseAttentionCase(
            name="runner_frozen_kv_mtp_decode_trtllm_mha_cuda_graph",
            backend="trtllm_mha",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(4, 7),
        ),
    )

    def test_projected_dense_decode_cases(self):
        for case_index, case in enumerate(self.DECODE_CASES):
            splits = 2 if case_index == 0 else 1
            with (
                self.subTest(case=case.name, backend=case.backend),
                envs.SGLANG_TRTLLM_MHA_DECODE_SEQ_LEN_SPLITS.override(splits),
            ):
                run_dense_attention_case(
                    self,
                    case,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                )

    @unittest.skipUnless(is_sm90_supported(), "FMHA-v2/XQA transition requires SM90")
    def test_decode_after_fmha_prefill(self):
        import flashinfer

        prefill_case = DenseAttentionCase(
            name="trtllm_mha_fmha_prefill_xqa_decode",
            backend="trtllm_mha",
            forward_mode=ForwardMode.EXTEND,
            num_heads=12,
            num_kv_heads=2,
            page_size=64,
            prefix_lens=(0,),
            extend_lens=(6,),
            sliding_window_size=511,
        )
        # XQA's multi-block dispatch uses the context capacity, even for a
        # short sequence. Keep it large enough to exercise its semaphores.
        max_context_len = 4096
        prefill = build_dense_attention_fixture(
            self,
            prefill_case,
            head_dim=64,
            hidden_size=128,
            max_context_len=max_context_len,
            dtype=torch.bfloat16,
            loc_layout="contiguous",
        )
        decode_case = replace(
            prefill_case,
            forward_mode=ForwardMode.DECODE,
            prefix_lens=(6,),
            extend_lens=(),
        )
        decode = replace(
            prefill,
            case=decode_case,
            forward_batch=_make_forward_batch(
                decode_case,
                prefill.runner,
                max_context_len=max_context_len,
                device="cuda",
            ),
            prefix_hidden=[prefill.input_hidden],
            input_hidden=torch.randn_like(prefill.input_hidden[:1]),
        )
        real_decode = flashinfer.decode.trtllm_batch_decode_with_kv_cache

        def decode_with_nan_output(*, query, **kwargs):
            # Stale workspace counters can leave heads unwritten. Do not let
            # an allocator-reused output silently hide missing kernel writes.
            kwargs["out"] = torch.full_like(query, float("nan"))
            return real_decode(query=query, **kwargs)

        run_dense_fixture_eager(prefill)
        with patch.object(
            flashinfer.decode,
            "trtllm_batch_decode_with_kv_cache",
            new=decode_with_nan_output,
        ):
            actual = run_dense_fixture_eager(decode)
        torch.testing.assert_close(
            actual,
            expected_dense_fixture_output(decode),
            atol=DENSE_ATOL,
            rtol=DENSE_RTOL,
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

    @unittest.skipUnless(
        is_sm100_supported(), "TRT-LLM context attention requires SM100"
    )
    def test_runner_mode_split_op_extend_cases(self):
        for case, static_num_tokens in self.SPLIT_OP_CASES:
            for breakable in (False, True):
                runner = "bcg" if breakable else "pcg"
                with self.subTest(case=case.name, backend=case.backend, runner=runner):
                    run_dense_split_op_extend_case(
                        self,
                        case,
                        breakable=breakable,
                        static_num_tokens=static_num_tokens,
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
            with (
                self.subTest(case=case.name, backend=case.backend),
                envs.SGLANG_TRTLLM_MHA_DECODE_SEQ_LEN_SPLITS.override(2),
            ):
                run_dense_frozen_kv_mtp_cuda_graph_runner_case(
                    self,
                    case,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                )

    # XQA has native page-128 kernels (any head layout). max_context_len must
    # be a page multiple so the kit's per-request slot ranges stay page-aligned.
    def test_page128_decode(self):
        case = DenseAttentionCase(
            name="trtllm_mha_xqa_decode_page128_boundary",
            backend="trtllm_mha",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=2,
            page_size=128,
            prefix_lens=(127, 128, 200),
        )
        run_dense_attention_case(
            self,
            case,
            head_dim=self.HEAD_DIM,
            hidden_size=self.HIDDEN_SIZE,
            max_context_len=512,
        )


@unittest.skipIf(
    not torch.cuda.is_available()
    or not is_flashinfer_available()
    or not is_sm100_supported(),
    "CUDA + FlashInfer TRT-LLM-GEN (SM100) are required",
)
class TestTRTLLMMHAPage128TrtllmGen(CustomTestCase):
    """page_size=128 on trtllm-gen (SM100) via dynamic tokens-per-page kernels.

    Those kernels only exist for GQA (q heads per kv head > 1) with equal QK/V
    head dims, so every positive case here is GQA; the MHA layout must fail at
    backend construction (see test_page128_mha_rejected_at_init). All cases
    pass max_context_len=512: the kit's per-request slot ranges start at
    ``page_size + req_idx * max_context_len``, so it must be a page multiple.
    """

    HEAD_DIM = 64
    HIDDEN_SIZE = 256
    MAX_CONTEXT_LEN = 512

    DECODE_CASES = (
        DenseAttentionCase(
            name="trtllm_gen_gqa_decode_page128_boundary",
            backend="trtllm_mha",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=2,
            page_size=128,
            prefix_lens=(127, 128, 200),
        ),
        DenseAttentionCase(
            name="trtllm_gen_gqa4_decode_page128_bsz1",
            backend="trtllm_mha",
            forward_mode=ForwardMode.DECODE,
            num_heads=8,
            num_kv_heads=2,
            page_size=128,
            prefix_lens=(300,),
        ),
    )

    EXTEND_CASES = (
        DenseAttentionCase(
            name="trtllm_gen_gqa_extend_page128",
            backend="trtllm_mha",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            num_kv_heads=2,
            page_size=128,
            prefix_lens=(0, 128),
            extend_lens=(130, 5),
        ),
    )

    CUDA_GRAPH_DECODE_CASES = (
        DenseAttentionCase(
            name="runner_cuda_graph_trtllm_gen_gqa_decode_page128",
            backend="trtllm_mha",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=2,
            page_size=128,
            prefix_lens=(127, 128, 200),
        ),
    )

    def test_page128_decode_cases(self):
        for case in self.DECODE_CASES:
            with self.subTest(case=case.name):
                run_dense_attention_case(
                    self,
                    case,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                    max_context_len=self.MAX_CONTEXT_LEN,
                )

    def test_page128_extend_cases(self):
        for case in self.EXTEND_CASES:
            with self.subTest(case=case.name):
                run_dense_attention_case(
                    self,
                    case,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                    max_context_len=self.MAX_CONTEXT_LEN,
                )

    def test_page128_cuda_graph_decode_cases(self):
        for case in self.CUDA_GRAPH_DECODE_CASES:
            with self.subTest(case=case.name):
                run_dense_cuda_graph_decode_case(
                    self,
                    case,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                    max_context_len=self.MAX_CONTEXT_LEN,
                )

    def test_page128_mha_rejected_at_init(self):
        # heads_per_kv == 1 has no page-128 trtllm-gen kernel; the backend must
        # refuse at construction (not fail mid-capture with a missing-kernel
        # RuntimeError from flashinfer).
        case = DenseAttentionCase(
            name="trtllm_gen_mha_decode_page128_rejected",
            backend="trtllm_mha",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=4,
            page_size=128,
            prefix_lens=(7,),
        )
        with self.assertRaisesRegex(ValueError, "dynamic tokens-per-page"):
            build_dense_attention_fixture(
                self,
                case,
                head_dim=self.HEAD_DIM,
                hidden_size=self.HIDDEN_SIZE,
                max_context_len=self.MAX_CONTEXT_LEN,
            )


if __name__ == "__main__":
    unittest.main()
