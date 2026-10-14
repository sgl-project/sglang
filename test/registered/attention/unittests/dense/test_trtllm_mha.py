import unittest
from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
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


def _mixed_extend_case(
    name,
    prefix_lens,
    extend_lens,
    *,
    num_heads=4,
    num_kv_heads=4,
    page_size=16,
    sliding_window_size=None,
):
    return DenseAttentionCase(
        name=name,
        backend="trtllm_mha",
        forward_mode=ForwardMode.EXTEND,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        page_size=page_size,
        prefix_lens=prefix_lens,
        extend_lens=extend_lens,
        sliding_window_size=sliding_window_size,
    )


# (prefill reqs, prefill tokens, tail reqs) for an extend batch.
def _decode_tail_split_point(extend_lens):
    num_prefill = next(
        (i + 1 for i in reversed(range(len(extend_lens))) if extend_lens[i] != 1), 0
    )
    num_tail = len(extend_lens) - num_prefill
    return num_prefill, sum(extend_lens[:num_prefill]), num_tail


# Records (requests, query rows) for each context and decode kernel call.
@contextmanager
def _record_trtllm_kernel_rows():
    import flashinfer

    calls = {"context": [], "decode": []}
    real_context = flashinfer.prefill.trtllm_batch_context_with_kv_cache
    real_decode = flashinfer.decode.trtllm_batch_decode_with_kv_cache

    def context(*, query, batch_size, **kwargs):
        calls["context"].append((batch_size, query.shape[0]))
        return real_context(query=query, batch_size=batch_size, **kwargs)

    def decode(*, query, seq_lens, **kwargs):
        calls["decode"].append((seq_lens.shape[0], query.shape[0]))
        return real_decode(query=query, seq_lens=seq_lens, **kwargs)

    with (
        patch.object(
            flashinfer.prefill, "trtllm_batch_context_with_kv_cache", new=context
        ),
        patch.object(
            flashinfer.decode, "trtllm_batch_decode_with_kv_cache", new=decode
        ),
    ):
        yield calls


@unittest.skipIf(
    not torch.cuda.is_available()
    or not is_flashinfer_available()
    or not is_sm100_supported(),
    "CUDA + FlashInfer TRT-LLM-GEN (SM100) are required",
)
class TestTRTLLMMHAMixedExtendDecodeTail(CustomTestCase):
    """Mixed-chunk extend batches end with the running requests' 1-token rows.

    The context kernel must see only the leading requests and the decode kernel
    only the trailing 1-token run; each case checks both the output and which
    kernel saw which rows, since the unsplit path produces the same output.
    """

    HEAD_DIM = 64
    HIDDEN_SIZE = 256
    MAX_CONTEXT_LEN = 256

    CASES = (
        # A chunk with a cached prefix beside decodes on both sides of a page edge.
        _mixed_extend_case(
            "chunk_beside_decodes",
            prefix_lens=(16, 15, 16, 17, 47),
            extend_lens=(20, 1, 1, 1, 1),
        ),
        _mixed_extend_case(
            "gqa6_page64",
            prefix_lens=(64, 63, 130, 1),
            extend_lens=(70, 1, 1, 1),
            num_heads=12,
            num_kv_heads=2,
            page_size=64,
        ),
        _mixed_extend_case(
            "two_prefills",
            prefix_lens=(0, 30, 7, 70),
            extend_lens=(17, 9, 1, 1),
        ),
        # The leading 1-token row is not in the tail and stays on the context kernel.
        _mixed_extend_case(
            "one_token_row_before_prefill",
            prefix_lens=(40, 0, 22, 9),
            extend_lens=(1, 13, 1, 1),
        ),
        _mixed_extend_case(
            "all_rows_one_token",
            prefix_lens=(3, 16, 31, 90),
            extend_lens=(1, 1, 1, 1),
        ),
        # No request has a prefix, so cu_seqlens_q aliases cu_seqlens_k.
        _mixed_extend_case(
            "one_token_prompt_no_prefix",
            prefix_lens=(0, 0),
            extend_lens=(9, 1),
        ),
        # Decode rows reach past the window.
        _mixed_extend_case(
            "sliding_window",
            prefix_lens=(50, 10, 100, 33),
            extend_lens=(40, 1, 1, 1),
            num_kv_heads=2,
            sliding_window_size=32,
        ),
        _mixed_extend_case("no_tail", prefix_lens=(10, 20), extend_lens=(5, 7)),
    )

    def _build(self, case, dtype=torch.float16):
        return build_dense_attention_fixture(
            self,
            case,
            head_dim=self.HEAD_DIM,
            hidden_size=self.HIDDEN_SIZE,
            max_context_len=self.MAX_CONTEXT_LEN,
            dtype=dtype,
        )

    def _run_case(self, case, *, out_buffer=False):
        fixture = self._build(case)
        if out_buffer:
            # Rows no kernel writes stay NaN and fail the comparison.
            fixture.forward_batch._attn_output = torch.full(
                (case.num_input_tokens, case.num_heads * self.HEAD_DIM),
                float("nan"),
                dtype=fixture.input_hidden.dtype,
                device="cuda",
            )
        with _record_trtllm_kernel_rows() as calls:
            actual = run_dense_fixture_eager(fixture)
        torch.testing.assert_close(
            actual,
            expected_dense_fixture_output(fixture),
            atol=DENSE_ATOL,
            rtol=DENSE_RTOL,
        )

        num_prefill, num_prefill_tokens, num_tail = _decode_tail_split_point(
            case.extend_lens
        )
        self.assertEqual(
            calls["context"], [(num_prefill, num_prefill_tokens)] if num_prefill else []
        )
        self.assertEqual(sum(reqs for reqs, _ in calls["decode"]), num_tail)
        self.assertEqual(sum(rows for _, rows in calls["decode"]), num_tail)
        if out_buffer:
            self.assertFalse(fixture.forward_batch._attn_output.isnan().any())

    def test_matches_reference(self):
        for case in self.CASES:
            for out_buffer in (False, True):
                with self.subTest(case=case.name, out_buffer=out_buffer):
                    self._run_case(case, out_buffer=out_buffer)

    def test_decode_seq_len_splits(self):
        # Tail rows run in KV-length groups and are scattered back into the tail.
        case = _mixed_extend_case(
            "decode_tail_seq_len_split",
            prefix_lens=(30, 90, 4, 61, 17, 33),
            extend_lens=(24, 1, 1, 1, 1, 1),
            num_heads=8,
            num_kv_heads=2,
        )
        for splits in (2, 8):
            for out_buffer in (False, True):
                with (
                    self.subTest(splits=splits, out_buffer=out_buffer),
                    envs.SGLANG_TRTLLM_MHA_DECODE_SEQ_LEN_SPLITS.override(splits),
                ):
                    self._run_case(case, out_buffer=out_buffer)

    def test_split_op_runners(self):
        # The split must still engage when attention runs as a graph split op
        # on a token-padded batch (the runner also runs one eager forward).
        case = _mixed_extend_case(
            "split_op_decode_tail",
            prefix_lens=(16, 30, 47),
            extend_lens=(20, 1, 1),
        )
        for breakable in (False, True):
            with (
                self.subTest(runner="bcg" if breakable else "pcg"),
                _record_trtllm_kernel_rows() as calls,
            ):
                run_dense_split_op_extend_case(
                    self,
                    case,
                    breakable=breakable,
                    static_num_tokens=32,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                )
                self.assertEqual(calls["context"], [(1, 20)] * 2)
                self.assertEqual(calls["decode"], [(2, 2)] * 2)

    def test_split_matches_unsplit(self):
        """FP8 KV, attention sinks and windows, which the reference does not
        model, against the same backend with the split turned off."""
        case = _mixed_extend_case(
            "decode_tail_split_vs_unsplit",
            prefix_lens=(64, 200, 9, 130, 63),
            extend_lens=(56, 1, 1, 1, 1),
            num_heads=12,
            num_kv_heads=2,
            page_size=64,
        )
        num_prefill, num_prefill_tokens, num_tail = _decode_tail_split_point(
            case.extend_lens
        )
        for kv_dtype in (torch.bfloat16, torch.float8_e4m3fn):
            for window in (-1, 40):
                for use_sinks in (False, True):
                    with self.subTest(
                        kv_dtype=kv_dtype, window=window, sinks=use_sinks
                    ):
                        fixture = self._build(case, dtype=torch.bfloat16)
                        if kv_dtype != torch.bfloat16:
                            self._use_kv_dtype(fixture, kv_dtype)
                        fixture.actual_module.attn.sliding_window_size = window
                        sinks = (
                            torch.randn(case.num_heads, device="cuda")
                            if use_sinks
                            else None
                        )
                        unsplit, unsplit_calls = self._run_attention(
                            fixture, split=False, sinks=sinks
                        )
                        split, split_calls = self._run_attention(
                            fixture, split=True, sinks=sinks
                        )
                        self.assertEqual(unsplit_calls["decode"], [])
                        self.assertEqual(
                            split_calls["context"], [(num_prefill, num_prefill_tokens)]
                        )
                        self.assertEqual(split_calls["decode"], [(num_tail, num_tail)])
                        # Same kernel over the same rows; one bf16 ulp leaves room for
                        # FlashInfer picking a kernel variant by batch size.
                        torch.testing.assert_close(
                            split[:num_prefill_tokens],
                            unsplit[:num_prefill_tokens],
                            atol=1e-5,
                            rtol=2**-8,
                        )
                        torch.testing.assert_close(
                            split[num_prefill_tokens:],
                            unsplit[num_prefill_tokens:],
                            atol=1e-2,
                            rtol=1e-2,
                        )

    @torch.no_grad()
    def test_long_context_tail_matches_fp64(self):
        case = replace(
            _mixed_extend_case(
                "long_context_decode_tail",
                prefix_lens=(8192, 8191, 8192, 8193),
                extend_lens=(31, 1, 1, 1),
                num_heads=12,
                num_kv_heads=2,
                page_size=64,
            ),
            forward_mode=ForwardMode.MIXED,
        )
        for kv_dtype in (torch.bfloat16, torch.float8_e4m3fn):
            for window in (-1, 512):
                with self.subTest(kv_dtype=kv_dtype, window=window):
                    fixture = build_dense_attention_fixture(
                        self,
                        case,
                        head_dim=128,
                        hidden_size=256,
                        max_context_len=16384,
                        dtype=torch.bfloat16,
                    )
                    layer = fixture.actual_module.attn
                    k_scale, v_scale = 1.0, 1.0
                    if kv_dtype != torch.bfloat16:
                        k_scale, v_scale = 0.5, 2.0
                        layer.k_scale_float, layer.v_scale_float = k_scale, v_scale
                        layer.k_scale = torch.tensor(k_scale, device="cuda")
                        layer.v_scale = torch.tensor(v_scale, device="cuda")
                        self._use_kv_dtype(fixture, kv_dtype)
                    layer.sliding_window_size = window
                    sinks = torch.randn(case.num_heads, device="cuda")
                    unsplit, _ = self._run_attention(fixture, split=False, sinks=sinks)
                    actual, calls = self._run_attention(
                        fixture, split=True, sinks=sinks
                    )
                    self.assertEqual(calls["context"], [(1, 31)])
                    self.assertEqual(calls["decode"], [(3, 3)])
                    q, _, _ = fixture.actual_module.project_qkv(fixture.input_hidden)
                    q = q.view(-1, case.num_heads, 128).to(kv_dtype).double()
                    k_cache, v_cache = fixture.runner.token_to_kv_pool.get_kv_buffer(0)
                    # Use the stored (possibly quantized) KV, not the unquantized inputs.
                    references = []
                    for req in range(1, case.batch_size):
                        seq_len = case.seq_lens[req]
                        start = max(0, seq_len - 1 - window) if window >= 0 else 0
                        locs = fixture.runner.req_to_token_pool.req_to_token[
                            req, start:seq_len
                        ].long()
                        k = k_cache[locs].double().repeat_interleave(6, dim=1) * k_scale
                        v = v_cache[locs].double().repeat_interleave(6, dim=1) * v_scale
                        row = 30 + req
                        scores = torch.einsum("hd,khd->hk", q[row], k) * layer.scaling
                        probs = torch.cat(
                            [scores, sinks.double()[:, None]], dim=1
                        ).softmax(-1)
                        expected = torch.einsum("hk,khd->hd", probs[:, :-1], v)
                        references.append(expected)
                    expected = torch.stack(references)
                    split_error = (
                        actual[31:].view_as(expected).double() - expected
                    ).norm() / expected.norm()
                    unsplit_error = (
                        unsplit[31:].view_as(expected).double() - expected
                    ).norm() / expected.norm()
                    # FP8 attention has internal rounding beyond quantizing Q/K/V.
                    # Bound total error and compare with the original context kernel.
                    limit = 3e-2 if kv_dtype == torch.float8_e4m3fn else 5e-3
                    self.assertLess(split_error.item(), limit)
                    self.assertLess(unsplit_error.item(), limit)
                    self.assertLessEqual(split_error.item(), 1.5 * unsplit_error.item())

    @torch.no_grad()
    def test_cuda_graph_replay(self):
        case = _mixed_extend_case(
            "mixed_graph_replay",
            prefix_lens=(64, 130, 63),
            extend_lens=(17, 1, 1),
            num_heads=12,
            num_kv_heads=2,
            page_size=64,
        )
        fixture = self._build(case, dtype=torch.bfloat16)
        self._use_kv_dtype(fixture, torch.float8_e4m3fn)
        backend, batch = fixture.backend, fixture.forward_batch
        layer = fixture.actual_module.attn
        q, k, v = fixture.actual_module.project_qkv(fixture.input_hidden)
        backend.init_forward_metadata(batch)
        backend.forward_extend(q, k, v, layer, batch)
        # Keep the captured metadata buffers alive while building the reference.
        captured_metadata = backend.forward_metadata
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = backend.forward_extend(
                q, None, None, layer, batch, save_kv_cache=False
            )
        for factor in (1.0, 0.5):
            q.mul_(factor)
            graph.replay()
            actual = output.clone()
            backend.init_forward_metadata(batch)
            backend.forward_metadata.mixed_prefill_reqs = -1
            expected = backend.forward_extend(
                q, None, None, layer, batch, save_kv_cache=False
            )
            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
        self.assertEqual(captured_metadata.mixed_prefill_reqs, 1)

    @staticmethod
    def _run_attention(fixture, *, split, sinks):
        layer = fixture.actual_module.attn
        q, k, v = fixture.actual_module.project_qkv(fixture.input_hidden)
        kwargs = {} if sinks is None else {"sinks": sinks}
        with (
            torch.no_grad(),
            forward_context(ForwardContext(attn_backend=fixture.backend)),
            _record_trtllm_kernel_rows() as calls,
        ):
            fixture.backend.init_forward_metadata(fixture.forward_batch)
            if not split:
                fixture.backend.forward_metadata.mixed_prefill_reqs = -1
            out = layer(q, k, v, fixture.forward_batch, **kwargs)
        return out.clone(), calls

    # Swaps in a KV pool of another dtype, a backend over it, and the prefix KV.
    def _use_kv_dtype(self, fixture, kv_dtype):
        runner = fixture.runner
        head_dim = fixture.actual_module.head_dim
        runner.kv_cache_dtype = kv_dtype
        runner.token_to_kv_pool = MHATokenToKVPool(
            size=runner.token_to_kv_pool.size,
            page_size=runner.page_size,
            dtype=kv_dtype,
            head_num=fixture.case.num_kv_heads,
            head_dim=head_dim,
            layer_num=1,
            device="cuda",
            enable_memory_saver=False,
            enable_alt_stream=False,
        )
        runner.init_kv_index_translator()
        fixture.backend = TRTLLMHAAttnBackend(runner)
        module = fixture.actual_module
        req_to_token = runner.req_to_token_pool.req_to_token
        for req_idx, prefix in enumerate(fixture.prefix_hidden):
            if prefix.shape[0] == 0:
                continue
            _, k, v = module.project_qkv(prefix)
            runner.token_to_kv_pool.set_kv_buffer(
                module.attn,
                req_to_token[req_idx, : prefix.shape[0]].long(),
                k.view(-1, fixture.case.num_kv_heads, head_dim),
                v.view(-1, fixture.case.num_kv_heads, head_dim),
                *fixture.backend._kv_write_scales(module.attn),
            )


@unittest.skipIf(
    not torch.cuda.is_available()
    or not is_flashinfer_available()
    or not (is_sm90_supported() or is_sm120_supported()),
    "FMHA-v2 prefill requires SM90 or SM120",
)
class TestTRTLLMMHAFmhaV2MixedExtend(CustomTestCase):
    def test_mixed_extend_matches_reference(self):
        case = _mixed_extend_case(
            "fmha_v2_mixed_extend",
            prefix_lens=(16, 15, 47),
            extend_lens=(20, 1, 1),
            num_kv_heads=2,
            page_size=64,
        )
        fixture = build_dense_attention_fixture(
            self,
            case,
            head_dim=64,
            hidden_size=256,
            max_context_len=256,
            dtype=torch.bfloat16,
        )
        actual = run_dense_fixture_eager(fixture)
        torch.testing.assert_close(
            actual,
            expected_dense_fixture_output(fixture),
            atol=DENSE_ATOL,
            rtol=DENSE_RTOL,
        )


if __name__ == "__main__":
    unittest.main()
