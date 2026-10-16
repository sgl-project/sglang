import types
import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention import trtllm_mha_backend
from sglang.srt.layers.attention.trtllm_mha_decode_groups import DecodeGroupPlans
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
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_MAX_CONTEXT_LEN,
    DENSE_ATOL,
    DENSE_RTOL,
    DenseAttentionCase,
    _make_forward_batch,
    build_dense_attention_fixture,
    expected_dense_output_from_inputs,
    make_dense_case_with_prefix_lens,
    make_dense_random_inputs,
    prepare_dense_runner_inputs,
    run_dense_attention_case,
    run_dense_forward,
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

    def _decode_forwards(self, case, later_prefix_lens, *, hoist):
        # One backend, one forward per length set; the later sets reuse freed metadata
        # buffers, so a plan kept from an earlier forward would be read stale.
        torch.manual_seed(0)
        with (
            envs.SGLANG_TRTLLM_MHA_DECODE_SEQ_LEN_SPLITS.override(4),
            envs.SGLANG_Q35MF_PDDEC_SPLIT_HOIST.override(hoist),
        ):
            fixture = build_dense_attention_fixture(
                self, case, head_dim=self.HEAD_DIM, hidden_size=self.HIDDEN_SIZE
            )
            outputs = []
            for i, prefix_lens in enumerate((case.prefix_lens,) + later_prefix_lens):
                step = make_dense_case_with_prefix_lens(
                    case, f"{case.name}_{i}", prefix_lens
                )
                inputs = make_dense_random_inputs(
                    step, fixture, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE
                )
                batch = _make_forward_batch(
                    step,
                    fixture.runner,
                    max_context_len=DEFAULT_MAX_CONTEXT_LEN,
                    device=DEFAULT_DEVICE,
                )
                prepare_dense_runner_inputs(
                    fixture,
                    step,
                    batch,
                    inputs,
                    max_context_len=DEFAULT_MAX_CONTEXT_LEN,
                )
                with (
                    torch.no_grad(),
                    forward_context(ForwardContext(attn_backend=fixture.backend)),
                ):
                    fixture.backend.init_forward_metadata(batch)
                    actual = run_dense_forward(fixture, batch, inputs)
                expected = expected_dense_output_from_inputs(
                    fixture, step, inputs, None
                )
                torch.testing.assert_close(
                    actual, expected, atol=DENSE_ATOL, rtol=DENSE_RTOL
                )
                outputs.append(actual)
        return outputs

    def test_split_hoist_matches_per_call_groups(self):
        """Length groups built once per forward give the per-call groups' output bit for
        bit, on every forward of a backend and inside a CUDA graph."""
        case = DenseAttentionCase(
            name="trtllm_mha_decode_split_hoist",
            backend="trtllm_mha",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=2,
            page_size=16,
            prefix_lens=(40, 7, 33, 7, 60, 2, 21),
        )
        later = ((5, 62, 12, 48, 9, 30, 17), (21, 2, 60, 7, 33, 7, 40))
        stock = self._decode_forwards(case, later, hoist=False)
        hoisted = self._decode_forwards(case, later, hoist=True)
        for i, (a, b) in enumerate(zip(stock, hoisted)):
            with self.subTest(forward=i):
                self.assertTrue(torch.equal(a, b))
        graph_case = self.CUDA_GRAPH_DECODE_CASES[0]
        with (
            self.subTest(case=graph_case.name, runner="cuda_graph"),
            envs.SGLANG_TRTLLM_MHA_DECODE_SEQ_LEN_SPLITS.override(2),
            envs.SGLANG_Q35MF_PDDEC_SPLIT_HOIST.override(True),
        ):
            run_dense_cuda_graph_decode_case(
                self,
                graph_case,
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


class TestTRTLLMMHADecodeGroupHoist(CustomTestCase):
    """Per-call length groups vs groups built once per forward, with a fake FMHA whose
    output depends on each request's inputs and on its group's size and slot."""

    @staticmethod
    def _fake_decode(
        *, query, block_tables, seq_lens, out=None, out_dtype=None, **kwargs
    ):
        n = seq_lens.shape[0]
        rows = query.float().view(n, -1, *query.shape[1:])
        slot = torch.arange(n, dtype=torch.float32).view(n, 1, 1, 1)
        value = (
            rows * (n + 1)
            + seq_lens.float().view(n, 1, 1, 1)
            + block_tables[:, :1].float().view(n, 1, 1, 1) / 8
            + slot / 4
        )
        if kwargs.get("mask") is not None:
            value = value + kwargs["mask"].float().sum() / 16
        value = value.view(query.shape)
        if out is None:
            return value.to(out_dtype)
        out.copy_(value)
        return out

    def _run(self, *, hoist, query, block_tables, seq_lens, q_len, out, mask):
        backend = types.SimpleNamespace(
            q_data_type=torch.bfloat16,
            workspace_buffer=None,
            max_context_len=4096,
            _multi_ctas_kv_counter_buffer=None,
            decode_seq_len_splits=4,
            decode_split_hoist=hoist,
            _decode_group_plans=DecodeGroupPlans(),
        )
        fake_flashinfer = types.SimpleNamespace(
            decode=types.SimpleNamespace(
                trtllm_batch_decode_with_kv_cache=self._fake_decode
            )
        )
        with mock.patch.object(
            trtllm_mha_backend, "flashinfer", fake_flashinfer, create=True
        ):
            return trtllm_mha_backend.TRTLLMHAAttnBackend._run_fixed_q_len_decode(
                backend,
                query,
                None,
                block_tables,
                seq_lens,
                bmm1_scale=1.0,
                bmm2_scale=1.0,
                window_left=-1,
                sinks=None,
                q_len_per_req=q_len,
                out=out,
                mask=mask,
            )

    def test_hoisted_groups_match_per_call_groups(self):
        # 7 requests in 4 groups (2, 2, 2, 1), with tied lengths across a group edge.
        seq_lens = torch.tensor([5, 3, 5, 9, 1, 3, 7], dtype=torch.int32)
        block_tables = torch.arange(7 * 3, dtype=torch.int32).view(7, 3) * 5 % 17
        for q_len in (1, 7):
            for use_out in (False, True):
                for use_mask in (False, True):
                    query = torch.randn(7 * q_len, 2, 8).to(torch.bfloat16)
                    mask = torch.ones(7, q_len, dtype=torch.int32) if use_mask else None
                    results = []
                    for hoist in (False, True):
                        out = torch.zeros_like(query) if use_out else None
                        result = self._run(
                            hoist=hoist,
                            query=query,
                            block_tables=block_tables,
                            seq_lens=seq_lens,
                            q_len=q_len,
                            out=out,
                            mask=mask,
                        )
                        if use_out:
                            self.assertEqual(result.data_ptr(), out.data_ptr())
                        results.append(result)
                    with self.subTest(q_len=q_len, out=use_out, mask=use_mask):
                        self.assertTrue(torch.equal(results[0], results[1]))


if __name__ == "__main__":
    unittest.main()
