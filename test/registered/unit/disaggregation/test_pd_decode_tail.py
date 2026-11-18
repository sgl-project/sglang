"""Decode-tail PD mode (RFC #50547): the transfer truncation and the tail
extend admission surface.

With `--disaggregation-decode-tail-tokens W`, the prefill transfers KV for
`[0, N - W)` of an N-token prompt and drops the handoff token at commit; the
decode worker admits the request through a real extend over the last W prompt
tokens (`prepare_for_tail_extend`) and samples the first token itself.
Short prompts use a metadata-only handoff. Speculative decoding and logprob
requests are outside this initial mode.
"""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.disaggregation.decode_schedule_batch_mixin as mixin_module
import sglang.srt.disaggregation.prefill as prefill_module
import sglang.srt.managers.scheduler as scheduler_module
from sglang.srt.arg_groups import pd_disaggregation_hook
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.configs.nemotron_h import NemotronHConfig
from sglang.srt.disaggregation import decode as decode_module
from sglang.srt.disaggregation.decode_schedule_batch_mixin import (
    ScheduleBatchDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.mode import FAKE_BOOTSTRAP_HOST, DisaggregationMode
from sglang.srt.disaggregation.utils import pre_alloc_fill_len, tail_transfer_input_len
from sglang.srt.managers.io_struct import GenerateReqInput, TokenizedGenerateReqInput
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TailTransferInputLenTest(CustomTestCase):
    def test_cap_stops_at_n_minus_w(self):
        self.assertEqual(tail_transfer_input_len(1000, 10), 990)

    def test_short_prompt_sends_only_metadata(self):
        self.assertEqual(tail_transfer_input_len(5, 10), 0)
        self.assertEqual(tail_transfer_input_len(1, 10), 0)
        self.assertEqual(tail_transfer_input_len(2, 10), 0)
        self.assertEqual(tail_transfer_input_len(11, 10), 1)

    def test_cap_is_identity_without_tail(self):
        self.assertEqual(tail_transfer_input_len(1000, 0), 1000)

    def test_rebootstrap_restores_the_entire_prefix(self):
        req = Req("resume", None, array("q", [10, 11, 12]), SamplingParams())
        req.output_ids.extend([20, 21])
        payload = req.build_rebootstrap_payload()
        self.assertEqual(payload["input_ids"], [10, 11, 12, 20])
        self.assertTrue(payload["disagg_rebootstrap"])
        self.assertEqual(
            tail_transfer_input_len(
                len(payload["input_ids"]),
                2,
                is_rebootstrap=payload["disagg_rebootstrap"],
            ),
            pre_alloc_fill_len(req),
        )
        self.assertEqual(list(req.output_ids), [20, 21])

    def test_rebootstrap_marker_preserves_request_splitting_and_ipc_default(self):
        request = GenerateReqInput(input_ids=[[1], [2]], disagg_rebootstrap=True)
        request.normalize_batch_and_arguments()
        self.assertTrue(request[0].disagg_rebootstrap)
        self.assertTrue(request[1].disagg_rebootstrap)
        self.assertFalse(TokenizedGenerateReqInput.__struct_defaults__[-1])

    def test_rebootstrap_before_first_sample_keeps_the_tail(self):
        req = Req("resume", None, array("q", [10, 11, 12]), SamplingParams())
        payload = req.build_rebootstrap_payload()
        self.assertFalse(payload["disagg_rebootstrap"])
        self.assertEqual(
            tail_transfer_input_len(
                len(payload["input_ids"]),
                2,
                is_rebootstrap=payload["disagg_rebootstrap"],
            ),
            1,
        )


class TailModelGuardTest(CustomTestCase):
    def test_recurrent_config_without_layer_types_is_rejected(self):
        hf_config = NemotronHConfig(layers_block_type=["mamba", "attention"])
        model_config = SimpleNamespace(
            hf_config=hf_config,
            hf_text_config=SimpleNamespace(),
            is_draft_model=False,
            attention_arch=AttentionArch.MHA,
            is_hybrid_swa=False,
        )
        cfg = SimpleNamespace(
            disaggregation_transfer_backend="mooncake",
            disaggregation_decode_allocation_policy="early",
            disaggregation_mode="prefill",
            disaggregation_decode_host_receive_threshold=1,
            dcp_size=1,
            disaggregation_decode_tail_tokens=4,
            max_prefill_tokens=128,
            chunked_prefill_size=64,
            speculative_algorithm=None,
            pp_size=1,
            dp_size=1,
            ep_size=1,
            attn_cp_size=1,
            enable_two_batch_overlap=False,
            enable_mixed_chunk=False,
            enable_lora=False,
            enable_hisparse=False,
            enable_pd_role_switch=False,
            disaggregation_decode_enable_radix_cache=False,
        )
        with (
            patch.object(pd_disaggregation_hook, "resolving_view", return_value=cfg),
            patch.object(pd_disaggregation_hook, "resolved_view", return_value=cfg),
            patch.object(
                pd_disaggregation_hook, "model_config_of", return_value=model_config
            ),
            self.assertRaisesRegex(ValueError, "recurrent state"),
        ):
            pd_disaggregation_hook.handle_pd_disaggregation(cfg)


class IsTailTransferTest(CustomTestCase):
    """The commit gate: tail mode on, and the transfer is not fake warmup."""

    def _assert_gate(self, tail_tokens, req, expected):
        gate = decode_module._is_tail_transfer
        with patch.object(
            decode_module,
            "get_disagg",
            return_value=SimpleNamespace(disaggregation_decode_tail_tokens=tail_tokens),
        ):
            # The patch must stay active across the call: the gate reads the
            # runtime config through get_disagg on every invocation.
            self.assertEqual(gate(req), expected)

    def _req(self, fake=False, with_handoff=False):
        req = SimpleNamespace(
            rid="chat-1",
            output_ids=[42] if with_handoff else [],
        )
        if fake:
            req.bootstrap_host = FAKE_BOOTSTRAP_HOST
        else:
            req.bootstrap_host = "10.0.0.5"
        return req

    def test_real_transfer_in_tail_mode(self):
        self._assert_gate(8, self._req(), True)

    def test_fake_warmup_transfers_keep_their_handoff_id(self):
        self._assert_gate(8, self._req(fake=True), False)

    def test_gate_closed_without_tail_mode(self):
        self._assert_gate(0, self._req(), False)

    def test_resumed_output_keeps_the_prebuilt_path(self):
        self._assert_gate(8, self._req(with_handoff=True), False)


class PrepareForTailExtendTest(CustomTestCase):
    """The batch construction: the transferred rows are the attention prefix,
    the tail W tokens are the extend, and nothing is re-allocated."""

    PROMPT_LEN = 64
    COMMITTED = 56

    def _make_req(self, index):
        return SimpleNamespace(
            rid=f"r{index}",
            kv=ReqKvInfo(
                req_pool_idx=index,
                kv_committed_len=self.COMMITTED,
                kv_allocated_len=self.PROMPT_LEN,
            ),
            extend_end=self.PROMPT_LEN,
            origin_input_ids=list(range(self.PROMPT_LEN)),
            output_ids=[],
            is_retracted=False,
            multimodal_inputs=None,
        )

    def _prepare(self, reqs):
        batch = SimpleNamespace(
            reqs=reqs,
            device="cpu",
            return_logprob=False,
            model_config=SimpleNamespace(vocab_size=128),
        )
        req_to_token = torch.zeros(len(reqs), 256, dtype=torch.int64)
        for i, req in enumerate(reqs):
            # The receiver wrote the transferred KV's row indices for
            # [0, committed) and preallocated the tail with the prompt grid.
            req_to_token[i, : req.extend_end] = torch.arange(req.extend_end) + 100 * (
                i + 1
            )
        batch.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)

        with patch.object(
            mixin_module.SamplingBatchInfo,
            "from_schedule_batch",
            staticmethod(lambda batch, vocab_size: SimpleNamespace()),
        ):
            ScheduleBatchDisaggregationDecodeMixin.prepare_for_tail_extend(batch)
        return batch

    def test_extend_covers_only_the_tail(self):
        req = self._make_req(0)
        # pre_alloc_fill_len with empty output_ids == the full prompt length.
        self.assertEqual(pre_alloc_fill_len(req), self.PROMPT_LEN)

        batch = self._prepare([req])
        self.assertEqual(batch.extend_num_tokens, 8)
        # The attention prefix is the transferred range, not the tree view.
        self.assertEqual(batch.prefix_lens, [self.COMMITTED])
        self.assertEqual(batch.extend_lens, [8])
        # input_ids are the tail prompt tokens, not the handoff token.
        self.assertEqual(batch.input_ids.tolist(), list(range(56, 64)))
        # The rows come from the preallocated grid — no fresh allocation.
        self.assertEqual(
            batch.out_cache_loc.tolist(),
            (torch.arange(56, 64) + 100).tolist(),
        )
        self.assertEqual(batch.seq_lens_cpu.tolist(), [self.PROMPT_LEN])
        self.assertEqual(req.kv.kv_committed_len, self.PROMPT_LEN)

    def test_metadata_only_handoff_extends_one_token(self):
        req = self._make_req(0)
        req.kv.truncate_committed(0)
        req.extend_end = 1
        req.origin_input_ids = [42]
        batch = self._prepare([req])
        self.assertEqual(batch.prefix_lens, [0])
        self.assertEqual(batch.input_ids.tolist(), [42])
        self.assertEqual(batch.extend_lens, [1])
        self.assertEqual(req.kv.kv_committed_len, 1)

    def test_two_requests_flatten_in_order(self):
        reqs = [self._make_req(0), self._make_req(1)]
        batch = self._prepare(reqs)
        self.assertEqual(batch.extend_num_tokens, 16)
        self.assertEqual(batch.prefix_lens, [self.COMMITTED, self.COMMITTED])
        self.assertEqual(
            batch.input_ids.tolist(), list(range(56, 64)) + list(range(56, 64))
        )
        self.assertEqual(
            batch.out_cache_loc.tolist(),
            (torch.arange(56, 64) + 100).tolist()
            + (torch.arange(56, 64) + 200).tolist(),
        )
        self.assertEqual(batch.seq_lens_cpu.tolist(), [64, 64])


class TailSchedulingTest(CustomTestCase):
    def test_ready_grammar_is_enqueued_before_tail_selection(self):
        grammar_req = object()
        tail_batch = object()
        enqueued = []

        def select_tail(running_batch):
            self.assertEqual(enqueued, [grammar_req])
            return tail_batch

        scheduler = SimpleNamespace(
            scheduler_stage_metrics=None,
            last_batch=None,
            grammar_manager=SimpleNamespace(
                has_waiting_grammars=lambda: True,
                get_ready_grammar_requests=lambda: [grammar_req],
            ),
            _add_request_to_queue=enqueued.append,
            _get_new_tail_extend_batch=select_tail,
        )
        plan = decode_module.SchedulerDisaggregationDecodeMixin.get_next_disagg_decode_batch_to_run(
            scheduler, object()
        )
        self.assertIs(plan.batch_to_run, tail_batch)


class StagingPrefetchTailBoundaryTest(CustomTestCase):
    """Staging reservations follow the shortened transfer, including zero."""

    def _prefetch(self, prompt_len, tail, *, rebootstrap=False):
        seen = []
        kv_mgr = SimpleNamespace(
            _prefetch_staging_reqs=lambda room, upto_tokens=None: seen.append(
                (room, upto_tokens)
            ),
            transfer_infos={7: {}},
        )
        scheduler = SimpleNamespace(
            disagg_prefill_bootstrap_queue=SimpleNamespace(kv_manager=kv_mgr)
        )
        req = SimpleNamespace(
            bootstrap_room=7,
            origin_input_ids=list(range(prompt_len)),
            disagg_rebootstrap=rebootstrap,
            extend_end=prompt_len,
        )
        with (
            patch.object(
                prefill_module,
                "get_disagg",
                return_value=SimpleNamespace(disaggregation_decode_tail_tokens=tail),
            ),
            patch.object(
                prefill_module.envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION,
                "get",
                return_value=False,
            ),
        ):
            prefill_module.SchedulerDisaggregationPrefillMixin.maybe_prefetch_staging_for_batch(
                scheduler, SimpleNamespace(reqs=[req])
            )
        return seen

    def test_tail_reserves_only_the_transferred_prefix(self):
        self.assertEqual(self._prefetch(10, 4), [(7, 6)])

    def test_metadata_only_handoff_reserves_no_staging_slot(self):
        self.assertEqual(self._prefetch(4, 10), [(7, 0)])
        self.assertEqual(self._prefetch(1, 10), [(7, 0)])

    def test_rebootstrap_reserves_the_full_restored_prefix(self):
        self.assertEqual(self._prefetch(10, 4, rebootstrap=True), [(7, 10)])

    def test_without_tail_keeps_the_previous_prefetch(self):
        self.assertEqual(self._prefetch(10, 0), [(7, None)])


class TailBoundaryRestoreTest(CustomTestCase):
    """A retracted tail request resumes with the transferred prefix only."""

    def _req(self, prompt_len, *, output_ids=(), rebootstrap=False):
        req = SimpleNamespace(
            rid="resume",
            origin_input_ids=[42] * prompt_len,
            output_ids=list(output_ids),
            disagg_rebootstrap=rebootstrap,
            kv=ReqKvInfo(req_pool_idx=0),
        )
        req.kv.set_row_len(prompt_len)
        return req

    def _truncate(self, req, tail):
        with patch.object(
            decode_module,
            "get_disagg",
            return_value=SimpleNamespace(disaggregation_decode_tail_tokens=tail),
        ):
            decode_module._truncate_to_tail_boundary(req)

    def test_resume_restores_the_tail_boundary(self):
        req = self._req(64)
        req.extend_end = 64
        self._truncate(req, 8)
        self.assertEqual(req.kv.kv_committed_len, 56)
        self.assertEqual(req.kv.kv_allocated_len, 64)
        self.assertEqual(req.extend_end, 64)

    def test_short_prompt_resume_commits_nothing(self):
        req = self._req(1)
        req.extend_end = 1
        self._truncate(req, 10)
        self.assertEqual(req.kv.kv_committed_len, 0)
        self.assertEqual(req.extend_end, 1)

    def test_rebootstrap_resume_keeps_the_full_prefix(self):
        req = self._req(10, rebootstrap=True)
        req.extend_end = 10
        self._truncate(req, 4)
        self.assertEqual(req.kv.kv_committed_len, 10)


class TailIntakeValidationTest(CustomTestCase):
    """Session turns pass the same Phase 1 gate as ordinary requests."""

    def _validate(self, **attrs):
        req = SimpleNamespace(rid="chat-1", **attrs)
        retired = []
        scheduler = SimpleNamespace(
            disaggregation_mode=DisaggregationMode.DECODE,
            request_retirer=SimpleNamespace(retire_unadmitted=retired.append),
        )
        with (
            patch.object(
                scheduler_module,
                "get_disagg",
                return_value=SimpleNamespace(disaggregation_decode_tail_tokens=4),
            ),
            patch.object(
                scheduler_module,
                "prepare_abort",
                lambda req, msg, status_code: setattr(req, "abort_msg", msg),
            ),
        ):
            rejected = scheduler_module.Scheduler._reject_unsupported_tail_request(
                scheduler, req
            )
        return rejected, retired, req

    def test_plain_request_is_admitted(self):
        rejected, retired, _ = self._validate()
        self.assertFalse(rejected)
        self.assertEqual(retired, [])

    def test_session_logprob_request_is_rejected(self):
        rejected, retired, req = self._validate(return_logprob=True)
        self.assertTrue(rejected)
        self.assertEqual(retired, [req])
        self.assertIn("does not support logprobs", req.abort_msg)

    def test_explicit_embedding_request_is_rejected(self):
        rejected, retired, req = self._validate(input_embeds=[[1.0]])
        self.assertTrue(rejected)
        self.assertEqual(retired, [req])


if __name__ == "__main__":
    unittest.main()
