"""Unit tests for external speculative decoding: wire layout, ZMQ transport
reply handling, the EXTERNAL server-args handler and the drafter context."""

import contextlib
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec.msgpack
import torch
import zmq

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import _handle_external
from sglang.srt.managers.req_types import FINISH_LENGTH, MultimodalInputs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.utils import validate_input_length
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.external_components.base_transport import (
    ExternalDraftTransport,
)
from sglang.srt.speculative.external_components.contract import (
    AllMessages,
    CancelInferenceRequest,
    CancelInferenceResponse,
    ConfigureDraftEngineRequest,
    ConfigureDraftEngineResponse,
    DraftContext,
    DraftTokensReply,
    DraftTokensRequest,
    PrefillCacheRequest,
    PrefillContext,
)
from sglang.srt.speculative.external_components.zmq_transport import ZmqTransport
from sglang.srt.speculative.external_worker import ExternalWorker

# -- Wire layout ----------------------------------------------------------------


class TestWireLayout(CustomTestCase):
    def test_messages_encode_to_documented_positional_arrays(self):
        """External drafters parse these arrays by position (tag first, then
        fields in declaration order), so a field reorder or rename of a tag
        breaks every drafter while still round-tripping in Python."""
        cases = [
            (
                ConfigureDraftEngineRequest(num_draft_tokens=7, protocol_version=1),
                ["ConfigureDraftEngineRequest", 7, 1],
            ),
            (ConfigureDraftEngineResponse(), ["ConfigureDraftEngineResponse"]),
            (
                PrefillCacheRequest(
                    prefill_contexts=[
                        PrefillContext(request_id=0, prompt_token_ids=[10, 20])
                    ]
                ),
                ["PrefillCacheRequest", [["PrefillContext", 0, [10, 20]]]],
            ),
            (
                DraftTokensRequest(
                    draft_contexts=[
                        DraftContext(request_id=5, seq_len=100, bonus_token_id=42)
                    ]
                ),
                ["DraftTokensRequest", [["DraftContext", 5, 100, 42]]],
            ),
            (
                DraftTokensReply(request_id=3, token_ids=[1, 2]),
                ["DraftTokensReply", 3, [1, 2]],
            ),
            (
                CancelInferenceRequest(request_ids=[1, 2]),
                ["CancelInferenceRequest", [1, 2]],
            ),
            (CancelInferenceResponse(), ["CancelInferenceResponse"]),
        ]
        encoder = msgspec.msgpack.Encoder()
        decoder = msgspec.msgpack.Decoder(AllMessages)
        for msg, wire in cases:
            with self.subTest(type(msg).__name__):
                payload = encoder.encode(msg)
                self.assertEqual(msgspec.msgpack.decode(payload), wire)
                self.assertEqual(decoder.decode(payload), msg)


# -- Transport validation tests -----------------------------------------------


class TestTransportReceiveDrafts(CustomTestCase):
    """Test ZmqTransport.receive_drafts validation without real ZMQ."""

    def _make_transport(self):
        return ZmqTransport(endpoint="tcp://127.0.0.1:5580")

    def _patch_recv(self, transport, replies):
        """Patch _recv to return replies in sequence."""
        it = iter(replies)
        transport._recv = MagicMock(side_effect=lambda **kw: next(it))

    def test_unexpected_request_id_raises(self):
        transport = self._make_transport()
        self._patch_recv(transport, [DraftTokensReply(request_id=99, token_ids=[1, 2])])
        with self.assertRaisesRegex(RuntimeError, "Unexpected request_id"):
            transport.receive_drafts(request_ids=[0], num_draft_tokens=2)

    def test_wrong_token_count_raises(self):
        transport = self._make_transport()
        self._patch_recv(transport, [DraftTokensReply(request_id=0, token_ids=[1, 2])])
        with self.assertRaisesRegex(RuntimeError, "2 tokens, expected 5"):
            transport.receive_drafts(request_ids=[0], num_draft_tokens=5)

    def test_empty_token_ids_returns_none(self):
        transport = self._make_transport()
        self._patch_recv(transport, [DraftTokensReply(request_id=0, token_ids=[])])
        result = transport.receive_drafts(request_ids=[0], num_draft_tokens=3)
        self.assertIsNone(result[0])

    def test_wrong_message_type_raises(self):
        transport = self._make_transport()
        self._patch_recv(transport, [ConfigureDraftEngineResponse()])
        with self.assertRaisesRegex(RuntimeError, "Expected DraftTokensReply"):
            transport.receive_drafts(request_ids=[0], num_draft_tokens=3)

    def test_out_of_order_replies(self):
        transport = self._make_transport()
        replies = [
            DraftTokensReply(request_id=2, token_ids=[7, 8]),
            DraftTokensReply(request_id=0, token_ids=[1, 2]),
            DraftTokensReply(request_id=1, token_ids=[4, 5]),
        ]
        self._patch_recv(transport, replies)
        result = transport.receive_drafts(request_ids=[0, 1, 2], num_draft_tokens=2)
        self.assertEqual(result[0], [1, 2])
        self.assertEqual(result[1], [4, 5])
        self.assertEqual(result[2], [7, 8])

    def test_cancel_ack_between_draft_replies_is_skipped(self):
        """The ack of a fire-and-forget cancel arrives among draft replies and
        must not be treated as a protocol error."""
        transport = self._make_transport()
        transport._send = MagicMock()
        transport.cancel_inference(request_ids=[5])
        replies = [
            DraftTokensReply(request_id=0, token_ids=[1, 2]),
            CancelInferenceResponse(),
            DraftTokensReply(request_id=1, token_ids=[4, 5]),
        ]
        self._patch_recv(transport, replies)
        result = transport.receive_drafts(request_ids=[0, 1], num_draft_tokens=2)
        self.assertEqual(result, {0: [1, 2], 1: [4, 5]})

    def test_cancelled_request_is_not_waited_for(self):
        """A cancelled request's draft may never be answered, so it is not
        waited for."""
        transport = self._make_transport()
        transport._send = MagicMock()
        transport.cancel_inference(request_ids=[1])
        self._patch_recv(transport, [DraftTokensReply(request_id=0, token_ids=[1, 2])])
        result = transport.receive_drafts(request_ids=[0, 1], num_draft_tokens=2)
        self.assertEqual(result, {0: [1, 2], 1: None})
        self.assertEqual(transport._recv.call_count, 1)

    def test_late_reply_for_cancelled_request_is_dropped(self):
        """A reply already sent for a since-cancelled request must not fail the read."""
        transport = self._make_transport()
        transport._send = MagicMock()
        transport.cancel_inference(request_ids=[1])
        replies = [
            DraftTokensReply(request_id=1, token_ids=[7, 8]),
            DraftTokensReply(request_id=0, token_ids=[1, 2]),
        ]
        self._patch_recv(transport, replies)
        result = transport.receive_drafts(request_ids=[0], num_draft_tokens=2)
        self.assertEqual(result, {0: [1, 2]})

    def test_cancel_ack_clears_cancelled_ids(self):
        """No reply for a cancelled id can follow its ack, so tracking ends there."""
        transport = self._make_transport()
        transport._send = MagicMock()
        transport.cancel_inference(request_ids=[1])
        replies = [
            CancelInferenceResponse(),
            DraftTokensReply(request_id=0, token_ids=[1, 2]),
            DraftTokensReply(request_id=1, token_ids=[7, 8]),
        ]
        self._patch_recv(transport, replies)
        transport.receive_drafts(request_ids=[0], num_draft_tokens=2)
        # After the ack no reply for request 1 can follow, so one is an error.
        with self.assertRaisesRegex(RuntimeError, "Unexpected request_id 1"):
            transport.receive_drafts(request_ids=[2], num_draft_tokens=2)

    def test_timeout_fails_waiting_requests(self):
        """A stalled drafter fails the requests still waiting instead of raising
        out of the scheduler, and their late replies are dropped."""
        transport = self._make_transport()
        replies = iter([DraftTokensReply(request_id=0, token_ids=[1, 2])])

        def recv(**kw):
            reply = next(replies, None)
            if reply is None:
                raise zmq.Again()
            return reply

        transport._recv = MagicMock(side_effect=recv)
        result = transport.receive_drafts(request_ids=[0, 1], num_draft_tokens=2)
        self.assertEqual(result, {0: [1, 2], 1: None})

        self._patch_recv(
            transport,
            [
                DraftTokensReply(request_id=1, token_ids=[7, 8]),
                DraftTokensReply(request_id=2, token_ids=[3, 4]),
            ],
        )
        result = transport.receive_drafts(request_ids=[2], num_draft_tokens=2)
        self.assertEqual(result, {2: [3, 4]})

    def test_cancel_ack_without_pending_cancel_raises(self):
        """Acks are only skipped while a cancel is pending."""
        transport = self._make_transport()
        self._patch_recv(transport, [CancelInferenceResponse()])
        with self.assertRaisesRegex(RuntimeError, "Expected DraftTokensReply"):
            transport.receive_drafts(request_ids=[0], num_draft_tokens=2)


# -- _handle_external server_args validation -----------------------------------


class TestHandleExternal(CustomTestCase):
    def _args(self, **overrides):
        fields = dict(
            model_path="dummy",
            speculative_algorithm="EXTERNAL",
            speculative_external_endpoint="tcp://127.0.0.1:5580",
        )
        return ServerArgs(**(fields | overrides))

    def test_unsupported_configs_fail(self):
        for fields, message in [
            ({"speculative_external_endpoint": None}, "speculative-external-endpoint"),
            ({"enable_dp_attention": True, "tp_size": 2, "dp_size": 2}, "DP attention"),
            ({"pp_size": 2}, "pipeline parallelism"),
            ({"dp_size": 2}, "data parallelism"),
            ({"disaggregation_mode": "decode"}, "PD disaggregation"),
            ({"disaggregation_mode": "prefill"}, "PD disaggregation"),
            ({"speculative_num_draft_tokens": 1}, "speculative-num-draft-tokens >= 2"),
        ]:
            with (
                self.subTest(fields=fields),
                self.assertRaisesRegex(ValueError, message),
            ):
                _handle_external(self._args(**fields))

    def test_defaults_applied(self):
        args = self._args(enable_mixed_chunk=True)
        _handle_external(args)

        self.assertEqual(resolution_result(args, "max_running_requests"), 48)
        self.assertFalse(resolution_result(args, "enable_mixed_chunk"))
        self.assertEqual(resolution_result(args, "speculative_eagle_topk"), 1)
        self.assertEqual(resolution_result(args, "speculative_num_draft_tokens"), 8)
        self.assertEqual(resolution_result(args, "speculative_num_steps"), 7)

    def test_explicit_values_preserved(self):
        args = self._args(speculative_num_draft_tokens=12, max_running_requests=128)
        _handle_external(args)

        self.assertEqual(resolution_result(args, "speculative_num_draft_tokens"), 12)
        self.assertEqual(resolution_result(args, "speculative_num_steps"), 11)
        self.assertEqual(resolution_result(args, "max_running_requests"), 128)


# -- Drafter context --------------------------------------------------------------


class _RecordingTransport(ExternalDraftTransport):
    def __init__(self, *, endpoint):
        self.prefill_contexts = []
        self.cancelled_ids = []

    def open(self):
        pass

    def close(self):
        pass

    def configure_draft_engine(self, *, num_draft_tokens, protocol_version=1):
        pass

    def prefill_cache(self, *, prefill_contexts):
        self.prefill_contexts.extend(prefill_contexts)

    def request_drafts(self, *, draft_contexts):
        pass

    def receive_drafts(self, *, request_ids, num_draft_tokens):
        return {}

    def cancel_inference(self, *, request_ids):
        self.cancelled_ids.extend(request_ids)


class TestDrafterContext(CustomTestCase):
    """The drafter's prefill context must be exactly the sequence that the next
    DraftContext's seq_len counts."""

    def setUp(self):
        override = get_context().override_server_args(
            device="cpu",
            speculative_algorithm="EXTERNAL",
            speculative_external_endpoint="tcp://127.0.0.1:5580",
            speculative_num_draft_tokens=4,
            disable_overlap_schedule=True,
        )
        override.install()
        self.addCleanup(override.restore)
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(get_parallel().override(tp_rank=0))
        stack.enter_context(
            patch(
                "sglang.srt.speculative.external_worker.ZmqTransport",
                _RecordingTransport,
            )
        )

    def _make_worker(self):
        target = SimpleNamespace(
            model_runner=None,
            forward_batch_generation=lambda batch: SimpleNamespace(
                logits_output=None,
                next_token_ids=torch.tensor([9]),
                can_run_cuda_graph=False,
            ),
        )
        return ExternalWorker(
            server_args=None, gpu_id=0, nccl_port=0, target_worker=target
        )

    def _prefill_then_draft_seq_len(self, worker, req):
        num_prefill_tokens = len(req.origin_input_ids) + len(req.output_ids)
        batch = SimpleNamespace(reqs=[req], seq_lens=torch.tensor([num_prefill_tokens]))
        worker._forward_extend(batch)
        req.output_ids.append(9)
        batch.seq_lens_cpu = torch.tensor([num_prefill_tokens])
        # Only this launch is in flight: the prefill result is processed.
        req.inflight_steps = 1
        seq_lens, bonus_tokens = worker._get_draft_positions(batch)
        self.assertEqual(bonus_tokens, [9])
        return seq_lens[0]

    def test_unprocessed_bonus_is_relayed(self):
        worker = self._make_worker()
        done = Req("done", "", array("q", [1, 2]), SamplingParams())
        done.output_ids = array("q", [7])
        done.inflight_steps = 1
        pending = Req("pending", "", array("q", [1, 2]), SamplingParams())
        pending.output_ids = array("q", [7])
        # The previous step's result is not processed yet.
        pending.inflight_steps = 2
        batch = SimpleNamespace(
            reqs=[done, pending],
            seq_lens_cpu=torch.tensor([3, 5]),
            spec_info=SimpleNamespace(
                accept_tokens=torch.tensor([7, 0, 0, 0, 8, 6, 0, 0]),
                accept_lens=torch.tensor([1, 2]),
            ),
        )
        _, bonus_tokens = worker._get_draft_positions(batch)
        self.assertEqual(bonus_tokens, [7, 6])

    def test_multimodal_and_retained_output(self):
        for multimodal in [False, True]:
            for output_ids in [[], [4, 5]]:
                with self.subTest(multimodal=multimodal, output_ids=output_ids):
                    worker = self._make_worker()
                    req = Req("r", "", array("q", [1, 2, 3]), SamplingParams())
                    if multimodal:
                        # The scheduler pads media tokens with cache ids.
                        req.origin_input_ids = array("q", [1] + [1 << 60] * 16 + [2, 3])
                        req.multimodal_inputs = MultimodalInputs(mm_items=[])
                    req.output_ids = array("q", output_ids)

                    seq_len = self._prefill_then_draft_seq_len(worker, req)

                    [context] = worker._transport.prefill_contexts
                    self.assertEqual(context.prompt_token_ids, [1, 2, 3] + output_ids)
                    self.assertEqual(seq_len, len(context.prompt_token_ids))

    def test_truncated_prompt(self):
        worker = self._make_worker()
        req = Req("r", "", array("q", [1, 2, 3, 4, 5, 6]), SamplingParams())
        validate_input_length(req, max_req_input_len=4, allow_auto_truncate=True)

        seq_len = self._prefill_then_draft_seq_len(worker, req)

        [context] = worker._transport.prefill_contexts
        self.assertEqual(context.prompt_token_ids, [1, 2, 3, 4])
        self.assertEqual(seq_len, 4)

    def test_request_aborted_at_admission_is_not_sent(self):
        worker = self._make_worker()
        req = Req("r", "", array("q", [1, 2, 3, 4, 5, 6]), SamplingParams())
        req.set_finish_with_abort("Input is too long")

        worker._forward_extend(SimpleNamespace(reqs=[req], seq_lens=torch.tensor([1])))

        self.assertEqual(worker._transport.prefill_contexts, [])
        self.assertEqual(worker._transport.cancelled_ids, [])

    def test_request_finished_in_prefill_is_cancelled_at_once(self):
        """A request that finishes in prefill (e.g. /health, max_new_tokens=1)
        is cancelled when it finishes, not at the next forward, and only once."""
        worker = self._make_worker()
        req = Req("r", "", array("q", [1, 2, 3]), SamplingParams())
        worker._forward_extend(SimpleNamespace(reqs=[req], seq_lens=torch.tensor([3])))
        [context] = worker._transport.prefill_contexts
        req.output_ids.append(9)
        req.finished_reason = FINISH_LENGTH(length=1)

        worker.note_request_finished(rid=req.rid, natural_stop=False)
        self.assertEqual(worker._transport.cancelled_ids, [context.request_id])

        worker._cancel_departed_requests(SimpleNamespace(reqs=[]))
        self.assertEqual(worker._transport.cancelled_ids, [context.request_id])

    def test_retraction_without_decode_step_resends_context(self):
        """A retraction the worker sees no decode step for (e.g. pausing with
        retract) still replaces the drafter's context."""
        worker = self._make_worker()
        req = Req("r", "", array("q", [1, 2, 3]), SamplingParams())
        self._prefill_then_draft_seq_len(worker, req)
        req.output_ids.extend([4, 5])
        req.reset_for_retract()
        # Cleared when the request is scheduled for prefill again.
        req.is_retracted = False

        seq_len = self._prefill_then_draft_seq_len(worker, req)
        worker._cancel_departed_requests(SimpleNamespace(reqs=[req]))

        old, new = worker._transport.prefill_contexts
        self.assertEqual(worker._transport.cancelled_ids, [old.request_id])
        self.assertNotEqual(new.request_id, old.request_id)
        self.assertEqual(new.prompt_token_ids, [1, 2, 3, 9, 4, 5])
        self.assertEqual(seq_len, len(new.prompt_token_ids))


# -- SpeculativeAlgorithm.EXTERNAL integration ---------------------------------


class TestExternalAlgorithmEnum(CustomTestCase):
    def test_from_string(self):
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        algo = SpeculativeAlgorithm.from_string("EXTERNAL")
        self.assertIs(algo, SpeculativeAlgorithm.EXTERNAL)

    def test_has_no_draft_kv(self):
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        self.assertFalse(SpeculativeAlgorithm.EXTERNAL.has_draft_kv())


if __name__ == "__main__":
    unittest.main(verbosity=3)
