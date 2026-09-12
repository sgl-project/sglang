"""A request rejected at intake must not be admitted to a PD handoff.

`Req.set_finish_with_abort()` records the rejection in `to_finish` (not
`finished_reason`) and replaces the prompt with a one-token stub, so
`req.finished()` stays False and the disaggregation admission queues let it
through: it completes a bootstrap handshake, reserves a metadata buffer,
initialises an RDMA sender and runs a forward pass on the stub before anything
unwinds it, and the decode worker sits on it until the transfer timeout.

Only disaggregation is affected. In NULL mode the stub costs one cheap forward
pass and the batch boundary returns the 400, so that path is deliberately left
alone -- `test_null_mode_is_deliberately_untouched` pins that.

The door must also not retire a *re-entering* request. "Abort method 3" sets
`to_finish` on a running request and `filter_batch` does not drop it, so a
retracted, preempted or resumed request can arrive carrying one while its queue
still owes a release. The prefill door detects that from the resources the
request still holds; the decode door cannot -- a retracted request holds none of
them by then -- so it gates on its own `is_retracted` / `is_rebootstrap` flags.
"""

import unittest
from array import array
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
from sglang.srt.disaggregation.utils import DisaggregationMode, is_unadmitted_reject
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

ERROR_MSG = (
    "Input length (1500 tokens) exceeds the maximum allowed length (1018 tokens)."
)


def _make_req(prompt_len=1500, session=None):
    req = Req(
        rid="admission-abort",
        origin_input_text="",
        origin_input_ids=array("q", list(range(prompt_len))),
        sampling_params=SamplingParams(max_new_tokens=8),
        session=session,
    )
    req.time_stats.trace_ctx = MagicMock()
    return req


def _make_scheduler():
    return SimpleNamespace(
        output_streamer=MagicMock(),
        beam_coordinator=MagicMock(),
        retire_unadmitted_request=MagicMock(),
    )


def _prefill_queue(sched):
    q = SimpleNamespace(
        scheduler=sched,
        queue=[],
        create_sender=MagicMock(return_value=True),
    )
    return q


def _decode_queue(sched):
    q = SimpleNamespace(
        scheduler=sched,
        retracted_queue=[],
        pending_reqs=[],
        _check_if_req_exceed_kv_capacity=MagicMock(return_value=False),
        _create_receiver_and_enqueue=MagicMock(
            return_value=SimpleNamespace(kv_receiver=MagicMock())
        ),
        _resolve_prefill_dp_rank=MagicMock(return_value=0),
    )
    return q


def _admit_prefill(q, req):
    PrefillBootstrapQueue.add(q, req, 8)


def _admit_decode(q, req, **kw):
    with patch(
        "sglang.srt.disaggregation.decode._is_fake_transfer", return_value=False
    ):
        DecodePreallocQueue.add(q, req, **kw)


class TestAdmissionAbortNotEnqueued(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        # set_finish_with_abort() logs on TP rank 0, and ParallelContext.tp_rank
        # reads through to the live process group -- none exists here.
        p = patch(
            "sglang.srt.managers.schedule_batch.get_parallel",
            return_value=SimpleNamespace(tp_rank=0),
        )
        p.start()
        self.addCleanup(p.stop)

    def test_rejected_request_is_retired_at_each_pd_door(self):
        for door in ("prefill", "decode"):
            with self.subTest(door=door):
                req = _make_req()
                req.set_finish_with_abort(ERROR_MSG)
                # Precondition: the rejection is pending, so `finished()` --
                # what every admission gate keys off -- is still False.
                self.assertIsInstance(req.to_finish, FINISH_ABORT)
                self.assertFalse(req.finished())
                self.assertEqual(len(req.origin_input_ids), 1)

                sched = _make_scheduler()
                if door == "prefill":
                    q = _prefill_queue(sched)
                    _admit_prefill(q, req)
                    q.create_sender.assert_not_called()
                    self.assertEqual(q.queue, [])
                else:
                    q = _decode_queue(sched)
                    _admit_decode(q, req)
                    q._create_receiver_and_enqueue.assert_not_called()
                    self.assertEqual(q.pending_reqs, [])
                sched.retire_unadmitted_request.assert_called_once_with(req)

    def test_reentering_prefill_request_is_not_retired(self):
        """A preempted prefill requeue must reach its queue, which owns release.

        Preemption runs `release_req`, so KV is gone by then; what survives is
        the metadata buffer (`finalize_bootstrap` allocated it before the
        request ever entered the running batch) and the host retraction backup.
        """
        for marker in ("metadata_buffer_index", "retraction_backup"):
            with self.subTest(marker=marker):
                req = _make_req(prompt_len=16)
                req.to_finish = FINISH_ABORT(
                    "Aborted by AbortReq.", HTTPStatus.SERVICE_UNAVAILABLE
                )
                # release_req already freed the KV row.
                self.assertFalse(req.kv.holds_kv)
                if marker == "metadata_buffer_index":
                    req.metadata_buffer_index = 7
                else:
                    req.kv.retraction_backup = object()
                self.assertFalse(is_unadmitted_reject(req))

                sched = _make_scheduler()
                q = _prefill_queue(sched)
                _admit_prefill(q, req)
                self.assertEqual(q.queue, [req])
                sched.retire_unadmitted_request.assert_not_called()
                self.assertIsInstance(req.to_finish, FINISH_ABORT)

    def test_retracted_decode_request_is_not_retired(self):
        """The decode door cannot sniff this one, so it must trust its flags.

        By the time `retract_decode` requeues, the request carries none of the
        markers `is_unadmitted_reject` reads: `release_req` nulled
        `kv.req_pool_idx`, `reset_for_retract` nulled `kv.mamba_pool_idx`, the
        decode metadata buffer lives on `DecodeRequest` rather than `Req`, and
        `add()` itself clears `retraction_mb_id`. Retiring it here would strand
        the host pages `release_req` allocated, which only `retraction_restore`
        or `retraction_discard` free.

        `backup=None` is the short-sequence case: `retraction_backup()` returns
        early for `seqlen <= 1` without setting it, so the resource predicate
        alone says "unadmitted" and only the flag keeps the request safe.
        """
        for flag in ("is_retracted", "is_rebootstrap"):
            for backup in (object(), None):
                with self.subTest(flag=flag, has_backup=backup is not None):
                    req = _make_req(prompt_len=16)
                    req.to_finish = FINISH_ABORT(
                        "Aborted by AbortReq.", HTTPStatus.SERVICE_UNAVAILABLE
                    )
                    # Exactly what retract_decode leaves behind.
                    self.assertFalse(req.kv.holds_kv)
                    self.assertFalse(req.kv.holds_mamba)
                    self.assertEqual(req.metadata_buffer_index, -1)
                    req.kv.retraction_backup = backup

                    sched = _make_scheduler()
                    q = _decode_queue(sched)
                    _admit_decode(q, req, **{flag: True})
                    sched.retire_unadmitted_request.assert_not_called()
                    self.assertIsInstance(req.to_finish, FINISH_ABORT)
                    if flag == "is_retracted":
                        self.assertEqual(q.retracted_queue, [req])
                    else:
                        q._create_receiver_and_enqueue.assert_called_once()

    def test_valid_request_still_admitted(self):
        for door in ("prefill", "decode"):
            with self.subTest(door=door):
                req = _make_req(prompt_len=16)
                sched = _make_scheduler()
                if door == "prefill":
                    q = _prefill_queue(sched)
                    _admit_prefill(q, req)
                    self.assertEqual(q.queue, [req])
                else:
                    q = _decode_queue(sched)
                    _admit_decode(q, req)
                    q._create_receiver_and_enqueue.assert_called_once()
                sched.retire_unadmitted_request.assert_not_called()

    def test_null_mode_is_deliberately_untouched(self):
        """The bug is disaggregation-only; NULL mode must keep enqueuing.

        There the one-token stub costs a cheap forward pass and the batch
        boundary promotes `to_finish` into the 400. Adding a guard here would
        skip `StreamingSession.find_active_slot`'s pre-abort detach, which only
        runs while scheduling.
        """
        req = _make_req()
        req.set_finish_with_abort(ERROR_MSG)
        sched = SimpleNamespace(
            disaggregation_mode=DisaggregationMode.NULL,
            waiting_queue=[],
            processed_tokens_counter=0,
            _set_or_validate_priority=MagicMock(return_value=True),
            _abort_on_queued_limit=MagicMock(return_value=False),
            _prefetch_kvcache=MagicMock(),
        )
        Scheduler._add_request_to_queue(sched, req)
        self.assertEqual(sched.waiting_queue, [req])
        self.assertIsInstance(req.to_finish, FINISH_ABORT)

    def test_retire_detaches_a_streaming_session(self):
        """Otherwise the session stays in-flight forever.

        `create_req` marks it in-flight and the pre-abort detach lives in
        `StreamingSession.find_active_slot`, which a retired request never
        reaches; a stuck flag fails every later request on that session.
        """
        session = MagicMock()
        session.streaming = True
        req = _make_req(session=session)
        req.set_finish_with_abort(ERROR_MSG)

        sched = SimpleNamespace(
            output_streamer=MagicMock(), beam_coordinator=MagicMock()
        )
        Scheduler.retire_unadmitted_request(sched, req)

        session.abort_req.assert_called_once()
        self.assertIsNone(req.session)
        sched.beam_coordinator.retire_group.assert_called_once_with(req)
        req.time_stats.trace_ctx.abort.assert_called_once()
        # The original 400 survives, rather than being replaced downstream.
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertEqual(req.finished_reason.status_code, HTTPStatus.BAD_REQUEST)
        self.assertIsNone(req.to_finish)
        sched.output_streamer.stream_output.assert_called_once()

    def test_retire_leaves_a_non_streaming_session_alone(self):
        session = MagicMock()
        session.streaming = False
        req = _make_req(session=session)
        req.set_finish_with_abort(ERROR_MSG)

        sched = SimpleNamespace(
            output_streamer=MagicMock(), beam_coordinator=MagicMock()
        )
        Scheduler.retire_unadmitted_request(sched, req)

        session.abort_req.assert_not_called()
        self.assertIs(req.session, session)


if __name__ == "__main__":
    unittest.main()
