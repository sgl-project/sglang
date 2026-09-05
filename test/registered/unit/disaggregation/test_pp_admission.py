import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation import prefill as prefill_module
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
from sglang.srt.disaggregation.pp_admission import (
    PPAdmissionMessage,
    PPAdmissionState,
    PPAdmissionVerdict,
    map_authoritative_polls,
    prepare_forward_message,
    publication_for_stage,
    route_aborts_to_failed,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPPAdmission(unittest.TestCase):
    def test_commit_rejects_request_without_metadata_reservation(self):
        req = SimpleNamespace(
            rid="request",
            metadata_buffer_index=-1,
            time_stats=SimpleNamespace(set_wait_queue_entry_time=lambda: None),
        )
        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue.queue = [req]
        queue.pp_size = 8
        queue.pp_rank = 3

        with patch.object(prefill_module, "_PP_ADMIT_FLOW", True):
            with self.assertRaisesRegex(
                AssertionError, "without a metadata reservation"
            ):
                queue.pop_bootstrapped(
                    return_failed_reqs=True,
                    pp_good_rids=["request"],
                    pp_bad_rids=[],
                )

    def test_prepare_reserves_capacity_and_never_overcommits(self):
        reqs = [
            SimpleNamespace(
                rid="a", metadata_buffer_index=-1, disagg_kv_sender=object()
            ),
            SimpleNamespace(
                rid="b", metadata_buffer_index=-1, disagg_kv_sender=object()
            ),
        ]
        slots = [7]
        allocator = SimpleNamespace(
            available_size=lambda: len(slots),
            alloc=lambda: slots.pop(),
        )
        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue.queue = reqs
        queue.scheduler = SimpleNamespace(
            attn_cp_cpu_group=object(), attn_tp_cpu_group=object()
        )
        queue.req_to_metadata_buffer_idx_allocator = allocator

        with patch.object(
            prefill_module,
            "poll_and_all_reduce_attn_cp_tp_group",
            return_value=[KVPoll.WaitingForInput, KVPoll.WaitingForInput],
        ):
            prepared, failed = queue.prepare_bootstrapped_rids()

        self.assertEqual(prepared, ["a"])
        self.assertEqual(failed, [])
        self.assertEqual(reqs[0].metadata_buffer_index, 7)
        self.assertEqual(reqs[1].metadata_buffer_index, -1)

    def test_first_stage_verdict_is_authoritative(self):
        verdict = PPAdmissionVerdict(("ready",), ("failed",))

        self.assertEqual(
            map_authoritative_polls(
                ["ready", "local-only", "failed"],
                verdict,
                admitted_poll="admitted",
                failed_poll="failed",
            ),
            ["admitted", None, "failed"],
        )

    def test_first_stage_publishes_applied_verdict(self):
        intended = PPAdmissionVerdict(("a", "b"), ())
        applied = PPAdmissionVerdict(("a",), ())

        self.assertEqual(publication_for_stage(True, intended, applied), applied)
        self.assertEqual(publication_for_stage(False, intended, applied), intended)

    def test_local_failure_report_reaches_uniform_outcome(self):
        verdict = PPAdmissionVerdict(("request",), ())
        message = PPAdmissionMessage(verdict)

        message = prepare_forward_message(message, verdict, ())
        message = prepare_forward_message(message, verdict, ("request",))
        message = prepare_forward_message(message, verdict, ())

        self.assertEqual(message.verdict, verdict)
        self.assertEqual(message.local_failures, ("request",))
        for state in (PPAdmissionState(), PPAdmissionState(), PPAdmissionState()):
            self.assertEqual(
                state.consume_uniform_failures(message.local_failures), ["request"]
            )
            self.assertEqual(state.consume_uniform_failures(message.local_failures), [])

    def test_forward_payload_preserves_admission_and_report_order(self):
        verdict = PPAdmissionVerdict(("a", "b"), ("c",))
        message = PPAdmissionMessage(verdict, ("old",))

        forwarded = prepare_forward_message(message, verdict, ("new", "old"))

        self.assertEqual(forwarded.verdict, verdict)
        self.assertEqual(forwarded.local_failures, ("old", "new"))
        self.assertEqual(forwarded.to_payload(), [["a", "b"], ["c"], ["old", "new"]])

    def test_abort_routing_overrides_admission(self):
        good, bad = route_aborts_to_failed(["keep", "abort"], ["failed"], ["abort"])

        self.assertEqual(good, ["keep"])
        self.assertEqual(bad, ["failed", "abort"])

    def test_malformed_payload_is_rejected(self):
        with self.assertRaises(ValueError):
            PPAdmissionVerdict.from_payload([["ready"]])
        with self.assertRaises(TypeError):
            PPAdmissionVerdict.from_payload([["ready"], None])
        with self.assertRaises(ValueError):
            PPAdmissionMessage.from_payload([[], [], [], []])
        with self.assertRaises(TypeError):
            PPAdmissionMessage.from_payload([[], [], None])

    def test_pp_admission_defaults_off(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(envs.SGLANG_PP_PD_ADMIT_FLOW.get())


if __name__ == "__main__":
    unittest.main()
