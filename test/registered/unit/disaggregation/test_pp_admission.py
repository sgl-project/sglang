import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.pp_admission import (
    PPAdmissionMessage,
    PPAdmissionState,
    PPAdmissionVerdict,
    map_authoritative_polls,
    merge_deferred_send,
    prepare_forward_message,
    publication_for_stage,
    route_aborts_to_failed,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPPAdmission(unittest.TestCase):
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

    def test_local_sender_initialization_can_be_deferred(self):
        state = PPAdmissionState(step=7)
        req = SimpleNamespace(pp_defer_body=None)

        state.defer_bootstrap(req)
        state.defer_bootstrap(req)

        self.assertEqual(req.pp_defer_body, 7)
        self.assertEqual(state.deferred_bootstrap, [req])

    def test_metadata_buffer_defer_is_reoffered_until_applied(self):
        state = PPAdmissionState()
        state.defer_verdict("deferred")
        intended = PPAdmissionVerdict(("new",), ())

        offered = intended.with_deferred(state.deferred_rids)
        self.assertEqual(offered.admitted, ("new", "deferred"))

        state.clear_applied(PPAdmissionVerdict(("deferred",), ()))
        self.assertNotIn("deferred", state.deferred_rids)

    def test_failed_verdict_overrides_deferred_admission(self):
        verdict = PPAdmissionVerdict((), ("aborted",))

        self.assertEqual(verdict.with_deferred(["aborted"]).admitted, ())

    def test_last_chunk_discards_stale_end_idx(self):
        pending = merge_deferred_send(None, last_chunk=False, end_idx=128)
        pending = merge_deferred_send(pending, last_chunk=True, end_idx=256)
        pending = merge_deferred_send(pending, last_chunk=False, end_idx=384)

        self.assertEqual(pending, (True, None))

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
