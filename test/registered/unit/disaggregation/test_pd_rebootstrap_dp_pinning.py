"""A PD true-retraction rebootstrap must go back to the prefill DP rank the
decode side resolved for the request.

When neither the client nor the router supplies ``disagg_prefill_dp_rank`` and
the prefill balances DP ranks by load, decode learns the rank from the bootstrap
server. If that rank is not kept on the request, the recompute ``/generate``
carries ``routed_dp_rank: null`` and the prefill load balancer may hand it to a
different DP rank than the one the decode receiver registered with: the
recomputed KV is offered by one rank while the receiver waits on another, and
the request only ends on the KV waiting timeout.
"""

import json
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeRequest
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

BOOTSTRAP_HOST = "prefill"
BOOTSTRAP_PORT = 8998


def _req():
    """A request that arrived without a client or router supplied rank."""
    req = Req(
        rid="r0",
        origin_input_text="hi",
        origin_input_ids=array("q", [1, 2]),
        sampling_params=SamplingParams(max_new_tokens=4),
        bootstrap_host=BOOTSTRAP_HOST,
        bootstrap_port=BOOTSTRAP_PORT,
        bootstrap_room=7,
    )
    req.output_ids = [3, 4]
    return req


def _queue():
    """A decode queue facing a prefill whose DP rank cannot be derived locally."""
    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue.pending_reqs = []
    queue.kv_manager = SimpleNamespace(
        prefill_info_table={
            f"{BOOTSTRAP_HOST}:{BOOTSTRAP_PORT}": SimpleNamespace(
                dp_size=4, follow_bootstrap_room=False
            )
        }
    )
    queue._check_if_req_exceed_kv_capacity = Mock(return_value=False)
    queue.created = []

    def create_receiver_and_enqueue(req, is_rebootstrap=False):
        entry = DecodeRequest(
            req=req,
            kv_receiver=Mock(conclude_state=None),
            is_rebootstrap=is_rebootstrap,
        )
        queue.created.append(entry)
        return entry

    queue._create_receiver_and_enqueue = create_receiver_and_enqueue
    return queue


class TestRebootstrapPrefillDpRankPinning(CustomTestCase):
    def setUp(self):
        # Early allocation is the default decode allocation policy.
        publish(
            ServerArgs(
                model_path="dummy",
                disaggregation_mode="decode",
                disaggregation_decode_allocation_policy="early",
            ),
            role="test",
        )
        self.addCleanup(reset_context)

    def test_recompute_payload_carries_the_resolved_rank(self):
        queue = _queue()
        req = _req()
        entry = DecodeRequest(req=req, kv_receiver=Mock(conclude_state=None))

        # What the pending-queue slow path does once the bootstrap server answers.
        queue._init_receiver(entry, 3)

        payload = req.build_rebootstrap_payload()
        # routed_dp_rank routes the recompute on the prefill DP controller.
        self.assertEqual(payload["routed_dp_rank"], 3)
        # disagg_prefill_dp_rank keeps the prefill sender from re-registering
        # the room under whichever rank happens to run it.
        self.assertEqual(payload["disagg_prefill_dp_rank"], 3)
        json.dumps(payload)

    def test_rebootstrap_rebinds_to_the_resolved_rank_without_a_new_query(self):
        queue = _queue()
        req = _req()

        queue.add(req)
        (first,) = queue.pending_reqs
        first.kv_receiver.init.assert_not_called()
        queue.pending_reqs.clear()
        queue._init_receiver(first, 3)

        queue.add(req, is_rebootstrap=True)

        # The bootstrap server's room -> rank entry may have expired by now, so
        # the rebootstrap must not wait in the pending queue to query it again.
        self.assertEqual(queue.pending_reqs, [])
        _, second = queue.created
        self.assertTrue(second.is_rebootstrap)
        second.kv_receiver.init.assert_called_once_with(3)

    def test_unresolved_request_pins_nothing(self):
        queue = _queue()
        req = _req()

        queue.add(req)

        self.assertEqual(len(queue.pending_reqs), 1)
        payload = req.build_rebootstrap_payload()
        self.assertIsNone(payload["routed_dp_rank"])
        self.assertIsNone(payload["disagg_prefill_dp_rank"])


if __name__ == "__main__":
    unittest.main()
