"""A PD true-retraction rebootstrap must be pinned to the prefill DP rank the
decode KV receiver is already bootstrapped against.

`KVReceiver.init(rank)` fetches the bootstrap infos of -- and registers this
decode rank's kv_args with -- exactly that prefill DP rank's TP ranks, and
nothing re-resolves it afterwards. If the recompute `/generate` does not carry
`routed_dp_rank`, the prefill DP controller round-robins it to an arbitrary DP:
the recomputed KV is then offered by one DP while the receiver waits on another
and both sides only unblock on the KV waiting timeout.
"""

import json
import unittest
from array import array
from types import SimpleNamespace

import numpy as np

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _sampling_params():
    return SimpleNamespace(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        ignore_eos=False,
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
        no_stop_trim=False,
    )


def _payload_req(resolved_dp_rank=None, disagg_prefill_dp_rank=None):
    """Carries exactly the attributes build_rebootstrap_payload reads."""
    return SimpleNamespace(
        rid="rid-0",
        origin_input_ids=np.array([1, 2], dtype=np.int32),
        output_ids=[np.int32(3), np.int32(4)],
        sampling_params=_sampling_params(),
        bootstrap_host="127.0.0.1",
        bootstrap_port=30000,
        bootstrap_room=7,
        priority=10,
        extra_key=None,
        cache_salt=None,
        routing_key=None,
        disagg_prefill_dp_rank=disagg_prefill_dp_rank,
        pd_resolved_prefill_dp_rank=resolved_dp_rank,
    )


def _queue_req(bootstrap_host="127.0.0.1"):
    return SimpleNamespace(
        rid="rid-0",
        bootstrap_host=bootstrap_host,
        bootstrap_port=30000,
        bootstrap_room=7,
        to_finish=None,
        finished_reason=None,
        pd_resolved_prefill_dp_rank=None,
    )


def _prealloc_queue(resolved_rank=None):
    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue.pending_reqs = []
    queue.inited = []

    def _create_receiver_and_enqueue(req, is_rebootstrap=False):
        return SimpleNamespace(
            req=req,
            is_rebootstrap=is_rebootstrap,
            kv_receiver=SimpleNamespace(init=queue.inited.append),
        )

    queue._create_receiver_and_enqueue = _create_receiver_and_enqueue
    queue._check_if_req_exceed_kv_capacity = lambda req: False
    queue._resolve_prefill_dp_rank = lambda req: resolved_rank
    queue.token_to_kv_pool_allocator = None
    return queue


class TestRebootstrapPayloadDpPinning(CustomTestCase):
    def test_payload_pins_routed_dp_rank_to_the_bound_prefill_dp(self):
        payload = Req.build_rebootstrap_payload(_payload_req(resolved_dp_rank=3))

        # routed_dp_rank is what DataParallelController.maybe_external_dp_rank_routing
        # reads; without it the recompute is round-robined to an arbitrary DP.
        self.assertEqual(payload["routed_dp_rank"], 3)
        # disagg_prefill_dp_rank must agree, or the prefill KV sender would
        # re-register a different dp_rank for this bootstrap_room.
        self.assertEqual(payload["disagg_prefill_dp_rank"], 3)
        json.dumps(payload)

    def test_bound_rank_wins_over_the_client_hint(self):
        # The bound rank is the one the receiver is actually connected to; a
        # stale client/router-supplied hint must not override it.
        payload = Req.build_rebootstrap_payload(
            _payload_req(resolved_dp_rank=2, disagg_prefill_dp_rank=0)
        )

        self.assertEqual(payload["routed_dp_rank"], 2)
        self.assertEqual(payload["disagg_prefill_dp_rank"], 2)

    def test_client_hint_is_used_when_no_rank_was_resolved(self):
        payload = Req.build_rebootstrap_payload(_payload_req(disagg_prefill_dp_rank=1))

        self.assertEqual(payload["routed_dp_rank"], 1)
        self.assertEqual(payload["disagg_prefill_dp_rank"], 1)

    def test_nothing_is_pinned_when_no_rank_is_known(self):
        # Keep the pre-pinning behaviour rather than sending a rank the prefill
        # DP controller would reject.
        payload = Req.build_rebootstrap_payload(_payload_req())

        self.assertIsNone(payload["routed_dp_rank"])
        self.assertIsNone(payload["disagg_prefill_dp_rank"])
        json.dumps(payload)

    def test_numpy_rank_is_coerced_to_plain_int_for_json(self):
        # A rank resolved through query_prefill_dp_ranks can be a numpy scalar,
        # which json.dumps refuses.
        payload = Req.build_rebootstrap_payload(
            _payload_req(resolved_dp_rank=np.int64(3))
        )

        self.assertIs(type(payload["routed_dp_rank"]), int)
        self.assertIs(type(payload["disagg_prefill_dp_rank"]), int)
        self.assertEqual(payload["routed_dp_rank"], 3)
        json.dumps(payload)

    def test_real_req_defaults_to_no_bound_rank(self):
        # The attribute must exist on every Req: build_rebootstrap_payload reads
        # it directly, not through getattr.
        req = Req(
            rid="r0",
            origin_input_text="hi",
            origin_input_ids=array("q", [1, 2]),
            sampling_params=SamplingParams(max_new_tokens=4),
        )
        self.assertIsNone(req.pd_resolved_prefill_dp_rank)


class TestInitKvReceiverRecordsRank(CustomTestCase):
    """`KVReceiver.init(rank)` is the only place the prefill DP rank is known,
    so the prealloc queue has to copy it onto the Req for the payload builder."""

    def setUp(self):
        publish(
            ServerArgs(model_path="dummy", disaggregation_mode="decode"), role="test"
        )
        self.addCleanup(reset_context)

    def test_init_records_the_rank_and_still_inits_the_receiver(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        inited = []
        decode_req = SimpleNamespace(
            req=SimpleNamespace(pd_resolved_prefill_dp_rank=None),
            kv_receiver=SimpleNamespace(init=inited.append),
        )

        queue._init_kv_receiver(decode_req, np.int64(2))

        self.assertEqual(decode_req.req.pd_resolved_prefill_dp_rank, 2)
        self.assertIs(type(decode_req.req.pd_resolved_prefill_dp_rank), int)
        self.assertEqual(inited, [np.int64(2)])

    def test_fast_path_add_records_the_resolved_rank(self):
        queue = _prealloc_queue(resolved_rank=3)
        req = _queue_req()

        queue.add(req)

        self.assertEqual(queue.inited, [3])
        self.assertEqual(req.pd_resolved_prefill_dp_rank, 3)
        self.assertEqual(queue.pending_reqs, [])

    def test_fake_transfer_is_pinned_to_the_rank_it_binds_to(self):
        queue = _prealloc_queue(resolved_rank=None)
        req = _queue_req(bootstrap_host=FAKE_BOOTSTRAP_HOST)

        queue.add(req)

        self.assertEqual(queue.inited, [0])
        self.assertEqual(req.pd_resolved_prefill_dp_rank, 0)

    def test_unresolved_request_keeps_no_rank_until_the_slow_path_runs(self):
        queue = _prealloc_queue(resolved_rank=None)
        req = _queue_req()

        queue.add(req)

        self.assertEqual(queue.inited, [])
        self.assertIsNone(req.pd_resolved_prefill_dp_rank)
        self.assertEqual(len(queue.pending_reqs), 1)


if __name__ == "__main__":
    unittest.main()
