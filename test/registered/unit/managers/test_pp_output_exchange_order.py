"""PP output exchange must not cross-deadlock with two microbatches in flight.

Each scheduler loop iteration, every PP rank sends one microbatch's output
downstream (the last rank returns it to rank 0, earlier ranks relay it on) and
receives another microbatch's output from upstream. NCCL p2p ops between a peer
pair are serialized on one stream, and a device send only completes on its own
while the peer's connection buffer has free slots (about two ops over NET/IB, so
any cross-node PP); after that it waits for the peer's matching recv. If every
rank posts its send first, each rank's send sits ahead of the recv its peer's
send is waiting for, and both GPUs hang. Model the worst case, where every send
must rendezvous with the peer's recv at the head of its stream.
"""

import unittest
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _posted_output_ops(pp_rank, spec_relay, send_msg, recv_msg):
    """Device p2p ops one PP2 rank posts for one output exchange, in stream order."""
    ops = []
    scheduler = SchedulerPPMixin()
    scheduler._pp_spec_relay = spec_relay
    scheduler.schedule_stream = None
    scheduler.copy_stream = SimpleNamespace(
        wait_stream=lambda stream: None, wait_event=lambda event: None
    )
    scheduler.copy_stream_ctx = nullcontext()
    scheduler.device_module = SimpleNamespace(
        Event=lambda: SimpleNamespace(record=lambda stream: None),
        current_stream=lambda: None,
    )
    scheduler._pp_prep_batch_result = lambda *args: SimpleNamespace(logits_output=None)

    def send(*args):
        ops.append(("send", send_msg))
        return []

    def recv():
        ops.append(("recv", recv_msg))
        return {}, None

    scheduler._pp_send_output_to_next_stage = send
    scheduler._pp_recv_dict_from_prev_stage = recv
    target = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        reqs=[object(), object()],
        contains_last_prefill_chunk=True,
        return_logprob=True,
    )
    with published_topology(role="scheduler", ranks={"world_rank": pp_rank}, pp_size=2):
        scheduler._pp_send_recv_and_preprocess_output_tensors(
            0, 1, [None, target], [None, None], deque(), None
        )
    return ops


class TestPPOutputExchangeOrder(CustomTestCase):
    def test_two_in_flight_output_exchange_does_not_cross_deadlock(self):
        for spec_relay in (False, True):
            with self.subTest(spec_relay=spec_relay):
                # Rank 1 (last) returns output A to rank 0 while rank 0 relays
                # output B (received last iteration) on to rank 1.
                streams = [
                    deque(_posted_output_ops(0, spec_relay, "relay-B", "output-A")),
                    deque(_posted_output_ops(1, spec_relay, "output-A", "relay-B")),
                ]
                self.assertEqual([len(s) for s in streams], [2, 2])
                while streams[0] or streams[1]:
                    heads = {streams[0][0], streams[1][0]}
                    kinds = sorted(kind for kind, _ in heads)
                    self.assertTrue(
                        kinds == ["recv", "send"] and len({m for _, m in heads}) == 1,
                        f"PP output exchange deadlocks: rank0={list(streams[0])}, "
                        f"rank1={list(streams[1])}",
                    )
                    streams[0].popleft()
                    streams[1].popleft()


if __name__ == "__main__":
    unittest.main()
