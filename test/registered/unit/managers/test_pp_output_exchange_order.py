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
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.scheduler_pp_mixin import (  # noqa: E402
    PPBatchMetadata,
    SchedulerPPMixin,
)
from sglang.srt.model_executor.forward_mode import ForwardMode  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _posted_output_ops(pp_rank, spec_relay, send_msg, recv_msg, **server_args):
    """Device p2p ops one PP2 rank posts for one output exchange, in stream order.

    Fences on this rank's own forward show up as ("wait", "own-forward").
    """
    ops = []
    scheduler = SchedulerPPMixin()
    scheduler._pp_spec_relay = spec_relay
    scheduler.pp_group = SimpleNamespace(is_first_rank=pp_rank == 0)
    scheduler.launch_event = "own-forward"
    scheduler.pp_comm_stream_ctx = nullcontext()
    scheduler.schedule_stream = None
    scheduler.copy_stream = SimpleNamespace(
        wait_stream=lambda stream: None, wait_event=lambda event: None
    )
    scheduler.copy_stream_ctx = nullcontext()
    scheduler.device_module = SimpleNamespace(
        Event=lambda: SimpleNamespace(record=lambda stream: None),
        current_stream=lambda: SimpleNamespace(
            wait_event=lambda event: ops.append(("wait", event))
        ),
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
    with published_topology(
        role="scheduler", ranks={"world_rank": pp_rank}, pp_size=2, **server_args
    ):
        scheduler._pp_send_recv_and_preprocess_output_tensors(
            0, 1, [None, target], [None, None], deque(), None
        )
    return ops


class TestPPOutputExchangeOrder(CustomTestCase):
    def test_decode_sends_current_activation_before_waiting_for_cpu_results(self):
        # Stop the real loop at the next slot's D2H wait: the downstream
        # stage must already have the current activation and control messages.
        batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
        scheduler = MagicMock(spec=SchedulerPPMixin)
        scheduler.pp_loop_size = 2
        scheduler.pp_group = SimpleNamespace(is_last_rank=False)
        scheduler.mbs = [None, batch]
        scheduler.running_mbs = [batch, batch]
        scheduler.last_mbs = [None, None]
        scheduler.mb_metadata = [None, None]
        scheduler.last_rank_comm_queue = deque()
        scheduler.send_req_work = []
        scheduler.increment_scheduler_iterations = MagicMock()
        scheduler.ingest_requests = MagicMock(return_value=["request"])
        scheduler.get_next_disagg_decode_batch_to_run = MagicMock(
            return_value=SimpleNamespace(running_batch=batch, batch_to_run=batch)
        )
        scheduler._pp_pd_get_retract_ids.return_value = ["retract"]
        scheduler._pp_pd_get_prealloc_ids.return_value = [["prealloc"], []]
        scheduler._pp_pd_get_decode_transferred_ids.return_value = {"transfer": False}
        result = object()
        scheduler._pp_launch_batch.return_value = (result, None)
        done = MagicMock()
        done.synchronize.side_effect = EOFError("next-slot D2H wait")
        scheduler._pp_commit_send_output_work_and_preprocess_output_tensors.return_value = (
            None,
            None,
            done,
        )
        scheduler._pp_pd_send_consensus_bootstrapped_ids.return_value = ([], None)
        scheduler._pp_pd_send_consensus_release_ids.return_value = ([], None)
        sent = []
        scheduler._pp_send_pyobj_to_next_stage.side_effect = lambda data, **kwargs: (
            sent.append(data) or []
        )
        scheduler._pp_send_proxy_to_next_stage.side_effect = sent.append

        with (
            published_topology(role="scheduler", pp_size=2, pp_async_batch_depth=0),
            self.assertRaisesRegex(EOFError, "next-slot D2H wait"),
        ):
            SchedulerPPMixin.event_loop_pp_disagg_decode(scheduler)

        self.assertEqual(
            sent,
            [
                ["request"],
                ["retract"],
                [["prealloc"], []],
                {"transfer": False},
                result,
            ],
        )
        scheduler._pp_commit_proxy_send_work.assert_called_once_with()
        scheduler._pp_process_batch_result.assert_not_called()

    def test_decode_readback_preserves_device_relay_and_future_tokens(self):
        # Exercise real result preparation and copy_to_cpu. Only transport,
        # stream/event and copy primitives are replaced in this CPU regression.
        for mode, spec, should_copy in (
            (ForwardMode.DECODE, SpeculativeAlgorithm.NONE, True),
            (ForwardMode.EXTEND, SpeculativeAlgorithm.NONE, False),
            (ForwardMode.DECODE, SpeculativeAlgorithm.EAGLE, False),
        ):
            with self.subTest(mode=mode, spec=spec):
                tokens = torch.tensor([7, 11], dtype=torch.int64)
                scheduler = SchedulerPPMixin()
                scheduler.pp_group = SimpleNamespace(is_first_rank=False)
                scheduler.launch_event = None
                scheduler.schedule_stream = None
                scheduler.copy_stream = MagicMock()
                scheduler.copy_stream_ctx = nullcontext()
                scheduler.device_module = MagicMock()
                scheduler.future_map = MagicMock()
                recv_done = object()
                scheduler._pp_recv_dict_from_prev_stage = lambda: (
                    {"next_token_ids": tokens},
                    recv_done,
                )
                scheduler._pp_send_output_to_next_stage = lambda *args: []
                batch = SimpleNamespace(
                    forward_mode=mode,
                    spec_algorithm=spec,
                    return_logprob=False,
                    reqs=[object(), object()],
                    req_pool_indices=torch.tensor([0, 1]),
                )
                with (
                    published_topology(
                        role="scheduler", ranks={"world_rank": 1}, pp_size=2
                    ),
                    patch(
                        "sglang.srt.managers.utils._async_d2h",
                        side_effect=lambda tensor: tensor.clone(),
                    ) as copy,
                ):
                    relay, result, done, _ = (
                        scheduler._pp_send_recv_and_preprocess_output_tensors(
                            0,
                            1,
                            [None, batch],
                            [None, PPBatchMetadata(True)],
                            deque(),
                            None,
                        )
                    )

                self.assertEqual(copy.call_count, int(should_copy))
                self.assertEqual(result.next_token_ids.tolist(), [7, 11])
                self.assertIs(result.copy_done, done if should_copy else None)
                self.assertIs(relay["next_token_ids"], tokens)
                self.assertIs(
                    scheduler.future_map.stash.call_args.args[1].bonus_tokens, tokens
                )
                scheduler.copy_stream.wait_event.assert_called_once_with(recv_done)
                if should_copy:
                    self.assertIsNot(result.next_token_ids, tokens)
                    done.record.assert_called_once_with()
                else:
                    self.assertIs(result.next_token_ids, tokens)

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

    def test_only_first_rank_fences_output_recv_on_its_forward(self):
        """With the fence on, rank 0 must not post its output recv before its own
        forward finishes: an early recv holds an SM through the forward and hangs
        flashinfer MLA."""
        with envs.SGLANG_ENABLE_PP_OUTPUT_RECV_AFTER_FORWARD.override(True):
            self.assertEqual(
                _posted_output_ops(0, False, "relay", "output"),
                [("send", "relay"), ("wait", "own-forward"), ("recv", "output")],
            )
            # Later ranks receive relays that already exist, the spec relay
            # exchanges before its forward, and depth > 0 receives before launch.
            for pp_rank, spec_relay, server_args in (
                (1, False, {}),
                (0, True, {}),
                (0, False, {"pp_async_batch_depth": 1}),
            ):
                with self.subTest(
                    pp_rank=pp_rank, spec_relay=spec_relay, **server_args
                ):
                    ops = _posted_output_ops(
                        pp_rank, spec_relay, "relay", "output", **server_args
                    )
                    self.assertNotIn(("wait", "own-forward"), ops)


if __name__ == "__main__":
    unittest.main()
