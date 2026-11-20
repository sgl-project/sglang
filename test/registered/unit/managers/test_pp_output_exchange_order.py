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

import pickle
import unittest
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.layers.logits_processor import LogitsProcessorOutput  # noqa: E402
from sglang.srt.managers import scheduler_pp_mixin as pp_mod  # noqa: E402
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin  # noqa: E402
from sglang.srt.managers.utils import GenerationBatchResult  # noqa: E402
from sglang.srt.model_executor.forward_mode import ForwardMode  # noqa: E402

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
    def test_numeric_logprob_rows_keep_independent_containers(self):
        for values in ([], [1, 2], [1.25, 2.5], [1, 2.5]):
            with self.subTest(values=values):
                expected = values.copy()
                result = GenerationBatchResult(
                    next_token_ids=torch.tensor([7]),
                    logits_output=LogitsProcessorOutput(
                        next_token_logits=None,
                        input_top_logprobs_val=[[values]],
                        next_token_top_logprobs_val=[values],
                    ),
                )
                wire = SchedulerPPMixin()._pp_prepare_tensor_dict(
                    result, SimpleNamespace(return_logprob=True)
                )
                values.append(99)
                values[0] = -1
                self.assertEqual(wire["input_top_logprobs_val"][0][0], expected)
                self.assertEqual(wire["next_token_top_logprobs_val"][0], expected)

    def test_numeric_prefix_does_not_hide_a_later_tensor(self):
        for dtype in (torch.float32, torch.int64):
            with self.subTest(dtype=dtype):
                base = torch.arange(100000, dtype=dtype)
                view = base.view(10000, 10)[5:7, :2]
                expected = view.clone()
                frozen = pp_mod._pp_freeze_output_metadata([7, view])
                self.assertEqual(frozen[0], 7)
                self.assertEqual(frozen[1].shape, view.shape)
                self.assertEqual(frozen[1].dtype, dtype)
                self.assertEqual(
                    frozen[1].untyped_storage().nbytes(),
                    view.numel() * view.element_size(),
                )
                base.zero_()
                torch.testing.assert_close(frozen[1], expected)

    def test_nested_logprob_views_serialize_only_their_own_storage(self):
        """A tiny PP logprob row must not serialize or retain its large base."""
        bases = [torch.arange(100000, dtype=d) for d in (torch.float32, torch.int64)]
        views = [base.view(10000, 10)[5:6, :2] for base in bases]
        expected = [view.clone() for view in views]
        result = GenerationBatchResult(
            next_token_ids=torch.tensor([7]),
            logits_output=LogitsProcessorOutput(
                next_token_logits=None,
                input_top_logprobs_val=[[views[0]]],
                input_top_logprobs_idx=[[views[1]]],
            ),
        )
        wire = SchedulerPPMixin()._pp_prepare_tensor_dict(
            result, SimpleNamespace(return_logprob=True)
        )
        fields = ("input_top_logprobs_val", "input_top_logprobs_idx")
        for field, view in zip(fields, views):
            frozen = wire[field][0][0]
            self.assertEqual(frozen.dtype, view.dtype)
            self.assertEqual(
                frozen.untyped_storage().nbytes(),
                frozen.numel() * frozen.element_size(),
            )
            self.assertNotEqual(frozen.data_ptr(), view.data_ptr())
        for base in bases:
            base.zero_()
        payload = pickle.dumps(wire)
        self.assertLess(len(payload), 4096)
        restored = pickle.loads(payload)  # Only this test's locally generated data.
        for field, values in zip(fields, expected):
            torch.testing.assert_close(restored[field][0][0], values)

    def test_nested_metadata_preserves_container_types_and_values(self):
        """Tuple/dict metadata must not bypass the same tensor ownership rule."""
        base = torch.arange(1000)
        wire = pp_mod._pp_freeze_output_metadata(
            {"rows": [(base[500:502], None, 7, False)]}
        )
        self.assertIsInstance(wire, dict)
        self.assertIsInstance(wire["rows"], list)
        self.assertIsInstance(wire["rows"][0], tuple)
        self.assertEqual(wire["rows"][0][1:], (None, 7, False))
        base.zero_()
        self.assertEqual(wire["rows"][0][0].tolist(), [500, 501])
        self.assertEqual(wire["rows"][0][0].untyped_storage().nbytes(), 16)

    def test_top_level_output_tensor_keeps_communication_fast_path(self):
        """The metadata fix must not clone top-level communication tensors."""
        ids = torch.tensor([7])
        result = GenerationBatchResult(
            next_token_ids=ids,
            logits_output=LogitsProcessorOutput(next_token_logits=None),
        )
        wire = SchedulerPPMixin()._pp_prepare_tensor_dict(
            result, SimpleNamespace(return_logprob=False)
        )
        self.assertEqual(list(wire), ["next_token_ids"])
        self.assertIs(wire["next_token_ids"], ids)

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


class TestPPOutputSkipWhileCapturingRoutedExperts(CustomTestCase):
    def test_middle_prefill_chunks_keep_their_output_message(self):
        """The output message carries a chunk's routed experts to the first
        stage; skipping it returns zeros for that chunk's tokens."""
        middle_chunk = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            reqs=[object()],
            contains_last_prefill_chunk=False,
            return_logprob=False,
        )
        with envs.SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM.override(True):
            for capturing in (False, True):
                features = SimpleNamespace(enable_return_routed_experts=capturing)
                with (
                    self.subTest(capturing=capturing),
                    mock.patch.object(
                        pp_mod,
                        "get_exec",
                        return_value=SimpleNamespace(features=features),
                    ),
                ):
                    self.assertEqual(
                        pp_mod._pp_can_skip_output_comm(middle_chunk), not capturing
                    )


if __name__ == "__main__":
    unittest.main()
