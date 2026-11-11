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
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.scheduler_pp_mixin import (  # noqa: E402
    SchedulerPPMixin,
    _pp_exchange_outputs_before_forward,
)
from sglang.srt.model_executor.forward_mode import ForwardMode  # noqa: E402

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _posted_output_ops(
    pp_rank,
    spec_relay,
    send_msg,
    recv_msg,
    *,
    mode=ForwardMode.EXTEND,
    pd_spec=False,
    has_output=True,
    npu=False,
    **server_args,
):
    """Device p2p ops one PP2 rank posts for one output exchange, in stream order.

    Fences on this rank's own forward show up as ("wait", "own-forward").
    """
    ops = []
    scheduler = SchedulerPPMixin()
    scheduler._pp_spec_relay = spec_relay
    scheduler._pp_spec_decode = pd_spec
    scheduler.pp_group = SimpleNamespace(
        is_first_rank=pp_rank == 0, is_last_rank=pp_rank == 1
    )
    scheduler.attn_tp_group = None
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

    def send(tensor_dict, **kwargs):
        ops.append(("send", tensor_dict["payload"]))
        return []

    def recv():
        ops.append(("recv", recv_msg))
        return {"payload": recv_msg}, None

    def send_recv(*, send_tensor_dict, **kwargs):
        ops.append(("send_recv", (send_tensor_dict["payload"], recv_msg)))
        return {"payload": recv_msg}

    scheduler._pp_send_dict_to_next_stage = send
    scheduler._pp_recv_dict_from_prev_stage = recv
    scheduler.pp_group.send_recv_tensor_dict = send_recv
    target = (
        SimpleNamespace(
            forward_mode=mode,
            reqs=[object(), object()],
            contains_last_prefill_chunk=True,
            return_logprob=True,
        )
        if mode is not None
        else None
    )
    payload = SimpleNamespace(tensors={"payload": send_msg})
    tail_outputs = deque([(None, payload)]) if target is not None else deque()
    with (
        patch("sglang.srt.managers.scheduler_pp_mixin._is_npu", npu),
        published_topology(
            role="scheduler", ranks={"world_rank": pp_rank}, pp_size=2, **server_args
        ),
    ):
        scheduler._pp_send_recv_and_preprocess_output_tensors(
            0,
            1,
            [target, target],
            [None, None],
            tail_outputs,
            payload if has_output else None,
        )
    return ops


class TestPPOutputExchangeOrder(CustomTestCase):
    def test_other_modes_do_not_initialize_pd_spec_resources(self):
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        for gate, pp_size, algorithm, mode, expected_relay in (
            (True, 2, SpeculativeAlgorithm.NONE, "decode", False),
            (True, 1, SpeculativeAlgorithm.EAGLE, "decode", False),
            (True, 2, SpeculativeAlgorithm.EAGLE, "null", True),
            (False, 2, SpeculativeAlgorithm.EAGLE, "decode", False),
        ):
            with self.subTest(
                gate=gate, pp_size=pp_size, algorithm=algorithm, mode=mode
            ):
                scheduler = SchedulerPPMixin()
                scheduler.spec_algorithm = algorithm
                # Stage-capacity initialization always inspects the allocator.
                # No draft worker or retirer: touching PD-only resources in
                # any of these modes must fail this test.
                scheduler.token_to_kv_pool_allocator = object()
                with (
                    envs.SGLANG_ENABLE_PP_SPEC.override(gate),
                    envs.SGLANG_PP_COMM_OVERLAP.override(False),
                    patch(
                        "sglang.srt.managers.scheduler_pp_mixin.get_parallel",
                        return_value=SimpleNamespace(
                            pp_size=pp_size, pp_async_batch_depth=0
                        ),
                    ),
                    patch(
                        "sglang.srt.managers.scheduler_pp_mixin.get_disagg",
                        return_value=SimpleNamespace(disaggregation_mode=mode),
                    ),
                ):
                    scheduler.init_pp_loop_state()
                self.assertFalse(scheduler._pp_spec_decode)
                self.assertEqual(scheduler._pp_spec_relay, expected_relay)
                self.assertEqual(len(scheduler.running_mbs), pp_size)
                self.assertTrue(
                    all(batch.reqs == [] for batch in scheduler.running_mbs)
                )

    def test_prebuilt_proposals_participate_in_output_ring(self):
        """A bootstrap proposal must be sent and received on every rank;
        ordinary prebuilt batches still have no output-ring traffic."""
        cases = (
            (False, ForwardMode.PREBUILT, False),
            (True, ForwardMode.PREBUILT, True),
            (False, ForwardMode.DECODE, True),
            (True, ForwardMode.DECODE, True),
            (False, ForwardMode.EXTEND, True),
            (True, ForwardMode.EXTEND, True),
            (False, None, False),
            (True, None, False),
        )
        for pd_spec, mode, has_output in cases:
            for pp_rank in (0, 1):
                for npu in (False, True):
                    with self.subTest(
                        pd_spec=pd_spec, mode=mode, rank=pp_rank, npu=npu
                    ):
                        ops = _posted_output_ops(
                            pp_rank,
                            pd_spec,
                            "proposal-out",
                            "proposal-in",
                            mode=mode,
                            pd_spec=pd_spec,
                            has_output=has_output,
                            npu=npu,
                        )
                        if not has_output:
                            self.assertEqual(ops, [])
                        elif npu:
                            self.assertEqual(
                                [op for op in ops if op[0] != "wait"],
                                [("send_recv", ("proposal-out", "proposal-in"))],
                            )
                        else:
                            expected = [
                                ("send", "proposal-out"),
                                ("recv", "proposal-in"),
                            ]
                            self.assertEqual(
                                ops, expected if pp_rank == 0 else expected[::-1]
                            )

    def test_bootstrap_launches_first_but_verify_waits_for_proposal(self):
        for depth in (0, 1):
            for spec_relay in (False, True):
                for last_rank in (False, True):
                    for mode in (ForwardMode.PREBUILT, ForwardMode.DECODE):
                        with self.subTest(
                            depth=depth, spec=spec_relay, last=last_rank, mode=mode
                        ):
                            actual = _pp_exchange_outputs_before_forward(
                                SimpleNamespace(
                                    forward_mode=mode, is_extend_in_batch=False
                                ),
                                spec_relay,
                                last_rank,
                                depth,
                            )
                            self.assertEqual(
                                actual,
                                depth > 0
                                or (
                                    spec_relay
                                    and not last_rank
                                    and mode == ForwardMode.DECODE
                                ),
                            )

    def test_bootstrap_payload_contains_only_the_proposal(self):
        scheduler = SchedulerPPMixin()
        scheduler._pp_spec_decode = scheduler._pp_spec_relay = True
        result = SimpleNamespace(
            accept_lens=None,
            next_verify_chain=object(),
            next_verify_parent_list=object(),
            next_verify_top_scores_index=object(),
        )
        payload = scheduler._pp_prepare_tensor_dict(
            result, SimpleNamespace(forward_mode=ForwardMode.PREBUILT)
        )
        self.assertEqual(
            payload,
            {
                "spec_next_chain": result.next_verify_chain,
                "spec_next_parents": result.next_verify_parent_list,
                "spec_next_top_scores": result.next_verify_top_scores_index,
            },
        )

    def test_bootstrap_result_retires_its_forward(self):
        """A proposal-only result retires the local allocation and request
        launch without a generated token or target verification result."""
        import torch

        from sglang.srt.managers.schedule_batch import ScheduleBatch
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.managers.scheduler_components.request_list import (
            KeyedReqList,
            ListKind,
        )
        from sglang.srt.managers.scheduler_pp_mixin import PPBatchMetadata
        from sglang.srt.mem_cache.kv.alloc.forward_steps import ForwardSteps
        from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

        scheduler = SchedulerPPMixin()
        scheduler._pp_spec_decode = True
        req = SimpleNamespace(rid="bootstrap", inflight_steps=0)
        batch = ScheduleBatch(reqs=[req], forward_mode=ForwardMode.PREBUILT)
        batch.add_inflight_step()
        steps = ForwardSteps([])
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(forward_steps=steps)
        scheduler.running_reqs = KeyedReqList(ListKind.DECODING)
        scheduler.prefill_launched_reqs = KeyedReqList(ListKind.PREFILL_LAUNCHED)
        scheduler.invariant_checker = SimpleNamespace(run=lambda *args: None)
        launch = steps.forward_launched()
        forward_done = object()
        steps.forward_recorded(forward_done)
        metadata = PPBatchMetadata(
            can_run_cuda_graph=False,
            fwd_batch=batch,
            forward_launch=launch,
            unified_forward_done=forward_done,
        )
        proposal = torch.tensor([[1, 2, 3]])
        with patch(
            "sglang.srt.managers.scheduler_pp_mixin.get_spec",
            return_value=SimpleNamespace(speculative_num_draft_tokens=3),
        ):
            result = scheduler._pp_prep_batch_result(
                batch, metadata, PPProxyTensors({"spec_next_chain": proposal})
            )

        self.assertTrue(steps.in_flight())
        self.assertEqual(req.inflight_steps, 1)
        Scheduler._on_step_processed(scheduler, batch, result)
        self.assertFalse(steps.in_flight())
        self.assertEqual(req.inflight_steps, 0)
        self.assertIs(result.unified_forward_done, forward_done)
        self.assertIsNone(result.next_token_ids)
        torch.testing.assert_close(batch.spec_info.tokens, proposal)

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
