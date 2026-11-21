"""CPU coverage for bounded admission updates, including the real HTTP schema."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import msgspec
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sglang.srt import runtime_context as rc
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.io_struct import GetInternalStateReq, SetInternalStateReq
from sglang.srt.managers.scheduler_components import internal_state as module
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def make_scheduler():
    return SimpleNamespace(
        max_running_requests=64,
        max_queued_requests=8,
        max_prefill_tokens=8192,
        dynamic_chunk_sizer=None,
        disaggregation_mode=DisaggregationMode.NULL,
        enable_priority_scheduling=False,
        waiting_queue=[],
        running_batch=SimpleNamespace(reqs=[object()]),
        metrics_reporter=SimpleNamespace(
            spec_total_num_accept_tokens=7, spec_total_num_forward_ct=3
        ),
        spec_algorithm=SimpleNamespace(
            is_none=lambda: True,
            is_dflash=lambda: False,
            is_dflash_family=lambda: True,
        ),
        draft_worker=Mock(),
    )


class InternalStateTestCase(CustomTestCase):
    def setUp(self):
        super().setUp()
        rc.reset_context()
        self.addCleanup(rc.reset_context)
        self.startup = ServerArgs(model_path="dummy", max_prefill_tokens=8192)
        rc.publish(self.startup, role="scheduler")
        self.scheduler = make_scheduler()
        self.controller = module.SchedulerInternalStateController(self.scheduler, False)

    def update(self, **values):
        return self.controller.set_internal_state(
            SetInternalStateReq(server_args=values)
        ).updated


class TestTrimtabHotKnobs(InternalStateTestCase):
    def test_success_and_repeated_updates_keep_startup_ceiling(self):
        active = self.scheduler.running_batch.reqs[:]
        for budget, queue in ((1, 0), (4096, 2), (8192, 10000), (8192, 10000)):
            self.assertTrue(
                self.update(max_prefill_tokens=budget, max_queued_requests=queue)
            )
            self.assertEqual(self.scheduler.max_prefill_tokens, budget)
            self.assertEqual(self.scheduler.max_queued_requests, queue)
            self.assertEqual(rc.get_schedule().max_prefill_tokens, budget)
            self.assertEqual(rc.get_schedule().max_queued_requests, queue)
            self.assertEqual(self.scheduler.running_batch.reqs, active)
        self.assertFalse(self.update(max_prefill_tokens=8193))
        self.assertEqual(self.startup.max_prefill_tokens, 8192)

    def test_invalid_values_are_rejected_without_writes(self):
        for key in ("max_prefill_tokens", "max_queued_requests"):
            for value in (
                True,
                False,
                -1,
                1.5,
                1.0,
                "4",
                None,
                float("nan"),
                float("inf"),
                -float("inf"),
            ):
                with self.subTest(key=key, value=value):
                    before = rc.get_context().resolved_server_args_dict()
                    self.assertFalse(self.update(**{key: value}))
                    self.assertEqual(self.scheduler.max_prefill_tokens, 8192)
                    self.assertEqual(self.scheduler.max_queued_requests, 8)
                    self.assertEqual(
                        rc.get_context().resolved_server_args_dict(), before
                    )
        self.assertFalse(self.update(max_prefill_tokens=0))

    def test_atomic_validation_in_either_key_order(self):
        for values in (
            {"max_queued_requests": 2, "max_prefill_tokens": 8193},
            {"max_prefill_tokens": 8193, "max_queued_requests": 2},
            {"max_queued_requests": 2, "unknown": 1},
            {"speculative_accept_threshold_single": 0.5, "max_prefill_tokens": 0},
            {"max_queued_requests": 2, "pp_max_micro_batch_size": 1000},
            {"dspark_clear_info_records": True, "max_prefill_tokens": 0},
        ):
            with self.subTest(values=values):
                before = rc.get_context().overrides_log()
                self.assertFalse(self.update(**values))
                self.assertEqual(rc.get_context().overrides_log(), before)
                self.assertEqual(self.scheduler.max_queued_requests, 8)
                self.assertEqual(
                    self.scheduler.metrics_reporter.spec_total_num_forward_ct, 3
                )
                self.scheduler.draft_worker.clear_info_records.assert_not_called()

    def test_existing_handlers_and_empty_update(self):
        self.assertTrue(self.update())
        self.assertTrue(self.update(max_queued_requests=4, pp_max_micro_batch_size=8))
        self.assertEqual(rc.get_parallel().pp_max_micro_batch_size, 8)
        self.assertTrue(self.update(speculative_accept_threshold_single=0.5))
        self.assertEqual(rc.get_spec().speculative_accept_threshold_single, 0.5)
        self.assertEqual(self.scheduler.metrics_reporter.spec_total_num_forward_ct, 0)
        self.assertTrue(self.update(dspark_force_budget_frac=None))
        self.scheduler.draft_worker.set_dspark_forced_budget_frac.assert_called_with(
            None
        )
        self.assertTrue(self.update(dspark_clear_info_records=True))
        self.scheduler.draft_worker.clear_info_records.assert_called_once()

    def test_boolean_numeric_values_are_rejected_for_existing_keys(self):
        for key in (
            "pp_max_micro_batch_size",
            "speculative_accept_threshold_single",
            "speculative_accept_threshold_acc",
            "dspark_force_budget_frac",
        ):
            with self.subTest(key=key):
                self.assertFalse(self.update(**{key: True}))
                self.assertFalse(self.update(**{key: False}))

    def test_dynamic_chunking_and_disaggregation_restrictions(self):
        self.scheduler.dynamic_chunk_sizer = object()
        self.assertFalse(self.update(max_prefill_tokens=4096, max_queued_requests=2))
        self.assertEqual(self.scheduler.max_queued_requests, 8)
        for mode in (DisaggregationMode.PREFILL, DisaggregationMode.DECODE):
            self.scheduler.disaggregation_mode = mode
            self.assertFalse(self.update(max_queued_requests=4))

    def test_priority_zero_rejected_until_upstream_empty_queue_fix(self):
        self.scheduler.enable_priority_scheduling = True
        self.assertFalse(self.update(max_queued_requests=0))
        self.assertTrue(self.update(max_queued_requests=1))

    def test_readback_reports_effective_values(self):
        self.assertTrue(self.update(max_queued_requests=2, max_prefill_tokens=2048))
        contributors = {
            name: Mock()
            for name in vars(module.SchedulerInternalStateController)
            if name.startswith("_add_")
        }
        with patch.multiple(module.SchedulerInternalStateController, **contributors):
            state = self.controller.get_internal_state(
                GetInternalStateReq()
            ).internal_state
        self.assertEqual(state["max_prefill_tokens"], 2048)
        self.assertEqual(state["max_queued_requests"], 2)

    def test_deferred_and_other_contributors_knobs_are_not_implemented(self):
        for key, value in (
            ("max_running_requests", 4),
            ("log_level", "debug"),
            ("chunked_prefill_size", 1024),
            ("prefill_max_requests", 2),
            ("schedule_policy", "fcfs"),
            ("schedule_conservativeness", 0.5),
        ):
            with self.subTest(key=key):
                self.assertFalse(self.update(max_queued_requests=2, **{key: value}))
                self.assertEqual(self.scheduler.max_queued_requests, 8)

    def test_lowering_queue_limit_does_not_drop_existing_requests(self):
        # Use the actual admission method, with only abort delivery replaced.
        maybe_stub_sgl_kernel()
        from sglang.srt.managers.scheduler import Scheduler

        for priority in (False, True):
            for lower_first in (False, True):
                with self.subTest(priority=priority, lower_first=lower_first):
                    s = self.scheduler
                    s.enable_priority_scheduling = priority
                    s.schedule_low_priority_values_first = lower_first
                    s.waiting_queue = [
                        SimpleNamespace(
                            rid=str(i),
                            priority=i,
                            time_stats=SimpleNamespace(wait_queue_entry_time=i),
                        )
                        for i in range(3)
                    ]
                    original = s.waiting_queue[:]
                    self.assertTrue(self.update(max_queued_requests=1))
                    self.assertEqual(s.waiting_queue, original)
                    rejected = []

                    def abort(req, **kwargs):
                        rejected.append(req.rid)
                        if req in s.waiting_queue:
                            s.waiting_queue.remove(req)

                    s._abort_unavailable = abort
                    incoming = SimpleNamespace(
                        rid="incoming", priority=-1 if lower_first else 4
                    )
                    aborted = Scheduler._abort_on_queued_limit(s, incoming)
                    self.assertEqual(aborted, not priority)
                    if not aborted:
                        s.waiting_queue.append(incoming)
                    self.assertEqual(len(s.waiting_queue), 3)
                    self.assertEqual(len(rejected), 1)

    def test_update_uses_existing_control_broadcast(self):
        maybe_stub_sgl_kernel()
        from sglang.srt.managers.scheduler_components import (
            request_receiver as receiver,
        )

        # Exercise routing, replacing only the distributed collective itself.
        request = SetInternalStateReq(server_args={"max_prefill_tokens": 4096})
        instance = receiver.SchedulerRequestReceiver.__new__(
            receiver.SchedulerRequestReceiver
        )
        group = SimpleNamespace(rank=0, cpu_group=object(), ranks=[0, 1])
        parallel = SimpleNamespace(
            attn_dp_enabled=True,
            attn_tp_rank=0,
            attn_cp_rank=0,
            tp_size=2,
            tp_group=group,
        )
        for local in (False, True):
            with (
                self.subTest(local=local),
                patch.object(receiver, "get_parallel", return_value=parallel),
                patch.object(
                    receiver, "is_control_broadcast_local", return_value=local
                ),
                patch.object(
                    receiver,
                    "attn_cp_tp_broadcast_pyobj",
                    side_effect=lambda value: value,
                ) as attn,
                patch.object(
                    receiver,
                    "broadcast_pyobj",
                    side_effect=lambda value, *args, **kwargs: value,
                ) as all_ranks,
            ):
                result = instance._broadcast_reqs_across_ranks([request])
                self.assertEqual(result, [request])
                if local:
                    self.assertEqual(attn.call_args_list[-1].args, ([request],))
                    all_ranks.assert_not_called()
                else:
                    all_ranks.assert_called_once_with(
                        [request], 0, group.cpu_group, src=0
                    )

    def test_zero_queue_without_priority_rejects_arrivals(self):
        maybe_stub_sgl_kernel()
        from sglang.srt.managers.scheduler import Scheduler

        self.assertTrue(self.update(max_queued_requests=0))
        self.scheduler._abort_unavailable = Mock()
        incoming = SimpleNamespace(rid="incoming")
        self.assertTrue(Scheduler._abort_on_queued_limit(self.scheduler, incoming))
        self.scheduler._abort_unavailable.assert_called_once_with(
            incoming, message="The request queue is full."
        )


class TestInternalStateHTTP(InternalStateTestCase):
    def setUp(self):
        super().setUp()
        from sglang.srt.entrypoints import http_server
        from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin

        self.received = []
        self.controllers = [
            self.controller,
            module.SchedulerInternalStateController(make_scheduler(), False),
        ]

        async def communicate(obj):
            # Serialize the real IPC struct and deliver to two controller instances.
            obj = msgspec.msgpack.decode(
                msgspec.msgpack.encode(obj), type=SetInternalStateReq
            )
            self.received.append(obj)
            return [
                controller.set_internal_state(obj) for controller in self.controllers
            ]

        manager = SimpleNamespace(
            auto_create_handle_loop=Mock(), set_internal_state_communicator=communicate
        )

        async def update(obj):
            return await TokenizerControlMixin.set_internal_state(manager, obj)

        manager.set_internal_state = update
        self.enterContext(
            patch.object(
                http_server, "_global_state", SimpleNamespace(tokenizer_manager=manager)
            )
        )
        app = FastAPI()
        app.add_api_route(
            "/set_internal_state",
            http_server.set_internal_state,
            methods=["POST", "PUT"],
        )
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def test_http_booleans_never_become_numbers(self):
        for method in (self.client.post, self.client.put):
            for key in (
                "max_prefill_tokens",
                "max_queued_requests",
                "pp_max_micro_batch_size",
                "speculative_accept_threshold_single",
                "speculative_accept_threshold_acc",
                "dspark_force_budget_frac",
            ):
                for value in (True, False):
                    with self.subTest(method=method.__name__, key=key, value=value):
                        response = method(
                            "/set_internal_state",
                            json={
                                "server_args": {"max_queued_requests": 3, key: value}
                            },
                        )
                        self.assertEqual(response.status_code, 200)
                        self.assertEqual(response.json(), [False, False])
                        self.assertIs(self.received[-1].server_args[key], value)
                        for controller in self.controllers:
                            self.assertEqual(
                                controller.scheduler.max_queued_requests, 8
                            )

    def test_http_invalid_numeric_values_do_not_apply_other_keys(self):
        for value in ("0", "-1", "1.0", "NaN", "Infinity", "-Infinity", "8193"):
            with self.subTest(value=value):
                response = self.client.post(
                    "/set_internal_state",
                    content='{"server_args":{"max_queued_requests":3,"max_prefill_tokens":'
                    + value
                    + "}}",
                    headers={"content-type": "application/json"},
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), [False, False])
                for controller in self.controllers:
                    self.assertEqual(controller.scheduler.max_queued_requests, 8)
                    self.assertEqual(controller.scheduler.max_prefill_tokens, 8192)

    def test_http_success_reaches_each_controller_and_reports_all_replies(self):
        response = self.client.post(
            "/set_internal_state",
            json={
                "server_args": {"max_queued_requests": 4, "max_prefill_tokens": 4096}
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [True, True])
        for controller in self.controllers:
            self.assertEqual(controller.scheduler.max_queued_requests, 4)
            self.assertEqual(controller.scheduler.max_prefill_tokens, 4096)

    def test_http_rejects_numeric_strings_and_structured_values(self):
        for value in ("4", [], {}):
            response = self.client.post(
                "/set_internal_state",
                json={"server_args": {"max_queued_requests": value}},
            )
            self.assertEqual(response.status_code, 422)
        self.assertEqual(self.received, [])

    def test_http_preserves_boolean_command_and_null_reset(self):
        response = self.client.post(
            "/set_internal_state",
            json={
                "server_args": {
                    "dspark_clear_info_records": True,
                    "dspark_force_budget_frac": None,
                }
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [True, True])
        for controller in self.controllers:
            controller.scheduler.draft_worker.clear_info_records.assert_called_once()
            controller.scheduler.draft_worker.set_dspark_forced_budget_frac.assert_called_with(
                None
            )


if __name__ == "__main__":
    unittest.main()
