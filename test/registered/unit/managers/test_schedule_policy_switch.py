"""CPU checks for runtime policy replacement and its scheduler bindings."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import msgspec
from pydantic import TypeAdapter, ValidationError

from sglang.srt.managers.io_struct import (
    SchedulePolicySwitchReqInput,
    SchedulePolicySwitchReqOutput,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components import schedule_policy_switch
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    ListKind,
    move,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.srt.runtime_context import get_context, get_schedule
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class FakeTreeCache:
    disable = False

    def supports_fast_match_prefix(self):
        return False

    def supports_prefix_sharing(self):
        return False


class TestSchedulePolicySwitch(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)
        # Exercise the real factory and handler without initializing model workers.
        self.scheduler = Scheduler.__new__(Scheduler)
        self.scheduler.tree_cache = FakeTreeCache()
        self.scheduler.enable_hierarchical_cache = False
        self.scheduler.enable_priority_scheduling = False
        self.scheduler.schedule_low_priority_values_first = False
        self.scheduler.dllm_config = None
        self.scheduler.chunked_prefill_size = 4096
        self.scheduler.forward_ct = 12
        self.scheduler.waiting_queue = IndexedReqList(ListKind.WAITING)
        self.scheduler.request_retirer = SimpleNamespace(release_observer=None)
        self.scheduler._schedule_policy_switch_in_progress = False
        self.scheduler.schedule_policy = "fcfs"
        self.scheduler.policy = self.scheduler._create_schedule_policy("fcfs")

    def switch(self, name):
        return schedule_policy_switch.handle_schedule_policy_switch(
            self.scheduler, SchedulePolicySwitchReqInput(new_schedule_policy=name)
        )

    def test_switch_restores_fcfs_order_without_changing_requests(self):
        requests = [
            SimpleNamespace(
                rid=rid,
                owner_list=None,
                time_stats=SimpleNamespace(wait_queue_entry_time=arrival),
                sampling_params=SimpleNamespace(max_new_tokens=budget),
            )
            for rid, arrival, budget in [("A", 1, 16), ("B", 2, 128), ("C", 3, 64)]
        ]
        scheduler = self.scheduler
        move(requests, scheduler.waiting_queue)
        self.assertTrue(self.switch("lof").success)
        scheduler.waiting_queue.reorder(
            scheduler.policy.calc_priority(scheduler.waiting_queue)
        )
        self.assertEqual([r.rid for r in scheduler.waiting_queue], ["B", "C", "A"])
        result = self.switch("fcfs")
        self.assertEqual(
            (result.old_schedule_policy, result.new_schedule_policy), ("lof", "fcfs")
        )
        scheduler.waiting_queue.reorder(
            scheduler.policy.calc_priority(scheduler.waiting_queue)
        )
        self.assertEqual(list(scheduler.waiting_queue), requests)
        self.assertTrue(all(r.owner_list is scheduler.waiting_queue for r in requests))

    def test_shortest_prefill_aging_survives_switching_away_and_back(self):
        get_context().override("test", shortest_prefill_aging_factor=0.5)
        for target, expected in [
            ("shortest-prefill-first", 0.5),
            ("fcfs", 0.0),
            ("shortest-prefill-first", 0.5),
        ]:
            with self.subTest(target=target):
                self.assertTrue(self.switch(target).success)
                self.assertEqual(
                    self.scheduler.policy.shortest_prefill_aging_factor, expected
                )
        self.assertEqual(get_schedule().shortest_prefill_aging_factor, 0.5)

    def test_shortest_prefill_load_scaling_survives_switching_away_and_back(self):
        get_context().override("test", shortest_prefill_boundary_halving_chunks=4.0)
        self.assertTrue(self.switch("shortest-prefill-first").success)
        original = self.scheduler.policy.shortest_prefill_load_scaling
        self.assertEqual(original.chunk_tokens, 4096)
        self.assertEqual(original.halving_tokens, 16384)
        self.assertTrue(self.switch("lof").success)
        self.assertIsNone(self.scheduler.policy.shortest_prefill_load_scaling)
        self.assertTrue(self.switch("shortest-prefill-first").success)
        self.assertEqual(self.scheduler.policy.shortest_prefill_load_scaling, original)

    def test_startup_still_rejects_options_for_the_wrong_policy(self):
        with get_context().override_server_args(shortest_prefill_aging_factor=0.5):
            with self.assertRaisesRegex(ValueError, "requires schedule_policy"):
                self.scheduler._create_schedule_policy("fcfs")
        with get_context().override_server_args(enable_uniboost_gamma_ada=True):
            with self.assertRaisesRegex(ValueError, "requires schedule_policy"):
                self.scheduler._create_schedule_policy("fcfs")

    def test_uniboost_controller_is_wired_and_cleared(self):
        get_context().override(
            "test", uniboost_gamma=17.0, enable_uniboost_gamma_ada=True
        )
        self.assertTrue(self.switch("uniboost").success)
        policy = self.scheduler.policy
        self.assertEqual(policy.uniboost_gamma, 17.0)
        self.assertIs(
            self.scheduler.request_retirer.release_observer.__self__,
            policy.uniboost_gamma_ada,
        )
        self.assertTrue(self.switch("fcfs").success)
        self.assertIsNone(self.scheduler.request_retirer.release_observer)
        self.assertTrue(self.switch("uniboost").success)
        self.assertEqual(self.scheduler.policy.uniboost_gamma, 17.0)
        self.assertIs(
            self.scheduler.request_retirer.release_observer.__self__,
            self.scheduler.policy.uniboost_gamma_ada,
        )

    def test_factory_preserves_parallel_clock_and_dllm_configuration(self):
        with get_context().override_server_args(
            tp_size=2, enable_uniboost_gamma_ada=True
        ):
            policy = self.scheduler._create_schedule_policy("uniboost")
            self.assertTrue(policy.uniboost_rank_invariant)
            self.assertEqual(policy.uniboost_gamma_ada.step_clock(), 12)
            self.scheduler.forward_ct = 15
            self.assertEqual(policy.uniboost_gamma_ada.step_clock(), 15)
        self.scheduler.dllm_config = object()
        self.assertFalse(
            self.scheduler._create_schedule_policy("lpm")._bounded_lpm_enabled
        )

    def test_lpm_departure_callback_follows_current_policy(self):
        self.assertTrue(self.switch("lpm").success)
        old = self.scheduler.policy
        old._on_bounded_state_change(True)
        self.assertIs(self.scheduler.waiting_queue._on_leave.__self__, old)
        self.assertTrue(self.switch("fcfs").success)
        self.assertIsNone(self.scheduler.waiting_queue._on_leave)
        self.assertTrue(self.switch("lpm").success)
        current = self.scheduler.policy
        current._on_bounded_state_change(True)
        self.assertIs(self.scheduler.waiting_queue._on_leave.__self__, current)
        req = SimpleNamespace(rid="member", owner_list=None)
        move([req], self.scheduler.waiting_queue)
        current._bounded_members = [req]
        move([req], IndexedReqList(ListKind.DECODING))
        self.assertEqual(current._bounded_members, [])

    def test_cache_fallback_reports_and_records_effective_policy(self):
        self.scheduler.tree_cache.disable = True
        result = self.switch("lpm")
        self.assertTrue(result.success)
        self.assertEqual(result.new_schedule_policy, "fcfs")
        self.assertEqual(self.scheduler.policy.policy.value, "fcfs")
        self.assertEqual(self.scheduler.schedule_policy, "fcfs")
        self.assertEqual(get_schedule().schedule_policy, "fcfs")

    def test_failed_creation_preserves_old_policy_and_callback(self):
        self.assertTrue(self.switch("lpm").success)
        old = self.scheduler.policy
        old._on_bounded_state_change(True)
        callback = self.scheduler.waiting_queue._on_leave
        result = self.switch("unknown")
        self.assertFalse(result.success)
        self.assertNotIn("restart required", result.message)
        self.assertIs(self.scheduler.policy, old)
        self.assertIs(self.scheduler.waiting_queue._on_leave, callback)
        self.assertEqual(get_schedule().schedule_policy, "lpm")
        self.assertFalse(self.scheduler._schedule_policy_switch_in_progress)

    def test_failed_config_update_does_not_replace_policy(self):
        old = self.scheduler.policy
        with patch.object(schedule_policy_switch, "get_context") as context:
            context.return_value.override.side_effect = ValueError("config unavailable")
            result = self.switch("lof")
        self.assertFalse(result.success)
        self.assertIs(self.scheduler.policy, old)
        self.assertEqual(self.scheduler.schedule_policy, "fcfs")

    def test_priority_scheduling_rejects_unsupported_target(self):
        self.scheduler.enable_priority_scheduling = True
        old = self.scheduler.policy
        self.assertFalse(self.switch("hrrn").success)
        self.assertIs(self.scheduler.policy, old)
        self.assertTrue(self.switch("lof").success)
        self.assertTrue(self.scheduler.policy.enable_priority_scheduling)

    def test_same_policy_and_reentrant_switch_do_not_replace_policy(self):
        old = self.scheduler.policy
        self.assertTrue(self.switch("fcfs").success)
        self.assertIs(self.scheduler.policy, old)
        self.scheduler._schedule_policy_switch_in_progress = True
        self.assertFalse(self.switch("lof").success)
        self.assertIs(self.scheduler.policy, old)
        self.assertTrue(self.scheduler._schedule_policy_switch_in_progress)

    def test_tensor_and_data_parallelism_allow_switching(self):
        for topology in (
            {},
            {"tp_size": 2},
            {"dp_size": 2},
            {"tp_size": 2, "dp_size": 2},
        ):
            with self.subTest(topology=topology):
                with get_context().override_server_args(**topology):
                    self.assertTrue(self.switch("lof").success)
                    self.assertTrue(self.switch("fcfs").success)

    def test_other_parallel_modes_reject_switching_without_changing_state(self):
        self.assertTrue(self.switch("lpm").success)
        old = self.scheduler.policy
        old._on_bounded_state_change(True)
        callback = self.scheduler.waiting_queue._on_leave
        for topology, reason in (
            ({"pp_size": 2}, "pipeline parallelism"),
            ({"tp_size": 2, "ep_size": 2}, "expert parallelism"),
            ({"tp_size": 2, "attn_dp_size": 2}, "DP attention"),
            ({"tp_size": 2, "attn_cp_size": 2}, "attention context parallelism"),
            ({"enable_prefill_cp": True}, "attention context parallelism"),
            ({"tp_size": 2, "dcp_size": 2}, "decode context parallelism"),
            ({"tp_size": 2, "moe_dp_size": 2}, "MoE data parallelism"),
            ({"tp_size": 2, "dwdp_size": 2}, "distributed weight data parallelism"),
            ({"tp_size": 2, "enable_layernorm_sp": True}, "sequence parallelism"),
        ):
            for target in ("lof", "lpm"):
                with self.subTest(topology=topology, target=target):
                    with get_context().override_server_args(
                        schedule_policy="lpm", **topology
                    ):
                        with patch.object(
                            self.scheduler, "_create_schedule_policy"
                        ) as create_policy:
                            result = self.switch(target)
                        create_policy.assert_not_called()
                        self.assertFalse(result.success)
                        self.assertIn(reason, result.message)
                        self.assertIs(self.scheduler.policy, old)
                        self.assertIs(self.scheduler.waiting_queue._on_leave, callback)
                        self.assertEqual(self.scheduler.schedule_policy, "lpm")
                        self.assertEqual(get_schedule().schedule_policy, "lpm")
                        self.assertFalse(
                            self.scheduler._schedule_policy_switch_in_progress
                        )

    def test_request_schema_accepts_uniboost(self):
        adapter = TypeAdapter(SchedulePolicySwitchReqInput)
        req = adapter.validate_json('{"new_schedule_policy":"uniboost"}')
        self.assertEqual(req.new_schedule_policy, "uniboost")
        decoded = msgspec.msgpack.decode(
            msgspec.msgpack.encode(req), type=SchedulePolicySwitchReqInput
        )
        self.assertEqual(decoded.new_schedule_policy, "uniboost")
        with self.assertRaises(ValidationError):
            adapter.validate_json('{"new_schedule_policy":"unknown"}')

    def test_http_route_switches_validates_and_uses_admin_auth(self):
        from fastapi.testclient import TestClient

        from sglang.srt.entrypoints import http_server

        async def communicate(obj):
            return [self.switch(obj.new_schedule_policy)]

        manager = SimpleNamespace(
            auto_create_handle_loop=Mock(),
            schedule_policy_switch_communicator=AsyncMock(side_effect=communicate),
            record_config_updates=get_context().override,
        )

        async def control(obj, request):
            return await TokenizerControlMixin.schedule_policy_switch(
                manager, obj, request
            )

        manager.schedule_policy_switch = control
        state = SimpleNamespace(tokenizer_manager=manager)
        with patch.object(http_server, "_global_state", state, create=True):
            client = TestClient(http_server.app)
            response = client.post(
                "/schedule_policy_switch", json={"new_schedule_policy": "lof"}
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["new_schedule_policy"], "lof")
            response = client.post(
                "/schedule_policy_switch", json={"new_schedule_policy": "unknown"}
            )
            self.assertEqual(response.status_code, 400)
            self.assertEqual(manager.schedule_policy_switch_communicator.await_count, 1)
            self.assertEqual(self.scheduler.schedule_policy, "lof")

            with get_context().override_server_args(pp_size=2, schedule_policy="lof"):
                response = client.post(
                    "/schedule_policy_switch", json={"new_schedule_policy": "fcfs"}
                )
            self.assertEqual(response.status_code, 400)
            self.assertIn("pipeline parallelism", response.json()["message"])
            self.assertEqual(self.scheduler.schedule_policy, "lof")

            get_context().override(
                "test", api_key="test-api", admin_api_key="test-admin"
            )
            response = client.post(
                "/schedule_policy_switch",
                json={"new_schedule_policy": "fcfs"},
                headers={"Authorization": "Bearer test-api"},
            )
            self.assertEqual(response.status_code, 401)
            self.assertEqual(self.scheduler.schedule_policy, "lof")
            response = client.post(
                "/schedule_policy_switch",
                json={"new_schedule_policy": "fcfs"},
                headers={"Authorization": "Bearer test-admin"},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(self.scheduler.schedule_policy, "fcfs")


class TestTokenizerPolicySwitch(CustomTestCase):
    def make_manager(self, results):
        return SimpleNamespace(
            auto_create_handle_loop=Mock(),
            schedule_policy_switch_communicator=AsyncMock(return_value=results),
            record_config_updates=Mock(),
        )

    def test_tokenizer_reports_effective_policy_after_fallback(self):
        manager = self.make_manager(
            [
                SchedulePolicySwitchReqOutput(
                    success=True, old_schedule_policy="lof", new_schedule_policy="fcfs"
                )
            ]
        )
        result = asyncio.run(
            TokenizerControlMixin.schedule_policy_switch(
                manager, SchedulePolicySwitchReqInput(new_schedule_policy="lpm")
            )
        )
        self.assertTrue(result.success)
        self.assertEqual(result.new_schedule_policy, "fcfs")
        manager.record_config_updates.assert_called_once_with(
            "tokenizer.schedule_policy_switch", schedule_policy="fcfs"
        )

    def test_failed_or_inconsistent_workers_do_not_update_tokenizer_config(self):
        success = SchedulePolicySwitchReqOutput(
            success=True, new_schedule_policy="fcfs"
        )
        failed = SchedulePolicySwitchReqOutput(
            success=False, message="rejected", new_schedule_policy="lpm"
        )
        different = SchedulePolicySwitchReqOutput(
            success=True, new_schedule_policy="lpm"
        )
        for results in ([], [success, failed], [success, different]):
            with self.subTest(results=results):
                manager = self.make_manager(results)
                result = asyncio.run(
                    TokenizerControlMixin.schedule_policy_switch(
                        manager, SchedulePolicySwitchReqInput(new_schedule_policy="lpm")
                    )
                )
                self.assertFalse(result.success)
                manager.record_config_updates.assert_not_called()


if __name__ == "__main__":
    unittest.main()
