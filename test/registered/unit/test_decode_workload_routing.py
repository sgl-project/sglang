"""CPU source-isolated routing tests; no GPU/server or transport claims.

Import the dependency-free policy directly. Compile only the real controller's
routing methods and DPBudget to avoid requiring CUDA imports on the test host.
"""

import ast
import importlib.util
import sys
import types
import unittest
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
MANAGERS = ROOT / "python/sglang/srt/managers"
spec = importlib.util.spec_from_file_location(
    "decode_workload_routing", MANAGERS / "decode_workload_routing.py"
)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


def controller_types(sent):
    tree = ast.parse((MANAGERS / "data_parallel_controller.py").read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in (
            "DPBudget",
            "LoadBalanceMethod",
        ):
            nodes.append(node)
        if isinstance(node, ast.ClassDef) and node.name == "DataParallelController":
            node.body = [
                n
                for n in node.body
                if isinstance(n, ast.FunctionDef)
                and n.name
                in ("maybe_external_dp_rank_routing", "total_tokens_scheduler")
            ]
            nodes.append(node)
    scope = {
        "Enum": Enum,
        "auto": auto,
        "Req": object,
        "is_fake_bootstrap_request": policy.is_fake_bootstrap_request,
        "sock_send": lambda socket, req: sent.append((socket, req)),
        "logger": SimpleNamespace(debug=lambda *a: None),
    }
    # Execute only AST nodes from the checked-in controller, never external input.
    exec(  # noqa: S102
        compile(
            ast.Module(body=nodes, type_ignores=[]),
            "controller_source_under_test",
            "exec",
        ),
        scope,
    )
    return scope["DataParallelController"], scope["DPBudget"], scope["LoadBalanceMethod"]


class TestDecodeWorkloadRouting(unittest.TestCase):
    def setUp(self):
        stub = types.ModuleType("sglang.srt.disaggregation.utils")
        stub.FAKE_BOOTSTRAP_HOST = "2.2.2.2"
        self.fake_module_patch = patch.dict(
            sys.modules, {"sglang.srt.disaggregation.utils": stub}
        )
        self.fake_module_patch.start()
        self.addCleanup(self.fake_module_patch.stop)
        self.sent = []
        cls, self.budget_cls, self.methods = controller_types(self.sent)
        self.controller = cls()
        self.controller.decode_workload_balancing = True
        self.controller.workers = ["rank0", "rank1", "rank2", "rank3"]
        self.controller.status = [True] * 4
        self.controller._active_workers = list(range(4))
        self.controller.dp_budget = self.budget_cls(4)

    def request(self, hint=0, length=10):
        return SimpleNamespace(
            routed_dp_rank=hint,
            input_ids=[1] * length,
            disagg_prefill_dp_rank=3,
            bootstrap_room=123,
            bootstrap_host="ctx",
            bootstrap_port=8998,
        )

    def test_override_gen_hint_preserves_ctx_and_room(self):
        c = self.controller
        c.dp_budget.total_tokens = [100, 80, 0, 50]
        req = self.request(0)
        c.total_tokens_scheduler(req)
        self.assertEqual(self.sent[0][0], "rank2")
        self.assertEqual(req.routed_dp_rank, 2)
        self.assertEqual(
            (
                req.disagg_prefill_dp_rank,
                req.bootstrap_room,
                req.bootstrap_host,
                req.bootstrap_port,
            ),
            (3, 123, "ctx", 8998),
        )

    def test_native_pd_warmup_keeps_explicit_gen_rank_without_ctx(self):
        req = self.request(hint=3)
        req.bootstrap_host = "2.2.2.2"
        req.disagg_prefill_dp_rank = None
        policy.validate_prefill_routing(req)
        self.controller.total_tokens_scheduler(req)
        self.assertEqual(self.sent[0][0], "rank3")
        self.assertEqual(req.routed_dp_rank, 3)

    def test_default_off_preserves_external_affinity(self):
        c = self.controller
        c.decode_workload_balancing = False
        c.dp_budget.total_tokens = [100, 80, 0, 50]
        c.total_tokens_scheduler(self.request(0))
        self.assertEqual(self.sent[0][0], "rank0")

    def test_unhealthy_or_absent_workers_never_selected(self):
        c = self.controller
        c.status[0] = False
        c.workers[1] = None
        c.dp_budget.total_tokens = [0, 0, 20, 30]
        c.total_tokens_scheduler(self.request())
        self.assertEqual(self.sent[0][0], "rank2")

    def test_empty_eligible_set_fails_closed(self):
        self.controller.status = [False] * 4
        with self.assertRaises(ValueError):
            self.controller.total_tokens_scheduler(self.request())
        self.assertEqual(self.sent, [])

    def test_request_count_breaks_equal_token_tie(self):
        self.controller.dp_budget.total_requests = [9, 7, 1, 8]
        self.controller.total_tokens_scheduler(self.request())
        self.assertEqual(self.sent[0][0], "rank2")

    def test_sixteen_frontend_burst_uses_one_budget(self):
        for _ in range(16):
            self.controller.total_tokens_scheduler(self.request(hint=0))
        self.assertEqual(self.controller.dp_budget.total_requests, [4] * 4)
        self.assertEqual(self.controller.dp_budget.total_tokens, [40] * 4)

    def test_hot_rank_avoided_after_snapshot(self):
        c = self.controller
        loads = [
            SimpleNamespace(
                timestamp=1.0,
                dp_rank=i,
                num_running_reqs=10,
                num_waiting_reqs=100 if i == 0 else 0,
                num_total_tokens=10000 if i == 0 else 10,
            )
            for i in range(4)
        ]
        c.dp_budget.update_budget(loads)
        for _ in range(16):
            c.total_tokens_scheduler(self.request(hint=0))
        self.assertNotIn("rank0", [s for s, _ in self.sent])
        # Same timestamp must not erase optimistic reservations.
        before = list(c.dp_budget.total_tokens)
        c.dp_budget.update_budget(loads)
        self.assertEqual(c.dp_budget.total_tokens, before)

    def test_dp8_and_dp16(self):
        for n in [8, 16]:
            budget = self.budget_cls(n)
            ranks = [
                budget.dispatch(
                    self.methods.TOTAL_TOKENS, 40000, eligible_ranks=range(n)
                )
                for _ in range(n * 2)
            ]
            self.assertEqual(ranks, list(range(n)) * 2)

    def test_invalid_rank_or_workload(self):
        for ranks, tokens in [([], 1), ([0, 0], 1), ([-1], 1), ([4], 1), ([0], -1)]:
            with self.assertRaises(ValueError):
                self.controller.dp_budget.dispatch(
                    self.methods.TOTAL_TOKENS, tokens, eligible_ranks=ranks
                )

    def test_opt_in_uses_existing_dispatch_once(self):
        c = self.controller
        with patch.object(
            c.dp_budget, "dispatch", wraps=c.dp_budget.dispatch
        ) as dispatch:
            c.total_tokens_scheduler(self.request(length=10))
        dispatch.assert_called_once_with(
            self.methods.TOTAL_TOKENS, estimated_tokens=10, eligible_ranks=[0, 1, 2, 3]
        )
        self.assertEqual(sum(c.dp_budget.total_requests), 1)
        self.assertEqual(sum(c.dp_budget.total_tokens), 10)
        self.assertEqual(len(self.sent), 1)

    def test_default_without_hint_uses_existing_balancing(self):
        c = self.controller
        c.decode_workload_balancing = False
        c.dp_budget.total_tokens = [100, 80, 0, 50]
        req = self.request(hint=None)
        c.total_tokens_scheduler(req)
        self.assertEqual(self.sent[0][0], "rank2")
        self.assertIsNone(req.routed_dp_rank)
        self.assertEqual(c.dp_budget.total_tokens, [100, 80, 10, 50])

    def test_inactive_rank_excluded_and_rank_tie_stable(self):
        c = self.controller
        c._active_workers = [3, 1]
        c.total_tokens_scheduler(self.request())
        self.assertEqual(self.sent[0][0], "rank1")

    def test_legacy_budget_methods_and_reservations(self):
        budget = self.controller.dp_budget
        budget.total_requests = [4, 1, 1, 3]
        self.assertEqual(budget.dispatch(self.methods.TOTAL_REQUESTS), 1)
        self.assertEqual(budget.total_requests, [4, 2, 1, 3])
        budget.total_tokens = [10, 20, 10, 30]
        self.assertEqual(budget.dispatch(self.methods.TOTAL_TOKENS, 7), 2)
        self.assertEqual(budget.total_tokens, [10, 20, 17, 30])
        self.assertIsNone(budget.dispatch(self.methods.ROUND_ROBIN))

    def test_explicit_hint_paths_do_not_reserve_budget(self):
        for enabled, host in [(False, "ctx"), (True, "2.2.2.2")]:
            with self.subTest(enabled=enabled, host=host):
                self.controller.decode_workload_balancing = enabled
                req = self.request(hint=3)
                req.bootstrap_host = host
                with patch.object(self.controller.dp_budget, "dispatch") as dispatch:
                    self.controller.total_tokens_scheduler(req)
                dispatch.assert_not_called()

    def config(self, **deltas):
        return SimpleNamespace(
            **(
                {
                    "disaggregation_decode_workload_balancing": True,
                    "disaggregation_mode": "decode",
                    "enable_dp_attention": True,
                    "dp_size": 8,
                    "load_balance_method": "total_tokens",
                    "disaggregation_decode_enable_radix_cache": False,
                    "max_ep_size": None,
                }
                | deltas
            )
        )

    def test_configuration_validation(self):
        policy.validate_decode_workload_config(self.config())
        for delta in [
            {"disaggregation_mode": "prefill"},
            {"enable_dp_attention": False},
            {"dp_size": 1},
            {"load_balance_method": "round_robin"},
            {"disaggregation_decode_enable_radix_cache": True},
        ]:
            with self.assertRaises(ValueError):
                policy.validate_decode_workload_config(self.config(**delta))
        policy.validate_decode_workload_config(
            self.config(
                disaggregation_decode_workload_balancing=False,
                disaggregation_mode="prefill",
            )
        )

    def test_capacity_cannot_grow_beyond_launch_budget(self):
        policy.validate_decode_workload_config(self.config(max_ep_size=8))
        with self.assertRaisesRegex(ValueError, "max-ep-size"):
            policy.validate_decode_workload_config(self.config(max_ep_size=16))
        policy.validate_decode_workload_config(
            self.config(disaggregation_decode_workload_balancing=False, max_ep_size=16)
        )

    def test_explicit_ctx_bootstrap_validation(self):
        policy.validate_prefill_routing(self.request())
        for name in ["disagg_prefill_dp_rank", "bootstrap_room", "bootstrap_host"]:
            req = self.request()
            setattr(req, name, None)
            with self.assertRaises(ValueError):
                policy.validate_prefill_routing(req)

    def test_default_bootstrap_port_is_preserved(self):
        req = self.request()
        req.bootstrap_port = None
        policy.validate_prefill_routing(req)
        self.controller.total_tokens_scheduler(req)
        self.assertIsNone(req.bootstrap_port)

    def test_invalid_ctx_rank_and_batched_fields(self):
        for rank in [-1, True, "3", [], [2, None]]:
            req = self.request()
            req.disagg_prefill_dp_rank = rank
            with self.assertRaises(ValueError):
                policy.validate_prefill_routing(req)
        req = self.request()
        req.disagg_prefill_dp_rank = [1, 2]
        req.bootstrap_room = [123, 124]
        policy.validate_prefill_routing(req)


if __name__ == "__main__":
    unittest.main()
