"""Config namespace bags.

publish snapshots resolved ``server_args`` into the ``get_exec()`` / ``get_memory()``
/ ... namespace bags (the single source of truth for config); bags are read-only
by bare assignment and fail closed until published.
"""

import dataclasses
import unittest
from unittest import mock

from sglang.srt import runtime_context as rc
from sglang.srt.arg_groups.arg_utils import NS, A
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@dataclasses.dataclass
class _DeepFake:
    a: A[int, NS("exec.moe.eplb")] = 1
    b: A[int, NS("exec.moe.eplb.tuning")] = 2


@dataclasses.dataclass
class _CollisionFake:
    # 'topk' is both a leaf on exec.moe and a subgroup of exec.moe -> collision.
    topk: A[int, NS("exec.moe")] = 8
    x: A[int, NS("exec.moe.topk")] = 1


_TOP = (
    rc.get_device,
    rc.get_model,
    rc.get_exec,
    rc.get_schedule,
    rc.get_memory,
    rc.get_spec,
    rc.get_lora,
    rc.get_mm,
    rc.get_disagg,
    rc.get_serving,
    rc.get_observability,
)
_EXEC_SUBS = (
    "kernel",
    "moe",
    "graph",
    "comm",
    "mamba",
    "overlap",
    "offload",
    "dllm",
    "deterministic",
    "features",
)


class TestConfigBags(CustomTestCase):
    def setUp(self):
        rc.reset_context()

    def tearDown(self):
        rc.reset_context()

    def _publish(self):
        sa = ServerArgs(model_path="dummy")
        rc.get_context().set_server_args(sa)
        return sa

    def test_fail_closed_before_publish(self):
        with self.assertRaises(ValueError):
            rc.get_exec()
        with self.assertRaises(ValueError):
            rc.get_memory()

    def test_bag_values_match_server_args(self):
        sa = self._publish()
        self.assertEqual(rc.get_exec().moe.moe_runner_backend, sa.moe_runner_backend)
        self.assertEqual(rc.get_exec().kernel.attention_backend, sa.attention_backend)
        self.assertEqual(rc.get_memory().hicache_ratio, sa.hicache_ratio)
        self.assertEqual(rc.get_schedule().page_size, sa.page_size)
        self.assertEqual(rc.get_serving().host, sa.host)
        self.assertEqual(rc.get_model().model_path, sa.model_path)

    def test_all_accessors_and_exec_subgroups(self):
        self._publish()
        for acc in _TOP:
            self.assertIsNotNone(acc())
        exec_cfg = rc.get_exec()
        for sub in _EXEC_SUBS:
            self.assertTrue(hasattr(exec_cfg, sub), f"exec.{sub} missing")

    def test_read_only_by_bare_assignment(self):
        self._publish()
        with self.assertRaises(AttributeError):
            rc.get_memory().hicache_ratio = 9.0

    def test_scoped_override_restores(self):
        sa = self._publish()
        original = sa.hicache_ratio
        with rc.get_memory().override(hicache_ratio=original + 1.0):
            self.assertEqual(rc.get_memory().hicache_ratio, original + 1.0)
        self.assertEqual(rc.get_memory().hicache_ratio, original)

    def test_unknown_leaf_raises(self):
        self._publish()
        with self.assertRaises(AttributeError):
            _ = rc.get_memory().definitely_not_a_field

    def test_reset_clears_bags(self):
        self._publish()
        rc.reset_context()
        with self.assertRaises(ValueError):
            rc.get_exec()


class TestConfigBagTree(CustomTestCase):
    def test_deep_nesting(self):
        bags = rc._build_config_bags(_DeepFake())
        self.assertEqual(bags["exec"].moe.eplb.a, 1)
        self.assertEqual(bags["exec"].moe.eplb.tuning.b, 2)

    def test_leaf_subgroup_collision_raises(self):
        with self.assertRaises(ValueError):
            rc._build_config_bags(_CollisionFake())


class TestRoleNamespaceEnforcement(CustomTestCase):
    """SGLANG_ROLE_NAMESPACES: off (default) is free; record collects the
    per-role read audit; enforce fails closed on reads outside the role's
    declared set."""

    def setUp(self):
        rc.reset_context()

    def tearDown(self):
        rc.reset_context()

    def _publish(self, role):
        rc.publish(ServerArgs(model_path="dummy"), role=role)

    def test_off_mode_ignores_declared_sets(self):
        self._publish("test")
        with mock.patch.dict(rc.ROLE_NAMESPACE_SETS, {"test": frozenset({"serving"})}):
            rc.get_exec()  # off mode: no enforcement despite the narrow set

    def test_enforce_blocks_reads_outside_the_declared_set(self):
        self._publish("test")
        with mock.patch.object(rc, "_ROLE_NS_MODE", "enforce"), mock.patch.dict(
            rc.ROLE_NAMESPACE_SETS, {"test": frozenset({"serving", "schedule"})}
        ):
            rc.get_serving()
            rc.get_schedule()
            with self.assertRaisesRegex(ValueError, "outside the declared set"):
                rc.get_exec()

    def test_enforce_full_tree_role_and_roleless_install_are_unrestricted(self):
        with mock.patch.object(rc, "_ROLE_NS_MODE", "enforce"):
            self._publish("scheduler")  # None in the table = full tree
            rc.get_exec()
            rc.get_mm()
            # A direct set_server_args install is roleless; enforcement only
            # keys off a recorded publish role.
            rc.get_context().set_server_args(ServerArgs(model_path="dummy"))
            rc.get_exec()

    def test_off_mode_bag_read_traces_under_torch_compile(self):
        # config_bag runs inside compiled model forwards; the mode gate must
        # stay a dead-branch-prunable check in the default "off" mode.
        import torch

        self._publish("test")

        @torch.compile(fullgraph=True, backend="eager", dynamic=False)
        def probe(x):
            if rc.get_schedule().max_running_requests is None:
                return x + 1
            return x * 2

        self.assertEqual(probe(torch.zeros(())).item(), 1.0)

    def test_record_mode_collects_the_audit(self):
        self._publish("test")
        with mock.patch.object(rc, "_ROLE_NS_MODE", "record"), mock.patch.object(
            rc, "_RECORDED_NS_READS", set()
        ):
            rc.get_exec()
            rc.get_disagg()
            self.assertIn(("test", "exec"), rc._RECORDED_NS_READS)
            self.assertIn(("test", "disagg"), rc._RECORDED_NS_READS)


if __name__ == "__main__":
    unittest.main()
