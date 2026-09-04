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
from sglang.srt.arg_groups.overrides import resolution_result
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
    def _callTestMethod(self, method):
        # No retry: CustomTestCase retries once in CI, but `addCleanup` runs
        # after the last attempt, so a retry of the dual-resolve case would
        # re-enter with the first attempt's leaked process state instead of
        # the pristine snapshot the sibling helper restores from.
        unittest.TestCase._callTestMethod(self, method)

    def setUp(self):
        rc.reset_context()

    def tearDown(self):
        rc.reset_context()

    def _publish(self):
        sa = ServerArgs(model_path="dummy")
        # Through publish, so the record is resolved the way a process resolves it.
        rc.publish(sa, role="test")
        return sa

    def test_fail_closed_before_publish(self):
        with self.assertRaises(ValueError):
            rc.get_exec()
        with self.assertRaises(ValueError):
            rc.get_memory()

    def test_the_bags_carry_what_resolution_produced(self):
        """The projection is faithful: each leaf is the *resolved* value.

        Two requirements the dummy-model shortcut cannot meet: the record must
        go through the real resolution pipeline (a dummy path returns at the
        dummy-model boundary with every sampled leaf still raw, so raw==raw
        would pass vacuously), and the reference must be an independent
        resolution of the same raw input -- a fresh, never-published record,
        resolved after restoring the process state (environ and the EnvField
        none-flags) the first resolution may have written -- so the assertion
        is "bag == what resolution produces", not "bag == the instance publish
        copied from". Reproducibility (`test_resolution_is_reproducible`)
        licenses the sibling as a stand-in for the pipeline's output.

        The reference's resolved values are read through `resolution_result`,
        because a record holds the user's raw input: the decision lives in the
        declarations, and the bags are where a process reads it. The
        raw-differs guard keeps the comparison meaningful -- every sampled leaf
        must have moved off its dataclass default -- and the last assertion is
        the other half of that invariant: the record still answers the raw
        input for a leaf resolution decided.
        """
        import dataclasses

        sa, reference = self._resolve_published_and_sibling()
        defaults = {f.name: f.default for f in dataclasses.fields(ServerArgs)}
        # Leaves resolution writes on this input on both CI device shapes
        # (CUDA host and CPU-only runner): each starts at a None default.
        sampled = (
            (lambda: rc.get_exec().kernel.attention_backend, "attention_backend"),
            (lambda: rc.get_schedule().page_size, "page_size"),
            (lambda: rc.get_schedule().chunked_prefill_size, "chunked_prefill_size"),
            (lambda: rc.get_schedule().mem_fraction_static, "mem_fraction_static"),
        )
        for accessor, leaf in sampled:
            with self.subTest(leaf=leaf):
                # The raw-differs guard: a sampled leaf that still sits on its
                # default (or has none to differ from) proves nothing.
                self.assertIsNot(defaults[leaf], dataclasses.MISSING)
                resolved = resolution_result(reference, leaf)
                self.assertNotEqual(resolved, defaults[leaf])
                self.assertEqual(accessor(), resolved)
        # The record is the raw input, so the field still reads as the default
        # for a leaf the bag now answers for.
        self.assertEqual(sa.page_size, defaults["page_size"])
        self.assertNotEqual(rc.get_schedule().page_size, sa.page_size)

    def test_passthrough_leaves_project_into_their_namespaces(self):
        """Thin projection smoke over leaves resolution does not move.

        `bag == instance` is all these equalities can claim (publish projected
        an unchanged field into serving/memory/moe/model) -- resolution
        faithfulness is `test_the_bags_carry_what_resolution_produced`'s job.
        """
        sa = self._publish()
        sampled = (
            (lambda: rc.get_serving().host, "host"),
            (lambda: rc.get_memory().hicache_write_policy, "hicache_write_policy"),
            (lambda: rc.get_exec().moe.moe_runner_backend, "moe_runner_backend"),
            (lambda: rc.get_model().model_path, "model_path"),
        )
        for accessor, leaf in sampled:
            with self.subTest(leaf=leaf):
                self.assertEqual(accessor(), getattr(sa, leaf))

    def _resolve_published_and_sibling(self):
        """Resolve a real mini config twice from the same pristine process
        state: publish the first record, hand back the never-published sibling
        as the reference."""
        import json
        import os
        import shutil
        import tempfile

        from sglang.srt.environ import EnvField, envs

        config_dir = tempfile.mkdtemp(prefix="bag_contract_")
        self.addCleanup(shutil.rmtree, config_dir, ignore_errors=True)
        with open(os.path.join(config_dir, "config.json"), "w") as handle:
            json.dump(
                {
                    "architectures": ["LlamaForCausalLM"],
                    "model_type": "llama",
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "num_hidden_layers": 2,
                    "vocab_size": 128,
                    "max_position_embeddings": 2048,
                },
                handle,
            )

        # Resolution can write process state os.environ does not carry (the
        # multimodal transport env sticky plus the EnvField descriptor flag
        # `set()` flips); snapshot both so the sibling resolves the same
        # pristine input. Walk the MRO: `vars(type(envs))` alone would miss
        # fields declared on a base class.
        env_fields = {}
        for klass in reversed(type(envs).__mro__):
            for name, field in vars(klass).items():
                if isinstance(field, EnvField):
                    env_fields[name] = field
        environ_before = dict(os.environ)
        none_flags_before = {
            name: field._set_to_none for name, field in env_fields.items()
        }

        def restore_process_state():
            os.environ.clear()
            os.environ.update(environ_before)
            for name, was_none in none_flags_before.items():
                getattr(type(envs), name)._set_to_none = was_none

        self.addCleanup(restore_process_state)

        def resolve():
            server_args = ServerArgs(
                model_path=config_dir, device="cuda", random_seed=42
            )
            # The reference has to be *resolved*, not merely constructed:
            # construction is inert, and the point of the sibling is to be an
            # independent run of the pipeline over the same raw input.
            server_args.resolve_once()
            return server_args

        sa = resolve()
        rc.publish(sa, role="scheduler")
        restore_process_state()
        return sa, resolve()

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
        self._publish()
        original = rc.get_memory().hicache_ratio
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
        with (
            mock.patch.object(rc, "_ROLE_NS_MODE", "enforce"),
            mock.patch.dict(
                rc.ROLE_NAMESPACE_SETS, {"test": frozenset({"serving", "schedule"})}
            ),
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
            rc.publish(ServerArgs(model_path="dummy"), role="test")
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
        with (
            mock.patch.object(rc, "_ROLE_NS_MODE", "record"),
            mock.patch.object(rc, "_RECORDED_NS_READS", set()),
        ):
            rc.get_exec()
            rc.get_disagg()
            self.assertIn(("test", "exec"), rc._RECORDED_NS_READS)
            self.assertIn(("test", "disagg"), rc._RECORDED_NS_READS)

    def test_enforce_rejects_roles_missing_from_the_table(self):
        # Fail closed: an unknown/misspelled role must not silently inherit
        # the full tree — rejected at publish, and defensively at read time.
        with mock.patch.object(rc, "_ROLE_NS_MODE", "enforce"):
            with self.assertRaisesRegex(ValueError, "no ROLE_NAMESPACE_SETS entry"):
                self._publish("not_a_registered_role")
            self._publish("scheduler")
            with mock.patch.dict(rc.ROLE_NAMESPACE_SETS, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "no ROLE_NAMESPACE_SETS entry"):
                    rc.get_exec()

    def test_mode_env_value_is_validated(self):
        self.assertEqual(rc._validated_role_ns_mode(" Enforce "), "enforce")
        with self.assertRaisesRegex(ValueError, "SGLANG_ROLE_NAMESPACES"):
            rc._validated_role_ns_mode("bogus")

    def test_record_mode_registers_the_exit_summary_at_publish(self):
        # A role that reads no bags must still emit its audit line; the exit
        # hook therefore registers at publish, not at the first read.
        with (
            mock.patch.object(rc, "_ROLE_NS_MODE", "record"),
            mock.patch.object(rc, "_RECORD_DUMP_REGISTERED", False),
        ):
            self._publish("test")
            self.assertTrue(rc._RECORD_DUMP_REGISTERED)

    def test_record_mode_bag_read_traces_under_torch_compile(self):
        # Recording has side effects dynamo must never trace; the
        # is_compiling() probe prunes them, keeping fullgraph capture legal
        # even in record mode.
        import torch

        self._publish("test")
        with (
            mock.patch.object(rc, "_ROLE_NS_MODE", "record"),
            mock.patch.object(rc, "_RECORDED_NS_READS", set()),
        ):

            @torch.compile(fullgraph=True, backend="eager", dynamic=False)
            def probe(x):
                if rc.get_schedule().max_running_requests is None:
                    return x + 1
                return x * 2

            self.assertEqual(probe(torch.zeros(())).item(), 1.0)


if __name__ == "__main__":
    unittest.main()
