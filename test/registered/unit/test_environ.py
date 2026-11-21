"""Unit tests for sglang.srt.environ: EnvField semantics and the deprecated-env registry."""

import json
import os
import re
import subprocess
import sys
import unittest
import warnings
from contextlib import ExitStack
from unittest.mock import patch

from sglang.srt.environ import EnvField, _DeprecatedEnv, envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=18, suite="base-a-test-cpu")


NV_MEGA_RUNTIME_CHILDREN = (
    "SGLANG_DSPARK_FUSED_ACCEPT_TAIL",
    "SGLANG_DSPARK_DEFER_CONTEXT_KV",
    "SGLANG_OPT_DSV41_DSPARK_DRAFT_PREP_FORK",
    "SGLANG_OPT_DSV41_DSPARK_POST_FOLD",
    "SGLANG_OPT_DSV41_DSPARK_SKIP_PREFILLED_INPUTS",
    "SGLANG_OPT_DSV41_BLOCKWISE_REJECTION",
    "SGLANG_OPT_DSV41_DSPARK_ONE_LAUNCH_COMMIT_STAGE",
    "SGLANG_OPT_MXFP4_DEFERRED_FINALIZED_SHAPE",
    "SGLANG_OPT_CUTEDSL_EP_DEFERRED_FINALIZE",
    "SGLANG_OPT_CUDA_GRAPH_FILL_FAST_PATH",
    "SGLANG_OPT_DSV41_DSPARK_PREPARE_INPUTS_ONCE",
    "SGLANG_OPT_DSV41_DSPARK_MID_FOLD",
    "SGLANG_OPT_DSV41_TARGET_VERIFY_METADATA_GLUE",
    "SGLANG_OPT_DSV41_TARGET_METADATA_EARLY",
    "SGLANG_OPT_SKIP_GLUED_REPLAY_VIEW",
    "SGLANG_OPT_DSV41_POST_FORWARD_CUDA_GRAPH",
    "SGLANG_OPT_DSV41_SAMPLING_ACCEPT_GRAPH",
    "SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH",
    "SGLANG_OPT_SPEC_RELAY_IN_PLACE",
    "SGLANG_OPT_SMALL_FIELD_COPY_VECTORIZED",
    "SGLANG_OPT_SPEC_VERIFY_REUSE_STAGED_INPUTS",
    "SGLANG_OPT_SPEC_POST_VERIFY_ONE_LAUNCH_LOAD",
    "SGLANG_OPT_TRTLLM_MHA_CHAIN_VERIFY",
    "SGLANG_OPT_SPEC_COMPACT_VERIFY_SAMPLING",
    "SGLANG_OPT_SPEC_PACKED_VERIFY_BCAST",
    "SGLANG_OPT_SCHEDULER_SHM_SYNC",
    "SGLANG_OPT_SCHEDULER_SHM_SYNC_MULTIHOST",
    "SGLANG_OPT_SCHEDULER_SHM_SYNC_LEADER_TCP",
    "SGLANG_ENABLE_GDN_PREFILL_GRAPH_EXTEND",
)


class TestEnvField(unittest.TestCase):
    def setUp(self):
        self.env_fields = {
            name: field
            for name, field in vars(type(envs)).items()
            if isinstance(field, EnvField)
        }
        envs.SGLANG_TEST_RETRACT.clear()
        self.addCleanup(envs.SGLANG_TEST_RETRACT.clear)

    def test_set_get_clear_is_set(self):
        self.assertFalse(envs.SGLANG_TEST_RETRACT.is_set())
        self.assertIs(envs.SGLANG_TEST_RETRACT.get(), False)

        envs.SGLANG_TEST_RETRACT.set(True)
        self.assertTrue(envs.SGLANG_TEST_RETRACT.is_set())
        self.assertIs(envs.SGLANG_TEST_RETRACT.get(), True)

        envs.SGLANG_TEST_RETRACT.clear()
        self.assertFalse(envs.SGLANG_TEST_RETRACT.is_set())
        self.assertIs(envs.SGLANG_TEST_RETRACT.get(), False)

    def test_set_to_none_is_distinct_from_clear(self):
        envs.SGLANG_TEST_RETRACT.set(None)
        self.assertTrue(envs.SGLANG_TEST_RETRACT.is_set())
        self.assertIsNone(envs.SGLANG_TEST_RETRACT.get())

        envs.SGLANG_TEST_RETRACT.clear()
        self.assertFalse(envs.SGLANG_TEST_RETRACT.is_set())
        self.assertIs(envs.SGLANG_TEST_RETRACT.get(), False)

    def test_override_restores_previous_state(self):
        envs.SGLANG_TEST_RETRACT.set(True)
        with envs.SGLANG_TEST_RETRACT.override(None):
            self.assertTrue(envs.SGLANG_TEST_RETRACT.is_set())
            self.assertIsNone(envs.SGLANG_TEST_RETRACT.get())
        self.assertIs(envs.SGLANG_TEST_RETRACT.get(), True)

        envs.SGLANG_TEST_RETRACT.set(None)
        with envs.SGLANG_TEST_RETRACT.override(True):
            self.assertIs(envs.SGLANG_TEST_RETRACT.get(), True)
        self.assertTrue(envs.SGLANG_TEST_RETRACT.is_set())
        self.assertIsNone(envs.SGLANG_TEST_RETRACT.get())

    def test_override_with_exit_stack(self):
        envs.SGLANG_TEST_RETRACT.set(None)
        exit_stack = ExitStack()
        exit_stack.enter_context(envs.SGLANG_TEST_RETRACT.override(False))
        self.assertIs(envs.SGLANG_TEST_RETRACT.get(), False)
        exit_stack.close()
        self.assertIsNone(envs.SGLANG_TEST_RETRACT.get())

    def test_override_is_inherited_by_subprocess(self):
        command = [
            sys.executable,
            "-c",
            "import os; print(os.getenv('SGLANG_TEST_RETRACT'))",
        ]
        with envs.SGLANG_TEST_RETRACT.override(True):
            output = subprocess.check_output(command).decode().strip()
            self.assertEqual(output, "True")

        output = subprocess.check_output(command).decode().strip()
        self.assertEqual(output, "None")

    def test_implicit_bool_raises(self):
        message = re.escape(
            "Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"
        )

        with self.assertRaisesRegex(RuntimeError, message):
            if envs.SGLANG_TEST_RETRACT:
                pass

        with self.assertRaisesRegex(RuntimeError, message):
            if (1 != 1) or envs.SGLANG_TEST_RETRACT:
                pass

        with self.assertRaisesRegex(RuntimeError, message):
            if envs.SGLANG_TEST_RETRACT or (1 == 1):
                pass

    def test_invalid_value_warns_and_returns_default(self):
        os.environ["SGLANG_TEST_RETRACT"] = "not-a-bool"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertIs(envs.SGLANG_TEST_RETRACT.get(), False)
        self.assertIn("Invalid value", str(caught[0].message))

    def test_mega_runtime_defaults_and_registry_scope(self):
        with ExitStack() as stack:
            for field in self.env_fields.values():
                stack.enter_context(field.override(False))
                field.clear()
            defaults = {name: field.get() for name, field in self.env_fields.items()}
            master = envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION
            self.assertIs(master.get(), False)
            for master_value in (False, True, False):
                with master.override(master_value):
                    for name, expected in defaults.items():
                        if name == master.name or name in NV_MEGA_RUNTIME_CHILDREN:
                            expected = master_value
                        with self.subTest(master=master_value, name=name):
                            self.assertEqual(self.env_fields[name].get(), expected)
                    for name in NV_MEGA_RUNTIME_CHILDREN:
                        with self.subTest(master=master_value, child=name):
                            child = self.env_fields[name]
                            self.assertIs(child.get(), master_value)
                            self.assertFalse(child.is_set())

    def test_mega_runtime_explicit_children_override_master(self):
        master = envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION
        for master_value, child_value in ((True, "0"), (True, "false"), (False, "1")):
            with master.override(master_value):
                for name in NV_MEGA_RUNTIME_CHILDREN:
                    child = self.env_fields[name]
                    with self.subTest(
                        master=master_value, child=name, value=child_value
                    ):
                        with child.override(child_value):
                            self.assertIs(child.get(), not master_value)
                            self.assertTrue(child.is_set())
                            child.clear()
                            self.assertIs(child.get(), master_value)

    def test_mega_runtime_does_not_override_explicit_none(self):
        with envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True):
            for name in NV_MEGA_RUNTIME_CHILDREN:
                with self.subTest(child=name):
                    child = self.env_fields[name]
                    with child.override(None):
                        self.assertIsNone(child.get())


class TestDeprecatedEnvRegistry(unittest.TestCase):
    def _apply(self, old_name, deprecation):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            deprecation.apply(old_name)
        return caught

    def test_removed_env_warns_without_forwarding(self):
        old_name = "SGLANG_TEST_REMOVED_ENV"
        os.environ[old_name] = "1"
        self.addCleanup(os.environ.pop, old_name, None)

        caught = self._apply(old_name, _DeprecatedEnv())
        self.assertIn(f"{old_name} is deprecated", str(caught[0].message))

    def test_renamed_env_forwards_value(self):
        old_name, new_name = "SGLANG_TEST_OLD_ENV", "SGLANG_TEST_NEW_ENV"
        os.environ[old_name] = "abc"
        self.addCleanup(os.environ.pop, old_name, None)
        self.addCleanup(os.environ.pop, new_name, None)

        caught = self._apply(old_name, _DeprecatedEnv(replacement=new_name))
        self.assertIn(new_name, str(caught[0].message))
        self.assertEqual(os.environ[new_name], "abc")

    def test_unset_env_is_a_no_op(self):
        caught = self._apply("SGLANG_TEST_UNSET_ENV", _DeprecatedEnv())
        self.assertEqual(len(caught), 0)

    def test_renamed_env_applies_transform(self):
        old_name, new_name = "SGLANG_TEST_OLD_ENV", "SGLANG_TEST_NEW_ENV"
        os.environ[old_name] = "abc"
        self.addCleanup(os.environ.pop, old_name, None)
        self.addCleanup(os.environ.pop, new_name, None)

        self._apply(old_name, _DeprecatedEnv(replacement=new_name, transform=str.upper))
        self.assertEqual(os.environ[new_name], "ABC")


class TestDocumentedBooleanKnobs(unittest.TestCase):
    """Documented booleans must accept `true`/`false`: compared to "1" or "0" at the
    read site, `true` was ignored and even turned the MLA RoPE fusion off."""

    DOCUMENTED_DEFAULTS = {
        "SGLANG_SKIP_P2P_CHECK": False,
        "SGLANG_FUSED_MLA_ENABLE_ROPE_FUSION": True,
        "SGLANG_BLOCK_NONZERO_RANK_CHILDREN": True,
    }

    def test_default_and_true_false_spellings(self):
        for name, default in self.DOCUMENTED_DEFAULTS.items():
            field = vars(type(envs))[name]
            with self.subTest(name=name), patch.dict(os.environ):
                os.environ.pop(name, None)
                self.assertIs(field.get(), default)
                for value, expected in (
                    ("true", True),
                    ("1", True),
                    ("false", False),
                    ("0", False),
                ):
                    os.environ[name] = value
                    self.assertIs(field.get(), expected)


class TestImportTimeBooleanKnobs(unittest.TestCase):
    """Modules that freeze a boolean at import must read it through the
    registry: a second parser at the read site accepted fewer spellings than
    the declaration, so `yes` set the knob for some readers and not for others."""

    def test_frozen_module_flags_follow_the_registry(self):
        deepgemm = "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM"
        script = (
            "import json\n"
            "from sglang.srt.batch_invariant_ops import batch_invariant_ops as ops\n"
            "from sglang.srt.batch_overlap import two_batch_overlap as tbo\n"
            "print(json.dumps([ops._ENABLE_MM_DEEPGEMM, "
            "ops._ENABLE_MM_FALLBACK_VARIANT, ops._ENABLE_MM_COMPARISON_TEST, "
            "tbo._tbo_debug]))\n"
        )
        overrides = {
            # The kill switch registered tests launch servers with.
            deepgemm: "0",
            "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "yes",
            "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_COMPARISON_TEST": "1",
            "SGLANG_TBO_DEBUG": "yes",
        }
        # One interpreter: the flags are read once, when the modules load.
        out = subprocess.check_output(
            [sys.executable, "-c", script], env={**os.environ, **overrides}, text=True
        )
        self.assertEqual(
            json.loads(out.strip().splitlines()[-1]), [False, True, True, True]
        )
        with patch.dict(os.environ):
            os.environ.pop(deepgemm, None)
            self.assertIs(vars(type(envs))[deepgemm].get(), True)


if __name__ == "__main__":
    unittest.main()
