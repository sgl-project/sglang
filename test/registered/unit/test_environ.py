"""Unit tests for sglang.srt.environ: EnvField semantics and the deprecated-env registry."""

import importlib.machinery
import os
import re
import subprocess
import sys
import tempfile
import unittest
import warnings
from contextlib import ExitStack
from unittest.mock import patch

from sglang.srt import environ
from sglang.srt.environ import EnvField, _DeprecatedEnv, envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


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


class TestKeepTritonBlackwellPtxas(unittest.TestCase):
    """FlashInfer's import must not swap Triton's bundled Blackwell ptxas for an
    older ptxas on PATH (13.0 fuses f32x2 mul/add despite --fmad=false)."""

    @staticmethod
    def _version(release, build=0):
        return f"echo 'Cuda compilation tools, release {release}, V{release}.{build}'"

    @staticmethod
    def _write_ptxas(path, script):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(f"#!/bin/sh\n{script}\n")
        os.chmod(path, 0o755)

    def _fake_toolchain(self, tmp, path_script):
        triton_dir = os.path.join(tmp, "triton")
        bundled = os.path.join(
            triton_dir, "backends", "nvidia", "bin", "ptxas-blackwell"
        )
        self._write_ptxas(bundled, self._version("13.3", 73))
        self._write_ptxas(os.path.join(tmp, "bin", "ptxas"), path_script)
        env_path = os.path.join(tmp, "bin") + os.pathsep + os.environ["PATH"]
        return os.path.join(triton_dir, "__init__.py"), bundled, env_path

    def _resolve(self, path_script, preset=None):
        with tempfile.TemporaryDirectory() as tmp:
            origin, bundled, env_path = self._fake_toolchain(tmp, path_script)
            spec = importlib.machinery.ModuleSpec("triton", None, origin=origin)
            with (
                patch.dict(os.environ, {"PATH": env_path}),
                patch.object(environ.importlib.util, "find_spec", return_value=spec),
                patch.object(environ, "_PTXAS_PROBE_TIMEOUT_S", 0.5),
            ):
                os.environ.pop("TRITON_PTXAS_BLACKWELL_PATH", None)
                if preset is not None:
                    os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = preset
                environ.keep_triton_blackwell_ptxas()
                return os.environ.get("TRITON_PTXAS_BLACKWELL_PATH"), bundled

    def test_older_path_ptxas_keeps_the_bundled_one(self):
        for stream in ("", " >&2"):
            with self.subTest(stream=stream):
                selected, bundled = self._resolve(self._version("13.0", 88) + stream)
                self.assertEqual(selected, bundled)

    def test_path_ptxas_not_provably_older_is_left_to_flashinfer(self):
        older = self._version("13.0", 88)
        for case, script in (
            ("newer release", self._version("13.4", 10)),
            ("same release, newer build", self._version("13.3", 99)),
            ("unparsable", "echo 'ptxas wrapper'"),
            ("failed", f"{older}; exit 1"),
            ("hung", f"sleep 3; {older}"),
        ):
            with self.subTest(case=case):
                selected, _ = self._resolve(script)
                self.assertIsNone(selected)

    def test_explicit_setting_is_kept(self):
        selected, _ = self._resolve(
            self._version("13.0", 88), preset="/opt/cuda/bin/ptxas"
        )
        self.assertEqual(selected, "/opt/cuda/bin/ptxas")

    def test_import_sglang_runs_before_flashinfer_patch(self):
        """Importing sglang alone must pin the bundled ptxas, so a later FlashInfer
        import keeps it instead of adopting the older ptxas on PATH."""
        with tempfile.TemporaryDirectory() as tmp:
            origin, bundled, env_path = self._fake_toolchain(
                tmp, self._version("13.0", 88)
            )
            script = f"""
import importlib.machinery, importlib.util, os, shutil
find_spec = importlib.util.find_spec
fake = importlib.machinery.ModuleSpec("triton", None, origin={origin!r})
importlib.util.find_spec = lambda name, package=None: (
    fake if name == "triton" else find_spec(name, package)
)
import sglang
# Mirrors flashinfer/triton/__init__.py _patch_triton_ptxas_blackwell for a CUDA 13 PATH ptxas.
if not os.environ.get("TRITON_PTXAS_BLACKWELL_PATH"):
    os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = shutil.which("ptxas")
print(os.environ["TRITON_PTXAS_BLACKWELL_PATH"])
"""
            env = dict(os.environ, PATH=env_path)
            env.pop("TRITON_PTXAS_BLACKWELL_PATH", None)
            result = subprocess.run(
                [sys.executable, "-c", script],
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip().splitlines()[-1], bundled)


if __name__ == "__main__":
    unittest.main()
