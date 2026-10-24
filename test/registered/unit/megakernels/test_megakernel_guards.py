"""CPU checks of --enable-nvidia-megakernel's startup resolution."""

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.megakernels import guards, runtime
from sglang.srt.megakernels import spec as registry
from sglang.srt.megakernels.guards import megakernel_disable_reasons
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fakes import ARCH, ISA, REPLICATION, server_args  # noqa: E402

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class TestMegakernelGuards(CustomTestCase):
    def setUp(self):
        registry_patch = patch.dict(registry._REGISTRY, clear=True)
        registry_patch.start()
        self.addCleanup(registry_patch.stop)

    def test_a_disabled_flag_looks_nothing_up_and_keeps_the_forward(self):
        with patch.object(guards, "plugin_reasons") as plugin:
            guards.handle_nvidia_megakernel(server_args())
        plugin.assert_not_called()
        model = SimpleNamespace()  # any attribute read would raise
        with patch.object(runtime, "megakernel_enabled", return_value=False):
            runtime.install_megakernel(
                model=model, hf_config=None, capture_tail_hooks=[], is_draft=False
            )
        self.assertEqual(vars(model), {})

    def test_an_unmet_tp_size_is_refused_and_an_empty_field_accepts_any(self):
        hf_config = SimpleNamespace(architectures=[ARCH], model_type="fake")

        def tp_reasons(path):
            registry._REGISTRY.clear()
            registry.register(architecture=ARCH, path=path)
            _, reasons = megakernel_disable_reasons(
                cfg=server_args(tp_size=8), hf_config=hf_config, isa=ISA
            )
            return [r for r in reasons if "tp_size" in r]

        self.assertTrue(tp_reasons("_fakes:SPEC"))
        self.assertEqual(tp_reasons("_fakes:ANY_TP"), [])


class TestFlagSetsEnv(CustomTestCase):
    """With the flag, a variable of FLAG_SETS_ENV that the served spec requires and
    the user left unset is set to the spec's value; a set value is kept."""

    def setUp(self):
        for patcher in (
            patch.dict(registry._REGISTRY, clear=True),
            patch.dict(os.environ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        for name in (REPLICATION, "SGLANG_DSA_TOPK_BROADCAST"):
            os.environ.pop(name, None)

    def resolve(self, path="_fakes:REPLICATED", **kw):
        registry.register(architecture=ARCH, path=path)
        hf_config = SimpleNamespace(architectures=[ARCH], model_type="fake")
        model_config = SimpleNamespace(hf_config=hf_config, dtype="bfloat16")
        with (
            patch.object(guards, "plugin_reasons", return_value=[]),
            patch.object(guards, "device_isa", return_value=ISA),
            patch.object(guards, "model_config_of", return_value=model_config),
        ):
            guards.handle_nvidia_megakernel(
                server_args(**{"enable_nvidia_megakernel": True, **kw})
            )

    def test_an_unset_variable_is_set_and_logged(self):
        with self.assertLogs(guards.logger, "INFO") as logs:
            self.resolve()
        self.assertTrue(envs.SGLANG_ENABLE_EMBED_REPLICATION.get())
        self.assertIn(f"set {REPLICATION}=True for fake", "\n".join(logs.output))

    def test_a_set_value_is_kept_and_checked(self):
        os.environ[REPLICATION] = "1"
        self.resolve()
        self.assertEqual(os.environ[REPLICATION], "1")
        os.environ[REPLICATION] = "0"
        with self.assertRaisesRegex(ValueError, f"{REPLICATION} False is not True"):
            self.resolve()
        self.assertEqual(os.environ[REPLICATION], "0")

    def test_without_the_flag_nothing_is_set(self):
        self.resolve(enable_nvidia_megakernel=False)
        self.assertNotIn(REPLICATION, os.environ)

    def test_a_spec_that_does_not_require_it_leaves_it_unset(self):
        self.resolve(path="_fakes:SPEC")
        self.assertNotIn(REPLICATION, os.environ)

    def test_a_refused_setup_leaves_the_environment_as_it_was(self):
        # SGLANG_DSA_TOPK_BROADCAST is not in FLAG_SETS_ENV: checked, never set
        with self.assertRaisesRegex(
            ValueError, "SGLANG_DSA_TOPK_BROADCAST False is not True"
        ):
            self.resolve(path="_fakes:REPLICATED_AND_BROADCAST")
        self.assertNotIn(REPLICATION, os.environ)
        self.assertNotIn("SGLANG_DSA_TOPK_BROADCAST", os.environ)

    def test_the_flag_sets_only_variables_read_when_a_model_is_built(self):
        self.assertEqual(registry.FLAG_SETS_ENV, {REPLICATION})


if __name__ == "__main__":
    unittest.main()
