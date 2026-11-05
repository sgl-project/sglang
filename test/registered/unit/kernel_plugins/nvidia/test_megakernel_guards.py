"""CPU checks of --enable-nvidia-megakernel's startup resolution."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.kernels.plugin_kernels.nvidia import spec as registry
from sglang.srt.kernel_plugins.nvidia import binding
from sglang.srt.kernel_plugins.nvidia.binding import (
    megakernel_disable_reasons,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fakes import ARCH, ISA, SPEC, server_args  # noqa: E402

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")

KIMI_ARCH = "KimiK3ForConditionalGeneration"
KIMI_MODULE = "sglang.srt.kernel_plugins.nvidia.kimi_k3.spec"


class TestMegakernelGuards(CustomTestCase):
    def setUp(self):
        registry_patch = patch.dict(registry._REGISTRY, clear=True)
        registry_patch.start()
        self.addCleanup(registry_patch.stop)

    def test_a_disabled_flag_looks_nothing_up_and_keeps_the_forward(self):
        cfg = server_args()
        with (
            patch.object(binding, "plugin_reasons") as plugin,
            patch.object(registry.importlib, "import_module") as import_module,
        ):
            binding.handle_nvidia_megakernel(cfg)
        plugin.assert_not_called()
        import_module.assert_not_called()
        model = SimpleNamespace()  # any attribute read would raise
        with patch.object(binding, "megakernel_enabled", return_value=False):
            binding.install_megakernel(
                model=model, hf_config=None, capture_tail_hooks=[], is_draft=False
            )
        self.assertEqual(vars(model), {})

    def test_builtin_registration_is_lazy_and_preserves_other_plugin_specs(self):
        external_specs = (
            ("DeepseekV4ForCausalLM", "external.deepseek:SPEC"),
            (KIMI_ARCH, "sgl_nvidia_mega.kimi_k3.spec:SPEC"),
            ("Qwen3_5MoeForConditionalGeneration", "external.qwen:SPEC"),
        )
        with patch.object(registry.importlib, "import_module") as import_module:
            for architecture, path in external_specs:
                registry.register(architecture=architecture, path=path)
            self.assertEqual(
                registry.registered_architectures(),
                sorted(architecture for architecture, _ in external_specs),
            )
            self.assertTrue(registry.builtin_spec_registered([KIMI_ARCH]))
            self.assertTrue(registry.spec_registered([KIMI_ARCH]))
            self.assertFalse(registry.builtin_spec_registered([ARCH]))
            self.assertNotIn(KIMI_ARCH, registry._REGISTRY)
            import_module.assert_not_called()

            import_module.return_value = SimpleNamespace(SPEC=SPEC)
            self.assertEqual(
                registry.find_spec(architectures=[KIMI_ARCH], isa=ISA), (SPEC, "")
            )
            import_module.assert_called_once_with(KIMI_MODULE)

            import_module.reset_mock()
            architectures = ["DeepseekV4ForCausalLM", KIMI_ARCH]
            self.assertEqual(
                registry.find_spec(architectures=architectures, isa=ISA), (SPEC, "")
            )
            import_module.assert_called_once_with(KIMI_MODULE)

            import_module.reset_mock()
            spec, reason = registry.find_spec(architectures=architectures, isa="sm_100")
            self.assertIsNone(spec)
            self.assertIn("not sm_100", reason)
            import_module.assert_called_once_with(KIMI_MODULE)

        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(
                architecture="DeepseekV4ForCausalLM", path="different.deepseek:SPEC"
            )

    def test_builtin_setup_needs_no_plugin_and_still_checks_requirements(self):
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[KIMI_ARCH]), dtype="bfloat16"
        )
        cfg = server_args(enable_nvidia_megakernel=True)
        with (
            patch.object(binding, "resolving_view", return_value=cfg),
            patch.object(binding, "model_config_of", return_value=model_config),
            patch.object(binding, "device_isa", return_value=ISA),
            patch.object(binding, "entry_points", return_value=[]) as entry_points,
            patch.object(
                registry.importlib,
                "import_module",
                return_value=SimpleNamespace(SPEC=SPEC),
            ),
            binding.envs.SGLANG_PLUGINS.override("unrelated_plugin"),
        ):
            binding.handle_nvidia_megakernel(cfg)
            cfg.tp_size = 8  # The fake spec requires TP4, even for the built-in path.
            with self.assertRaisesRegex(ValueError, "tp_size"):
                binding.handle_nvidia_megakernel(cfg)
            entry_points.assert_not_called()

    def test_other_models_still_require_the_external_plugin_and_allowlist(self):
        model_config = SimpleNamespace(hf_config=SimpleNamespace(architectures=[ARCH]))
        cfg = server_args(enable_nvidia_megakernel=True)
        with (
            patch.object(binding, "resolving_view", return_value=cfg),
            patch.object(binding, "model_config_of", return_value=model_config),
            patch.object(binding, "entry_points", return_value=[]),
            binding.envs.SGLANG_PLUGINS.override(""),
        ):
            with self.assertRaisesRegex(ValueError, "plugin is not installed"):
                binding.handle_nvidia_megakernel(cfg)
        with (
            patch.object(binding, "resolving_view", return_value=cfg),
            patch.object(binding, "model_config_of", return_value=model_config),
            patch.object(
                binding,
                "entry_points",
                return_value=[SimpleNamespace(name=registry.ENTRY_POINT)],
            ),
            binding.envs.SGLANG_PLUGINS.override("unrelated_plugin"),
        ):
            with self.assertRaisesRegex(ValueError, "does not list"):
                binding.handle_nvidia_megakernel(cfg)

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


if __name__ == "__main__":
    unittest.main()
