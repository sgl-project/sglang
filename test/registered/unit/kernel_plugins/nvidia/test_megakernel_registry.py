"""CPU checks of lazy megakernel registration and device target selection."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.kernels.plugin_kernels.megakernel import spec as registry
from sglang.kernels.plugin_kernels.nvidia import spec as legacy_spec
from sglang.srt.kernel_plugins.nvidia import binding
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, stage="base-a", runner_config="cpu")

ARCH = "FakeMegaForCausalLM"
OTHER_ARCH = "OtherMegaForCausalLM"
CUDA = "sm_103"
OOT = "fake_accelerator"


def make_spec(name, *isas):
    return registry.MegakernelSpec(
        name=name,
        requires=registry.Requirements(isas=frozenset(isas), tp_sizes=frozenset()),
        features=frozenset(),
        numerics=registry.Numerics(kind=registry.NumericsKind.BITWISE),
        max_rows=1,
        build=lambda model, config, context: None,
        api_version=registry.MEGAKERNEL_API_VERSION,
    )


class TestMegakernelRegistry(unittest.TestCase):
    def setUp(self):
        registry_patch = patch.dict(registry._REGISTRY, clear=True)
        registry_patch.start()
        self.addCleanup(registry_patch.stop)
        registry.register(architecture=ARCH, path="default_provider:SPEC")
        registry.register(architecture=ARCH, target=OOT, path="target_provider:SPEC")

    def test_legacy_imports_share_the_contract_and_registry(self):
        self.assertIs(legacy_spec.MegakernelSpec, registry.MegakernelSpec)
        self.assertIs(legacy_spec.BuiltMegakernel, registry.BuiltMegakernel)
        self.assertIs(legacy_spec.Step, registry.Step)
        self.assertEqual(
            legacy_spec.MEGAKERNEL_API_VERSION, registry.MEGAKERNEL_API_VERSION
        )
        legacy_spec.register(architecture=OTHER_ARCH, path="legacy_provider:SPEC")
        spec = make_spec(CUDA, CUDA)
        with patch.object(
            registry.importlib, "import_module", return_value=SimpleNamespace(SPEC=spec)
        ):
            self.assertEqual(
                registry.find_spec(architectures=[OTHER_ARCH], isa=CUDA), (spec, "")
            )
        self.assertTrue(legacy_spec.spec_registered([ARCH], target=OOT))

    def test_registration_metadata_is_lazy_and_target_specific(self):
        with patch.object(registry.importlib, "import_module") as import_module:
            self.assertTrue(registry.spec_registered([ARCH]))
            self.assertTrue(registry.spec_registered([ARCH], target=OOT))
            self.assertFalse(registry.spec_registered([ARCH], target=CUDA))
            self.assertEqual(registry.registered_architectures(), [ARCH])
        import_module.assert_not_called()

    def test_duplicate_paths_are_idempotent_and_conflicts_are_target_specific(self):
        registry.register(architecture=ARCH, target=OOT, path="target_provider:SPEC")
        with self.assertRaisesRegex(ValueError, f"{ARCH} on {OOT}.*target_provider"):
            registry.register(architecture=ARCH, target=OOT, path="other_provider:SPEC")
        registry.register(architecture=ARCH, target=CUDA, path="other_provider:SPEC")

    def test_exact_target_wins_and_cuda_keeps_the_architecture_default(self):
        for isa, module in ((OOT, "target_provider"), (CUDA, "default_provider")):
            with self.subTest(isa=isa):
                spec = make_spec(isa, CUDA, OOT)
                with patch.object(
                    registry.importlib,
                    "import_module",
                    return_value=SimpleNamespace(SPEC=spec),
                ) as import_module:
                    self.assertEqual(
                        registry.find_spec(architectures=[ARCH], isa=isa), (spec, "")
                    )
                import_module.assert_called_once_with(module)

    def test_exact_only_lookup_skips_defaults_on_all_architectures(self):
        registry.register(architecture=OTHER_ARCH, path="other_provider:SPEC")
        spec = make_spec(OOT, OOT)
        with patch.object(
            registry.importlib, "import_module", return_value=SimpleNamespace(SPEC=spec)
        ) as import_module:
            self.assertEqual(
                registry.find_spec(
                    architectures=[OTHER_ARCH, ARCH], isa=OOT, allow_default=False
                ),
                (spec, ""),
            )
        import_module.assert_called_once_with("target_provider")

    def test_exact_only_miss_does_not_import_the_architecture_default(self):
        with patch.object(registry.importlib, "import_module") as import_module:
            spec, reason = registry.find_spec(
                architectures=[ARCH], isa=CUDA, allow_default=False
            )
        self.assertIsNone(spec)
        self.assertIn(CUDA, reason)
        import_module.assert_not_called()

    def test_broken_exact_target_refuses_without_trying_defaults(self):
        registry.register(architecture=OTHER_ARCH, path="other_provider:SPEC")
        with patch.object(
            registry.importlib,
            "import_module",
            side_effect=ImportError("provider unavailable"),
        ) as import_module:
            spec, reason = registry.find_spec(architectures=[ARCH, OTHER_ARCH], isa=OOT)
        self.assertIsNone(spec)
        self.assertIn(f"{ARCH} on {OOT}", reason)
        self.assertIn("provider unavailable", reason)
        import_module.assert_called_once_with("target_provider")

    def test_draft_ignores_unrelated_targets_and_refuses_a_broken_match(self):
        registry.register(
            architecture=OTHER_ARCH, target=OOT, path="target_provider:SPEC"
        )
        config = SimpleNamespace(architectures=[OTHER_ARCH])
        with (
            patch.object(binding, "device_isa", return_value=CUDA) as device_isa,
            patch.object(binding, "_gather_tp", side_effect=lambda result: [result]),
            patch.object(
                registry.importlib,
                "import_module",
                side_effect=ImportError("provider unavailable"),
            ),
        ):
            self.assertIsNone(
                binding._find_spec_on_every_rank(hf_config=config, is_draft=True)
            )
            device_isa.return_value = OOT
            with self.assertRaisesRegex(ValueError, "provider unavailable"):
                binding._find_spec_on_every_rank(hf_config=config, is_draft=True)


if __name__ == "__main__":
    unittest.main()
