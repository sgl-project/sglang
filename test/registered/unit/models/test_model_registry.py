"""Unit tests for class registration and lazy loading in ``_ModelRegistry``.

With ``SGLANG_EXTERNAL_MODEL_PACKAGE`` set, the registry holds only the external
package's models at import time and imports the built-in ``sglang.srt.models``
package on the first lookup that needs it; an out-of-tree architecture must
resolve without paying for that import.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch.nn as nn

from sglang.srt.models import registry as registry_module
from sglang.srt.models.registry import _ModelRegistry
from sglang.test.test_utils import CustomTestCase


class _DefaultModel(nn.Module):
    pass


class _ExternalModel(nn.Module):
    pass


class _ExternalOverride(nn.Module):
    pass


class _RuntimeModel(nn.Module):
    def __init__(self):
        raise AssertionError("Registration must not instantiate the model")


DEFAULT_PACKAGE = "sglang.srt.models"
FAKE_PACKAGES = {
    DEFAULT_PACKAGE: {"DefaultArch": _DefaultModel, "SharedArch": _DefaultModel},
    "another.models": {"DefaultArch": _ExternalModel},
}
EXTERNAL_MODELS = {"ExternalArch": _ExternalModel, "SharedArch": _ExternalOverride}


class TestLazyDefaultModels(CustomTestCase):
    def setUp(self):
        self.import_calls = []
        patcher = patch.object(
            registry_module, "import_model_classes", side_effect=self._fake_import
        )
        self.addCleanup(patcher.stop)
        patcher.start()
        # The state ``registry.py`` builds when SGLANG_EXTERNAL_MODEL_PACKAGE is set.
        self.registry = _ModelRegistry(
            models=dict(EXTERNAL_MODELS), _default_models_loaded=False
        )

    def _fake_import(self, package_name, strict=False):
        self.import_calls.append(package_name)
        return dict(FAKE_PACKAGES[package_name])

    def test_external_arch_resolves_without_importing_defaults(self):
        self.assertTrue(self.registry.is_arch_supported("ExternalArch"))
        model_cls, arch = self.registry.resolve_model_cls(["ExternalArch"])
        self.assertIs(model_cls, _ExternalModel)
        self.assertEqual(arch, "ExternalArch")
        self.assertEqual(self.import_calls, [])

    def test_default_arch_imports_defaults_once(self):
        model_cls, arch = self.registry.resolve_model_cls(["DefaultArch"])
        self.assertIs(model_cls, _DefaultModel)
        self.assertEqual(arch, "DefaultArch")
        self.assertFalse(self.registry.is_arch_supported("MissingArch"))
        self.registry.resolve_model_cls("DefaultArch")
        self.assertEqual(self.import_calls, [DEFAULT_PACKAGE])

    def test_external_models_override_defaults(self):
        self.assertEqual(
            set(self.registry.get_supported_archs()),
            {"ExternalArch", "SharedArch", "DefaultArch"},
        )
        self.assertIs(
            self.registry.resolve_model_cls("SharedArch")[0], _ExternalOverride
        )

    def test_register_checks_duplicates_against_defaults(self):
        with self.assertRaises(ValueError):
            self.registry.register("another.models")
        self.assertEqual(self.import_calls, [DEFAULT_PACKAGE, "another.models"])

    def test_register_class_loads_defaults_to_check_new_name(self):
        self.registry.register_class("RuntimeArch", _RuntimeModel)

        self.assertEqual(self.import_calls, [DEFAULT_PACKAGE])
        self.assertIs(self.registry.resolve_model_cls("RuntimeArch")[0], _RuntimeModel)
        self.assertIs(self.registry.resolve_model_cls("DefaultArch")[0], _DefaultModel)
        self.assertIs(
            self.registry.resolve_model_cls("SharedArch")[0], _ExternalOverride
        )
        self.assertEqual(
            set(self.registry.get_supported_archs()),
            {"RuntimeArch", "ExternalArch", "SharedArch", "DefaultArch"},
        )
        self.assertEqual(self.import_calls, [DEFAULT_PACKAGE])

    def test_register_class_rejects_deferred_builtin_duplicate(self):
        with self.assertRaisesRegex(ValueError, "DefaultArch already registered"):
            self.registry.register_class("DefaultArch", _RuntimeModel)

        self.assertEqual(self.import_calls, [DEFAULT_PACKAGE])
        self.assertIs(self.registry.resolve_model_cls("DefaultArch")[0], _DefaultModel)
        self.assertIs(
            self.registry.resolve_model_cls("SharedArch")[0], _ExternalOverride
        )

    def test_register_class_rejects_external_duplicate_without_loading_defaults(self):
        for arch in EXTERNAL_MODELS:
            with self.subTest(arch=arch):
                with self.assertRaisesRegex(ValueError, "already registered"):
                    self.registry.register_class(arch, _RuntimeModel)
                self.assertIs(
                    self.registry.resolve_model_cls(arch)[0], EXTERNAL_MODELS[arch]
                )
        self.assertEqual(self.registry.models, EXTERNAL_MODELS)
        self.assertEqual(self.import_calls, [])

    def test_register_class_overwrite_survives_default_loading(self):
        for arch in ("RuntimeArch", "DefaultArch", "ExternalArch", "SharedArch"):
            with self.subTest(arch=arch):
                registry = _ModelRegistry(
                    models=dict(EXTERNAL_MODELS), _default_models_loaded=False
                )
                self.import_calls.clear()
                registry.register_class(arch, _RuntimeModel, overwrite=True)

                self.assertTrue(registry.is_arch_supported(arch))
                for architectures in (arch, [arch]):
                    self.assertEqual(
                        registry.resolve_model_cls(architectures), (_RuntimeModel, arch)
                    )
                self.assertEqual(self.import_calls, [])

                self.assertEqual(
                    set(registry.get_supported_archs()),
                    {"DefaultArch", "ExternalArch", "SharedArch", arch},
                )
                self.assertEqual(
                    registry.resolve_model_cls(["MissingArch", arch]),
                    (_RuntimeModel, arch),
                )
                for external_arch, external_cls in EXTERNAL_MODELS.items():
                    if external_arch != arch:
                        self.assertIs(
                            registry.resolve_model_cls(external_arch)[0], external_cls
                        )
                self.assertEqual(self.import_calls, [DEFAULT_PACKAGE])

    def test_registered_class_preserves_architecture_order_on_lazy_lookup(self):
        self.registry.register_class("RuntimeArch", _RuntimeModel, overwrite=True)

        self.assertEqual(
            self.registry.resolve_model_cls(["DefaultArch", "RuntimeArch"]),
            (_DefaultModel, "DefaultArch"),
        )
        self.assertEqual(
            self.registry.resolve_model_cls(["RuntimeArch", "DefaultArch"]),
            (_RuntimeModel, "RuntimeArch"),
        )
        self.assertEqual(self.import_calls, [DEFAULT_PACKAGE])

    def test_package_registration_after_lazy_class_override(self):
        self.registry.register_class("DefaultArch", _RuntimeModel, overwrite=True)
        with self.assertRaisesRegex(ValueError, "already registered"):
            self.registry.register("another.models")
        self.assertIs(self.registry.resolve_model_cls("DefaultArch")[0], _RuntimeModel)

        self.registry.register("another.models", overwrite=True)
        self.assertIs(self.registry.resolve_model_cls("DefaultArch")[0], _ExternalModel)
        self.assertIs(
            self.registry.resolve_model_cls("SharedArch")[0], _ExternalOverride
        )
        self.assertEqual(
            self.import_calls, [DEFAULT_PACKAGE, "another.models", "another.models"]
        )


class TestRegisterClass(CustomTestCase):
    def setUp(self):
        self.registry = _ModelRegistry()

    def test_register_class_and_alias(self):
        for arch in ("_RuntimeModel", "CustomArchitecture"):
            with self.subTest(arch=arch):
                registry = _ModelRegistry()
                registry.register_class(arch, _RuntimeModel)

                self.assertTrue(registry.is_arch_supported(arch))
                self.assertEqual(set(registry.get_supported_archs()), {arch})
                for architectures in (arch, [arch], ["MissingArch", arch]):
                    model_cls, resolved_arch = registry.resolve_model_cls(architectures)
                    self.assertIs(model_cls, _RuntimeModel)
                    self.assertEqual(resolved_arch, arch)

    def test_register_runtime_constructed_class_without_importing_package(self):
        runtime_cls = type(
            "GeneratedModel", (_RuntimeModel,), {"__module__": "runtime_only_model"}
        )
        with patch.object(
            registry_module,
            "import_model_classes",
            side_effect=AssertionError("Class registration must not import a package"),
        ):
            self.registry.register_class("GeneratedArch", runtime_cls)
            self.assertIs(
                self.registry.resolve_model_cls("GeneratedArch")[0], runtime_cls
            )

    def test_duplicate_registration_preserves_original_class(self):
        self.registry.register_class("RuntimeArch", _RuntimeModel)
        for cls in (_RuntimeModel, _ExternalModel):
            with self.subTest(cls=cls):
                with self.assertRaisesRegex(
                    ValueError, "Set overwrite=True to replace"
                ):
                    self.registry.register_class("RuntimeArch", cls)
                self.assertEqual(self.registry.models, {"RuntimeArch": _RuntimeModel})

    def test_explicit_overwrite_replaces_class(self):
        self.registry.register_class("RuntimeArch", _ExternalModel)
        self.registry.register_class("RuntimeArch", _RuntimeModel, overwrite=True)
        self.assertEqual(
            self.registry.resolve_model_cls("RuntimeArch"),
            (_RuntimeModel, "RuntimeArch"),
        )

    def test_class_registration_after_package_registration(self):
        with patch.object(
            registry_module,
            "import_model_classes",
            return_value={"PackageArch": _ExternalModel},
        ):
            self.registry.register("another.models")

        with self.assertRaisesRegex(ValueError, "already registered"):
            self.registry.register_class("PackageArch", _RuntimeModel)
        self.assertIs(self.registry.resolve_model_cls("PackageArch")[0], _ExternalModel)

        self.registry.register_class("PackageArch", _RuntimeModel, overwrite=True)
        self.assertIs(self.registry.resolve_model_cls("PackageArch")[0], _RuntimeModel)

    def test_package_registration_after_class_registration(self):
        self.registry.register_class("PackageArch", _RuntimeModel)
        with patch.object(
            registry_module,
            "import_model_classes",
            return_value={"PackageArch": _ExternalModel},
        ):
            with self.assertRaisesRegex(ValueError, "already registered"):
                self.registry.register("another.models")
            self.assertIs(
                self.registry.resolve_model_cls("PackageArch")[0], _RuntimeModel
            )

            self.registry.register("another.models", overwrite=True)
            self.assertIs(
                self.registry.resolve_model_cls("PackageArch")[0], _ExternalModel
            )

    def test_duplicate_check_depends_on_successful_builtin_imports(self):
        """A skipped module cannot reserve the architecture its EntryClass exports."""
        working_module = f"{DEFAULT_PACKAGE}.working_model"
        optional_module = f"{DEFAULT_PACKAGE}.optional_model"
        package = SimpleNamespace(__path__=[])
        modules = {
            DEFAULT_PACKAGE: package,
            working_module: SimpleNamespace(EntryClass=_DefaultModel),
            optional_module: SimpleNamespace(EntryClass=_ExternalModel),
        }
        for import_error in (
            None,
            ImportError("Optional accelerator dependency is unavailable"),
            RuntimeError("Accelerator initialization failed"),
        ):
            with self.subTest(import_error=import_error):
                registry = _ModelRegistry(_default_models_loaded=False)

                def import_module(name):
                    if name == optional_module and import_error is not None:
                        raise import_error
                    return modules[name]

                with (
                    # Exercise real discovery without reading or clearing its
                    # process-wide cache, which may already contain built-ins.
                    patch.object(
                        registry_module,
                        "import_model_classes",
                        registry_module.import_model_classes.__wrapped__,
                    ),
                    patch.object(
                        registry_module.importlib,
                        "import_module",
                        side_effect=import_module,
                    ),
                    patch.object(
                        registry_module.pkgutil,
                        "iter_modules",
                        return_value=[
                            (None, working_module, False),
                            (None, optional_module, False),
                        ],
                    ),
                    patch.object(
                        registry_module.envs.SGLANG_DISABLED_MODEL_ARCHS,
                        "get",
                        return_value=[],
                    ),
                ):
                    arch = _ExternalModel.__name__
                    if import_error is None:
                        with self.assertRaisesRegex(ValueError, "already registered"):
                            registry.register_class(arch, _RuntimeModel)
                        expected_cls = _ExternalModel
                    else:
                        with self.assertLogs(registry_module.logger, level="WARNING"):
                            registry.register_class(arch, _RuntimeModel)
                        expected_cls = _RuntimeModel

                    self.assertIs(registry.resolve_model_cls(arch)[0], expected_cls)
                    self.assertIs(
                        registry.resolve_model_cls(_DefaultModel.__name__)[0],
                        _DefaultModel,
                    )
                    self.assertEqual(
                        set(registry.get_supported_archs()),
                        {arch, _DefaultModel.__name__},
                    )

    def test_invalid_class_does_not_replace_registration(self):
        self.registry.register_class("RuntimeArch", _RuntimeModel)
        for cls in (None, object, nn.Module(), "package.Model"):
            with self.subTest(cls=cls):
                with self.assertRaisesRegex(TypeError, "subclass of torch.nn.Module"):
                    self.registry.register_class("RuntimeArch", cls, overwrite=True)
                self.assertEqual(self.registry.models, {"RuntimeArch": _RuntimeModel})

    def test_invalid_architecture_does_not_change_registry(self):
        for arch in (None, 1, ["RuntimeArch"]):
            with self.subTest(arch=arch):
                with self.assertRaisesRegex(TypeError, "architecture must be a string"):
                    self.registry.register_class(arch, _RuntimeModel)
                self.assertEqual(self.registry.models, {})


if __name__ == "__main__":
    unittest.main()
