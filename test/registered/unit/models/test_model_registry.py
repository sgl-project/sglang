"""Unit tests for lazy default-model loading in ``_ModelRegistry``.

With ``SGLANG_EXTERNAL_MODEL_PACKAGE`` set, the registry holds only the external
package's models at import time and imports the built-in ``sglang.srt.models``
package on the first lookup that needs it; an out-of-tree architecture must
resolve without paying for that import.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import unittest
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


if __name__ == "__main__":
    unittest.main()
