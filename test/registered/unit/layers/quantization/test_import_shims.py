"""Pins the quantization package's public import surface.

The refactor series moves implementations between modules and leaves re-export
shims behind. Nothing else in the tree fails loudly when such a shim is
forgotten: a moved symbol just stops being importable from the path that 251
files, the docs, and a string-addressed kernel registry still use.

Adding a shim means adding its old path to `LEGACY_IMPORT_PATHS` below. Nothing
here should ever be deleted, only appended to; a path that no longer needs to
resolve is removed by the gated shim-removal phase, not by whoever moved the
symbol.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import ast
import importlib
import inspect
import unittest
from pathlib import Path

from sglang.test.test_utils import CustomTestCase

PACKAGE = "sglang.srt.layers.quantization"

# Documented in docs/docs/developer_guide/quantization_contribution_guide.mdx and
# imported by 251 files. These names are API, not shims, and never move.
BASE_CONFIG_SYMBOLS = (
    "QuantizationConfig",
    "QuantizeMethodBase",
    "LinearMethodBase",
    "FusedMoEMethodBase",
    "method_has_implemented_embedding",
)

# Every config class the package root re-exports. `configs/model_config.py`
# iterates the whole registry and `models/deepseek_nextn.py` imports
# `Fp8Config` straight from the root, so the root stays a stable surface even
# once the implementations live in per-method packages.
ROOT_CONFIG_EXPORTS = (
    "AWQCPUConfig",
    "AWQConfig",
    "AWQMarlinConfig",
    "AWQXPUConfig",
    "AutoRoundConfig",
    "BitsAndBytesConfig",
    "BlockInt8Config",
    "CPUGPTQConfig",
    "CompressedTensorsConfig",
    "Fp8Config",
    "GGUFConfig",
    "GPTQAscendConfig",
    "GPTQConfig",
    "GPTQMarlinConfig",
    "GPTQXPUConfig",
    "HummingConfig",
    "MlxQuantizationConfig",
    "ModelOptFp4Config",
    "ModelOptFp8Config",
    "ModelOptMixedPrecisionConfig",
    "ModelSlimConfig",
    "MoeWNA16Config",
    "Mxfp4Config",
    "Mxfp4W4A4Config",
    "Mxfp4W4A8Config",
    "NvFp4OnlineConfig",
    "PetitNvFp4Config",
    "QuantizationConfig",
    "QuarkConfig",
    "QuarkInt4Fp8Config",
    "W4AFp8Config",
    "W8A8Fp8Config",
    "W8A8Int8Config",
)

# Non-class root API. The method tables are read by name elsewhere.
ROOT_API = (
    "QUANTIZATION_METHODS",
    "BASE_QUANTIZATION_METHODS",
    "CPU_QUANTIZATION_METHODS",
    "get_quantization_config",
)

# The registry a shim PR appends to: module path -> names that must resolve
# from it.
LEGACY_IMPORT_PATHS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (f"{PACKAGE}.base_config", BASE_CONFIG_SYMBOLS),
    (PACKAGE, ROOT_CONFIG_EXPORTS + ROOT_API),
)


def _module_defines(module_path: str, name: str) -> bool:
    """Is `name` bound anywhere in the module's source, nesting included?

    `importlib` cannot answer this for a symbol whose definition sits behind an
    optional third-party import, so read the source instead.
    """
    spec = importlib.util.find_spec(module_path)
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name:
                return True
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            if node.id == name:
                return True
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if any((a.asname or a.name) == name for a in node.names):
                return True
    return False


class TestLegacyImportPaths(CustomTestCase):
    """Every path in the registry still resolves every name it promises."""

    def test_every_registered_path_resolves(self):
        for module_path, names in LEGACY_IMPORT_PATHS:
            module = importlib.import_module(module_path)
            for name in names:
                with self.subTest(module=module_path, name=name):
                    self.assertTrue(
                        hasattr(module, name),
                        f"{module_path}.{name} is no longer importable; if the "
                        f"symbol moved, leave a re-export shim behind",
                    )

    def test_root_config_exports_are_the_classes_they_name(self):
        # Catches a shim wired to the wrong class, which `hasattr` would miss.
        module = importlib.import_module(PACKAGE)
        for name in ROOT_CONFIG_EXPORTS:
            with self.subTest(name=name):
                obj = getattr(module, name)
                self.assertTrue(inspect.isclass(obj), f"{name} is not a class")
                self.assertEqual(obj.__name__, name)

    def test_from_import_form_works(self):
        # 251 files use `from ... import <names>` rather than getattr, and a
        # module-level `__getattr__` has to serve that form too.
        from sglang.srt.layers.quantization import (  # noqa: F401
            QUANTIZATION_METHODS,
            Fp8Config,
            QuantizationConfig,
            get_quantization_config,
        )
        from sglang.srt.layers.quantization.base_config import (  # noqa: F401
            FusedMoEMethodBase,
            LinearMethodBase,
            QuantizeMethodBase,
            method_has_implemented_embedding,
        )

    def test_submodule_import_still_falls_through(self):
        # A module-level `__getattr__` must raise AttributeError for unknown
        # names, or `from <package> import <submodule>` stops working.
        from sglang.srt.layers.quantization import (  # noqa: F401
            fp4_utils,
            fp8,
            fp8_utils,
            marlin_utils_fp8,
        )

        module = importlib.import_module(PACKAGE)
        with self.assertRaises(AttributeError):
            module.NotAQuantizationSymbol


class TestStringAddressedKernelTargets(CustomTestCase):
    """Kernel specs address their implementation by `"module:attr"` string, so a
    moved symbol is invisible to every linter and only fails when the kernel is
    selected on hardware that has the backend installed.
    """

    def _quantization_targets(self):
        import sglang.kernels.ops.gemm  # noqa: F401  (registers the specs)
        from sglang.kernels import registry

        return [
            spec
            for spec in registry.all_specs()
            if (spec.target or "").startswith(PACKAGE)
        ]

    def test_targets_exist(self):
        specs = self._quantization_targets()
        self.assertTrue(specs, "expected at least gemm.bmm_fp8 to target the package")
        for spec in specs:
            with self.subTest(op=spec.op, target=spec.target):
                module_path, _, attr = spec.target.partition(":")
                # The attribute itself may sit behind an optional third-party
                # import (bmm_fp8 needs flashinfer), so accept either a live
                # attribute or a definition visible in the source.
                module = importlib.import_module(module_path)
                self.assertTrue(
                    hasattr(module, attr) or _module_defines(module_path, attr),
                    f"{spec.op} targets {spec.target}, which no longer names "
                    f"anything in {module_path}",
                )


if __name__ == "__main__":
    unittest.main()
