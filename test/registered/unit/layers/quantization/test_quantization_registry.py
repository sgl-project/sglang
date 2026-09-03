"""Unit tests for the quantization method registry — CPU-only, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import subprocess
import sys
import unittest

from sglang.srt.layers.quantization import QUANTIZATION_METHODS
from sglang.srt.layers.quantization.registry import (
    CPU_SUPPORTED_METHOD_SPECS,
    PLATFORM_OVERRIDE_SPECS,
    QUANTIZATION_METHOD_SPECS,
    all_config_class_specs,
    all_method_names,
)
from sglang.srt.server_args import QUANTIZATION_CHOICES
from sglang.test.test_utils import CustomTestCase


def _every_spec():
    """(table name, method name, spec) for every entry in every table."""
    for name, spec in QUANTIZATION_METHOD_SPECS.items():
        yield "base", name, spec
    for condition, overrides in PLATFORM_OVERRIDE_SPECS:
        for name, spec in overrides.items():
            yield condition, name, spec
    for name, spec in CPU_SUPPORTED_METHOD_SPECS.items():
        yield "cpu", name, spec


class TestSpecsResolve(CustomTestCase):
    """Specs are strings, so a wrong module path or class name is invisible to
    the linters and only shows up when that method is selected. Resolve all of
    them here, including the tables this platform does not activate.
    """

    def test_every_spec_resolves_to_its_named_class(self):
        import importlib

        for table, name, spec in _every_spec():
            with self.subTest(table=table, method=name):
                module_path, sep, class_name = spec.rpartition(":")
                self.assertTrue(sep, f"{spec!r} is not 'module.path:ClassName'")
                cls = getattr(importlib.import_module(module_path), class_name, None)
                self.assertIsNotNone(cls, f"{spec!r} does not resolve")
                self.assertEqual(cls.__name__, class_name)

    def test_config_classes_are_reachable_from_the_package_root(self):
        # The package has always re-exported these; they now come from
        # __getattr__ instead of eager imports.
        import sglang.srt.layers.quantization as quantization

        for class_name in all_config_class_specs():
            with self.subTest(cls=class_name):
                self.assertEqual(getattr(quantization, class_name).__name__, class_name)

    def test_unknown_attribute_raises_attribute_error(self):
        # Must be AttributeError specifically: `from <pkg> import <submodule>`
        # relies on the failure falling through to the import machinery.
        import sglang.srt.layers.quantization as quantization

        with self.assertRaises(AttributeError):
            quantization.NotAQuantizationConfig

        from sglang.srt.layers.quantization import fp8_utils  # noqa: F401


class TestActiveTable(CustomTestCase):
    """`ModelConfig._verify_quantization` iterates the table and takes the first
    checkpoint match, so its order is behavior, and it reads the table through
    `[*...]` and `.items()`.
    """

    def test_order_follows_the_spec_table(self):
        names = [*QUANTIZATION_METHODS]
        base = [n for n in names if n in QUANTIZATION_METHOD_SPECS]
        self.assertEqual(base, [n for n in QUANTIZATION_METHOD_SPECS if n in names])

    def test_reads_like_the_dict_it_replaced(self):
        self.assertIn("fp8", QUANTIZATION_METHODS)
        self.assertNotIn("no_such_method", QUANTIZATION_METHODS)
        self.assertEqual(QUANTIZATION_METHODS["fp8"].__name__, "Fp8Config")
        self.assertEqual([*QUANTIZATION_METHODS], list(QUANTIZATION_METHODS.keys()))
        self.assertEqual(
            len([*QUANTIZATION_METHODS.items()]), len(QUANTIZATION_METHODS)
        )

    def test_active_table_is_a_subset_of_the_platform_independent_names(self):
        self.assertEqual(set(QUANTIZATION_METHODS) - set(all_method_names()), set())


class TestCliChoicesVsRegistry(CustomTestCase):
    """A CLI choice naming no registered method is a crash: `--quantization
    marlin` passed argparse and then died in `get_quantization_config`. The
    reverse is deliberate -- a method needing an already-quantized checkpoint
    is reached through that checkpoint's `quant_method` and must not be
    offered on the CLI -- so only one direction is asserted as an equality.
    """

    # Accepted by the CLI but not quantization methods: resolved before the
    # registry is ever consulted.
    SENTINELS = ("unquant",)

    def test_no_choice_is_missing_from_the_registry(self):
        registered = set(all_method_names())
        self.assertEqual(
            [c for c in QUANTIZATION_CHOICES if c not in registered],
            list(self.SENTINELS),
        )

    def test_checkpoint_only_methods_stay_off_the_cli(self):
        # blockwise_int8 is registered for checkpoint auto-detection only:
        # BlockInt8LinearMethod asserts both a weight_block_size and an
        # int8-serialized checkpoint, and the former may only be set when the
        # latter holds.
        self.assertIn("blockwise_int8", all_method_names())
        self.assertNotIn("blockwise_int8", QUANTIZATION_CHOICES)


class TestPlatformConditions(CustomTestCase):
    """Override rows name their condition as a string, so a new row with an
    unhandled name would only surface on the platform it targets.
    """

    def test_every_override_condition_has_a_handler(self):
        from sglang.srt.layers.quantization import _platform_conditions

        handlers = set(_platform_conditions())
        named = {condition for condition, _ in PLATFORM_OVERRIDE_SPECS}
        self.assertEqual(named - handlers, set())


class TestImportIsLazy(CustomTestCase):
    """Importing the package used to import all 28 config modules -- measured
    at +1984 modules -- so every third-party quantization dependency came with
    it. A stray top-level config import in the package `__init__`, or a
    membership test that falls through to `__getitem__`, silently brings it all
    back.
    """

    def test_membership_and_key_iteration_resolve_nothing(self):
        code = (
            "import sys;"
            "import sglang.srt.layers.quantization as q;"
            "assert ('fp8' in q.QUANTIZATION_METHODS) is True;"
            "assert ('nope' in q.QUANTIZATION_METHODS) is False;"
            "assert [*q.QUANTIZATION_METHODS] == list(q.QUANTIZATION_METHODS.keys());"
            "print([m for m in sys.modules if m.startswith('sglang.srt.layers.quantization.')])"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout
        self.assertEqual(
            sorted(eval(out.strip())),
            ["sglang.srt.layers.quantization.registry"],
            "`in` / key iteration must not import a config module",
        )

    def test_importing_the_package_pulls_in_no_config_module(self):
        code = (
            "import sys;"
            "import sglang.srt.layers.quantization as q;"
            "print([m for m in sys.modules if m.startswith('sglang.srt.layers.quantization.')])"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout
        loaded = eval(out.strip())
        self.assertEqual(
            sorted(loaded),
            ["sglang.srt.layers.quantization.registry"],
            "importing the quantization package must not import any config module",
        )

    def test_importing_server_args_pulls_in_no_quantization_module(self):
        # server_args is imported by CLI tooling and the router; the whole
        # quantization package costs ~1.5 s on top of what it already loads.
        code = (
            "import sys;"
            "import sglang.srt.server_args;"
            "print([m for m in sys.modules if m.startswith('sglang.srt.layers.quantization')])"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout
        self.assertEqual(sorted(eval(out.strip())), [])


if __name__ == "__main__":
    unittest.main()
