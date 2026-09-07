"""Two family modules must never declare the same field for the same architecture.

An architecture claimed by two family modules is normal -- ``Qwen3NextForCausalLM``
gets its attention shape from ``qwen3_5`` and its MoE runner from ``qwen3_moe``.
Two modules declaring the *same* field for it is not: nobody owns that value,
and which module supplies it is decided by nothing more deliberate than the
order the imports happen to be in. That is a defect in the declarations, so
this forbids it outright rather than choosing a winner.

The ordering follows from the rule and is not itself pinned. ``__init__.py`` is
a list of imports, importing is what registers, and the gate applies matching
declarations in registration order with the last writer winning -- so an
overlap would make an import list into a behavioural statement, which tools
reorder freely. With no overlap the list can be sorted however anyone likes.

The declared-field sets are read with the chain ratchet's own extractor rather
than a second implementation of the same scan, for the reason its docstring
gives: two censuses of one thing that disagree are worse than either alone.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import ast
import importlib.util
import pathlib
import sys
import unittest

from sglang.srt.arg_groups import model_overrides
from sglang.srt.arg_groups.model_override_base import (
    _MODEL_OVERRIDE_FNS,
    MODEL_OVERRIDES,
)
from sglang.test.test_utils import CustomTestCase

_RATCHET = pathlib.Path(__file__).resolve().parent / "test_chain_read_ratchet.py"


def _returned_field_names(fn):
    spec = importlib.util.spec_from_file_location("_chain_ratchet_for_split", _RATCHET)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = pathlib.Path(sys.modules[fn.__module__].__file__).read_text(
        encoding="utf-8-sig"
    )
    body = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == fn.__name__
    )
    return module._returned_field_names(body)


class TestModelOverrideSplit(CustomTestCase):
    def test_no_field_is_declared_by_two_family_modules(self):
        contested = {
            arch: fns for arch, fns in _MODEL_OVERRIDE_FNS.items() if len(fns) > 1
        }
        self.assertTrue(contested, "the scan found no architecture with two claimants")
        for arch, fns in sorted(contested.items()):
            with self.subTest(architecture=arch):
                seen: dict[str, str] = {}
                for fn in fns:
                    for field in _returned_field_names(fn):
                        earlier = seen.get(field)
                        self.assertIsNone(
                            earlier,
                            f"{arch}: {fn.__module__}.{fn.__name__} and {earlier} "
                            f"both declare {field!r}, so which one wins now depends "
                            f"on the order of the imports in "
                            f"arg_groups/model_overrides/__init__.py",
                        )
                        seen[field] = f"{fn.__module__}.{fn.__name__}"

    def test_the_constant_table_does_not_contest_a_callable(self):
        """``MODEL_OVERRIDES`` applies before the callables, so a field it and a
        callable both name is decided by that ordering instead."""
        for arch, const in sorted(MODEL_OVERRIDES.items()):
            for fn in _MODEL_OVERRIDE_FNS.get(arch, ()):
                with self.subTest(architecture=arch, fn=fn.__name__):
                    self.assertFalse(
                        set(const) & _returned_field_names(fn),
                        f"{arch}: MODEL_OVERRIDES and {fn.__name__} both declare "
                        f"{sorted(set(const) & _returned_field_names(fn))}",
                    )

    def test_the_import_list_names_every_family_module(self):
        """Importing is what registers, so a module missing from the list is a
        family that silently stops applying -- and the tests that import a
        provider directly would not notice."""
        package = pathlib.Path(model_overrides.__file__).parent
        on_disk = {
            path.stem for path in package.glob("*.py") if path.stem != "__init__"
        }
        imported = {
            alias.name
            for node in ast.walk(ast.parse((package / "__init__.py").read_text()))
            if isinstance(node, ast.ImportFrom)
            and node.module == "sglang.srt.arg_groups.model_overrides"
            for alias in node.names
        }
        self.assertEqual(on_disk, imported)

    def test_every_declaration_comes_from_its_own_family_module(self):
        """The split itself: nothing was left behind in overrides.py."""
        for arch, fns in _MODEL_OVERRIDE_FNS.items():
            for fn in fns:
                with self.subTest(architecture=arch, fn=fn.__name__):
                    self.assertTrue(
                        fn.__module__.startswith(
                            "sglang.srt.arg_groups.model_overrides."
                        ),
                        f"{fn.__name__} still lives in {fn.__module__}",
                    )


if __name__ == "__main__":
    unittest.main()
