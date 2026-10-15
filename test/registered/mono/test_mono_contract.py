"""CPU test of the monokernel plugin's contract with SGLang (RFC #43949 section 10): every module under
kernels/ops/mono declares the SGLang names it imports in ``SGLANG_READS``, and every declared name still exists. An
SGLang refactor that breaks the plugin fails here and names the file to repair."""

import ast
import unittest
from pathlib import Path

import sglang
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOT = Path(sglang.__file__).parent
PLUGIN = ROOT / "kernels/ops/mono"


def srt_imports(tree: ast.Module) -> set[str]:
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("sglang.srt"):
            names |= {f"{n.module}.{a.name}" for a in n.names}
        elif isinstance(n, ast.Import):
            names |= {a.name for a in n.names if a.name.startswith("sglang.srt")}
    return names


def declared(tree: ast.Module) -> set[str]:
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "SGLANG_READS" for t in n.targets
        ):
            return set(ast.literal_eval(n.value))
    return set()


def module_source(module: str):
    rel = Path(*module.split(".")[1:])
    for p in (ROOT / rel.with_suffix(".py"), ROOT / rel / "__init__.py"):
        if p.is_file():
            return p
    return None


def top_level_names(path: Path) -> set[str]:
    out = set()
    for n in ast.parse(path.read_text()).body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.Assign):
            out |= {t.id for t in n.targets if isinstance(t, ast.Name)}
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            out.add(n.target.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out |= {(a.asname or a.name).split(".")[0] for a in n.names}
    return out


class TestSglangReads(CustomTestCase):
    def test_every_srt_import_is_declared(self):
        for path in sorted(PLUGIN.rglob("*.py")):
            tree = ast.parse(path.read_text())
            used, decl = srt_imports(tree), declared(tree)
            rel = path.relative_to(ROOT)
            self.assertEqual(used - decl, set(), f"{rel}: add these to SGLANG_READS")
            self.assertEqual(
                decl - used,
                set(),
                f"{rel}: SGLANG_READS lists names it no longer imports",
            )

    def test_every_declared_name_exists(self):
        for path in sorted(PLUGIN.rglob("*.py")):
            for name in sorted(declared(ast.parse(path.read_text()))):
                module, _, attr = name.rpartition(".")
                src = module_source(module)
                self.assertIsNotNone(src, f"{path.relative_to(ROOT)}: {module} is gone")
                self.assertIn(
                    attr,
                    top_level_names(src),
                    f"{path.relative_to(ROOT)}: {name} is gone; repair this module",
                )


if __name__ == "__main__":
    unittest.main()
