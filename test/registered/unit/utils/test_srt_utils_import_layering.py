"""sglang.srt.utils is the leaf layer of srt: its modules import no srt domain
package at module level.

A helper that needs a domain package (managers, layers, multimodal, ...) belongs
to that package, as ``sglang.srt.<pkg>.utils`` or a sibling module. Imports
inside a function or under ``TYPE_CHECKING`` do not run at import time and are
outside this check.
"""

import ast
import unittest
from pathlib import Path

import sglang
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PACKAGE_ROOT = Path(next(iter(sglang.__path__)))
_UTILS_DIR = _PACKAGE_ROOT / "srt" / "utils"

# srt infrastructure that sglang.srt.utils may build on. Each imports no srt
# domain package at module level itself.
_ALLOWED = (
    "sglang.srt.utils",
    "sglang.srt.environ",
    "sglang.srt.runtime_context",
    "sglang.srt.platforms",
    "sglang.srt.observability.func_timer",
)

# Modules that predate the rule, with the srt packages each imports. Moving a
# module to its owner removes its entry; the test fails on a stale entry, so the
# list only shrinks.
_KNOWN_VIOLATIONS = {
    "crash_dump.py": {"sglang.srt.sampling"},
    "cuda_vmm_transport_utils.py": {"sglang.srt.multimodal"},
    "hf_transformers/common.py": {"sglang.srt.configs"},
    "hf_transformers/config.py": {"sglang.srt.configs", "sglang.srt.connector"},
    "hf_transformers/gguf_native.py": {"sglang.srt.configs"},
    "hf_transformers/processor.py": {"sglang.srt.multimodal"},
    "hf_transformers/tokenizer.py": {"sglang.srt.connector"},
    "host_shared_memory.py": {"sglang.srt.distributed"},
    "offloader.py": {"sglang.srt.distributed", "sglang.srt.layers"},
    "profile_utils.py": {"sglang.srt.managers", "sglang.srt.model_executor"},
    "request_logger.py": {"sglang.srt.sampling"},
    "weight_checker.py": {"sglang.srt.managers", "sglang.srt.mem_cache"},
    "weight_checker_comparator.py": {"sglang.srt.layers"},
}


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _module_level_imports(path: Path) -> list[str]:
    """Absolute names of the modules imported when ``path`` is imported."""
    package = path.relative_to(_PACKAGE_ROOT.parent).parent.parts
    modules = []
    pending = list(ast.parse(path.read_text()).body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            pending.extend(node.orelse)
            continue
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - node.level + 1]
                name = ".".join(base + ((node.module,) if node.module else ()))
            else:
                name = node.module
            modules.append(name)
        else:
            pending.extend(
                c for c in ast.iter_child_nodes(node) if isinstance(c, ast.stmt)
            )
    return modules


def _srt_package(module: str) -> str:
    return ".".join(module.split(".")[:3])


def _domain_imports() -> dict[str, set[str]]:
    found = {}
    for path in sorted(_UTILS_DIR.rglob("*.py")):
        rel = path.relative_to(_UTILS_DIR).as_posix()
        packages = {
            _srt_package(module)
            for module in _module_level_imports(path)
            if module.startswith("sglang.srt.")
            and not any(
                module == allowed or module.startswith(allowed + ".")
                for allowed in _ALLOWED
            )
        }
        if packages:
            found[rel] = packages
    return found


class TestSrtUtilsImportLayering(CustomTestCase):
    def test_no_new_domain_imports(self):
        new = [
            f"srt/utils/{rel}: {sorted(packages - _KNOWN_VIOLATIONS.get(rel, set()))}"
            for rel, packages in _domain_imports().items()
            if packages - _KNOWN_VIOLATIONS.get(rel, set())
        ]
        self.assertFalse(
            new,
            "sglang.srt.utils imports an srt domain package at module level; move "
            "the helper to the package that owns it, or import inside the "
            "function:\n" + "\n".join(new),
        )

    def test_known_violations_are_current(self):
        found = _domain_imports()
        stale = [
            f"srt/utils/{rel}: {sorted(packages - found.get(rel, set()))}"
            for rel, packages in _KNOWN_VIOLATIONS.items()
            if packages - found.get(rel, set())
        ]
        self.assertFalse(
            stale,
            "these imports are gone; drop them from _KNOWN_VIOLATIONS:\n"
            + "\n".join(stale),
        )


if __name__ == "__main__":
    unittest.main()
