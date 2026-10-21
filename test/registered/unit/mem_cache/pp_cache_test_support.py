"""Load focused CPU definitions without the optional model/runtime stack."""

import __future__
import ast
import importlib.util
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4] / "python/sglang/srt"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@cache
def source_nodes(path):
    return ast.parse((ROOT / path).read_text()).body


def functions(path, names, namespace, *, cls=None):
    """Compile production definitions with explicit runtime dependencies."""
    nodes = source_nodes(path)
    if cls is not None:
        nodes = next(
            n for n in nodes if isinstance(n, ast.ClassDef) and n.name == cls
        ).body
    selected = [
        n
        for n in nodes
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    if {n.name for n in selected} != set(names):
        raise LookupError(f"Missing production definitions: {path}: {names}")
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]),
            str(ROOT / path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace


def method(path, cls, name, extra=None):
    return functions(path, (name,), dict(extra or {}), cls=cls)[name]


lease = load("_scheduler_cache_lease", "mem_cache/pp_cache_prefix.py")
with patch.dict(sys.modules, {"sglang.srt.mem_cache.pp_cache_prefix": lease}):
    m = load("_scheduler_pp_runtime", "managers/pp_prefetch_scheduler.py")
environ = load("_pp_environ", "environ.py")
with patch.dict(sys.modules, {"sglang.srt.environ": environ}):
    config = load("_pp_cache_config", "mem_cache/pp_prefetch_config.py")
register_cpu_ci = load("_pp_ci_registry", "../test/ci/ci_register.py").register_cpu_ci


@dataclass(frozen=True)
class CacheRequestHandle:
    rid: str
    attempt_id: int = 0
