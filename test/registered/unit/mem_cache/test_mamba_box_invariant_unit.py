import ast
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _mem_cache_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "python" / "sglang" / "srt" / "mem_cache"
        if candidate.is_dir():
            return candidate
    raise RuntimeError("could not locate the sglang mem_cache source directory")


MEM_CACHE = _mem_cache_dir()
COMPONENTS = MEM_CACHE / "unified_cache_components"

OWNED_KV_TOKENS = {"swa_evicted_seqlen", "kv_allocated_len", "cache_protected_len"}

FULL_SWA_FILES = [
    COMPONENTS / "full_component.py",
    COMPONENTS / "swa_component.py",
    MEM_CACHE / "swa_radix_cache.py",
    MEM_CACHE / "radix_cache.py",
]


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _names_and_attrs(tree: ast.Module) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
    return out


def _req_attrs(tree: ast.Module) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "req"
        ):
            out.add(node.attr)
    return out


def test_mamba_component_does_not_read_owned_kv_or_cache_fields():
    """mamba box must not reach into owned-kv / cache req state (op22 invariant)."""
    tree = _parse(COMPONENTS / "mamba_component.py")
    assert not (_req_attrs(tree) & {"kv", "cache"})
    assert not (_names_and_attrs(tree) & OWNED_KV_TOKENS)


@pytest.mark.parametrize("path", FULL_SWA_FILES, ids=lambda p: p.name)
def test_full_swa_cache_does_not_reference_mamba(path: Path):
    """full/swa cache code must carry no mamba references (op22 invariant)."""
    tree = _parse(path)
    assert "mamba" not in _req_attrs(tree)
    mamba_ids = {
        name
        for name in _names_and_attrs(tree)
        if name == "mamba" or name.startswith("mamba_")
    }
    assert (
        not mamba_ids
    ), f"{path.name} references mamba identifiers: {sorted(mamba_ids)}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
