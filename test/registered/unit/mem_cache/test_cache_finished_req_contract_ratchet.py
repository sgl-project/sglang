"""Ratchet guard: every in-tree `cache_finished_req` honors the return contract.

`release_kv_cache` frees `[unhandled_kv_start, kv_allocated_len)` on the
backend's behalf, and treats a `None` return as the deprecated legacy contract
(ceil the committed length instead). That fallback is what keeps externally
registered backends working, but it also means an in-tree backend that drops its
tail free and forgets to return leaks the tail silently instead of crashing.

The set of backends is enumerated by parsing the source rather than by counting
them by hand, which has been wrong every time it was tried. AST parsing is used
over runtime reflection because instantiating these backends needs GPU pools,
mamba pools, the C++ extension, and optional lmcache/flexkv deps; a
"skip what won't instantiate" enumeration would silently drop exactly the
backends most worth guarding.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import ast
import unittest
from pathlib import Path

import sglang.srt
from sglang.test.test_utils import CustomTestCase

_SRT_ROOT = Path(next(iter(sglang.srt.__path__)))

_ABSTRACT_DEFINITION = ("mem_cache/base_prefix_cache.py", "BasePrefixCache")

# StreamingSession delegates to `self.inner`, which may be an externally
# registered legacy backend, so it must be able to pass a `None` through.
_WRAPPER_DEFINITION = ("session/streaming_session.py", "StreamingSession")

_PINNED_DEFINITIONS = frozenset(
    {
        _ABSTRACT_DEFINITION,
        _WRAPPER_DEFINITION,
        ("mem_cache/chunk_cache.py", "ChunkCache"),
        ("mem_cache/chunk_cache.py", "PureSWAChunkCache"),
        ("mem_cache/mamba_radix_cache.py", "MambaRadixCache"),
        ("mem_cache/pure_swa_radix_cache.py", "PureSWARadixCache"),
        ("mem_cache/radix_cache.py", "RadixCache"),
        ("mem_cache/radix_cache_cpp.py", "RadixCacheCpp"),
        ("mem_cache/storage/flexkv/flexkv_radix_cache.py", "FlexKVRadixCache"),
        ("mem_cache/storage/lmcache/lmc_radix_cache.py", "LMCRadixCache"),
        ("mem_cache/swa_radix_cache.py", "SWARadixCache"),
        ("mem_cache/unified_radix_cache.py", "UnifiedRadixCache"),
    }
)


def _find_definitions() -> dict[tuple[str, str], ast.FunctionDef]:
    definitions: dict[tuple[str, str], ast.FunctionDef] = {}
    for path in sorted(_SRT_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for class_node in ast.walk(tree):
            if not isinstance(class_node, ast.ClassDef):
                continue
            for node in class_node.body:
                if (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == "cache_finished_req"
                ):
                    rel = path.relative_to(_SRT_ROOT).as_posix()
                    definitions[(rel, class_node.name)] = node
    return definitions


def _always_exits(body: list[ast.stmt]) -> bool:
    last = body[-1]
    if isinstance(last, (ast.Return, ast.Raise)):
        return True
    if isinstance(last, ast.If):
        return (
            bool(last.orelse)
            and _always_exits(last.body)
            and _always_exits(last.orelse)
        )
    if isinstance(last, ast.With):
        return _always_exits(last.body)
    if isinstance(last, ast.Try):
        if last.finalbody and _always_exits(last.finalbody):
            return True
        if last.orelse:
            tail_ok = _always_exits(last.orelse)
        else:
            tail_ok = _always_exits(last.body)
        return tail_ok and all(_always_exits(h.body) for h in last.handlers)
    return False


class TestCacheFinishedReqContractRatchet(CustomTestCase):
    def test_cache_finished_req_definitions_match_the_pin(self):
        """Every class defining cache_finished_req is pinned, and every pin still exists."""
        found = set(_find_definitions())

        grown = found - _PINNED_DEFINITIONS
        self.assertFalse(
            grown,
            f"New cache_finished_req definitions {sorted(grown)} are not pinned. "
            "They must return CacheFinishedReqResult(unhandled_kv_start=...) "
            "instead of freeing the tail KV range themselves; see "
            "release_kv_cache in mem_cache/common.py. Then add them here.",
        )
        shrunk = _PINNED_DEFINITIONS - found
        self.assertFalse(
            shrunk,
            f"Pinned cache_finished_req definitions {sorted(shrunk)} no longer "
            "exist; drop them from the pin.",
        )

    def test_cache_finished_req_never_returns_bare(self):
        """No backend can return None implicitly, which release_kv_cache reads as legacy."""
        for key, node in sorted(_find_definitions().items()):
            if key == _ABSTRACT_DEFINITION:
                continue
            with self.subTest(definition=key):
                bare_returns = [
                    child
                    for child in ast.walk(node)
                    if isinstance(child, ast.Return) and child.value is None
                ]
                self.assertEqual(
                    bare_returns,
                    [],
                    f"{key} has a bare `return`, which release_kv_cache reads as "
                    "the deprecated legacy contract and silently leaks the KV "
                    "tail. Return CacheFinishedReqResult(...) instead.",
                )
                self.assertTrue(
                    _always_exits(node.body),
                    f"{key} can fall off its end and implicitly return None, "
                    "which release_kv_cache reads as the deprecated legacy "
                    "contract and silently leaks the KV tail.",
                )

    def test_cache_finished_req_return_annotation_is_pinned(self):
        """Only the abstract signature and the session wrapper may return Optional."""
        for key, node in sorted(_find_definitions().items()):
            with self.subTest(definition=key):
                annotation = ast.unparse(node.returns) if node.returns else None
                if key in (_ABSTRACT_DEFINITION, _WRAPPER_DEFINITION):
                    expected = "Optional[CacheFinishedReqResult]"
                else:
                    expected = "CacheFinishedReqResult"
                self.assertEqual(
                    annotation,
                    expected,
                    f"{key} must be annotated `-> {expected}`.",
                )


if __name__ == "__main__":
    unittest.main()
