from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import ast
import types
import unittest
from pathlib import Path
from typing import Callable, List

import sglang.srt
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ReqKvInfo  # noqa: E402
from sglang.srt.mem_cache.allocation_sizing import (  # noqa: E402
    publish_kv_bookkeeping_page_size,
)
from sglang.srt.runtime_context import get_flags  # noqa: E402

_PUBLISH_FN = "publish_kv_bookkeeping_page_size"
_SRT_ROOT = Path(next(iter(sglang.srt.__path__)))
_CONFIGURATOR = _SRT_ROOT / "mem_cache" / "kv_cache_configurator.py"
_MLX_STUB = _SRT_ROOT / "hardware_backend" / "mlx" / "model_runner_stub.py"


def _make_allocator(*, page_size: int, uses_legacy: bool) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        page_size=page_size,
        uses_legacy_real_length_alloc=uses_legacy,
    )


class TestPublishKvBookkeepingPageSize(CustomTestCase):
    def setUp(self):
        self._saved = get_flags().kv_bookkeeping_page_size
        self._saved_published = get_flags().kv_bookkeeping_page_size_published

    def tearDown(self):
        get_flags().kv_bookkeeping_page_size = self._saved
        get_flags().kv_bookkeeping_page_size_published = self._saved_published

    def test_page_aligned_allocator_publishes_its_page_size(self):
        """A page-aligned allocator makes its page the bookkeeping modulus."""
        publish_kv_bookkeeping_page_size(
            allocator=_make_allocator(page_size=64, uses_legacy=False)
        )

        self.assertEqual(get_flags().kv_bookkeeping_page_size, 64)

    def test_legacy_allocator_publishes_an_unconstrained_modulus(self):
        """A legacy real-length allocator must exempt itself without a branch in the setter."""
        publish_kv_bookkeeping_page_size(
            allocator=_make_allocator(page_size=64, uses_legacy=True)
        )

        self.assertEqual(get_flags().kv_bookkeeping_page_size, 1)
        ReqKvInfo(kv_allocated_len=7, swa_evicted_seqlen=3)

    def test_republishing_the_same_value_is_idempotent(self):
        """configure() runs once per worker (target, draft, extra runners) against one allocator."""
        allocator = _make_allocator(page_size=64, uses_legacy=False)

        publish_kv_bookkeeping_page_size(allocator=allocator)
        publish_kv_bookkeeping_page_size(allocator=allocator)

        self.assertEqual(get_flags().kv_bookkeeping_page_size, 64)

    def test_conflicting_publish_fails_loudly(self):
        """One process-wide value cannot serve two allocators with different page semantics."""
        publish_kv_bookkeeping_page_size(
            allocator=_make_allocator(page_size=64, uses_legacy=False)
        )

        with self.assertRaises(AssertionError):
            publish_kv_bookkeeping_page_size(
                allocator=_make_allocator(page_size=16, uses_legacy=False)
            )

    def test_publishing_after_a_legacy_allocator_fails_loudly(self):
        """A legacy allocator resolves to 1, which must not be read back as "never published"."""
        publish_kv_bookkeeping_page_size(
            allocator=_make_allocator(page_size=64, uses_legacy=True)
        )
        self.assertEqual(get_flags().kv_bookkeeping_page_size, 1)

        with self.assertRaises(AssertionError):
            publish_kv_bookkeeping_page_size(
                allocator=_make_allocator(page_size=64, uses_legacy=False)
            )

    def test_publishing_a_legacy_allocator_after_an_aligned_one_fails_loudly(self):
        """The conflict must be caught in both orders, not only aligned-then-legacy."""
        publish_kv_bookkeeping_page_size(
            allocator=_make_allocator(page_size=64, uses_legacy=False)
        )

        with self.assertRaises(AssertionError):
            publish_kv_bookkeeping_page_size(
                allocator=_make_allocator(page_size=64, uses_legacy=True)
            )

    def test_non_bool_capability_attribute_fails_loudly(self):
        """A mangled attribute must not be coerced into a silent page-aligned answer."""
        allocator = types.SimpleNamespace(
            page_size=64, uses_legacy_real_length_alloc=None
        )

        with self.assertRaises(AssertionError):
            publish_kv_bookkeeping_page_size(allocator=allocator)


class TestPublishSiteCoversEveryPoolPath(CustomTestCase):
    def _find_function(self, tree: ast.Module, name: str) -> ast.FunctionDef:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        self.fail(f"{name} not found")

    def _calls_publish(self, node: ast.AST) -> bool:
        return any(
            isinstance(call.func, ast.Name) and call.func.id == _PUBLISH_FN
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
        )

    def test_publish_runs_in_configure_not_in_the_allocator_builder(self):
        """_init_pools returns early on the unified-pool path, so publishing there would skip it."""
        tree = ast.parse(_CONFIGURATOR.read_text())

        self.assertTrue(self._calls_publish(self._find_function(tree, "configure")))
        self.assertFalse(
            self._calls_publish(
                self._find_function(tree, "_build_token_to_kv_pool_allocator")
            )
        )

    def _statement_index(
        self,
        body: List[ast.stmt],
        predicate: Callable[[ast.Call], bool],
        *,
        what: str,
    ) -> int:
        for i, stmt in enumerate(body):
            for call in ast.walk(stmt):
                if isinstance(call, ast.Call) and predicate(call):
                    return i
        self.fail(f"{what} not found in the function body")

    def _unnested_publish_index(self, body: List[ast.stmt], *, what: str) -> int:
        for i, stmt in enumerate(body):
            if (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Name)
                and stmt.value.func.id == _PUBLISH_FN
            ):
                return i
        self.fail(
            f"{what}: the publish call is not an unconditional statement of the "
            "function body; nesting it under a branch leaves paths unpublished"
        )

    def test_publish_runs_unconditionally_after_the_pools_are_built(self):
        """Publishing before _init_pools, or under a branch, would leave paths unpublished."""
        tree = ast.parse(_CONFIGURATOR.read_text())
        configure = self._find_function(tree, "configure")

        init_pools_at = self._statement_index(
            configure.body,
            lambda call: isinstance(call.func, ast.Attribute)
            and call.func.attr == "_init_pools",
            what="_init_pools",
        )
        publish_at = self._unnested_publish_index(configure.body, what="configure")

        self.assertGreater(publish_at, init_pools_at)

    def test_the_mlx_stub_publishes_its_own_allocator_unconditionally(self):
        """The MLX stub builds pools without the configurator, so it must publish for itself."""
        tree = ast.parse(_MLX_STUB.read_text())
        initialize = self._find_function(tree, "initialize")

        allocator_at = self._statement_index(
            initialize.body,
            lambda call: isinstance(call.func, ast.Name)
            and call.func.id == "TokenToKVPoolAllocator",
            what="TokenToKVPoolAllocator construction",
        )
        publish_at = self._unnested_publish_index(initialize.body, what="MLX initialize")

        self.assertGreater(publish_at, allocator_at)


if __name__ == "__main__":
    unittest.main()
