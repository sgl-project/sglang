"""Unit tests for the unified runtime_context (Global Context Object). CPU-only.

Covers the lifecycle (set/get/reset), parallel accessors + in-place attention
fill, the flags hierarchy with symmetric attribute read/write + freeze discipline
+ scoped override, the lazy BufferStore, ctx.metrics, and the from_model_runner
factory (engine fields off the runner, attention dims + tp_group off the getters).
"""

import dataclasses
import unittest
from unittest import mock

import torch

from sglang.srt.runtime_context import (
    MODEL_OVERRIDABLE_FIELDS,
    BufferSpec,
    BufferStore,
    ParallelContext,
    RuntimeContext,
    apply_model_overrides,
    build_config_only_context,
    build_context,
    get_attn_tp_size,
    get_context,
    get_flags,
    get_server_args,
    resolve_parallel_context,
    get_tp_rank,
    get_tp_size,
    has_context,
    reset_context,
    set_context,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PREDICATE = (
    "sglang.srt.layers.moe.utils.should_use_flashinfer_cutlass_moe_fp4_allgather"
)


def _publish(**parallel_kw) -> None:
    set_context(RuntimeContext(parallel=ParallelContext(**parallel_kw)))


def _ws_spec() -> BufferSpec:
    return BufferSpec(
        name="t.ws", shape=16, dtype=torch.uint8, device=torch.device("cpu")
    )


class TestLifecycle(unittest.TestCase):
    def setUp(self):
        reset_context()

    def tearDown(self):
        reset_context()

    def test_get_before_init_raises(self):
        self.assertFalse(has_context())
        with self.assertRaises(ValueError):
            get_context()
        with self.assertRaises(ValueError):
            get_tp_size()

    def test_set_publishes(self):
        _publish(tp_size=4, tp_rank=2)
        self.assertTrue(has_context())
        self.assertEqual(get_tp_size(), 4)
        self.assertEqual(get_tp_rank(), 2)

    def test_reset_clears_drops_group_and_releases_buffers(self):
        set_context(RuntimeContext(parallel=ParallelContext(tp_group="G")))
        ctx = get_context()
        ctx.buffers.register(_ws_spec())
        ctx.buffers.get_persistent_buffer("t.ws")
        reset_context()
        self.assertFalse(has_context())
        self.assertIsNone(ctx.parallel.tp_group)  # group reference dropped
        self.assertEqual(ctx.buffers._cache, {})  # buffers released
        with self.assertRaises(ValueError):
            get_context()

    def test_republish_overwrites(self):
        _publish(tp_size=2)
        _publish(tp_size=8)
        self.assertEqual(get_tp_size(), 8)


class TestParallel(unittest.TestCase):
    def setUp(self):
        reset_context()

    def tearDown(self):
        reset_context()

    def test_in_place_attention_fill(self):
        _publish(tp_size=8, tp_rank=3, pp_size=2, dp_size=4)
        self.assertEqual(get_tp_size(), 8)
        self.assertIsNone(get_attn_tp_size())  # default until filled
        # in-place fill (no dataclasses.replace, no re-publish)
        get_context().parallel.attn_tp_size = 4
        get_context().parallel.tp_group = "TPG"
        self.assertEqual(get_attn_tp_size(), 4)
        self.assertEqual(get_context().parallel.tp_group, "TPG")
        self.assertEqual(get_tp_size(), 8)  # engine dim unchanged


class TestFlags(unittest.TestCase):
    def setUp(self):
        reset_context()

    def tearDown(self):
        reset_context()

    def test_default_false(self):
        set_context(RuntimeContext())
        self.assertFalse(get_flags().attn.use_mla_backend)
        self.assertFalse(get_flags().moe.use_cutlass_fp4_allgather)
        self.assertFalse(get_flags().is_extend_in_batch)

    def test_hierarchy_and_symmetric_write(self):
        set_context(RuntimeContext())
        get_flags().attn.use_mla_backend = True  # write == read shape
        self.assertTrue(get_flags().attn.use_mla_backend)
        self.assertTrue(get_flags().attn.use_mla_backend)
        get_flags().is_extend_in_batch = True  # flat per-forward
        self.assertTrue(get_flags().is_extend_in_batch)

    def test_freeze_materializes_moe_and_locks_static(self):
        set_context(RuntimeContext())
        get_flags().attn.use_mla_backend = True
        with mock.patch(_PREDICATE, return_value=True):
            get_context().freeze()
        self.assertTrue(get_flags().moe.use_cutlass_fp4_allgather)  # materialized
        with self.assertRaises(RuntimeError):  # static write after freeze
            get_flags().attn.use_mla_backend = False
        self.assertTrue(get_flags().attn.use_mla_backend)  # unchanged
        get_flags().is_extend_in_batch = True  # per-forward still writable
        self.assertTrue(get_flags().is_extend_in_batch)

    def test_typo_raises(self):
        set_context(RuntimeContext())
        with self.assertRaises(AttributeError):
            get_flags().attn.use_mla_backened = True  # not a declared flag

    def test_override_scoped_and_restores(self):
        set_context(RuntimeContext())
        get_flags().attn.use_mla_backend = True
        with mock.patch(_PREDICATE, return_value=False):
            get_context().freeze()  # frozen — override must still work
        with get_flags().attn.override(use_mla_backend=False):
            self.assertFalse(get_flags().attn.use_mla_backend)
        self.assertTrue(get_flags().attn.use_mla_backend)  # restored

        class _Boom(Exception):
            pass

        with self.assertRaises(_Boom):
            with get_flags().attn.override(use_mla_backend=False):
                raise _Boom()
        self.assertTrue(get_flags().attn.use_mla_backend)  # restored despite exception

    def test_override_unknown_flag_raises(self):
        set_context(RuntimeContext())
        with self.assertRaises(AttributeError):
            with get_flags().attn.override(bogus=1):
                pass

    def test_override_invalid_key_does_not_leak_valid_change(self):
        # transactional: an invalid key aborts before mutating the valid one
        set_context(RuntimeContext())
        attn = get_flags().attn
        with self.assertRaises(AttributeError):
            with attn.override(use_mla_backend=True, bogus=1):
                pass
        self.assertFalse(attn.use_mla_backend)  # not left flipped

    def test_multi_runner_fresh_unfrozen(self):
        set_context(RuntimeContext())
        c1 = get_context()
        with mock.patch(_PREDICATE, return_value=True):
            c1.freeze()
        set_context(RuntimeContext())  # second runner re-publishes fresh
        c2 = get_context()
        self.assertIsNot(c2, c1)
        get_flags().attn.use_mla_backend = True  # unfrozen → ok
        self.assertTrue(get_flags().attn.use_mla_backend)

    def test_structs_are_dataclasses(self):
        # flat per-forward field lives directly on Flags, not under a wrapper
        flags = RuntimeContext().flags
        self.assertIn("is_extend_in_batch", flags.__dataclass_fields__)
        self.assertNotIn("is_extend_in_batch", flags.attn.__dataclass_fields__)


class TestBuffers(unittest.TestCase):
    def test_register_does_not_allocate(self):
        store = BufferStore()
        store.register(_ws_spec())
        self.assertEqual(store._cache, {})

    def test_get_persistent_buffer_allocates_once_and_caches(self):
        store = BufferStore()
        store.register(_ws_spec())
        self.assertIs(
            store.get_persistent_buffer("t.ws"), store.get_persistent_buffer("t.ws")
        )

    def test_shape_dtype_device(self):
        store = BufferStore()
        store.register(_ws_spec())
        buf = store.get_persistent_buffer("t.ws")
        self.assertEqual(tuple(buf.shape), (16,))
        self.assertEqual(buf.dtype, torch.uint8)
        self.assertEqual(buf.device.type, "cpu")

    def test_release_reallocates(self):
        store = BufferStore()
        store.register(_ws_spec())
        a = store.get_persistent_buffer("t.ws")
        store.release()
        self.assertEqual(store._cache, {})
        self.assertIsNot(a, store.get_persistent_buffer("t.ws"))

    def test_register_idempotent(self):
        store = BufferStore()
        store.register(_ws_spec())
        store.register(_ws_spec())
        self.assertIs(
            store.get_persistent_buffer("t.ws"), store.get_persistent_buffer("t.ws")
        )

    def test_shape_is_element_count_for_any_dtype(self):
        # shape is an element count passed straight to torch.zeros — correct for
        # any dtype (no bytes/element conflation).
        store = BufferStore()
        store.register(
            BufferSpec(
                name="t.f16", shape=8, dtype=torch.float16, device=torch.device("cpu")
            )
        )
        buf = store.get_persistent_buffer("t.f16")
        self.assertEqual(buf.numel(), 8)
        self.assertEqual(buf.dtype, torch.float16)

    def test_temporary_buffer_fresh_per_call_not_cached(self):
        # temporary: allocate fresh each call, sized per-forward, never cached
        store = BufferStore()
        store.register(
            BufferSpec(name="tmp", dtype=torch.float16, device=torch.device("cpu"))
        )
        a = store.get_temporary_buffer("tmp", (4, 8))  # "forward 1"
        b = store.get_temporary_buffer("tmp", (4, 8))  # "forward 2", same size
        self.assertIsNot(a, b)  # fresh each call, not cached
        self.assertEqual(tuple(a.shape), (4, 8))
        self.assertEqual(a.dtype, torch.float16)
        c = store.get_temporary_buffer("tmp", (2, 8))  # "forward 3", different size
        self.assertEqual(tuple(c.shape), (2, 8))
        self.assertEqual(store._cache, {})  # nothing cached

    def test_get_persistent_buffer_on_temporary_spec_raises(self):
        # a temporary spec (no fixed shape) must not be fetched via the persistent path
        store = BufferStore()
        store.register(
            BufferSpec(name="tmp", dtype=torch.uint8, device=torch.device("cpu"))
        )
        with self.assertRaises(AssertionError):
            store.get_persistent_buffer("tmp")


class TestMetrics(unittest.TestCase):
    def setUp(self):
        reset_context()

    def tearDown(self):
        reset_context()

    def test_metrics_round_trip(self):
        set_context(RuntimeContext())
        self.assertIsNone(get_context().metrics.pre_model_load_memory)
        get_context().metrics.pre_model_load_memory = 42.0
        self.assertEqual(get_context().metrics.pre_model_load_memory, 42.0)


class TestConstruction(unittest.TestCase):
    """P1c construction rework: the pure assembler build_context (server_args by
    reference, use_mla -> flags.attn, memory -> metrics), the degenerate
    build_config_only_context, and resolve_parallel_context (engine dims off the
    runner, attention dims + tp_group off the getters). No GPU/distributed."""

    def setUp(self):
        reset_context()

    def tearDown(self):
        reset_context()

    def test_build_context_assembles_by_reference(self):
        sa = object()  # the SAME object must come back out (identity, no copy)
        parallel = ParallelContext(tp_size=2, tp_rank=1)
        ctx = build_context(
            server_args=sa,
            parallel=parallel,
            use_mla_backend=True,
            pre_model_load_memory=42.0,
        )
        self.assertIs(ctx.server_args, sa)
        self.assertIs(ctx.parallel, parallel)
        self.assertTrue(ctx.flags.attn.use_mla_backend)
        self.assertEqual(ctx.metrics.pre_model_load_memory, 42.0)
        self.assertFalse(ctx._frozen)  # build does not publish or freeze

    def test_build_config_only_context_is_degenerate(self):
        sa = object()
        ctx = build_config_only_context(server_args=sa)
        self.assertIs(ctx.server_args, sa)
        self.assertEqual(ctx.parallel.tp_size, 1)  # defaults
        self.assertFalse(ctx.flags.attn.use_mla_backend)
        self.assertIsNone(ctx.metrics.pre_model_load_memory)

    def test_resolve_parallel_context_reads_runner_and_getters(self):
        class FakeMR:
            tp_size = 2
            tp_rank = 1
            pp_size = 1
            pp_rank = 0
            dp_size = 1
            dp_rank = None
            moe_ep_size = 1
            moe_ep_rank = 0
            moe_dp_size = 1
            moe_dp_rank = None

        getters = {
            "get_attention_cp_size": 1,
            "get_attention_cp_rank": 0,
            "get_attention_tp_size": 2,
            "get_attention_tp_rank": 1,
            "get_attention_dp_size": 1,
            "get_attention_dp_rank": 0,
        }
        with mock.patch("sglang.srt.distributed.get_tp_group", return_value="TPG"):
            with mock.patch.multiple(
                "sglang.srt.layers.dp_attention",
                **{k: (lambda v=v: v) for k, v in getters.items()},
            ):
                parallel = resolve_parallel_context(FakeMR())
        self.assertEqual(parallel.tp_size, 2)
        self.assertEqual(parallel.tp_rank, 1)
        self.assertEqual(parallel.attn_tp_size, 2)
        self.assertEqual(parallel.tp_group, "TPG")


class TestServerArgsShim(unittest.TestCase):
    """P1b: ctx.server_args root field + the identity-preserving
    get_global_server_args() delegating shim (context path wins; legacy global is
    the pre-publish fallback; exact unset-raise message preserved)."""

    def setUp(self):
        reset_context()
        self._clear_legacy()

    def tearDown(self):
        reset_context()
        self._clear_legacy()

    @staticmethod
    def _clear_legacy():
        import sglang.srt.server_args as sa

        sa._global_server_args = None

    def test_shim_returns_context_server_args_by_identity(self):
        from sglang.srt.server_args import get_global_server_args

        sentinel = object()  # the shim returns it verbatim, no isinstance check
        set_context(RuntimeContext(server_args=sentinel))
        self.assertIs(get_global_server_args(), sentinel)
        self.assertIs(get_server_args(), sentinel)
        self.assertIs(get_context().server_args, sentinel)

    def test_shim_falls_back_to_legacy_global_when_no_context(self):
        from sglang.srt.server_args import (
            get_global_server_args,
            set_global_server_args_for_scheduler,
        )

        self.assertFalse(has_context())
        sentinel = object()
        set_global_server_args_for_scheduler(sentinel)
        self.assertIs(get_global_server_args(), sentinel)

    def test_shim_uninit_raises_exact_message(self):
        from sglang.srt.server_args import get_global_server_args

        with self.assertRaises(ValueError) as cm:
            get_global_server_args()
        self.assertEqual(str(cm.exception), "Global server args is not set yet!")

    def test_in_place_mutation_visible_through_both_handles(self):
        # identity guarantee: writing THROUGH the shim's return value is visible on
        # ctx.server_args (the same live object) — the basis for keeping the 9+ in-
        # place ModelRunner mutations working without flipping their call-sites.
        from types import SimpleNamespace

        from sglang.srt.server_args import get_global_server_args

        sa = SimpleNamespace(use_mla_backend=False)
        set_context(RuntimeContext(server_args=sa))
        get_global_server_args().use_mla_backend = True
        self.assertTrue(get_context().server_args.use_mla_backend)
        self.assertIs(get_global_server_args(), sa)


class TestModelOverrideContract(unittest.TestCase):
    """P2a: the bounded model-override write surface — the 24-field whitelist, the
    transactional apply_model_overrides, and the scoped ctx.override_server_args."""

    def setUp(self):
        reset_context()

    def tearDown(self):
        reset_context()

    def test_whitelist_membership(self):
        self.assertEqual(len(MODEL_OVERRIDABLE_FIELDS), 24)
        for f in ("use_mla_backend", "attention_backend", "moe_a2a_backend"):
            self.assertIn(f, MODEL_OVERRIDABLE_FIELDS)
        # excluded by design: runtime-reload + B3/B4 runtime-driven fields
        for f in ("model_path", "load_format", "kv_cache_dtype", "enable_torch_compile"):
            self.assertNotIn(f, MODEL_OVERRIDABLE_FIELDS)

    def test_apply_sets_whitelisted_fields(self):
        from types import SimpleNamespace

        sa = SimpleNamespace(use_mla_backend=False, page_size=1)
        apply_model_overrides(sa, {"use_mla_backend": True, "page_size": 64})
        self.assertTrue(sa.use_mla_backend)
        self.assertEqual(sa.page_size, 64)

    def test_apply_rejects_non_whitelisted_key(self):
        from types import SimpleNamespace

        sa = SimpleNamespace(bogus=0)
        with self.assertRaises(KeyError):
            apply_model_overrides(sa, {"bogus": 1})

    def test_apply_is_transactional_no_partial_write(self):
        # a bad key aborts BEFORE any valid key is mutated
        from types import SimpleNamespace

        sa = SimpleNamespace(use_mla_backend=False)
        with self.assertRaises(KeyError):
            apply_model_overrides(sa, {"use_mla_backend": True, "bogus": 1})
        self.assertFalse(sa.use_mla_backend)  # not left flipped

    def test_override_server_args_scoped_and_restores(self):
        from types import SimpleNamespace

        sa = SimpleNamespace(use_mla_backend=False, page_size=1)
        set_context(RuntimeContext(server_args=sa))
        with get_context().override_server_args(use_mla_backend=True, page_size=64):
            self.assertTrue(sa.use_mla_backend)
            self.assertEqual(sa.page_size, 64)
        self.assertFalse(sa.use_mla_backend)  # restored
        self.assertEqual(sa.page_size, 1)

    def test_override_server_args_restores_on_exception(self):
        from types import SimpleNamespace

        class _Boom(Exception):
            pass

        sa = SimpleNamespace(use_mla_backend=False)
        set_context(RuntimeContext(server_args=sa))
        with self.assertRaises(_Boom):
            with get_context().override_server_args(use_mla_backend=True):
                raise _Boom()
        self.assertFalse(sa.use_mla_backend)  # restored despite exception

    def test_override_server_args_transactional_no_leak(self):
        from types import SimpleNamespace

        sa = SimpleNamespace(use_mla_backend=False)
        set_context(RuntimeContext(server_args=sa))
        with self.assertRaises(KeyError):
            with get_context().override_server_args(use_mla_backend=True, bogus=1):
                pass
        self.assertFalse(sa.use_mla_backend)  # invalid key aborted before mutating


class TestCaptureFlags(unittest.TestCase):
    """P2a: flags.capture is a NON-frozen group (B4 home) — freeze() must NOT lock
    it, so the capture-time enable_torch_compile write stays legal post-freeze."""

    def setUp(self):
        reset_context()

    def tearDown(self):
        reset_context()

    def test_capture_group_present_and_default_none(self):
        set_context(RuntimeContext())
        self.assertIsNone(get_flags().capture.enable_torch_compile)

    def test_capture_writable_after_freeze(self):
        set_context(RuntimeContext())
        with mock.patch(_PREDICATE, return_value=False):
            get_context().freeze()
        # static groups are locked now; capture must remain writable (B4 post-freeze)
        with self.assertRaises(RuntimeError):
            get_flags().attn.use_mla_backend = True
        get_flags().capture.enable_torch_compile = False  # legal post-freeze
        self.assertEqual(get_flags().capture.enable_torch_compile, False)


if __name__ == "__main__":
    unittest.main()
