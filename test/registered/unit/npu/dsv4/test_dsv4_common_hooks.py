"""Unit tests for dsv4_common_hooks module-level functions."""

import sys
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

# Mock NPU-only modules before importing the source module.
for _ in (
    "torch_npu",
    "torch_npu.contrib",
    "sgl_kernel_npu",
    "sgl_kernel_npu.mem_cache",
    "sgl_kernel_npu.mem_cache.allocator",
    "sglang.srt.utils.hf_transformers_patches",
    "sglang.global_config",
    "sglang.lang.api",
    "sglang.lang.backend.runtime_endpoint",
    "sglang.lang.choices",
    "sglang.utils",
    "sglang.srt.configs",
    "sglang.srt.configs.model_config",
    "sglang.srt.dllm.config",
    "sglang.srt.layers.attention.base_attn_backend",
    "sglang.srt.layers.radix_attention",
    "sglang.srt.layers.utils.cp_utils",
    "sglang.srt.mem_cache",
    "sglang.srt.mem_cache.swa_memory_pool",
    "sglang.srt.mem_cache.allocation",
    "sglang.srt.mem_cache.allocator",
    "sglang.srt.model_executor",
    "sglang.srt.hardware_backend.npu.allocator_npu",
    # NOTE: dsv4_common_hooks is NOT mocked — it is the module under test.
    "sglang.srt.runtime_context",
    "sglang.srt.speculative.spec_info",
):
    sys.modules.setdefault(_, MagicMock())
# triton may be unavailable on CPU/Windows; use the real one if importable,
# otherwise mock it as a package so triton.backends / triton.language resolve.
try:
    import triton  # noqa: F401
except ImportError:
    import types as _types

    _triton = _types.ModuleType("triton")
    _triton.__path__ = []
    sys.modules["triton"] = _triton
    sys.modules.setdefault("triton.backends", MagicMock())
    sys.modules.setdefault("triton.language", MagicMock())

# get_schedule().c128_page_size must be a real int for dsv4_common_hooks.
sys.modules["sglang.srt.runtime_context"].get_schedule = lambda: SimpleNamespace(
    c128_page_size=1
)


# --- DSV4OutCacheLoc dataclass (used to create bundles in tests) ---


@dataclass
class DSV4OutCacheLoc:
    out_full_loc: "object"
    out_swa_loc: "object"
    out_c4_loc: "object"
    out_c128_loc: "object"


# sglang.version
_ver = type(sys)("sglang.version")
_ver.__version__ = "0.0.0.dev0"
sys.modules["sglang.version"] = _ver

import torch  # noqa: E402

# Clear any sibling test module's mock so we import the real module under test.
sys.modules.pop("sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks", None)

from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (  # noqa: E402
    _write_dsv4_tables,
    _write_per_req,
    _write_per_req_slice,
    dsv4_prealloc_kwargs,
    dsv4_unwrap_prealloc,
    maybe_build_dsv4_verify_bundle,
    maybe_write_dsv4_decode,
    maybe_write_dsv4_extend,
    write_dsv4_prealloc_tables,
)

_MOD = "sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks"


from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=3, suite="base-a-test-1-npu-a2")


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _make_bundle(
    *,
    out_full_loc=None,
    out_swa_loc=None,
    out_c4_loc=None,
    out_c128_loc=None,
):
    """Create a DSV4OutCacheLoc bundle with sensible tensor defaults."""
    return DSV4OutCacheLoc(
        out_full_loc=(
            out_full_loc if out_full_loc is not None else torch.tensor([1, 2, 3])
        ),
        out_swa_loc=(
            out_swa_loc if out_swa_loc is not None else torch.tensor([10, 11, 12])
        ),
        out_c4_loc=(out_c4_loc if out_c4_loc is not None else torch.tensor([20, 21])),
        out_c128_loc=(out_c128_loc if out_c128_loc is not None else torch.tensor([30])),
    )


def _make_pool(n_reqs=2, max_len=64, device="cpu"):
    """Fake req_to_token_pool with a write_c128 MagicMock and 2D int32 tables."""
    pool = SimpleNamespace()
    pool.write_c128 = MagicMock()
    base = torch.arange(n_reqs * max_len, dtype=torch.int32, device=device).reshape(
        n_reqs, max_len
    )
    pool.req_to_c128_sidecar = base.clone()
    pool.c128_page_size = 1
    return pool


def _make_req(*, pool_idx=0):
    """Fake Req with a pool index."""
    req = SimpleNamespace()
    req.req_pool_idx = pool_idx
    return req


def _make_batch(
    *,
    bundle=None,
    pool=None,
    req_pool_indices_cpu=None,
    reqs=None,
    seq_lens_cpu=None,
    token_to_kv_pool_allocator=None,
    out_cache_loc=None,
):
    """Fake ScheduleBatch."""
    batch = SimpleNamespace()
    batch.out_cache_loc_dsv4 = bundle
    batch.req_to_token_pool = pool
    indices = (
        req_pool_indices_cpu
        if req_pool_indices_cpu is not None
        else torch.tensor([0, 1])
    )
    batch.req_pool_indices_cpu = indices
    batch.req_pool_indices = MagicMock()
    batch.req_pool_indices.cpu.return_value = indices
    batch.reqs = reqs if reqs is not None else [_make_req(), _make_req()]
    batch.seq_lens_cpu = (
        seq_lens_cpu if seq_lens_cpu is not None else torch.tensor([8, 12])
    )
    if token_to_kv_pool_allocator is None:
        token_to_kv_pool_allocator = MagicMock()
        token_to_kv_pool_allocator.translate_loc_from_full_to_swa = lambda x: x
    batch.token_to_kv_pool_allocator = token_to_kv_pool_allocator
    batch.out_cache_loc = (
        out_cache_loc if out_cache_loc is not None else torch.tensor([1, 2, 3])
    )
    return batch


# ---------------------------------------------------------------------------
#  _write_per_req
# ---------------------------------------------------------------------------


class TestWritePerReq(unittest.TestCase):
    """Tests for the module-level _write_per_req helper."""

    def test_none_flat_loc_is_noop(self):
        """flat_loc None -> write_fn never called."""
        write_fn = MagicMock()
        _write_per_req(write_fn, torch.tensor([0, 1]), None, lambda i: (0, 4))
        write_fn.assert_not_called()

    def test_empty_flat_loc_is_noop(self):
        """flat_loc empty -> write_fn never called."""
        write_fn = MagicMock()
        _write_per_req(
            write_fn,
            torch.tensor([0, 1]),
            torch.empty(0, dtype=torch.int64),
            lambda i: (0, 4),
        )
        write_fn.assert_not_called()

    def test_distributes_flat_loc_across_reqs(self):
        """Two reqs with windows [0,2) and [2,5) -> correct chunks written."""
        write_fn = MagicMock()
        flat_loc = torch.tensor([100, 101, 102, 103, 104])
        bounds = [(0, 2), (2, 5)]
        _write_per_req(write_fn, torch.tensor([3, 7]), flat_loc, lambda i: bounds[i])
        self.assertEqual(write_fn.call_count, 2)
        first_args = write_fn.call_args_list[0].args
        self.assertEqual(first_args[0], (3, slice(0, 2)))
        self.assertTrue(
            torch.equal(first_args[1], torch.tensor([100, 101], dtype=torch.int32))
        )
        second_args = write_fn.call_args_list[1].args
        self.assertEqual(second_args[0], (7, slice(2, 5)))
        self.assertTrue(
            torch.equal(
                second_args[1],
                torch.tensor([102, 103, 104], dtype=torch.int32),
            )
        )

    def test_zero_alloc_len_skipped(self):
        """Req with lo == hi -> skipped, no write_fn call."""
        write_fn = MagicMock()
        flat_loc = torch.tensor([100])
        _write_per_req(write_fn, torch.tensor([0, 1]), flat_loc, lambda i: (0, 0))
        write_fn.assert_not_called()

    def test_negative_alloc_len_clamped_to_zero(self):
        """hi < lo -> alloc_len clamped to 0, skipped."""
        write_fn = MagicMock()
        flat_loc = torch.tensor([100])
        _write_per_req(write_fn, torch.tensor([0]), flat_loc, lambda i: (5, 2))
        write_fn.assert_not_called()

    def test_chunk_converted_to_int32(self):
        """Chunks are converted to int32 regardless of flat_loc dtype."""
        write_fn = MagicMock()
        flat_loc = torch.tensor([100, 101], dtype=torch.int64)
        _write_per_req(write_fn, torch.tensor([0]), flat_loc, lambda i: (0, 2))
        self.assertEqual(write_fn.call_args.args[1].dtype, torch.int32)

    def test_mixed_zero_and_nonzero(self):
        """Batch with one zero and one nonzero req -> only nonzero writes."""
        write_fn = MagicMock()
        flat_loc = torch.tensor([100, 101])
        bounds = [(0, 0), (0, 2)]
        _write_per_req(write_fn, torch.tensor([0, 1]), flat_loc, lambda i: bounds[i])
        self.assertEqual(write_fn.call_count, 1)
        self.assertEqual(write_fn.call_args.args[0], (1, slice(0, 2)))


# ---------------------------------------------------------------------------
#  _write_per_req_slice
# ---------------------------------------------------------------------------


class TestWritePerReqSlice(unittest.TestCase):
    """Tests for the module-level _write_per_req_slice helper."""

    def test_ratio_one_uses_raw_positions(self):
        """ratio=1 -> bounds are (prefix[i], seq[i])."""
        write_fn = MagicMock()
        flat_loc = torch.tensor([10, 11, 12, 13])
        _write_per_req_slice(
            write_fn,
            torch.tensor([0, 1]),
            prefix_lens_cpu=torch.tensor([0, 2]),
            seq_lens_cpu=torch.tensor([2, 4]),
            flat_loc=flat_loc,
            ratio=1,
        )
        self.assertEqual(write_fn.call_count, 2)
        first_args = write_fn.call_args_list[0].args
        self.assertEqual(first_args[0], (0, slice(0, 2)))
        second_args = write_fn.call_args_list[1].args
        self.assertEqual(second_args[0], (1, slice(2, 4)))

    def test_ratio_four_uses_compressed_positions(self):
        """ratio=4 -> bounds are (prefix//4, seq//4)."""
        write_fn = MagicMock()
        flat_loc = torch.tensor([20, 21])
        _write_per_req_slice(
            write_fn,
            torch.tensor([0, 1]),
            prefix_lens_cpu=torch.tensor([0, 8]),
            seq_lens_cpu=torch.tensor([4, 12]),
            flat_loc=flat_loc,
            ratio=4,
        )
        self.assertEqual(write_fn.call_count, 2)
        first_args = write_fn.call_args_list[0].args
        self.assertEqual(first_args[0], (0, slice(0, 1)))
        second_args = write_fn.call_args_list[1].args
        self.assertEqual(second_args[0], (1, slice(2, 3)))

    def test_ratio_128_uses_compressed_positions(self):
        """ratio=128 -> bounds are (prefix//128, seq//128)."""
        write_fn = MagicMock()
        flat_loc = torch.tensor([30])
        _write_per_req_slice(
            write_fn,
            torch.tensor([0]),
            prefix_lens_cpu=torch.tensor([0]),
            seq_lens_cpu=torch.tensor([256]),
            flat_loc=flat_loc,
            ratio=128,
        )
        self.assertEqual(write_fn.call_count, 1)
        self.assertEqual(write_fn.call_args.args[0], (0, slice(0, 2)))

    def test_none_flat_loc_is_noop(self):
        """flat_loc None -> no-op (delegates to _write_per_req)."""
        write_fn = MagicMock()
        _write_per_req_slice(
            write_fn,
            torch.tensor([0]),
            prefix_lens_cpu=torch.tensor([0]),
            seq_lens_cpu=torch.tensor([8]),
            flat_loc=None,
            ratio=4,
        )
        write_fn.assert_not_called()

    def test_same_ratio_bucket_skipped(self):
        """prefix and seq in same ratio bucket -> alloc_len 0, skipped."""
        write_fn = MagicMock()
        flat_loc = torch.tensor([20])
        _write_per_req_slice(
            write_fn,
            torch.tensor([0]),
            prefix_lens_cpu=torch.tensor([4]),
            seq_lens_cpu=torch.tensor([5]),
            flat_loc=flat_loc,
            ratio=4,
        )
        write_fn.assert_not_called()


# ---------------------------------------------------------------------------
#  maybe_write_dsv4_extend
# ---------------------------------------------------------------------------


class TestMaybeWriteDsv4Extend(unittest.TestCase):
    """Tests for maybe_write_dsv4_extend."""

    def test_bundle_none_is_noop(self):
        """batch.out_cache_loc_dsv4 None -> no writes."""
        batch = _make_batch(bundle=None, pool=_make_pool())
        maybe_write_dsv4_extend(
            batch,
            torch.tensor([0, 1]),
            torch.tensor([0, 0]),
            torch.tensor([8, 12]),
        )
        batch.req_to_token_pool.write_c128.assert_not_called()

    def test_pool_without_write_c128_is_noop(self):
        """Pool lacks write_c128 -> no-op."""
        batch = _make_batch(
            bundle=_make_bundle(),
            pool=SimpleNamespace(),
        )
        maybe_write_dsv4_extend(
            batch,
            torch.tensor([0, 1]),
            torch.tensor([0, 0]),
            torch.tensor([8, 12]),
        )

    def test_writes_c128(self):
        """Normal extend -> only write_c128 is called (C128 sidecar write)."""
        pool = _make_pool()
        bundle = _make_bundle(
            out_swa_loc=torch.arange(256, dtype=torch.int64),
            out_c4_loc=torch.arange(64, dtype=torch.int64),
            out_c128_loc=torch.tensor([30, 31], dtype=torch.int64),
        )
        batch = _make_batch(
            bundle=bundle,
            pool=pool,
            req_pool_indices_cpu=torch.tensor([0]),
            reqs=[_make_req(pool_idx=0)],
            seq_lens_cpu=torch.tensor([256]),
        )
        maybe_write_dsv4_extend(
            batch,
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([256]),
        )
        pool.write_c128.assert_called_once()


# ---------------------------------------------------------------------------
#  maybe_write_dsv4_decode
# ---------------------------------------------------------------------------


class TestMaybeWriteDsv4Decode(unittest.TestCase):
    """Tests for maybe_write_dsv4_decode."""

    def test_bundle_none_is_noop(self):
        """batch.out_cache_loc_dsv4 None -> no writes."""
        batch = _make_batch(bundle=None, pool=_make_pool())
        maybe_write_dsv4_decode(batch, torch.tensor([9]), 1)
        batch.req_to_token_pool.write_c128.assert_not_called()

    def test_pool_without_write_c128_is_noop(self):
        """Pool lacks write_c128 -> no-op."""
        batch = _make_batch(
            bundle=_make_bundle(),
            pool=SimpleNamespace(),
        )
        maybe_write_dsv4_decode(batch, torch.tensor([9]), 1)

    def test_writes_c128(self):
        """Normal decode -> only write_c128 is called (C128 sidecar write)."""
        pool = _make_pool()
        bundle = _make_bundle(
            out_swa_loc=torch.tensor([10]),
            out_c4_loc=torch.tensor([20]),
            out_c128_loc=torch.tensor([30]),
        )
        batch = _make_batch(
            bundle=bundle,
            pool=pool,
            req_pool_indices_cpu=torch.tensor([0]),
            reqs=[_make_req(pool_idx=0)],
            seq_lens_cpu=torch.tensor([256]),
        )
        maybe_write_dsv4_decode(batch, torch.tensor([256]), 1)
        pool.write_c128.assert_called_once()

    def test_prefix_lens_clamped_to_zero(self):
        """seq_lens_cpu < token_per_req -> prefix_lens clamped to 0."""
        pool = _make_pool()
        bundle = _make_bundle(
            out_swa_loc=torch.tensor([10, 11]),
            out_c4_loc=torch.tensor([20]),
            out_c128_loc=torch.tensor([]),
        )
        batch = _make_batch(
            bundle=bundle,
            pool=pool,
            req_pool_indices_cpu=torch.tensor([0]),
            reqs=[_make_req(pool_idx=0)],
            seq_lens_cpu=torch.tensor([0]),
        )
        maybe_write_dsv4_decode(batch, torch.tensor([0]), 1)
        # prefix = max(0, 0-1) = 0; c128 bounds (0, 0) -> skipped
        pool.write_c128.assert_not_called()


# ---------------------------------------------------------------------------
#  maybe_build_dsv4_verify_bundle
# ---------------------------------------------------------------------------


class TestMaybeBuildDsv4VerifyBundle(unittest.TestCase):
    """Tests for maybe_build_dsv4_verify_bundle."""

    def test_pool_without_req_to_c128_sidecar_returns_none(self):
        """Pool lacks req_to_c128_sidecar -> returns None."""
        batch = _make_batch(
            bundle=_make_bundle(),
            pool=SimpleNamespace(),
        )
        result = maybe_build_dsv4_verify_bundle(batch, draft_token_num=4)
        self.assertIsNone(result)

    def test_reserve_bundle_none_returns_none(self):
        """batch.out_cache_loc_dsv4 None -> returns None."""
        pool = _make_pool()
        batch = _make_batch(
            bundle=None,
            pool=pool,
            req_pool_indices_cpu=torch.tensor([0]),
            seq_lens_cpu=torch.tensor([0]),
        )
        result = maybe_build_dsv4_verify_bundle(batch, draft_token_num=4)
        self.assertIsNone(result)

    def test_returns_same_type_as_reserve_bundle(self):
        """Result is same type as the reserve bundle."""
        pool = _make_pool(n_reqs=2, max_len=64)
        bundle = _make_bundle()
        batch = _make_batch(
            bundle=bundle,
            pool=pool,
            req_pool_indices_cpu=torch.tensor([0]),
            seq_lens_cpu=torch.tensor([0]),
            out_cache_loc=torch.tensor([1, 2, 3]),
        )
        result = maybe_build_dsv4_verify_bundle(batch, draft_token_num=4)
        self.assertIsInstance(result, DSV4OutCacheLoc)
        self.assertTrue(torch.equal(result.out_full_loc, torch.tensor([1, 2, 3])))

    def test_flatten_interval_correct_slices(self):
        """out_full_loc, out_swa_loc (passthrough), out_c4_loc derived from out_cache_loc."""
        pool = _make_pool(n_reqs=2, max_len=64)
        bundle = _make_bundle()
        batch = _make_batch(
            bundle=bundle,
            pool=pool,
            req_pool_indices_cpu=torch.tensor([0]),
            seq_lens_cpu=torch.tensor([0]),
            out_cache_loc=torch.tensor([3, 7, 11], dtype=torch.int64),
        )
        result = maybe_build_dsv4_verify_bundle(batch, draft_token_num=4)
        self.assertTrue(
            torch.equal(
                result.out_full_loc,
                torch.tensor([3, 7, 11], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                result.out_swa_loc,
                torch.tensor([3, 7, 11], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(result.out_c4_loc, torch.tensor([0, 1, 2], dtype=torch.int64))
        )

    def test_empty_intervals_return_empty_tensor(self):
        """All intervals empty (draft_token_num=0) -> empty tensors."""
        pool = _make_pool(n_reqs=2, max_len=64)
        bundle = _make_bundle()
        batch = _make_batch(
            bundle=bundle,
            pool=pool,
            req_pool_indices_cpu=torch.tensor([0]),
            seq_lens_cpu=torch.tensor([4]),
            out_cache_loc=torch.tensor([], dtype=torch.int64),
        )
        result = maybe_build_dsv4_verify_bundle(batch, draft_token_num=0)
        self.assertEqual(result.out_swa_loc.numel(), 0)
        self.assertEqual(result.out_c4_loc.numel(), 0)
        self.assertEqual(result.out_c128_loc.numel(), 0)

    def test_multiple_reqs_concatenated(self):
        """Two reqs with non-empty c128 intervals -> c128 chunks concatenated."""
        pool = _make_pool(n_reqs=2, max_len=64)
        bundle = _make_bundle()
        batch = _make_batch(
            bundle=bundle,
            pool=pool,
            req_pool_indices_cpu=torch.tensor([0, 1]),
            seq_lens_cpu=torch.tensor([128, 128]),
            out_cache_loc=torch.tensor([99]),
        )
        result = maybe_build_dsv4_verify_bundle(batch, draft_token_num=128)
        # c128: req0 table[0, 1:2]=[1] + req1 table[1, 1:2]=[65]
        expected_c128 = torch.tensor([1, 65], dtype=torch.int32)
        self.assertTrue(torch.equal(result.out_c128_loc, expected_c128))

    def test_c128_interval_with_large_seq(self):
        """c128 interval correctly sliced when seq crosses 128 boundary."""
        pool = _make_pool(n_reqs=2, max_len=512)
        bundle = _make_bundle()
        batch = _make_batch(
            bundle=bundle,
            pool=pool,
            req_pool_indices_cpu=torch.tensor([0]),
            seq_lens_cpu=torch.tensor([128]),
            out_cache_loc=torch.tensor([99]),
        )
        result = maybe_build_dsv4_verify_bundle(batch, draft_token_num=128)
        # c128: start=128//128=1, end=256//128=2 -> table[0, 1:2]
        self.assertEqual(result.out_c128_loc.numel(), 1)
        self.assertEqual(result.out_c128_loc.item(), 1)


# ---------------------------------------------------------------------------
#  _write_dsv4_tables
# ---------------------------------------------------------------------------


class TestWriteDsv4Tables(unittest.TestCase):
    """Tests for the shared _write_dsv4_tables helper."""

    def test_write_c128_called(self):
        """Normal call -> write_c128 called with out_c128_loc."""
        pool = _make_pool()
        bundle = _make_bundle(
            out_c128_loc=torch.tensor([1, 2, 3], dtype=torch.int64),
        )
        _write_dsv4_tables(
            pool,
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([128]),
            bundle,
        )
        pool.write_c128.assert_called_once()

    def test_empty_c128_loc_no_write(self):
        """Empty out_c128_loc -> write_c128 not called."""
        pool = _make_pool()
        bundle = _make_bundle(
            out_c128_loc=torch.tensor([], dtype=torch.int64),
        )
        _write_dsv4_tables(
            pool,
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([4]),
            bundle,
        )
        pool.write_c128.assert_not_called()


# ---------------------------------------------------------------------------
#  write_dsv4_prealloc_tables
# ---------------------------------------------------------------------------


class TestWriteDsv4PreallocTables(unittest.TestCase):
    """Tests for write_dsv4_prealloc_tables (disagg prealloc path)."""

    def test_bundle_none_is_noop(self):
        """Bundle None -> no-op."""
        pool = _make_pool()
        req = _make_req(pool_idx=0)
        write_dsv4_prealloc_tables(pool, req, 0, 4, None)
        pool.write_c128.assert_not_called()

    def test_pool_without_write_c128_is_noop(self):
        """Pool lacks write_c128 -> no-op."""
        pool = SimpleNamespace()
        req = _make_req(pool_idx=0)
        write_dsv4_prealloc_tables(pool, req, 0, 4, _make_bundle())
        # No crash.

    def test_writes_tables_for_single_req(self):
        """Normal: writes c128 table for one req."""
        pool = _make_pool()
        req = _make_req(pool_idx=3)
        bundle = _make_bundle(
            out_c128_loc=torch.arange(128, dtype=torch.int64),
        )
        write_dsv4_prealloc_tables(pool, req, 0, 128, bundle)
        pool.write_c128.assert_called_once()


# ---------------------------------------------------------------------------
#  dsv4_unwrap_prealloc
# ---------------------------------------------------------------------------


class TestDsv4UnwrapPrealloc(unittest.TestCase):
    """Tests for dsv4_unwrap_prealloc."""

    def test_none_kv_loc_returns_none(self):
        """kv_loc None -> returns None unchanged."""
        pool = _make_pool()
        req = _make_req(pool_idx=0)
        result = dsv4_unwrap_prealloc(None, pool, req, 0, 4)
        self.assertIsNone(result)

    def test_plain_tensor_passes_through(self):
        """Non-bundle (plain tensor) -> passes through unchanged."""
        pool = _make_pool()
        req = _make_req(pool_idx=0)
        kv_loc = torch.tensor([1, 2, 3])
        result = dsv4_unwrap_prealloc(kv_loc, pool, req, 0, 4)
        self.assertIs(result, kv_loc)

    def test_bundle_unwrapped_and_tables_written(self):
        """Bundle -> writes tables, returns out_full_loc."""
        pool = _make_pool()
        req = _make_req(pool_idx=0)
        bundle = _make_bundle(
            out_c128_loc=torch.arange(128, dtype=torch.int64),
        )
        result = dsv4_unwrap_prealloc(bundle, pool, req, 0, 128)
        self.assertTrue(torch.equal(result, torch.tensor([1, 2, 3])))
        pool.write_c128.assert_called_once()


# ---------------------------------------------------------------------------
#  dsv4_prealloc_kwargs
# ---------------------------------------------------------------------------


class TestDsv4PreallocKwargs(unittest.TestCase):
    """Tests for dsv4_prealloc_kwargs."""

    def test_non_dsv4_allocator_returns_empty(self):
        """Allocator without c128_attn_allocator -> empty dict."""
        allocator = SimpleNamespace()
        req = _make_req(pool_idx=0)
        pool = _make_pool()
        result = dsv4_prealloc_kwargs(allocator, req, 4, pool, device="cpu")
        self.assertEqual(result, {})

    def test_dsv4_allocator_returns_kwargs(self):
        """DSV4 allocator -> returns req_pool_indices and req_to_token_pool."""
        allocator = MagicMock()
        req = _make_req(pool_idx=5)
        pool = _make_pool()
        result = dsv4_prealloc_kwargs(allocator, req, 8, pool, device="cpu")
        self.assertIn("req_pool_indices", result)
        self.assertIn("req_to_token_pool", result)
        self.assertIs(result["req_to_token_pool"], pool)
        self.assertEqual(result["req_pool_indices"].item(), 5)


if __name__ == "__main__":
    unittest.main()
