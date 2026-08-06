"""Unit tests for dsv4_common_hooks module-level functions."""

import os
import sys
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock

# Ensure the workspace-local sglang copy (D:\sglang\python) is imported,
# not a stale editable-install copy that may differ.
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "python"
    ),
)

# ---- Mock modules unavailable on CPU/Windows before any sglang import ----
_mock = MagicMock()
for _mod in (
    "torch_npu",
    "triton",
    "triton.language",
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
    "sglang.srt.utils",
    "aiohttp",
    "sglang.test.ci.ci_register",
    "sglang.test.test_utils",
):
    sys.modules[_mod] = _mock

# --- DSV4OutCacheLoc dataclass (used to create bundles in tests) ---


@dataclass
class DSV4OutCacheLoc:
    out_full_loc: "object"
    out_swa_loc: "object"
    out_c4_loc: "object"
    out_c128_loc: "object"
    out_c4_state_loc: Optional[object] = None
    out_c128_state_loc: Optional[object] = None


# sglang.version
_ver = type(sys)("sglang.version")
_ver.__version__ = "0.0.0.dev0"
sys.modules["sglang.version"] = _ver

import torch  # noqa: E402

from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (  # noqa: E402
    _free_state_range,
    _write_dsv4_tables,
    _write_per_req,
    _write_per_req_slice,
    _write_state_tail_per_req,
    dsv4_prealloc_kwargs,
    dsv4_unwrap_prealloc,
    maybe_build_dsv4_verify_bundle,
    maybe_evict_dsv4_state,
    maybe_evict_dsv4_state_on_swa,
    maybe_write_dsv4_decode,
    maybe_write_dsv4_extend,
    write_dsv4_prealloc_tables,
)

_MOD = "sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks"


def register_npu_ci(est_time, suite=None, nightly=False, disabled=None):
    def decorator(cls):
        return cls

    return decorator


class CustomTestCase(unittest.TestCase):
    pass


register_npu_ci(est_time=3, suite="stage-a-unit-test-npu")


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _make_bundle(
    *,
    out_full_loc=None,
    out_swa_loc=None,
    out_c4_loc=None,
    out_c128_loc=None,
    out_c4_state_loc=None,
    out_c128_state_loc=None,
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
        out_c4_state_loc=out_c4_state_loc,
        out_c128_state_loc=out_c128_state_loc,
    )


def _make_pool(n_reqs=2, max_len=64, device="cpu", with_state=True):
    """Fake req_to_token_pool with write MagicMocks and 2D int32 tables."""
    pool = SimpleNamespace()
    pool.write_swa = MagicMock()
    pool.write_c4 = MagicMock()
    pool.write_c128 = MagicMock()
    if with_state:
        pool.write_c4_state = MagicMock()
        pool.write_c128_state = MagicMock()
    base = torch.arange(n_reqs * max_len, dtype=torch.int32, device=device).reshape(
        n_reqs, max_len
    )
    pool.req_to_token_swa = base.clone()
    pool.req_to_token_c4 = base.clone()
    pool.req_to_token_c128 = base.clone()
    pool.req_to_token_c4_state = base.clone()
    pool.req_to_token_c128_state = base.clone()
    return pool


def _make_req(
    *,
    c4_off=0,
    c128_off=0,
    c4_write_off=None,
    c128_write_off=None,
    pool_idx=0,
):
    """Fake Req with compress-state alloc/write offsets."""
    req = SimpleNamespace()
    req.c4_state_alloc_offset = c4_off
    req.c128_state_alloc_offset = c128_off
    if c4_write_off is not None:
        req.c4_state_write_offset = c4_write_off
    if c128_write_off is not None:
        req.c128_state_write_offset = c128_write_off
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
    tree_cache=None,
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
    batch.token_to_kv_pool_allocator = token_to_kv_pool_allocator
    batch.tree_cache = tree_cache or SimpleNamespace(page_size=1)
    batch.out_cache_loc = (
        out_cache_loc if out_cache_loc is not None else torch.tensor([1, 2, 3])
    )
    return batch


def _make_allocator(*, with_state=True):
    """Fake token_to_kv_pool_allocator with c4/c128 state allocators."""
    alloc = SimpleNamespace()
    if with_state:
        alloc.c4_state_attn_allocator = MagicMock()
        alloc.c128_state_attn_allocator = MagicMock()
    else:
        alloc.c4_state_attn_allocator = None
        alloc.c128_state_attn_allocator = None
    return alloc


# ---------------------------------------------------------------------------
#  _write_per_req
# ---------------------------------------------------------------------------


class TestWritePerReq(CustomTestCase):
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


class TestWritePerReqSlice(CustomTestCase):
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
#  _write_state_tail_per_req
# ---------------------------------------------------------------------------


class TestWriteStateTailPerReq(CustomTestCase):
    """Tests for the module-level _write_state_tail_per_req helper."""

    def test_writes_offset_to_seq(self):
        """Bounds are (state_alloc_offsets[i], seq_lens_cpu[i])."""
        write_fn = MagicMock()
        flat_loc = torch.tensor([50, 51, 52])
        _write_state_tail_per_req(
            write_fn,
            torch.tensor([0, 1]),
            state_alloc_offsets=[2, 0],
            seq_lens_cpu=torch.tensor([4, 3]),
            flat_loc=flat_loc,
        )
        self.assertEqual(write_fn.call_count, 2)
        first_args = write_fn.call_args_list[0].args
        self.assertEqual(first_args[0], (0, slice(2, 4)))
        self.assertTrue(
            torch.equal(first_args[1], torch.tensor([50, 51], dtype=torch.int32))
        )
        second_args = write_fn.call_args_list[1].args
        self.assertEqual(second_args[0], (1, slice(0, 3)))
        self.assertTrue(
            torch.equal(second_args[1], torch.tensor([52], dtype=torch.int32))
        )

    def test_offset_ge_seq_skipped(self):
        """offset >= seq -> alloc_len 0, skipped."""
        write_fn = MagicMock()
        flat_loc = torch.tensor([50])
        _write_state_tail_per_req(
            write_fn,
            torch.tensor([0]),
            state_alloc_offsets=[5],
            seq_lens_cpu=torch.tensor([5]),
            flat_loc=flat_loc,
        )
        write_fn.assert_not_called()

    def test_none_flat_loc_is_noop(self):
        """flat_loc None -> no-op."""
        write_fn = MagicMock()
        _write_state_tail_per_req(
            write_fn,
            torch.tensor([0]),
            state_alloc_offsets=[0],
            seq_lens_cpu=torch.tensor([4]),
            flat_loc=None,
        )
        write_fn.assert_not_called()


# ---------------------------------------------------------------------------
#  maybe_write_dsv4_extend
# ---------------------------------------------------------------------------


class TestMaybeWriteDsv4Extend(CustomTestCase):
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
        batch.req_to_token_pool.write_swa.assert_not_called()
        batch.req_to_token_pool.write_c4.assert_not_called()
        batch.req_to_token_pool.write_c128.assert_not_called()

    def test_pool_without_write_c4_is_noop(self):
        """Pool lacks write_c4 -> no-op."""
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

    def test_writes_swa_c4_c128(self):
        """Normal extend -> swa, c4, c128 writes all called."""
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
        self.assertEqual(pool.write_swa.call_count, 1)
        self.assertEqual(pool.write_c4.call_count, 1)
        self.assertEqual(pool.write_c128.call_count, 1)

    def test_state_writes_when_state_loc_present(self):
        """out_c4_state_loc not None and pool has write_c4_state -> state writes."""
        pool = _make_pool()
        bundle = _make_bundle(
            out_swa_loc=torch.arange(256, dtype=torch.int64),
            out_c4_loc=torch.arange(64, dtype=torch.int64),
            out_c128_loc=torch.tensor([30, 31], dtype=torch.int64),
            out_c4_state_loc=torch.arange(256, dtype=torch.int64),
            out_c128_state_loc=torch.arange(256, dtype=torch.int64),
        )
        batch = _make_batch(
            bundle=bundle,
            pool=pool,
            req_pool_indices_cpu=torch.tensor([0]),
            reqs=[_make_req(c4_off=0, c128_off=0, pool_idx=0)],
            seq_lens_cpu=torch.tensor([256]),
        )
        maybe_write_dsv4_extend(
            batch,
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([256]),
        )
        pool.write_c4_state.assert_called_once()
        pool.write_c128_state.assert_called_once()

    def test_state_writes_skipped_when_state_loc_none(self):
        """out_c4_state_loc None -> state writes skipped."""
        pool = _make_pool()
        bundle = _make_bundle(
            out_swa_loc=torch.arange(256, dtype=torch.int64),
            out_c4_loc=torch.arange(64, dtype=torch.int64),
            out_c128_loc=torch.tensor([30, 31], dtype=torch.int64),
            out_c4_state_loc=None,
            out_c128_state_loc=None,
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
        pool.write_c4_state.assert_not_called()
        pool.write_c128_state.assert_not_called()

    def test_default_state_offsets_from_reqs(self):
        """No explicit offsets -> uses c4_state_write_offset first, falls back to alloc_offset."""
        pool = _make_pool()
        bundle = _make_bundle(
            out_swa_loc=torch.arange(256, dtype=torch.int64),
            out_c4_loc=torch.arange(64, dtype=torch.int64),
            out_c128_loc=torch.tensor([30, 31], dtype=torch.int64),
            out_c4_state_loc=torch.arange(256, dtype=torch.int64),
            out_c128_state_loc=torch.arange(256, dtype=torch.int64),
        )
        reqs = [_make_req(c4_off=2, c128_off=3, pool_idx=0)]
        batch = _make_batch(
            bundle=bundle,
            pool=pool,
            req_pool_indices_cpu=torch.tensor([0]),
            reqs=reqs,
            seq_lens_cpu=torch.tensor([256]),
        )
        maybe_write_dsv4_extend(
            batch,
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([256]),
        )
        first_state_call = pool.write_c4_state.call_args_list[0].args
        self.assertEqual(first_state_call[0], (0, slice(2, 256)))

    def test_write_offset_takes_priority_over_alloc_offset(self):
        """c4_state_write_offset takes priority over c4_state_alloc_offset."""
        pool = _make_pool()
        bundle = _make_bundle(
            out_swa_loc=torch.arange(256, dtype=torch.int64),
            out_c4_loc=torch.arange(64, dtype=torch.int64),
            out_c128_loc=torch.tensor([30, 31], dtype=torch.int64),
            out_c4_state_loc=torch.arange(256, dtype=torch.int64),
            out_c128_state_loc=torch.arange(256, dtype=torch.int64),
        )
        reqs = [
            _make_req(
                c4_off=2, c128_off=3, c4_write_off=10, c128_write_off=11, pool_idx=0
            )
        ]
        batch = _make_batch(
            bundle=bundle,
            pool=pool,
            req_pool_indices_cpu=torch.tensor([0]),
            reqs=reqs,
            seq_lens_cpu=torch.tensor([256]),
        )
        maybe_write_dsv4_extend(
            batch,
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([256]),
        )
        first_state_call = pool.write_c4_state.call_args_list[0].args
        self.assertEqual(first_state_call[0], (0, slice(10, 256)))
        c128_state_call = pool.write_c128_state.call_args_list[0].args
        self.assertEqual(c128_state_call[0], (0, slice(11, 256)))

    def test_explicit_state_offsets_used(self):
        """Explicit c4_state_alloc_offsets override req defaults."""
        pool = _make_pool()
        bundle = _make_bundle(
            out_swa_loc=torch.arange(256, dtype=torch.int64),
            out_c4_loc=torch.arange(64, dtype=torch.int64),
            out_c128_loc=torch.tensor([30, 31], dtype=torch.int64),
            out_c4_state_loc=torch.arange(256, dtype=torch.int64),
            out_c128_state_loc=torch.arange(256, dtype=torch.int64),
        )
        batch = _make_batch(
            bundle=bundle,
            pool=pool,
            req_pool_indices_cpu=torch.tensor([0]),
            reqs=[_make_req(c4_off=99, c128_off=99, pool_idx=0)],
            seq_lens_cpu=torch.tensor([256]),
        )
        maybe_write_dsv4_extend(
            batch,
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([256]),
            c4_state_alloc_offsets=[3],
            c128_state_alloc_offsets=[5],
        )
        first_c4_state_call = pool.write_c4_state.call_args_list[0].args
        self.assertEqual(first_c4_state_call[0], (0, slice(3, 256)))

    def test_pool_without_state_write_methods_skips_state(self):
        """Pool lacks write_c4_state -> state writes skipped (hasattr check)."""
        pool = _make_pool(with_state=False)
        bundle = _make_bundle(
            out_swa_loc=torch.arange(256, dtype=torch.int64),
            out_c4_loc=torch.arange(64, dtype=torch.int64),
            out_c128_loc=torch.tensor([30, 31], dtype=torch.int64),
            out_c4_state_loc=torch.tensor([40]),
            out_c128_state_loc=torch.tensor([50]),
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
        # No crash -- state writes skipped since pool lacks write_c4_state.

    def test_req_without_offset_attr_defaults_zero(self):
        """Req without c4_state_alloc_offset -> getattr defaults to 0."""
        pool = _make_pool()
        bundle = _make_bundle(
            out_swa_loc=torch.arange(256, dtype=torch.int64),
            out_c4_loc=torch.arange(64, dtype=torch.int64),
            out_c128_loc=torch.tensor([30, 31], dtype=torch.int64),
            out_c4_state_loc=torch.arange(256, dtype=torch.int64),
            out_c128_state_loc=torch.arange(256, dtype=torch.int64),
        )
        req = SimpleNamespace(req_pool_idx=0)
        batch = _make_batch(
            bundle=bundle,
            pool=pool,
            req_pool_indices_cpu=torch.tensor([0]),
            reqs=[req],
            seq_lens_cpu=torch.tensor([256]),
        )
        maybe_write_dsv4_extend(
            batch,
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([256]),
        )
        first_state_call = pool.write_c4_state.call_args_list[0].args
        self.assertEqual(first_state_call[0], (0, slice(0, 256)))


# ---------------------------------------------------------------------------
#  maybe_write_dsv4_decode
# ---------------------------------------------------------------------------


class TestMaybeWriteDsv4Decode(CustomTestCase):
    """Tests for maybe_write_dsv4_decode."""

    def test_bundle_none_is_noop(self):
        """batch.out_cache_loc_dsv4 None -> no writes."""
        batch = _make_batch(bundle=None, pool=_make_pool())
        maybe_write_dsv4_decode(batch, torch.tensor([9]), 1)
        batch.req_to_token_pool.write_swa.assert_not_called()

    def test_pool_without_write_c4_is_noop(self):
        """Pool lacks write_c4 -> no-op."""
        batch = _make_batch(
            bundle=_make_bundle(),
            pool=SimpleNamespace(),
        )
        maybe_write_dsv4_decode(batch, torch.tensor([9]), 1)

    def test_writes_swa_c4_c128(self):
        """Normal decode -> swa, c4, c128 writes all called."""
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
        self.assertEqual(pool.write_swa.call_count, 1)
        self.assertEqual(pool.write_c4.call_count, 1)
        self.assertEqual(pool.write_c128.call_count, 1)

    def test_state_writes_when_state_loc_present(self):
        """out_c4_state_loc not None -> state writes with ratio=1."""
        pool = _make_pool()
        bundle = _make_bundle(
            out_swa_loc=torch.tensor([10]),
            out_c4_loc=torch.tensor([20]),
            out_c128_loc=torch.tensor([30]),
            out_c4_state_loc=torch.tensor([40]),
            out_c128_state_loc=torch.tensor([50]),
        )
        batch = _make_batch(
            bundle=bundle,
            pool=pool,
            req_pool_indices_cpu=torch.tensor([0]),
            reqs=[_make_req(pool_idx=0)],
            seq_lens_cpu=torch.tensor([256]),
        )
        maybe_write_dsv4_decode(batch, torch.tensor([256]), 1)
        pool.write_c4_state.assert_called_once()
        pool.write_c128_state.assert_called_once()

    def test_state_writes_skipped_when_state_loc_none(self):
        """out_c4_state_loc None -> state writes skipped."""
        pool = _make_pool()
        bundle = _make_bundle(
            out_swa_loc=torch.tensor([10]),
            out_c4_loc=torch.tensor([20]),
            out_c128_loc=torch.tensor([30]),
            out_c4_state_loc=None,
            out_c128_state_loc=None,
        )
        batch = _make_batch(
            bundle=bundle,
            pool=pool,
            req_pool_indices_cpu=torch.tensor([0]),
            reqs=[_make_req(pool_idx=0)],
            seq_lens_cpu=torch.tensor([256]),
        )
        maybe_write_dsv4_decode(batch, torch.tensor([256]), 1)
        pool.write_c4_state.assert_not_called()
        pool.write_c128_state.assert_not_called()

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
        # prefix = max(0, 0-1) = 0; swa bounds (0, 0) -> skipped
        pool.write_swa.assert_not_called()

    def test_pool_without_state_write_methods_skips_state(self):
        """Pool lacks write_c4_state -> state writes skipped."""
        pool = _make_pool(with_state=False)
        bundle = _make_bundle(
            out_swa_loc=torch.tensor([10]),
            out_c4_loc=torch.tensor([20]),
            out_c128_loc=torch.tensor([30]),
            out_c4_state_loc=torch.tensor([40]),
            out_c128_state_loc=torch.tensor([50]),
        )
        batch = _make_batch(
            bundle=bundle,
            pool=pool,
            req_pool_indices_cpu=torch.tensor([0]),
            reqs=[_make_req(pool_idx=0)],
            seq_lens_cpu=torch.tensor([256]),
        )
        maybe_write_dsv4_decode(batch, torch.tensor([256]), 1)
        # No crash -- state writes skipped.


# ---------------------------------------------------------------------------
#  maybe_build_dsv4_verify_bundle
# ---------------------------------------------------------------------------


class TestMaybeBuildDsv4VerifyBundle(CustomTestCase):
    """Tests for maybe_build_dsv4_verify_bundle."""

    def test_pool_without_req_to_token_c4_returns_none(self):
        """Pool lacks req_to_token_c4 -> returns None."""
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
        """swa interval [seq, seq+draft) sliced from the table."""
        pool = _make_pool(n_reqs=2, max_len=64)
        bundle = _make_bundle()
        batch = _make_batch(
            bundle=bundle,
            pool=pool,
            req_pool_indices_cpu=torch.tensor([0]),
            seq_lens_cpu=torch.tensor([0]),
            out_cache_loc=torch.tensor([99]),
        )
        result = maybe_build_dsv4_verify_bundle(batch, draft_token_num=4)
        # swa: table[0, 0:4] = [0, 1, 2, 3]
        self.assertTrue(
            torch.equal(
                result.out_swa_loc,
                torch.tensor([0, 1, 2, 3], dtype=torch.int32),
            )
        )
        # c4: table[0, 0:1] = [0]
        self.assertTrue(
            torch.equal(result.out_c4_loc, torch.tensor([0], dtype=torch.int32))
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
            out_cache_loc=torch.tensor([99]),
        )
        result = maybe_build_dsv4_verify_bundle(batch, draft_token_num=0)
        self.assertEqual(result.out_swa_loc.numel(), 0)
        self.assertEqual(result.out_c4_loc.numel(), 0)
        self.assertEqual(result.out_c128_loc.numel(), 0)
        self.assertEqual(result.out_c4_state_loc.numel(), 0)
        self.assertEqual(result.out_c128_state_loc.numel(), 0)

    def test_multiple_reqs_concatenated(self):
        """Two reqs with non-empty intervals -> chunks concatenated."""
        pool = _make_pool(n_reqs=2, max_len=64)
        bundle = _make_bundle()
        batch = _make_batch(
            bundle=bundle,
            pool=pool,
            req_pool_indices_cpu=torch.tensor([0, 1]),
            seq_lens_cpu=torch.tensor([0, 4]),
            out_cache_loc=torch.tensor([99]),
        )
        result = maybe_build_dsv4_verify_bundle(batch, draft_token_num=4)
        # swa: req0 table[0, 0:4]=[0,1,2,3] + req1 table[1, 4:8]=[68,69,70,71]
        expected_swa = torch.tensor([0, 1, 2, 3, 68, 69, 70, 71], dtype=torch.int32)
        self.assertTrue(torch.equal(result.out_swa_loc, expected_swa))
        # c4: req0 table[0, 0:1]=[0] + req1 table[1, 1:2]=[65]
        expected_c4 = torch.tensor([0, 65], dtype=torch.int32)
        self.assertTrue(torch.equal(result.out_c4_loc, expected_c4))

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
#  _free_state_range
# ---------------------------------------------------------------------------


class TestFreeStateRange(CustomTestCase):
    """Tests for the module-level _free_state_range helper."""

    def test_state_allocator_none_is_noop(self):
        """state_allocator None -> no-op."""
        pool = _make_pool()
        req = _make_req(c4_off=0, pool_idx=0)
        _free_state_range(
            None,
            pool,
            "req_to_token_c4_state",
            req,
            "c4_state_alloc_offset",
            10,
        )
        self.assertEqual(req.c4_state_alloc_offset, 0)

    def test_pool_without_table_attr_is_noop(self):
        """Pool lacks the table attr -> no-op."""
        allocator = MagicMock()
        pool = SimpleNamespace()
        req = _make_req(c4_off=0, pool_idx=0)
        _free_state_range(
            allocator,
            pool,
            "req_to_token_c4_state",
            req,
            "c4_state_alloc_offset",
            10,
        )
        allocator.free.assert_not_called()

    def test_watermark_le_offset_is_noop(self):
        """watermark <= offset -> no-op."""
        allocator = MagicMock()
        pool = _make_pool()
        req = _make_req(c4_off=5, pool_idx=0)
        _free_state_range(
            allocator,
            pool,
            "req_to_token_c4_state",
            req,
            "c4_state_alloc_offset",
            5,
        )
        allocator.free.assert_not_called()
        self.assertEqual(req.c4_state_alloc_offset, 5)

    def test_frees_slots_and_advances_offset(self):
        """Normal: frees [offset, watermark) and advances offset."""
        allocator = MagicMock()
        pool = _make_pool(n_reqs=2, max_len=64)
        req = _make_req(c4_off=2, pool_idx=0)
        _free_state_range(
            allocator,
            pool,
            "req_to_token_c4_state",
            req,
            "c4_state_alloc_offset",
            6,
        )
        allocator.free.assert_called_once()
        freed = allocator.free.call_args.args[0]
        self.assertTrue(
            torch.equal(freed, torch.tensor([2, 3, 4, 5], dtype=torch.int64))
        )
        self.assertEqual(req.c4_state_alloc_offset, 6)

    def test_offset_defaults_to_zero(self):
        """Req without offset attr -> getattr defaults to 0."""
        allocator = MagicMock()
        pool = _make_pool(n_reqs=2, max_len=64)
        req = SimpleNamespace(req_pool_idx=0)
        _free_state_range(
            allocator,
            pool,
            "req_to_token_c4_state",
            req,
            "c4_state_alloc_offset",
            4,
        )
        allocator.free.assert_called_once()
        freed = allocator.free.call_args.args[0]
        self.assertTrue(torch.equal(freed, torch.tensor([1, 2, 3], dtype=torch.int64)))
        self.assertEqual(req.c4_state_alloc_offset, 4)


# ---------------------------------------------------------------------------
#  maybe_evict_dsv4_state
# ---------------------------------------------------------------------------


class TestMaybeEvictDsv4State(CustomTestCase):
    """Tests for maybe_evict_dsv4_state."""

    def test_allocator_without_c4_state_attr_is_noop(self):
        """Allocator lacks c4_state_attn_allocator -> no-op."""
        batch = _make_batch(
            pool=_make_pool(),
            token_to_kv_pool_allocator=SimpleNamespace(),
        )
        req = _make_req(pool_idx=0)
        maybe_evict_dsv4_state(batch, req, pre_len=100)
        # No crash.

    def test_both_state_allocators_none_is_noop(self):
        """Both state allocators None -> no-op."""
        allocator = _make_allocator(with_state=False)
        batch = _make_batch(
            pool=_make_pool(),
            token_to_kv_pool_allocator=allocator,
        )
        req = _make_req(pool_idx=0)
        maybe_evict_dsv4_state(batch, req, pre_len=100)
        # No crash.

    def test_c4_eviction_frees_and_advances(self):
        """pre_len=30, page_size=1 -> c4_watermark=6, c128_watermark=0."""
        allocator = _make_allocator()
        pool = _make_pool(n_reqs=2, max_len=64)
        batch = _make_batch(
            pool=pool,
            token_to_kv_pool_allocator=allocator,
            tree_cache=SimpleNamespace(page_size=1),
        )
        req = _make_req(c4_off=0, c128_off=0, pool_idx=0)
        maybe_evict_dsv4_state(batch, req, pre_len=30)
        # c4_watermark = max(0, 30-24) = 6 -> free [0, 6), advance to 6
        allocator.c4_state_attn_allocator.free.assert_called_once()
        self.assertEqual(req.c4_state_alloc_offset, 6)
        # c128_watermark = max(0, 30-192) = 0 -> 0 <= 0 -> no-op
        allocator.c128_state_attn_allocator.free.assert_not_called()

    def test_both_c4_and_c128_evicted_at_large_pre_len(self):
        """pre_len=200, page_size=1 -> c4_watermark=176, c128_watermark=8."""
        allocator = _make_allocator()
        pool = _make_pool(n_reqs=2, max_len=256)
        batch = _make_batch(
            pool=pool,
            token_to_kv_pool_allocator=allocator,
            tree_cache=SimpleNamespace(page_size=1),
        )
        req = _make_req(c4_off=0, c128_off=0, pool_idx=0)
        maybe_evict_dsv4_state(batch, req, pre_len=200)
        allocator.c4_state_attn_allocator.free.assert_called_once()
        allocator.c128_state_attn_allocator.free.assert_called_once()
        self.assertEqual(req.c4_state_alloc_offset, 176)
        self.assertEqual(req.c128_state_alloc_offset, 8)

    def test_watermark_page_aligned(self):
        """pre_len=30, page_size=4 -> c4_watermark=(6//4)*4=4."""
        allocator = _make_allocator()
        pool = _make_pool(n_reqs=2, max_len=64)
        batch = _make_batch(
            pool=pool,
            token_to_kv_pool_allocator=allocator,
            tree_cache=SimpleNamespace(page_size=4),
        )
        req = _make_req(c4_off=0, pool_idx=0)
        maybe_evict_dsv4_state(batch, req, pre_len=30)
        # c4_watermark = (6 // 4) * 4 = 4 -> free [0, 4), advance to 4
        # table[0, 0:4] = [0,1,2,3], zero-slot filtered -> [1,2,3] (3 slots)
        freed = allocator.c4_state_attn_allocator.free.call_args.args[0]
        self.assertEqual(freed.numel(), 3)
        self.assertEqual(req.c4_state_alloc_offset, 4)

    def test_pre_len_below_c4_retention_no_eviction(self):
        """pre_len < 24 -> c4_watermark=0, no eviction."""
        allocator = _make_allocator()
        pool = _make_pool(n_reqs=2, max_len=64)
        batch = _make_batch(
            pool=pool,
            token_to_kv_pool_allocator=allocator,
            tree_cache=SimpleNamespace(page_size=1),
        )
        req = _make_req(c4_off=0, pool_idx=0)
        maybe_evict_dsv4_state(batch, req, pre_len=10)
        # c4_watermark = max(0, 10-24) = 0 -> 0 <= 0 -> no-op
        allocator.c4_state_attn_allocator.free.assert_not_called()

    def test_advances_from_nonzero_offset(self):
        """Existing offset=3, watermark=6 -> free [3, 6), advance to 6."""
        allocator = _make_allocator()
        pool = _make_pool(n_reqs=2, max_len=64)
        batch = _make_batch(
            pool=pool,
            token_to_kv_pool_allocator=allocator,
            tree_cache=SimpleNamespace(page_size=1),
        )
        req = _make_req(c4_off=3, c128_off=0, pool_idx=0)
        maybe_evict_dsv4_state(batch, req, pre_len=30)
        freed = allocator.c4_state_attn_allocator.free.call_args.args[0]
        self.assertTrue(torch.equal(freed, torch.tensor([3, 4, 5], dtype=torch.int64)))
        self.assertEqual(req.c4_state_alloc_offset, 6)


# ---------------------------------------------------------------------------
#  maybe_evict_dsv4_state_on_swa
# ---------------------------------------------------------------------------


class TestMaybeEvictDsv4StateOnSwa(CustomTestCase):
    """Tests for maybe_evict_dsv4_state_on_swa."""

    def test_allocator_without_c4_state_attr_is_noop(self):
        """Allocator lacks c4_state_attn_allocator -> no-op."""
        allocator = SimpleNamespace()
        pool = _make_pool()
        req = _make_req(pool_idx=0)
        maybe_evict_dsv4_state_on_swa(allocator, pool, req, 100)
        # No crash.

    def test_both_state_allocators_none_is_noop(self):
        """Both state allocators None -> no-op via _free_state_range."""
        allocator = _make_allocator(with_state=False)
        pool = _make_pool()
        req = _make_req(c4_off=0, c128_off=0, pool_idx=0)
        maybe_evict_dsv4_state_on_swa(allocator, pool, req, 100)
        # No crash -- _free_state_range returns early when allocator is None.

    def test_frees_c4_and_c128_state(self):
        """Normal: frees [0, watermark) for both c4 and c128."""
        allocator = _make_allocator()
        pool = _make_pool(n_reqs=2, max_len=64)
        req = _make_req(c4_off=0, c128_off=0, pool_idx=0)
        maybe_evict_dsv4_state_on_swa(allocator, pool, req, 10)
        allocator.c4_state_attn_allocator.free.assert_called_once()
        allocator.c128_state_attn_allocator.free.assert_called_once()
        self.assertEqual(req.c4_state_alloc_offset, 10)
        self.assertEqual(req.c128_state_alloc_offset, 10)

    def test_watermark_le_offset_is_noop(self):
        """watermark <= offset -> no free for that pool."""
        allocator = _make_allocator()
        pool = _make_pool(n_reqs=2, max_len=64)
        req = _make_req(c4_off=10, c128_off=10, pool_idx=0)
        maybe_evict_dsv4_state_on_swa(allocator, pool, req, 10)
        allocator.c4_state_attn_allocator.free.assert_not_called()
        allocator.c128_state_attn_allocator.free.assert_not_called()

    def test_advances_from_nonzero_offset(self):
        """offset=3, watermark=10 -> free [3, 10), advance to 10."""
        allocator = _make_allocator()
        pool = _make_pool(n_reqs=2, max_len=64)
        req = _make_req(c4_off=3, c128_off=3, pool_idx=0)
        maybe_evict_dsv4_state_on_swa(allocator, pool, req, 10)
        freed = allocator.c4_state_attn_allocator.free.call_args.args[0]
        self.assertTrue(
            torch.equal(
                freed,
                torch.tensor([3, 4, 5, 6, 7, 8, 9], dtype=torch.int64),
            )
        )
        self.assertEqual(req.c4_state_alloc_offset, 10)


# ---------------------------------------------------------------------------
#  _free_state_range — zero-slot filtering
# ---------------------------------------------------------------------------


class TestFreeStateRangeZeroFilter(CustomTestCase):
    """Tests for zero-slot filtering in _free_state_range."""

    def test_zero_slots_filtered_out(self):
        """Table entries with value 0 (uninitialized) are not freed."""
        allocator = MagicMock()
        pool = SimpleNamespace()
        table = torch.zeros(2, 8, dtype=torch.int32)
        table[0, 1] = 5
        table[0, 3] = 7
        pool.req_to_token_c4_state = table
        req = _make_req(c4_off=0, pool_idx=0)
        _free_state_range(
            allocator,
            pool,
            "req_to_token_c4_state",
            req,
            "c4_state_alloc_offset",
            4,
        )
        allocator.free.assert_called_once()
        freed = allocator.free.call_args.args[0]
        self.assertTrue(torch.equal(freed, torch.tensor([5, 7], dtype=torch.int64)))
        self.assertEqual(req.c4_state_alloc_offset, 4)

    def test_all_zero_slots_no_free_call(self):
        """All slots are 0 -> free not called, but offset still advances."""
        allocator = MagicMock()
        pool = SimpleNamespace()
        pool.req_to_token_c4_state = torch.zeros(2, 8, dtype=torch.int32)
        req = _make_req(c4_off=0, pool_idx=0)
        _free_state_range(
            allocator,
            pool,
            "req_to_token_c4_state",
            req,
            "c4_state_alloc_offset",
            4,
        )
        allocator.free.assert_not_called()
        self.assertEqual(req.c4_state_alloc_offset, 4)


# ---------------------------------------------------------------------------
#  _write_dsv4_tables
# ---------------------------------------------------------------------------


class TestWriteDsv4Tables(CustomTestCase):
    """Tests for the shared _write_dsv4_tables helper."""

    def test_bundle_none_is_noop(self):
        """Bundle None -> no writes (caller checks, but _write_dsv4_tables trusts caller)."""
        pool = _make_pool()
        bundle = _make_bundle()
        _write_dsv4_tables(
            pool,
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([4]),
            bundle,
            c4_state_offsets=[0],
            c128_state_offsets=[0],
        )
        self.assertEqual(pool.write_swa.call_count, 1)

    def test_writes_all_five_tables(self):
        """Normal call -> swa, c4, c128, c4_state, c128_state all written."""
        pool = _make_pool()
        bundle = _make_bundle(
            out_swa_loc=torch.arange(4, dtype=torch.int64),
            out_c4_loc=torch.tensor([20], dtype=torch.int64),
            out_c128_loc=torch.tensor([]),
            out_c4_state_loc=torch.arange(4, dtype=torch.int64),
            out_c128_state_loc=torch.tensor([]),
        )
        _write_dsv4_tables(
            pool,
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([4]),
            bundle,
            c4_state_offsets=[0],
            c128_state_offsets=[0],
        )
        pool.write_swa.assert_called_once()
        pool.write_c4.assert_called_once()
        # c128 has empty flat_loc -> _write_per_req returns early
        pool.write_c128.assert_not_called()
        pool.write_c4_state.assert_called_once()
        # c128_state has empty flat_loc -> skipped
        pool.write_c128_state.assert_not_called()


# ---------------------------------------------------------------------------
#  write_dsv4_prealloc_tables
# ---------------------------------------------------------------------------


class TestWriteDsv4PreallocTables(CustomTestCase):
    """Tests for write_dsv4_prealloc_tables (disagg prealloc path)."""

    def test_bundle_none_is_noop(self):
        """Bundle None -> no-op."""
        pool = _make_pool()
        req = _make_req(pool_idx=0)
        write_dsv4_prealloc_tables(pool, req, 0, 4, None)
        pool.write_swa.assert_not_called()

    def test_pool_without_write_c4_is_noop(self):
        """Pool lacks write_c4 -> no-op."""
        pool = SimpleNamespace()
        req = _make_req(pool_idx=0)
        write_dsv4_prealloc_tables(pool, req, 0, 4, _make_bundle())
        # No crash.

    def test_writes_tables_for_single_req(self):
        """Normal: writes swa/c4/state tables for one req."""
        pool = _make_pool()
        req = _make_req(c4_off=0, c128_off=0, pool_idx=3)
        bundle = _make_bundle(
            out_swa_loc=torch.arange(4, dtype=torch.int64),
            out_c4_loc=torch.tensor([20], dtype=torch.int64),
            out_c128_loc=torch.tensor([]),
            out_c4_state_loc=torch.arange(4, dtype=torch.int64),
            out_c128_state_loc=torch.tensor([]),
        )
        write_dsv4_prealloc_tables(pool, req, 0, 4, bundle)
        pool.write_swa.assert_called_once()
        # Verify req_pool_idx from req is used
        swa_args = pool.write_swa.call_args.args
        self.assertEqual(swa_args[0][0], 3)
        self.assertEqual(swa_args[0][1], slice(0, 4))


# ---------------------------------------------------------------------------
#  dsv4_unwrap_prealloc
# ---------------------------------------------------------------------------


class TestDsv4UnwrapPrealloc(CustomTestCase):
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
        req = _make_req(c4_off=0, c128_off=0, pool_idx=0)
        bundle = _make_bundle(
            out_swa_loc=torch.arange(4, dtype=torch.int64),
            out_c4_loc=torch.tensor([20], dtype=torch.int64),
            out_c128_loc=torch.tensor([]),
            out_c4_state_loc=torch.arange(4, dtype=torch.int64),
            out_c128_state_loc=torch.tensor([]),
        )
        result = dsv4_unwrap_prealloc(bundle, pool, req, 0, 4)
        self.assertTrue(torch.equal(result, torch.tensor([1, 2, 3])))
        pool.write_swa.assert_called_once()


# ---------------------------------------------------------------------------
#  dsv4_prealloc_kwargs
# ---------------------------------------------------------------------------


class TestDsv4PreallocKwargs(CustomTestCase):
    """Tests for dsv4_prealloc_kwargs."""

    def test_non_dsv4_allocator_returns_empty(self):
        """Allocator without c4_attn_allocator -> empty dict."""
        allocator = SimpleNamespace()
        req = _make_req(pool_idx=0)
        pool = _make_pool()
        result = dsv4_prealloc_kwargs(allocator, req, 4, pool, device="cpu")
        self.assertEqual(result, {})

    def test_dsv4_allocator_returns_kwargs(self):
        """DSV4 allocator -> returns req_pool_indices, dsv4_state_lens, req_to_token_pool."""
        allocator = MagicMock()
        allocator.compute_dsv4_state_lens_extend.return_value = "state_lens"
        req = _make_req(pool_idx=5)
        pool = _make_pool()
        result = dsv4_prealloc_kwargs(allocator, req, 8, pool, device="cpu")
        self.assertIn("req_pool_indices", result)
        self.assertIn("dsv4_state_lens", result)
        self.assertIn("req_to_token_pool", result)
        self.assertEqual(result["dsv4_state_lens"], "state_lens")
        self.assertIs(result["req_to_token_pool"], pool)
        self.assertEqual(result["req_pool_indices"].item(), 5)
        allocator.compute_dsv4_state_lens_extend.assert_called_once_with(
            [req], [8], [0]
        )


if __name__ == "__main__":
    unittest.main()
