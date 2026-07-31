"""Unit tests for DSV4NPUTokenToKVPoolAllocator and module-level helpers."""

import sys
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, patch

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
    "sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks",
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


# --- DSV4OutCacheLoc dataclass (real, used by stub module) ---


@dataclass
class DSV4OutCacheLoc:
    out_full_loc: "object"
    out_swa_loc: "object"
    out_c4_loc: "object"
    out_c128_loc: "object"
    out_c4_state_loc: Optional[object] = None
    out_c128_state_loc: Optional[object] = None


_fb_info_mod = type(sys)("sglang.srt.model_executor.forward_batch_info")
_fb_info_mod.DSV4OutCacheLoc = DSV4OutCacheLoc
sys.modules["sglang.srt.model_executor.forward_batch_info"] = _fb_info_mod

# --- SWATokenToKVPoolAllocator stub (real class — DSV4 allocator inherits it) ---


class _SWATokenToKVPoolAllocatorStub:
    """Stub for SWATokenToKVPoolAllocator — methods overridable via patch.object."""

    def __init__(self, **kwargs):
        pass

    def alloc_extend(self, *a, **k):
        return None

    def alloc_decode(self, *a, **k):
        return None

    def alloc_extend_swa_tail(self, *a, **k):
        return None

    def translate_loc_from_full_to_swa(self, x):
        return x

    def free(self, idx):
        pass

    def clear(self):
        pass


_swa_mod = type(sys)("sglang.srt.mem_cache.allocator.swa")
_swa_mod.SWATokenToKVPoolAllocator = _SWATokenToKVPoolAllocatorStub
sys.modules["sglang.srt.mem_cache.allocator.swa"] = _swa_mod

# sglang.version
_ver = type(sys)("sglang.version")
_ver.__version__ = "0.0.0.dev0"
sys.modules["sglang.version"] = _ver

import torch  # noqa: E402

from sglang.srt.hardware_backend.npu.dsv4.dsv4_allocator import (  # noqa: E402
    DSV4NPUTokenToKVPoolAllocator,
    get_last_loc,
)

_MOD = "sglang.srt.hardware_backend.npu.dsv4.dsv4_allocator"


from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=3, suite="base-a-test-1-npu-a2")


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _make_allocator(device="cpu"):
    """Create a DSV4NPUTokenToKVPoolAllocator bypassing __init__.

    c128_attn_allocator is a MagicMock; device / _empty_loc / _cur_pool are real.
    """
    alloc = object.__new__(DSV4NPUTokenToKVPoolAllocator)
    alloc.device = device
    alloc.page_size = 1
    alloc._empty_loc = torch.empty((0,), dtype=torch.int64, device=device)
    alloc._cur_req_to_token_pool = None
    alloc.c128_attn_allocator = MagicMock()
    alloc.c128_attn_allocator.page_size = 1
    alloc.c128_attn_allocator.available_size.return_value = 1024
    alloc.c128_page_refcount = torch.zeros(10, dtype=torch.int32, device=device)
    return alloc


def _make_req(*, committed=0, allocated=0, pool_idx=0):
    """SimpleNamespace req for allocator tests."""
    req = SimpleNamespace()
    req.kv_committed_len = committed
    req.kv = SimpleNamespace(kv_allocated_len=allocated)
    req.req_pool_idx = pool_idx
    return req


def _make_pool(n_reqs=2, max_len=64, device="cpu"):
    """Fake req_to_token_pool with 2D int32 tensor tables.

    Slot IDs start from 1 (page 0 is the kernel's skip sentinel in the real
    allocator; ``free()`` filters ``slots > 0`` so a 0 would be silently dropped).
    """
    pool = SimpleNamespace()
    base = torch.arange(1, n_reqs * max_len + 1, dtype=torch.int32, device=device)
    pool.req_to_token_c4 = base[: n_reqs * max_len].reshape(n_reqs, max_len).clone()
    pool.req_to_token_c128 = base[: n_reqs * max_len].reshape(n_reqs, max_len).clone()
    pool.req_to_c128_sidecar = base[: n_reqs * max_len].reshape(n_reqs, max_len).clone()
    pool.c128_page_size = 1
    pool.kernel_page_size = 1
    return pool


# ---------------------------------------------------------------------------
#  get_last_loc
# ---------------------------------------------------------------------------


class TestGetLastLoc(unittest.TestCase):
    """Tests for the module-level get_last_loc helper.

    Signature: get_last_loc(req_to_c128_sidecar, req_pool_indices, prefix_lens,
    page_size). ``page_size`` = c-pool page size in compressed-token units; the
    sidecar stores the page base id for each token, so the returned slot id is
    ``page_id * page_size + pos % page_size``.
    """

    def test_fresh_req_returns_minus_one(self):
        """prefix_lens == 0 → returns -1 for every req."""
        table = torch.tensor([[10, 11, 12], [20, 21, 22]])
        indices = torch.tensor([0, 1])
        prefix = torch.tensor([0, 0])
        result = get_last_loc(table, indices, prefix, 1)
        self.assertTrue(torch.equal(result, torch.tensor([-1, -1])))

    def test_extending_req_looks_up_table(self):
        """prefix_lens > 0 → looks up req_to_c128_sidecar[req, prefix-1]."""
        table = torch.tensor([[10, 11, 12], [20, 21, 22]])
        indices = torch.tensor([0, 1])
        prefix = torch.tensor([2, 1])
        result = get_last_loc(table, indices, prefix, 1)
        self.assertTrue(torch.equal(result, torch.tensor([11, 20])))

    def test_mixed_batch(self):
        """Batch with one fresh and one extending req."""
        table = torch.tensor([[10, 11, 12], [20, 21, 22]])
        indices = torch.tensor([0, 1])
        prefix = torch.tensor([3, 0])
        result = get_last_loc(table, indices, prefix, 1)
        self.assertTrue(torch.equal(result, torch.tensor([12, -1])))

    def test_dtype_matches_prefix(self):
        """Result dtype matches prefix_lens dtype."""
        table = torch.tensor([[10, 11, 12], [20, 21, 22]])
        indices = torch.tensor([0, 1])
        prefix = torch.tensor([1, 1], dtype=torch.int32)
        result = get_last_loc(table, indices, prefix, 1)
        self.assertEqual(result.dtype, torch.int32)
        self.assertTrue(torch.equal(result, torch.tensor([10, 20], dtype=torch.int32)))

    def test_multi_token_page(self):
        """page_size > 1 → slot id = page_id * page_size + pos % page_size."""
        # Sidecar holds one page-base id per token; token 3 (page 1 of size 2)
        # lives in page 20 → slot 20*2 + 1 = 41.
        table = torch.tensor([[10, 20, 40, 40], [30, 30, 40, 40]])
        indices = torch.tensor([0])
        prefix = torch.tensor([4])
        result = get_last_loc(table, indices, prefix, 2)
        self.assertTrue(torch.equal(result, torch.tensor([41])))


# ---------------------------------------------------------------------------
#  Module-level dispatch / reserve functions
# ---------------------------------------------------------------------------


class TestModuleFunctions(unittest.TestCase):
    """Tests for alloc_paged_token_slots_extend_npu and _reserve_extend."""

    @patch(_MOD + ".is_deepseek_v4", return_value=True)
    @patch(_MOD + ".alloc_paged_token_slots_reserve_extend")
    @patch(_MOD + ".alloc_paged_token_slots_extend")
    def test_dsv4_calls_reserve(self, mock_extend, mock_reserve, mock_is_dsv4):
        """is_deepseek_v4 True → dispatches to reserve_extend."""
        batch = SimpleNamespace(model_config=SimpleNamespace(hf_config=object()))
        mock_reserve.return_value = "reserved"
        from sglang.srt.hardware_backend.npu.dsv4.dsv4_allocator import (
            alloc_paged_token_slots_extend_npu,
        )

        result = alloc_paged_token_slots_extend_npu(1, 2, batch=batch, extra="kw")
        self.assertEqual(result, "reserved")
        mock_reserve.assert_called_once_with(1, 2, batch=batch, extra="kw")
        mock_extend.assert_not_called()

    @patch(_MOD + ".is_deepseek_v4", return_value=False)
    @patch(_MOD + ".alloc_paged_token_slots_reserve_extend")
    @patch(_MOD + ".alloc_paged_token_slots_extend")
    def test_non_dsv4_calls_extend(self, mock_extend, mock_reserve, mock_is_dsv4):
        """is_deepseek_v4 False → dispatches to standard extend."""
        batch = SimpleNamespace(model_config=SimpleNamespace(hf_config=object()))
        mock_extend.return_value = "extended"
        from sglang.srt.hardware_backend.npu.dsv4.dsv4_allocator import (
            alloc_paged_token_slots_extend_npu,
        )

        result = alloc_paged_token_slots_extend_npu(1, 2, batch=batch)
        self.assertEqual(result, "extended")
        mock_extend.assert_called_once_with(1, 2, batch=batch)
        mock_reserve.assert_not_called()

    @patch(_MOD + ".alloc_paged_token_slots_reserve_extend")
    @patch(_MOD + ".alloc_paged_token_slots_extend")
    def test_no_batch_calls_extend(self, mock_extend, mock_reserve):
        """batch=None → standard extend path regardless of model."""
        mock_extend.return_value = "extended"
        from sglang.srt.hardware_backend.npu.dsv4.dsv4_allocator import (
            alloc_paged_token_slots_extend_npu,
        )

        result = alloc_paged_token_slots_extend_npu(1, 2, batch=None)
        self.assertEqual(result, "extended")
        mock_extend.assert_called_once()

    @patch(_MOD + ".alloc_paged_token_slots_extend")
    @patch(_MOD + ".maybe_write_dsv4_extend")
    def test_reserve_forwards_and_writes_tables(self, mock_hook, mock_extend):
        """reserve_extend forwards args to extend and writes DSV4 KV tables."""
        batch = SimpleNamespace(req_pool_indices_cpu=torch.tensor([0]))
        prefix_cpu = torch.tensor([0])
        seq_cpu = torch.tensor([8])
        mock_extend.return_value = "loc"
        from sglang.srt.hardware_backend.npu.dsv4.dsv4_allocator import (
            alloc_paged_token_slots_reserve_extend,
        )

        result = alloc_paged_token_slots_reserve_extend(
            "tree",
            prefix_cpu,
            prefix_cpu,
            seq_cpu,
            seq_cpu,
            torch.tensor([-1]),
            8,
            req_pool_indices=torch.tensor([7]),
            batch=batch,
        )
        self.assertEqual(result, "loc")
        mock_extend.assert_called_once_with(
            "tree",
            prefix_cpu,
            prefix_cpu,
            seq_cpu,
            seq_cpu,
            torch.tensor([-1]),
            8,
            req_pool_indices=torch.tensor([7]),
            batch=batch,
        )
        mock_hook.assert_called_once_with(
            batch, batch.req_pool_indices_cpu, prefix_cpu, seq_cpu
        )

    @patch(_MOD + ".alloc_paged_token_slots_extend")
    @patch(_MOD + ".maybe_write_dsv4_extend")
    def test_reserve_without_batch(self, mock_hook, mock_extend):
        """batch=None → skips the DSV4 table write hook."""
        mock_extend.return_value = "loc"
        from sglang.srt.hardware_backend.npu.dsv4.dsv4_allocator import (
            alloc_paged_token_slots_reserve_extend,
        )

        result = alloc_paged_token_slots_reserve_extend(
            "tree",
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([8]),
            torch.tensor([8]),
            torch.tensor([-1]),
            8,
            batch=None,
        )
        self.assertEqual(result, "loc")
        mock_hook.assert_not_called()


# ---------------------------------------------------------------------------
#  __init__
# ---------------------------------------------------------------------------


def _make_kvcache(c128_size=50, kernel_page_size=1):
    """Fake kvcache with a c128 KV pool (the only pool the allocator uses)."""
    kvcache = SimpleNamespace()
    kvcache.c128_size = c128_size
    kvcache.c128_kv_pool = SimpleNamespace(kernel_page_size=kernel_page_size)
    return kvcache


class TestInit(unittest.TestCase):
    """Tests for DSV4NPUTokenToKVPoolAllocator.__init__.

    Upstream removed the compressor-state allocators; the constructor now only
    creates the C128 KV sub-allocator + the C128 page refcount tensor.
    """

    def _construct(self, mock_ctor, kvcache=None, **kwargs):
        mock_ctor.return_value.num_pages = 25
        kvcache = kvcache or _make_kvcache()
        return DSV4NPUTokenToKVPoolAllocator(
            size=100,
            size_swa=50,
            page_size=1,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=kvcache,
            need_sort=False,
            **kwargs,
        )

    @patch(_MOD + ".NPUPagedTokenToKVPoolAllocator")
    def test_creates_c128_allocator_and_empty_loc(self, mock_ctor):
        """c128 KV allocator created with pool args; no state allocators."""
        kvcache = _make_kvcache()
        alloc = self._construct(mock_ctor, kvcache)
        self.assertEqual(mock_ctor.call_count, 1)
        c128_call = mock_ctor.call_args_list[0]
        self.assertEqual(c128_call.args[0], 50)
        self.assertEqual(c128_call.kwargs["page_size"], 1)
        self.assertIs(c128_call.kwargs["kvcache"], kvcache.c128_kv_pool)
        self.assertEqual(c128_call.kwargs["dtype"], torch.bfloat16)
        self.assertEqual(c128_call.kwargs["device"], "cpu")
        self.assertEqual(c128_call.kwargs["need_sort"], False)
        # state allocators removed upstream
        self.assertFalse(hasattr(alloc, "c4_state_attn_allocator"))
        self.assertFalse(hasattr(alloc, "c128_state_attn_allocator"))
        # c128_page_refcount sized num_pages + 1
        self.assertEqual(alloc.c128_page_refcount.numel(), 26)
        self.assertEqual(alloc.c128_page_refcount.dtype, torch.int32)
        self.assertEqual(alloc._empty_loc.numel(), 0)
        self.assertEqual(alloc._empty_loc.dtype, torch.int64)
        self.assertIsNone(alloc._cur_req_to_token_pool)

    @patch(_MOD + ".NPUPagedTokenToKVPoolAllocator")
    def test_c128_pool_size_and_page_forwarded(self, mock_ctor):
        """c128_size and kernel_page_size forwarded to the c128 allocator."""
        kvcache = _make_kvcache(c128_size=99, kernel_page_size=4)
        self._construct(mock_ctor, kvcache)
        call = mock_ctor.call_args_list[0]
        self.assertEqual(call.args[0], 99)
        self.assertEqual(call.kwargs["page_size"], 4)


# ---------------------------------------------------------------------------
#  _compute_c_extend_counts / _pool_exhausted (static)
# ---------------------------------------------------------------------------


class TestComputeCExtendCounts(unittest.TestCase):
    """Tests for DSV4NPUTokenToKVPoolAllocator._compute_c_extend_counts."""

    def test_normal_batch(self):
        """Sum of (seq//ratio - prefix//ratio) across batch."""
        prefix = torch.tensor([0, 4, 8])
        seq = torch.tensor([8, 12, 16])
        self.assertEqual(
            DSV4NPUTokenToKVPoolAllocator._compute_c_extend_counts(prefix, seq, 4),
            6,
        )

    def test_ratio_128(self):
        """ratio=128 compressed-token counting."""
        prefix = torch.tensor([0, 128])
        seq = torch.tensor([256, 256])
        self.assertEqual(
            DSV4NPUTokenToKVPoolAllocator._compute_c_extend_counts(prefix, seq, 128),
            3,
        )

    def test_none_tensors_return_zero(self):
        """None prefix/seq → 0 new tokens."""
        self.assertEqual(
            DSV4NPUTokenToKVPoolAllocator._compute_c_extend_counts(None, None, 4),
            0,
        )

    def test_no_new_compressed_tokens(self):
        """prefix and seq within same ratio bucket → 0 new tokens."""
        prefix = torch.tensor([4, 8])
        seq = torch.tensor([5, 10])
        self.assertEqual(
            DSV4NPUTokenToKVPoolAllocator._compute_c_extend_counts(prefix, seq, 4),
            0,
        )

    def test_clamps_negative_diff_to_zero(self):
        """prefix > seq (shouldn't happen) → clamped to 0."""
        prefix = torch.tensor([10])
        seq = torch.tensor([5])
        self.assertEqual(
            DSV4NPUTokenToKVPoolAllocator._compute_c_extend_counts(prefix, seq, 4),
            0,
        )


class TestPoolExhausted(unittest.TestCase):
    """Tests for DSV4NPUTokenToKVPoolAllocator._pool_exhausted."""

    def test_returns_runtime_error(self):
        """Always returns a RuntimeError instance."""
        err = DSV4NPUTokenToKVPoolAllocator._pool_exhausted(4, "KV", 10, 5)
        self.assertIsInstance(err, RuntimeError)

    def test_message_contains_details(self):
        """Message includes ratio, kind, need, and available."""
        err = DSV4NPUTokenToKVPoolAllocator._pool_exhausted(128, "state", 3, 1)
        msg = str(err)
        self.assertIn("c128", msg)
        self.assertIn("state", msg)
        self.assertIn("need 3", msg)
        self.assertIn("available=1", msg)


# ---------------------------------------------------------------------------
#  _alloc_c_extend
# ---------------------------------------------------------------------------


class TestAllocCExtend(unittest.TestCase):
    """Tests for DSV4NPUTokenToKVPoolAllocator._alloc_c_extend."""

    def test_returns_empty_when_no_new_compressed_tokens(self):
        """prefix and seq in same ratio bucket → empty result."""
        alloc = _make_allocator()
        prefix = torch.tensor([8])
        seq = torch.tensor([9])
        result = alloc._alloc_c_extend(
            alloc.c128_attn_allocator,
            prefix,
            prefix,
            seq,
            seq,
            torch.tensor([0]),
            torch.int64,
            ratio=4,
        )
        self.assertEqual(result.numel(), 0)

    def test_normal_alloc(self):
        """Normal extend → calls sub-allocator.alloc_extend and returns result."""
        alloc = _make_allocator()
        alloc._cur_req_to_token_pool = _make_pool()
        prefix = torch.tensor([0])
        seq = torch.tensor([8])
        expected = torch.tensor([10, 11], dtype=torch.int32)
        alloc.c128_attn_allocator.alloc_extend.return_value = expected
        result = alloc._alloc_c_extend(
            alloc.c128_attn_allocator,
            prefix,
            prefix,
            seq,
            seq,
            torch.tensor([0]),
            torch.int64,
            ratio=4,
        )
        self.assertTrue(torch.equal(result, expected))
        call = alloc.c128_attn_allocator.alloc_extend.call_args
        self.assertEqual(call.args[5], 2)

    def test_pool_exhausted_raises(self):
        """Sub-allocator returns None → RuntimeError."""
        alloc = _make_allocator()
        alloc._cur_req_to_token_pool = _make_pool()
        alloc.c128_attn_allocator.alloc_extend.return_value = None
        alloc.c128_attn_allocator.available_size.return_value = 1
        prefix = torch.tensor([0])
        seq = torch.tensor([8])
        with self.assertRaises(RuntimeError):
            alloc._alloc_c_extend(
                alloc.c128_attn_allocator,
                prefix,
                prefix,
                seq,
                seq,
                torch.tensor([0]),
                torch.int64,
                ratio=4,
            )

    def test_missing_pool_asserts(self):
        """_cur_req_to_token_pool None → AssertionError."""
        alloc = _make_allocator()
        alloc._cur_req_to_token_pool = None
        prefix = torch.tensor([0])
        seq = torch.tensor([8])
        with self.assertRaises(AssertionError):
            alloc._alloc_c_extend(
                alloc.c128_attn_allocator,
                prefix,
                prefix,
                seq,
                seq,
                torch.tensor([0]),
                torch.int64,
                ratio=4,
            )


# ---------------------------------------------------------------------------
#  alloc_extend (public API)
# ---------------------------------------------------------------------------


class TestAllocExtend(unittest.TestCase):
    """Tests for DSV4NPUTokenToKVPoolAllocator.alloc_extend."""

    def test_returns_none_when_super_returns_none(self):
        """super().alloc_extend returns None → alloc_extend returns None."""
        alloc = _make_allocator()
        prefix = torch.tensor([0])
        seq = torch.tensor([8])
        last_loc = torch.tensor([-1])
        with patch.object(
            _SWATokenToKVPoolAllocatorStub, "alloc_extend", return_value=None
        ):
            result = alloc.alloc_extend(
                prefix,
                prefix,
                seq,
                seq,
                last_loc,
                8,
                req_pool_indices=torch.tensor([0]),
            )
        self.assertIsNone(result)

    @patch.object(DSV4NPUTokenToKVPoolAllocator, "_alloc_compressed_kv")
    @patch.object(
        DSV4NPUTokenToKVPoolAllocator,
        "translate_loc_from_full_to_swa",
        return_value=torch.tensor([9, 10]),
    )
    @patch.object(
        _SWATokenToKVPoolAllocatorStub,
        "alloc_extend",
        return_value=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
    )
    def test_stashes_pool_and_delegates(self, mock_super, mock_swa, mock_ack):
        """super().alloc_extend succeeds → _alloc_compressed_kv called with full_loc."""
        alloc = _make_allocator()
        mock_ack.return_value = "bundle"
        pool = _make_pool()
        prefix = torch.tensor([0])
        seq = torch.tensor([8])
        last_loc = torch.tensor([-1])
        result = alloc.alloc_extend(
            prefix,
            prefix,
            seq,
            seq,
            last_loc,
            8,
            req_pool_indices=torch.tensor([0]),
            req_to_token_pool=pool,
        )
        self.assertEqual(result, "bundle")
        self.assertIs(alloc._cur_req_to_token_pool, pool)
        mock_swa.assert_called_once()
        args = mock_ack.call_args.args
        self.assertTrue(torch.equal(args[0], torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])))
        self.assertEqual(args[2].item(), 0)


# ---------------------------------------------------------------------------
#  alloc_decode (public API)
# ---------------------------------------------------------------------------


class TestAllocDecode(unittest.TestCase):
    """Tests for DSV4NPUTokenToKVPoolAllocator.alloc_decode."""

    def test_returns_none_when_super_returns_none(self):
        """super().alloc_decode returns None → alloc_decode returns None."""
        alloc = _make_allocator()
        seq = torch.tensor([9])
        last_loc = torch.tensor([8])
        with patch.object(
            _SWATokenToKVPoolAllocatorStub, "alloc_decode", return_value=None
        ):
            result = alloc.alloc_decode(
                seq,
                seq,
                last_loc,
                req_pool_indices=torch.tensor([0]),
            )
        self.assertIsNone(result)

    @patch.object(DSV4NPUTokenToKVPoolAllocator, "_alloc_compressed_kv")
    @patch.object(
        DSV4NPUTokenToKVPoolAllocator,
        "translate_loc_from_full_to_swa",
        return_value=torch.tensor([99]),
    )
    @patch.object(
        _SWATokenToKVPoolAllocatorStub,
        "alloc_decode",
        return_value=torch.tensor([42]),
    )
    def test_derives_prefix_lens_and_delegates(self, mock_super, mock_swa, mock_ack):
        """prefix_lens = seq_lens - 1 passed to _alloc_compressed_kv."""
        alloc = _make_allocator()
        alloc._cur_req_to_token_pool = _make_pool()
        mock_ack.return_value = "bundle"
        seq = torch.tensor([9])
        seq_cpu = torch.tensor([9])
        last_loc = torch.tensor([8])
        result = alloc.alloc_decode(
            seq,
            seq_cpu,
            last_loc,
            req_pool_indices=torch.tensor([0]),
        )
        self.assertEqual(result, "bundle")
        args = mock_ack.call_args.args
        self.assertEqual(args[2].item(), 8)
        self.assertEqual(args[3].item(), 8)


# ---------------------------------------------------------------------------
#  free
# ---------------------------------------------------------------------------


class TestFree(unittest.TestCase):
    """Tests for DSV4NPUTokenToKVPoolAllocator.free.

    Upstream moved C128 KV page recycling to the sidecar refcount path:
    ``free(req=, req_to_token_pool=)`` releases the req's non-zero
    ``req_to_c128_sidecar`` pages, zeroes the row, and clears the fixed C128
    request state bank via ``get_kvcache().clear_c128_req_state``.
    """

    def test_free_index_only_calls_super(self):
        """free(free_index) → only super().free is called."""
        alloc = _make_allocator()
        idx = torch.tensor([1, 2, 3])
        with patch.object(_SWATokenToKVPoolAllocatorStub, "free") as super_free:
            alloc.free(idx)
            super_free.assert_called_once_with(idx)

    def test_no_req_returns_early(self):
        """free(free_index) without req → c128 release not touched."""
        alloc = _make_allocator()
        with patch.object(alloc, "release_c128_pages") as release:
            alloc.free(torch.tensor([1, 2]))
            release.assert_not_called()

    def test_req_free_releases_c128_pages(self):
        """free(req=, pool=) → releases non-zero sidecar pages, zeroes the row,
        and clears the fixed C128 request state."""
        alloc = _make_allocator()
        kvcache = MagicMock()
        alloc.get_kvcache = MagicMock(return_value=kvcache)
        pool = _make_pool(n_reqs=2, max_len=8)
        req = _make_req(committed=10, allocated=10, pool_idx=0)
        with patch.object(alloc, "release_c128_pages") as release:
            alloc.free(req=req, req_to_token_pool=pool)
        row = pool.req_to_c128_sidecar[0]
        self.assertTrue(torch.equal(release.call_args[0][0], torch.tensor(list(range(1, 9)))))
        self.assertTrue(torch.equal(row, torch.zeros(8, dtype=torch.int32)))
        kvcache.clear_c128_req_state.assert_called_once_with(0)

    def test_zero_kv_len_no_free(self):
        """kv_len == 0 → release/clear not called."""
        alloc = _make_allocator()
        kvcache = MagicMock()
        alloc.get_kvcache = MagicMock(return_value=kvcache)
        pool = _make_pool()
        req = _make_req(committed=0, allocated=0, pool_idx=0)
        with patch.object(alloc, "release_c128_pages") as release:
            alloc.free(req=req, req_to_token_pool=pool)
        release.assert_not_called()
        kvcache.clear_c128_req_state.assert_not_called()

    def test_none_req_pool_idx_skipped(self):
        """req_pool_idx None → returns early."""
        alloc = _make_allocator()
        kvcache = MagicMock()
        alloc.get_kvcache = MagicMock(return_value=kvcache)
        pool = _make_pool()
        req = _make_req(committed=5, allocated=5, pool_idx=None)
        with patch.object(alloc, "release_c128_pages") as release:
            alloc.free(req=req, req_to_token_pool=pool)
        release.assert_not_called()
        kvcache.clear_c128_req_state.assert_not_called()

    def test_uses_max_of_committed_and_allocated(self):
        """kv_len = max(committed, allocated) gates the release."""
        alloc = _make_allocator()
        kvcache = MagicMock()
        alloc.get_kvcache = MagicMock(return_value=kvcache)
        pool = _make_pool(n_reqs=2, max_len=8)
        # committed=0 but allocated=12 → still releases the full sidecar row.
        req = _make_req(committed=0, allocated=12, pool_idx=0)
        with patch.object(alloc, "release_c128_pages") as release:
            alloc.free(req=req, req_to_token_pool=pool)
        release.assert_called_once()


# ---------------------------------------------------------------------------
#  _derive_c4_loc_from_full / _alloc_compressed_kv / _wrap_full_alloc
# ---------------------------------------------------------------------------


class TestDeriveC4LocFromFull(unittest.TestCase):
    """Tests for DSV4NPUTokenToKVPoolAllocator._derive_c4_loc_from_full."""

    def test_derives_c4_from_full(self):
        """Full slots closing a 4-token group map to //4 c4 slots."""
        out_full = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        self.assertTrue(
            torch.equal(
                DSV4NPUTokenToKVPoolAllocator._derive_c4_loc_from_full(out_full),
                torch.tensor([0, 1]),
            )
        )

    def test_empty_full_loc(self):
        """No slots → empty c4 loc."""
        out_full = torch.empty(0, dtype=torch.int64)
        result = DSV4NPUTokenToKVPoolAllocator._derive_c4_loc_from_full(out_full)
        self.assertEqual(result.numel(), 0)


class TestAllocCompressedKV(unittest.TestCase):
    """Tests for DSV4NPUTokenToKVPoolAllocator._alloc_compressed_kv."""

    def test_bundles_full_swa_c4_c128(self):
        """Bundles full/swa/c4/c128 into a DSV4OutCacheLoc; no state fields."""
        alloc = _make_allocator()
        alloc._cur_req_to_token_pool = _make_pool()
        out_full = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int64)
        out_swa = torch.tensor([9, 10], dtype=torch.int64)
        prefix = torch.tensor([0])
        seq = torch.tensor([256])
        c128_result = torch.tensor([50, 51], dtype=torch.int32)
        alloc.c128_attn_allocator.alloc_extend.return_value = c128_result
        result = alloc._alloc_compressed_kv(
            out_full,
            out_swa,
            prefix,
            prefix,
            seq,
            seq,
            torch.int64,
            torch.tensor([0]),
        )
        self.assertIsInstance(result, DSV4OutCacheLoc)
        self.assertTrue(torch.equal(result.out_full_loc, out_full))
        self.assertTrue(torch.equal(result.out_swa_loc, out_swa))
        self.assertTrue(torch.equal(result.out_c4_loc, torch.tensor([0, 1])))
        self.assertTrue(torch.equal(result.out_c128_loc, c128_result))
        self.assertIsNone(result.out_c4_state_loc)
        self.assertIsNone(result.out_c128_state_loc)

    def test_requires_req_pool_indices(self):
        """req_pool_indices None → AssertionError."""
        alloc = _make_allocator()
        with self.assertRaises(AssertionError):
            alloc._alloc_compressed_kv(
                torch.tensor([1]),
                torch.tensor([1]),
                torch.tensor([0]),
                torch.tensor([0]),
                torch.tensor([256]),
                torch.tensor([256]),
                torch.int64,
                None,
            )


class TestWrapFullAlloc(unittest.TestCase):
    """Tests for DSV4NPUTokenToKVPoolAllocator._wrap_full_alloc."""

    @patch.object(DSV4NPUTokenToKVPoolAllocator, "_alloc_compressed_kv")
    @patch.object(
        DSV4NPUTokenToKVPoolAllocator,
        "translate_loc_from_full_to_swa",
        return_value=torch.tensor([9]),
    )
    def test_none_returns_none(self, mock_swa, mock_ack):
        """out_full_loc None → returns None without translate/compressed_kv."""
        alloc = _make_allocator()
        result = alloc._wrap_full_alloc(
            None,
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([8]),
            torch.tensor([8]),
            torch.int64,
            torch.tensor([0]),
        )
        self.assertIsNone(result)
        mock_swa.assert_not_called()
        mock_ack.assert_not_called()

    @patch.object(DSV4NPUTokenToKVPoolAllocator, "_alloc_compressed_kv")
    @patch.object(
        DSV4NPUTokenToKVPoolAllocator,
        "translate_loc_from_full_to_swa",
        return_value=torch.tensor([9]),
    )
    def test_translates_and_delegates(self, mock_swa, mock_ack):
        """out_full_loc not None → translate + _alloc_compressed_kv."""
        alloc = _make_allocator()
        mock_ack.return_value = "bundle"
        full_loc = torch.tensor([1, 2])
        result = alloc._wrap_full_alloc(
            full_loc,
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([8]),
            torch.tensor([8]),
            torch.int64,
            torch.tensor([0]),
        )
        self.assertEqual(result, "bundle")
        mock_swa.assert_called_once_with(full_loc)
        mock_ack.assert_called_once()


# ---------------------------------------------------------------------------
#  alloc_extend_swa_tail (public API)
# ---------------------------------------------------------------------------


class TestAllocExtendSwaTail(unittest.TestCase):
    """Tests for DSV4NPUTokenToKVPoolAllocator.alloc_extend_swa_tail."""

    def test_returns_none_when_super_returns_none(self):
        """super().alloc_extend_swa_tail returns None → returns None."""
        alloc = _make_allocator()
        prefix = torch.tensor([0])
        seq = torch.tensor([512])
        last_loc = torch.tensor([-1])
        with patch.object(
            _SWATokenToKVPoolAllocatorStub, "alloc_extend_swa_tail", return_value=None
        ):
            result = alloc.alloc_extend_swa_tail(
                prefix,
                prefix,
                seq,
                seq,
                last_loc,
                512,
                128,
                req_pool_indices=torch.tensor([0]),
            )
        self.assertIsNone(result)

    @patch.object(DSV4NPUTokenToKVPoolAllocator, "_alloc_compressed_kv")
    @patch.object(
        DSV4NPUTokenToKVPoolAllocator,
        "translate_loc_from_full_to_swa",
        return_value=torch.tensor([9]),
    )
    @patch.object(
        _SWATokenToKVPoolAllocatorStub,
        "alloc_extend_swa_tail",
        return_value=torch.tensor([1, 2, 3]),
    )
    def test_delegates_to_wrap_full_alloc(self, mock_super, mock_swa, mock_ack):
        """super succeeds → _wrap_full_alloc translates + delegates."""
        alloc = _make_allocator()
        mock_ack.return_value = "bundle"
        pool = _make_pool()
        prefix = torch.tensor([0])
        seq = torch.tensor([512])
        last_loc = torch.tensor([-1])
        result = alloc.alloc_extend_swa_tail(
            prefix,
            prefix,
            seq,
            seq,
            last_loc,
            512,
            128,
            req_pool_indices=torch.tensor([0]),
            req_to_token_pool=pool,
        )
        self.assertEqual(result, "bundle")
        self.assertIs(alloc._cur_req_to_token_pool, pool)
        args = mock_ack.call_args.args
        self.assertTrue(torch.equal(args[0], torch.tensor([1, 2, 3])))


# ---------------------------------------------------------------------------
#  clear
# ---------------------------------------------------------------------------


class TestClear(unittest.TestCase):
    """Tests for clear."""


if __name__ == "__main__":
    unittest.main()
