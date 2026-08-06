"""Unit tests for DSV4NPUTokenToKVPoolAllocator and module-level helpers."""

import os
import sys
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, patch

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
    "sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks",
    "sglang.srt.runtime_context",
    "sglang.srt.speculative.spec_info",
    "sglang.srt.utils",
    "aiohttp",
    "sglang.test.ci.ci_register",
    "sglang.test.test_utils",
):
    sys.modules[_mod] = _mock

# --- DSV4OutCacheLoc / DSV4StateLens dataclasses (real, used by stub module) ---


@dataclass
class DSV4OutCacheLoc:
    out_full_loc: "object"
    out_swa_loc: "object"
    out_c4_loc: "object"
    out_c128_loc: "object"
    out_c4_state_loc: Optional[object] = None
    out_c128_state_loc: Optional[object] = None


@dataclass
class DSV4StateLens:
    c4_prefix_lens: "object"
    c4_prefix_lens_cpu: "object"
    c4_seq_lens: "object"
    c4_seq_lens_cpu: "object"
    c4_extend_num_tokens: int
    c128_prefix_lens: "object"
    c128_prefix_lens_cpu: "object"
    c128_seq_lens: "object"
    c128_seq_lens_cpu: "object"
    c128_extend_num_tokens: int


_fb_info_mod = type(sys)("sglang.srt.model_executor.forward_batch_info")
_fb_info_mod.DSV4OutCacheLoc = DSV4OutCacheLoc
_fb_info_mod.DSV4StateLens = DSV4StateLens
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


def _make_allocator(device="cpu", with_state=True):
    """Create a DSV4NPUTokenToKVPoolAllocator bypassing __init__.

    Sub-allocators are MagicMocks; device / _empty_loc / _cur_pool are real.
    """
    alloc = object.__new__(DSV4NPUTokenToKVPoolAllocator)
    alloc.device = device
    alloc.page_size = 1
    alloc._empty_loc = torch.empty((0,), dtype=torch.int64, device=device)
    alloc._cur_req_to_token_pool = None
    alloc.c4_attn_allocator = MagicMock()
    alloc.c128_attn_allocator = MagicMock()
    if with_state:
        alloc.c4_state_attn_allocator = MagicMock()
        alloc.c128_state_attn_allocator = MagicMock()
    else:
        alloc.c4_state_attn_allocator = None
        alloc.c128_state_attn_allocator = None
    return alloc


def _make_req(
    *,
    c4_kv=None,
    c128_kv=None,
    c4_off=None,
    c128_off=None,
    committed=0,
    allocated=0,
    pool_idx=0,
):
    """SimpleNamespace req; state attrs set only when given (getattr defaults 0)."""
    req = SimpleNamespace()
    if c4_kv is not None:
        req.c4_state_kv_len = c4_kv
    if c128_kv is not None:
        req.c128_state_kv_len = c128_kv
    if c4_off is not None:
        req.c4_state_alloc_offset = c4_off
    if c128_off is not None:
        req.c128_state_alloc_offset = c128_off
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
    pool.req_to_token_c4_state = (
        base[: n_reqs * max_len].reshape(n_reqs, max_len).clone()
    )
    pool.req_to_token_c128_state = (
        base[: n_reqs * max_len].reshape(n_reqs, max_len).clone()
    )
    return pool


# ---------------------------------------------------------------------------
#  get_last_loc
# ---------------------------------------------------------------------------


class TestGetLastLoc(CustomTestCase):
    """Tests for the module-level get_last_loc helper."""

    def test_fresh_req_returns_minus_one(self):
        """prefix_lens == 0 → returns -1 for every req."""
        table = torch.tensor([[10, 11, 12], [20, 21, 22]])
        indices = torch.tensor([0, 1])
        prefix = torch.tensor([0, 0])
        result = get_last_loc(table, indices, prefix)
        self.assertTrue(torch.equal(result, torch.tensor([-1, -1])))

    def test_extending_req_looks_up_table(self):
        """prefix_lens > 0 → looks up req_to_token[req, prefix-1]."""
        table = torch.tensor([[10, 11, 12], [20, 21, 22]])
        indices = torch.tensor([0, 1])
        prefix = torch.tensor([2, 1])
        result = get_last_loc(table, indices, prefix)
        self.assertTrue(torch.equal(result, torch.tensor([11, 20])))

    def test_mixed_batch(self):
        """Batch with one fresh and one extending req."""
        table = torch.tensor([[10, 11, 12], [20, 21, 22]])
        indices = torch.tensor([0, 1])
        prefix = torch.tensor([3, 0])
        result = get_last_loc(table, indices, prefix)
        self.assertTrue(torch.equal(result, torch.tensor([12, -1])))

    def test_dtype_matches_prefix(self):
        """Result dtype matches prefix_lens dtype."""
        table = torch.tensor([[10, 11, 12], [20, 21, 22]])
        indices = torch.tensor([0, 1])
        prefix = torch.tensor([1, 1], dtype=torch.int32)
        result = get_last_loc(table, indices, prefix)
        self.assertEqual(result.dtype, torch.int32)
        self.assertTrue(torch.equal(result, torch.tensor([10, 20], dtype=torch.int32)))


# ---------------------------------------------------------------------------
#  Module-level dispatch / reserve functions
# ---------------------------------------------------------------------------


class TestModuleFunctions(CustomTestCase):
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
    def test_reserve_computes_state_lens(self, mock_hook, mock_extend):
        """reserve_extend computes dsv4_state_lens from allocator when None."""
        allocator = MagicMock()
        allocator.compute_dsv4_state_lens_reserve.return_value = "state_lens"
        batch = SimpleNamespace(
            token_to_kv_pool_allocator=allocator,
            reqs=[_make_req()],
            req_pool_indices_cpu=torch.tensor([0]),
        )
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
            batch=batch,
        )
        self.assertEqual(result, "loc")
        allocator.compute_dsv4_state_lens_reserve.assert_called_once_with(
            batch.reqs, prefix_cpu, seq_cpu
        )
        mock_hook.assert_called_once()

    @patch(_MOD + ".alloc_paged_token_slots_extend")
    @patch(_MOD + ".maybe_write_dsv4_extend")
    def test_reserve_without_batch(self, mock_hook, mock_extend):
        """batch=None → skips state-lens computation and hook."""
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

    @patch(_MOD + ".alloc_paged_token_slots_extend")
    @patch(_MOD + ".maybe_write_dsv4_extend")
    def test_reserve_allocator_without_compute_method(self, mock_hook, mock_extend):
        """Allocator without compute_dsv4_state_lens_reserve → dsv4_state_lens stays None."""
        allocator = SimpleNamespace()
        batch = SimpleNamespace(
            token_to_kv_pool_allocator=allocator,
            reqs=[_make_req()],
            req_pool_indices_cpu=torch.tensor([0]),
        )
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
            batch=batch,
        )
        self.assertEqual(result, "loc")
        # hasattr(allocator, "compute_dsv4_state_lens_reserve") is False
        # → dsv4_state_lens stays None → forwarded as None
        self.assertIsNone(mock_extend.call_args.kwargs["dsv4_state_lens"])
        mock_hook.assert_called_once()

    @patch(_MOD + ".alloc_paged_token_slots_extend")
    @patch(_MOD + ".maybe_write_dsv4_extend")
    def test_reserve_pre_provided_state_lens(self, mock_hook, mock_extend):
        """dsv4_state_lens pre-provided → if branch skipped, value forwarded as-is."""
        allocator = MagicMock()
        allocator.compute_dsv4_state_lens_reserve.return_value = "should_not_be_used"
        pre_provided = "pre_provided_lens"
        batch = SimpleNamespace(
            token_to_kv_pool_allocator=allocator,
            reqs=[_make_req()],
            req_pool_indices_cpu=torch.tensor([0]),
        )
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
            batch=batch,
            dsv4_state_lens=pre_provided,
        )
        self.assertEqual(result, "loc")
        # if branch skipped → compute_dsv4_state_lens_reserve NOT called
        allocator.compute_dsv4_state_lens_reserve.assert_not_called()
        # pre-provided value forwarded to alloc_paged_token_slots_extend as-is
        self.assertIs(mock_extend.call_args.kwargs["dsv4_state_lens"], pre_provided)
        mock_hook.assert_called_once()


# ---------------------------------------------------------------------------
#  __init__
# ---------------------------------------------------------------------------


def _make_kvcache(
    *,
    with_state=True,
    c4_state_pool="c4s_pool",
    c128_state_pool="c128s_pool",
    c4_state_pool_size=10,
    c128_state_pool_size=20,
    compression_ratios=None,
):
    """Fake kvcache with c4/c128 KV pool and optional compress-state pools."""
    kvcache = SimpleNamespace()
    kvcache.c4_size = 100
    kvcache.c4_kv_pool = "c4_kv"
    kvcache.c128_size = 50
    kvcache.c128_kv_pool = "c128_kv"
    if with_state:
        kvcache.compression_ratios = compression_ratios or [4, 128]
        kvcache.compress_state_pools = [c4_state_pool, c128_state_pool]
        kvcache.c4_state_pool_size = c4_state_pool_size
        kvcache.c128_state_pool_size = c128_state_pool_size
    return kvcache


class TestInit(CustomTestCase):
    """Tests for DSV4NPUTokenToKVPoolAllocator.__init__."""

    @patch(_MOD + ".NPUPagedTokenToKVPoolAllocator")
    def test_creates_kv_allocators_and_empty_loc(self, mock_ctor):
        """c4/c128 KV allocators created with correct pool sizes; no state → None."""
        kvcache = _make_kvcache(with_state=False)
        alloc = DSV4NPUTokenToKVPoolAllocator(
            size=100,
            size_swa=50,
            page_size=1,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=kvcache,
            need_sort=False,
        )
        self.assertEqual(mock_ctor.call_count, 2)
        c4_call = mock_ctor.call_args_list[0]
        self.assertEqual(c4_call.args[0], 100)
        self.assertEqual(c4_call.kwargs["kvcache"], "c4_kv")
        self.assertEqual(c4_call.kwargs["page_size"], 1)
        c128_call = mock_ctor.call_args_list[1]
        self.assertEqual(c128_call.args[0], 50)
        self.assertEqual(c128_call.kwargs["kvcache"], "c128_kv")
        self.assertIsNone(alloc.c4_state_attn_allocator)
        self.assertIsNone(alloc.c128_state_attn_allocator)
        self.assertEqual(alloc._empty_loc.numel(), 0)
        self.assertEqual(alloc._empty_loc.dtype, torch.int64)
        self.assertIsNone(alloc._cur_req_to_token_pool)

    @patch(_MOD + ".NPUPagedTokenToKVPoolAllocator")
    def test_no_compress_state_pools_attr(self, mock_ctor):
        """kvcache without compress_state_pools → state allocators stay None."""
        kvcache = SimpleNamespace(
            c4_size=100,
            c4_kv_pool="c4_kv",
            c128_size=50,
            c128_kv_pool="c128_kv",
        )
        alloc = DSV4NPUTokenToKVPoolAllocator(
            size=100,
            size_swa=50,
            page_size=1,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=kvcache,
            need_sort=False,
        )
        self.assertIsNone(alloc.c4_state_attn_allocator)
        self.assertIsNone(alloc.c128_state_attn_allocator)
        self.assertEqual(mock_ctor.call_count, 2)

    @patch(_MOD + ".NPUPagedTokenToKVPoolAllocator")
    def test_empty_state_pools_list(self, mock_ctor):
        """compress_state_pools = [] (falsy) → state allocators stay None."""
        kvcache = _make_kvcache(with_state=False)
        kvcache.compress_state_pools = []
        alloc = DSV4NPUTokenToKVPoolAllocator(
            size=100,
            size_swa=50,
            page_size=1,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=kvcache,
            need_sort=False,
        )
        self.assertIsNone(alloc.c4_state_attn_allocator)
        self.assertIsNone(alloc.c128_state_attn_allocator)

    @patch(_MOD + ".NPUPagedTokenToKVPoolAllocator")
    def test_creates_both_state_allocators(self, mock_ctor):
        """Both state pools non-None and sizes > 0 → both state allocators created."""
        kvcache = _make_kvcache(
            c4_state_pool="c4s_pool",
            c128_state_pool="c128s_pool",
            c4_state_pool_size=10,
            c128_state_pool_size=20,
        )
        alloc = DSV4NPUTokenToKVPoolAllocator(
            size=100,
            size_swa=50,
            page_size=1,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=kvcache,
            need_sort=False,
        )
        self.assertEqual(mock_ctor.call_count, 4)
        self.assertIsNotNone(alloc.c4_state_attn_allocator)
        self.assertIsNotNone(alloc.c128_state_attn_allocator)
        # Call 2: c4 state allocator
        c4s_call = mock_ctor.call_args_list[2]
        self.assertEqual(c4s_call.args[0], 10)
        self.assertEqual(c4s_call.kwargs["kvcache"], "c4s_pool")
        # Call 3: c128 state allocator
        c128s_call = mock_ctor.call_args_list[3]
        self.assertEqual(c128s_call.args[0], 20)
        self.assertEqual(c128s_call.kwargs["kvcache"], "c128s_pool")

    @patch(_MOD + ".NPUPagedTokenToKVPoolAllocator")
    def test_state_pool_none_stays_none(self, mock_ctor):
        """c4 state pool is None → c4 state allocator stays None; c128 created."""
        kvcache = _make_kvcache(
            c4_state_pool=None,
            c128_state_pool="c128s_pool",
        )
        alloc = DSV4NPUTokenToKVPoolAllocator(
            size=100,
            size_swa=50,
            page_size=1,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=kvcache,
            need_sort=False,
        )
        self.assertIsNone(alloc.c4_state_attn_allocator)
        self.assertIsNotNone(alloc.c128_state_attn_allocator)
        self.assertEqual(mock_ctor.call_count, 3)

    @patch(_MOD + ".NPUPagedTokenToKVPoolAllocator")
    def test_state_pool_size_zero_stays_none(self, mock_ctor):
        """c4_state_pool_size == 0 → c4 state allocator stays None."""
        kvcache = _make_kvcache(
            c4_state_pool="c4s_pool",
            c128_state_pool="c128s_pool",
            c4_state_pool_size=0,
            c128_state_pool_size=20,
        )
        alloc = DSV4NPUTokenToKVPoolAllocator(
            size=100,
            size_swa=50,
            page_size=1,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=kvcache,
            need_sort=False,
        )
        self.assertIsNone(alloc.c4_state_attn_allocator)
        self.assertIsNotNone(alloc.c128_state_attn_allocator)

    @patch(_MOD + ".NPUPagedTokenToKVPoolAllocator")
    def test_first_state_pool_skips_none_entry(self, mock_ctor):
        """Duplicate ratio with first entry None → next() finds second entry."""
        kvcache = _make_kvcache(
            compression_ratios=[4, 4, 128],
            c4_state_pool=None,
            c128_state_pool="c128s_pool",
        )
        # Override state_pools to have 3 entries matching 3 ratios
        kvcache.compress_state_pools = [None, "c4s_pool_2", "c128s_pool"]
        alloc = DSV4NPUTokenToKVPoolAllocator(
            size=100,
            size_swa=50,
            page_size=1,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=kvcache,
            need_sort=False,
        )
        self.assertIsNotNone(alloc.c4_state_attn_allocator)
        self.assertIsNotNone(alloc.c128_state_attn_allocator)
        # c4 state allocator used "c4s_pool_2" (the non-None entry)
        c4s_call = mock_ctor.call_args_list[2]
        self.assertEqual(c4s_call.kwargs["kvcache"], "c4s_pool_2")


# ---------------------------------------------------------------------------
#  _compute_c_extend_counts / _pool_exhausted (static)
# ---------------------------------------------------------------------------


class TestComputeCExtendCounts(CustomTestCase):
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


class TestPoolExhausted(CustomTestCase):
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
#  compute_dsv4_state_lens_extend
# ---------------------------------------------------------------------------


class TestComputeStateLensExtend(CustomTestCase):
    """Tests for compute_dsv4_state_lens_extend."""

    def test_returns_none_when_no_state_allocator(self):
        """c4_state_attn_allocator None → returns None."""
        alloc = _make_allocator(with_state=False)
        self.assertIsNone(
            alloc.compute_dsv4_state_lens_extend([_make_req()], [130], [0])
        )

    def test_fresh_req_tail_within_short_window_adds_128_for_c4(self):
        """seq=130 prefix=0, tail=2 (≤3, ≥128) → c4_count=130, c128=2."""
        alloc = _make_allocator()
        req = _make_req()
        result = alloc.compute_dsv4_state_lens_extend([req], [130], [0])
        self.assertIsNotNone(result)
        self.assertEqual(req.c4_state_kv_len, 130)
        self.assertEqual(req.c128_state_kv_len, 2)
        self.assertEqual(req.c4_state_alloc_offset, 0)
        self.assertEqual(req.c128_state_alloc_offset, 128)
        self.assertEqual(req.c4_state_write_offset, 0)
        self.assertEqual(req.c128_state_write_offset, 128)

    def test_fresh_req_tail_outside_short_window(self):
        """seq=132 prefix=0, tail=4 (>3) → c4_count=4, c128=4."""
        alloc = _make_allocator()
        req = _make_req()
        alloc.compute_dsv4_state_lens_extend([req], [132], [0])
        self.assertEqual(req.c4_state_kv_len, 4)
        self.assertEqual(req.c128_state_kv_len, 4)
        self.assertEqual(req.c4_state_alloc_offset, 128)
        self.assertEqual(req.c128_state_alloc_offset, 128)
        self.assertEqual(req.c4_state_write_offset, 128)
        self.assertEqual(req.c128_state_write_offset, 128)

    def test_fresh_req_seq_below_128(self):
        """seq=100 prefix=0 (<128) → c4=tail=100, c128=tail=100."""
        alloc = _make_allocator()
        req = _make_req()
        alloc.compute_dsv4_state_lens_extend([req], [100], [0])
        self.assertEqual(req.c4_state_kv_len, 100)
        self.assertEqual(req.c128_state_kv_len, 100)

    def test_fresh_req_accumulates_across_calls(self):
        """State kv_len accumulates from previous value."""
        alloc = _make_allocator()
        req = _make_req(c4_kv=10, c128_kv=5)
        alloc.compute_dsv4_state_lens_extend([req], [132], [0])
        self.assertEqual(req.c4_state_kv_len, 14)
        self.assertEqual(req.c128_state_kv_len, 9)

    def test_fresh_req_multi_req_extend_num_tokens(self):
        """extend_num_tokens = sum of per-req c4_count."""
        alloc = _make_allocator()
        r1, r2 = _make_req(), _make_req()
        result = alloc.compute_dsv4_state_lens_extend([r1, r2], [132, 130], [0, 0])
        self.assertEqual(result.c4_extend_num_tokens, 4 + 130)
        self.assertEqual(result.c128_extend_num_tokens, 4 + 2)

    def test_fresh_req_pack_returns_tensors_on_device(self):
        """Packed tensors are on the allocator's device."""
        alloc = _make_allocator(device="cpu")
        r = _make_req()
        result = alloc.compute_dsv4_state_lens_extend([r], [130], [0])
        self.assertEqual(result.c4_prefix_lens.device.type, "cpu")
        self.assertEqual(result.c4_seq_lens.device.type, "cpu")
        self.assertEqual(result.c4_seq_lens.item(), 130)
        self.assertEqual(result.c4_prefix_lens.item(), 0)

    # --- prefix_len > 0: c4_count = min(c4_alloc_len, chunk_len) ---

    def test_prefix_gt_zero_caps_count_at_chunk_len(self):
        """prefix > 0, chunk_len < c4_alloc_len → c4_count = chunk_len."""
        alloc = _make_allocator()
        req = _make_req()
        # seq=132, prefix=130, tail=4, c4_alloc_len=4, chunk_len=2
        # c4_count = min(4, 2) = 2
        result = alloc.compute_dsv4_state_lens_extend([req], [132], [130])
        self.assertIsNotNone(result)
        self.assertEqual(req.c4_state_kv_len, 2)
        self.assertEqual(req.c128_state_kv_len, 2)
        self.assertEqual(req.c4_state_write_offset, 132 - 2)
        self.assertEqual(req.c128_state_write_offset, 132 - 2)
        # c4_state_alloc_offset NOT set when prefix > 0
        self.assertFalse(hasattr(req, "c4_state_alloc_offset"))

    def test_prefix_gt_zero_capped_at_alloc_len(self):
        """prefix > 0, chunk_len > c4_alloc_len → c4_count = c4_alloc_len."""
        alloc = _make_allocator()
        req = _make_req()
        # seq=132, prefix=100, tail=4, c4_alloc_len=4, chunk_len=32
        # c4_count = min(4, 32) = 4
        result = alloc.compute_dsv4_state_lens_extend([req], [132], [100])
        self.assertIsNotNone(result)
        self.assertEqual(req.c4_state_kv_len, 4)
        self.assertEqual(req.c128_state_kv_len, 4)

    def test_prefix_gt_zero_does_not_set_alloc_offset(self):
        """prefix > 0 → c4_state_alloc_offset not touched."""
        alloc = _make_allocator()
        req = _make_req()
        alloc.compute_dsv4_state_lens_extend([req], [132], [100])
        self.assertFalse(hasattr(req, "c4_state_alloc_offset"))

    def test_prefix_gt_zero_accumulates(self):
        """prefix > 0 with existing c4_state_kv_len."""
        alloc = _make_allocator()
        req = _make_req(c4_kv=10, c128_kv=5)
        # seq=132, prefix=130, chunk_len=2, c4_count=2
        alloc.compute_dsv4_state_lens_extend([req], [132], [130])
        self.assertEqual(req.c4_state_kv_len, 12)
        self.assertEqual(req.c128_state_kv_len, 7)


# ---------------------------------------------------------------------------
#  compute_dsv4_state_lens_decode
# ---------------------------------------------------------------------------


class TestComputeStateLensDecode(CustomTestCase):
    """Tests for compute_dsv4_state_lens_decode."""

    def test_returns_none_when_no_state_allocator(self):
        """c4_state_attn_allocator None → returns None."""
        alloc = _make_allocator(with_state=False)
        self.assertIsNone(alloc.compute_dsv4_state_lens_decode([_make_req()]))

    def test_one_slot_per_req(self):
        """Each req gets exactly 1 new state slot per pool."""
        alloc = _make_allocator()
        r1 = _make_req(c4_kv=10, c128_kv=5)
        r2 = _make_req(c4_kv=20, c128_kv=15)
        result = alloc.compute_dsv4_state_lens_decode([r1, r2])
        self.assertEqual(r1.c4_state_kv_len, 11)
        self.assertEqual(r1.c128_state_kv_len, 6)
        self.assertEqual(r2.c4_state_kv_len, 21)
        self.assertEqual(r2.c128_state_kv_len, 16)
        self.assertEqual(result.c4_extend_num_tokens, 2)
        self.assertEqual(result.c128_extend_num_tokens, 2)
        self.assertEqual(result.c4_prefix_lens.tolist(), [10, 20])
        self.assertEqual(result.c4_seq_lens.tolist(), [11, 21])


# ---------------------------------------------------------------------------
#  compute_dsv4_state_lens_reserve
# ---------------------------------------------------------------------------


class TestComputeStateLensReserve(CustomTestCase):
    """Tests for compute_dsv4_state_lens_reserve."""

    def test_returns_none_when_no_state_allocator(self):
        """c4_state_attn_allocator None → returns None."""
        alloc = _make_allocator(with_state=False)
        self.assertIsNone(
            alloc.compute_dsv4_state_lens_reserve([_make_req()], [0], [8])
        )

    def test_reserve_len_per_req(self):
        """reserve = seq - prefix per req."""
        alloc = _make_allocator()
        r1 = _make_req(c4_kv=3, c128_kv=2)
        r2 = _make_req(c4_kv=0, c128_kv=0)
        result = alloc.compute_dsv4_state_lens_reserve([r1, r2], [0, 4], [8, 12])
        self.assertEqual(r1.c4_state_kv_len, 3 + 8)
        self.assertEqual(r1.c128_state_kv_len, 2 + 8)
        self.assertEqual(r2.c4_state_kv_len, 0 + 8)
        self.assertEqual(r2.c128_state_kv_len, 0 + 8)
        self.assertEqual(result.c4_extend_num_tokens, 8 + 8)
        self.assertEqual(result.c128_extend_num_tokens, 8 + 8)

    def test_zero_reserve(self):
        """seq == prefix → 0 new state slots."""
        alloc = _make_allocator()
        r = _make_req(c4_kv=5, c128_kv=5)
        result = alloc.compute_dsv4_state_lens_reserve([r], [8], [8])
        self.assertEqual(r.c4_state_kv_len, 5)
        self.assertEqual(result.c4_extend_num_tokens, 0)


# ---------------------------------------------------------------------------
#  _alloc_c_extend
# ---------------------------------------------------------------------------


class TestAllocCExtend(CustomTestCase):
    """Tests for DSV4NPUTokenToKVPoolAllocator._alloc_c_extend."""

    def test_returns_empty_when_no_new_compressed_tokens(self):
        """prefix and seq in same ratio bucket → empty result."""
        alloc = _make_allocator()
        prefix = torch.tensor([8])
        seq = torch.tensor([9])
        result = alloc._alloc_c_extend(
            alloc.c4_attn_allocator,
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
        alloc.c4_attn_allocator.alloc_extend.return_value = expected
        result = alloc._alloc_c_extend(
            alloc.c4_attn_allocator,
            prefix,
            prefix,
            seq,
            seq,
            torch.tensor([0]),
            torch.int64,
            ratio=4,
        )
        self.assertTrue(torch.equal(result, expected))
        call = alloc.c4_attn_allocator.alloc_extend.call_args
        self.assertEqual(call.args[5], 2)

    def test_pool_exhausted_raises(self):
        """Sub-allocator returns None → RuntimeError."""
        alloc = _make_allocator()
        alloc._cur_req_to_token_pool = _make_pool()
        alloc.c4_attn_allocator.alloc_extend.return_value = None
        alloc.c4_attn_allocator.available_size.return_value = 1
        prefix = torch.tensor([0])
        seq = torch.tensor([8])
        with self.assertRaises(RuntimeError):
            alloc._alloc_c_extend(
                alloc.c4_attn_allocator,
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
                alloc.c4_attn_allocator,
                prefix,
                prefix,
                seq,
                seq,
                torch.tensor([0]),
                torch.int64,
                ratio=4,
            )


# ---------------------------------------------------------------------------
#  _alloc_state_extend
# ---------------------------------------------------------------------------


class TestAllocStateExtend(CustomTestCase):
    """Tests for DSV4NPUTokenToKVPoolAllocator._alloc_state_extend."""

    def test_returns_empty_when_allocator_none(self):
        """allocator None → returns _empty_loc."""
        alloc = _make_allocator(with_state=False)
        result = alloc._alloc_state_extend(
            None,
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([1]),
            torch.tensor([0]),
            torch.int64,
            1,
            ratio=4,
        )
        self.assertEqual(result.numel(), 0)

    def test_returns_empty_when_zero_extend_tokens(self):
        """state_extend_num_tokens == 0 → returns _empty_loc."""
        alloc = _make_allocator()
        result = alloc._alloc_state_extend(
            alloc.c4_state_attn_allocator,
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.int64,
            0,
            ratio=4,
        )
        self.assertEqual(result.numel(), 0)

    def test_normal_alloc(self):
        """Normal state extend → calls sub-allocator.alloc_extend."""
        alloc = _make_allocator()
        alloc._cur_req_to_token_pool = _make_pool()
        expected = torch.tensor([5, 6], dtype=torch.int32)
        alloc.c4_state_attn_allocator.alloc_extend.return_value = expected
        result = alloc._alloc_state_extend(
            alloc.c4_state_attn_allocator,
            raw_prefix_lens=torch.tensor([0]),
            state_prefix_lens=torch.tensor([0]),
            state_prefix_lens_cpu=torch.tensor([0]),
            state_seq_lens=torch.tensor([2]),
            state_seq_lens_cpu=torch.tensor([2]),
            req_pool_indices=torch.tensor([0]),
            last_loc_dtype=torch.int64,
            state_extend_num_tokens=2,
            ratio=4,
        )
        self.assertTrue(torch.equal(result, expected))

    def test_pool_exhausted_raises(self):
        """Sub-allocator returns None → RuntimeError."""
        alloc = _make_allocator()
        alloc._cur_req_to_token_pool = _make_pool()
        alloc.c4_state_attn_allocator.alloc_extend.return_value = None
        alloc.c4_state_attn_allocator.available_size.return_value = 0
        with self.assertRaises(RuntimeError):
            alloc._alloc_state_extend(
                alloc.c4_state_attn_allocator,
                torch.tensor([0]),
                torch.tensor([0]),
                torch.tensor([0]),
                torch.tensor([2]),
                torch.tensor([2]),
                torch.tensor([0]),
                torch.int64,
                2,
                ratio=4,
            )

    def test_missing_pool_asserts(self):
        """_cur_req_to_token_pool None → AssertionError."""
        alloc = _make_allocator()
        alloc._cur_req_to_token_pool = None
        with self.assertRaises(AssertionError):
            alloc._alloc_state_extend(
                alloc.c4_state_attn_allocator,
                torch.tensor([0]),
                torch.tensor([0]),
                torch.tensor([0]),
                torch.tensor([2]),
                torch.tensor([2]),
                torch.tensor([0]),
                torch.int64,
                2,
                ratio=4,
            )


# ---------------------------------------------------------------------------
#  _alloc_c_and_state
# ---------------------------------------------------------------------------


class TestAllocCAndState(CustomTestCase):
    """Tests for DSV4NPUTokenToKVPoolAllocator._alloc_c_and_state."""

    def _args(self, **over):
        defaults = dict(
            out_full_loc=torch.tensor([1, 2]),
            out_swa_loc=torch.tensor([3, 4]),
            prefix_lens=torch.tensor([0]),
            prefix_lens_cpu=torch.tensor([0]),
            seq_lens=torch.tensor([8]),
            seq_lens_cpu=torch.tensor([8]),
            last_loc_dtype=torch.int64,
            req_pool_indices=torch.tensor([0]),
            dsv4_state_lens=None,
        )
        defaults.update(over)
        return defaults

    def test_requires_req_pool_indices(self):
        """req_pool_indices None → AssertionError."""
        alloc = _make_allocator()
        with self.assertRaises(AssertionError):
            alloc._alloc_c_and_state(**{**self._args(), "req_pool_indices": None})

    @patch.object(DSV4NPUTokenToKVPoolAllocator, "_alloc_c_extend")
    @patch.object(DSV4NPUTokenToKVPoolAllocator, "_alloc_state_extend")
    def test_without_state_lens(self, mock_state, mock_c):
        """dsv4_state_lens None → state locs are _empty_loc, c locs populated."""
        alloc = _make_allocator()
        mock_c.side_effect = [torch.tensor([10]), torch.tensor([20])]
        mock_state.return_value = torch.empty(0, dtype=torch.int64)
        result = alloc._alloc_c_and_state(**self._args())
        self.assertIsInstance(result, DSV4OutCacheLoc)
        self.assertTrue(torch.equal(result.out_full_loc, torch.tensor([1, 2])))
        self.assertTrue(torch.equal(result.out_c4_loc, torch.tensor([10])))
        self.assertTrue(torch.equal(result.out_c128_loc, torch.tensor([20])))
        self.assertEqual(result.out_c4_state_loc.numel(), 0)
        self.assertEqual(result.out_c128_state_loc.numel(), 0)
        mock_state.assert_not_called()

    @patch.object(DSV4NPUTokenToKVPoolAllocator, "_alloc_c_extend")
    @patch.object(DSV4NPUTokenToKVPoolAllocator, "_alloc_state_extend")
    def test_with_state_lens(self, mock_state, mock_c):
        """dsv4_state_lens set → all six loc fields populated."""
        alloc = _make_allocator()
        mock_c.side_effect = [torch.tensor([10]), torch.tensor([20])]
        mock_state.side_effect = [torch.tensor([30]), torch.tensor([40])]
        state_lens = DSV4StateLens(
            c4_prefix_lens=torch.tensor([0]),
            c4_prefix_lens_cpu=torch.tensor([0]),
            c4_seq_lens=torch.tensor([2]),
            c4_seq_lens_cpu=torch.tensor([2]),
            c4_extend_num_tokens=2,
            c128_prefix_lens=torch.tensor([0]),
            c128_prefix_lens_cpu=torch.tensor([0]),
            c128_seq_lens=torch.tensor([2]),
            c128_seq_lens_cpu=torch.tensor([2]),
            c128_extend_num_tokens=2,
        )
        result = alloc._alloc_c_and_state(**self._args(dsv4_state_lens=state_lens))
        self.assertTrue(torch.equal(result.out_c4_state_loc, torch.tensor([30])))
        self.assertTrue(torch.equal(result.out_c128_state_loc, torch.tensor([40])))
        self.assertEqual(mock_state.call_count, 2)
        self.assertEqual(mock_c.call_count, 2)


# ---------------------------------------------------------------------------
#  alloc_extend (public API)
# ---------------------------------------------------------------------------


class TestAllocExtend(CustomTestCase):
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

    @patch.object(DSV4NPUTokenToKVPoolAllocator, "_alloc_c_and_state")
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
    def test_stashes_pool_and_delegates(self, mock_super, mock_swa, mock_cas):
        """super().alloc_extend succeeds → _alloc_c_and_state called with full_loc."""
        alloc = _make_allocator()
        mock_cas.return_value = "bundle"
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
        args = mock_cas.call_args.args
        self.assertTrue(torch.equal(args[0], torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])))
        self.assertEqual(args[2].item(), 0)


# ---------------------------------------------------------------------------
#  alloc_decode (public API)
# ---------------------------------------------------------------------------


class TestAllocDecode(CustomTestCase):
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

    @patch.object(DSV4NPUTokenToKVPoolAllocator, "_alloc_c_and_state")
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
    def test_derives_prefix_lens_and_delegates(self, mock_super, mock_swa, mock_cas):
        """prefix_lens = seq_lens - 1 passed to _alloc_c_and_state."""
        alloc = _make_allocator()
        alloc._cur_req_to_token_pool = _make_pool()
        mock_cas.return_value = "bundle"
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
        args = mock_cas.call_args.args
        self.assertEqual(args[2].item(), 8)
        self.assertEqual(args[3].item(), 8)


# ---------------------------------------------------------------------------
#  free
# ---------------------------------------------------------------------------


class TestFree(CustomTestCase):
    """Tests for DSV4NPUTokenToKVPoolAllocator.free."""

    def test_free_index_only_calls_super(self):
        """free(free_index) → only super().free is called."""
        alloc = _make_allocator()
        idx = torch.tensor([1, 2, 3])
        with patch.object(_SWATokenToKVPoolAllocatorStub, "free") as super_free:
            alloc.free(idx)
            super_free.assert_called_once_with(idx)

    def test_no_req_no_c_pool_free(self):
        """free(index) without req → c4/c128 allocators not called."""
        alloc = _make_allocator()
        alloc.free(torch.tensor([1, 2]))
        alloc.c4_attn_allocator.free.assert_not_called()
        alloc.c128_attn_allocator.free.assert_not_called()

    def test_req_free_frees_kv_pools(self):
        """free(req=, pool=) → c4 KV slots [0, kv_len//4) freed."""
        alloc = _make_allocator()
        pool = _make_pool(n_reqs=2, max_len=32)
        req = _make_req(committed=10, allocated=10, pool_idx=0)
        alloc.free(req=req, req_to_token_pool=pool)
        call4 = alloc.c4_attn_allocator.free.call_args[0][0]
        self.assertTrue(torch.equal(call4, torch.tensor([1, 2], dtype=torch.int64)))
        alloc.c128_attn_allocator.free.assert_not_called()

    def test_req_free_frees_state_pools_tail_only(self):
        """State pools free only [alloc_offset, kv_len) tail."""
        alloc = _make_allocator()
        pool = _make_pool(n_reqs=2, max_len=32)
        req = _make_req(
            committed=10,
            allocated=10,
            pool_idx=0,
            c4_off=3,
            c128_off=5,
        )
        alloc.free(req=req, req_to_token_pool=pool)
        c4_call = alloc.c4_state_attn_allocator.free.call_args[0][0]
        self.assertEqual(len(c4_call), 7)
        c128_call = alloc.c128_state_attn_allocator.free.call_args[0][0]
        self.assertEqual(len(c128_call), 5)

    def test_state_allocator_none_skipped(self):
        """State allocators None → KV free still works, no crash."""
        alloc = _make_allocator(with_state=False)
        pool = _make_pool(n_reqs=2, max_len=32)
        req = _make_req(committed=10, allocated=10, pool_idx=0)
        alloc.free(req=req, req_to_token_pool=pool)

    def test_zero_kv_len_no_free(self):
        """kv_len == 0 → no KV or state free calls."""
        alloc = _make_allocator()
        pool = _make_pool()
        req = _make_req(committed=0, allocated=0, pool_idx=0)
        alloc.free(req=req, req_to_token_pool=pool)
        alloc.c4_attn_allocator.free.assert_not_called()

    def test_uses_max_of_committed_and_allocated(self):
        """kv_len = max(committed, allocated)."""
        alloc = _make_allocator()
        pool = _make_pool(n_reqs=2, max_len=32)
        req = _make_req(committed=5, allocated=12, pool_idx=0)
        alloc.free(req=req, req_to_token_pool=pool)
        call4 = alloc.c4_attn_allocator.free.call_args[0][0]
        self.assertEqual(len(call4), 3)

    def test_kv_len_le_offset_skips_state_free(self):
        """kv_len <= alloc_offset → state free skipped."""
        alloc = _make_allocator()
        pool = _make_pool(n_reqs=2, max_len=32)
        req = _make_req(committed=3, allocated=3, pool_idx=0, c4_off=5, c128_off=5)
        alloc.free(req=req, req_to_token_pool=pool)
        alloc.c4_state_attn_allocator.free.assert_not_called()
        alloc.c128_state_attn_allocator.free.assert_not_called()


# ---------------------------------------------------------------------------
#  _wrap_full_alloc
# ---------------------------------------------------------------------------


class TestWrapFullAlloc(CustomTestCase):
    """Tests for DSV4NPUTokenToKVPoolAllocator._wrap_full_alloc."""

    @patch.object(DSV4NPUTokenToKVPoolAllocator, "_alloc_c_and_state")
    @patch.object(
        DSV4NPUTokenToKVPoolAllocator,
        "translate_loc_from_full_to_swa",
        return_value=torch.tensor([9]),
    )
    def test_none_returns_none(self, mock_swa, mock_cas):
        """out_full_loc None → returns None without translate/c_and_state."""
        alloc = _make_allocator()
        result = alloc._wrap_full_alloc(
            None,
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([8]),
            torch.tensor([8]),
            torch.int64,
            torch.tensor([0]),
            None,
        )
        self.assertIsNone(result)
        mock_swa.assert_not_called()
        mock_cas.assert_not_called()

    @patch.object(DSV4NPUTokenToKVPoolAllocator, "_alloc_c_and_state")
    @patch.object(
        DSV4NPUTokenToKVPoolAllocator,
        "translate_loc_from_full_to_swa",
        return_value=torch.tensor([9]),
    )
    def test_translates_and_delegates(self, mock_swa, mock_cas):
        """out_full_loc not None → translate + _alloc_c_and_state."""
        alloc = _make_allocator()
        mock_cas.return_value = "bundle"
        full_loc = torch.tensor([1, 2])
        result = alloc._wrap_full_alloc(
            full_loc,
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([8]),
            torch.tensor([8]),
            torch.int64,
            torch.tensor([0]),
            None,
        )
        self.assertEqual(result, "bundle")
        mock_swa.assert_called_once_with(full_loc)
        mock_cas.assert_called_once()


# ---------------------------------------------------------------------------
#  alloc_extend_swa_tail (public API)
# ---------------------------------------------------------------------------


class TestAllocExtendSwaTail(CustomTestCase):
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

    @patch.object(DSV4NPUTokenToKVPoolAllocator, "_alloc_c_and_state")
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
    def test_delegates_to_wrap_full_alloc(self, mock_super, mock_swa, mock_cas):
        """super succeeds → _wrap_full_alloc translates + delegates."""
        alloc = _make_allocator()
        mock_cas.return_value = "bundle"
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
        args = mock_cas.call_args.args
        self.assertTrue(torch.equal(args[0], torch.tensor([1, 2, 3])))


# ---------------------------------------------------------------------------
#  clear
# ---------------------------------------------------------------------------


class TestClear(CustomTestCase):
    """Tests for clear."""


if __name__ == "__main__":
    unittest.main()
