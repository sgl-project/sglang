"""Unit tests for DSV4NPUReqToTokenPool."""

import os
import sys
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

# Ensure the workspace-local sglang copy (D:\sglang\python) is imported,
# not a stale editable-install copy that may differ. Four ".." climb from
# test/registered/npu/dsv4/ up to the repo root, then into python/.
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "..",
        "python",
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

# --- TorchMemorySaverAdapter stub (real class — used by the pool __init__) ---


class _NoopMemorySaverAdapter:
    @contextmanager
    def region(self, tag, enable_cpu_backup=False):
        yield

    @property
    def enabled(self):
        return False


class _TorchMemorySaverAdapter:
    @staticmethod
    def create(enable):
        return _NoopMemorySaverAdapter()


_adapt_mod = type(sys)("sglang.srt.utils.torch_memory_saver_adapter")
_adapt_mod.TorchMemorySaverAdapter = _TorchMemorySaverAdapter
sys.modules["sglang.srt.utils.torch_memory_saver_adapter"] = _adapt_mod

# --- ReqToTokenPool stub (real base class — DSV4NPUReqToTokenPool inherits it) ---
# Mirrors the production base: sets size/_alloc_size/max_context_len/device and
# creates req_to_token / free_slots / req_generation, plus write/available_size/
# free/clear. The aux tables themselves are created by the DSV4 subclass.

import torch  # noqa: E402


class _ReqToTokenPoolStub:
    def __init__(self, size, max_context_len, device, enable_memory_saver):
        self.size = size
        # +1 padding row at index 0 (cuda-graph dummy slot).
        self._alloc_size = size + 1
        self.max_context_len = max_context_len
        self.device = device
        self.req_to_token = torch.zeros(
            (self._alloc_size, max_context_len), dtype=torch.int32, device=device
        )
        self.free_slots = list(range(1, self._alloc_size))
        self.req_generation = torch.zeros(self._alloc_size, dtype=torch.int64)

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def free(self, req):
        assert req.req_pool_idx is not None, "request must have req_pool_idx"
        self.free_slots.append(req.req_pool_idx)
        req.req_pool_idx = None

    def clear(self):
        self.free_slots = list(range(1, self._alloc_size))
        self.req_generation.zero_()


_mp_mod = type(sys)("sglang.srt.mem_cache.memory_pool")
_mp_mod.ReqToTokenPool = _ReqToTokenPoolStub
sys.modules["sglang.srt.mem_cache.memory_pool"] = _mp_mod

# --- DecodeReqToTokenPool stub (real base for DSV4NPUDecodeReqToTokenPool) ---
# Mirrors production sglang/srt/disaggregation/decode.py: _alloc_size adds the
# pre-allocated in-flight prefill slots on top of the size + 1 padding row.


class _DecodeReqToTokenPoolStub:
    def __init__(
        self, size, max_context_len, device, enable_memory_saver, pre_alloc_size
    ):
        self.size = size
        # +1 padding row at index 0; pre_alloc_size extra slots for in-flight
        # prefill transfers.
        self._alloc_size = size + pre_alloc_size + 1
        self.max_context_len = max_context_len
        self.device = device
        self.pre_alloc_size = pre_alloc_size
        self.req_to_token = torch.zeros(
            (self._alloc_size, max_context_len), dtype=torch.int32, device=device
        )
        self.free_slots = list(range(1, self._alloc_size))
        self.req_generation = torch.zeros(self._alloc_size, dtype=torch.int64)

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def free(self, req):
        assert req.req_pool_idx is not None, "request must have req_pool_idx"
        self.free_slots.append(req.req_pool_idx)
        req.req_pool_idx = None

    def clear(self):
        self.free_slots = list(range(1, self._alloc_size))
        self.req_generation.zero_()


_decode_mod = type(sys)("sglang.srt.disaggregation.decode")
_decode_mod.DecodeReqToTokenPool = _DecodeReqToTokenPoolStub
sys.modules["sglang.srt.disaggregation.decode"] = _decode_mod

# sglang.version
_ver = type(sys)("sglang.version")
_ver.__version__ = "0.0.0.dev0"
sys.modules["sglang.version"] = _ver

from sglang.srt.hardware_backend.npu.dsv4.dsv4_req_to_token_pool import (  # noqa: E402
    DSV4NPUDecodeReqToTokenPool,
    DSV4NPUReqToTokenPool,
)


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


def _make_pool(size=4, max_context_len=64, device="cpu"):
    """Construct a real DSV4NPUReqToTokenPool on CPU with memory saver off."""
    return DSV4NPUReqToTokenPool(
        size=size,
        max_context_len=max_context_len,
        device=device,
        enable_memory_saver=False,
    )


def _make_decode_pool(size=4, max_context_len=64, pre_alloc_size=2, device="cpu"):
    """Construct a real DSV4NPUDecodeReqToTokenPool (disagg-decode variant)."""
    return DSV4NPUDecodeReqToTokenPool(
        size=size,
        max_context_len=max_context_len,
        device=device,
        enable_memory_saver=False,
        pre_alloc_size=pre_alloc_size,
    )


def _make_req(pool_idx=0):
    """SimpleNamespace req carrying only req_pool_idx (all free() needs)."""
    req = SimpleNamespace()
    req.req_pool_idx = pool_idx
    return req


_AUX_TABLES = (
    "req_to_token_swa",
    "req_to_token_c4",
    "req_to_token_c128",
    "req_to_token_c4_state",
    "req_to_token_c128_state",
)


# ---------------------------------------------------------------------------
#  __init__
# ---------------------------------------------------------------------------


class TestInit(CustomTestCase):
    """Tests for DSV4NPUReqToTokenPool.__init__."""

    def test_creates_all_five_aux_tables(self):
        """All five auxiliary tables exist and are int32 tensors."""
        pool = _make_pool()
        for name in _AUX_TABLES:
            table = getattr(pool, name)
            self.assertIsInstance(table, torch.Tensor)
            self.assertEqual(table.dtype, torch.int32)

    def test_table_rows_match_alloc_size(self):
        """Each aux table has _alloc_size (= size + 1) rows."""
        pool = _make_pool(size=4)
        for name in _AUX_TABLES:
            self.assertEqual(getattr(pool, name).shape[0], pool._alloc_size)
            self.assertEqual(getattr(pool, name).shape[0], 5)

    def test_swa_and_state_columns_equal_max_context_len(self):
        """swa + both state tables have one column per raw token."""
        pool = _make_pool(max_context_len=64)
        for name in (
            "req_to_token_swa",
            "req_to_token_c4_state",
            "req_to_token_c128_state",
        ):
            self.assertEqual(getattr(pool, name).shape[1], 64)

    def test_c4_columns_divided_by_ratio_4(self):
        """c4 table has max_context_len // 4 columns."""
        pool = _make_pool(max_context_len=64)
        self.assertEqual(pool.req_to_token_c4.shape[1], 16)

    def test_c128_columns_divided_by_ratio_128(self):
        """c128 table has max_context_len // 128 columns."""
        pool = _make_pool(max_context_len=256)
        self.assertEqual(pool.req_to_token_c128.shape[1], 2)

    def test_c4_floors_to_one_when_below_ratio(self):
        """max_context_len < 4 → c4 columns clamped to 1 via max(1, ...)."""
        pool = _make_pool(max_context_len=2)
        self.assertEqual(pool.req_to_token_c4.shape[1], 1)

    def test_c128_floors_to_one_when_below_ratio(self):
        """max_context_len < 128 → c128 columns clamped to 1 via max(1, ...)."""
        pool = _make_pool(max_context_len=64)
        self.assertEqual(pool.req_to_token_c128.shape[1], 1)

    def test_dsv4_allocator_starts_none(self):
        """_dsv4_allocator is None until register_dsv4_allocator wires it."""
        pool = _make_pool()
        self.assertIsNone(pool._dsv4_allocator)

    def test_aux_tables_zero_initialized(self):
        """All five aux tables start zeroed so unallocated cols map to block 0."""
        pool = _make_pool()
        for name in _AUX_TABLES:
            self.assertTrue(bool((getattr(pool, name) == 0).all()))

    def test_base_attributes_set(self):
        """super().__init__ sets size/_alloc_size/device/req_to_token/etc."""
        pool = _make_pool(size=4, max_context_len=64)
        self.assertEqual(pool.size, 4)
        self.assertEqual(pool._alloc_size, 5)
        self.assertEqual(pool.max_context_len, 64)
        self.assertEqual(pool.device, "cpu")
        self.assertEqual(pool.req_to_token.shape, (5, 64))
        self.assertEqual(pool.req_to_token.dtype, torch.int32)
        self.assertEqual(pool.req_generation.shape, (5,))
        self.assertEqual(pool.req_generation.dtype, torch.int64)
        self.assertEqual(pool.free_slots, [1, 2, 3, 4])

    def test_tables_on_specified_device(self):
        """Aux tables are placed on the requested device."""
        pool = _make_pool(device="cpu")
        for name in _AUX_TABLES:
            self.assertEqual(getattr(pool, name).device.type, "cpu")


# ---------------------------------------------------------------------------
#  write_* helpers
# ---------------------------------------------------------------------------


class TestWriteHelpers(CustomTestCase):
    """Tests for write_swa / write_c4 / write_c128 / write_c4_state /
    write_c128_state."""

    def test_write_swa(self):
        pool = _make_pool(max_context_len=8)
        values = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int32)
        pool.write_swa(0, values)
        self.assertTrue(torch.equal(pool.req_to_token_swa[0], values))

    def test_write_c4(self):
        pool = _make_pool(max_context_len=8)
        values = torch.tensor([10, 11], dtype=torch.int32)
        pool.write_c4(0, values)
        self.assertTrue(torch.equal(pool.req_to_token_c4[0], values))

    def test_write_c128(self):
        pool = _make_pool(max_context_len=8)
        values = torch.tensor([20], dtype=torch.int32)
        pool.write_c128(0, values)
        self.assertTrue(torch.equal(pool.req_to_token_c128[0], values))

    def test_write_c4_state(self):
        pool = _make_pool(max_context_len=8)
        values = torch.tensor([30, 31, 32, 33, 34, 35, 36, 37], dtype=torch.int32)
        pool.write_c4_state(0, values)
        self.assertTrue(torch.equal(pool.req_to_token_c4_state[0], values))

    def test_write_c128_state(self):
        pool = _make_pool(max_context_len=8)
        values = torch.tensor([40, 41, 42, 43, 44, 45, 46, 47], dtype=torch.int32)
        pool.write_c128_state(0, values)
        self.assertTrue(torch.equal(pool.req_to_token_c128_state[0], values))

    def test_write_swa_does_not_affect_other_tables(self):
        """Writing swa leaves c4/c128/state tables and other rows untouched."""
        pool = _make_pool(max_context_len=8)
        pool.write_swa(0, torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int32))
        for name in (
            "req_to_token_c4",
            "req_to_token_c128",
            "req_to_token_c4_state",
            "req_to_token_c128_state",
        ):
            self.assertTrue(bool((getattr(pool, name)[0] == 0).all()))
        # swa row 1 still zero.
        self.assertTrue(bool((pool.req_to_token_swa[1] == 0).all()))


# ---------------------------------------------------------------------------
#  register_dsv4_allocator
# ---------------------------------------------------------------------------


class TestRegisterAllocator(CustomTestCase):
    """Tests for register_dsv4_allocator."""

    def test_sets_dsv4_allocator(self):
        pool = _make_pool()
        alloc = MagicMock()
        pool.register_dsv4_allocator(alloc)
        self.assertIs(pool._dsv4_allocator, alloc)

    def test_overwrites_previous_allocator(self):
        pool = _make_pool()
        pool.register_dsv4_allocator(MagicMock())
        second = MagicMock()
        pool.register_dsv4_allocator(second)
        self.assertIs(pool._dsv4_allocator, second)


# ---------------------------------------------------------------------------
#  free
# ---------------------------------------------------------------------------


class TestFree(CustomTestCase):
    """Tests for DSV4NPUReqToTokenPool.free."""

    def test_free_without_allocator_only_super(self):
        """_dsv4_allocator None → only super().free runs (slot freed)."""
        pool = _make_pool(size=4)
        self.assertIsNone(pool._dsv4_allocator)
        req = _make_req(pool_idx=3)
        before = pool.available_size()
        pool.free(req)
        self.assertIsNone(req.req_pool_idx)
        self.assertEqual(pool.available_size(), before + 1)
        self.assertIn(3, pool.free_slots)

    def test_free_with_allocator_delegates(self):
        """With allocator wired, free(req) delegates req + pool ref to it."""
        pool = _make_pool(size=4)
        alloc = MagicMock()
        pool.register_dsv4_allocator(alloc)
        req = _make_req(pool_idx=3)
        pool.free(req)
        alloc.free.assert_called_once_with(req=req, req_to_token_pool=pool)

    def test_free_passes_pool_ref_as_self(self):
        """req_to_token_pool kwarg is the pool instance itself."""
        pool = _make_pool(size=4)
        alloc = MagicMock()
        pool.register_dsv4_allocator(alloc)
        req = _make_req(pool_idx=2)
        pool.free(req)
        self.assertIs(alloc.free.call_args.kwargs["req_to_token_pool"], pool)
        self.assertIs(alloc.free.call_args.kwargs["req"], req)

    def test_free_runs_allocator_before_super(self):
        """allocator.free runs while req_pool_idx is still set (before super)."""
        pool = _make_pool(size=4)
        alloc = MagicMock()
        pool.register_dsv4_allocator(alloc)
        seen = {}

        def snap(req, req_to_token_pool):
            seen["idx"] = req.req_pool_idx

        alloc.free.side_effect = snap
        req = _make_req(pool_idx=5)
        pool.free(req)
        self.assertEqual(seen["idx"], 5)
        self.assertIsNone(req.req_pool_idx)

    def test_free_super_appends_slot_and_clears_idx(self):
        """super().free still runs after allocator: slot appended, idx cleared."""
        pool = _make_pool(size=4)
        alloc = MagicMock()
        pool.register_dsv4_allocator(alloc)
        req = _make_req(pool_idx=4)
        pool.free(req)
        self.assertIsNone(req.req_pool_idx)
        self.assertIn(4, pool.free_slots)


# ---------------------------------------------------------------------------
#  clear (inherited — documented invariant: aux tables NOT zeroed)
# ---------------------------------------------------------------------------


class TestClear(CustomTestCase):
    """Tests for the inherited clear() as observed through the DSV4 pool."""

    def test_clear_resets_free_slots(self):
        """clear() restores free_slots to the full range."""
        pool = _make_pool(size=4, max_context_len=8)
        pool.free_slots.pop()
        self.assertEqual(len(pool.free_slots), 3)
        pool.clear()
        self.assertEqual(pool.free_slots, [1, 2, 3, 4])

    def test_clear_resets_req_generation(self):
        """clear() zeroes req_generation."""
        pool = _make_pool(size=4, max_context_len=8)
        pool.req_generation.fill_(7)
        pool.clear()
        self.assertTrue(bool((pool.req_generation == 0).all()))

    def test_clear_does_not_zero_aux_tables(self):
        """Documented invariant: clear() leaves the five aux tables untouched."""
        pool = _make_pool(size=4, max_context_len=8)
        pool.req_to_token_swa[0] = torch.tensor(
            [1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int32
        )
        pool.req_to_token_c4[0] = torch.tensor([9, 10], dtype=torch.int32)
        pool.clear()
        self.assertTrue(
            torch.equal(
                pool.req_to_token_swa[0],
                torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                pool.req_to_token_c4[0], torch.tensor([9, 10], dtype=torch.int32)
            )
        )


# ---------------------------------------------------------------------------
#  DSV4NPUDecodeReqToTokenPool (disagg-decode variant) — exercises the shared
#  mixin (_init_dsv4_tables / _dsv4_free / write_* / register_dsv4_allocator)
#  over a DecodeReqToTokenPool base instead of ReqToTokenPool.
# ---------------------------------------------------------------------------


class TestDecodeInit(CustomTestCase):
    """Tests for DSV4NPUDecodeReqToTokenPool.__init__."""

    def test_creates_all_five_aux_tables(self):
        """Mixin _init_dsv4_tables runs on top of the decode base too."""
        pool = _make_decode_pool()
        for name in _AUX_TABLES:
            table = getattr(pool, name)
            self.assertIsInstance(table, torch.Tensor)
            self.assertEqual(table.dtype, torch.int32)

    def test_alloc_size_includes_pre_alloc(self):
        """_alloc_size = size + pre_alloc_size + 1; all 5 tables have that many rows."""
        pool = _make_decode_pool(size=4, pre_alloc_size=2)
        self.assertEqual(pool._alloc_size, 7)
        for name in _AUX_TABLES:
            self.assertEqual(getattr(pool, name).shape[0], 7)

    def test_c4_c128_column_ratios_hold(self):
        """c4 = max_context_len // 4, c128 = max(1, // 128) on the decode pool."""
        pool = _make_decode_pool(max_context_len=64)
        self.assertEqual(pool.req_to_token_c4.shape[1], 16)
        self.assertEqual(pool.req_to_token_c128.shape[1], 1)

    def test_dsv4_allocator_starts_none(self):
        pool = _make_decode_pool()
        self.assertIsNone(pool._dsv4_allocator)

    def test_aux_tables_zero_initialized(self):
        pool = _make_decode_pool()
        for name in _AUX_TABLES:
            self.assertTrue(bool((getattr(pool, name) == 0).all()))

    def test_base_decode_attributes_set(self):
        """Decode base sets size/_alloc_size/pre_alloc_size/req_to_token/etc."""
        pool = _make_decode_pool(size=4, max_context_len=64, pre_alloc_size=2)
        self.assertEqual(pool.size, 4)
        self.assertEqual(pool._alloc_size, 7)
        self.assertEqual(pool.pre_alloc_size, 2)
        self.assertEqual(pool.max_context_len, 64)
        self.assertEqual(pool.device, "cpu")
        self.assertEqual(pool.req_to_token.shape, (7, 64))
        self.assertEqual(pool.req_to_token.dtype, torch.int32)
        self.assertEqual(pool.req_generation.shape, (7,))
        self.assertEqual(pool.req_generation.dtype, torch.int64)
        self.assertEqual(pool.free_slots, [1, 2, 3, 4, 5, 6])

    def test_tables_on_specified_device(self):
        pool = _make_decode_pool(device="cpu")
        for name in _AUX_TABLES:
            self.assertEqual(getattr(pool, name).device.type, "cpu")


class TestDecodeWriteHelpers(CustomTestCase):
    """Confirms the mixin write_* helpers are inherited via the decode MRO."""

    def test_write_swa(self):
        pool = _make_decode_pool(max_context_len=8)
        values = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int32)
        pool.write_swa(0, values)
        self.assertTrue(torch.equal(pool.req_to_token_swa[0], values))


class TestDecodeFree(CustomTestCase):
    """Tests for DSV4NPUDecodeReqToTokenPool.free (mixin _dsv4_free + base)."""

    def test_free_without_allocator_only_super(self):
        """No allocator -> _dsv4_free no-ops, decode base free still runs."""
        pool = _make_decode_pool(size=4, pre_alloc_size=2)
        self.assertIsNone(pool._dsv4_allocator)
        req = _make_req(pool_idx=3)
        before = pool.available_size()
        pool.free(req)
        self.assertIsNone(req.req_pool_idx)
        self.assertEqual(pool.available_size(), before + 1)
        self.assertIn(3, pool.free_slots)

    def test_free_with_allocator_delegates(self):
        """With allocator wired, free(req) delegates req + pool ref to it."""
        pool = _make_decode_pool(size=4, pre_alloc_size=2)
        alloc = MagicMock()
        pool.register_dsv4_allocator(alloc)
        req = _make_req(pool_idx=3)
        pool.free(req)
        alloc.free.assert_called_once_with(req=req, req_to_token_pool=pool)

    def test_free_runs_allocator_before_super(self):
        """allocator.free runs while req_pool_idx is still set (before base free)."""
        pool = _make_decode_pool(size=4, pre_alloc_size=2)
        alloc = MagicMock()
        pool.register_dsv4_allocator(alloc)
        seen = {}

        def snap(req, req_to_token_pool):
            seen["idx"] = req.req_pool_idx

        alloc.free.side_effect = snap
        req = _make_req(pool_idx=5)
        pool.free(req)
        self.assertEqual(seen["idx"], 5)
        self.assertIsNone(req.req_pool_idx)


class TestDecodeClear(CustomTestCase):
    """clear() on the decode pool: base resets, aux tables NOT zeroed."""

    def test_clear_resets_free_slots(self):
        pool = _make_decode_pool(size=4, max_context_len=8, pre_alloc_size=2)
        pool.free_slots.pop()
        pool.clear()
        self.assertEqual(pool.free_slots, [1, 2, 3, 4, 5, 6])

    def test_clear_does_not_zero_aux_tables(self):
        """Documented invariant holds for the decode pool too."""
        pool = _make_decode_pool(size=4, max_context_len=8, pre_alloc_size=2)
        pool.req_to_token_swa[0] = torch.tensor(
            [1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int32
        )
        pool.req_to_token_c4[0] = torch.tensor([9, 10], dtype=torch.int32)
        pool.clear()
        self.assertTrue(
            torch.equal(
                pool.req_to_token_swa[0],
                torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                pool.req_to_token_c4[0], torch.tensor([9, 10], dtype=torch.int32)
            )
        )


if __name__ == "__main__":
    unittest.main()
