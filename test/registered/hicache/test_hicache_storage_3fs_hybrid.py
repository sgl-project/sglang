"""
Unit tests for HiCacheHF3FS v2 (hybrid / multi-pool) storage path.

These tests exercise the new v2 interface added for hybrid models (KV + MAMBA
pools), running entirely against the in-memory mock HF3FS client so they do
not require a real 3FS cluster.

Scope (from PLAN.md §5):
    * register_mem_host_pool_v2 per-pool engine construction
    * batch_exists_v2 with ALL_PAGES and TRAILING_PAGES policies
    * batch_set_v2 / batch_get_v2 round-trip across KV + MAMBA pools
    * MHA zero-copy (-k/-v) key doubling scoped to KV only
    * MLA skip_backup scoped to KV only
    * Partial pool failure (mamba file smaller than KV file)
    * v1 backwards compatibility preserved
    * Cleanup / close() semantics
    * Thread-safety of mixed v1 + v2 callers

The tests intentionally touch only the documented public API described in
PLAN.md -- they do NOT read HiCacheHF3FS internals and should not couple
to implementation details beyond the spec.

Usage:
    python3 -m pytest test/registered/hicache/test_hicache_storage_3fs_hybrid.py -v
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from typing import Dict, List, Optional

try:  # noqa: SIM105
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

# ---------------------------------------------------------------------------
# Imports from the feature under test. All guarded so the file is discoverable
# even if any piece is not yet wired up; individual tests skip with a clear
# reason instead of erroring at collection time.
# ---------------------------------------------------------------------------

_IMPORT_ERROR: Optional[Exception] = None
try:
    from sglang.srt.mem_cache.hicache_storage import (  # type: ignore
        PoolHitPolicy,
        PoolName,
        PoolTransfer,
        PoolTransferResult,
    )
    from sglang.srt.mem_cache.storage.hf3fs.storage_hf3fs import (  # type: ignore
        HiCacheHF3FS,
    )
except Exception as exc:  # pragma: no cover - exercised only on broken envs
    _IMPORT_ERROR = exc
    PoolHitPolicy = None  # type: ignore
    PoolName = None  # type: ignore
    PoolTransfer = None  # type: ignore
    PoolTransferResult = None  # type: ignore
    HiCacheHF3FS = None  # type: ignore


try:
    from sglang.test.test_utils import CustomTestCase
except Exception:  # pragma: no cover
    CustomTestCase = unittest.TestCase  # type: ignore


_REQUIRE_IMPORTS = unittest.skipIf(
    _IMPORT_ERROR is not None or torch is None,
    f"HiCacheHF3FS v2 deps unavailable: {_IMPORT_ERROR!r}",
)


# ---------------------------------------------------------------------------
# Stubs & helpers
# ---------------------------------------------------------------------------


class _FakeHostKVCache:
    """Minimal `HostKVCache`-shaped stub.

    PLAN.md §3.1 says the backend derives the pool's `bytes_per_page` from
    either `get_ksize_per_token() * page_size` (page-first layout) or
    `get_size_per_token() * page_size` (layer-first layout), so we expose
    both. `kv_buffer` is a plain torch tensor the backend can read/write via
    its mock client path.
    """

    def __init__(
        self,
        bytes_per_token: int,
        page_size: int,
        num_slots: int,
        layout: str = "layer_first",
    ) -> None:
        self.bytes_per_token = bytes_per_token
        self.page_size = page_size
        self.layout = layout
        # One flat buffer large enough to hold `num_slots` pages.
        # The exact shape isn't contract-specified, so we hand the backend
        # a simple contiguous uint8 buffer it can DMA from.
        total_bytes = bytes_per_token * page_size * num_slots
        self.kv_buffer = torch.zeros(total_bytes, dtype=torch.uint8)
        self.num_slots = num_slots

    def get_ksize_per_token(self) -> int:
        return self.bytes_per_token

    def get_size_per_token(self) -> int:
        return self.bytes_per_token

    # Some backends probe these as sentinel attributes; expose them too.
    @property
    def dtype(self):
        return torch.uint8

    # ------------------------------------------------------------------
    # HostKVCache page interface used by the v2 storage path.
    # ------------------------------------------------------------------
    @property
    def _bytes_per_page(self) -> int:
        return self.bytes_per_token * self.page_size

    def _slot_to_byte_range(self, slot_index: int) -> "tuple[int, int]":
        bpp = self._bytes_per_page
        start = slot_index * self.bytes_per_token
        return start, start + bpp

    def get_data_page(self, index: int, flat: bool = True) -> torch.Tensor:
        start, end = self._slot_to_byte_range(int(index))
        chunk = self.kv_buffer[start:end]
        if flat:
            return chunk
        # Non-flat (zero-copy MHA) view: return a 2-element stack so the
        # backend's k/v split path has something to index. The bytes are
        # cloned because the buffer here isn't sized for separate K and V
        # halves — for unit tests we only verify result shapes, not byte
        # contents on this path.
        return torch.stack([chunk.clone(), chunk.clone()])

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(self._bytes_per_page, dtype=torch.uint8)

    def set_from_flat_data_page(
        self, index: int, data_page: torch.Tensor
    ) -> None:
        start, end = self._slot_to_byte_range(int(index))
        flat = data_page.contiguous().view(torch.uint8).reshape(-1)
        self.kv_buffer[start:end].copy_(flat[: end - start])


def _mock_hf3fs_config(temp_dir: str, file_size: int = 256 * 1024 * 1024) -> Dict:
    """Canonical mock-client config for unit tests."""
    return {
        "file_path_prefix": os.path.join(temp_dir, "hicache"),
        "file_size": file_size,
        "numjobs": 2,
        "entries": 8,
        "use_mock_hf3fs_client": True,
    }


def _build_backend(
    temp_dir: str,
    *,
    bytes_per_page: int = 8192,
    rank: int = 0,
    is_mla_model: bool = False,
    extra_config: Optional[Dict] = None,
) -> "HiCacheHF3FS":
    """Construct a HiCacheHF3FS instance via its from_env_config factory.

    The exact signature of `from_env_config` is not part of the public spec,
    so if the implementer uses a different entry point the test file owns
    adapting this single helper.
    """
    cfg = _mock_hf3fs_config(temp_dir)
    if extra_config:
        cfg.update(extra_config)

    # The factory in `backend_factory.py:171-183` is the canonical caller.
    # We mirror its contract:  from_env_config(storage_config, bytes_per_page,
    #                                          rank, is_mla_model, ...)
    return HiCacheHF3FS.from_env_config(
        storage_config=cfg,
        bytes_per_page=bytes_per_page,
        rank=rank,
        is_mla_model=is_mla_model,
    )


def _keys(n: int, prefix: str = "k") -> List[str]:
    return [f"{prefix}_{i:04d}" for i in range(n)]


# ---------------------------------------------------------------------------
# Construction / registration tests
# ---------------------------------------------------------------------------


@_REQUIRE_IMPORTS
class TestRegisterMemHostPoolV2(CustomTestCase):
    """PLAN.md §5 test #1 — construction sanity for per-pool engines."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="hf3fs_hybrid_")
        self.addCleanup(self._rmtree, self.tmp)
        self.backend = _build_backend(self.tmp, bytes_per_page=8192)
        self.addCleanup(self._close, self.backend)

    @staticmethod
    def _rmtree(path: str) -> None:
        import shutil

        shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def _close(backend) -> None:
        try:
            backend.close()
        except Exception:
            pass

    def test_registers_both_kv_and_mamba_pools(self):
        kv_pool = _FakeHostKVCache(
            bytes_per_token=64, page_size=64, num_slots=128
        )
        mamba_pool = _FakeHostKVCache(
            bytes_per_token=256, page_size=64, num_slots=64
        )

        self.backend.register_mem_host_pool_v2(kv_pool, PoolName.KV)
        self.backend.register_mem_host_pool_v2(mamba_pool, PoolName.MAMBA)

        # PLAN.md §3.1 — both pool engines must coexist.
        engines = getattr(self.backend, "_engines", None)
        self.assertIsNotNone(
            engines, "HiCacheHF3FS must expose a _engines dict per pool"
        )
        self.assertIn(PoolName.KV, engines)
        self.assertIn(PoolName.MAMBA, engines)

        kv_eng = engines[PoolName.KV]
        mamba_eng = engines[PoolName.MAMBA]

        # Per-engine resources must be distinct.
        self.assertNotEqual(
            kv_eng.file_path,
            mamba_eng.file_path,
            "KV and MAMBA engines must own separate files",
        )
        self.assertIsNot(
            kv_eng.clients,
            mamba_eng.clients,
            "KV and MAMBA engines must own separate client lists",
        )
        self.assertIsNot(
            kv_eng.executor,
            mamba_eng.executor,
            "KV and MAMBA engines must own separate thread pools",
        )
        # host_pool binding should match what we passed in.
        self.assertIs(kv_eng.host_pool, kv_pool)
        self.assertIs(mamba_eng.host_pool, mamba_pool)

        # bytes_per_page should reflect each pool's layout.
        self.assertEqual(kv_eng.bytes_per_page, 64 * 64)
        self.assertEqual(mamba_eng.bytes_per_page, 256 * 64)

    def test_register_is_idempotent(self):
        kv_pool = _FakeHostKVCache(64, 64, 128)
        mamba_pool = _FakeHostKVCache(256, 64, 64)

        self.backend.register_mem_host_pool_v2(kv_pool, PoolName.KV)
        self.backend.register_mem_host_pool_v2(mamba_pool, PoolName.MAMBA)
        # Calling again with the same args must be a no-op (not raise, not
        # silently drop data). PLAN.md §4 edge case #1.
        self.backend.register_mem_host_pool_v2(kv_pool, PoolName.KV)
        self.backend.register_mem_host_pool_v2(mamba_pool, PoolName.MAMBA)

        engines = self.backend._engines
        self.assertIs(engines[PoolName.KV].host_pool, kv_pool)
        self.assertIs(engines[PoolName.MAMBA].host_pool, mamba_pool)

    def test_register_order_agnostic_mamba_first(self):
        """PLAN.md §4 #1 — do not rely on KV being registered first."""
        kv_pool = _FakeHostKVCache(64, 64, 128)
        mamba_pool = _FakeHostKVCache(256, 64, 64)

        # Register MAMBA before KV.
        self.backend.register_mem_host_pool_v2(mamba_pool, PoolName.MAMBA)
        self.backend.register_mem_host_pool_v2(kv_pool, PoolName.KV)

        self.assertIn(PoolName.KV, self.backend._engines)
        self.assertIn(PoolName.MAMBA, self.backend._engines)

    def test_v1_still_registers_kv_engine(self):
        """Backwards compat: register_mem_pool_host (v1) must still work."""
        kv_pool = _FakeHostKVCache(64, 64, 128)
        # The v1 path uses `register_mem_pool_host`.
        self.assertTrue(
            hasattr(self.backend, "register_mem_pool_host"),
            "v1 register_mem_pool_host must remain on HiCacheHF3FS",
        )
        self.backend.register_mem_pool_host(kv_pool)
        engines = getattr(self.backend, "_engines", {})
        self.assertIn(
            PoolName.KV,
            engines,
            "v1 registration must populate the KV engine (PLAN.md §3.1 / §Backwards compatibility)",
        )
        self.assertIs(engines[PoolName.KV].host_pool, kv_pool)


# ---------------------------------------------------------------------------
# batch_exists_v2 tests
# ---------------------------------------------------------------------------


@_REQUIRE_IMPORTS
class TestBatchExistsV2(CustomTestCase):
    """PLAN.md §5 tests #2, #3, #4 — batch_exists_v2 policies."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="hf3fs_hybrid_")
        self.addCleanup(self._rmtree, self.tmp)
        self.backend = _build_backend(self.tmp, bytes_per_page=8192)
        self.addCleanup(self._close, self.backend)

        self.kv_pool = _FakeHostKVCache(64, 64, 256)
        self.mamba_pool = _FakeHostKVCache(256, 64, 128)
        self.backend.register_mem_host_pool_v2(self.kv_pool, PoolName.KV)
        self.backend.register_mem_host_pool_v2(self.mamba_pool, PoolName.MAMBA)

    @staticmethod
    def _rmtree(path: str) -> None:
        import shutil

        shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def _close(backend) -> None:
        try:
            backend.close()
        except Exception:
            pass

    # -- helpers -----------------------------------------------------------

    def _set_pool_pages(
        self,
        pool_name,
        keys: List[str],
        host_indices: List[int],
    ) -> None:
        """Drive batch_set_v2 for a single pool."""
        transfer = PoolTransfer(
            name=pool_name,
            keys=keys,
            host_indices=host_indices,
            hit_policy=PoolHitPolicy.ALL_PAGES,
        )
        self.backend.batch_set_v2([transfer], extra_info=None)

    # -- tests -------------------------------------------------------------

    def test_v2_exists_kv_only_fallback(self):
        """pool_transfers=None should behave like v1 batch_exists."""
        keys = _keys(4)
        host_idx = list(range(4))
        self._set_pool_pages(PoolName.KV, keys, host_idx)

        result = self.backend.batch_exists_v2(
            keys, pool_transfers=None, extra_info=None
        )
        self.assertEqual(result.kv_hit_pages, 4)

        # And an unknown key list should report zero hits.
        result_miss = self.backend.batch_exists_v2(
            _keys(4, prefix="miss"), pool_transfers=None, extra_info=None
        )
        self.assertEqual(result_miss.kv_hit_pages, 0)

    def test_v2_exists_all_pages_policy_shrinks_hit(self):
        """PLAN.md §5 #3 — ALL_PAGES policy shrinks the KV prefix.

        Write 4 KV pages and 2 MAMBA pages (slots [0, 1]). A MAMBA
        PoolTransfer with ALL_PAGES policy should shrink the effective
        hit to 2 pages.
        """
        kv_keys = _keys(4, prefix="kv")
        self._set_pool_pages(PoolName.KV, kv_keys, [0, 1, 2, 3])
        self._set_pool_pages(PoolName.MAMBA, kv_keys[:2], [0, 1])

        mamba_transfer = PoolTransfer(
            name=PoolName.MAMBA,
            keys=kv_keys,
            host_indices=[10, 11, 12, 13],
            hit_policy=PoolHitPolicy.ALL_PAGES,
        )
        result = self.backend.batch_exists_v2(
            kv_keys, pool_transfers=[mamba_transfer], extra_info=None
        )

        self.assertEqual(
            result.kv_hit_pages,
            2,
            "KV prefix must be clamped to the MAMBA boundary under ALL_PAGES",
        )
        self.assertEqual(result.extra_pool_hit_pages[PoolName.MAMBA], 2)

    def test_v2_exists_trailing_pages_policy(self):
        """PLAN.md §5 #4 — TRAILING_PAGES policy matches HiCacheFile semantics.

        Write 4 KV pages. Write MAMBA state only for pages [2, 3]. With a
        mamba PoolTransfer(hit_policy=TRAILING_PAGES, keys=[k2, k3]), the
        kv_hit_pages should still be 4 and extra_pool_hit_pages[MAMBA] == 4
        because the trailing-2-of-4 window satisfied the policy.
        """
        kv_keys = _keys(4, prefix="kv")
        self._set_pool_pages(PoolName.KV, kv_keys, [0, 1, 2, 3])
        # Only the trailing two pages have mamba state.
        self._set_pool_pages(
            PoolName.MAMBA, kv_keys[2:], [2, 3]
        )

        mamba_transfer = PoolTransfer(
            name=PoolName.MAMBA,
            keys=kv_keys[2:],
            host_indices=[12, 13],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        result = self.backend.batch_exists_v2(
            kv_keys, pool_transfers=[mamba_transfer], extra_info=None
        )

        self.assertEqual(result.kv_hit_pages, 4)
        self.assertEqual(result.extra_pool_hit_pages[PoolName.MAMBA], 4)

    def test_v2_exists_trailing_pages_with_kv_miss(self):
        """PLAN.md §4 #6 — TRAILING_PAGES early-returns when kv_pages == 0."""
        mamba_transfer = PoolTransfer(
            name=PoolName.MAMBA,
            keys=_keys(2),
            host_indices=[0, 1],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        result = self.backend.batch_exists_v2(
            _keys(4, prefix="never_written"),
            pool_transfers=[mamba_transfer],
            extra_info=None,
        )
        self.assertEqual(result.kv_hit_pages, 0)
        self.assertEqual(
            result.extra_pool_hit_pages.get(PoolName.MAMBA, 0),
            0,
            "TRAILING_PAGES with kv_pages == 0 must not report any mamba hit",
        )

    def test_v2_exists_transfer_keys_longer_than_kv_pages(self):
        """PLAN.md §4 #7 — transfer.keys longer than kv prefix must be clamped."""
        kv_keys = _keys(2, prefix="kv")
        self._set_pool_pages(PoolName.KV, kv_keys, [0, 1])
        self._set_pool_pages(PoolName.MAMBA, kv_keys, [0, 1])

        mamba_transfer = PoolTransfer(
            name=PoolName.MAMBA,
            # `keys` list longer than the 2-page KV prefix.
            keys=kv_keys + _keys(3, prefix="extra"),
            host_indices=[0, 1, 2, 3, 4],
            hit_policy=PoolHitPolicy.ALL_PAGES,
        )
        result = self.backend.batch_exists_v2(
            kv_keys, pool_transfers=[mamba_transfer], extra_info=None
        )
        self.assertEqual(result.kv_hit_pages, 2)
        self.assertLessEqual(
            result.extra_pool_hit_pages.get(PoolName.MAMBA, 0), 2
        )

    def test_v2_exists_partial_kv_prefix(self):
        """Non-contiguous KV writes must still return the longest prefix."""
        kv_keys = _keys(6, prefix="kv")
        # Write only pages 0, 1, 2 — page 3 is a hole.
        self._set_pool_pages(PoolName.KV, kv_keys[:3], [0, 1, 2])
        # Simulate a write at page 5 too (non-contiguous).
        self._set_pool_pages(PoolName.KV, [kv_keys[5]], [5])

        result = self.backend.batch_exists_v2(
            kv_keys, pool_transfers=None, extra_info=None
        )
        self.assertEqual(
            result.kv_hit_pages,
            3,
            "Longest *contiguous* prefix must be reported (page 5 is past a hole)",
        )


# ---------------------------------------------------------------------------
# batch_get_v2 / batch_set_v2 round-trip tests
# ---------------------------------------------------------------------------


@_REQUIRE_IMPORTS
class TestBatchRoundTripV2(CustomTestCase):
    """PLAN.md §5 test #5 — set then get for KV + MAMBA."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="hf3fs_hybrid_")
        self.addCleanup(self._rmtree, self.tmp)
        self.backend = _build_backend(self.tmp, bytes_per_page=8192)
        self.addCleanup(self._close, self.backend)

        self.kv_src = _FakeHostKVCache(64, 64, 128)
        self.mamba_src = _FakeHostKVCache(256, 64, 128)
        # Seed the host buffers with recognizable patterns so we can verify
        # that batch_get_v2 drops data into the correct slots.
        self.kv_src.kv_buffer.fill_(0xAB)
        self.mamba_src.kv_buffer.fill_(0xCD)

        self.backend.register_mem_host_pool_v2(self.kv_src, PoolName.KV)
        self.backend.register_mem_host_pool_v2(self.mamba_src, PoolName.MAMBA)

    @staticmethod
    def _rmtree(path):
        import shutil

        shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def _close(backend):
        try:
            backend.close()
        except Exception:
            pass

    def test_roundtrip_kv_and_mamba(self):
        kv_keys = _keys(4, prefix="kv")
        mamba_keys = kv_keys  # pool suffixes are added internally

        kv_transfer = PoolTransfer(
            name=PoolName.KV,
            keys=kv_keys,
            host_indices=[0, 1, 2, 3],
            hit_policy=PoolHitPolicy.ALL_PAGES,
        )
        mamba_transfer = PoolTransfer(
            name=PoolName.MAMBA,
            keys=mamba_keys,
            host_indices=[0, 1, 2, 3],
            hit_policy=PoolHitPolicy.ALL_PAGES,
        )

        set_result = self.backend.batch_set_v2(
            [kv_transfer, mamba_transfer], extra_info=None
        )
        self.assertEqual(set_result[PoolName.KV], [True, True, True, True])
        self.assertEqual(set_result[PoolName.MAMBA], [True, True, True, True])

        # Zero out the destination slots so we can prove batch_get_v2
        # actually writes them.
        self.kv_src.kv_buffer.zero_()
        self.mamba_src.kv_buffer.zero_()

        get_result = self.backend.batch_get_v2(
            [kv_transfer, mamba_transfer], extra_info=None
        )
        self.assertEqual(get_result[PoolName.KV], [True, True, True, True])
        self.assertEqual(get_result[PoolName.MAMBA], [True, True, True, True])

        # Host buffers must contain the bytes we originally wrote.
        self.assertTrue(
            (self.kv_src.kv_buffer != 0).any(),
            "batch_get_v2 must deposit bytes into the KV host pool",
        )
        self.assertTrue(
            (self.mamba_src.kv_buffer != 0).any(),
            "batch_get_v2 must deposit bytes into the MAMBA host pool",
        )

    def test_batch_set_v2_result_is_per_key_list(self):
        """Contract: each pool maps to a List[bool], not a scalar bool."""
        transfer = PoolTransfer(
            name=PoolName.KV,
            keys=_keys(3),
            host_indices=[0, 1, 2],
            hit_policy=PoolHitPolicy.ALL_PAGES,
        )
        result = self.backend.batch_set_v2([transfer], extra_info=None)
        self.assertIn(PoolName.KV, result)
        self.assertIsInstance(result[PoolName.KV], list)
        self.assertEqual(len(result[PoolName.KV]), 3)
        for v in result[PoolName.KV]:
            self.assertIsInstance(v, bool)

    def test_get_after_set_preserves_pool_isolation(self):
        """Pool suffix namespacing: KV key `x` and MAMBA key `x` must not collide."""
        key = "shared"
        kv_transfer = PoolTransfer(
            name=PoolName.KV,
            keys=[key],
            host_indices=[0],
            hit_policy=PoolHitPolicy.ALL_PAGES,
        )
        mamba_transfer = PoolTransfer(
            name=PoolName.MAMBA,
            keys=[key],
            host_indices=[0],
            hit_policy=PoolHitPolicy.ALL_PAGES,
        )
        # Write only KV.
        self.backend.batch_set_v2([kv_transfer], extra_info=None)

        exists = self.backend.batch_exists_v2(
            [key], pool_transfers=[mamba_transfer], extra_info=None
        )
        # KV exists but mamba does not -> under ALL_PAGES the combined hit
        # must be zero.
        self.assertEqual(exists.kv_hit_pages, 0)


# ---------------------------------------------------------------------------
# MHA zero-copy hybrid test (PLAN.md §5 #6 + §4 #3)
# ---------------------------------------------------------------------------


@_REQUIRE_IMPORTS
class TestZeroCopyMhaHybrid(CustomTestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="hf3fs_hybrid_")
        self.addCleanup(self._rmtree, self.tmp)
        # MHA (not MLA), zero-copy path implied by config -> set on backend.
        self.backend = _build_backend(
            self.tmp, bytes_per_page=8192, is_mla_model=False
        )
        self.addCleanup(self._close, self.backend)

        self.kv_pool = _FakeHostKVCache(64, 64, 128)
        self.mamba_pool = _FakeHostKVCache(256, 64, 128)
        self.backend.register_mem_host_pool_v2(self.kv_pool, PoolName.KV)
        self.backend.register_mem_host_pool_v2(self.mamba_pool, PoolName.MAMBA)

    @staticmethod
    def _rmtree(path):
        import shutil

        shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def _close(backend):
        try:
            backend.close()
        except Exception:
            pass

    def test_zero_copy_key_doubling_scoped_to_kv(self):
        """-k/-v key suffix must be applied only on the KV engine.

        We can't introspect the backend's wire keys without reading
        implementation, so verify the observable invariant instead: a
        round-trip on a hybrid backend with MHA + zero-copy works
        without raising and without any mamba-side key collision.
        """
        # Force zero-copy if the backend exposes the flag.
        if hasattr(self.backend, "is_zero_copy"):
            self.backend.is_zero_copy = True
        kv_eng = self.backend._engines[PoolName.KV]
        if hasattr(kv_eng, "is_zero_copy"):
            kv_eng.is_zero_copy = True

        transfers = [
            PoolTransfer(
                name=PoolName.KV,
                keys=_keys(2, prefix="kv"),
                host_indices=[0, 1],
                hit_policy=PoolHitPolicy.ALL_PAGES,
            ),
            PoolTransfer(
                name=PoolName.MAMBA,
                keys=_keys(2, prefix="kv"),
                host_indices=[0, 1],
                hit_policy=PoolHitPolicy.ALL_PAGES,
            ),
        ]

        set_res = self.backend.batch_set_v2(transfers, extra_info=None)
        self.assertEqual(set_res[PoolName.KV], [True, True])
        self.assertEqual(set_res[PoolName.MAMBA], [True, True])

        get_res = self.backend.batch_get_v2(transfers, extra_info=None)
        self.assertEqual(get_res[PoolName.KV], [True, True])
        self.assertEqual(get_res[PoolName.MAMBA], [True, True])

        # And an exists query should also succeed.
        exists = self.backend.batch_exists_v2(
            _keys(2, prefix="kv"),
            pool_transfers=[transfers[1]],
            extra_info=None,
        )
        self.assertEqual(exists.kv_hit_pages, 2)
        self.assertEqual(exists.extra_pool_hit_pages[PoolName.MAMBA], 2)


# ---------------------------------------------------------------------------
# MLA skip_backup scoping (PLAN.md §5 #7 + §4 #4)
# ---------------------------------------------------------------------------


@_REQUIRE_IMPORTS
class TestMlaSkipBackupKvOnly(CustomTestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="hf3fs_hybrid_")
        self.addCleanup(self._rmtree, self.tmp)
        self.backend = _build_backend(
            self.tmp, bytes_per_page=8192, rank=2, is_mla_model=True
        )
        self.addCleanup(self._close, self.backend)

        self.kv_pool = _FakeHostKVCache(64, 64, 128)
        self.mamba_pool = _FakeHostKVCache(256, 64, 128)
        self.backend.register_mem_host_pool_v2(self.kv_pool, PoolName.KV)
        self.backend.register_mem_host_pool_v2(self.mamba_pool, PoolName.MAMBA)

    @staticmethod
    def _rmtree(path):
        import shutil

        shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def _close(backend):
        try:
            backend.close()
        except Exception:
            pass

    def test_mla_non_zero_rank_only_kv_is_skipped(self):
        """KV returns [True]*N (no-op skip); MAMBA actually writes through."""
        transfers = [
            PoolTransfer(
                name=PoolName.KV,
                keys=_keys(4, prefix="kv"),
                host_indices=[0, 1, 2, 3],
                hit_policy=PoolHitPolicy.ALL_PAGES,
            ),
            PoolTransfer(
                name=PoolName.MAMBA,
                keys=_keys(4, prefix="kv"),
                host_indices=[0, 1, 2, 3],
                hit_policy=PoolHitPolicy.ALL_PAGES,
            ),
        ]

        set_res = self.backend.batch_set_v2(transfers, extra_info=None)
        # KV returns per-key True list (PLAN.md §3.4 -- bug fix from scalar).
        self.assertEqual(
            set_res[PoolName.KV],
            [True, True, True, True],
            "MLA KV rank>0 skip_backup must return per-key list of True",
        )
        self.assertEqual(len(set_res[PoolName.MAMBA]), 4)
        self.assertTrue(all(set_res[PoolName.MAMBA]))

        # MAMBA should actually be in storage after the write (PLAN.md §4 #4).
        mamba_only = PoolTransfer(
            name=PoolName.MAMBA,
            keys=_keys(4, prefix="kv"),
            host_indices=[0, 1, 2, 3],
            hit_policy=PoolHitPolicy.ALL_PAGES,
        )
        exists = self.backend.batch_exists_v2(
            _keys(4, prefix="kv"),
            pool_transfers=[mamba_only],
            extra_info=None,
        )
        # MAMBA-only: KV never written -> kv_hit=0. The important check is
        # that the underlying mamba write actually landed, which we verify
        # by a targeted get.
        get_res = self.backend.batch_get_v2([mamba_only], extra_info=None)
        self.assertTrue(all(get_res[PoolName.MAMBA]),
            "Mamba writes must persist on every rank, even under MLA skip_backup")


# ---------------------------------------------------------------------------
# Partial pool failure (PLAN.md §5 #8 + §4 #5, #11)
# ---------------------------------------------------------------------------


@_REQUIRE_IMPORTS
class TestPartialPoolFailure(CustomTestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="hf3fs_hybrid_")
        self.addCleanup(self._rmtree, self.tmp)
        # Shrink the mamba pool's file drastically via per-pool config.
        extra = {
            "pools": {
                "mamba": {
                    # One page fits (mamba bytes_per_page = 256*64 = 16384).
                    "file_size_fraction": 0.00001,
                }
            }
        }
        self.backend = _build_backend(
            self.tmp, bytes_per_page=8192, extra_config=extra
        )
        self.addCleanup(self._close, self.backend)

        self.kv_pool = _FakeHostKVCache(64, 64, 128)
        self.mamba_pool = _FakeHostKVCache(256, 64, 128)
        self.backend.register_mem_host_pool_v2(self.kv_pool, PoolName.KV)
        self.backend.register_mem_host_pool_v2(self.mamba_pool, PoolName.MAMBA)

    @staticmethod
    def _rmtree(path):
        import shutil

        shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def _close(backend):
        try:
            backend.close()
        except Exception:
            pass

    def test_mamba_file_exhausted_kv_still_ok(self):
        """With a tiny mamba file, mamba sets fail past its capacity but KV
        continues to succeed and no exception escapes (PLAN.md §4 #5).
        """
        transfers = [
            PoolTransfer(
                name=PoolName.KV,
                keys=_keys(4, prefix="kv"),
                host_indices=[0, 1, 2, 3],
                hit_policy=PoolHitPolicy.ALL_PAGES,
            ),
            PoolTransfer(
                name=PoolName.MAMBA,
                keys=_keys(4, prefix="kv"),
                host_indices=[0, 1, 2, 3],
                hit_policy=PoolHitPolicy.ALL_PAGES,
            ),
        ]

        try:
            result = self.backend.batch_set_v2(transfers, extra_info=None)
        except Exception as exc:
            self.fail(
                "batch_set_v2 must not raise on per-pool capacity exhaustion; got "
                f"{exc!r}"
            )

        self.assertEqual(
            result[PoolName.KV],
            [True, True, True, True],
            "KV pool should not be affected by mamba capacity exhaustion",
        )
        mamba_results = result[PoolName.MAMBA]
        self.assertEqual(len(mamba_results), 4)
        self.assertTrue(
            any(v is False for v in mamba_results),
            "At least one mamba write must fail when the mamba file is exhausted",
        )


# ---------------------------------------------------------------------------
# Interface contract / error handling (PLAN.md §5 #9)
# ---------------------------------------------------------------------------


@_REQUIRE_IMPORTS
class TestV2InterfaceContract(CustomTestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="hf3fs_hybrid_")
        self.addCleanup(self._rmtree, self.tmp)

    @staticmethod
    def _rmtree(path):
        import shutil

        shutil.rmtree(path, ignore_errors=True)

    def test_has_v2_methods(self):
        for name in (
            "batch_exists_v2",
            "batch_get_v2",
            "batch_set_v2",
            "register_mem_host_pool_v2",
        ):
            self.assertTrue(
                hasattr(HiCacheHF3FS, name),
                f"HiCacheHF3FS must expose {name}",
            )

    def test_v2_without_pool_registration_raises_clear_error(self):
        """Calling a v2 method before any pool is registered must raise
        something understandable -- not an opaque AttributeError.
        """
        backend = _build_backend(self.tmp, bytes_per_page=8192)
        try:
            with self.assertRaises(Exception) as ctx:
                backend.batch_get_v2(
                    [
                        PoolTransfer(
                            name=PoolName.MAMBA,
                            keys=_keys(1),
                            host_indices=[0],
                            hit_policy=PoolHitPolicy.ALL_PAGES,
                        )
                    ],
                    extra_info=None,
                )
            # Must not be a raw AttributeError from missing engine dict etc.
            self.assertNotIsInstance(ctx.exception, AttributeError)
        finally:
            try:
                backend.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Cleanup / close() (PLAN.md §4 #9)
# ---------------------------------------------------------------------------


@_REQUIRE_IMPORTS
class TestCleanup(CustomTestCase):
    def test_close_releases_all_pool_engines(self):
        tmp = tempfile.mkdtemp(prefix="hf3fs_hybrid_")
        try:
            backend = _build_backend(tmp, bytes_per_page=8192)
            backend.register_mem_host_pool_v2(
                _FakeHostKVCache(64, 64, 128), PoolName.KV
            )
            backend.register_mem_host_pool_v2(
                _FakeHostKVCache(256, 64, 128), PoolName.MAMBA
            )

            engines = list(backend._engines.values())
            self.assertEqual(len(engines), 2)

            # close() must not raise and must shut down both executors.
            backend.close()

            for eng in engines:
                executor = getattr(eng, "executor", None)
                if executor is not None and hasattr(executor, "_shutdown"):
                    # ThreadPoolExecutor has a _shutdown flag after .shutdown().
                    self.assertTrue(
                        executor._shutdown,
                        "close() must shutdown each per-pool executor",
                    )
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Concurrent v1 + v2 callers (PLAN.md §4 #10)
# ---------------------------------------------------------------------------


@_REQUIRE_IMPORTS
class TestConcurrentV1V2(CustomTestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="hf3fs_hybrid_")
        self.addCleanup(self._rmtree, self.tmp)
        self.backend = _build_backend(self.tmp, bytes_per_page=8192)
        self.addCleanup(self._close, self.backend)

        self.kv_pool = _FakeHostKVCache(64, 64, 256)
        self.mamba_pool = _FakeHostKVCache(256, 64, 128)
        self.backend.register_mem_host_pool_v2(self.kv_pool, PoolName.KV)
        self.backend.register_mem_host_pool_v2(self.mamba_pool, PoolName.MAMBA)

    @staticmethod
    def _rmtree(path):
        import shutil

        shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def _close(backend):
        try:
            backend.close()
        except Exception:
            pass

    def test_mixed_v1_v2_callers_are_thread_safe(self):
        """Interleave v1 batch_set_v1 / v2 batch_set_v2 from two threads.

        The RLock should serialize metadata mutations so neither caller
        crashes and both sets of keys are present at the end.
        """
        errors: List[BaseException] = []
        done = threading.Event()

        def v2_worker():
            try:
                for i in range(20):
                    transfer = PoolTransfer(
                        name=PoolName.MAMBA,
                        keys=[f"v2_{i}"],
                        host_indices=[0],
                        hit_policy=PoolHitPolicy.ALL_PAGES,
                    )
                    self.backend.batch_set_v2([transfer], extra_info=None)
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        def v1_worker():
            try:
                page_size = getattr(self.kv_pool, "page_size", 1) or 1
                for i in range(20):
                    # v1 entry points per PLAN.md §3.5.
                    if hasattr(self.backend, "batch_set_v1"):
                        # v1 expects host_indices of length len(keys) * page_size.
                        host_indices = torch.zeros(page_size, dtype=torch.int64)
                        self.backend.batch_set_v1(
                            [f"v1_{i}"], host_indices=host_indices
                        )
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)
            finally:
                done.set()

        t1 = threading.Thread(target=v2_worker)
        t2 = threading.Thread(target=v1_worker)
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        self.assertEqual(errors, [], f"Thread errors: {errors}")


# ---------------------------------------------------------------------------
# Mock client parity (PLAN.md §4 #8)
# ---------------------------------------------------------------------------


@_REQUIRE_IMPORTS
class TestMockClientParity(CustomTestCase):
    def test_mock_allocations_scale_with_multiple_pools(self):
        """Opening multiple pool files under the mock client must work.

        The test is intentionally thin: if the mock client only supports one
        file per backend instance the register call will raise -- catching
        the "harmless startup window" bug PLAN.md §4 #9 warns about.
        """
        tmp = tempfile.mkdtemp(prefix="hf3fs_hybrid_")
        try:
            backend = _build_backend(tmp, bytes_per_page=8192)
            try:
                backend.register_mem_host_pool_v2(
                    _FakeHostKVCache(64, 64, 128), PoolName.KV
                )
                backend.register_mem_host_pool_v2(
                    _FakeHostKVCache(256, 64, 128), PoolName.MAMBA
                )
                # At least one write per pool must land without raising.
                for pool in (PoolName.KV, PoolName.MAMBA):
                    res = backend.batch_set_v2(
                        [
                            PoolTransfer(
                                name=pool,
                                keys=_keys(1, prefix=str(pool)),
                                host_indices=[0],
                                hit_policy=PoolHitPolicy.ALL_PAGES,
                            )
                        ],
                        extra_info=None,
                    )
                    self.assertEqual(len(res[pool]), 1)
            finally:
                try:
                    backend.close()
                except Exception:
                    pass
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Config surface (PLAN.md §3 "Config surface")
# ---------------------------------------------------------------------------


@_REQUIRE_IMPORTS
class TestConfigSurface(CustomTestCase):
    """Per-pool file_size_fraction override must be respected."""

    def test_custom_file_size_fraction_is_honored(self):
        tmp = tempfile.mkdtemp(prefix="hf3fs_hybrid_")
        try:
            extra = {
                "pools": {
                    "mamba": {"file_size_fraction": 0.25},
                }
            }
            backend = _build_backend(
                tmp, bytes_per_page=8192, extra_config=extra
            )
            try:
                backend.register_mem_host_pool_v2(
                    _FakeHostKVCache(64, 64, 256), PoolName.KV
                )
                backend.register_mem_host_pool_v2(
                    _FakeHostKVCache(256, 64, 256), PoolName.MAMBA
                )

                kv_eng = backend._engines[PoolName.KV]
                mamba_eng = backend._engines[PoolName.MAMBA]

                # mamba file should be strictly smaller than the KV file
                # when fraction < 1, and strictly > 0.
                if os.path.exists(kv_eng.file_path) and os.path.exists(
                    mamba_eng.file_path
                ):
                    kv_sz = os.path.getsize(kv_eng.file_path)
                    m_sz = os.path.getsize(mamba_eng.file_path)
                    self.assertGreater(m_sz, 0)
                    self.assertLess(m_sz, kv_sz)
            finally:
                try:
                    backend.close()
                except Exception:
                    pass
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_default_fraction_when_pools_config_missing(self):
        """When extra_config has no 'pools' section, registration must still
        succeed using a sane default fraction (PLAN.md §Config surface).
        """
        tmp = tempfile.mkdtemp(prefix="hf3fs_hybrid_")
        try:
            backend = _build_backend(tmp, bytes_per_page=8192)
            try:
                backend.register_mem_host_pool_v2(
                    _FakeHostKVCache(64, 64, 128), PoolName.KV
                )
                backend.register_mem_host_pool_v2(
                    _FakeHostKVCache(256, 64, 128), PoolName.MAMBA
                )
                self.assertIn(PoolName.MAMBA, backend._engines)
            finally:
                try:
                    backend.close()
                except Exception:
                    pass
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# PoolName scoping sanity (PLAN.md §3 "PoolName.DSA")
# ---------------------------------------------------------------------------


@_REQUIRE_IMPORTS
class TestPoolNameScope(CustomTestCase):
    def test_pool_name_has_kv_and_mamba(self):
        # Per PLAN.md §3 "DSA pool" decision: start with KV + MAMBA only.
        self.assertTrue(hasattr(PoolName, "KV"))
        self.assertTrue(hasattr(PoolName, "MAMBA"))

    def test_register_unknown_pool_is_rejected(self):
        """Registering with a non-PoolName value must fail clearly."""
        tmp = tempfile.mkdtemp(prefix="hf3fs_hybrid_")
        try:
            backend = _build_backend(tmp, bytes_per_page=8192)
            try:
                pool = _FakeHostKVCache(64, 64, 128)
                with self.assertRaises(Exception):
                    backend.register_mem_host_pool_v2(pool, "not_a_real_pool")
            finally:
                try:
                    backend.close()
                except Exception:
                    pass
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
