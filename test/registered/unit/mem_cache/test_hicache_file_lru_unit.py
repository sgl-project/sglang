"""
Unit tests for HiCacheFile LRU/eviction logic (max_size cap, free-space
watermark, MLA owner gating, pre-reservation under concurrency) and the
CP-aware file-key suffix.

The eviction logic lives in ``LRUFileEvictor`` (mem_cache/storage/file/); these
tests drive it end-to-end through ``HiCacheFile`` and inspect the wired-up
evictor via ``backend._evictor``.

These are pure CPU tests; they do not launch a server or need CUDA.
Run with:
    python3 -m pytest test/registered/unit/mem_cache/test_hicache_file_lru_unit.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import os
import shutil
import tempfile
import time
import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    MetadataCache,
)
from sglang.srt.mem_cache.storage.file.lru_file_evictor import _parse_size_to_bytes
from sglang.test.test_utils import CustomTestCase


def _t(n_bytes: int, fill: int = 0) -> torch.Tensor:
    """Build a uint8 CPU tensor of n_bytes filled with `fill`."""
    return torch.full((n_bytes,), fill, dtype=torch.uint8)


def _make_config(
    *,
    tp_rank=0,
    tp_size=1,
    pp_rank=0,
    pp_size=1,
    attn_cp_rank=0,
    attn_cp_size=1,
    is_mla=False,
    model="testmodel",
    extra_config=None,
) -> HiCacheStorageConfig:
    return HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=pp_rank,
        pp_size=pp_size,
        attn_cp_rank=attn_cp_rank,
        attn_cp_size=attn_cp_size,
        is_mla_model=is_mla,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name=model,
        extra_config=extra_config,
    )


class _BackendBuilder:
    """Build a HiCacheFile with explicit config in a fresh temp dir."""

    def __init__(self, base_tmp: str):
        self.base_tmp = base_tmp

    def __call__(
        self,
        *,
        max_size=None,
        min_free=None,
        eviction_ratio=None,
        tp_rank=0,
        tp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla=False,
        model="testmodel",
        subdir=None,
        metadata_ttl=None,
        enable_metadata_cache=None,
    ) -> HiCacheFile:
        # Each backend gets its own subdir so MLA / non-MLA tests don't
        # contaminate each other's file_path.
        d = os.path.join(
            self.base_tmp, subdir or f"r{tp_rank}_t{tp_size}_{int(time.time_ns())}"
        )
        os.makedirs(d, exist_ok=True)
        cfg = _make_config(
            tp_rank=tp_rank,
            tp_size=tp_size,
            attn_cp_rank=attn_cp_rank,
            attn_cp_size=attn_cp_size,
            is_mla=is_mla,
            model=model,
            extra_config={
                "max_size": max_size,
                "eviction_ratio": eviction_ratio,
                "min_free_space": min_free,
                "metadata_ttl": metadata_ttl,
                "enable_metadata_cache": enable_metadata_cache,
            },
        )
        return HiCacheFile(cfg, file_path=d)


class TestParseSize(CustomTestCase):
    def test_zero_and_none(self):
        self.assertEqual(_parse_size_to_bytes(None), 0)
        self.assertEqual(_parse_size_to_bytes("0"), 0)
        self.assertEqual(_parse_size_to_bytes(""), 0)
        self.assertEqual(_parse_size_to_bytes("none"), 0)

    def test_units(self):
        self.assertEqual(_parse_size_to_bytes("1024"), 1024)
        self.assertEqual(_parse_size_to_bytes("1k"), 1000)
        self.assertEqual(_parse_size_to_bytes("1Ki"), 1024)
        self.assertEqual(_parse_size_to_bytes("1Mi"), 1 << 20)
        self.assertEqual(_parse_size_to_bytes("2Gi"), 2 * (1 << 30))
        self.assertEqual(_parse_size_to_bytes("1.5G"), int(1.5 * 10**9))

    def test_invalid_returns_zero(self):
        self.assertEqual(_parse_size_to_bytes("abc"), 0)
        self.assertEqual(_parse_size_to_bytes("10XY"), 0)


class HiCacheFileLRUTestBase(CustomTestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="hicache_lru_unit_")
        self.make_backend = _BackendBuilder(self.tmpdir)
        # Neutralise env vars so user shell can't leak settings into tests.
        self._env_overrides = [
            envs.SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE.override("0"),
            envs.SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE.override("0"),
        ]
        for cm in self._env_overrides:
            cm.__enter__()

    def tearDown(self):
        for cm in self._env_overrides:
            cm.__exit__(None, None, None)
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestEnvDefaults(CustomTestCase):
    """Verify the env var defaults match the documented opt-in behavior."""

    def test_min_free_space_default_is_zero(self):
        # Default must keep eviction off so existing users are unaffected.
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE", None)
            self.assertEqual(
                envs.SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE.get(),
                "0",
            )

    def test_max_size_default_is_none(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE", None)
            self.assertIsNone(envs.SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE.get())


class TestEvictionDisabledByDefault(HiCacheFileLRUTestBase):
    def test_no_config_no_eviction(self):
        b = self.make_backend(max_size="0", min_free="0")
        self.assertFalse(b._evictor.enabled)
        # Set/get should still work as raw file storage.
        self.assertTrue(b.set("k1", _t(50)))
        self.assertTrue(b.exists("k1"))
        # No tracking happens.
        self.assertEqual(len(b._evictor._lru), 0)
        self.assertEqual(b._evictor._total_bytes, 0)


class TestCapBasedEviction(HiCacheFileLRUTestBase):
    def test_basic_lru_evicts_oldest(self):
        b = self.make_backend(max_size="300", eviction_ratio=1.0)
        self.assertTrue(b.set("a", _t(100)))
        self.assertTrue(b.set("b", _t(100)))
        self.assertTrue(b.set("c", _t(100)))
        self.assertEqual(b._evictor._total_bytes, 300)
        # Adding "d" forces eviction of "a" (oldest).
        self.assertTrue(b.set("d", _t(100)))
        self.assertLessEqual(b._evictor._total_bytes, 300)
        self.assertFalse(b.exists("a"))
        for k in ("b", "c", "d"):
            self.assertTrue(b.exists(k), f"{k} should still be present")

    def test_get_touches_recency(self):
        b = self.make_backend(max_size="300", eviction_ratio=1.0)
        b.set("a", _t(100))
        b.set("b", _t(100))
        b.set("c", _t(100))
        # Access "a" -> now "b" is the LRU.
        b.get("a", target_location=_t(100))
        # Inserting "d" should evict "b", not "a".
        b.set("d", _t(100))
        self.assertTrue(b.exists("a"), "a was just-accessed and must survive")
        self.assertFalse(b.exists("b"), "b should be the new LRU and got evicted")

    def test_value_larger_than_cap_rejected(self):
        b = self.make_backend(max_size="100")
        self.assertFalse(b.set("too_big", _t(200)))
        self.assertFalse(b.exists("too_big"))
        self.assertEqual(b._evictor._total_bytes, 0)
        self.assertEqual(len(b._evictor._lru), 0)

    def test_eviction_ratio_drops_to_watermark(self):
        # ratio=0.5 -> evict down to ~50% of the cap before adding.
        b = self.make_backend(max_size="400", eviction_ratio=0.5)
        for k in ("a", "b", "c", "d"):
            b.set(k, _t(100))
        self.assertEqual(b._evictor._total_bytes, 400)
        # target = 0.5*400 - 100 = 100, then +100 -> 200.
        b.set("e", _t(100))
        self.assertLessEqual(b._evictor._total_bytes, 200)

    def test_repeated_set_same_key_is_noop(self):
        b = self.make_backend(max_size="300")
        self.assertTrue(b.set("a", _t(100)))
        self.assertEqual(b._evictor._total_bytes, 100)
        # Same key, different value -- fast path skips rewrite.
        self.assertTrue(b.set("a", _t(100)))
        self.assertEqual(b._evictor._total_bytes, 100)
        self.assertEqual(len(b._evictor._lru), 1)

    def test_clear_resets_state(self):
        b = self.make_backend(max_size="300")
        b.set("a", _t(100))
        b.set("b", _t(100))
        self.assertEqual(b._evictor._total_bytes, 200)
        self.assertTrue(b.clear())
        self.assertEqual(b._evictor._total_bytes, 0)
        self.assertEqual(len(b._evictor._lru), 0)
        self.assertFalse(b.exists("a"))


class TestScanExistingFiles(HiCacheFileLRUTestBase):
    def test_scan_seeds_lru_in_mtime_order(self):
        # Pre-create files, then check older mtimes land at the LRU front.
        d = tempfile.mkdtemp(prefix="hicache_seed_", dir=self.tmpdir)
        cfg = _make_config(
            model="seedmodel",
            extra_config={"max_size": "1000", "min_free_space": "0"},
        )
        # Files must end with the expected suffix for the rank/model.
        suffix = f"_seedmodel_0_1"
        # Create older "old.bin" first, then newer "new.bin".
        old_path = os.path.join(d, f"old{suffix}.bin")
        new_path = os.path.join(d, f"new{suffix}.bin")
        with open(old_path, "wb") as f:
            f.write(b"x" * 50)
        # Force older mtime on old_path.
        old_t = time.time() - 100
        os.utime(old_path, (old_t, old_t))
        with open(new_path, "wb") as f:
            f.write(b"y" * 70)
        b = HiCacheFile(cfg, file_path=d)
        self.assertEqual(b._evictor._total_bytes, 50 + 70)
        # First key in _lru should be the oldest (front = LRU).
        keys = list(b._evictor._lru.keys())
        self.assertEqual(keys[0], f"old{suffix}")
        self.assertEqual(keys[1], f"new{suffix}")


class TestCPSuffix(HiCacheFileLRUTestBase):
    """Distinct CP ranks must not share a file key."""

    def test_cp_disabled_has_no_cp_suffix(self):
        b = self.make_backend(attn_cp_size=1, attn_cp_rank=0)
        self.assertNotIn("_cp", b.config_suffix)

    def test_distinct_cp_ranks_get_distinct_suffix(self):
        b0 = self.make_backend(attn_cp_rank=0, attn_cp_size=8, subdir="cp")
        b1 = self.make_backend(attn_cp_rank=1, attn_cp_size=8, subdir="cp")
        self.assertNotEqual(b0.config_suffix, b1.config_suffix)
        self.assertTrue(b0.config_suffix.endswith("_cp0_8"))
        self.assertTrue(b1.config_suffix.endswith("_cp1_8"))
        # Same logical key maps to different files per CP rank -> no write race.
        self.assertNotEqual(b0._get_suffixed_key("k"), b1._get_suffixed_key("k"))

    def test_cp_suffix_applies_to_mla(self):
        # MLA drops tp from the suffix; the CP tag keeps ranks isolated.
        b0 = self.make_backend(is_mla=True, attn_cp_rank=0, attn_cp_size=4, subdir="m")
        b1 = self.make_backend(is_mla=True, attn_cp_rank=3, attn_cp_size=4, subdir="m")
        self.assertTrue(b0.config_suffix.endswith("_cp0_4"))
        self.assertTrue(b1.config_suffix.endswith("_cp3_4"))
        self.assertNotEqual(b0.config_suffix, b1.config_suffix)


class TestMLAOwnerGating(HiCacheFileLRUTestBase):
    def test_mla_rank0_owns_eviction(self):
        b = self.make_backend(max_size="200", is_mla=True, tp_rank=0, tp_size=2)
        self.assertTrue(b._evictor.is_storage_owner)
        self.assertTrue(b._evictor.enabled)

    def test_mla_rank1_skips_eviction(self):
        b = self.make_backend(max_size="200", is_mla=True, tp_rank=1, tp_size=2)
        self.assertFalse(b._evictor.is_storage_owner)
        self.assertFalse(b._evictor.enabled)
        # Non-owner MLA ranks must not create new files when eviction is on.
        self.assertFalse(b.set("a", _t(50)))
        self.assertFalse(b.exists("a"))
        self.assertEqual(len(b._evictor._lru), 0)
        self.assertEqual(b._evictor._total_bytes, 0)

    def test_mla_rank1_can_touch_existing_file(self):
        # Non-owner ranks may still touch existing files, just not create new ones.
        b = self.make_backend(max_size="200", is_mla=True, tp_rank=1, tp_size=2)
        path = os.path.join(b.file_path, f"{b._get_suffixed_key('a')}.bin")
        with open(path, "wb") as f:
            f.write(b"x" * 50)
        self.assertTrue(b.set("a", _t(50)))
        self.assertTrue(b.exists("a"))
        self.assertEqual(len(b._evictor._lru), 0)

    def test_non_mla_each_rank_owns_its_files(self):
        # Non-MLA: even rank > 0 is its own owner because suffix isolates files.
        b = self.make_backend(max_size="200", is_mla=False, tp_rank=3, tp_size=4)
        self.assertTrue(b._evictor.is_storage_owner)
        self.assertTrue(b._evictor.enabled)


class TestTrackOrTouch(HiCacheFileLRUTestBase):
    def test_set_fast_path_adopts_external_file(self):
        # A file written by another rank should be adopted on the next set().
        b = self.make_backend(max_size="500")
        # Manually drop a suffixed file with the right name on disk.
        suffixed = b._get_suffixed_key("xkey")
        path = os.path.join(b.file_path, f"{suffixed}.bin")
        with open(path, "wb") as f:
            f.write(b"a" * 80)
        self.assertEqual(b._evictor._total_bytes, 0)
        self.assertNotIn(suffixed, b._evictor._lru)
        # set() should hit the fast path and adopt the file.
        self.assertTrue(b.set("xkey", _t(80)))
        self.assertIn(suffixed, b._evictor._lru)
        self.assertEqual(b._evictor._total_bytes, 80)

    def test_get_adopts_external_file(self):
        b = self.make_backend(max_size="500")
        suffixed = b._get_suffixed_key("ykey")
        path = os.path.join(b.file_path, f"{suffixed}.bin")
        with open(path, "wb") as f:
            f.write(b"\x00" * 64)
        # get() should return the data and also adopt the file.
        out = b.get("ykey", target_location=_t(64))
        self.assertIsNotNone(out)
        self.assertIn(suffixed, b._evictor._lru)
        self.assertEqual(b._evictor._total_bytes, 64)


class TestMinFreeSpaceWatermark(HiCacheFileLRUTestBase):
    def test_refuses_when_fs_would_drop_below_min_free(self):
        # Force statvfs to report a tiny free figure so the watermark trips.
        b = self.make_backend(max_size="0", min_free="100")
        # 150B free, writing 100B leaves 50B < 100B watermark -> refuse.
        b._evictor._fs_stats = lambda: (1024, 150)
        self.assertFalse(b.set("nope", _t(100)))
        self.assertFalse(b.exists("nope"))

    def test_evicts_to_satisfy_min_free(self):
        b = self.make_backend(max_size="0", min_free="100")
        # Pre-seed LRU with one 80B entry that is on disk.
        suffixed = b._get_suffixed_key("victim")
        path = os.path.join(b.file_path, f"{suffixed}.bin")
        with open(path, "wb") as f:
            f.write(b"v" * 80)
        b._evictor._lru[suffixed] = 80
        b._evictor._total_bytes = 80
        # 130 free; +60 write needs evicting the 80B victim to clear the watermark.
        free = [130]

        def fake_fs_stats():
            return (1024, free[0])

        original_remove = os.remove

        def tracked_remove(p):
            # Simulate tmpfs immediate free on unlink.
            if os.path.exists(p):
                free[0] += os.path.getsize(p)
            return original_remove(p)

        with mock.patch.object(
            b._evictor, "_fs_stats", side_effect=fake_fs_stats
        ), mock.patch("os.remove", side_effect=tracked_remove):
            self.assertTrue(b.set("newk", _t(60)))
        self.assertFalse(b.exists("victim"))
        self.assertTrue(b.exists("newk"))


class TestUnlinkFailureDoesNotWedge(HiCacheFileLRUTestBase):
    def test_unremovable_lru_front_file_does_not_block_eviction(self):
        """A single un-removable LRU-front file must not wedge all eviction.

        Regression: when ``os.remove`` raised a non-``FileNotFoundError``
        ``OSError`` (EPERM/EACCES/EBUSY/EROFS/EIO, plausible on NFS/FUSE/ro
        remounts), ``_evict_one_lru_locked`` re-pinned the victim at the LRU
        *front* and returned ``"stop"``. Every later eviction pass then popped
        that same file first, reclaimed nothing, and left the cap exceeded, so
        ``reserve`` refused all new writes forever -- one stuck cold file
        permanently disabled the whole file cache. The un-removable entry must
        instead be skipped so the removable newer entries behind it still get
        evicted.
        """
        b = self.make_backend(max_size="300", eviction_ratio=1.0)
        for k in ("a", "b", "c"):
            self.assertTrue(b.set(k, _t(100)))
        self.assertEqual(b._evictor._total_bytes, 300)
        # "a" is the LRU front; make only its unlink fail.
        a_path = os.path.join(b.file_path, f"{b._get_suffixed_key('a')}.bin")
        real_remove = os.remove

        def guarded_remove(p):
            if os.path.abspath(p) == os.path.abspath(a_path):
                raise PermissionError(13, "simulated EPERM")
            return real_remove(p)

        with mock.patch("os.remove", side_effect=guarded_remove):
            admitted = b.set("d", _t(100))

        # New write is admitted (not wedged): eviction skipped the stuck "a" and
        # reclaimed a removable neighbour instead.
        self.assertTrue(
            admitted, "d must be admitted; the stuck file must not wedge eviction"
        )
        self.assertTrue(b.exists("d"))
        self.assertTrue(b.exists("a"), "the un-removable file itself survives on disk")
        # Exactly one removable neighbour was evicted to make room.
        self.assertNotEqual(
            b.exists("b"),
            b.exists("c"),
            "exactly one of the removable neighbours (b/c) should be evicted",
        )
        self.assertLessEqual(b._evictor._total_bytes, 300)

    def test_all_unremovable_refuses_without_infinite_loop(self):
        """If every entry is un-removable, reserve refuses (bounded, no spin)."""
        b = self.make_backend(max_size="200", eviction_ratio=1.0)
        for k in ("a", "b"):
            self.assertTrue(b.set(k, _t(100)))
        self.assertEqual(b._evictor._total_bytes, 200)

        with mock.patch("os.remove", side_effect=PermissionError(13, "EPERM")):
            # Would spin forever if the skip budget in _evict_while were unbounded.
            admitted = b.set("c", _t(100))

        self.assertFalse(admitted, "nothing evictable -> refuse the new write")
        self.assertTrue(b.exists("a") and b.exists("b"))
        self.assertEqual(b._evictor._total_bytes, 200)


class TestPreReservationConcurrency(HiCacheFileLRUTestBase):
    def test_pre_reservation_visible_during_write(self):
        """An in-flight reservation must not be evicted by a concurrent set()."""
        b = self.make_backend(max_size="100", eviction_ratio=1.0)
        pending = b._get_suffixed_key("A")
        with b._evictor._lock:
            b._evictor._lru[pending] = 60
            b._evictor._pending_writes.add(pending)
            b._evictor._total_bytes = 60

        self.assertFalse(b.set("B", _t(60)))
        self.assertIn(pending, b._evictor._lru)
        self.assertIn(pending, b._evictor._pending_writes)
        self.assertEqual(b._evictor._total_bytes, sum(b._evictor._lru.values()))
        self.assertLessEqual(b._evictor._total_bytes, 100)


class TestMetadataCache(CustomTestCase):
    def test_metadata_cache_basic(self):
        cache = MetadataCache(ttl_seconds=1.0)
        cache.add("k1")
        self.assertTrue(cache.contains("k1"))
        self.assertFalse(cache.contains("k2"))
        cache.remove("k1")
        self.assertFalse(cache.contains("k1"))

    def test_metadata_cache_ttl(self):
        cache = MetadataCache(ttl_seconds=0.1)
        cache.add("k1")
        self.assertTrue(cache.contains("k1"))
        time.sleep(0.2)
        self.assertFalse(cache.contains("k1"))

    def test_metadata_cache_hard_ttl(self):
        cache = MetadataCache(ttl_seconds=0.3)
        cache.add("k1")
        time.sleep(0.15)
        # Try updating k1
        cache.add("k1")
        # Expiry is still 0.3s from original timestamp, i.e. 0.15s from now.
        time.sleep(0.2)
        self.assertFalse(cache.contains("k1"))

    def test_metadata_cache_infinite_ttl(self):
        cache = MetadataCache(ttl_seconds=-1.0)
        cache.add("k1")
        time.sleep(0.3)
        self.assertTrue(cache.contains("k1"))


class TestHiCacheFileMetadataIntegration(HiCacheFileLRUTestBase):
    def test_disabled_by_default(self):
        b = self.make_backend()
        self.assertIsNone(b.metadata_cache)
        self.assertFalse(b.enable_metadata_cache)

    def test_startup_scanning_populates_cache(self):
        d = tempfile.mkdtemp(prefix="hicache_metadata_seed_", dir=self.tmpdir)
        cfg = _make_config(
            model="seedmodel",
            extra_config={"metadata_ttl": 5.0, "enable_metadata_cache": True},
        )
        suffix = f"_seedmodel_0_1"

        # Pre-create a suffixed bin file on disk
        with open(os.path.join(d, f"k1{suffix}.bin"), "wb") as f:
            f.write(b"data")

        b = HiCacheFile(cfg, file_path=d)
        # It should be found in metadata cache on startup
        self.assertTrue(b.metadata_cache.contains(f"k1{suffix}"))

    def test_write_and_read_populates_cache(self):
        b = self.make_backend(metadata_ttl=5.0, enable_metadata_cache=True)
        suffix = b.config_suffix

        self.assertFalse(b.metadata_cache.contains(f"k1{suffix}"))
        b.set("k1", _t(50))
        # After set, it must be in the metadata cache
        self.assertTrue(b.metadata_cache.contains(f"k1{suffix}"))

        # Evict manually from metadata cache and call get
        b.metadata_cache.clear()
        self.assertFalse(b.metadata_cache.contains(f"k1{suffix}"))
        b.get("k1", target_location=_t(50))
        # Get should populate it back
        self.assertTrue(b.metadata_cache.contains(f"k1{suffix}"))

    def test_eviction_removes_from_metadata_cache(self):
        # max_size=200, so setting three 100B tensors will evict the oldest
        b = self.make_backend(
            max_size="200",
            eviction_ratio=1.0,
            metadata_ttl=-1.0,
            enable_metadata_cache=True,
        )
        suffix = b.config_suffix

        b.set("k1", _t(100))
        b.set("k2", _t(100))
        self.assertTrue(b.metadata_cache.contains(f"k1{suffix}"))
        self.assertTrue(b.metadata_cache.contains(f"k2{suffix}"))

        # Forces eviction of k1
        b.set("k3", _t(100))
        self.assertFalse(b.metadata_cache.contains(f"k1{suffix}"))
        self.assertTrue(b.metadata_cache.contains(f"k2{suffix}"))
        self.assertTrue(b.metadata_cache.contains(f"k3{suffix}"))

    def test_batch_exists_bypass_scandir(self):
        b = self.make_backend(metadata_ttl=5.0, enable_metadata_cache=True)
        suffix = b.config_suffix

        b.set("k1", _t(50))
        b.set("k2", _t(50))

        # Now patch os.scandir and os.path.exists
        with mock.patch("os.scandir") as mock_scandir, mock.patch(
            "os.path.exists"
        ) as mock_exists:
            mock_exists.return_value = True

            # batch_exists_v2 for k1 and k2 should hit the metadata cache and NOT call os.scandir or os.path.exists
            res = b.batch_exists_v2(["k1", "k2"])
            self.assertEqual(res.kv_hit_pages, 2)
            mock_scandir.assert_not_called()
            mock_exists.assert_not_called()

            # Querying "k3" (miss) should fall back to os.path.exists once but still NOT call os.scandir
            res = b.batch_exists_v2(["k3"])
            self.assertEqual(
                res.kv_hit_pages, 1
            )  # since mock_exists returns True, k3 exists physically
            mock_scandir.assert_not_called()
            mock_exists.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
