"""Unit tests for srt/mem_cache/hicache_storage.py

Covers:
- PoolTransferResult semantics
- HiCacheFile key / path generation:
  1. Page hash key:
     the input key to HiCacheFile is a page hash key, i.e. the logical cache key
     for one cached page. Its defining property is that it is computed as a
     prefix hash chain rather than as an independent per-page hash:
       k0 = hash(page0_tokens)
       k1 = hash(k0, page1_tokens)
       k2 = hash(k1, page2_tokens)
     This is needed because RoPE makes KV cache position-sensitive, so the same
     page tokens under different prefixes must map to different cache keys.

  2. Key derivation:
     _get_suffixed_key appends config_suffix to the page hash key, namespacing it
     by model / parallelism configuration. config_suffix is:
       _{model_name} for MLA without PP,
       _{model_name}_{tp_rank}_{tp_size} for non-MLA without PP,
     with _{pp_size}_{pp_rank} appended when pp_size > 1.
     _get_component_key then derives the final component key from that suffixed
     key: KV / None / "__default__" keep the same key, while non-KV pools insert
     a ".<pool_name>" infix before the config suffix.

  3. Path derivation:
     _get_component_path maps the component key to the on-disk file path
     "<storage_dir>/<component_key>.bin". HiCacheFile is therefore a
     directory-based storage backend, not a single file.

  Example:
    page hash chain: k0 -> k1 -> k2
    config suffix: "_mymodel_0_1"
    suffixed key for k2: "k2_mymodel_0_1"
    KV component key for k2: "k2_mymodel_0_1"
    Mamba component key for k2: "k2.mamba_mymodel_0_1"
    KV component path for k2: "<storage_dir>/k2_mymodel_0_1.bin"
- HiCacheFile.set / get / exists / clear  (file-system I/O)
- HiCacheFile.batch_get / batch_set  (v1 batch wrappers)
- HiCacheFile.batch_exists_v2  (KV-only, ALL_PAGES, TRAILING_PAGES, contiguous-prefix semantics)
- HiCacheFile.batch_get_v2 / batch_set_v2 / _batch_io_v2  (v2 multi-pool I/O pipeline)

All tests are CPU-only and use a temporary directory so they never touch
real storage and do not require a GPU.
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.test.test_utils import CustomTestCase

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_config(
    tp_rank: int = 0,
    tp_size: int = 1,
    pp_rank: int = 0,
    pp_size: int = 1,
    model_name: str = "meta-llama/Llama-3.2-1B",
    is_mla_model: bool = False,
) -> HiCacheStorageConfig:
    return HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=pp_rank,
        pp_size=pp_size,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=is_mla_model,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name=model_name,
    )


def _make_storage(tmp_dir: str, **config_kwargs) -> HiCacheFile:
    cfg = _make_config(**config_kwargs)
    return HiCacheFile(cfg, file_path=tmp_dir)


# ---------------------------------------------------------------------------
# PoolTransferResult
# ---------------------------------------------------------------------------


class TestPoolTransferResult(CustomTestCase):
    def test_empty(self):
        r = PoolTransferResult.empty()
        self.assertEqual(r.kv_hit_pages, 0)
        self.assertEqual(r.extra_pool_hit_pages, {})

    def test_update_kv_hit_pages_uses_max(self):
        r = PoolTransferResult.empty()
        r.update_kv_hit_pages(5)
        r.update_kv_hit_pages(3)  # smaller → should not overwrite
        self.assertEqual(r.kv_hit_pages, 5)
        r.update_kv_hit_pages(8)  # larger → should overwrite
        self.assertEqual(r.kv_hit_pages, 8)

    def test_update_extra_pool_hit_pages(self):
        r = PoolTransferResult.empty()
        r.update_extra_pool_hit_pages({"mamba": [True, False, True]})
        self.assertEqual(r.extra_pool_hit_pages["mamba"], 2)

    def test_update_extra_pool_hit_pages_overwrites(self):
        r = PoolTransferResult.empty()
        r.update_extra_pool_hit_pages({"mamba": [True, True]})
        r.update_extra_pool_hit_pages({"mamba": [False]})
        self.assertEqual(r.extra_pool_hit_pages["mamba"], 0)


# ---------------------------------------------------------------------------
# HiCacheFile key / path generation
# ---------------------------------------------------------------------------


class TestHiCacheFileKeyGeneration(CustomTestCase):
    """Tests for config_suffix, _get_suffixed_key, _get_component_key, _get_component_path.

    Note: despite its name, HiCacheFile is NOT a single file — it is a
    directory-based storage backend where each cached page is stored as an
    individual '<component_key>.bin' file under a root storage directory
    (file_path).  The key generation methods tested here determine the file
    name for each page inside that directory.
    """

    def setUp(self):
        self._tmp_obj = tempfile.TemporaryDirectory()
        self.tmp = self._tmp_obj.name
        self.addCleanup(self._tmp_obj.cleanup)

    # config_suffix construction ------------------------------------------------

    def test_suffix_non_mla(self):
        # Non-MLA: suffix = _{model_name}_{tp_rank}_{tp_size}
        s = _make_storage(self.tmp, model_name="org/model", tp_rank=1, tp_size=4)
        self.assertEqual(s.config_suffix, "_org-model_1_4")

    def test_suffix_mla_no_tp_fields(self):
        # Non-MLA models encode TP-specific differences in the cache object
        # itself: each TP rank stores a different head shard of the K/V cache,
        # so the cache cannot be shared across TP ranks. MLA instead encodes
        # TP-specific differences in the projection matrices: the cache stores
        # a head-agnostic low-rank latent KV, and each TP rank consumes the
        # same cached latent with its own projection weights, so the cache can
        # be shared across TP ranks. Therefore MLA omits TP rank/size from the
        # storage suffix and uses suffix = _{model_name}.
        s = _make_storage(
            self.tmp, model_name="org/model", tp_rank=0, tp_size=4, is_mla_model=True
        )
        self.assertEqual(s.config_suffix, "_org-model")

    def test_suffix_pp_appended_when_pp_size_gt_1(self):
        # PP enabled: suffix = _{model_name}_{tp_rank}_{tp_size}_{pp_size}_{pp_rank}
        s = _make_storage(self.tmp, pp_size=2, pp_rank=1)
        self.assertEqual(s.config_suffix, "_meta-llama-Llama-3.2-1B_0_1_2_1")

    def test_suffix_no_pp_when_pp_size_eq_1(self):
        # PP disabled: suffix = _{model_name}_{tp_rank}_{tp_size}
        s = _make_storage(self.tmp, pp_size=1, pp_rank=0)
        self.assertEqual(s.config_suffix, "_meta-llama-Llama-3.2-1B_0_1")

    def test_slash_in_model_name_replaced(self):
        # Slashes in model name are replaced with hyphens
        s = _make_storage(self.tmp, model_name="org/subdir/model")
        self.assertEqual(s.config_suffix, "_org-subdir-model_0_1")

    # _get_suffixed_key ---------------------------------------------------------

    def test_get_suffixed_key(self):
        # key is a SHA-256 hex digest identifying one cached page.  Pages are
        # hashed in a chained fashion (each digest covers the current page's
        # tokens plus the prior page's digest), so the key sequence forms a
        # content-addressed prefix chain that enables contiguous-prefix matching.
        # _get_suffixed_key appends config_suffix (model name + TP/PP ranks) to
        # namespace the key, preventing collisions between different model or
        # parallelism configurations sharing the same storage directory.
        s = _make_storage(self.tmp, model_name="mymodel", tp_rank=0, tp_size=1)
        # SHA-256 hex digest of token_ids=[1, 2, 3, 4] (one page)
        page_hash_key = (
            "cf97adeedb59e05bfd73a2b4c2a8885708c4f4f70c84c64b27120e72ab733b72"
        )
        self.assertEqual(
            s._get_suffixed_key(page_hash_key),
            "cf97adeedb59e05bfd73a2b4c2a8885708c4f4f70c84c64b27120e72ab733b72_mymodel_0_1",
        )

    # _get_component_key --------------------------------------------------------

    def test_component_key_kv_same_as_suffixed(self):
        # KV is the primary pool; its component key is identical to the plain
        # suffixed key (no pool infix inserted).
        s = _make_storage(self.tmp)
        page_hash_key = (
            "cf97adeedb59e05bfd73a2b4c2a8885708c4f4f70c84c64b27120e72ab733b72"
        )
        self.assertEqual(
            s._get_component_key(page_hash_key, PoolName.KV),
            s._get_suffixed_key(page_hash_key),
        )

    def test_component_key_default_same_as_suffixed(self):
        # "__default__" is a magic string that _get_component_key treats
        # identically to PoolName.KV (no pool infix is inserted).
        # It is used by callers that pass a component_name without knowing
        # whether a specific pool is configured — e.g. legacy code paths that
        # pre-date the PoolName enum.  The test locks this aliasing behaviour
        # so that a refactor removing the magic string would be caught here.
        s = _make_storage(self.tmp)
        page_hash_key = (
            "cf97adeedb59e05bfd73a2b4c2a8885708c4f4f70c84c64b27120e72ab733b72"
        )
        self.assertEqual(
            s._get_component_key(page_hash_key, "__default__"),
            s._get_suffixed_key(page_hash_key),
        )

    def test_component_key_none_same_as_suffixed(self):
        # None component name falls back to the same path as KV — no pool infix.
        s = _make_storage(self.tmp)
        page_hash_key = (
            "cf97adeedb59e05bfd73a2b4c2a8885708c4f4f70c84c64b27120e72ab733b72"
        )
        self.assertEqual(
            s._get_component_key(page_hash_key, None),
            s._get_suffixed_key(page_hash_key),
        )
        # literal: confirms the exact resulting key string
        self.assertEqual(
            s._get_component_key(page_hash_key, None),
            "cf97adeedb59e05bfd73a2b4c2a8885708c4f4f70c84c64b27120e72ab733b72"
            "_meta-llama-Llama-3.2-1B_0_1",
        )

    def test_component_key_mamba_adds_pool_infix(self):
        # Non-KV pools get a "<hash>.<pool_name>" infix before the config suffix.
        s = _make_storage(self.tmp)
        page_hash_key = (
            "cf97adeedb59e05bfd73a2b4c2a8885708c4f4f70c84c64b27120e72ab733b72"
        )
        self.assertEqual(
            s._get_component_key(page_hash_key, PoolName.MAMBA),
            "cf97adeedb59e05bfd73a2b4c2a8885708c4f4f70c84c64b27120e72ab733b72"
            ".mamba_meta-llama-Llama-3.2-1B_0_1",
        )

    # _get_component_path -------------------------------------------------------

    def test_component_path(self):
        s = _make_storage(self.tmp)
        page_hash_key = (
            "cf97adeedb59e05bfd73a2b4c2a8885708c4f4f70c84c64b27120e72ab733b72"
        )
        expected = os.path.join(
            self.tmp,
            "cf97adeedb59e05bfd73a2b4c2a8885708c4f4f70c84c64b27120e72ab733b72"
            "_meta-llama-Llama-3.2-1B_0_1.bin",
        )
        self.assertEqual(s._get_component_path(page_hash_key), expected)
        # literal: confirms the full path structure — <storage_dir>/<hash><config_suffix>.bin
        self.assertEqual(
            s._get_component_path(page_hash_key),
            os.path.join(
                self.tmp,
                "cf97adeedb59e05bfd73a2b4c2a8885708c4f4f70c84c64b27120e72ab733b72"
                "_meta-llama-Llama-3.2-1B_0_1.bin",
            ),
        )

    def test_component_path_with_pool_name(self):
        # Non-KV pools (e.g. Mamba) get a "<hash>.<pool_name>" infix before
        # the config suffix, resulting in a distinct file path so that KV and
        # pool pages stored under the same page hash never collide.
        s = _make_storage(self.tmp)
        page_hash_key = (
            "cf97adeedb59e05bfd73a2b4c2a8885708c4f4f70c84c64b27120e72ab733b72"
        )
        expected = os.path.join(
            self.tmp,
            "cf97adeedb59e05bfd73a2b4c2a8885708c4f4f70c84c64b27120e72ab733b72"
            ".mamba_meta-llama-Llama-3.2-1B_0_1.bin",
        )
        self.assertEqual(s._get_component_path(page_hash_key, PoolName.MAMBA), expected)


# ---------------------------------------------------------------------------
# HiCacheFile set / get / exists / clear
# ---------------------------------------------------------------------------


class TestHiCacheFileIO(CustomTestCase):
    def setUp(self):
        self._tmp_obj = tempfile.TemporaryDirectory()
        self.tmp = self._tmp_obj.name
        self.addCleanup(self._tmp_obj.cleanup)
        self.storage = _make_storage(self.tmp)

    def _random_tensor(self, shape=(4, 8)):
        return torch.randn(*shape, dtype=torch.float32)

    def test_exists_returns_false_for_missing_key(self):
        missing_page_hash_key = "deadbeef_page_not_written"
        self.assertFalse(self.storage.exists(missing_page_hash_key))

    def test_set_creates_file_and_exists_returns_true(self):
        page_hash_key = "cafebabe_page_0001"
        t = self._random_tensor()
        result = self.storage.set(page_hash_key, t)
        self.assertTrue(result)
        self.assertTrue(self.storage.exists(page_hash_key))

    def test_set_idempotent_for_existing_key(self):
        # First write: tensor A is stored.
        page_hash_key = "cafebabe_page_idem"
        tensor_a = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        self.storage.set(page_hash_key, tensor_a)

        # Second write with a *different* tensor B: set() should return True
        # (early-return / skip path) and NOT overwrite the file.
        tensor_b = torch.tensor([9.0, 9.0, 9.0, 9.0], dtype=torch.float32)
        result2 = self.storage.set(page_hash_key, tensor_b)
        self.assertTrue(result2)

        # Read back — must still be tensor_a, proving set() really skipped.
        buf = torch.zeros(4, dtype=torch.float32)
        loaded = self.storage.get(page_hash_key, buf)
        self.assertIsNotNone(loaded)
        self.assertTrue(torch.allclose(tensor_a, loaded))

    def test_get_returns_none_for_missing_key(self):
        missing_page_hash_key = "deadbeef_page_not_written"
        buf = self._random_tensor()
        result = self.storage.get(missing_page_hash_key, buf)
        self.assertIsNone(result)

    def test_set_then_get_roundtrip(self):
        page_hash_key = "cafebabe_page_roundtrip"
        original = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        self.storage.set(page_hash_key, original)

        buf = torch.zeros(4, dtype=torch.float32)
        loaded = self.storage.get(page_hash_key, buf)
        self.assertIsNotNone(loaded)
        self.assertTrue(torch.allclose(original, loaded))

    def test_clear_removes_all_files(self):
        page_hash_keys = [f"cafebabe_page_{i:03d}" for i in range(3)]
        for page_hash_key in page_hash_keys:
            self.storage.set(page_hash_key, self._random_tensor())
        self.storage.clear()
        for page_hash_key in page_hash_keys:
            self.assertFalse(self.storage.exists(page_hash_key))

    def test_batch_exists_returns_zero_when_all_missing(self):
        missing_page_hash_keys = [
            "deadbeef_page_0",
            "deadbeef_page_1",
            "deadbeef_page_2",
        ]
        self.assertEqual(self.storage.batch_exists(missing_page_hash_keys), 0)

    def test_batch_exists_returns_contiguous_prefix_count(self):
        t = self._random_tensor()
        self.storage.set("cafebabe_page_0", t)
        self.storage.set("cafebabe_page_1", t)
        # cafebabe_page_2 is missing — prefix chain breaks here
        self.storage.set("cafebabe_page_3", t)
        page_hash_keys = [
            "cafebabe_page_0",
            "cafebabe_page_1",
            "cafebabe_page_2",
            "cafebabe_page_3",
        ]
        self.assertEqual(self.storage.batch_exists(page_hash_keys), 2)

    def test_batch_exists_returns_full_count_when_all_present(self):
        t = self._random_tensor()
        page_hash_keys = [f"cafebabe_page_{i}" for i in range(4)]
        for page_hash_key in page_hash_keys:
            self.storage.set(page_hash_key, t)
        self.assertEqual(self.storage.batch_exists(page_hash_keys), 4)


# ---------------------------------------------------------------------------
# HiCacheFile.batch_exists_v2
# ---------------------------------------------------------------------------


class TestBatchExistsV2(CustomTestCase):
    """Tests for the per-pool prefix-hit logic in batch_exists_v2."""

    def setUp(self):
        self._tmp_obj = tempfile.TemporaryDirectory()
        self.tmp = self._tmp_obj.name
        self.addCleanup(self._tmp_obj.cleanup)
        self.storage = _make_storage(self.tmp)
        self.t = torch.randn(4, dtype=torch.float32)

    def _write_kv(self, key: str):
        self.storage.set(key, self.t)

    def _write_component(self, key: str, pool_name: str):
        """Write a non-KV component (e.g. mamba) using the component path directly."""
        component_key = f"{key}.{pool_name}"
        self.storage.set(component_key, self.t)

    # KV-only (no pool_transfers) -----------------------------------------------

    def test_kv_only_all_missing(self):
        keys = ["k0", "k1", "k2"]
        result = self.storage.batch_exists_v2(keys)
        self.assertEqual(result.kv_hit_pages, 0)
        self.assertEqual(result.extra_pool_hit_pages, {})

    def test_kv_only_full_prefix(self):
        keys = [f"p{i}" for i in range(4)]
        for k in keys:
            self._write_kv(k)
        result = self.storage.batch_exists_v2(keys)
        self.assertEqual(result.kv_hit_pages, 4)

    def test_kv_only_partial_prefix(self):
        # pages 0, 1 present; page 2 missing → prefix chain breaks at index 2.
        # page 3 is present but lies beyond the gap and must NOT be included.
        keys = ["a0", "a1", "a2", "a3"]
        self._write_kv("a0")
        self._write_kv("a1")
        # a2 is absent
        self._write_kv("a3")
        result = self.storage.batch_exists_v2(keys)
        self.assertEqual(result.kv_hit_pages, 2)

    def test_kv_only_partial_prefix_a3_absent_same_result(self):
        # Whether a3 is present or not must not affect the prefix length,
        # because the chain already breaks at the missing a2.
        keys = ["a0", "a1", "a2", "a3"]
        self._write_kv("a0")
        self._write_kv("a1")
        # a2 and a3 both absent
        result = self.storage.batch_exists_v2(keys)
        self.assertEqual(result.kv_hit_pages, 2)

    # ALL_PAGES policy ----------------------------------------------------------
    # ALL_PAGES is used by pools where every page is independently required,
    # e.g. DSA (Differential Sliding-window Attention): each KV page holds a
    # distinct attention window and cannot be skipped.  The policy trims
    # kv_hit_pages to the longest prefix where every page has a matching pool
    # file.  (Mamba SSM state uses TRAILING_PAGES instead, because it is a
    # single cumulative state stored only at the node boundary.)

    def test_all_pages_pool_shrinks_prefix_when_pool_page_missing(self):
        # KV has 4 pages; DSA pool missing page 2 → boundary trimmed to 2.
        keys = [f"b{i}" for i in range(4)]
        for k in keys:
            self._write_kv(k)
        self._write_component("b0", "dsa")
        self._write_component("b1", "dsa")
        # b2.dsa is absent; b3.dsa is absent

        transfer = PoolTransfer(
            name="dsa",
            keys=keys,
            hit_policy=PoolHitPolicy.ALL_PAGES,
        )
        result = self.storage.batch_exists_v2(keys, pool_transfers=[transfer])
        self.assertEqual(result.kv_hit_pages, 2)

    def test_all_pages_pool_full_match(self):
        keys = [f"c{i}" for i in range(3)]
        for k in keys:
            self._write_kv(k)
            self._write_component(k, "dsa")

        transfer = PoolTransfer(
            name="dsa",
            keys=keys,
            hit_policy=PoolHitPolicy.ALL_PAGES,
        )
        result = self.storage.batch_exists_v2(keys, pool_transfers=[transfer])
        self.assertEqual(result.kv_hit_pages, 3)

    def test_all_pages_pool_does_not_expand_beyond_kv(self):
        # DSA pool has all 4 pages but KV only has 2; final_pages must not exceed kv_pages.
        keys = [f"d{i}" for i in range(4)]
        self._write_kv("d0")
        self._write_kv("d1")
        # d2, d3 KV missing
        for k in keys:
            self._write_component(k, "dsa")

        transfer = PoolTransfer(
            name="dsa",
            keys=keys,
            hit_policy=PoolHitPolicy.ALL_PAGES,
        )
        result = self.storage.batch_exists_v2(keys, pool_transfers=[transfer])
        self.assertEqual(result.kv_hit_pages, 2)

    # TRAILING_PAGES policy -----------------------------------------------------

    def test_trailing_pages_only_last_n_required(self):
        # 4 KV pages present; mamba only has last 2 (pages 2, 3)
        # With trailing_pages policy and transfer.keys covering 2 entries → should hit 4
        #
        # Note: only len(transfer.keys) is consumed by batch_exists_v2 to derive
        # the trailing window size; the key values themselves are not used for
        # file lookup in the exists path.  The semantic contract for transfer.keys
        # content under TRAILING_PAGES is currently undefined — callers should
        # pass the actual trailing page hashes (as mamba_archive_transfers does)
        # to stay compatible with future implementations.
        keys = [f"e{i}" for i in range(4)]
        for k in keys:
            self._write_kv(k)
        self._write_component("e2", "mamba")
        self._write_component("e3", "mamba")

        # transfer.keys has 2 entries (trailing window = 2)
        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            keys=["e2", "e3"],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        result = self.storage.batch_exists_v2(keys, pool_transfers=[transfer])
        # last 2 pages of the 4-page KV prefix must exist in mamba → hit 4
        self.assertEqual(result.kv_hit_pages, 4)

    def test_trailing_pages_partial_prefix_when_tail_absent(self):
        # 4 KV pages; mamba has only pages 0, 1.
        # TRAILING_PAGES checks whether the last 'trailing' pages of any
        # prefix_len sub-window exist.  With trailing=2:
        #   prefix_len=4: needs pages 2,3 in mamba → absent
        #   prefix_len=3: needs pages 1,2 in mamba → f1 present, f2 absent
        #   prefix_len=2: needs pages 0,1 in mamba → both present → boundary=2
        # So final_pages = min(kv=4, boundary=2) = 2.
        keys = [f"f{i}" for i in range(4)]
        for k in keys:
            self._write_kv(k)
        self._write_component("f0", "mamba")
        self._write_component("f1", "mamba")

        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            keys=["f2", "f3"],  # trailing window = 2
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        result = self.storage.batch_exists_v2(keys, pool_transfers=[transfer])
        # The algorithm finds the longest prefix where the last 2 mamba pages exist.
        # prefix_len=2 satisfies (pages 0,1) → boundary=2 → final=2
        self.assertEqual(result.kv_hit_pages, 2)

    def test_trailing_pages_zero_when_no_mamba_at_all(self):
        # 3 KV pages; no mamba component written at all → boundary=0 → final=0
        keys = [f"h{i}" for i in range(3)]
        for k in keys:
            self._write_kv(k)

        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            keys=["h1", "h2"],  # trailing window = 2
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        result = self.storage.batch_exists_v2(keys, pool_transfers=[transfer])
        self.assertEqual(result.kv_hit_pages, 0)

    # No-KV-hit early exit ------------------------------------------------------

    def test_no_kv_hit_skips_pool_evaluation(self):
        # All KV missing → pool_transfers should not be evaluated (final_pages = 0)
        keys = ["g0", "g1"]
        self._write_component("g0", "mamba")

        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            keys=keys,
            hit_policy=PoolHitPolicy.ALL_PAGES,
        )
        result = self.storage.batch_exists_v2(keys, pool_transfers=[transfer])
        self.assertEqual(result.kv_hit_pages, 0)

    # Edge cases ---------------------------------------------------------------

    def test_empty_keys_returns_zero_hit(self):
        # Empty key list: both kv_hit_pages and extra_pool_hit_pages must be empty.
        result = self.storage.batch_exists_v2([])
        self.assertEqual(result.kv_hit_pages, 0)
        self.assertEqual(result.extra_pool_hit_pages, {})

    def test_kv_zero_hit_does_not_include_kv_in_hit_count(self):
        # When kv_pages == 0, the source guards hit_count insertion with
        # ``if kv_pages``, so KV must NOT appear as a key in extra_pool_hit_pages.
        keys = ["missing0", "missing1"]
        result = self.storage.batch_exists_v2(keys)
        self.assertNotIn(PoolName.KV, result.extra_pool_hit_pages)

    def test_multiple_pool_transfers_final_pages_is_minimum(self):
        # KV: 4 pages present.
        # DSA (ALL_PAGES): pages 0-2 present → boundary=3.
        # Mamba (TRAILING_PAGES, trailing=2): only pages 2,3 present → boundary=4.
        # final_pages = min(4, 3, 4) = 3.
        keys = [f"m{i}" for i in range(4)]
        for k in keys:
            self._write_kv(k)
        # DSA: pages 0,1,2 → boundary 3 (page 3 missing)
        for k in ["m0", "m1", "m2"]:
            self._write_component(k, "dsa")
        # Mamba: pages 2,3 → trailing=2 satisfied at prefix_len=4
        self._write_component("m2", "mamba")
        self._write_component("m3", "mamba")

        dsa_transfer = PoolTransfer(
            name="dsa",
            keys=keys,
            hit_policy=PoolHitPolicy.ALL_PAGES,
        )
        mamba_transfer = PoolTransfer(
            name=PoolName.MAMBA,
            keys=["m2", "m3"],  # trailing window = 2
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        result = self.storage.batch_exists_v2(
            keys, pool_transfers=[dsa_transfer, mamba_transfer]
        )
        self.assertEqual(result.kv_hit_pages, 3)

    # Unrelated-file isolation --------------------------------------------------

    def test_batch_exists_v2_ignores_unrelated_files(self):
        # batch_exists_v2 scans the storage directory with os.scandir and
        # matches only files whose names are in the pre-computed target set.
        # An unrelated file must never be counted as a cache hit.
        self._write_kv("r0")

        # Plant a file that looks like a .bin but has nothing to do with any
        # queried key.
        unrelated_path = os.path.join(self.tmp, "random_unrelated_file.bin")
        with open(unrelated_path, "wb") as f:
            f.write(b"\x00" * 16)

        result = self.storage.batch_exists_v2(["r0"])
        self.assertEqual(result.kv_hit_pages, 1)

        result_no_kv = self.storage.batch_exists_v2(["missing_key"])
        self.assertEqual(result_no_kv.kv_hit_pages, 0)


# ---------------------------------------------------------------------------
# HiCacheFile.batch_get / batch_set  (v1 batch wrappers)
# ---------------------------------------------------------------------------


class TestHiCacheFileBatchV1(CustomTestCase):
    """Tests for the v1 batch_get / batch_set convenience wrappers.

    These methods delegate to get() / set() per-entry, so they exercise the
    same file-system I/O path.  We verify correctness of the delegation and
    the early-exit-on-failure semantics of batch_set.
    """

    def setUp(self):
        self._tmp_obj = tempfile.TemporaryDirectory()
        self.tmp = self._tmp_obj.name
        self.addCleanup(self._tmp_obj.cleanup)
        self.storage = _make_storage(self.tmp)

    def _tensor(self):
        return torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)

    def test_batch_set_writes_all_and_returns_true(self):
        keys = ["cafebabe_bs0", "cafebabe_bs1", "cafebabe_bs2"]
        values = [self._tensor() for _ in keys]
        ok = self.storage.batch_set(keys, values)
        self.assertTrue(ok)
        for k in keys:
            self.assertTrue(self.storage.exists(k))

    def test_batch_get_roundtrip(self):
        keys = ["cafebabe_bg0", "cafebabe_bg1"]
        values = [
            torch.tensor([float(i)] * 4, dtype=torch.float32) for i in range(len(keys))
        ]
        self.storage.batch_set(keys, values)

        bufs = [torch.zeros(4, dtype=torch.float32) for _ in keys]
        results = self.storage.batch_get(keys, bufs)
        for original, loaded in zip(values, results):
            self.assertIsNotNone(loaded)
            self.assertTrue(torch.allclose(original, loaded))

    def test_batch_get_missing_key_returns_none(self):
        # One key exists, one does not.
        self.storage.set("cafebabe_bg_ok", self._tensor())
        bufs = [
            torch.zeros(4, dtype=torch.float32),
            torch.zeros(4, dtype=torch.float32),
        ]
        results = self.storage.batch_get(["cafebabe_bg_ok", "deadbeef_missing"], bufs)
        self.assertIsNotNone(results[0])
        self.assertIsNone(results[1])

    def test_batch_set_returns_false_on_write_failure(self):
        # Use mock.patch.object rather than passing a non-tensor so that the
        # failure is explicit and not tied to a specific internal exception type.
        # This remains valid even if set() adds input validation in the future.
        t = self._tensor()
        with patch.object(self.storage, "set", side_effect=[True, False]):
            ok = self.storage.batch_set(["cafebabe_fail0", "cafebabe_fail1"], [t, t])
        self.assertFalse(ok)


# ---------------------------------------------------------------------------
# HiCacheFile.batch_get_v2 / batch_set_v2 / _batch_io_v2
# ---------------------------------------------------------------------------


def _make_host_pool(page_size: int = 1, page_numel: int = 4) -> MagicMock:
    """Return a mock host pool that stores flat data pages as plain tensors.

    The mock implements the subset of the HostKVCache interface used by
    _read_page / _write_page:
      - get_dummy_flat_data_page() → a zero tensor of shape (page_numel,)
      - get_data_page(offset, flat=True) → a ones tensor of shape (page_numel,)
      - set_from_flat_data_page(offset, data) → recorded by mock

    ``page_size`` controls the stride used by ``_batch_io_v2`` when indexing
    into ``host_indices``  (``host_indices[i * page_size]``).  Set it to >1
    when you want to verify that the stride logic is correct.
    ``page_numel`` is the number of float32 elements per flat page; it should
    be self-consistent with the tensors your test writes to storage.
    """
    pool = MagicMock()
    pool.page_size = page_size
    pool.get_dummy_flat_data_page.return_value = torch.zeros(
        page_numel, dtype=torch.float32
    )
    pool.get_data_page.return_value = torch.ones(page_numel, dtype=torch.float32)
    return pool


class TestHiCacheFileBatchV2IO(CustomTestCase):
    """Tests for the v2 batch I/O pipeline: _batch_io_v2, batch_get_v2, batch_set_v2.

    host pools are replaced with lightweight mocks so no GPU or real KV cache
    is required.  The tests verify:
    - batch_set_v2 writes each (key, page) pair to storage and returns success flags.
    - batch_get_v2 reads each (key, page) pair back and calls set_from_flat_data_page.
    - _batch_io_v2 returns [False, ...] when host_indices length mismatches.
    - Multiple PoolTransfers in one call are each processed independently.
    """

    def setUp(self):
        self._tmp_obj = tempfile.TemporaryDirectory()
        self.tmp = self._tmp_obj.name
        self.addCleanup(self._tmp_obj.cleanup)
        self.storage = _make_storage(self.tmp)

    def _register_pool(self, name, pool):
        """Register a mock host pool under the given name."""
        self.storage.register_mem_host_pool_v2(pool, name)

    def test_batch_set_v2_writes_pages_and_returns_true_flags(self):
        pool = _make_host_pool(page_size=1)
        self._register_pool(PoolName.KV, pool)

        keys = ["cafebabe_v2s0", "cafebabe_v2s1"]
        transfer = PoolTransfer(
            name=PoolName.KV,
            keys=keys,
            host_indices=torch.tensor([0, 1]),
        )
        results = self.storage.batch_set_v2([transfer])
        self.assertEqual(results[PoolName.KV], [True, True])
        # Files must now exist in storage (under the KV component path, which
        # is the plain suffixed key since KV has no pool infix).
        for k in keys:
            self.assertTrue(self.storage.exists(k))

    def test_batch_get_v2_reads_pages_and_calls_set_from_flat(self):
        # Use page_size=2 so host_indices has 2 slots per key.
        # _batch_io_v2 picks offset = host_indices[i * page_size]:
        #   key 0 → host_indices[0 * 2] = host_indices[0] = 10
        #   key 1 → host_indices[1 * 2] = host_indices[2] = 20
        # If the impl were wrong (e.g. host_indices[i]), it would pick [10, 11]
        # instead of [10, 20], and the assertion below would catch it.
        pool = _make_host_pool(page_size=2)
        self._register_pool(PoolName.KV, pool)

        keys = ["cafebabe_v2g0", "cafebabe_v2g1"]

        # Write pages first (page_size=2 → host_indices needs 2*len(keys)=4 entries).
        write_transfer = PoolTransfer(
            name=PoolName.KV,
            keys=keys,
            host_indices=torch.tensor([10, 11, 20, 21]),
        )
        self.storage.batch_set_v2([write_transfer])

        # Read back and verify call count + exact offsets passed to set_from_flat_data_page.
        pool.reset_mock()
        read_transfer = PoolTransfer(
            name=PoolName.KV,
            keys=keys,
            host_indices=torch.tensor([10, 11, 20, 21]),
        )
        results = self.storage.batch_get_v2([read_transfer])
        self.assertEqual(results[PoolName.KV], [True, True])

        # set_from_flat_data_page must be called exactly once per key.
        self.assertEqual(pool.set_from_flat_data_page.call_count, len(keys))

        # Verify that the page_offset argument (first positional arg) follows
        # host_indices[i * page_size] stride — not the simpler host_indices[i].
        offsets = [call.args[0] for call in pool.set_from_flat_data_page.call_args_list]
        self.assertEqual(offsets, [10, 20])

    def test_batch_get_v2_returns_false_for_missing_page(self):
        pool = _make_host_pool(page_size=1)
        self._register_pool(PoolName.KV, pool)

        # "deadbeef_v2missing" was never written.
        transfer = PoolTransfer(
            name=PoolName.KV,
            keys=["deadbeef_v2missing"],
            host_indices=torch.tensor([0]),
        )
        results = self.storage.batch_get_v2([transfer])
        self.assertEqual(results[PoolName.KV], [False])
        # A missing page must NOT call set_from_flat_data_page — the host
        # buffer must remain untouched when storage returns None.
        pool.set_from_flat_data_page.assert_not_called()

    def test_batch_io_v2_returns_false_list_on_index_length_mismatch(self):
        pool = _make_host_pool(page_size=1)
        self._register_pool(PoolName.KV, pool)

        # 2 keys but host_indices has only 1 entry → mismatch
        transfer = PoolTransfer(
            name=PoolName.KV,
            keys=["cafebabe_mm0", "cafebabe_mm1"],
            host_indices=torch.tensor([0]),  # too short
        )
        results = self.storage.batch_set_v2([transfer])
        self.assertEqual(results[PoolName.KV], [False, False])

    def test_batch_io_v2_get_returns_false_list_on_index_length_mismatch(self):
        # Mirrors the set mismatch test for batch_get_v2, so that if set/get
        # are ever split into separate _batch_io implementations the guard is
        # independently covered for both paths.
        pool = _make_host_pool(page_size=1)
        self._register_pool(PoolName.KV, pool)

        transfer = PoolTransfer(
            name=PoolName.KV,
            keys=["cafebabe_get_mm0", "cafebabe_get_mm1"],
            host_indices=torch.tensor([0]),  # too short (expected 2)
        )
        results = self.storage.batch_get_v2([transfer])
        self.assertEqual(results[PoolName.KV], [False, False])

    def test_batch_set_v2_multiple_transfers_processed_independently(self):
        kv_pool = _make_host_pool(page_size=1)
        mamba_pool = _make_host_pool(page_size=1)
        self._register_pool(PoolName.KV, kv_pool)
        self._register_pool(PoolName.MAMBA, mamba_pool)

        kv_keys = ["cafebabe_multi_kv0"]
        mamba_keys = ["cafebabe_multi_mb0"]

        transfers = [
            PoolTransfer(
                name=PoolName.KV,
                keys=kv_keys,
                host_indices=torch.tensor([0]),
            ),
            PoolTransfer(
                name=PoolName.MAMBA,
                keys=mamba_keys,
                host_indices=torch.tensor([0]),
            ),
        ]
        results = self.storage.batch_set_v2(transfers)
        self.assertEqual(results[PoolName.KV], [True])
        self.assertEqual(results[PoolName.MAMBA], [True])


if __name__ == "__main__":
    unittest.main(verbosity=2)
