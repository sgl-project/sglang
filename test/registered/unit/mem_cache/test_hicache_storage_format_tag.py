"""A host pool with its own page byte format namespaces its storage keys.

The unified pool's host mirror stores whole page envelopes, whose bytes are
token-major entries. A page persisted under another format has the same size,
so only the key can keep it from being read back as a unified page: the
storage model name carries the pool's format tag, and an untagged page misses.

CPU-only.

    python -m pytest test/registered/unit/mem_cache/test_hicache_storage_format_tag.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.cache_controller import storage_model_name
from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.srt.mem_cache.memory_pool_host import LogicalHostPool
from sglang.srt.mem_cache.pool_host.base import HostKVCache
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.unified import UnifiedPageEnvelopeHostPool
from sglang.test.test_utils import CustomTestCase


def _config(model_name):
    return HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name=model_name,
    )


def _host_pool(dtype, *, envelope=False):
    pool_type = UnifiedPageEnvelopeHostPool if envelope else MHATokenToKVPoolHost
    pool = pool_type.__new__(pool_type)
    store_dtype = torch.uint8 if dtype.itemsize == 1 else dtype
    if envelope:
        pool.dtype = torch.uint8
        pool.device_pool = SimpleNamespace(store_dtype=store_dtype, dtype=dtype)
    else:
        pool.dtype = store_dtype
        pool._value_dtype = dtype
    return pool


class TestStorageModelName(CustomTestCase):
    def test_default_pools_tag_the_model_name_with_dtype(self):
        self.assertIsNone(HostKVCache.storage_format_tag)
        pool = _host_pool(torch.bfloat16)
        self.assertEqual(
            storage_model_name("org/model", pool), "org/model-dtype_bfloat16"
        )
        self.assertEqual(storage_model_name(None, pool), "dtype_bfloat16")

    def test_logical_anchor_pool_keeps_the_model_name(self):
        """A LogicalHostPool storage anchor keeps the untagged model name."""
        pool = LogicalHostPool(size=4, page_size=2)
        self.assertEqual(storage_model_name("org/model", pool), "org/model")
        self.assertIsNone(storage_model_name(None, pool))

    def test_unified_pool_tags_the_model_name(self):
        tag = UnifiedPageEnvelopeHostPool.storage_format_tag
        pool = _host_pool(torch.bfloat16, envelope=True)
        self.assertIsNotNone(tag)
        self.assertEqual(
            storage_model_name("org/model", pool),
            f"org/model-{tag}-dtype_bfloat16",
        )
        self.assertEqual(storage_model_name(None, pool), f"{tag}-dtype_bfloat16")

    def test_untagged_page_misses_under_the_unified_name(self):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        untagged = HiCacheFile(
            _config(storage_model_name("org/model", _host_pool(torch.bfloat16))),
            file_path=root,
        )
        tagged = HiCacheFile(
            _config(
                storage_model_name(
                    "org/model", _host_pool(torch.bfloat16, envelope=True)
                )
            ),
            file_path=root,
        )
        self.assertTrue(untagged.set("page-0", torch.zeros(64, dtype=torch.uint8)))
        self.assertTrue(untagged.exists("page-0"))
        self.assertFalse(tagged.exists("page-0"))

    def test_file_pages_are_isolated_by_persisted_dtype(self):
        """Same-sized BF16/FP16 and raw FP8 pages must never alias on disk."""
        for envelope in (False, True):
            with self.subTest(envelope=envelope), tempfile.TemporaryDirectory() as root:
                readers = []
                legacy = HiCacheFile(_config("org/model"), file_path=root)
                legacy.set("page-0", torch.zeros(64, dtype=torch.uint8))
                for index, dtype in enumerate(
                    (
                        torch.bfloat16,
                        torch.float16,
                        torch.float8_e4m3fn,
                        torch.float8_e5m2,
                        torch.float8_e4m3fnuz,
                    ),
                    start=1,
                ):
                    model_name = storage_model_name(
                        "org/model", _host_pool(dtype, envelope=envelope)
                    )
                    store = HiCacheFile(_config(model_name), file_path=root)
                    self.assertFalse(store.exists("page-0"))
                    payload = torch.full((64,), index, dtype=torch.uint8)
                    self.assertTrue(store.set("page-0", payload))
                    readers.append((model_name, payload))
                for model_name, payload in readers:
                    reader = HiCacheFile(_config(model_name), file_path=root)
                    self.assertTrue(reader.exists("page-0"))
                    result = reader.get("page-0", torch.empty_like(payload))
                    self.assertIsNotNone(result)
                    torch.testing.assert_close(result, payload)


class TestHostStorageDtype(CustomTestCase):
    def test_controller_uses_host_storage_dtype(self):
        from sglang.srt.managers import cache_controller
        from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
        from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

        controller = SimpleNamespace(
            enable_storage_metrics=False,
            # The allocation facade does not need a dtype; its storage anchor
            # owns the persisted representation.
            mem_pool_host=SimpleNamespace(layout="page_first"),
            mem_pool_device=SimpleNamespace(dtype=torch.float16),
            get_attn_cp_rank_and_size=lambda: (0, 1),
            _build_unified_plan=lambda *_: None,
        )
        parallel = SimpleNamespace(
            tp_rank=0, tp_size=1, pp_rank=0, pp_size=1, attn_dcp_size=1, attn_dcp_rank=0
        )
        cases = (
            # A typed host representation wins over the device value dtype.
            (torch.bfloat16, torch.float8_e4m3fn, "bfloat16"),
            (torch.float8_e4m3fn, torch.bfloat16, "float8_e4m3fn"),
            (torch.uint8, torch.bfloat16, "uint8"),
            # Raw FP8 bytes still require different interpretation.
            (torch.uint8, torch.float8_e4m3fn, "float8_e4m3fn"),
            (torch.uint8, torch.float8_e5m2, "float8_e5m2"),
            (torch.uint8, torch.float8_e4m3fnuz, "float8_e4m3fnuz"),
        )
        with (
            patch.object(cache_controller, "get_parallel", return_value=parallel),
            patch.object(
                cache_controller, "is_dp_attention_enabled", return_value=False
            ),
        ):
            for pool_type in (MHATokenToKVPoolHost, MLATokenToKVPoolHost):
                for buffer_dtype, value_dtype, expected in cases:
                    with self.subTest(pool_type=pool_type, expected=expected):
                        # Exercise the real host dtype contract without allocating
                        # GPU memory or host transfer buffers.
                        host_pool = pool_type.__new__(pool_type)
                        host_pool.dtype = buffer_dtype
                        host_pool._value_dtype = value_dtype
                        controller.storage_host_pool = host_pool
                        config = (
                            cache_controller.HiCacheController._generate_storage_config(
                                controller
                            )
                        )
                        self.assertEqual(config.model_name, f"dtype_{expected}")

    def test_controller_accepts_logical_storage_anchor(self):
        from sglang.srt.managers import cache_controller
        from sglang.srt.mem_cache.memory_pool_host import LogicalHostPool

        controller = SimpleNamespace(
            enable_storage_metrics=False,
            mem_pool_host=SimpleNamespace(layout="page_first"),
            mem_pool_device=SimpleNamespace(dtype=torch.float8_e4m3fn),
            storage_host_pool=LogicalHostPool(size=16, page_size=1),
            get_attn_cp_rank_and_size=lambda: (0, 1),
            _build_unified_plan=lambda *_: None,
        )
        parallel = SimpleNamespace(
            tp_rank=0, tp_size=1, pp_rank=0, pp_size=1, attn_dcp_size=1, attn_dcp_rank=0
        )
        with (
            patch.object(cache_controller, "get_parallel", return_value=parallel),
            patch.object(
                cache_controller, "is_dp_attention_enabled", return_value=False
            ),
        ):
            config = cache_controller.HiCacheController._generate_storage_config(
                controller
            )
        self.assertIsNone(config.model_name)

    def test_host_construction_preserves_buffer_value_dtype(self):
        from sglang.srt.mem_cache.memory_pool import (
            MHATokenToKVPool,
            MLATokenToKVPool,
        )
        from sglang.srt.mem_cache.pool_host.base import host_memory_budget_scope
        from sglang.srt.mem_cache.pool_host.common import (
            ALLOC_MEMORY_FUNCS,
            alloc_with_pin_memory,
        )
        from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
        from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

        for dtype in (
            torch.bfloat16,
            torch.float16,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            torch.float8_e4m3fnuz,
        ):
            for is_mla in (False, True):
                with (
                    self.subTest(dtype=dtype, is_mla=is_mla),
                    host_memory_budget_scope(1024 * 1024),
                    patch.dict(ALLOC_MEMORY_FUNCS, {"cpu": alloc_with_pin_memory}),
                ):
                    device_cls = MLATokenToKVPool if is_mla else MHATokenToKVPool
                    geometry = (
                        dict(kv_lora_rank=4, qk_rope_head_dim=2)
                        if is_mla
                        else dict(head_num=1, head_dim=4)
                    )
                    device_pool = device_cls(
                        size=8,
                        page_size=2,
                        dtype=dtype,
                        layer_num=1,
                        device="cpu",
                        enable_memory_saver=False,
                        **geometry,
                    )
                    info = device_pool.get_device_pool_infos()[0].buffer_info
                    self.assertEqual(info.value_dtype, dtype)
                    host_cls = MLATokenToKVPoolHost if is_mla else MHATokenToKVPoolHost
                    host_pool = host_cls(
                        info,
                        host_to_device_ratio=2,
                        host_size=0,
                        page_size=2,
                        layout="page_first",
                        pin_memory=False,
                        device_capacity_tokens=8,
                        start_layer=0,
                        end_layer=1,
                        host_token_capacity=16,
                        **(dict(target_layer_num=1) if is_mla else {}),
                    )
                    self.addCleanup(host_pool.destroy)
                    self.assertFalse(hasattr(host_pool, "device_pool"))
                    self.assertEqual(host_pool.get_storage_dtype(), dtype)

    def test_controller_preserves_dtype_and_dcp_shard_identity(self):
        from sglang.srt.managers import cache_controller
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
        from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

        host_pool = MLATokenToKVPoolHost.__new__(MLATokenToKVPoolHost)
        host_pool.dtype = torch.uint8
        host_pool._value_dtype = torch.float8_e5m2
        controller = SimpleNamespace(
            enable_storage_metrics=False,
            mem_pool_host=SimpleNamespace(layout="page_first"),
            mem_pool_device=MLATokenToKVPool.__new__(MLATokenToKVPool),
            storage_host_pool=host_pool,
            get_attn_cp_rank_and_size=lambda: (0, 1),
            _build_unified_plan=lambda *_: None,
            page_size=16,
        )
        parallel = SimpleNamespace(
            tp_rank=3, tp_size=4, pp_rank=0, pp_size=1, attn_dcp_size=2, attn_dcp_rank=1
        )
        with (
            patch.object(cache_controller, "get_parallel", return_value=parallel),
            patch.object(
                cache_controller, "is_dp_attention_enabled", return_value=False
            ),
        ):
            config = cache_controller.HiCacheController._generate_storage_config(
                controller
            )
        self.assertEqual(config.model_name, "dtype_float8_e5m2")
        self.assertEqual((config.dcp_size, config.dcp_rank), (2, 1))
        self.assertEqual(config.logical_page_size, 16)
        self.assertFalse(config.is_storage_writer)

    def test_unified_views_preserve_fp8_encoding_in_host_buffers(self):
        from dataclasses import replace

        from sglang.srt.mem_cache.kv.layout.mha import MHASubPoolSpec, MHAView
        from sglang.srt.mem_cache.kv.layout.mla import MLASubPoolSpec, MLAView
        from sglang.srt.mem_cache.pool_host.base import host_memory_budget_scope
        from sglang.srt.mem_cache.pool_host.common import (
            ALLOC_MEMORY_FUNCS,
            alloc_with_pin_memory,
        )
        from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
        from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
        from sglang.test.unified_allocator_fixtures import arena_of

        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz):
            for is_mla in (False, True):
                with (
                    self.subTest(dtype=dtype, is_mla=is_mla),
                    host_memory_budget_scope(1024 * 1024),
                    patch.dict(ALLOC_MEMORY_FUNCS, {"cpu": alloc_with_pin_memory}),
                ):
                    spec_cls = MLASubPoolSpec if is_mla else MHASubPoolSpec
                    spec = spec_cls(
                        name="full",
                        layer_num=1,
                        grow_direction="down",
                        store_dtype=torch.uint8,
                        **(
                            dict(kv_lora_rank=32, qk_rope_head_dim=32)
                            if is_mla
                            else dict(head_num=1, head_dim=32, kv_cache_dtype=dtype)
                        ),
                    )
                    arena = arena_of(
                        [spec, replace(spec, name="swa", grow_direction="up")],
                        total_bytes=spec.entry_bytes() * 10,
                        device="cpu",
                        page_size=2,
                    )
                    view_cls = MLAView if is_mla else MHAView
                    view = view_cls(
                        arena=arena,
                        spec=spec,
                        page_size=2,
                        **(
                            dict(kv_cache_dtype=dtype)
                            if is_mla
                            else dict(model_layer_ids=(0,), enable_alt_stream=False)
                        ),
                    )
                    info = view.get_device_pool_infos()[0].buffer_info
                    self.assertEqual(info.dtype, torch.uint8)
                    host_cls = MLATokenToKVPoolHost if is_mla else MHATokenToKVPoolHost
                    host = host_cls(
                        info,
                        host_to_device_ratio=2,
                        host_size=0,
                        page_size=2,
                        layout="page_first",
                        pin_memory=False,
                        device_capacity_tokens=view.size,
                        start_layer=0,
                        end_layer=1,
                        host_token_capacity=16,
                        **(dict(target_layer_num=1) if is_mla else {}),
                    )
                    self.addCleanup(host.destroy)
                    self.assertEqual(host.get_storage_dtype(), dtype)

    def test_page_envelope_reports_persisted_value_dtype(self):
        from sglang.srt.mem_cache.kv.layout.page import PageEnvelopeHostLayout
        from sglang.srt.mem_cache.pool_host.unified import UnifiedPageEnvelopeHostPool

        for store_dtype, value_dtype, expected in (
            (torch.bfloat16, torch.float16, torch.bfloat16),
            (torch.float8_e4m3fn, torch.bfloat16, torch.float8_e4m3fn),
            (torch.uint8, torch.float8_e4m3fn, torch.float8_e4m3fn),
            (torch.uint8, torch.float8_e5m2, torch.float8_e5m2),
        ):
            with self.subTest(store_dtype=store_dtype, value_dtype=value_dtype):
                host_pool = UnifiedPageEnvelopeHostPool.__new__(
                    UnifiedPageEnvelopeHostPool
                )
                host_pool.dtype = torch.uint8
                host_pool.device_pool = SimpleNamespace(
                    dtype=value_dtype, store_dtype=store_dtype
                )
                self.assertEqual(host_pool.get_storage_dtype(), expected)
                host_pool.device_pool = PageEnvelopeHostLayout(
                    buffer=torch.empty(1, 8, dtype=torch.uint8),
                    page_size=1,
                    capacity=1,
                    store_dtype=store_dtype,
                    value_dtype=value_dtype,
                )
                self.assertEqual(host_pool.get_storage_dtype(), expected)

    def test_page_envelope_rejects_missing_dtype_metadata(self):
        """Missing encoding metadata must not merge raw FP8 keyspaces."""
        from sglang.srt.mem_cache.kv.layout.page import PageEnvelopeHostLayout

        host_pool = UnifiedPageEnvelopeHostPool.__new__(UnifiedPageEnvelopeHostPool)
        host_pool.dtype = torch.uint8
        for store_dtype, value_dtype in (
            (None, None),
            (torch.uint8, None),
            (None, torch.float8_e4m3fn),
        ):
            with self.subTest(store_dtype=store_dtype, value_dtype=value_dtype):
                host_pool.device_pool = PageEnvelopeHostLayout(
                    buffer=torch.empty(1, 8, dtype=torch.uint8),
                    page_size=1,
                    capacity=1,
                    store_dtype=store_dtype,
                    value_dtype=value_dtype,
                )
                with self.assertRaisesRegex(ValueError, "store_dtype and value_dtype"):
                    storage_model_name("org/model", host_pool)


if __name__ == "__main__":
    unittest.main()
