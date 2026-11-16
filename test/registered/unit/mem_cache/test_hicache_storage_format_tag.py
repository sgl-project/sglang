"""Page format and resolved logical KV dtype namespace storage keys.

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

from sglang.srt.managers import cache_controller
from sglang.srt.managers.cache_controller import storage_model_name
from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.srt.mem_cache.memory_pool_host import LogicalHostPool
from sglang.srt.mem_cache.pool_host.base import HostKVCache
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


class TestStorageModelName(CustomTestCase):
    def test_default_pools_tag_the_model_name_with_dtype(self):
        self.assertIsNone(HostKVCache.storage_format_tag)
        pool = SimpleNamespace(storage_format_tag=None)
        self.assertEqual(
            storage_model_name("org/model", pool, torch.bfloat16),
            "org/model-dtype_bfloat16",
        )
        self.assertEqual(
            storage_model_name(None, pool, torch.bfloat16), "dtype_bfloat16"
        )

    def test_logical_anchor_pool_uses_the_server_dtype(self):
        """A logical anchor has no physical dtype but still needs isolated keys."""
        pool = LogicalHostPool(size=4, page_size=2)
        self.assertEqual(
            storage_model_name("org/model", pool, torch.bfloat16),
            "org/model-dtype_bfloat16",
        )

    def test_unified_pool_tags_the_model_name(self):
        tag = UnifiedPageEnvelopeHostPool.storage_format_tag
        self.assertIsNotNone(tag)
        self.assertEqual(
            storage_model_name(
                "org/model", UnifiedPageEnvelopeHostPool, torch.bfloat16
            ),
            f"org/model-{tag}-dtype_bfloat16",
        )
        self.assertEqual(
            storage_model_name(None, UnifiedPageEnvelopeHostPool, torch.bfloat16),
            f"{tag}-dtype_bfloat16",
        )

    def test_untagged_page_misses_under_the_unified_name(self):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        untagged = HiCacheFile(
            _config(storage_model_name("org/model", HostKVCache, torch.bfloat16)),
            file_path=root,
        )
        tagged = HiCacheFile(
            _config(
                storage_model_name(
                    "org/model", UnifiedPageEnvelopeHostPool, torch.bfloat16
                )
            ),
            file_path=root,
        )
        self.assertTrue(untagged.set("page-0", torch.zeros(64, dtype=torch.uint8)))
        self.assertTrue(untagged.exists("page-0"))
        self.assertFalse(tagged.exists("page-0"))

    def test_file_pages_are_isolated_by_logical_dtype(self):
        """Same-sized BF16/FP16 and raw FP8 pages must never alias on disk."""
        for host_pool in (HostKVCache, UnifiedPageEnvelopeHostPool):
            with (
                self.subTest(host_pool=host_pool),
                tempfile.TemporaryDirectory() as root,
            ):
                readers = []
                legacy = HiCacheFile(_config("org/model"), file_path=root)
                legacy.set("page-0", torch.zeros(64, dtype=torch.uint8))
                for index, dtype in enumerate(
                    (
                        torch.bfloat16,
                        torch.float16,
                        torch.float8_e4m3fn,
                        torch.float8_e5m2,
                    ),
                    start=1,
                ):
                    model_name = storage_model_name("org/model", host_pool, dtype)
                    store = HiCacheFile(_config(model_name), file_path=root)
                    self.assertFalse(store.exists("page-0"))
                    payload = torch.full((64,), index, dtype=torch.uint8)
                    self.assertTrue(store.set("page-0", payload))
                    readers.append((dtype, payload))
                # Fresh readers recompute the name and restore persisted bytes.
                for dtype, payload in readers:
                    reader = HiCacheFile(
                        _config(storage_model_name("org/model", host_pool, dtype)),
                        file_path=root,
                    )
                    self.assertTrue(reader.exists("page-0"))
                    result = reader.get("page-0", torch.empty_like(payload))
                    self.assertIsNotNone(result)
                    torch.testing.assert_close(result, payload)

    def test_controller_uses_resolved_device_dtype(self):
        """Host/FP8 backing bytes must not replace the logical dtype in keys."""
        controller = SimpleNamespace(
            enable_storage_metrics=False,
            mem_pool_host=SimpleNamespace(layout="page_first"),
            storage_host_pool=LogicalHostPool(size=4, page_size=2),
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
            for routed in (False, True):
                for dtype in (
                    torch.bfloat16,
                    torch.float16,
                    torch.float8_e4m3fn,
                    torch.float8_e5m2,
                    None,
                    "auto",
                ):
                    with self.subTest(routed=routed, dtype=dtype):
                        view = SimpleNamespace(store_dtype=torch.uint8)
                        if dtype is not None:
                            view.dtype = dtype
                        controller.mem_pool_device = (
                            SimpleNamespace(
                                layer_router=SimpleNamespace(
                                    sub_pools=lambda: [SimpleNamespace(name="kv")],
                                    view=lambda _: view,
                                )
                            )
                            if routed
                            else view
                        )
                        if dtype is None or dtype == "auto":
                            with self.assertRaisesRegex(ValueError, "resolved logical"):
                                cache_controller.HiCacheController._generate_storage_config(
                                    controller, "org/model"
                                )
                        else:
                            config = cache_controller.HiCacheController._generate_storage_config(
                                controller, "org/model"
                            )
                            self.assertEqual(
                                config.model_name,
                                storage_model_name("org/model", HostKVCache, dtype),
                            )


if __name__ == "__main__":
    unittest.main()
