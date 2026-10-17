"""Storage keys separate host pools with incompatible page byte formats.

CPU-only.

    python -m pytest test/registered/unit/mem_cache/test_hicache_storage_format_tag.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import shutil
import tempfile
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.cache_controller import storage_model_name
from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import LogicalHostPool
from sglang.srt.mem_cache.pool_host.base import HostKVCache
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.unified import UnifiedPageEnvelopeHostPool
from sglang.test.test_utils import CustomTestCase


def _config(model_name, *, mla=False):
    return HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=mla,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name=model_name,
    )


class TestStorageModelName(CustomTestCase):
    def test_default_pools_keep_the_model_name(self):
        self.assertIsNone(HostKVCache.storage_format_tag)
        pool = SimpleNamespace(storage_format_tag=None)
        self.assertEqual(storage_model_name("org/model", pool), "org/model")
        self.assertIsNone(storage_model_name(None, pool))

    def test_logical_anchor_pool_keeps_the_model_name(self):
        """A LogicalHostPool storage anchor keeps the untagged model name."""
        pool = LogicalHostPool(size=4, page_size=2)
        self.assertEqual(storage_model_name("org/model", pool), "org/model")

    def test_unified_pool_tags_the_model_name(self):
        tag = UnifiedPageEnvelopeHostPool.storage_format_tag
        self.assertIsNotNone(tag)
        self.assertEqual(
            storage_model_name("org/model", UnifiedPageEnvelopeHostPool),
            f"org/model-{tag}",
        )
        self.assertEqual(storage_model_name(None, UnifiedPageEnvelopeHostPool), tag)

    def test_untagged_page_misses_under_the_unified_name(self):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        untagged = HiCacheFile(_config("org/model"), file_path=root)
        tagged = HiCacheFile(
            _config(storage_model_name("org/model", UnifiedPageEnvelopeHostPool)),
            file_path=root,
        )
        self.assertTrue(untagged.set("page-0", torch.zeros(64, dtype=torch.uint8)))
        self.assertTrue(untagged.exists("page-0"))
        self.assertFalse(tagged.exists("page-0"))


class TestDenseStoragePageFormats(CustomTestCase):
    def _host_pool(self, mla, layout):
        device_type = MLATokenToKVPool if mla else MHATokenToKVPool
        host_type = MLATokenToKVPoolHost if mla else MHATokenToKVPoolHost
        geometry = (
            dict(kv_lora_rank=2, qk_rope_head_dim=2)
            if mla
            else dict(head_num=2, head_dim=4)
        )
        device = device_type(
            size=4,
            page_size=2,
            dtype=torch.float16,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
            **geometry,
        )
        host = host_type(
            device,
            host_to_device_ratio=2,
            host_size=0,
            page_size=2,
            layout=layout,
            pin_memory=False,
        )
        self.addCleanup(host.destroy)
        return host

    def test_file_restore_requires_compatible_page_format(self):
        for mla in (False, True):
            layouts = ["layer_first", "page_first_direct", "page_first"]
            if not mla:
                layouts.append("page_head")
            hosts = {layout: self._host_pool(mla, layout) for layout in layouts}
            for source_layout, source in hosts.items():
                with tempfile.TemporaryDirectory() as directory:
                    writer = HiCacheFile(
                        _config(storage_model_name("org/model", source), mla=mla),
                        file_path=directory,
                    )
                    payload = source.get_dummy_flat_data_page()
                    payload.copy_(torch.arange(payload.numel()).reshape(payload.shape))
                    source.set_from_flat_data_page(0, payload)
                    self.assertTrue(writer.set("page-0", source.get_data_page(0)))
                    for target_layout, target in hosts.items():
                        with self.subTest(
                            mla=mla, source=source_layout, target=target_layout
                        ):
                            reader = HiCacheFile(
                                _config(
                                    storage_model_name("org/model", target), mla=mla
                                ),
                                file_path=directory,
                            )
                            compatible = source_layout == target_layout or {
                                source_layout,
                                target_layout,
                            } == {"layer_first", "page_first_direct"}
                            self.assertEqual(reader.exists("page-0"), compatible)
                            restored = reader.get(
                                "page-0", target.get_dummy_flat_data_page()
                            )
                            if compatible:
                                self.assertIsNotNone(restored)
                                target.set_from_flat_data_page(0, restored)
                                torch.testing.assert_close(
                                    target.get_data_page(0), payload, rtol=0, atol=0
                                )
                            else:
                                self.assertIsNone(restored)

    def test_untagged_files_are_not_restored(self):
        for mla in (False, True):
            with self.subTest(mla=mla), tempfile.TemporaryDirectory() as directory:
                host = self._host_pool(mla, "layer_first")
                legacy = HiCacheFile(_config("org/model", mla=mla), file_path=directory)
                self.assertTrue(legacy.set("page-0", host.get_dummy_flat_data_page()))
                tagged = HiCacheFile(
                    _config(storage_model_name("org/model", host), mla=mla),
                    file_path=directory,
                )
                self.assertFalse(tagged.exists("page-0"))


if __name__ == "__main__":
    unittest.main()
