"""Pageable host allocation and configuration regression tests."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.hicache_hook import validate_pageable_hicache
from sglang.srt.mem_cache.pool_host import common
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPageableHiCache(unittest.TestCase):
    def test_pageable_allocator_never_registers(self):
        with (
            patch.dict(os.environ, {"SGLANG_HICACHE_PAGEABLE": "1"}),
            patch.object(
                common, "_cuda_host_register", side_effect=AssertionError("registered")
            ),
        ):
            buffer = common.alloc_with_host_register(
                (3, 8), torch.float32, "cpu", True, common.HostTensorAllocator()
            )
            self.assertFalse(buffer.is_pinned())
            self.assertFalse(common.resolve_host_pin_memory(True))
            self.assertEqual(tuple(buffer.shape), (3, 8))

    def test_default_registration_is_preserved(self):
        with (
            patch.dict(os.environ, {"SGLANG_HICACHE_PAGEABLE": "0"}),
            patch.object(common, "_cuda_host_register") as register,
        ):
            buffer = common.alloc_with_host_register(
                (3, 8), torch.float32, "cpu", True, common.HostTensorAllocator()
            )
            register.assert_called_once_with(buffer, None)
            self.assertTrue(common.resolve_host_pin_memory(True))

    def test_attention_and_gdn_pools_keep_pageable_lifecycle(self):
        from sglang.srt.mem_cache.device_pool_info import MambaStateBufferInfo
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
        from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
        from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
        from sglang.srt.mem_cache.pool_host.qsa import QSAIndexerPoolHost

        with (
            patch.dict(os.environ, {"SGLANG_HICACHE_PAGEABLE": "1"}),
            patch.object(
                common, "_cuda_host_register", side_effect=AssertionError("registered")
            ),
            patch.object(
                common,
                "_cuda_host_unregister",
                side_effect=AssertionError("unregistered"),
            ),
        ):
            device = MHATokenToKVPool(
                size=8,
                page_size=2,
                dtype=torch.float16,
                head_num=1,
                head_dim=4,
                layer_num=2,
                device="cpu",
                enable_memory_saver=False,
            )
            host = MHATokenToKVPoolHost(
                device.get_device_pool_infos()[0].buffer_info,
                device_capacity_tokens=8,
                start_layer=0,
                end_layer=2,
                host_to_device_ratio=2,
                host_size=0,
                page_size=2,
                layout="page_first_direct",
                pin_memory=True,
            )
            gdn = MambaPoolHost(
                MambaStateBufferInfo(
                    temporal=torch.zeros((2, 5, 2, 2)),
                    conv=(torch.zeros((2, 5, 3, 2)),),
                ),
                host_to_device_ratio=2,
                host_size=0,
                device_capacity=4,
                layout="page_first_direct",
                pin_memory=True,
            )
            qsa = QSAIndexerPoolHost(
                SimpleNamespace(
                    qsa_compress_ratio=2,
                    qsa_compressed_k_buffer_pool=[
                        torch.zeros((5, 4), dtype=torch.uint8)
                    ],
                ),
                host,
            )
            try:
                for pool in (host, gdn, qsa):
                    self.assertFalse(pool.pin_memory)
            finally:
                qsa.destroy()
                gdn.destroy()
                host.destroy()

    def test_rejects_token_granular_host_access(self):
        from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool

        with patch.dict(os.environ, {"SGLANG_HICACHE_PAGEABLE": "1"}):
            with self.assertRaisesRegex(ValueError, "whole-page"):
                DeepSeekV4PagedHostPool(
                    pool_name="partial",
                    device_buffers=[],
                    item_bytes=4,
                    num_host_pages=4,
                    slot_page_size=4,
                    layout="page_first_direct",
                    device="cpu",
                    pin_memory=True,
                )

    def test_runtime_storage_attachment_is_rejected(self):
        from sglang.srt.managers.cache_controller import HiCacheController

        with patch.dict(os.environ, {"SGLANG_HICACHE_PAGEABLE": "1"}):
            with self.assertRaisesRegex(ValueError, "L3 storage"):
                HiCacheController.attach_storage_backend(object(), "file")

    def test_hisparse_is_rejected_before_host_pool_construction(self):
        from sglang.srt.arg_groups.hicache_hook import handle_hicache

        with (
            patch.dict(os.environ, {"SGLANG_HICACHE_PAGEABLE": "1"}),
            patch(
                "sglang.srt.arg_groups.hicache_hook.resolving_view",
                side_effect=lambda x: x,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "HiSparse"):
                handle_hicache(SimpleNamespace(enable_hisparse=True))

    def test_configuration_guards(self):
        config = dict(
            device="cuda",
            hicache_io_backend="direct",
            hicache_mem_layout="page_first_direct",
            hicache_storage_backend=None,
            enable_unified_memory=False,
        )
        with (
            patch.dict(os.environ, {"SGLANG_HICACHE_PAGEABLE": "1"}),
            patch(
                "sglang.srt.arg_groups.hicache_hook.resolving_view",
                side_effect=lambda x: x,
            ),
        ):
            validate_pageable_hicache(SimpleNamespace(**config))
            for key, value in [
                ("device", "cpu"),
                ("hicache_io_backend", "kernel"),
                ("hicache_mem_layout", "page_first"),
                ("hicache_storage_backend", "mooncake"),
                ("enable_unified_memory", True),
            ]:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_pageable_hicache(
                        SimpleNamespace(**(config | {key: value}))
                    )


if __name__ == "__main__":
    unittest.main()
