"""Direct CPU regression for same-backend ``StorageAttachment.attach`` updates.

This complements the full HTTP lifecycle case without starting a model or server.
"""

import json
import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.mem_cache.unified_cache.storage_attachment import StorageAttachment

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestHiCacheStorageAttachment(CustomTestCase):
    def _make_attachment(self):
        controller = SimpleNamespace(
            storage_backend_type="file",
            write_policy="write_back",
        )
        cache = SimpleNamespace(
            enable_storage=True,
            cache_controller=controller,
            host_memory_mode="cache",
            token_to_kv_pool_allocator=SimpleNamespace(),
            is_write_back=True,
            tree_core=SimpleNamespace(is_write_through_compatible=lambda: True),
            prefetch_stop_policy="timeout",
            write_through_threshold=2,
            prefetch_threshold=256,
            prefetch_timeout_base=3.0,
            prefetch_timeout_per_page=0.01,
            hicache_storage_pass_prefix_keys=True,
        )
        return StorageAttachment(cache), cache, controller

    def test_malformed_same_backend_update_preserves_state(self):
        attachment, cache, controller = self._make_attachment()
        cache_state = vars(cache).copy()
        controller_state = vars(controller).copy()

        attached, message = attachment.attach(
            storage_backend="file",
            storage_backend_extra_config_json="{",
            hicache_storage_prefetch_policy="wait_complete",
            hicache_write_policy="write_through_selective",
        )

        self.assertFalse(attached)
        self.assertEqual(
            message,
            "Failed to parse storage_backend_extra_config_json '{': "
            "Expecting property name enclosed in double quotes: "
            "line 1 column 2 (char 1)",
        )
        self.assertEqual(vars(cache), cache_state)
        self.assertEqual(vars(controller), controller_state)

        # A different backend is rejected before attempting to parse its config.
        attached, message = attachment.attach(
            storage_backend="mooncake",
            storage_backend_extra_config_json="{",
            hicache_storage_prefetch_policy="wait_complete",
            hicache_write_policy="write_through_selective",
        )
        self.assertFalse(attached)
        self.assertEqual(
            message,
            "HiCache storage backend is already enabled with backend 'file'. "
            "Cannot attach different backend 'mooncake'. Detach first.",
        )
        self.assertEqual(vars(cache), cache_state)
        self.assertEqual(vars(controller), controller_state)

    def test_valid_same_backend_update_preserves_config(self):
        attachment, cache, controller = self._make_attachment()
        original_config = {
            "enable_storage": cache.enable_storage,
            "prefetch_threshold": cache.prefetch_threshold,
            "prefetch_timeout_base": cache.prefetch_timeout_base,
            "prefetch_timeout_per_page": cache.prefetch_timeout_per_page,
            "hicache_storage_pass_prefix_keys": (
                cache.hicache_storage_pass_prefix_keys
            ),
        }
        new_extra_config = json.dumps(
            {
                "prefetch_threshold": 512,
                "prefetch_timeout_base": 4,
                "prefetch_timeout_per_ki_token": 0.02,
                "hicache_storage_pass_prefix_keys": False,
            }
        )

        attached, message = attachment.attach(
            storage_backend="file",
            storage_backend_extra_config_json=new_extra_config,
            hicache_storage_prefetch_policy="wait_complete",
            hicache_write_policy="write_through_selective",
        )

        self.assertTrue(attached)
        self.assertEqual(
            message,
            "HiCache storage backend already enabled with same backend; "
            "policies updated.",
        )
        self.assertEqual(cache.prefetch_stop_policy, "wait_complete")
        self.assertEqual(controller.write_policy, "write_through_selective")
        self.assertEqual(cache.write_through_threshold, 2)
        self.assertFalse(cache.is_write_back)
        self.assertEqual(controller.storage_backend_type, "file")
        self.assertEqual(
            {
                "enable_storage": cache.enable_storage,
                "prefetch_threshold": cache.prefetch_threshold,
                "prefetch_timeout_base": cache.prefetch_timeout_base,
                "prefetch_timeout_per_page": cache.prefetch_timeout_per_page,
                "hicache_storage_pass_prefix_keys": (
                    cache.hicache_storage_pass_prefix_keys
                ),
            },
            original_config,
        )


if __name__ == "__main__":
    unittest.main()
