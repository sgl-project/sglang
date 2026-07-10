from array import array
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.managers.cache_controller import HiCacheController, RetentionOperation
from sglang.srt.managers.io_struct import KvHintEnvelope, KvRetentionHint
from sglang.srt.mem_cache import common
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestKvRetentionHints(CustomTestCase):
    def test_committed_prefix_is_page_aligned_and_uses_tree_hashes(self):
        backend = SimpleNamespace(
            _can_retain_groups=lambda: True,
            retention_max_hints=4,
            retention_max_pages=8192,
            retention_max_ttl_seconds=86400,
        )
        cache = object.__new__(HiRadixCache)
        cache.page_size = 2
        cache.is_eagle = False
        cache.enable_storage = True
        cache._retention_unsupported_logged = False
        cache._retention_apply_failed_logged = False
        cache._retention_bounded_logged = False
        cache.cache_controller = SimpleNamespace(
            storage_backend=backend,
            storage_backend_type="mooncake",
            write_policy="write_through",
            retain_storage=Mock(),
        )
        node = SimpleNamespace(get_prefix_hash_values=lambda _: ["page0", "page1"])
        cache.match_prefix = Mock(
            return_value=SimpleNamespace(
                device_indices=[0, 1, 2, 3], last_device_node=node
            )
        )
        req = SimpleNamespace(
            origin_input_ids=array("q", [11, 22, 33, 44, 55]),
            extra_key="tenant",
            kv_hints=KvHintEnvelope(
                retention=[KvRetentionHint(prefix_tokens=5, ttl_seconds=300)]
            ),
        )

        queued = cache.apply_kv_hints(req)

        self.assertEqual(queued, 2)
        cache.cache_controller.retain_storage.assert_called_once_with(
            ["page0", "page1"], 300
        )
        self.assertEqual(cache.apply_kv_hints(req), 0)
        cache.cache_controller.retain_storage.assert_called_once()
        matched_key = cache.match_prefix.call_args.args[0].key
        self.assertEqual(list(matched_key.token_ids), [11, 22, 33, 44])
        self.assertEqual(matched_key.extra_key, "tenant")

    def test_retention_is_bounded_across_the_request(self):
        backend = SimpleNamespace(
            _can_retain_groups=lambda: True,
            retention_max_hints=1,
            retention_max_pages=1,
            retention_max_ttl_seconds=15,
        )
        cache = object.__new__(HiRadixCache)
        cache.page_size = 2
        cache.is_eagle = False
        cache.enable_storage = True
        cache._retention_unsupported_logged = False
        cache._retention_apply_failed_logged = False
        cache._retention_bounded_logged = False
        cache.cache_controller = SimpleNamespace(
            storage_backend=backend,
            storage_backend_type="mooncake",
            write_policy="write_through",
            retain_storage=Mock(),
        )
        node = SimpleNamespace(get_prefix_hash_values=lambda _: ["page0", "page1"])
        cache.match_prefix = Mock(
            return_value=SimpleNamespace(
                device_indices=[0, 1, 2, 3], last_device_node=node
            )
        )
        req = SimpleNamespace(
            origin_input_ids=array("q", [11, 22, 33, 44]),
            extra_key=None,
            kv_hints=KvHintEnvelope(
                retention=[
                    KvRetentionHint(prefix_tokens=4, ttl_seconds=30),
                    KvRetentionHint(prefix_tokens=4, ttl_seconds=10),
                ]
            ),
        )

        self.assertEqual(cache.apply_kv_hints(req), 1)
        cache.cache_controller.retain_storage.assert_called_once_with(["page0"], 15)

    def test_controller_dispatches_retention_on_backup_worker(self):
        controller = object.__new__(HiCacheController)
        controller.backup_queue = Queue()
        controller.storage_backend = SimpleNamespace(retain_pages=Mock())
        controller.storage_stop_event = Mock()
        controller.storage_stop_event.is_set.side_effect = [False, True]

        controller.retain_storage(["page0"], 300)

        operation = controller.backup_queue.queue[0]
        self.assertIsInstance(operation, RetentionOperation)
        controller.storage_backend.retain_pages.assert_not_called()
        controller.backup_thread_func()
        controller.storage_backend.retain_pages.assert_called_once_with(["page0"], 300)

    def test_unsupported_prefix_cache_warns_once(self):
        req = SimpleNamespace(
            kv_hints=KvHintEnvelope(
                retention=[KvRetentionHint(prefix_tokens=2, ttl_seconds=300)]
            )
        )
        common._unsupported_cache_kv_hints_logged = False

        with patch.object(common.logger, "warning") as warning:
            common.maybe_apply_kv_hints(req, object())
            common.maybe_apply_kv_hints(req, object())

        warning.assert_called_once()
