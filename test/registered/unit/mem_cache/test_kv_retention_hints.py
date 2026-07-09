from array import array
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.managers.io_struct import KvHintEnvelope, KvRetentionHint
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestKvRetentionHints(CustomTestCase):
    def test_committed_prefix_is_page_aligned_and_uses_tree_hashes(self):
        backend = SimpleNamespace(
            _can_use_group_semantics=lambda: True,
            retain_pages=Mock(return_value=2),
        )
        cache = object.__new__(HiRadixCache)
        cache.page_size = 2
        cache.is_eagle = False
        cache.enable_storage = True
        cache._retention_unsupported_logged = False
        cache.cache_controller = SimpleNamespace(
            storage_backend=backend,
            storage_backend_type="mooncake",
            write_policy="write_through",
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

        accepted = cache.apply_kv_hints(req)

        self.assertEqual(accepted, 2)
        backend.retain_pages.assert_called_once_with(["page0", "page1"], 300)
        matched_key = cache.match_prefix.call_args.args[0].key
        self.assertEqual(list(matched_key.token_ids), [11, 22, 33, 44])
        self.assertEqual(matched_key.extra_key, "tenant")
