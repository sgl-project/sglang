import types
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.decode_kvcache_offload_manager import (  # noqa: E402
    DecodeKVCacheOffloadManager,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDecodeKVCacheOffloadManager(unittest.TestCase):
    def test_plain_controller_backup_uses_base_write_signature(self):
        class PlainController:
            storage_page_size = 2

            def get_hash_str(self, tokens, prior):
                return str(tokens)

            def write_storage(
                self, host_indices, token_ids, hash_value=None, prefix_keys=None
            ):
                self.write_args = (host_indices, token_ids)
                self.write_kwargs = {
                    "hash_value": hash_value,
                    "prefix_keys": prefix_keys,
                }
                return 7

        manager = object.__new__(DecodeKVCacheOffloadManager)
        manager.is_hybrid_mamba = False
        manager.page_size = 4
        manager.ongoing_backup = {}
        manager.cache_controller = PlainController()
        req = types.SimpleNamespace(rid="request-1")

        manager._trigger_backup(
            req=req,
            host_indices="host-indices",
            incremental_tokens=[1, 2, 3, 4],
            start_time=1.0,
            prior_hash=None,
            storage_prior_hash=None,
        )

        self.assertEqual(
            manager.cache_controller.write_args,
            ("host-indices", [1, 2, 3, 4]),
        )
        self.assertEqual(
            manager.cache_controller.write_kwargs["hash_value"],
            ["[1, 2, 3, 4]"],
        )
        self.assertEqual(
            manager.ongoing_backup[7],
            ("request-1", "host-indices", None, 1.0),
        )


if __name__ == "__main__":
    unittest.main()
