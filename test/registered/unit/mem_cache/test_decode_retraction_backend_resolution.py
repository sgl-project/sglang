import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.mem_cache import kv_cache_builder
from sglang.srt.runtime_context import (
    get_context,
    get_disagg,
    get_memory,
    get_parallel,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeMHATokenToKVPool:
    pass


class TestDecodeRetractionBackendResolution(unittest.TestCase):
    @staticmethod
    def _make_mha_worker():
        kv_cache = _FakeMHATokenToKVPool()
        allocator = SimpleNamespace(get_kvcache=lambda: kv_cache)
        return SimpleNamespace(
            is_hybrid_swa=False,
            model_runner=SimpleNamespace(model_config=object()),
            get_memory_pool=lambda: (None, allocator),
        )

    def _resolve(self, *, is_hip: bool, configured_backend=None):
        override = get_context().override_server_args(
            disaggregation_mode="decode",
            disaggregation_decode_retraction_backup=configured_backend,
            disaggregation_decode_enable_radix_cache=False,
            disaggregation_decode_enable_offload_kvcache=False,
            enable_priority_scheduling=False,
            disable_priority_preemption=False,
            hicache_ratio=None,
            enable_hierarchical_cache=False,
        )
        with override, get_parallel().override(dcp_enabled=False), patch.object(
            kv_cache_builder, "is_hip", return_value=is_hip
        ), patch.object(
            kv_cache_builder, "MHATokenToKVPool", _FakeMHATokenToKVPool
        ), patch.object(
            kv_cache_builder, "uses_ssm_state", return_value=False
        ):
            backend = kv_cache_builder.resolve_decode_retraction_backup(
                tp_worker=self._make_mha_worker()
            )
            return (
                backend,
                get_disagg().disaggregation_decode_retraction_backup,
                get_memory().hicache_ratio,
            )

    def test_rocm_auto_backend_falls_back_to_cpu_tensor(self):
        self.assertEqual(
            self._resolve(is_hip=True),
            ("cpu_tensor", "cpu_tensor", 2.0),
        )

    def test_non_rocm_auto_backend_keeps_host_pool(self):
        self.assertEqual(
            self._resolve(is_hip=False),
            (
                "host_pool",
                "host_pool",
                kv_cache_builder.BACKUP_ONLY_HICACHE_RATIO,
            ),
        )

    def test_rocm_explicit_host_pool_is_preserved(self):
        self.assertEqual(
            self._resolve(is_hip=True, configured_backend="host_pool"),
            (
                "host_pool",
                "host_pool",
                kv_cache_builder.BACKUP_ONLY_HICACHE_RATIO,
            ),
        )


if __name__ == "__main__":
    unittest.main()
