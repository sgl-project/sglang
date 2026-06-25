import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_register_cpu_ci():
    repo_root = Path(__file__).resolve().parents[4]
    module_path = repo_root / "python/sglang/test/ci/ci_register.py"
    spec = importlib.util.spec_from_file_location("_ci_register", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.register_cpu_ci


register_cpu_ci = _load_register_cpu_ci()
register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _load_base_module():
    repo_root = Path(__file__).resolve().parents[4]
    module_path = repo_root / "python/sglang/srt/mem_cache/pool_host/base.py"

    stub_modules = {
        "psutil": types.ModuleType("psutil"),
        "torch": types.ModuleType("torch"),
        "sglang": types.ModuleType("sglang"),
        "sglang.srt": types.ModuleType("sglang.srt"),
        "sglang.srt.environ": types.ModuleType("sglang.srt.environ"),
        "sglang.srt.mem_cache": types.ModuleType("sglang.srt.mem_cache"),
        "sglang.srt.mem_cache.memory_pool": types.ModuleType(
            "sglang.srt.mem_cache.memory_pool"
        ),
        "sglang.srt.mem_cache.pool_host": types.ModuleType(
            "sglang.srt.mem_cache.pool_host"
        ),
        "sglang.srt.mem_cache.pool_host.common": types.ModuleType(
            "sglang.srt.mem_cache.pool_host.common"
        ),
        "sglang.srt.utils": types.ModuleType("sglang.srt.utils"),
    }
    stub_modules["torch"].distributed = types.SimpleNamespace(
        is_available=lambda: False,
        is_initialized=lambda: False,
    )
    stub_modules["psutil"].virtual_memory = lambda: types.SimpleNamespace(available=0)
    stub_modules["sglang.srt.environ"].envs = types.SimpleNamespace(
        SGLANG_HICACHE_LOCAL_PROCESS_COUNT=types.SimpleNamespace(get=lambda: 0)
    )
    stub_modules["sglang.srt.mem_cache.memory_pool"].KVCache = object
    stub_modules["sglang.srt.mem_cache.pool_host.common"]._cuda_host_unregister = (
        lambda _: None
    )
    stub_modules["sglang.srt.mem_cache.pool_host.common"].get_allocator_from_storage = (
        lambda _: None
    )
    stub_modules["sglang.srt.utils"].is_cuda = lambda: False
    stub_modules["sglang.srt.utils"].is_hip = lambda: False

    previous_modules = {name: sys.modules.get(name) for name in stub_modules}
    try:
        sys.modules.update(stub_modules)
        spec = importlib.util.spec_from_file_location(
            "_hicache_host_base_under_test",
            module_path,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous_module in previous_modules.items():
            if previous_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module


base = _load_base_module()


class TestHiCacheHostMemoryGuard(unittest.TestCase):
    def test_get_local_hicache_process_count_uses_hicache_env(self):
        with patch.object(
            base.envs.SGLANG_HICACHE_LOCAL_PROCESS_COUNT,
            "get",
            return_value=4,
        ):
            self.assertEqual(base.get_local_hicache_process_count(), 4)

    def test_validate_hicache_host_memory_accounts_for_local_processes(self):
        requested_bytes = 20 * 10**9
        available_bytes = 100 * 10**9

        with (
            patch.object(base, "get_local_hicache_process_count", return_value=4),
            patch.object(
                base.psutil,
                "virtual_memory",
                return_value=types.SimpleNamespace(
                    available=available_bytes + base.HICACHE_HOST_MEMORY_RESERVE_BYTES
                ),
            ),
        ):
            actual_available, aggregate_requested, local_process_count = (
                base.validate_hicache_host_memory(
                    requested_bytes,
                    description="test pool",
                )
            )

        self.assertEqual(actual_available, available_bytes)
        self.assertEqual(aggregate_requested, 80 * 10**9)
        self.assertEqual(local_process_count, 4)

    def test_get_local_hicache_process_count_uses_local_size_env(self):
        with patch.dict(base.os.environ, {"LOCAL_SIZE": "4"}):
            self.assertEqual(base.get_local_hicache_process_count(), 4)

    def test_validate_hicache_host_memory_rejects_aggregate_overcommit(self):
        requested_bytes = 30 * 10**9
        available_bytes = 100 * 10**9

        with (
            patch.object(base, "get_local_hicache_process_count", return_value=4),
            patch.object(
                base.psutil,
                "virtual_memory",
                return_value=types.SimpleNamespace(
                    available=available_bytes + base.HICACHE_HOST_MEMORY_RESERVE_BYTES
                ),
            ),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "120.00 GB total.*100.00 GB free",
            ):
                base.validate_hicache_host_memory(
                    requested_bytes,
                    description="test pool",
                )


if __name__ == "__main__":
    unittest.main()
