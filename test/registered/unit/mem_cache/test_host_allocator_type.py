"""The host-allocator kind a config asks for, before any pool exists."""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.mem_cache.pool_host import common as pool_host_common
from sglang.srt.mem_cache.pool_host.common import (
    allocator_type_of,
    get_allocator_type,
    host_allocator_owns_memory,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_OWNED_TYPES = ("shm", "mooncake", "mori")


def _cfg(storage_backend, extra_config=None):
    return SimpleNamespace(
        hicache_storage_backend=storage_backend,
        hicache_storage_backend_extra_config=extra_config,
    )


class TestHostAllocatorType(unittest.TestCase):
    # Every --hicache-storage-backend choice, plus the unset default.
    STORAGE_BACKEND_TYPES = {
        None: "default",
        "file": "file",
        "sim": "sim",
        "mooncake": "mooncake",
        "hf3fs": "hf3fs",
        "nixl": "nixl",
        "aibrix": "aibrix",
        "dynamic": "dynamic",
        "eic": "eic",
        "simm": "simm",
        "mori": "mori",
        "shm": "shm",
    }

    def test_each_storage_backend_names_its_allocator_type(self):
        for backend, expected in self.STORAGE_BACKEND_TYPES.items():
            with self.subTest(backend=backend):
                self.assertEqual(allocator_type_of(_cfg(backend)), expected)

    def test_exactly_shm_mooncake_and_mori_own_their_host_memory(self):
        for backend, allocator_type in self.STORAGE_BACKEND_TYPES.items():
            with self.subTest(backend=backend):
                self.assertEqual(
                    host_allocator_owns_memory(_cfg(backend)),
                    allocator_type in _OWNED_TYPES,
                )

    def test_a_dynamic_backend_takes_its_allocator_from_extra_config(self):
        self.assertEqual(
            allocator_type_of(_cfg("dynamic", '{"allocator": "shm"}')), "shm"
        )
        self.assertTrue(
            host_allocator_owns_memory(_cfg("dynamic", '{"allocator": "shm"}'))
        )

    def test_an_unreadable_extra_config_leaves_the_backend_name(self):
        for extra_config in ("", "not json", "[]", '{"allocator": "file"}'):
            with self.subTest(extra_config=extra_config):
                self.assertEqual(
                    allocator_type_of(_cfg("dynamic", extra_config)), "dynamic"
                )
                self.assertFalse(
                    host_allocator_owns_memory(_cfg("dynamic", extra_config))
                )

    def test_extra_config_only_reroutes_the_dynamic_backend(self):
        self.assertEqual(
            allocator_type_of(_cfg("file", '{"allocator": "shm"}')), "file"
        )

    def test_the_published_allocator_type_reads_the_memory_bag(self):
        with mock.patch.object(
            pool_host_common, "get_memory", return_value=_cfg("shm")
        ):
            self.assertEqual(get_allocator_type(), "shm")


if __name__ == "__main__":
    unittest.main()
