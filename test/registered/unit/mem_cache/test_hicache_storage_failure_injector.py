"""Unit tests for the HiCache storage failure-injection proxy.

The proxy must delegate the full HiCacheStorage interface to the wrapped
backend, including the attach-time finalization hook: the controllers call
``storage_backend.on_host_pools_registered()`` on whatever the storage
factory returned, which is the injector whenever test env vars are set.
"""

import inspect
import unittest
from unittest.mock import MagicMock

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageFailureInjector,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestFailureInjectorDelegation(unittest.TestCase):
    def test_on_host_pools_registered_is_delegated(self):
        backend = MagicMock(spec=HiCacheStorage)
        injector = HiCacheStorageFailureInjector(backend, 0.0, 0.0)
        injector.on_host_pools_registered()
        backend.on_host_pools_registered.assert_called_once_with()

    def test_injector_overrides_every_public_storage_method(self):
        # Guards the proxy contract: any public HiCacheStorage method
        # without an override on the injector would silently hit the
        # base-class default instead of the wrapped backend (exactly the
        # on_host_pools_registered miss this file grew out of).
        base_methods = {
            name
            for name, member in inspect.getmembers(HiCacheStorage, inspect.isfunction)
            if not name.startswith("_")
        }
        injector_overrides = set(HiCacheStorageFailureInjector.__dict__)
        missing = base_methods - injector_overrides
        self.assertEqual(
            missing, set(), f"injector fails to delegate: {sorted(missing)}"
        )


if __name__ == "__main__":
    unittest.main()
