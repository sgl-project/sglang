"""BaseHostCache.destroy must pair the cudaHostRegister done at allocation."""

import unittest
from unittest import mock

import torch

from sglang.srt.state_capturer import base as base_mod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHostCacheDestroy(CustomTestCase):
    def _host_cache(self):
        cache = object.__new__(base_mod.BaseHostCache)
        cache.buffer = torch.zeros((4, 2, 3), dtype=torch.int32)
        cache.name = "test"
        return cache

    def test_unregisters_once_and_is_idempotent(self):
        cache = self._host_cache()
        buffer = cache.buffer
        with (
            mock.patch.object(base_mod, "_is_cuda", True),
            mock.patch.object(base_mod, "_cuda_host_unregister") as m_unreg,
        ):
            cache.destroy()
            cache.destroy()
        m_unreg.assert_called_once_with(buffer)
        self.assertIsNone(cache.buffer)

    def test_capturer_destroy_reaches_host_cache(self):
        capturer = object.__new__(base_mod.BaseTopkCapturer)
        capturer.host_cache = self._host_cache()
        with (
            mock.patch.object(base_mod, "_is_cuda", True),
            mock.patch.object(base_mod, "_cuda_host_unregister") as m_unreg,
        ):
            capturer.destroy()
        m_unreg.assert_called_once()


if __name__ == "__main__":
    unittest.main()
