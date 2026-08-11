import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.mem_cache.memory_pool_host as memory_pool_host
from sglang.srt.mem_cache.memory_pool_host import (
    MambaPoolHost,
    _ascend_hicache_mamba_io_mode,
)


class TestAscendMambaAsyncConfig(unittest.TestCase):
    def test_accepts_explicit_sync_and_async_modes(self):
        for mode in ("sync", "async"):
            with self.subTest(mode=mode), patch.dict(
                os.environ, {"SGLANG_ASCEND_HICACHE_MAMBA_IO": mode}
            ):
                self.assertEqual(_ascend_hicache_mamba_io_mode(), mode)

    def test_rejects_auto_mode(self):
        with patch.dict(
            os.environ, {"SGLANG_ASCEND_HICACHE_MAMBA_IO": "auto"}
        ), self.assertRaisesRegex(ValueError, "must be one of"):
            _ascend_hicache_mamba_io_mode()

    @patch.object(memory_pool_host, "_is_npu", True)
    @patch.object(
        memory_pool_host,
        "_ascend_hicache_mamba_io_mode",
        return_value="async",
    )
    @patch.object(memory_pool_host, "transfer_state_dim_exchange", None)
    def test_async_requires_native_operator(self, _mock_mode):
        pool = MambaPoolHost.__new__(MambaPoolHost)

        with self.assertRaisesRegex(
            RuntimeError, "transfer_state_dim_exchange from sgl-kernel-npu"
        ):
            pool._configure_ascend_mamba_io()

    def test_conv_only_components_skip_empty_temporal_state(self):
        pool = MambaPoolHost.__new__(MambaPoolHost)
        pool.temporal_state_elem_size = 0
        pool.temporal_buffer = torch.empty(4, 3, 1, 0)
        pool.conv_buffer = [torch.empty(4, 3, 1, 2, 4)]
        device_pool = SimpleNamespace(
            mamba_cache=SimpleNamespace(
                temporal=torch.empty(3, 4, 0),
                conv=[torch.empty(3, 4, 2, 4)],
            )
        )

        device_states, host_states = pool._state_components(device_pool)

        self.assertEqual(len(device_states), 1)
        self.assertEqual(len(host_states), 1)
        self.assertIs(device_states[0], device_pool.mamba_cache.conv[0])
        self.assertIs(host_states[0], pool.conv_buffer[0])


if __name__ == "__main__":
    unittest.main()
