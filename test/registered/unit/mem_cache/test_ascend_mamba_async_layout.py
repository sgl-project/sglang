import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
    @patch.object(
        memory_pool_host, "transfer_state_per_layer_direct_pf_lf", None
    )
    @patch.object(
        memory_pool_host, "transfer_state_all_layer_direct_lf_pf", None
    )
    def test_async_requires_native_operator(self, _mock_mode):
        pool = MambaPoolHost.__new__(MambaPoolHost)

        with self.assertRaisesRegex(
            RuntimeError, "per-layer PF->LF and all-layer LF->PF"
        ):
            pool._configure_ascend_mamba_io()

    def test_async_h2d_is_dispatched_per_component_from_copy_helper(self):
        pool = MambaPoolHost.__new__(MambaPoolHost)
        host = torch.empty(4, 3, 1, 2, 4)
        device_layers = torch.empty(3, 5, 2, 4)
        host_indices = torch.tensor([1, 3])
        device_indices = torch.tensor([0, 4])

        transfer_op = MagicMock()
        with patch.object(
            memory_pool_host,
            "transfer_state_per_layer_direct_pf_lf",
            transfer_op,
        ), patch.object(
            memory_pool_host,
            "_ascend_hicache_mamba_io_mode",
            return_value="async",
        ):
            pool._copy_tensor_pf_lf(
                src=host,
                dst=device_layers[2],
                src_indices=host_indices,
                dst_indices=device_indices,
                layer_id=2,
                num_layers=3,
                io_backend="kernel_ascend",
            )

        transfer_op.assert_called_once_with(
            src=host,
            dst=device_layers[2],
            src_indices=host_indices,
            dst_indices=device_indices,
            layer_id=2,
        )

    def test_async_d2h_is_dispatched_per_component_from_copy_helper(self):
        pool = MambaPoolHost.__new__(MambaPoolHost)
        device_layers = torch.empty(3, 5, 2, 4)
        host = torch.empty(4, 3, 1, 2, 4)
        device_indices = torch.tensor([0, 4])
        host_indices = torch.tensor([1, 3])

        transfer_op = MagicMock()
        with patch.object(
            memory_pool_host,
            "transfer_state_all_layer_direct_lf_pf",
            transfer_op,
        ), patch.object(
            memory_pool_host,
            "_ascend_hicache_mamba_io_mode",
            return_value="async",
        ):
            pool._copy_tensor_all_layers_lf_pf(
                src_layers=device_layers,
                dst=host,
                src_indices=device_indices,
                dst_indices=host_indices,
                num_layers=3,
                io_backend="kernel_ascend",
                src_ptrs=torch.empty(0),
            )

        transfer_op.assert_called_once_with(
            device_states=[device_layers],
            host_states=[host],
            device_indices=device_indices,
            host_indices=host_indices,
        )

    def test_sync_h2d_torch_fallback_is_preserved(self):
        pool = MambaPoolHost.__new__(MambaPoolHost)
        host = torch.arange(4 * 3 * 1 * 2, dtype=torch.float32).reshape(4, 3, 1, 2)
        device_layers = torch.zeros(3, 5, 2)
        host_indices = torch.tensor([1, 3])
        device_indices = torch.tensor([0, 4])

        with patch.object(
            memory_pool_host,
            "_ascend_hicache_mamba_io_mode",
            return_value="sync",
        ):
            pool._copy_tensor_pf_lf(
                src=host,
                dst=device_layers[2],
                src_indices=host_indices,
                dst_indices=device_indices,
                layer_id=2,
                num_layers=3,
                io_backend="kernel_ascend",
            )

        torch.testing.assert_close(
            device_layers[2, device_indices], host[host_indices, 2, 0]
        )

    def test_copy_helpers_keep_static_signatures(self):
        self.assertIsInstance(
            MambaPoolHost.__dict__["_copy_tensor_pf_lf"], staticmethod
        )
        self.assertIsInstance(
            MambaPoolHost.__dict__["_copy_tensor_all_layers_lf_pf"], staticmethod
        )

    def test_conv_only_load_skips_empty_temporal_component(self):
        pool = MambaPoolHost.__new__(MambaPoolHost)
        pool.layout = "page_first_direct"
        pool.temporal_state_elem_size = 0
        pool.num_mamba_layers = 3
        pool.temporal_buffer = torch.empty(4, 3, 1, 0)
        pool.conv_state_shapes = [torch.Size([2, 4])]
        pool.conv_buffer = [torch.empty(4, 3, 1, 2, 4)]
        pool._copy_tensor_pf_lf = MagicMock()
        device_pool = SimpleNamespace(
            mamba_cache=SimpleNamespace(
                temporal=torch.empty(3, 4, 0),
                conv=[torch.empty(3, 4, 2, 4)],
            )
        )

        pool.load_to_device_per_layer(
            device_pool=device_pool,
            host_indices=torch.tensor([1]),
            device_indices=torch.tensor([2]),
            layer_id=1,
            io_backend="kernel_ascend",
        )

        pool._copy_tensor_pf_lf.assert_called_once()
        call = pool._copy_tensor_pf_lf.call_args.kwargs
        self.assertIs(call["src"], pool.conv_buffer[0])
        self.assertIs(call["dst"], device_pool.mamba_cache.conv[0][1])


if __name__ == "__main__":
    unittest.main()
