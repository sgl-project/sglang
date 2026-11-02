"""CPU checks for NPU DWDP storage ownership and partial resource cleanup."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.hardware_backend.npu.dwdp.manager import (
    NpuDwdpManager,
    NpuWeightBuffer,
    _WeightSpec,
)
from sglang.srt.hardware_backend.npu.dwdp.transport import (
    NpuWeightTransport,
    _Allocation,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="stage-a-unit-test-npu")


class TestNpuDwdpValidation(unittest.TestCase):
    def test_rejects_incompatible_execution_before_weight_migration(self):
        for field, value, message in (
            ("tp_size", 4, "DWDP=TP"),
            ("pp_size", 2, "PP1"),
            ("attn_cp_size", 2, "CP1"),
            ("attn_dcp_size", 2, "DCP1"),
            ("lora_paths", ["adapter"], "LoRA"),
            ("cpu_offload_gb", 1, "CPU offload"),
            ("offload_group_size", 1, "CPU offload"),
            ("enable_pd_role_switch", True, "PD role switching"),
        ):
            with self.subTest(field=field):
                manager = NpuDwdpManager.__new__(NpuDwdpManager)
                manager.dwdp_size = 2
                manager._group = SimpleNamespace(world_size=2)
                manager._parallel = SimpleNamespace(
                    tp_size=2,
                    pp_size=1,
                    attn_tp_size=1,
                    attn_cp_size=1,
                    attn_dcp_size=1,
                )
                manager._server_args = SimpleNamespace(
                    enable_lora=None,
                    lora_paths=None,
                    cpu_offload_gb=0,
                    offload_group_size=0,
                    enable_memory_saver=False,
                    enable_pd_role_switch=False,
                )
                target = (
                    manager._parallel
                    if hasattr(manager._parallel, field)
                    else manager._server_args
                )
                setattr(target, field, value)
                with (
                    patch(
                        "sglang.srt.model_executor.cuda_graph_config.cuda_graph_fully_disabled",
                        return_value=True,
                    ),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    manager._validate(None)


class TestNpuWeightBuffer(unittest.TestCase):
    def test_moe_order_and_two_slots_across_different_widths_and_dtypes(self):
        specs = {
            2: {"w13_weight": _WeightSpec((4, 32, 16), torch.bfloat16, 2)},
            5: {"w13_weight": _WeightSpec((4, 64, 16), torch.int8, 1)},
            7: {"w13_weight": _WeightSpec((4, 96, 16), torch.float16, 2)},
        }
        buffer = NpuWeightBuffer(specs, "cpu")
        first = buffer.get_full_tensor(2, "w13_weight")
        second = buffer.get_full_tensor(5, "w13_weight")
        third = buffer.get_full_tensor(7, "w13_weight")
        self.assertEqual(first.data_ptr(), third.data_ptr())
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        self.assertEqual([buffer.buffer_index_for_layer(i) for i in specs], [0, 1, 0])
        self.assertEqual(first.shape, specs[2]["w13_weight"].shape)
        self.assertEqual(third.shape, specs[7]["w13_weight"].shape)
        self.assertEqual(
            sum(x.numel() for x in buffer.buffers), 4 * 96 * 16 * 2 + 4 * 64 * 16
        )
        first.fill_(1)
        second.fill_(3)
        torch.testing.assert_close(
            first.view(torch.uint8).reshape(-1),
            third.view(torch.uint8).reshape(-1)[: first.numel() * first.element_size()],
        )
        torch.testing.assert_close(second, torch.full_like(second, 3))
        buffer.release()
        buffer.release()
        self.assertEqual(buffer.buffers, [])

    def test_weight_regions_are_aligned_disjoint_and_cover_all_experts(self):
        specs = {
            0: {
                "w13_weight": _WeightSpec((6, 5, 7), torch.float16, 2),
                "w2_weight": _WeightSpec((6, 7, 3), torch.int8, 1),
            }
        }
        buffer = NpuWeightBuffer(specs, "cpu")
        w13 = buffer.get_full_tensor(0, "w13_weight")
        w2 = buffer.get_full_tensor(0, "w2_weight")
        self.assertEqual((w2.data_ptr() - w13.data_ptr()) % 512, 0)
        w13.fill_(2)
        w2.fill_(5)
        torch.testing.assert_close(w13, torch.full_like(w13, 2))
        self.assertTrue(w13.is_contiguous() and w2.is_contiguous())


class TestNpuTransportLifetime(unittest.TestCase):
    def make_transport(self):
        transport = NpuWeightTransport.__new__(NpuWeightTransport)
        transport.rt = SimpleNamespace(
            unmap_mem=Mock(return_value=0),
            release_mem_address=Mock(return_value=0),
            free_physical=Mock(return_value=0),
            memcpy_async=Mock(return_value=0),
        )
        transport._local = {(0, "w13_weight"): _Allocation(11, 4096, 8192, True)}
        transport._allocations = list(transport._local.values())
        transport._imports = [_Allocation(12, 4096, 16384, True)]
        transport._peer_shards = {(1, 0, "w13_weight"): (16384, 4096)}
        transport._exports = {}
        return transport

    def test_actual_acl_copy_uses_supplied_stream_and_local_or_imported_source(self):
        transport = self.make_transport()
        stream = SimpleNamespace(npu_stream=999)
        transport.copy_shard(0, 1, (0, "w13_weight"), 32768, 1024, stream)
        transport.rt.memcpy_async.assert_called_with(32768, 1024, 16384, 1024, 3, 999)
        transport.copy_shard(0, 0, (0, "w13_weight"), 32768, 1024, stream)
        transport.rt.memcpy_async.assert_called_with(32768, 1024, 8192, 1024, 3, 999)
        with self.assertRaises(ValueError):
            transport.copy_shard(0, 1, (0, "w13_weight"), 32768, 8192, stream)

    def test_cleanup_retries_only_unfinished_steps(self):
        transport = self.make_transport()
        transport.rt.release_mem_address.side_effect = [123, 0, 0]
        with self.assertRaisesRegex(RuntimeError, "release expert address"):
            transport.close()
        transport.close()
        transport.close()
        self.assertEqual(transport.rt.unmap_mem.call_count, 2)
        self.assertEqual(transport.rt.free_physical.call_count, 2)
        self.assertEqual(transport.rt.release_mem_address.call_count, 3)
        self.assertEqual(transport._allocations, [])
        self.assertEqual(transport._imports, [])


if __name__ == "__main__":
    unittest.main()
