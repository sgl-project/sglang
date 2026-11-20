"""CPU checks for the shared NPU C8 cache layout and INT8 scale contracts."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.hardware_backend.npu import memory_pool_npu as pools
from sglang.srt.layers.attention.dsa import dsa_npu_indexer as indexer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestNpuDsaC8(CustomTestCase):
    def setUp(self):
        super().setUp()
        parallel = patch.object(
            pools,
            "get_parallel",
            return_value=SimpleNamespace(attn_dcp_size=1, attn_dcp_rank=0),
        )
        parallel.start()
        self.addCleanup(parallel.stop)

    def make_pool(self, dtype):
        return pools.NPUMLATokenToKVPool(
            size=128,
            page_size=128,
            dtype=dtype,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
            index_head_dim=128,
            indexer_layer_ids=(0,),
            kv_cache_dim=656,
        )

    def test_fp8_and_int8_allocate_the_same_packed_main_cache(self):
        for dtype, scale_dtype in (
            (torch.int8, torch.float16),
            (torch.float8_e4m3fn, torch.float32),
        ):
            with self.subTest(dtype=dtype):
                pool = self.make_pool(dtype)
                self.assertTrue(pool.dsa_kv_cache_store_c8)
                self.assertEqual(pool.k_buffer.shape, (2, 2, 128, 1, 656))
                self.assertEqual(pool.k_buffer.dtype, dtype)
                self.assertEqual(pool.v_buffer.numel(), 0)
                self.assertEqual(pool.v_buffer.dtype, torch.bfloat16)
                self.assertEqual(pool.index_k_scale_buffer.shape, (1, 2, 128, 1))
                self.assertEqual(pool.index_k_scale_buffer.dtype, scale_dtype)
                # Only the active indexer layer owns K and scale buffers.
                self.assertEqual(len(pool._get_disagg_buffer_entries()), 4)
                self.assertEqual(pool.get_state_layer_ids(), [0, 0])
                expected_bytes = (
                    2 * 656 + 128 + torch.empty((), dtype=scale_dtype).element_size()
                ) * 256
                self.assertEqual(pool.get_kv_size_bytes(), expected_bytes)

    def test_int8_write_preserves_rope_and_scale_bytes_at_the_requested_slots(self):
        pool = self.make_pool(torch.int8)
        latent = torch.randn(2, 1, 512, dtype=torch.bfloat16)
        rope = torch.randn(2, 1, 64, dtype=torch.bfloat16)
        quantized = torch.arange(1024).to(torch.int8).view(2, 1, 512)
        scales = torch.tensor([[0.125, 0.25, 0.5, 1.0], [2.0, 4.0, 8.0, 16.0]])

        def scatter(cache, indices, values):
            cache[indices.flatten().long()] = values

        vendor = SimpleNamespace(
            npu_dynamic_block_quant=Mock(return_value=(quantized, scales)),
            npu_scatter_nd_update_=Mock(side_effect=scatter),
        )
        loc = torch.tensor([7, 129])
        with patch.object(pools, "torch_npu", vendor, create=True):
            pool.set_kv_buffer(SimpleNamespace(layer_id=0), loc, latent, rope)
        quant_args = vendor.npu_dynamic_block_quant.call_args
        self.assertEqual(quant_args.args[0].shape, (2, 1, 512))
        self.assertEqual(
            quant_args.kwargs,
            dict(dst_type=torch.int8, row_block_size=1, col_block_size=128),
        )
        rows = pool.k_buffer[0].view(-1, 656)[loc]
        torch.testing.assert_close(rows[:, :512], quantized.view(2, 512))
        torch.testing.assert_close(
            rows[:, 512:640].contiguous().view(torch.bfloat16), rope.view(2, 64)
        )
        torch.testing.assert_close(
            rows[:, 640:].contiguous().view(torch.float32), scales
        )
        self.assertFalse(pool.k_buffer[0].view(-1, 656)[0].any())
        # Retraction/offload receives the full packed record and the separate LI scale.
        buffers = pool._get_cpu_offload_layer_buffers(0)
        self.assertEqual(buffers[0].shape, (256, 1, 656))
        self.assertEqual(buffers[-1].dtype, torch.float16)

    def test_indexer_quantizes_after_hadamard_and_casts_scales_for_each_dtype(self):
        x = torch.randn(2, 3, 128, dtype=torch.bfloat16)
        hadamard = indexer.create_npu_hadamard_128(128, "cpu")
        rotated = (x @ hadamard).reshape(-1, 128)

        def dynamic_quant(input_, dst_type):
            return (
                torch.zeros(input_.shape, dtype=dst_type),
                torch.arange(1, input_.shape[0] + 1, dtype=torch.float32),
            )

        vendor = SimpleNamespace(npu_dynamic_quant=Mock(side_effect=dynamic_quant))
        with patch.object(indexer, "torch_npu", vendor, create=True):
            for dtype, scale_dtype in (
                (torch.int8, torch.float16),
                (torch.float8_e4m3fn, torch.float32),
            ):
                with self.subTest(dtype=dtype):
                    q, scale = indexer._quantize_npu_indexer_activation(
                        x, hadamard, dtype, scale_dtype
                    )
                    empty_q, empty_scale = indexer._quantize_npu_indexer_activation(
                        x[:0], hadamard, dtype, scale_dtype
                    )
                    self.assertEqual(q.shape, x.shape)
                    self.assertEqual(q.dtype, dtype)
                    self.assertEqual(scale.dtype, scale_dtype)
                    torch.testing.assert_close(
                        scale,
                        torch.arange(1, 7, dtype=scale_dtype).view(2, 3),
                    )
                    self.assertEqual(empty_q.shape, (0, 3, 128))
                    self.assertEqual(empty_scale.dtype, scale_dtype)

        self.assertEqual(vendor.npu_dynamic_quant.call_count, 2)
        for call in vendor.npu_dynamic_quant.call_args_list:
            torch.testing.assert_close(call.args[0], rotated)


if __name__ == "__main__":
    unittest.main()
