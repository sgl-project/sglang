"""DFlash prefix commits retain only one compressed MLA row per token."""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.runtime_context import reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.parallel_groups import publish

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class TestMLAPrefixValidPool(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def _inputs(self, dtype=torch.bfloat16):
        pool = MLATokenToKVPool(
            size=31,
            page_size=1,
            dtype=dtype,
            kv_lora_rank=128,
            qk_rope_head_dim=64,
            layer_num=2,
            device="cuda",
            enable_memory_saver=False,
        )
        loc = torch.tensor([[5, 6, 7], [9, 10, 11]], device="cuda")
        lengths = torch.tensor([2, 1], dtype=torch.int32, device="cuda")
        latent = torch.randn(6, 1, 128, dtype=torch.bfloat16, device="cuda")
        rope = torch.randn(6, 1, 64, dtype=torch.bfloat16, device="cuda")
        return pool, loc, lengths, latent, rope

    def _assert_committed(self, pool, latent, rope, rows, slots):
        expected = torch.zeros_like(pool.kv_buffer[1])
        source = torch.cat((latent, rope), dim=-1).to(pool.dtype).view(pool.store_dtype)
        expected[slots] = source[rows]
        self.assertTrue(torch.equal(pool.kv_buffer[1], expected))
        self.assertEqual(torch.count_nonzero(pool.kv_buffer[0]).item(), 0)

    def test_compressed_prefix_rows_and_dtype_conversion(self):
        for dtype in (torch.bfloat16, fp8_dtype):
            with self.subTest(dtype=dtype):
                pool, loc, lengths, latent, rope = self._inputs(dtype)
                self.assertEqual(tuple(pool.kv_buffer[0].shape), (32, 1, 192))
                self.assertEqual(
                    pool.get_kv_size_bytes(), 32 * 192 * 2 * dtype.itemsize
                )
                pool.set_mla_kv_buffer_prefix_valid(
                    SimpleNamespace(layer_id=0),
                    loc,
                    lengths,
                    latent,
                    rope,
                    layer_id_override=1,
                )
                self._assert_committed(pool, latent, rope, [0, 1, 3], [5, 6, 9])

    def test_replay_obeys_updated_commit_lengths(self):
        pool, loc, lengths, latent, rope = self._inputs()
        layer = SimpleNamespace(layer_id=1)
        # Compile the existing MLA scatter before recording the graph.
        pool.set_mla_kv_buffer_prefix_valid(layer, loc, lengths, latent, rope)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            pool.set_mla_kv_buffer_prefix_valid(layer, loc, lengths, latent, rope)
        pool.kv_buffer[1].zero_()
        lengths.copy_(torch.tensor([0, 3], dtype=torch.int32, device="cuda"))
        graph.replay()
        self._assert_committed(pool, latent, rope, [3, 4, 5], [9, 10, 11])

    def test_rejects_expanded_kv_rows(self):
        pool, loc, lengths, latent, rope = self._inputs()
        with self.assertRaisesRegex(ValueError, "compressed latent/shared RoPE"):
            pool.set_mla_kv_buffer_prefix_valid(
                SimpleNamespace(layer_id=0),
                loc,
                lengths,
                latent.expand(-1, 2, -1),
                rope,
            )


if __name__ == "__main__":
    unittest.main()
