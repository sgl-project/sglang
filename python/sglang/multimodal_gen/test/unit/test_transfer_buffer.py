# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TransferTensorBuffer."""

import unittest

import torch

from sglang.multimodal_gen.runtime.disaggregation.transport.buffer import (
    TransferTensorBuffer,
)


class TestTransferTensorBuffer(unittest.TestCase):
    """Test tensor buffer allocation and memory operations."""

    def test_basic_alloc_free(self):
        buf = TransferTensorBuffer(pool_size=1 << 20, role_name="test")
        handle = buf.allocate(size=4096, request_id="req-1")
        self.assertIsNotNone(handle)
        self.assertEqual(handle.request_id, "req-1")
        self.assertTrue(buf.free(handle))

    def test_alloc_failure(self):
        buf = TransferTensorBuffer(
            pool_size=1 << 20, min_block_size=1 << 20, role_name="test"
        )
        # Pool has only 1 slot
        h1 = buf.allocate(size=1 << 20, request_id="req-1")
        self.assertIsNotNone(h1)
        h2 = buf.allocate(size=1 << 20, request_id="req-2")
        self.assertIsNone(h2)
        buf.free(h1)
        h3 = buf.allocate(size=1 << 20, request_id="req-3")
        self.assertIsNotNone(h3)

    def test_pool_is_pinned(self):
        buf = TransferTensorBuffer(pool_size=1 << 20, role_name="test")
        self.assertTrue(buf._pool.is_pinned())

    def test_pool_data_ptr(self):
        buf = TransferTensorBuffer(pool_size=1 << 20, role_name="test")
        self.assertGreater(buf.pool_data_ptr, 0)

    def test_free_slots_count(self):
        buf = TransferTensorBuffer(
            pool_size=4 << 20, min_block_size=1 << 20, role_name="test"
        )
        self.assertEqual(buf.free_slots_count(1 << 20), 4)
        h = buf.allocate(size=1 << 20, request_id="r1")
        self.assertEqual(buf.free_slots_count(1 << 20), 3)
        buf.free(h)
        self.assertEqual(buf.free_slots_count(1 << 20), 4)

    def test_stats(self):
        buf = TransferTensorBuffer(pool_size=1 << 20, role_name="encoder")
        stats = buf.get_stats()
        self.assertEqual(stats["role"], "encoder")
        self.assertGreater(stats["pool_size"], 0)


class TestTransferTensorBufferIO(unittest.TestCase):
    """Test tensor write/read operations on CPU."""

    def test_write_read_cpu_tensor(self):
        """Write a CPU tensor to slot, read it back."""
        buf = TransferTensorBuffer(pool_size=1 << 20, role_name="test")
        handle = buf.allocate(size=1 << 20, request_id="req-1")

        src = torch.randn(4, 8, dtype=torch.float32)
        nbytes = buf.write_tensor(handle, "test", src, byte_offset=0)
        self.assertEqual(nbytes, 4 * 8 * 4)

        dst = buf.read_tensor(handle, shape=[4, 8], dtype=torch.float32, byte_offset=0)
        self.assertTrue(torch.allclose(src, dst))
        buf.free(handle)

    def test_write_read_bfloat16(self):
        buf = TransferTensorBuffer(pool_size=1 << 20, role_name="test")
        handle = buf.allocate(size=1 << 20, request_id="req-1")

        src = torch.randn(2, 16, dtype=torch.bfloat16)
        buf.write_tensor(handle, "embeds", src, byte_offset=0)

        dst = buf.read_tensor(
            handle, shape=[2, 16], dtype=torch.bfloat16, byte_offset=0
        )
        self.assertTrue(torch.allclose(src, dst))
        buf.free(handle)

    def test_write_multiple_at_offsets(self):
        buf = TransferTensorBuffer(pool_size=1 << 20, role_name="test")
        handle = buf.allocate(size=1 << 20, request_id="req-1")

        t1 = torch.randn(8, dtype=torch.float32)  # 32 bytes
        t2 = torch.randn(4, dtype=torch.float32)  # 16 bytes

        buf.write_tensor(handle, "a", t1, byte_offset=0)
        buf.write_tensor(handle, "b", t2, byte_offset=512)  # aligned offset

        r1 = buf.read_tensor(handle, [8], torch.float32, byte_offset=0)
        r2 = buf.read_tensor(handle, [4], torch.float32, byte_offset=512)

        self.assertTrue(torch.allclose(t1, r1))
        self.assertTrue(torch.allclose(t2, r2))
        buf.free(handle)

    def test_write_exceeds_slot_raises(self):
        buf = TransferTensorBuffer(
            pool_size=1 << 20, min_block_size=1024, role_name="test"
        )
        handle = buf.allocate(size=1024, request_id="req-1")

        big = torch.randn(512, dtype=torch.float32)  # 2048 bytes > 1024
        with self.assertRaises(ValueError):
            buf.write_tensor(handle, "big", big, byte_offset=0)
        buf.free(handle)


class TestTransferTensorBufferBatchIO(unittest.TestCase):
    """Test batch write/read with manifest."""

    def test_write_read_manifest(self):
        buf = TransferTensorBuffer(pool_size=4 << 20, role_name="test")
        handle = buf.allocate(size=1 << 20, request_id="req-1")

        tensors = {
            "prompt_embeds": torch.randn(1, 16, 64, dtype=torch.bfloat16),
            "latents": torch.randn(1, 4, 8, 8, dtype=torch.float32),
            "timesteps": torch.tensor([999.0, 950.0, 900.0]),
        }

        manifest = buf.write_tensors_from_gpu(handle, tensors)
        self.assertIn("prompt_embeds", manifest)
        self.assertIn("latents", manifest)
        self.assertIn("timesteps", manifest)

        result = buf.read_tensors_from_manifest(handle, manifest)
        self.assertTrue(
            torch.allclose(tensors["prompt_embeds"], result["prompt_embeds"])
        )
        self.assertTrue(torch.allclose(tensors["latents"], result["latents"]))
        self.assertTrue(torch.allclose(tensors["timesteps"], result["timesteps"]))
        buf.free(handle)

    def test_write_read_with_list_tensors(self):
        buf = TransferTensorBuffer(pool_size=4 << 20, role_name="test")
        handle = buf.allocate(size=1 << 20, request_id="req-1")

        tensors = {
            "embeds": [
                torch.randn(1, 8, dtype=torch.float32),
                torch.randn(1, 8, dtype=torch.float32),
            ],
        }

        manifest = buf.write_tensors_from_gpu(handle, tensors)
        result = buf.read_tensors_from_manifest(handle, manifest)

        self.assertIsInstance(result["embeds"], list)
        self.assertEqual(len(result["embeds"]), 2)
        for i in range(2):
            self.assertTrue(torch.allclose(tensors["embeds"][i], result["embeds"][i]))
        buf.free(handle)

    def test_write_skips_none(self):
        buf = TransferTensorBuffer(pool_size=4 << 20, role_name="test")
        handle = buf.allocate(size=1 << 20, request_id="req-1")

        tensors = {
            "a": torch.randn(4, dtype=torch.float32),
            "b": None,
        }
        manifest = buf.write_tensors_from_gpu(handle, tensors)
        self.assertIn("a", manifest)
        self.assertNotIn("b", manifest)
        buf.free(handle)

    def test_realistic_encoder_output(self):
        """Simulate encoder→denoiser transfer for Wan2.1-like model."""
        # ~60MB total: prompt_embeds + latents
        pool_size = 256 << 20  # 256 MiB
        buf = TransferTensorBuffer(pool_size=pool_size, role_name="encoder")
        handle = buf.allocate(size=64 << 20, request_id="wan-001")
        self.assertIsNotNone(handle)

        tensors = {
            "prompt_embeds": [torch.randn(1, 512, 4096, dtype=torch.bfloat16)],
            "latents": torch.randn(1, 16, 21, 30, 52, dtype=torch.bfloat16),
            "timesteps": torch.tensor(
                [999.0 - i * 20 for i in range(50)], dtype=torch.float32
            ),
        }

        manifest = buf.write_tensors_from_gpu(handle, tensors)

        result = buf.read_tensors_from_manifest(handle, manifest)
        self.assertTrue(
            torch.allclose(tensors["prompt_embeds"][0], result["prompt_embeds"][0])
        )
        self.assertTrue(torch.allclose(tensors["latents"], result["latents"]))
        self.assertTrue(torch.allclose(tensors["timesteps"], result["timesteps"]))
        buf.free(handle)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestTransferTensorBufferGPU(unittest.TestCase):
    """Test GPU-backed TransferTensorBuffer (GPUDirect RDMA path)."""

    def test_gpu_pool_allocation(self):
        buf = TransferTensorBuffer(pool_size=1 << 20, role_name="test", device="cuda:0")
        self.assertEqual(buf.device, "cuda:0")
        self.assertTrue(buf._pool.is_cuda)
        handle = buf.allocate(size=4096, request_id="req-1")
        self.assertIsNotNone(handle)
        buf.free(handle)

    def test_gpu_write_read_roundtrip(self):
        """Write a GPU tensor to GPU pool, read it back — no D2H/H2D."""
        buf = TransferTensorBuffer(pool_size=1 << 20, role_name="test", device="cuda:0")
        handle = buf.allocate(size=1 << 20, request_id="req-1")

        src = torch.randn(4, 8, dtype=torch.float32, device="cuda:0")
        nbytes = buf.write_tensor(handle, "test", src, byte_offset=0)
        self.assertEqual(nbytes, 4 * 8 * 4)

        dst = buf.read_tensor(
            handle, shape=[4, 8], dtype=torch.float32, byte_offset=0, device="cuda:0"
        )
        self.assertTrue(dst.is_cuda)
        self.assertTrue(torch.allclose(src, dst))
        buf.free(handle)

    def test_gpu_write_read_bfloat16(self):
        buf = TransferTensorBuffer(pool_size=1 << 20, role_name="test", device="cuda:0")
        handle = buf.allocate(size=1 << 20, request_id="req-1")

        src = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda:0")
        buf.write_tensor(handle, "embeds", src, byte_offset=0)

        dst = buf.read_tensor(
            handle, shape=[2, 16], dtype=torch.bfloat16, byte_offset=0, device="cuda:0"
        )
        self.assertTrue(torch.allclose(src, dst))
        buf.free(handle)

    def test_gpu_batch_write_read_manifest(self):
        buf = TransferTensorBuffer(pool_size=4 << 20, role_name="test", device="cuda:0")
        handle = buf.allocate(size=1 << 20, request_id="req-1")

        tensors = {
            "prompt_embeds": torch.randn(
                1, 16, 64, dtype=torch.bfloat16, device="cuda:0"
            ),
            "latents": torch.randn(1, 4, 8, 8, dtype=torch.float32, device="cuda:0"),
        }

        manifest = buf.write_tensors_from_gpu(handle, tensors)
        result = buf.read_tensors_from_manifest(handle, manifest, device="cuda:0")

        self.assertTrue(
            torch.allclose(tensors["prompt_embeds"], result["prompt_embeds"])
        )
        self.assertTrue(torch.allclose(tensors["latents"], result["latents"]))
        buf.free(handle)

    def test_gpu_pool_read_to_cpu(self):
        """Read from GPU pool to CPU (cross-device)."""
        buf = TransferTensorBuffer(pool_size=1 << 20, role_name="test", device="cuda:0")
        handle = buf.allocate(size=1 << 20, request_id="req-1")

        src = torch.randn(4, 8, dtype=torch.float32, device="cuda:0")
        buf.write_tensor(handle, "test", src, byte_offset=0)

        dst = buf.read_tensor(
            handle, shape=[4, 8], dtype=torch.float32, byte_offset=0, device="cpu"
        )
        self.assertFalse(dst.is_cuda)
        self.assertTrue(torch.allclose(src.cpu(), dst))
        buf.free(handle)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestTransferEngineGPUDirect(unittest.TestCase):
    """Test supports_gpu_direct property on engine classes."""

    def test_mock_engine_no_gpu_direct(self):
        from sglang.multimodal_gen.runtime.disaggregation.transport.engine import (
            MockTransferEngine,
        )

        engine = MockTransferEngine()
        self.assertFalse(engine.supports_gpu_direct)


if __name__ == "__main__":
    unittest.main()
