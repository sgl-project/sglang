"""The packed Ulysses Q/K/V input exchange must be bit-identical to the
unpacked path. The collective is emulated in-process with exact
``all_to_all_single`` chunk semantics (rank r's j-th chunk goes to rank j's
r-th chunk); the pack kernel and unpack views run unmodified on CUDA."""

import math
import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers import usp as usp_mod
from sglang.test.test_utils import CustomTestCase

_USP = "sglang.multimodal_gen.runtime.layers.usp"


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestA2AStagingBuffer(CustomTestCase):
    def setUp(self):
        super().setUp()
        usp_mod._A2A_STAGING_BUFFERS.clear()

    def tearDown(self):
        usp_mod._A2A_STAGING_BUFFERS.clear()
        super().tearDown()

    def test_cache_capacity_is_bounded_across_shapes(self):
        device = torch.device("cuda", torch.cuda.current_device())
        role = "test_role"
        shapes = ((2, 3), (4, 5), (5, 4), (1, 7), (3, 11), (2, 4))

        with torch.no_grad():
            for shape in shapes:
                actual = usp_mod._a2a_staging_buffer(
                    role, shape, torch.bfloat16, device
                )
                self.assertEqual(actual.shape, shape)
                self.assertTrue(actual.is_contiguous())

        key = (role, torch.bfloat16, device.index)
        self.assertEqual(list(usp_mod._A2A_STAGING_BUFFERS), [key])
        self.assertEqual(
            usp_mod._A2A_STAGING_BUFFERS[key].numel(),
            max(math.prod(shape) for shape in shapes),
        )
        retained_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in usp_mod._A2A_STAGING_BUFFERS.values()
        )
        self.assertEqual(
            retained_bytes,
            max(math.prod(shape) for shape in shapes)
            * torch.empty((), dtype=torch.bfloat16).element_size(),
        )
        self.assertLess(
            retained_bytes,
            sum(math.prod(shape) for shape in shapes)
            * torch.empty((), dtype=torch.bfloat16).element_size(),
        )

    def test_smaller_shape_reuses_larger_backing_buffer(self):
        device = torch.device("cuda", torch.cuda.current_device())

        with torch.no_grad():
            large = usp_mod._a2a_staging_buffer(
                "test_role", (8, 16), torch.float16, device
            )
            small = usp_mod._a2a_staging_buffer(
                "test_role", (2, 7), torch.float16, device
            )

        self.assertEqual(large.untyped_storage().data_ptr(), small.data_ptr())
        self.assertEqual(small.shape, (2, 7))

    def test_role_and_dtype_are_separate_cache_keys(self):
        device = torch.device("cuda", torch.cuda.current_device())

        with torch.no_grad():
            usp_mod._a2a_staging_buffer("input", (8,), torch.float16, device)
            usp_mod._a2a_staging_buffer("output", (8,), torch.float16, device)
            usp_mod._a2a_staging_buffer("input", (8,), torch.bfloat16, device)

        self.assertEqual(len(usp_mod._A2A_STAGING_BUFFERS), 3)

    def test_bypass_paths_do_not_replace_cached_storage(self):
        cuda_device = torch.device("cuda", torch.cuda.current_device())

        with torch.no_grad():
            cached = usp_mod._a2a_staging_buffer(
                "test_role", (8,), torch.float16, cuda_device
            )
        key = ("test_role", torch.float16, cuda_device.index)
        cached_storage = usp_mod._A2A_STAGING_BUFFERS[key]

        with torch.enable_grad():
            grad_buffer = usp_mod._a2a_staging_buffer(
                "test_role", (16,), torch.float16, cuda_device
            )
        with (
            torch.no_grad(),
            patch(f"{_USP}.torch.compiler.is_compiling", return_value=True),
        ):
            compile_buffer = usp_mod._a2a_staging_buffer(
                "test_role", (16,), torch.float16, cuda_device
            )
        with (
            torch.no_grad(),
            patch(f"{_USP}.torch.cuda.is_current_stream_capturing", return_value=True),
        ):
            capture_buffer = usp_mod._a2a_staging_buffer(
                "test_role", (16,), torch.float16, cuda_device
            )
        with torch.no_grad():
            cpu_buffer = usp_mod._a2a_staging_buffer(
                "cpu", (8,), torch.float16, torch.device("cpu")
            )

        self.assertEqual(list(usp_mod._A2A_STAGING_BUFFERS), [key])
        self.assertIs(usp_mod._A2A_STAGING_BUFFERS[key], cached_storage)
        self.assertEqual(cached.numel(), 8)
        self.assertEqual(grad_buffer.device.type, "cuda")
        self.assertEqual(compile_buffer.device.type, "cuda")
        self.assertEqual(capture_buffer.device.type, "cuda")
        self.assertEqual(cpu_buffer.device.type, "cpu")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestPackedQKVInputA2A(CustomTestCase):
    def _run_all_ranks(self, fn, world):
        sends, recvs = [], None

        def fake_a2a(x, role=None):
            if recvs is None:  # recording pass
                sends.append(x.detach().clone())
                return torch.empty_like(x)
            return recvs.pop(0).reshape(x.shape)

        with (
            patch(f"{_USP}._usp_all_to_all_single", fake_a2a),
            patch(f"{_USP}.get_ulysses_parallel_world_size", return_value=world),
        ):
            for r in range(world):
                fn(r)
            recvs = [
                torch.cat([s.flatten().chunk(world)[r] for s in sends])
                for r in range(world)
            ]
            return [fn(r) for r in range(world)]  # replay pass

    def test_packed_matches_unpacked_bitwise(self):
        for world, b, s_global, h_global, d in ((4, 1, 128, 8, 64), (2, 2, 48, 6, 32)):
            torch.manual_seed(1234)
            s_local, h_local = s_global // world, h_global // world
            full = [
                torch.randn(
                    b, s_global, h_global, d, dtype=torch.bfloat16, device="cuda"
                )
                for _ in range(3)
            ]
            shards = [
                tuple(t[:, r * s_local : (r + 1) * s_local].contiguous() for t in full)
                for r in range(world)
            ]
            packed = self._run_all_ranks(
                lambda r: usp_mod._usp_input_all_to_all_qkv(*shards[r]), world
            )
            for r in range(world):
                for i in range(3):
                    spec = full[i][:, :, r * h_local : (r + 1) * h_local].contiguous()
                    self.assertTrue(
                        torch.equal(packed[r][i], spec), f"rank{r} qkv[{i}]"
                    )
                    self.assertTrue(packed[r][i].is_contiguous())


if __name__ == "__main__":
    unittest.main()
