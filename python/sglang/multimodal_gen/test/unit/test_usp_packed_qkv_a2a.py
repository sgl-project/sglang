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


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestPackedQKVRowViewPredicate(CustomTestCase):
    """`_packed_qkv_row_view_is_free` decides which path the a2a takes.

    It is the whole safety argument for accepting strided input: the pack
    kernel reads q/k/v through explicit row/head strides, so only head_size
    needs unit stride, but `view(b * s_local, h, d)` still has to be free. If
    the predicate admits a layout where that view would copy, the kernel reads
    the wrong rows and attention is silently wrong -- no exception.
    """

    H, D, WORLD = 8, 64, 4

    def _merged_qkv_chunk(self, b, s):
        """q/k/v as chunks of one packed projection, the FLUX.2 layout.

        Non-contiguous, unit stride on head_size, batch/seq mergeable -- the
        case this PR exists to accept.
        """
        inner = self.H * self.D
        hidden = torch.randn(b, s, 3 * inner, device="cuda", dtype=torch.bfloat16)
        q, k, v = hidden.chunk(3, dim=-1)
        return tuple(t.unflatten(-1, (self.H, self.D)) for t in (q, k, v))

    def test_strided_merged_projection_is_accepted(self):
        q, k, v = self._merged_qkv_chunk(1, 32)
        self.assertFalse(q.is_contiguous(), "setup: q must be strided")
        self.assertTrue(usp_mod._packed_qkv_row_view_is_free(q))
        self.assertTrue(usp_mod._can_use_packed_qkv_a2a_4d(q, k, v, self.WORLD))
        # the row view the pack kernel takes must genuinely not copy
        rows = q.shape[0] * q.shape[1]
        self.assertEqual(q.view(rows, self.H, self.D).data_ptr(), q.data_ptr())

    def test_batched_sequence_slice_is_rejected(self):
        """b > 1 sliced along seq: stride(0) still spans the unsliced rows, so
        merging batch and seq would copy. Must stay on the unpacked path."""
        full = torch.randn(2, 64, self.H, self.D, device="cuda", dtype=torch.bfloat16)
        sl = full[:, 32:]
        self.assertEqual(sl.stride(-1), 1, "setup: head_size is unit stride")
        self.assertNotEqual(sl.stride(0), sl.shape[1] * sl.stride(1))
        self.assertFalse(usp_mod._packed_qkv_row_view_is_free(sl))
        self.assertFalse(usp_mod._can_use_packed_qkv_a2a_4d(sl, sl, sl, self.WORLD))

    def test_contiguous_still_accepted(self):
        """No regression for the layouts that already qualified."""
        t = torch.randn(1, 32, self.H, self.D, device="cuda", dtype=torch.bfloat16)
        self.assertTrue(t.is_contiguous())
        self.assertTrue(usp_mod._packed_qkv_row_view_is_free(t))
        self.assertTrue(usp_mod._can_use_packed_qkv_a2a_4d(t, t, t, self.WORLD))

    def test_non_unit_head_stride_is_rejected(self):
        """head_size must be unit-stride; the kernel cannot express a gap."""
        wide = torch.randn(
            1, 32, self.H, 2 * self.D, device="cuda", dtype=torch.bfloat16
        )
        t = wide[..., ::2]
        self.assertNotEqual(t.stride(-1), 1)
        self.assertFalse(usp_mod._packed_qkv_row_view_is_free(t))


if __name__ == "__main__":
    unittest.main()
