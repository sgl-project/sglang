"""GGUF route weighting preserves per-slot rounding and skips inactive data."""

import unittest

import sgl_kernel  # noqa: F401
import torch
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType

from sglang.kernels.ops.moe.gguf_moe_vec import gguf_moe_a8_vec
from sglang.kernels.ops.moe.gguf_moe_weights_aot import resolve_gguf_moe_weights
from sglang.srt.utils import is_musa
from sglang.test.ci.ci_register import register_cuda_ci, register_musa_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="nightly", runner_config="1-gpu-large")
# backend-specific: MUSA uses the native weighting operator in its serving path.
register_musa_ci(est_time=6, suite="nightly-musa-1-gpu", nightly=True)


class TestGgufMoeWeights(CustomTestCase):
    @unittest.skipIf(is_musa(), "Local GGUF vector projection is CUDA-only")
    @torch.inference_mode()
    def test_vector_formats_match_native_and_replay_inactive_routes(self):
        """Every vector format retains real provider math and skips negative operands."""
        from sgl_kernel.quantization import ggml_moe_a8_vec

        generator = torch.Generator(device="cuda").manual_seed(314)
        base_ids = torch.tensor(
            [[0, 3], [1, 2], [3, 0]], dtype=torch.int32, device="cuda"
        )
        # These valid packed bytes give finite, nonzero scales, integer codes and
        # codebook indices, including IQ1_M's distributed FP16 scale bits.
        for quant_type in (
            2,
            3,
            6,
            7,
            8,
            10,
            11,
            12,
            13,
            14,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            29,
        ):
            block_size, block_bytes = GGML_QUANT_SIZES[GGMLQuantizationType(quant_type)]
            packed = torch.empty(
                4, 17, 256 // block_size * block_bytes, dtype=torch.uint8, device="cuda"
            )
            for expert in range(4):
                packed[expert].fill_(0x31 + expert)
            dtypes = (
                (torch.float16, torch.bfloat16, torch.float32)
                if quant_type in (2, 8, 10, 29)
                else (torch.float16,)
            )
            for dtype in dtypes:
                with self.subTest(quant_type=quant_type, dtype=dtype):
                    inputs = (
                        torch.randn(
                            3, 256, generator=generator, device="cuda", dtype=dtype
                        )
                        / 4
                    )
                    original = inputs.clone()
                    ids = base_ids.clone()
                    expected = ggml_moe_a8_vec(
                        inputs, packed, ids, 2, quant_type, 17, 3
                    )
                    self.assertTrue(bool(torch.isfinite(expected).all()))
                    self.assertGreater(float(expected.abs().max()), 0.0)
                    actual = gguf_moe_a8_vec(inputs, packed, ids, 2, quant_type, 17, 3)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    # The down projection has one input per original route, while
                    # its IDs retain [original_tokens, original_topk] shape.
                    down = inputs.repeat_interleave(2, dim=0)
                    torch.testing.assert_close(
                        gguf_moe_a8_vec(down, packed, ids, 1, quant_type, 17, 6),
                        ggml_moe_a8_vec(down, packed, ids, 1, quant_type, 17, 6),
                        rtol=0,
                        atol=0,
                    )
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = gguf_moe_a8_vec(
                            inputs, packed, ids, 2, quant_type, 17, 3
                        )
                    original_pointer = ids.data_ptr()
                    for routes in (
                        [[-1, 3], [1, -3], [-7, -1]],
                        [[-1, -3], [-7, -1], [-1, -1]],
                        base_ids,
                    ):
                        ids.copy_(
                            torch.as_tensor(routes, dtype=ids.dtype, device=ids.device)
                        )
                        inputs.copy_(original)
                        inputs[(ids < 0).all(dim=1)] = float("nan")
                        captured.fill_(float("nan"))
                        graph.replay()
                        live = (ids >= 0).flatten()
                        self.assertTrue(bool(torch.isfinite(captured).all()))
                        self.assertEqual(int(torch.count_nonzero(captured[~live])), 0)
                        torch.testing.assert_close(
                            captured[live], expected[live], rtol=0, atol=0
                        )
                        self.assertEqual(ids.data_ptr(), original_pointer)
                    empty = gguf_moe_a8_vec(
                        inputs[:0], packed, ids[:0], 2, quant_type, 17, 0
                    )
                    self.assertEqual(empty.shape, (0, 17))

    def test_rounding_and_poisoned_routes(self):
        """Weights round per slot before summation; dropped NaNs contribute zero."""
        device = "musa" if is_musa() else "cuda"
        providers = {}
        native = resolve_gguf_moe_weights(device)
        if device == "musa":
            self.assertIsNotNone(native, "MUSA kernel tests require a rebuilt provider")
        else:
            from sglang.kernels.ops.moe.gguf_moe_weights import apply_gguf_moe_weights

            providers["triton"] = apply_gguf_moe_weights
        if native is not None:
            providers["aot"] = native
        generator = torch.Generator(device=device).manual_seed(836)
        for provider, apply_weights in providers.items():
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                for weight_dtype in (torch.float16, torch.bfloat16, torch.float32):
                    for strided in (False, True):
                        with self.subTest(
                            provider=provider,
                            dtype=dtype,
                            weight_dtype=weight_dtype,
                            strided=strided,
                        ):
                            outputs = torch.randn(
                                12, 257, generator=generator, device=device, dtype=dtype
                            )
                            weights = torch.randn(
                                4,
                                3,
                                generator=generator,
                                device=device,
                                dtype=weight_dtype,
                            )
                            ids = torch.tensor(
                                [[0, 1, 2], [0, -3, 2], [-1, -1, -1], [1, 0, -1]],
                                device=device,
                                dtype=torch.int64 if strided else torch.int32,
                            )
                            if strided:
                                weights = weights.t().contiguous().t()
                                ids = ids.t().contiguous().t()
                            inactive = ids < 0
                            outputs[inactive.flatten()] = float("nan")
                            weights[inactive] = float("nan")
                            expected = outputs.clone()
                            expected[inactive.flatten()] = 0
                            expected.view(4, 3, 257).mul_(
                                weights.masked_fill(inactive, 0).unsqueeze(-1)
                            )
                            apply_weights(outputs, weights, ids)
                            torch.testing.assert_close(
                                outputs, expected, rtol=0, atol=0
                            )
                            self.assertTrue(bool(torch.isfinite(outputs).all()))
                            ids.fill_(-1)
                            weights.fill_(float("nan"))
                            outputs.fill_(float("nan"))
                            apply_weights(outputs, weights, ids)
                            self.assertEqual(int(torch.count_nonzero(outputs)), 0)
                            apply_weights(outputs[:0], weights[:0], ids[:0])

    @unittest.skipIf(is_musa(), "CUDA fast-math regression")
    def test_subnormal_operands_and_products(self):
        """A tiny score can produce a normal result even in the SM90 fast-math build."""
        native = resolve_gguf_moe_weights("cuda")
        if native is None:
            self.skipTest("Native GGUF weighting requires a rebuilt kernel provider")
        for dtype in (torch.float32, torch.bfloat16):
            for weight_dtype in (torch.float32, torch.bfloat16):
                for strided in (False, True):
                    with self.subTest(
                        dtype=dtype, weight_dtype=weight_dtype, strided=strided
                    ):
                        outputs = (
                            torch.tensor(
                                [1e38, 1e38, float("nan"), 1e-40, 2.0**-126, -0.0],
                                dtype=dtype,
                                device="cuda",
                            )
                            .view(6, 1)
                            .expand(6, 257)
                            .clone()
                        )
                        weights = torch.tensor(
                            [[1e-40, -1e-40, float("nan")], [1e38, 0.5, 1.0]],
                            dtype=weight_dtype,
                            device="cuda",
                        )
                        ids = torch.tensor(
                            [[0, 1, -1], [2, 3, 4]],
                            dtype=torch.int64 if strided else torch.int32,
                            device="cuda",
                        )
                        if strided:
                            weights = weights.t().contiguous().t()
                            ids = ids.t().contiguous().t()
                        # FP64 exactly represents products of FP32 operands;
                        # round to FP32 before narrowing to the output dtype.
                        expected = outputs.cpu().double().view(2, 3, 257)
                        expected = expected * weights.cpu().double().unsqueeze(-1)
                        expected[ids.cpu() < 0] = 0
                        expected = expected.float().to(dtype).view_as(outputs).cuda()
                        self.assertGreater(float(expected[0, 0]), 1e-3)
                        self.assertGreater(float(expected[3, 0]), 1e-3)
                        self.assertNotEqual(float(expected[4, 0]), 0.0)
                        native(outputs, weights, ids)
                        torch.testing.assert_close(outputs, expected, rtol=0, atol=0)
                        self.assertTrue(
                            torch.equal(outputs.signbit(), expected.signbit())
                        )


if __name__ == "__main__":
    unittest.main()
