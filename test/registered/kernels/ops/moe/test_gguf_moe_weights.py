"""GGUF route weighting preserves per-slot rounding and skips inactive data."""

import unittest

import sgl_kernel  # noqa: F401
import torch

from sglang.kernels.ops.moe.gguf_moe_weights_aot import resolve_gguf_moe_weights
from sglang.srt.utils import is_musa
from sglang.test.ci.ci_register import register_cuda_ci, register_musa_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=6, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_musa_ci(est_time=6, stage="base-b", runner_config="1-gpu-musa")


class TestGgufMoeWeights(CustomTestCase):
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
