import unittest
import warnings

import torch
import torch.nn.functional as F

from sglang.srt.layers.muse_spark.muse_spark_layers import MuseSpark1xRMSNorm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestMuseSparkFloatWeightNorm(CustomTestCase):
    def setUp(self):
        torch.manual_seed(123)
        warnings.filterwarnings(
            "ignore", message="Mismatch dtype between input and weight"
        )

    def make_norm(self, hidden, dtype, centered=False, device="cuda", has_weight=True):
        norm = MuseSpark1xRMSNorm(
            hidden, eps=1e-6, zero_centered_gamma=centered, has_weight=has_weight
        ).to(device=device, dtype=dtype)
        if norm.weight is not None:
            with torch.no_grad():
                norm.weight.uniform_(-0.2, 0.2)
                if not centered:
                    norm.weight.add_(1)
        norm.precompute_effective_weight()
        return norm

    @torch.inference_mode()
    def test_float_streams_and_effective_weights(self):
        for n, hidden in ((1, 64), (8, 128), (128, 8192), (2048, 8192)):
            for weight_dtype in (torch.bfloat16, torch.float16):
                for centered in (False, True):
                    norm = self.make_norm(hidden, weight_dtype, centered)
                    weight = norm._get_effective_weight().clone()
                    for scale in (0.0, 1e-5, 1.0, 1000.0):
                        x = (
                            torch.randn(n, hidden, device="cuda", dtype=torch.float32)
                            * scale
                        )
                        actual = norm(x)
                        expected = F.rms_norm(x, (hidden,), weight, norm.eps)
                        precise = F.rms_norm(
                            x.double(), (hidden,), weight.double(), norm.eps
                        ).float()
                        self.assertEqual(actual.dtype, x.dtype)
                        torch.testing.assert_close(
                            actual, expected, rtol=2e-6, atol=1e-6
                        )
                        torch.testing.assert_close(
                            actual, precise, rtol=2e-6, atol=1e-6
                        )
                    torch.testing.assert_close(
                        norm._get_effective_weight(), weight, rtol=0, atol=0
                    )

    @torch.inference_mode()
    def test_existing_dtype_cpu_and_empty_paths(self):
        for device in ("cpu", "cuda"):
            for dtype in (torch.bfloat16, torch.float16, torch.float32):
                for has_weight in (False, True):
                    norm = self.make_norm(
                        128, dtype, device=device, has_weight=has_weight
                    )
                    x = torch.randn(8, 128, device=device, dtype=dtype)
                    expected = F.rms_norm(
                        x, (128,), norm._get_effective_weight(), norm.eps
                    ).to(dtype)
                    torch.testing.assert_close(norm(x), expected, rtol=0, atol=0)
                    empty = torch.empty(0, 128, device=device, dtype=dtype)
                    self.assertIs(norm(empty), empty)

    @torch.inference_mode()
    def test_graph_replay_reads_new_inputs_and_weights(self):
        norm = self.make_norm(8192, torch.bfloat16)
        x = torch.randn(128, 8192, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                norm(x)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = norm(x)
        for scale in (0.25, 1.0, 4.0):
            x.normal_().mul_(scale)
            norm.weight.uniform_(0.1, 2.0)
            graph.replay()
            expected = F.rms_norm(x, (8192,), norm.weight, norm.eps)
            torch.testing.assert_close(out, expected, rtol=2e-6, atol=1e-6)
        graph.reset()

    def test_autograd_remains_connected(self):
        norm = self.make_norm(128, torch.bfloat16)
        x = torch.randn(8, 128, device="cuda", requires_grad=True)
        y = norm(x)
        y.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(norm.weight.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())
        self.assertTrue(torch.isfinite(norm.weight.grad).all().item())


if __name__ == "__main__":
    unittest.main()
