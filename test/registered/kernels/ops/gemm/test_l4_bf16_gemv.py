"""Opt-in L4 shape selection and numerical/graph regression tests."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.gemm import hopper_bf16_gemv as gemv
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# Selection checks run on existing CI hardware; L4 numerical coverage is local.
register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")


class FakeArch:
    def __init__(self, major, minor):
        self.major, self.minor = major, minor

    def is_sm(self, major, minor=None):
        return self.major == major and (minor is None or self.minor == minor)


class TestL4Selection(CustomTestCase):
    def test_reject_unsupported_device_and_deterministic_mode(self):
        from sglang.srt.layers.quantization import unquant as u

        config = SimpleNamespace(
            kernel=SimpleNamespace(bf16_gemm_backend="gemv"),
            deterministic=SimpleNamespace(enable_deterministic_inference=True),
        )
        with (
            patch.object(u, "get_exec", return_value=config),
            patch.multiple(
                u, _is_gfx95=False, _skip_aiter_unquant=False, _use_l4_bf16_gemv=False
            ),
            patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)),
        ):
            with patch.object(gemv, "is_l4_bf16_gemv_supported", return_value=True):
                with self.assertRaisesRegex(ValueError, "deterministic inference"):
                    u.initialize_bf16_gemm_config()
            with patch.object(gemv, "is_l4_bf16_gemv_supported", return_value=False):
                with self.assertRaisesRegex(ValueError, "requires SM90"):
                    u.initialize_bf16_gemm_config()

    def test_scope_and_existing_architectures(self):
        for major, minor, supported in [(8, 9, True), (8, 9, False), (8, 0, False)]:
            with (
                self.subTest(major=major, minor=minor, supported=supported),
                patch.object(gemv, "current_arch", return_value=FakeArch(major, minor)),
                patch.object(gemv, "is_l4_bf16_gemv_supported", return_value=supported),
            ):
                self.assertEqual(gemv.use_hopper_bf16_gemv(1, 2048, 4096), supported)
                for shape in [
                    (2, 2048, 4096),
                    (1, 2048, 4097),
                    (1, 2049, 4096),
                    (1, 4096, 4096),
                ]:
                    self.assertFalse(gemv.use_hopper_bf16_gemv(*shape))
        for arch in [(9, 0), (12, 0)]:
            with patch.object(gemv, "current_arch", return_value=FakeArch(*arch)):
                self.assertTrue(gemv.use_hopper_bf16_gemv(1, 2048, 4096))
                self.assertFalse(gemv.use_hopper_bf16_gemv(2, 2048, 4096))


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_name() == "NVIDIA L4",
    "L4 required",
)
class TestL4Numerics(CustomTestCase):
    def test_gradient_requests_fall_back(self):
        from sglang.srt.layers.quantization import unquant as u

        with torch.enable_grad():
            for input_grad, weight_grad in [(True, False), (False, True), (True, True)]:
                with self.subTest(input_grad=input_grad, weight_grad=weight_grad):
                    x = torch.randn(1, 4096, dtype=torch.bfloat16, device="cuda")
                    w = torch.randn(2048, 4096, dtype=torch.bfloat16, device="cuda")
                    x.requires_grad_(input_grad)
                    w.requires_grad_(weight_grad)
                    ref_x = x.detach().clone().requires_grad_(input_grad)
                    ref_w = w.detach().clone().requires_grad_(weight_grad)
                    with (
                        patch.multiple(
                            u,
                            _BF16_GEMM_BACKEND=u.Bf16GemmBackend.GEMV,
                            _use_l4_bf16_gemv=True,
                            _use_hopper_bf16_gemv=gemv.use_hopper_bf16_gemv,
                        ),
                        patch.object(
                            u, "_hopper_bf16_gemv", wraps=gemv.hopper_bf16_gemv
                        ) as call,
                    ):
                        result = u.UnquantizedLinearMethod().apply(
                            SimpleNamespace(weight=w), x, None
                        )
                    self.assertEqual(call.call_count, 0)
                    reference = torch.nn.functional.linear(ref_x, ref_w)
                    self.assertTrue(torch.equal(result, reference))
                    upstream = torch.randn_like(result)
                    result.backward(upstream)
                    reference.backward(upstream)
                    if input_grad:
                        self.assertTrue(torch.equal(x.grad, ref_x.grad))
                    if weight_grad:
                        self.assertTrue(torch.equal(w.grad, ref_w.grad))

    @torch.inference_mode()
    def test_linear_fallbacks(self):
        from sglang.srt.layers.quantization import unquant as u

        x = torch.randn(1, 4096, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(2048, 4096, dtype=torch.bfloat16, device="cuda")
        shifted_x = torch.empty(4097, dtype=x.dtype, device=x.device)[1:].view(1, 4096)
        shifted_x.copy_(x)
        shifted_w = torch.empty(w.numel() + 1, dtype=w.dtype, device=w.device)[
            1:
        ].view_as(w)
        shifted_w.copy_(w)
        strided_w = torch.empty(2048, 4097, dtype=w.dtype, device=w.device)[:, :4096]
        strided_w.copy_(w)
        cases = [
            ("eligible", x, w, None, 1),
            ("batch", x.repeat(2, 1), w, None, 0),
            ("prefill", x.repeat(256, 1), w, None, 0),
            ("shape", x, w[:1024], None, 0),
            ("3d", x.view(1, 1, 4096), w, None, 0),
            ("bias", x, w, torch.ones(2048, dtype=x.dtype, device=x.device), 0),
            ("offset", shifted_x, w, None, 0),
            ("weight_offset", x, shifted_w, None, 0),
            ("stride", x, strided_w, None, 0),
            ("fp32", x.float(), w.float(), None, 0),
        ]
        with (
            patch.multiple(
                u,
                _BF16_GEMM_BACKEND=u.Bf16GemmBackend.GEMV,
                _use_l4_bf16_gemv=True,
                _use_hopper_bf16_gemv=gemv.use_hopper_bf16_gemv,
            ),
            patch.object(u, "_hopper_bf16_gemv", wraps=gemv.hopper_bf16_gemv) as call,
        ):
            for name, inp, weight, bias, expected_calls in cases:
                with self.subTest(case=name):
                    call.reset_mock()
                    result = u.UnquantizedLinearMethod().apply(
                        SimpleNamespace(weight=weight), inp, bias
                    )
                    self.assertEqual(call.call_count, expected_calls)
                    if expected_calls == 0:
                        self.assertTrue(
                            torch.equal(
                                result, torch.nn.functional.linear(inp, weight, bias)
                            )
                        )

    @torch.inference_mode()
    def test_reference_and_graph(self):
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        for seed in range(4):
            for scale in [0.0, 0.01, 1.0, 10.0]:
                with self.subTest(seed=seed, scale=scale):
                    torch.manual_seed(seed)
                    x = (
                        torch.randn(1, 4096, device="cuda", dtype=torch.bfloat16)
                        * scale
                    )
                    w = (
                        torch.randn(2048, 4096, device="cuda", dtype=torch.bfloat16)
                        * 0.05
                    )
                    y = gemv.hopper_bf16_gemv(x, w)
                    baseline = torch.nn.functional.linear(x, w)
                    self.assertTrue(torch.isfinite(y).all().item())
                    for dtype in (torch.float32, torch.float64):
                        ref = torch.nn.functional.linear(x.to(dtype), w.to(dtype))
                        error = (y.to(dtype) - ref).abs().max().item()
                        baseline_error = (baseline.to(dtype) - ref).abs().max().item()
                        self.assertLessEqual(error, max(2 * baseline_error, 1e-2))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = gemv.hopper_bf16_gemv(x, w)
        x.zero_()
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(output.count_nonzero().item(), 0)
        x.fill_(1)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(output, gemv.hopper_bf16_gemv(x, w)))


if __name__ == "__main__":
    unittest.main()
