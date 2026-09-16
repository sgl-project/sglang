"""WO-A quantized handoff preserves native WO-B output and graph inputs."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "requires gfx950")
class TestWoAMxfp8Epilogue(unittest.TestCase):
    def setUp(self):
        from sglang.srt.models.deepseek_v4 import _apply_wo_a_bf16_matmul
        from sglang.srt.runtime_context import get_context

        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)
        self.project = _apply_wo_a_bf16_matmul
        torch.manual_seed(39186)

    def operands(self, rows):
        x = torch.randn(rows, 4, 4096, dtype=torch.bfloat16, device="cuda")[:, 1:3]
        w = torch.randn(2, 1024, 4096, dtype=torch.bfloat16, device="cuda") * 0.015625
        return x, w

    def quant_reference(self, y):
        groups = y.float().reshape(y.shape[0], 64, 32)
        amax = groups.abs().amax(-1).clamp(min=1e-10)
        exponent = (torch.ceil(torch.log2(amax / 448.0)) + 127).clamp(1, 254)
        scale = torch.exp2(exponent - 127)
        q = (groups / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        return q.reshape(y.shape[0], 2048), exponent.to(torch.uint8)

    @torch.inference_mode()
    def test_strided_mutable_graph_quantization(self):
        from sglang.kernels.ops.attention.dsv4.wo_a_bf16 import wo_a_bf16_small_batch
        from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import Mxfp8Activation
        from sglang.srt.environ import envs

        with envs.SGLANG_OPT_HIP_WO_A_MXFP8_EPILOGUE.override(True):
            for rows in range(2, 9):
                with self.subTest(rows=rows):
                    x, w = self.operands(rows)

                    def project():
                        return self.project(
                            x, w, is_decode=True, is_target_verify=True, fp8_grid=True
                        )

                    project()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = project()
                    self.assertIsInstance(output, Mxfp8Activation)
                    for magnitude in (0.0, 1e-12, 1.0, 128.0):
                        x.normal_().mul_(magnitude)
                        w.normal_(std=0.015625)
                        graph.replay()
                        q, scales = self.quant_reference(wo_a_bf16_small_batch(x, w))
                        torch.testing.assert_close(
                            output.q.view(torch.uint8),
                            q.view(torch.uint8),
                            rtol=0,
                            atol=0,
                        )
                        torch.testing.assert_close(output.scale, scales, rtol=0, atol=0)

    @torch.inference_mode()
    def test_native_consumer_matches_bf16_handoff(self):
        from sglang.kernels.ops.quantization.mxfp8_native_amd_gfx95 import (
            mxfp8_native_blockscaled_linear,
            prepare_mxfp8_native_weight,
        )
        from sglang.srt.environ import envs

        weight = torch.randn(5120, 2048, device="cuda").to(torch.float8_e4m3fn)
        scale = torch.full((160, 64), 0.015625, device="cuda")
        qw, qs, wb = prepare_mxfp8_native_weight(weight, scale, [32, 32])
        for rows in (2, 4, 6, 8):
            with self.subTest(rows=rows):
                x, w = self.operands(rows)
                with envs.SGLANG_OPT_HIP_WO_A_MXFP8_EPILOGUE.override(False):
                    y = self.project(
                        x, w, is_decode=True, is_target_verify=True, fp8_grid=True
                    )
                expected = mxfp8_native_blockscaled_linear(y.flatten(1), qw, qs, wb)
                with envs.SGLANG_OPT_HIP_WO_A_MXFP8_EPILOGUE.override(True):
                    operand = self.project(
                        x, w, is_decode=True, is_target_verify=True, fp8_grid=True
                    )
                actual = mxfp8_native_blockscaled_linear(
                    operand.q, qw, qs, wb, input_scale=operand.scale
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_unqualified_consumers_keep_bf16(self):
        from sglang.srt.environ import envs
        from sglang.srt.runtime_context import get_forward

        x, w = self.operands(6)
        for reason in ("disabled", "consumer", "sequence_parallel"):
            with (
                self.subTest(reason=reason),
                envs.SGLANG_OPT_HIP_WO_A_MXFP8_EPILOGUE.override(reason != "disabled"),
                get_forward().scoped(sp_active=reason == "sequence_parallel"),
                patch(
                    "sglang.kernels.ops.attention.dsv4.wo_a_bf16_hip.wo_a_bf16_small_batch_mxfp8_hip",
                    side_effect=AssertionError("unexpected quantized handoff"),
                ),
            ):
                output = self.project(
                    x,
                    w,
                    is_decode=True,
                    is_target_verify=True,
                    fp8_grid=reason != "consumer",
                )
                self.assertIsInstance(output, torch.Tensor)
                self.assertEqual(output.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
