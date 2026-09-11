"""Check the MXFP8 linear layer against an FP32 dequantization reference."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.srt.layers.quantization.fp8_utils import (
    Fp8GemmRunnerBackend,
    can_serve_block_fp8_as_mxfp8,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-c", runner_config="4-gpu-gb300")

BLOCK = 32
DEVICE = "cuda"
OPT_IN = Fp8GemmRunnerBackend.FLASHINFER_CUTEDSL
SKIP_MSG = "MXFP8 dense kernels unavailable (needs Blackwell + FlashInfer)"


def _opt_in():
    """Select the FlashInfer cute-dsl MXFP8 kernel the way `--fp8-gemm-backend` would."""
    return patch.object(fp8_utils, "FP8_GEMM_RUNNER_BACKEND", OPT_IN)


def _mxfp8_available():
    with _opt_in():
        return can_serve_block_fp8_as_mxfp8([BLOCK, BLOCK], "ue8m0")


def _quant_block32(w: torch.Tensor):
    """fp8 e4m3 weight with one ue8m0 scale per 32x32 block (ceil rule), like the checkpoint."""
    n, k = w.shape
    blocks = w.float().view(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-30)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    q = (blocks / scale).clamp(-448, 448).to(torch.float8_e4m3fn).view(n, k)
    return q, scale.view(n // BLOCK, k // BLOCK)


def _dequant_block32(q: torch.Tensor, scale: torch.Tensor):
    n, k = q.shape
    return (
        q.float().view(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
        * scale.view(n // BLOCK, 1, k // BLOCK, 1)
    ).view(n, k)


def _block32_config():
    return Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[BLOCK, BLOCK],
        scale_fmt="ue8m0",
    )


def _build_layer(method: Fp8LinearMethod, q: torch.Tensor, scale: torch.Tensor):
    """Create the layer through the linear method and load the checkpoint tensors through
    the parameters' own loaders (the scale arrives as e8m0, the way the checkpoint stores it)."""
    n, k = q.shape
    layer = torch.nn.Module()
    method.create_weights(
        layer=layer,
        input_size_per_partition=k,
        output_partition_sizes=[n],
        input_size=k,
        output_size=n,
        params_dtype=torch.bfloat16,
        skip_block_quant_check=True,  # no parallel state in a unit test
        weight_loader=lambda *a, **kw: None,
    )
    layer = layer.to(DEVICE)
    layer.weight.load_column_parallel_weight(q, tp_rank=0)
    layer.weight_scale_inv.load_column_parallel_weight(
        scale.to(torch.float8_e8m0fnu), tp_rank=0
    )
    method.process_weights_after_loading(layer)
    return layer


class _OptInCase(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not _mxfp8_available():
            raise unittest.SkipTest(SKIP_MSG)
        cls._patch = _opt_in()
        cls._patch.start()
        torch.manual_seed(0)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "_patch"):
            cls._patch.stop()


class TestLinearNumerics(_OptInCase):
    """Check decode and prefill outputs at the deployed dense projection shapes."""

    SHAPES = [(1792, 5120), (5120, 576)]
    MS = [1, 17, 300]

    def test_fused_wo_a_quantized_input(self):
        from flashinfer import mxfp8_quantize

        from sglang.kernels.ops.attention.dsv4.wo_a_bf16_small_batch import (
            _quantize_partial,
            _wo_a_reduce,
            wo_a_bf16_small_batch,
            wo_a_bf16_small_batch_mxfp8,
        )
        from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput

        method = Fp8LinearMethod(_block32_config())
        w = torch.randn(5120, 2048, device=DEVICE, dtype=torch.bfloat16) / 2048**0.5
        qweight, scale = _quant_block32(w)
        layer = _build_layer(method, qweight, scale)
        for rows in range(2, 9):
            for magnitude in (0.0, 1e-37, 1e-7, 1.0, 448.0, 1e10):
                partial = torch.randn(8, rows, 2, 1024, device=DEVICE) * magnitude
                bf16 = torch.empty(rows, 2048, dtype=torch.bfloat16, device=DEVICE)
                _wo_a_reduce[(rows * 8,)](partial, bf16, rows * 2048, num_warps=4)
                expected_q, expected_s = mxfp8_quantize(bf16, True, alignment=32)
                actual_q, actual_s = _quantize_partial(partial)
                torch.testing.assert_close(
                    actual_q.view(torch.uint8),
                    expected_q.view(torch.uint8),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(actual_s, expected_s, rtol=0, atol=0)
                actual = method.apply(layer, Mxfp8SwizzledInput(actual_q, actual_s))
                expected = method.apply(layer, bf16)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

            x = torch.randn(rows, 64, 512, device=DEVICE, dtype=torch.bfloat16)[
                :, :16
            ].view(rows, 2, 4096)
            wo_a = (
                torch.randn(2, 1024, 4096, device=DEVICE, dtype=torch.bfloat16) * 0.02
            )
            bf16 = wo_a_bf16_small_batch(x, wo_a).flatten(1)
            q, s = wo_a_bf16_small_batch_mxfp8(x, wo_a)
            expected_q, expected_s = mxfp8_quantize(bf16, True, alignment=32)
            torch.testing.assert_close(
                q.view(torch.uint8), expected_q.view(torch.uint8), rtol=0, atol=0
            )
            torch.testing.assert_close(s, expected_s, rtol=0, atol=0)

        # Ordinary block-FP8 tuples must retain their existing dispatch path.
        sentinel = torch.empty(0)
        with patch.object(
            method, "w8a8_block_fp8_linear", return_value=sentinel
        ) as legacy:
            self.assertIs(method.apply(layer, (q, s)), sentinel)
            legacy.assert_called_once()

        partial = torch.randn(8, 6, 2, 1024, device=DEVICE)
        _quantize_partial(partial)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            q, s = _quantize_partial(partial)
            output = method.apply(layer, Mxfp8SwizzledInput(q, s))
        for _ in range(3):
            partial.normal_()
            # The replay must regenerate scale padding as well as live rows.
            s.fill_(255)
            graph.replay()
            bf16 = torch.empty(6, 2048, dtype=torch.bfloat16, device=DEVICE)
            _wo_a_reduce[(48,)](partial, bf16, 6 * 2048, num_warps=4)
            eq, es = mxfp8_quantize(bf16, True, alignment=32)
            torch.testing.assert_close(
                q.view(torch.uint8), eq.view(torch.uint8), rtol=0, atol=0
            )
            torch.testing.assert_close(s, es, rtol=0, atol=0)
            torch.testing.assert_close(
                output, method.apply(layer, bf16), rtol=0, atol=0
            )

    def test_against_dequantized_reference(self):
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

        method = Fp8LinearMethod(_block32_config())
        for n, k in self.SHAPES:
            w = torch.randn(n, k, device=DEVICE, dtype=torch.bfloat16) / (k**0.5)
            q, s = _quant_block32(w)
            layer = _build_layer(method, q, s)
            w_deq = _dequant_block32(q, s)
            for m in self.MS:
                x = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
                xq, xs = sglang_per_token_group_quant_fp8(x, BLOCK, scale_ue8m0=True)
                x_deq = (
                    xq.float().view(m, k // BLOCK, BLOCK)
                    * xs.view(m, k // BLOCK, 1).float()
                )
                ref = x_deq.view(m, k) @ w_deq.t()
                out = method.apply(layer, x)
                self.assertEqual(out.dtype, torch.bfloat16)
                amax = ref.abs().max().item()
                error = (out.float() - ref).abs().max().item() / amax
                self.assertLess(error, 1e-2, (n, k, m, error))


if __name__ == "__main__":
    unittest.main()
