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
