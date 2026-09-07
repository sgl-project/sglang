"""32-wide-K ue8m0 block-fp8 linears served by the FlashInfer MXFP8 dense kernels (opt-in).

A checkpoint whose dense weights carry one ue8m0 scale per 32x32 block and whose
activations are quantized per 32 along K with ue8m0 scales is an MXFP8 operand: the
block scale repeated over its 32 rows is the per-row [N, K // 32] e8m0 layout the MXFP8
GEMMs read. With an explicit `--fp8-gemm-backend flashinfer_*` on Blackwell,
Fp8LinearMethod routes such layers to that kernel instead of the Triton block kernel,
keeping the FlashInfer autotuner out of the tactic choice. These tests pin (1) the scale
expansion, (2) the opt-in dispatch conditions, (3) that the activation quantizer of the
MXFP8 path emits the same fp8 values and scales as the block path's, (4) that a row's
output does not depend on the batch it is computed in, and (5) that the linear's output
stays within the Triton path's distance of an fp32 reference at decode and prefill row
counts.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.srt.layers.quantization.fp8_utils import (
    Fp8GemmRunnerBackend,
    block_fp8_scale_to_mxfp8_e8m0,
    can_serve_block_fp8_as_mxfp8,
    resolve_block_fp8_mxfp8_backend,
    triton_w8a8_block_fp8_linear,
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
        cls._patch.stop()


class TestBlockScaleExpansion(CustomTestCase):
    def test_bit_exact_expansion_and_partial_block(self):
        n, k = 1000, 544  # N not a multiple of 32, K a multiple of 32
        scale = torch.exp2(
            torch.randint(-20, 20, ((n + 31) // 32, k // 32), device=DEVICE).float()
        )
        e8m0 = block_fp8_scale_to_mxfp8_e8m0(scale, (n, k), [BLOCK, BLOCK])
        self.assertEqual(tuple(e8m0.shape), (n, k // 32))
        self.assertEqual(e8m0.dtype, torch.uint8)
        expected = (
            (torch.log2(scale) + 127).to(torch.uint8).repeat_interleave(32, dim=0)[:n]
        )
        self.assertTrue(torch.equal(e8m0, expected))
        back = torch.exp2(e8m0.float() - 127)
        self.assertTrue(torch.equal(back, scale.repeat_interleave(32, dim=0)[:n]))

    def test_rejects_non_power_of_two_and_bad_block(self):
        scale = torch.full((4, 4), 1.5, device=DEVICE)
        with self.assertRaises(ValueError):
            block_fp8_scale_to_mxfp8_e8m0(scale, (128, 128), [BLOCK, BLOCK])
        with self.assertRaises(ValueError):
            block_fp8_scale_to_mxfp8_e8m0(
                torch.ones(1, 1, device=DEVICE), (128, 128), [128, 128]
            )


class TestDispatch(_OptInCase):
    def test_conditions(self):
        self.assertTrue(can_serve_block_fp8_as_mxfp8([BLOCK, BLOCK], "ue8m0"))
        self.assertTrue(resolve_block_fp8_mxfp8_backend().is_flashinfer_cutedsl())
        self.assertFalse(can_serve_block_fp8_as_mxfp8([128, 128], "ue8m0"))
        self.assertFalse(can_serve_block_fp8_as_mxfp8([BLOCK, BLOCK], None))
        self.assertFalse(can_serve_block_fp8_as_mxfp8(None, "ue8m0"))
        # opt-in only: auto and triton keep the Triton block kernel
        for off in (Fp8GemmRunnerBackend.AUTO, Fp8GemmRunnerBackend.TRITON):
            with patch.object(fp8_utils, "FP8_GEMM_RUNNER_BACKEND", off):
                self.assertFalse(can_serve_block_fp8_as_mxfp8([BLOCK, BLOCK], "ue8m0"))
                self.assertTrue(resolve_block_fp8_mxfp8_backend().is_unsupported())
        with patch.object(
            fp8_utils,
            "FP8_GEMM_RUNNER_BACKEND",
            Fp8GemmRunnerBackend.FLASHINFER_CUTLASS,
        ):
            self.assertTrue(resolve_block_fp8_mxfp8_backend().is_flashinfer_cutlass())

    def test_method_flags_and_layer_readiness(self):
        method = Fp8LinearMethod(_block32_config())
        self.assertTrue(method.block_fp8_as_mxfp8)
        self.assertFalse(method.use_mxfp8)
        self.assertTrue(method.mxfp8_dense_backend.is_flashinfer_cutedsl())
        w = torch.randn(256, 512, device=DEVICE, dtype=torch.bfloat16)
        layer = _build_layer(method, *_quant_block32(w))
        self.assertTrue(layer.block_fp8_mxfp8_ready)
        self.assertTrue(hasattr(layer, "weight_scale_inv_swizzled"))
        # the block scales survive for the Triton fallback and for direct readers
        self.assertEqual(tuple(layer.weight_scale_inv.shape), (8, 16))
        self.assertEqual(layer.weight_scale_inv.dtype, torch.float32)
        # a layer whose weight the model reads directly keeps the plain path
        method2 = Fp8LinearMethod(_block32_config())
        layer2 = torch.nn.Module()
        layer2.skip_aiter_bpreshuffle = True
        method2.create_weights(
            layer=layer2,
            input_size_per_partition=512,
            output_partition_sizes=[256],
            input_size=512,
            output_size=256,
            params_dtype=torch.bfloat16,
            skip_block_quant_check=True,
            weight_loader=lambda *a, **kw: None,
        )
        layer2 = layer2.to(DEVICE)
        q, s = _quant_block32(w)
        layer2.weight.load_column_parallel_weight(q, tp_rank=0)
        layer2.weight_scale_inv.load_column_parallel_weight(
            s.to(torch.float8_e8m0fnu), tp_rank=0
        )
        method2.process_weights_after_loading(layer2)
        self.assertFalse(layer2.block_fp8_mxfp8_ready)

    def test_off_by_default(self):
        with patch.object(
            fp8_utils, "FP8_GEMM_RUNNER_BACKEND", Fp8GemmRunnerBackend.AUTO
        ):
            method = Fp8LinearMethod(_block32_config())
            self.assertFalse(method.block_fp8_as_mxfp8)
            w = torch.randn(256, 512, device=DEVICE, dtype=torch.bfloat16)
            q, s = _quant_block32(w)
            layer = _build_layer(method, q, s)
            self.assertFalse(getattr(layer, "block_fp8_mxfp8_ready", False))
            x = torch.randn(3, 512, device=DEVICE, dtype=torch.bfloat16)
            out_triton = triton_w8a8_block_fp8_linear(
                x, q, [BLOCK, BLOCK], s, None, None, act_scale_ue8m0=True
            )
            self.assertTrue(torch.equal(method.apply(layer, x), out_triton))


class TestActivationQuantEquivalence(_OptInCase):
    """The MXFP8 path quantizes activations with FlashInfer's mxfp8_quantize; the block
    path with sglang's per-32 ue8m0 group quant. Same fp8 values, same scales wherever a
    block has a nonzero value (an all-zero block quantizes to zeros under any scale)."""

    def test_values_and_scales(self):
        from flashinfer import mxfp8_quantize

        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

        k = 1024
        cases = [
            torch.randn(64, k, device=DEVICE, dtype=torch.bfloat16) * s
            for s in (1e-3, 1.0, 1e3)
        ]
        x = torch.randn(8, k, device=DEVICE, dtype=torch.bfloat16)
        x[:, ::32] = (
            448.0 * torch.exp2(torch.arange(-4, 4, device=DEVICE).repeat(4).float())
        ).to(torch.bfloat16)
        cases.append(x)  # block amax exactly at 448 * 2^e
        x = torch.randn(8, k, device=DEVICE, dtype=torch.bfloat16)
        x[:, ::32] = torch.exp2(
            torch.arange(-8, 8, device=DEVICE).repeat(2).float()
        ).to(torch.bfloat16)
        x[:, 1::32] = 0
        cases.append(x)  # block amax exactly a power of two
        x = torch.randn(8, k, device=DEVICE, dtype=torch.bfloat16)
        x[2] = 0
        x[:, :32] = 0
        cases.append(x)  # zero row and zero block
        x = torch.randn(8, k, device=DEVICE, dtype=torch.bfloat16)
        x[0, 5] = 3e4
        cases.append(x)  # outlier
        for x in cases:
            q_s, s_s = sglang_per_token_group_quant_fp8(x, BLOCK, scale_ue8m0=True)
            q_f, s_f = mxfp8_quantize(x, is_sf_swizzled_layout=False, alignment=BLOCK)
            self.assertTrue(torch.equal(q_s.view(torch.uint8), q_f.view(torch.uint8)))
            e_s = (s_s.view(x.shape[0], -1).float().view(torch.int32) >> 23).to(
                torch.uint8
            )
            e_f = s_f.view(torch.uint8).reshape(x.shape[0], k // BLOCK)
            nonzero = x.view(x.shape[0], k // BLOCK, BLOCK).abs().amax(-1) > 0
            self.assertTrue(torch.equal(e_s[nonzero], e_f[nonzero]))


class TestBatchInvariance(_OptInCase):
    """A row's output must not depend on the batch it is computed in (the gate that
    keeps identical requests bitwise identical across batch shapes). Row 0 alone against
    row 0 inside batches of every size up to 64 (the decode regime), eager and under
    CUDA-graph capture at the decode graph sizes."""

    def test_row_alone_equals_row_in_batch(self):
        method = Fp8LinearMethod(_block32_config())
        for n, k in [(1792, 5120), (5120, 576)]:
            w = torch.randn(n, k, device=DEVICE, dtype=torch.bfloat16) / (k**0.5)
            layer = _build_layer(method, *_quant_block32(w))
            x0 = torch.randn(1, k, device=DEVICE, dtype=torch.bfloat16)
            solo = method.apply(layer, x0)[0]
            for m in range(2, 65):
                xb = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
                xb[0] = x0[0]
                self.assertTrue(
                    torch.equal(method.apply(layer, xb)[0], solo), (n, k, m)
                )
            for m in (1, 4):
                xb = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
                xb[0] = x0[0]
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    method.apply(layer, xb)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        out = method.apply(layer, xb)
                torch.cuda.synchronize()
                graph.replay()
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(out[0], solo), (n, k, "graph", m))


class TestLinearNumerics(_OptInCase):
    """The layer's output against the Triton block path and an fp32 reference built from the
    same fp8 activations and weights, at decode and prefill row counts and at the dense
    projection shapes of a 40-layer TP4 deployment."""

    SHAPES = [(1792, 5120), (8192, 1280), (5120, 2048), (1152, 5120), (5120, 576)]
    MS = [1, 7, 32, 300, 4096]

    def test_against_triton_and_reference(self):
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

        method = Fp8LinearMethod(_block32_config())
        for n, k in self.SHAPES:
            w = torch.randn(n, k, device=DEVICE, dtype=torch.bfloat16) / (k**0.5)
            q, s = _quant_block32(w)
            layer = _build_layer(method, q, s)
            self.assertTrue(layer.block_fp8_mxfp8_ready, (n, k))
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
                out_triton = triton_w8a8_block_fp8_linear(
                    x, q, [BLOCK, BLOCK], s, None, None, act_scale_ue8m0=True
                )
                amax = ref.abs().max().item()
                err_mx = (out.float() - ref).abs().max().item() / amax
                err_tr = (out_triton.float() - ref).abs().max().item() / amax
                diff = (out.float() - out_triton.float()).abs().max().item() / amax
                # both paths differ from the fp32 reference only by bf16 output rounding
                # and fp32 accumulation order (~1e-2 is one bf16 ulp near the max)
                self.assertLess(
                    err_mx, max(1.5 * err_tr, 1e-2), (n, k, m, err_mx, err_tr)
                )
                self.assertLess(diff, 8e-3, (n, k, m, diff))


if __name__ == "__main__":
    unittest.main()
