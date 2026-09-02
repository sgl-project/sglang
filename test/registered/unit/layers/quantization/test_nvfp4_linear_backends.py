"""Numerics for the NVFP4 dense-linear GEMM backends (--fp4-gemm-backend).

Real layer path (ColumnParallelLinear -> weight_loader -> weight processing
-> forward) per SM100 backend vs a dequantized-reference matmul; a merged
two-shard case guards the per-partition scale gathering of fused layers.
"""

import unittest
from unittest import mock

import torch
from flashinfer import fp4_quantize

from sglang.srt.layers.quantization import fp4_utils
from sglang.srt.layers.quantization.fp4_utils import Fp4GemmRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.layer_ut_utils import (
    assert_output_close,
    init_single_process_dist,
    load_linear_weights,
    make_tp1_column_parallel_linear,
)
from sglang.test.quant_ref_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    dequantize_nvfp4_to_dtype,
    quantize_nvfp4_shard,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b", runner_config="4-gpu-b200")

# (M, N, K). The second shape hits the padding paths: N=160 is not a multiple
# of 128 (TRTLLM shuffle pad) and K=336 is neither a multiple of 32 (CUTLASS
# K pad) nor K/16 a multiple of 4 (TRTLLM scale pad).
SHAPES = [
    (64, 256, 512),
    (5, 160, 336),
    (128, 1024, 1024),
]

ACT_SCALE = 1.0 / (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX)


def _make_quantized_layer(n: int, k: int):
    """NVFP4 checkpoint-format weights through the real weight_loader."""
    quant_config = ModelOptFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        group_size=16,
        use_per_token_activation=False,
        packed_modules_mapping={},
    )
    layer = make_tp1_column_parallel_linear(quant_config, n, k)

    w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / 10
    w_q, sf_linear, gs, w_dequant = quantize_nvfp4_shard(w)
    load_linear_weights(
        layer,
        weight=w_q,
        weight_scale=sf_linear,
        weight_scale_2=(1.0 / gs).clone(),
        # Calibrated activation amax stand-in (inputs are randn/10).
        input_scale=torch.tensor(ACT_SCALE, device="cuda"),
    )
    return layer, w_dequant


def _make_merged_layer(n_half: int, k: int):
    """Two fused output shards (gate_up_proj) loaded per shard; exercises the
    per-partition scale_2 / input_scale gathering that fused-QKV regressions hit."""
    from sglang.srt.layers.linear import MergedColumnParallelLinear

    quant_config = ModelOptFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        group_size=16,
        use_per_token_activation=False,
        packed_modules_mapping={"gate_up_proj": ["gate_proj", "up_proj"]},
    )
    layer = MergedColumnParallelLinear(
        input_size=k,
        output_sizes=[n_half, n_half],
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=quant_config,
        prefix="model.layers.0.mlp.gate_up_proj",
        tp_rank=0,
        tp_size=1,
    ).cuda()

    # process_weights_after_loading collapses shard scale_2 with max() without
    # requanting block scales, so shards must share one gs (modelopt fused
    # exports ship equal scale_2).
    shards = [
        torch.randn((n_half, k), device="cuda", dtype=torch.bfloat16) / 10
        for _ in (0, 1)
    ]
    shared_gs = (
        FLOAT8_E4M3_MAX
        * FLOAT4_E2M1_MAX
        / max(w.abs().max().to(torch.float32) for w in shards)
    )
    dequants = []
    for shard_id, w in enumerate(shards):
        w_q, sf_linear, gs, w_dequant = quantize_nvfp4_shard(w, gs=shared_gs)
        load_linear_weights(
            layer,
            shard_id=shard_id,
            weight=w_q,
            weight_scale=sf_linear,
            weight_scale_2=(1.0 / gs).clone(),
            input_scale=torch.tensor(ACT_SCALE, device="cuda"),
        )
        dequants.append(w_dequant)
    return layer, torch.cat(dequants, dim=0)


@unittest.skipIf(get_device_sm() < 100, "NVFP4 dense GEMM backends require SM100+")
class TestNvFp4LinearBackends(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist()

    def _run_backend(self, backend: str, build_layer=_make_quantized_layer):
        torch.manual_seed(7)
        for m, n, k in SHAPES:
            with self.subTest(backend=backend, shape=(m, n, k)):
                with mock.patch.object(
                    fp4_utils,
                    "FP4_GEMM_RUNNER_BACKEND",
                    Fp4GemmRunnerBackend(backend),
                ):
                    layer, w_dequant = build_layer(n, k)
                    layer.quant_method.process_weights_after_loading(layer)

                    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / 10
                    out, _ = layer(x)
                    self._assert_matches(layer, x, out, w_dequant)

    def _assert_matches(self, layer, x, out, w_dequant):
        x_gs = layer.input_scale_inv.data.float()
        x_q, x_sf = fp4_quantize(x, x_gs)
        x_dequant = dequantize_nvfp4_to_dtype(x_q, x_sf, x_gs, torch.float32)
        ref = x_dequant @ w_dequant.T
        assert_output_close(self, out, ref, rtol=5e-2, atol=5e-2)

    def test_merged_shards(self):
        torch.manual_seed(7)
        with mock.patch.object(
            fp4_utils,
            "FP4_GEMM_RUNNER_BACKEND",
            Fp4GemmRunnerBackend("flashinfer_cutedsl"),
        ):
            layer, w_dequant = _make_merged_layer(256, 512)
            layer.quant_method.process_weights_after_loading(layer)
            x = torch.randn((16, 512), device="cuda", dtype=torch.bfloat16) / 10
            out, _ = layer(x)
            self._assert_matches(layer, x, out, w_dequant)

    def test_flashinfer_cutedsl(self):
        self._run_backend("flashinfer_cutedsl")

    def test_flashinfer_cutlass(self):
        self._run_backend("flashinfer_cutlass")

    def test_flashinfer_cudnn(self):
        self._run_backend("flashinfer_cudnn")

    def test_flashinfer_trtllm(self):
        self._run_backend("flashinfer_trtllm")


if __name__ == "__main__":
    unittest.main()
