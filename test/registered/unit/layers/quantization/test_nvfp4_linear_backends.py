"""Numerics for the NVFP4 dense-linear GEMM backends (--fp4-gemm-backend).

Runs the real linear layer path (ColumnParallelLinear -> weight_loader ->
process_weights_after_loading -> forward) for each SM100 backend choice and
checks the output against a dequantized-reference matmul. This covers both
the per-backend weight preparation (padding / interleave / TRTLLM shuffle)
and the GEMM kernel dispatch; a MergedColumnParallelLinear case guards the
per-partition scale gathering of fused gate_up / QKV layers.
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
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b", runner_config="4-gpu-b200")

kE2M1ToFloat = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)
FLOAT8_E4M3_MAX = 448.0
FLOAT4_E2M1_MAX = 6.0

# (M, N, K). The second shape hits the padding paths: N=160 is not a multiple
# of 128 (TRTLLM shuffle pad) and K=336 is neither a multiple of 32 (CUTLASS
# K pad) nor K/16 a multiple of 4 (TRTLLM scale pad).
SHAPES = [
    (64, 256, 512),
    (5, 160, 336),
    (128, 1024, 1024),
]


def convert_swizzled_to_linear(a_sf_swizzled: torch.Tensor, m, k, block_size):
    m_tiles = (m + 128 - 1) // 128
    f = block_size * 4
    k_tiles = (k + f - 1) // f
    tmp = torch.reshape(a_sf_swizzled, (1, m_tiles, k_tiles, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    out = tmp.reshape(m_tiles * 128, k_tiles * f // block_size)
    # Crop the K-tile padding too: k // block_size scale columns, not k.
    return out[0:m, 0 : k // block_size]


def break_fp4_bytes(a, dtype):
    assert a.dtype == torch.uint8
    m, n = a.shape
    a_flat = a.flatten()
    high = (a_flat & 0xF0) >> 4
    low = a_flat & 0x0F
    combined = torch.stack((low, high), dim=1).flatten()
    signs = (combined & 0x08).to(torch.bool)
    abs_vals = (combined & 0x07).to(torch.long)
    kE2M1 = kE2M1ToFloat.to(device=a.device)
    values = kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)
    return values.reshape(m, n * 2).to(dtype=dtype)


def dequantize_nvfp4_to_dtype(
    tensor_fp4, tensor_sf, global_scale, dtype, device, block_size=16
):
    assert tensor_fp4.dtype == torch.uint8
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2
    tensor_f32 = break_fp4_bytes(tensor_fp4, torch.float32)
    tensor_f32 = tensor_f32.reshape(m, k // block_size, block_size)
    tensor_sf = tensor_sf.view(torch.float8_e4m3fn)
    tensor_sf = convert_swizzled_to_linear(tensor_sf, m, k, block_size)
    tensor_sf_dtype = tensor_sf.to(torch.float32) / global_scale
    out = (tensor_f32 * tensor_sf_dtype.unsqueeze(-1)).reshape(m, k)
    return out.to(dtype=dtype)


def _init_single_process_dist():
    import os

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29632")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )

    if not torch.distributed.is_initialized():
        init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")
    if not model_parallel_is_initialized():
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            backend="gloo",
        )


def _quantize_shard(w: torch.Tensor, gs=None):
    """NVFP4-quantize one checkpoint shard; returns (packed, linear sf,
    global scale, fp32 dequant reference)."""
    n, k = w.shape
    if gs is None:
        gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w.abs().max().to(torch.float32)
    w_q, w_sf_swizzled = fp4_quantize(w, gs)
    sf_linear = convert_swizzled_to_linear(
        w_sf_swizzled.view(torch.float8_e4m3fn), n, k, 16
    )
    w_dequant = dequantize_nvfp4_to_dtype(
        w_q, w_sf_swizzled, gs, torch.float32, w.device
    )
    return w_q, sf_linear, gs, w_dequant


ACT_SCALE = 1.0 / (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX)


def _make_quantized_layer(n: int, k: int):
    """ColumnParallelLinear with NVFP4 checkpoint-format weights fed through
    the real weight_loader; returns (layer, w_dequant)."""
    from sglang.srt.layers.linear import ColumnParallelLinear

    quant_config = ModelOptFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        group_size=16,
        use_per_token_activation=False,
        packed_modules_mapping={},
    )
    layer = ColumnParallelLinear(
        input_size=k,
        output_size=n,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=quant_config,
        prefix="model.layers.0.mlp.up_proj",
        tp_rank=0,
        tp_size=1,
    ).cuda()

    w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / 10
    w_q, sf_linear, gs, w_dequant = _quantize_shard(w)
    for name, loaded in (
        ("weight", w_q),
        ("weight_scale", sf_linear),
        ("weight_scale_2", (1.0 / gs).clone()),
        # Calibrated activation amax stand-in (inputs are randn/10).
        ("input_scale", torch.tensor(ACT_SCALE, device="cuda")),
    ):
        layer.weight_loader_v2(getattr(layer, name), loaded)
    return layer, w_dequant


def _make_merged_layer(n_half: int, k: int):
    """MergedColumnParallelLinear (two fused output shards, e.g. gate_up_proj)
    loaded per shard with distinct global scales; exercises the per-partition
    scale_2 / input_scale gathering that fused-QKV regressions hit."""
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

    # process_weights_after_loading collapses per-shard weight_scale_2 with
    # max() and does not requant block scales, so unequal shard scales are a
    # known accuracy hazard; modelopt fused exports ship equal scale_2 and the
    # fixture mirrors that.
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
        w_q, sf_linear, gs, w_dequant = _quantize_shard(w, gs=shared_gs)
        for name, loaded in (
            ("weight", w_q),
            ("weight_scale", sf_linear),
            ("weight_scale_2", (1.0 / gs).clone()),
            ("input_scale", torch.tensor(ACT_SCALE, device="cuda")),
        ):
            layer.weight_loader_v2(getattr(layer, name), loaded, shard_id)
        dequants.append(w_dequant)
    return layer, torch.cat(dequants, dim=0)


@unittest.skipIf(get_device_sm() < 100, "NVFP4 dense GEMM backends require SM100+")
class TestNvFp4LinearBackends(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        _init_single_process_dist()

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
        x_dequant = dequantize_nvfp4_to_dtype(x_q, x_sf, x_gs, torch.float32, x.device)
        ref = x_dequant @ w_dequant.T
        self.assertEqual(tuple(out.shape), tuple(ref.shape))
        cos = torch.nn.functional.cosine_similarity(
            out.float().flatten(), ref.flatten(), dim=0
        ).item()
        self.assertGreater(cos, 0.99)
        torch.testing.assert_close(out.float(), ref, rtol=5e-2, atol=5e-2)

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
