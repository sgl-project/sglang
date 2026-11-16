"""MXFP8 W8A8 dense linear on XPU, served both ways.

An MXFP8 checkpoint (e4m3 weights, UE8M0 block-32 scales) reaches the XPU GEMM
through two quantization methods: the compressed-tensors ``mxfp8-quantized``
scheme (``CompressedTensorsW8A8MXFp8``) and the native ``fp8`` method with
``use_mxfp8`` (``Fp8LinearMethod``, ``Mxfp8DenseGemmBackend.XPU_SCALED_MM``).
Both must load the same checkpoint tensors and produce the same output.
"""

import sys
from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch

from sglang.srt.utils import is_xpu
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=20, suite="stage-b-test-1-gpu-xpu")

if is_xpu():
    from compressed_tensors.quantization import QuantizationArgs

    from sglang.srt.layers.quantization import fp8_utils
    from sglang.srt.layers.quantization.compressed_tensors.schemes import (
        CompressedTensorsW8A8MXFp8,
    )
    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
    from sglang.srt.runtime_context import SpawnRanks, publish, reset_context
    from sglang.srt.server_args import prepare_server_args

pytestmark = pytest.mark.skipif(not is_xpu(), reason="XPU required")

DEVICE = "xpu"
GROUP_SIZE = 32
# Dynamic MXFP8 activation quantization against a bf16 matmul on dequantized weights.
REL_TOL = 0.05


def _mxfp8_quantize(w: torch.Tensor):
    """[N, K] -> e4m3 [N, K] and UE8M0 scale bytes [N, K // 32]."""
    blocks = w.float().reshape(w.shape[0], -1, GROUP_SIZE)
    amax = blocks.abs().amax(-1).clamp(min=1e-12)
    exp = torch.ceil(torch.log2(amax / torch.finfo(torch.float8_e4m3fn).max))
    q = (blocks / torch.exp2(exp)[..., None]).to(torch.float8_e4m3fn).reshape(w.shape)
    return q, (exp + 127).to(torch.uint8)


def _dequant(q: torch.Tensor, scale_u8: torch.Tensor) -> torch.Tensor:
    scale = torch.exp2(scale_u8.float() - 127)
    blocks = q.float().reshape(q.shape[0], -1, GROUP_SIZE)
    return (blocks * scale[..., None]).reshape(q.shape)


def _copy_loader(param, loaded_weight, *args, **kwargs):
    param.data.copy_(loaded_weight)


def _compressed_tensors_layer(w_q, w_s, k, n):
    weight_args = QuantizationArgs(
        num_bits=8, type="float", strategy="group", group_size=GROUP_SIZE
    )
    input_args = QuantizationArgs(
        num_bits=8,
        type="float",
        strategy="group",
        group_size=GROUP_SIZE,
        dynamic=True,
    )
    scheme = CompressedTensorsW8A8MXFp8(weight_args, input_args)
    layer = torch.nn.Module()
    scheme.create_weights(
        layer, k, [n], k, n, torch.bfloat16, weight_loader=_copy_loader
    )
    layer.to(DEVICE)
    layer.weight.data.copy_(w_q)
    layer.weight_scale.data.copy_(w_s)
    scheme.process_weights_after_loading(layer)
    return lambda x: scheme.apply_weights(layer, x)


def _fp8_method_layer(w_q, w_s, k, n):
    config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        use_mxfp8=True,
        weight_block_size=[1, GROUP_SIZE],
    )
    method = Fp8LinearMethod(config)
    layer = torch.nn.Module()
    layer.tp_group = None  # single rank, as a LinearBase with tp_size 1
    layer.deepgemm_mxfp8_input = False  # LinearBase default; only DeepSeek-V4 sets it
    method.create_weights(
        layer,
        input_size_per_partition=k,
        output_partition_sizes=[n],
        input_size=k,
        output_size=n,
        params_dtype=torch.bfloat16,
        weight_loader=_copy_loader,
    )
    layer.to(DEVICE)
    layer.weight.data.copy_(w_q)
    layer.weight_scale_inv.data.copy_(w_s)
    method.process_weights_after_loading(layer)
    return method, lambda x: method.apply(layer, x)


@contextmanager
def _fp8_gemm_backend_from_cli(backend):
    """Parse --fp8-gemm-backend through the CLI, publish it, and initialize from it."""
    server_args = prepare_server_args(
        ["--model-path", "dummy", "--fp8-gemm-backend", backend]
    )
    reset_context()
    publish(server_args, role="test", ranks=SpawnRanks(world_rank=0, dp_rank=None))
    try:
        with patch.object(fp8_utils, "FP8_GEMM_RUNNER_BACKEND", None):
            fp8_utils.initialize_fp8_gemm_config()
            yield
    finally:
        reset_context()


def test_fp8_method_selects_xpu_scaled_mm():
    config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        use_mxfp8=True,
        weight_block_size=[1, GROUP_SIZE],
    )
    # TODO: torch.xpu.get_device_capability
    assert config.get_min_capability() == 20
    method = Fp8LinearMethod(config)
    assert method.mxfp8_dense_backend.is_xpu_scaled_mm()


@pytest.mark.parametrize(
    "x_shape, n",
    [
        ((16, 256), 128),
        ((1, 512), 256),
        ((2, 33, 256), 96),
    ],
)
def test_both_methods_match(x_shape, n):
    torch.manual_seed(0)
    k = x_shape[-1]
    w_q, w_s = _mxfp8_quantize(torch.randn(n, k, dtype=torch.bfloat16))
    w_q, w_s = w_q.to(DEVICE), w_s.to(DEVICE)
    x = torch.randn(*x_shape, dtype=torch.bfloat16, device=DEVICE)

    out_ct = _compressed_tensors_layer(w_q, w_s, k, n)(x)
    method, run_fp8 = _fp8_method_layer(w_q, w_s, k, n)
    out_fp8 = run_fp8(x)

    assert method.mxfp8_dense_backend.is_xpu_scaled_mm()
    assert out_ct.shape == out_fp8.shape == (*x_shape[:-1], n)
    assert out_ct.dtype == out_fp8.dtype == torch.bfloat16
    # Same kernels on the same checkpoint tensors: identical output.
    torch.testing.assert_close(out_ct, out_fp8, rtol=0, atol=0)

    ref = x.float().reshape(-1, k) @ _dequant(w_q, w_s).t()
    rel = (out_fp8.float().reshape(-1, n) - ref).norm() / ref.norm()
    assert rel < REL_TOL, f"relative error {rel:.4f} vs dequantized reference"


def test_explicit_xpu_scaled_mm_backend_matches_auto():
    # --fp8-gemm-backend=xpu_scaled_mm must serve both methods with the same kernel
    # the auto default picks on XPU: bit-identical output on the same checkpoint.
    torch.manual_seed(0)
    k, n = 256, 128
    w_q, w_s = _mxfp8_quantize(torch.randn(n, k, dtype=torch.bfloat16))
    w_q, w_s = w_q.to(DEVICE), w_s.to(DEVICE)
    x = torch.randn(8, k, dtype=torch.bfloat16, device=DEVICE)

    auto_ct = _compressed_tensors_layer(w_q, w_s, k, n)(x)
    _, run_auto_fp8 = _fp8_method_layer(w_q, w_s, k, n)
    auto_fp8 = run_auto_fp8(x)

    with _fp8_gemm_backend_from_cli("xpu_scaled_mm"):
        assert fp8_utils.get_fp8_gemm_runner_backend().is_xpu_scaled_mm()
        explicit_ct = _compressed_tensors_layer(w_q, w_s, k, n)(x)
        method, run_explicit_fp8 = _fp8_method_layer(w_q, w_s, k, n)
        explicit_fp8 = run_explicit_fp8(x)

    assert method.mxfp8_dense_backend.is_xpu_scaled_mm()
    torch.testing.assert_close(explicit_ct, auto_ct, rtol=0, atol=0)
    torch.testing.assert_close(explicit_fp8, auto_fp8, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
