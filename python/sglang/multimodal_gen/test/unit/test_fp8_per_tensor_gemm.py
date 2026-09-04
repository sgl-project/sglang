# SPDX-License-Identifier: Apache-2.0
"""Online fp8 per-tensor GEMM path (``Fp8Config(per_tensor_online=True)``):
load-time scalar weight scale, dynamic scalar activation scale into
torch._scaled_mm, the prequantized (fp8, scale) input from the fused SwiGLU +
quant kernel, and the per-layer fallback to the per-channel path."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _init_parallel() -> None:
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
        model_parallel_is_initialized,
    )
    from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
        ensure_distributed_env_defaults,
    )

    if not model_parallel_is_initialized():
        ensure_distributed_env_defaults()
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


def test_per_tensor_linear_matches_bf16_and_accepts_prequantized() -> None:
    _init_parallel()
    from sglang.kernels.ops.diffusion import silu_mul_per_tensor_fp8
    from sglang.multimodal_gen.runtime.layers.linear import RowParallelLinear
    from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config

    in_f, out_f, rows = 512, 384, 200
    layer = RowParallelLinear(
        in_f,
        out_f,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=Fp8Config(per_tensor_online=True),
        prefix="mlp.fc2",
    ).to("cuda")
    assert layer.quant_method.per_tensor_online
    g = torch.Generator(device="cpu").manual_seed(1)
    weight = (torch.randn(out_f, in_f, generator=g) * 0.02).to("cuda", torch.bfloat16)
    with torch.no_grad():
        layer.weight.copy_(weight)
    layer.quant_method.process_weights_after_loading(layer)
    assert layer.fp8_per_tensor
    assert layer.quant_method.accepts_fp8_per_tensor_input(layer)
    assert layer.weight_scale.numel() == 1 and layer.weight.dtype != torch.bfloat16

    x = torch.randn(rows, in_f, generator=g).to("cuda", torch.bfloat16)
    out, _ = layer(x)
    ref = x.float() @ weight.float().t()
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    assert rel < 0.05, rel  # fp8 e4m3 per-tensor on gaussian data

    # the prequantized path: fused SwiGLU + quant feeding the same linear
    hidden = torch.randn(rows, 2 * in_f, generator=g).to("cuda", torch.bfloat16)
    act = torch.nn.functional.silu(hidden[:, :in_f]) * hidden[:, in_f:]
    out_tensor, _ = layer(act)
    out_tuple, _ = layer(silu_mul_per_tensor_fp8(hidden))
    assert torch.equal(out_tensor, out_tuple)


def test_default_config_keeps_channelwise() -> None:
    _init_parallel()
    from sglang.multimodal_gen.runtime.layers.linear import RowParallelLinear
    from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config

    layer = RowParallelLinear(
        256,
        128,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=Fp8Config(),
        prefix="mlp.fc2",
    ).to("cuda")
    assert not layer.quant_method.per_tensor_online
    layer.quant_method.process_weights_after_loading(layer)
    assert not layer.fp8_per_tensor
    assert layer.weight_scale.numel() == 128


def test_unaligned_layer_falls_back_to_channelwise() -> None:
    """torch._scaled_mm needs 16-aligned K and N; such a layer keeps the
    per-channel path even when the config asks for per-tensor."""
    _init_parallel()
    from sglang.multimodal_gen.runtime.layers.linear import RowParallelLinear
    from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config

    layer = RowParallelLinear(
        8,
        128,
        bias=True,
        params_dtype=torch.bfloat16,
        quant_config=Fp8Config(per_tensor_online=True),
        prefix="adaln.linear",
    ).to("cuda")
    layer.quant_method.process_weights_after_loading(layer)
    assert not layer.fp8_per_tensor
    x = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
    out, _ = layer(x)
    assert out.shape == (4, 128)
