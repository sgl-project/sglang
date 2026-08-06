import sys

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.quantization import fp8 as diffusion_fp8_quant
from sglang.multimodal_gen.runtime.layers.quantization import (
    modelopt_quant as diffusion_modelopt_quant,
)
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import (
    Fp8Config,
    Fp8LinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp8Config,
    ModelOptFp8LinearMethod,
)
from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.fp8_kernel import (
    scaled_fp8_quant,
    static_quant_fp8,
)
from sglang.srt.layers.quantization.fp8_utils import (
    cutlass_fp8_supported,
    input_to_float8,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=80, suite="nightly-kernel-1-gpu", nightly=True)

DEVICE = "cuda"
DTYPE = torch.bfloat16
MAX_FP8_DIFF = 5e-4
TEST_CASES = [
    pytest.param(19, 150, 80, id="misaligned_projection_shape"),
    pytest.param(512, 3072, 4096, id="flux2_added_kv_projection_shape"),
]


def _modelopt_fp8_supported() -> bool:
    return torch.cuda.is_available() and cutlass_fp8_supported()


def _calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


def _dequantize_fp8_input(qinput: torch.Tensor, x_scale: torch.Tensor) -> torch.Tensor:
    return qinput.to(torch.float32) * x_scale.to(torch.float32)


def _dequantize_fp8_weight(
    weight: torch.Tensor, weight_scale: torch.Tensor
) -> torch.Tensor:
    if weight_scale.ndim == 0 or weight_scale.numel() == 1:
        scale = weight_scale.to(torch.float32)
    else:
        scale = weight_scale.to(torch.float32).reshape(-1, 1).t()
    return weight.to(torch.float32) * scale


def _build_layer(
    weight_q: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor,
) -> tuple[torch.nn.Module, ModelOptFp8LinearMethod]:
    output_size, input_size = weight_q.shape
    method = ModelOptFp8LinearMethod(
        ModelOptFp8Config(is_checkpoint_fp8_serialized=True)
    )
    layer = torch.nn.Module()
    method.create_weights(
        layer=layer,
        input_size_per_partition=input_size,
        output_partition_sizes=[output_size],
        input_size=input_size,
        output_size=output_size,
        params_dtype=DTYPE,
        weight_loader=lambda *args, **kwargs: None,
    )
    layer = layer.to(device=DEVICE)

    layer.weight.data.copy_(weight_q)
    layer.weight_scale.data.copy_(weight_scale.reshape_as(layer.weight_scale))
    layer.input_scale.data.copy_(input_scale.reshape_as(layer.input_scale))
    method.process_weights_after_loading(layer)
    return layer, method


def test_sm100_static_fp8_routes_to_scaled_mm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diffusion_modelopt_quant, "cutlass_fp8_supported", lambda: True)
    monkeypatch.setattr(diffusion_modelopt_quant, "is_sm100_supported", lambda: True)

    method = ModelOptFp8LinearMethod(
        ModelOptFp8Config(is_checkpoint_fp8_serialized=True)
    )
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.ones((8, 4)), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(torch.tensor(0.5), requires_grad=False)
    layer.input_scale = torch.nn.Parameter(torch.tensor(0.25), requires_grad=False)
    x = torch.ones((2, 8))
    expected = torch.full((2, 4), 7.0)

    def fake_scaled_mm(**kwargs):
        assert kwargs["input"] is x
        assert kwargs["weight_scale"].ndim == 0
        assert kwargs["input_scale"].ndim == 0
        return expected

    monkeypatch.setattr(
        diffusion_modelopt_quant,
        "apply_fp8_linear_scaled_mm",
        fake_scaled_mm,
    )
    monkeypatch.setattr(
        diffusion_modelopt_quant,
        "apply_fp8_linear",
        lambda **kwargs: pytest.fail("generic FP8 path must not run on SM100"),
    )

    assert method.enable_sm100_scaled_mm
    assert method.apply(layer, x) is expected


def test_serialized_fp8_config_routes_to_scaled_mm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover the quant_method=fp8 format emitted by the minWM builder."""
    monkeypatch.setattr(diffusion_fp8_quant, "cutlass_fp8_supported", lambda: True)
    monkeypatch.setattr(diffusion_fp8_quant, "is_sm100_supported", lambda: True)

    method = Fp8LinearMethod(
        Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="static")
    )
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.ones((8, 4)), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(torch.tensor(0.5), requires_grad=False)
    layer.input_scale = torch.nn.Parameter(torch.tensor(0.25), requires_grad=False)
    x = torch.ones((2, 8))
    expected = torch.full((2, 4), 7.0)

    def fake_scaled_mm(**kwargs):
        assert kwargs["input"] is x
        assert kwargs["weight_scale"].ndim == 0
        assert kwargs["input_scale"].ndim == 0
        return expected

    monkeypatch.setattr(
        diffusion_fp8_quant,
        "apply_fp8_linear_scaled_mm",
        fake_scaled_mm,
    )
    monkeypatch.setattr(
        diffusion_fp8_quant,
        "apply_fp8_linear",
        lambda **kwargs: pytest.fail("generic FP8 path must not run on SM100"),
    )

    assert method.enable_sm100_scaled_mm
    assert method.apply(layer, x) is expected


def test_serialized_fp8_config_keeps_scalar_scales_on_sm100(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diffusion_fp8_quant, "cutlass_fp8_supported", lambda: True)
    monkeypatch.setattr(diffusion_fp8_quant, "is_sm100_supported", lambda: True)
    monkeypatch.setattr(
        diffusion_fp8_quant, "get_tensor_model_parallel_world_size", lambda: 1
    )

    method = Fp8LinearMethod(
        Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="static")
    )
    layer = torch.nn.Module()
    method.create_weights(
        layer=layer,
        input_size_per_partition=8,
        output_partition_sizes=[4],
        input_size=8,
        output_size=4,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *args, **kwargs: None,
    )
    layer = layer.to(device=DEVICE)
    layer.weight.data.fill_(1)
    layer.weight_scale.data.fill_(0.5)
    layer.input_scale.data.fill_(0.25)

    method.process_weights_after_loading(layer)

    assert tuple(layer.weight.shape) == (8, 4)
    assert tuple(layer.weight.stride()) == (1, 8)
    assert layer.weight_scale.ndim == 0
    assert layer.input_scale.ndim == 0


def test_scaled_mm_helper_uses_jit_per_tensor_quant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = torch.ones((2, 16))
    weight = torch.ones((16, 16)).t()
    weight_scale = torch.tensor(0.5)
    input_scale = torch.tensor(0.25)
    qinput = torch.ones_like(x, dtype=torch.float8_e4m3fn)
    expected = torch.full((2, 16), 3.0)

    def fake_quant(actual_input, actual_scale):
        assert actual_input.shape == x.shape
        assert actual_input.stride() == x.stride()
        assert actual_input.data_ptr() == x.data_ptr()
        assert actual_scale is input_scale
        return qinput, actual_scale

    def fake_scaled_mm(actual_input, actual_weight, **kwargs):
        assert actual_input is qinput
        assert actual_weight is weight
        assert kwargs["scale_a"].data_ptr() == input_scale.data_ptr()
        assert kwargs["scale_b"].data_ptr() == weight_scale.data_ptr()
        return expected

    monkeypatch.setattr(fp8_utils, "scaled_fp8_quant", fake_quant)
    monkeypatch.setattr(torch, "_scaled_mm", fake_scaled_mm)

    actual = fp8_utils.apply_fp8_linear_scaled_mm(
        input=x,
        weight=weight,
        weight_scale=weight_scale,
        input_scale=input_scale,
    )
    torch.testing.assert_close(actual, expected)


def test_scaled_mm_helper_uses_triton_static_quant_for_large_m(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = torch.ones((8192, 16))
    weight = torch.ones((16, 16)).t()
    weight_scale = torch.tensor(0.5)
    input_scale = torch.tensor(0.25)
    qinput = torch.ones_like(x, dtype=torch.float8_e4m3fn)
    expected = torch.full((8192, 16), 3.0)

    def fake_quant(actual_input, actual_scale, repeat_scale):
        assert actual_input.shape == x.shape
        assert actual_input.data_ptr() == x.data_ptr()
        assert actual_scale is input_scale
        assert repeat_scale is False
        return qinput, actual_scale

    def fail_jit_quant(*_args, **_kwargs):
        raise AssertionError("large-M static quant must not use the flat JIT kernel")

    monkeypatch.setattr(fp8_utils, "static_quant_fp8", fake_quant)
    monkeypatch.setattr(fp8_utils, "scaled_fp8_quant", fail_jit_quant)
    monkeypatch.setattr(torch, "_scaled_mm", lambda *_args, **_kwargs: expected)

    actual = fp8_utils.apply_fp8_linear_scaled_mm(
        input=x,
        weight=weight,
        weight_scale=weight_scale,
        input_scale=input_scale,
    )
    torch.testing.assert_close(actual, expected)


def test_scaled_mm_helper_scopes_deterministic_fill_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = torch.ones((2, 16))
    weight = torch.ones((16, 16)).t()
    weight_scale = torch.tensor(0.5)
    input_scale = torch.tensor(0.25)
    qinput = torch.ones_like(x, dtype=torch.float8_e4m3fn)
    expected = torch.full((2, 16), 3.0)
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_fill = torch.utils.deterministic.fill_uninitialized_memory

    def fake_quant(actual_input, actual_scale):
        assert not torch.utils.deterministic.fill_uninitialized_memory
        return qinput, actual_scale

    def fake_scaled_mm(*args, **kwargs):
        assert not torch.utils.deterministic.fill_uninitialized_memory
        return expected

    monkeypatch.setattr(fp8_utils, "scaled_fp8_quant", fake_quant)
    monkeypatch.setattr(torch, "_scaled_mm", fake_scaled_mm)
    try:
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = True
        actual = fp8_utils.apply_fp8_linear_scaled_mm(
            input=x,
            weight=weight,
            weight_scale=weight_scale,
            input_scale=input_scale,
        )
        torch.testing.assert_close(actual, expected)
        assert torch.utils.deterministic.fill_uninitialized_memory
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = previous_fill
        torch.use_deterministic_algorithms(
            previous_enabled,
            warn_only=previous_warn_only,
        )


@pytest.mark.skipif(
    not _modelopt_fp8_supported(),
    reason="Diffusion ModelOpt FP8 scaled mm correctness requires CUDA FP8 support",
)
@pytest.mark.parametrize("m,n,k", TEST_CASES)
def test_checkpoint_processing(m: int, n: int, k: int) -> None:
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(20260410 + m + n + k)

    weight = torch.randn((n, k), device=DEVICE, dtype=DTYPE, generator=generator)
    weight_q, weight_scale = input_to_float8(weight)
    input_scale = torch.tensor(1.0, device=DEVICE, dtype=torch.float32)

    layer, method = _build_layer(weight_q, weight_scale, input_scale)

    assert tuple(layer.weight.shape) == (k, n)
    assert tuple(layer.weight.stride()) == (1, k)
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.input_scale.ndim == 0
    expected_scale_shape = () if method.enable_sm100_scaled_mm else (n, 1)
    assert tuple(layer.weight_scale.shape) == expected_scale_shape

    expected_weight = weight_q.t().to(torch.float32) * weight_scale.to(torch.float32)
    actual_weight = _dequantize_fp8_weight(layer.weight, layer.weight_scale)
    torch.testing.assert_close(actual_weight, expected_weight, atol=0.0, rtol=0.0)


@pytest.mark.skipif(
    not _modelopt_fp8_supported(),
    reason="Diffusion ModelOpt FP8 scaled mm correctness requires CUDA FP8 support",
)
@pytest.mark.parametrize("m,n,k", TEST_CASES)
def test_shape_correctness(m: int, n: int, k: int) -> None:
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(20260410 + m + n + k)

    x = torch.randn((m, k), device=DEVICE, dtype=DTYPE, generator=generator)
    weight = torch.randn((n, k), device=DEVICE, dtype=DTYPE, generator=generator)
    weight_q, weight_scale = input_to_float8(weight)
    _, input_scale = input_to_float8(x)

    layer, method = _build_layer(weight_q, weight_scale, input_scale)

    if method.enable_sm100_scaled_mm:
        qinput, x_scale = scaled_fp8_quant(x.contiguous(), layer.input_scale)
    else:
        qinput, x_scale = static_quant_fp8(
            x.contiguous(),
            layer.input_scale,
            repeat_scale=method.cutlass_fp8_supported,
        )
    expected = torch.matmul(
        _dequantize_fp8_input(qinput, x_scale),
        _dequantize_fp8_weight(layer.weight, layer.weight_scale),
    )

    actual = method.apply(layer, x)
    diff = _calc_diff(actual, expected.to(dtype=DTYPE))
    assert diff < MAX_FP8_DIFF, f"{m=}, {n=}, {k=}, {diff=:.6f}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
