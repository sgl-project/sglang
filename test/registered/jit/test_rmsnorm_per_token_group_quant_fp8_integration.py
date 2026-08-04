from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, NamedTuple

import pytest
import torch

from sglang.jit_kernel.rmsnorm_per_token_group_quant_fp8 import (
    rmsnorm_per_token_group_quant_fp8,
)
from sglang.jit_kernel.tests.rmsnorm_per_token_group_quant_fp8_test_utils import (
    DEVICE,
    DTYPE,
    EPS,
    GROUP_SIZE,
    TRACE_HIDDEN_SIZE,
)
from sglang.jit_kernel.tests.rmsnorm_per_token_group_quant_fp8_test_utils import (
    make_input as _make_input,
)
from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.test.ci.ci_register import register_cuda_ci

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
    from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
        DeepseekMLAForwardMixin,
    )

register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


class _MlaHarness(NamedTuple):
    mla: DeepseekMLAForwardMixin
    quant_method: Fp8LinearMethod
    x: torch.Tensor
    weight_scale: torch.Tensor
    packed_scale_cols: int
    server_args: SimpleNamespace


@pytest.fixture
def mla_harness(monkeypatch: pytest.MonkeyPatch) -> _MlaHarness:
    from sglang.srt.layers import deep_gemm_wrapper
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
    from sglang.srt.models.deepseek_common.attention_forward_methods import (
        forward_mla,
    )

    if not deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
        pytest.skip("requires Blackwell DeepGEMM with packed UE8M0 scales")

    hidden_size = 640
    packed_scale_cols = (hidden_size // GROUP_SIZE + 3) // 4
    quant_method = Fp8LinearMethod.__new__(Fp8LinearMethod)
    quant_method.w8a8_block_fp8_prequantized_linear = lambda **_: None
    quant_method.block_quant = True
    quant_method.use_marlin = False
    quant_method.use_mxfp8 = False
    quant_method.weight_block_size = [GROUP_SIZE, GROUP_SIZE]

    weight_scale_storage = torch.empty(
        packed_scale_cols, 128, device=DEVICE, dtype=torch.int32
    )
    weight_scale = weight_scale_storage.T[:64]
    weight_scale.format_ue8m0 = True

    class FakeMLA(forward_mla.DeepseekMLAForwardMixin):
        pass

    server_args = SimpleNamespace(
        enable_lora=False,
        enable_deterministic_inference=False,
    )
    monkeypatch.setattr(forward_mla, "get_server_args", lambda: server_args)

    mla = FakeMLA()
    mla.q_lora_rank = hidden_size
    mla.q_a_layernorm = RMSNorm(
        hidden_size, eps=EPS, weight_dtype=torch.bfloat16
    ).cuda()
    mla.q_b_proj = SimpleNamespace(
        quant_method=quant_method,
        input_scale=None,
        orig_dtype=DTYPE,
        weight=torch.empty(64, hidden_size, device=DEVICE, dtype=fp8_dtype),
        weight_scale_inv=weight_scale,
    )
    mla.use_min_latency_q_b_gemm = False
    mla.use_dsa = True

    # Seed the load-static eligibility flag, as process_weights_after_loading
    # would once the layer's weights are finalized.
    mla.q_b_proj._fused_rmsnorm_fp8_enabled = (
        quant_method._is_fused_rmsnorm_fp8_layer(mla.q_b_proj)
    )

    return _MlaHarness(
        mla=mla,
        quant_method=quant_method,
        x=_make_input(6, hidden_size),
        weight_scale=weight_scale,
        packed_scale_cols=packed_scale_cols,
        server_args=server_args,
    )


def test_fp8_fused_rmsnorm_gate_is_live_per_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.srt.layers.quantization import fp8 as fp8_module
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

    hidden_size = 640
    x = _make_input(2, hidden_size)
    norm_weight = torch.randn(hidden_size, device=DEVICE, dtype=DTYPE)
    layer = SimpleNamespace(
        weight=torch.empty(64, hidden_size, device=DEVICE, dtype=fp8_dtype)
    )
    quant_method = Fp8LinearMethod.__new__(Fp8LinearMethod)
    monkeypatch.setattr(
        Fp8LinearMethod,
        "_can_apply_packed_ue8m0_input",
        lambda self, candidate_layer: candidate_layer is layer,
    )

    gate_results = iter((False, True))
    gate_calls = []

    def live_gate(*, input_dtype: torch.dtype, hidden_size: int) -> bool:
        gate_calls.append((input_dtype, hidden_size))
        return next(gate_results)

    output_q = torch.empty_like(x, dtype=fp8_dtype)
    output_s = torch.empty(2, 2, device=DEVICE, dtype=torch.int32)
    output_norm = torch.empty_like(x)
    monkeypatch.setattr(
        fp8_module,
        "can_use_rmsnorm_per_token_group_quant_fp8",
        live_gate,
    )
    monkeypatch.setattr(
        fp8_module,
        "rmsnorm_per_token_group_quant_fp8",
        lambda input, weight, eps: (output_q, output_s, output_norm),
    )

    assert (
        quant_method.maybe_prepare_fused_rmsnorm_input(layer, x, norm_weight, EPS)
        is None
    )
    prepared = quant_method.maybe_prepare_fused_rmsnorm_input(
        layer, x, norm_weight, EPS
    )

    assert prepared is not None
    linear_input, normalized = prepared
    assert linear_input.data is output_q
    assert linear_input.scale is output_s
    assert normalized is output_norm
    assert gate_calls == [(DTYPE, hidden_size), (DTYPE, hidden_size)]


def test_prequantized_input_never_falls_back_to_triton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.srt.layers.quantization import fp8_utils
    from sglang.srt.layers.quantization.fp8_utils import PackedUe8m0LinearInput

    m, n, k = 2, 65, 640
    block_size = [GROUP_SIZE, GROUP_SIZE]
    input_bf16 = torch.randn(m, k, device=DEVICE, dtype=DTYPE)
    input_fp8 = input_bf16.to(fp8_dtype)
    input_scale = torch.empty(m, 2, device=DEVICE, dtype=torch.int32)
    weight = torch.empty(n, k, device=DEVICE, dtype=fp8_dtype)
    weight_scale = torch.empty(1, 1, device=DEVICE, dtype=torch.float32)
    fallback_output = torch.empty(m, n, device=DEVICE, dtype=DTYPE)

    def triton_fallback(
        input, weight, block_size, weight_scale, input_scale=None, bias=None
    ):
        assert input_scale is None
        return fallback_output

    monkeypatch.setattr(fp8_utils, "triton_w8a8_block_fp8_linear", triton_fallback)

    assert (
        fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback(
            input_bf16, weight, block_size, weight_scale
        )
        is fallback_output
    )
    with pytest.raises(ValueError, match="N divisible by 64"):
        fp8_utils.deepgemm_w8a8_block_fp8_linear_prequantized(
            PackedUe8m0LinearInput(input_fp8, input_scale),
            weight,
            block_size,
            weight_scale,
        )


def test_deepgemm_prequantized_input_parity() -> None:
    from sglang.srt.layers import deep_gemm_wrapper
    from sglang.srt.layers.quantization.fp8_utils import (
        PackedUe8m0LinearInput,
        deepgemm_w8a8_block_fp8_linear_prequantized,
        deepgemm_w8a8_block_fp8_linear_with_fallback,
        quant_weight_ue8m0,
        transform_scale_ue8m0,
    )

    if not deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
        pytest.skip("requires Blackwell DeepGEMM with packed UE8M0 scales")

    torch.manual_seed(31)
    # Match the fused activation contract exactly (M=6, K=2048) while keeping
    # N small enough for a focused kernel test.
    m, n, k = 6, 256, TRACE_HIDDEN_SIZE
    block_size = [GROUP_SIZE, GROUP_SIZE]
    x = torch.randn(m, k, device=DEVICE, dtype=DTYPE)
    norm_weight = torch.randn(k, device=DEVICE, dtype=DTYPE)
    weight_bf16 = torch.randn(n, k, device=DEVICE, dtype=DTYPE)
    weight_fp8, weight_scale_raw = quant_weight_ue8m0(
        weight_bf16, weight_block_size=block_size
    )
    weight_scale = transform_scale_ue8m0(weight_scale_raw, mn=n)

    x_fp8, x_scale, x_norm = rmsnorm_per_token_group_quant_fp8(x, norm_weight, EPS)
    output_from_bf16 = deepgemm_w8a8_block_fp8_linear_with_fallback(
        x_norm, weight_fp8, block_size, weight_scale
    )
    output_from_prequant = deepgemm_w8a8_block_fp8_linear_prequantized(
        PackedUe8m0LinearInput(x_fp8, x_scale),
        weight_fp8,
        block_size,
        weight_scale,
    )
    torch.cuda.synchronize()

    assert output_from_bf16.dtype == DTYPE
    assert output_from_prequant.dtype == DTYPE
    assert torch.equal(output_from_prequant, output_from_bf16)


def test_deepseek_mla_prepares_and_consumes_packed_input(
    mla_harness: _MlaHarness,
) -> None:
    from sglang.srt.layers.quantization.fp8_utils import PackedUe8m0LinearInput

    prepared = mla_harness.quant_method.maybe_prepare_fused_rmsnorm_input(
        mla_harness.mla.q_b_proj,
        mla_harness.x,
        mla_harness.mla.q_a_layernorm.weight,
        EPS,
    )

    assert prepared is not None
    linear_input, normalized = prepared
    assert isinstance(linear_input, PackedUe8m0LinearInput)
    assert linear_input.data.shape == mla_harness.x.shape
    assert linear_input.scale.shape == (
        mla_harness.x.shape[0],
        mla_harness.packed_scale_cols,
    )
    assert normalized.shape == mla_harness.x.shape

    seen = {}

    def fake_prequantized_deepgemm(**kwargs):
        seen.update(kwargs)
        return torch.empty(mla_harness.x.shape[0], 64, device=DEVICE, dtype=DTYPE)

    mla_harness.quant_method.w8a8_block_fp8_prequantized_linear = (
        fake_prequantized_deepgemm
    )
    mla_harness.quant_method.apply(mla_harness.mla.q_b_proj, linear_input)
    assert seen["input"] is linear_input


@pytest.mark.parametrize(
    "broken_contract",
    ("activation_dtype", "norm_dtype"),
)
def test_fused_rmsnorm_rejects_incompatible_tensors(
    mla_harness: _MlaHarness,
    broken_contract: str,
) -> None:
    x = mla_harness.x
    norm_weight = mla_harness.mla.q_a_layernorm.weight
    if broken_contract == "activation_dtype":
        x = x.half()
    else:
        norm_weight = norm_weight.half()

    assert (
        mla_harness.quant_method.maybe_prepare_fused_rmsnorm_input(
            mla_harness.mla.q_b_proj,
            x,
            norm_weight,
            EPS,
        )
        is None
    )


@pytest.mark.parametrize(
    "fallback_reason",
    (
        "lora",
        "deterministic_inference",
        "deepgemm_scale_mode_off",
        "missing_prequantized_consumer",
        "unsupported_output_size",
        "unpacked_weight_scale",
    ),
)
def test_deepseek_mla_uses_unfused_fallback_when_contract_is_not_met(
    mla_harness: _MlaHarness,
    monkeypatch: pytest.MonkeyPatch,
    fallback_reason: str,
) -> None:
    if fallback_reason == "lora":
        mla_harness.server_args.enable_lora = True
    elif fallback_reason == "deterministic_inference":
        mla_harness.server_args.enable_deterministic_inference = True
    elif fallback_reason == "deepgemm_scale_mode_off":
        # LongCat toggles this mode after constructing its attention modules;
        # the live per-forward gate must observe it and fall back.
        from sglang.srt.layers import deep_gemm_wrapper

        monkeypatch.setattr(deep_gemm_wrapper, "DEEPGEMM_SCALE_UE8M0", False)
    elif fallback_reason == "missing_prequantized_consumer":
        mla_harness.quant_method.w8a8_block_fp8_prequantized_linear = None
    elif fallback_reason == "unsupported_output_size":
        mla_harness.mla.q_b_proj.weight = torch.empty(
            65,
            mla_harness.x.shape[1],
            device=DEVICE,
            dtype=fp8_dtype,
        )
    else:  # unpacked_weight_scale
        mla_harness.weight_scale.format_ue8m0 = False

    # Re-derive the load-static eligibility flag for the (mutated) layer, as
    # process_weights_after_loading would after loading its weights.
    q_b_proj = mla_harness.mla.q_b_proj
    q_b_proj._fused_rmsnorm_fp8_enabled = (
        mla_harness.quant_method._is_fused_rmsnorm_fp8_layer(q_b_proj)
    )

    q_fallback, q_norm_fallback = mla_harness.mla._q_a_layernorm_with_optional_quant(
        mla_harness.x
    )

    assert isinstance(q_fallback, torch.Tensor)
    assert q_norm_fallback is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
