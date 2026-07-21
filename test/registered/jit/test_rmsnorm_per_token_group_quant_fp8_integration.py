import sys
from types import SimpleNamespace

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

register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


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
        quant_method.maybe_prepare_fused_rmsnorm_input(
            layer,
            x,
            norm_weight,
            EPS,
            need_normalized_output=True,
        )
        is None
    )
    prepared = quant_method.maybe_prepare_fused_rmsnorm_input(
        layer,
        x,
        norm_weight,
        EPS,
        need_normalized_output=True,
    )

    assert prepared is not None
    assert prepared.linear_input.data is output_q
    assert prepared.linear_input.scale is output_s
    assert prepared.normalized_input is output_norm
    assert gate_calls == [(DTYPE, hidden_size), (DTYPE, hidden_size)]


def test_prequantized_input_never_falls_back_to_triton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.srt.layers.quantization import fp8_utils

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
    with pytest.raises(ValueError, match="fall back before RMSNorm/quantization"):
        fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback(
            input_fp8,
            weight,
            block_size,
            weight_scale,
            input_scale=input_scale,
        )


def test_deepgemm_prequantized_input_parity() -> None:
    from sglang.srt.layers import deep_gemm_wrapper
    from sglang.srt.layers.quantization.fp8_utils import (
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
    output_from_prequant = deepgemm_w8a8_block_fp8_linear_with_fallback(
        x_fp8,
        weight_fp8,
        block_size,
        weight_scale,
        input_scale=x_scale,
    )

    @torch.compile(fullgraph=True)
    def compiled_prequantized_deepgemm(
        input: torch.Tensor,
        input_scale: torch.Tensor,
        weight: torch.Tensor,
        packed_weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        return deepgemm_w8a8_block_fp8_linear_with_fallback(
            input,
            weight,
            block_size,
            packed_weight_scale,
            input_scale=input_scale,
        )

    output_from_compiled_prequant = compiled_prequantized_deepgemm(
        x_fp8, x_scale, weight_fp8, weight_scale
    )
    torch.cuda.synchronize()

    assert output_from_bf16.dtype == DTYPE
    assert output_from_prequant.dtype == DTYPE
    assert torch.equal(output_from_prequant, output_from_bf16)
    assert torch.equal(output_from_compiled_prequant, output_from_bf16)


def test_deepseek_mla_capability_dispatch_and_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.srt.layers import deep_gemm_wrapper
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.layers.quantization.fp8 import (
        DeepGemmUe8m0Input,
        Fp8LinearMethod,
    )
    from sglang.srt.models.deepseek_common.attention_forward_methods import (
        forward_mla,
    )

    DeepseekMLAForwardMixin = forward_mla.DeepseekMLAForwardMixin

    if not deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
        pytest.skip("requires Blackwell DeepGEMM with packed UE8M0 scales")

    hidden_size = 640
    packed_scale_cols = (hidden_size // GROUP_SIZE + 3) // 4
    quant_method = Fp8LinearMethod.__new__(Fp8LinearMethod)
    quant_method._accepts_packed_ue8m0_input = True
    quant_method.block_quant = True
    quant_method.use_marlin = False
    quant_method.use_mxfp8 = False
    quant_method.weight_block_size = [GROUP_SIZE, GROUP_SIZE]
    weight_scale_storage = torch.empty(
        packed_scale_cols, 128, device=DEVICE, dtype=torch.int32
    )
    weight_scale = weight_scale_storage.T[:64]
    weight_scale.format_ue8m0 = True

    class FakeMLA(DeepseekMLAForwardMixin):
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
        weight=torch.empty(64, hidden_size, device=DEVICE, dtype=torch.float8_e4m3fn),
        weight_scale_inv=weight_scale,
    )
    mla.use_min_latency_q_b_gemm = False
    mla.use_dsa = True

    x = _make_input(6, hidden_size)
    prepared = quant_method.maybe_prepare_fused_rmsnorm_input(
        mla.q_b_proj,
        x,
        mla.q_a_layernorm.weight,
        EPS,
        need_normalized_output=True,
    )
    assert prepared is not None
    assert isinstance(prepared.linear_input, DeepGemmUe8m0Input)
    assert prepared.linear_input.data.shape == x.shape
    assert prepared.linear_input.scale.shape == (x.shape[0], packed_scale_cols)
    assert prepared.normalized_input.shape == x.shape

    seen = {}

    def fake_deepgemm(**kwargs):
        seen.update(kwargs)
        return torch.empty(x.shape[0], 64, device=DEVICE, dtype=DTYPE)

    quant_method.w8a8_block_fp8_linear = fake_deepgemm
    quant_method.apply(mla.q_b_proj, prepared.linear_input)
    assert seen["input"] is prepared.linear_input.data
    assert seen["input_scale"] is prepared.linear_input.scale

    quant_method._accepts_packed_ue8m0_input = False
    with pytest.raises(ValueError, match="live DeepGEMM block-FP8 consumer"):
        quant_method.apply(mla.q_b_proj, prepared.linear_input)
    quant_method._accepts_packed_ue8m0_input = True

    assert (
        quant_method.maybe_prepare_fused_rmsnorm_input(
            mla.q_b_proj,
            x.half(),
            mla.q_a_layernorm.weight,
            EPS,
            need_normalized_output=True,
        )
        is None
    )

    bf16_norm = mla.q_a_layernorm
    mla.q_a_layernorm = RMSNorm(hidden_size, eps=EPS, weight_dtype=torch.float16).cuda()
    assert (
        quant_method.maybe_prepare_fused_rmsnorm_input(
            mla.q_b_proj,
            x,
            mla.q_a_layernorm.weight,
            EPS,
            need_normalized_output=True,
        )
        is None
    )
    mla.q_a_layernorm = bf16_norm

    server_args.enable_lora = True
    q_fallback, _ = mla._q_a_layernorm_with_optional_quant(x)
    assert isinstance(q_fallback, torch.Tensor)
    server_args.enable_lora = False
    server_args.enable_deterministic_inference = True
    q_fallback, _ = mla._q_a_layernorm_with_optional_quant(x)
    assert isinstance(q_fallback, torch.Tensor)
    server_args.enable_deterministic_inference = False

    valid_weight = mla.q_b_proj.weight
    mla.q_b_proj.weight = torch.empty(
        65, hidden_size, device=DEVICE, dtype=torch.float8_e4m3fn
    )
    q_fallback, q_norm_fallback = mla._q_a_layernorm_with_optional_quant(x)
    assert isinstance(q_fallback, torch.Tensor)
    assert q_norm_fallback is None
    mla.q_b_proj.weight = valid_weight

    weight_scale.format_ue8m0 = False
    q_fallback, q_norm_fallback = mla._q_a_layernorm_with_optional_quant(x)
    assert isinstance(q_fallback, torch.Tensor)
    assert q_norm_fallback is None
    weight_scale.format_ue8m0 = True

    @torch.compile(fullgraph=True)
    def compiled(input: torch.Tensor):
        return mla._q_a_layernorm_with_optional_quant(input)

    q_pair, q_norm = compiled(x)
    torch.cuda.synchronize()
    assert isinstance(q_pair, DeepGemmUe8m0Input)
    assert q_pair.data.shape == x.shape
    assert q_pair.scale.shape == (x.shape[0], packed_scale_cols)
    assert q_norm.shape == x.shape

    # LongCat changes the scale mode after constructing its attention modules.
    # The next forward must observe the live state and take the unfused path.
    monkeypatch.setattr(deep_gemm_wrapper, "DEEPGEMM_SCALE_UE8M0", False)
    q_fallback, q_norm_fallback = compiled(x)
    assert isinstance(q_fallback, torch.Tensor)
    assert q_norm_fallback is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
