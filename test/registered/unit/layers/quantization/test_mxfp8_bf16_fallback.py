import sys
from types import SimpleNamespace

import pytest
import torch

import sglang.srt.layers.quantization.fp8 as fp8_quant
from sglang.srt.layers.quantization.mxfp8_block_convert import (
    dequant_mxfp8_to_bf16,
)
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=45, suite="stage-b-test-1-gpu-small-amd")
register_amd_ci(est_time=45, suite="stage-b-test-1-gpu-small-amd-mi35x")


def _mxfp8_values(shape):
    values = torch.linspace(-2, 2, torch.tensor(shape).prod().item()).reshape(shape)
    return values.to(torch.float8_e4m3fn)


def test_dequant_mxfp8_supports_dense_and_expert_shapes():
    dense_weight = _mxfp8_values((3, 64))
    dense_scale = torch.tensor([[127, 128], [126, 127], [128, 126]], dtype=torch.uint8)
    dense = dequant_mxfp8_to_bf16(dense_weight, dense_scale)

    expert_weight = dense_weight.reshape(1, 3, 64).expand(2, -1, -1).contiguous()
    expert_scale = dense_scale.reshape(1, 3, 2).expand(2, -1, -1).contiguous()
    experts = dequant_mxfp8_to_bf16(expert_weight, expert_scale)

    assert dense.dtype == torch.bfloat16
    assert experts.dtype == torch.bfloat16
    torch.testing.assert_close(experts[0], dense, rtol=0, atol=0)
    torch.testing.assert_close(experts[1], dense, rtol=0, atol=0)


def test_dense_uses_aiter_block_fp8_when_native_mxfp8_is_unavailable(monkeypatch):
    block_linear = object()
    monkeypatch.setattr(fp8_quant, "_is_hip", True)
    monkeypatch.setattr(fp8_quant, "_is_cuda", False)
    monkeypatch.setattr(fp8_quant, "_use_aiter", True)
    monkeypatch.setattr(fp8_quant, "cutlass_fp8_supported", lambda: False)
    monkeypatch.setattr(
        fp8_quant,
        "resolve_mxfp8_dense_gemm_backend",
        lambda: SimpleNamespace(is_unsupported=lambda: True),
    )
    monkeypatch.setattr(
        fp8_quant, "dispatch_w8a8_block_fp8_linear", lambda: block_linear
    )

    method = fp8_quant.Fp8LinearMethod(
        SimpleNamespace(
            use_mxfp8=True,
            weight_block_size=[1, 32],
            is_checkpoint_fp8_serialized=True,
        )
    )

    assert method.convert_mxfp8_to_block is True
    assert method.emulate_mxfp8 is False
    assert method.w8a8_block_fp8_linear is block_linear


def test_dense_emulation_preserves_mxfp8_storage_and_dequantizes_in_forward():
    method = object.__new__(fp8_quant.Fp8LinearMethod)
    method.emulate_mxfp8 = True
    method.convert_mxfp8_to_block = False
    method.use_mxfp8 = True
    method.block_quant = True

    weight = _mxfp8_values((4, 64))
    scale = torch.full((4, 2), 127, dtype=torch.uint8)
    expected_weight = dequant_mxfp8_to_bf16(weight, scale)
    layer = SimpleNamespace(
        weight=torch.nn.Parameter(weight, requires_grad=False),
        weight_scale_inv=torch.nn.Parameter(scale, requires_grad=False),
        input_scale=None,
    )

    method.process_weights_after_loading_block_quant(layer)

    assert method.use_mxfp8 is True
    assert method.block_quant is True
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight_scale_inv.dtype == torch.uint8

    x = torch.randn(2, 64, dtype=torch.bfloat16)
    torch.testing.assert_close(
        method.apply(layer, x),
        torch.nn.functional.linear(x, expected_weight),
        rtol=0,
        atol=0,
    )


def _test_absorbed_kv_b_proj_uses_exact_mxfp8_dequant_before_split(
    *, emulate_mxfp8: bool, convert_mxfp8_to_block: bool
):
    qk_nope_head_dim = 4
    v_head_dim = 2
    num_heads = 2
    weight = _mxfp8_values((num_heads * (qk_nope_head_dim + v_head_dim), 64))
    scale = torch.tensor(
        [[126 + (row % 3), 127 + (row % 2)] for row in range(weight.shape[0])],
        dtype=torch.uint8,
    )
    expected = dequant_mxfp8_to_bf16(weight, scale)
    expected_kc, expected_vc = expected.unflatten(
        0, (-1, qk_nope_head_dim + v_head_dim)
    ).split([qk_nope_head_dim, v_head_dim], dim=1)

    kv_b_proj = SimpleNamespace(
        weight=torch.nn.Parameter(weight, requires_grad=False),
        weight_scale_inv=torch.nn.Parameter(scale, requires_grad=False),
        quant_method=SimpleNamespace(
            use_mxfp8=True,
            emulate_mxfp8=emulate_mxfp8,
            convert_mxfp8_to_block=convert_mxfp8_to_block,
        ),
    )
    self_attn = SimpleNamespace(
        kv_b_proj=kv_b_proj,
        qk_nope_head_dim=qk_nope_head_dim,
        v_head_dim=v_head_dim,
        w_scale=99.0,
        w_kc=None,
        w_vc=None,
    )
    loader = DeepseekV2WeightLoaderMixin.__new__(DeepseekV2WeightLoaderMixin)
    loader.config = SimpleNamespace(
        num_hidden_layers=1,
        architectures=["HunyuanV4ForCausalLM"],
    )
    loader.quant_config = SimpleNamespace(
        weight_block_size=[1, 32], get_name=lambda: "mxfp8"
    )
    loader.model = SimpleNamespace(
        start_layer=0,
        end_layer=1,
        layers=[SimpleNamespace(self_attn=self_attn)],
    )

    loader.post_load_weights(weight_names=["model.layers.0.self_attn.kv_b_proj.weight"])

    assert self_attn.w_scale == 1.0
    assert self_attn.w_kc.dtype == torch.bfloat16
    assert self_attn.w_vc.dtype == torch.bfloat16
    torch.testing.assert_close(
        self_attn.w_kc,
        expected_kc.transpose(1, 2).contiguous().transpose(1, 2),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        self_attn.w_vc,
        expected_vc.contiguous().transpose(1, 2),
        rtol=0,
        atol=0,
    )


def test_absorbed_kv_b_proj_uses_exact_mxfp8_dequant_for_emulation():
    _test_absorbed_kv_b_proj_uses_exact_mxfp8_dequant_before_split(
        emulate_mxfp8=True, convert_mxfp8_to_block=False
    )


def test_absorbed_kv_b_proj_uses_exact_mxfp8_dequant_for_block_conversion():
    _test_absorbed_kv_b_proj_uses_exact_mxfp8_dequant_before_split(
        emulate_mxfp8=False, convert_mxfp8_to_block=True
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
