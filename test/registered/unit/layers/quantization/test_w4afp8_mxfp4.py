from types import SimpleNamespace

import pytest
import torch

from sglang.srt.configs.kimi_k25 import KimiK25Config
from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config, W4AFp8MoEMethod
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_w4afp8_config_distinguishes_int4_and_mxfp4():
    int4_config = W4AFp8Config()
    assert int4_config.moe_weight_format == "int4"
    assert int4_config.group_size == 128

    mxfp4_config = W4AFp8Config.from_config(
        {
            "quant_method": "w4afp8",
            "weight_format": "mxfp4",
            "group_size": 32,
            "activation_scheme": "dynamic",
        }
    )
    assert mxfp4_config.moe_weight_format == "mxfp4"
    assert mxfp4_config.group_size == 32

    with pytest.raises(ValueError, match="requires group_size=32"):
        W4AFp8Config(moe_weight_format="mxfp4", group_size=128)

    with pytest.raises(ValueError, match="Conflicting W4AFP8"):
        W4AFp8Config.from_config(
            {
                "quant_method": "w4afp8",
                "weight_format": "mxfp4",
                "moe_weight_format": "int4",
            }
        )


def test_w4afp8_config_keeps_kimi_dense_linears_in_bfloat16(monkeypatch):
    from sglang.srt.layers.quantization import w4afp8
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    dense_layer = LinearBase.__new__(LinearBase)
    for spelling in ("bfloat16", "bf16"):
        config = W4AFp8Config.from_config(
            {
                "quant_method": "w4afp8",
                "moe_weight_format": "mxfp4",
                "linear_weight_format": spelling,
                "group_size": 32,
            }
        )
        assert config.linear_weight_format == "bfloat16"
        assert config.is_checkpoint_fp8_serialized is False
        assert isinstance(
            config.get_quant_method(dense_layer, "model.layers.0.self_attn.q_proj"),
            UnquantizedLinearMethod,
        )

    monkeypatch.setattr(w4afp8, "Fp8LinearMethod", lambda _: "fp8")
    glm_config = W4AFp8Config.from_config(
        {
            "quant_method": "w4afp8",
            "moe_weight_format": "mxfp4",
            "group_size": 32,
        }
    )
    assert glm_config.linear_weight_format == "fp8"
    assert glm_config.is_checkpoint_fp8_serialized is True
    assert glm_config.get_quant_method(dense_layer, "model.layers.0.q_proj") == "fp8"

    with pytest.raises(ValueError, match="linear weight format"):
        W4AFp8Config(linear_weight_format="float32")


def test_kimi_nested_quantization_config_is_visible_at_outer_level():
    quantization_config = {
        "quant_method": "w4afp8",
        "moe_weight_format": "mxfp4",
        "linear_weight_format": "bfloat16",
        "group_size": 32,
    }
    config = KimiK25Config(
        text_config={"quantization_config": dict(quantization_config)}
    )

    assert config.text_config.quantization_config == quantization_config
    assert config.quantization_config == quantization_config


def test_expert_major_residual_scale_matches_reference():
    topk_ids = torch.tensor([[2, 0], [1, 2], [0, 2]], dtype=torch.int32)
    residual = torch.tensor([0.25, 0.5, 1.0], dtype=torch.float32)

    (actual,) = W4AFp8MoEMethod._expert_major_residual_scales(
        topk_ids, residual
    )
    route_scale = residual[topk_ids.long()] * 64.0
    expected = torch.cat(
        [route_scale[topk_ids == expert_id] for expert_id in range(3)]
    )
    torch.testing.assert_close(actual, expected)


def test_mxfp4_preprocess_chunks_only_on_expert_dimension(monkeypatch):
    from sglang.srt.layers.quantization import w4afp8

    monkeypatch.setattr(w4afp8, "MXFP4_PREPROCESS_EXPERT_CHUNK", 2)
    weight = torch.arange(5 * 3 * 2, dtype=torch.uint8).reshape(5, 3, 2)
    scale = torch.arange(5 * 3, dtype=torch.uint8).reshape(5, 3, 1)
    chunk_sizes = []

    def fake_preprocess(weight_chunk, scale_chunk):
        chunk_sizes.append(weight_chunk.shape[0])
        residual = weight_chunk[:, 0, 0].to(torch.float32)
        return weight_chunk + 1, scale_chunk + 2, residual

    actual = W4AFp8MoEMethod._preprocess_mxfp4_in_chunks(
        fake_preprocess, weight, scale
    )
    assert chunk_sizes == [2, 2, 1]
    torch.testing.assert_close(actual[0], weight + 1)
    torch.testing.assert_close(actual[1], scale + 2)
    torch.testing.assert_close(actual[2], weight[:, 0, 0].to(torch.float32))


def test_mxfp4_forward_uses_pr3738_contract(monkeypatch):
    method = W4AFp8MoEMethod(
        W4AFp8Config(moe_weight_format="mxfp4", group_size=32)
    )
    method.moe_runner_config = SimpleNamespace(routed_scaling_factor=2.5)

    layer = SimpleNamespace(
        moe_ep_size=1,
        moe_tp_size=8,
        moe_tp_rank=3,
        w13_weight=torch.zeros((2, 4, 2), dtype=torch.uint8),
        w2_weight=torch.zeros((2, 4, 2), dtype=torch.uint8),
        w13_weight_scale=torch.arange(8, dtype=torch.uint8).reshape(2, 4),
        w2_weight_scale=torch.arange(8, dtype=torch.uint8).reshape(2, 4),
        w13_weight_residual=torch.tensor([0.25, 0.5], dtype=torch.float32),
        w2_weight_residual=torch.tensor([1.0, 2.0], dtype=torch.float32),
        mxfp4_fc2_act_global=torch.ones((), dtype=torch.float32),
    )
    x = torch.randn(2, 4, dtype=torch.bfloat16)
    topk_ids = torch.tensor([[1, 0], [1, 1]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.6, 0.4], [0.7, 0.3]], dtype=torch.float32)
    dispatch_output = SimpleNamespace(
        hidden_states=x,
        topk_output=(topk_weights, topk_ids, None),
    )

    captured = {}

    class ActivationType:
        Swiglu = "swiglu"

    def fake_cutlass_fused_moe(**kwargs):
        captured.update(kwargs)
        kwargs["output"].zero_()

    monkeypatch.setattr(
        method,
        "_get_flashinfer_mxfp4_fp8_helpers",
        lambda: (fake_cutlass_fused_moe, object(), ActivationType),
    )
    result = method._apply_mxfp4_fp8(layer, dispatch_output)

    assert result.hidden_states.shape == x.shape
    assert captured["use_w4_group_scaling"] is True
    assert captured["use_wfp4afp8_humming"] is True
    assert captured["use_packed_weights"] is False
    assert captured["profile_ids"] is None
    assert captured["tp_size"] == 8
    assert captured["tp_rank"] == 3
    assert captured["ep_size"] == 1
    assert captured["ep_rank"] == 0
    torch.testing.assert_close(
        captured["token_final_scales"], topk_weights * 2.5
    )

    quant_scales = captured["quant_scales"]
    assert len(quant_scales) == 5
    assert quant_scales[0].dtype == torch.int32
    assert quant_scales[3].dtype == torch.int32
    torch.testing.assert_close(
        quant_scales[1], torch.tensor([16.0, 32.0, 32.0, 32.0])
    )
    torch.testing.assert_close(
        quant_scales[4], torch.tensor([64.0, 128.0, 128.0, 128.0])
    )
