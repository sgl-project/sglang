import json
import logging
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


def test_expert_major_residual_scale_compacts_local_ids_before_sentinel():
    topk_ids = torch.tensor([[1, -1], [0, 1]], dtype=torch.int32)
    fc1_residual = torch.tensor([0.25, 0.5], dtype=torch.float32)
    fc2_residual = torch.tensor([1.0, 2.0], dtype=torch.float32)

    fc1_actual, fc2_actual = W4AFp8MoEMethod._expert_major_residual_scales(
        topk_ids, fc1_residual, fc2_residual
    )

    # Stable expert-major order is route indices [2, 0, 3], followed by the
    # remote route at index 1 with a neutral scale.
    torch.testing.assert_close(
        fc1_actual, torch.tensor([16.0, 32.0, 32.0, 1.0])
    )
    torch.testing.assert_close(
        fc2_actual, torch.tensor([64.0, 128.0, 128.0, 1.0])
    )


def test_expert_major_residual_scale_handles_empty_and_all_remote_routes():
    residual = torch.tensor([0.25, 0.5], dtype=torch.float32)

    (empty,) = W4AFp8MoEMethod._expert_major_residual_scales(
        torch.empty((0, 2), dtype=torch.int32), residual
    )
    assert empty.shape == (0,)

    (all_remote,) = W4AFp8MoEMethod._expert_major_residual_scales(
        torch.full((2, 2), -1, dtype=torch.int32), residual
    )
    torch.testing.assert_close(all_remote, torch.ones(4))


@pytest.mark.parametrize(
    "residuals, error_match",
    [
        ((torch.ones(2, 1),), "1D tensors"),
        ((torch.ones(2), torch.ones(3)), "one value per local expert"),
    ],
)
def test_expert_major_residual_scale_rejects_bad_residual_shape(
    residuals, error_match
):
    with pytest.raises(ValueError, match=error_match):
        W4AFp8MoEMethod._expert_major_residual_scales(
            torch.tensor([[0, 1]], dtype=torch.int32), *residuals
        )


@pytest.mark.parametrize("invalid_id", [-2, 2])
def test_expert_major_residual_scale_rejects_invalid_local_id(invalid_id):
    with pytest.raises(ValueError, match="local IDs"):
        W4AFp8MoEMethod._expert_major_residual_scales(
            torch.tensor([[0, invalid_id]], dtype=torch.int32), torch.ones(2)
        )


@pytest.mark.parametrize(
    "residual_name, residual",
    [
        ("w13_weight_residual", torch.ones(31)),
        ("w2_weight_residual", torch.ones(32, 1)),
    ],
)
def test_mxfp4_residual_shape_must_match_local_weight_count(
    residual_name, residual
):
    layer = SimpleNamespace(
        w13_weight=torch.empty((32, 1, 1)),
        w2_weight=torch.empty((32, 1, 1)),
        w13_weight_residual=torch.ones(32),
        w2_weight_residual=torch.ones(32),
    )
    setattr(layer, residual_name, residual)

    with pytest.raises(ValueError, match=residual_name):
        W4AFp8MoEMethod._validate_mxfp4_residual_shapes(layer)


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


def _valid_ep_server_args(**overrides):
    values = {
        "disable_shared_experts_fusion": True,
        "enable_eplb": False,
        "ep_num_redundant_experts": 0,
        "init_expert_location": "trivial",
        "elastic_ep_backend": None,
        "enable_elastic_expert_backup": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _ep_method_and_layer():
    method = W4AFp8MoEMethod(
        W4AFp8Config(moe_weight_format="mxfp4", group_size=32)
    )
    method.moe_runner_config = SimpleNamespace(
        num_experts=256,
        num_local_experts=32,
        num_fused_shared_experts=0,
        routed_scaling_factor=1.0,
    )
    layer = SimpleNamespace(
        moe_ep_size=8,
        moe_ep_rank=3,
        moe_tp_size=1,
        moe_tp_rank=0,
        _num_global_routed=256,
        _num_local_routed=32,
    )
    return method, layer


def test_mxfp4_ep_topology_accepts_standard_local_id_contract():
    method, layer = _ep_method_and_layer()
    method._validate_mxfp4_ep_topology(
        layer,
        server_args=_valid_ep_server_args(),
        runner_backend="auto",
        a2a_backend="none",
    )


@pytest.mark.parametrize(
    "server_overrides, runner_backend, a2a_backend, layer_overrides, config_overrides, error_match",
    [
        ({}, "triton", "none", {}, {}, "moe_runner_backend"),
        ({}, "auto", "deepep", {}, {}, "moe_a2a_backend"),
        ({}, "auto", "none", {"moe_tp_size": 2}, {}, "expected TP1"),
        (
            {"disable_shared_experts_fusion": False},
            "auto",
            "none",
            {},
            {},
            "shared-expert fusion",
        ),
        ({"enable_eplb": True}, "auto", "none", {}, {}, "EPLB"),
        (
            {"ep_num_redundant_experts": 1},
            "auto",
            "none",
            {},
            {},
            "ep_num_redundant_experts",
        ),
        (
            {"init_expert_location": "random"},
            "auto",
            "none",
            {},
            {},
            "init_expert_location",
        ),
        (
            {"elastic_ep_backend": "nixl"},
            "auto",
            "none",
            {},
            {},
            "elastic_ep_backend",
        ),
        (
            {"enable_elastic_expert_backup": True},
            "auto",
            "none",
            {},
            {},
            "elastic expert backup",
        ),
        (
            {},
            "auto",
            "none",
            {"_num_local_routed": 31},
            {},
            "partition is not uniform",
        ),
        (
            {},
            "auto",
            "none",
            {},
            {"num_fused_shared_experts": 1},
            "num_fused_shared_experts",
        ),
    ],
)
def test_mxfp4_ep_topology_rejects_unsupported_configuration(
    server_overrides,
    runner_backend,
    a2a_backend,
    layer_overrides,
    config_overrides,
    error_match,
):
    method, layer = _ep_method_and_layer()
    for name, value in layer_overrides.items():
        setattr(layer, name, value)
    for name, value in config_overrides.items():
        setattr(method.moe_runner_config, name, value)

    with pytest.raises(RuntimeError, match=error_match):
        method._validate_mxfp4_ep_topology(
            layer,
            server_args=_valid_ep_server_args(**server_overrides),
            runner_backend=runner_backend,
            a2a_backend=a2a_backend,
        )


def test_mxfp4_ep_topology_marker_is_once_per_rank(monkeypatch, caplog):
    from sglang.srt import distributed
    from sglang.srt.layers.quantization import w4afp8

    method, layer = _ep_method_and_layer()
    monkeypatch.setattr(distributed, "get_tensor_model_parallel_rank", lambda: 3)
    w4afp8._MXFP4_TOPOLOGY_LOGGED.clear()
    caplog.set_level(logging.WARNING, logger=w4afp8.__name__)

    method._log_mxfp4_ep_topology(layer)
    method._log_mxfp4_ep_topology(layer)

    records = [
        record.message
        for record in caplog.records
        if record.message.startswith("W4AFP8_MXFP4_EP_TOPOLOGY ")
    ]
    assert len(records) == 1
    payload = json.loads(records[0].split(" ", 1)[1])
    assert payload == {
        "dispatcher": "standard",
        "ep_rank": 3,
        "ep_size": 8,
        "expert_id_space": "local_or_minus_one",
        "flashinfer_ep_rank": 0,
        "flashinfer_ep_size": 1,
        "flashinfer_tp_rank": 0,
        "flashinfer_tp_size": 1,
        "global_routed_experts": 256,
        "local_routed_experts": 32,
        "moe_tp_rank": 0,
        "moe_tp_size": 1,
        "outer_tp_rank": 3,
    }
    w4afp8._MXFP4_TOPOLOGY_LOGGED.clear()


def test_mxfp4_ep_forward_preserves_sentinel_and_uses_local_kernel(monkeypatch):
    method, layer = _ep_method_and_layer()
    method._mxfp4_ep_topology_validated = True
    method.moe_runner_config.routed_scaling_factor = 2.5
    layer.w13_weight = torch.zeros((32, 4, 2), dtype=torch.uint8)
    layer.w2_weight = torch.zeros((32, 4, 2), dtype=torch.uint8)
    layer.w13_weight_scale = torch.arange(128, dtype=torch.uint8).reshape(32, 4)
    layer.w2_weight_scale = torch.arange(128, dtype=torch.uint8).reshape(32, 4)
    layer.w13_weight_residual = torch.arange(1, 33, dtype=torch.float32) / 4
    layer.w2_weight_residual = torch.arange(1, 33, dtype=torch.float32)
    layer.mxfp4_fc2_act_global = torch.ones((), dtype=torch.float32)

    x = torch.randn(2, 4, dtype=torch.bfloat16)
    topk_ids = torch.tensor([[1, -1], [0, 1]], dtype=torch.int32)
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
    assert captured["tp_size"] == 1
    assert captured["tp_rank"] == 0
    assert captured["ep_size"] == 1
    assert captured["ep_rank"] == 0
    torch.testing.assert_close(captured["token_selected_experts"], topk_ids)
    torch.testing.assert_close(
        captured["token_final_scales"], topk_weights * 2.5
    )
    torch.testing.assert_close(
        captured["quant_scales"][1],
        torch.tensor([16.0, 32.0, 32.0, 1.0]),
    )
    torch.testing.assert_close(
        captured["quant_scales"][4],
        torch.tensor([64.0, 128.0, 128.0, 1.0]),
    )


def test_mxfp4_ep_all_remote_rank_starts_with_exact_zero_output(monkeypatch):
    method, layer = _ep_method_and_layer()
    method._mxfp4_ep_topology_validated = True
    layer.w13_weight = torch.zeros((32, 4, 2), dtype=torch.uint8)
    layer.w2_weight = torch.zeros((32, 4, 2), dtype=torch.uint8)
    layer.w13_weight_scale = torch.zeros((32, 4), dtype=torch.uint8)
    layer.w2_weight_scale = torch.zeros((32, 4), dtype=torch.uint8)
    layer.w13_weight_residual = torch.ones(32, dtype=torch.float32)
    layer.w2_weight_residual = torch.ones(32, dtype=torch.float32)
    layer.mxfp4_fc2_act_global = torch.ones((), dtype=torch.float32)

    class ActivationType:
        Swiglu = "swiglu"

    def fake_cutlass_fused_moe(**kwargs):
        # An all-invalid local kernel is allowed to skip every output row.
        assert torch.all(kwargs["token_selected_experts"] == -1)

    monkeypatch.setattr(
        method,
        "_get_flashinfer_mxfp4_fp8_helpers",
        lambda: (fake_cutlass_fused_moe, object(), ActivationType),
    )
    dispatch_output = SimpleNamespace(
        hidden_states=torch.ones((2, 4), dtype=torch.bfloat16),
        topk_output=(
            torch.full((2, 2), 0.5, dtype=torch.float32),
            torch.full((2, 2), -1, dtype=torch.int32),
            None,
        ),
    )

    result = method._apply_mxfp4_fp8(layer, dispatch_output)
    assert torch.count_nonzero(result.hidden_states).item() == 0


def test_standard_dispatcher_ep_rank3_maps_global_ids_to_local_or_minus_one(
    monkeypatch,
):
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import standard
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.moe.utils import MoeRunnerBackend

    monkeypatch.setattr(
        standard, "get_moe_expert_parallel_world_size", lambda: 8
    )
    monkeypatch.setattr(standard, "get_moe_expert_parallel_rank", lambda: 3)
    monkeypatch.setattr(
        standard, "get_moe_runner_backend", lambda: MoeRunnerBackend.AUTO
    )
    monkeypatch.setattr(standard, "get_device", lambda: torch.device("cpu"))

    dispatcher = standard.StandardDispatcher(
        MoeRunnerConfig(
            num_experts=256,
            num_local_experts=32,
            num_fused_shared_experts=0,
        )
    )
    topk_output = StandardTopKOutput(
        topk_weights=torch.ones((1, 4)),
        topk_ids=torch.tensor([[95, 96, 127, 128]], dtype=torch.int32),
        router_logits=torch.empty(0),
    )
    actual = dispatcher.dispatch(torch.ones((1, 4)), topk_output)
    torch.testing.assert_close(
        actual.topk_output.topk_ids,
        torch.tensor([[-1, 0, 31, -1]], dtype=torch.int32),
    )
