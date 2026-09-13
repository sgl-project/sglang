"""Regression tests for Qwen4Exp fused shared-expert checkpoint loading."""

from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

from sglang.srt.models.qwen4_exp import (
    Qwen4ExpForConditionalGeneration,
    _qwen4_exp_fused_shared_expert_mapping,
)
from sglang.srt.models.qwen4_exp_mtp import Qwen4ExpForCausalLMMTP


class _RecordingParameter:
    def __init__(self, calls):
        self.calls = calls

    def weight_loader(self, param, loaded_weight, mapped_name, *, shard_id, expert_id):
        assert param is self
        self.calls.append((mapped_name, shard_id, expert_id, loaded_weight.item()))


class _RecordingStackedParameter:
    def __init__(self, calls, name):
        self.calls = calls
        self.name = name

    def weight_loader(self, param, loaded_weight, shard_id=None):
        assert param is self
        self.calls.append((self.name, shard_id, loaded_weight.item()))


class _RecordingMTPExpertParameter:
    def __init__(self, calls):
        self.calls = calls

    def weight_loader(
        self,
        param,
        loaded_weight,
        mapped_name,
        shard_id=None,
        expert_id=None,
    ):
        assert param is self
        self.calls.append(
            (mapped_name, shard_id, expert_id, loaded_weight.flatten().tolist())
        )


def _make_fused_model(num_layers=48):
    calls = []
    params = {}
    modules = {}
    for layer_id in range(num_layers):
        prefix = f"model.layers.{layer_id}.mlp.experts"
        params[f"{prefix}.w13_weight"] = _RecordingParameter(calls)
        params[f"{prefix}.w2_weight"] = _RecordingParameter(calls)
        modules[f"model.layers.{layer_id}.mlp"] = SimpleNamespace(
            enable_shared_expert_fusion=True
        )

    model = SimpleNamespace(
        config=SimpleNamespace(
            num_experts=512,
            tie_word_embeddings=False,
            encoder_only=False,
            text_config=SimpleNamespace(split_ngram_parts=512),
        ),
        language_model_only=True,
        start_layer=0,
        end_layer=num_layers,
        named_parameters=lambda remove_duplicate=False: params.items(),
        named_buffers=lambda: (),
        named_modules=lambda: modules.items(),
        modules=lambda: (),
        _load_qwen4_exp_ple_buffer=lambda *args: False,
    )
    return model, calls


def _make_unfused_model():
    calls = []
    params = {
        "model.layers.0.mlp.shared_expert.gate_up_proj.weight": (
            _RecordingStackedParameter(calls, "gate_up_proj")
        ),
        "model.layers.0.mlp.shared_expert.down_proj.weight": (
            _RecordingStackedParameter(calls, "down_proj")
        ),
    }
    model = SimpleNamespace(
        config=SimpleNamespace(
            num_experts=512,
            tie_word_embeddings=False,
            encoder_only=False,
            text_config=SimpleNamespace(split_ngram_parts=512),
        ),
        language_model_only=True,
        start_layer=0,
        end_layer=1,
        named_parameters=lambda remove_duplicate=False: params.items(),
        named_buffers=lambda: (),
        named_modules=lambda: (
            (
                "model.layers.0.mlp",
                SimpleNamespace(enable_shared_expert_fusion=False),
            ),
        ),
        modules=lambda: (),
        _load_qwen4_exp_ple_buffer=lambda *args: False,
    )
    return model, calls


def _make_fused_mtp_model(num_experts=2):
    calls = []
    params = {
        "model.layers.0.mlp.experts.w13_weight": _RecordingMTPExpertParameter(calls),
        "model.layers.0.mlp.experts.w2_weight": _RecordingMTPExpertParameter(calls),
    }
    model = SimpleNamespace(
        config=SimpleNamespace(num_experts=num_experts),
        named_parameters=lambda: params.items(),
        modules=lambda: (SimpleNamespace(num_fused_shared_experts=1),),
    )
    return model, calls


def test_shared_expert_projections_load_into_fused_slot():
    model, calls = _make_fused_model(num_layers=1)
    weights = [
        (
            "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
            torch.tensor(1.0),
        ),
        (
            "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
            torch.tensor(2.0),
        ),
        (
            "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
            torch.tensor(3.0),
        ),
    ]

    loaded = Qwen4ExpForConditionalGeneration.load_weights(model, weights)

    assert calls == [
        ("model.layers.0.mlp.experts.w13_weight", "w1", 512, 1.0),
        ("model.layers.0.mlp.experts.w13_weight", "w3", 512, 2.0),
        ("model.layers.0.mlp.experts.w2_weight", "w2", 512, 3.0),
    ]
    assert loaded == {
        "model.layers.0.mlp.experts.w13_weight",
        "model.layers.0.mlp.experts.w2_weight",
    }


def test_all_48_layers_consume_all_144_shared_projection_names():
    model, calls = _make_fused_model()
    projections = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}
    weights = [
        (
            (
                f"model.language_model.layers.{layer_id}.mlp.shared_expert."
                f"{projection}.weight"
            ),
            torch.tensor(float(layer_id)),
        )
        for layer_id in range(48)
        for projection in projections
    ]

    Qwen4ExpForConditionalGeneration.load_weights(model, weights)

    assert len(calls) == 144
    assert {(name, shard, expert) for name, shard, expert, _ in calls} == {
        (
            (
                f"model.layers.{layer_id}.mlp.experts."
                f"{'w2' if projection == 'down_proj' else 'w13'}_weight"
            ),
            shard,
            512,
        )
        for layer_id in range(48)
        for projection, shard in projections.items()
    }


def test_shared_expert_uses_separate_parameters_when_fusion_is_disabled():
    model, calls = _make_unfused_model()
    weights = [
        (
            "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
            torch.tensor(1.0),
        ),
        (
            "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
            torch.tensor(2.0),
        ),
        (
            "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
            torch.tensor(3.0),
        ),
    ]

    loaded = Qwen4ExpForConditionalGeneration.load_weights(model, weights)

    assert calls == [
        ("gate_up_proj", 0, 1.0),
        ("gate_up_proj", 1, 2.0),
        ("down_proj", None, 3.0),
    ]
    assert loaded == {
        "model.layers.0.mlp.shared_expert.gate_up_proj.weight",
        "model.layers.0.mlp.shared_expert.down_proj.weight",
    }


def test_mapping_does_not_capture_routed_visual_or_nonfused_weights():
    params = {"model.layers.0.mlp.experts.w13_weight": object()}

    assert (
        _qwen4_exp_fused_shared_expert_mapping(
            "model.layers.0.mlp.experts.0.gate_proj.weight", params, 512
        )
        is None
    )
    assert (
        _qwen4_exp_fused_shared_expert_mapping(
            "visual.layers.0.mlp.shared_expert.gate_proj.weight", params, 512
        )
        is None
    )
    assert (
        _qwen4_exp_fused_shared_expert_mapping(
            "model.layers.0.mlp.shared_expert.gate_proj.weight", {}, 512
        )
        is None
    )


def test_mtp_shared_expert_loads_after_fused_routed_expert_weights():
    model, calls = _make_fused_mtp_model()
    weights = [
        (
            "mtp.layers.0.mlp.experts.gate_up_proj",
            torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]),
        ),
        (
            "mtp.layers.0.mlp.experts.down_proj",
            torch.tensor([[[5.0]], [[6.0]]]),
        ),
        ("mtp.layers.0.mlp.shared_expert.gate_proj.weight", torch.tensor(7.0)),
        ("mtp.layers.0.mlp.shared_expert.up_proj.weight", torch.tensor(8.0)),
        ("mtp.layers.0.mlp.shared_expert.down_proj.weight", torch.tensor(9.0)),
    ]

    loaded = Qwen4ExpForCausalLMMTP.load_weights(model, weights)

    shared_expert_calls = [call for call in calls if call[2] == 2]
    assert shared_expert_calls == [
        ("model.layers.0.mlp.experts.w13_weight", "w1", 2, [7.0]),
        ("model.layers.0.mlp.experts.w13_weight", "w3", 2, [8.0]),
        ("model.layers.0.mlp.experts.w2_weight", "w2", 2, [9.0]),
    ]
    assert loaded == {
        "model.layers.0.mlp.experts.w13_weight",
        "model.layers.0.mlp.experts.w2_weight",
    }


def test_mtp_missing_parameter_is_not_reported_as_loaded():
    model = SimpleNamespace(
        config=SimpleNamespace(num_experts=None),
        named_parameters=lambda: (),
        modules=lambda: (),
    )

    loaded = Qwen4ExpForCausalLMMTP.load_weights(
        model, [("mtp.required.weight", torch.tensor(1.0))]
    )

    assert loaded == set()


def test_mtp_required_shared_expert_missing_fails_closed():
    model = SimpleNamespace(
        config=SimpleNamespace(num_experts=512),
        named_parameters=lambda: (),
        modules=lambda: (SimpleNamespace(num_fused_shared_experts=1),),
    )

    with pytest.raises(
        ValueError, match="Required MTP shared-expert parameter could not be loaded"
    ):
        Qwen4ExpForCausalLMMTP.load_weights(
            model,
            [("mtp.layers.0.mlp.shared_expert.gate_proj.weight", torch.tensor(1.0))],
        )
