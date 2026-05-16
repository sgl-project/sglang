from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.multimodal_gen.runtime.loader import fsdp_load, weight_utils
from sglang.multimodal_gen.runtime.loader.component_loaders import transformer_loader
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping


class TinyModel(nn.Module):
    param_names_mapping = {r"^hf_weight$": "linear.weight"}

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)


class TwoParamModel(nn.Module):
    param_names_mapping = {
        r"^w1$": "linear_1.weight",
        r"^w2$": "linear_2.weight",
    }

    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(2, 2, bias=False)
        self.linear_2 = nn.Linear(2, 2, bias=False)


def test_streamed_state_dict_load_consumes_weights_without_materializing_source_dict():
    model = TinyModel()
    weight = torch.arange(4, dtype=torch.float32).reshape(2, 2)

    fsdp_load.load_model_from_streamed_state_dict(
        model,
        iter([("hf_weight", weight)]),
        torch.device("cpu"),
        torch.float32,
        strict=True,
        param_names_mapping=get_param_names_mapping(model.param_names_mapping),
    )

    assert torch.equal(model.linear.weight, weight)
    assert model.reverse_param_names_mapping["linear.weight"] == (
        "hf_weight",
        None,
        None,
    )


def test_streamed_state_dict_load_copies_reused_streamer_buffers():
    model = TwoParamModel()
    staging_buffer = torch.ones(2, 2)

    def reused_buffer_iterator():
        yield "w1", staging_buffer
        staging_buffer.fill_(7)
        yield "w2", staging_buffer

    fsdp_load.load_model_from_streamed_state_dict(
        model,
        reused_buffer_iterator(),
        torch.device("cpu"),
        torch.float32,
        strict=True,
        param_names_mapping=get_param_names_mapping(model.param_names_mapping),
    )

    assert torch.equal(model.linear_1.weight, torch.ones(2, 2))
    assert torch.equal(model.linear_2.weight, torch.full((2, 2), 7.0))


def test_streamed_state_dict_load_rejects_merged_mapping():
    model = TinyModel()
    mapping = get_param_names_mapping({r"^w1$": ("linear.weight", 0, 2)})

    with pytest.raises(ValueError, match="merged checkpoint parameters"):
        fsdp_load.load_model_from_streamed_state_dict(
            model,
            iter([("w1", torch.ones(2, 2))]),
            torch.device("cpu"),
            torch.float32,
            strict=True,
            param_names_mapping=mapping,
        )


def test_runai_distributed_streaming_fast_path_gating(monkeypatch):
    monkeypatch.setattr(
        transformer_loader, "can_use_runai_distributed_streamer", lambda: True
    )
    monkeypatch.setattr(
        transformer_loader.envs,
        "SGLANG_RUNAI_DISTRIBUTED_MODEL_STREAMER_MIN_WEIGHT_GB",
        8.0,
    )
    server_args = SimpleNamespace(use_fsdp_inference=False)
    component_server_args = SimpleNamespace(dit_cpu_offload=False)
    quant_spec = SimpleNamespace(runtime_quant_config=None, post_load_hooks=[])

    enabled, reason = transformer_loader._should_use_runai_distributed_streaming(
        server_args,
        component_server_args,
        TinyModel,
        quant_spec,
        [],
    )
    assert enabled
    assert reason == ""

    class MergedMappingModel(TinyModel):
        param_names_mapping = {r"^w1$": ("linear.weight", 0, 2)}

    enabled, reason = transformer_loader._should_use_runai_distributed_streaming(
        server_args,
        component_server_args,
        MergedMappingModel,
        quant_spec,
        [],
    )
    assert not enabled
    assert reason == "merged parameter mapping is required"

    quant_spec.runtime_quant_config = object()
    enabled, reason = transformer_loader._should_use_runai_distributed_streaming(
        server_args,
        component_server_args,
        TinyModel,
        quant_spec,
        [],
    )
    assert not enabled
    assert reason == "quantized transformer load is enabled"


def test_runai_distributed_streaming_skips_small_checkpoints(monkeypatch, tmp_path):
    monkeypatch.setattr(
        transformer_loader, "can_use_runai_distributed_streamer", lambda: True
    )
    monkeypatch.setattr(
        transformer_loader.envs,
        "SGLANG_RUNAI_DISTRIBUTED_MODEL_STREAMER_MIN_WEIGHT_GB",
        1.0,
    )
    checkpoint = tmp_path / "small.safetensors"
    checkpoint.write_bytes(b"0")

    enabled, reason = transformer_loader._should_use_runai_distributed_streaming(
        SimpleNamespace(use_fsdp_inference=False),
        SimpleNamespace(dit_cpu_offload=False),
        TinyModel,
        SimpleNamespace(runtime_quant_config=None, post_load_hooks=[]),
        [str(checkpoint)],
    )

    assert not enabled
    assert "checkpoint is too small" in reason


def test_runai_distributed_streamer_env_fallback(monkeypatch):
    monkeypatch.setattr(weight_utils, "HAS_RUNAI_MODEL_STREAMER", True)
    monkeypatch.setattr(
        weight_utils.envs, "SGLANG_USE_RUNAI_MODEL_STREAMER", True
    )
    monkeypatch.setattr(
        weight_utils.envs,
        "SGLANG_USE_RUNAI_DISTRIBUTED_MODEL_STREAMER",
        True,
    )
    monkeypatch.setattr(weight_utils.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(weight_utils.torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(weight_utils.current_platform, "is_cuda_alike", lambda: True)

    assert weight_utils.can_use_runai_distributed_streamer()

    monkeypatch.setattr(
        weight_utils.envs,
        "SGLANG_USE_RUNAI_DISTRIBUTED_MODEL_STREAMER",
        False,
    )
    assert not weight_utils.can_use_runai_distributed_streamer()
