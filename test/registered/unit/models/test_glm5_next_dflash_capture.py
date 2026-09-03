import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from sglang.srt.models.glm5_next import (
    Glm5NextForConditionalGeneration,
    Glm5NextModel,
)
from sglang.srt.models.glm5_next_nextn import Glm5NextModelNextN
from sglang.test.ci.ci_register import register_cpu_ci
from torch import nn

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_glm5_next_dflash_contracts_mhc_hidden_state():
    model = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(mhc=True, hc_mult=4)
    model.dflash_capture = True

    hidden_states = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    residual = torch.full_like(hidden_states, 2)

    actual = model._prepare_aux_hidden_state(hidden_states, residual)
    expected = (hidden_states + residual).unflatten(-1, (4, -1)).mean(dim=-2)

    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 3)


def test_glm5_next_eagle_capture_keeps_mhc_hidden_state():
    model = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(mhc=True, hc_mult=4)
    model.dflash_capture = False

    hidden_states = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    residual = torch.full_like(hidden_states, 2)

    actual = model._prepare_aux_hidden_state(hidden_states, residual)

    torch.testing.assert_close(actual, hidden_states + residual)


def test_glm5_next_dflash_contracts_mhc_hidden_state_without_residual():
    # GLM-5.3-Flash runs with mhc=True, where MHCLayerCommunicator folds the
    # residual into the widened hidden state and returns residual=None.
    model = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(mhc=True, hc_mult=4)
    model.dflash_capture = True

    hidden_states = torch.arange(24, dtype=torch.float32).reshape(2, 12)

    actual = model._prepare_aux_hidden_state(hidden_states, None)
    expected = hidden_states.unflatten(-1, (4, -1)).mean(dim=-2)

    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 3)


def test_glm5_next_eagle_capture_without_residual():
    model = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(mhc=False, hc_mult=1)
    model.dflash_capture = False

    hidden_states = torch.arange(24, dtype=torch.float32).reshape(2, 12)

    actual = model._prepare_aux_hidden_state(hidden_states, None)

    torch.testing.assert_close(actual, hidden_states)


def test_glm5_next_dflash_maps_target_layers_to_capture_points():
    model = Glm5NextForConditionalGeneration.__new__(Glm5NextForConditionalGeneration)
    nn.Module.__init__(model)
    model.pp_group = SimpleNamespace(is_last_rank=True)
    model.model = SimpleNamespace(dflash_capture=False, layers_to_capture=[])
    model.capture_aux_hidden_states = False

    model.set_dflash_layers_to_capture([5, 14, 24, 33, 42])

    assert model.capture_aux_hidden_states
    assert model.model.dflash_capture
    assert model.model.layers_to_capture == [6, 15, 25, 34, 43]


def test_glm5_nextn_builds_glm_decoder_layer(monkeypatch):
    decoder = object()
    decoder_cls = MagicMock(return_value=decoder)
    monkeypatch.setattr(
        "sglang.srt.models.glm5_next_nextn.Glm5NextDecoderLayer", decoder_cls
    )
    model = Glm5NextModelNextN.__new__(Glm5NextModelNextN)
    config = SimpleNamespace(num_hidden_layers=45, mhc=True)

    actual = model._build_decoder(
        config,
        quant_config=None,
        moe_quant_config_override=None,
        prefix="model.decoder",
        alt_stream=None,
    )

    assert actual is decoder
    decoder_cls.assert_called_once()
    call = decoder_cls.call_args.kwargs
    assert call["config"] is not config
    assert not call["config"].mhc
    assert call | {"config": config} == {
        "config": config,
        "layer_id": 45,
        "quant_config": None,
        "moe_quant_config_override": None,
        "is_nextn": True,
        "prefix": "model.decoder",
        "alt_stream": None,
    }


def test_glm5_nextn_preserves_modelopt_quantization_for_routed_experts():
    model = Glm5NextModelNextN.__new__(Glm5NextModelNextN)
    quant_config = MagicMock()

    assert model._resolve_modelopt_nextn_quant_config(quant_config) is quant_config


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
