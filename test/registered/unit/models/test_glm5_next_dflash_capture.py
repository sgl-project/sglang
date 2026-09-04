import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.models.glm5_next import (
    Glm5NextForConditionalGeneration,
    Glm5NextModel,
)
from sglang.test.ci.ci_register import register_cpu_ci

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


@pytest.mark.parametrize("mhc", [True, False])
def test_glm5_next_pp_proxy_matches_mhc_residual_contract(mhc):
    model = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(mhc=mhc)
    model.pp_group = SimpleNamespace(is_first_rank=False, is_last_rank=False)
    model.start_layer = model.end_layer = 0
    model.first_k_dense_replace = 0
    model.dflash_capture = False
    model.dsa_enable_prefill_cp = False
    model.mla_enable_prefill_cp = False
    model.layers_to_capture = []
    model.enable_a2a_moe = False

    hidden_states = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    proxy_tensors = {"hidden_states": hidden_states}
    if not mhc:
        proxy_tensors["residual"] = torch.full_like(hidden_states, 2)

    output = model.forward(
        input_ids=torch.empty(0, dtype=torch.long),
        positions=torch.empty(0, dtype=torch.long),
        forward_batch=SimpleNamespace(can_run_tbo=False, attn_cp_metadata=None),
        pp_proxy_tensors=PPProxyTensors(proxy_tensors),
    )

    expected_keys = {"hidden_states"} if mhc else {"hidden_states", "residual"}
    assert set(output.tensors) == expected_keys
    torch.testing.assert_close(output["hidden_states"], hidden_states)
    if not mhc:
        torch.testing.assert_close(output["residual"], proxy_tensors["residual"])


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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
