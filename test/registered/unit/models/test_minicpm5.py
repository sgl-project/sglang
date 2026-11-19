from types import SimpleNamespace

import pytest
from torch import nn

from sglang.srt.model_executor.model_runner_components.attention_backend_setup import (
    configure_aux_hidden_state_capture,
)
from sglang.srt.models.glm4_moe import Glm4MoeModel
from sglang.srt.models.minicpm5 import MiniCPM5ForCausalLM, MiniCPM5Model
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def test_dspark_captures_at_the_next_glm4_moe_layer_boundary():
    model = MiniCPM5Model.__new__(MiniCPM5Model)
    nn.Module.__init__(model)
    model.layers_to_capture = []
    target = MiniCPM5ForCausalLM.__new__(MiniCPM5ForCausalLM)
    nn.Module.__init__(target)
    target.pp_group = SimpleNamespace(is_last_rank=True)
    target.model = model
    target.capture_aux_hidden_states = False

    def configure(layer_ids):
        configure_aux_hidden_state_capture(
            model=target,
            eagle_use_aux_hidden_state=False,
            eagle_aux_hidden_state_layer_ids=None,
            dflash_use_aux_hidden_state=True,
            dflash_target_layer_ids=layer_ids,
            is_dspark=True,
        )

    assert MiniCPM5Model.forward is Glm4MoeModel.forward
    assert not target.capture_aux_hidden_states
    with pytest.raises(ValueError, match="explicit layer_ids"):
        configure(None)
    configure([1, 7, 13, 19, 25])
    assert target.capture_aux_hidden_states
    assert model.layers_to_capture == [2, 8, 14, 20, 26]
