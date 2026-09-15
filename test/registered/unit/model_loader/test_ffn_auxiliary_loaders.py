"""FFN naming compatibility outside the model-owned checkpoint loaders."""

import re
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.elastic_ep import expert_backup_client
from sglang.srt.layers.quantization.quark import weights as quark_weights
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _model(shapes):
    experts = nn.Module()
    for name, shape in shapes.items():
        experts.register_parameter(name, nn.Parameter(torch.zeros(shape)))
    layer = nn.Module()
    layer.checkpoint_name_mapping = {"mlp": "ffn"}
    layer.ffn = nn.Module()
    layer.ffn.experts = experts
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([layer])
    model.config = SimpleNamespace(intermediate_size=32, num_local_experts=1)
    return model, experts


def test_quark_expert_checkpoint_populates_ffn_weights_scales_and_biases():
    model, experts = _model(
        {
            "w13_weight": (1, 64, 2),
            "w2_weight": (1, 4, 16),
            "w13_weight_scale": (1, 64, 1),
            "w2_weight_scale": (1, 4, 1),
            "w13_weight_bias": (1, 64),
            "w2_weight_bias": (1, 4),
            "w13_input_scale": (1,),
            "w2_input_scale": (1,),
        }
    )
    values = {
        "gate_up_proj.weight": torch.arange(128.0).reshape(64, 2),
        "down_proj.weight": torch.full((4, 16), 3.0),
        "gate_up_proj.weight_scale": torch.arange(64.0).reshape(64, 1),
        "down_proj.weight_scale": torch.full((4, 1), 5.0),
        "gate_up_proj.bias": torch.arange(64.0),
        "down_proj.bias": torch.full((4,), 7.0),
        "gate_up_proj.input_scale": torch.tensor(8.0),
        "down_proj.input_scale": torch.tensor(9.0),
    }
    external_prefix = "model.layers.0.mlp.experts.0."
    pattern = re.compile(
        r"^(.*\.mlp\.experts)\.(\d+)\.(gate_up_proj|down_proj)\."
        r"(weight|weight_scale|input_scale|bias)$"
    )
    parallel = SimpleNamespace(
        moe_tp_rank=0, moe_tp_size=1, moe_ep_rank=0, moe_ep_size=1
    )
    pointers = {name: value.data_ptr() for name, value in model.named_parameters()}
    with (
        patch.object(quark_weights, "get_parallel", return_value=parallel),
        patch.object(quark_weights, "_is_cuda", False),
    ):
        loaded = quark_weights._load_gptoss_quark_expert_weights(
            model,
            [(external_prefix + name, value) for name, value in values.items()],
            pattern,
        )
    assert loaded == set(pointers)
    for target, source in (
        ("w13_weight", "gate_up_proj.weight"),
        ("w13_weight_scale", "gate_up_proj.weight_scale"),
        ("w13_weight_bias", "gate_up_proj.bias"),
    ):
        expected = torch.cat([values[source][0::2], values[source][1::2]])
        torch.testing.assert_close(getattr(experts, target)[0], expected)
    for target, source in (
        ("w2_weight", "down_proj.weight"),
        ("w2_weight_scale", "down_proj.weight_scale"),
        ("w2_weight_bias", "down_proj.bias"),
        ("w13_input_scale", "gate_up_proj.input_scale"),
        ("w2_input_scale", "down_proj.input_scale"),
    ):
        torch.testing.assert_close(getattr(experts, target)[0], values[source])
    assert pointers == {
        name: value.data_ptr() for name, value in model.named_parameters()
    }


def test_elastic_backup_targets_current_ffn_storage_from_checkpoint_names():
    model, experts = _model({"w13_weight": (1, 8, 4)})
    registered, transfers = [], []
    engine = SimpleNamespace(
        register_memory=lambda ptr, size: registered.append((ptr, size)) or 0,
        batch_transfer_sync_read=lambda session, local, remote, sizes: (
            transfers.append((session, local, remote, sizes)) or 0
        ),
    )
    client = expert_backup_client.ExpertBackupClient.__new__(
        expert_backup_client.ExpertBackupClient
    )
    client._get_model = lambda: model
    client.use_backup = True
    client.engine_num = 1
    client.moe_ep_size, client.moe_ep_rank = 1, 0
    client.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(n_routed_experts=1, num_hidden_layers=1)
    )
    client.session_id_list = ["fixture"]
    client.dram_map_list = [
        {
            "model.layers.0.mlp.experts.0.gate_proj.weight": SimpleNamespace(
                weight_ptr=123, byte_size=64
            )
        }
    ]
    locations = SimpleNamespace(logical_to_all_physical=lambda *_: [0])
    with patch(
        "sglang.srt.distributed.parallel_state.get_mooncake_transfer_engine",
        return_value=SimpleNamespace(engine=engine),
    ):
        client.start_transfer_client()
    with (
        patch.object(
            expert_backup_client,
            "get_global_expert_location_metadata",
            return_value=locations,
        ),
        patch.object(
            expert_backup_client,
            "get_exec",
            return_value=SimpleNamespace(
                moe=SimpleNamespace(ep_num_redundant_experts=0)
            ),
        ),
    ):
        client.update_weights()
    assert registered == [(experts.w13_weight.data_ptr(), 128)]
    assert transfers == [
        ("fixture", [experts.w13_weight[0, :4].data_ptr()], [123], [64])
    ]


if __name__ == "__main__":
    import sys

    import pytest

    args = ["-x" if arg == "-f" else arg for arg in sys.argv[1:]]
    sys.exit(pytest.main([__file__, "-v", *args]))
