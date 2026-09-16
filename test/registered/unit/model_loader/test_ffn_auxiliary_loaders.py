"""FFN naming compatibility outside the model-owned checkpoint loaders."""

import re
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from test_ffn_model_loaders import (
    _fixture,
    _packed_loader,
)
from torch import nn

from sglang.srt.elastic_ep import expert_backup_client
from sglang.srt.layers.quantization.quark import weights as quark_weights
from sglang.srt.model_loader import loader as loader_module
from sglang.srt.model_loader.loader import (
    BitsAndBytesModelLoader,
    QuantizedRLModelLoader,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _model(shapes):
    experts = nn.Module()
    for name, shape in shapes.items():
        experts.register_parameter(name, nn.Parameter(torch.zeros(shape)))
    layer = nn.Module()
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


class TestAuxiliaryFFNMetadata(unittest.TestCase):
    def test_bitsandbytes_metadata_attaches_to_loaded_ffn_parameters(self):
        model, ffn = _fixture("qwen3", "Qwen3ForCausalLM")
        ffn.gate_up_proj.weight.pack_factor = 1
        ffn.down_proj.weight.pack_factor = 1
        prefix = "model.layers.0.mlp"
        weights = [
            (f"{prefix}.{name}.weight", torch.full(shape, value))
            for name, shape, value in (
                ("gate_proj", (6, 4), 2.0),
                ("up_proj", (6, 4), 3.0),
                ("down_proj", (4, 6), 4.0),
            )
        ]
        states = {name: SimpleNamespace(shape=weight.shape) for name, weight in weights}
        loader = BitsAndBytesModelLoader.__new__(BitsAndBytesModelLoader)
        loader.target_modules = []
        config = SimpleNamespace(
            model_path="unused",
            revision=None,
            hf_config=SimpleNamespace(model_type="qwen3"),
        )
        with (
            patch.object(
                loader,
                "_get_quantized_weights_iterator",
                return_value=(iter(weights), states),
            ),
            patch.object(loader_module.current_platform, "empty_cache"),
        ):
            loader._load_weights(config, model)
        self.assertIs(
            ffn.gate_up_proj.weight.bnb_quant_state[0],
            states[f"{prefix}.gate_proj.weight"],
        )
        self.assertIs(
            ffn.gate_up_proj.weight.bnb_quant_state[1],
            states[f"{prefix}.up_proj.weight"],
        )
        torch.testing.assert_close(
            ffn.gate_up_proj.weight.bnb_shard_offsets, torch.tensor([0, 24, 48])
        )
        torch.testing.assert_close(
            ffn.gate_up_proj.weight, torch.cat((weights[0][1], weights[1][1]))
        )
        self.assertIs(
            ffn.down_proj.weight.bnb_quant_state[0],
            states[f"{prefix}.down_proj.weight"],
        )

    def test_quantized_rl_updates_storage_and_scale_together(self):
        model, ffn = _fixture("qwen3", "Qwen3ForCausalLM")
        for proj, shape in ((ffn.gate_up_proj, (12, 4)), (ffn.down_proj, (4, 6))):
            proj.weight = nn.Parameter(
                torch.zeros(shape, dtype=torch.float8_e4m3fn), requires_grad=False
            )
            proj.weight_scale = nn.Parameter(
                torch.zeros(1, shape[0]), requires_grad=False
            )
        ffn.gate_up_proj.weight.weight_loader = _packed_loader
        internal = "model.layers.0.ffn"
        external = "model.layers.0.mlp"
        params = dict(model.named_parameters())
        model.original_weights_rebuild_keys = {
            name: {"shape": param.shape, "stride": param.stride()}
            for name, param in params.items()
            if name.endswith(".weight")
        }
        model.recorded_loader = {}
        pointers = {name: param.data_ptr() for name, param in params.items()}
        gate = torch.full((6, 4), 2.0, dtype=torch.bfloat16)
        up = torch.full((6, 4), 3.0, dtype=torch.bfloat16)
        down = torch.full((4, 6), 4.0, dtype=torch.bfloat16)
        # Isolate only the GPU quantizer: storage reset, load, copy-back, scale
        # association and scale writes all execute the production implementation.
        kernel = ModuleType("sglang.kernels.ops.quantization.fp8_kernel")
        kernel.per_token_group_quant_fp8 = lambda weight, group_size: (
            weight.to(torch.float8_e4m3fn),
            torch.full((weight.shape[0], 1), weight[0, 0].item()),
        )
        with (
            patch.dict(sys.modules, {kernel.__name__: kernel}),
            patch.object(
                loader_module,
                "get_parallel",
                return_value=SimpleNamespace(tp_rank=0, tp_size=1),
            ),
        ):
            updated, last = QuantizedRLModelLoader.rebinding_and_load_weights(
                model,
                model.load_weights,
                [
                    (f"{external}.gate_proj.weight", gate),
                    (f"{external}.up_proj.weight", up),
                    (f"{external}.down_proj.weight", down),
                ],
            )
        self.assertEqual(
            set(updated),
            {f"{internal}.gate_up_proj.weight", f"{internal}.down_proj.weight"},
        )
        self.assertFalse(last)
        for name, param in model.named_parameters():
            self.assertEqual(param.data_ptr(), pointers[name], name)
        torch.testing.assert_close(
            ffn.gate_up_proj.weight.float(), torch.cat((gate, up)).float()
        )
        torch.testing.assert_close(ffn.down_proj.weight.float(), down.float())
        torch.testing.assert_close(
            ffn.gate_up_proj.weight_scale, torch.tensor([[2.0] * 6 + [3.0] * 6])
        )
        torch.testing.assert_close(ffn.down_proj.weight_scale, torch.full((1, 4), 4.0))


if __name__ == "__main__":
    import pytest

    args = ["-x" if arg == "-f" else arg for arg in sys.argv[1:]]
    sys.exit(pytest.main([__file__, "-v", *args]))
