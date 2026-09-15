"""Registered module paths and external quantization metadata must agree.

These tests exercise configuration selection, not CUDA/NPU quantization kernels.
"""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch.nn as nn

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.auto_round import AutoRoundConfig
from sglang.srt.layers.quantization.awq import AWQConfig
from sglang.srt.layers.quantization.bitsandbytes import BitsAndBytesConfig
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.gguf import GGUFConfig
from sglang.srt.layers.quantization.gptq import GPTQConfig
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptMixedPrecisionConfig,
)
from sglang.srt.layers.quantization.quark.quark import QuarkConfig
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.quantization.utils import get_dynamic_override
from sglang.srt.models.inkling_common.quantization.config import (
    InklingModelOptNvfp4Config,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PARENT = "model.layers.0"
INTERNAL = PARENT + ".ffn"
EXTERNAL = PARENT + ".mlp"
PACKED = {"gate_up_proj": ["gate_proj", "up_proj"]}


class DeclaredLayer(nn.Module):
    checkpoint_name_mapping = {"mlp": "ffn"}


class DeclaredSharedExperts(nn.Module):
    checkpoint_name_mapping = {"shared_mlp": "shared_ffn"}


def bind(config):
    # Declarations must be available before even nn.Module.__init__ runs.
    config.register_checkpoint_names(DeclaredLayer.__new__(DeclaredLayer), PARENT)
    return config


def linear():
    layer = LinearBase.__new__(LinearBase)
    nn.Module.__init__(layer)
    return layer


def test_nested_names_are_scoped_and_configs_are_independent():
    config = bind(Fp8Config(ignored_layers=[EXTERNAL + ".shared_mlp.down_proj"]))
    config.register_checkpoint_names(
        DeclaredSharedExperts.__new__(DeclaredSharedExperts), INTERNAL
    )
    path = INTERNAL + ".shared_ffn.down_proj"
    resolved = config
    assert isinstance(
        resolved.get_quant_method(linear(), path), UnquantizedLinearMethod
    )
    for unchanged in (
        "projector.mlp.0",
        "visual.ffn.down_proj",
        "model.layers.1.ffn.down_proj",
    ):
        assert not config.match_layer(unchanged, lambda name: name.startswith(EXTERNAL))
    native = Fp8Config(ignored_layers=[INTERNAL])
    assert isinstance(
        native.get_quant_method(linear(), INTERNAL), UnquantizedLinearMethod
    )
    assert config.ignored_layers == [EXTERNAL + ".shared_mlp.down_proj"]


@pytest.mark.parametrize("mapping", [{"feed_forward": "ffn"}, {"mlp": "other_ffn"}])
def test_conflicting_declarations_are_rejected_without_changing_names(mapping):
    class Conflicting(nn.Module):
        checkpoint_name_mapping = mapping

    config = bind(Fp8Config())
    names = config._checkpoint_names
    before = names.rules.copy(), names.forward_rules.copy()
    with pytest.raises(ValueError, match="Conflicting checkpoint names"):
        config.register_checkpoint_names(Conflicting(), PARENT)
    assert (names.rules, names.forward_rules) == before
    assert names.checkpoint_name(INTERNAL + ".weight") == EXTERNAL + ".weight"
    assert names.internal_name(EXTERNAL + ".weight") == INTERNAL + ".weight"
    config.register_checkpoint_names(DeclaredLayer.__new__(DeclaredLayer), PARENT)
    assert (names.rules, names.forward_rules) == before


@pytest.mark.parametrize("num_experts,first_dense", [(1, 1), (2, 0), (2, 1)])
def test_bailing_v3_ffn_construction_uses_checkpoint_skip_rules(
    num_experts, first_dense
):
    from sglang.srt.models import bailing_moe_v3 as bailing

    quant_config = Fp8Config(ignored_layers=[EXTERNAL + ".down_proj"])
    selected_methods = []

    def make_ffn(*args, prefix, quant_config, **kwargs):
        ffn = nn.Module()
        ffn.down_proj = linear()
        ffn.down_proj.prefix = prefix + ".down_proj"
        selected_methods.append(
            quant_config.get_quant_method(ffn.down_proj, ffn.down_proj.prefix)
        )
        return ffn

    config = SimpleNamespace(
        attention_type=0,
        hidden_size=4,
        intermediate_size=6,
        num_experts=num_experts,
        first_k_dense_replace=first_dense,
        num_hidden_layers=2,
    )
    with (
        patch.object(bailing, "BailingKDA", return_value=nn.Module()),
        patch.object(bailing, "BailingMLP", side_effect=make_ffn),
        patch.object(bailing, "BailingMoE", side_effect=make_ffn),
        patch.object(bailing, "enable_moe_dense_fully_dp", return_value=False),
        patch.object(bailing, "RMSNorm", side_effect=lambda *a, **k: nn.Identity()),
        patch.object(bailing.LayerScatterModes, "init_new", return_value=None),
        patch.object(bailing, "LayerCommunicator", return_value=None),
    ):
        layer = bailing.BailingMoELinearDecoderLayer(
            config, quant_config=quant_config, prefix=PARENT
        )
    assert layer.mlp.down_proj.prefix == EXTERNAL + ".down_proj"
    assert len(selected_methods) == 1
    assert isinstance(selected_methods[0], UnquantizedLinearMethod)


@pytest.mark.parametrize(
    "config",
    [
        Fp8Config(ignored_layers=["mlp"]),
        AWQConfig(4, 128, True, modules_to_not_convert=["mlp"]),
        BitsAndBytesConfig(llm_int8_skip_modules=["mlp"]),
        GGUFConfig(modules_to_not_convert=["mlp"]),
        ModelOptFp4Config(exclude_modules=["*.mlp.*"]),
        InklingModelOptNvfp4Config(group_size=16, exclude_modules=["mlp"]),
    ],
)
def test_real_method_selection_keeps_external_skip_rules(config):
    bind(config)
    prefix = INTERNAL + ".down_proj"
    resolved = config
    assert type(resolved) is type(config)
    method = resolved.get_quant_method(linear(), prefix)
    assert isinstance(method, UnquantizedLinearMethod)


def test_fp8_packed_skip_requires_all_shards():
    config = bind(
        Fp8Config(
            ignored_layers=[EXTERNAL + ".gate_proj", EXTERNAL + ".up_proj"],
            packed_modules_mapping=PACKED,
        )
    )
    assert isinstance(
        config.get_quant_method(linear(), INTERNAL + ".gate_up_proj"),
        UnquantizedLinearMethod,
    )
    config.ignored_layers.pop()
    with pytest.raises(ValueError, match="some but not all"):
        config.get_quant_method(linear(), INTERNAL + ".gate_up_proj")


def test_gguf_skip_keeps_undeclared_paths_and_source_rules():
    config = bind(GGUFConfig(modules_to_not_convert=["mlp"]))
    assert isinstance(
        config.get_quant_method(linear(), "projector.mlp.0"),
        UnquantizedLinearMethod,
    )
    assert not isinstance(
        config.get_quant_method(linear(), "model.layers.1.ffn.down_proj"),
        UnquantizedLinearMethod,
    )
    assert config.modules_to_not_convert == ["mlp"]


def test_gptq_regex_overrides_and_negative_method_selection():
    dynamic = {
        r"-:model\.layers\.0\.mlp\.down_proj$": {},
        r"+:.*\.mlp\.gate_up_proj$": {"bits": 8},
    }
    config = bind(GPTQConfig(4, 128, False, False, dynamic))
    assert (
        get_dynamic_override(
            config,
            INTERNAL + ".gate_up_proj",
            "bits",
        )
        == 8
    )
    assert isinstance(
        config.get_quant_method(linear(), INTERNAL + ".down_proj"),
        UnquantizedLinearMethod,
    )
    assert config.dynamic == dynamic


def test_mixed_precision_queries_and_delegates_share_late_declarations():
    config = bind(
        ModelOptMixedPrecisionConfig.from_config(
            {
                "quant_algo": "MIXED_PRECISION",
                "ignore": [EXTERNAL + ".shared_mlp.down_proj"],
                "quantized_layers": {
                    EXTERNAL + ".gate_proj": {"quant_algo": "FP8"},
                    EXTERNAL + ".up_proj": {"quant_algo": "FP8"},
                },
            }
        )
    )
    config.update_packed_modules_mapping(PACKED)
    assert config.resolve_quant_algo(INTERNAL + ".gate_up_proj") == "FP8"
    config.register_checkpoint_names(
        DeclaredSharedExperts.__new__(DeclaredSharedExperts), INTERNAL
    )
    prefix = INTERNAL + ".shared_ffn.down_proj"
    resolved = config
    assert resolved.is_layer_excluded(prefix)
    config.quantized_layers[EXTERNAL + ".up_proj"]["quant_algo"] = "NVFP4"
    with pytest.raises(ValueError, match="Mixed quant_algo"):
        config.resolve_quant_algo(INTERNAL + ".gate_up_proj")


def test_compressed_tensors_exact_and_regex_targets_and_ignore():
    target = r"re:.*\.mlp\.gate_proj$"
    config = bind(
        CompressedTensorsConfig.from_config(
            {
                "format": "float-quantized",
                "ignore": [r"re:.*\.mlp\.down_proj$"],
                "config_groups": {
                    "g": {
                        "targets": [target, EXTERNAL + ".up_proj"],
                        "weights": {
                            "num_bits": 8,
                            "type": "float",
                            "strategy": "channel",
                            "symmetric": True,
                        },
                    }
                },
            }
        )
    )
    assert (
        config.get_scheme_dict(linear(), INTERNAL + ".gate_proj")
        == config.target_scheme_map[target]
    )
    assert config.get_scheme_dict(linear(), INTERNAL + ".up_proj") is not None
    assert config.get_scheme_dict(linear(), INTERNAL + ".down_proj") is None


def test_quark_glob_and_packed_consistency():
    config = bind(
        QuarkConfig(
            quant_config={
                "packed_modules_mapping": PACKED,
                "layer_quant_config": {"*.mlp.*_proj": {"selected": True}},
                "layer_type_quant_config": {},
                "global_quant_config": {"selected": False},
            }
        )
    )
    assert config._find_matched_config(INTERNAL + ".gate_up_proj", linear()) == {
        "selected": True
    }
    assert config._find_matched_config("visual.ffn.down_proj", linear()) == {
        "selected": False
    }


def test_autoround_extra_config_names_are_not_rewritten():
    extra = {EXTERNAL + ".down_proj": {"bits": 16}, r".*\.mlp\.gate_proj$": {"bits": 8}}
    config = bind(AutoRoundConfig(4, 128, extra_config=deepcopy(extra)))
    assert config.get_layer_config(linear(), INTERNAL + ".down_proj")[0] == 16
    assert config.get_layer_config(linear(), INTERNAL + ".gate_proj")[0] == 8
    assert config.extra_config == extra


def test_humming_positive_and_negative_regex_queries():
    from sglang.srt.layers.quantization import humming

    # Schema/kernel dependencies are not needed to verify query selection.
    config = bind(humming.HummingConfig.__new__(humming.HummingConfig))
    external = {
        "quant_method": "gptq",
        "bits": 4,
        "dynamic": {
            r"-:.*\.mlp\.down_proj$": {},
            r"+:.*\.mlp\.gate_proj$": {"bits": 8},
        },
    }
    config.full_config = external
    config.packed_modules_mapping = {}
    down = config
    assert down.is_layer_skipped(down.full_config, INTERNAL + ".down_proj")
    gate = config
    with patch.object(
        humming, "_build_checkpoint_weight_schema", side_effect=lambda cfg: cfg["bits"]
    ):
        assert (
            gate.get_layer_weight_schema(gate.full_config, INTERNAL + ".gate_proj") == 8
        )


@pytest.mark.parametrize("family", ["phi", "persimmon"])
def test_dense_decoder_propagates_prefix_into_quantized_children(family):
    from sglang.srt.models import persimmon, phi

    module, decoder, attention, children = (
        (phi, phi.PhiLayer, "PhiAttention", ("fc1", "fc2"))
        if family == "phi"
        else (
            persimmon,
            persimmon.PersimmonDecoderLayer,
            "PersimmonAttention",
            ("dense_h_to_4h", "dense_4h_to_h"),
        )
    )
    config = SimpleNamespace(
        hidden_size=4, intermediate_size=8, hidden_act="gelu", layer_norm_eps=1e-6
    )
    quant_config = Fp8Config(ignored_layers=[EXTERNAL])
    seen = []

    def build_linear(*args, quant_config, prefix, **kwargs):
        seen.append(prefix)
        assert isinstance(
            quant_config.get_quant_method(linear(), prefix),
            UnquantizedLinearMethod,
        )
        return nn.Identity()

    with (
        patch.object(module, attention, return_value=nn.Identity()),
        patch.object(module, "ColumnParallelLinear", side_effect=build_linear),
        patch.object(module, "RowParallelLinear", side_effect=build_linear),
    ):
        layer = decoder(config, quant_config=quant_config, prefix=PARENT)
    assert seen == [EXTERNAL + "." + child for child in children]
    assert "mlp" in layer._modules


def test_shared_linear_keeps_registered_prefix_and_original_config():
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

    config = bind(Fp8Config(ignored_layers=[EXTERNAL + ".down_proj"]))
    original = Fp8Config.get_quant_method
    seen = []

    def select(self, layer, prefix):
        assert self is config
        seen.append(prefix)
        return original(self, layer, prefix)

    with patch.object(Fp8Config, "get_quant_method", select):
        skipped = LinearBase(4, 4, quant_config=config, prefix=INTERNAL + ".down_proj")
        quantized = LinearBase(
            4, 4, quant_config=config, prefix=INTERNAL + ".gate_up_proj"
        )
    assert isinstance(skipped.quant_method, UnquantizedLinearMethod)
    assert isinstance(quantized.quant_method, Fp8LinearMethod)
    assert seen == [INTERNAL + ".down_proj", INTERNAL + ".gate_up_proj"]
    assert config.ignored_layers == [EXTERNAL + ".down_proj"]


def test_external_ffn_selector_does_not_match_a_renamed_mlp():
    config = bind(ModelOptFp4Config(exclude_modules=["*.ffn.*"]))
    resolved = config
    assert not resolved.is_layer_excluded(INTERNAL + ".down_proj")


def test_compressed_tensors_matches_expert_projection_targets():
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    config = bind(
        CompressedTensorsConfig.from_config(
            {
                "format": "float-quantized",
                "ignore": [],
                "config_groups": {
                    "g": {
                        "targets": [r"re:.*\.mlp\.experts\.0\..*_proj$"],
                        "weights": {
                            "num_bits": 8,
                            "type": "float",
                            "strategy": "channel",
                            "symmetric": True,
                        },
                    }
                },
            }
        )
    )
    moe = FusedMoE.__new__(FusedMoE)
    nn.Module.__init__(moe)
    prefix = INTERNAL + ".experts"
    resolved = config
    for projection in ["gate_proj", "up_proj", "down_proj"]:
        name = prefix + ".0." + projection
        assert resolved.get_scheme_dict(moe, name)["weights"].num_bits == 8


def test_compressed_tensors_external_ffn_regex_cannot_shadow_mlp_rule():
    config = bind(
        CompressedTensorsConfig.from_config(
            {
                "format": "pack-quantized",
                "ignore": [],
                "config_groups": {
                    "native_ffn": {
                        "targets": [r"re:.*\.ffn\..*"],
                        "weights": {
                            "num_bits": 4,
                            "type": "int",
                            "strategy": "channel",
                            "symmetric": True,
                        },
                    },
                    "renamed_mlp": {
                        "targets": [r"re:.*\.mlp\..*"],
                        "weights": {
                            "num_bits": 8,
                            "type": "int",
                            "strategy": "channel",
                            "symmetric": True,
                        },
                    },
                },
            }
        )
    )
    prefix = INTERNAL + ".down_proj"
    resolved = config
    assert resolved.get_scheme_dict(linear(), prefix)["weights"].num_bits == 8


def test_exact_metadata_is_shared_and_updated_by_subtree_without_changing_source():
    from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8Config

    first = EXTERNAL + ".down_proj.weight"
    second = "model.layers.1.mlp.down_proj.weight"
    metadata = {first: "FLOAT", second: "W8A8_DYNAMIC", "quant_method": "w8a8_int8"}
    config = bind(W8A8Int8Config(metadata))
    resolved0 = config
    internal0 = INTERNAL + ".down_proj.weight"
    assert resolved0.quant_description[internal0] == "FLOAT"
    config.register_checkpoint_names(
        DeclaredLayer.__new__(DeclaredLayer), "model.layers.1"
    )
    resolved1 = config
    assert resolved1.quant_description is resolved0.quant_description
    assert (
        resolved1.quant_description["model.layers.1.ffn.down_proj.weight"]
        == "W8A8_DYNAMIC"
    )
    assert first not in list(resolved1.quant_description)
    assert second not in list(resolved1.quant_description)
    assert metadata == {
        first: "FLOAT",
        second: "W8A8_DYNAMIC",
        "quant_method": "w8a8_int8",
    }
    assert config.quant_description[internal0] == metadata[first]


@pytest.mark.parametrize("target", [r"re:.*\.mlp\..*", "Linear"])
@pytest.mark.parametrize("ignored", [False, True])
def test_compressed_tensors_sparsity_filters_target_keys_before_matching(
    target, ignored
):
    scheme = object()
    config = bind(
        CompressedTensorsConfig(
            target_scheme_map={},
            ignore=[],
            quant_format="dense",
            sparsity_scheme_map={target: scheme},
            sparsity_ignore_list=[target] if ignored else [],
        )
    )
    prefix = INTERNAL + ".down_proj"
    resolved = config
    for selected, name in ((config, EXTERNAL + ".down_proj"), (resolved, prefix)):
        with patch.object(selected, "supports_cutlass_24", return_value=False) as check:
            assert selected.get_linear_scheme(linear(), name) is None
        assert check.call_args.kwargs["sparsity_scheme"] is (
            None if ignored else scheme
        )
    assert config.sparsity_ignore_list == ([target] if ignored else [])


def test_humming_environment_metadata_stays_on_humming_configs():
    from sglang.srt.environ import envs
    from sglang.srt.layers.quantization.base_config import QuantizationConfig
    from sglang.srt.layers.quantization.humming import HummingConfig

    config = bind(HummingConfig.__new__(HummingConfig))
    config.full_config = {"quant_method": "gptq", "bits": 4}
    config.packed_modules_mapping = {}
    policy = {"dynamic": {r"-:.*\.mlp\.down_proj$": {}}}
    prefix = INTERNAL + ".down_proj"
    with (
        envs.SGLANG_HUMMING_INPUT_QUANT_CONFIG.override(json.dumps(policy)),
        envs.SGLANG_HUMMING_ONLINE_QUANT_CONFIG.override(json.dumps(policy)),
    ):
        resolved = config
        fp8 = bind(Fp8Config())
    for field in ("humming_input_quant_config", "humming_online_quant_config"):
        assert field not in vars(QuantizationConfig)
        assert field not in vars(fp8)
        assert resolved.is_layer_skipped(policy, prefix)
    assert policy == {"dynamic": {r"-:.*\.mlp\.down_proj$": {}}}


@pytest.mark.parametrize("use_kt", [False, True])
@pytest.mark.parametrize("renamed", [False, True])
def test_mxfp4_humming_config_is_bound_before_weights_and_kt(
    monkeypatch, use_kt, renamed
):
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe import MoeA2ABackend, MoeRunnerBackend
    from sglang.srt.layers.moe.fused_moe_triton import layer as module
    from sglang.srt.layers.quantization.humming_utils import humming_is_layer_skipped
    from sglang.srt.layers.quantization.mxfp4_humming_moe import Mxfp4HummingMoEMethod
    from sglang.srt.runtime_context import get_context, get_flags, get_parallel

    prefix = INTERNAL + ".experts"
    policy = {"a_dtype": "float8e4m3", "dynamic": {r"-:.*\.mlp\.experts$": {}}}
    config = bind(Fp8Config()) if renamed else Fp8Config()
    events = []
    with envs.SGLANG_HUMMING_INPUT_QUANT_CONFIG.override(json.dumps(policy)):
        method = Mxfp4HummingMoEMethod(None, prefix)

    def check_policy():
        assert humming_is_layer_skipped(method.input_quant_config, prefix) is renamed
        assert method.prefix == prefix
        if not renamed:
            assert method.input_quant_config["a_dtype"] == "float8e4m3"

    def select(resolved, layer, prefix):
        assert resolved is layer.quant_config
        assert "humming_input_quant_config" not in vars(resolved)
        events.append("select")
        return method

    def create_weights(**kwargs):
        check_policy()
        events.append("weights")

    def create_runner(layer, config):
        method.runner = SimpleNamespace()

    def wrap(gpu_method, kt_config):
        assert gpu_method is method
        check_policy()
        events.append("kt")
        # CPU/GPU expert execution is outside this metadata-binding test.
        return gpu_method

    monkeypatch.setattr(Fp8Config, "get_quant_method", select)
    monkeypatch.setattr(method, "create_weights", create_weights)
    monkeypatch.setattr(method, "create_moe_runner", create_runner)
    monkeypatch.setattr(
        module, "create_moe_dispatcher", lambda config: SimpleNamespace()
    )
    monkeypatch.setattr(
        module,
        "create_kt_config_from_server_args",
        lambda *args: object() if use_kt else None,
    )
    monkeypatch.setattr(module, "KTEPWrapperMethod", wrap)
    with (
        get_context().override_server_args(model_path="dummy"),
        get_flags().moe.override(
            runner_backend=MoeRunnerBackend.HUMMING, a2a_backend=MoeA2ABackend.NONE
        ),
        get_parallel().override(
            moe_ep_size=1,
            moe_ep_rank=0,
            moe_tp_size=1,
            moe_tp_rank=0,
            tp_size=1,
            tp_rank=0,
        ),
    ):
        layer = module.FusedMoE(
            num_experts=2,
            hidden_size=4,
            intermediate_size=8,
            layer_id=0,
            quant_config=config,
            prefix=prefix,
        )
    assert layer.quant_method is method
    assert events == (["select", "kt", "weights"] if use_kt else ["select", "weights"])
    assert policy == {"a_dtype": "float8e4m3", "dynamic": {r"-:.*\.mlp\.experts$": {}}}


def test_mxfp4_humming_passes_input_config_to_weight_preparation():
    from unittest.mock import Mock

    import torch

    from sglang.srt.environ import envs
    from sglang.srt.layers.quantization import humming_utils
    from sglang.srt.layers.quantization.mxfp4_humming_moe import Mxfp4HummingMoEMethod

    policy = {"a_dtype": "float8e4m3"}
    with envs.SGLANG_HUMMING_INPUT_QUANT_CONFIG.override(json.dumps(policy)):
        method = Mxfp4HummingMoEMethod(Mock(), INTERNAL + ".experts")
    layer = nn.Module()
    layer.w13_weight_scale_inv = nn.Parameter(torch.ones(2), requires_grad=False)
    layer.w2_weight_scale_inv = nn.Parameter(torch.ones(2), requires_grad=False)
    # No layer.quant_config is needed to pass the method-owned metadata.
    with patch.object(humming_utils, "prepare_humming_moe_layer") as prepare:
        method.process_weights_after_loading(layer)
    prepare.assert_called_once_with(
        layer, {"quant_method": "mxfp4"}, input_quant_config=method.input_quant_config
    )
    assert method.input_quant_config == policy
    assert layer.w13_weight_scale.dtype == torch.float8_e8m0fnu
    assert layer.w2_weight_scale.dtype == torch.float8_e8m0fnu


if __name__ == "__main__":
    import sys

    args = ["-x" if arg == "-f" else arg for arg in sys.argv[1:]]
    sys.exit(pytest.main([__file__, "-v", *args]))
