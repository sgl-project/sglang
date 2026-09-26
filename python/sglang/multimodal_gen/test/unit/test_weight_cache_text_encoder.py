# SPDX-License-Identifier: Apache-2.0
"""Real H3 loader capability, pure admission and multi-component identity."""

import copy
import json
import pickle
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.loader.component_loaders import text_encoder_loader
from sglang.multimodal_gen.runtime.loader.native_encoder_state import (
    MINIMAX_H3_TEXT_ENCODER,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline import (
    MiniMaxH3Pipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.prepare import prepare_pipeline
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.weight_cache import identity
from sglang.multimodal_gen.runtime.weight_cache.placement import (
    requested_component_names,
)
from sglang.multimodal_gen.test.unit.test_weight_cache_admission import (
    prepared_wan as prepared_wan,
)
from sglang.multimodal_gen.test.unit.test_weight_cache_minimax_h3 import (  # noqa: F401
    h3_args,
)


def variant(args, **overrides):
    values = pickle.loads(args._raw_inputs)
    values.update(overrides)
    with patch.object(ServerArgs, "_adjust_network_ports"):
        return ServerArgs(**values)


@pytest.fixture
def h3_both(h3_args):
    root = Path(h3_args.model_path) / "FL2VA"
    index = json.loads((root / "model_index.json").read_text())
    index["text_encoder"] = ["transformers", "Qwen3VLForConditionalGeneration"]
    (root / "model_index.json").write_text(json.dumps(index))
    te = root / "text_encoder"
    (te / "config.json").write_text(json.dumps(MINIMAX_H3_TEXT_ENCODER.expected_config))
    (te / "generation_config.json").write_text("{}")
    save_file({}, te / "model.safetensors")
    (te / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"unused": "model.safetensors"}})
    )
    ModelRegistry.resolve_model_cls("MiniMaxH3Qwen3VLEncoder")
    args = variant(h3_args, weight_cache_components=["dit", "text_encoder"])
    args.weight_cache_allow_weak_checkpoint_identity = True
    return args


def test_h3_two_component_preparation_is_pure_frozen_and_complete(h3_both):
    with (
        patch.object(torch.nn.Module, "__init__", side_effect=AssertionError("module")),
        patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA")),
        patch.object(
            text_encoder_loader,
            "get_encoder_data_parallel_group",
            side_effect=AssertionError("rank"),
        ),
        patch.object(
            text_encoder_loader,
            "get_local_torch_device",
            side_effect=AssertionError("rank"),
        ),
    ):
        prepared = prepare_pipeline(MiniMaxH3Pipeline, h3_both, required=True)
    assert prepared.cached_component_names == ("transformer", "text_encoder")
    component = prepared.component("text_encoder")
    assert component.contract is MINIMAX_H3_TEXT_ENCODER
    fingerprint = component.fingerprint_fields()
    json.dumps(fingerprint)  # No config objects, tensors or process groups on the wire.
    assert fingerprint["recipe"]["selected_lm_layer"] == 50
    _, files = identity.consumed_files(prepared)
    assert set(files) >= {
        "text_encoder/config.json",
        "text_encoder/generation_config.json",
        "text_encoder/model.safetensors",
        "text_encoder/model.safetensors.index.json",
        "transformer/config.json",
        "model_index.json",
    }
    args = copy.deepcopy(h3_both)
    prepared.apply_config(args)
    assert args.pipeline_config.text_encoder_configs[0].num_hidden_layers == 50
    assert h3_both.model_paths == {}
    args.pipeline_config.text_encoder_configs[0].arch_config.num_hidden_layers = 1
    assert component.fingerprint_fields() == fingerprint
    with (
        patch.object(identity, "environment_identity", return_value={}),
        patch.object(
            identity.current_platform, "get_device_uuid", return_value="test-gpu"
        ),
    ):
        plan = identity.compatibility_plan(prepared, h3_both)
        assert plan.to_dict()["requested"] == ["transformer", "text_encoder"]
        Path(prepared.model_path, "text_encoder/model.safetensors").write_bytes(
            b"changed"
        )
        assert identity.compatibility_plan(prepared, h3_both) != plan


def test_default_h3_still_caches_only_dit(h3_both):
    args = variant(
        h3_both,
        weight_cache_components=["dit"],
        component_residency={"text_encoder": "layerwise-offload"},
    )
    prepared = prepare_pipeline(MiniMaxH3Pipeline, args, required=True)
    assert prepared.cached_component_names == ("transformer",)


def test_other_pipeline_cannot_inherit_h3_text_encoder_support(prepared_wan):
    args, pipeline, prepare = prepared_wan
    args.weight_cache_components = ["dit", "text_encoder"]
    with pytest.raises(ValueError, match="audited binding"):
        prepare(pipeline, args, required=True)


def test_encoder_metadata_list_is_frozen(h3_both):
    prepared = prepare_pipeline(MiniMaxH3Pipeline, h3_both, required=True)
    component = prepared.component("text_encoder")
    files = component.consumed_files()
    Path(prepared.model_path, "text_encoder/generation_config.json").unlink()
    Path(prepared.model_path, "text_encoder/model.safetensors.index.json").unlink()
    assert component.consumed_files() == files
    with pytest.raises(FileNotFoundError):
        identity.checkpoint_identity(prepared, h3_both)


@pytest.mark.parametrize("explicit", [False, True])
def test_meta_encoder_uses_ordinary_attention_fallback_policy(h3_both, explicit):
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.layers.attention import selector
    from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
    from sglang.multimodal_gen.test.unit.test_text_encoder_load_recipe import (
        TinyEncoder,
    )

    if explicit:
        h3_both = variant(h3_both, component_attention_backends={"text_encoder": "fa"})
    component = prepare_pipeline(MiniMaxH3Pipeline, h3_both, required=True).component(
        "text_encoder"
    )

    def construct(*args, **kwargs):
        context = selector.get_component_attn_backend_context()
        assert context.allow_global_backend_fallback
        assert context.require_backend_selection is explicit
        # H3 language layers select FA, while the head_size=72 vision tower
        # selects SDPA. Only a component-specific override requires strict FA.
        selector._record_component_attn_backend("fa", None)
        selector._record_component_attn_backend("torch_sdpa", None)
        with torch.device("meta"):
            return TinyEncoder(SimpleNamespace(width=3))

    with (
        patch.object(text_encoder_loader, "get_folding_tp_group", return_value=None),
        patch.object(
            text_encoder_loader, "use_tensor_parallel_group", return_value=nullcontext()
        ),
        patch.object(text_encoder_loader, "initialize_model", side_effect=construct),
    ):
        if explicit:
            with pytest.raises(
                selector.ComponentAttentionBackendNotAppliedError, match="torch_sdpa"
            ):
                component.loader().build_prepared_meta(
                    component.recipe, attention_backend=AttentionBackendEnum.FA
                )
        else:
            model = component.loader().build_prepared_meta(
                component.recipe, attention_backend=AttentionBackendEnum.FA
            )
            assert model.weight.is_meta and not model.training


@pytest.mark.parametrize(
    "selectors", [["text_encoder", "dit"], ["transformer", "text_encoder"]]
)
def test_selector_alias_order_is_canonical(h3_both, selectors):
    args = variant(h3_both, weight_cache_components=selectors)
    assert requested_component_names(args) == ("transformer", "text_encoder")


@pytest.mark.parametrize(
    "selectors",
    [
        [],
        ["text_encoder"],
        ["dit", "transformer"],
        ["dit", "dit"],
        ["dit", "vae"],
        ["dit", "text_encoder_2"],
    ],
)
def test_invalid_component_selection_fails_before_loading(h3_both, selectors):
    with pytest.raises(ValueError):
        variant(h3_both, weight_cache_components=selectors)


@pytest.mark.parametrize(
    "overrides",
    [
        {"text_encoder_cpu_offload": True},
        {"cpu_offload_components": ["text_encoder"]},
        {"layerwise_offload_components": ["text_encoder"]},
        {"component_residency": {"text_encoder": "component-offload"}},
        {"component_residency": {"text_encoder": "layerwise-offload"}},
    ],
)
def test_cached_encoder_rejects_explicit_offload(h3_both, overrides):
    with pytest.raises(ValueError, match="offload|resident"):
        variant(h3_both, **overrides)


@pytest.mark.parametrize(
    "variant",
    [
        "raw",
        "class",
        "layers",
        "quant",
        "dtype",
        "projection",
        "runtime",
        "attention",
        "folding",
    ],
)
def test_h3_encoder_contract_rejects_unreviewed_state(h3_both, variant):
    component = prepare_pipeline(MiniMaxH3Pipeline, h3_both, required=True).component(
        "text_encoder"
    )
    recipe = component.recipe.thaw()
    attention = "fa"
    if variant == "raw":
        recipe.hf_config["extension"] = True
    elif variant == "class":
        recipe.model_cls = torch.nn.Linear
    elif variant == "layers":
        recipe.config.arch_config.num_hidden_layers = 49
    elif variant == "quant":
        recipe.config.quant_config = "fp8"
    elif variant == "dtype":
        recipe.dtype = "fp16"
    elif variant == "projection":
        recipe.config.arch_config.conditioning_projection_path = "/unreviewed"
    elif variant == "runtime":
        recipe.server_args.enable_torch_compile = True
    elif variant == "attention":
        attention = "torch_sdpa"
    elif variant == "folding":
        recipe.config.parallel_folding_mode = "world"
    with pytest.raises(ValueError):
        component.contract.validate_supported(recipe.freeze(), attention=attention)
