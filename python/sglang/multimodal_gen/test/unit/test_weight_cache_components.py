# SPDX-License-Identifier: Apache-2.0
"""Component capabilities are loader-owned, frozen, and separately admitted."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import pytest
import torch

from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
    GenericComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader import (
    TransformerLoader,
)
from sglang.multimodal_gen.runtime.loader.native_dit_state import (
    MINIMAX_H3,
    QWEN_IMAGE,
    WAN,
    for_model,
)
from sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline import (
    MiniMaxH3Pipeline,
)
from sglang.multimodal_gen.runtime.pipelines.qwen_image import QwenImagePipeline
from sglang.multimodal_gen.runtime.pipelines_core.prepare import prepare_pipeline
from sglang.multimodal_gen.test.unit.test_weight_cache_admission import (
    prepared_wan as prepared_wan,
)
from sglang.multimodal_gen.test.unit.test_weight_cache_minimax_h3 import (
    h3_args as h3_args,
)
from sglang.multimodal_gen.test.unit.test_weight_cache_qwen_image import (
    prepared_qwen as prepared_qwen,
)


@pytest.mark.parametrize(
    "fixture_name,contract",
    [("prepared_wan", WAN), ("prepared_qwen", QWEN_IMAGE), ("h3_args", MINIMAX_H3)],
)
def test_existing_rows_share_actual_loader_capability(request, fixture_name, contract):
    fixture = request.getfixturevalue(fixture_name)
    if fixture_name == "prepared_wan":
        args, pipeline, _ = fixture
    else:
        args = fixture
        pipeline = (
            QwenImagePipeline if fixture_name == "prepared_qwen" else MiniMaxH3Pipeline
        )
    prepared = prepare_pipeline(pipeline, args, required=True)
    spec = next(spec for spec in prepared.specs if spec.module_name == "transformer")
    loader = ComponentLoader.for_component_type(
        spec.load_module_name,
        spec.transformers_or_diffusers,
        spec.architecture,
        loader_cls=pipeline.component_loaders.get(spec.module_name),
    )
    with (
        patch.object(torch.nn.Module, "__init__", side_effect=AssertionError("module")),
        patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA")),
    ):
        component = loader.prepare_weight_cache(
            spec,
            prepared.component("transformer").recipe.thaw().server_args,
            planned_device=torch.device("cuda", 0),
        )
    assert component.contract is contract
    assert component.loader_cls is TransformerLoader
    assert component.name == "transformer"
    assert (
        component.recipe.thaw().weight_files
        == prepared.component("transformer").recipe.thaw().weight_files
    )
    fingerprint = component.fingerprint_fields()
    assert fingerprint["contract"] == contract.contract_id
    assert fingerprint["recipe"]["dtype"] == "torch.bfloat16"
    with patch.object(
        ComponentLoader, "for_component_type", side_effect=AssertionError("rediscovery")
    ):
        assert type(component.loader()) is TransformerLoader


def test_generic_loader_has_no_implicit_cache_capability():
    with pytest.raises(ValueError, match="no audited weight-cache capability"):
        GenericComponentLoader("diffusers", "Unknown").prepare_weight_cache(None, None)


def test_preparation_never_discovers_unrelated_loaders(prepared_wan):
    args, pipeline, prepare = prepared_wan
    with (
        patch.object(ComponentLoader, "_loaders_registered", False),
        patch.object(
            ComponentLoader,
            "_ensure_loaders_registered",
            side_effect=AssertionError("discovery"),
        ),
        patch.object(torch.nn.Module, "__init__", side_effect=AssertionError("module")),
        patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA")),
    ):
        assert (
            prepare(pipeline, args, required=True).component("transformer").loader_cls
            is TransformerLoader
        )
        with pytest.raises(ValueError, match="No registered component loader"):
            ComponentLoader.for_component_type(
                "unregistered_test_component", "diffusers", discover_loaders=False
            )


def test_state_contract_rejects_same_named_unregistered_class():
    fake = type(WAN.model_name, (), {"__module__": WAN.model_module})
    with pytest.raises(ValueError, match="resolved model"):
        for_model(fake)


def test_pipeline_binding_and_state_capability_are_both_required(prepared_wan):
    from sglang.multimodal_gen.runtime.weight_cache import policy

    args, pipeline, prepare = prepared_wan
    binding = policy.for_pipeline(pipeline)
    wrong = msgspec.structs.replace(
        binding,
        components=(
            msgspec.structs.replace(
                binding.components[0], contract_id=QWEN_IMAGE.contract_id
            ),
        ),
    )
    with patch.object(policy, "for_pipeline", return_value=wrong):
        with pytest.raises(ValueError, match="differs from audited pipeline binding"):
            prepare(pipeline, args, required=True)


def test_prepared_component_collection_and_identity_cover_every_input(prepared_wan):
    from sglang.multimodal_gen.runtime.weight_cache import identity

    args, pipeline, prepare = prepared_wan
    args.weight_cache_allow_weak_checkpoint_identity = True
    prepared = prepare(pipeline, args, required=True)
    first = prepared.cached_components[0]
    other_file = Path(prepared.model_path) / "test_encoder.weights"
    other_file.write_bytes(b"test-only second component")
    second = SimpleNamespace(
        name="text_encoder",
        consumed_files=lambda: (other_file, *first.consumed_files()),
        fingerprint_fields=lambda: {"contract": "test.encoder.v1"},
    )
    multi = msgspec.structs.replace(prepared, cached_components=(first, second))
    assert multi.cached_component_names == ("transformer", "text_encoder")
    assert multi.component("text_encoder") is second
    assert multi.component("vae") is None
    _, names = identity.consumed_files(multi)
    assert len(names) == len(set(names))
    assert "test_encoder.weights" in names
    with (
        patch.object(identity, "environment_identity", return_value={}),
        patch.object(
            identity.current_platform, "get_device_uuid", return_value="test-gpu"
        ),
    ):
        plan = identity.compatibility_plan(multi, args)
        fields = plan.to_dict()
        assert set(fields["components"]) == {"transformer", "text_encoder"}
        assert fields["components"]["text_encoder"]["checkpoint_files"] == sorted(
            [
                "test_encoder.weights",
                *fields["components"]["transformer"]["checkpoint_files"],
            ]
        )
        second.fingerprint_fields = lambda: {"contract": "test.encoder.v2"}
        assert identity.compatibility_plan(multi, args) != plan
    with pytest.raises(ValueError, match="distinct"):
        msgspec.structs.replace(prepared, cached_components=(first, first))
    with pytest.raises(ValueError, match="absent"):
        msgspec.structs.replace(
            prepared,
            cached_components=(msgspec.structs.replace(first, name="missing"),),
        )
