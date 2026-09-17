# SPDX-License-Identifier: Apache-2.0
"""Component capabilities are loader-owned, frozen, and separately admitted."""

from unittest.mock import patch

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
            prepared.transformer.thaw().server_args,
            planned_device=torch.device("cuda", 0),
        )
    assert component.contract is contract
    assert component.loader_cls is TransformerLoader
    assert component.name == "transformer"
    assert (
        component.recipe.thaw().weight_files == prepared.transformer.thaw().weight_files
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


def test_state_contract_rejects_same_named_unregistered_class():
    fake = type(WAN.model_name, (), {"__module__": WAN.model_module})
    with pytest.raises(ValueError, match="resolved model"):
        for_model(fake)
