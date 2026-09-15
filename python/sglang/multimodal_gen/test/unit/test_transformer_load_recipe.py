# SPDX-License-Identifier: Apache-2.0
"""Preparation decisions survive discovery/argument changes before loading."""

import unittest
from contextlib import nullcontext
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from sglang.multimodal_gen.runtime.loader import fsdp_load
from sglang.multimodal_gen.runtime.loader.component_loaders import transformer_loader
from sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader import (
    ResolvedTransformerLoad,
    TransformerLoader,
)
from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (
    TransformerQuantLoadSpec,
)
from sglang.multimodal_gen.runtime.loader.weight_load_plan import WeightLoadPlan
from sglang.multimodal_gen.utils import get_mixed_precision_state


def _resolved():
    return ResolvedTransformerLoad(
        model_cls=nn.Linear,
        init_params={"in_features": 3, "out_features": 4},
        weight_files=("/frozen/checkpoint.safetensors",),
        server_args=SimpleNamespace(tp_size=1),
        component_name="transformer",
        component_starts_on_cpu=False,
        quant_spec=TransformerQuantLoadSpec([], None, None, torch.bfloat16),
        weight_load_plan=WeightLoadPlan(torch.device("cpu")),
        checkpoint_key_filter=None,
        quantized_attn_backend=None,
    )


class TestTransformerLoadRecipe(unittest.TestCase):
    def test_prepared_load_uses_shared_finalization_without_resolving_again(self):
        recipe = _resolved()
        args = Mock()
        args.model_paths = {"transformer": "/frozen"}
        args.should_use_fsdp_for_component.return_value = False
        args.requested_component_attention_backend.return_value = None
        recipe.server_args = args
        frozen = Mock()
        frozen.thaw.return_value = recipe
        loader = TransformerLoader()
        model = nn.Linear(3, 4, dtype=torch.bfloat16)
        with (
            patch.object(
                loader,
                "component_load_precision",
                side_effect=AssertionError("resolve again"),
            ),
            patch.object(
                loader,
                "_load_customized_with_context",
                side_effect=AssertionError("ordinary resolver"),
            ),
            patch.object(
                loader,
                "component_attention_backend_context",
                return_value=nullcontext(),
            ),
            patch.object(loader, "target_device", return_value=torch.device("cpu")),
            patch.object(
                loader, "materialize_customized", return_value=model
            ) as materialize,
            patch(
                "sglang.multimodal_gen.runtime.loader.component_loaders.component_loader.current_platform.get_available_gpu_memory",
                side_effect=[16.0, 15.5],
            ),
        ):
            loaded, memory = loader.load_prepared(frozen, attention_backend=object())
        self.assertIs(loaded, model)
        self.assertFalse(model.training)
        self.assertFalse(model.weight.requires_grad)
        self.assertEqual(memory, 0.5)
        materialize.assert_called_once_with(recipe)

    def test_frozen_recipe_has_no_shared_mutable_resolution_state(self):
        resolved = _resolved()
        frozen = resolved.freeze()
        resolved.init_params["out_features"] = 999
        resolved.server_args.tp_size = 8
        first = frozen.thaw()
        self.assertEqual(first.init_params["out_features"], 4)
        self.assertEqual(first.server_args.tp_size, 1)
        first.init_params["out_features"] = 888
        self.assertEqual(frozen.thaw().init_params["out_features"], 4)
        with self.assertRaises(FrozenInstanceError):
            frozen._payload = b"mutated"

    def test_materialize_uses_frozen_class_config_and_files(self):
        frozen = _resolved().freeze()
        loader = TransformerLoader()
        model = nn.Linear(3, 4, dtype=torch.bfloat16)
        with (
            patch.object(
                transformer_loader,
                "get_diffusers_component_config",
                side_effect=AssertionError("config rediscovery"),
            ),
            patch.object(
                transformer_loader,
                "resolve_transformer_checkpoint_files",
                side_effect=AssertionError("file rediscovery"),
            ),
            patch.object(loader, "load_state_dict_model", return_value=model) as load,
        ):
            self.assertIs(loader.materialize_customized(frozen), model)
        self.assertIs(load.call_args.kwargs["model_cls"], nn.Linear)
        self.assertEqual(
            load.call_args.kwargs["init_params"], {"in_features": 3, "out_features": 4}
        )
        self.assertEqual(
            load.call_args.kwargs["weight_files"], ["/frozen/checkpoint.safetensors"]
        )

    def test_meta_construction_installs_same_policy_without_weight_reads(self):
        class InspectPolicy(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.empty(2))
                self.constructed_dtype = get_mixed_precision_state().param_dtype

        with patch.object(
            fsdp_load,
            "safetensors_weights_iterator",
            side_effect=AssertionError("weight read"),
        ):
            model, policy = fsdp_load.initialize_model_for_inference(
                InspectPolicy,
                {},
                param_dtype=torch.bfloat16,
            )
        self.assertTrue(model.weight.is_meta)
        self.assertEqual(model.weight.dtype, torch.bfloat16)
        self.assertEqual(model.constructed_dtype, policy.param_dtype)

    def test_unquantized_backend_planning_does_not_probe_cuda(self):
        with patch.object(
            transformer_loader.current_platform,
            "is_blackwell",
            side_effect=AssertionError("CUDA probe"),
        ):
            self.assertIsNone(
                transformer_loader._default_quantized_attention_backend(
                    _resolved().quant_spec, SimpleNamespace()
                )
            )


if __name__ == "__main__":
    unittest.main()
