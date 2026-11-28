"""CPU regressions for fused weight halves split across chunked weight updates."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.models.bailing_moe_linear import BailingMoeV2_5ForCausalLM
from sglang.srt.models.bailing_moe_nextn import BailingMoeForCausalLMNextN
from sglang.srt.models.bailing_moe_v3 import BailingMoeV3ForCausalLM
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
from sglang.srt.models.glm5_next_nextn import Glm5NextForConditionalGenerationNextN
from sglang.srt.models.kimi_k25_eagle3 import Eagle3DeepseekV2ForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _register_param(root: nn.Module, full_name: str, param: nn.Parameter) -> None:
    parts = full_name.split(".")
    curr = root
    for part in parts[:-1]:
        child = curr._modules.get(part)
        if child is None:
            child = nn.Module()
            curr.add_module(part, child)
        curr = child
    curr.register_parameter(parts[-1], param)


def _updater(model: nn.Module) -> WeightUpdater:
    return WeightUpdater(
        tp_rank=0,
        device="cpu",
        gpu_id=0,
        model_config=SimpleNamespace(dtype=torch.float32),
        custom_weight_loaders={},
        get_model=lambda: model,
        update_model_fields=MagicMock(),
        recapture_cuda_graph=None,
        get_model_runner=lambda: None,
    )


def _build_qkv_a_fixture(kind: str, *, quantized: bool = False, suffix: str = "weight"):
    quant = SimpleNamespace(get_name=lambda: "awq") if quantized else None
    param = nn.Parameter(torch.zeros((3, 3), dtype=torch.float32), requires_grad=False)

    if kind == "glm5_next":
        model = Glm5NextForConditionalGeneration.__new__(
            Glm5NextForConditionalGeneration
        )
        nn.Module.__init__(model)
        model.config = SimpleNamespace(
            n_routed_experts=0, q_lora_rank=2, num_hidden_layers=2
        )
        model.quant_config = quant
        model.num_fused_shared_experts = 0
        model.fuse_qkv_a_proj = True
        model.encoder_only = False
        model.language_only = False
        model.model = nn.Module()
        model.model.layers = nn.ModuleList()
        source = destination = "model.layers.0.self_attn"
        missing_source = "model.layers.1.self_attn"
    elif kind == "glm5_next_nextn":
        model = Glm5NextForConditionalGenerationNextN.__new__(
            Glm5NextForConditionalGenerationNextN
        )
        nn.Module.__init__(model)
        model.config = SimpleNamespace(
            n_routed_experts=0,
            q_lora_rank=2,
            num_hidden_layers=2,
            num_nextn_predict_layers=1,
        )
        model.quant_config = quant
        model.num_fused_shared_experts = 0
        model.fuse_qkv_a_proj = True
        model.model = nn.Module()
        model.model.decoder = nn.Module()
        source = "model.language_model.layers.2.self_attn"
        destination = "model.decoder.self_attn"
        missing_source = "model.language_model.layers.1.self_attn"
    elif kind == "bailing_moe_v3":
        model = BailingMoeV3ForCausalLM.__new__(BailingMoeV3ForCausalLM)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(
            num_experts=0,
            q_lora_rank=2,
            num_hidden_layers=2,
            tie_word_embeddings=False,
        )
        model.quant_config = quant
        model.num_fused_shared_experts = 0
        model.post_load_weights = lambda **kwargs: None
        source = destination = "model.layers.0.attention"
        missing_source = "model.layers.1.attention"
    elif kind == "bailing_moe_linear":
        model = BailingMoeV2_5ForCausalLM.__new__(BailingMoeV2_5ForCausalLM)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(
            num_experts=0,
            q_lora_rank=2,
            num_hidden_layers=2,
            tie_word_embeddings=False,
        )
        model.quant_config = quant
        model.post_load_weights = lambda **kwargs: None
        source = destination = "model.layers.0.attention"
        missing_source = "model.layers.1.attention"
    elif kind in ("bailing_nextn_v3", "bailing_nextn_v25"):
        model = BailingMoeForCausalLMNextN.__new__(BailingMoeForCausalLMNextN)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(
            num_experts=0,
            q_lora_rank=2,
            num_hidden_layers=2,
            num_nextn_predict_layers=1,
            tie_word_embeddings=False,
        )
        model.quant_config = quant
        model.num_fused_shared_experts = 0
        model.base_load_weights_func = (
            BailingMoeV3ForCausalLM.load_weights
            if kind == "bailing_nextn_v3"
            else BailingMoeV2_5ForCausalLM.load_weights
        )
        model.post_load_weights_func = None
        source = "model.layers.2.attention"
        destination = "model.decoder.attention"
        missing_source = "model.layers.1.attention"
    elif kind == "kimi_k25_eagle3":
        model = Eagle3DeepseekV2ForCausalLM.__new__(Eagle3DeepseekV2ForCausalLM)
        nn.Module.__init__(model)
        model.post_load_weights = lambda: None
        source = "layers.0.self_attn"
        destination = "model.midlayer.self_attn"
        missing_source = "layers.1.self_attn"
    else:
        raise ValueError(f"unknown fixture kind: {kind}")

    _register_param(model, f"{destination}.fused_qkv_a_proj_with_mqa.{suffix}", param)
    if kind in (
        "bailing_moe_v3",
        "bailing_moe_linear",
        "bailing_nextn_v3",
        "bailing_nextn_v25",
    ):
        model.model.layer_group_size = 1
    return model, param, source, destination, missing_source


_PAIR_FIXTURES = (
    "glm5_next",
    "glm5_next_nextn",
    "bailing_moe_v3",
    "bailing_moe_linear",
    "bailing_nextn_v3",
    "bailing_nextn_v25",
    "kimi_k25_eagle3",
)


class TestChunkedWeightHalves(CustomTestCase):
    def test_qkv_a_halves_pair_across_calls_with_buffer_reuse(self):
        """All covered QKV-A loaders must pair split halves using original values."""
        for kind in _PAIR_FIXTURES:
            quant_modes = (False,) if kind == "kimi_k25_eagle3" else (False, True)
            for quantized in quant_modes:
                for reverse in (False, True):
                    with self.subTest(kind=kind, quantized=quantized, reverse=reverse):
                        suffix = "qweight" if quantized else "weight"
                        model, target, source, _, _ = _build_qkv_a_fixture(
                            kind, quantized=quantized, suffix=suffix
                        )
                        axis = 1 if quantized else 0
                        q = torch.ones((3, 2) if quantized else (2, 3))
                        kv = torch.full((3, 1) if quantized else (1, 3), 2.0)
                        expected = torch.cat((q, kv), dim=axis)
                        halves = [
                            (f"{source}.q_a_proj.{suffix}", q),
                            (f"{source}.kv_a_proj_with_mqa.{suffix}", kv),
                        ]
                        if reverse:
                            halves.reverse()

                        with patch.object(
                            DeepseekV2WeightLoaderMixin, "post_load_weights"
                        ):
                            model.load_weights([halves[0]])
                            torch.testing.assert_close(target, torch.zeros(3, 3))
                            # Simulate sender reusing its bucket buffer between chunks.
                            halves[0][1].fill_(9.0)
                            model.load_weights([halves[1]])

                        torch.testing.assert_close(target, expected)
                        with patch.object(DefaultModelLoader, "postprocess_weights"):
                            self.assertIsNone(
                                _updater(model).end_weight_update(run_post_load=False)
                            )

    def test_already_fused_qkv_a_weight_loads_directly_without_pending(self):
        """Incoming fused_qkv_a_proj_with_mqa tensors must load directly and not be intercepted."""
        for kind in _PAIR_FIXTURES:
            with self.subTest(kind=kind):
                model, target, source, _, _ = _build_qkv_a_fixture(kind)
                fused = torch.full((3, 3), 7.0)

                with patch.object(DeepseekV2WeightLoaderMixin, "post_load_weights"):
                    model.load_weights(
                        [(f"{source}.fused_qkv_a_proj_with_mqa.weight", fused)]
                    )

                torch.testing.assert_close(target, fused)
                self.assertEqual(model.discard_pending_weight_halves(), [])

    def test_glm5_next_quark_scale_pairs_across_calls(self):
        """Quark weight_scale halves map to weight_scale_inv across chunked calls."""
        for kind in ("glm5_next", "glm5_next_nextn"):
            with self.subTest(kind=kind):
                model, target, source, _, _ = _build_qkv_a_fixture(
                    kind, suffix="weight_scale_inv"
                )
                q_scale = torch.ones((2, 3))
                kv_scale = torch.full((1, 3), 3.0)
                expected = torch.cat((q_scale, kv_scale), dim=0)

                with patch.object(DeepseekV2WeightLoaderMixin, "post_load_weights"):
                    model.load_weights([(f"{source}.q_a_proj.weight_scale", q_scale)])
                    torch.testing.assert_close(target, torch.zeros(3, 3))
                    q_scale.fill_(9.0)
                    model.load_weights(
                        [(f"{source}.kv_a_proj_with_mqa.weight_scale", kv_scale)]
                    )

                torch.testing.assert_close(target, expected)
                self.assertEqual(model.discard_pending_weight_halves(), [])

    def test_begin_and_end_weight_update_lifecycle_for_qkv_a_loaders(self):
        """Session boundaries must drop stale halves and fail on unpaired halves."""
        for kind in _PAIR_FIXTURES:
            with self.subTest(kind=kind):
                model, target, source, destination, missing_source = (
                    _build_qkv_a_fixture(kind)
                )
                q_name = f"{source}.q_a_proj.weight"
                kv_name = f"{source}.kv_a_proj_with_mqa.weight"

                # A half for a layer not owned by this rank must not pollute pending.
                with patch.object(DeepseekV2WeightLoaderMixin, "post_load_weights"):
                    model.load_weights(
                        [(f"{missing_source}.q_a_proj.weight", torch.ones((2, 3)))]
                    )
                self.assertEqual(model.discard_pending_weight_halves(), [])

                # A stale half from an earlier load must be dropped on begin_weight_update.
                with patch.object(DeepseekV2WeightLoaderMixin, "post_load_weights"):
                    model.load_weights([(q_name, torch.ones((2, 3)))])
                with patch.object(DefaultModelLoader, "restore_weights_before_loading"):
                    _updater(model).begin_weight_update()

                # Exercise WeightUpdater submodule discovery via model.modules().
                wrapper = nn.Module()
                wrapper.add_module("language_model", model)
                updater = _updater(wrapper)

                with patch.object(DeepseekV2WeightLoaderMixin, "post_load_weights"):
                    model.load_weights([(kv_name, torch.full((1, 3), 2.0))])
                torch.testing.assert_close(target, torch.zeros(3, 3))

                with patch.object(DefaultModelLoader, "postprocess_weights"):
                    error = updater.end_weight_update(run_post_load=False)
                    self.assertIsNotNone(error)
                    self.assertIn(f"{destination}.kv_a_proj_with_mqa.weight", error)
                    self.assertIsNone(updater.end_weight_update(run_post_load=False))


if __name__ == "__main__":
    unittest.main()
