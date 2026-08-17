from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from torch import nn

from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.models.encoders import gemma_3
from sglang.srt.models import siglip


def _vision_config():
    return SimpleNamespace(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_hidden_layers=1,
        layer_norm_eps=1e-6,
    )


def test_siglip_encoder_propagates_attention_backend():
    with (
        patch.object(siglip, "VisionAttention", return_value=nn.Identity()) as attn,
        patch.object(siglip, "SiglipMLP", return_value=nn.Identity()),
    ):
        siglip.SiglipEncoder(
            _vision_config(),
            qkv_backend="sdpa",
        )

    assert attn.call_count == 1
    assert attn.call_args.kwargs["qkv_backend"] == "sdpa"


def test_gemma3_uses_srt_siglip_with_stable_backend():
    config = SimpleNamespace(vision_config=object(), text_config=object())
    folding_group = object()

    with (
        patch.object(gemma_3, "get_tp_group", return_value=folding_group),
        patch.object(
            gemma_3,
            "SiglipVisionModel",
            return_value=nn.Identity(),
        ) as vision_model,
        patch.object(
            gemma_3,
            "Gemma3MultiModalProjector",
            return_value=nn.Identity(),
        ),
        patch.object(
            gemma_3,
            "Gemma3TextModel",
            return_value=nn.Identity(),
        ),
    ):
        model = gemma_3.Gemma3ForConditionalGeneration(config)

    vision_model.assert_called_once_with(
        config=config.vision_config,
        qkv_backend="sdpa",
        quant_config=None,
        prefix="vision_tower",
    )
    assert model._vision_tensor_parallel_group is folding_group


def test_gemma3_restores_vision_tensor_parallel_group():
    model = gemma_3.Gemma3ForConditionalGeneration.__new__(
        gemma_3.Gemma3ForConditionalGeneration
    )
    nn.Module.__init__(model)
    folding_group = object()
    active_group = object()
    model._vision_tensor_parallel_group = folding_group
    events = []

    @contextmanager
    def use_group(group):
        events.append(("enter", group))
        yield
        events.append(("exit", group))

    with (
        patch.object(gemma_3, "get_tp_group", return_value=active_group),
        patch.object(
            gemma_3,
            "patch_tensor_parallel_group",
            side_effect=use_group,
        ) as patch_group,
    ):
        with model._vision_parallel_context():
            events.append(("forward", folding_group))

    patch_group.assert_called_once_with(folding_group)
    assert events == [
        ("enter", folding_group),
        ("forward", folding_group),
        ("exit", folding_group),
    ]


def test_gemma3_maps_hf_siglip_projection_name():
    map_name = get_param_names_mapping(
        gemma_3.Gemma3ForConditionalGeneration.param_names_mapping
    )

    mapped, _, _ = map_name("vision_tower.encoder.layers.0.self_attn.out_proj.weight")

    assert mapped == "vision_tower.vision_model.encoder.layers.0.self_attn.proj.weight"
