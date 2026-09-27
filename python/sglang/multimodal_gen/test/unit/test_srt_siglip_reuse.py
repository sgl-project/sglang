from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from transformers import Siglip2VisionConfig

from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.models.encoders import gemma_3
from sglang.multimodal_gen.runtime.models.encoders.base import (
    EncoderTensorParallelMixin,
)
from sglang.multimodal_gen.runtime.models.encoders.hunyuan_image3 import (
    HunyuanImage3VisionModel,
    _PerImageSDPA,
)
from sglang.srt.models import siglip, siglip2


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

    with (
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
    assert isinstance(model, EncoderTensorParallelMixin)
    assert not hasattr(model, "_vision_tensor_parallel_group")


def test_gemma3_maps_hf_siglip_projection_name():
    map_name = get_param_names_mapping(
        gemma_3.Gemma3ForConditionalGeneration.param_names_mapping
    )

    mapped, _, _ = map_name("vision_tower.encoder.layers.0.self_attn.out_proj.weight")

    assert mapped == "vision_tower.vision_model.encoder.layers.0.self_attn.proj.weight"


@pytest.mark.parametrize("replicated", [False, True])
def test_siglip2_propagates_backend_and_parallelism(replicated):
    config = Siglip2VisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_hidden_layers=1,
        num_patches=16,
    )
    with (
        patch.object(siglip2, "VisionAttention", return_value=nn.Identity()) as attn,
        patch.object(
            siglip2, "ColumnParallelLinear", return_value=nn.Identity()
        ) as fc1,
        patch.object(siglip2, "RowParallelLinear", return_value=nn.Identity()) as fc2,
    ):
        kwargs = (
            {"qkv_backend": "sdpa", "use_data_parallel": True} if replicated else {}
        )
        siglip2.Siglip2Model(config, **kwargs)

    assert attn.call_args.kwargs["qkv_backend"] == ("sdpa" if replicated else None)
    assert attn.call_args.kwargs["use_data_parallel"] is replicated
    for linear in (fc1, fc2):
        assert linear.call_args.kwargs["tp_size"] == (1 if replicated else None)
        assert linear.call_args.kwargs["tp_rank"] == (0 if replicated else None)


@pytest.mark.parametrize("mask_ndim", [1, 2, 3])
def test_hunyuan_siglip2_preserves_padded_layout(mask_ndim):
    # Exercise the processor boundary independently of device-specific kernels.
    model = HunyuanImage3VisionModel.__new__(HunyuanImage3VisionModel)
    nn.Module.__init__(model)
    batch = 1 if mask_ndim == 1 else 2
    pixels = torch.arange(batch * 8 * 3, dtype=torch.float32).reshape(batch, 8, 3)
    shapes = torch.tensor([[2, 3], [1, 4]])[:batch]
    mask = torch.arange(8)[None, :] < shapes.prod(dim=1)[:, None]
    input_mask = mask.reshape(-1) if mask_ndim == 1 else mask
    if mask_ndim == 3:
        input_mask = input_mask[:, None, :]
    packed = pixels[mask]

    with patch.object(
        siglip2.Siglip2Model, "forward", return_value=packed[None]
    ) as forward:
        output = model(pixels, input_mask, shapes)

    torch.testing.assert_close(forward.call_args.kwargs["pixel_values_packed"], packed)
    assert forward.call_args.kwargs["spatial_shapes"].device.type == "cpu"
    assert forward.call_args.kwargs["cu_seqlens"].tolist() == (
        [0, 6] if batch == 1 else [0, 6, 10]
    )
    assert forward.call_args.kwargs["max_seqlen"] == 6
    torch.testing.assert_close(output[mask], packed)
    assert torch.count_nonzero(output[~mask]) == 0


def test_hunyuan_siglip2_attention_does_not_mix_images():
    generator = torch.Generator().manual_seed(42)
    q, k, v = [torch.randn(10, 2, 8, generator=generator) for _ in range(3)]
    attention = _PerImageSDPA()
    boundaries = torch.tensor([0, 6, 10], dtype=torch.int32)
    actual = attention(q, k, v, boundaries)
    expected = torch.cat(
        [
            torch.nn.functional.scaled_dot_product_attention(
                q[start:end].transpose(0, 1),
                k[start:end].transpose(0, 1),
                v[start:end].transpose(0, 1),
            ).transpose(0, 1)
            for start, end in ((0, 6), (6, 10))
        ]
    )
    torch.testing.assert_close(actual, expected)
    v[6:] += 100
    torch.testing.assert_close(attention(q, k, v, boundaries)[:6], actual[:6])


def test_hunyuan_siglip2_loads_unfused_checkpoint_weights():
    model = HunyuanImage3VisionModel(
        dict(
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_hidden_layers=1,
            num_patches=16,
            patch_size=2,
            use_return_dict=True,
        )
    )
    checkpoint = {}
    expected = {}
    for name, param in model.named_parameters():
        value = torch.arange(param.numel(), dtype=param.dtype).reshape(param.shape)
        expected[name] = value
        checkpoint_name = name.removeprefix("vision_model.")
        if "attn.qkv_proj" in name:
            for projection, weight in zip(("q", "k", "v"), value.chunk(3)):
                checkpoint[
                    checkpoint_name.replace("attn.qkv_proj", f"{projection}_proj")
                ] = weight
        else:
            checkpoint[checkpoint_name.replace("attn.proj", "out_proj")] = value

    loaded = model.load_weights(checkpoint.items())

    assert loaded == set(expected)
    for name, param in model.named_parameters():
        torch.testing.assert_close(param, expected[name], rtol=0, atol=0)
