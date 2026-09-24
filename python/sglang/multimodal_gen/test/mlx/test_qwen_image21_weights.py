# SPDX-License-Identifier: Apache-2.0

import sys

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
mlx_utils = pytest.importorskip("mlx.utils")

from sglang.multimodal_gen.runtime.models.dits.qwen_image21_mlx import (
    QwenImage21Transformer,
    build_layout,
    load_transformer,
)
from sglang.multimodal_gen.runtime.models.encoders.qwen3vl_mlx import (
    Qwen3VLTextEncoder,
    load_encoders,
)
from sglang.multimodal_gen.runtime.models.encoders.qwen3vl_vision_mlx import (
    Qwen3VLVisionEncoder,
)

QUANTIZATION = dict(method="affine", bits=4, group_size=64)


def quantize(model, embeddings=False):
    model.set_dtype(mx.bfloat16)
    nn.quantize(
        model,
        bits=4,
        group_size=64,
        class_predicate=lambda p, m: (
            (isinstance(m, nn.Linear) or embeddings and isinstance(m, nn.Embedding))
            and m.weight.shape[1] % 64 == 0
        ),
    )
    return dict(mlx_utils.tree_flatten(model.parameters()))


def test_transformer_loader_preserves_fused_gate_up(tmp_path):
    mx.random.seed(91)
    config = dict(
        num_layers=2, attention_head_dim=16, num_attention_heads=4, context_in_dim=64
    )
    model = QwenImage21Transformer(**config)
    weights = quantize(model)
    converted = {}
    for name, value in weights.items():
        if ".img_mlp.gate_layer." in name:
            converted[name.replace(".gate_layer.", ".gate_up.")] = mx.concatenate(
                (value, weights[name.replace(".gate_layer.", ".proj.")]), axis=0
            )
        elif ".img_mlp.proj." not in name:
            converted[name] = value
    path = tmp_path / "transformer.safetensors"
    mx.save_safetensors(str(path), converted)
    loaded = load_transformer(str(path), config, QUANTIZATION)
    layout = build_layout([False] * 9, [(1, 2, 4)], (4, 6, 6))
    embeds = mx.random.normal((1, 9, 64)).astype(mx.bfloat16)
    latents = mx.random.normal((1, 8, 64)).astype(mx.bfloat16)
    expected = model(
        latents,
        mx.array([721.4]),
        layout.target_rope,
        model.prepare_conditioning(embeds, layout),
    )
    actual = loaded(
        latents,
        mx.array([721.4]),
        layout.target_rope,
        loaded.prepare_conditioning(embeds, layout),
    )
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("with_images", [False, True])
@pytest.mark.parametrize("quantized_embeddings", [False, True])
def test_encoder_loader_preserves_quantized_and_dense_layers(
    tmp_path, with_images, quantized_embeddings
):
    mx.random.seed(17)
    text_config = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    vision_config = dict(
        hidden_size=128 if quantized_embeddings else 144,
        intermediate_size=272,
        depth=3,
        num_heads=2,
        out_hidden_size=64,
        deepstack_visual_indexes=[0, 1, 2],
    )
    text = Qwen3VLTextEncoder(**text_config, mrope_section=(3, 3, 2))
    vision = Qwen3VLVisionEncoder(**vision_config)
    weights = {f"model.{k}": v for k, v in quantize(text, quantized_embeddings).items()}
    weights.update(
        {
            f"model.visual.{k}": v
            for k, v in quantize(vision, quantized_embeddings).items()
        }
    )
    name = "model.visual.patch_embed.proj.weight"
    weights[name] = weights[name].transpose(0, 4, 1, 2, 3)
    weights["lm_head.weight"] = mx.zeros((128, 64), dtype=mx.bfloat16)
    path = tmp_path / "encoder.safetensors"
    mx.save_safetensors(str(path), weights)
    text_config["rope_scaling"] = dict(mrope_section=[3, 3, 2])
    actual_text, actual_vision = load_encoders(
        str(path),
        dict(text_config=text_config, vision_config=vision_config),
        QUANTIZATION,
        with_images,
    )
    ids = mx.array([[3, 9, 5, 8]])
    assert mx.array_equal(actual_text(ids), text(ids)).item()
    assert isinstance(actual_text.layers[0].self_attn.q_proj, nn.QuantizedLinear)
    if with_images:
        assert isinstance(actual_vision.blocks[0].mlp.linear_fc2, nn.Linear)
        pixels = mx.random.normal((16, 1536)).astype(mx.bfloat16)
        expected, expected_deepstack = vision(pixels, [(1, 4, 4)])
        actual, actual_deepstack = actual_vision(pixels, [(1, 4, 4)])
        assert mx.array_equal(actual, expected).item()
        assert all(
            mx.array_equal(a, b).item()
            for a, b in zip(actual_deepstack, expected_deepstack)
        )
        if quantized_embeddings:
            packed = actual_vision.pos_embed
            assert isinstance(packed, nn.QuantizedEmbedding)
            dense_weight = mx.dequantize(
                packed.weight, packed.scales, packed.biases, group_size=64, bits=4
            )
            dense = nn.Embedding(*dense_weight.shape)
            dense.weight = dense_weight
            actual_vision.pos_embed = dense
            dense_output, dense_deepstack = actual_vision(pixels, [(1, 4, 4)])
            assert mx.array_equal(actual, dense_output).item()
            assert all(
                mx.array_equal(a, b).item()
                for a, b in zip(actual_deepstack, dense_deepstack)
            )
    else:
        assert actual_vision is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
