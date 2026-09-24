# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

# the native tensor core has no torch or serving dependency
_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "runtime/hardware_backend/mlx/qwen_image21.py"
)
_SPEC = importlib.util.spec_from_file_location("qwen21_mlx_test_model", _MODEL_PATH)
model_module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = model_module
_SPEC.loader.exec_module(model_module)


def joint_reference(model, latents, embeddings, timestep, layout, condition):
    prefix = model.txt_in(embeddings)[:, layout.text_indices]
    if condition is not None:
        prefix[:, layout.image_indices] = model.img_in(condition)
    target = model.img_in(latents)
    prefix_len = prefix.shape[1]
    temb = model.time_text_embed(timestep, target.dtype)
    target_mod = model.prepare_modulation(temb)
    prefix_mod = model.prepare_modulation(
        model.time_text_embed(mx.zeros((1,)), target.dtype)
    )
    modulation = [
        mx.concatenate(
            (
                mx.broadcast_to(a, (1, prefix_len, a.shape[-1])),
                mx.broadcast_to(b, target.shape),
            ),
            axis=1,
        )
        for a, b in zip(prefix_mod, target_mod)
    ]
    x = mx.concatenate((prefix, target), axis=1)
    rope = tuple(
        mx.concatenate((a, b)) for a, b in zip(layout.prefix_rope, layout.target_rope)
    )
    length = x.shape[1]
    allowed = [[False] * length for _ in range(length)]
    for start, end, image in layout.segments:
        for row in range(start, end):
            for col in range(end if image else row + 1):
                allowed[row][col] = True
    for row in range(prefix_len, length):
        allowed[row] = [True] * length
    mask = mx.array(allowed)
    for block in model.transformer_blocks:
        q, k, v = block.attn.qkv(block.img_norm1(x) * (1 + modulation[0]), rope)
        attended = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=block.attn.head_dim**-0.5, mask=mask
        )
        x = x + block.attn.output(attended) * modulation[1]
        x = x + block.img_mlp(block.img_norm2(x) * (1 + modulation[2])) * modulation[3]
    return model.proj_out(model.norm_out(x[:, prefix_len:], temb))


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("bits", [None, 4, 8])
@pytest.mark.parametrize("with_images", [False, True])
def test_prefix_cache_matches_block_causal_attention(dtype, bits, with_images):
    mx.random.seed(42)
    model = model_module.QwenImage21Transformer(
        num_layers=3, attention_head_dim=16, num_attention_heads=4, context_in_dim=64
    )
    model.set_dtype(dtype)
    if bits:
        nn.quantize(
            model,
            bits=bits,
            group_size=64,
            class_predicate=lambda path, module: isinstance(module, nn.Linear),
        )
    slots = [False, True, False, False, True, False] if with_images else [False] * 5
    shapes = [(1, 2, 2), (1, 1, 3), (1, 3, 2)] if with_images else [(1, 3, 2)]
    layout = model_module.build_layout(slots, shapes, (4, 6, 6))
    embeddings = mx.random.normal((1, len(slots), 64)).astype(dtype)
    latents = mx.random.normal((1, 6, 64)).astype(dtype)
    condition = mx.random.normal((1, 7, 64)).astype(dtype) if with_images else None
    caches = model.prepare_conditioning(embeddings, layout, condition)
    mx.eval(caches)
    before = [mx.array(value) for pair in caches for value in pair]
    compiled = mx.compile(model)

    for time in (900.0, 200.0):
        timestep = mx.array([time])
        expected = joint_reference(
            model, latents, embeddings, timestep, layout, condition
        )
        actual = model(latents, timestep, layout.target_rope, caches)
        compiled_actual = compiled(latents, timestep, layout.target_rope, caches)
        atol, rtol = (2e-6, 2e-5) if dtype == mx.float32 else (0.02, 0.02)
        assert mx.allclose(expected, actual, atol=atol, rtol=rtol).item()
        assert mx.allclose(actual, compiled_actual, atol=atol, rtol=rtol).item()

    second = model.prepare_conditioning(embeddings + 0.25, layout, condition)
    assert not mx.array_equal(caches[0][0], second[0][0]).item()
    after = [value for pair in caches for value in pair]
    assert all(mx.array_equal(a, b).item() for a, b in zip(before, after))


def test_reference_images_keep_block_positions():
    layout = model_module.build_layout([False, True, False], [(1, 2, 2), (1, 2, 2)])
    assert layout.text_indices.tolist() == [0, 1, 1, 1, 1, 2]
    assert layout.image_indices.tolist() == [1, 2, 3, 4]
    assert layout.segments == ((0, 1, False), (1, 5, True), (5, 6, False))
    assert mx.allclose(layout.target_rope[0][:, 0], mx.cos(mx.array(4.0))).item()


@pytest.mark.parametrize("with_images", [False, True])
def test_compiled_denoising_preserves_bf16_outputs(with_images):
    mx.random.seed(20260924)
    model = model_module.QwenImage21Transformer(
        num_layers=2, num_attention_heads=2, context_in_dim=256
    )
    model.set_dtype(mx.bfloat16)
    nn.quantize(
        model,
        bits=4,
        group_size=64,
        class_predicate=lambda path, module: isinstance(module, nn.Linear),
    )
    slots = [False, True, False] if with_images else [False] * 37
    shapes = [(1, 8, 8), (1, 16, 16)] if with_images else [(1, 16, 16)]
    layout = model_module.build_layout(slots, shapes)
    embeddings = mx.random.normal((1, len(slots), 256)).astype(mx.bfloat16)
    condition = (
        mx.random.normal((1, 64, 64)).astype(mx.bfloat16) if with_images else None
    )
    # mirror strict checkpoint loading before capturing parameter values
    mx.eval(model.parameters())
    caches = model.prepare_conditioning(embeddings, layout, condition)
    mx.eval(caches)
    compiled = model.compile_denoise()
    for time in (1000.0, 511.23, 20.0):
        latents = mx.random.normal((1, 256, 64)).astype(mx.bfloat16)
        timestep = mx.array([time])
        expected = model(latents, timestep, layout.target_rope, caches)
        actual = compiled(latents, timestep, layout.target_rope, caches)
        assert mx.array_equal(actual, expected).item()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
