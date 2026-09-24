# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")
hf = pytest.importorskip("transformers.models.qwen3_vl.modeling_qwen3_vl")

_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "runtime/hardware_backend/mlx/qwen3vl_text.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "qwen3vl_mlx_text_test_model", _MODEL_PATH
)
model_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(model_module)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_images", [False, True])
def test_encoder_matches_transformers_pre_norm_and_deepstack(dtype, with_images):
    torch.manual_seed(19)
    settings = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rope_theta=5000000,
        rms_norm_eps=1e-6,
    )
    config = hf.Qwen3VLTextConfig(
        **settings,
        rope_scaling={
            "rope_type": "default",
            "mrope_section": [3, 3, 2],
            "mrope_interleaved": True,
        },
    )
    config._attn_implementation = "sdpa"
    reference = hf.Qwen3VLTextModel(config).eval().to(dtype)
    reference.rotary_emb = hf.Qwen3VLTextRotaryEmbedding(config)
    model = model_module.Qwen3VLTextEncoder(**settings, mrope_section=(3, 3, 2))
    mlx_dtype = mx.float32 if dtype == torch.float32 else mx.bfloat16
    with torch.no_grad():
        reference.norm.weight.fill_(3)
    model.load_weights(
        [
            (name, mx.array(value.float().numpy()).astype(mlx_dtype))
            for name, value in reference.state_dict().items()
        ]
    )
    input_ids = torch.randint(0, 128, (2, 11))
    valid = torch.ones_like(input_ids)
    valid[1, :2] = 0
    positions = torch.arange(11)[None, None].expand(3, 2, -1).clone()
    torch_kwargs, mlx_kwargs = {}, {}
    if with_images:
        positions[1, :, 3:7] = torch.tensor([3, 3, 4, 4])
        positions[2, :, 3:7] = torch.tensor([3, 4, 3, 4])
        visual_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        visual_mask[:, 3:7] = True
        deepstack = [torch.randn(8, 64).to(dtype) * 0.01 for _ in range(3)]
        torch_kwargs = dict(
            visual_pos_masks=visual_mask, deepstack_visual_embeds=deepstack
        )
        mlx_kwargs = dict(
            visual_positions=mx.array(
                visual_mask.flatten().nonzero().flatten().numpy()
            ),
            deepstack_visual_embeds=[
                mx.array(x.float().numpy()).astype(mlx_dtype) for x in deepstack
            ],
        )
    captured = []
    handle = reference.norm.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone())
    )
    with torch.no_grad():
        reference(
            input_ids,
            position_ids=positions,
            attention_mask=valid,
            use_cache=False,
            **torch_kwargs,
        )
    handle.remove()
    actual = model(
        mx.array(input_ids.numpy()),
        position_ids=mx.array(positions.numpy()),
        attention_mask=mx.array(valid.numpy()),
        **mlx_kwargs,
    )
    # padded queries are discarded by the prompt stage; valid queries are the contract
    actual_valid = np.array(actual.astype(mx.float32))[valid.bool().numpy()]
    expected = captured[0].float().numpy()[valid.bool().numpy()]
    np.testing.assert_allclose(
        actual_valid,
        expected,
        atol=2e-7 if dtype == torch.float32 else 0.002,
        rtol=2e-5 if dtype == torch.float32 else 0.015,
    )


def test_multimodal_rope_keeps_bfloat16_operation_order():
    torch.manual_seed(23)
    positions = torch.randint(0, 1024, (3, 2, 7))
    x = torch.randn(2, 7, 4, 128).to(torch.bfloat16)
    frequency = 5000000.0 ** (-torch.arange(0, 128, 2).float() / 128)
    angles = positions[..., None].float() * frequency
    selected = angles[0].clone()
    selected[..., 1:60:3] = angles[1, ..., 1:60:3]
    selected[..., 2:60:3] = angles[2, ..., 2:60:3]
    cos, sin = (
        selected.cos().to(x.dtype)[:, :, None],
        selected.sin().to(x.dtype)[:, :, None],
    )
    first, second = x.chunk(2, dim=-1)
    expected = torch.cat(
        (first * cos - second * sin, second * cos + first * sin), dim=-1
    )
    rope = model_module.interleaved_rope(
        mx.array(positions.numpy()), 128, 5000000, (24, 20, 20), mx.bfloat16
    )
    actual = model_module.apply_rope(
        mx.array(x.float().numpy()).astype(mx.bfloat16), rope
    )
    np.testing.assert_array_equal(
        np.array(actual.astype(mx.float32)), expected.float().numpy()
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
