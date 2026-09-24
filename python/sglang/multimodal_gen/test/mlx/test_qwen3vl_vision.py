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
    Path(__file__).resolve().parents[2]
    / "runtime/hardware_backend/mlx/qwen3vl_vision.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "qwen3vl_mlx_vision_test_model", _MODEL_PATH
)
model_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(model_module)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_vision_matches_transformers_and_isolates_reference_images(dtype):
    torch.manual_seed(31)
    settings = dict(
        hidden_size=288,
        intermediate_size=384,
        out_hidden_size=64,
        num_heads=4,
        depth=3,
        in_channels=3,
        patch_size=4,
        temporal_patch_size=2,
        spatial_merge_size=2,
        num_position_embeddings=64,
        deepstack_visual_indexes=[0, 2],
    )
    config = hf.Qwen3VLVisionConfig(**settings)
    config._attn_implementation = "sdpa"
    reference = hf.Qwen3VLVisionModel(config).eval().to(dtype)
    # from_pretrained retains fp32 rotary buffers; Module.to(dtype) does not
    reference.rotary_pos_emb = hf.Qwen3VLVisionRotaryEmbedding(
        config.hidden_size // config.num_heads // 2
    )
    model = model_module.Qwen3VLVisionEncoder(**settings)
    mlx_dtype = mx.float32 if dtype == torch.float32 else mx.bfloat16
    weights = []
    for key, value in reference.state_dict().items():
        if key == "patch_embed.proj.weight":
            value = value.permute(0, 2, 3, 4, 1).contiguous()
        weights.append((key, mx.array(value.float().numpy()).astype(mlx_dtype)))
    model.load_weights(weights, strict=True)
    grid = [(1, 4, 6), (1, 6, 4)]
    pixels = torch.randn(48, 3 * 2 * 4 * 4).to(dtype)
    # exercise the production 72-wide fused attention against fp32-accumulating SDPA
    with (
        torch.no_grad(),
        torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH),
    ):
        expected, deepstack = reference(pixels, torch.tensor(grid))
    x = mx.array(pixels.float().numpy()).astype(mlx_dtype)
    actual, actual_deepstack = model(x, grid)
    assert len(actual_deepstack) == len(deepstack) == 2
    for observed, target in zip([actual, *actual_deepstack], [expected, *deepstack]):
        np.testing.assert_allclose(
            np.array(observed.astype(mx.float32)),
            target.float().numpy(),
            atol=1e-6 if dtype == torch.float32 else 0.002,
            rtol=2e-5 if dtype == torch.float32 else 0.015,
        )
    isolated, isolated_deepstack = model(x[:24], grid[:1])
    for together, alone in zip(
        [actual, *actual_deepstack], [isolated, *isolated_deepstack]
    ):
        np.testing.assert_allclose(
            np.array(together[:6].astype(mx.float32)),
            np.array(alone.astype(mx.float32)),
            atol=3e-7 if dtype == torch.float32 else 0.002,
            rtol=1e-5,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
