# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")
F = torch.nn.functional

_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "runtime/hardware_backend/mlx/qwen_image21_vae.py"
)
_SPEC = importlib.util.spec_from_file_location("qwen21_mlx_vae_test_model", _MODEL_PATH)
model_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(model_module)


def torch_tensor(value):
    return torch.from_numpy(np.array(value.astype(mx.float32))).to(
        torch.bfloat16 if value.dtype == mx.bfloat16 else torch.float32
    )


def assert_close(actual, expected, atol=1e-5):
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)),
        expected.float().numpy(),
        atol=atol,
        rtol=1e-5,
    )


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_channel_norm_keeps_low_precision_cast(dtype):
    mx.random.seed(19)
    x = mx.random.normal((2, 7, 5, 32)).astype(dtype)
    norm = model_module.ChannelNorm(32)
    norm.gamma = mx.random.uniform(0.5, 2.0, (32,)).astype(dtype)
    value = torch_tensor(x)
    expected = F.normalize(value.float(), dim=-1).to(value.dtype)
    expected = expected * 32**0.5 * torch_tensor(norm.gamma)
    assert_close(norm(x), expected, atol=1e-6 if dtype == mx.float32 else 0)
    assert mx.array_equal(norm(mx.zeros_like(x)), mx.zeros_like(x)).item()


@pytest.mark.parametrize("channels", [(8, 8), (8, 16), (16, 8)])
@pytest.mark.parametrize("temporal", [1, 2])
def test_shortcuts_match_single_frame_video_layout(channels, temporal):
    in_channels, out_channels = channels
    x = mx.arange(2 * 4 * 6 * in_channels, dtype=mx.float32).reshape(
        2, 4, 6, in_channels
    )
    video = torch_tensor(x).permute(0, 3, 1, 2).unsqueeze(2)
    padded = F.pad(video, (0, 0, 0, 0, temporal - 1, 0))
    down = padded.reshape(2, in_channels, 1, temporal, 2, 2, 3, 2)
    down = down.permute(0, 1, 3, 5, 7, 2, 4, 6).reshape(2, out_channels, -1, 1, 2, 3)
    down = down.mean(dim=2).squeeze(2).permute(0, 2, 3, 1)
    assert_close(
        model_module.AverageDownsample(in_channels, out_channels, temporal, 2)(x),
        down,
        0,
    )

    up = video.repeat_interleave(out_channels * temporal * 4 // in_channels, dim=1)
    up = up.reshape(2, out_channels, temporal, 2, 2, 1, 4, 6)
    up = up.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(2, out_channels, temporal, 8, 12)
    up = up[:, :, -1].permute(0, 2, 3, 1)
    assert_close(
        model_module.DuplicateUpsample(in_channels, out_channels, temporal)(x), up, 0
    )


def test_residual_block_matches_pytorch_convolutions():
    mx.random.seed(11)
    block = model_module.ResidualBlock(16, 32)
    x = mx.random.normal((1, 8, 6, 16))

    def conv(value, layer, padding):
        return F.conv2d(
            value,
            torch_tensor(layer.weight).permute(0, 3, 1, 2),
            torch_tensor(layer.bias),
            padding=padding,
        )

    def norm(value, layer):
        return (
            F.normalize(value, dim=1)
            * layer.scale
            * torch_tensor(layer.gamma)[None, :, None, None]
        )

    value = torch_tensor(x).permute(0, 3, 1, 2)
    residual = conv(value, block.conv_shortcut, 0)
    value = conv(F.silu(norm(value, block.norm1)), block.conv1, 1)
    value = conv(F.silu(norm(value, block.norm2)), block.conv2, 1) + residual
    assert_close(block(x), value.permute(0, 2, 3, 1))


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_native_vae_encodes_and_decodes_alpha(dtype):
    mx.random.seed(12)
    model = model_module.QwenImage21VAE(
        base_dim=4,
        decoder_base_dim=4,
        z_dim=4,
        dim_mult=(1, 2, 2),
        num_res_blocks=1,
        temperal_downsample=(False, True),
    )
    model.set_dtype(dtype)
    image = mx.random.uniform(-1, 1, (1, 16, 12, 4)).astype(dtype)
    latents = model.encode(image)
    assert latents.shape == (1, 4, 3, 4)
    changed_alpha = mx.concatenate((image[..., :3], -image[..., 3:]), axis=-1)
    assert not mx.allclose(latents, model.encode(changed_alpha)).item()
    decoded = model.decode(latents)
    assert decoded.shape == image.shape
    assert mx.all(mx.isfinite(decoded)).item()
    assert mx.max(mx.abs(decoded)).item() <= 1
    assert mx.max(decoded[..., 3]).item() > mx.min(decoded[..., 3]).item()
    compiled = model.compile_decode()
    for value in (latents, latents + 0.25):
        expected, actual = model.decode(value), compiled(value)
        if dtype == mx.bfloat16:
            assert mx.array_equal(expected, actual).item()
        else:
            assert mx.allclose(expected, actual, atol=1e-6, rtol=1e-5).item()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
