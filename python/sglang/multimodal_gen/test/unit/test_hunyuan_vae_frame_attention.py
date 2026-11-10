# SPDX-License-Identifier: Apache-2.0

import weakref
from unittest.mock import patch

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from sglang.multimodal_gen.runtime.layers.parallel_conv import (
    disable_spatial_parallel_decode,
)
from sglang.multimodal_gen.runtime.models.vaes.hunyuanvae import (
    HunyuanVAEAttention,
    HunyuanVideoMidBlock3D,
    HunyuanVideoUpsampleCausal3D,
    prepare_causal_attention_mask,
)


@pytest.mark.parametrize("frames", [1, 3, 7])
@pytest.mark.parametrize("batch_size,heads", [(1, 1), (2, 2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_frame_attention_matches_dense_causal_mask(frames, batch_size, heads, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the production SDPA path")
    torch.manual_seed(0)
    attention = HunyuanVAEAttention(32, heads, 16, 1e-6, 4, True).cuda().to(dtype)
    hidden = torch.randn(batch_size, frames * 15, 32, device="cuda", dtype=dtype)
    mask = prepare_causal_attention_mask(frames, 15, dtype, hidden.device)
    with torch.inference_mode():
        expected = attention(hidden, attention_mask=mask)
        actual = attention(hidden, num_frames=frames)
    # unmasked prefixes may select a different fused SDPA reduction kernel
    tolerance = max(1e-6, torch.finfo(dtype).eps)
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("batch_size,heads", [(1, 1), (2, 2)])
def test_bf16_frame_attention_matches_dense_with_same_math_backend(batch_size, heads):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the production SDPA path")
    torch.manual_seed(0)
    attention = HunyuanVAEAttention(32, heads, 16, 1e-6, 4, True).cuda().bfloat16()
    hidden = torch.randn(batch_size, 105, 32, device="cuda", dtype=torch.bfloat16)
    mask = prepare_causal_attention_mask(7, 15, hidden.dtype, hidden.device)
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        expected = attention(hidden, attention_mask=mask)
        actual = attention(hidden, num_frames=7)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_spatial_attention_avoids_dense_mask_and_preserves_height_shards():
    torch.manual_seed(0)
    attention = HunyuanVAEAttention(8, 2, 4, 1e-6, 2, True)
    stage = HunyuanVideoMidBlock3D.__new__(HunyuanVideoMidBlock3D)
    torch.nn.Module.__init__(stage)
    stage.spatial_parallel = True
    hidden = torch.randn(1, 8, 3, 5, 3)
    local = hidden[:, :, :, :2]
    mask = prepare_causal_attention_mask(3, 15, hidden.dtype, hidden.device)
    with torch.inference_mode():
        expected = attention(hidden.permute(0, 2, 3, 4, 1).flatten(1, 3), mask)
        expected = expected.unflatten(1, (3, 5, 3)).permute(0, 4, 1, 2, 3)
        with (
            patch(
                "sglang.multimodal_gen.runtime.models.vaes.hunyuanvae.gather_variable_height",
                return_value=(hidden, [2, 3]),
            ) as gather,
            patch(
                "sglang.multimodal_gen.runtime.models.vaes.hunyuanvae.chunk_height_by_sizes",
                side_effect=lambda value, sizes: value[:, :, :, : sizes[0]],
            ) as chunk,
            patch(
                "sglang.multimodal_gen.runtime.models.vaes.hunyuanvae.prepare_causal_attention_mask",
                side_effect=AssertionError(
                    "spatial decode must not build a dense mask"
                ),
            ),
        ):
            actual = stage._run_attention(attention, local)
    gather.assert_called_once_with(local)
    assert chunk.call_args.args[1] == [2, 3]
    torch.testing.assert_close(actual, expected[:, :, :, :2])


def test_frame_attention_preserves_gradient_flow():
    torch.manual_seed(0)
    attention = HunyuanVAEAttention(8, 2, 4, 1e-6, 2, True).double()
    hidden = torch.randn(1, 12, 8, dtype=torch.float64, requires_grad=True)
    mask = prepare_causal_attention_mask(3, 4, hidden.dtype, hidden.device)
    expected = attention(hidden, attention_mask=mask)
    expected_grad = torch.autograd.grad(expected.square().sum(), hidden)[0]
    actual = attention(hidden, num_frames=3)
    actual_grad = torch.autograd.grad(actual.square().sum(), hidden)[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad)


def test_disabled_spatial_decode_preserves_dense_attention():
    torch.manual_seed(0)
    attention = HunyuanVAEAttention(8, 2, 4, 1e-6, 2, True)
    stage = HunyuanVideoMidBlock3D.__new__(HunyuanVideoMidBlock3D)
    torch.nn.Module.__init__(stage)
    hidden = torch.randn(1, 8, 3, 5, 3)
    with torch.inference_mode():
        stage.spatial_parallel = False
        expected = stage._run_attention(attention, hidden)
        stage.spatial_parallel = True
        with (
            disable_spatial_parallel_decode(),
            patch.object(attention, "forward", wraps=attention.forward) as forward,
        ):
            actual = stage._run_attention(attention, hidden)
    assert "num_frames" not in forward.call_args.kwargs
    assert forward.call_args.kwargs["attention_mask"].shape == (1, 45, 45)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_upsample_releases_intermediates_before_convolution():
    upsample = HunyuanVideoUpsampleCausal3D(4)
    hidden = torch.randn(1, 4, 3, 5, 7)
    interpolate = torch.nn.functional.interpolate
    buffers = []

    def track_interpolate(*args, **kwargs):
        result = interpolate(*args, **kwargs)
        buffers.append(weakref.ref(result))
        return result

    def check_input(value):
        assert all(buffer() is None for buffer in buffers)
        return value

    with (
        torch.inference_mode(),
        patch(
            "sglang.multimodal_gen.runtime.models.vaes.hunyuanvae.F.interpolate",
            side_effect=track_interpolate,
        ),
        patch.object(upsample.conv, "forward", side_effect=check_input),
    ):
        actual = upsample(hidden)
    expected = torch.cat(
        (
            interpolate(hidden[:, :, 0], scale_factor=2, mode="nearest").unsqueeze(2),
            interpolate(hidden[:, :, 1:].contiguous(), scale_factor=2, mode="nearest"),
        ),
        dim=2,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
