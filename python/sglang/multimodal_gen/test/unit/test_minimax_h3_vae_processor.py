# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MiniMax-H3 visual VAE tensor postprocessing."""

import torch

from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.processor import (
    VAEProcessor,
    get_denormalize_transform,
)


def _processor(
    *, custom_reverse: bool = False, custom_reverse_inplace: bool = False
) -> VAEProcessor:
    kwargs = {}
    if custom_reverse:
        kwargs["transform_rev"] = get_denormalize_transform("simple")
    if custom_reverse_inplace:
        kwargs["transform_rev_inplace"] = get_denormalize_transform(
            "simple", inplace=True
        )
    return VAEProcessor(
        vae_ratio=16,
        vae_ratio_t=4,
        clip_length=17,
        frame_overlap=0,
        token_overlap=3,
        tokens_chunk_size=5,
        isolated_last_frame=False,
        latent_patch_size=1,
        crop_mode="top_left",
        pixel_norm_type="simple",
        use_3d_conv=True,
        **kwargs,
    )


def test_revert_tensor_preserves_input_by_default():
    processor = _processor()
    tensor = torch.rand(1, 3, 5, 8, 8)
    original = tensor.clone()

    output = processor.revert_tensor(tensor)

    torch.testing.assert_close(tensor, original, rtol=0, atol=0)
    assert output.untyped_storage().data_ptr() != tensor.untyped_storage().data_ptr()


def test_runtime_owned_revert_matches_out_of_place_and_reuses_storage():
    processor = _processor()
    tensor = torch.rand(1, 3, 5, 8, 8)
    expected = processor.revert_tensor(tensor)
    runtime_owned_tensor = tensor.clone()
    input_storage = runtime_owned_tensor.untyped_storage().data_ptr()

    output = processor.revert_tensor(runtime_owned_tensor, runtime_owned=True)

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert output.untyped_storage().data_ptr() == input_storage


def test_runtime_owned_revert_uses_explicit_inplace_transform():
    processor = _processor(custom_reverse=True, custom_reverse_inplace=True)
    tensor = torch.rand(1, 3, 5, 8, 8)
    expected = processor.revert_tensor(tensor)
    runtime_owned_tensor = tensor.clone()
    input_storage = runtime_owned_tensor.untyped_storage().data_ptr()

    output = processor.revert_tensor(runtime_owned_tensor, runtime_owned=True)

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert output.untyped_storage().data_ptr() == input_storage


def test_runtime_owned_revert_falls_back_for_custom_transform():
    processor = _processor(custom_reverse=True)
    tensor = torch.rand(1, 3, 5, 8, 8)
    original = tensor.clone()

    output = processor.revert_tensor(tensor, runtime_owned=True)

    torch.testing.assert_close(tensor, original, rtol=0, atol=0)
    assert output.untyped_storage().data_ptr() != tensor.untyped_storage().data_ptr()
