from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.models.dits.ltx_2 import (
    LTX2AudioVideoRotaryPosEmbed,
    LTX2VideoTransformer3DModel,
)


def _build_fake_model():
    video_rope = LTX2AudioVideoRotaryPosEmbed(
        dim=8,
        patch_size=1,
        patch_size_t=1,
        base_num_frames=16,
        base_height=4,
        base_width=4,
        scale_factors=(8, 32, 32),
        theta=10000.0,
        causal_offset=1,
        modality="video",
        double_precision=False,
        rope_type="interleaved",
        num_attention_heads=2,
    )
    audio_rope = LTX2AudioVideoRotaryPosEmbed(
        dim=8,
        patch_size=1,
        patch_size_t=1,
        base_num_frames=16,
        sampling_rate=16000,
        hop_length=160,
        scale_factors=(4,),
        theta=10000.0,
        causal_offset=1,
        modality="audio",
        double_precision=False,
        rope_type="interleaved",
        num_attention_heads=2,
    )
    return SimpleNamespace(
        rope=video_rope,
        audio_rope=audio_rope,
        _maybe_quantize_video_rope_coords=lambda coords, device, dtype: coords.to(
            device=device, dtype=dtype
        ),
    )


def test_ltx2_cross_attn_anchor_coords_use_global_first_token():
    fake_model = _build_fake_model()
    device = torch.device("cpu")

    cross_video_coords, cross_audio_coords = (
        LTX2VideoTransformer3DModel._prepare_cross_attn_anchor_coords(
            fake_model,
            batch_size=1,
            height=2,
            width=2,
            fps=24.0,
            hidden_device=device,
            hidden_dtype=torch.float32,
            audio_device=device,
        )
    )

    full_video_coords = fake_model.rope.prepare_video_coords(
        batch_size=1,
        num_frames=3,
        height=2,
        width=2,
        device=device,
        fps=24.0,
        start_frame=0,
    )
    full_audio_coords = fake_model.audio_rope.prepare_audio_coords(
        batch_size=1,
        num_frames=4,
        device=device,
        start_frame=0,
    )
    shifted_video_coords = fake_model.rope.prepare_video_coords(
        batch_size=1,
        num_frames=3,
        height=2,
        width=2,
        device=device,
        fps=24.0,
        start_frame=3,
    )
    shifted_audio_coords = fake_model.audio_rope.prepare_audio_coords(
        batch_size=1,
        num_frames=4,
        device=device,
        start_frame=4,
    )

    assert cross_video_coords.shape == (1, 3, 1, 2)
    assert cross_audio_coords.shape == (1, 1, 1, 2)
    assert torch.equal(cross_video_coords, full_video_coords[:, :, 0:1, :])
    assert torch.equal(cross_audio_coords, full_audio_coords[:, :, 0:1, :])
    assert not torch.equal(cross_video_coords, shifted_video_coords[:, :, 0:1, :])
    assert not torch.equal(cross_audio_coords, shifted_audio_coords[:, :, 0:1, :])
