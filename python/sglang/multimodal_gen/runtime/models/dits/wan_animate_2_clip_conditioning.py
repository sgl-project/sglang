# SPDX-License-Identifier: Apache-2.0
"""Per-clip contracts between the Wan-Animate-2 stages and the DiT.

Lives on the model side so the DiT can type its ``clip_cond`` / ``reference_kv`` kwargs
without importing a pipeline stage: the before-denoising stage builds the conditioning, the
DiT's ``build_reference_kv`` turns it into the reference K/V the denoising stage hands to
every ``forward`` of that clip.
"""

import msgspec
import torch


class WanAnimate2ClipConditioning(msgspec.Struct, frozen=True, kw_only=True):
    """Per-clip conditioning consumed by the DiT's forward_ref / forward_gen."""

    # Inputs for ``forward_ref`` (reference video = the motion source)
    # VAE latents of this clip's reference-video frames,
    # [16, reference_video_latent_t, reference_video_latent_h, reference_video_latent_w] fp32.
    reference_video_latents: torch.Tensor
    # Generation token grid (latent_t, latent_h // 2, latent_w // 2): the DiT patchifies the
    # latent with patch_size (1, 2, 2), so each spatial side halves.
    generation_video_grid_sizes: tuple[int, int, int]
    # CLIP tokens of the clip's first reference-video frame, [1, 257, 1280] bf16.
    reference_video_frame_0_image_embeddings: torch.Tensor
    # [20, reference_video_latent_t, reference_video_latent_h, reference_video_latent_w] bf16.
    # Channels: [0:4] given-frame mask, all ones (every reference frame is known);
    #           [4:20] VAE latent, the same tensor as reference_video_latents.
    # Time: the clip's reference-video latent frames; no reference-image frame in front.
    # forward_ref cats it under reference_video_latents (16 ch) for the 36-channel DiT input.
    reference_video_condition: torch.Tensor
    # T5 embedding of prompt_ref, [L_prompt_ref, 4096] bf16.
    prompt_ref_embeddings: torch.Tensor

    # Inputs for: ``forward_gen``
    # [20, latent_t, latent_h, latent_w] bf16.
    # Channels: [0:4] given-frame mask (1 = known pixels, 0 = to generate);
    #           [4:20] VAE latent of the given pixels, encoded zeros elsewhere.
    # Time: frame 0 is the reference image (mask = all ones); frames 1..latent_t are the clip's
    #       own clip_latent_t frames, given only for the num_frames_conditioning frames fed
    #       back from the previous clip (none for clip 0).
    # forward_gen cats it under the noise latent (16 ch) for the 36-channel DiT input.
    generation_condition: torch.Tensor
    # Reference-video token grid (reference_video_latent_t, _h // 2, _w // 2); same halving.
    reference_video_grid_sizes: tuple[int, int, int]
    # Token grid of a full-length clip (the configured clip_len); the attention layout and
    # block mask are sized from it even on the shorter last clip, so one compiled mask
    # serves the whole request.
    full_clip_grid_sizes: tuple[int, int, int]

    # Pixel-frame count of this clip (the last clip may be shorter than clip_len).
    num_frames: int
    # Seeded initial noise for the denoising loop, [16, latent_t, latent_h, latent_w] fp32.
    init_noise: torch.Tensor


class WanAnimate2ReferenceKV(msgspec.Struct, frozen=True):
    """One clip's post-RoPE reference-video K and V, keyed by DiT block index.

    Returned by ``WanAnimate2Transformer3DModel.build_reference_kv`` and owned by the caller
    for that clip only; the DiT keeps no copy between calls.
    """

    # [1, S_reference_video, local_heads, head_dim] in the DiT param dtype. Under Ulysses SP
    # this is the full sequence with sharded heads, the layout forward_gen uses after its a2a.
    k: dict[int, torch.Tensor]
    v: dict[int, torch.Tensor]
