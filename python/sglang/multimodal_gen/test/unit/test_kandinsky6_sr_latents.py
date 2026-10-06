# SPDX-License-Identifier: Apache-2.0
"""Pure-function unit test for a numerical gap found by the SGLang-vs-Diffusers audit,
pinned directly against ``tiled.py`` (no external reference, no model weights - see
``test_kandinsky6_sr_reference_parity.py`` for the k6_video-backed comparisons).

GAP 3 (``tiled.py`` initial-latent dtype): the reference casts the LQ latent to the DiT's
loaded parameter dtype *before* building the initial latent (so a bf16-loaded checkpoint's
LQ-noise mix rounds in bf16), but ``build_chunk_latent`` hard-coded ``dtype=torch.float32``
regardless of the DiT's precision.

GAP 2 (the pi-Flow last-segment ``raw_dst`` bug that used to live in
``latents.piflow_schedule``) no longer applies: that hand-written schedule function was
removed when the denoising stage started driving the bundle's own ``PiflowScheduler`` object
directly (``PiflowScheduler._policy_step`` already implements the fixed convention -- see
``sampling.denoise_with_scheduler``), so there is no local re-derivation of the pi-Flow
schedule left to regress.
"""

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.sampling import (
    DitSpec,
    SamplingSpec,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.tiled import (
    build_chunk_latent,
)


def _dit_spec(dtype: torch.dtype | None) -> DitSpec:
    return DitSpec(
        instruct_type="noise",
        visual_cond=False,
        in_visual_dim=4,
        use_motion_score=False,
        patch_size=(1, 1, 1),
        dtype=dtype,
    )


def _tiny_spec() -> SamplingSpec:
    return SamplingSpec(
        tiling_scale=2,
        tiles_batch_size=1,
        seed=1,
        num_steps=5,
        tile_min_overlap=0.2,
        visual_size=512,
        scale_factor=(1.0, 1.0, 1.0),
        scheduler_scale=5.0,
        lq_noise_scale=0.7,
        lq_noise_type="linear",
        lq_channel_noise_scale=0.0,
        cap_noise_timestep=False,
        piflow=None,
    )


def test_build_chunk_latent_dtype_follows_the_dit_not_a_hardcoded_fp32():
    """Fails on the pre-fix code, which passed ``dtype=torch.float32`` into
    ``build_initial_latent`` unconditionally: the bf16 case below would come back
    fp32 instead of bf16."""
    lq_tile = torch.randn(4, 4, 4, 4)  # [T, H, W, C], starts fp32 regardless of DiT
    spec = _tiny_spec()

    for dit_dtype in (torch.bfloat16, torch.float32):
        chunk = build_chunk_latent(
            [lq_tile],
            seed=1,
            dit_spec=_dit_spec(dit_dtype),
            spec=spec,
            device=torch.device("cpu"),
        )
        assert chunk.dtype == dit_dtype, (
            f"expected the DiT's own dtype {dit_dtype}, got {chunk.dtype} "
            "(the pre-fix code always produced float32)"
        )


def test_build_chunk_latent_keeps_the_lq_dtype_when_the_dit_spec_has_none():
    """``DitSpec.dtype=None`` (a parameter-less / mocked DiT) must not force any
    cast - the initial latent keeps the caller's own LQ dtype, matching
    ``build_initial_latent``'s documented ``dtype=None`` contract."""
    lq_tile = torch.randn(4, 4, 4, 4, dtype=torch.float64)
    chunk = build_chunk_latent(
        [lq_tile],
        seed=1,
        dit_spec=_dit_spec(None),
        spec=_tiny_spec(),
        device=torch.device("cpu"),
    )
    assert chunk.dtype == torch.float64
