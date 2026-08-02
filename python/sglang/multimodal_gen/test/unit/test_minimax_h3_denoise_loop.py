# SPDX-License-Identifier: Apache-2.0
"""Numerical contract for request-static H3 denoise metadata."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_minimax_h3_euler_ancestral import (
    _minimax_h3_euler_eta0_step,
    _minimax_h3_rf_v_to_x0,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_QUALITY_PROFILES,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
    MiniMaxH3DenoiseBranch,
    _build_local_embedding_layout,
    _minimax_h3_update_target_rows_,
    minimax_h3_denoise_loop,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
    minimax_h3_packed_sequence,
    minimax_h3_packed_sequence_ref2va_blocks,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
    MiniMaxH3DenoisingStage,
)


def _branch(
    mode: str, token_tags: torch.Tensor | None = None
) -> MiniMaxH3DenoiseBranch:
    common = dict(text_len=3, latent_t=2, latent_h=4, latent_w=4, audio_t=3)
    if mode == "t2va":
        packed = minimax_h3_packed_sequence(
            **common,
            include_keyframe_cond=False,
        )
    elif mode == "fl2va":
        packed = minimax_h3_packed_sequence(
            **common,
            include_keyframe_cond=True,
            keyframe_frame_indices=[0, -1],
            frame_count=5,
        )
    else:
        packed = minimax_h3_packed_sequence_ref2va_blocks(
            **common,
            ref_blocks=[
                {"kind": "image", "latent_h": 4, "latent_w": 4},
                {"kind": "audio", "ref_audio_t": 2},
            ],
        )
    return MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.zeros(3, 5120),
        token_tags=packed["token_tags"] if token_tags is None else token_tags,
        device=torch.device("cpu"),
    )


def test_precomputed_timestep_plan_matches_full_unique_reference():
    """Preplanning must preserve fp32 collisions and every packed row class."""

    for mode in ("t2va", "fl2va", "ref2va"):
        branch = _branch(mode)
        assert branch.static_kwargs["skip_mask_out_condition"]
        assert "token_tags" not in branch.static_kwargs
        assert not bool((branch.static_kwargs["block_token_tags"] < 0).any())
        torch.testing.assert_close(
            branch.static_kwargs["img_pos_for_infer_output_info"]["position_ids"],
            branch.img_pos_dev[branch.update_mask_dev],
            rtol=0,
            atol=0,
        )
        video_steps = [0.75, 0.1]
        audio_steps = [0.625, 0.2]
        plan = branch.prepare_timestep_plan(
            video_timesteps=video_steps,
            audio_timesteps=audio_steps,
            imgvid_cond_noise_aug=0.6,
            audio_ref_cond_noise_aug=0.4,
        )

        assert branch.static_kwargs["packed_seq_params"]["cu_seqlens_q_host"] == tuple(
            int(value)
            for value in branch.static_kwargs["packed_seq_params"][
                "cu_seqlens_q"
            ].tolist()
        )
        assert branch.static_kwargs["refiner_packed_seq_params"][
            "cu_seqlens_q_host"
        ] == (0, 3, 3)

        for step, (video_t, audio_t) in enumerate(
            zip(video_steps, audio_steps, strict=True)
        ):
            reference = torch.full((branch.seq_len,), video_t, dtype=torch.float32)
            reference[branch.img_cond_seq_idx] = max(video_t, 0.6)
            reference[branch.audio_target_seq_idx] = audio_t
            reference[branch.audio_ref_seq_idx] = max(audio_t, 0.4)
            expected = torch.unique(reference, sorted=True, return_inverse=True)
            torch.testing.assert_close(plan[step][0], expected[0], rtol=0, atol=0)
            torch.testing.assert_close(plan[step][1], expected[1], rtol=0, atol=0)
            torch.testing.assert_close(
                plan[step][2],
                branch.static_kwargs["block_token_tags"]
                + expected[1] * MINIMAX_H3_ADALN_MODALITY_NUM,
                rtol=0,
                atol=0,
            )

        repeated_plan = branch.prepare_timestep_plan(
            video_timesteps=[0.0, 0.1, 0.2],
            audio_timesteps=[0.0, 0.2, 0.4],
            imgvid_cond_noise_aug=0.999,
            audio_ref_cond_noise_aug=1.0,
        )
        assert repeated_plan[1][1] is repeated_plan[2][1]
        assert repeated_plan[1][2] is repeated_plan[2][2]


def test_inplace_target_update_matches_scheduler_math():
    generator = torch.Generator().manual_seed(7)
    for sigma_curr, sigma_next in ((1.0, 0.7), (0.2, 0.0), (0.0, 0.0)):
        state = torch.randn(11, 32, generator=generator)
        velocity = torch.randn(11, 32, generator=generator)
        timestep = torch.tensor(1.0 - sigma_curr)
        ratio = torch.tensor(0.0 if sigma_curr == 0.0 else sigma_next / sigma_curr)
        denoised = _minimax_h3_rf_v_to_x0(state, velocity, timestep)
        expected = _minimax_h3_euler_eta0_step(
            state,
            denoised,
            sigma_curr=sigma_curr,
            sigma_next=sigma_next,
            sigma_ratio=ratio,
        )

        actual = state.clone()
        velocity_scratch = velocity.clone()
        _minimax_h3_update_target_rows_(
            actual,
            velocity_scratch,
            sigma_t=1.0 - timestep,
            sigma_curr=sigma_curr,
            sigma_ratio=ratio,
            one_minus_sigma_ratio=1.0 - ratio,
            denoised_scratch=torch.empty_like(actual),
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_denoise_loop_reuses_inference_mode_model_outputs():
    branch = _branch("t2va")
    video_rows = torch.zeros(branch.img_pos.shape[0], 96)
    audio_rows = torch.zeros(branch.audio_pos.shape[0], 32)
    video_target_rows = int(branch.update_mask.sum())

    def model(**_kwargs):
        return torch.zeros(video_target_rows, 96), torch.zeros_like(audio_rows)

    final_video, final_audio = minimax_h3_denoise_loop(
        model=model,
        positive=branch,
        initial_video_rows=video_rows,
        initial_audio_rows=audio_rows,
        keyframe_cond_rows=None,
        sigmas_video=[1.0, 0.0],
        sigmas_audio=[1.0, 0.0],
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(final_video, video_rows, rtol=0, atol=0)
    torch.testing.assert_close(final_audio, audio_rows, rtol=0, atol=0)


def test_local_text_layout_is_a_contiguous_prefix_per_ulysses_rank():
    for mode in ("t2va", "fl2va", "ref2va"):
        branch = _branch(mode)
        text_len = int(branch.static_kwargs["prompt_embeds"].shape[0])
        for world_size in (1, 2, 4, 8):
            for rank in range(world_size):
                layout = _build_local_embedding_layout(
                    seq_len=branch.seq_len,
                    text_pos=torch.arange(text_len),
                    img_pos=branch.img_pos,
                    audio_pos=branch.audio_pos,
                    world_size=world_size,
                    rank=rank,
                    device=torch.device("cpu"),
                )
                start = int(layout["text_source_start"])
                stop = int(layout["text_source_stop"])
                row_start = rank * (branch.seq_len // world_size)
                expected = torch.nonzero(
                    (torch.arange(text_len) >= row_start)
                    & (
                        torch.arange(text_len)
                        < row_start + branch.seq_len // world_size
                    )
                ).view(-1)
                assert expected.tolist() == list(range(start, stop))


def test_rank_local_token_tags_match_reference_slice():
    for mode in ("t2va", "fl2va", "ref2va"):
        seq_len = _branch(mode).seq_len
        token_tags = torch.arange(seq_len, dtype=torch.long) - seq_len // 2
        for world_size in (1, 2, 4, 8):
            for rank in range(world_size):
                with patch(
                    "sglang.multimodal_gen.runtime.pipelines_core.stages."
                    "model_specific_stages.minimax_h3.denoise_loop._ulysses_ctx",
                    return_value=(world_size, rank),
                ):
                    branch = _branch(mode, token_tags=token_tags)
                local_rows = branch.seq_len // world_size
                expected = token_tags[
                    rank * local_rows : (rank + 1) * local_rows
                ].clamp(min=0)
                torch.testing.assert_close(
                    branch.static_kwargs["block_token_tags"], expected, rtol=0, atol=0
                )


def test_quality_profiles_own_cache_dit_configs():
    stage = object.__new__(MiniMaxH3DenoisingStage)
    for profile_name, (warmup, threshold, max_cached) in {
        name: value
        for name, value in MINIMAX_H3_QUALITY_PROFILES.items()
        if value is not None
    }.items():
        stage._minimax_h3_quality_profile = profile_name
        assert stage._cache_dit_requested()
        assert stage._cache_dit_scm_masks(50) == ("none", "dynamic", None, None)
        config = stage._build_cache_dit_config(
            50,
            steps_computation_mask=None,
            scm_policy="dynamic",
        )
        assert config.Fn_compute_blocks == 1
        assert config.Bn_compute_blocks == 0
        assert config.max_warmup_steps == warmup
        assert config.residual_diff_threshold == threshold
        assert config.max_continuous_cached_steps == max_cached
        assert not config.enable_taylorseer


def test_quality_profile_transitions_respect_explicit_lossless():
    stage = object.__new__(MiniMaxH3DenoisingStage)
    transformer = object()
    stage.transformer = transformer
    stage._cache_dit_enabled = True
    stage._cached_num_steps = 50
    stage._minimax_h3_quality_profile = "high"
    stage._minimax_h3_cache_mode = "high"

    def batch(quality, explicit=False):
        return SimpleNamespace(
            sampling_params=SimpleNamespace(
                quality=quality,
                _explicit_fields={"quality"} if explicit else set(),
            ),
            is_warmup=False,
        )

    def enable(stage_arg, _num_steps, _batch):
        stage_arg._cache_dit_enabled = True

    with (
        patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages."
            "minimax_h3.stages.denoising.DenoisingStage._cache_dit_requested",
            return_value=True,
        ),
        patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages."
            "minimax_h3.stages.denoising.disable_cache_on_transformer",
            return_value=transformer,
        ) as disable,
        patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages."
            "minimax_h3.stages.denoising.DenoisingStage._maybe_enable_cache_dit",
            autospec=True,
            side_effect=enable,
        ) as enable_cache,
    ):
        stage._maybe_enable_cache_dit(50, batch("lossless", explicit=True))
        assert not stage._cache_dit_enabled
        assert stage._minimax_h3_cache_mode is None

        stage._maybe_enable_cache_dit(50, batch("lossless"))
        assert stage._cache_dit_enabled
        assert stage._minimax_h3_cache_mode == "generic"

        stage._maybe_enable_cache_dit(50, batch("medium", explicit=True))
        assert stage._cache_dit_enabled
        assert stage._minimax_h3_quality_profile == "medium"
        assert stage._minimax_h3_cache_mode == "medium"

    assert disable.call_count == 2
    assert enable_cache.call_count == 2
