# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 cfg-distilled full denoise loop.

Per step, the positive presentation is forwarded exactly once. Video and audio
target rows chain through the Euler-eta0 update while visual and audio condition
rows stay pinned to their noised step-0 anchors.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any, Callable

import torch

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_ctx,
    get_ulysses_ctx,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionMetadata,
)
from sglang.multimodal_gen.runtime.managers.forward_context import (
    set_forward_context,
)

MINIMAX_H3_IMGVID_COND_TIMESTEP = 0.999
# ref2va audio reference anchor timestep
MINIMAX_H3_AUDIO_REF_COND_TIMESTEP = 1.0
# Packed row widths: video rows are [1,2,2]-patchified 24-channel latents
# (24 * 1 * 2 * 2 = 96); audio rows carry the 32-dim audio latent.
MINIMAX_H3_VIDEO_ROW_WIDTH = 96
MINIMAX_H3_AUDIO_ROW_WIDTH = 32
_MINIMAX_H3_SUBBLOCK_QUERY_BLOCK_SIZE = 64


def _minimax_h3_subblock_video_query_indices(
    packed: dict[str, torch.Tensor],
    text_video_token_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Return packed video rows plus any Qwen rows marked as video."""
    latent_video_pos = packed["video_pos"].view(-1).to(dtype=torch.long)
    if text_video_token_mask is None:
        return latent_video_pos
    text_pos = packed["text_pos"].view(-1).to(dtype=torch.long)
    text_video_token_mask = text_video_token_mask.view(-1).to(
        device=text_pos.device,
        dtype=torch.bool,
    )
    if text_video_token_mask.shape[0] != text_pos.shape[0]:
        raise ValueError(
            "text_video_token_mask must align with packed text rows "
            f"({text_pos.shape[0]}), got {text_video_token_mask.shape[0]}"
        )
    return torch.cat([text_pos[text_video_token_mask], latent_video_pos])


def _minimax_h3_subblock_sparse_query_block_mask(
    video_query_indices: torch.Tensor,
    *,
    used_len: int,
) -> torch.Tensor:
    """Return True for pure-video 64-row Q blocks and False otherwise."""
    if used_len < 0:
        raise ValueError(f"used_len must be non-negative, got {used_len}")
    if video_query_indices.ndim != 1:
        raise ValueError(
            f"video_query_indices must be rank 1, got {list(video_query_indices.shape)}"
        )
    video_query_indices = video_query_indices.to(dtype=torch.long)
    if video_query_indices.numel():
        first = int(video_query_indices.min())
        last = int(video_query_indices.max())
        if first < 0 or last >= used_len:
            raise ValueError(
                "video_query_indices must be first-segment-relative and in "
                f"[0, {used_len}), got min={first}, max={last}"
            )
        if torch.unique(video_query_indices).numel() != video_query_indices.numel():
            raise ValueError("video_query_indices must not contain duplicates")
    block_size = _MINIMAX_H3_SUBBLOCK_QUERY_BLOCK_SIZE
    num_query_blocks = -(-used_len // block_size)
    video_rows_per_block = torch.bincount(
        torch.div(video_query_indices, block_size, rounding_mode="floor"),
        minlength=num_query_blocks,
    )
    block_ids = torch.arange(
        num_query_blocks, device=video_query_indices.device, dtype=torch.long
    )
    real_rows_per_block = (used_len - block_ids * block_size).clamp(
        min=0, max=block_size
    )
    # BSA is block-granular. Only blocks whose real rows are all video may use
    # the sparse budget; mixed boundary blocks stay dense so their non-video
    # rows retain exact attention in the single heterogeneous BSA call.
    return video_rows_per_block == real_rows_per_block


@torch.inference_mode()
def _minimax_h3_update_target_rows_(
    state: torch.Tensor,
    velocity: torch.Tensor,
    *,
    sigma_t: torch.Tensor,
    sigma_curr: float,
    sigma_ratio: torch.Tensor,
    one_minus_sigma_ratio: torch.Tensor,
    denoised_scratch: torch.Tensor,
) -> None:
    torch.mul(sigma_t, velocity, out=denoised_scratch)
    torch.add(state, denoised_scratch, out=denoised_scratch)
    if sigma_curr == 0.0:
        return
    torch.mul(one_minus_sigma_ratio, denoised_scratch, out=velocity)
    torch.mul(sigma_ratio, state, out=state)
    torch.add(state, velocity, out=state)


def _build_local_embedding_layout(
    *,
    seq_len: int,
    text_pos: torch.Tensor,
    img_pos: torch.Tensor,
    audio_pos: torch.Tensor,
    world_size: int,
    rank: int,
    device: torch.device,
) -> dict[str, torch.Tensor | int]:
    if seq_len % world_size:
        raise ValueError(
            f"packed seq_len {seq_len} not divisible by the combined "
            f"sequence-parallel world size {world_size}"
        )
    local_seq_len = seq_len // world_size
    row_start = rank * local_seq_len
    row_stop = row_start + local_seq_len

    def local_ids(pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        source_ids = torch.nonzero(
            (pos >= row_start) & (pos < row_stop),
            as_tuple=False,
        ).view(-1)
        return source_ids.to(device), pos.index_select(0, source_ids).to(device)

    text_source_start = min(row_start, int(text_pos.shape[0]))
    text_source_stop = min(row_stop, int(text_pos.shape[0]))
    _, img_global_ids = local_ids(img_pos)
    _, audio_global_ids = local_ids(audio_pos)
    return {
        "text_source_start": text_source_start,
        "text_source_stop": text_source_stop,
        "img_global_ids": img_global_ids,
        "img_row_ids": img_global_ids - row_start,
        "audio_global_ids": audio_global_ids,
        "audio_row_ids": audio_global_ids - row_start,
    }


class MiniMaxH3DenoiseBranch:
    """Static per-branch state: packed layout + fixed forward kwargs.

    `packed` is a minimax_h3_packed_sequence(...) result (or equivalent layout
    dict); `text_embeddings` is the branch's [text_len, 5120] hidden states;
    `token_tags` must already carry any fl2va vision-span overrides, and
    `video_query_indices` explicitly identifies the only rows eligible for
    SubBlock sparsity.
    """

    def __init__(
        self,
        *,
        packed: dict[str, torch.Tensor],
        text_embeddings: torch.Tensor,
        token_tags: torch.Tensor,
        device: torch.device,
        video_query_indices: torch.Tensor | None = None,
    ) -> None:
        seq_len = int(packed["seq_len"])
        self.seq_len = seq_len
        self.img_pos = packed["img_pos"].view(-1).to(torch.long)
        self.audio_pos = packed["audio_pos"].view(-1).to(torch.long)
        self.update_mask = packed["update_mask"].view(-1).to(torch.bool)
        # ref2va: audio_pos may include reference-audio anchor rows
        # (audio_update_mask False); absent means all rows are targets.
        if "audio_update_mask" in packed:
            self.audio_update_mask = packed["audio_update_mask"].view(-1).to(torch.bool)
        else:
            self.audio_update_mask = torch.ones(
                self.audio_pos.shape[0], dtype=torch.bool
            )
        # Packed H3 layouts place reference rows before the generated suffix
        # for both modalities. Keep the suffix boundary once so the denoise
        # hot path can update contiguous views instead of gathering and
        # scattering the same static index tensors on every step.
        self.video_target_start = int((~self.update_mask).sum())
        self.audio_target_start = int((~self.audio_update_mask).sum())
        self.video_target_slice = slice(self.video_target_start, None)
        self.audio_target_slice = slice(self.audio_target_start, None)
        text_pos = packed["text_pos"].view(-1).to(torch.long)
        text_len = int(text_pos.shape[0])
        if list(text_embeddings.shape)[0] != text_len:
            raise ValueError(
                f"text_embeddings rows {list(text_embeddings.shape)} != "
                f"packed text_len {text_len}"
            )
        if int(token_tags.view(-1).shape[0]) != seq_len:
            raise ValueError(
                f"token_tags length {int(token_tags.view(-1).shape[0])} != "
                f"seq_len {seq_len}"
            )
        cu = packed["cu_seqlens"].to(torch.int32)
        self.img_pos_dev = self.img_pos.to(device)
        self.audio_pos_dev = self.audio_pos.to(device)
        self.update_mask_dev = self.update_mask.to(device)
        self.audio_update_mask_dev = self.audio_update_mask.to(device)
        # Resolve the remaining step-static packed-sequence and anchor row
        # sets once, keeping nonzero-driven work out of the hot loop.
        self.img_cond_seq_idx = self.img_pos_dev[~self.update_mask_dev]
        self.img_target_seq_idx = self.img_pos_dev[self.update_mask_dev]
        self.audio_target_seq_idx = self.audio_pos_dev[self.audio_update_mask_dev]
        self.audio_ref_seq_idx = self.audio_pos_dev[~self.audio_update_mask_dev]
        self.cond_row_idx = torch.nonzero(~self.update_mask_dev).view(-1)
        self.audio_ref_row_idx = torch.nonzero(~self.audio_update_mask_dev).view(-1)
        # rows that keep the video timestep each step: text, padding, and
        # video target rows — everything the three overwrite sets do not cover
        self.n_video_timestep_rows = (
            seq_len
            - int(self.img_cond_seq_idx.numel())
            - int(self.audio_target_seq_idx.numel())
            - int(self.audio_ref_seq_idx.numel())
        )
        # persistent packed-row buffers; every img/audio position is fully
        # rewritten by index_copy_ on the first forward_kwargs() call, then
        # only the target-row subset each step after (condition/reference
        # rows never change post-priming -- see forward_kwargs).
        self._x_buffer_primed = False
        self.x_buffer = torch.zeros(
            1, seq_len, MINIMAX_H3_VIDEO_ROW_WIDTH, dtype=torch.float32, device=device
        )
        self.audio_x_buffer = torch.zeros(
            1, seq_len, MINIMAX_H3_AUDIO_ROW_WIDTH, dtype=torch.float32, device=device
        )
        text_pos_dev = text_pos.to(device)
        ulysses_world_size, ulysses_rank = get_ulysses_ctx()
        ring_world_size, ring_rank = get_ring_ctx()
        # Combined SP-local rank/world size: the group coordinator lays out
        # ring as the outer (slower-varying) dimension and Ulysses as the
        # inner one (see set_seq_parallel_pg_by_sp_groups), so this matches
        # minimax_h3.py's row_start = ring_rank*ring_chunk_len +
        # ulysses_rank*local_seq_len exactly -- `_build_local_embedding_layout`
        # below needs the same combined rank, not just the Ulysses component.
        sp_world_size = ulysses_world_size * ring_world_size
        sp_rank = ring_rank * ulysses_world_size + ulysses_rank
        token_tags_host = token_tags.view(-1).to(dtype=torch.long)
        subblock_sparse_query_block_mask = (
            _minimax_h3_subblock_sparse_query_block_mask(
                video_query_indices.view(-1).to(dtype=torch.long),
                used_len=int(cu[1]),
            ).to(device)
            if video_query_indices is not None
            else None
        )
        local_seq_len = seq_len // sp_world_size
        local_row_start = sp_rank * local_seq_len
        local_row_stop = local_row_start + local_seq_len
        self.local_row_slice = slice(local_row_start, local_row_stop)
        self.block_token_tags = (
            token_tags_host[local_row_start:local_row_stop].clamp(min=0).to(device)
        )
        self.static_kwargs: dict[str, Any] = {
            # Cast the fp64 position grid on the host. MiniMaxH3Rope casts it to
            # fp32 as its first op anyway, so the values are identical; doing the
            # cast on CPU also avoids platforms that cannot execute fp64 on
            # device (e.g. Iluvatar CoreX returns zeros for device fp64).
            "img_position_ids": packed["img_position_ids"][None]
            .to(torch.float32)
            .to(device),
            "update_mask": self.update_mask_dev,
            "block_token_tags": self.block_token_tags,
            "skip_mask_out_condition": True,
            "prompt_embeds": text_embeddings.to(device),
            "img_pos_info": {"position_ids": self.img_pos_dev},
            "audio_pos_info": {"position_ids": self.audio_pos_dev},
            "text_pos_info": {"position_ids": text_pos_dev},
            "img_pos_for_infer_output_info": {"position_ids": self.img_target_seq_idx},
            "local_embedding_layout": _build_local_embedding_layout(
                seq_len=seq_len,
                text_pos=text_pos,
                img_pos=self.img_pos,
                audio_pos=self.audio_pos,
                world_size=sp_world_size,
                rank=sp_rank,
                device=device,
            ),
            "packed_seq_params": {
                "cu_seqlens_q": cu.to(device),
                "cu_seqlens_q_host": tuple(int(value) for value in cu.tolist()),
                "max_seqlen_q": int(cu[1]),
            },
            "refiner_packed_seq_params": {
                "cu_seqlens_q": torch.tensor(
                    [0, text_len, text_len], dtype=torch.int32, device=device
                ),
                "cu_seqlens_q_host": (0, text_len, text_len),
                "max_seqlen_q": text_len,
            },
        }
        if subblock_sparse_query_block_mask is not None:
            self.static_kwargs["subblock_sparse_query_block_mask"] = (
                subblock_sparse_query_block_mask
            )

    def forward_kwargs(
        self,
        *,
        video_rows: torch.Tensor,
        audio_rows: torch.Tensor,
        step_timesteps: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> dict[str, Any]:
        x = self.x_buffer
        audio_x = self.audio_x_buffer
        if not self._x_buffer_primed:
            # First step: condition/reference rows have just been pinned into
            # video_rows/audio_rows (see minimax_h3_denoise_loop) and never
            # change again, so this is the only step that needs the full
            # img/audio extent written into the persistent buffers.
            x[0].index_copy_(0, self.img_pos_dev, video_rows)
            audio_x[0].index_copy_(0, self.audio_pos_dev, audio_rows)
            self._x_buffer_primed = True
        else:
            # Later steps: only the target-row subset changed since the
            # buffers were primed; rewriting condition/reference rows again
            # would just copy the same bytes already sitting there.
            x[0].index_copy_(
                0, self.img_target_seq_idx, video_rows[self.video_target_slice]
            )
            audio_x[0].index_copy_(
                0, self.audio_target_seq_idx, audio_rows[self.audio_target_slice]
            )
        unique_timesteps, inverse_indices, block_combined_indices = step_timesteps
        return {
            **self.static_kwargs,
            "x": x,
            "audio_x": audio_x,
            "unique_timesteps": unique_timesteps,
            "inverse_indices": inverse_indices,
            "block_combined_indices": block_combined_indices,
        }

    def _expand_step_timesteps(
        self,
        *,
        t_video: float,
        t_audio: float,
        imgvid_cond_timestep: float,
        audio_ref_cond_timestep: float,
        inverse_indices_by_pattern: dict[tuple[int, ...], torch.Tensor],
        block_combined_by_pattern: dict[tuple[int, ...], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build step-local timestep and AdaLN index tensors.

        Packed-sequence timestep semantics: non-media rows (text and padding)
        inherit the current video timestep, condition rows pin their noise-aug
        timesteps. The row->timestep layout is step-static, so instead of
        materializing the full timestep tensor and paying a device-syncing
        torch.unique per step, the at-most-four candidate values are deduped
        in fp32 on the host — torch.unique on the candidate tensor keeps exact
        fp32 collision semantics — and inverse indices are index_fill'ed from
        the static position sets.
        """
        candidates: list[float] = []
        fill_groups: list[tuple[torch.Tensor, int]] = []
        base_slot = -1
        if self.n_video_timestep_rows > 0:
            base_slot = len(candidates)
            candidates.append(float(t_video))
        for seq_idx, value in (
            (self.img_cond_seq_idx, imgvid_cond_timestep),
            (self.audio_target_seq_idx, t_audio),
            (self.audio_ref_seq_idx, audio_ref_cond_timestep),
        ):
            if seq_idx.numel() > 0:
                fill_groups.append((seq_idx, len(candidates)))
                candidates.append(float(value))
        unique_cpu, slot_to_unique = torch.unique(
            torch.tensor(candidates, dtype=torch.float32),
            sorted=True,
            return_inverse=True,
        )
        device = self.img_pos_dev.device
        base_index = int(slot_to_unique[base_slot]) if base_slot >= 0 else 0
        pattern = tuple(slot_to_unique.tolist())
        inverse_indices = inverse_indices_by_pattern.get(pattern)
        if inverse_indices is None:
            inverse_indices = torch.full(
                (self.seq_len,), base_index, dtype=torch.long, device=device
            )
            for seq_idx, slot in fill_groups:
                inverse_indices.index_fill_(0, seq_idx, int(slot_to_unique[slot]))
            inverse_indices_by_pattern[pattern] = inverse_indices
        block_combined = block_combined_by_pattern.get(pattern)
        if block_combined is None:
            block_combined = torch.add(
                self.block_token_tags,
                inverse_indices[self.local_row_slice],
                alpha=MINIMAX_H3_ADALN_MODALITY_NUM,
            )
            block_combined_by_pattern[pattern] = block_combined
        return unique_cpu.to(device), inverse_indices, block_combined

    def prepare_timestep_plan(
        self,
        *,
        video_timesteps: list[float],
        audio_timesteps: list[float],
        imgvid_cond_noise_aug: float,
        audio_ref_cond_noise_aug: float,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Stage every step's packed timestep state before denoising."""
        if len(video_timesteps) != len(audio_timesteps):
            raise ValueError("video/audio timestep plans must have equal length")
        inverse_indices_by_pattern: dict[tuple[int, ...], torch.Tensor] = {}
        block_combined_by_pattern: dict[tuple[int, ...], torch.Tensor] = {}
        return [
            self._expand_step_timesteps(
                t_video=t_video,
                t_audio=t_audio,
                imgvid_cond_timestep=max(t_video, imgvid_cond_noise_aug),
                audio_ref_cond_timestep=max(t_audio, audio_ref_cond_noise_aug),
                inverse_indices_by_pattern=inverse_indices_by_pattern,
                block_combined_by_pattern=block_combined_by_pattern,
            )
            for t_video, t_audio in zip(video_timesteps, audio_timesteps)
        ]


def minimax_h3_denoise_loop(
    *,
    model: Any,
    model_forward: (
        Callable[[Any, dict[str, Any], int], tuple[torch.Tensor, torch.Tensor]] | None
    ) = None,
    positive: MiniMaxH3DenoiseBranch,
    initial_video_rows: torch.Tensor,
    initial_audio_rows: torch.Tensor,
    keyframe_cond_rows: torch.Tensor | None,
    audio_ref_rows: torch.Tensor | None = None,
    sigmas_video: list[float],
    sigmas_audio: list[float],
    device: torch.device,
    imgvid_cond_noise_aug_for_inference: float = MINIMAX_H3_IMGVID_COND_TIMESTEP,
    audio_cond_noise_aug_for_inference: float = MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    attn_metadata: AttentionMetadata | None = None,
    on_step: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
    step_profiler: Callable[[int], AbstractContextManager] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full denoise loop; returns final (video_rows, audio_rows).

    ``initial_video_rows`` covers all image rows of the positive layout. For a
    conditional task, pass ``keyframe_cond_rows`` and/or ``audio_ref_rows`` to
    pin those rows across every step. The model's raw positive velocity is the
    update signal; MiniMax H3 only supports cfg-distilled checkpoints.
    ``model_forward`` is the native-stage hook for residency/BCG runners and
    receives the zero-based loop step; the default keeps this helper
    independently testable with a plain callable.
    """
    if len(sigmas_video) != len(sigmas_audio):
        raise ValueError("video/audio sigma schedules must have equal length")
    if len(sigmas_video) < 2:
        raise ValueError("sigma schedules need at least 2 entries")
    n_cond = positive.video_target_start
    if keyframe_cond_rows is None:
        if n_cond != 0:
            raise ValueError(
                f"layout has {n_cond} cond rows but keyframe_cond_rows is None"
            )
    else:
        if int(keyframe_cond_rows.shape[0]) != n_cond:
            raise ValueError(
                f"keyframe_cond_rows {int(keyframe_cond_rows.shape[0])} != "
                f"layout cond rows {n_cond}"
            )
    video_rows = initial_video_rows.to(device=device, dtype=torch.float32, copy=True)
    audio_rows = initial_audio_rows.to(device=device, dtype=torch.float32, copy=True)
    if int(video_rows.shape[0]) != int(positive.img_pos.shape[0]):
        raise ValueError(
            f"initial video rows {int(video_rows.shape[0])} != positive layout "
            f"rows {int(positive.img_pos.shape[0])}"
        )
    if int(audio_rows.shape[0]) != int(positive.audio_pos.shape[0]):
        raise ValueError(
            f"initial audio rows {int(audio_rows.shape[0])} != positive layout "
            f"rows {int(positive.audio_pos.shape[0])}"
        )
    n_audio_ref = positive.audio_target_start
    if audio_ref_rows is None:
        if n_audio_ref != 0:
            raise ValueError(
                f"layout has {n_audio_ref} audio ref rows but audio_ref_rows is None"
            )
        audio_anchor = None
    else:
        if int(audio_ref_rows.shape[0]) != n_audio_ref:
            raise ValueError(
                f"audio_ref_rows {int(audio_ref_rows.shape[0])} != layout "
                f"audio ref rows {n_audio_ref}"
            )
        audio_anchor = audio_ref_rows.to(device=device, dtype=torch.float32)
    cond_anchor = (
        keyframe_cond_rows.to(device=device, dtype=torch.float32)
        if keyframe_cond_rows is not None
        else None
    )
    if cond_anchor is not None:
        video_rows.index_copy_(0, positive.cond_row_idx, cond_anchor)
    if audio_anchor is not None:
        audio_rows.index_copy_(0, positive.audio_ref_row_idx, audio_anchor)

    num_steps = len(sigmas_video) - 1
    video_target_slice = positive.video_target_slice
    audio_target_slice = positive.audio_target_slice
    video_timesteps = [1.0 - sigma for sigma in sigmas_video[:-1]]
    audio_timesteps = [1.0 - sigma for sigma in sigmas_audio[:-1]]
    # One H2D copy per schedule, preserving the previous Python-float
    # subtraction followed by fp32 conversion.
    video_step_t = torch.tensor(video_timesteps, dtype=torch.float32, device=device)
    audio_step_t = torch.tensor(audio_timesteps, dtype=torch.float32, device=device)
    timestep_plan = positive.prepare_timestep_plan(
        video_timesteps=video_timesteps,
        audio_timesteps=audio_timesteps,
        imgvid_cond_noise_aug=float(imgvid_cond_noise_aug_for_inference),
        audio_ref_cond_noise_aug=float(audio_cond_noise_aug_for_inference),
    )
    # Every step's timesteps are settled by now. Rebuilding AdaLN reads all
    # 24.2 GiB of adaln_proj whatever is missing, so fill the whole request in
    # one pass here instead of topping up step by step inside the loop.
    model.prepare_adaln_plans([entry[0] for entry in timestep_plan])

    # match the scheduler's device-fp32 math once, then reuse one denoised
    # scratch per modality instead of allocating intermediates every step
    video_sigmas = torch.tensor(sigmas_video, dtype=torch.float32, device=device)
    audio_sigmas = torch.tensor(sigmas_audio, dtype=torch.float32, device=device)
    video_sigma_ratios = video_sigmas[1:] / video_sigmas[:-1]
    audio_sigma_ratios = audio_sigmas[1:] / audio_sigmas[:-1]
    video_sigma_t = 1.0 - video_step_t
    audio_sigma_t = 1.0 - audio_step_t
    video_one_minus_sigma_ratios = 1.0 - video_sigma_ratios
    audio_one_minus_sigma_ratios = 1.0 - audio_sigma_ratios
    video_denoised_scratch = torch.empty_like(video_rows[video_target_slice])
    audio_denoised_scratch = torch.empty_like(audio_rows[audio_target_slice])
    for step in range(num_steps):
        step_cm = step_profiler(step) if step_profiler is not None else nullcontext()
        with step_cm:
            s_v = sigmas_video[step]
            s_a = sigmas_audio[step]

            fk = positive.forward_kwargs(
                video_rows=video_rows,
                audio_rows=audio_rows,
                step_timesteps=timestep_plan[step],
            )
            if attn_metadata is not None:
                attn_metadata.current_timestep = step
            if model_forward is None and attn_metadata is not None:
                forward_cm: AbstractContextManager = set_forward_context(
                    current_timestep=step,
                    attn_metadata=attn_metadata,
                )
            else:
                forward_cm = nullcontext()
            with forward_cm, torch.inference_mode():
                if model_forward is None:
                    v_video, v_audio = model(**fk)
                else:
                    v_video, v_audio = model_forward(model, fk, step)
                # The model outputs are inference tensors. Keep their disposable
                # fp32 velocity updates in the same context so ``out=velocity``
                # can reuse the output storage without an extra clone.
                mv_video_t = v_video.float()
                mv_audio_t = v_audio[audio_target_slice].float()

                video_target = video_rows[video_target_slice]
                _minimax_h3_update_target_rows_(
                    video_target,
                    mv_video_t,
                    sigma_t=video_sigma_t[step],
                    sigma_curr=s_v,
                    sigma_ratio=video_sigma_ratios[step],
                    one_minus_sigma_ratio=video_one_minus_sigma_ratios[step],
                    denoised_scratch=video_denoised_scratch,
                )

                audio_target = audio_rows[audio_target_slice]
                _minimax_h3_update_target_rows_(
                    audio_target,
                    mv_audio_t,
                    sigma_t=audio_sigma_t[step],
                    sigma_curr=s_a,
                    sigma_ratio=audio_sigma_ratios[step],
                    one_minus_sigma_ratio=audio_one_minus_sigma_ratios[step],
                    denoised_scratch=audio_denoised_scratch,
                )
            if on_step is not None:
                on_step(step, video_rows, audio_rows)

    return video_rows, audio_rows


__all__ = [
    "MINIMAX_H3_AUDIO_REF_COND_TIMESTEP",
    "MINIMAX_H3_AUDIO_ROW_WIDTH",
    "MINIMAX_H3_IMGVID_COND_TIMESTEP",
    "MINIMAX_H3_VIDEO_ROW_WIDTH",
    "MiniMaxH3DenoiseBranch",
    "minimax_h3_denoise_loop",
]
