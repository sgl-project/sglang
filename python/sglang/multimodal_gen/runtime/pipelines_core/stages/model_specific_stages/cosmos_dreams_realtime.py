# SPDX-License-Identifier: Apache-2.0
"""Cosmos-Dreams realtime (tick) session stages.

One realtime request generates one autoregressive block. The first tick
commits the conditioning image as latent frame 0 and denoises the first
``chunk_size`` frames; every later tick denoises the next block against the
session's committed K/V history; the causal Wan VAE decodes only the new
latent frames. Actions arrive per tick as raw rows for the pixel steps of the
block through ``condition_inputs[ACTION_ROWS_CONDITION]``. Without a session
the stages behave exactly like the offline pipeline.
"""

from __future__ import annotations

from typing import Any

import torch

from sglang.multimodal_gen.configs.models.dits.cosmos_dreams import (
    CosmosDreamsManifest,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.models.dits.cosmos_dreams import KVPair
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3 import (
    Cosmos3DecodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos_dreams import (
    EXTRA_DOMAIN_ID,
    EXTRA_GEOMETRY,
    EXTRA_TEXT_IDS,
    EXTRA_TEXT_MASK,
    CosmosDreamsPrepareStage,
    CosmosDreamsRolloutStage,
    PreparedConditioning,
    _RolloutContext,
    iter_ar_chunk_ranges,
    load_action_rows,
    normalize_action_rows,
    pad_action_rows,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime.vae import (
    RealtimeVAEDecodeState,
)
from sglang.multimodal_gen.runtime.realtime.session import BaseRealtimeState
from sglang.multimodal_gen.runtime.server_args import ServerArgs

# Raw action rows for the pixel steps of one block, in condition_inputs.
ACTION_ROWS_CONDITION = "action_rows"
EXTRA_TICK_ACTION_ROWS = "cosmos_dreams_tick_action_rows"


class CosmosDreamsSessionState(BaseRealtimeState):
    """Rollout carry-over between the ticks of one realtime session."""

    def __init__(self) -> None:
        super().__init__()
        self.conditioning: PreparedConditioning | None = None
        self.seed: int | None = None
        self.fps: float | None = None
        self.context: _RolloutContext | None = None
        self.history: list[KVPair] | None = None
        self.next_frame: int = 0
        self.pending_image_latent: torch.Tensor | None = None

    def begin(
        self, conditioning: PreparedConditioning, *, seed: int, fps: float
    ) -> None:
        self.dispose()
        self.conditioning = conditioning
        self.seed = seed
        self.fps = fps
        self.pending_image_latent = conditioning.image_latent

    def dispose(self) -> None:
        self.conditioning = None
        self.seed = None
        self.fps = None
        self.context = None
        self.history = None
        self.next_frame = 0
        self.pending_image_latent = None


def block_frames(
    realtime_chunk_size: int | None, manifest: CosmosDreamsManifest
) -> int:
    """Latent frames denoised by one tick; a positive multiple of the trained chunk."""
    frames = (
        manifest.chunk_size if realtime_chunk_size is None else int(realtime_chunk_size)
    )
    if frames <= 0 or frames % manifest.chunk_size:
        raise ValueError(
            "Cosmos-Dreams realtime chunk size must be a positive multiple of the "
            f"trained chunk size {manifest.chunk_size}, got {realtime_chunk_size}."
        )
    return frames


def tick_action_rows(
    action_rows: Any,
    *,
    manifest: CosmosDreamsManifest,
    embodiment: str,
    frames: int,
) -> torch.Tensor | None:
    """Normalized, zero-padded ``[frames * A, D]`` rows for one block.

    Row ``i`` drives pixel step ``i`` of the block. ``None`` conditions every
    frame of the block on the null action.
    """
    if action_rows is None:
        return None
    contract = manifest.action_contract.embodiments[embodiment]
    rows = load_action_rows(action_rows)
    if rows.shape[-1] != contract.raw_action_dim:
        raise ValueError(
            f"Cosmos-Dreams embodiment {embodiment!r} requires raw action dimension "
            f"{contract.raw_action_dim}, got {rows.shape[-1]}."
        )
    expected = frames * manifest.action_tokens_per_frame
    if rows.shape[0] != expected:
        raise ValueError(
            f"Cosmos-Dreams tick needs exactly {expected} action rows for {frames} latent "
            f"frames ({manifest.action_tokens_per_frame} pixel steps each), got {rows.shape[0]}."
        )
    rows = normalize_action_rows(rows, contract.normalizer.transform)
    return pad_action_rows(rows, manifest.max_action_dim)


def block_actions(
    rows: torch.Tensor | None,
    *,
    frames: int,
    action_tokens_per_frame: int,
    model_action_dim: int,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """``[1, frames * A, D]`` action block and the block-local null-action frames."""
    if rows is None:
        zeros = torch.zeros(1, frames * action_tokens_per_frame, model_action_dim)
        return zeros, tuple(range(frames))
    return rows.unsqueeze(0), ()


class CosmosDreamsRealtimePrepareStage(CosmosDreamsPrepareStage):
    """Prepare the session on the first tick and restore it on later ticks."""

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if batch.session is None:
            return super().forward(batch, server_args)
        device = get_local_torch_device()
        state = batch.session.get_or_create_state(CosmosDreamsSessionState)
        if batch.block_idx == 0:
            if batch.preprocessed_image is None:
                raise ValueError(
                    "Cosmos-Dreams realtime sessions start from a conditioning image; "
                    "send first_frame with the init message."
                )
            if not isinstance(batch.seed, int):
                raise ValueError(
                    f"Cosmos-Dreams requires a single integer seed, got {batch.seed!r}."
                )
            state.begin(
                self._prepare_conditioning(batch, device),
                seed=batch.seed,
                fps=float(batch.fps),
            )
        elif state.conditioning is None:
            raise ValueError(
                f"Cosmos-Dreams realtime tick {batch.block_idx} arrived without session state."
            )
        prepared = state.conditioning
        geometry = prepared.geometry
        batch.extra[EXTRA_GEOMETRY] = geometry
        batch.extra[EXTRA_TEXT_IDS] = prepared.text_ids
        batch.extra[EXTRA_TEXT_MASK] = prepared.text_mask
        batch.extra[EXTRA_DOMAIN_ID] = prepared.domain_id
        batch.height, batch.width = geometry.height, geometry.width
        # Frame 0 is committed from the session, never re-encoded per tick.
        batch.image_latent = None
        frames = block_frames(batch.realtime_chunk_size, self.manifest)
        rows = tick_action_rows(
            batch.condition_inputs.get(ACTION_ROWS_CONDITION),
            manifest=self.manifest,
            embodiment=prepared.embodiment,
            frames=frames,
        )
        batch.extra[EXTRA_TICK_ACTION_ROWS] = (
            None if rows is None else rows.to(device=device)
        )
        self.log_info(
            f"Cosmos-Dreams tick {batch.block_idx}: latent frames "
            f"[{state.next_frame}, {state.next_frame + frames + (1 if state.next_frame == 0 else 0)}), "
            f"{geometry.height}x{geometry.width}, actions={'yes' if rows is not None else 'null'}"
        )
        return batch


class CosmosDreamsRealtimeRolloutStage(CosmosDreamsRolloutStage):
    """Denoise one block per tick against the session's committed history."""

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if batch.session is None:
            return super().forward(batch, server_args)
        with self.use_declared_component(
            component_name="transformer", module=self.transformer, phase="denoise"
        ):
            with torch.no_grad():
                batch.latents = self._tick(batch)
        return batch

    def _session_context(self, state: CosmosDreamsSessionState) -> _RolloutContext:
        if state.context is not None:
            return state.context
        prepared = state.conditioning
        if prepared is None or state.fps is None:
            raise ValueError("Cosmos-Dreams realtime tick has no prepared session.")
        device = get_local_torch_device()
        text_kv, _ = self.transformer.encode_und_kv(
            prepared.text_ids, prepared.text_mask
        )
        state.context = _RolloutContext(
            text_kv=text_kv,
            fps=state.fps,
            domain_ids=torch.tensor(
                [prepared.domain_id], device=device, dtype=torch.long
            ),
            tokens_per_frame=prepared.geometry.tokens_per_frame(
                self.manifest.action_tokens_per_frame
            ),
            latent_channels=self.transformer.latent_channel,
            geometry=prepared.geometry,
            device=device,
            dtype=torch.bfloat16,
        )
        return state.context

    def _tick(self, batch: Req) -> torch.Tensor:
        manifest = self.manifest
        state = batch.session.get_or_create_state(CosmosDreamsSessionState)
        context = self._session_context(state)
        frames = block_frames(batch.realtime_chunk_size, manifest)
        rows: torch.Tensor | None = batch.extra[EXTRA_TICK_ACTION_ROWS]
        action_count = manifest.action_tokens_per_frame
        outputs: list[torch.Tensor] = []
        if state.next_frame == 0:
            outputs.append(self._commit_first_frame(state, context))
        block_start = state.next_frame
        if (block_start - 1) % manifest.chunk_size:
            raise ValueError(
                f"Cosmos-Dreams session frontier {block_start} is off the chunk partition."
            )
        seed = state.seed
        assert seed is not None
        for chunk_start, chunk_end in iter_ar_chunk_ranges(
            block_start, block_start + frames, manifest.chunk_size
        ):
            chunk_rows = None
            if rows is not None:
                first = (chunk_start - block_start) * action_count
                chunk_rows = rows[
                    first : first + (chunk_end - chunk_start) * action_count
                ]
            action, null_indexes = block_actions(
                chunk_rows,
                frames=chunk_end - chunk_start,
                action_tokens_per_frame=action_count,
                model_action_dim=manifest.max_action_dim,
            )
            action = action.to(device=context.device, dtype=context.dtype)
            generator = torch.Generator(device=context.device).manual_seed(
                seed + chunk_start
            )
            noise = torch.randn(
                (
                    1,
                    context.latent_channels,
                    chunk_end - chunk_start,
                    context.geometry.latent_height,
                    context.geometry.latent_width,
                ),
                generator=generator,
                device=context.device,
                dtype=context.dtype,
            )
            clean_chunk = self._denoise_chunk(
                context,
                noise,
                seed=seed,
                history=state.history,
                frame_start=chunk_start,
                action=action,
                null_indexes=null_indexes,
            )
            # Every frame is committed: the next tick reads all of them.
            for local_idx, frame_idx in enumerate(range(chunk_start, chunk_end)):
                state.history = self._commit_clean_frame(
                    context,
                    state.history,
                    clean_chunk[:, :, local_idx : local_idx + 1],
                    frame_idx=frame_idx,
                    action=action[
                        :, local_idx * action_count : (local_idx + 1) * action_count
                    ],
                    null_action=local_idx in null_indexes,
                )
            outputs.append(clean_chunk)
            state.next_frame = chunk_end
        self.log_info(
            f"Committed latent frames [{block_start}, {state.next_frame}) of the session"
        )
        return torch.cat(outputs, dim=2)

    def _commit_first_frame(
        self, state: CosmosDreamsSessionState, context: _RolloutContext
    ) -> torch.Tensor:
        latent = state.pending_image_latent
        if latent is None:
            raise ValueError(
                "Cosmos-Dreams session has no conditioning latent for frame 0."
            )
        latent = latent.to(device=context.device, dtype=context.dtype)
        null_action = torch.zeros(
            1,
            self.manifest.action_tokens_per_frame,
            self.manifest.max_action_dim,
            device=context.device,
            dtype=context.dtype,
        )
        state.history = self._commit_clean_frame(
            context,
            state.history,
            latent,
            frame_idx=0,
            action=null_action,
            null_action=True,
        )
        state.pending_image_latent = None
        state.next_frame = 1
        return latent


class CosmosDreamsCausalDecodingStage(Cosmos3DecodingStage):
    """Decode only the tick's latent frames with the persistent causal VAE cache."""

    def forward(self, batch: Req, server_args: ServerArgs):
        if batch.session is None:
            return super().forward(batch, server_args)
        decode_state = batch.session.get_or_create_state(RealtimeVAEDecodeState)
        decode_state.reset_causal_decode_state = self.vae.reset_causal_decode_state
        if batch.block_idx == 0:
            self.vae.reset_causal_decode_state()
        with self.use_declared_component(component_name="vae", module=self.vae):
            with torch.no_grad():
                video = self.vae.causal_decode(self._denormalize_latents(batch.latents))
        output = self._postprocess_tensor(video)
        self.log_debug("Decoded realtime chunk: %s", tuple(output.shape))
        return OutputBatch(output=output, metrics=batch.metrics)
