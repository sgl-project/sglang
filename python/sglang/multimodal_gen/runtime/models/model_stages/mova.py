# SPDX-License-Identifier: Apache-2.0
"""
MoVA-specific pipeline stages.

Sequence Parallelism (SP) Support:
- Video latents are sharded along the sequence dimension (T*H*W) after patchify
- Audio latents are sharded along the sequence dimension (L) after patchify
- USPAttention handles all-to-all communication internally
- Latents are gathered before unpatchify to restore full sequence
"""

from __future__ import annotations

from typing import Iterable

import torch
from diffusers.utils.torch_utils import randn_tensor
from tqdm.auto import tqdm

from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_world_group,
)
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    cfg_model_parallel_all_reduce,
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_classifier_free_guidance_rank,
    get_sp_parallel_rank,
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context

# Both audio and video DiT use the same sinusoidal_embedding_1d function
# Import from mova_video_dit where it's defined (mova_audio_dit re-exports it)
from sglang.multimodal_gen.runtime.models.dits.mova_video_dit import (
    sinusoidal_embedding_1d,
)

# Create aliases for backward compatibility
video_sinusoidal_embedding_1d = sinusoidal_embedding_1d
audio_sinusoidal_embedding_1d = sinusoidal_embedding_1d
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import (
    _ensure_tensor_decode_output,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs, get_global_server_args
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class MovaLatentPreparationStage(PipelineStage):
    """Prepare video/audio noise latents for MoVA."""

    def __init__(self, audio_vae, require_vae_embedding: bool = True) -> None:
        super().__init__()
        self.audio_vae = audio_vae
        self.require_vae_embedding = require_vae_embedding

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        batch_size = batch.batch_size
        num_frames = batch.num_frames
        if num_frames is None:
            raise ValueError("num_frames is required for MoVA")

        audio_num_samples = int(self.audio_vae.sample_rate * num_frames / batch.fps)
        batch.audio_num_samples = audio_num_samples

        video_shape = server_args.pipeline_config.prepare_latent_shape(
            batch, batch_size, num_frames
        )
        audio_shape = server_args.pipeline_config.prepare_audio_latent_shape(
            batch_size, audio_num_samples, self.audio_vae
        )

        device = get_local_torch_device()
        generator = batch.generator
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        dit_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision]
        batch.latents = randn_tensor(
            video_shape, generator=generator, device=device, dtype=dit_dtype
        )
        batch.audio_latents = randn_tensor(
            audio_shape, generator=generator, device=device, dtype=dit_dtype
        )

        if batch.image_latent is not None:
            batch.y = batch.image_latent.to(device=device, dtype=dit_dtype)
        elif self.require_vae_embedding:
            raise ValueError("MoVA requires reference image latents for denoising")
        return batch


class MovaTimestepPreparationStage(PipelineStage):
    """Prepare paired timesteps for MoVA."""

    def __init__(self, scheduler) -> None:
        super().__init__()
        self.scheduler = scheduler

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        self.scheduler.set_timesteps(
            batch.num_inference_steps,
            denoising_strength=1.0,
            shift=getattr(batch, "sigma_shift", self.scheduler.shift),
        )
        self.scheduler.set_pair_postprocess_by_name(
            "dual_sigma_shift",
            visual_shift=getattr(batch, "visual_shift", 5.0),
            audio_shift=getattr(batch, "audio_shift", 5.0),
        )
        paired = self.scheduler.get_pairs()
        batch.paired_timesteps = paired
        batch.timesteps = paired
        return batch


class MovaDenoisingStage(PipelineStage):
    """Run MoVA dual-tower denoising loop."""

    def __init__(self, video_dit, video_dit2, audio_dit, dual_tower_bridge, scheduler):
        super().__init__()
        self.video_dit = video_dit
        self.video_dit2 = video_dit2
        self.audio_dit = audio_dit
        self.dual_tower_bridge = dual_tower_bridge
        self.scheduler = scheduler

    @property
    def parallelism_type(self) -> StageParallelismType:
        if get_global_server_args().enable_cfg_parallel:
            return StageParallelismType.CFG_PARALLEL
        return StageParallelismType.REPLICATED

    def _predict(
        self,
        visual_dit,
        visual_latents,
        audio_latents,
        y,
        context,
        timestep,
        audio_timestep,
        video_fps,
    ):
        # Set forward context for distributed attention (USPAttention)
        with set_forward_context(
            current_timestep=0,  # Not used by MoVA but required by context
            attn_metadata=None,  # MoVA doesn't use special attention metadata
        ):
            return self.inference_single_step(
                visual_dit=visual_dit,
                visual_latents=visual_latents,
                audio_latents=audio_latents,
                y=y,
                context=context,
                timestep=timestep,
                audio_timestep=audio_timestep,
                video_fps=video_fps,
            )

    def _cfg_combine(self, pos, neg, guidance_scale, cfg_rank, enable_cfg_parallel):
        if not enable_cfg_parallel:
            return neg + guidance_scale * (pos - neg)
        if cfg_rank == 0:
            partial = guidance_scale * pos
        else:
            partial = (1 - guidance_scale) * neg
        return cfg_model_parallel_all_reduce(partial)

    def progress_bar(
        self, iterable: Iterable | None = None, total: int | None = None
    ) -> tqdm:
        """
        Create a progress bar for the denoising process.
        """
        local_rank = get_world_group().local_rank
        disable = local_rank != 0
        return tqdm(iterable=iterable, total=total, disable=disable)

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        paired_timesteps = batch.paired_timesteps
        if paired_timesteps is None:
            raise ValueError("paired_timesteps must be set for MoVA")

        y = batch.y if batch.y is not None else batch.image_latent
        if getattr(self.video_dit, "require_vae_embedding", False) and y is None:
            raise ValueError("MoVA requires reference image latents for denoising")

        cur_visual_dit = self.video_dit
        switched = False
        boundary_ratio = server_args.pipeline_config.boundary_ratio
        total_steps = paired_timesteps.shape[0]
        cfg_rank = get_classifier_free_guidance_rank()
        enable_cfg_parallel = server_args.enable_cfg_parallel

        with self.progress_bar(total=total_steps) as progress_bar:
            for idx_step in range(total_steps):
                pair_t = paired_timesteps[idx_step]
                if pair_t.shape == (2,):
                    timestep, audio_timestep = pair_t
                else:
                    timestep = pair_t
                    audio_timestep = pair_t

                if (
                    not switched
                    and boundary_ratio is not None
                    and self.video_dit2 is not None
                    and timestep.item()
                    < boundary_ratio * self.scheduler.num_train_timesteps
                ):
                    cur_visual_dit = self.video_dit2
                    switched = True

                timestep = timestep.unsqueeze(0).to(device=get_local_torch_device())
                audio_timestep = audio_timestep.unsqueeze(0).to(
                    device=get_local_torch_device()
                )

                if not batch.do_classifier_free_guidance:
                    visual_noise_pred, audio_noise_pred = self._predict(
                        cur_visual_dit,
                        batch.latents,
                        batch.audio_latents,
                        y,
                        batch.prompt_embeds[0],
                        timestep,
                        audio_timestep,
                        batch.fps,
                    )
                else:
                    if enable_cfg_parallel and cfg_rank == 0:
                        pos = self._predict(
                            cur_visual_dit,
                            batch.latents,
                            batch.audio_latents,
                            y,
                            batch.prompt_embeds[0],
                            timestep,
                            audio_timestep,
                            batch.fps,
                        )
                        neg = (None, None)
                    elif enable_cfg_parallel and cfg_rank != 0:
                        pos = (None, None)
                        neg = self._predict(
                            cur_visual_dit,
                            batch.latents,
                            batch.audio_latents,
                            y,
                            batch.negative_prompt_embeds[0],
                            timestep,
                            audio_timestep,
                            batch.fps,
                        )
                    else:
                        pos = self._predict(
                            cur_visual_dit,
                            batch.latents,
                            batch.audio_latents,
                            y,
                            batch.prompt_embeds[0],
                            timestep,
                            audio_timestep,
                            batch.fps,
                        )
                        neg = self._predict(
                            cur_visual_dit,
                            batch.latents,
                            batch.audio_latents,
                            y,
                            batch.negative_prompt_embeds[0],
                            timestep,
                            audio_timestep,
                            batch.fps,
                        )

                    if enable_cfg_parallel:
                        visual_noise_pred = self._cfg_combine(
                            pos[0] if pos[0] is not None else neg[0],
                            neg[0] if neg[0] is not None else pos[0],
                            batch.guidance_scale,
                            cfg_rank,
                            enable_cfg_parallel,
                        )
                        audio_noise_pred = self._cfg_combine(
                            pos[1] if pos[1] is not None else neg[1],
                            neg[1] if neg[1] is not None else pos[1],
                            batch.guidance_scale,
                            cfg_rank,
                            enable_cfg_parallel,
                        )
                    else:
                        visual_noise_pred = self._cfg_combine(
                            pos[0],
                            neg[0],
                            batch.guidance_scale,
                            cfg_rank,
                            enable_cfg_parallel,
                        )
                        audio_noise_pred = self._cfg_combine(
                            pos[1],
                            neg[1],
                            batch.guidance_scale,
                            cfg_rank,
                            enable_cfg_parallel,
                        )

                next_timestep = (
                    paired_timesteps[idx_step + 1, 0]
                    if idx_step + 1 < total_steps
                    else None
                )
                next_audio_timestep = (
                    paired_timesteps[idx_step + 1, 1]
                    if idx_step + 1 < total_steps
                    else None
                )
                batch.latents = self.scheduler.step_from_to(
                    visual_noise_pred, timestep, next_timestep, batch.latents
                )
                batch.audio_latents = self.scheduler.step_from_to(
                    audio_noise_pred,
                    audio_timestep,
                    next_audio_timestep,
                    batch.audio_latents,
                )

        for dit in filter(None, [self.video_dit, self.video_dit2, self.audio_dit]):
            if isinstance(dit, OffloadableDiTMixin):
                dit.prepare_for_next_denoise()

        return batch

    def _shard_sequence_for_sp(
        self, x: torch.Tensor, dim: int = 1
    ) -> tuple[torch.Tensor, int]:
        """
        Shard tensor along sequence dimension for Sequence Parallelism.

        Args:
            x: Input tensor
            dim: Dimension to shard along

        Returns:
            (sharded_tensor, pad_len)
        """
        sp_size = get_sp_world_size()
        if sp_size <= 1:
            return x, 0

        sp_rank = get_sp_parallel_rank()
        seq_len = x.shape[dim]

        # Pad if needed
        pad_len = (sp_size - (seq_len % sp_size)) % sp_size
        if pad_len > 0:
            pad_shape = list(x.shape)
            pad_shape[dim] = pad_len
            pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=dim)

        # Shard
        chunk_size = x.shape[dim] // sp_size
        start = sp_rank * chunk_size
        end = start + chunk_size
        idx = [slice(None)] * x.dim()
        idx[dim] = slice(start, end)
        return x[tuple(idx)], pad_len

    def _gather_sequence_from_sp(
        self, x: torch.Tensor, pad_len: int, dim: int = 1
    ) -> torch.Tensor:
        """
        Gather tensor along sequence dimension after Sequence Parallelism.

        Args:
            x: Sharded tensor
            pad_len: Padding length that was added during sharding
            dim: Dimension to gather along

        Returns:
            Gathered tensor with padding removed
        """
        sp_size = get_sp_world_size()
        if sp_size <= 1:
            return x

        gathered = sequence_model_parallel_all_gather(x, dim=dim)
        if pad_len > 0:
            idx = [slice(None)] * gathered.dim()
            idx[dim] = slice(0, gathered.shape[dim] - pad_len)
            gathered = gathered[tuple(idx)]
        return gathered

    def inference_single_step(
        self,
        visual_dit,
        visual_latents: torch.Tensor,
        audio_latents: torch.Tensor,
        y,
        context: torch.Tensor,
        timestep: torch.Tensor,
        audio_timestep: torch.Tensor,
        video_fps: float,
    ):
        """
        Single inference step for MoVA dual-tower denoising.

        Supports Sequence Parallelism (SP):
        - After patchify, sequences are sharded across SP ranks
        - USPAttention handles distributed attention communication
        - Before unpatchify, sequences are gathered back
        """
        model_dtype = visual_dit.time_embedding.fc_in.weight.dtype
        device = visual_latents.device
        sp_size = get_sp_world_size()

        visual_context = context.to(device=device, dtype=model_dtype)
        audio_context = context.to(device=device, dtype=model_dtype)
        with torch.autocast(
            device_type=current_platform.device_type, dtype=torch.float32
        ):
            visual_t = visual_dit.time_embedding(
                video_sinusoidal_embedding_1d(visual_dit.freq_dim, timestep)
            )
            visual_t_mod, _ = visual_dit.time_projection(visual_t)
            visual_t_mod = visual_t_mod.unflatten(1, (6, visual_dit.dim))

            audio_t = self.audio_dit.time_embedding(
                audio_sinusoidal_embedding_1d(self.audio_dit.freq_dim, audio_timestep)
            )
            audio_t_mod, _ = self.audio_dit.time_projection(audio_t)
            audio_t_mod = audio_t_mod.unflatten(1, (6, self.audio_dit.dim))

        visual_t = visual_t.to(model_dtype)
        visual_t_mod = visual_t_mod.to(model_dtype)
        audio_t = audio_t.to(model_dtype)
        audio_t_mod = audio_t_mod.to(model_dtype)

        visual_context_emb = visual_dit.text_embedding(visual_context)
        audio_context_emb = self.audio_dit.text_embedding(audio_context)

        visual_x = visual_latents.to(model_dtype)
        audio_x = audio_latents.to(model_dtype)

        if getattr(visual_dit, "require_vae_embedding", False):
            visual_x = torch.cat([visual_x, y], dim=1)

        # Patchify visual latents
        visual_x, (t, h, w) = visual_dit.patchify(visual_x)
        grid_size = (t, h, w)
        full_visual_seq_len = t * h * w

        # Build visual freqs for full sequence
        visual_dit._init_freqs()
        visual_freqs = tuple(freq.to(visual_x.device) for freq in visual_dit.freqs)
        visual_freqs = (
            torch.cat(
                [
                    visual_freqs[0][:t].view(t, 1, 1, -1).expand(t, h, w, -1),
                    visual_freqs[1][:h].view(1, h, 1, -1).expand(t, h, w, -1),
                    visual_freqs[2][:w].view(1, 1, w, -1).expand(t, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(full_visual_seq_len, 1, -1)
            .to(visual_x.device)
        )

        # Patchify audio latents
        audio_x, (f,) = self.audio_dit.patchify(audio_x, None)
        full_audio_seq_len = f

        # Build audio freqs for full sequence
        self.audio_dit._init_freqs()
        audio_freqs = (
            torch.cat(
                [
                    self.audio_dit.freqs[0][:f].view(f, -1).expand(f, -1),
                    self.audio_dit.freqs[1][:f].view(f, -1).expand(f, -1),
                    self.audio_dit.freqs[2][:f].view(f, -1).expand(f, -1),
                ],
                dim=-1,
            )
            .reshape(full_audio_seq_len, 1, -1)
            .to(audio_x.device)
        )

        # Shard sequences for SP
        visual_x, visual_pad_len = self._shard_sequence_for_sp(visual_x, dim=1)
        audio_x, audio_pad_len = self._shard_sequence_for_sp(audio_x, dim=1)

        # Shard freqs to match local sequence length
        visual_freqs, _ = self._shard_sequence_for_sp(visual_freqs, dim=0)
        audio_freqs, _ = self._shard_sequence_for_sp(audio_freqs, dim=0)

        # Forward through dual-tower DiT
        visual_x, audio_x = self.forward_dual_tower_dit(
            visual_dit=visual_dit,
            visual_x=visual_x,
            audio_x=audio_x,
            visual_context=visual_context_emb,
            audio_context=audio_context_emb,
            visual_t_mod=visual_t_mod,
            audio_t_mod=audio_t_mod,
            visual_freqs=visual_freqs,
            audio_freqs=audio_freqs,
            grid_size=grid_size,
            video_fps=video_fps,
            full_visual_seq_len=full_visual_seq_len,
            full_audio_seq_len=full_audio_seq_len,
        )

        # Gather sequences back from SP before head/unpatchify
        visual_x = self._gather_sequence_from_sp(visual_x, visual_pad_len, dim=1)
        audio_x = self._gather_sequence_from_sp(audio_x, audio_pad_len, dim=1)

        visual_output = visual_dit.head(visual_x, visual_t)
        visual_output = visual_dit.unpatchify(visual_output, grid_size)

        audio_output = self.audio_dit.head(audio_x, audio_t)
        audio_output = self.audio_dit.unpatchify(audio_output, (f,))

        return visual_output.float(), audio_output.float()

    def forward_dual_tower_dit(
        self,
        visual_dit,
        visual_x: torch.Tensor,
        audio_x: torch.Tensor,
        visual_context: torch.Tensor,
        audio_context: torch.Tensor,
        visual_t_mod: torch.Tensor,
        audio_t_mod: torch.Tensor,
        visual_freqs: torch.Tensor,
        audio_freqs: torch.Tensor,
        grid_size: tuple[int, int, int],
        video_fps: float,
        full_visual_seq_len: int,
        full_audio_seq_len: int,
        condition_scale: float | None = 1.0,
        a2v_condition_scale: float | None = None,
        v2a_condition_scale: float | None = None,
    ):
        """
        Forward pass through dual-tower DiT with cross-modal interaction.

        Sequence Parallelism (SP) Support:
        - visual_x and audio_x are already sharded along sequence dimension
        - visual_freqs and audio_freqs match the local sequence length
        - USPAttention in self-attention handles distributed communication
        - LocalAttention in cross-attention operates on local sequence vs replicated context
        - Cross-modal attention (dual_tower_bridge) uses LocalAttention (no SP communication)

        Args:
            full_visual_seq_len: Full visual sequence length before SP sharding
            full_audio_seq_len: Full audio sequence length before SP sharding
        """
        min_layers = min(len(visual_dit.blocks), len(self.audio_dit.blocks))
        visual_layers = len(visual_dit.blocks)
        sp_size = get_sp_world_size()

        # Build RoPE frequencies for cross-attention if needed (only used when SP == 1)
        # When SP > 1, we rebuild freqs inside the loop after gathering full sequences
        if getattr(self.dual_tower_bridge, "apply_cross_rope", False) and sp_size == 1:
            visual_rope_cos_sin, audio_rope_cos_sin = (
                self.dual_tower_bridge.build_aligned_freqs(
                    video_fps=video_fps,
                    grid_size=grid_size,
                    audio_steps=full_audio_seq_len,
                    device=visual_x.device,
                    dtype=visual_x.dtype,
                )
            )
        else:
            visual_rope_cos_sin = None
            audio_rope_cos_sin = None

        for layer_idx in range(min_layers):
            visual_block = visual_dit.blocks[layer_idx]
            audio_block = self.audio_dit.blocks[layer_idx]

            # Cross-modal interaction via dual tower bridge
            # Bridge operations (PerFrameAttentionPooling, RoPE) expect full sequences
            # When SP is enabled, we need to gather before bridge and shard after
            if self.dual_tower_bridge.should_interact(layer_idx, "a2v"):
                if sp_size > 1:
                    # Gather sequences for bridge operations
                    visual_x_full = sequence_model_parallel_all_gather(visual_x, dim=1)
                    audio_x_full = sequence_model_parallel_all_gather(audio_x, dim=1)
                    # Remove padding if any (use full_*_seq_len to trim)
                    visual_x_full = visual_x_full[:, :full_visual_seq_len, :]
                    audio_x_full = audio_x_full[:, :full_audio_seq_len, :]

                    # Call bridge on full sequences (use full freqs, not sharded)
                    if getattr(self.dual_tower_bridge, "apply_cross_rope", False):
                        # Rebuild full freqs for bridge
                        visual_rope_full, audio_rope_full = (
                            self.dual_tower_bridge.build_aligned_freqs(
                                video_fps=video_fps,
                                grid_size=grid_size,
                                audio_steps=full_audio_seq_len,
                                device=visual_x.device,
                                dtype=visual_x.dtype,
                            )
                        )
                    else:
                        visual_rope_full = None
                        audio_rope_full = None

                    visual_x_full, audio_x_full = self.dual_tower_bridge(
                        layer_idx,
                        visual_x_full,
                        audio_x_full,
                        x_freqs=visual_rope_full,
                        y_freqs=audio_rope_full,
                        a2v_condition_scale=a2v_condition_scale,
                        v2a_condition_scale=v2a_condition_scale,
                        condition_scale=condition_scale,
                        video_grid_size=grid_size,
                    )

                    # Shard back for DiT blocks
                    visual_x, _ = self._shard_sequence_for_sp(visual_x_full, dim=1)
                    audio_x, _ = self._shard_sequence_for_sp(audio_x_full, dim=1)
                else:
                    # No SP, call bridge directly
                    visual_x, audio_x = self.dual_tower_bridge(
                        layer_idx,
                        visual_x,
                        audio_x,
                        x_freqs=visual_rope_cos_sin,
                        y_freqs=audio_rope_cos_sin,
                        a2v_condition_scale=a2v_condition_scale,
                        v2a_condition_scale=v2a_condition_scale,
                        condition_scale=condition_scale,
                        video_grid_size=grid_size,
                    )

            # Self-attention and FFN in DiT blocks
            visual_x = visual_block(
                visual_x, visual_context, visual_t_mod, visual_freqs
            )
            audio_x = audio_block(audio_x, audio_context, audio_t_mod, audio_freqs)

        # Process remaining visual layers (if visual has more layers than audio)
        for layer_idx in range(min_layers, visual_layers):
            visual_block = visual_dit.blocks[layer_idx]
            visual_x = visual_block(
                visual_x, visual_context, visual_t_mod, visual_freqs
            )

        return visual_x, audio_x


class MovaDecodingStage(PipelineStage):
    """Decode video and audio outputs for MoVA."""

    def __init__(self, video_vae, audio_vae) -> None:
        super().__init__()
        self.video_vae = video_vae
        self.audio_vae = audio_vae

    @property
    def parallelism_type(self) -> StageParallelismType:
        if get_global_server_args().enable_cfg_parallel:
            return StageParallelismType.MAIN_RANK_ONLY
        return StageParallelismType.REPLICATED

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        self.video_vae = self.video_vae.to(get_local_torch_device())
        self.audio_vae = self.audio_vae.to(get_local_torch_device())

        video_latents = server_args.pipeline_config.denormalize_video_latents(
            batch.latents, self.video_vae
        )

        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32
        ) and not server_args.disable_autocast

        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=vae_dtype,
            enabled=vae_autocast_enabled,
        ):
            if server_args.pipeline_config.vae_tiling:
                self.video_vae.enable_tiling()
            if not vae_autocast_enabled:
                video_latents = video_latents.to(vae_dtype)
            decode_output = self.video_vae.decode(video_latents)
            video = _ensure_tensor_decode_output(decode_output)

        video = (video / 2 + 0.5).clamp(0, 1)

        with torch.autocast(
            device_type=current_platform.device_type, dtype=torch.float32
        ):
            audio = self.audio_vae.decode(batch.audio_latents)
        output_batch = OutputBatch(
            output=video,
            audio=audio,
            audio_sample_rate=getattr(self.audio_vae, "sample_rate", None),
            timings=batch.timings,
        )
        return output_batch
