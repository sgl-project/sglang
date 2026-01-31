# SPDX-License-Identifier: Apache-2.0
"""
MOVA-specific pipeline stages.

Sequence Parallelism (SP) Support:
- Video latents are sharded along the sequence dimension (T*H*W) after patchify
- Audio latents are sharded along the sequence dimension (L) after patchify
- USPAttention handles all-to-all communication internally
- Latents are gathered before unpatchify to restore full sequence
"""

from __future__ import annotations

import functools
import inspect
import os
from collections.abc import Iterable

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
    get_cfg_group,
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
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs, get_global_server_args
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import StageProfiler
from sglang.multimodal_gen.runtime.utils.profiler import SGLDiffusionProfiler
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class MOVALatentPreparationStage(PipelineStage):
    """Prepare video/audio noise latents for MOVA."""

    def __init__(self, audio_vae, require_vae_embedding: bool = True) -> None:
        super().__init__()
        self.audio_vae = audio_vae
        self.require_vae_embedding = require_vae_embedding

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        batch_size = batch.batch_size
        num_frames = batch.num_frames
        if num_frames is None:
            raise ValueError("num_frames is required for MOVA")

        audio_num_samples = int(self.audio_vae.sample_rate * num_frames / batch.fps)

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
            raise ValueError("MOVA requires reference image latents for denoising")
        return batch


class MOVATimestepPreparationStage(PipelineStage):
    """Prepare paired timesteps for MOVA."""

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


class MOVADenoisingStage(PipelineStage):
    """Run MOVA dual-tower denoising loop."""

    def __init__(self, video_dit, video_dit_2, audio_dit, dual_tower_bridge, scheduler):
        super().__init__()
        self.video_dit = video_dit
        self.video_dit_2 = video_dit_2
        self.audio_dit = audio_dit
        self.dual_tower_bridge = dual_tower_bridge
        self.scheduler = scheduler
        self._cache_dit_enabled = False
        self._cached_num_steps = None
        self._torch_compiled = False

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
        timestep_index: int,
        attn_metadata,
        forward_batch: Req | None = None,
    ):
        # Set forward context for distributed attention (USPAttention)
        with set_forward_context(
            current_timestep=timestep_index,
            attn_metadata=attn_metadata,
            forward_batch=forward_batch,
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

    def compile_module_with_torch_compile(self, module, server_args: ServerArgs):
        if not server_args.enable_torch_compile or module is None:
            return module
        if not hasattr(module, "forward"):
            return module
        try:
            import torch._inductor.config as _inductor_cfg

            _inductor_cfg.reorder_for_compute_comm_overlap = True
        except ImportError:
            pass
        mode = os.environ.get("SGLANG_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs")
        logger.info("Compiling %s with mode: %s", module.__class__.__name__, mode)
        compiled_forward = torch.compile(getattr(module, "forward"), mode=mode)
        setattr(module, "forward", compiled_forward)
        return module

    def _maybe_compile_dits(self, server_args: ServerArgs):
        if self._torch_compiled or not server_args.enable_torch_compile:
            return
        for module in filter(None, [self.video_dit, self.video_dit_2, self.audio_dit]):
            self.compile_module_with_torch_compile(module, server_args)
        self._torch_compiled = True

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify denoising stage inputs."""
        result = VerificationResult()
        result.add_check("y", batch.y, V.is_tensor)
        result.add_check("paired_timesteps", batch.paired_timesteps, V.is_tensor)
        result.add_check("latents", batch.latents, V.is_tensor)
        result.add_check("audio_latents", batch.audio_latents, V.is_tensor)
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)
        result.add_check(
            "negative_prompt_embeds",
            batch.negative_prompt_embeds,
            lambda x: not batch.do_classifier_free_guidance or V.list_not_empty(x),
        )
        result.add_check(
            "num_inference_steps", batch.num_inference_steps, V.positive_int
        )
        result.add_check("guidance_scale", batch.guidance_scale, V.non_negative_float)
        result.add_check(
            "guidance_rescale", batch.guidance_rescale, V.non_negative_float
        )
        result.add_check(
            "do_classifier_free_guidance",
            batch.do_classifier_free_guidance,
            V.bool_value,
        )
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify denoising stage outputs."""
        result = VerificationResult()
        result.add_check("latents", batch.latents, V.is_tensor)
        result.add_check("audio_latents", batch.audio_latents, V.is_tensor)
        return result

    def progress_bar(
        self, iterable: Iterable | None = None, total: int | None = None
    ) -> tqdm:
        """
        Create a progress bar for the denoising process.
        """
        local_rank = get_world_group().local_rank
        disable = local_rank != 0
        return tqdm(iterable=iterable, total=total, disable=disable)

    def step_profile(self):
        profiler = SGLDiffusionProfiler.get_instance()
        if profiler:
            profiler.step_denoising_step()

    def rescale_noise_cfg(
        self, noise_cfg, noise_pred_text, guidance_rescale=0.0
    ) -> torch.Tensor:
        """
        Rescale noise prediction according to guidance_rescale.

        Based on findings of "Common Diffusion Noise Schedules and Sample Steps are Flawed"
        (https://arxiv.org/pdf/2305.08891.pdf), Section 3.4.
        """
        std_text = noise_pred_text.std(
            dim=list(range(1, noise_pred_text.ndim)), keepdim=True
        )
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        noise_cfg = (
            guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
        )
        return noise_cfg

    def prepare_extra_func_kwargs(self, func, kwargs) -> dict[str, object]:
        if not kwargs:
            return {}

        if isinstance(func, functools.partial) and func.args:
            func = getattr(func.args[0], "_original_forward", func)

        target_func = inspect.unwrap(func)
        params = inspect.signature(target_func).parameters
        return {k: v for k, v in kwargs.items() if k in params}

    def _build_attn_metadata(
        self, i: int, batch: Req, server_args: ServerArgs
    ) -> object | None:
        return None

    def _manage_device_placement(
        self,
        model_to_use: torch.nn.Module | None,
        model_to_offload: torch.nn.Module | None,
        server_args: ServerArgs,
    ):
        if not server_args.dit_cpu_offload:
            return

        if (
            model_to_offload is not None
            and next(model_to_offload.parameters()).device.type == "cuda"
        ):
            model_to_offload.to("cpu")

        if (
            model_to_use is not None
            and next(model_to_use.parameters()).device.type == "cpu"
        ):
            model_to_use.to(get_local_torch_device())

    def _select_visual_dit(
        self, timestep: float, boundary_ratio: float | None, server_args: ServerArgs
    ):
        if boundary_ratio is None or self.video_dit_2 is None:
            self._manage_device_placement(self.video_dit, None, server_args)
            return self.video_dit

        boundary_timestep = boundary_ratio * self.scheduler.num_train_timesteps
        if timestep >= boundary_timestep:
            current_model = self.video_dit
            model_to_offload = self.video_dit_2
        else:
            current_model = self.video_dit_2
            model_to_offload = self.video_dit

        self._manage_device_placement(current_model, model_to_offload, server_args)
        return current_model

    def _apply_guidance_rescale(
        self,
        noise_pred,
        noise_pred_text,
        guidance_rescale,
        cfg_rank,
        enable_cfg_parallel,
    ):
        if guidance_rescale <= 0.0:
            return noise_pred
        if enable_cfg_parallel:
            std_cfg = noise_pred.std(dim=list(range(1, noise_pred.ndim)), keepdim=True)
            if cfg_rank == 0:
                assert noise_pred_text is not None
                std_text = noise_pred_text.std(
                    dim=list(range(1, noise_pred_text.ndim)), keepdim=True
                )
            else:
                std_text = torch.empty_like(std_cfg)
            std_text = get_cfg_group().broadcast(std_text, src=0)
            noise_pred_rescaled = noise_pred * (std_text / std_cfg)
            return guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * (
                noise_pred
            )
        return self.rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale)

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        self._maybe_compile_dits(server_args)
        self._manage_device_placement(self.audio_dit, None, server_args)

        paired_timesteps = batch.paired_timesteps
        if paired_timesteps is None:
            raise ValueError("paired_timesteps must be set for MOVA")

        y = batch.y if batch.y is not None else batch.image_latent
        if getattr(self.video_dit, "require_vae_embedding", False) and y is None:
            raise ValueError("MOVA requires reference image latents for denoising")

        boundary_ratio = server_args.pipeline_config.boundary_ratio
        total_steps = paired_timesteps.shape[0]
        cfg_rank = get_classifier_free_guidance_rank()
        enable_cfg_parallel = server_args.enable_cfg_parallel

        is_warmup = batch.is_warmup
        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step_from_to,
            getattr(batch, "extra_step_kwargs", None) or {},
        )

        timings = getattr(batch, "timings", None)
        perf_dump_path_provided = getattr(batch, "perf_dump_path", None) is not None

        with self.progress_bar(total=total_steps) as progress_bar:
            for idx_step in range(total_steps):
                with StageProfiler(
                    f"denoising_step_{idx_step}",
                    logger=logger,
                    timings=timings,
                    perf_dump_path_provided=perf_dump_path_provided,
                ):
                    pair_t = paired_timesteps[idx_step]
                    if getattr(pair_t, "shape", None) == (2,):
                        timestep, audio_timestep = pair_t
                    else:
                        timestep = pair_t
                        audio_timestep = pair_t

                    cur_visual_dit = self._select_visual_dit(
                        timestep.item(), boundary_ratio, server_args
                    )

                    timestep = timestep.unsqueeze(0).to(device=get_local_torch_device())
                    audio_timestep = audio_timestep.unsqueeze(0).to(
                        device=get_local_torch_device()
                    )

                    attn_metadata = self._build_attn_metadata(
                        idx_step, batch, server_args
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
                            idx_step,
                            attn_metadata,
                            batch,
                        )
                    else:
                        if enable_cfg_parallel:
                            if cfg_rank == 0:
                                pos = self._predict(
                                    cur_visual_dit,
                                    batch.latents,
                                    batch.audio_latents,
                                    y,
                                    batch.prompt_embeds[0],
                                    timestep,
                                    audio_timestep,
                                    batch.fps,
                                    idx_step,
                                    attn_metadata,
                                    batch,
                                )
                                neg = (None, None)
                            else:
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
                                    idx_step,
                                    attn_metadata,
                                    batch,
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
                                idx_step,
                                attn_metadata,
                                batch,
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
                                idx_step,
                                attn_metadata,
                                batch,
                            )

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

                            if batch.guidance_rescale > 0.0:
                                visual_noise_pred = self._apply_guidance_rescale(
                                    visual_noise_pred,
                                    pos[0] if pos[0] is not None else None,
                                    batch.guidance_rescale,
                                    cfg_rank,
                                    enable_cfg_parallel,
                                )
                                audio_noise_pred = self._apply_guidance_rescale(
                                    audio_noise_pred,
                                    pos[1] if pos[1] is not None else None,
                                    batch.guidance_rescale,
                                    cfg_rank,
                                    enable_cfg_parallel,
                                )

                        if idx_step + 1 < total_steps:
                            next_pair_t = paired_timesteps[idx_step + 1]
                            if getattr(next_pair_t, "shape", None) == (2,):
                                next_timestep, next_audio_timestep = next_pair_t
                            else:
                                next_timestep = next_pair_t
                                next_audio_timestep = next_pair_t
                        else:
                            next_timestep = None
                            next_audio_timestep = None

                        batch.latents = self.scheduler.step_from_to(
                            visual_noise_pred,
                            timestep,
                            next_timestep,
                            batch.latents,
                            **extra_step_kwargs,
                        )
                        batch.audio_latents = self.scheduler.step_from_to(
                            audio_noise_pred,
                            audio_timestep,
                            next_audio_timestep,
                            batch.audio_latents,
                            **extra_step_kwargs,
                        )

                    if progress_bar is not None:
                        progress_bar.update()
                    if not is_warmup and hasattr(self, "step_profile"):
                        self.step_profile()

        for dit in filter(None, [self.video_dit, self.video_dit_2, self.audio_dit]):
            if isinstance(dit, OffloadableDiTMixin):
                dit.prepare_for_next_req()

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
        Single inference step for MOVA dual-tower denoising.

        Supports Sequence Parallelism (SP):
        - After patchify, sequences are sharded across SP ranks
        - USPAttention handles distributed attention communication
        - Before unpatchify, sequences are gathered back
        """
        model_dtype = visual_dit.time_embedding.fc_in.weight.dtype
        device = visual_latents.device

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
        visual_rope_cos_sin, audio_rope_cos_sin = (
            self.dual_tower_bridge.build_aligned_freqs(
                video_fps=video_fps,
                grid_size=grid_size,
                audio_steps=full_audio_seq_len,
                device=visual_x.device,
                dtype=visual_x.dtype,
            )
        )
        if visual_rope_cos_sin is not None:
            visual_rope_cos_sin = [
                self._shard_sequence_for_sp(rope_cos_sin, dim=1)[0]
                for rope_cos_sin in visual_rope_cos_sin
            ]
        if audio_rope_cos_sin is not None:
            audio_rope_cos_sin = [
                self._shard_sequence_for_sp(rope_cos_sin, dim=1)[0]
                for rope_cos_sin in audio_rope_cos_sin
            ]

        for layer_idx in range(min_layers):
            visual_block = visual_dit.blocks[layer_idx]
            audio_block = self.audio_dit.blocks[layer_idx]

            # Cross-modal interaction via dual tower bridge
            # Bridge operations (PerFrameAttentionPooling, RoPE) expect full sequences
            # When SP is enabled, we need to gather before bridge and shard after
            if self.dual_tower_bridge.should_interact(layer_idx, "a2v"):
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


class MOVADecodingStage(PipelineStage):
    """Decode video and audio outputs for MOVA."""

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
