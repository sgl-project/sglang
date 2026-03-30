import os
from collections import deque
from typing import List, Optional, Union

import torch
from diffusers.utils import is_ftfy_available
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor

from sglang.multimodal_gen.runtime.distributed import divide, get_tp_world_size
from sglang.multimodal_gen.runtime.distributed.group_coordinator import (
    get_local_torch_device,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.kv_cache import KVCacheManager
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import TextEncodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import StageProfiler
from sglang.multimodal_gen.runtime.utils.profiler import SGLDiffusionProfiler
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

if is_ftfy_available():
    import ftfy

import html

import regex as re

logger = init_logger(__name__)


def _maybe_compile_module(
    module: object, module_name: str, server_args: ServerArgs, fullgraph: bool = False
) -> object:
    if not server_args.enable_torch_compile or not isinstance(module, torch.nn.Module):
        return module
    marker = "_sglang_torch_compile_module_enabled"
    if getattr(module, marker, False):
        return module

    compile_mode = os.environ.get(
        "SGLANG_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs"
    )
    try:
        module.compile(mode=compile_mode, fullgraph=fullgraph, dynamic=None)
        setattr(module, marker, True)
    except Exception as e:
        logger.warning(
            "Failed to compile %s, falling back to eager: %s", module_name, e
        )
    return module


def _maybe_compile_method(
    module: object,
    method_name: str,
    module_name: str,
    server_args: ServerArgs,
    fullgraph: bool = False,
) -> None:
    if not server_args.enable_torch_compile:
        return
    compile_mode = os.environ.get(
        "SGLANG_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs"
    )
    marker = f"_sglang_torch_compile_{method_name}_enabled"
    if getattr(module, marker, False):
        return
    method = getattr(module, method_name, None)
    if not callable(method):
        return

    try:
        compiled_method = torch.compile(
            method, mode=compile_mode, fullgraph=fullgraph, dynamic=None
        )
        setattr(module, method_name, compiled_method)
        setattr(module, marker, True)
    except Exception as e:
        logger.warning(
            "Failed to compile %s.%s, falling back to eager: %s",
            module_name,
            method_name,
            e,
        )


class KreaRealtimeVideoTextEncodingStage(TextEncodingStage):

    def __init__(self, text_encoders, tokenizers) -> None:
        """
        Initialize the prompt encoding stage.

        """
        super().__init__(text_encoders, tokenizers)
        self.interpolation_steps = 4

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify text encoding stage inputs."""
        result = VerificationResult()
        result.add_check("prompt", batch.prompt, V.string_or_list_strings)
        result.add_check("prompt_embeds", batch.prompt_embeds, V.is_list)
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify text encoding stage outputs."""
        result = VerificationResult()
        result.add_check(
            "prompt_embeds", batch.prompt_embeds, V.list_of_tensors_min_dims(2)
        )
        if batch.debug:
            logger.debug(f"{batch.prompt_embeds=}")
        return result

    @torch.no_grad()
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ):
        assert len(self.tokenizers) == len(self.text_encoders)
        assert len(self.text_encoders) == len(
            server_args.pipeline_config.text_encoder_configs
        )

        assert batch.prompt is not None
        # encode new prompt
        if batch.session.is_prompt_changed(batch.prompt):
            prompt_text: str | list[str] = batch.prompt
            all_indices: list[int] = list(range(len(self.text_encoders)))
            prompt_embeds_list, _ = self.encode_text(
                prompt_text,
                server_args,
                encoder_index=all_indices,
                return_attention_mask=False,
            )

            # interpolate embeds for prompt change
            interpolate_embeds = None
            if batch.session.last_embeds:
                interpolate_embeds = self.interpolate_embeds(
                    batch.session.last_embeds, prompt_embeds_list
                )

            # update session
            batch.session.save_prompt_changed(
                batch.prompt, prompt_embeds_list, interpolate_embeds
            )

        # set correct embeds
        curr_embeds = batch.session.get_current_embeds()
        batch.prompt_embeds.extend(curr_embeds)

        return batch

    def interpolate_embeds(self, prev_embeds: torch.Tensor, curr_embeds: torch.Tensor):
        assert len(prev_embeds) == len(curr_embeds)
        interpolated_embeds_list = []
        for i in range(len(prev_embeds)):
            assert prev_embeds[i].shape == curr_embeds[i].shape
            x = torch.lerp(
                prev_embeds[i],
                curr_embeds[i],
                torch.linspace(0, 1, steps=self.interpolation_steps)
                .unsqueeze(1)
                .unsqueeze(2)
                .to(prev_embeds[i]),
            )
            interpolated_embeds_list.append(
                list(x.chunk(self.interpolation_steps, dim=0))
            )
        # change shape from [num_embeds, interpolation_steps, ...] to [interpolation_steps, num_embeds, ...]
        result = []
        for step_idx in range(self.interpolation_steps):
            step_embeds = []
            for layer_chunks in interpolated_embeds_list:
                step_embeds.append(layer_chunks[step_idx])
            result.append(step_embeds)
        return result


class KreaRealtimeVideoBeforeDenoisingStage(PipelineStage):
    def __init__(self, tokenizer, transformer, vae, vae_dtype: torch.dtype) -> None:
        super().__init__()
        self.vae = vae
        self.tokenizer = tokenizer
        self.transformer = transformer
        self.video_processor = VideoProcessor(vae_scale_factor=8)
        self._vae_latents_mean = torch.tensor(
            self.vae.config.latents_mean,
            device=self.vae.device,
            dtype=vae_dtype,
        ).view(1, self.vae.config.z_dim, 1, 1, 1)
        self._vae_latents_std = 1.0 / torch.tensor(
            self.vae.config.latents_std,
            device=self.vae.device,
            dtype=vae_dtype,
        ).view(1, self.vae.config.z_dim, 1, 1, 1)
        self._maybe_enable_torch_compile()

    def _maybe_enable_torch_compile(self) -> None:
        _maybe_compile_module(
            self.transformer,
            "KreaTransformer",
            self.server_args,
            fullgraph=False,
        )

    def _reset_vae_encode_cache(self):
        """Initialize VAE encoder-side cache state even if clear_cache was monkeypatched."""
        if hasattr(self.vae, "_original_clear_cache"):
            self.vae._original_clear_cache()
        else:
            self.vae.clear_cache()

        # Defensive reset for encoder cache fields used by Wan VAE encode().
        if not hasattr(self.vae, "_enc_conv_idx"):
            self.vae._enc_conv_idx = 0
        if hasattr(self.vae, "_enc_conv_num"):
            self.vae._enc_feat_map = [None] * self.vae._enc_conv_num
        elif hasattr(self.vae, "_enc_feat_map"):
            self.vae._enc_feat_map = [None] * len(self.vae._enc_feat_map)
        else:
            self.vae._enc_feat_map = [None] * 55

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ):
        # step1 TextEncoder
        device = get_local_torch_device()
        default_num_inference_steps = 4
        num_inference_steps = batch.num_inference_steps
        if num_inference_steps is None:
            num_inference_steps = default_num_inference_steps
        elif num_inference_steps <= 0:
            logger.warning(
                "Invalid num_inference_steps=%s for krea realtime; fallback to %s",
                num_inference_steps,
                default_num_inference_steps,
            )
            num_inference_steps = default_num_inference_steps
        batch.num_inference_steps = num_inference_steps

        transformer_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision]
        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        block_idx = batch.block_idx
        num_frames_per_block = self.transformer.config.arch_config.num_frames_per_block
        kv_cache_num_frames = self.transformer.config.arch_config.kv_cache_num_frames
        has_video_condition = batch.input_video is not None
        strength = 0.7 if has_video_condition else 1.0
        frame_seq_length = self.transformer.config.arch_config.frame_seq_length
        # step2 set timesteps
        batch.timesteps, batch.all_timesteps, batch.sigmas = self.prepare_timesteps(
            5, strength, num_inference_steps
        )

        # step3 prepare latents
        # video to video
        if batch.input_video is not None:
            if isinstance(batch.input_video, list):
                conditioning_frames = batch.input_video
            else:
                conditioning_frames = [batch.input_video]

            if len(conditioning_frames) < num_frames_per_block:
                conditioning_frames = conditioning_frames + [
                    conditioning_frames[-1]
                ] * (num_frames_per_block - len(conditioning_frames))
            video = (
                self.video_processor.preprocess(
                    conditioning_frames,
                    batch.height,
                    batch.width,
                )
                .unsqueeze(0)
                .to(vae_dtype)
            )
            batch.current_start_frame = block_idx * num_frames_per_block
            init_latents = self.encode_frames(
                video,
                vae_dtype,
                device,
                None,
            )
            init_latents = init_latents[:, :, -num_frames_per_block:]

            strength = batch.timesteps[0] / 1000.0
            noise = randn_tensor(
                init_latents.shape,
                device=self.transformer.device,
                dtype=transformer_dtype,
                generator=batch.generator,
            )

            init_latents = init_latents * (1.0 - strength) + noise * strength
            init_latents = init_latents.to(transformer_dtype).contiguous()

            batch.latents = init_latents
        else:
            # text to video
            # For realtime chunked generation, we need enough latent blocks to index
            # the current block; otherwise slicing can produce an empty frame range.
            effective_num_blocks = max(batch.num_blocks, batch.block_idx + 1)
            init_latents = self.prepare_latents(
                1,
                server_args.pipeline_config.vae_config.arch_config.scale_factor_spatial,
                server_args.pipeline_config.dit_config.arch_config.num_channels_latents,
                batch.height,
                batch.width,
                effective_num_blocks,
                num_frames_per_block,
                transformer_dtype,
                self.transformer.device,
                batch.generator,
                batch.latents,
            )
            init_latents = init_latents.contiguous()
            start_frame = block_idx * num_frames_per_block
            end_frame = start_frame + num_frames_per_block

            # Extract single block from full latent buffer
            # final_latents shape: [B, C, total_frames, H, W]
            # Extract frames along the time dimension (dim=2)
            batch.latents = init_latents[:, :, start_frame:end_frame, :, :]
            if batch.latents.shape[2] == 0:
                raise RuntimeError(
                    "Krea realtime latents are empty after slicing. "
                    f"block_idx={block_idx}, num_frames_per_block={num_frames_per_block}, "
                    f"effective_num_blocks={effective_num_blocks}, "
                    f"start_frame={start_frame}, end_frame={end_frame}, "
                    f"init_latents_frames={init_latents.shape[2]}"
                )
            batch.current_start_frame = start_frame
        # step4 setup kvcache
        num_heads = divide(self.transformer.num_attention_heads, get_tp_world_size())
        head_dim = self.transformer.attention_head_dim
        num_blocks = len(self.transformer.blocks)
        sa_max_size = (kv_cache_num_frames + num_frames_per_block) * frame_seq_length

        batch.local_attn_size = kv_cache_num_frames + num_frames_per_block
        for block in self.transformer.blocks:
            block.attn1.local_attn_size = -1
        for block in self.transformer.blocks:
            block.attn1.num_frame_per_block = num_frames_per_block

        sink_size = self.transformer.blocks[0].attn1.sink_size
        manager = batch.session.kv_cache_manager
        if manager is None:
            manager = KVCacheManager(
                num_blocks=num_blocks,
                sa_batch_size=1,
                sa_max_size=sa_max_size,
                sa_num_heads=num_heads,
                sa_head_dim=head_dim,
                dtype=transformer_dtype,
                device=self.transformer.device,
                sink_size=sink_size,
                frame_seq_length=frame_seq_length,
            )
            batch.session.kv_cache_manager = manager
        else:
            manager.reset_self_attn()
            if batch.update_prompt_embeds:
                manager.reset_cross_attn()

        # step5 recomputeKVCache
        if batch.block_idx != 0:
            context_frames = self.get_context_frames(
                kv_cache_num_frames,
                num_frames_per_block,
                batch,
                vae_dtype,
            )
            block_mask = self.transformer._prepare_blockwise_causal_attn_mask(
                self.transformer.device,
                num_frames=context_frames.shape[2],
                frame_seqlen=frame_seq_length,
                num_frame_per_block=num_frames_per_block,
                local_attn_size=-1,
            )
            self.transformer.block_mask = block_mask
            context_timestep = torch.zeros(
                (context_frames.shape[0], context_frames.shape[2]),
                device=self.transformer.device,
                dtype=torch.int64,
            )
            with set_forward_context(
                current_timestep=0,
                attn_metadata=None,
                forward_batch=batch,
            ):
                self.transformer(
                    hidden_states=context_frames.to(transformer_dtype),
                    timestep=context_timestep,
                    encoder_hidden_states=batch.prompt_embeds[0].to(transformer_dtype),
                    kv_cache=manager.self_attn_caches,
                    crossattn_cache=manager.cross_attn_caches,
                    current_start=0,
                    cache_start=None,
                )
            self.transformer.block_mask = None
        return batch

    def get_context_frames(
        self,
        kv_cache_num_frames,
        num_frames_per_block,
        batch,
        vae_dtype,
    ):
        current_kv_cache_num_frames = kv_cache_num_frames
        total_frames_generated = (batch.block_idx - 1) * num_frames_per_block

        if total_frames_generated < current_kv_cache_num_frames:
            context_frames = batch.session.current_denoised_latents[
                :, :, :current_kv_cache_num_frames
            ]

        else:
            context_frames = batch.session.current_denoised_latents
            context_frames = context_frames[:, :, 1:][
                :, :, -current_kv_cache_num_frames + 1 :
            ]
            first_frame_latent = self.prepare_frame_latents(
                frames=batch.session.frame_cache_context[0],
                dtype=vae_dtype,
            )
            first_frame_latent = first_frame_latent.to(batch.latents)
            context_frames = torch.cat((first_frame_latent, context_frames), dim=2)

        return context_frames

    def prepare_frame_latents(self, frames, dtype):
        self._reset_vae_encode_cache()
        frames = frames.to(device=self.vae.device, dtype=dtype).contiguous()
        latents = retrieve_latents(self.vae.encode(frames), sample_mode="argmax")
        latents = (latents - self._vae_latents_mean) * self._vae_latents_std

        return latents

    def encode_frames(
        self,
        video: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        latents: Optional[torch.Tensor] = None,
    ):
        if latents is not None:
            return latents.to(device, dtype)

        self._reset_vae_encode_cache()

        init_latents = [
            retrieve_latents(
                self.vae.encode(
                    vid.unsqueeze(0)
                    .transpose(2, 1)
                    .to(device=self.vae.device, dtype=dtype)
                    .contiguous()
                ),
                sample_mode="argmax",
            )
            for vid in video
        ]
        init_latents = torch.cat(init_latents, dim=0).to(dtype)
        init_latents = (init_latents - self._vae_latents_mean) * self._vae_latents_std

        return init_latents

    def prepare_timesteps(self, shift, strength, num_inference_steps):
        sigmas = torch.linspace(1.0, 0.0, 1001, device=self.transformer.device)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        all_timesteps = sigmas * 1000.0
        zero_padded_timesteps = torch.cat(
            [
                all_timesteps,
                torch.tensor(
                    [0],
                    device=self.transformer.device,
                    dtype=all_timesteps.dtype,
                ),
            ]
        )
        denoising_steps = torch.linspace(
            strength * 1000,
            0,
            num_inference_steps,
            dtype=torch.float32,
        ).to(torch.long)
        timesteps = zero_padded_timesteps[1000 - denoising_steps]

        return timesteps, all_timesteps, sigmas

    def prepare_latents(
        self,
        batch_size: int,
        vae_scale_factor: float = 8.0,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_blocks: int = 9,
        num_frames_per_block: int = 3,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = num_blocks * num_frames_per_block
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // vae_scale_factor,
            int(width) // vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(
            shape,
            generator=generator,
            device=self.transformer.device,
            dtype=dtype,
        )
        return latents


class KreaRealtimeVideoDenoisingStage(PipelineStage):
    def __init__(self, transformer, scheduler, vae, vae_dtype: torch.dtype):
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler
        self.vae = vae
        self.video_processor = VideoProcessor(vae_scale_factor=8)
        self._vae_latents_mean = torch.tensor(
            self.vae.config.latents_mean,
            device=self.vae.device,
            dtype=vae_dtype,
        ).view(1, self.vae.config.z_dim, 1, 1, 1)
        self._vae_latents_std = 1.0 / torch.tensor(
            self.vae.config.latents_std,
            device=self.vae.device,
            dtype=vae_dtype,
        ).view(1, self.vae.config.z_dim, 1, 1, 1)
        self._maybe_enable_torch_compile()

    def _maybe_enable_torch_compile(self) -> None:
        _maybe_compile_module(
            self.transformer,
            "KreaTransformer",
            self.server_args,
            fullgraph=False,
        )

    def forward(self, batch: Req, server_args: ServerArgs):
        latents = batch.latents
        prompt_embeds = batch.prompt_embeds[0]
        manager = batch.session.kv_cache_manager
        kv_cache = manager.self_attn_caches
        crossattn_cache = manager.cross_attn_caches
        current_start_frame = batch.current_start_frame
        timesteps = batch.timesteps
        all_timesteps = batch.all_timesteps
        sigmas = batch.sigmas
        num_inference_steps = batch.num_inference_steps
        kv_cache_num_frames = self.transformer.config.arch_config.kv_cache_num_frames
        num_frames_per_block = self.transformer.config.arch_config.num_frames_per_block
        frame_seq_length = self.transformer.config.arch_config.frame_seq_length
        self.transformer_dtype = PRECISION_TO_TYPE[
            server_args.pipeline_config.dit_precision
        ]
        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        is_warmup = batch.is_warmup
        # Precompute sigma for each denoising step once to avoid per-step sync.
        step_timestep_ids = torch.argmin(
            (all_timesteps.unsqueeze(0) - timesteps.unsqueeze(1)).abs(), dim=1
        )
        step_sigmas = sigmas[step_timestep_ids]
        # Iterate over each timestep
        for i, t in enumerate(timesteps):
            with StageProfiler(
                f"denoising_step_{i}",
                logger=logger,
                metrics=batch.metrics,
                perf_dump_path_provided=batch.perf_dump_path is not None,
            ):
                # Step 1: predict noise
                noise_pred = self.predict_noise(
                    latents=latents,
                    timestep=t,
                    timestep_index=i,
                    prompt_embeds=prompt_embeds,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start_frame=current_start_frame,
                    kv_cache_num_frames=kv_cache_num_frames,
                    num_frames_per_block=num_frames_per_block,
                    seq_length=32760,
                    frame_seq_length=frame_seq_length,
                    batch=batch,
                )

                # Step 2: update latents
                latents = self.update_latents(
                    latents=latents,
                    noise_pred=noise_pred,
                    sigma_t=step_sigmas[i],
                )

                # Step 3: if not the last step, add noise for the next timestep
                # This is a common practice in samplers like DDIM
                if i < (num_inference_steps - 1):
                    # Prepare noise
                    sample = latents.transpose(1, 2).squeeze(0)
                    noise = randn_tensor(
                        sample.shape,
                        device=latents.device,
                        dtype=latents.dtype,
                        generator=batch.generator,
                    )

                    # Add noise to latents
                    latents = (
                        self.add_noise(
                            sample=sample,
                            noise=noise,
                            sigma_t=step_sigmas[i + 1],
                        )
                        .unsqueeze(0)
                        .transpose(1, 2)
                    )
            if not is_warmup:
                self.step_profile()
        batch.latents = latents
        batch.session.current_denoised_latents = latents
        if batch.session.frame_cache_context is None:
            frame_cache_len = 1 + (kv_cache_num_frames - 1) * 4
            batch.session.frame_cache_context = deque(maxlen=frame_cache_len)

        # Disable clearing cache
        if batch.block_idx == 0:
            if not hasattr(self.vae, "_original_clear_cache"):
                self.vae._original_clear_cache = self.vae.clear_cache
            self.vae._original_clear_cache()
            self.vae.clear_cache = lambda: None
            decoder_cache_len = getattr(self.vae, "_conv_num", 55)
            self.vae._feat_map = [None] * decoder_cache_len

        if batch.block_idx != 0:
            self.vae._feat_map = batch.session.decoder_cache

        latents = batch.latents.to(device=self.vae.device, dtype=vae_dtype)
        latents = latents / self._vae_latents_std + self._vae_latents_mean

        videos = self.vae.decode(latents)

        batch.session.decoder_cache = self.vae._feat_map
        batch.session.frame_cache_context.extend(videos.split(1, dim=2))
        videos = self.video_processor.postprocess_video(videos, output_type="np")

        output_batch = OutputBatch(
            output=videos,
            trajectory_timesteps=batch.trajectory_timesteps,
            trajectory_latents=batch.trajectory_latents,
            trajectory_decoded=None,
            metrics=batch.metrics,
        )
        return output_batch

    def step_profile(self):
        profiler = SGLDiffusionProfiler.get_instance()
        if profiler:
            profiler.step_denoising_step()

    def add_noise(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        sigma_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to a sample.

        Uses the formula: noisy_sample = (1 - sigma) * sample + sigma * noise

        Args:
            sample: clean sample
            noise: noise tensor
            sigma_t: scalar sigma at the current step

        Returns:
            The noisy sample.
        """
        sigma = sigma_t.to(device=sample.device).reshape(1, 1, 1, 1)

        # Add noise
        noisy_sample = (
            (1 - sigma.double()) * sample.double() + sigma.double() * noise.double()
        ).type_as(noise)

        return noisy_sample

    def predict_noise(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        timestep_index: int,
        prompt_embeds: torch.Tensor,
        kv_cache: list,
        crossattn_cache: list,
        current_start_frame: int,
        kv_cache_num_frames: int,
        num_frames_per_block: int,
        seq_length: int,
        frame_seq_length: int,
        batch: Req,
    ) -> torch.Tensor:
        # Compute the effective start frame (not exceeding cache capacity)
        start_frame = min(current_start_frame, kv_cache_num_frames)
        prompt_embeds = prompt_embeds.to(self.transformer_dtype)
        # Call the Transformer to predict noise
        with set_forward_context(
            current_timestep=timestep_index,
            attn_metadata=None,
            forward_batch=batch,
        ):
            noise_pred = self.transformer(
                hidden_states=latents,
                timestep=timestep.expand(latents.shape[0], num_frames_per_block),
                encoder_hidden_states=prompt_embeds,
                kv_cache=kv_cache,
                seq_len=seq_length,
                crossattn_cache=crossattn_cache,
                current_start=start_frame * frame_seq_length,
                start_frame=start_frame,
                cache_start=None,
            )

        return noise_pred

    def update_latents(
        self,
        latents: torch.Tensor,
        noise_pred: torch.Tensor,
        sigma_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update latents based on the predicted noise.

        Uses: latents = latents - sigma * noise_pred

        Args:
            latents: Current latent representation
            noise_pred: Predicted noise
            sigma_t: scalar sigma at the current step

        Returns:
            Updated latents
        """
        # Use float64 for numerical stability
        latents_dtype = latents.dtype
        sigma_t = sigma_t.to(device=latents.device)
        latents = (latents.double() - sigma_t.double() * noise_pred.double()).to(
            latents_dtype
        )

        return latents


def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample",
):
    if hasattr(encoder_output, "sample") and sample_mode == "sample":
        return encoder_output.sample(generator)
    elif hasattr(encoder_output, "mode") and sample_mode == "argmax":
        return encoder_output.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text
