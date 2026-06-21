# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass

import torch

from sglang.multimodal_gen.configs.pipeline_configs.ideogram import (
    LATENT_SCALE,
    LATENT_SHIFT,
)
from sglang.multimodal_gen.configs.sample.ideogram import IDEOGRAM4_PRESETS
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.layers.attention import build_varlen_mask_meta
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import (
    _ensure_tensor_decode_output,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingContext,
    DenoisingStage,
    DenoisingStepState,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.nvtx_pytorch_hooks import maybe_nvtx_range
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

SEQUENCE_PADDING_INDICATOR = -1
OUTPUT_IMAGE_INDICATOR = 2
LLM_TOKEN_INDICATOR = 3
IMAGE_POSITION_OFFSET = 65536


@dataclass(frozen=True)
class LogitNormalSchedule:
    mean: float
    std: float = 1.0
    logsnr_min: float = -15.0
    logsnr_max: float = 18.0

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        t = t.to(torch.float64)
        z = torch.special.ndtri(t)
        y = self.mean + self.std * z
        t_ = 1 - torch.special.expit(y)
        t_min = 1.0 / (1 + math.exp(0.5 * self.logsnr_max))
        t_max = 1.0 / (1 + math.exp(0.5 * self.logsnr_min))
        return t_.clamp(t_min, t_max).to(torch.float32)


@dataclass(frozen=True)
class Ideogram4TextEncodingFingerprint:
    prompt: object
    height: int
    width: int
    num_outputs_per_prompt: int
    max_text_tokens: int
    patch_size: int
    ae_scale_factor: int


def get_schedule_for_resolution(image_resolution, known_mean: float, std: float):
    num_pixels = image_resolution[0] * image_resolution[1]
    known_pixels = 512 * 512
    mean = known_mean + 0.5 * math.log(num_pixels / known_pixels)
    return LogitNormalSchedule(mean=mean, std=std)


def make_step_intervals(num_steps: int) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, num_steps + 1, dtype=torch.float32)


class Ideogram4Scheduler:
    order = 1
    init_noise_sigma = 1.0
    num_train_timesteps = 1

    def __init__(self) -> None:
        self.timesteps = torch.empty(0, dtype=torch.float32)
        self._begin_index = None

    def set_begin_index(self, begin_index: int) -> None:
        self._begin_index = begin_index

    def set_timesteps(self, num_inference_steps: int, device=None) -> None:
        self.timesteps = torch.arange(
            num_inference_steps - 1,
            -1,
            -1,
            dtype=torch.float32,
            device=device or get_local_torch_device(),
        )

    def scale_model_input(self, sample: torch.Tensor, timestep=None) -> torch.Tensor:
        return sample

    def step(self, model_output, timestep, sample, return_dict=False, **kwargs):
        raise RuntimeError("Ideogram4DenoisingStage applies its custom scheduler step")


class Ideogram4TextEncodingStage(TextEncodingStage):
    deduplicated_extra_tensor_tree_output_keys = ("ideogram4",)

    def __init__(self, text_encoder, tokenizer) -> None:
        super().__init__([text_encoder], [tokenizer])

    def _tokenize(self, prompt: str, max_text_tokens: int):
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.tokenizers[0].apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        encoded = self.tokenizers[0](
            text, return_tensors="pt", add_special_tokens=False
        )
        token_ids = encoded["input_ids"][0]
        num_text_tokens = int(token_ids.shape[0])
        if num_text_tokens > max_text_tokens:
            raise ValueError(
                f"prompt has {num_text_tokens} tokens, exceeds max_text_tokens={max_text_tokens}"
            )
        return token_ids, num_text_tokens

    def _build_inputs(self, prompts: list[str], height: int, width: int, server_args):
        cfg = server_args.pipeline_config
        tokenized = [self._tokenize(p, cfg.max_text_tokens) for p in prompts]
        batch_size = len(prompts)
        patch = cfg.patch_size * cfg.ae_scale_factor
        if height < 256 or height > 2048 or width < 256 or width > 2048:
            raise ValueError("height/width must be between 256 and 2048")
        if height % patch != 0 or width % patch != 0:
            raise ValueError(
                f"height/width must be divisible by patch_size*ae_scale_factor={patch}"
            )
        grid_h = height // patch
        grid_w = width // patch
        num_image_tokens = grid_h * grid_w
        max_text_tokens = max(num_text for _, num_text in tokenized)
        total_seq_len = max_text_tokens + num_image_tokens
        device = get_local_torch_device()

        h_idx = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
        w_idx = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
        t_idx = torch.zeros_like(h_idx)
        image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET

        token_ids = torch.zeros(batch_size, total_seq_len, dtype=torch.long)
        text_position_ids = torch.zeros(batch_size, total_seq_len, 3, dtype=torch.long)
        position_ids = torch.zeros(batch_size, total_seq_len, 3, dtype=torch.long)
        segment_ids = torch.full(
            (batch_size, total_seq_len), SEQUENCE_PADDING_INDICATOR, dtype=torch.long
        )
        indicator = torch.zeros(batch_size, total_seq_len, dtype=torch.long)

        for b, (toks, num_text) in enumerate(tokenized):
            pad_len = max_text_tokens - num_text
            total_unpadded = num_text + num_image_tokens
            offset = pad_len
            token_ids[b, offset : offset + num_text] = toks
            text_pos = torch.arange(num_text)
            text_pos_3d = torch.stack([text_pos, text_pos, text_pos], dim=1)
            text_position_ids[b, offset : offset + num_text] = text_pos_3d
            position_ids[b, offset : offset + num_text] = text_pos_3d
            position_ids[b, offset + num_text :] = image_pos
            indicator[b, offset : offset + num_text] = LLM_TOKEN_INDICATOR
            indicator[b, offset + num_text :] = OUTPUT_IMAGE_INDICATOR
            segment_ids[b, offset : offset + total_unpadded] = 1

        return {
            "token_ids": token_ids.to(device),
            "text_position_ids": text_position_ids.to(device),
            "position_ids": position_ids.to(device),
            "segment_ids": segment_ids.to(device),
            "indicator": indicator.to(device),
            "num_image_tokens": num_image_tokens,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "max_text_tokens": max_text_tokens,
        }

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        prompts = batch.prompt if isinstance(batch.prompt, list) else [batch.prompt]
        prompts = [p or " " for p in prompts]
        if batch.num_outputs_per_prompt > 1:
            prompts = [
                prompt
                for prompt in prompts
                for _ in range(batch.num_outputs_per_prompt)
            ]
        inputs = self._build_inputs(prompts, batch.height, batch.width, server_args)
        with self.use_declared_component(
            component_name="text_encoder", module=self.text_encoders[0]
        ) as text_encoder:
            llm_features = text_encoder.encode_ideogram_features(
                inputs["token_ids"],
                inputs["text_position_ids"],
                inputs["indicator"],
                LLM_TOKEN_INDICATOR,
            )
        batch.prompt_embeds = [llm_features]
        batch.prompt_embeds_mask = [
            (inputs["indicator"] == LLM_TOKEN_INDICATOR).to(torch.bool)
        ]
        batch.extra["ideogram4"] = inputs
        return batch

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("prompt", batch.prompt, V.string_or_list_strings)
        result.add_check("height", batch.height, V.positive_int)
        result.add_check("width", batch.width, V.positive_int)
        result.add_check(
            "num_outputs_per_prompt", batch.num_outputs_per_prompt, V.positive_int
        )
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check(
            "prompt_embeds", batch.prompt_embeds, V.list_of_tensors_min_dims(2)
        )
        result.add_check(
            "prompt_embeds_mask",
            batch.prompt_embeds_mask,
            V.list_of_tensors_min_dims(2),
        )
        result.add_check(
            "ideogram4_extra",
            batch.extra.get("ideogram4"),
            lambda x: isinstance(x, dict),
        )
        return result

    def build_dedup_fingerprint(
        self, batch: Req, server_args: ServerArgs
    ) -> Ideogram4TextEncodingFingerprint:
        cfg = server_args.pipeline_config
        return Ideogram4TextEncodingFingerprint(
            prompt=self.freeze_for_dedup(batch.prompt),
            height=int(batch.height),
            width=int(batch.width),
            num_outputs_per_prompt=int(batch.num_outputs_per_prompt),
            max_text_tokens=int(cfg.max_text_tokens),
            patch_size=int(cfg.patch_size),
            ae_scale_factor=int(cfg.ae_scale_factor),
        )


class Ideogram4DenoisingStage(DenoisingStage):
    def __init__(self, transformer, unconditional_transformer, pipeline=None) -> None:
        super().__init__(
            transformer=transformer,
            scheduler=Ideogram4Scheduler(),
            pipeline=pipeline,
        )
        self.unconditional_transformer = unconditional_transformer
        self._maybe_torch_compile(self.unconditional_transformer)

    def _component_name_for_stage_module(self, module, default_name: str) -> str:
        if module is self.unconditional_transformer:
            return "unconditional_transformer"
        return super()._component_name_for_stage_module(module, default_name)

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        return [
            ComponentUse(
                stage_name=stage_name,
                component_name="transformer",
                phase="transformer",
                preferred_ready_after_request=True,
                memory_intensive=True,
            ),
            ComponentUse(
                stage_name=stage_name,
                component_name="unconditional_transformer",
                phase="unconditional_transformer",
                memory_intensive=True,
            ),
        ]

    def _maybe_enable_cache_dit_and_torch_compile(
        self, num_inference_steps: int | tuple[int, int], batch: Req
    ) -> None:
        self._maybe_enable_cache_dit(num_inference_steps, batch)
        for transformer in filter(
            None, [self.transformer, self.unconditional_transformer]
        ):
            self._maybe_torch_compile(transformer)

    def _manage_unconditional_transformer_use_site(self, batch: Req) -> None:
        manager = self._component_residency_manager
        if manager is None:
            return
        use = self._declared_component_use(
            component_name="unconditional_transformer",
            phase="unconditional_transformer",
        )
        manager.begin_use(use, module=self.unconditional_transformer)

    def _manage_dit_use_site(
        self,
        current_model: torch.nn.Module,
        current_phase: str,
        batch: Req,
    ) -> None:
        if self._component_residency_manager is None:
            return
        super()._manage_dit_use_site(current_model, current_phase, batch)

    def _run_ideogram_transformer(
        self, current_model: torch.nn.Module, call_kwargs: dict
    ) -> torch.Tensor:
        runner = self._maybe_get_bcg_runner(current_model)
        if runner is not None:
            return self._bcg_run(runner, call_kwargs, current_model)
        return current_model(**call_kwargs)

    def _preprocess_sp_latents(self, batch: Req, server_args: ServerArgs):
        batch.did_sp_shard_latents = False

    def _postprocess_sp_latents(
        self,
        batch: Req,
        latents: torch.Tensor,
        trajectory_tensor: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return latents, trajectory_tensor

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        return VerificationResult()

    def _prepare_denoising_loop(
        self, batch: Req, server_args: ServerArgs
    ) -> DenoisingContext:
        preset = getattr(batch, "preset", "V4_DEFAULT_20")
        if preset not in IDEOGRAM4_PRESETS:
            raise ValueError(
                f"Unknown Ideogram 4 preset {preset!r}; expected one of {sorted(IDEOGRAM4_PRESETS)}"
            )
        preset_cfg = IDEOGRAM4_PRESETS[preset]
        num_steps = int(preset_cfg["num_steps"])
        device = get_local_torch_device()
        schedule = get_schedule_for_resolution(
            (batch.height, batch.width),
            known_mean=float(preset_cfg["mu"]),
            std=float(preset_cfg["std"]),
        )
        step_intervals = make_step_intervals(num_steps).to(device)
        guidance_schedule = torch.as_tensor(
            preset_cfg["guidance_schedule"], dtype=torch.float32, device=device
        )
        schedule_values = schedule(step_intervals)
        schedule_deltas = schedule_values[:-1] - schedule_values[1:]

        self.scheduler.set_timesteps(num_steps, device=device)
        batch.scheduler = self.scheduler
        batch.timesteps = self.scheduler.timesteps
        batch.num_inference_steps = num_steps

        ctx = super()._prepare_denoising_loop(batch, server_args)
        # ideogram fp8 denoising keeps explicit fp32 latent/scheduler math;
        # wrapping the full loop in bf16 autocast collapses latent variance
        ctx.autocast_enabled = False

        data = batch.extra["ideogram4"]
        z = ctx.latents.to(device, dtype=torch.float32)
        llm_features = batch.prompt_embeds[0]
        batch_size = z.shape[0]
        max_text_tokens = data["max_text_tokens"]
        num_image_tokens = data["num_image_tokens"]
        latent_dim = z.shape[-1]
        text_z_padding = torch.zeros(
            batch_size,
            max_text_tokens,
            latent_dim,
            dtype=torch.float32,
            device=z.device,
        )
        neg_position_ids = data["position_ids"][:, max_text_tokens:]
        neg_segment_ids = data["segment_ids"][:, max_text_tokens:]
        neg_indicator = data["indicator"][:, max_text_tokens:]
        attn_mask = data["segment_ids"] > 0
        neg_attn_mask = neg_segment_ids > 0
        neg_llm_features = torch.zeros(
            batch_size,
            num_image_tokens,
            llm_features.shape[-1],
            dtype=llm_features.dtype,
            device=z.device,
        )
        ctx.latents = z
        ctx.extra.update(
            {
                "ideogram4_schedule_values": schedule_values,
                "ideogram4_schedule_deltas": schedule_deltas,
                "ideogram4_guidance_schedule": guidance_schedule,
                "ideogram4_text_z_padding": text_z_padding,
                "ideogram4_attn_mask": attn_mask,
                "ideogram4_attn_mask_meta": build_varlen_mask_meta(attn_mask),
                "ideogram4_neg_position_ids": neg_position_ids,
                "ideogram4_neg_segment_ids": neg_segment_ids,
                "ideogram4_neg_indicator": neg_indicator,
                "ideogram4_neg_attn_mask": neg_attn_mask,
                "ideogram4_neg_attn_mask_meta": build_varlen_mask_meta(neg_attn_mask),
                "ideogram4_neg_llm_features": neg_llm_features,
            }
        )
        return ctx

    def _run_denoising_step(
        self,
        ctx: DenoisingContext,
        step: DenoisingStepState,
        batch: Req,
        server_args: ServerArgs,
    ) -> None:
        data = batch.extra["ideogram4"]
        z = ctx.latents.to(dtype=torch.float32)
        llm_features = batch.prompt_embeds[0]
        max_text_tokens = data["max_text_tokens"]
        num_image_tokens = data["num_image_tokens"]
        schedule_values = ctx.extra["ideogram4_schedule_values"]
        schedule_deltas = ctx.extra["ideogram4_schedule_deltas"]
        guidance_schedule = ctx.extra["ideogram4_guidance_schedule"]
        i = step.t_int

        t_val = schedule_values[i + 1]
        t = t_val.expand(z.shape[0])
        pos_z = torch.cat([ctx.extra["ideogram4_text_z_padding"], z], dim=1)
        use_nvtx = self.current_use_nvtx

        with maybe_nvtx_range("predict_noise", use_nvtx):
            with set_forward_context(
                current_timestep=i,
                attn_metadata=step.attn_metadata,
                forward_batch=batch,
            ):
                pos_out = self._run_ideogram_transformer(
                    step.current_model,
                    dict(
                        llm_features=llm_features,
                        x=pos_z,
                        t=t,
                        position_ids=data["position_ids"],
                        segment_ids=data["segment_ids"],
                        indicator=data["indicator"],
                        attn_mask=ctx.extra["ideogram4_attn_mask"],
                        attn_mask_meta=ctx.extra["ideogram4_attn_mask_meta"],
                    ),
                )
                pos_v = pos_out[:, max_text_tokens : max_text_tokens + num_image_tokens]

            self._manage_unconditional_transformer_use_site(batch)
            with set_forward_context(
                current_timestep=i,
                attn_metadata=step.attn_metadata,
                forward_batch=batch,
            ):
                neg_v = self._run_ideogram_transformer(
                    self.unconditional_transformer,
                    dict(
                        llm_features=ctx.extra["ideogram4_neg_llm_features"],
                        x=z,
                        t=t,
                        position_ids=ctx.extra["ideogram4_neg_position_ids"],
                        segment_ids=ctx.extra["ideogram4_neg_segment_ids"],
                        indicator=ctx.extra["ideogram4_neg_indicator"],
                        attn_mask=ctx.extra["ideogram4_neg_attn_mask"],
                        attn_mask_meta=ctx.extra["ideogram4_neg_attn_mask_meta"],
                    ),
                )

        with maybe_nvtx_range("scheduler_step", use_nvtx):
            velocity = (
                guidance_schedule[i] * pos_v + (1.0 - guidance_schedule[i]) * neg_v
            )
            ctx.latents = z + velocity * schedule_deltas[i]


class Ideogram4DecodingStage(PipelineStage):
    @property
    def role_affinity(self):
        from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType

        return RoleType.DECODER

    def __init__(self, vae) -> None:
        super().__init__()
        self.vae = vae

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        return [
            ComponentUse(
                self._component_stage_name(stage_name),
                "vae",
                target_dtype=PRECISION_TO_TYPE[
                    server_args.pipeline_config.vae_precision
                ],
                keep_ready_after_warmup=True,
            )
        ]

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        data = batch.extra["ideogram4"]
        latents = batch.latents.to(get_local_torch_device())
        cfg = server_args.pipeline_config
        patch = cfg.patch_size
        shift = torch.tensor(LATENT_SHIFT, device=latents.device, dtype=latents.dtype)
        scale = torch.tensor(LATENT_SCALE, device=latents.device, dtype=latents.dtype)
        z = latents * scale.to(latents.dtype) + shift.to(latents.dtype)
        batch_size = z.shape[0]
        grid_h = data["grid_h"]
        grid_w = data["grid_w"]
        ae_channels = z.shape[-1] // (patch * patch)
        z = z.view(batch_size, grid_h, grid_w, patch, patch, ae_channels)
        z = z.permute(0, 5, 1, 3, 2, 4).contiguous()
        z = z.view(batch_size, ae_channels, grid_h * patch, grid_w * patch)
        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        with self.use_declared_component(component_name="vae", module=self.vae) as vae:
            z = z.to(vae_dtype)
            decoded = vae.decode(z)
            frames = _ensure_tensor_decode_output(decoded)
        frames = (frames / 2 + 0.5).clamp(0, 1)
        return OutputBatch(
            output=frames,
            trajectory_timesteps=batch.trajectory_timesteps,
            trajectory_latents=batch.trajectory_latents,
            rollout_trajectory_data=batch.rollout_trajectory_data,
            trajectory_decoded=None,
            metrics=batch.metrics,
            noise_pred=None,
        )
