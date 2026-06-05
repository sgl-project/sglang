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
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import (
    _ensure_tensor_decode_output,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
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
            scheduler=None,
            transformer_2=unconditional_transformer,
            pipeline=pipeline,
        )

    def _component_name_for_stage_module(self, module, default_name: str) -> str:
        if default_name == "transformer_2":
            return "unconditional_transformer"
        return super()._component_name_for_stage_module(module, default_name)

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        return VerificationResult()

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        preset = getattr(batch, "preset", "V4_DEFAULT_20")
        if preset not in IDEOGRAM4_PRESETS:
            raise ValueError(
                f"Unknown Ideogram 4 preset {preset!r}; expected one of {sorted(IDEOGRAM4_PRESETS)}"
            )
        preset_cfg = IDEOGRAM4_PRESETS[preset]
        num_steps = int(preset_cfg["num_steps"])
        schedule = get_schedule_for_resolution(
            (batch.height, batch.width),
            known_mean=float(preset_cfg["mu"]),
            std=float(preset_cfg["std"]),
        )
        step_intervals = make_step_intervals(num_steps).to(get_local_torch_device())
        gw_per_step = torch.as_tensor(
            preset_cfg["guidance_schedule"],
            dtype=torch.float32,
            device=get_local_torch_device(),
        )

        data = batch.extra["ideogram4"]
        z = batch.latents.to(get_local_torch_device(), dtype=torch.float32)
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
        neg_llm_features = torch.zeros(
            batch_size,
            num_image_tokens,
            llm_features.shape[-1],
            dtype=llm_features.dtype,
            device=z.device,
        )

        with (
            self.use_declared_component(
                component_name="transformer", module=self.transformer
            ) as transformer,
            self.use_declared_component(
                component_name="unconditional_transformer",
                module=self.transformer_2,
            ) as unconditional_transformer,
        ):
            for i in self.progress_bar(
                range(num_steps - 1, -1, -1),
                total=num_steps,
                disable=batch.suppress_logs,
            ):
                t_val = float(schedule(step_intervals[i + 1].unsqueeze(0)).item())
                s_val = float(schedule(step_intervals[i].unsqueeze(0)).item())
                t = torch.full(
                    (batch_size,), t_val, dtype=torch.float32, device=z.device
                )
                pos_z = torch.cat([text_z_padding, z], dim=1)
                with set_forward_context(
                    current_timestep=i,
                    attn_metadata=None,
                    forward_batch=batch,
                ):
                    pos_out = transformer(
                        llm_features=llm_features,
                        x=pos_z,
                        t=t,
                        position_ids=data["position_ids"],
                        segment_ids=data["segment_ids"],
                        indicator=data["indicator"],
                    )
                    pos_v = pos_out[:, max_text_tokens:]
                    neg_v = unconditional_transformer(
                        llm_features=neg_llm_features,
                        x=z,
                        t=t,
                        position_ids=neg_position_ids,
                        segment_ids=neg_segment_ids,
                        indicator=neg_indicator,
                    )
                z = z + (gw_per_step[i] * pos_v + (1.0 - gw_per_step[i]) * neg_v) * (
                    s_val - t_val
                )

        batch.latents = z
        return batch


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
