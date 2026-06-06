# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_classifier_free_guidance_rank,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.sana_wm_refiner_transformer import (
    pack_latents,
    unpack_latents,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm import (
    SanaWMDecodingStage,
    sana_wm_broadcast_tensor_dict_from_tp_rank0,
    sana_wm_is_tp_rank0,
    sana_wm_stage_tp_world_size,
    log_sana_wm_tensor_stats,
    sana_wm_diagnostics_enabled,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)

# Distilled 3-step sigma schedule, matches NVlabs `inference_sana_wm.py`.
STAGE_2_DISTILLED_SIGMA_VALUES: tuple[float, ...] = (0.909375, 0.725, 0.421875, 0.0)

# Default Gemma-3 token budget for the refiner prompt encoder.
_REFINER_TEXT_MAX_LENGTH = 1024


class _OfficialLayerwiseModule(nn.Module, LayerwiseOffloadableModuleMixin):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        self.layerwise_offload_managers = []

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError as exc:
            wrapped = self.__dict__.get("module")
            if wrapped is not None:
                return getattr(wrapped, name)
            modules = self.__dict__.get("_modules", {})
            wrapped = modules.get("module")
            if wrapped is not None:
                return getattr(wrapped, name)
            raise exc


class OfficialDiffusersLTX2RefinerModule(_OfficialLayerwiseModule):
    """Thin offload wrapper around Diffusers' official LTX-2 refiner module."""

    layer_names = ["module.transformer_blocks"]


class OfficialGemma3TextEncoderModule(_OfficialLayerwiseModule):
    layer_names = [
        "module.language_model.layers",
        "module.model.language_model.layers",
    ]


def _truthy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def sana_wm_skip_refiner_enabled(
    batch: Req | None = None,
    pipeline_config: Any | None = None,
) -> bool:
    if _truthy_flag(getattr(pipeline_config, "sana_wm_skip_refiner", False)):
        return True
    if _truthy_flag(os.getenv("SGLANG_SANA_WM_SKIP_REFINER", "")):
        return True
    if batch is None:
        return False
    extra = getattr(batch, "extra", None) or {}
    return any(
        _truthy_flag(value)
        for value in (
            extra.get("skip_refiner"),
            extra.get("sana_wm_skip_refiner"),
        )
    )


def default_sana_wm_refiner_dtype(server_args: ServerArgs) -> torch.dtype:
    precision = getattr(server_args.pipeline_config, "dit_precision", "bf16")
    return PRECISION_TO_TYPE.get(precision, torch.bfloat16)


def _refiner_prompt_value(batch: Req) -> Any:
    extra = getattr(batch, "extra", None) or {}
    prompt = extra.get("refiner_prompt")
    return getattr(batch, "prompt", None) if prompt is None else prompt


def _refiner_prompt_matches_batch(batch_size: int | None):
    def validator(value: Any) -> bool:
        if batch_size is None:
            return False
        if isinstance(value, str):
            return True
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return len(value) in (1, batch_size)
        return False

    return validator


def _refiner_seeds_match_batch(batch_size: int | None):
    def validator(value: Any) -> bool:
        if value is None:
            return True
        if batch_size is None:
            return False
        if not isinstance(value, (list, tuple)):
            return False
        if not all(isinstance(item, int) for item in value):
            return False
        return len(value) in (1, batch_size)

    return validator


def _refiner_fps_valid(value: Any) -> bool:
    return value is None or (isinstance(value, (int, float)) and value > 0)


def _is_current_cfg_main_rank() -> bool:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    try:
        return get_classifier_free_guidance_rank() == 0
    except AssertionError:
        return True


def _configured_tp_size(server_args: Any) -> int:
    try:
        return max(int(getattr(server_args, "tp_size", 1) or 1), 1)
    except (TypeError, ValueError):
        return 1


def _is_current_world_main_rank() -> bool:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def _is_current_refiner_execution_rank(server_args: Any) -> bool:
    if getattr(server_args, "enable_cfg_parallel", False):
        return _is_current_cfg_main_rank()
    if _configured_tp_size(server_args) > 1:
        return _is_current_world_main_rank()
    return True


def _pack_text_embeds(
    text_hidden_states: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    padding_side: str = "left",
    scale_factor: int = 8,
    eps: float = 1e-6,
) -> torch.Tensor:
    batch_size, seq_len, hidden_dim, _ = text_hidden_states.shape
    device = text_hidden_states.device
    original_dtype = text_hidden_states.dtype

    token_indices = torch.arange(seq_len, device=device).unsqueeze(0)
    if padding_side == "right":
        mask = token_indices < sequence_lengths[:, None]
    elif padding_side == "left":
        start_indices = seq_len - sequence_lengths[:, None]
        mask = token_indices >= start_indices
    else:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")
    mask = mask[:, :, None, None]

    masked = text_hidden_states.masked_fill(~mask, 0.0)
    valid_positions = (sequence_lengths * hidden_dim).view(batch_size, 1, 1, 1)
    masked_mean = masked.sum(dim=(1, 2), keepdim=True) / (valid_positions + eps)
    x_min = text_hidden_states.masked_fill(~mask, float("inf")).amin(
        dim=(1, 2), keepdim=True
    )
    x_max = text_hidden_states.masked_fill(~mask, float("-inf")).amax(
        dim=(1, 2), keepdim=True
    )
    normalized = (text_hidden_states - masked_mean) / (x_max - x_min + eps)
    normalized = (normalized * scale_factor).flatten(2)

    flat_mask = mask.squeeze(-1).expand(-1, -1, normalized.shape[-1])
    return normalized.masked_fill(~flat_mask, 0.0).to(dtype=original_dtype)


def _refiner_config_value(transformer: nn.Module, name: str) -> Any:
    transformer = _unwrap_diffusers_ltx2_refiner(transformer)
    config = getattr(transformer, "config", None)
    if config is not None:
        if isinstance(config, dict) and name in config:
            return config[name]
        if hasattr(config, name):
            return getattr(config, name)
    return getattr(transformer, name)


def _unwrap_diffusers_ltx2_refiner(transformer: nn.Module) -> nn.Module:
    if isinstance(transformer, OfficialDiffusersLTX2RefinerModule):
        return transformer.module
    return transformer


def _uses_diffusers_ltx2_refiner(transformer: nn.Module) -> bool:
    transformer = _unwrap_diffusers_ltx2_refiner(transformer)
    return transformer.__class__.__name__ == "LTX2VideoTransformer3DModel"


def _uses_native_sana_wm_refiner(transformer: nn.Module) -> bool:
    return transformer.__class__.__name__ == "SanaWMLTX2VideoRefiner"


def _uses_tp_parallel_refiner(transformer: nn.Module, server_args: Any) -> bool:
    return (
        _uses_native_sana_wm_refiner(transformer)
        and _configured_tp_size(server_args) > 1
        and not getattr(server_args, "enable_cfg_parallel", False)
    )


def _uses_native_refiner_tp_group(transformer: nn.Module, server_args: Any) -> bool:
    return _uses_native_sana_wm_refiner(transformer) and _configured_tp_size(
        server_args
    ) > 1


def _as_additive_attention_mask(
    attention_mask: torch.Tensor | None,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.ndim == 2:
        return ((1 - attention_mask.to(dtype)) * -10000.0).unsqueeze(1)
    return attention_mask.to(dtype)


def _forward_diffusers_video_only(
    transformer: nn.Module,
    *,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_attention_mask: torch.Tensor | None,
    num_frames: int,
    height: int,
    width: int,
    fps: float,
    n_context_tokens: int,
) -> torch.Tensor:
    batch_size = hidden_states.size(0)
    encoder_attention_mask = _as_additive_attention_mask(
        encoder_attention_mask, hidden_states.dtype
    )

    video_coords = transformer.rope.prepare_video_coords(
        batch_size, num_frames, height, width, hidden_states.device, fps=fps
    )
    video_rotary_emb = transformer.rope(video_coords, device=hidden_states.device)

    hidden_states = transformer.proj_in(hidden_states)
    temb, embedded_timestep = transformer.time_embed(
        timestep.flatten(),
        batch_size=batch_size,
        hidden_dtype=hidden_states.dtype,
    )
    temb = temb.view(batch_size, -1, temb.size(-1))
    embedded_timestep = embedded_timestep.view(
        batch_size, -1, embedded_timestep.size(-1)
    )

    encoder_hidden_states = transformer.caption_projection(encoder_hidden_states)
    encoder_hidden_states = encoder_hidden_states.view(
        batch_size, -1, hidden_states.size(-1)
    )

    for block in transformer.transformer_blocks:
        hidden_states = _forward_diffusers_video_block(
            block=block,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            video_rotary_emb=video_rotary_emb,
            encoder_attention_mask=encoder_attention_mask,
            n_context_tokens=n_context_tokens,
        )

    scale_shift_values = (
        transformer.scale_shift_table[None, None] + embedded_timestep[:, :, None]
    )
    shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
    hidden_states = transformer.norm_out(hidden_states)
    hidden_states = hidden_states * (1 + scale) + shift
    return transformer.proj_out(hidden_states)


def _forward_diffusers_video_block(
    *,
    block: nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    video_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    encoder_attention_mask: torch.Tensor | None,
    n_context_tokens: int,
) -> torch.Tensor:
    batch_size = hidden_states.size(0)

    norm_hidden_states = block.norm1(hidden_states)
    num_ada_params = block.scale_shift_table.shape[0]
    ada_values = block.scale_shift_table[None, None].to(temb.device) + temb.reshape(
        batch_size, temb.size(1), num_ada_params, -1
    )
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_values.unbind(
        dim=2
    )
    norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

    attn_hidden_states = _streaming_diffusers_self_attention(
        attn=block.attn1,
        hidden_states=norm_hidden_states,
        query_rotary_emb=video_rotary_emb,
        n_context_tokens=n_context_tokens,
    )
    hidden_states = hidden_states + attn_hidden_states * gate_msa

    norm_hidden_states = block.norm2(hidden_states)
    attn_hidden_states = block.attn2(
        norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        query_rotary_emb=None,
        attention_mask=encoder_attention_mask,
    )
    hidden_states = hidden_states + attn_hidden_states

    norm_hidden_states = block.norm3(hidden_states) * (1 + scale_mlp) + shift_mlp
    hidden_states = hidden_states + block.ff(norm_hidden_states) * gate_mlp
    return hidden_states


def _streaming_diffusers_self_attention(
    *,
    attn: nn.Module,
    hidden_states: torch.Tensor,
    query_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    n_context_tokens: int,
) -> torch.Tensor:
    """SANA-WM sink/current streaming mask using Diffusers LTX-2 attention."""

    sequence_length = hidden_states.shape[1]
    if n_context_tokens <= 0 or n_context_tokens >= sequence_length:
        return attn(
            hidden_states=hidden_states,
            encoder_hidden_states=None,
            query_rotary_emb=query_rotary_emb,
        )

    from diffusers.models.attention_dispatch import dispatch_attention_fn
    from diffusers.models.transformers.transformer_ltx2 import (
        apply_interleaved_rotary_emb,
        apply_split_rotary_emb,
    )

    to_gate_logits = getattr(attn, "to_gate_logits", None)
    gate_logits = to_gate_logits(hidden_states) if to_gate_logits is not None else None

    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    query = attn.norm_q(query)
    key = attn.norm_k(key)

    if attn.rope_type == "interleaved":
        query = apply_interleaved_rotary_emb(query, query_rotary_emb)
        key = apply_interleaved_rotary_emb(key, query_rotary_emb)
    elif attn.rope_type == "split":
        query = apply_split_rotary_emb(query, query_rotary_emb)
        key = apply_split_rotary_emb(key, query_rotary_emb)
    else:
        raise ValueError(f"Unsupported LTX-2 RoPE type: {attn.rope_type}")

    query = query.unflatten(2, (attn.heads, -1))
    key = key.unflatten(2, (attn.heads, -1))
    value = value.unflatten(2, (attn.heads, -1))

    processor = attn.processor
    backend = getattr(processor, "_attention_backend", None)
    parallel_config = getattr(processor, "_parallel_config", None)
    context_hidden_states = dispatch_attention_fn(
        query[:, :n_context_tokens],
        key[:, :n_context_tokens],
        value[:, :n_context_tokens],
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        backend=backend,
        parallel_config=parallel_config,
    )
    current_hidden_states = dispatch_attention_fn(
        query[:, n_context_tokens:],
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        backend=backend,
        parallel_config=parallel_config,
    )

    hidden_states = torch.cat([context_hidden_states, current_hidden_states], dim=1)
    hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

    if gate_logits is not None:
        hidden_states = hidden_states.unflatten(2, (attn.heads, -1))
        gates = 2.0 * torch.sigmoid(gate_logits)
        hidden_states = hidden_states * gates.unsqueeze(-1)
        hidden_states = hidden_states.flatten(2, 3)

    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)
    return hidden_states


class SanaWMLTX2RefinerStage(PipelineStage):
    def __init__(
        self,
        *,
        transformer: nn.Module,
        connectors: nn.Module,
        text_encoder: nn.Module,
        tokenizer: Any,
        dtype: torch.dtype,
        text_max_sequence_length: int = _REFINER_TEXT_MAX_LENGTH,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.connectors = connectors
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.text_max_sequence_length = int(text_max_sequence_length)

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

    @property
    def parallelism_type(self) -> StageParallelismType:
        if _uses_tp_parallel_refiner(self.transformer, self.server_args):
            return StageParallelismType.REPLICATED
        if getattr(self.server_args, "enable_cfg_parallel", False):
            return StageParallelismType.MAIN_RANK_ONLY
        if _configured_tp_size(self.server_args) > 1:
            return StageParallelismType.MAIN_RANK_ONLY_AND_SEND_TO_OTHERS
        return StageParallelismType.REPLICATED

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        if sana_wm_skip_refiner_enabled(
            pipeline_config=getattr(server_args, "pipeline_config", None)
        ):
            return []
        if (
            not _uses_tp_parallel_refiner(self.transformer, server_args)
            and not _is_current_refiner_execution_rank(server_args)
        ):
            return []

        stage_name = self._component_stage_name(stage_name)
        if (
            _uses_native_refiner_tp_group(self.transformer, server_args)
            and not sana_wm_is_tp_rank0()
        ):
            return [
                ComponentUse(
                    stage_name=stage_name,
                    component_name="transformer_2",
                    target_dtype=self.dtype,
                    memory_intensive=True,
                    allow_prefetch=False,
                )
            ]

        return [
            ComponentUse(
                stage_name=stage_name,
                component_name="text_encoder_2",
                target_dtype=self.dtype,
            ),
            ComponentUse(
                stage_name=stage_name,
                component_name="connectors",
                target_dtype=self.dtype,
            ),
            ComponentUse(
                stage_name=stage_name,
                component_name="transformer_2",
                target_dtype=self.dtype,
                memory_intensive=True,
                allow_prefetch=False,
            ),
        ]

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify SANA-WM LTX-2 refiner stage inputs."""
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        result.add_check(
            "extra",
            getattr(batch, "extra", None),
            lambda x: x is None or isinstance(x, dict),
        )

        latents = batch.latents if isinstance(batch.latents, torch.Tensor) else None
        batch_size = (
            int(latents.shape[0])
            if latents is not None and latents.dim() == 5
            else None
        )
        if not sana_wm_skip_refiner_enabled(
            batch,
            pipeline_config=getattr(server_args, "pipeline_config", None),
        ):
            result.add_check(
                "refiner_prompt",
                _refiner_prompt_value(batch),
                _refiner_prompt_matches_batch(batch_size),
            )
            result.add_check(
                "fps",
                getattr(batch, "fps", 16),
                _refiner_fps_valid,
            )
            result.add_check(
                "seeds",
                getattr(batch, "seeds", None),
                _refiner_seeds_match_batch(batch_size),
            )
        return result

    @staticmethod
    def _prompts_for_batch(batch: Req, batch_size: int) -> list[str]:
        prompt = batch.extra.get("refiner_prompt") if batch.extra else None
        if prompt is None:
            prompt = batch.prompt
        if isinstance(prompt, str):
            return [prompt] * batch_size
        if isinstance(prompt, list) and all(isinstance(p, str) for p in prompt):
            if len(prompt) == batch_size:
                return prompt
            if len(prompt) == 1:
                return prompt * batch_size
        raise ValueError(
            "SANA-WM refiner requires a string prompt or one prompt per batch item."
        )

    @torch.inference_mode()
    def _encode_prompts(
        self,
        prompts: list[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._should_broadcast_prompt_encoding_over_tp():
            payload = None
            if sana_wm_is_tp_rank0():
                prompt_embeds, attention_mask = self._encode_prompts_local(
                    prompts, device
                )
                payload = {
                    "prompt_embeds": prompt_embeds,
                    "attention_mask": attention_mask,
                }
            payload = sana_wm_broadcast_tensor_dict_from_tp_rank0(payload)
            outputs = (
                payload["prompt_embeds"].to(device=device, dtype=self.dtype),
                payload["attention_mask"].to(device=device),
            )
        else:
            return self._encode_prompts_local(prompts, device)
        return outputs

    def _should_broadcast_prompt_encoding_over_tp(self) -> bool:
        server_args = getattr(self, "server_args", None)
        return (
            _uses_native_refiner_tp_group(self.transformer, server_args)
            and sana_wm_stage_tp_world_size() > 1
        )

    @torch.inference_mode()
    def _encode_prompts_local(
        self,
        prompts: list[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not prompts:
            raise ValueError("SANA-WM refiner requires at least one prompt.")

        tokenizer = self.tokenizer
        if getattr(tokenizer, "padding_side", "right") != "left":
            tokenizer.padding_side = "left"
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        text_inputs = tokenizer(
            [prompt.strip() for prompt in prompts],
            padding="max_length",
            max_length=self.text_max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)

        with self.use_declared_component(
            component_name="text_encoder_2", module=self.text_encoder
        ):
            text_backbone = getattr(self.text_encoder, "model", self.text_encoder)
            outputs = text_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        per_layer_hidden = getattr(outputs, "hidden_states", None)
        if per_layer_hidden is None:
            raise RuntimeError(
                "SANA-WM refiner text encoder must return per-layer hidden_states."
            )
        stacked = torch.stack(per_layer_hidden, dim=-1)  # (B, L, D, n_layers)
        seq_lengths = attention_mask.sum(dim=-1)
        log_sana_wm_tensor_stats("refiner.text_hidden_states_stacked", stacked)
        prompt_embeds = _pack_text_embeds(
            stacked,
            seq_lengths,
            padding_side=tokenizer.padding_side,
        ).to(dtype=self.dtype)
        log_sana_wm_tensor_stats("refiner.prompt_embeds_packed", prompt_embeds)

        with self.use_declared_component(
            component_name="connectors", module=self.connectors
        ):
            video_text_embedding, _, video_attention_mask = self.connectors(
                prompt_embeds, attention_mask
            )
        log_sana_wm_tensor_stats("refiner.video_text_embedding", video_text_embedding)
        log_sana_wm_tensor_stats("refiner.video_attention_mask", video_attention_mask)
        return (
            video_text_embedding.to(device=device, dtype=self.dtype),
            video_attention_mask.to(device=device),
        )

    @torch.inference_mode()
    def _encode_prompt(
        self,
        prompt: str,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._encode_prompts([prompt], device)

    def _predict_current_x0(
        self,
        *,
        sink: torch.Tensor,
        noisy_current: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        sigma: torch.Tensor,
        fps: float,
        n_context_tokens: int,
        step_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        full_latent = torch.cat([sink, noisy_current], dim=2)
        batch_size, _, num_frames, height, width = full_latent.shape
        patch_size = int(_refiner_config_value(self.transformer, "patch_size"))
        patch_size_t = int(_refiner_config_value(self.transformer, "patch_size_t"))
        latent_tokens = pack_latents(full_latent, patch_size, patch_size_t)

        raw_timestep = torch.zeros(
            batch_size,
            latent_tokens.shape[1],
            1,
            dtype=torch.float32,
            device=latent_tokens.device,
        )
        raw_timestep[:, n_context_tokens:, 0] = sigma.float()

        with self.use_declared_component(
            component_name="transformer_2", module=self.transformer
        ):
            if _uses_diffusers_ltx2_refiner(self.transformer):
                model_timestep = raw_timestep.squeeze(-1) * float(
                    _refiner_config_value(self.transformer, "timestep_scale_multiplier")
                )
                velocity_tokens = _forward_diffusers_video_only(
                    self.transformer,
                    hidden_states=latent_tokens.to(self.dtype),
                    encoder_hidden_states=prompt_embeds,
                    timestep=model_timestep,
                    encoder_attention_mask=prompt_attention_mask,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    fps=fps,
                    n_context_tokens=n_context_tokens,
                )
            else:
                additive_mask = _as_additive_attention_mask(
                    prompt_attention_mask, self.dtype
                )
                with set_forward_context(
                    current_timestep=step_idx,
                    attn_metadata=None,
                ):
                    velocity_tokens = self.transformer(
                        hidden_states=latent_tokens.to(self.dtype),
                        encoder_hidden_states=prompt_embeds,
                        timestep=raw_timestep.squeeze(-1),
                        encoder_attention_mask=additive_mask,
                        num_frames=num_frames,
                        height=height,
                        width=width,
                        fps=fps,
                        n_context_tokens=n_context_tokens,
                    )

        current_tokens = latent_tokens[:, n_context_tokens:, :]
        denoised = latent_tokens.float() - velocity_tokens.float() * raw_timestep
        return current_tokens, denoised[:, n_context_tokens:, :].to(self.dtype)

    @torch.inference_mode()
    def _refine_batch(
        self,
        latents: torch.Tensor,
        prompts: list[str],
        *,
        fps: float,
        seeds: list[int],
        sink_size: int = 1,
    ) -> torch.Tensor:
        device = get_local_torch_device()
        z = latents.to(device=device, dtype=self.dtype)
        if z.shape[2] <= sink_size:
            raise ValueError(
                f"Stage-1 latent has {z.shape[2]} frames but sink_size={sink_size}."
            )
        batch_size = int(z.shape[0])
        if len(prompts) != batch_size:
            raise ValueError(
                "SANA-WM refiner prompt batch does not match latent batch: "
                f"{len(prompts)} prompts for batch {batch_size}."
            )
        if len(seeds) != batch_size:
            raise ValueError(
                "SANA-WM refiner seed batch does not match latent batch: "
                f"{len(seeds)} seeds for batch {batch_size}."
            )
        self.log_info(
            "SANA-WM refiner start: latent=%s, fps=%.3f, seeds=%s, "
            "sink_size=%d, sigmas=%s, diagnostics=%s",
            tuple(z.shape),
            fps,
            seeds,
            sink_size,
            STAGE_2_DISTILLED_SIGMA_VALUES,
            "on" if sana_wm_diagnostics_enabled() else "off",
        )
        log_sana_wm_tensor_stats("refiner.input_latent", z)

        prompt_embeds, prompt_attention_mask = self._encode_prompts(prompts, device)

        sigmas = torch.tensor(
            STAGE_2_DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=device
        )
        start_sigma = float(sigmas[0])
        sink = z[:, :, :sink_size].contiguous()
        current = z[:, :, sink_size:].contiguous()
        log_sana_wm_tensor_stats("refiner.sink_latent", sink)
        log_sana_wm_tensor_stats("refiner.current_latent_clean", current)
        eps = torch.cat(
            [
                torch.randn(
                    current[index : index + 1].shape,
                    generator=torch.Generator(device=device).manual_seed(int(seed)),
                    device=device,
                    dtype=self.dtype,
                )
                for index, seed in enumerate(seeds)
            ],
            dim=0,
        )
        noisy = (1.0 - start_sigma) * current + start_sigma * eps
        log_sana_wm_tensor_stats("refiner.current_latent_noisy_initial", noisy)

        patch_size = int(_refiner_config_value(self.transformer, "patch_size"))
        patch_size_t = int(_refiner_config_value(self.transformer, "patch_size_t"))

        sink_tokens = pack_latents(sink, patch_size, patch_size_t)
        n_context_tokens = sink_tokens.shape[1]

        for step_idx in range(len(sigmas) - 1):
            sigma = sigmas[step_idx]
            noisy_tokens, denoised = self._predict_current_x0(
                sink=sink,
                noisy_current=noisy,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                sigma=sigma,
                fps=fps,
                n_context_tokens=n_context_tokens,
                step_idx=step_idx,
            )
            velocity_tokens = (noisy_tokens.float() - denoised.float()) / sigma.float()
            next_tokens = (
                noisy_tokens.float()
                + velocity_tokens * (sigmas[step_idx + 1] - sigma).float()
            )
            noisy = unpack_latents(
                next_tokens.to(self.dtype),
                num_frames=noisy.shape[2],
                height=noisy.shape[3],
                width=noisy.shape[4],
                patch_size=patch_size,
                patch_size_t=patch_size_t,
            )
            if sana_wm_diagnostics_enabled():
                velocity_5d = unpack_latents(
                    velocity_tokens,
                    num_frames=noisy.shape[2],
                    height=noisy.shape[3],
                    width=noisy.shape[4],
                    patch_size=patch_size,
                    patch_size_t=patch_size_t,
                )
                log_sana_wm_tensor_stats(
                    f"refiner.step_{step_idx}.velocity_current",
                    velocity_5d.to(self.dtype),
                )
                log_sana_wm_tensor_stats(f"refiner.step_{step_idx}.current_latent", noisy)

        refined = torch.cat([sink, noisy], dim=2)
        log_sana_wm_tensor_stats("refiner.output_latent", refined)
        return refined

    @torch.inference_mode()
    def _refine_one(
        self,
        latent: torch.Tensor,
        prompt: str,
        *,
        fps: float,
        seed: int,
        sink_size: int = 1,
    ) -> torch.Tensor:
        return self._refine_batch(
            latent,
            [prompt],
            fps=fps,
            seeds=[seed],
            sink_size=sink_size,
        )

    @torch.inference_mode()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if batch.latents is None:
            raise ValueError("SANA-WM refiner requires batch.latents from stage 1.")
        if batch.latents.ndim != 5:
            raise ValueError(
                "SANA-WM refiner expects 5D latents shaped (B, C, T, H, W), "
                f"got {tuple(batch.latents.shape)}."
            )

        if sana_wm_skip_refiner_enabled(
            batch,
            pipeline_config=getattr(server_args, "pipeline_config", None),
        ):
            if batch.extra is None:
                batch.extra = {}
            batch.extra["sana_wm_refiner_applied"] = False
            self.log_info(
                "SANA-WM LTX-2 refiner skipped by config or request flag."
            )
            return batch

        batch_size = int(batch.latents.shape[0])
        prompts = self._prompts_for_batch(batch, batch_size)
        fps = float(getattr(batch, "fps", 16) or 16)

        seeds: list[int]
        if batch.seeds is not None and len(batch.seeds) == batch_size:
            seeds = [int(s) for s in batch.seeds]
        elif batch.seeds is not None and len(batch.seeds) == 1:
            seeds = [int(batch.seeds[0])] * batch_size
        else:
            seeds = [int(getattr(batch, "seed", 0) or 0)] * batch_size

        refined = self._refine_batch(
            batch.latents,
            prompts,
            fps=fps,
            seeds=seeds,
        )
        batch.latents = refined.to(
            device=batch.latents.device, dtype=batch.latents.dtype
        )
        if batch.extra is None:
            batch.extra = {}
        batch.extra["sana_wm_refiner_applied"] = True
        self.log_info("SANA-WM LTX-2 refiner applied to stage-1 latents.")
        return batch


class SanaWMRefinerDecodingStage(SanaWMDecodingStage):
    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs):
        self._drop_refiner_sink = bool(
            (getattr(batch, "extra", None) or {}).get("sana_wm_refiner_applied", True)
        )
        try:
            return super().forward(batch, server_args)
        finally:
            self._drop_refiner_sink = True

    @torch.no_grad()
    def decode(
        self,
        latents: torch.Tensor,
        server_args: ServerArgs,
        *,
        vae_dtype: torch.dtype,
    ) -> torch.Tensor:
        frames = super().decode(latents, server_args, vae_dtype=vae_dtype)
        log_sana_wm_tensor_stats("refiner.decode.frames_with_sink", frames)
        if frames.ndim != 5:
            raise ValueError(
                "SANA-WM refiner decoding expects decoded video shaped "
                f"(B, C, T, H, W), got {tuple(frames.shape)}."
            )
        if frames.shape[2] <= 1:
            raise ValueError(
                "SANA-WM refiner decoding expected a sink frame plus refined "
                f"frames, got temporal length {frames.shape[2]}."
            )
        if not getattr(self, "_drop_refiner_sink", True):
            log_sana_wm_tensor_stats("refiner.decode.frames_output", frames)
            return frames
        frames = frames[:, :, 1:].contiguous()
        log_sana_wm_tensor_stats("refiner.decode.frames_output", frames)
        return frames
