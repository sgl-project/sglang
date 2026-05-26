# SPDX-License-Identifier: Apache-2.0
#
# SANA-WM stage-2 LTX-2 latent refiner.
#
# Runs a 3-step Euler loop on top of the SANA-WM stage-1 latent using the
# upstream LTX-2 video-only refiner:
#   * `transformer_2` -> `SanaWMLTX2VideoRefiner`
#   * `connectors`    -> sglang `LTX2TextConnectors`
#   * `text_encoder_2`/`tokenizer_2` -> sglang `Gemma3ForConditionalGeneration`
#                                       + HF AutoTokenizer
#
# All four modules are loaded by `PipelineComponentLoader` and handed to the
# stage by `SanaWMTwoStagePipeline.create_pipeline_stages` (see
# `runtime/pipelines/sana_wm_pipeline.py`). The stage does not load any
# weights itself, and it does not import raw `diffusers`/`transformers`
# model classes.
#
# Sink/current split: the first stage-1 latent frame is preserved as the
# clean "anchor" (sink) and only the remaining frames are denoised. The
# refiner forward feeds the packed (sink + noisy current) token sequence with
# `n_context_tokens` set so streaming SLA isolates the two halves the same
# way NVlabs' `inference_sana_wm.py` does.

from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.models.dits.sana_wm_refiner_transformer import (
    pack_latents,
    unpack_latents,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm import (
    SanaWMDecodingStage,
    log_sana_wm_tensor_stats,
    sana_wm_diagnostics_enabled,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)

# Distilled 3-step sigma schedule, matches NVlabs `inference_sana_wm.py`.
STAGE_2_DISTILLED_SIGMA_VALUES: tuple[float, ...] = (0.909375, 0.725, 0.421875, 0.0)

# Default Gemma-3 token budget for the refiner prompt encoder.
_REFINER_TEXT_MAX_LENGTH = 1024


def _skip_refiner_enabled() -> bool:
    return os.getenv("SGLANG_SANA_WM_SKIP_REFINER", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def default_sana_wm_refiner_dtype(server_args: ServerArgs) -> torch.dtype:
    precision = getattr(server_args.pipeline_config, "dit_precision", "bf16")
    return PRECISION_TO_TYPE.get(precision, torch.bfloat16)


def _pack_text_embeds(
    text_hidden_states: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    padding_side: str = "left",
    scale_factor: int = 8,
    eps: float = 1e-6,
) -> torch.Tensor:
    """SANA-WM-specific text-embed pooling.

    Stacks per-layer Gemma-3 hidden states (`text_hidden_states` shape
    `(B, L, D, n_layers)`), applies a masked min-max normalization across the
    (token, layer) axes, scales by `scale_factor`, then flattens layers into
    the channel dim, padded positions zeroed out. Matches NVlabs upstream.
    """
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
        raise ValueError(
            f"padding_side must be 'left' or 'right', got {padding_side}"
        )
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


class SanaWMLTX2RefinerStage(PipelineStage):
    """Run the SANA-WM stage-2 LTX-2 refiner before VAE decode.

    Modules are injected by `SanaWMTwoStagePipeline`:
      * `transformer` (`SanaWMLTX2VideoRefiner`) -- video-only LTX-2 forward
      * `connectors` (`LTX2TextConnectors`)
      * `text_encoder` (`Gemma3ForConditionalGeneration`)
      * `tokenizer` (HF `AutoTokenizer` for the refiner Gemma-3)
    """

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

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        if _skip_refiner_enabled():
            return []

        # Declare every component this stage forwards through so
        # ComponentResidencyManager moves them onto GPU before the stage runs.
        # Without this, `dit_cpu_offload=True` keeps the refiner sub-modules on
        # CPU and the first matmul fails with "mat2 is on cpu" vs cuda inputs.
        # The tokenizer stays on CPU (no nn.Module weights to ferry).
        stage_name = self._component_stage_name(stage_name)
        return [
            ComponentUse(
                stage_name=stage_name,
                component_name="transformer_2",
                target_dtype=self.dtype,
                memory_intensive=True,
            ),
            ComponentUse(
                stage_name=stage_name,
                component_name="connectors",
                target_dtype=self.dtype,
            ),
            ComponentUse(
                stage_name=stage_name,
                component_name="text_encoder_2",
                target_dtype=self.dtype,
            ),
        ]

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
    def _encode_prompt(
        self,
        prompt: str,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokenizer = self.tokenizer
        if getattr(tokenizer, "padding_side", "right") != "left":
            tokenizer.padding_side = "left"
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        text_inputs = tokenizer(
            [prompt.strip()],
            padding="max_length",
            max_length=self.text_max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)

        outputs = self.text_encoder(
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

        video_text_embedding, _, video_attention_mask = self.connectors(
            prompt_embeds, attention_mask
        )
        log_sana_wm_tensor_stats(
            "refiner.video_text_embedding", video_text_embedding
        )
        log_sana_wm_tensor_stats(
            "refiner.video_attention_mask", video_attention_mask
        )
        return (
            video_text_embedding.to(device=device, dtype=self.dtype),
            video_attention_mask.to(device=device),
        )

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
        device = get_local_torch_device()
        z = latent.to(device=device, dtype=self.dtype)
        if z.shape[2] <= sink_size:
            raise ValueError(
                f"Stage-1 latent has {z.shape[2]} frames but sink_size={sink_size}."
            )
        self.log_info(
            "SANA-WM refiner start: latent=%s, fps=%.3f, seed=%d, "
            "sink_size=%d, sigmas=%s, diagnostics=%s",
            tuple(z.shape),
            fps,
            seed,
            sink_size,
            STAGE_2_DISTILLED_SIGMA_VALUES,
            "on" if sana_wm_diagnostics_enabled() else "off",
        )
        log_sana_wm_tensor_stats("refiner.input_latent", z)

        prompt_embeds, prompt_attention_mask = self._encode_prompt(prompt, device)
        # Reshape mask to additive bias the same way `_forward_video_only` did.
        if prompt_attention_mask.ndim == 2:
            additive = (1 - prompt_attention_mask.to(self.dtype)) * -10000.0
            additive = additive.unsqueeze(1)
        else:
            additive = prompt_attention_mask.to(self.dtype)

        sigmas = torch.tensor(
            STAGE_2_DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=device
        )
        start_sigma = float(sigmas[0])
        sink = z[:, :, :sink_size].contiguous()
        current = z[:, :, sink_size:].contiguous()
        log_sana_wm_tensor_stats("refiner.sink_latent", sink)
        log_sana_wm_tensor_stats("refiner.current_latent_clean", current)
        gen = torch.Generator(device=device).manual_seed(int(seed))
        eps = torch.randn(
            current.shape, generator=gen, device=device, dtype=self.dtype
        )
        noisy = (1.0 - start_sigma) * current + start_sigma * eps
        log_sana_wm_tensor_stats("refiner.current_latent_noisy_initial", noisy)

        patch_size = int(self.transformer.patch_size)
        patch_size_t = int(self.transformer.patch_size_t)
        B = z.shape[0]

        sink_tokens = pack_latents(sink, patch_size, patch_size_t)
        n_context_tokens = sink_tokens.shape[1]

        for step_idx in range(len(sigmas) - 1):
            sigma = sigmas[step_idx]
            full_latent = torch.cat([sink, noisy], dim=2)
            T_full = full_latent.shape[2]
            H_full = full_latent.shape[3]
            W_full = full_latent.shape[4]

            full_tokens = pack_latents(full_latent, patch_size, patch_size_t)
            L_full = full_tokens.shape[1]

            timestep = torch.zeros(
                B, L_full, dtype=torch.float32, device=device
            )
            timestep[:, n_context_tokens:] = sigma

            # The framework's attention layer (layers/attention/layer.py)
            # reads attn metadata from the active forward context. The refiner
            # DiT goes through LTX2Attention -> framework attn, so we must
            # establish a forward context here -- same pattern as DenoisingStage
            # and TextEncodingStage. attn_metadata=None is acceptable since
            # this is an inference forward (no KV cache / paged attention).
            with set_forward_context(
                current_timestep=step_idx,
                attn_metadata=None,
            ):
                velocity_tokens = self.transformer(
                    hidden_states=full_tokens.to(self.dtype),
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    encoder_attention_mask=additive,
                    num_frames=T_full,
                    height=H_full,
                    width=W_full,
                    fps=fps,
                    n_context_tokens=n_context_tokens,
                )
            velocity_5d = unpack_latents(
                velocity_tokens,
                num_frames=T_full,
                height=H_full,
                width=W_full,
                patch_size=patch_size,
                patch_size_t=patch_size_t,
            )
            velocity_current = velocity_5d[:, :, sink_size:].to(self.dtype)
            dt = (sigmas[step_idx + 1] - sigma).float()
            noisy = (noisy.float() + velocity_current.float() * dt).to(self.dtype)
            log_sana_wm_tensor_stats(
                f"refiner.step_{step_idx}.velocity_current", velocity_current
            )
            log_sana_wm_tensor_stats(
                f"refiner.step_{step_idx}.current_latent", noisy
            )

        refined = torch.cat([sink, noisy], dim=2)
        log_sana_wm_tensor_stats("refiner.output_latent", refined)
        return refined

    @torch.inference_mode()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if batch.latents is None:
            raise ValueError("SANA-WM refiner requires batch.latents from stage 1.")
        if batch.latents.ndim != 5:
            raise ValueError(
                "SANA-WM refiner expects 5D latents shaped (B, C, T, H, W), "
                f"got {tuple(batch.latents.shape)}."
            )

        if _skip_refiner_enabled():
            if batch.extra is None:
                batch.extra = {}
            batch.extra["sana_wm_refiner_applied"] = False
            self.log_info(
                "SANA-WM LTX-2 refiner skipped by SGLANG_SANA_WM_SKIP_REFINER."
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

        refined: list[torch.Tensor] = []
        for idx, (prompt, seed) in enumerate(zip(prompts, seeds, strict=True)):
            refined.append(
                self._refine_one(
                    batch.latents[idx : idx + 1],
                    prompt,
                    fps=fps,
                    seed=seed,
                )
            )
        batch.latents = torch.cat(refined, dim=0).to(
            device=batch.latents.device, dtype=batch.latents.dtype
        )
        if batch.extra is None:
            batch.extra = {}
        batch.extra["sana_wm_refiner_applied"] = True
        self.log_info("SANA-WM LTX-2 refiner applied to stage-1 latents.")
        return batch


class SanaWMRefinerDecodingStage(SanaWMDecodingStage):
    """Decode refined latents and drop the clean sink anchor frame."""

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
        # Match NVlabs `inference_sana_wm.py`: decode with the clean sink anchor,
        # then drop the first frame from the returned video.
        frames = frames[:, :, 1:].contiguous()
        log_sana_wm_tensor_stats("refiner.decode.frames_output", frames)
        return frames
