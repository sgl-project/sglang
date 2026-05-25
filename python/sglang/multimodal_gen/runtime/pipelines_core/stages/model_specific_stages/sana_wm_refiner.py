# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""SANA-WM specific LTX-2 video-only refinement stage."""

from __future__ import annotations

import gc
import os
from pathlib import Path

import torch
from torch import nn

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)

STAGE_2_DISTILLED_SIGMA_VALUES: tuple[float, ...] = (0.909375, 0.725, 0.421875, 0.0)


class SanaWMLTX2RefinerStage(PipelineStage):
    """Run the official SANA-WM LTX-2 latent refiner before VAE decode."""

    def __init__(
        self,
        *,
        refiner_root: str,
        refiner_gemma_root: str,
        dtype: torch.dtype,
        text_max_sequence_length: int = 1024,
    ) -> None:
        super().__init__()
        self.refiner_root = refiner_root
        self.refiner_gemma_root = refiner_gemma_root
        self.dtype = dtype
        self.text_max_sequence_length = text_max_sequence_length
        self._refiner: _DiffusersLTX2VideoRefiner | None = None

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

    def _load_refiner(self) -> "_DiffusersLTX2VideoRefiner":
        if self._refiner is None:
            self._validate_refiner_layout()
            self.log_info("Loading SANA-WM LTX-2 refiner from %s", self.refiner_root)
            self._refiner = _DiffusersLTX2VideoRefiner(
                refiner_root=self.refiner_root,
                gemma_root=self.refiner_gemma_root,
                dtype=self.dtype,
                device=get_local_torch_device(),
                text_max_sequence_length=self.text_max_sequence_length,
            )
        return self._refiner

    def _validate_refiner_layout(self) -> None:
        required_paths = [
            "transformer/config.json",
            "connectors/config.json",
            os.path.join("text_encoder", "config.json"),
        ]
        missing = [
            rel_path
            for rel_path in required_paths
            if not os.path.exists(os.path.join(self.refiner_root, rel_path))
        ]
        if missing:
            raise ValueError(
                "SANA-WM two-stage inference requires the upstream refiner "
                f"directory. Missing under {self.refiner_root}: {missing}"
            )
        if not os.path.exists(os.path.join(self.refiner_gemma_root, "config.json")):
            raise ValueError(
                "SANA-WM refiner text encoder not found at "
                f"{self.refiner_gemma_root}."
            )

    @staticmethod
    def _prompts_for_batch(batch: Req, batch_size: int) -> list[str]:
        prompt = batch.extra.get("refiner_prompt") if batch.extra else None
        if prompt is None:
            prompt = batch.prompt
        if isinstance(prompt, str):
            return [prompt] * batch_size
        if isinstance(prompt, list) and all(isinstance(item, str) for item in prompt):
            if len(prompt) == batch_size:
                return prompt
            if len(prompt) == 1:
                return prompt * batch_size
        raise ValueError(
            "SANA-WM refiner requires a string prompt or one prompt per batch item."
        )

    @staticmethod
    def _seeds_for_batch(batch: Req, batch_size: int) -> list[int]:
        if batch.seeds is not None:
            if len(batch.seeds) == batch_size:
                return [int(seed) for seed in batch.seeds]
            if len(batch.seeds) == 1:
                return [int(batch.seeds[0])] * batch_size
        seed = getattr(batch, "seed", 42)
        return [int(seed)] * batch_size

    @torch.inference_mode()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if batch.latents is None:
            raise ValueError("SANA-WM refiner requires batch.latents from stage 1.")
        if batch.latents.ndim != 5:
            raise ValueError(
                "SANA-WM refiner expects 5D latents shaped (B, C, T, H, W), "
                f"got {tuple(batch.latents.shape)}."
            )

        refiner = self._load_refiner()
        batch_size = int(batch.latents.shape[0])
        prompts = self._prompts_for_batch(batch, batch_size)
        seeds = self._seeds_for_batch(batch, batch_size)
        fps = float(getattr(batch, "fps", 16) or 16)
        progress = not bool(getattr(batch, "is_warmup", False))

        refined_latents: list[torch.Tensor] = []
        for item_idx, (prompt, seed) in enumerate(zip(prompts, seeds, strict=True)):
            refined = refiner.refine_latents(
                batch.latents[item_idx : item_idx + 1],
                prompt,
                fps=fps,
                seed=seed,
                progress=progress and batch_size == 1,
            )
            refined_latents.append(refined)

        batch.latents = torch.cat(refined_latents, dim=0).to(
            device=batch.latents.device, dtype=batch.latents.dtype
        )
        batch.extra["sana_wm_refiner_applied"] = True
        self.log_info("SANA-WM LTX-2 refiner applied to stage-1 latents.")
        return batch


class SanaWMRefinerDecodingStage(DecodingStage):
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
        # Match NVlabs inference_sana_wm.py: decode with the clean sink anchor,
        # then remove that first frame from the returned video.
        return frames[:, :, 1:].contiguous()


class _DiffusersLTX2VideoRefiner(nn.Module):
    """Thin adapter around diffusers LTX-2 modules for SANA-WM refinement."""

    def __init__(
        self,
        refiner_root: str | Path,
        gemma_root: str | Path,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
        text_max_sequence_length: int = 1024,
    ) -> None:
        super().__init__()
        self.refiner_root = Path(refiner_root)
        self.gemma_root = Path(gemma_root)
        self.dtype = dtype
        self.device = torch.device(device)
        self.text_max_sequence_length = int(text_max_sequence_length)
        self.transformer, self.connectors = self._load_diffusers_components()

    def _load_diffusers_components(self) -> tuple[nn.Module, nn.Module]:
        from diffusers.models.transformers.transformer_ltx2 import (
            LTX2VideoTransformer3DModel,
        )
        from diffusers.pipelines.ltx2 import LTX2TextConnectors

        transformer = LTX2VideoTransformer3DModel.from_pretrained(
            self.refiner_root,
            subfolder="transformer",
            torch_dtype=self.dtype,
        ).eval()
        connectors = LTX2TextConnectors.from_pretrained(
            self.refiner_root,
            subfolder="connectors",
            torch_dtype=self.dtype,
        ).eval()
        return transformer, connectors

    @torch.inference_mode()
    def refine_latents(
        self,
        sana_latent: torch.Tensor,
        prompt: str,
        *,
        fps: float,
        sink_size: int = 1,
        seed: int = 42,
        progress: bool = True,
    ) -> torch.Tensor:
        if sana_latent.shape[2] <= sink_size:
            raise ValueError(
                f"Stage-1 latent has {sana_latent.shape[2]} frames but "
                f"sink_size={sink_size}."
            )

        self.transformer.to("cpu")
        _empty_cuda_cache()
        prompt_embeds, prompt_attention_mask = self._encode_prompt(prompt)

        self.transformer.to(self.device)
        z = sana_latent.to(device=self.device, dtype=self.dtype)
        sigmas = torch.tensor(
            STAGE_2_DISTILLED_SIGMA_VALUES,
            dtype=torch.float32,
            device=self.device,
        )
        start_sigma = float(sigmas[0])
        sink = z[:, :, :sink_size].contiguous()
        current = z[:, :, sink_size:].contiguous()
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        eps = torch.randn(
            current.shape, generator=generator, device=self.device, dtype=self.dtype
        )
        noisy = (1.0 - start_sigma) * current + start_sigma * eps

        step_iter = range(len(sigmas) - 1)
        if progress:
            from tqdm.auto import tqdm

            step_iter = tqdm(step_iter, desc="sana_wm_refiner", unit="step")

        for step_idx in step_iter:
            sigma = sigmas[step_idx]
            denoised = self._predict_current_x0(
                sink=sink,
                noisy_current=noisy,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                sigma=sigma,
                fps=fps,
            )
            noisy_tokens = _pack_latents(
                noisy,
                patch_size=self.transformer.config.patch_size,
                patch_size_t=self.transformer.config.patch_size_t,
            )
            velocity = (noisy_tokens.float() - denoised.float()) / sigma.float()
            next_tokens = noisy_tokens.float() + velocity * (
                sigmas[step_idx + 1] - sigma
            ).float()
            noisy = _unpack_latents(
                next_tokens.to(self.dtype),
                num_frames=noisy.shape[2],
                height=noisy.shape[3],
                width=noisy.shape[4],
                patch_size=self.transformer.config.patch_size,
                patch_size_t=self.transformer.config.patch_size_t,
            )

        return torch.cat([sink, noisy], dim=2)

    @torch.inference_mode()
    def _encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

        tokenizer = AutoTokenizer.from_pretrained(self.gemma_root)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        text_inputs = tokenizer(
            [prompt.strip()],
            padding="max_length",
            max_length=self.text_max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)

        text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
            self.gemma_root,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        ).eval()
        text_encoder.to(self.device)
        text_backbone = getattr(text_encoder, "model", text_encoder)
        outputs = text_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = torch.stack(outputs.hidden_states, dim=-1)
        sequence_lengths = attention_mask.sum(dim=-1)
        prompt_embeds = _pack_text_embeds(
            hidden_states,
            sequence_lengths,
            device=self.device,
            padding_side=tokenizer.padding_side,
        ).to(dtype=self.dtype)

        del text_encoder, text_backbone, outputs, hidden_states
        _empty_cuda_cache()

        self.connectors.to(self.device)
        connector_prompt_embeds, _, connector_attention_mask = self.connectors(
            prompt_embeds, attention_mask
        )
        self.connectors.to("cpu")
        del prompt_embeds, attention_mask
        _empty_cuda_cache()

        return connector_prompt_embeds.to(
            device=self.device, dtype=self.dtype
        ), connector_attention_mask.to(device=self.device)

    def _predict_current_x0(
        self,
        *,
        sink: torch.Tensor,
        noisy_current: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        sigma: torch.Tensor,
        fps: float,
    ) -> torch.Tensor:
        full_latent = torch.cat([sink, noisy_current], dim=2)
        batch_size, _, num_frames, height, width = full_latent.shape
        latent_tokens = _pack_latents(
            full_latent,
            patch_size=self.transformer.config.patch_size,
            patch_size_t=self.transformer.config.patch_size_t,
        )
        n_context_tokens = _pack_latents(
            sink,
            patch_size=self.transformer.config.patch_size,
            patch_size_t=self.transformer.config.patch_size_t,
        ).shape[1]

        raw_timestep = torch.zeros(
            batch_size,
            latent_tokens.shape[1],
            1,
            dtype=torch.float32,
            device=self.device,
        )
        raw_timestep[:, n_context_tokens:, 0] = sigma.float()
        model_timestep = raw_timestep.squeeze(-1) * float(
            self.transformer.config.timestep_scale_multiplier
        )

        velocity = self._forward_video_only(
            hidden_states=latent_tokens,
            encoder_hidden_states=prompt_embeds,
            timestep=model_timestep,
            encoder_attention_mask=prompt_attention_mask,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
            n_context_tokens=n_context_tokens,
        )
        denoised = latent_tokens.float() - velocity.float() * raw_timestep
        return denoised[:, n_context_tokens:, :].to(self.dtype)

    def _forward_video_only(
        self,
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
        transformer = self.transformer
        batch_size = hidden_states.size(0)

        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(hidden_states.dtype)
            ) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

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
            hidden_states = _forward_video_block(
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


def _forward_video_block(
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

    attn_hidden_states = _streaming_self_attention(
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
    return hidden_states + block.ff(norm_hidden_states) * gate_mlp


def _streaming_self_attention(
    *,
    attn: nn.Module,
    hidden_states: torch.Tensor,
    query_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    n_context_tokens: int,
) -> torch.Tensor:
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
    return attn.to_out[1](hidden_states)


def _pack_text_embeds(
    text_hidden_states: torch.Tensor,
    sequence_lengths: torch.Tensor,
    device: str | torch.device,
    padding_side: str = "left",
    scale_factor: int = 8,
    eps: float = 1e-6,
) -> torch.Tensor:
    batch_size, seq_len, hidden_dim, _ = text_hidden_states.shape
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

    masked_hidden_states = text_hidden_states.masked_fill(~mask, 0.0)
    valid_positions = (sequence_lengths * hidden_dim).view(batch_size, 1, 1, 1)
    masked_mean = masked_hidden_states.sum(dim=(1, 2), keepdim=True) / (
        valid_positions + eps
    )
    x_min = text_hidden_states.masked_fill(~mask, float("inf")).amin(
        dim=(1, 2), keepdim=True
    )
    x_max = text_hidden_states.masked_fill(~mask, float("-inf")).amax(
        dim=(1, 2), keepdim=True
    )
    normalized = (text_hidden_states - masked_mean) / (x_max - x_min + eps)
    normalized = normalized * scale_factor
    normalized = normalized.flatten(2)
    mask_flat = mask.squeeze(-1).expand(-1, -1, normalized.shape[-1])
    return normalized.masked_fill(~mask_flat, 0.0).to(dtype=original_dtype)


def _pack_latents(
    latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1
) -> torch.Tensor:
    batch_size, _, num_frames, height, width = latents.shape
    post_patch_num_frames = num_frames // patch_size_t
    post_patch_height = height // patch_size
    post_patch_width = width // patch_size
    latents = latents.reshape(
        batch_size,
        -1,
        post_patch_num_frames,
        patch_size_t,
        post_patch_height,
        patch_size,
        post_patch_width,
        patch_size,
    )
    return latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)


def _unpack_latents(
    latents: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
    patch_size: int = 1,
    patch_size_t: int = 1,
) -> torch.Tensor:
    batch_size = latents.size(0)
    latents = latents.reshape(
        batch_size,
        num_frames,
        height,
        width,
        -1,
        patch_size_t,
        patch_size,
        patch_size,
    )
    return (
        latents.permute(0, 4, 1, 5, 2, 6, 3, 7)
        .flatten(6, 7)
        .flatten(4, 5)
        .flatten(2, 3)
    )


def _empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def default_sana_wm_refiner_dtype(server_args: ServerArgs) -> torch.dtype:
    return PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision]
