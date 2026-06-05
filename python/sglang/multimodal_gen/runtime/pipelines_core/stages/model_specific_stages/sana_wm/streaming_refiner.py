# SPDX-License-Identifier: Apache-2.0
"""SANA-WM chunked streaming LTX-2 refiner — S2b.

Runs the refiner block-by-block carrying a refiner KV cache: a ``sink`` prefix
(captured pre-RoPE, re-RoPE'd per block at shifted positions) plus a sliding
``history`` of refined-block K/V (post-RoPE), injected into each ``attn1`` as a
KV prefix. Ports the reference NVlabs ``RefinerChunkRunner``
(minimal-sanawm/refiner.py), operating on mg's already-loaded diffusers LTX-2
refiner transformer.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.models.dits.sana_wm_refiner_transformer import (
    pack_latents,
    unpack_latents,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from . import parity_probe
from .refiner import (
    STAGE_2_DISTILLED_SIGMA_VALUES,
    SanaWMLTX2RefinerStage,
    _as_additive_attention_mask,
    _unwrap_diffusers_ltx2_refiner,
    log_sana_wm_tensor_stats,
    sana_wm_skip_refiner_enabled,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


# --------------------------------------------------------------------------- #
# Per-attn1 KV-prefix / capture hooks (port of refiner.py:753-788)
# --------------------------------------------------------------------------- #
def set_kv_prefix_on_blocks(transformer: nn.Module, kv_prefix_per_layer) -> None:
    if kv_prefix_per_layer is None:
        clear_kv_prefix_on_blocks(transformer)
        return
    for block, prefix in zip(transformer.transformer_blocks, kv_prefix_per_layer):
        block.attn1._tf_kv_prefix = prefix


def clear_kv_prefix_on_blocks(transformer: nn.Module) -> None:
    for block in transformer.transformer_blocks:
        block.attn1._tf_kv_prefix = None


def set_capture_flag_on_blocks(transformer: nn.Module, mode: str, *, enable: bool) -> None:
    if mode == "pre_rope":
        attr, clear_attr = "_kv_cache_capture", "_cached_kv_pre"
    elif mode == "post_rope":
        attr, clear_attr = "_tf_capture_kv", "_cached_kv_post"
    else:
        raise ValueError(f"unsupported capture mode: {mode}")
    for block in transformer.transformer_blocks:
        setattr(block.attn1, attr, bool(enable))
        if enable and hasattr(block.attn1, clear_attr):
            setattr(block.attn1, clear_attr, None)


def collect_captured_kv_from_blocks(transformer: nn.Module, mode: str):
    attr = "_cached_kv_pre" if mode == "pre_rope" else "_cached_kv_post"
    out = []
    for block in transformer.transformer_blocks:
        cached = getattr(block.attn1, attr, None)
        if cached is None:
            raise RuntimeError(f"missing captured KV on {attr}")
        out.append(cached)
        setattr(block.attn1, attr, None)
    return out


# --------------------------------------------------------------------------- #
# Absolute-position RoPE (port of refiner.py:721-750)
# --------------------------------------------------------------------------- #
def build_rotary_emb_for_absolute_positions(
    *, transformer, batch_size, frame_positions, height, width, device, fps
):
    rope = transformer.rope
    patch_size_t = int(rope.patch_size_t)
    patch_size = int(rope.patch_size)
    f_positions = torch.tensor(frame_positions, dtype=torch.float32, device=device)
    if patch_size_t > 1:
        f_positions = f_positions[::patch_size_t]
    grid_h = torch.arange(0, height, patch_size, dtype=torch.float32, device=device)
    grid_w = torch.arange(0, width, patch_size, dtype=torch.float32, device=device)
    grid = torch.meshgrid(f_positions, grid_h, grid_w, indexing="ij")
    grid = torch.stack(grid, dim=0)
    patch_delta = torch.tensor(
        (patch_size_t, patch_size, patch_size), dtype=grid.dtype, device=device
    )
    patch_ends = grid + patch_delta.view(3, 1, 1, 1)
    latent_coords = (
        torch.stack([grid, patch_ends], dim=-1)
        .flatten(1, 3)
        .unsqueeze(0)
        .repeat(batch_size, 1, 1, 1)
    )
    scale = torch.tensor(rope.scale_factors, device=device)
    broadcast_shape = [1] * latent_coords.ndim
    broadcast_shape[1] = -1
    pixel_coords = latent_coords * scale.view(*broadcast_shape)
    pixel_coords[:, 0, ...] = (
        pixel_coords[:, 0, ...] + rope.causal_offset - rope.scale_factors[0]
    ).clamp(min=0)
    pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / float(fps)
    return rope(pixel_coords, device=device)


# --------------------------------------------------------------------------- #
# Self-attention with KV-prefix injection + capture (port of refiner.py:464-576)
# --------------------------------------------------------------------------- #
def streaming_self_attention(*, attn, hidden_states, query_rotary_emb, n_context_tokens):
    sequence_length = hidden_states.shape[1]
    has_streaming_hooks = (
        getattr(attn, "_kv_cache_capture", False)
        or getattr(attn, "_tf_capture_kv", False)
        or getattr(attn, "_tf_kv_prefix", None) is not None
    )
    if n_context_tokens >= sequence_length and not has_streaming_hooks:
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

    # diffusers 0.38+ always defines `to_gate_logits`; 0.37 only on gated variants
    # (the SANA-WM refiner is ungated) -> getattr so 0.37 works too.
    _to_gate_logits = getattr(attn, "to_gate_logits", None)
    gate_logits = _to_gate_logits(hidden_states) if _to_gate_logits is not None else None
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)
    query = attn.norm_q(query)
    key = attn.norm_k(key)
    if getattr(attn, "_kv_cache_capture", False):
        attn._cached_kv_pre = (key.detach().clone(), value.detach().clone())

    if attn.rope_type == "interleaved":
        query = apply_interleaved_rotary_emb(query, query_rotary_emb)
        key = apply_interleaved_rotary_emb(key, query_rotary_emb)
    elif attn.rope_type == "split":
        query = apply_split_rotary_emb(query, query_rotary_emb)
        key = apply_split_rotary_emb(key, query_rotary_emb)
    else:
        raise ValueError(f"Unsupported LTX-2 RoPE type: {attn.rope_type}")
    if getattr(attn, "_tf_capture_kv", False):
        attn._cached_kv_post = (key.detach().clone(), value.detach().clone())

    tf_prefix = getattr(attn, "_tf_kv_prefix", None)
    if isinstance(tf_prefix, dict) and tf_prefix.get("mode") == "rf_shifted_sink":
        prefix_k_parts = []
        prefix_v_parts = []
        sink_k_pre = tf_prefix.get("sink_k_pre")
        sink_v = tf_prefix.get("sink_v")
        if sink_k_pre is not None and sink_v is not None and sink_k_pre.shape[1] > 0:
            sink_pe = tf_prefix.get("sink_pe")
            if sink_pe is None:
                raise RuntimeError("rf_shifted_sink prefix requires sink_pe")
            if attn.rope_type == "interleaved":
                sink_k = apply_interleaved_rotary_emb(sink_k_pre.to(key.dtype), sink_pe)
            else:
                sink_k = apply_split_rotary_emb(sink_k_pre.to(key.dtype), sink_pe)
            prefix_k_parts.append(sink_k)
            prefix_v_parts.append(sink_v.to(value.dtype))
        history_k = tf_prefix.get("history_k")
        history_v = tf_prefix.get("history_v")
        if history_k is not None and history_v is not None and history_k.shape[1] > 0:
            prefix_k_parts.append(history_k.to(key.dtype))
            prefix_v_parts.append(history_v.to(value.dtype))
        if prefix_k_parts:
            key = torch.cat([*prefix_k_parts, key], dim=1)
            value = torch.cat([*prefix_v_parts, value], dim=1)

    query = query.unflatten(2, (attn.heads, -1))
    key = key.unflatten(2, (attn.heads, -1))
    value = value.unflatten(2, (attn.heads, -1))
    processor = attn.processor
    backend = getattr(processor, "_attention_backend", None)
    parallel_config = getattr(processor, "_parallel_config", None)
    if n_context_tokens <= 0 or n_context_tokens >= query.shape[1]:
        hidden_states = dispatch_attention_fn(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
            backend=backend, parallel_config=parallel_config,
        )
    else:
        context = dispatch_attention_fn(
            query[:, :n_context_tokens], key[:, :n_context_tokens], value[:, :n_context_tokens],
            attn_mask=None, dropout_p=0.0, is_causal=False,
            backend=backend, parallel_config=parallel_config,
        )
        current = dispatch_attention_fn(
            query[:, n_context_tokens:], key, value,
            attn_mask=None, dropout_p=0.0, is_causal=False,
            backend=backend, parallel_config=parallel_config,
        )
        hidden_states = torch.cat([context, current], dim=1)
    hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
    if gate_logits is not None:
        hidden_states = hidden_states.unflatten(2, (attn.heads, -1))
        hidden_states = hidden_states * (2.0 * torch.sigmoid(gate_logits)).unsqueeze(-1)
        hidden_states = hidden_states.flatten(2, 3)
    hidden_states = attn.to_out[0](hidden_states)
    return attn.to_out[1](hidden_states)


def forward_video_block(
    *, block, hidden_states, encoder_hidden_states, temb, video_rotary_emb,
    encoder_attention_mask, n_context_tokens,
):
    batch = hidden_states.size(0)
    norm_hidden_states = block.norm1(hidden_states)
    num_ada_params = block.scale_shift_table.shape[0]
    ada_values = block.scale_shift_table[None, None].to(temb.device) + temb.reshape(
        batch, temb.size(1), num_ada_params, -1
    )
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_values.unbind(dim=2)
    norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
    attn_hidden_states = streaming_self_attention(
        attn=block.attn1, hidden_states=norm_hidden_states,
        query_rotary_emb=video_rotary_emb, n_context_tokens=n_context_tokens,
    )
    hidden_states = hidden_states + attn_hidden_states * gate_msa
    norm_hidden_states = block.norm2(hidden_states)
    attn_hidden_states = block.attn2(
        norm_hidden_states, encoder_hidden_states=encoder_hidden_states,
        query_rotary_emb=None, attention_mask=encoder_attention_mask,
    )
    hidden_states = hidden_states + attn_hidden_states
    norm_hidden_states = block.norm3(hidden_states) * (1 + scale_mlp) + shift_mlp
    return hidden_states + block.ff(norm_hidden_states) * gate_mlp


class _RefinerCore:
    """Adapts the unwrapped diffusers refiner transformer to the
    DiffusersLTX2Refiner interface RefinerChunkRunner expects."""

    def __init__(self, transformer: nn.Module, device: torch.device, dtype: torch.dtype):
        self.transformer = transformer
        self.device = device
        self.dtype = dtype

    def _forward_video_only_with_rope(
        self, *, hidden_states, encoder_hidden_states, timestep, encoder_attention_mask,
        video_rotary_emb, n_context_tokens,
    ):
        transformer = self.transformer
        batch = hidden_states.size(0)
        if encoder_attention_mask is not None:
            encoder_attention_mask = _as_additive_attention_mask(
                encoder_attention_mask, hidden_states.dtype
            )
        hidden_states = transformer.proj_in(hidden_states)
        temb, embedded_timestep = transformer.time_embed(
            timestep.flatten(), batch_size=batch, hidden_dtype=hidden_states.dtype,
        )
        temb = temb.view(batch, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(batch, -1, embedded_timestep.size(-1))
        encoder_hidden_states = transformer.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch, -1, hidden_states.size(-1))
        for block in transformer.transformer_blocks:
            hidden_states = forward_video_block(
                block=block, hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states, temb=temb,
                video_rotary_emb=video_rotary_emb,
                encoder_attention_mask=encoder_attention_mask,
                n_context_tokens=n_context_tokens,
            )
        scale_shift = transformer.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        shift, scale = scale_shift[:, :, 0], scale_shift[:, :, 1]
        hidden_states = transformer.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        return transformer.proj_out(hidden_states)

    def _predict_x0_active_block(
        self, *, active, active_positions, sigma_cur, prompt_embeds,
        prompt_attention_mask, fps, kv_prefix_per_layer,
    ):
        ps = int(self.transformer.config.patch_size)
        pst = int(self.transformer.config.patch_size_t)
        latent_tokens = pack_latents(active, ps, pst)
        batch, seq_len, _ = latent_tokens.shape
        timestep = torch.full(
            (batch, seq_len),
            float(sigma_cur) * float(self.transformer.config.timestep_scale_multiplier),
            dtype=torch.float32, device=self.device,
        )
        video_rotary_emb = build_rotary_emb_for_absolute_positions(
            transformer=self.transformer, batch_size=batch, frame_positions=active_positions,
            height=int(active.shape[3]), width=int(active.shape[4]), device=self.device, fps=float(fps),
        )
        set_kv_prefix_on_blocks(self.transformer, kv_prefix_per_layer)
        try:
            velocity = self._forward_video_only_with_rope(
                hidden_states=latent_tokens, encoder_hidden_states=prompt_embeds,
                timestep=timestep, encoder_attention_mask=prompt_attention_mask,
                video_rotary_emb=video_rotary_emb, n_context_tokens=0,
            )
        finally:
            clear_kv_prefix_on_blocks(self.transformer)
        raw_sigma = torch.full((batch, seq_len, 1), float(sigma_cur), dtype=torch.float32, device=self.device)
        denoised = latent_tokens.float() - velocity.float() * raw_sigma
        return unpack_latents(
            denoised.to(self.dtype), num_frames=int(active.shape[2]),
            height=int(active.shape[3]), width=int(active.shape[4]), patch_size=ps, patch_size_t=pst,
        )

    def _capture_block_kv(
        self, *, clean_block, frame_positions, prompt_embeds, prompt_attention_mask,
        fps, capture_mode, kv_prefix_per_layer,
    ):
        ps = int(self.transformer.config.patch_size)
        pst = int(self.transformer.config.patch_size_t)
        latent_tokens = pack_latents(clean_block, ps, pst)
        batch, seq_len, _ = latent_tokens.shape
        timestep = torch.zeros(batch, seq_len, dtype=torch.float32, device=self.device)
        video_rotary_emb = build_rotary_emb_for_absolute_positions(
            transformer=self.transformer, batch_size=batch, frame_positions=frame_positions,
            height=int(clean_block.shape[3]), width=int(clean_block.shape[4]), device=self.device, fps=float(fps),
        )
        set_kv_prefix_on_blocks(self.transformer, kv_prefix_per_layer)
        set_capture_flag_on_blocks(self.transformer, capture_mode, enable=True)
        try:
            _ = self._forward_video_only_with_rope(
                hidden_states=latent_tokens, encoder_hidden_states=prompt_embeds,
                timestep=timestep, encoder_attention_mask=prompt_attention_mask,
                video_rotary_emb=video_rotary_emb, n_context_tokens=0,
            )
        finally:
            set_capture_flag_on_blocks(self.transformer, capture_mode, enable=False)
            clear_kv_prefix_on_blocks(self.transformer)
        return collect_captured_kv_from_blocks(self.transformer, capture_mode)


class RefinerChunkRunner:
    """Port of the reference RefinerChunkRunner (refiner.py:579-718)."""

    def __init__(
        self, refiner: _RefinerCore, *, prompt_embeds, prompt_attention_mask, fps,
        sigmas, source_sink_frames, block_size, kv_max_frames, seed, spatial_shape,
    ):
        self.refiner = refiner
        self.prompt_embeds = prompt_embeds
        self.prompt_attention_mask = prompt_attention_mask
        self.fps = float(fps)
        self.sigmas = sigmas
        self.source_sink_frames = int(source_sink_frames)
        self.block_size = int(block_size)
        self.kv_max_frames = int(kv_max_frames)
        self.max_history_frames = self.kv_max_frames - self.source_sink_frames
        self.generator = torch.Generator(device=refiner.device).manual_seed(int(seed))
        self.device = refiner.device
        self.dtype = refiner.dtype
        self.height, self.width = int(spatial_shape[0]), int(spatial_shape[1])
        transformer = refiner.transformer
        self.tokens_per_frame = (
            int(self.height // transformer.config.patch_size)
            * int(self.width // transformer.config.patch_size)
            * int(transformer.config.patch_size_t)
        )
        self.sink_kv_pre = None
        self.history_kv_post = [None] * len(transformer.transformer_blocks)
        self.history_frames = 0

    @torch.inference_mode()
    def refine_block(self, *, block_idx, clean_block, block_start, block_end, sink_seed_frames=None):
        # parity harness (env-gated, no-op in prod): per-block input/config/output
        # checksums; both the batch stage and the realtime stage run through here.
        _probe_dir = parity_probe.probe_dir(
            parity_probe.ENV_RT_DUMP, parity_probe.ENV_FORK_DUMP
        )
        _probe = None
        if _probe_dir:
            _ck = parity_probe.checksum
            _probe = {
                "block_idx": int(block_idx),
                "block_start": int(block_start),
                "block_end": int(block_end),
                "clean_block": _ck(clean_block),
                "sink_seed_frames": _ck(sink_seed_frames),
                "prompt_embeds": _ck(self.prompt_embeds),
                "prompt_attention_mask": _ck(self.prompt_attention_mask),
                "fps": self.fps,
                "sigmas": [float(s) for s in self.sigmas],
                "source_sink_frames": self.source_sink_frames,
                "block_size": self.block_size,
                "kv_max_frames": self.kv_max_frames,
                "history_frames": self.history_frames,
                "generator_state": float(
                    self.generator.get_state().double().sum().item()
                ),
            }
        del block_idx
        refiner = self.refiner
        if block_start < self.source_sink_frames:
            raise ValueError("refiner block overlaps source sink")
        if self.sink_kv_pre is None:
            if self.source_sink_frames == 0:
                self.sink_kv_pre = [(None, None) for _ in self.history_kv_post]
            elif sink_seed_frames is None:
                raise ValueError("first refine_block call requires sink_seed_frames")
            else:
                self.sink_kv_pre = refiner._capture_block_kv(
                    clean_block=sink_seed_frames.contiguous(),
                    frame_positions=list(range(self.source_sink_frames)),
                    prompt_embeds=self.prompt_embeds,
                    prompt_attention_mask=self.prompt_attention_mask,
                    fps=self.fps, capture_mode="pre_rope", kv_prefix_per_layer=None,
                )
        batch = int(clean_block.shape[0])
        sink_rope_offset = block_start - self.history_frames - self.source_sink_frames
        sink_pe = None
        if self.source_sink_frames > 0:
            sink_pe = build_rotary_emb_for_absolute_positions(
                transformer=refiner.transformer, batch_size=batch,
                frame_positions=list(range(sink_rope_offset, sink_rope_offset + self.source_sink_frames)),
                height=self.height, width=self.width, device=self.device, fps=self.fps,
            )
        kv_prefix_per_layer = []
        for layer_idx, sink_kv in enumerate(self.sink_kv_pre):
            history = self.history_kv_post[layer_idx]
            kv_prefix_per_layer.append({
                "mode": "rf_shifted_sink",
                "sink_k_pre": sink_kv[0], "sink_v": sink_kv[1], "sink_pe": sink_pe,
                "history_k": history[0] if history is not None else None,
                "history_v": history[1] if history is not None else None,
            })
        sigma0 = float(self.sigmas[0].item())
        eps = torch.randn(clean_block.shape, generator=self.generator, device=self.device, dtype=self.dtype)
        x_t = ((1.0 - sigma0) * clean_block.float() + sigma0 * eps.float()).to(self.dtype)
        active_positions = list(range(int(block_start), int(block_end)))
        for level in range(int(self.sigmas.numel()) - 1):
            sigma_cur = float(self.sigmas[level].item())
            sigma_next = float(self.sigmas[level + 1].item())
            pred_x0 = refiner._predict_x0_active_block(
                active=x_t, active_positions=active_positions, sigma_cur=sigma_cur,
                prompt_embeds=self.prompt_embeds, prompt_attention_mask=self.prompt_attention_mask,
                fps=self.fps, kv_prefix_per_layer=kv_prefix_per_layer,
            )
            if sigma_cur <= 1.0e-6:
                x_t = pred_x0.to(self.dtype)
            else:
                ratio = sigma_next / sigma_cur
                x_t = (ratio * x_t.float() + (1.0 - ratio) * pred_x0.float()).to(self.dtype)
        block_kv_post = refiner._capture_block_kv(
            clean_block=x_t, frame_positions=active_positions,
            prompt_embeds=self.prompt_embeds, prompt_attention_mask=self.prompt_attention_mask,
            fps=self.fps, capture_mode="post_rope", kv_prefix_per_layer=kv_prefix_per_layer,
        )
        for layer_idx, new_kv in enumerate(block_kv_post):
            old = self.history_kv_post[layer_idx]
            self.history_kv_post[layer_idx] = new_kv if old is None else (
                torch.cat([old[0], new_kv[0]], dim=1),
                torch.cat([old[1], new_kv[1]], dim=1),
            )
        self.history_frames += int(block_end - block_start)
        if self.max_history_frames > 0 and self.history_frames > self.max_history_frames:
            keep_tokens = self.max_history_frames * self.tokens_per_frame
            for layer_idx, old in enumerate(self.history_kv_post):
                if old is not None:
                    self.history_kv_post[layer_idx] = (old[0][:, -keep_tokens:], old[1][:, -keep_tokens:])
            self.history_frames = self.max_history_frames
        if _probe is not None:
            _probe["refined"] = parity_probe.checksum(x_t)
            parity_probe.dump_obj(
                _probe_dir, f"refiner_probe_{_probe['block_idx']:03d}", _probe
            )
        return x_t


class SanaWMStreamingRefinerStage(SanaWMLTX2RefinerStage):
    """Chunked streaming LTX-2 refiner — refines stage-1 latents block-by-block
    carrying a sink/history KV cache. Inherits loading/encode/residency from
    SanaWMLTX2RefinerStage; only the refine loop changes."""

    def __init__(self, *, block_size: int = 3, kv_max_frames: int = 11,
                 sink_size: int = 1, seed: int = 42, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.block_size = int(block_size)
        self.kv_max_frames = int(kv_max_frames)
        self.sink_size = int(sink_size)
        self.seed = int(seed)

    @torch.inference_mode()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if sana_wm_skip_refiner_enabled():
            return batch
        if batch.latents is None or batch.latents.ndim != 5:
            raise ValueError("SANA-WM streaming refiner expects 5D stage-1 latents.")
        device = get_local_torch_device()
        latents = batch.latents.to(device=device, dtype=self.dtype).clone()
        B, C, T, H, W = latents.shape
        prompt = self._prompts_for_batch(batch, B)[0]
        fps = float((batch.extra or {}).get("fps", getattr(batch, "fps", 16.0)) or 16.0)

        prompt_embeds, prompt_mask = self._encode_prompt(prompt, device)
        sigmas = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=device)

        n_active = T - self.sink_size
        if n_active <= 0:
            self.log_info("SANA-WM streaming refiner: no active frames (T=%d <= sink=%d); skipping.",
                          T, self.sink_size)
            return batch
        n_blocks = math.ceil(n_active / self.block_size)
        self.log_info(
            "SANA-WM streaming refiner: latent=%s, sink=%d, block=%d, blocks=%d, kv_max=%d, seed=%d",
            tuple(latents.shape), self.sink_size, self.block_size, n_blocks, self.kv_max_frames, self.seed,
        )

        import time
        t0 = time.perf_counter()
        with self.use_declared_component(
            component_name="transformer_2", module=self.transformer
        ) as transformer_mod:
            self.transformer = transformer_mod
            unwrapped = _unwrap_diffusers_ltx2_refiner(self.transformer)
            core = _RefinerCore(unwrapped, device, self.dtype)
            runner = RefinerChunkRunner(
                core, prompt_embeds=prompt_embeds, prompt_attention_mask=prompt_mask,
                fps=fps, sigmas=sigmas, source_sink_frames=self.sink_size,
                block_size=self.block_size, kv_max_frames=self.kv_max_frames,
                seed=self.seed, spatial_shape=(H, W),
            )
            for i in range(n_blocks):
                start_f = self.sink_size + i * self.block_size
                end_f = min(start_f + self.block_size, T)
                sink_seed = latents[:, :, : self.sink_size] if i == 0 else None
                refined = runner.refine_block(
                    block_idx=i, clean_block=latents[:, :, start_f:end_f].contiguous(),
                    block_start=start_f, block_end=end_f, sink_seed_frames=sink_seed,
                )
                latents[:, :, start_f:end_f] = refined.to(latents.dtype)

        log_sana_wm_tensor_stats("stream.refiner.latents", latents)
        self.log_info(
            "SANA-WM streaming refiner applied (%d blocks) in %.4f s.",
            n_blocks, time.perf_counter() - t0,
        )
        batch.latents = latents
        if batch.extra is None:
            batch.extra = {}
        batch.extra["sana_wm_refiner_applied"] = True
        return batch
