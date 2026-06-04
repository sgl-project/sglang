# SPDX-License-Identifier: Apache-2.0
"""SANA-WM streaming (self-forcing) denoising stage — S1c.

Drives the DiT's chunk-causal ``forward_long`` autoregressively: the video is
generated chunk-by-chunk, each chunk denoised with a short self-forcing schedule
while a rolling per-block KV cache carries recurrent GDN state (and a softmax
K/V concat-window) across chunks.

Key differences from the dense ``SanaWMDenoisingStage`` (one-shot bidirectional):
  * a fresh ``FlowMatchEulerDiscreteScheduler(shift=1.0)`` with explicit sigmas
    (NOT the request scheduler, whose shift would warp the sigmas);
  * ``transformer.forward_long`` per chunk with a per-block 10-slot ``kv_cache``;
  * the KV accumulator concatenates softmax cache slots on **dim=1** (our
    ``(B, N, H, D)`` softmax cache layout), not the reference's dim=2;
  * the condition frame (frame 0) is derived from a condition mask, re-pinned
    every step in chunk 0.
"""

from __future__ import annotations

import time

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.sana_wm import (
    _NUM_STREAM_CACHE_SLOTS,
    _SLOT_CAM_K,
    _SLOT_CAM_V,
    _SLOT_FFN_TCONV,
    _SLOT_K,
    _SLOT_SHORTCONV,
    _SLOT_TYPE_FLAG,
    _SLOT_V,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_sana_wm_self_forcing import (
    SanaWMSelfForcingSamplerConfig,
    SanaWMSelfForcingScheduler,
)
from .sana_wm_base import (
    SanaWMDenoisingStage,
    _align_sana_wm_cfg_text_conditions,
    _cat_optional_tensors,
    _first_tensor,
    _to_device_dtype,
    log_sana_wm_tensor_stats,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

# --- debug parity harness (gated by env; no-op in production) ---
import os as _os
from pathlib import Path as _Path

_SANAWM_INJECT_DIR = _os.environ.get("SANAWM_INJECT_DIR")
_SANAWM_FORK_DUMP_DIR = _os.environ.get("SANAWM_FORK_DUMP_DIR")


class SanaWMStreamingDenoisingStage(SanaWMDenoisingStage):
    """Autoregressive self-forcing streaming variant of SanaWMDenoisingStage."""

    # Chunk schedule + KV cache: delegate to SanaWMSelfForcingScheduler.
    @staticmethod
    def _autoregressive_segments(total_frames: int, num_frame_per_block: int) -> list[int]:
        return SanaWMSelfForcingScheduler.create_autoregressive_segments(
            total_frames, num_frame_per_block
        )

    @staticmethod
    def _accumulate_kv_cache(
        kv_cache: list,
        chunk_idx: int,
        chunk_indices: list[int],
        num_cached_blocks: int,
        sink_token: bool,
        num_blocks: int,
    ) -> tuple[list, int]:
        return SanaWMSelfForcingScheduler.accumulate_kv_cache(
            kv_cache, chunk_idx, chunk_indices, num_cached_blocks, sink_token, num_blocks
        )

    @staticmethod
    def _evict_stale_kv_cache(
        kv_cache: list, chunk_idx: int, valid: list[int], num_cached_blocks: int, num_blocks: int
    ) -> None:
        SanaWMSelfForcingScheduler.evict_stale_kv_cache(
            kv_cache, chunk_idx, valid, num_cached_blocks, num_blocks
        )

    # ----------------------------------------------------------------- #
    # Streaming denoising loop
    # ----------------------------------------------------------------- #
    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if batch.latents is None or batch.latents.ndim != 5:
            raise ValueError(
                "SANA-WM streaming denoising expects 5D latents (B, C, T, H, W)."
            )

        pcfg = server_args.pipeline_config
        device = get_local_torch_device()
        target_dtype = PRECISION_TO_TYPE.get(getattr(pcfg, "dit_precision", "bf16"), torch.bfloat16)

        # .clone() detaches from the loader's InferenceMode tensor so the
        # per-chunk in-place latent updates below are allowed.
        latents = batch.latents.to(device=device, dtype=target_dtype).clone()
        init_latents = latents.clone()
        B, C, total_frames, H, W = latents.shape

        def _iload(_name):
            return torch.load(f"{_SANAWM_INJECT_DIR}/{_name}.pt", map_location=device)

        def _fdump(_name, _t):
            if _SANAWM_FORK_DUMP_DIR and _t is not None:
                _Path(_SANAWM_FORK_DUMP_DIR).mkdir(parents=True, exist_ok=True)
                torch.save(_t.detach().float().cpu(), f"{_SANAWM_FORK_DUMP_DIR}/{_name}.pt")

        if _SANAWM_INJECT_DIR:  # parity harness: run the OFFICIAL's exact stage-1 inputs
            latents = _iload("z_full_initial").to(device=device, dtype=target_dtype).clone()
            init_latents = latents.clone()
            B, C, total_frames, H, W = latents.shape

        _fdump("init_noise", init_latents)  # parity harness: seeded pre-noise (cond @ frame 0)

        sampler_cfg = SanaWMSelfForcingSamplerConfig.from_pipeline_config(pcfg)
        num_frame_per_block = sampler_cfg.num_frame_per_block
        num_cached_blocks = sampler_cfg.num_cached_blocks
        sink_token = sampler_cfg.sink_token
        explicit_sigmas = SanaWMSelfForcingScheduler.build_per_chunk_sigmas(
            sampler_cfg.denoising_step_list
        )

        # Streaming uses its OWN cfg scale (official StreamingGenerationConfig.cfg_scale=1.0
        # => no CFG on the distilled 4-step model). The general guidance_scale (e.g. 4.5)
        # is for the dense path; using it here ran CFG=4.5 vs the reference's none.
        cfg_scale = sampler_cfg.streaming_cfg_scale
        do_cfg = bool(batch.do_classifier_free_guidance) and cfg_scale > 1.0
        if server_args.enable_cfg_parallel and do_cfg:
            raise NotImplementedError(
                "SANA-WM streaming does not support CFG parallel; run replicated."
            )

        # --- text conditioning ---
        pos_embeds = _to_device_dtype(
            _first_tensor(pcfg.get_pos_prompt_embeds(batch)), device=device, dtype=target_dtype
        )
        pos_mask = _to_device_dtype(_first_tensor(batch.prompt_attention_mask), device=device)
        if pos_embeds is None:
            raise ValueError("SANA-WM streaming requires positive prompt embeds.")
        neg_embeds = neg_mask = None
        if do_cfg:
            neg_embeds = _to_device_dtype(
                _first_tensor(pcfg.get_neg_prompt_embeds(batch)), device=device, dtype=target_dtype
            )
            neg_mask = _to_device_dtype(_first_tensor(batch.negative_attention_mask), device=device)
            if neg_embeds is None:
                raise ValueError("SANA-WM streaming CFG requires negative prompt embeds.")
            pos_embeds, neg_embeds, pos_mask, neg_mask = _align_sana_wm_cfg_text_conditions(
                pos_embeds, neg_embeds, pos_mask, neg_mask
            )
        embeds_in = torch.cat([neg_embeds, pos_embeds], dim=0) if do_cfg else pos_embeds
        mask_in = _cat_optional_tensors(neg_mask, pos_mask) if do_cfg else pos_mask
        if _SANAWM_INJECT_DIR and not do_cfg:
            _cond = _iload("cond").to(device=device, dtype=target_dtype)
            while _cond.dim() > embeds_in.dim():
                _cond = _cond.squeeze(1)
            embeds_in = _cond
            mask_in = _iload("cond_mask").to(device=device)

        # --- camera / plücker (FULL length; forward_long windows internally) ---
        extra = batch.extra or {}
        camera_conditions = _to_device_dtype(
            extra.get("camera_conditions"), device=device, dtype=target_dtype
        )
        chunk_plucker = _to_device_dtype(
            extra.get("chunk_plucker"), device=device, dtype=target_dtype
        )
        cam_in = (
            torch.cat([camera_conditions, camera_conditions], dim=0)
            if do_cfg and camera_conditions is not None
            else camera_conditions
        )
        plk_in = (
            torch.cat([chunk_plucker, chunk_plucker], dim=0)
            if do_cfg and chunk_plucker is not None
            else chunk_plucker
        )
        if _SANAWM_INJECT_DIR:
            cam_in = _iload("raymap").to(device=device, dtype=target_dtype)
            plk_in = _iload("chunk_plucker").to(device=device, dtype=target_dtype)
            if do_cfg:
                cam_in = torch.cat([cam_in, cam_in], dim=0)
                plk_in = torch.cat([plk_in, plk_in], dim=0)

        # parity harness: full-length conditioning fed to forward_long (windowed
        # internally per chunk via [start_f:end_f]).
        _fdump("cond_embeds", embeds_in)
        _fdump("cond_mask", mask_in)
        _fdump("cond_camera", cam_in)
        _fdump("cond_plucker", plk_in)

        scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0)

        chunk_indices = self._autoregressive_segments(total_frames, num_frame_per_block)
        num_chunks = len(chunk_indices) - 1
        if num_chunks < 1:
            raise ValueError(
                f"streaming needs >= {num_frame_per_block} latent frames, got {total_frames}."
            )

        start_time = time.perf_counter()
        with self.use_declared_component(
            component_name="transformer", module=self.transformer
        ) as transformer:
            assert transformer is not None
            self.transformer = transformer
            num_blocks = len(transformer.blocks)
            if _SANAWM_FORK_DUMP_DIR:  # parity harness: weights fingerprint
                _fp = {
                    n: float(p.detach().float().abs().sum().item())
                    for n, p in transformer.named_parameters()
                }
                _Path(_SANAWM_FORK_DUMP_DIR).mkdir(parents=True, exist_ok=True)
                torch.save(_fp, f"{_SANAWM_FORK_DUMP_DIR}/dit_fingerprint.pt")
            kv_cache = [
                [[None] * _NUM_STREAM_CACHE_SLOTS for _ in range(num_blocks)]
                for _ in range(num_chunks)
            ]

            self.log_info(
                "SANA-WM streaming denoise: latent=%s, chunks=%d (block=%d frames), "
                "steps/chunk=%d, cfg=%s",
                tuple(latents.shape),
                num_chunks,
                num_frame_per_block,
                len(explicit_sigmas),
                do_cfg,
            )

            for chunk_idx in self.progress_bar(range(num_chunks)):
                chunk_kv, sink_num = self._accumulate_kv_cache(
                    kv_cache, chunk_idx, chunk_indices, num_cached_blocks, sink_token, num_blocks
                )
                start_f = chunk_indices[chunk_idx]
                end_f = chunk_indices[chunk_idx + 1]
                chunk_frames = end_f - start_f
                frame_index = (
                    torch.arange(start_f, end_f, device=device, dtype=torch.long)
                    if sink_num > 0
                    else None
                )

                # Condition mask: frame 0 only (chunk 0). Re-pinned each step.
                cond_mask = torch.zeros(
                    B, C, chunk_frames, H, W, device=device, dtype=torch.float32
                )
                cond_local = []
                if start_f == 0:
                    cond_mask[:, :, 0] = 1.0
                    cond_local = [0]

                scheduler.set_timesteps(sigmas=explicit_sigmas, device=device)
                for t in scheduler.timesteps:
                    chunk_lat = latents[:, :, start_f:end_f]
                    lat_in = torch.cat([chunk_lat, chunk_lat], dim=0) if do_cfg else chunk_lat
                    ts_tensor = (1.0 - cond_mask) * t.to(
                        device=device, dtype=torch.float32
                    ).view(1, 1, 1, 1, 1)
                    ts_in = torch.cat([ts_tensor, ts_tensor], dim=0) if do_cfg else ts_tensor
                    model_ts = ts_in[:, :1, :, 0, 0]  # (B|2B, chunk_frames)

                    with set_forward_context(
                        current_timestep=int(t.item()) if t.ndim == 0 else 0,
                        attn_metadata=None,
                        forward_batch=batch,
                    ):
                        noise_pred, _ = transformer.forward_long(
                            hidden_states=lat_in.to(target_dtype),
                            encoder_hidden_states=embeds_in,
                            timestep=model_ts,
                            encoder_attention_mask=mask_in,
                            camera_conditions=cam_in,
                            chunk_plucker=plk_in,
                            kv_cache=chunk_kv,
                            save_kv_cache=False,
                            start_f=start_f,
                            end_f=end_f,
                            frame_index=frame_index,
                        )
                    if do_cfg:
                        noise_uncond, noise_text = noise_pred.chunk(2)
                        noise_pred = noise_uncond + cfg_scale * (noise_text - noise_uncond)

                    if chunk_idx == 0:  # parity harness: per-step model output
                        _fdump(f"noise_pred_c0_t{int(t.item())}", noise_pred)

                    denoised = scheduler.step(
                        -noise_pred.reshape(B, C, -1).transpose(1, 2),
                        t,
                        chunk_lat.reshape(B, C, -1).transpose(1, 2),
                        per_token_timesteps=ts_tensor.reshape(B, C, -1)[:, 0],
                        return_dict=False,
                    )[0]
                    latents[:, :, start_f:end_f] = (
                        denoised.transpose(1, 2).reshape(chunk_lat.shape).to(latents.dtype)
                    )
                    for loc in cond_local:
                        latents[:, :, start_f + loc] = init_latents[:, :, start_f + loc]

                # Clean pass (t=0) to write this chunk's KV for the next chunk.
                chunk_lat = latents[:, :, start_f:end_f]
                lat_in = torch.cat([chunk_lat, chunk_lat], dim=0) if do_cfg else chunk_lat
                ts_zero = torch.zeros(
                    lat_in.shape[0], 1, chunk_frames, device=device, dtype=torch.float32
                )
                with set_forward_context(
                    current_timestep=0, attn_metadata=None, forward_batch=batch
                ):
                    _, updated_cache = transformer.forward_long(
                        hidden_states=lat_in.to(target_dtype),
                        encoder_hidden_states=embeds_in,
                        timestep=ts_zero,
                        encoder_attention_mask=mask_in,
                        camera_conditions=cam_in,
                        chunk_plucker=plk_in,
                        kv_cache=chunk_kv,
                        save_kv_cache=True,
                        start_f=start_f,
                        end_f=end_f,
                        frame_index=frame_index,
                    )
                kv_cache[chunk_idx] = updated_cache
                _fdump(f"stage1_{chunk_idx:03d}_{start_f}_{end_f}", latents[:, :, start_f:end_f])

        log_sana_wm_tensor_stats("stream.output_latents", latents)
        self.log_info(
            "SANA-WM streaming denoise finished in %.4f s; first_frame_max_delta=%.6g",
            time.perf_counter() - start_time,
            float((latents[:, :, :1] - init_latents[:, :, :1]).abs().max().item()),
        )
        batch.latents = pcfg.post_denoising_loop(latents, batch)
        return batch


class SanaWMStreamingDecodingStage(DecodingStage):
    """Streaming causal-VAE decode.

    Decodes the latents chunk-by-chunk over the SAME autoregressive grid the
    denoise stage used, carrying a per-conv decoder cache across chunks so the
    causal LTX-2 VAE produces seam-free frames as generation proceeds (the
    `decode_per_frame_with_cache` equivalent at chunk granularity). Subclasses
    DecodingStage directly (NOT SanaWMDecodingStage, whose long-video config
    re-enables the stateless tiled decode). `forward` is inherited; only `decode`
    is overridden.
    """

    @torch.no_grad()
    def decode(
        self,
        latents: torch.Tensor,
        server_args: ServerArgs,
        *,
        vae_dtype: torch.dtype,
    ) -> torch.Tensor:
        if not hasattr(self.vae, "decode_chunk"):
            raise ValueError(
                "SANA-WM streaming decode requires AutoencoderKLCausalLTX2Video "
                "(decode_chunk). Point --component_paths.vae at the ltx2_causal_vae "
                "weights when streaming."
            )
        device = get_local_torch_device()
        latents = latents.to(device)
        pcfg = server_args.pipeline_config
        num_frame_per_block = int(getattr(pcfg, "num_frame_per_block", 3))
        total_frames = latents.shape[2]
        segments = SanaWMStreamingDenoisingStage._autoregressive_segments(
            total_frames, num_frame_per_block
        )
        conv_cache = self.vae.reset_decoder_cache()
        chunks = []
        for i in range(len(segments) - 1):
            s, e = segments[i], segments[i + 1]
            z = self.scale_and_shift(latents[:, :, s:e].to(vae_dtype), server_args)
            # No autocast: match the official decode (VAE already runs in vae_dtype,
            # z is cast once above) for consistent rounding across chunk boundaries.
            pixel = self.vae.decode_chunk(z, conv_cache)
            pixel = (pixel / 2 + 0.5).clamp(0, 1)
            chunks.append(pixel.float().cpu())

        frames = torch.cat(chunks, dim=2)
        log_sana_wm_tensor_stats("stream.decode.frames", frames)
        return frames
