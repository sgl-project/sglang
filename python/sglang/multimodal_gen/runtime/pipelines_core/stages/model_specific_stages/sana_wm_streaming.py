# SPDX-License-Identifier: Apache-2.0
"""SANA-WM streaming (self-forcing) denoising stage — S1c.

Drives the DiT's chunk-causal ``forward_long`` autoregressively: the video is
generated chunk-by-chunk, each chunk denoised with a short self-forcing schedule
while a rolling per-block KV cache carries recurrent GDN state (and a softmax
K/V concat-window) across chunks. Ports the reference NVlabs
``SelfForcingFlowEulerSampler`` (minimal-sanawm/scheduler.py).

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
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm import (
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


class SanaWMStreamingDenoisingStage(SanaWMDenoisingStage):
    """Autoregressive self-forcing streaming variant of SanaWMDenoisingStage."""

    # ----------------------------------------------------------------- #
    # Chunk schedule + KV cache accumulation (port of scheduler.py:201-404)
    # ----------------------------------------------------------------- #
    @staticmethod
    def _autoregressive_segments(total_frames: int, num_frame_per_block: int) -> list[int]:
        base = int(num_frame_per_block)
        remained = total_frames % base
        num_chunks = total_frames // base
        chunk_indices = [0]
        for idx in range(num_chunks):
            cur = chunk_indices[-1] + base + (remained if idx == 0 else 0)
            chunk_indices.append(cur)
        return chunk_indices

    @staticmethod
    def _accumulate_kv_cache(
        kv_cache: list,
        chunk_idx: int,
        chunk_indices: list[int],
        num_cached_blocks: int,
        sink_token: bool,
        num_blocks: int,
    ) -> tuple[list, int]:
        """Build chunk ``chunk_idx``'s read-only KV prefix from prior chunks.

        GDN/STATE blocks (type flag > 0.5) copy-forward the PREVIOUS chunk's
        recurrent state; softmax/CONCAT blocks concatenate the rolling-window +
        sink K/V along **dim=1** (token axis of our (B,N,H,D) softmax cache)."""
        if chunk_idx == 0:
            return kv_cache[0], 0

        cur = kv_cache[chunk_idx]
        start_chunk = max(chunk_idx - num_cached_blocks, 0) if num_cached_blocks > 0 else 0
        valid = list(range(start_chunk, chunk_idx))
        sink_num = 0
        if sink_token and num_cached_blocks > 0:
            sink_start = max(chunk_idx - num_cached_blocks + 1, 0)
            if sink_start > 0:
                valid = [0] + list(range(sink_start, chunk_idx))
                sink_num = chunk_indices[1] - chunk_indices[0]

        for block_id in range(num_blocks):
            prev_last = kv_cache[chunk_idx - 1][block_id]
            type_flag = prev_last[_SLOT_TYPE_FLAG]
            if type_flag is not None and float(type_flag.item()) > 0.5:
                # STATE (GDN) block: carry the previous chunk's recurrent state.
                cur[block_id] = [
                    prev_last[_SLOT_K],
                    prev_last[_SLOT_V],
                    prev_last[_SLOT_CAM_K],
                    prev_last[_SLOT_CAM_V],
                    prev_last[_SLOT_SHORTCONV],
                    None,
                    prev_last[_SLOT_TYPE_FLAG],
                    None,
                    None,
                    prev_last[_SLOT_FFN_TCONV],
                ]
                continue

            # CONCAT (softmax) block: concat cached K/V over the valid window.
            acc: list[torch.Tensor | None] = [None] * _NUM_STREAM_CACHE_SLOTS
            for idx in valid:
                prev = kv_cache[idx][block_id]
                if prev[_SLOT_K] is None:
                    continue
                for slot in (_SLOT_K, _SLOT_V, _SLOT_CAM_K, _SLOT_CAM_V):
                    if prev[slot] is None:
                        continue
                    acc[slot] = (
                        prev[slot].clone()
                        if acc[slot] is None
                        else torch.cat([acc[slot], prev[slot]], dim=1)  # (B,N,H,D) token axis
                    )
            cur[block_id] = [
                acc[_SLOT_K],
                acc[_SLOT_V],
                acc[_SLOT_CAM_K],
                acc[_SLOT_CAM_V],
                prev_last[_SLOT_SHORTCONV],
                None,
                prev_last[_SLOT_TYPE_FLAG],
                None,
                None,
                prev_last[_SLOT_FFN_TCONV],
            ]

        SanaWMStreamingDenoisingStage._evict_stale_kv_cache(
            kv_cache, chunk_idx, valid, num_cached_blocks, num_blocks
        )
        return cur, sink_num

    @staticmethod
    def _evict_stale_kv_cache(
        kv_cache: list, chunk_idx: int, valid: list[int], num_cached_blocks: int, num_blocks: int
    ) -> None:
        if num_cached_blocks <= 0:
            return
        keep = set(valid)
        keep.add(chunk_idx)
        for stale in range(chunk_idx):
            if stale in keep:
                continue
            kv_cache[stale] = [
                [None] * _NUM_STREAM_CACHE_SLOTS for _ in range(num_blocks)
            ]

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

        num_frame_per_block = int(getattr(pcfg, "num_frame_per_block", 3))
        num_cached_blocks = int(getattr(pcfg, "num_cached_blocks", 2))
        sink_token = bool(getattr(pcfg, "sink_token", True))
        schedule = list(getattr(pcfg, "denoising_step_list", (1000, 960, 889, 727, 0)))
        if len(schedule) < 2 or schedule[-1] != 0:
            raise ValueError(f"denoising_step_list must end with 0, got {schedule}")
        explicit_sigmas = [float(t) / 1000.0 for t in schedule[:-1]]

        cfg_scale = float(getattr(batch, "guidance_scale", 1.0) or 1.0)
        do_cfg = bool(batch.do_classifier_free_guidance) and cfg_scale > 1.0
        if server_args.enable_cfg_parallel and do_cfg:
            raise NotImplementedError(
                "SANA-WM streaming does not support CFG parallel; run replicated."
            )

        # --- text conditioning (mirror the dense stage) ---
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
        vae_autocast = (vae_dtype != torch.float32) and not server_args.disable_autocast

        conv_cache = self.vae.reset_decoder_cache()
        chunks = []
        for i in range(len(segments) - 1):
            s, e = segments[i], segments[i + 1]
            z = self.scale_and_shift(latents[:, :, s:e].to(vae_dtype), server_args)
            with torch.autocast(
                device_type=current_platform.device_type,
                dtype=vae_dtype,
                enabled=vae_autocast,
            ):
                pixel = self.vae.decode_chunk(z, conv_cache)
            pixel = (pixel / 2 + 0.5).clamp(0, 1)
            chunks.append(pixel.float().cpu())

        frames = torch.cat(chunks, dim=2)
        log_sana_wm_tensor_stats("stream.decode.frames", frames)
        return frames
