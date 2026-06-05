# SPDX-License-Identifier: Apache-2.0
"""SANA-WM streaming (self-forcing) denoising stage — S1c.

Generates video chunk-by-chunk via the DiT's chunk-causal ``forward_long``, a
rolling per-block KV cache carrying recurrent GDN state + softmax K/V window.

Gotchas vs the dense ``SanaWMDenoisingStage`` (one-shot bidirectional):
  * fresh ``FlowMatchEulerDiscreteScheduler(shift=1.0)`` with explicit sigmas
    (NOT the request scheduler, whose shift would warp the sigmas);
  * KV accumulator concats softmax cache slots on **dim=1** (our
    ``(B, N, H, D)`` softmax cache layout), not the reference's dim=2;
  * the condition frame (frame 0) is re-pinned every step in chunk 0.
"""

from __future__ import annotations

import time

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.sana_wm import (
    _NUM_STREAM_CACHE_SLOTS,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.self_forcing import (
    SanaWMSelfForcingSamplerConfig,
    SanaWMSelfForcingSampler,
)
from .base import (
    _align_sana_wm_cfg_text_conditions,
    _cat_optional_tensors,
    _first_tensor,
    _to_device_dtype,
    log_sana_wm_tensor_stats,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.causal_denoising import (
    CausalDMDDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.realtime.causal_state import RealtimeCausalDiTState
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

# --- debug parity harness (gated by env; no-op in production) ---
import os as _os

from . import parity_probe

_SANAWM_INJECT_DIR = _os.environ.get(parity_probe.ENV_INJECT)


def self_forcing_denoise_chunk(
    *,
    transformer,
    scheduler,
    sigmas,
    get_chunk,
    set_chunk,
    cond_mask: torch.Tensor,
    embeds: torch.Tensor,
    mask,
    camera_conditions,
    chunk_plucker,
    chunk_kv,
    start_f: int,
    end_f: int,
    frame_index,
    do_cfg: bool,
    cfg_scale: float,
    target_dtype: torch.dtype,
    device,
    forward_ctx=None,
    on_noise_pred=None,
) -> list:
    """Self-forcing denoise of ONE chunk + the clean t=0 KV pass; returns the chunk's updated per-block KV cache.

    SINGLE parity-locked implementation shared by the offline and realtime paths
    (it used to be copy-pasted — the same drift failure mode as the four parity bugs).

    ``get_chunk``/``set_chunk`` are caller closures so each path keeps its EXACT
    tensor lifecycle (kernel-visible layouts, and therefore bitwise behavior,
    unchanged); ``set_chunk`` owns the dtype cast and cond-frame re-pinning.
    ``forward_ctx`` (optional) maps ``int timestep -> context manager`` (batch
    wraps forward_long in set_forward_context; realtime does not).
    """
    from contextlib import nullcontext

    ctx = forward_ctx if forward_ctx is not None else (lambda _ts: nullcontext())

    scheduler.set_timesteps(sigmas=sigmas, device=device)
    for t in scheduler.timesteps:
        chunk_lat = get_chunk()
        B, C = chunk_lat.shape[0], chunk_lat.shape[1]
        lat_in = torch.cat([chunk_lat, chunk_lat], dim=0) if do_cfg else chunk_lat
        ts_tensor = (1.0 - cond_mask) * t.to(
            device=device, dtype=torch.float32
        ).view(1, 1, 1, 1, 1)
        ts_in = torch.cat([ts_tensor, ts_tensor], dim=0) if do_cfg else ts_tensor
        model_ts = ts_in[:, :1, :, 0, 0]  # (B|2B, chunk_frames)

        with ctx(int(t.item()) if t.ndim == 0 else 0):
            noise_pred, _ = transformer.forward_long(
                hidden_states=lat_in.to(target_dtype),
                encoder_hidden_states=embeds,
                timestep=model_ts,
                encoder_attention_mask=mask,
                camera_conditions=camera_conditions,
                chunk_plucker=chunk_plucker,
                kv_cache=chunk_kv,
                save_kv_cache=False,
                start_f=start_f,
                end_f=end_f,
                frame_index=frame_index,
            )
        if do_cfg:
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + cfg_scale * (noise_text - noise_uncond)

        if on_noise_pred is not None:
            on_noise_pred(int(t.item()), noise_pred)

        denoised = scheduler.step(
            -noise_pred.reshape(B, C, -1).transpose(1, 2),
            t,
            chunk_lat.reshape(B, C, -1).transpose(1, 2),
            per_token_timesteps=ts_tensor.reshape(B, C, -1)[:, 0],
            return_dict=False,
        )[0]
        set_chunk(denoised.transpose(1, 2).reshape(chunk_lat.shape))

    # Clean pass (t=0) to write this chunk's KV for the next chunk.
    chunk_lat = get_chunk()
    lat_in = torch.cat([chunk_lat, chunk_lat], dim=0) if do_cfg else chunk_lat
    ts_zero = torch.zeros(
        lat_in.shape[0], 1, chunk_lat.shape[2], device=device, dtype=torch.float32
    )
    with ctx(0):
        _, updated_cache = transformer.forward_long(
            hidden_states=lat_in.to(target_dtype),
            encoder_hidden_states=embeds,
            timestep=ts_zero,
            encoder_attention_mask=mask,
            camera_conditions=camera_conditions,
            chunk_plucker=chunk_plucker,
            kv_cache=chunk_kv,
            save_kv_cache=True,
            start_f=start_f,
            end_f=end_f,
            frame_index=frame_index,
        )
    return updated_cache


class SanaWMStreamCacheState(RealtimeCausalDiTState):
    """Per-session streaming DiT state, framework-pattern (cf. LingBot).

    The Wan-shaped ``kv_cache`` field stays None — SANA-WM's cache is the
    heterogeneous per-block 10-slot list (GDN recurrent matrix states + softmax
    concat windows + conv tails), carried in the fields below."""

    def __init__(self):
        super().__init__()
        # kv[chunk][block][slot] — grown per chunk, stale chunks evicted.
        self.stream_kv_cache: list = []
        self.chunk_indices: list[int] = [0]
        # Growing stage-1 latent buffer (cond frame + denoised chunks).
        self.latents: torch.Tensor | None = None
        # Per-session flow-Euler scheduler (fresh shift=1.0; persists across ticks).
        self.scheduler: FlowMatchEulerDiscreteScheduler | None = None

    def dispose(self) -> None:
        super().dispose()
        self.stream_kv_cache = []
        self.chunk_indices = [0]
        self.latents = None
        self.scheduler = None


class SanaWMStreamingDenoisingStage(CausalDMDDenoisingStage):
    """Autoregressive self-forcing streaming denoise — SANA-WM's causal-DMD variant.

    Same family as ``LingBotWorldCausalDMDDenoisingStage`` (offline whole-clip vs
    realtime per-chunk). Differences from the Wan-style base are the cache contract
    (heterogeneous 10-slot list instead of preallocated ``CausalSelfAttentionKVCache``),
    the front-loaded chunk grid (chunk 0 carries the remainder), and in-chunk
    condition-frame pinning instead of KV warm-up — hence the cache-management overrides.
    """

    def __init__(self, transformer, scheduler=None, *, keep_resident: bool = False) -> None:
        # Skip CausalDMDDenoisingStage.__init__: it reads Wan-specific arch
        # fields (num_frames_per_block / sliding_window_num_frames / sink_size)
        # that SANA-WM defines per-run via the pipeline config instead.
        DenoisingStage.__init__(self, transformer, scheduler)
        self.num_transformer_blocks = len(transformer.blocks)
        # Realtime pipelines keep the DiT device-resident for the session's
        # lifetime (the per-tick offload round-trip would dominate latency);
        # the offline pipeline keeps the default offload behavior.
        self._keep_resident = bool(keep_resident)

    def component_uses(self, server_args: ServerArgs, stage_name: str | None = None):
        if not self._keep_resident:
            return super().component_uses(server_args, stage_name)
        from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
            ComponentUse,
        )

        stage_name = self._component_stage_name(stage_name)
        return [
            ComponentUse(
                stage_name,
                "transformer",
                target_dtype=PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision],
                memory_intensive=True,
                keep_ready_after_warmup=True,
            ),
        ]

    # Chunk schedule + KV cache: delegate to SanaWMSelfForcingSampler.
    @staticmethod
    def _autoregressive_segments(total_frames: int, num_frame_per_block: int) -> list[int]:
        return SanaWMSelfForcingSampler.create_autoregressive_segments(
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
        return SanaWMSelfForcingSampler.accumulate_kv_cache(
            kv_cache, chunk_idx, chunk_indices, num_cached_blocks, sink_token, num_blocks
        )

    @staticmethod
    def _evict_stale_kv_cache(
        kv_cache: list, chunk_idx: int, valid: list[int], num_cached_blocks: int, num_blocks: int
    ) -> None:
        SanaWMSelfForcingSampler.evict_stale_kv_cache(
            kv_cache, chunk_idx, valid, num_cached_blocks, num_blocks
        )

    # Realtime per-chunk path (sessions): per-session state in SanaWMStreamCacheState.
    @torch.no_grad()
    def _forward_realtime_chunk(self, batch: Req, server_args: ServerArgs) -> Req:
        if batch.latents is None or batch.latents.ndim != 5:
            raise ValueError(
                "SANA-WM realtime denoising expects this tick's pre-noised chunk "
                "latents (B, C, n, H, W) from the latent-preparation stage."
            )
        pcfg = server_args.pipeline_config
        device = get_local_torch_device()
        target_dtype = PRECISION_TO_TYPE.get(
            getattr(pcfg, "dit_precision", "bf16"), torch.bfloat16
        )
        state = batch.session.get_or_create_state(SanaWMStreamCacheState)
        if batch.block_idx == 0 and state.latents is not None:
            state.dispose()  # session restart on chunk 0 (mirrors the base stage)

        sc = self._resolve_stream_conditioning(
            batch, server_args, device=device, target_dtype=target_dtype
        )
        sampler_cfg = sc.sampler_cfg
        incoming = batch.latents.to(device=device, dtype=target_dtype).clone()
        plan = list(batch.extra.get("sana_wm_chunk_plan") or [incoming.shape[2]])
        if sum(plan) != incoming.shape[2]:
            raise ValueError(
                f"chunk plan {plan} does not cover the incoming {incoming.shape[2]} frames"
            )
        if state.scheduler is None:
            state.scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0)

        # Device-only move, NO dtype cast: Module.to(dtype=...) would cast the
        # DiT's complex RoPE buffers to real, discarding the imaginary part
        # (parity root cause #1). No use_declared_component round-trip either —
        # the DiT stays device-resident for the session's lifetime.
        transformer = self.transformer.to(device=device).eval()
        num_blocks = len(transformer.blocks)
        _dump_dir = parity_probe.probe_dir(parity_probe.ENV_RT_DUMP)
        if _dump_dir and state.chunk_idx == 0:  # parity harness
            parity_probe.dump_tensor(_dump_dir, "cond_embeds", sc.embeds)
            parity_probe.dump_tensor(_dump_dir, "cond_mask", sc.mask)
            parity_probe.dump_obj(
                _dump_dir, "dit_fingerprint", parity_probe.weights_fingerprint(transformer)
            )

        offset = 0
        for n in plan:
            chunk_lat = incoming[:, :, offset : offset + n]
            offset += n
            chunk_idx = state.chunk_idx
            # chunk 0's incoming includes the conditioning frame at index 0.
            start_f = state.chunk_indices[-1] if chunk_idx > 0 else 0
            end_f = (start_f + n) if chunk_idx > 0 else n
            state.chunk_indices.append(end_f)
            state.stream_kv_cache.append(
                [[None] * _NUM_STREAM_CACHE_SLOTS for _ in range(num_blocks)]
            )
            chunk_kv, sink_num = self._accumulate_kv_cache(
                state.stream_kv_cache, chunk_idx, state.chunk_indices,
                sampler_cfg.num_cached_blocks, sampler_cfg.sink_token, num_blocks,
            )
            # Evict entries outside the accumulate window (sink + last
            # num_cached_blocks) — unbounded sessions leak concat K/V otherwise.
            if chunk_idx > 0 and sampler_cfg.num_cached_blocks > 0:
                start_chunk = max(chunk_idx - sampler_cfg.num_cached_blocks, 0)
                valid = list(range(start_chunk, chunk_idx))
                if sampler_cfg.sink_token:
                    sink_start = max(chunk_idx - sampler_cfg.num_cached_blocks + 1, 0)
                    if sink_start > 0:
                        valid = [0] + list(range(sink_start, chunk_idx))
                self._evict_stale_kv_cache(
                    state.stream_kv_cache, chunk_idx, valid,
                    sampler_cfg.num_cached_blocks, num_blocks,
                )
            if _dump_dir:  # parity harness
                parity_probe.dump_obj(
                    _dump_dir, f"kv_probe_{chunk_idx:03d}",
                    parity_probe.kv_cache_checksums(chunk_kv, sink_num),
                )
                parity_probe.dump_tensor(
                    _dump_dir, f"cond_camera_{chunk_idx:03d}", sc.camera
                )
                parity_probe.dump_tensor(
                    _dump_dir, f"cond_plucker_{chunk_idx:03d}", sc.plucker
                )
            frame_index = (
                torch.arange(start_f, end_f, device=device, dtype=torch.long)
                if sink_num > 0
                else None
            )

            B, C = chunk_lat.shape[0], chunk_lat.shape[1]
            cond_mask = torch.zeros(
                B, C, end_f - start_f, *chunk_lat.shape[3:],
                device=device, dtype=torch.float32,
            )
            cond_local = []
            if chunk_idx == 0:
                cond_mask[:, :, 0] = 1.0
                cond_local = [0]
            init_chunk = chunk_lat.clone()
            _local = {"lat": chunk_lat}

            def _get_chunk(_local=_local):
                return _local["lat"]

            def _set_chunk(denoised, _local=_local, init_chunk=init_chunk, cond_local=cond_local):
                lat = denoised.to(target_dtype)
                for loc in cond_local:
                    lat[:, :, loc] = init_chunk[:, :, loc]
                _local["lat"] = lat

            def _on_noise_pred(ts_int, noise_pred, chunk_idx=chunk_idx):
                if _dump_dir and chunk_idx == 0:  # parity harness
                    parity_probe.dump_tensor(
                        _dump_dir, f"noise_pred_c0_t{ts_int}", noise_pred
                    )

            # Same forward context the mega-stage held across the tick (the
            # per-chunk denoise core relies on an ambient context here; the
            # offline path supplies per-step contexts instead).
            with set_forward_context(
                current_timestep=batch.block_idx,
                attn_metadata=None,
                forward_batch=batch,
            ):
                state.stream_kv_cache[chunk_idx] = self_forcing_denoise_chunk(
                    transformer=transformer,
                    scheduler=state.scheduler,
                    sigmas=sc.explicit_sigmas,
                    get_chunk=_get_chunk,
                    set_chunk=_set_chunk,
                    cond_mask=cond_mask,
                    embeds=sc.embeds,
                    mask=sc.mask,
                    camera_conditions=sc.camera,
                    chunk_plucker=sc.plucker,
                    chunk_kv=chunk_kv,
                    start_f=start_f,
                    end_f=end_f,
                    frame_index=frame_index,
                    do_cfg=sc.do_cfg,
                    cfg_scale=sc.cfg_scale,
                    target_dtype=target_dtype,
                    device=device,
                    on_noise_pred=_on_noise_pred,
                )
            chunk_lat = _local["lat"]
            state.latents = (
                chunk_lat
                if state.latents is None
                else torch.cat([state.latents, chunk_lat], dim=2)
            )
            parity_probe.dump_tensor(  # parity harness
                _dump_dir, f"stage1_{chunk_idx:03d}_{start_f}_{end_f}", chunk_lat
            )
            state.chunk_idx += 1
            state.current_chunk_start_frame = end_f

        # Downstream chain stages (refiner/decode) consume the growing buffer.
        batch.latents = state.latents
        return batch

    def _resolve_stream_conditioning(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        device: torch.device,
        target_dtype: torch.dtype,
        iload=None,
    ):
        """Resolve sampler config + text/camera conditioning (shared by the offline loop and realtime path)."""
        from types import SimpleNamespace

        pcfg = server_args.pipeline_config
        sampler_cfg = SanaWMSelfForcingSamplerConfig.from_pipeline_config(pcfg)
        explicit_sigmas = SanaWMSelfForcingSampler.build_per_chunk_sigmas(
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
        if _SANAWM_INJECT_DIR and iload is not None and not do_cfg:
            _cond = iload("cond").to(device=device, dtype=target_dtype)
            while _cond.dim() > embeds_in.dim():
                _cond = _cond.squeeze(1)
            embeds_in = _cond
            mask_in = iload("cond_mask").to(device=device)

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
        if _SANAWM_INJECT_DIR and iload is not None:
            cam_in = iload("raymap").to(device=device, dtype=target_dtype)
            plk_in = iload("chunk_plucker").to(device=device, dtype=target_dtype)
            if do_cfg:
                cam_in = torch.cat([cam_in, cam_in], dim=0)
                plk_in = torch.cat([plk_in, plk_in], dim=0)

        return SimpleNamespace(
            sampler_cfg=sampler_cfg,
            explicit_sigmas=explicit_sigmas,
            cfg_scale=cfg_scale,
            do_cfg=do_cfg,
            embeds=embeds_in,
            mask=mask_in,
            camera=cam_in,
            plucker=plk_in,
        )

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        # LingBot-style dispatch: realtime sessions denoise ONE chunk per call
        # with per-session state; otherwise run the whole clip offline.
        if batch.session is not None:
            return self._forward_realtime_chunk(batch, server_args)
        return self._forward_offline(batch, server_args)

    @torch.no_grad()
    def _forward_offline(self, batch: Req, server_args: ServerArgs) -> Req:
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

        _dump_dir = parity_probe.probe_dir(parity_probe.ENV_FORK_DUMP)

        def _fdump(_name, _t):
            parity_probe.dump_tensor(_dump_dir, _name, _t)

        if _SANAWM_INJECT_DIR:  # parity harness: run the OFFICIAL's exact stage-1 inputs
            latents = _iload("z_full_initial").to(device=device, dtype=target_dtype).clone()
            init_latents = latents.clone()
            B, C, total_frames, H, W = latents.shape

        _fdump("init_noise", init_latents)  # parity harness: seeded pre-noise (cond @ frame 0)

        sc = self._resolve_stream_conditioning(
            batch, server_args, device=device, target_dtype=target_dtype, iload=_iload
        )
        sampler_cfg = sc.sampler_cfg
        num_frame_per_block = sampler_cfg.num_frame_per_block
        num_cached_blocks = sampler_cfg.num_cached_blocks
        sink_token = sampler_cfg.sink_token
        explicit_sigmas = sc.explicit_sigmas
        cfg_scale = sc.cfg_scale
        do_cfg = sc.do_cfg
        embeds_in, mask_in, cam_in, plk_in = sc.embeds, sc.mask, sc.camera, sc.plucker

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
            if _dump_dir:  # parity harness: weights fingerprint
                parity_probe.dump_obj(
                    _dump_dir,
                    "dit_fingerprint",
                    parity_probe.weights_fingerprint(transformer),
                )
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
                if _dump_dir:  # parity harness: accumulated-KV checksums
                    parity_probe.dump_obj(
                        _dump_dir,
                        f"kv_probe_{chunk_idx:03d}",
                        parity_probe.kv_cache_checksums(chunk_kv, sink_num),
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

                # Buffer-view closures: the shared loop reads/writes views of
                # the full-latent buffer (kernel-visible layouts unchanged).
                def _get_chunk(start_f=start_f, end_f=end_f):
                    return latents[:, :, start_f:end_f]

                def _set_chunk(
                    denoised, start_f=start_f, end_f=end_f, cond_local=cond_local
                ):
                    latents[:, :, start_f:end_f] = denoised.to(latents.dtype)
                    for loc in cond_local:
                        latents[:, :, start_f + loc] = init_latents[
                            :, :, start_f + loc
                        ]

                def _forward_ctx(ts_int):
                    return set_forward_context(
                        current_timestep=ts_int,
                        attn_metadata=None,
                        forward_batch=batch,
                    )

                def _on_noise_pred(ts_int, noise_pred, chunk_idx=chunk_idx):
                    if chunk_idx == 0:  # parity harness: per-step model output
                        _fdump(f"noise_pred_c0_t{ts_int}", noise_pred)

                kv_cache[chunk_idx] = self_forcing_denoise_chunk(
                    transformer=transformer,
                    scheduler=scheduler,
                    sigmas=explicit_sigmas,
                    get_chunk=_get_chunk,
                    set_chunk=_set_chunk,
                    cond_mask=cond_mask,
                    embeds=embeds_in,
                    mask=mask_in,
                    camera_conditions=cam_in,
                    chunk_plucker=plk_in,
                    chunk_kv=chunk_kv,
                    start_f=start_f,
                    end_f=end_f,
                    frame_index=frame_index,
                    do_cfg=do_cfg,
                    cfg_scale=cfg_scale,
                    target_dtype=target_dtype,
                    device=device,
                    forward_ctx=_forward_ctx,
                    on_noise_pred=_on_noise_pred,
                )
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
    """Streaming causal-VAE decode over the SAME autoregressive grid the denoise stage used.

    Carries a per-conv decoder cache across chunks so the causal LTX-2 VAE produces
    seam-free frames (the `decode_per_frame_with_cache` equivalent at chunk granularity).
    Subclasses DecodingStage directly (NOT SanaWMDecodingStage, whose long-video config
    re-enables the stateless tiled decode).
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
