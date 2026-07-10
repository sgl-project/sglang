# SPDX-License-Identifier: Apache-2.0
"""SANA-WM realtime stage chain.

Per-tick pipeline:

    RealtimeInputValidationStage          (framework)
    SanaWMRealtimeTextEncodingStage       (framework cache via MRO)
    SanaWMCondFrameEncodeStage            -> batch.image_latent (+ inputs snapshot)
    SanaWMRealtimeLatentPrepStage         -> batch.latents = this tick's pre-noised
                                             chunk(s); batch.extra["sana_wm_chunk_plan"]
    SanaWMCameraCondStage                 -> batch.extra camera_conditions/chunk_plucker
    SanaWMStreamingDenoisingStage         (session path; RealtimeCausalDiTState)
    SanaWMChunkedRefinerChainStage        -> batch.latents = refined buffer
    SanaWMCausalDecodeChainStage          -> OutputBatch (decodes past its frontier)

Chain stages share ``SanaWMRealtimeStage`` for first-frame image handling and
causal VAE decode helpers; model-specific chunk planning stays in this module.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.sana_wm_components import (
    compute_chunk_plucker,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.realtime.session import BaseRealtimeState
from sglang.multimodal_gen.runtime.realtime.states import (
    RealtimeCausalDecodeState,
    get_realtime_causal_dit_state,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

from . import parity_probe
from .base import (
    _SANA_WM_DEFAULT_ROTATION_SPEED_DEG,
    _SANA_WM_DEFAULT_TRANSLATION_SPEED,
    SanaWMBeforeDenoisingStage,
    configure_sana_wm_ltx2_vae_for_long_video,
    snap_sana_wm_num_frames,
)
from .realtime_stage import (
    DEFAULT_REFINER_BLOCK_SIZE,
    DEFAULT_REFINER_KV_MAX_FRAMES,
    SANA_WM_HEIGHT,
    SANA_WM_WIDTH,
    SanaWMRealtimeStage,
    _motion_param,
)
from .streaming import SanaWMStreamingDenoisingStage
from .streaming_refiner import RefinerChunkRunner


# --------------------------------------------------------------------- #
# Per-session state blobs (one per owning stage)
# --------------------------------------------------------------------- #
class SanaWMSessionInputsState(BaseRealtimeState):
    """Session input snapshot — written by the cond-encode stage, read by the
    camera stage (crop geometry) and the refiner stage (prompt)."""

    def __init__(self):
        super().__init__()
        self.image = None
        self.intrinsics_image = None
        self.src_size = None
        self.resized_size = None
        self.crop_offset = None
        self.target_height: int | None = None
        self.target_width: int | None = None
        self.prompt: str = ""
        self.open_ended = False
        self.latent_t: int | None = None  # None => open-ended
        self.num_frame_per_block = DEFAULT_REFINER_BLOCK_SIZE
        self.sink_size = 1
        self.camera_actions: list[list[str]] = []
        self.max_camera_actions = 0  # unlimited (sessions stream past any horizon)
        self.static_c2w: np.ndarray | None = None
        self.intrinsics_raw: np.ndarray | None = None
        self.translation_speed = _SANA_WM_DEFAULT_TRANSLATION_SPEED
        self.rotation_speed_deg = _SANA_WM_DEFAULT_ROTATION_SPEED_DEG
        self.first_latent: torch.Tensor | None = None

    def dispose(self):
        super().dispose()
        self.__init__()


class SanaWMNoiseState(BaseRealtimeState):
    """Seeded noise discipline — full-horizon buffer for fixed N (sliced per
    chunk, matching the offline draw bitwise), seeded fallback otherwise."""

    def __init__(self):
        super().__init__()
        self.noise_buffer: torch.Tensor | None = None
        self.generator: torch.Generator | None = None
        self.segments: list[int] | None = None  # front-loaded grid (fixed N)

    def dispose(self):
        super().dispose()
        self.noise_buffer = None
        self.generator = None
        self.segments = None


class SanaWMRefinerChainState(BaseRealtimeState):
    def __init__(self):
        super().__init__()
        self.runner: RefinerChunkRunner | None = None
        self.refined_full: torch.Tensor | None = None
        self.next_ref_idx = 0
        self.block_size = DEFAULT_REFINER_BLOCK_SIZE
        self.sink_size = 1

    def dispose(self):
        super().dispose()
        self.runner = None
        self.refined_full = None
        self.next_ref_idx = 0
        self.block_size = DEFAULT_REFINER_BLOCK_SIZE
        self.sink_size = 1


# --------------------------------------------------------------------- #
# Chain stages
# --------------------------------------------------------------------- #
class SanaWMCondFrameEncodeStage(SanaWMRealtimeStage):
    """Encode the first frame and write the session input snapshot."""

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        session = self.require_session(batch, context="SANA-WM realtime chain")
        self._pipeline_config = server_args.pipeline_config
        device = get_local_torch_device()
        weight_dtype = PRECISION_TO_TYPE.get(
            getattr(server_args.pipeline_config, "dit_precision", "bf16"),
            torch.bfloat16,
        )
        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]

        self.vae = self.vae.to(device=device, dtype=vae_dtype).eval()
        configure_sana_wm_ltx2_vae_for_long_video(self.vae, server_args.pipeline_config)

        st = session.get_or_create_state(SanaWMSessionInputsState)
        if batch.block_idx == 0 or st.image is None:
            st.dispose()
            (
                st.image,
                st.intrinsics_image,
                st.src_size,
                st.resized_size,
                st.crop_offset,
            ) = self._prepare_image(batch)
            st.target_width, st.target_height = st.image.size
            st.prompt = str(batch.prompt)
            st.open_ended = self._is_open_ended(batch)
            if st.open_ended:
                st.latent_t = None
            else:
                num_frames = snap_sana_wm_num_frames(int(batch.num_frames), stride=8)
                batch.num_frames = num_frames
                st.latent_t = (num_frames - 1) // 8 + 1
            st.num_frame_per_block = int(
                batch.extra.get("sana_wm_num_frame_per_block", 3)
            )
            st.sink_size = int(batch.extra.get("sana_wm_sink_size", 1))
            st.translation_speed = _motion_param(
                batch, "translation_speed", _SANA_WM_DEFAULT_TRANSLATION_SPEED
            )
            st.rotation_speed_deg = _motion_param(
                batch, "rotation_speed_deg", _SANA_WM_DEFAULT_ROTATION_SPEED_DEG
            )
            st.static_c2w = self._prepare_static_camera(
                batch,
                num_frames=(
                    ((st.latent_t - 1) * 8 + 1)
                    if st.latent_t is not None
                    else st.num_frame_per_block * 8 + 1
                ),
                translation_speed=st.translation_speed,
                rotation_speed_deg=st.rotation_speed_deg,
            )
            st.first_latent = self._get_first_frame_latent(
                batch,
                st.image,
                device=device,
                vae_dtype=vae_dtype,
                latent_dtype=weight_dtype,
            )
        batch.image_latent = st.first_latent
        return batch


class SanaWMRealtimeLatentPrepStage(SanaWMRealtimeStage):
    """This tick's chunk PLAN + pre-noised latents.

    Fixed N: front-loaded segments + a full-horizon seeded buffer sliced per
    chunk (bitwise-matching the offline draw); past the horizon / open-ended:
    uniform chunks from the seeded fallback generator. The plan holds more than
    one chunk only when the refiner block grid lags the stage-1 chunk grid (the
    option-b boundary tick) so every tick emits frames."""

    def __init__(self, *, use_refiner: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        self.use_refiner = bool(use_refiner)

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        device = get_local_torch_device()
        weight_dtype = PRECISION_TO_TYPE.get(
            getattr(server_args.pipeline_config, "dit_precision", "bf16"),
            torch.bfloat16,
        )
        session = self.require_session(batch, context="SANA-WM realtime chain")
        inputs = session.get_or_create_state(SanaWMSessionInputsState)
        noise = session.get_or_create_state(SanaWMNoiseState)
        cache = get_realtime_causal_dit_state(session)
        first_latent = batch.image_latent
        if first_latent is None:
            raise ValueError("cond-frame latent missing (run the encode stage first)")

        nfpb = int(inputs.num_frame_per_block)
        if batch.block_idx == 0 or (
            noise.noise_buffer is None and noise.generator is None
        ):
            noise.dispose()
            seed = batch.seed[0] if isinstance(batch.seed, list) else batch.seed
            if seed is not None:
                noise.generator = torch.Generator(device=device).manual_seed(int(seed))
            if inputs.latent_t is not None:
                noise.segments = SanaWMStreamingDenoisingStage._autoregressive_segments(
                    int(inputs.latent_t), nfpb
                )
                gen = (
                    batch.generator[0]
                    if isinstance(batch.generator, list)
                    else batch.generator
                ) or noise.generator
                buf = torch.randn(
                    1,
                    first_latent.shape[1],
                    int(inputs.latent_t),
                    first_latent.shape[-2],
                    first_latent.shape[-1],
                    device=device,
                    dtype=weight_dtype,
                    generator=gen,
                )
                buf[:, :, :1] = first_latent
                noise.noise_buffer = buf
                parity_probe.dump_tensor(  # parity harness (no-op in prod)
                    parity_probe.probe_dir(parity_probe.ENV_RT_DUMP),
                    "noise_buffer",
                    buf,
                )

        # Chunk plan: enough stage-1 chunks for >=1 refiner block.
        stage1_cur = cache.chunk_indices[-1]
        chunk_idx = cache.chunk_idx

        def _next_len(idx: int) -> int:
            if noise.segments is not None and idx + 1 < len(noise.segments):
                # Front-loaded grid (chunk 0's segment already includes the
                # conditioning frame).
                return noise.segments[idx + 1] - noise.segments[idx]
            n = nfpb
            if idx == 0:
                # Uniform (open-ended) grid: chunk 0 = cond frame + nfpb new
                # frames, matching the retired engine's step(n_frames=nfpb).
                n += first_latent.shape[2]
            return n

        plan: list[int] = []
        sim = stage1_cur
        idx = chunk_idx
        if self.use_refiner and chunk_idx > 0:
            sink = int(inputs.sink_size)
            block = int(inputs.num_frame_per_block)
            refined_cur = sink + max(0, (sim - sink)) // block * block
            target = refined_cur + block
            while sim < target:
                n = _next_len(idx)
                plan.append(n)
                sim += n
                idx += 1
        else:
            plan.append(_next_len(idx))

        pieces = []
        pos = stage1_cur
        for i, n in enumerate(plan):
            is_chunk0 = (chunk_idx + i) == 0
            new_frames = n - (first_latent.shape[2] if is_chunk0 else 0)
            lo = pos + (first_latent.shape[2] if is_chunk0 else 0)
            hi = lo + new_frames
            if noise.noise_buffer is not None and hi <= noise.noise_buffer.shape[2]:
                draw = noise.noise_buffer[:, :, lo:hi].to(device, weight_dtype)
            else:
                draw = torch.randn(
                    first_latent.shape[0],
                    first_latent.shape[1],
                    new_frames,
                    first_latent.shape[-2],
                    first_latent.shape[-1],
                    device=device,
                    dtype=weight_dtype,
                    generator=noise.generator,
                )
            pieces.append(
                torch.cat([first_latent.to(device, weight_dtype), draw], dim=2)
                if is_chunk0
                else draw
            )
            pos = hi
        batch.latents = torch.cat(pieces, dim=2)
        batch.extra["sana_wm_chunk_plan"] = plan
        return batch


class SanaWMCameraCondStage(SanaWMRealtimeStage):
    """Camera conditioning windows — appends this tick's actions and rebuilds
    the full-length raymap/chunk_plucker through the END of the planned chunks
    (single-sourced through the batch helpers; parity RC#3)."""

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        device = get_local_torch_device()
        weight_dtype = PRECISION_TO_TYPE.get(
            getattr(server_args.pipeline_config, "dit_precision", "bf16"),
            torch.bfloat16,
        )
        session = self.require_session(batch, context="SANA-WM realtime chain")
        inputs = session.get_or_create_state(SanaWMSessionInputsState)
        cache = get_realtime_causal_dit_state(session)
        plan = list(batch.extra.get("sana_wm_chunk_plan") or [])
        target_latent = cache.chunk_indices[-1] + sum(plan)
        if cache.chunk_idx == 0 and plan:
            target_latent = max(target_latent, plan[0])

        self._append_realtime_camera_actions(batch, inputs)
        camera, plucker = self._build_camera_windows(
            batch,
            inputs,
            target_latent=target_latent,
            device=device,
            dtype=weight_dtype,
        )
        batch.extra["camera_conditions"] = camera
        batch.extra["chunk_plucker"] = plucker
        return batch

    def _build_camera_windows(
        self,
        batch: Req,
        inputs: SanaWMSessionInputsState,
        *,
        target_latent: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Port of the mega-stage's ``_update_camera_tensors`` (value-identical
        ops; returns instead of writing legacy state). Coverage runs through the
        END of the planned chunks; within a fixed horizon ``max()`` keeps it
        EXACTLY at num_frames — bitwise-identical camera tensors."""
        if (
            inputs.src_size is None
            or inputs.resized_size is None
            or inputs.crop_offset is None
        ):
            raise ValueError("SANA-WM crop metadata is not initialized")
        needed = (int(target_latent) - 1) * 8 + 1
        if inputs.open_ended:
            num_frames = needed
        else:
            num_frames = max(int(batch.num_frames), needed)
        condition_inputs = batch.condition_inputs or {}
        if "translation_speed" in condition_inputs:
            inputs.translation_speed = float(condition_inputs["translation_speed"])
        if "rotation_speed_deg" in condition_inputs:
            inputs.rotation_speed_deg = float(condition_inputs["rotation_speed_deg"])
        c2w = self._camera_from_state(
            inputs,
            num_frames=num_frames,
            translation_speed=inputs.translation_speed,
            rotation_speed_deg=inputs.rotation_speed_deg,
        )
        intrinsics_raw = self._prepare_intrinsics(
            batch, inputs, num_frames=num_frames, device=device
        )
        inputs.intrinsics_raw = intrinsics_raw
        vae_time_stride = 8
        pixel_h = int(inputs.target_height or batch.height or SANA_WM_HEIGHT)
        pixel_w = int(inputs.target_width or batch.width or SANA_WM_WIDTH)
        latent_h = pixel_h // 32
        latent_w = pixel_w // 32
        latent_t = (num_frames - 1) // vae_time_stride + 1
        camera_to_world = (
            torch.from_numpy(np.asarray(c2w, dtype=np.float32))
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
        )
        intrinsics_vec4 = (
            torch.from_numpy(np.asarray(intrinsics_raw, dtype=np.float32))
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
        )
        intrinsics_vec4 = (
            SanaWMBeforeDenoisingStage._transform_intrinsics_for_condition_image(
                intrinsics_vec4,
                {
                    "source_size": inputs.src_size,
                    "resized_size": inputs.resized_size,
                    "crop_offset": inputs.crop_offset,
                },
            )
        )
        rel_poses = SanaWMBeforeDenoisingStage._relative_camera_poses(camera_to_world)
        intrinsics_latent = SanaWMBeforeDenoisingStage._scale_intrinsics_to_latent(
            intrinsics_vec4,
            pixel_h=pixel_h,
            pixel_w=pixel_w,
            latent_h=latent_h,
            latent_w=latent_w,
        )
        original_camera = SanaWMBeforeDenoisingStage._flatten_camera_conditions(
            rel_poses, intrinsics_latent
        )
        raymap = SanaWMBeforeDenoisingStage._latent_frame_camera_conditions(
            original_camera,
            num_frames=num_frames,
            latent_frames=latent_t,
            vae_temporal_stride=vae_time_stride,
        ).to(device=device, dtype=dtype)
        chunk_plucker = compute_chunk_plucker(
            original_camera,
            HW=(latent_t, latent_h, latent_w),
            vae_temporal_stride=vae_time_stride,
            patch_size=(1, 1, 1),
        ).to(device=device, dtype=dtype)
        return raymap, chunk_plucker


class SanaWMChunkedRefinerChainStage(SanaWMRealtimeStage):
    """Chunked LTX-2 refiner on the uniform ``sink + i*block`` grid — refines
    COMPLETE blocks only (option b) and hands the refined buffer downstream."""

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        device = get_local_torch_device()
        session = self.require_session(batch, context="SANA-WM realtime chain")
        inputs = session.get_or_create_state(SanaWMSessionInputsState)
        st = session.get_or_create_state(SanaWMRefinerChainState)
        stage1 = batch.latents  # growing stage-1 buffer from the denoise stage
        if stage1 is None:
            raise ValueError("refiner stage expects the stage-1 latent buffer")

        with set_forward_context(
            current_timestep=batch.block_idx, attn_metadata=None, forward_batch=batch
        ):
            if st.runner is None:
                st.block_size = int(inputs.num_frame_per_block)
                st.sink_size = int(inputs.sink_size)
                first_latent = batch.image_latent
                st.refined_full = first_latent.detach().clone()
                st.runner = self._build_chain_refiner_runner(
                    inputs,
                    st,
                    batch,
                    device=device,
                    spatial_shape=(
                        int(first_latent.shape[3]),
                        int(first_latent.shape[4]),
                    ),
                )

            frontier = st.refined_full.shape[2]
            end_f = stage1.shape[2]
            while True:
                block_start = frontier
                block_end = block_start + st.block_size  # complete blocks only
                if block_end > end_f:
                    break
                sink_seed = (
                    stage1[:, :, : st.sink_size]
                    if block_start == st.sink_size
                    else None
                )
                refined = st.runner.refine_block(
                    block_idx=st.next_ref_idx,
                    clean_block=stage1[:, :, block_start:block_end].contiguous(),
                    block_start=block_start,
                    block_end=block_end,
                    sink_seed_frames=sink_seed,
                )
                st.refined_full = torch.cat(
                    [st.refined_full, refined.to(st.refined_full.dtype)], dim=2
                )
                st.next_ref_idx += 1
                frontier = block_end

        batch.latents = st.refined_full
        return batch

    def _build_chain_refiner_runner(
        self, inputs, st, batch, *, device, spatial_shape
    ) -> RefinerChunkRunner:
        # Reuse the parity-validated builder via a state adapter that exposes
        # prompt/sink/block/kv_max in the legacy state shape it expects.
        legacy = SimpleNamespace(
            prompt=inputs.prompt,
            sink_size=st.sink_size,
            refiner_block_size=st.block_size,
            refiner_kv_max_frames=int(
                batch.extra.get(
                    "sana_wm_refiner_kv_max_frames", DEFAULT_REFINER_KV_MAX_FRAMES
                )
            ),
        )
        return self._build_refiner_runner(
            legacy,
            batch,
            device=device,
            spatial_shape=spatial_shape,
            seed=int(batch.extra.get("sana_wm_refiner_seed", batch.seed)),
            fps=float(batch.fps),
        )


class SanaWMCausalDecodeChainStage(SanaWMRealtimeStage):
    """Causal-VAE chunk decode past this session's decode frontier."""

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        device = get_local_torch_device()
        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]

        self.vae = self.vae.to(device=device, dtype=vae_dtype).eval()
        configure_sana_wm_ltx2_vae_for_long_video(self.vae, server_args.pipeline_config)
        session = self.require_session(batch, context="SANA-WM realtime chain")
        st = session.get_or_create_state(RealtimeCausalDecodeState)
        src = batch.latents
        if src is None or src.shape[2] <= st.next_dec_idx:
            return self._empty_output(batch)
        with set_forward_context(
            current_timestep=batch.block_idx, attn_metadata=None, forward_batch=batch
        ):
            frames = self._decode_chunk(
                src[:, :, st.next_dec_idx :], st, server_args, vae_dtype=vae_dtype
            )
        st.next_dec_idx = src.shape[2]
        return OutputBatch(output=frames.to(torch.float32), metrics=batch.metrics)
