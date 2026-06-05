# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import (
    scale_and_shift_latents,
)

from .base import (
    SanaWMBeforeDenoisingStage,
    sana_wm_normalize_vae_latents,
)
from .refiner import (
    STAGE_2_DISTILLED_SIGMA_VALUES,
    SanaWMLTX2RefinerStage,
    _unwrap_diffusers_ltx2_refiner,
)
from .streaming_refiner import (
    RefinerChunkRunner,
    _RefinerCore,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

from .utils import (
    TARGET_HEIGHT,
    TARGET_WIDTH,
    action_string_to_c2w,
    normalize_camera_actions,
    estimate_intrinsics_with_pi3x,
    load_camera,
    load_intrinsics,
    pil_to_model_tensor,
    resize_and_center_crop,
)

logger = init_logger(__name__)

# Local aliases for the SANA-WM pixel resolution (single source: utils.TARGET_HEIGHT/WIDTH).
SANA_WM_HEIGHT = TARGET_HEIGHT
SANA_WM_WIDTH = TARGET_WIDTH
DEFAULT_REFINER_BLOCK_SIZE = 3
DEFAULT_REFINER_KV_MAX_FRAMES = 11


@contextmanager
def _deterministic_vae_encode_context():
    prev_benchmark = torch.backends.cudnn.benchmark
    prev_deterministic = torch.backends.cudnn.deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark = prev_benchmark
        torch.backends.cudnn.deterministic = prev_deterministic


def _normalize_camera_actions(payload: Any) -> list[list[str]]:
    return normalize_camera_actions(payload, allow_none=True)


def _actions_to_action_string(actions: list[list[str]]) -> str:
    if not actions:
        return "none-1"

    segments: list[str] = []
    current = tuple(sorted(set(actions[0])))
    count = 0
    for frame_actions in actions:
        normalized = tuple(sorted(set(frame_actions)))
        if normalized == current:
            count += 1
            continue
        key = "".join(current) if current else "none"
        segments.append(f"{key}-{count}")
        current = normalized
        count = 1
    key = "".join(current) if current else "none"
    segments.append(f"{key}-{count}")
    return ",".join(segments)


def _normalize_intrinsics_array(arr: Any, num_frames: int) -> np.ndarray:
    intrinsics = np.asarray(arr, dtype=np.float32)
    if intrinsics.shape == (4,):
        return np.broadcast_to(intrinsics, (num_frames, 4)).copy()
    if intrinsics.shape == (3, 3):
        vec = np.array(
            [
                intrinsics[0, 0],
                intrinsics[1, 1],
                intrinsics[0, 2],
                intrinsics[1, 2],
            ],
            dtype=np.float32,
        )
        return np.broadcast_to(vec, (num_frames, 4)).copy()
    if intrinsics.ndim == 2 and intrinsics.shape[1] == 4:
        if intrinsics.shape[0] < num_frames:
            pad = np.broadcast_to(intrinsics[-1:], (num_frames - intrinsics.shape[0], 4))
            intrinsics = np.concatenate([intrinsics, pad], axis=0)
        return intrinsics[:num_frames].copy()
    if intrinsics.ndim == 3 and intrinsics.shape[1:] == (3, 3):
        if intrinsics.shape[0] < num_frames:
            pad = np.broadcast_to(
                intrinsics[-1:], (num_frames - intrinsics.shape[0], 3, 3)
            )
            intrinsics = np.concatenate([intrinsics, pad], axis=0)
        intrinsics = intrinsics[:num_frames]
        return np.stack(
            [
                intrinsics[:, 0, 0],
                intrinsics[:, 1, 1],
                intrinsics[:, 0, 2],
                intrinsics[:, 1, 2],
            ],
            axis=1,
        ).astype(np.float32)
    raise ValueError(
        "intrinsics must have shape (4,), (3,3), (F,4), or (F,3,3), "
        f"got {intrinsics.shape}"
    )


def _motion_param(batch: Req, name: str, default: float) -> float:
    value = (batch.condition_inputs or {}).get(name)
    if value is None:
        value = batch.extra.get(f"sana_wm_{name}", default)
    return float(value)


class SanaWMRealtimeStage(PipelineStage):
    def __init__(
        self,
        *,
        transformer: torch.nn.Module,
        vae: torch.nn.Module,
        model_path: str,
        refiner_stage: SanaWMLTX2RefinerStage | None = None,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.vae = vae
        self.model_path = model_path
        # Injected by the pipeline; we reuse its transformer / text encoder to
        # drive the RefinerChunkRunner. None -> stage-1-only output.
        self.refiner_stage = refiner_stage
        self.first_frame_latent_cache = None

    @property
    def role_affinity(self):
        return RoleType.MONOLITHIC

    @property
    def parallelism_type(self) -> StageParallelismType:
        # realtime session state contains runtime-only iterators and runners
        return StageParallelismType.REPLICATED

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        return [
            ComponentUse(
                stage_name,
                "transformer",
                target_dtype=PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision],
                memory_intensive=True,
                keep_ready_after_warmup=True,
            ),
            ComponentUse(
                stage_name,
                "vae",
                target_dtype=PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision],
                keep_ready_after_warmup=True,
            ),
        ]

    def _empty_output(self, batch: Req) -> OutputBatch:
        output = torch.empty(
            (1, 3, 0, int(batch.height or SANA_WM_HEIGHT), int(batch.width or SANA_WM_WIDTH)),
            dtype=torch.float32,
            device=get_local_torch_device(),
        )
        return OutputBatch(output=output, metrics=batch.metrics)

    def _prepare_image(
        self, batch: Req
    ) -> tuple[Image.Image, Image.Image, tuple[int, int], tuple[int, int], tuple[int, int]]:
        if isinstance(batch.condition_image, Image.Image):
            original = batch.condition_image.convert("RGB")
        elif batch.image_path is not None and isinstance(batch.image_path, str):
            original = Image.open(batch.image_path).convert("RGB")
        else:
            raise ValueError("SANA-WM realtime requires a first-frame image")
        cropped, src_size, resized_size, crop_offset = resize_and_center_crop(
            original, SANA_WM_HEIGHT, SANA_WM_WIDTH
        )
        return cropped, original, src_size, resized_size, crop_offset

    def _prepare_static_camera(
        self,
        batch: Req,
        *,
        num_frames: int,
        translation_speed: float,
        rotation_speed_deg: float,
    ) -> np.ndarray | None:
        condition_inputs = batch.condition_inputs or {}
        camera_path = condition_inputs.get("camera_path")
        if camera_path is not None:
            c2w = load_camera(Path(str(camera_path)))
        elif condition_inputs.get("camera") is not None:
            c2w = np.asarray(condition_inputs["camera"], dtype=np.float32)
        elif condition_inputs.get("action") is not None:
            c2w = action_string_to_c2w(
                str(condition_inputs["action"]),
                translation_speed=translation_speed,
                rotation_speed_deg=rotation_speed_deg,
            )
        else:
            return None

        if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
            raise ValueError(f"camera trajectory must have shape (F,4,4), got {c2w.shape}")
        if c2w.shape[0] < num_frames:
            pad = np.broadcast_to(c2w[-1:], (num_frames - c2w.shape[0], 4, 4))
            c2w = np.concatenate([c2w, pad], axis=0)
        return c2w[:num_frames].astype(np.float32)

    def _prepare_intrinsics(
        self,
        batch: Req,
        state,
        *,
        num_frames: int,
        device: torch.device,
    ) -> np.ndarray:
        condition_inputs = batch.condition_inputs or {}
        if condition_inputs.get("intrinsics_path") is not None:
            return load_intrinsics(Path(str(condition_inputs["intrinsics_path"])), num_frames)
        if condition_inputs.get("intrinsics") is not None:
            return _normalize_intrinsics_array(condition_inputs["intrinsics"], num_frames)
        if state.intrinsics_raw is not None:
            cached = state.intrinsics_raw
            if cached.shape[0] < num_frames:
                # Open-ended growth: hold the last row (fixed-horizon sessions
                # cache the full-length array, so this never triggers).
                pad = np.broadcast_to(
                    cached[-1:], (num_frames - cached.shape[0],) + cached.shape[1:]
                )
                cached = np.concatenate([cached, pad], axis=0)
            return cached
        if state.intrinsics_image is None:
            raise ValueError("SANA-WM image is not initialized")
        estimated = estimate_intrinsics_with_pi3x(state.intrinsics_image, device)
        return np.broadcast_to(estimated, (num_frames, 4)).copy()

    def _append_realtime_camera_actions(
        self, batch: Req, state
    ) -> None:
        actions = _normalize_camera_actions(
            (batch.condition_inputs or {}).get("camera_actions")
        )
        if actions:
            state.camera_actions.extend(actions)
            if (
                state.max_camera_actions > 0
                and len(state.camera_actions) > state.max_camera_actions
            ):
                del state.camera_actions[state.max_camera_actions :]

    def _camera_from_state(
        self,
        state,
        *,
        num_frames: int,
        translation_speed: float,
        rotation_speed_deg: float,
    ) -> np.ndarray:
        if state.static_c2w is not None:
            c2w = state.static_c2w
            if c2w.shape[0] < num_frames:
                # Open-ended growth: hold the last pose (fixed-horizon sessions
                # always precompute the full trajectory, so this never triggers).
                pad = np.broadcast_to(
                    c2w[-1:], (num_frames - c2w.shape[0], 4, 4)
                )
                c2w = np.concatenate([c2w, pad], axis=0)
            return c2w[:num_frames]

        num_actions = max(0, num_frames - 1)
        actions = list(state.camera_actions[:num_actions])
        if len(actions) < num_actions:
            actions.extend([[] for _ in range(num_actions - len(actions))])
        return action_string_to_c2w(
            _actions_to_action_string(actions),
            translation_speed=translation_speed,
            rotation_speed_deg=rotation_speed_deg,
        )[:num_frames]

    @torch.inference_mode()
    def _encode_first_frame(
        self,
        image: Image.Image,
        *,
        device: torch.device,
        vae_dtype: torch.dtype,
        latent_dtype: torch.dtype,
    ) -> torch.Tensor:
        if hasattr(self.vae, "enable_tiling"):
            self.vae.enable_tiling()
        image_tensor = pil_to_model_tensor(image, device=device, dtype=vae_dtype)
        with _deterministic_vae_encode_context():
            # Must use the SAME shared core as the batch path's _vae_encode_image:
            # drifting this was a parity root cause.
            z = SanaWMBeforeDenoisingStage._extract_vae_latents(
                self.vae.encode(image_tensor.to(device=device, dtype=vae_dtype))
            ).float()
        z = sana_wm_normalize_vae_latents(
            self.vae, z, getattr(self, "_pipeline_config", None)
        )
        return z.to(device=device, dtype=latent_dtype)

    def _first_frame_cache_key(
        self,
        batch: Req,
        *,
        device: torch.device,
        vae_dtype: torch.dtype,
        latent_dtype: torch.dtype,
    ) -> tuple | None:
        if batch.image_path is None or not isinstance(batch.image_path, str):
            return None
        stat = os.stat(batch.image_path)
        return (
            batch.image_path,
            stat.st_mtime_ns,
            stat.st_size,
            device.type,
            device.index,
            vae_dtype,
            latent_dtype,
        )

    @torch.inference_mode()
    def _get_first_frame_latent(
        self,
        batch: Req,
        image: Image.Image,
        *,
        device: torch.device,
        vae_dtype: torch.dtype,
        latent_dtype: torch.dtype,
    ) -> torch.Tensor:
        cache_key = self._first_frame_cache_key(
            batch,
            device=device,
            vae_dtype=vae_dtype,
            latent_dtype=latent_dtype,
        )
        if (
            cache_key is not None
            and self.first_frame_latent_cache is not None
            and self.first_frame_latent_cache[0] == cache_key
        ):
            return self.first_frame_latent_cache[1]

        first_latent = self._encode_first_frame(
            image,
            device=device,
            vae_dtype=vae_dtype,
            latent_dtype=latent_dtype,
        )
        if cache_key is not None:
            self.first_frame_latent_cache = (cache_key, first_latent.detach())
        return first_latent

    def _ensure_conv_cache(self, state) -> dict:
        """Lazily allocate the per-conv decoder cache threaded across chunks."""
        if state.conv_cache is None:
            state.conv_cache = self.vae.reset_decoder_cache()
        return state.conv_cache

    def _scale_and_shift(
        self,
        latents: torch.Tensor,
        server_args: ServerArgs,
    ) -> torch.Tensor:
        """De-normalize latents before VAE decode (shared implementation)."""
        return scale_and_shift_latents(latents, server_args, self.vae)

    @torch.inference_mode()
    def _decode_chunk(
        self,
        latents: torch.Tensor,
        state,
        server_args: ServerArgs,
        *,
        vae_dtype: torch.dtype,
    ) -> torch.Tensor:
        if latents.shape[2] == 0:
            return torch.empty(
                (latents.shape[0], 3, 0, SANA_WM_HEIGHT, SANA_WM_WIDTH),
                dtype=torch.float32,
                device=latents.device,
            )
        conv_cache = self._ensure_conv_cache(state)
        z = self._scale_and_shift(latents.to(vae_dtype), server_args)
        decoded = self.vae.decode_chunk(z, conv_cache)
        return (decoded / 2 + 0.5).clamp(0, 1)

    def _build_refiner_runner(
        self,
        state,
        batch: Req,
        *,
        device: torch.device,
        spatial_shape: tuple[int, int],
        seed: int,
        fps: float,
    ) -> RefinerChunkRunner:
        """Build the chunked LTX-2 refiner runner that carries sink/history KV across blocks."""
        rs = self.refiner_stage
        if rs is None:
            raise RuntimeError("SANA-WM realtime refiner stage is not initialized")
        # Keep refiner sub-modules resident: the realtime stage runs the refiner
        # directly, outside the offline stage's use_declared_component context.
        for _mod in (rs.text_encoder, rs.connectors, rs.transformer):
            if _mod is not None:
                _mod.to(device)
        prompt_embeds, prompt_mask = rs._encode_prompt(state.prompt, device)
        unwrapped = _unwrap_diffusers_ltx2_refiner(rs.transformer)
        core = _RefinerCore(unwrapped, device, rs.dtype)
        sigmas = torch.tensor(
            STAGE_2_DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=device
        )
        return RefinerChunkRunner(
            core,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_mask,
            fps=fps,
            sigmas=sigmas,
            source_sink_frames=state.sink_size,
            block_size=state.refiner_block_size,
            kv_max_frames=state.refiner_kv_max_frames,
            seed=seed,
            spatial_shape=spatial_shape,
        )

    @staticmethod
    def _is_open_ended(batch: Req) -> bool:
        """Open-ended = the init request carried no num_frames.

        The adapter flags this via condition_inputs (build_sampling_params strips
        None fields, so batch.num_frames would otherwise carry the SamplingParams
        default and be indistinguishable from an explicit request)."""
        condition_inputs = batch.condition_inputs or {}
        if bool(condition_inputs.get("sana_wm_open_ended")):
            return True
        num_frames = getattr(batch, "num_frames", None)
        return num_frames is None or int(num_frames) <= 0

