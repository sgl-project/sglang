# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
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
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)

# Incremental stage-1 session, chunked LTX-2 refiner runner, and causal-VAE chunk decode.
from .realtime import (
    SanaWMRealtimeSession,
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
from sglang.multimodal_gen.runtime.realtime.session import BaseRealtimeState
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

from .utils import (
    action_string_to_c2w,
    estimate_intrinsics_with_pi3x,
    load_camera,
    load_intrinsics,
    pil_to_model_tensor,
    prepare_camera_conditions,
    resize_and_center_crop,
    snap_num_frames,
    transform_intrinsics_for_crop,
)

logger = init_logger(__name__)

SANA_WM_HEIGHT = 704
SANA_WM_WIDTH = 1280
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


def _as_int_tuple(values: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if values is None:
        return default
    return tuple(int(v) for v in values)


def _normalize_camera_actions(payload: Any) -> list[list[str]]:
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise ValueError("camera_actions must be list[list[str]]")
    out: list[list[str]] = []
    for item in payload:
        if not isinstance(item, list):
            raise ValueError("camera_actions must be list[list[str]]")
        out.append([str(key).lower() for key in item])
    return out


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


def _vae_scaling_factor(vae: torch.nn.Module) -> float | torch.Tensor:
    config = getattr(vae, "config", None)
    arch = getattr(config, "arch_config", None)
    value = getattr(arch, "scaling_factor", None)
    if value is None:
        value = getattr(config, "scaling_factor", None)
    is_zero = False
    if isinstance(value, torch.Tensor):
        is_zero = bool(value.numel() == 1 and value.item() == 0)
    elif value is not None:
        is_zero = value == 0
    if value is None or is_zero:
        value = getattr(vae, "scaling_factor", 1.0)
    return value


def _vae_stats(
    vae: torch.nn.Module,
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    channels = int(tensor.shape[1])
    mean = getattr(vae, "latents_mean", None)
    std = getattr(vae, "latents_std", None)
    if mean is None:
        mean = torch.zeros(channels, device=tensor.device, dtype=tensor.dtype)
    if std is None:
        std = torch.ones(channels, device=tensor.device, dtype=tensor.dtype)
    mean = mean.view(1, -1, 1, 1, 1).to(tensor.device, tensor.dtype)
    std = std.view(1, -1, 1, 1, 1).to(tensor.device, tensor.dtype)
    return mean, std


class SanaWMStreamingState(BaseRealtimeState):
    def __init__(self):
        super().__init__()
        self.initialized = False
        self.prompt: str = ""
        self.image: Image.Image | None = None
        self.intrinsics_image: Image.Image | None = None
        self.src_size: tuple[int, int] | None = None
        self.resized_size: tuple[int, int] | None = None
        self.crop_offset: tuple[int, int] | None = None
        self.intrinsics_raw: np.ndarray | None = None
        self.camera_actions: list[list[str]] = []
        self.max_camera_actions = 0
        self.static_c2w: np.ndarray | None = None
        self.latents: torch.Tensor | None = None
        # refined_full accumulates sink + refined frames across steps (refiner path).
        self.refined_full: torch.Tensor | None = None
        self.rollover_first_latent: torch.Tensor | None = None
        # Incremental stage-1 session.
        self.session: SanaWMRealtimeSession | None = None
        # This step's camera tensors for session.step.
        self.raymap: torch.Tensor | None = None
        self.chunk_plucker: torch.Tensor | None = None
        self.stage1_chunks = 0
        self.stage1_idx = 0
        self.produced_until = 0
        self.tick = 0
        self.use_refiner = False
        self.refiner_runner: RefinerChunkRunner | None = None
        self.sink_size = 1
        self.refiner_block_size = DEFAULT_REFINER_BLOCK_SIZE
        self.refiner_kv_max_frames = DEFAULT_REFINER_KV_MAX_FRAMES
        self.n_blocks = 0
        self.next_ref_idx = 0
        self.next_dec_idx = 0
        # Per-conv decoder cache threaded through vae.decode_chunk across chunks.
        self.conv_cache: dict | None = None
        self.latent_t = 0
        self.translation_speed = 0.04
        self.rotation_speed_deg = 1.2

    def dispose(self):
        self.conv_cache = None
        self.initialized = False
        self.prompt = ""
        self.image = None
        self.intrinsics_image = None
        self.src_size = None
        self.resized_size = None
        self.crop_offset = None
        self.intrinsics_raw = None
        self.camera_actions = []
        self.max_camera_actions = 0
        self.static_c2w = None
        self.session = None
        self.raymap = None
        self.chunk_plucker = None
        self.stage1_chunks = 0
        self.stage1_idx = 0
        self.produced_until = 0
        self.tick = 0
        self.use_refiner = False
        self.refiner_runner = None
        self.sink_size = 1
        self.refiner_block_size = DEFAULT_REFINER_BLOCK_SIZE
        self.refiner_kv_max_frames = DEFAULT_REFINER_KV_MAX_FRAMES
        self.n_blocks = 0
        self.next_ref_idx = 0
        self.next_dec_idx = 0
        self.latent_t = 0
        self.translation_speed = 0.04
        self.rotation_speed_deg = 1.2
        self.latents = None
        self.refined_full = None
        self.rollover_first_latent = None


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
        # When the pipeline loads the LTX-2 refiner sub-modules it injects a
        # configured refiner stage here; we reuse its loaded transformer / text
        # encoder to drive the RefinerChunkRunner. None -> stage-1-only output.
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
        state: SanaWMStreamingState,
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
            return state.intrinsics_raw
        if state.intrinsics_image is None:
            raise ValueError("SANA-WM image is not initialized")
        estimated = estimate_intrinsics_with_pi3x(state.intrinsics_image, device)
        return np.broadcast_to(estimated, (num_frames, 4)).copy()

    def _append_realtime_camera_actions(
        self, batch: Req, state: SanaWMStreamingState
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
        state: SanaWMStreamingState,
        *,
        num_frames: int,
        translation_speed: float,
        rotation_speed_deg: float,
    ) -> np.ndarray:
        if state.static_c2w is not None:
            return state.static_c2w[:num_frames]

        num_actions = max(0, num_frames - 1)
        actions = list(state.camera_actions[:num_actions])
        if len(actions) < num_actions:
            actions.extend([[] for _ in range(num_actions - len(actions))])
        return action_string_to_c2w(
            _actions_to_action_string(actions),
            translation_speed=translation_speed,
            rotation_speed_deg=rotation_speed_deg,
        )[:num_frames]

    def _update_camera_tensors(
        self,
        batch: Req,
        state: SanaWMStreamingState,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if state.session is None:
            return
        if state.src_size is None or state.resized_size is None or state.crop_offset is None:
            raise ValueError("SANA-WM crop metadata is not initialized")

        num_frames = int(batch.num_frames)
        condition_inputs = batch.condition_inputs or {}
        if "translation_speed" in condition_inputs:
            state.translation_speed = float(condition_inputs["translation_speed"])
        if "rotation_speed_deg" in condition_inputs:
            state.rotation_speed_deg = float(condition_inputs["rotation_speed_deg"])
        c2w = self._camera_from_state(
            state,
            num_frames=num_frames,
            translation_speed=state.translation_speed,
            rotation_speed_deg=state.rotation_speed_deg,
        )
        intrinsics_raw = self._prepare_intrinsics(
            batch,
            state,
            num_frames=num_frames,
            device=device,
        )
        state.intrinsics_raw = intrinsics_raw
        intrinsics = transform_intrinsics_for_crop(
            intrinsics_raw,
            state.src_size,
            state.resized_size,
            state.crop_offset,
        )
        camera = prepare_camera_conditions(c2w, intrinsics)
        raymap = camera["raymap"].unsqueeze(0).to(device=device, dtype=dtype)
        chunk_plucker = camera["chunk_plucker"].unsqueeze(0).to(
            device=device, dtype=dtype
        )
        # Carried into session.step as the next chunk's camera conditioning.
        state.raymap = raymap
        state.chunk_plucker = chunk_plucker

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
            posterior = self.vae.encode(image_tensor.to(device=device, dtype=vae_dtype)).latent_dist
            z = posterior.mode()
        mean, std = _vae_stats(self.vae, z)
        scaling_factor = _vae_scaling_factor(self.vae)
        z = (z - mean) * scaling_factor / std
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

    def _ensure_conv_cache(self, state: SanaWMStreamingState) -> dict:
        """Lazily allocate the per-conv decoder cache threaded across chunks."""
        if state.conv_cache is None:
            state.conv_cache = self.vae.reset_decoder_cache()
        return state.conv_cache

    def _scale_and_shift(
        self,
        latents: torch.Tensor,
        server_args: ServerArgs,
    ) -> torch.Tensor:
        """De-normalize latents before VAE decode."""
        scaling_factor, shift_factor = server_args.pipeline_config.get_decode_scale_and_shift(
            latents.device, latents.dtype, self.vae
        )
        if isinstance(scaling_factor, torch.Tensor):
            latents = latents / scaling_factor.to(latents.device, latents.dtype)
        else:
            latents = latents / scaling_factor
        if shift_factor is not None:
            if isinstance(shift_factor, torch.Tensor):
                latents = latents + shift_factor.to(latents.device, latents.dtype)
            else:
                latents = latents + shift_factor
        return latents

    @torch.inference_mode()
    def _decode_chunk(
        self,
        latents: torch.Tensor,
        state: SanaWMStreamingState,
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
        state: SanaWMStreamingState,
        batch: Req,
        *,
        device: torch.device,
        spatial_shape: tuple[int, int],
        seed: int,
        fps: float,
    ) -> RefinerChunkRunner:
        """Build the chunked LTX-2 refiner runner from the loaded refiner stage.

        Encodes the prompt through the refiner's Gemma-3, wraps the unwrapped
        diffusers LTX-2 refiner transformer in ``_RefinerCore``, and constructs the
        ``RefinerChunkRunner`` that carries the sink/history KV across blocks.
        """
        rs = self.refiner_stage
        if rs is None:
            raise RuntimeError("SANA-WM realtime refiner stage is not initialized")
        # Keep the refiner sub-modules resident on the model device. The offline
        # refiner stage moves them per-call via use_declared_component / the
        # offload manager, but the realtime session encodes the prompt and runs
        # the refiner directly outside that context.
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

    def _initialize_state(
        self,
        batch: Req,
        state: SanaWMStreamingState,
        *,
        transformer: torch.nn.Module,
        device: torch.device,
        weight_dtype: torch.dtype,
        vae_dtype: torch.dtype,
        first_latent: torch.Tensor | None = None,
        intrinsics_raw: np.ndarray | None = None,
        translation_speed: float | None = None,
        rotation_speed_deg: float | None = None,
    ) -> None:
        image, intrinsics_image, src_size, resized_size, crop_offset = self._prepare_image(batch)
        state.image = image
        state.intrinsics_image = intrinsics_image
        state.src_size = src_size
        state.resized_size = resized_size
        state.crop_offset = crop_offset
        state.intrinsics_raw = (
            intrinsics_raw.copy() if intrinsics_raw is not None else None
        )
        state.prompt = str(batch.prompt)

        num_frames = snap_num_frames(int(batch.num_frames), stride=8)
        batch.num_frames = num_frames
        state.max_camera_actions = max(0, num_frames - 1)
        self._append_realtime_camera_actions(batch, state)
        if first_latent is None:
            first_latent = self._get_first_frame_latent(
                batch,
                image,
                device=device,
                vae_dtype=vae_dtype,
                latent_dtype=weight_dtype,
            )
        else:
            first_latent = first_latent.to(device=device, dtype=weight_dtype)
        latent_t = (num_frames - 1) // 8 + 1
        generator = batch.generator[0] if isinstance(batch.generator, list) else batch.generator
        if generator is None:
            generator = torch.Generator(device=device).manual_seed(int(batch.seed))
        latents = torch.randn(
            1,
            first_latent.shape[1],
            latent_t,
            first_latent.shape[-2],
            first_latent.shape[-1],
            device=device,
            dtype=weight_dtype,
            generator=generator,
        )
        latents[:, :, :1] = first_latent

        cond = batch.prompt_embeds[0].to(device=device, dtype=weight_dtype)
        cond_mask = (
            batch.prompt_attention_mask[0]
            if isinstance(batch.prompt_attention_mask, list)
            else batch.prompt_attention_mask
        )
        if cond_mask is None:
            cond_mask = torch.ones(cond.shape[0], cond.shape[2], device=device)
        cond_mask = cond_mask.to(device=device)
        neg = None
        if batch.do_classifier_free_guidance and batch.negative_prompt_embeds:
            neg = batch.negative_prompt_embeds[0].to(device=device, dtype=weight_dtype)
            if neg.shape[0] == 0:
                neg = None
        if neg is None:
            neg = torch.zeros_like(cond)
        neg_mask = None
        if batch.do_classifier_free_guidance:
            if batch.negative_attention_mask:
                neg_mask = batch.negative_attention_mask[0].to(device=device)
                if neg_mask.shape[0] == 0:
                    neg_mask = None
            if neg_mask is None:
                neg_mask = torch.ones_like(cond_mask)

        state.translation_speed = _motion_param(
            batch,
            "translation_speed",
            0.04 if translation_speed is None else translation_speed,
        )
        state.rotation_speed_deg = _motion_param(
            batch,
            "rotation_speed_deg",
            1.2 if rotation_speed_deg is None else rotation_speed_deg,
        )
        state.static_c2w = self._prepare_static_camera(
            batch,
            num_frames=num_frames,
            translation_speed=state.translation_speed,
            rotation_speed_deg=state.rotation_speed_deg,
        )

        num_frame_per_block = int(batch.extra.get("sana_wm_num_frame_per_block", 3))
        state.sink_size = int(batch.extra.get("sana_wm_sink_size", 1))
        state.refiner_block_size = num_frame_per_block
        state.refiner_kv_max_frames = DEFAULT_REFINER_KV_MAX_FRAMES

        # Incremental stage-1 session.
        session = SanaWMRealtimeSession(
            transformer,
            denoising_step_list=_as_int_tuple(
                batch.extra.get("sana_wm_denoising_step_list"),
                (1000, 960, 889, 727, 0),
            ),
            num_frame_per_block=num_frame_per_block,
            num_cached_blocks=int(batch.extra.get("sana_wm_num_cached_blocks", 2)),
            sink_token=state.sink_size > 0,
            cfg_scale=float(batch.guidance_scale),
            device=device,
            dtype=weight_dtype,
            vae=self.vae,
        )
        session.reset(
            first_latent,
            cond,
            neg_embeds=neg if batch.do_classifier_free_guidance else None,
            pos_mask=cond_mask,
            neg_mask=neg_mask,
            # Seeded full pre-noise (frame 0 = condition slot) sliced per chunk,
            # so streaming output is reproducible (matches the offline draw).
            noise_buffer=latents,
            seed=(None if batch.seed is None else int(
                batch.seed[0] if isinstance(batch.seed, list) else batch.seed
            )),
            # Front-loaded autoregressive segmentation (chunk 0 carries the
            # remainder) so the chunk boundaries / KV / camera windowing align
            # instead of using uniform blocks.
            total_latent_frames=latent_t,
        )
        state.session = session
        self._update_camera_tensors(batch, state, device=device, dtype=weight_dtype)

        # Stage-1 grows session.latents chunk-by-chunk; the cap is latent_t.
        # Number of stage-1 chunks after the first (which carries the cond frame):
        # chunk 0 -> 1 + nfpb frames, each later chunk -> nfpb frames.
        state.latents = session.latents
        state.latent_t = latent_t
        state.stage1_chunks = max(
            0, math.ceil((latent_t - 1) / num_frame_per_block)
        )
        active_frames = latent_t - state.sink_size
        if active_frames <= 0:
            raise ValueError(
                f"SANA-WM active latent frames must be positive, got {active_frames}"
            )
        state.n_blocks = math.ceil(active_frames / state.refiner_block_size)
        state.rollover_first_latent = first_latent.detach().clone()

        if self.refiner_stage is not None:
            state.refiner_runner = self._build_refiner_runner(
                state,
                batch,
                device=device,
                spatial_shape=(int(first_latent.shape[3]), int(first_latent.shape[4])),
                seed=int(batch.extra.get("sana_wm_refiner_seed", batch.seed)),
                fps=float(batch.fps),
            )
            # refined buffer seeds the sink frame(s) (left unrefined).
            state.refined_full = first_latent.detach().clone()
            state.use_refiner = True
            logger.info(
                "SANA-WM realtime uses Stage-1 plus LTX-2 refiner streaming."
            )
        else:
            state.use_refiner = False
            logger.info("SANA-WM realtime uses Stage-1 only (no refiner stage).")

        state.conv_cache = self.vae.reset_decoder_cache()
        state.initialized = True
        logger.info(
            "SANA-WM realtime initialized: latent_t=%s stage1_chunks=%s "
            "refiner=%s decode_blocks=%s",
            latent_t,
            state.stage1_chunks,
            state.use_refiner,
            state.n_blocks,
        )

    def _advance_stage1(
        self,
        state: SanaWMStreamingState,
    ) -> tuple[int, int] | None:
        """Generate ONE stage-1 chunk via OUR incremental session.

        Returns ``(start_f, end_f)`` of the freshly produced latent frames in
        ``session.latents`` (which grows in place), or ``None`` once the latent
        horizon ``latent_t`` is reached (triggering a rollover upstream).
        """
        session = state.session
        if session is None or state.produced_until >= state.latent_t:
            return None
        start_f = session.latents.shape[2]
        session.step(
            camera_conditions=state.raymap,
            chunk_plucker=state.chunk_plucker,
            n_frames=state.refiner_block_size,
            decode=False,
        )
        end_f = session.latents.shape[2]
        state.latents = session.latents
        state.produced_until = end_f
        state.stage1_idx += 1
        return int(start_f), int(end_f)

    def _store_rollover_first_latent(
        self,
        state: SanaWMStreamingState,
        latents: torch.Tensor,
        frame_idx: int,
    ) -> None:
        state.rollover_first_latent = (
            latents[:, :, frame_idx : frame_idx + 1].detach().clone()
        )

    def _run_stage1_only_tick(
        self,
        state: SanaWMStreamingState,
        server_args: ServerArgs,
        *,
        vae_dtype: torch.dtype,
    ) -> torch.Tensor | None:
        advanced = self._advance_stage1(state)
        state.tick += 1
        if advanced is None or state.session is None:
            return None
        _, end_f = advanced
        src = state.session.latents
        seg = src[:, :, state.next_dec_idx : end_f]
        frames = self._decode_chunk(seg, state, server_args, vae_dtype=vae_dtype)
        self._store_rollover_first_latent(state, src, end_f - 1)
        state.next_dec_idx = end_f
        return frames

    def _run_refiner_tick(
        self,
        state: SanaWMStreamingState,
        server_args: ServerArgs,
        *,
        vae_dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if state.session is None or state.refined_full is None:
            return None
        if state.refiner_runner is None:
            raise RuntimeError("SANA-WM refiner runner is not initialized")

        advanced = self._advance_stage1(state)
        state.tick += 1
        if advanced is None:
            return None
        _, end_f = advanced

        # Refine the freshly generated stage-1 chunk, carrying sink/history KV.
        # ``refined_full`` holds sink frame(s) + all refined frames so far, so its
        # length is the next block's start.
        start_f = state.refined_full.shape[2]
        clean = state.session.latents[:, :, start_f:end_f].contiguous()
        sink_seed = (
            state.session.latents[:, :, : state.sink_size]
            if start_f == state.sink_size
            else None
        )
        refined = state.refiner_runner.refine_block(
            block_idx=start_f,
            clean_block=clean,
            block_start=start_f,
            block_end=end_f,
            sink_seed_frames=sink_seed,
        )
        state.refined_full = torch.cat(
            [state.refined_full, refined.to(state.refined_full.dtype)], dim=2
        )
        state.next_ref_idx += 1

        src = state.refined_full
        seg = src[:, :, state.next_dec_idx : end_f]
        frames = self._decode_chunk(seg, state, server_args, vae_dtype=vae_dtype)
        self._store_rollover_first_latent(state, src, end_f - 1)
        state.next_dec_idx = end_f
        return frames

    @torch.inference_mode()
    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        if batch.session is None:
            raise ValueError("SANA-WM realtime pipeline requires a realtime session")

        state = batch.session.get_or_create_state(SanaWMStreamingState)
        assert isinstance(state, SanaWMStreamingState)

        device = get_local_torch_device()
        weight_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision]
        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        transformer = self.transformer.to(device=device, dtype=weight_dtype).eval()
        self.vae = self.vae.to(device=device, dtype=vae_dtype).eval()
        batch.enable_sequence_shard = int(server_args.sp_degree or 1) > 1

        with set_forward_context(
            current_timestep=batch.block_idx,
            attn_metadata=None,
            forward_batch=batch,
        ):
            if batch.block_idx == 0 or not state.initialized:
                state.dispose()
                self._initialize_state(
                    batch,
                    state,
                    transformer=transformer,
                    device=device,
                    weight_dtype=weight_dtype,
                    vae_dtype=vae_dtype,
                )
            else:
                self._append_realtime_camera_actions(batch, state)
                self._update_camera_tensors(
                    batch,
                    state,
                    device=device,
                    dtype=weight_dtype,
                )

            if state.use_refiner:
                frames = self._run_refiner_tick(state, server_args, vae_dtype=vae_dtype)
            else:
                frames = self._run_stage1_only_tick(state, server_args, vae_dtype=vae_dtype)

            if frames is None and state.rollover_first_latent is not None:
                first_latent = state.rollover_first_latent
                intrinsics_raw = (
                    state.intrinsics_raw.copy()
                    if state.intrinsics_raw is not None
                    else None
                )
                translation_speed = state.translation_speed
                rotation_speed_deg = state.rotation_speed_deg
                logger.info(
                    "SANA-WM realtime horizon rollover: block_idx=%s, latent_t=%s",
                    batch.block_idx,
                    state.latent_t,
                )
                state.dispose()
                self._initialize_state(
                    batch,
                    state,
                    transformer=transformer,
                    device=device,
                    weight_dtype=weight_dtype,
                    vae_dtype=vae_dtype,
                    first_latent=first_latent,
                    intrinsics_raw=intrinsics_raw,
                    translation_speed=translation_speed,
                    rotation_speed_deg=rotation_speed_deg,
                )
                if state.use_refiner:
                    frames = self._run_refiner_tick(state, server_args, vae_dtype=vae_dtype)
                else:
                    frames = self._run_stage1_only_tick(state, server_args, vae_dtype=vae_dtype)

            if frames is None:
                return self._empty_output(batch)
            return OutputBatch(output=frames.to(torch.float32), metrics=batch.metrics)
