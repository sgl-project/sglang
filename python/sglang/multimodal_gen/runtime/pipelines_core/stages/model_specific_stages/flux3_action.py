# SPDX-License-Identifier: Apache-2.0
"""FLUX 3 Action stages: observation -> conditioning -> joint video/action denoising.

The policy follows the FLUX Action reference implementation
(https://github.com/black-forest-labs/flux-action): the camera frames are
composed onto one canvas and VAE-encoded as the ``video_cond`` stream, the
normalized state is the ``<action>_cond`` token, and future video latents are
denoised jointly with the action chunk. Only the actions are returned.

Everything that does not depend on the noised streams is computed once: the
text context per caption (cached across requests), the conditioning streams
per request, and the target streams' mode blocks per step (shared by the
conditional and unconditional CFG passes, since mode blocks never see text).
"""

from __future__ import annotations

import copy
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import msgspec
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from sglang.multimodal_gen.configs.pipeline_configs.flux3_action import (
    Flux3ActionPipelineConfig,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.models.dits.flux3 import (
    Flux3SegmentState,
    Flux3Transformer,
)
from sglang.multimodal_gen.runtime.models.encoders.flux3_text_encoder import (
    Flux3TextEncoder,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.models.vaes.flux3_video_vae import Flux3VideoVAE
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.vla import (
    synchronize_vla_action_tensor,
    vla_options,
    vla_state,
    vla_timings,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

DROID_CAMERA_HW = (360, 640)
DROID_COMPOSITE_HW = (540, 640)
LATENT_CHANNELS = 96
TEMPORAL_DOWNSAMPLE = 4
GRAY_LEVEL = 128


class Flux3ActionObservation(msgspec.Struct, frozen=True):
    canvas: torch.Tensor  # (3, Hc, Wc) in [-1, 1]
    state: torch.Tensor  # (D,) fp32, dataset units (gripper as stored)
    prompt: str


# ---------------------------------------------------------------- position ids
def _times_to_ids(seconds: torch.Tensor) -> torch.Tensor:
    """Seconds -> ids on the shared 10 ms clock."""
    return (seconds * 1000 // 10).to(torch.int64)


def _cartesian_ids(
    t: torch.Tensor, h: int, w: int, l_coord: torch.Tensor
) -> torch.Tensor:
    return torch.cartesian_prod(t, torch.arange(h), torch.arange(w), l_coord)


def pack_video(latent: torch.Tensor, first_frame: int, fps: float):
    """Latent ``(1, C, T, h, w)`` -> tokens ``(1, T*h*w, C)`` and ids; frame ``i`` at ``i * 4 / fps`` s."""
    _, _, t, h, w = latent.shape
    seconds = (
        torch.arange(first_frame, first_frame + t).float() * TEMPORAL_DOWNSAMPLE / fps
    )
    ids = _cartesian_ids(t=_times_to_ids(seconds), h=h, w=w, l_coord=torch.arange(1))
    return rearrange(latent, "b c t h w -> b (t h w) c"), ids[None]


def pack_action(values: torch.Tensor, seconds: torch.Tensor):
    """``(1, D, K)`` values at ``seconds (K,)`` -> tokens ``(1, K, D)`` with ids ``(t, 0, 0, 0)``."""
    ids = _cartesian_ids(t=_times_to_ids(seconds), h=1, w=1, l_coord=torch.arange(1))
    return values.transpose(1, 2), ids[None]


def text_ids(length: int) -> torch.Tensor:
    return _cartesian_ids(t=torch.arange(1), h=1, w=1, l_coord=torch.arange(length))[
        None
    ]


# ---------------------------------------------------------------- observation
def _as_image(value: Any) -> torch.Tensor:
    """A request image -> float ``(3, H, W)`` in ``[0, 1]``.

    Arrays and PIL images are HWC: uint8 pixels, or floats already in
    ``[0, 1]``. Tensors are uint8 ``(H, W, 3)`` or float ``(3, H, W)`` in
    ``[0, 1]`` (the reference policy's conventions).
    """
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.uint8 and value.ndim == 3 and value.shape[-1] == 3:
            return value.permute(2, 0, 1).float().div_(255.0)
        if value.is_floating_point() and value.ndim == 3 and value.shape[0] == 3:
            return _check_unit_range(value.float())
        raise ValueError(
            "image tensors must be uint8 (H, W, 3) or float (3, H, W), "
            f"got {value.dtype} {tuple(value.shape)}"
        )
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"expected an HWC RGB image, got shape {array.shape}")
    tensor = torch.from_numpy(np.require(array, requirements=["C", "W"]))
    if array.dtype == np.uint8:
        return tensor.permute(2, 0, 1).float().div_(255.0)
    if np.issubdtype(array.dtype, np.floating):
        return _check_unit_range(tensor.permute(2, 0, 1).float())
    raise ValueError(f"unsupported image dtype {array.dtype}")


def _check_unit_range(image: torch.Tensor) -> torch.Tensor:
    if image.numel() and (image.min() < 0.0 or image.max() > 1.0):
        raise ValueError("float images must be in [0, 1]")
    return image


def _canonical_camera(name: str, aliases: dict[str, str]) -> str:
    name = name.removeprefix("observation.images.").removeprefix("images.")
    return aliases.get(name, name)


def _pad_composite(composite: torch.Tensor, canvas_hw: tuple[int, int]) -> torch.Tensor:
    """DROID composite ``(3, 540, 640)`` in ``[0, 1]`` -> canvas ``(3, Hc, Wc)`` in ``[-1, 1]`` (reflect pad)."""
    if tuple(composite.shape) != (3, *DROID_COMPOSITE_HW):
        raise ValueError(f"composite must be 3x540x640, got {tuple(composite.shape)}")
    pad_right = canvas_hw[1] - composite.shape[-1]
    pad_bottom = canvas_hw[0] - composite.shape[-2]
    if pad_right < 0 or pad_bottom < 0:
        raise ValueError(f"canvas {canvas_hw} is smaller than the DROID composite")
    canvas = F.pad(composite[None], (0, pad_right, 0, pad_bottom), mode="reflect")[0]
    return canvas.mul_(2.0).sub_(1.0)


def _grid_canvas(cams: list[torch.Tensor], canvas_hw: tuple[int, int]) -> torch.Tensor:
    cols = int(np.ceil(np.sqrt(len(cams))))
    rows = int(np.ceil(len(cams) / cols))
    cell_h, cell_w = canvas_hw[0] // rows, canvas_hw[1] // cols
    canvas = cams[0].new_zeros(3, *canvas_hw)
    for i, cam in enumerate(cams):
        r, c = divmod(i, cols)
        canvas[:, r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w] = (
            F.interpolate(
                cam[None],
                size=(cell_h, cell_w),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )[0]
        )
    return canvas


def _compose_canvas(
    cams: list[torch.Tensor], layout: str, canvas_hw: tuple[int, int]
) -> torch.Tensor:
    """Camera frames ``(3, H, W)`` in ``[0, 1]`` (layout order) -> canvas in ``[-1, 1]``."""
    if layout == "droid":
        if len(cams) != 3 or any(tuple(c.shape[-2:]) != DROID_CAMERA_HW for c in cams):
            raise ValueError(
                "droid layout needs three 360x640 cameras [wrist, left, right]"
            )
        wrist, left, right = cams
        half = (DROID_CAMERA_HW[0] // 2, DROID_CAMERA_HW[1] // 2)
        bottom = torch.cat(
            [
                F.interpolate(
                    cam[None], size=half, mode="bilinear", align_corners=False
                )[0]
                for cam in (left, right)
            ],
            dim=-1,
        )
        return _pad_composite(torch.cat([wrist, bottom], dim=-2), canvas_hw=canvas_hw)
    if layout == "single":
        if len(cams) != 1:
            raise ValueError(f"single layout needs exactly one camera, got {len(cams)}")
        canvas = F.interpolate(
            cams[0][None],
            size=canvas_hw,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0]
    elif layout == "side_by_side":
        if len(cams) != 2 or canvas_hw[1] % 2:
            raise ValueError(
                "side_by_side layout needs two cameras and an even canvas width"
            )
        canvas = torch.cat(
            [
                F.interpolate(
                    cam[None],
                    size=(canvas_hw[0], canvas_hw[1] // 2),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )[0]
                for cam in cams
            ],
            dim=-1,
        )
    elif layout == "grid":
        canvas = _grid_canvas(cams, canvas_hw)
    else:
        raise ValueError(f"unknown camera layout {layout!r}")
    return canvas.mul_(2.0).sub_(1.0)


def _observation_canvas(
    observation: dict[str, Any], config: Flux3ActionPipelineConfig
) -> torch.Tensor:
    # OpenPI clients may also send cameras as top-level "observation.images.<name>".
    named = {
        k: v for k, v in observation.items() if k.startswith("observation.images.")
    }
    named.update(observation.get("images") or {})
    images = {
        _canonical_camera(name, config.camera_aliases): value
        for name, value in named.items()
    }
    if "composite" in images:
        if config.camera_layout != "droid":
            raise ValueError("a composite image requires the droid camera layout")
        return _pad_composite(
            _as_image(images["composite"]), canvas_hw=config.canvas_hw
        )
    missing = [key for key in config.image_keys if key not in images]
    if missing:
        raise KeyError(
            f"observation lacks cameras {missing}; expected {list(config.image_keys)} "
            "or a droid 'composite'"
        )
    cams = [_as_image(images[key]) for key in config.image_keys]
    if config.camera_layout == "grid":
        hw = (max(c.shape[-2] for c in cams), max(c.shape[-1] for c in cams))
        cams = [
            F.interpolate(
                c[None], size=hw, mode="bilinear", align_corners=False, antialias=True
            )[0]
            if tuple(c.shape[-2:]) != hw
            else c
            for c in cams
        ]
    return _compose_canvas(
        cams, layout=config.camera_layout, canvas_hw=config.canvas_hw
    )


def _observation_state(observation: dict[str, Any], state_dim: int) -> torch.Tensor:
    state = observation.get("state")
    if state is None:
        state = observation.get("observation.state")
    if state is None:
        raise KeyError("observation lacks 'state'")
    state = torch.as_tensor(np.asarray(state, dtype=np.float32)).reshape(-1)
    if state.shape != (state_dim,) or not torch.isfinite(state).all():
        raise ValueError(
            f"state must be {state_dim} finite values, got {tuple(state.shape)}"
        )
    return state


def parse_observation(
    observation: dict[str, Any], config: Flux3ActionPipelineConfig
) -> Flux3ActionObservation:
    prompt = observation.get("prompt") or observation.get("task") or ""
    if isinstance(prompt, (list, tuple)):
        if len(prompt) != 1:
            raise ValueError("FLUX 3 Action serves one observation per request")
        prompt = prompt[0]
    return Flux3ActionObservation(
        canvas=_observation_canvas(observation, config),
        state=_observation_state(observation, state_dim=config.state_dim),
        prompt=str(prompt),
    )


def _warmup_observation(config: Flux3ActionPipelineConfig) -> Flux3ActionObservation:
    canvas = torch.full((3, *config.canvas_hw), GRAY_LEVEL / 255.0 * 2.0 - 1.0)
    return Flux3ActionObservation(
        canvas=canvas, state=torch.zeros(config.state_dim), prompt=""
    )


# ---------------------------------------------------------------- action space
def _flip_gripper(x: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    """``x -> 1 - x`` on the gripper dims (self-inverse)."""
    if not dims:
        return x
    x = x.clone()
    x[..., list(dims)] = 1.0 - x[..., list(dims)]
    return x


def _bounds(stats: dict[str, list[float]], like: torch.Tensor):
    q01 = torch.as_tensor(stats["q01"], dtype=like.dtype, device=like.device)
    q99 = torch.as_tensor(stats["q99"], dtype=like.dtype, device=like.device)
    span = q99 - q01
    return q01, torch.where(span > 1e-6, span, torch.ones_like(span))


def normalize(x: torch.Tensor, stats, clip: float) -> torch.Tensor:
    if stats is None:
        return x
    q01, span = _bounds(stats, x)
    return (2.0 * (x - q01) / span - 1.0).clamp_(-clip, clip)


def denormalize(x: torch.Tensor, stats) -> torch.Tensor:
    if stats is None:
        return x
    q01, span = _bounds(stats, x)
    return (x + 1.0) * span / 2.0 + q01


def targets_to_actions(
    targets: torch.Tensor, state: torch.Tensor, config: Flux3ActionPipelineConfig
) -> torch.Tensor:
    """Normalized targets ``(K, D)`` and the observed state ``(D,)`` -> absolute commands (dataset units)."""
    flipped_state = _flip_gripper(state, dims=config.gripper_flip_dims)
    actions = denormalize(targets, stats=config.action_normalization)
    if config.action_parameterization == "joint_delta":
        integrated = flipped_state[None] + torch.cumsum(actions, dim=0)
        if config.absolute_action_dims:
            dims = list(config.absolute_action_dims)
            integrated[..., dims] = actions[..., dims]
        actions = integrated
    return _flip_gripper(actions, dims=config.gripper_flip_dims)


# ---------------------------------------------------------------- samplers
# Flow matching over a dict of streams: x_t = t * eps + (1 - t) * x0, velocity
# eps - x0; the solver state stays fp32.
Samples = dict[str, torch.Tensor]
Predictor = Callable[[Samples, float], Samples]
NUM_TRAIN_TIMESTEPS = 1000


def cosmos_unipc(
    samples: Samples,
    predict: Predictor,
    *,
    scheduler: FlowUniPCMultistepScheduler,
    n_steps: int,
    shift: float,
) -> Samples:
    """Solve with a private copy of ``scheduler`` per stream (UniPC keeps per-stream history)."""
    device = next(iter(samples.values())).device
    schedulers = {k: copy.deepcopy(scheduler) for k in samples}
    for stream_scheduler in schedulers.values():
        stream_scheduler.set_timesteps(n_steps, device=device, shift=shift)
        # Ticks can repeat at high step counts; index by step, not by tick.
        stream_scheduler.set_begin_index(0)
    for tick in next(iter(schedulers.values())).timesteps:
        # The reference feeds float32(tick) / 1000 to the model.
        t = torch.tensor(float(tick), dtype=torch.float32) / NUM_TRAIN_TIMESTEPS
        velocity = predict(samples, t.item())
        samples = {
            k: schedulers[k].step(velocity[k], tick, samples[k], return_dict=False)[0]
            for k in samples
        }
    return samples


# ---------------------------------------------------------------- stages
class Flux3ActionPreprocessStage(PipelineStage):
    """Observation dict -> canvas, state and prompt."""

    def __init__(self, config: Flux3ActionPipelineConfig):
        super().__init__()
        self.config = config

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        start = time.perf_counter()
        state = vla_state(batch)
        if batch.is_warmup:
            state["flux3_observation"] = _warmup_observation(self.config)
        else:
            observation = dict(state.get("observation") or {})
            observation.setdefault("prompt", batch.prompt)
            state["flux3_observation"] = parse_observation(observation, self.config)
        vla_timings(batch)["preprocess_ms"] = (time.perf_counter() - start) * 1000
        return batch


class Flux3ActionTextEncodingStage(PipelineStage):
    """Prompt (and CFG negative) -> DiT-encoded text contexts, cached per caption."""

    def __init__(
        self,
        config: Flux3ActionPipelineConfig,
        transformer: Flux3Transformer,
        text_encoder: Flux3TextEncoder,
    ):
        super().__init__()
        self.config = config
        self.transformer = transformer
        self.text_encoder = text_encoder
        self._contexts: OrderedDict[str, Flux3SegmentState] = OrderedDict()
        self._cached_tokens = 0

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        # Contexts are cached per caption: both components run only on a miss.
        return [
            ComponentUse(
                stage_name=stage_name,
                component_name=name,
                allow_prefetch=False,
                start_at_stage_entry=False,
            )
            for name in ("text_encoder", "transformer")
        ]

    def _encode(self, captions: list[str], device: torch.device) -> dict:
        with self.use_declared_component(
            component_name="text_encoder", module=self.text_encoder
        ):
            # One caption per forward: batching changes bf16 GEMM results.
            encoded = [self.text_encoder.encode([c])[0] for c in captions]
        contexts = {}
        with (
            self.use_declared_component(
                component_name="transformer", module=self.transformer
            ),
            set_forward_context(current_timestep=0, attn_metadata=None),
        ):
            for caption, ctx in zip(captions, encoded):
                contexts[caption] = self.transformer.encode_context(
                    ctx=ctx.to(device), ctx_ids=text_ids(ctx.shape[1]).to(device)
                )
        return contexts

    def _contexts_for(
        self, captions: list[str], device: torch.device, *, use_cache: bool
    ) -> tuple[list[Flux3SegmentState], bool]:
        """DiT-encoded text contexts of ``captions`` and whether all were cached."""
        cached = self._contexts if use_cache else {}
        todo = [c for c in dict.fromkeys(captions) if c not in cached]
        fresh = self._encode(todo, device) if todo else {}
        contexts = [fresh.get(c) or cached[c] for c in captions]
        if use_cache:
            self._remember(captions, fresh)
        return contexts, not todo

    def _remember(
        self, captions: list[str], fresh: dict[str, Flux3SegmentState]
    ) -> None:
        for caption, context in fresh.items():
            self._contexts[caption] = context
            self._cached_tokens += context.length
        for caption in captions:
            self._contexts.move_to_end(caption)
        while (
            self._cached_tokens > self.config.caption_cache_max_tokens
            and self._contexts
        ):
            _, evicted = self._contexts.popitem(last=False)
            self._cached_tokens -= evicted.length

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        start = time.perf_counter()
        state = vla_state(batch)
        observation: Flux3ActionObservation = state["flux3_observation"]
        options = vla_options(batch)
        guidance = self.config.resolve_guidance(
            options.get("guidance_scale"), options.get("guidance_scale_action")
        )
        captions = [observation.prompt]
        if any(g != 1.0 for g in guidance.values()):
            captions.append("")
        use_cache = bool(options.get("enable_prefix_cache", True))
        state["flux3_contexts"], hit = self._contexts_for(
            captions, get_local_torch_device(), use_cache=use_cache
        )
        state["flux3_guidance"] = guidance
        state["cache"] = {
            "enabled": use_cache,
            "hit": hit,
            "scope": "caption",
            "mode": "exact",
        }
        vla_timings(batch)["text_ms"] = (time.perf_counter() - start) * 1000
        return batch


class Flux3ActionObservationEncodingStage(PipelineStage):
    """Observed frame (VAE) and robot state -> DiT-encoded conditioning streams."""

    def __init__(
        self,
        config: Flux3ActionPipelineConfig,
        transformer: Flux3Transformer,
        vae: Flux3VideoVAE,
    ):
        super().__init__()
        self.config = config
        self.transformer = transformer
        self.vae = vae

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        return [
            ComponentUse(stage_name=stage_name, component_name="vae"),
            ComponentUse(
                stage_name=stage_name,
                component_name="transformer",
                start_at_stage_entry=False,
            ),
        ]

    def _state_tokens(self, state: torch.Tensor, device: torch.device):
        cfg = self.config
        flipped = _flip_gripper(state, dims=cfg.gripper_flip_dims)
        token = normalize(
            flipped, stats=cfg.state_normalization, clip=cfg.normalization_clip
        )
        values = (token[None, :, None] * cfg.action_scale).to(device)
        return pack_action(values, seconds=torch.zeros(1))

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        start = time.perf_counter()
        cfg = self.config
        device = get_local_torch_device()
        state = vla_state(batch)
        observation: Flux3ActionObservation = state["flux3_observation"]
        h, w = cfg.latent_hw
        with self.use_declared_component(component_name="vae", module=self.vae):
            frame = observation.canvas.to(device, torch.bfloat16)[None]
            latent = self.vae.encode_frame(frame)[..., :h, :w]
        video, video_ids = pack_video(latent, first_frame=0, fps=cfg.fps)
        action, action_ids = self._state_tokens(observation.state, device)
        zero = torch.zeros(1, device=device)
        with (
            self.use_declared_component(
                component_name="transformer", module=self.transformer
            ),
            set_forward_context(current_timestep=0, attn_metadata=None),
        ):
            state["flux3_conditioning"] = [
                self.transformer.encode_stream(
                    name="video_cond", x=video, ids=video_ids.to(device), timesteps=zero
                ),
                self.transformer.encode_stream(
                    name=f"{cfg.action_modality}_cond",
                    x=action,
                    ids=action_ids.to(device),
                    timesteps=zero,
                ),
            ]
        vla_timings(batch)["observation_ms"] = (time.perf_counter() - start) * 1000
        return batch


class Flux3ActionDenoisingStage(PipelineStage):
    """Joint video + action flow matching from noise; stores the action chunk."""

    def __init__(
        self,
        config: Flux3ActionPipelineConfig,
        transformer: Flux3Transformer,
        scheduler: FlowUniPCMultistepScheduler,
    ):
        super().__init__()
        self.config = config
        self.transformer = transformer
        self.scheduler = scheduler

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        return [
            ComponentUse(
                stage_name=self._component_stage_name(stage_name),
                component_name="transformer",
                phase="denoise",
                preferred_ready_after_request=True,
                memory_intensive=True,
            )
        ]

    def _noised_streams(self, seed: int):
        cfg = self.config
        h, w = cfg.latent_hw
        # Latent frames 1.. of the (chunk + 1)-frame window.
        n_pred = cfg.action_horizon // TEMPORAL_DOWNSAMPLE
        # Draw order (video, then action) on a CPU generator matches the reference.
        rng = torch.Generator().manual_seed(seed)
        video_noise = torch.randn(1, LATENT_CHANNELS, n_pred, h, w, generator=rng)
        action_noise = torch.randn(1, cfg.action_dim, cfg.action_horizon, generator=rng)
        video, video_ids = pack_video(video_noise, first_frame=1, fps=cfg.fps)
        seconds = (torch.arange(cfg.action_horizon).float() + 1) / cfg.fps
        action, action_ids = pack_action(action_noise, seconds=seconds)
        return {"video": video, cfg.action_modality: action}, {
            "video": video_ids,
            cfg.action_modality: action_ids,
        }

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        cfg = self.config
        device = get_local_torch_device()
        state = vla_state(batch)
        observation: Flux3ActionObservation = state["flux3_observation"]
        contexts: list[Flux3SegmentState] = state["flux3_contexts"]
        conditioning: list[Flux3SegmentState] = state["flux3_conditioning"]
        guidance: dict[str, float] = state["flux3_guidance"]
        steps = batch.num_inference_steps
        seed = batch.seed[0] if isinstance(batch.seed, list) else batch.seed
        horizon = batch.action_horizon or cfg.action_horizon
        if horizon > cfg.action_horizon:
            raise ValueError(
                f"action_horizon {horizon} exceeds the policy chunk {cfg.action_horizon}"
            )

        start = time.perf_counter()
        samples, ids = self._noised_streams(seed)
        samples = {k: v.to(device) for k, v in samples.items()}
        ropes = {k: self.transformer.rope(v.to(device)) for k, v in ids.items()}
        video_cond, action_cond = conditioning
        order = list(samples)  # joint sequence: video, video_cond, action, action_cond
        step = 0

        def predict(
            current: dict[str, torch.Tensor], t: float
        ) -> dict[str, torch.Tensor]:
            nonlocal step
            timestep = torch.full((1,), t, device=device, dtype=torch.float32)
            with set_forward_context(
                current_timestep=step, attn_metadata=None, forward_batch=batch
            ):
                targets = {
                    name: self.transformer.encode_stream(
                        name=name,
                        x=current[name].to(torch.bfloat16),
                        ids=None,
                        timesteps=timestep,
                        rope=ropes[name],
                    )
                    for name in order
                }
                streams = [targets["video"], video_cond, targets[order[1]], action_cond]
                preds = [
                    self.transformer.denoise(
                        context=ctx, streams=streams, targets=order
                    )
                    for ctx in contexts
                ]
            step += 1
            if len(preds) == 1:
                return {k: v.float() for k, v in preds[0].items()}
            cond, uncond = preds
            return {
                k: (uncond[k] + guidance[k] * (cond[k] - uncond[k])).float()
                for k in order
            }

        with self.use_declared_component(
            component_name="transformer", module=self.transformer
        ):
            result = cosmos_unipc(
                samples,
                predict,
                scheduler=self.scheduler,
                n_steps=steps,
                shift=cfg.sampler_shift,
            )
        targets = result[cfg.action_modality][0].float() / cfg.action_scale
        actions = targets_to_actions(
            targets, state=observation.state.to(device), config=cfg
        )
        state["actions"] = actions[None, :horizon]
        synchronize_vla_action_tensor(actions)
        vla_timings(batch)["denoise_ms"] = (time.perf_counter() - start) * 1000
        return batch
