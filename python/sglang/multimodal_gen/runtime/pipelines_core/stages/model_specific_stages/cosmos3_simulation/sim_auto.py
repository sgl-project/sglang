# SPDX-License-Identifier: Apache-2.0
"""Cosmos3-Nano-Sim-Auto rollout stage.

Runs after the Multiview-AV input, tokenization and latent stages and before
the multiview decoding stage. It encodes one caption document per camera,
then generates the clip chunk by chunk with the student's fixed-step SDE
schedule: each chunk's targets are denoised against the KV ring of committed
chunks through the replay block mask, and a clean refresh forward commits
the chunk's K/V to its ring slot. The denoised camera-major RGB latents (and
LiDAR latents) are left in ``batch.latents`` packed the way the latent stage
packed its targets.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import msgspec
import torch

from sglang.multimodal_gen.configs.models.dits.cosmos3_sim_auto import (
    Cosmos3SimAutoManifest,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3_multiview import (
    pack_state,
    unpack_state,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3_sim_auto import (
    Cosmos3SimAutoTransformer,
    KVPair,
    current_ulysses_group,
    local_head_group,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3_simulation.sim_auto_attention import (
    REPLAY_BACKENDS,
    ReplayMask,
    build_replay_mask,
    build_replay_plan,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3_simulation.sim_auto_layout import (
    PASS_CLEAN,
    PASS_NOISY,
    CaptionMetadata,
    Chunk,
    SensorGrid,
    StreamItem,
    TokenMetadata,
    build_caption_documents,
    build_chunk_items,
    build_chunk_schedule,
    caption_metadata,
    chunk_token_count,
    chunk_token_metadata,
    committed_metadata,
    lidar_num_sweeps,
    lidar_sensor_grid,
    rgb_sensor_grid,
    ring_slot_for_step,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_simulation.multiview import (
    EXTRA_CAMERAS,
    EXTRA_CAPTION_LENGTHS,
    EXTRA_FRAMES_PER_VIEW,
    EXTRA_KNOWN_VIEWS,
    EXTRA_LATENT_FRAMES_PER_VIEW,
    EXTRA_LIDAR_CONDITION_FRAMES,
    EXTRA_LOCAL_CONDITION_INDEXES,
    EXTRA_NUM_VIEWS,
    EXTRA_PACKED_SHAPES,
    EXTRA_TRANSFORMER_KWARGS,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Final latents in the reference's layout, for latent-level comparisons:
# ``vision_latent`` [1, C, V*T, h, w] camera-major, ``lidar_latent`` [1, C_l, S, h_l, w_l].
EXTRA_SIM_AUTO_LATENTS = "cosmos3_sim_auto_latents"

# The reference mixes seed, sample index and step into one SDE noise seed
# (omni_mot_causal_model._run_distilled_ar_sampler); matching it keeps parity
# runs comparable at the latent level.
SDE_SAMPLE_SEED_STRIDE = 1_000_003
SDE_STEP_SEED_STRIDE = 9_176

VelocityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def initial_noise(
    shape: Sequence[int], *, seed: int, sample_index: int, device: torch.device
) -> torch.Tensor:
    """FP32 standard normal noise from ``Generator(device).manual_seed(seed + sample_index)``."""
    generator = torch.Generator(device=device).manual_seed(
        int(seed) + int(sample_index)
    )
    return torch.randn(
        tuple(shape), device=device, dtype=torch.float32, generator=generator
    )


def sde_step_seed(seed: int, sample_index: int, step_index: int) -> int:
    """Seed of the noise reinjected after ``step_index`` (0-based); steps count from one."""
    return (
        int(seed)
        + int(sample_index) * SDE_SAMPLE_SEED_STRIDE
        + (int(step_index) + 1) * SDE_STEP_SEED_STRIDE
    )


def fixed_step_sample(
    velocity_fn: VelocityFn,
    noise: torch.Tensor,
    *,
    t_list: Sequence[float],
    num_train_timesteps: int,
    seed: int,
    sample_index: int,
    sample_type: str,
) -> torch.Tensor:
    """The student schedule on a flat ``[B, N]`` FP32 state, as the reference runs it.

    ``velocity_fn(x, timestep)`` receives the state and a ``[B, 1]`` FP32 timestep
    equal to ``sigma * num_train_timesteps``. Each step is Euler on ``x0``; the
    SDE variant re-noises to the next sigma with a per-step seeded draw.
    """
    if noise.ndim != 2:
        raise ValueError(
            f"Fixed-step sampling expects a flat [B, N] state, got {tuple(noise.shape)}."
        )
    if sample_type not in ("sde", "ode"):
        raise ValueError(
            f"Unsupported fixed-step sample_type {sample_type!r}; expected 'sde' or 'ode'."
        )
    sigmas = [float(value) for value in t_list]
    if sigmas[-1] != 0.0:
        sigmas.append(0.0)
    x = noise.float()
    device = x.device
    sigma_tensor = torch.tensor(sigmas, dtype=torch.float32, device=device)
    deltas = torch.tensor(
        [b - a for a, b in zip(sigmas[:-1], sigmas[1:])],
        dtype=torch.float32,
        device=device,
    )
    for step_index, (sigma_cur, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        timestep = torch.full(
            (x.shape[0], 1),
            sigma_cur * float(num_train_timesteps),
            dtype=torch.float32,
            device=device,
        )
        velocity = velocity_fn(x, timestep).float()
        if velocity.shape != x.shape:
            raise ValueError(
                f"velocity_fn returned {tuple(velocity.shape)} for state {tuple(x.shape)}."
            )
        x0 = x - sigma_tensor[step_index] * velocity
        if sigma_next == 0.0:
            x = x0
            continue
        if sample_type == "ode":
            x = x + deltas[step_index] * velocity
            continue
        generator = torch.Generator(device=device).manual_seed(
            sde_step_seed(seed, sample_index, step_index)
        )
        eps = torch.empty_like(x).normal_(generator=generator)
        x = (1.0 - sigma_tensor[step_index + 1]) * x0 + sigma_tensor[
            step_index + 1
        ] * eps
    return x


def camera_major_slice(
    latents: torch.Tensor,
    *,
    num_views: int,
    frames_per_view: int,
    frames: torch.Tensor,
) -> torch.Tensor:
    """Select ``frames`` of every view from camera-major ``[1, C, V*T, h, w]`` -> ``[1, C, V*n, h, w]``.

    ``frames`` is a long index on the latents' device: the per-step callers keep
    one, since building it from host ints is a blocking upload.
    """
    grid = latents.view(
        1, latents.shape[1], num_views, frames_per_view, *latents.shape[3:]
    )
    return grid.index_select(3, frames).reshape(
        1, latents.shape[1], num_views * frames.numel(), *latents.shape[3:]
    )


def camera_major_write(
    latents: torch.Tensor,
    values: torch.Tensor,
    *,
    num_views: int,
    frames_per_view: int,
    frames: torch.Tensor,
) -> None:
    """Write ``[1, C, V*n, h, w]`` values into ``frames`` (a device long index) of every view of a camera-major tensor."""
    grid = latents.view(
        1, latents.shape[1], num_views, frames_per_view, *latents.shape[3:]
    )
    grid.index_copy_(
        3,
        frames,
        values.view(1, values.shape[1], num_views, frames.numel(), *values.shape[3:]),
    )


class ReplayRing:
    """Per-layer K/V of committed chunks: ``cache_chunks`` slots of ``frame_unit`` tokens each.

    Slot ``s`` occupies ring tokens ``[s * frame_unit, (s + 1) * frame_unit)``;
    the first ``num_tokens`` of a populated slot hold committed K/V and the
    rest is zero padding no plan ever admits.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        cache_chunks: int,
        frame_unit: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if min(num_layers, cache_chunks, frame_unit, num_kv_heads, head_dim) < 1:
            raise ValueError("ReplayRing dimensions must be positive.")
        self.cache_chunks = cache_chunks
        self.frame_unit = frame_unit
        self.capacity = cache_chunks * frame_unit
        self.keys = [
            torch.zeros(
                1, self.capacity, num_kv_heads, head_dim, device=device, dtype=dtype
            )
            for _ in range(num_layers)
        ]
        self.values = [torch.zeros_like(key) for key in self.keys]
        self.slots: dict[int, TokenMetadata] = {}

    def history(self) -> list[tuple[int, TokenMetadata]]:
        """Populated slots as ``(ring base, committed metadata)`` in slot order."""
        return [
            (slot * self.frame_unit, self.slots[slot]) for slot in sorted(self.slots)
        ]

    def layer_kv(self) -> list[KVPair]:
        return list(zip(self.keys, self.values, strict=True))

    def signature(self) -> tuple[tuple[int, int, bool], ...]:
        """What the visibility plan depends on: slot, token count and whether it is the first chunk."""
        return tuple(
            (slot, meta.num_tokens, bool(int(meta.step.min()) == 0))
            for slot, meta in sorted(self.slots.items())
        )

    def publish(self, slot: int, kv: Sequence[KVPair], metadata: TokenMetadata) -> None:
        num_tokens = metadata.num_tokens
        if not 0 <= slot < self.cache_chunks or not 0 < num_tokens <= self.frame_unit:
            raise ValueError(
                f"Cannot publish {num_tokens} tokens into ring slot {slot} of {self.frame_unit} tokens."
            )
        if len(kv) != len(self.keys):
            raise ValueError(f"Expected {len(self.keys)} K/V layers, got {len(kv)}.")
        base = slot * self.frame_unit
        for (key, value), key_pool, value_pool in zip(
            kv, self.keys, self.values, strict=True
        ):
            if (
                tuple(key.shape) != (1, num_tokens, *key_pool.shape[2:])
                or key.shape != value.shape
            ):
                raise ValueError(
                    f"Ring publish expects [1, {num_tokens}, H_kv, D] K/V, got {tuple(key.shape)}."
                )
            key_pool[:, base : base + self.frame_unit].zero_()
            value_pool[:, base : base + self.frame_unit].zero_()
            key_pool[:, base : base + num_tokens].copy_(key)
            value_pool[:, base : base + num_tokens].copy_(value)
        self.slots[slot] = metadata


class _LidarInputs(msgspec.Struct, eq=False, kw_only=True):
    """LiDAR control latents (model dtype), the fp32 target buffer and the measured prefix sweeps."""

    control: torch.Tensor
    target: torch.Tensor
    condition_sweeps: tuple[int, ...]


class _Session(msgspec.Struct, eq=False, kw_only=True):
    """Everything one rollout keeps between chunks."""

    num_views: int
    joint: bool
    fps: float
    latent_frames: int
    schedule: tuple[Chunk, ...]
    rgb_grid: SensorGrid
    lidar_grid: SensorGrid | None
    text_extent: int
    seed: int
    rig_view_ids: torch.Tensor
    rgb_control: torch.Tensor
    rgb_target: torch.Tensor
    rgb_condition_frames: tuple[int, ...]
    lidar_control: torch.Tensor | None
    lidar_target: torch.Tensor | None
    lidar_condition_sweeps: tuple[int, ...]
    captions: CaptionMetadata
    und_kv: list[KVPair]
    ring: ReplayRing
    masks: dict[tuple[Any, ...], ReplayMask]
    device: torch.device


class _FlatLayout(msgspec.Struct, eq=False, kw_only=True):
    """How the chunk's generated frames pack into the sampler's flat ``[1, N]`` state.

    The reference flattens the noisy RGB frames camera-major ``[1, C, V*n, h, w]``
    followed by the noisy LiDAR sweeps ``[1, C_l, m, h_l, w_l]``.
    """

    num_views: int
    noisy_rgb: torch.Tensor
    noisy_sweeps: torch.Tensor
    # chunk-local device indexes, built once per chunk instead of every step
    local_rgb: torch.Tensor
    local_sweeps: torch.Tensor
    rgb_shape: tuple[int, ...]
    lidar_shape: tuple[int, ...]
    rgb_size: int
    lidar_size: int

    @classmethod
    def of(
        cls,
        items: Sequence[StreamItem],
        latents: Sequence[torch.Tensor],
        *,
        num_views: int,
        noisy_rgb: tuple[int, ...],
        noisy_sweeps: tuple[int, ...],
    ) -> _FlatLayout:
        rgb_item, rgb_base = items[1], latents[1]
        rgb_shape = (
            1,
            rgb_base.shape[1],
            num_views * len(noisy_rgb),
            *rgb_base.shape[3:],
        )
        local_sweeps: list[int] = []
        lidar_shape: tuple[int, ...] = (1, 0)
        if noisy_sweeps:
            lidar_item, lidar_base = items[3], latents[3]
            local_sweeps = [lidar_item.frames.index(sweep) for sweep in noisy_sweeps]
            lidar_shape = (
                1,
                lidar_base.shape[1],
                len(noisy_sweeps),
                *lidar_base.shape[3:],
            )
        device = rgb_base.device
        return cls(
            num_views=num_views,
            noisy_rgb=torch.tensor(noisy_rgb, dtype=torch.long, device=device),
            noisy_sweeps=torch.tensor(noisy_sweeps, dtype=torch.long, device=device),
            local_rgb=torch.tensor(
                [rgb_item.frames.index(frame) for frame in noisy_rgb],
                dtype=torch.long,
                device=device,
            ),
            local_sweeps=torch.tensor(local_sweeps, dtype=torch.long, device=device),
            rgb_shape=rgb_shape,
            lidar_shape=lidar_shape,
            rgb_size=math.prod(rgb_shape) if noisy_rgb else 0,
            lidar_size=math.prod(lidar_shape) if noisy_sweeps else 0,
        )

    @property
    def size(self) -> int:
        return self.rgb_size + self.lidar_size

    def scatter(
        self, latents: Sequence[torch.Tensor], x: torch.Tensor
    ) -> list[torch.Tensor]:
        """The chunk's item latents with the flat state written into the generated frames."""
        current = list(latents)
        if self.rgb_size:
            noisy = latents[1].clone()
            camera_major_write(
                noisy,
                x[:, : self.rgb_size].reshape(self.rgb_shape).to(noisy.dtype),
                num_views=self.num_views,
                frames_per_view=latents[1].shape[2] // self.num_views,
                frames=self.local_rgb,
            )
            current[1] = noisy
        if self.lidar_size:
            noisy_lidar = latents[3].clone()
            noisy_lidar.index_copy_(
                2,
                self.local_sweeps,
                x[:, self.rgb_size :].reshape(self.lidar_shape).to(noisy_lidar.dtype),
            )
            current[3] = noisy_lidar
        return current

    def gather(self, targets: Sequence[torch.Tensor]) -> torch.Tensor:
        """The generated frames of the target predictions as one flat FP32 state."""
        parts = []
        if self.rgb_size:
            parts.append(
                camera_major_slice(
                    targets[0],
                    num_views=self.num_views,
                    frames_per_view=targets[0].shape[2] // self.num_views,
                    frames=self.local_rgb,
                ).reshape(1, -1)
            )
        if self.lidar_size:
            parts.append(targets[1].index_select(2, self.local_sweeps).reshape(1, -1))
        return torch.cat(parts, dim=1).float()

    def write_back(self, session: _Session, denoised: torch.Tensor) -> None:
        if self.rgb_size:
            camera_major_write(
                session.rgb_target,
                denoised[:, : self.rgb_size].reshape(self.rgb_shape),
                num_views=self.num_views,
                frames_per_view=session.latent_frames,
                frames=self.noisy_rgb,
            )
        if self.lidar_size:
            session.lidar_target.index_copy_(
                2,
                self.noisy_sweeps,
                denoised[:, self.rgb_size :].reshape(self.lidar_shape),
            )


class Cosmos3SimAutoRolloutStage(PipelineStage):
    """Chunk-causal denoising with a KV ring of committed clean chunks."""

    parallelism_type = StageParallelismType.REPLICATED

    def __init__(
        self,
        *,
        transformer: Cosmos3SimAutoTransformer,
        manifest: Cosmos3SimAutoManifest,
        attention_backend: str,
    ) -> None:
        super().__init__()
        if not isinstance(transformer, Cosmos3SimAutoTransformer):
            raise TypeError(
                f"Cosmos3-Nano-Sim-Auto requires Cosmos3SimAutoTransformer, got {type(transformer).__name__}."
            )
        if attention_backend not in REPLAY_BACKENDS:
            raise ValueError(
                "Cosmos3-Nano-Sim-Auto attention backend must be one of "
                f"{list(REPLAY_BACKENDS)}, got {attention_backend!r}."
            )
        self.transformer = transformer
        self.manifest = manifest
        self.attention_backend = attention_backend
        # Ring and caption K/V hold the rank's head group; the rollout itself
        # runs replicated on every Ulysses rank.
        self.ulysses = current_ulysses_group()

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check(
            "control_latents", batch.extra.get("control_latents"), V.not_none
        )
        result.add_check(EXTRA_CAMERAS, batch.extra.get(EXTRA_CAMERAS), V.not_none)
        result.add_check("cond_text_ids", batch.extra.get("cond_text_ids"), V.is_tensor)
        return result

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        return [
            ComponentUse(
                self._component_stage_name(stage_name),
                "transformer",
                phase="denoise",
                preferred_ready_after_request=True,
                memory_intensive=True,
                start_at_stage_entry=False,
            )
        ]

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        with self.use_declared_component(
            component_name="transformer", module=self.transformer, phase="denoise"
        ):
            with torch.no_grad():
                self._rollout(batch)
        return batch

    # -- preparation -------------------------------------------------------------

    def _rollout(self, batch: Req) -> None:
        session = self._prepare(batch, get_local_torch_device())
        self.log_info(
            f"Sim-Auto rollout: {session.num_views} cameras x {session.latent_frames} latent frames in "
            f"{len(session.schedule)} chunks, {'joint RGB+LiDAR' if session.joint else 'RGB'}, "
            f"ring {session.ring.cache_chunks} x {session.ring.frame_unit} tokens, captions "
            f"{session.captions.num_tokens} tokens (text extent {session.text_extent}), seed {session.seed}, "
            f"attention {self.attention_backend}"
            + (
                f", ulysses {self.ulysses.world_size}"
                if self.ulysses is not None
                else ""
            )
        )
        for chunk in session.schedule:
            self._run_chunk(session, chunk)
        targets = [session.rgb_target]
        if session.lidar_target is not None:
            targets.append(session.lidar_target)
        batch.latents = pack_state(targets)
        batch.raw_latent_shape = tuple(batch.latents.shape)
        batch.extra[EXTRA_SIM_AUTO_LATENTS] = {
            "vision_latent": session.rgb_target,
            "lidar_latent": session.lidar_target,
        }

    def _prepare(self, batch: Req, device: torch.device) -> _Session:
        manifest = self.manifest
        extra = batch.extra
        if extra.get(EXTRA_KNOWN_VIEWS):
            raise ValueError(
                "Cosmos3-Nano-Sim-Auto does not support view completion; give every camera a WSM control."
            )
        local_indexes = [
            int(i) for i in (extra.get(EXTRA_LOCAL_CONDITION_INDEXES) or [])
        ]
        if local_indexes not in ([], [0]):
            raise ValueError(
                f"Cosmos3-Nano-Sim-Auto conditions on latent frame 0 only, got {local_indexes}."
            )
        seed = batch.seed
        if not isinstance(seed, int):
            raise ValueError(
                f"Cosmos3-Nano-Sim-Auto requires a single integer seed, got {seed!r}."
            )
        num_views = int(extra[EXTRA_NUM_VIEWS])
        frames_per_view = int(extra[EXTRA_FRAMES_PER_VIEW])
        fps = float(batch.fps)
        kwargs = extra[EXTRA_TRANSFORMER_KWARGS]
        rgb_control = extra["control_latents"][0].to(device)
        conditions = unpack_state(
            extra["condition_latents"], extra[EXTRA_PACKED_SHAPES]
        )
        lidar = self._prepare_lidar(
            kwargs["lidar_control_latents"],
            conditions[1] if len(conditions) > 1 else None,
            measured=extra.get(EXTRA_LIDAR_CONDITION_FRAMES),
            frames_per_view=frames_per_view,
            fps=fps,
            device=device,
        )
        rgb_grid, lidar_grid, schedule = self._geometry(
            batch,
            rgb_control=rgb_control,
            lidar=lidar,
            fps=fps,
            latent_frames=int(extra[EXTRA_LATENT_FRAMES_PER_VIEW]),
        )
        und_kv, captions, text_extent = self._encode_captions(
            extra["cond_text_ids"].to(device),
            tuple(int(length) for length in extra[EXTRA_CAPTION_LENGTHS]["cond"]),
            horizon_seconds=frames_per_view / fps,
        )
        return _Session(
            num_views=num_views,
            joint=lidar is not None,
            fps=fps,
            latent_frames=int(extra[EXTRA_LATENT_FRAMES_PER_VIEW]),
            schedule=schedule,
            rgb_grid=rgb_grid,
            lidar_grid=lidar_grid,
            text_extent=text_extent,
            seed=seed,
            rig_view_ids=kwargs["rig_view_ids"].to(device),
            rgb_control=rgb_control,
            rgb_target=conditions[0].to(device=device, dtype=torch.float32).clone(),
            rgb_condition_frames=(0,) if local_indexes else (),
            lidar_control=None if lidar is None else lidar.control,
            lidar_target=None if lidar is None else lidar.target,
            lidar_condition_sweeps=() if lidar is None else lidar.condition_sweeps,
            captions=captions,
            und_kv=und_kv,
            ring=self._ring(
                und_kv,
                schedule,
                num_views=num_views,
                rgb_grid=rgb_grid,
                lidar_grid=lidar_grid,
                fps=fps,
            ),
            masks={},
            device=device,
        )

    def _prepare_lidar(
        self,
        control: torch.Tensor | None,
        condition: torch.Tensor | None,
        *,
        measured: torch.Tensor | None,
        frames_per_view: int,
        fps: float,
        device: torch.device,
    ) -> _LidarInputs | None:
        if control is None:
            return None
        manifest = self.manifest
        if not math.isclose(fps, manifest.default_fps):
            raise ValueError(
                f"Joint requests need fps={manifest.default_fps:g}, got {fps:g}."
            )
        if condition is None:
            raise ValueError(
                "Joint requests need packed LiDAR condition latents from the latent stage."
            )
        target = condition.to(device=device, dtype=torch.float32).clone()
        sweeps = int(target.shape[2])
        expected = lidar_num_sweeps(
            frames_per_view, camera_fps=fps, lidar_fps=manifest.lidar_fps
        )
        if sweeps != expected:
            raise ValueError(
                f"LiDAR latents hold {sweeps} sweeps, the clip covers {expected}."
            )
        condition_sweeps: tuple[int, ...] = ()
        if measured is not None:
            if int(measured.shape[1]) != 1:
                raise ValueError(
                    "Cosmos3-Nano-Sim-Auto conditions on exactly one measured LiDAR sweep."
                )
            condition_sweeps = (0,)
        return _LidarInputs(
            control=control.to(device), target=target, condition_sweeps=condition_sweeps
        )

    def _geometry(
        self,
        batch: Req,
        *,
        rgb_control: torch.Tensor,
        lidar: _LidarInputs | None,
        fps: float,
        latent_frames: int,
    ) -> tuple[SensorGrid, SensorGrid | None, tuple[Chunk, ...]]:
        manifest = self.manifest
        rgb_grid = rgb_sensor_grid(
            height=int(batch.height),
            width=int(batch.width),
            vae_spatial_compression=manifest.vae_spatial_compression_factor,
            patch_size=manifest.latent_patch_size,
        )
        if tuple(rgb_control.shape[3:]) != (
            rgb_grid.latent_height,
            rgb_grid.latent_width,
        ):
            raise ValueError(
                f"WSM latents {tuple(rgb_control.shape)} do not match the {batch.width}x{batch.height} canvas."
            )
        lidar_grid = None
        if lidar is not None:
            lidar_grid = lidar_sensor_grid(
                latent_height=int(lidar.control.shape[3]),
                latent_width=int(lidar.control.shape[4]),
                patch_hw=self.transformer.lidar_patch_hw,
            )
        schedule = build_chunk_schedule(
            rgb_latents=latent_frames,
            lidar_sweeps=None if lidar is None else int(lidar.target.shape[2]),
            frames_per_chunk=manifest.frames_per_chunk,
            rgb_seconds_per_frame=manifest.rgb_seconds_per_frame(fps),
            lidar_seconds_per_frame=manifest.lidar_seconds_per_frame
            if lidar is not None
            else None,
        )
        return rgb_grid, lidar_grid, schedule

    def _ring(
        self,
        und_kv: list[KVPair],
        schedule: tuple[Chunk, ...],
        *,
        num_views: int,
        rgb_grid: SensorGrid,
        lidar_grid: SensorGrid | None,
        fps: float,
    ) -> ReplayRing:
        """One slot per retained chunk, sized for the largest chunk of this request."""
        manifest = self.manifest
        frame_unit = max(
            chunk_token_count(
                build_chunk_items(
                    chunk,
                    num_views=num_views,
                    rgb_grid=rgb_grid,
                    rgb_seconds_per_frame=manifest.rgb_seconds_per_frame(fps),
                    lidar_grid=lidar_grid,
                    lidar_seconds_per_frame=manifest.lidar_seconds_per_frame
                    if lidar_grid is not None
                    else None,
                )
            )
            for chunk in schedule
        )
        return ReplayRing(
            num_layers=self.transformer.num_hidden_layers,
            cache_chunks=manifest.cache_chunks,
            frame_unit=frame_unit,
            num_kv_heads=und_kv[0][0].shape[2],
            head_dim=und_kv[0][0].shape[3],
            device=und_kv[0][0].device,
            dtype=und_kv[0][0].dtype,
        )

    def _encode_captions(
        self,
        text_ids: torch.Tensor,
        lengths: tuple[int, ...],
        *,
        horizon_seconds: float,
    ) -> tuple[list[KVPair], CaptionMetadata, int]:
        """Per-camera caption K/V laid out ``[sink_v | caption_v]`` per view; the sink is the caption's prefix.

        Captions are encoded independently with positions rewound to zero, so
        the first ``text_sink_tokens`` K/V of a caption are exactly those of a
        separately encoded sink document.
        """
        if sum(lengths) != text_ids.shape[1] or any(length < 1 for length in lengths):
            raise ValueError(
                f"Caption lengths {lengths} must partition the {text_ids.shape[1]} prompt tokens."
            )
        sink_tokens = min(self.manifest.text_sink_tokens, min(lengths))
        per_view = [
            self.transformer.encode_caption(ids)
            for ids in text_ids.split(list(lengths), dim=1)
        ]
        und_kv: list[KVPair] = []
        for layer in range(self.transformer.num_hidden_layers):
            keys, values = [], []
            for layers in per_view:
                k, v = layers[layer]
                if sink_tokens:
                    keys.append(k[:, :sink_tokens])
                    values.append(v[:, :sink_tokens])
                keys.append(k)
                values.append(v)
            und_kv.append((torch.cat(keys, dim=1), torch.cat(values, dim=1)))
        und_kv = local_head_group(und_kv, self.ulysses)
        documents = build_caption_documents(
            lengths=lengths, sink_tokens=sink_tokens, horizon_seconds=horizon_seconds
        )
        captions = caption_metadata(documents)
        if captions.num_tokens > self.manifest.text_cache_max_len:
            raise ValueError(
                f"Per-camera captions need {captions.num_tokens} tokens but the artifact's "
                f"text_cache_max_len is {self.manifest.text_cache_max_len}."
            )
        return und_kv, captions, max(lengths)

    # -- chunk execution ------------------------------------------------------------

    def _chunk_items(self, session: _Session, chunk: Chunk) -> tuple[StreamItem, ...]:
        return build_chunk_items(
            chunk,
            num_views=session.num_views,
            rgb_grid=session.rgb_grid,
            rgb_seconds_per_frame=self.manifest.rgb_seconds_per_frame(session.fps),
            lidar_grid=session.lidar_grid,
            lidar_seconds_per_frame=self.manifest.lidar_seconds_per_frame
            if session.joint
            else None,
            rgb_condition_frames=session.rgb_condition_frames,
            lidar_condition_sweeps=session.lidar_condition_sweeps,
        )

    def _item_latents(
        self, session: _Session, items: Sequence[StreamItem]
    ) -> list[torch.Tensor]:
        latents = []
        for item in items:
            if item.is_lidar:
                source = (
                    session.lidar_control if item.is_control else session.lidar_target
                )
                latents.append(source[:, :, list(item.frames)])
            else:
                source = session.rgb_control if item.is_control else session.rgb_target
                latents.append(
                    camera_major_slice(
                        source,
                        num_views=session.num_views,
                        frames_per_view=session.latent_frames,
                        frames=torch.tensor(
                            item.frames, dtype=torch.long, device=source.device
                        ),
                    )
                )
        return latents

    def _block_mask(
        self,
        session: _Session,
        chunk: Chunk,
        items: Sequence[StreamItem],
        metadata: TokenMetadata,
        pass_kind: str,
    ) -> ReplayMask:
        # Plans depend on the chunk shape, on which ring slots are populated and
        # on whether chunk 0 is still inside the cross-view window (chunk 1
        # only); absolute time otherwise shifts every stream together.
        key = (
            pass_kind,
            min(chunk.step, 2),
            tuple(
                (item.kind, len(item.frames), item.condition_frames) for item in items
            ),
            session.ring.signature(),
        )
        mask = session.masks.get(key)
        if mask is None:
            plan = build_replay_plan(
                current=metadata,
                history=session.ring.history(),
                history_tokens=session.ring.capacity,
                captions=session.captions,
                window_seconds=self.manifest.cross_view_past_window_seconds,
            )
            mask = build_replay_mask(
                plan, device=session.device, backend=self.attention_backend
            )
            session.masks[key] = mask
        return mask

    def _forward(
        self,
        session: _Session,
        *,
        items: Sequence[StreamItem],
        latents: Sequence[torch.Tensor],
        timestep: torch.Tensor | None,
        mask: ReplayMask,
        step_index: int,
    ):
        with set_forward_context(current_timestep=step_index, attn_metadata=None):
            return self.transformer(
                items=items,
                latents=latents,
                timestep=timestep,
                text_extent=session.text_extent,
                und_kv=session.und_kv,
                history_kv=session.ring.layer_kv(),
                mask=mask,
                rig_view_ids=session.rig_view_ids,
            )

    def _run_chunk(self, session: _Session, chunk: Chunk) -> None:
        items = self._chunk_items(session, chunk)
        rgb_target = items[1]
        lidar_target = items[3] if len(items) == 4 else None
        noisy_rgb = rgb_target.generated_frames
        noisy_sweeps = lidar_target.generated_frames if lidar_target is not None else ()
        if noisy_rgb or noisy_sweeps:
            self._denoise(
                session, chunk, items, noisy_rgb=noisy_rgb, noisy_sweeps=noisy_sweeps
            )
        if chunk.step != session.schedule[-1].step:
            self._commit(session, chunk, items)
        self.log_info(
            f"Chunk {chunk.step + 1}/{len(session.schedule)}: RGB latents {list(chunk.rgb_frames)}"
            f"{', LiDAR sweeps %s' % list(chunk.lidar_sweeps) if chunk.lidar_sweeps else ''}, "
            f"{len(noisy_rgb) * session.num_views + len(noisy_sweeps)} generated frames, "
            f"ring slots {sorted(session.ring.slots)}"
        )

    def _denoise(
        self,
        session: _Session,
        chunk: Chunk,
        items: Sequence[StreamItem],
        *,
        noisy_rgb: tuple[int, ...],
        noisy_sweeps: tuple[int, ...],
    ) -> None:
        """Fixed-step SDE over the chunk's generated frames; history K/V is read but never written."""
        manifest = self.manifest
        latents = self._item_latents(session, items)
        layout = _FlatLayout.of(
            items,
            latents,
            num_views=session.num_views,
            noisy_rgb=noisy_rgb,
            noisy_sweeps=noisy_sweeps,
        )
        metadata = chunk_token_metadata(items, step=chunk.step, pass_kind=PASS_NOISY)
        mask = self._block_mask(session, chunk, items, metadata, PASS_NOISY)
        # The reference seeds RGB rollouts by the chunk's first latent and joint rollouts by the step.
        sample_index = chunk.step if session.joint else chunk.rgb_frames[0]
        step_counter = [0]

        def velocity(x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            output = self._forward(
                session,
                items=items,
                latents=layout.scatter(latents, x),
                timestep=timestep.reshape(1),
                mask=mask,
                step_index=step_counter[0],
            )
            step_counter[0] += 1
            return layout.gather(output.targets)

        noise = initial_noise(
            (1, layout.size),
            seed=session.seed,
            sample_index=sample_index,
            device=session.device,
        )
        denoised = fixed_step_sample(
            velocity,
            noise,
            t_list=manifest.t_list,
            num_train_timesteps=manifest.num_train_timesteps,
            seed=session.seed,
            sample_index=sample_index,
            sample_type=manifest.sample_type,
        )
        if not torch.isfinite(denoised).all():
            raise FloatingPointError(
                f"Cosmos3-Nano-Sim-Auto produced non-finite latents in chunk {chunk.step}."
            )
        layout.write_back(session, denoised)

    def _commit(
        self, session: _Session, chunk: Chunk, items: Sequence[StreamItem]
    ) -> None:
        """Clean refresh forward (no time embed) whose K/V becomes the chunk's ring slot."""
        metadata = chunk_token_metadata(items, step=chunk.step, pass_kind=PASS_CLEAN)
        mask = self._block_mask(session, chunk, items, metadata, PASS_CLEAN)
        output = self._forward(
            session,
            items=items,
            latents=self._item_latents(session, items),
            timestep=None,
            mask=mask,
            step_index=0,
        )
        slot = ring_slot_for_step(
            chunk.step,
            cache_chunks=self.manifest.cache_chunks,
            sink_chunks=self.manifest.sink_chunks,
        )
        session.ring.publish(slot, output.current_kv, committed_metadata(metadata))
