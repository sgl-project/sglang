# SPDX-License-Identifier: Apache-2.0
"""Praxis-1 stage: observation -> robot commands, computed by the native policy."""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any

import numpy as np

from sglang.multimodal_gen.configs.pipeline_configs.praxis import (
    RIGS,
    PraxisPipelineConfig,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.vla import (
    vla_options,
    vla_state,
    vla_timings,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.vla import praxis_bimanual
from sglang.multimodal_gen.runtime.vla.praxis_policy import PraxisEpisode, PraxisPolicy
from sglang.multimodal_gen.runtime.vla.praxis_sampler import SamplerSettings

# Request and response conversion of each embodiment.
_POSE8_DIM = 8
_PASSTHROUGH_KEYS = ("descriptor", "robot_state.q", "reset_context")
MAX_SESSIONS = 32


def to_uint8(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.uint8:
        # Float images are in [0, 1].
        array = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    return array


def _single_arm_state(*, rows: np.ndarray) -> dict[str, np.ndarray]:
    rows = rows.reshape(-1, _POSE8_DIM)
    return {"robot_state.ee_pose": rows[:, :7], "grip": rows[:, 7]}


def _bimanual_state(*, rows: np.ndarray) -> dict[str, np.ndarray]:
    rows = rows.reshape(-1, 2 * _POSE8_DIM)
    state: dict[str, np.ndarray] = {}
    for index, side in enumerate(("left", "right")):
        arm = rows[:, index * _POSE8_DIM : (index + 1) * _POSE8_DIM]
        state[f"robot_state.{side}.ee_pose"] = arm[:, :7]
        state[f"grip.{side}"] = arm[:, 7]
    return state


def reference_observation(
    *, observation: dict[str, Any], embodiment: str = "single_arm"
) -> dict[str, Any]:
    """Request observation -> the observation dict of the reference ``Session.step``."""
    images = observation.get("images") or {}
    reference: dict[str, Any] = {}
    for view in RIGS[embodiment].image_keys:
        value = images.get(view)
        if value is None:
            value = observation.get(f"image.{view}")
        if value is not None:
            reference[f"image.{view}"] = to_uint8(value)
    state = observation.get("state")
    if state is not None:
        rows = np.asarray(state, dtype=np.float32)
        convert = _bimanual_state if embodiment == "bimanual" else _single_arm_state
        reference.update(convert(rows=rows))
    else:
        for key in RIGS[embodiment].state_keys:
            if key in observation:
                reference[key] = np.asarray(observation[key], dtype=np.float32)
    prompt = observation.get("prompt")
    if prompt:
        reference["task"] = prompt
    for key in _PASSTHROUGH_KEYS:
        if key in observation:
            reference[key] = observation[key]
    return reference


def commands_to_actions(
    *, commands: list[dict[str, Any]], embodiment: str = "single_arm"
) -> np.ndarray:
    """Reference commands -> rows of ``[xyz(3), rotation matrix row-major(9), grip(1)]`` per arm."""
    if embodiment == "bimanual":
        return np.asarray(
            [
                [
                    value
                    for side in ("left", "right")
                    for value in (
                        *c[f"robot_command.{side}"]["pose"],
                        c[f"target_grip.{side}"],
                    )
                ]
                for c in commands
            ],
            dtype=np.float32,
        )
    return np.asarray(
        [[*c["robot_command"]["pose"], c["target_grip"]] for c in commands],
        dtype=np.float32,
    )


def plain(value: Any) -> Any:
    """Numpy values inside the commands -> Python values, so the response encodes as JSON."""
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def response_extras(
    *, commands: list[dict[str, Any]], frames: np.ndarray | None
) -> dict[str, Any]:
    """Model-specific output of a call; the predicted video is present only when it was decoded."""
    extras: dict[str, Any] = {"commands": plain(commands)}
    if frames is not None:
        extras["video"] = frames
    return extras


class SessionTable:
    """Reference sessions by id; the least recently used one is dropped beyond the limit."""

    def __init__(self, *, max_sessions: int):
        self._max_sessions = max_sessions
        self._sessions: OrderedDict[str, Any] = OrderedDict()

    def get(self, *, session_id: str) -> Any | None:
        session = self._sessions.get(session_id)
        if session is not None:
            self._sessions.move_to_end(session_id)
        return session

    def put(self, *, session_id: str, session: Any) -> None:
        self._sessions[session_id] = session
        self._sessions.move_to_end(session_id)
        while len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)


# Size of the views of a robot client for the antialiased checkpoint: the fit size of the tiles.
_WARMUP_VIEW_HW = (234, 416)
# The first warm-up call has no context (depth 0); the second one carries the executed frames (depth 1).
_WARMUP_STACKS = (1, 25)


def synthetic_observation(*, step: int, stack: int) -> dict[str, Any]:
    """A synthetic single-arm observation: ``stack`` frames per view, shaped like a robot client request."""
    height, width = _WARMUP_VIEW_HW
    t = step * 24 + np.arange(stack)
    frames = np.zeros((stack, height, width, 3), np.uint8)
    frames[..., 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, None, :]
    frames[..., 1] = np.linspace(0, 255, height, dtype=np.uint8)[None, :, None]
    frames[..., 2] = (t * 37 % 255)[:, None, None]
    pose = np.zeros((stack, 7), np.float32)
    pose[:, 0], pose[:, 2], pose[:, 3] = 0.4 + 0.0005 * t, 0.4, 1.0
    return {
        "image.exterior": frames,
        "image.wrist": frames.copy(),
        "robot_state.ee_pose": pose,
        "grip": np.zeros(stack, np.float32),
        "task": "warmup",
    }


class PraxisPolicyStage(PipelineStage):
    def __init__(self, *, config: PraxisPipelineConfig, policy: PraxisPolicy):
        super().__init__()
        self.config = config
        self._policy = policy
        self._sessions = SessionTable(max_sessions=MAX_SESSIONS)

    def _sampler(self, *, batch: Req) -> SamplerSettings:
        options = vla_options(batch)
        video = options.get("guidance_scale")
        action = options.get("guidance_scale_action")
        config = self.config
        return SamplerSettings(
            num_steps=int(
                batch.num_inference_steps or config.default_num_inference_steps
            ),
            t_shift=config.sampler_shift,
            cfg_scale=config.guidance_scale_action if action is None else float(action),
            cfg_video_scale=config.guidance_scale if video is None else float(video),
        )

    def _new_episode(self, *, batch: Req) -> PraxisEpisode:
        seed = batch.seed[0] if isinstance(batch.seed, list) else batch.seed
        return self._policy.episode(
            seed=int(seed),
            max_context_chunks=self.config.max_context_chunks,
            exec_horizon=self.config.action_horizon,
        )

    def _episode(self, *, batch: Req, observation: dict[str, Any]) -> PraxisEpisode:
        session_id = observation.get("session_id")
        episode = (
            None
            if session_id is None
            else self._sessions.get(session_id=str(session_id))
        )
        if episode is None:
            episode = self._new_episode(batch=batch)
            if session_id is not None:
                self._sessions.put(session_id=str(session_id), session=episode)
        return episode

    def _warmup(self, *, batch: Req, settings: SamplerSettings):
        """Run both context depths once, so the kernels of each are compiled before the first request."""
        episode = self._new_episode(batch=batch)
        make_observation = (
            praxis_bimanual.warmup_obs
            if self.config.embodiment == "bimanual"
            else synthetic_observation
        )
        for step, stack in enumerate(_WARMUP_STACKS):
            result = episode.step(make_observation(step=step, stack=stack), settings)
        return result

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        start = time.perf_counter()
        state = vla_state(batch)
        observation = dict(state.get("observation") or {})
        observation.setdefault("prompt", batch.prompt)
        settings = self._sampler(batch=batch)
        frames = None
        timings = vla_timings(batch)
        if batch.is_warmup:
            result = self._warmup(batch=batch, settings=settings)
        else:
            episode = self._episode(batch=batch, observation=observation)
            result = episode.step(
                reference_observation(
                    observation=observation, embodiment=self.config.embodiment
                ),
                settings,
            )
            timings["step_ms"] = (time.perf_counter() - start) * 1000
            if observation.get("return_video"):
                decode_start = time.perf_counter()
                frames = episode.decode_video(result)
                frames = frames.permute(0, 2, 3, 1).contiguous().cpu().numpy()
                timings["decode_ms"] = (time.perf_counter() - decode_start) * 1000
        # The shared postprocess expects a leading batch dimension.
        state["actions"] = commands_to_actions(
            commands=result.commands, embodiment=self.config.embodiment
        )[None]
        state["extras"] = response_extras(commands=result.commands, frames=frames)
        timings["policy_ms"] = (time.perf_counter() - start) * 1000
        return batch
