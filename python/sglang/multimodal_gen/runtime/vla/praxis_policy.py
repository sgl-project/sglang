# Copied and adapted from: https://github.com/runwayml/praxis-1-oss

# SPDX-License-Identifier: Apache-2.0
"""Praxis-1 policy: the transformer, the autoencoder and the text encoder behind one episode of calls.

One call of :meth:`PraxisEpisode.step` turns an observation into the robot commands of one chunk: it builds
the model inputs (canvas, latents of the executed context, proprio, prompt), samples the window with the
transformer and decodes the actions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import msgspec
import numpy as np
import torch

from sglang.multimodal_gen.runtime.vla import praxis_bimanual
from sglang.multimodal_gen.runtime.vla.praxis_bimanual import (
    BimanualChunkInputs,
    BimanualSession,
)
from sglang.multimodal_gen.runtime.vla.praxis_geometry import NormStats
from sglang.multimodal_gen.runtime.vla.praxis_rig import (
    ChunkInputs,
    PraxisSession,
    q99_dims,
)
from sglang.multimodal_gen.runtime.vla.praxis_sampler import (
    ACTION_SALT,
    VIDEO_SALT,
    CfgParallel,
    SamplerSettings,
    sample_window_joint,
    seeded_normal,
)

CONTROL_MODE = "droid_impedance"


class PolicyLayout(msgspec.Struct, frozen=True, eq=False):
    """What the policy reads from the checkpoint config."""

    embodiment: str
    stats: NormStats
    dims: np.ndarray
    fps: float
    num_chunks: int
    chunk_latents: int
    action_dim: int
    tile_size: tuple[int, int]
    aspect_tol: float
    antialias: bool
    blank_caption: str

    @classmethod
    def from_model_config(cls, config: dict[str, Any]) -> PolicyLayout:
        video = config["video"]
        if config["embodiment"] == "bimanual":
            praxis_bimanual.canvas_size(config)
            if not video["antialias"]:
                raise ValueError("The bimanual rig needs an antialiased canvas")
        return cls(
            embodiment=str(config["embodiment"]),
            stats=NormStats.from_config(config["normalization"]),
            dims=q99_dims(config),
            fps=float(config["chunking"]["grid_hz"]),
            num_chunks=int(config["chunking"]["n_chunks"]),
            chunk_latents=int(config["chunking"]["latents_per_chunk"]),
            action_dim=int(config["action_expert"]["action_patch_dim"]),
            tile_size=(int(video["tile_height"]), int(video["tile_width"])),
            aspect_tol=float(video["aspect_tol"]),
            antialias=bool(video["antialias"]),
            blank_caption=str(config["text_encoder"]["blank_caption"]),
        )


class StepResult(msgspec.Struct, frozen=True, eq=False):
    """Outputs of one call; ``rows`` are the normalized action rows of the sampled chunk."""

    commands: list[dict]
    rows: np.ndarray
    video: torch.Tensor
    inputs: ChunkInputs | BimanualChunkInputs


def _to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Copy a small CPU tensor without a host synchronize."""
    return tensor.pin_memory().to(device, non_blocking=True)


class PraxisPolicy:
    def __init__(
        self,
        *,
        transformer: torch.nn.Module,
        vae: Any,
        text_encoder: Any,
        layout: PolicyLayout,
        cfg_parallel: CfgParallel | None = None,
    ) -> None:
        self.cfg_parallel = cfg_parallel
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.layout = layout
        self.device = vae.device

    def episode(
        self, *, seed: int, max_context_chunks: int, exec_horizon: int
    ) -> PraxisEpisode:
        return PraxisEpisode(
            policy=self,
            seed=seed,
            max_context_chunks=max_context_chunks,
            exec_horizon=exec_horizon,
        )

    def sample_chunk(
        self,
        *,
        text: torch.Tensor,
        text_uncond: torch.Tensor | None,
        inputs: ChunkInputs | BimanualChunkInputs,
        seed: int,
        settings: SamplerSettings,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Denoise the window of the chunk at depth ``len(cleans)``; return the video latent and actions of its last chunk."""
        layout, device = self.layout, self.device
        depth = len(inputs.cleans)
        window = depth + 1
        if window > layout.num_chunks:
            raise ValueError(
                f"depth {depth} is out of range for {layout.num_chunks} chunks"
            )
        proprio = inputs.proprio_all.to(device, torch.float32)
        if proprio.shape[:2] != (1, window):
            raise ValueError(f"proprio_all {tuple(proprio.shape)} for {window} chunks")
        dtype = torch.bfloat16
        anchor = inputs.anchor.to(device, dtype)
        clean = (
            torch.cat([c.to(device, dtype) for c in inputs.cleans], dim=2)
            if depth
            else None
        )
        frames = window * layout.chunk_latents
        _, channels, _, height, width = anchor.shape
        noise_video = _to_device(
            seeded_normal(
                shape=(1, channels, frames, height, width), seed=seed, salt=VIDEO_SALT
            ),
            device,
        )
        noise_action = _to_device(
            seeded_normal(
                shape=(frames, layout.action_dim), seed=seed, salt=ACTION_SALT
            ),
            device,
        )
        video, actions = sample_window_joint(
            velocity=self._velocity_fn(
                anchor=anchor, clean=clean, proprio=proprio, window=window
            ),
            noise_video=noise_video,
            noise_action=noise_action,
            settings=settings,
            text=text.to(device, dtype),
            text_uncond=None if text_uncond is None else text_uncond.to(device, dtype),
            parallel=self.cfg_parallel,
        )
        start = depth * layout.chunk_latents
        return video[:, :, start:].contiguous(), actions[start:].contiguous()[None]

    def _velocity_fn(
        self,
        *,
        anchor: torch.Tensor,
        clean: torch.Tensor | None,
        proprio: torch.Tensor,
        window: int,
    ) -> Callable[..., tuple[torch.Tensor | None, torch.Tensor | None]]:
        device, dtype = self.device, torch.bfloat16

        def velocity(text, x_video, x_action, t_cur, decode, act=True):
            video, action = self.transformer(
                x_video.to(dtype),
                text,
                torch.full((1, window), t_cur, dtype=dtype, device=device),
                anchor=anchor,
                clean=clean,
                noised_actions=x_action[None],
                action_timestep=torch.full((1, window), t_cur, device=device),
                proprio=proprio,
                decode=decode,
                act=act,
            )
            return (
                None if video is None else video.float(),
                None if action is None else action.float()[0],
            )

        return velocity


class PraxisEpisode:
    """The context state of one robot episode."""

    def __init__(
        self,
        *,
        policy: PraxisPolicy,
        seed: int,
        max_context_chunks: int,
        exec_horizon: int,
    ) -> None:
        layout = policy.layout
        self._policy = policy
        self.seed = int(seed)
        shared = dict(
            encode=policy.vae.encode,
            stats=layout.stats,
            device=policy.device,
            max_context_chunks=max_context_chunks,
            fps=layout.fps,
            dims=layout.dims,
            exec_horizon=exec_horizon,
            n_chunks=layout.num_chunks,
            tile_size=layout.tile_size,
        )
        self.session = (
            BimanualSession(**shared)
            if layout.embodiment == "bimanual"
            else PraxisSession(
                **shared,
                control_mode=CONTROL_MODE,
                aspect_tol=layout.aspect_tol,
                antialias=layout.antialias,
            )
        )

    @torch.no_grad()
    def step(self, obs: dict[str, Any], settings: SamplerSettings) -> StepResult:
        """One policy call on ``obs``: the commands of the chunk and its predicted video latent."""
        policy, session = self._policy, self.session
        inputs = session.prepare(obs)
        encoder = policy.text_encoder
        text = encoder.encode(inputs.prompt)
        text_uncond = (
            encoder.encode(policy.layout.blank_caption) if settings.guided else None
        )
        video, actions = policy.sample_chunk(
            text=text,
            text_uncond=text_uncond,
            inputs=inputs,
            seed=self.seed + inputs.chunk_index,
            settings=settings,
        )
        commands = session.finish(inputs, actions)
        rows, _ = session.decode_actions(inputs, actions)
        return StepResult(commands=commands, rows=rows, video=video, inputs=inputs)

    @torch.no_grad()
    def decode_video(self, result: StepResult) -> torch.Tensor:
        """uint8 frames ``(len(commands), 3, H, W)`` of the predicted video, one per command row.

        The decoder is causal over latents. With the clean context latents in front, the chunk gives
        one frame per command row; without context its first latent decodes to one frame.
        """
        cleans = result.inputs.cleans
        latent = result.video
        if cleans:
            latent = torch.cat(
                [*cleans, latent.to(cleans[0].device, cleans[0].dtype)], dim=2
            )
        frames = self._policy.vae.decode(latent)
        return frames[-len(result.rows) :][: len(result.commands)]
