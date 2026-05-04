# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from PIL import Image

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.srt.ug.context import UGContextBundle
from sglang.srt.ug.denoiser import UGLatentFlowMiddleBridge, UGMiddleBridge
from sglang.srt.ug.interleaved import UGGSegmentResult
from sglang.srt.ug.sampling import build_bagel_denoise_schedule


class BAGELLatentFlowGSegmentExecutor:
    """BAGEL G mechanics for the UG middle protocol.

    The generic UG stages only pass the SRT-owned U context into this executor.
    Latent preparation, schedule stepping, velocity calls, and image decoding
    stay here on the BAGEL/diffusion side of the boundary.
    """

    def __call__(
        self,
        *,
        bridge: UGMiddleBridge,
        contexts: UGContextBundle,
        batch: Req,
        server_args: ServerArgs,
    ) -> UGGSegmentResult:
        if getattr(bridge, "g_kind", None) != "latent_flow":
            raise ValueError(
                "BAGEL latent-flow executor requires g_kind='latent_flow', got "
                f"{getattr(bridge, 'g_kind', None)!r}"
            )
        latent_bridge = cast(UGLatentFlowMiddleBridge, bridge)
        self._prepare_latents(
            bridge=latent_bridge,
            contexts=contexts,
            batch=batch,
            server_args=server_args,
        )
        self._denoise(bridge=latent_bridge, contexts=contexts, batch=batch)
        image = self._decode_image(
            bridge=latent_bridge,
            contexts=contexts,
            batch=batch,
        )
        metadata = {"g_kind": "latent_flow"}
        if "ug_latent_shape" in batch.extra:
            metadata["latent_shape"] = batch.extra["ug_latent_shape"]
        return UGGSegmentResult(type="image", image=image, metadata=metadata)

    @staticmethod
    def _prepare_latents(
        *,
        bridge: UGLatentFlowMiddleBridge,
        contexts: UGContextBundle,
        batch: Req,
        server_args: ServerArgs,
    ) -> None:
        cfg = server_args.pipeline_config
        prepared = bridge.prepare_g_latents(
            contexts=contexts,
            sampling_params=batch.sampling_params,
            seed=batch.seed,
        )
        if prepared is not None:
            batch.latents = prepared.latent_tokens
            batch.extra["ug_latent_position_ids"] = prepared.latent_position_ids
            batch.extra["ug_latent_shape"] = prepared.latent_shape
            return

        height = int(batch.height)
        width = int(batch.width)
        latent_height = height // cfg.latent_downsample
        latent_width = width // cfg.latent_downsample
        if latent_height <= 0 or latent_width <= 0:
            raise ValueError(
                f"UG latent shape is empty for height={height}, width={width}, "
                f"latent_downsample={cfg.latent_downsample}"
            )

        num_tokens = latent_height * latent_width
        latent_dim = cfg.latent_channel * cfg.latent_patch_size * cfg.latent_patch_size
        generator = torch.Generator(device="cpu").manual_seed(int(batch.seed))
        batch.latents = torch.randn(
            1,
            num_tokens,
            latent_dim,
            generator=generator,
            dtype=torch.float32,
        )
        batch.extra["ug_latent_position_ids"] = torch.arange(num_tokens)
        batch.extra["ug_latent_shape"] = (latent_height, latent_width, latent_dim)

    @staticmethod
    def _denoise(
        *,
        bridge: UGLatentFlowMiddleBridge,
        contexts: UGContextBundle,
        batch: Req,
    ) -> None:
        params = batch.sampling_params
        x_t = batch.latents
        if x_t is None:
            raise ValueError("UG G segment requires latents from latent preparation")
        num_steps = int(params.num_inference_steps)
        if num_steps <= 0:
            raise ValueError(f"num_inference_steps must be positive, got {num_steps}")

        schedule = build_bagel_denoise_schedule(
            num_inference_steps=num_steps,
            timestep_shift=params.timestep_shift,
            device=x_t.device,
        )
        trajectory_latents = []
        trajectory_timesteps = []

        for i, timestep in enumerate(schedule.timesteps):
            trajectory_latents.append(x_t)
            trajectory_timesteps.append(timestep)
            velocity = bridge.predict_g_velocity(
                contexts=contexts,
                latent_tokens=x_t,
                timestep=timestep.reshape(1),
                latent_position_ids=batch.extra["ug_latent_position_ids"],
                sampling_params=params,
            )
            x_t = x_t - velocity.to(x_t) * schedule.dts[i].to(x_t)

        batch.latents = x_t
        if batch.return_trajectory_latents:
            if trajectory_latents:
                batch.trajectory_latents = torch.stack(trajectory_latents)
                batch.trajectory_timesteps = torch.stack(trajectory_timesteps)
            else:
                batch.trajectory_latents = x_t[:0]
                batch.trajectory_timesteps = schedule.timesteps[:0]

    @staticmethod
    def _decode_image(
        *,
        bridge: UGLatentFlowMiddleBridge,
        contexts: UGContextBundle,
        batch: Req,
    ) -> Image.Image:
        image = bridge.decode_g_latents(
            contexts=contexts,
            latent_tokens=batch.latents,
            sampling_params=batch.sampling_params,
        )
        if image is None:
            value = int(batch.latents.mean().abs().item() * 255) % 255
            image = Image.fromarray(
                np.full(
                    (int(batch.height), int(batch.width), 3),
                    value,
                    dtype=np.uint8,
                )
            )
        if isinstance(image, Image.Image):
            return image
        array = np.asarray(image)
        if array.ndim == 4:
            array = array[0]
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array)
