# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.models.vision_utils import load_image
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.srt.ug.denoiser import UGDenoiserBridge
from sglang.srt.ug.interleaved import UGInputSegment, UGInterleavedRequest
from sglang.srt.ug.runtime import UGInterleavedMessage


class UGContextStage(PipelineStage):
    def __init__(self, bridge: UGDenoiserBridge) -> None:
        super().__init__()
        self.bridge = bridge

    @property
    def role_affinity(self):
        return RoleType.ENCODER

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        pipeline_config = server_args.pipeline_config
        unsupported = pipeline_config.validate_runtime(
            num_gpus=server_args.num_gpus,
            enable_cfg_parallel=server_args.enable_cfg_parallel,
            disagg_mode=server_args.disagg_mode,
        )
        if unsupported:
            raise ValueError(f"Unsupported UGPipeline runtime settings: {unsupported}")

        if batch.height is None:
            batch.height = pipeline_config.default_height
        if batch.width is None:
            batch.width = pipeline_config.default_width

        interleaved_messages = batch.extra.get("ug_interleaved_messages")
        if interleaved_messages is not None:
            messages = _normalize_pipeline_interleaved_messages(interleaved_messages)
            batch.extra["ug_interleaved_messages"] = messages
            batch.extra["ug_contexts"] = self.bridge.build_contexts_from_messages(
                messages=messages
            )
            batch.extra["ug_pre_image_segments"] = batch.extra[
                "ug_contexts"
            ].full.metadata.get("pre_image_segments", [])
            return batch

        if batch.condition_image is None and batch.image_path is not None:
            if isinstance(batch.image_path, list):
                if len(batch.image_path) != 1:
                    raise ValueError("UGPipeline MVP supports at most one input image")
                batch.condition_image = load_image(batch.image_path[0])
            else:
                batch.condition_image = load_image(batch.image_path)

        if batch.condition_image is not None and not isinstance(
            batch.condition_image, Image.Image
        ):
            raise TypeError(
                f"UGPipeline expects a PIL image input, got {type(batch.condition_image)}"
            )

        batch.extra["ug_contexts"] = self.bridge.build_contexts(
            prompt=batch.prompt,
            image=batch.condition_image,
        )
        batch.extra["ug_pre_image_segments"] = batch.extra[
            "ug_contexts"
        ].full.metadata.get("pre_image_segments", [])
        return batch


class UGLatentStage(PipelineStage):
    def __init__(self, bridge: UGDenoiserBridge) -> None:
        super().__init__()
        self.bridge = bridge

    @property
    def role_affinity(self):
        return RoleType.DENOISER

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        cfg = server_args.pipeline_config
        contexts = batch.extra.get("ug_contexts")
        if contexts is not None:
            prepared = self.bridge.prepare_latents(
                contexts=contexts,
                sampling_params=batch.sampling_params,
                seed=batch.seed,
            )
            if prepared is not None:
                batch.latents = prepared.latent_tokens
                batch.extra["ug_latent_position_ids"] = prepared.latent_position_ids
                batch.extra["ug_latent_shape"] = prepared.latent_shape
                return batch

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
        return batch


class UGDenoiseStage(PipelineStage):
    def __init__(self, bridge: UGDenoiserBridge) -> None:
        super().__init__()
        self.bridge = bridge

    @property
    def role_affinity(self):
        return RoleType.DENOISER

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        params = batch.sampling_params
        x_t = batch.latents
        if x_t is None:
            raise ValueError("UGDenoiseStage requires latents from UGLatentStage")
        num_steps = int(params.num_inference_steps)
        if num_steps <= 0:
            raise ValueError(f"num_inference_steps must be positive, got {num_steps}")

        timesteps = torch.linspace(1, 0, num_steps, dtype=x_t.dtype)
        timesteps = (
            params.timestep_shift
            * timesteps
            / (1 + (params.timestep_shift - 1) * timesteps)
        )
        dts = timesteps[:-1] - timesteps[1:]
        trajectory_latents = []
        trajectory_timesteps = []

        for i, timestep in enumerate(timesteps[:-1]):
            trajectory_latents.append(x_t)
            trajectory_timesteps.append(timestep)
            velocity = self.bridge.predict_velocity(
                contexts=batch.extra["ug_contexts"],
                latent_tokens=x_t,
                timestep=timestep.reshape(1),
                latent_position_ids=batch.extra["ug_latent_position_ids"],
                sampling_params=params,
            )
            x_t = x_t - velocity.to(x_t) * dts[i].to(x_t)

        batch.latents = x_t
        if batch.return_trajectory_latents:
            if trajectory_latents:
                batch.trajectory_latents = torch.stack(trajectory_latents)
                batch.trajectory_timesteps = torch.stack(trajectory_timesteps)
            else:
                batch.trajectory_latents = x_t[:0]
                batch.trajectory_timesteps = timesteps[:0]
        return batch


class UGDecodeStage(PipelineStage):
    def __init__(self, bridge: UGDenoiserBridge) -> None:
        super().__init__()
        self.bridge = bridge

    @property
    def role_affinity(self):
        return RoleType.DECODER

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        contexts = batch.extra.get("ug_contexts")
        image = None
        if contexts is not None:
            image = self.bridge.decode_latents(
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
        batch.output = _image_to_numpy_batch(image)
        image_for_append = (
            image
            if isinstance(image, Image.Image)
            else Image.fromarray(batch.output[0])
        )
        if contexts is not None:
            self.bridge.append_generated_image(
                contexts=contexts,
                image=image_for_append,
            )
            batch.extra["ug_post_image_segment"] = self.bridge.decode_next_segment(
                contexts=contexts
            )
            batch.extra["ug_output_segments"] = _build_ug_output_segments(
                pre_image_segments=batch.extra.get("ug_pre_image_segments", []),
                image=image_for_append,
                post_image_segment=batch.extra["ug_post_image_segment"],
            )
        return batch


def _image_to_numpy_batch(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
    else:
        array = np.asarray(image)
    if array.ndim == 3:
        array = array[None, ...]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _normalize_pipeline_interleaved_messages(messages) -> list[UGInterleavedMessage]:
    if isinstance(messages, UGInterleavedRequest):
        messages = messages.to_legacy_segments()
    normalized = []
    for message in messages:
        if isinstance(message, UGInterleavedMessage):
            normalized.append(message)
            continue
        if isinstance(message, UGInputSegment):
            message = message.to_legacy_segment()
        if not isinstance(message, dict):
            raise TypeError(f"UG interleaved message must be a dict: {message!r}")
        message_type = message.get("type")
        if message_type == "text":
            content = message.get("text", message.get("content"))
        elif message_type == "image":
            content = message.get("image", message.get("content"))
            if content is None:
                raise ValueError("UG image message is missing content")
            if not isinstance(content, Image.Image):
                content = load_image(content)
        else:
            raise ValueError(
                f"Unsupported UG interleaved message type: {message_type!r}"
            )
        if content is None:
            raise ValueError(f"UG {message_type} message is missing content")
        normalized.append(UGInterleavedMessage(type=message_type, content=content))
    if not normalized:
        raise ValueError("UG interleaved messages must not be empty")
    return normalized


def _build_ug_output_segments(
    *,
    pre_image_segments,
    image: Image.Image,
    post_image_segment,
) -> list[dict]:
    output_segments = list(pre_image_segments)
    output_segments.append({"type": "image", "image": image})
    if post_image_segment.type == "text":
        output_segments.append({"type": "text", "text": post_image_segment.text or ""})
    elif post_image_segment.type != "done":
        raise ValueError(
            "UG interleaved output expected text or done after generated image, "
            f"got {post_image_segment.type}"
        )
    return output_segments
