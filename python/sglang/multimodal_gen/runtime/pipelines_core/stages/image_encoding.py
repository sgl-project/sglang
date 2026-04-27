# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Image encoding stages for I2V diffusion pipelines.

This module contains implementations of image encoding stages for diffusion pipelines.
"""

import inspect
from dataclasses import dataclass
from typing import Any

import numpy as np
import PIL
import PIL.Image
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput

from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    qwen_image_postprocess_text,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.vaes.common import ParallelTiledVAE
from sglang.multimodal_gen.runtime.models.vision_utils import (
    normalize,
    numpy_to_pt,
    pil_to_numpy,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


@dataclass(frozen=True)
class ImageEncodingFingerprint:
    image_source: Any
    prompt: Any
    negative_prompt: Any
    do_classifier_free_guidance: bool
    height: int | None
    width: int | None
    num_frames: int | None


@dataclass(frozen=True)
class LTX2ImageEncodingFingerprint:
    image_source: Any
    height: int | None
    width: int | None
    num_frames: int | None
    latent_dtype: str
    condition_encoder_subdir: str
    encode_sample_mode: str


@dataclass(frozen=True)
class ImageVAEEncodingFingerprint:
    image_source: Any
    height: int | None
    width: int | None
    num_frames: int | None
    encode_sample_mode: str
    vae_precision: Any
    vae_tiling: bool


def _freeze_image_source_value(value):
    """Build a hashable identity fragment for image inputs.

    Image inputs are often PIL/numpy/tensor objects. For file paths we can use
    the path value; for in-memory objects we only dedup when the exact same
    object instance is shared by multiple requests. This avoids expensive image
    hashing and avoids treating two mutable image objects as equivalent just
    because they currently have the same shape.
    """
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_image_source_value(item) for item in value)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return ("object", id(value))


def _build_image_source_fingerprint(batch: Req, *, prefer_vae_image: bool = False):
    """Return the image input fragment used by image encoding fingerprints."""
    if batch.image_path is not None:
        return ("path", PipelineStage.freeze_for_dedup(batch.image_path))
    image = (
        batch.vae_image if prefer_vae_image and batch.vae_image is not None else None
    )
    if image is None:
        image = batch.condition_image
    return ("image", _freeze_image_source_value(image))


class ImageEncodingStage(PipelineStage):
    """
    Stage for encoding image prompts into embeddings for diffusion models.

    This stage handles the encoding of image prompts into the embedding space
    expected by the diffusion model.
    """

    deduplicated_output_fields = (
        "image_embeds",
        "prompt_embeds",
        "negative_prompt_embeds",
    )

    def __init__(
        self,
        image_processor,
        image_encoder=None,
        text_encoder=None,
    ) -> None:
        """
        Initialize the prompt encoding stage.

        Args:
            text_encoder: An encoder to encode input_ids and pixel values
        """
        super().__init__()
        self.image_processor = image_processor
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

    def load_model(self):
        if self.server_args.image_encoder_cpu_offload:
            device = get_local_torch_device()
            self.move_to_device(device)

    def offload_model(self):
        if self.server_args.image_encoder_cpu_offload:
            self.move_to_device("cpu")

    def move_to_device(self, device):
        if self.server_args.use_fsdp_inference:
            return
        fields = [
            "image_processor",
            "image_encoder",
        ]
        for field in fields:
            processor = getattr(self, field, None)
            if processor and hasattr(processor, "to"):
                setattr(self, field, processor.to(device))

    def encoding_qwen_image_edit(self, outputs, image_inputs):
        # encoder hidden state
        prompt_embeds = qwen_image_postprocess_text(outputs, image_inputs, 64)
        return prompt_embeds

    @torch.no_grad()
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Encode the prompt into image encoder hidden states.
        """

        if batch.condition_image is None:
            return batch
        cuda_device = get_local_torch_device()

        self.load_model()

        image_processor_kwargs = (
            server_args.pipeline_config.prepare_image_processor_kwargs(batch)
        )
        per_prompt_images = image_processor_kwargs.pop("per_prompt_images", None)
        texts = image_processor_kwargs.pop("text", None)

        if per_prompt_images is None:
            per_prompt_images = [batch.condition_image]
            texts = [None] if texts is None else texts

        all_prompt_embeds = []
        all_neg_prompt_embeds = []

        image_processor_call_params = inspect.signature(
            self.image_processor.__call__
        ).parameters
        image_processor_kwargs = {
            k: v
            for k, v in image_processor_kwargs.items()
            if k in image_processor_call_params
        }

        for idx, prompt_images in enumerate(per_prompt_images):
            if not prompt_images:
                continue

            cur_kwargs = image_processor_kwargs.copy()
            if texts and idx < len(texts) and "text" in image_processor_call_params:
                cur_kwargs["text"] = [texts[idx]]

            image_inputs = self.image_processor(
                images=prompt_images, return_tensors="pt", **cur_kwargs
            ).to(cuda_device)

            if self.image_encoder:
                # if an image encoder is provided
                with set_forward_context(current_timestep=0, attn_metadata=None):
                    outputs = self.image_encoder(
                        **image_inputs,
                        **server_args.pipeline_config.image_encoder_extra_args,
                    )
                    image_embeds = server_args.pipeline_config.postprocess_image(
                        outputs
                    )
                batch.image_embeds.append(image_embeds)
            elif self.text_encoder:
                # if a text encoder is provided, e.g. Qwen-Image-Edit
                # 1. neg prompt embeds
                if batch.do_classifier_free_guidance:
                    neg_image_processor_kwargs = (
                        server_args.pipeline_config.prepare_image_processor_kwargs(
                            batch, neg=True
                        )
                    )
                    neg_image_processor_kwargs.pop("per_prompt_images", None)
                    neg_texts = neg_image_processor_kwargs.pop("text", None)
                    if neg_texts and idx < len(neg_texts):
                        neg_image_processor_kwargs["text"] = [neg_texts[idx]]
                    neg_image_inputs = self.image_processor(
                        images=prompt_images,
                        return_tensors="pt",
                        **neg_image_processor_kwargs,
                    ).to(cuda_device)

                with set_forward_context(current_timestep=0, attn_metadata=None):
                    outputs = self.text_encoder(
                        input_ids=image_inputs.input_ids,
                        attention_mask=image_inputs.attention_mask,
                        pixel_values=image_inputs.pixel_values,
                        image_grid_thw=image_inputs.image_grid_thw,
                        output_hidden_states=True,
                    )
                    if batch.do_classifier_free_guidance:
                        neg_outputs = self.text_encoder(
                            input_ids=neg_image_inputs.input_ids,
                            attention_mask=neg_image_inputs.attention_mask,
                            pixel_values=neg_image_inputs.pixel_values,
                            image_grid_thw=neg_image_inputs.image_grid_thw,
                            output_hidden_states=True,
                        )

                all_prompt_embeds.append(
                    self.encoding_qwen_image_edit(outputs, image_inputs)
                )
                if batch.do_classifier_free_guidance:
                    all_neg_prompt_embeds.append(
                        self.encoding_qwen_image_edit(neg_outputs, neg_image_inputs)
                    )

        if all_prompt_embeds:
            batch.prompt_embeds.append(torch.cat(all_prompt_embeds, dim=0))
        if all_neg_prompt_embeds:
            batch.negative_prompt_embeds.append(torch.cat(all_neg_prompt_embeds, dim=0))

        self.offload_model()

        return batch

    def build_dedup_fingerprint(
        self, batch: Req, server_args: ServerArgs
    ) -> ImageEncodingFingerprint:
        return ImageEncodingFingerprint(
            image_source=_build_image_source_fingerprint(batch),
            prompt=self.freeze_for_dedup(batch.prompt),
            negative_prompt=self.freeze_for_dedup(batch.negative_prompt),
            do_classifier_free_guidance=bool(batch.do_classifier_free_guidance),
            height=batch.height,
            width=batch.width,
            num_frames=batch.num_frames,
        )

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify image encoding stage inputs."""
        result = VerificationResult()
        if batch.debug:
            logger.debug(f"{batch.condition_image=}")
            logger.debug(f"{batch.image_embeds=}")
        result.add_check("pil_image", batch.condition_image, V.not_none)
        result.add_check("image_embeds", batch.image_embeds, V.is_list)
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify image encoding stage outputs."""
        result = VerificationResult()
        # result.add_check("image_embeds", batch.image_embeds, V.list_of_tensors_dims(3))
        return result


class LTX2ImageEncodingStage(PipelineStage):
    """Encode ``batch.image_path`` into packed token latents for LTX-2 TI2V.

    Runs before denoising. Populates:
      - ``batch.condition_image`` (resized PIL image)
      - ``batch.image_latent``    (packed [B, S0, D] token latents)
      - ``batch.ltx2_num_image_tokens``
    """

    deduplicated_output_fields = (
        "condition_image",
        "image_latent",
        "ltx2_num_image_tokens",
    )

    def __init__(self, vae=None, **kwargs) -> None:
        super().__init__()
        self.vae = vae
        self._condition_image_encoder = None
        self._condition_image_encoder_dir = None

    # -- device management (mirrors ImageVAEEncodingStage) ---------------

    def load_model(self):
        device = get_local_torch_device()
        if self._condition_image_encoder is not None:
            self._condition_image_encoder = self._condition_image_encoder.to(device)
        else:
            self.vae = self.vae.to(device)

    def offload_model(self):
        if self.server_args.vae_cpu_offload:
            self.vae = self.vae.to("cpu")
            if self._condition_image_encoder is not None:
                self._condition_image_encoder = self._condition_image_encoder.to("cpu")

    # -- lazy condition encoder (LTX-2.3) --------------------------------

    def _ensure_condition_image_encoder(self, server_args: ServerArgs) -> bool:
        """Load LTX-2.3 condition-encoder weights on first call. Returns True if available."""
        import json
        import os

        arch_config = server_args.pipeline_config.vae_config.arch_config
        encoder_subdir = str(getattr(arch_config, "condition_encoder_subdir", ""))
        if not encoder_subdir:
            return False

        vae_model_path = server_args.model_paths["vae"]
        encoder_dir = os.path.join(vae_model_path, encoder_subdir)
        if (
            self._condition_image_encoder is not None
            and self._condition_image_encoder_dir == encoder_dir
        ):
            return True

        config_path = os.path.join(encoder_dir, "config.json")
        weights_path = os.path.join(encoder_dir, "model.safetensors")
        if not os.path.exists(config_path) or not os.path.exists(weights_path):
            raise ValueError(
                f"LTX-2 condition encoder files not found under {encoder_dir}"
            )

        from safetensors.torch import load_file as safetensors_load_file

        from sglang.multimodal_gen.runtime.models.vaes.ltx_2_3_condition_encoder import (
            LTX23VideoConditionEncoder,
        )

        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        self._condition_image_encoder = LTX23VideoConditionEncoder(config)
        self._condition_image_encoder.load_state_dict(
            safetensors_load_file(weights_path), strict=True
        )
        self._condition_image_encoder_dir = encoder_dir
        return True

    # -- image preprocessing ---------------------------------------------

    @staticmethod
    def _apply_video_codec_compression(
        img_array: np.ndarray, crf: int = 33
    ) -> np.ndarray:
        """Single H.264 frame round-trip to simulate compression artifacts."""
        from io import BytesIO

        import av

        if crf == 0:
            return img_array
        h, w = img_array.shape[0] // 2 * 2, img_array.shape[1] // 2 * 2
        img_array = img_array[:h, :w]
        buf = BytesIO()
        container = av.open(buf, mode="w", format="mp4")
        stream = container.add_stream(
            "libx264", rate=1, options={"crf": str(crf), "preset": "veryfast"}
        )
        stream.height, stream.width = h, w
        frame = av.VideoFrame.from_ndarray(img_array, format="rgb24").reformat(
            format="yuv420p"
        )
        container.mux(stream.encode(frame))
        container.mux(stream.encode())
        container.close()
        buf.seek(0)
        container = av.open(buf)
        decoded = next(container.decode(container.streams.video[0]))
        container.close()
        return decoded.to_ndarray(format="rgb24")

    @staticmethod
    def _pil_to_video_tensor(
        img: PIL.Image.Image,
        *,
        width: int,
        height: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Scale-to-cover, center-crop, normalize to [1, C, 1, H, W] in [-1, 1]."""
        import math

        arr = np.array(img).astype(np.uint8)[..., :3]
        t = (
            torch.from_numpy(arr.astype(np.float32))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=device)
        )
        src_h, src_w = t.shape[2], t.shape[3]
        scale = max(height / src_h, width / src_w)
        new_h, new_w = math.ceil(src_h * scale), math.ceil(src_w * scale)
        t = torch.nn.functional.interpolate(
            t, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
        top, left = (new_h - height) // 2, (new_w - width) // 2
        t = t[:, :, top : top + height, left : left + width]
        return ((t / 127.5 - 1.0).to(dtype=dtype)).unsqueeze(2)

    # -- encode paths ----------------------------------------------------

    def _vae_encode(
        self,
        video_condition: torch.Tensor,
        server_args: ServerArgs,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """VAE encode → sample → per-channel normalize (LTX-2 convention)."""
        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32
        ) and not server_args.disable_autocast

        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=vae_dtype,
            enabled=vae_autocast_enabled,
        ):
            try:
                if server_args.pipeline_config.vae_tiling:
                    self.vae.enable_tiling()
            except Exception:
                pass
            latent_dist = self.vae.encode(video_condition)
            if isinstance(latent_dist, AutoencoderKLOutput):
                latent_dist = latent_dist.latent_dist

        mode = server_args.pipeline_config.vae_config.encode_sample_mode()
        if mode == "argmax":
            latent = latent_dist.mode()
        elif mode == "sample":
            if generator is None:
                raise ValueError("Generator must be provided for VAE sampling.")
            latent = latent_dist.sample(generator)
        else:
            raise ValueError(f"Unsupported encode_sample_mode: {mode}")

        mean = self.vae.latents_mean.view(1, -1, 1, 1, 1).to(latent)
        std = self.vae.latents_std.view(1, -1, 1, 1, 1).to(latent)
        return (latent - mean) / std

    def _condition_encode(
        self, video_condition: torch.Tensor, server_args: ServerArgs
    ) -> torch.Tensor:
        """LTX-2.3 condition-image encoder path (bypasses VAE)."""
        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32
        ) and not server_args.disable_autocast

        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=vae_dtype,
            enabled=vae_autocast_enabled,
        ):
            return self._condition_image_encoder(video_condition)

    @staticmethod
    def _normalize_ltx2_image_paths(image_path: str | list[str]) -> list[str]:
        image_paths = image_path if isinstance(image_path, list) else [image_path]
        if len(image_paths) > 2:
            raise ValueError(
                "LTX-2 TI2V currently supports at most two conditioning images "
                "([first_frame, last_frame])."
            )
        return image_paths

    @staticmethod
    def _normalize_ltx2_image_latents(
        image_latent: torch.Tensor | list[torch.Tensor] | None,
    ) -> list[torch.Tensor]:
        if image_latent is None:
            return []
        return image_latent if isinstance(image_latent, list) else [image_latent]

    # -- forward ---------------------------------------------------------

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if batch.image_path is None:
            return batch
        image_paths = self._normalize_ltx2_image_paths(batch.image_path)

        vae_sf = int(server_args.pipeline_config.vae_scale_factor)
        patch = int(server_args.pipeline_config.patch_size)
        expected_tokens = (int(batch.height) // vae_sf // patch) * (
            int(batch.width) // vae_sf // patch
        )
        if (
            batch.image_latent is not None
            and int(getattr(batch, "ltx2_num_image_tokens", 0)) > 0
        ):
            # Re-encode if resolution changed (e.g. two-stage upsample between stages)
            existing_latents = self._normalize_ltx2_image_latents(batch.image_latent)
            if len(existing_latents) == len(image_paths) and all(
                int(latent.shape[1]) == expected_tokens for latent in existing_latents
            ):
                return batch
            # Resolution or reference-count mismatch — clear and re-encode below
            batch.image_latent = None
            batch.ltx2_num_image_tokens = 0

        batch.ltx2_num_image_tokens = 0
        batch.image_latent = None

        if self.vae is None:
            raise ValueError("VAE must be provided for LTX-2 TI2V.")

        from sglang.multimodal_gen.runtime.models.vision_utils import load_image

        # 1. Load images, apply codec compression, resize for condition_image
        conditioned_imgs = []
        for image_path in image_paths:
            img = load_image(image_path)
            arr = np.array(img).astype(np.uint8)[..., :3]
            arr = self._apply_video_codec_compression(arr, crf=33)
            conditioned_img = PIL.Image.fromarray(arr)
            conditioned_imgs.append(conditioned_img)
        batch.condition_image = [
            img.resize(
                (int(batch.width), int(batch.height)),
                resample=PIL.Image.Resampling.BILINEAR,
            )
            for img in conditioned_imgs
        ]
        if len(batch.condition_image) == 1:
            batch.condition_image = batch.condition_image[0]

        # 2. Load encoder(s) to device, cast to encode_dtype
        use_condition_encoder = self._ensure_condition_image_encoder(server_args)
        self.load_model()

        device = get_local_torch_device()
        encode_dtype = batch.latents.dtype

        # Cast the active encoder to the latent precision (must match original
        # behavior — running in a different dtype shifts the encoded latents).
        if use_condition_encoder:
            self._condition_image_encoder = self._condition_image_encoder.to(
                dtype=encode_dtype
            )
        else:
            self.vae = self.vae.to(dtype=encode_dtype)

        packed_latents = []
        for conditioned_img in conditioned_imgs:
            video_condition = self._pil_to_video_tensor(
                conditioned_img,
                width=int(batch.width),
                height=int(batch.height),
                device=device,
                dtype=encode_dtype,
            )

            # 3. Encode
            if use_condition_encoder:
                latent = self._condition_encode(video_condition, server_args).to(
                    dtype=encode_dtype
                )
            else:
                latent = self._vae_encode(video_condition, server_args, batch.generator)

            packed = server_args.pipeline_config.maybe_pack_latents(
                latent, latent.shape[0], batch
            )
            if not (isinstance(packed, torch.Tensor) and packed.ndim == 3):
                raise ValueError("Expected packed image latents [B, S0, D].")
            if int(packed.shape[1]) != expected_tokens:
                raise ValueError(
                    f"LTX-2 conditioning token count mismatch: "
                    f"{packed.shape[1]=} {expected_tokens=}."
                )
            packed_latents.append(packed)

        # Restore VAE to its config dtype (shared with decoding stage)
        if not use_condition_encoder:
            original_dtype = PRECISION_TO_TYPE[
                server_args.pipeline_config.vae_precision
            ]
            self.vae = self.vae.to(dtype=original_dtype)

        batch.image_latent = (
            packed_latents[0] if len(packed_latents) == 1 else packed_latents
        )
        batch.ltx2_num_image_tokens = int(packed_latents[0].shape[1])

        if batch.debug:
            logger.info(
                "LTX2 TI2V: %d tokens (shape=%s) for %sx%s",
                batch.ltx2_num_image_tokens,
                tuple(packed_latents[0].shape),
                batch.width,
                batch.height,
            )

        self.offload_model()
        return batch

    def build_dedup_fingerprint(
        self, batch: Req, server_args: ServerArgs
    ) -> LTX2ImageEncodingFingerprint | int:
        if batch.image_path is None or batch.image_latent is not None:
            return id(batch)

        sample_mode = server_args.pipeline_config.vae_config.encode_sample_mode()
        arch_config = server_args.pipeline_config.vae_config.arch_config
        encoder_subdir = str(getattr(arch_config, "condition_encoder_subdir", ""))
        if not encoder_subdir and sample_mode == "sample":
            return id(batch)

        latent_dtype = batch.latents.dtype if batch.latents is not None else None
        return LTX2ImageEncodingFingerprint(
            image_source=_build_image_source_fingerprint(batch),
            height=batch.height,
            width=batch.width,
            num_frames=batch.num_frames,
            latent_dtype=str(latent_dtype),
            condition_encoder_subdir=encoder_subdir,
            encode_sample_mode=sample_mode,
        )


class ImageVAEEncodingStage(PipelineStage):
    """
    Stage for encoding pixel representations into latent space.

    This stage handles the encoding of pixel representations into the final
    input format (e.g., image_latents).
    """

    deduplicated_output_fields = (
        "image_latent",
        "condition_image_latent_ids",
        "vae_image_sizes",
    )

    def __init__(self, vae: ParallelTiledVAE, **kwargs) -> None:
        super().__init__()
        self.vae: ParallelTiledVAE = vae

    def load_model(self):
        self.vae = self.vae.to(get_local_torch_device())

    def offload_model(self):
        if self.server_args.vae_cpu_offload:
            self.vae = self.vae.to("cpu")

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Encode pixel representations into latent space.
        """

        if batch.condition_image is None:
            return batch

        self.load_model()
        num_frames = batch.num_frames

        images = (
            batch.vae_image if batch.vae_image is not None else batch.condition_image
        )
        if not isinstance(images, list):
            images = [images]

        all_image_latents = []
        prepare_condition_image_latent_ids = getattr(
            server_args.pipeline_config, "prepare_condition_image_latent_ids", None
        )
        condition_latents = [] if callable(prepare_condition_image_latent_ids) else None
        for image in images:
            image = self.preprocess(
                image,
            ).to(get_local_torch_device(), dtype=torch.float32)

            # (B, C, H, W) -> (B, C, 1, H, W)
            image = image.unsqueeze(2)

            if num_frames == 1:
                video_condition = image
            else:
                video_condition = torch.cat(
                    [
                        image,
                        image.new_zeros(
                            image.shape[0],
                            image.shape[1],
                            num_frames - 1,
                            image.shape[3],
                            image.shape[4],
                        ),
                    ],
                    dim=2,
                )
            video_condition = video_condition.to(
                device=get_local_torch_device(), dtype=torch.float32
            )

            # Setup VAE precision
            vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
            vae_autocast_enabled = (
                vae_dtype != torch.float32
            ) and not server_args.disable_autocast

            # Encode Image
            with torch.autocast(
                device_type=current_platform.device_type,
                dtype=vae_dtype,
                enabled=vae_autocast_enabled,
            ):
                if server_args.pipeline_config.vae_tiling:
                    self.vae.enable_tiling()
                # if server_args.vae_sp:
                #     self.vae.enable_parallel()
                if not vae_autocast_enabled:
                    video_condition = video_condition.to(vae_dtype)
                latent_dist: DiagonalGaussianDistribution = self.vae.encode(
                    video_condition
                )
                # for auto_encoder from diffusers
                if isinstance(latent_dist, AutoencoderKLOutput):
                    latent_dist = latent_dist.latent_dist

            generator = batch.generator
            if generator is None:
                raise ValueError("Generator must be provided")

            sample_mode = server_args.pipeline_config.vae_config.encode_sample_mode()

            latent_condition = self.retrieve_latents(
                latent_dist, generator, sample_mode=sample_mode
            )
            latent_condition = server_args.pipeline_config.postprocess_vae_encode(
                latent_condition, self.vae
            )
            normalized_latent_condition = (
                server_args.pipeline_config.normalize_vae_encode(
                    latent_condition, self.vae
                )
            )
            if normalized_latent_condition is None:
                scaling_factor, shift_factor = (
                    server_args.pipeline_config.get_decode_scale_and_shift(
                        device=latent_condition.device,
                        dtype=latent_condition.dtype,
                        vae=self.vae,
                    )
                )

                # apply shift & scale if needed
                if isinstance(shift_factor, torch.Tensor):
                    shift_factor = shift_factor.to(latent_condition.device)

                if isinstance(scaling_factor, torch.Tensor):
                    scaling_factor = scaling_factor.to(latent_condition.device)

                latent_condition -= shift_factor
                latent_condition = latent_condition * scaling_factor
            else:
                latent_condition = normalized_latent_condition

            if condition_latents is not None:
                condition_latents.append(latent_condition)

            image_latent = server_args.pipeline_config.postprocess_image_latent(
                latent_condition, batch
            )
            all_image_latents.append(image_latent)

        batch.image_latent = torch.cat(all_image_latents, dim=1)
        if condition_latents is not None:
            prepare_condition_image_latent_ids(condition_latents, batch)

        self.offload_model()
        return batch

    def build_dedup_fingerprint(
        self, batch: Req, server_args: ServerArgs
    ) -> ImageVAEEncodingFingerprint | int:
        if batch.condition_image is None:
            return id(batch)

        sample_mode = server_args.pipeline_config.vae_config.encode_sample_mode()
        if sample_mode == "sample":
            return id(batch)

        return ImageVAEEncodingFingerprint(
            image_source=_build_image_source_fingerprint(batch, prefer_vae_image=True),
            height=batch.height,
            width=batch.width,
            num_frames=batch.num_frames,
            encode_sample_mode=sample_mode,
            vae_precision=server_args.pipeline_config.vae_precision,
            vae_tiling=bool(server_args.pipeline_config.vae_tiling),
        )

    def retrieve_latents(
        self,
        encoder_output: DiagonalGaussianDistribution,
        generator: torch.Generator | None = None,
        sample_mode: str = "sample",
    ):
        if sample_mode == "sample":
            return encoder_output.sample(generator)
        elif sample_mode == "argmax":
            return encoder_output.mode()
        else:
            raise AttributeError("Could not access latents of provided encoder_output")

    def preprocess(
        self,
        image: torch.Tensor | PIL.Image.Image,
    ) -> torch.Tensor:
        if isinstance(image, PIL.Image.Image):
            image = pil_to_numpy(image)  # to np
            image = numpy_to_pt(image)  # to pt

        do_normalize = True
        if image.min() < 0:
            do_normalize = False
        if do_normalize:
            image = normalize(image)

        return image

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify encoding stage inputs."""
        result = VerificationResult()

        assert batch.condition_image is None or (
            isinstance(batch.condition_image, PIL.Image.Image)
            or isinstance(batch.condition_image, torch.Tensor)
            or isinstance(batch.condition_image, list)
        )
        assert batch.height is not None and isinstance(batch.height, int)
        assert batch.width is not None and isinstance(batch.width, int)
        assert batch.num_frames is not None and isinstance(batch.num_frames, int)

        result.add_check("generator", batch.generator, V.generator_or_list_generators)
        result.add_check("height", batch.height, V.positive_int)
        result.add_check("width", batch.width, V.positive_int)
        result.add_check("num_frames", batch.num_frames, V.positive_int)
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify encoding stage outputs."""
        result = VerificationResult()
        # result.add_check(
        #     "image_latent", batch.image_latent, [V.is_tensor, V.with_dims(5)]
        # )
        return result
