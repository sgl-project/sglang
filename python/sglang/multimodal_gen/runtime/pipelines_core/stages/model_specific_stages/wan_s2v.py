# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import os
import tempfile
from urllib.parse import urlparse
from urllib.request import urlopen

import PIL.Image
import torch
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from torchvision import transforms

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.models.vaes.common import (
    DiagonalGaussianDistribution,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators,
    VerificationResult,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
V = StageValidators


class WanS2VBeforeDenoisingStage(PipelineStage):
    """Prepare Wan S2V conditions for the standard denoising loop."""

    def __init__(self, transformer, vae, text_encoder, tokenizer, audio_encoder):
        super().__init__()
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.audio_encoder = audio_encoder

    @property
    def parallelism_type(self) -> StageParallelismType:
        return StageParallelismType.MAIN_RANK_ONLY

    def _get_single_path(self, value, field_name: str) -> str:
        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError(f"Wan S2V expects exactly one {field_name}")
            value = value[0]
        if not isinstance(value, str) or not value:
            raise ValueError(f"Wan S2V expects {field_name} as a non-empty string")
        if value.lower().startswith(("http://", "https://")):
            return self._download_remote_input(value, field_name)
        if not os.path.exists(value):
            raise FileNotFoundError(f"{field_name} not found: {value}")
        return value

    def _download_remote_input(self, url: str, field_name: str) -> str:
        parsed = urlparse(url)
        ext = os.path.splitext(parsed.path)[1] or (
            ".wav" if field_name == "audio_path" else ".jpg"
        )
        cache_dir = os.path.join(
            tempfile.gettempdir(), "sglang_wan_s2v_remote_inputs"
        )
        os.makedirs(cache_dir, exist_ok=True)
        target_path = os.path.join(
            cache_dir, f"{field_name}_{hashlib.sha256(url.encode()).hexdigest()[:16]}{ext}"
        )
        if os.path.exists(target_path):
            return target_path
        with urlopen(url) as response, open(target_path, "wb") as fout:
            fout.write(response.read())
        return target_path

    def _generate_seed_and_generator(self, batch: Req) -> None:
        seed = batch.seed if batch.seed is not None else 42
        batch.seeds = [seed]
        generator_device = (
            "cpu" if batch.generator_device == "cpu" else current_platform.device_type
        )
        batch.generator = torch.Generator(generator_device).manual_seed(seed)

    def _encode_text_prompt(
        self,
        prompt: str,
        *,
        server_args: ServerArgs,
    ) -> torch.Tensor:
        if hasattr(self.text_encoder, "encode_prompt"):
            prompt_embeds = self.text_encoder.encode_prompt(
                prompt,
                get_local_torch_device(),
            )
            return prompt_embeds.to(get_local_torch_device())
        encoder_config = server_args.pipeline_config.text_encoder_configs[0]
        preprocess_func = server_args.pipeline_config.preprocess_text_funcs[0]
        postprocess_func = server_args.pipeline_config.postprocess_text_funcs[0]
        text_encoder_extra_arg = (
            server_args.pipeline_config.text_encoder_extra_args[0]
            if server_args.pipeline_config.text_encoder_extra_args
            else {}
        )
        processed_prompt = preprocess_func(prompt)
        tok_kwargs = dict(encoder_config.tokenizer_kwargs)
        tok_kwargs.update(text_encoder_extra_arg or {})
        text_inputs = server_args.pipeline_config.tokenize_prompt(
            [processed_prompt], self.tokenizer, tok_kwargs
        )
        encoder_device = next(self.text_encoder.parameters()).device
        text_inputs = text_inputs.to(encoder_device)
        outputs = self.text_encoder(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs.get("attention_mask"),
            output_hidden_states=True,
        )
        prompt_embeds = postprocess_func(outputs, text_inputs)
        attention_mask = text_inputs.get("attention_mask")
        if (
            attention_mask is not None
            and prompt_embeds.ndim == 3
            and prompt_embeds.shape[0] == attention_mask.shape[0]
        ):
            valid_tokens = int(attention_mask[0].sum().item())
            prompt_embeds = prompt_embeds[:, :valid_tokens]
        return prompt_embeds.to(get_local_torch_device())

    def _retrieve_latents(
        self,
        encoder_output: DiagonalGaussianDistribution | AutoencoderKLOutput,
    ) -> torch.Tensor:
        if isinstance(encoder_output, AutoencoderKLOutput):
            encoder_output = encoder_output.latent_dist
        if isinstance(encoder_output, DiagonalGaussianDistribution):
            return encoder_output.mode()
        raise TypeError(
            f"Unexpected VAE encoder output type: {type(encoder_output).__name__}"
        )

    def _encode_video_to_latents(
        self, video: torch.Tensor, server_args: ServerArgs
    ) -> torch.Tensor:
        vae_dtype = torch.float32
        if server_args.pipeline_config.vae_precision == "bf16":
            vae_dtype = torch.bfloat16
        elif server_args.pipeline_config.vae_precision == "fp16":
            vae_dtype = torch.float16
        self.vae = self.vae.to(device=get_local_torch_device(), dtype=vae_dtype)
        video = video.to(device=get_local_torch_device(), dtype=vae_dtype)
        if hasattr(self.vae, "encode_video"):
            return self.vae.encode_video(video)
        latent_dist = self.vae.encode(video)
        latents = self._retrieve_latents(latent_dist)
        latents = server_args.pipeline_config.postprocess_vae_encode(latents, self.vae)
        scaling_factor, shift_factor = (
            server_args.pipeline_config.get_decode_scale_and_shift(
                device=latents.device,
                dtype=latents.dtype,
                vae=self.vae,
            )
        )
        if shift_factor is not None:
            if isinstance(shift_factor, torch.Tensor):
                latents = latents - shift_factor.to(latents.device, latents.dtype)
            else:
                latents = latents - shift_factor
        if scaling_factor is not None:
            if isinstance(scaling_factor, torch.Tensor):
                latents = latents * scaling_factor.to(latents.device, latents.dtype)
            else:
                latents = latents * scaling_factor
        return latents

    def _load_pose_cond(
        self,
        pose_video: str | None,
        *,
        infer_frames: int,
        size: tuple[int, int],
        server_args: ServerArgs,
    ) -> torch.Tensor:
        from decord import VideoReader

        height, width = size
        if pose_video is not None:
            vr = VideoReader(pose_video)
            original_fps = vr.get_avg_fps()
            total_frames = len(vr)
            interval = max(1, round(original_fps / self.transformer.fps))
            required_span = (infer_frames - 1) * interval
            start_frame = 0
            sampled_indices = []
            for i in range(infer_frames):
                idx = start_frame + i * interval
                if idx >= total_frames:
                    break
                sampled_indices.append(idx)
            pose_seq = vr.get_batch(sampled_indices).asnumpy()
            resize = transforms.Resize(min(height, width))
            crop = transforms.CenterCrop((height, width))
            cond_tensor = torch.from_numpy(pose_seq).permute(0, 3, 1, 2) / 255.0
            cond_tensor = cond_tensor * 2 - 1
            cond_tensor = crop(resize(cond_tensor)).permute(1, 0, 2, 3).unsqueeze(0)
            padding_frames = infer_frames - cond_tensor.shape[2]
            if padding_frames > 0:
                cond_tensor = torch.cat(
                    [
                        cond_tensor,
                        -torch.ones([1, 3, padding_frames, height, width]),
                    ],
                    dim=2,
                )
        else:
            cond_tensor = -torch.ones([1, 3, infer_frames, height, width])

        cond_tensor = torch.cat([cond_tensor[:, :, :1], cond_tensor], dim=2)
        cond_latents = self._encode_video_to_latents(cond_tensor, server_args)[:, :, 1:]
        if pose_video is None:
            cond_latents = cond_latents * 0
        return cond_latents

    def _decode_source_audio(
        self, audio_path: str
    ) -> tuple[torch.Tensor | None, int | None]:
        if sf is None:
            return None, None
        try:
            audio_np, sample_rate = sf.read(
                audio_path, dtype="float32", always_2d=False
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to load source audio for output muxing: %s", exc)
            return None, None
        if audio_np is None:
            return None, None
        return torch.from_numpy(audio_np), int(sample_rate)

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("image_path", batch.image_path, V.not_none)
        result.add_check("audio_path", batch.audio_path, V.not_none)
        result.add_check("num_frames", batch.num_frames, V.positive_int)
        return result

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if not getattr(self.transformer, "supports_standard_denoising", False):
            raise RuntimeError(
                "Wan S2V standard denoising path requested, but transformer does not support it"
            )

        image_path = self._get_single_path(batch.image_path, "image_path")
        audio_path = self._get_single_path(batch.audio_path, "audio_path")
        pose_video_path = batch.pose_video_path
        if pose_video_path is not None:
            pose_video_path = self._get_single_path(pose_video_path, "pose_video_path")

        self._generate_seed_and_generator(batch)
        if batch.num_clip not in (None, 1):
            raise NotImplementedError(
                "Native Wan S2V standard denoising currently supports only a single clip"
            )

        infer_frames = self.transformer._normalize_infer_frames(batch.num_frames)
        height, width = self.transformer.get_generation_size(image_path=image_path)

        resize = transforms.Resize(min(height, width))
        crop = transforms.CenterCrop((height, width))
        to_tensor = transforms.ToTensor()
        model_pic = crop(resize(PIL.Image.open(image_path).convert("RGB")))
        ref_pixel_values = to_tensor(model_pic).unsqueeze(1).unsqueeze(0) * 2 - 1.0
        ref_latents = self._encode_video_to_latents(ref_pixel_values, server_args)

        motion_frames = self.transformer.motion_frames
        motion_pixels = torch.zeros([1, 3, motion_frames, height, width])
        motion_latents = self._encode_video_to_latents(motion_pixels, server_args)
        lat_motion_frames = (motion_frames + 3) // 4

        audio_input, max_num_repeat = self.audio_encoder.encode_audio(
            audio_path,
            infer_frames=infer_frames,
            fps=self.transformer.fps,
            dtype=self.transformer.param_dtype,
            m=self.transformer.audio_sample_m,
        )
        if max_num_repeat < 1:
            raise ValueError(f"Audio path produced no valid clips: {audio_path}")

        cond_states = self._load_pose_cond(
            pose_video_path,
            infer_frames=infer_frames,
            size=(height, width),
            server_args=server_args,
        ).to(dtype=self.transformer.param_dtype, device=get_local_torch_device())

        negative_prompt = batch.negative_prompt
        if not negative_prompt:
            negative_prompt = self.transformer.get_default_negative_prompt()

        prompt_embeds = self._encode_text_prompt(
            batch.prompt or "", server_args=server_args
        )
        negative_prompt_embeds = self._encode_text_prompt(
            negative_prompt, server_args=server_args
        )

        lat_target_frames = (infer_frames + 3 + motion_frames) // 4 - lat_motion_frames
        latent_shape = (1, 16, lat_target_frames, height // 8, width // 8)
        latents = self.transformer.prepare_standard_s2v_latents(
            latent_shape=latent_shape,
            generator=batch.generator,
        )

        if isinstance(prompt_embeds, list):
            prompt_embeds = prompt_embeds[0]
        if isinstance(negative_prompt_embeds, list):
            negative_prompt_embeds = negative_prompt_embeds[0]
        if prompt_embeds.ndim == 3 and prompt_embeds.shape[0] == 1:
            prompt_embeds = prompt_embeds[0]
        if negative_prompt_embeds.ndim == 3 and negative_prompt_embeds.shape[0] == 1:
            negative_prompt_embeds = negative_prompt_embeds[0]

        batch.prompt_embeds = [prompt_embeds]
        if batch.do_classifier_free_guidance:
            batch.negative_prompt_embeds = [negative_prompt_embeds]
        batch.latents = latents
        batch.raw_latent_shape = latents.shape
        batch.height = int(height)
        batch.width = int(width)
        batch.audio, batch.audio_sample_rate = self._decode_source_audio(audio_path)
        batch.extra["wan_s2v"] = {
            "ref_latents": ref_latents,
            "motion_latents": motion_latents,
            "cond_states": cond_states,
            "audio_input": audio_input[..., :infer_frames],
            "motion_frames": [motion_frames, lat_motion_frames],
            "drop_motion_frames": bool(self.transformer.drop_first_motion),
            "infer_frames": int(infer_frames),
        }
        if server_args.vae_cpu_offload:
            self.vae = self.vae.to("cpu")
        if getattr(server_args, "audio_encoder_cpu_offload", False):
            self.audio_encoder = self.audio_encoder.to("cpu")
        return batch

