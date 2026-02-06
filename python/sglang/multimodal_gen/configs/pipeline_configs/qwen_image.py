# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from dataclasses import dataclass, field
from typing import Callable

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.qwenimage import (
    QwenImageDitConfig,
    QwenImageEditPlus_2511_DitConfig,
)
from sglang.multimodal_gen.configs.models.encoders.qwen_image import Qwen2_5VLConfig
from sglang.multimodal_gen.configs.models.vaes.qwenimage import QwenImageVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ImagePipelineConfig,
    ModelTaskType,
    maybe_unpad_latents,
    shard_rotary_emb_for_sp,
)
from sglang.multimodal_gen.runtime.models.vision_utils import resize
from sglang.multimodal_gen.utils import calculate_dimensions


def _extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor):
    bool_mask = mask.bool()
    valid_lengths = bool_mask.sum(dim=1)
    selected = hidden_states[bool_mask]
    split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

    return split_result


def qwen_image_preprocess_text(prompt):
    prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

    template = prompt_template_encode
    txt = template.format(prompt)
    return txt


def qwen_image_postprocess_text(outputs, _text_inputs, drop_idx=34):
    # squeeze the batch dim
    hidden_states = outputs.hidden_states[-1]
    split_hidden_states = _extract_masked_hidden(
        hidden_states, _text_inputs.attention_mask
    )
    split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
    max_seq_len = max([e.size(0) for e in split_hidden_states])
    prompt_embeds = torch.stack(
        [
            torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))])
            for u in split_hidden_states
        ]
    )
    return prompt_embeds


# Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.QwenImagePipeline._pack_latents
def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(
        batch_size, num_channels_latents, height // 2, 2, width // 2, 2
    )
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(
        batch_size, (height // 2) * (width // 2), num_channels_latents * 4
    )

    return latents


@dataclass
class QwenImagePipelineConfig(ImagePipelineConfig):
    """Configuration for the QwenImage pipeline."""

    should_use_guidance: bool = False
    task_type: ModelTaskType = ModelTaskType.T2I

    vae_tiling: bool = False

    vae_sp: bool = False

    dit_config: DiTConfig = field(default_factory=QwenImageDitConfig)
    # VAE
    vae_config: VAEConfig = field(default_factory=QwenImageVAEConfig)

    enable_autocast: bool = False

    # Text encoding stage
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (Qwen2_5VLConfig(),)
    )

    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))

    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (qwen_image_preprocess_text,)
    )

    postprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (qwen_image_postprocess_text,)
    )
    text_encoder_extra_args: list[dict] = field(
        default_factory=lambda: [
            dict(
                padding=True,
                truncation=True,
            ),
            None,
        ]
    )

    def prepare_sigmas(self, sigmas, num_inference_steps):
        return self._prepare_sigmas(sigmas, num_inference_steps)

    def prepare_image_processor_kwargs(self, batch, neg=False):
        prompt = batch.prompt if not neg else batch.negative_prompt
        if prompt:
            prompt_template_encode = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
            txt = prompt_template_encode.format(batch.prompt)
            return dict(text=[txt], padding=True)
        else:
            return {}

    def get_vae_scale_factor(self):
        return self.vae_config.arch_config.vae_scale_factor

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        vae_scale_factor = self.vae_config.arch_config.vae_scale_factor
        height = 2 * (batch.height // (vae_scale_factor * 2))
        width = 2 * (batch.width // (vae_scale_factor * 2))
        num_channels_latents = self.dit_config.arch_config.in_channels // 4
        shape = (batch_size, 1, num_channels_latents, height, width)
        return shape

    def maybe_pack_latents(self, latents, batch_size, batch):
        height = 2 * (
            batch.height // (self.vae_config.arch_config.vae_scale_factor * 2)
        )
        width = 2 * (batch.width // (self.vae_config.arch_config.vae_scale_factor * 2))
        num_channels_latents = self.dit_config.arch_config.in_channels // 4
        # pack latents
        return _pack_latents(latents, batch_size, num_channels_latents, height, width)

    def get_decode_scale_and_shift(self, device, dtype, vae):
        vae_arch_config = self.vae_config.arch_config
        scaling_factor = 1.0 / torch.tensor(
            vae_arch_config.latents_std, device=device
        ).view(1, vae_arch_config.z_dim, 1, 1, 1).to(device, dtype)
        shift_factor = (
            torch.tensor(vae_arch_config.latents_mean)
            .view(1, vae_arch_config.z_dim, 1, 1, 1)
            .to(device, dtype)
        )
        return scaling_factor, shift_factor

    @staticmethod
    def get_freqs_cis(img_shapes, txt_seq_lens, rotary_emb, device, dtype):
        # img_shapes: for global entire image
        img_freqs, txt_freqs = rotary_emb(img_shapes, txt_seq_lens, device=device)

        # flashinfer RoPE expects a float32 cos/sin cache concatenated on the last dim
        img_cos_half = img_freqs.real.to(dtype=torch.float32).contiguous()
        img_sin_half = img_freqs.imag.to(dtype=torch.float32).contiguous()
        txt_cos_half = txt_freqs.real.to(dtype=torch.float32).contiguous()
        txt_sin_half = txt_freqs.imag.to(dtype=torch.float32).contiguous()

        img_cos_sin_cache = torch.cat([img_cos_half, img_sin_half], dim=-1)
        txt_cos_sin_cache = torch.cat([txt_cos_half, txt_sin_half], dim=-1)
        return img_cos_sin_cache, txt_cos_sin_cache

    def _prepare_cond_kwargs(self, batch, prompt_embeds, rotary_emb, device, dtype):
        batch_size = prompt_embeds[0].shape[0]
        height = batch.height
        width = batch.width
        vae_scale_factor = self.vae_config.arch_config.vae_scale_factor

        img_shapes = [
            [
                (
                    1,
                    height // vae_scale_factor // 2,
                    width // vae_scale_factor // 2,
                )
            ]
        ] * batch_size
        txt_seq_lens = [prompt_embeds[0].shape[1]]

        freqs_cis = self.get_freqs_cis(
            img_shapes, txt_seq_lens, rotary_emb, device, dtype
        )

        img_cache, txt_cache = freqs_cis
        img_cache = shard_rotary_emb_for_sp(img_cache)
        return {
            "txt_seq_lens": txt_seq_lens,
            "freqs_cis": (img_cache, txt_cache),
        }

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return self._prepare_cond_kwargs(
            batch, batch.prompt_embeds, rotary_emb, device, dtype
        )

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return self._prepare_cond_kwargs(
            batch, batch.negative_prompt_embeds, rotary_emb, device, dtype
        )

    def post_denoising_loop(self, latents, batch):
        # unpack latents for qwen-image
        (
            latents,
            batch_size,
            channels,
            height,
            width,
        ) = self._unpad_and_unpack_latents(latents, batch)
        latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)
        return latents


@dataclass
class QwenImageEditPipelineConfig(QwenImagePipelineConfig):
    """Configuration for the QwenImageEdit pipeline."""

    task_type: ModelTaskType = ModelTaskType.I2I

    def _prepare_edit_cond_kwargs(
        self, batch, prompt_embeds, rotary_emb, device, dtype
    ):
        batch_size = batch.latents.shape[0]
        assert batch_size == 1
        height = batch.height
        width = batch.width
        image_size = batch.original_condition_image_size
        edit_width, edit_height, _ = calculate_dimensions(
            1024 * 1024, image_size[0] / image_size[1]
        )
        vae_scale_factor = self.get_vae_scale_factor()

        img_shapes = [
            [
                (
                    1,
                    height // vae_scale_factor // 2,
                    width // vae_scale_factor // 2,
                ),
                (
                    1,
                    edit_height // vae_scale_factor // 2,
                    edit_width // vae_scale_factor // 2,
                ),
            ],
        ] * batch_size
        txt_seq_lens = [prompt_embeds[0].shape[1]]
        freqs_cis = QwenImagePipelineConfig.get_freqs_cis(
            img_shapes, txt_seq_lens, rotary_emb, device, dtype
        )

        # perform sp shard on noisy image tokens
        noisy_img_seq_len = (
            1 * (height // vae_scale_factor // 2) * (width // vae_scale_factor // 2)
        )

        img_cache, txt_cache = freqs_cis
        noisy_img_cache = shard_rotary_emb_for_sp(img_cache[:noisy_img_seq_len, :])
        img_cache = torch.cat(
            [noisy_img_cache, img_cache[noisy_img_seq_len:, :]], dim=0
        ).to(device=device)
        return {
            "txt_seq_lens": txt_seq_lens,
            "freqs_cis": (img_cache, txt_cache),
        }

    def preprocess_condition_image(
        self, image, target_width, target_height, _vae_image_processor
    ):
        return resize(image, target_height, target_width, resize_mode="default"), (
            target_width,
            target_height,
        )

    def postprocess_image_latent(self, latent_condition, batch):
        batch_size = batch.batch_size
        if batch_size > latent_condition.shape[0]:
            if batch_size % latent_condition.shape[0] == 0:
                # expand init_latents for batch_size
                additional_image_per_prompt = batch_size // latent_condition.shape[0]
                image_latents = latent_condition.repeat(
                    additional_image_per_prompt, 1, 1, 1
                )
            else:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {latent_condition.shape[0]} to {batch_size} text prompts."
                )
        else:
            image_latents = latent_condition
        image_latent_height, image_latent_width = image_latents.shape[3:]
        num_channels_latents = self.dit_config.arch_config.in_channels // 4
        image_latents = _pack_latents(
            image_latents,
            batch_size,
            num_channels_latents,
            image_latent_height,
            image_latent_width,
        )

        return image_latents

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return self._prepare_edit_cond_kwargs(
            batch, batch.prompt_embeds, rotary_emb, device, dtype
        )

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return self._prepare_edit_cond_kwargs(
            batch, batch.negative_prompt_embeds, rotary_emb, device, dtype
        )

    def calculate_condition_image_size(self, image, width, height) -> tuple[int, int]:
        calculated_width, calculated_height, _ = calculate_dimensions(
            1024 * 1024, width / height
        )
        return calculated_width, calculated_height

    def slice_noise_pred(self, noise, latents):
        # remove noise over input image
        noise = noise[:, : latents.size(1)]
        return noise


CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 1024 * 1024


@dataclass
class QwenImageEditPlusPipelineConfig(QwenImageEditPipelineConfig):
    task_type: ModelTaskType = ModelTaskType.I2I

    def _get_condition_image_sizes(self, batch) -> list[tuple[int, int]]:
        image = batch.condition_image
        if not isinstance(image, list):
            image = [image]

        condition_image_sizes = []
        for img in image:
            image_width, image_height = img.size
            edit_width, edit_height, _ = calculate_dimensions(
                VAE_IMAGE_SIZE, image_width / image_height
            )
            condition_image_sizes.append((edit_width, edit_height))

        return condition_image_sizes

    def prepare_image_processor_kwargs(self, batch, neg=False) -> dict:
        prompt = batch.prompt if not neg else batch.negative_prompt
        prompt_list = [prompt] if isinstance(prompt, str) else prompt
        image_list = batch.condition_image

        prompt_template_encode = (
            "<|im_start|>system\nDescribe the key features of the input image "
            "(color, shape, size, texture, objects, background), then explain how "
            "the user's text instruction should alter or modify the image. Generate "
            "a new image that meets the user's requirements while maintaining "
            "consistency with the original input where appropriate.<|im_end|>\n"
            "<|im_start|>user\n{}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
        if isinstance(image_list, list):
            base_img_prompt = ""
            for i, img in enumerate(image_list):
                base_img_prompt += img_prompt_template.format(i + 1)
        txt = [prompt_template_encode.format(base_img_prompt + p) for p in prompt_list]
        return dict(text=txt, padding=True)

    def prepare_calculated_size(self, image):
        return self.calculate_vae_image_size(image, image.width, image.height)

    def resize_condition_image(self, images, target_width, target_height):
        if not isinstance(images, list):
            images = [images]
        new_images = []
        for img, width, height in zip(images, target_width, target_height):
            new_images.append(resize(img, height, width, resize_mode="default"))
        return new_images

    def calculate_condition_image_size(self, image, width, height) -> tuple[int, int]:
        calculated_width, calculated_height, _ = calculate_dimensions(
            CONDITION_IMAGE_SIZE, width / height
        )
        return calculated_width, calculated_height

    def calculate_vae_image_size(self, image, width, height) -> tuple[int, int]:
        calculated_width, calculated_height, _ = calculate_dimensions(
            VAE_IMAGE_SIZE, width / height
        )
        return calculated_width, calculated_height

    def preprocess_vae_image(self, batch, vae_image_processor):
        if not isinstance(batch.condition_image, list):
            batch.condition_image = [batch.condition_image]
        new_images = []
        vae_image_sizes = []
        for img in batch.condition_image:
            width, height = self.calculate_vae_image_size(img, img.width, img.height)
            new_images.append(vae_image_processor.preprocess(img, height, width))
            vae_image_sizes.append((width, height))
        batch.vae_image = new_images
        batch.vae_image_sizes = vae_image_sizes
        return batch

    def _prepare_edit_cond_kwargs(
        self, batch, prompt_embeds, rotary_emb, device, dtype
    ):
        batch_size = batch.latents.shape[0]
        assert batch_size == 1
        height = batch.height
        width = batch.width

        vae_scale_factor = self.get_vae_scale_factor()

        img_shapes = [
            [
                (1, height // vae_scale_factor // 2, width // vae_scale_factor // 2),
                *[
                    (
                        1,
                        vae_height // vae_scale_factor // 2,
                        vae_width // vae_scale_factor // 2,
                    )
                    for vae_width, vae_height in batch.vae_image_sizes
                ],
            ],
        ] * batch_size
        txt_seq_lens = [prompt_embeds[0].shape[1]]

        freqs_cis = QwenImageEditPlusPipelineConfig.get_freqs_cis(
            img_shapes, txt_seq_lens, rotary_emb, device, dtype
        )

        # perform sp shard on noisy image tokens
        noisy_img_seq_len = (
            1 * (height // vae_scale_factor // 2) * (width // vae_scale_factor // 2)
        )

        if isinstance(freqs_cis[0], torch.Tensor) and freqs_cis[0].dim() == 2:
            img_cache, txt_cache = freqs_cis
            noisy_img_cache = shard_rotary_emb_for_sp(img_cache[:noisy_img_seq_len, :])
            img_cache = torch.cat(
                [noisy_img_cache, img_cache[noisy_img_seq_len:, :]], dim=0
            ).to(device=device)
            return {
                "txt_seq_lens": txt_seq_lens,
                "freqs_cis": (img_cache, txt_cache),
                "img_shapes": img_shapes,
            }

        (img_cos, img_sin), (txt_cos, txt_sin) = freqs_cis
        noisy_img_cos = shard_rotary_emb_for_sp(img_cos[:noisy_img_seq_len, :])
        noisy_img_sin = shard_rotary_emb_for_sp(img_sin[:noisy_img_seq_len, :])

        # concat back the img_cos for input image (since it is not sp-shared later)
        img_cos = torch.cat([noisy_img_cos, img_cos[noisy_img_seq_len:, :]], dim=0).to(
            device=device
        )
        img_sin = torch.cat([noisy_img_sin, img_sin[noisy_img_seq_len:, :]], dim=0).to(
            device=device
        )

        return {
            "txt_seq_lens": txt_seq_lens,
            "freqs_cis": ((img_cos, img_sin), (txt_cos, txt_sin)),
            "img_shapes": img_shapes,
        }


@dataclass
class QwenImageEditPlus_2511_PipelineConfig(QwenImageEditPlusPipelineConfig):
    dit_config: DiTConfig = field(default_factory=QwenImageEditPlus_2511_DitConfig)


@dataclass
class QwenImageLayeredPipelineConfig(QwenImageEditPipelineConfig):
    resolution: int = 640  # TODO: allow user to set resolution
    vae_precision: str = "bf16"

    def _prepare_edit_cond_kwargs(
        self, batch, prompt_embeds, rotary_emb, device, dtype
    ):
        batch_size = batch.latents.shape[0]
        assert batch_size == 1
        height = batch.height
        width = batch.width
        image_size = batch.original_condition_image_size

        vae_scale_factor = self.get_vae_scale_factor()

        img_shapes = batch.img_shapes
        txt_seq_lens = batch.txt_seq_lens

        freqs_cis = QwenImageEditPlusPipelineConfig.get_freqs_cis(
            img_shapes, txt_seq_lens, rotary_emb, device, dtype
        )

        # perform sp shard on noisy image tokens
        noisy_img_seq_len = (
            1 * (height // vae_scale_factor // 2) * (width // vae_scale_factor // 2)
        )

        img_cache, txt_cache = freqs_cis
        noisy_img_cache = shard_rotary_emb_for_sp(img_cache[:noisy_img_seq_len, :])
        img_cache = torch.cat(
            [noisy_img_cache, img_cache[noisy_img_seq_len:, :]], dim=0
        ).to(device=device)

        return {
            "txt_seq_lens": txt_seq_lens,
            "img_shapes": img_shapes,
            "freqs_cis": (img_cache, txt_cache),
            "additional_t_cond": torch.tensor([0], device=device, dtype=torch.long),
        }

    def _unpad_and_unpack_latents(self, latents, batch):
        vae_scale_factor = self.vae_config.arch_config.vae_scale_factor
        channels = self.dit_config.arch_config.in_channels
        batch_size = latents.shape[0]
        layers = batch.num_frames

        height = 2 * (int(batch.height) // (vae_scale_factor * 2))
        width = 2 * (int(batch.width) // (vae_scale_factor * 2))

        latents = maybe_unpad_latents(latents, batch)
        latents = latents.view(
            batch_size, layers + 1, height // 2, width // 2, channels // 4, 2, 2
        )
        latents = latents.permute(0, 1, 4, 2, 5, 3, 6)

        latents = latents.reshape(
            batch_size, layers + 1, channels // (2 * 2), height, width
        )
        latents = latents.permute(0, 2, 1, 3, 4)  # (b, c, f, h, w)
        return latents, batch_size, channels, height, width

    def allow_set_num_frames(self):
        return True

    def post_denoising_loop(self, latents, batch):
        # unpack latents for qwen-image
        (
            latents,
            batch_size,
            channels,
            height,
            width,
        ) = self._unpad_and_unpack_latents(latents, batch)
        b, c, f, h, w = latents.shape
        latents = latents[:, :, 1:]  # remove the first frame as it is the origin input
        latents = latents.permute(0, 2, 1, 3, 4).view(-1, c, 1, h, w)
        # latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)
        return latents
