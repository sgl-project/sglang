# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
from dataclasses import dataclass, field
from typing import Callable

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.zimage import ZImageDitConfig
from sglang.multimodal_gen.configs.models.encoders import (
    BaseEncoderOutput,
    TextEncoderConfig,
)
from sglang.multimodal_gen.configs.models.vaes.flux import FluxVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ImagePipelineConfig,
    ModelTaskType,
    PipelineConfig,
)


def zimage_preprocess_text(prompt: str):
    messages = [
        {"role": "user", "content": prompt},
    ]
    return messages


def zimage_postprocess_text(outputs: BaseEncoderOutput, _text_inputs) -> torch.Tensor:
    device = outputs.hidden_states[-2].device
    prompt_mask = _text_inputs.attention_mask.to(device).bool()
    return outputs.hidden_states[-2][0][prompt_mask[0]]


class TransformersModelConfig(EncoderConfig):
    tokenizer_kwargs: dict = field(default_factory=lambda: {})


@dataclass
class ZImagePipelineConfig(ImagePipelineConfig):

    should_use_guidance: bool = False
    task_type: ModelTaskType = ModelTaskType.T2I

    dit_config: DiTConfig = field(default_factory=ZImageDitConfig)
    vae_config: VAEConfig = field(default_factory=FluxVAEConfig)
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (TextEncoderConfig(),)
    )

    preprocess_text_funcs: tuple[Callable, ...] = field(
        default_factory=lambda: (zimage_preprocess_text,)
    )

    postprocess_text_funcs: tuple[Callable, ...] = field(
        default_factory=lambda: (zimage_postprocess_text,)
    )

    def tokenize_prompt(self, prompts: list[str], tokenizer, tok_kwargs) -> dict:
        # flatten to 1-d list
        inputs = tokenizer.apply_chat_template(
            prompts,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
            padding="max_length",
            max_length=512,  # TODO (yhyang201): set max length according to config
            truncation=True,
            return_tensors="pt",
            return_dict=True,
        )
        return inputs

    def shard_latents_for_sp(self, batch, latents):
        if latents.dim() == 5:
            return PipelineConfig.shard_latents_for_sp(self, batch, latents)
        return super().shard_latents_for_sp(batch, latents)

    def gather_latents_for_sp(self, latents):
        if latents.dim() == 5:
            return PipelineConfig.gather_latents_for_sp(self, latents)
        return super().gather_latents_for_sp(latents)

    def post_denoising_loop(self, latents, batch):
        bs, channels, num_frames, height, width = latents.shape
        raw_latent_shape = getattr(batch, "raw_latent_shape", None)
        if raw_latent_shape is not None and num_frames > raw_latent_shape[2]:
            latents = latents[:, :, : raw_latent_shape[2], :, :]
            num_frames = raw_latent_shape[2]
        if num_frames != 1:
            return latents[:, :, 0, :, :]
        return latents.view(bs, channels, height, width)

    def get_freqs_cis(self, prompt_embeds, width, height, device, rotary_emb, batch):
        def create_coordinate_grid(size, start=None, device=None):
            if start is None:
                start = (0 for _ in size)

            axes = [
                torch.arange(x0, x0 + span, dtype=torch.int32, device=device)
                for x0, span in zip(start, size)
            ]
            grids = torch.meshgrid(axes, indexing="ij")
            return torch.stack(grids, dim=-1)

        PATCH_SIZE = 2
        F_PATCH_SIZE = 1
        SEQ_MULTI_OF = 32

        cap_ori_len = prompt_embeds.size(0)
        cap_padding_len = (-cap_ori_len) % SEQ_MULTI_OF
        cap_padded_pos_ids = create_coordinate_grid(
            size=(cap_ori_len + cap_padding_len, 1, 1),
            start=(1, 0, 0),
            device=device,
        ).flatten(0, 2)

        C = self.dit_config.num_channels_latents
        F = 1
        H = height // self.vae_config.arch_config.spatial_compression_ratio
        W = width // self.vae_config.arch_config.spatial_compression_ratio

        pH, pW = PATCH_SIZE, PATCH_SIZE
        pF = F_PATCH_SIZE
        F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW
        image_ori_len = F_tokens * H_tokens * W_tokens
        image_padding_len = (-image_ori_len) % SEQ_MULTI_OF

        image_ori_pos_ids = create_coordinate_grid(
            size=(F_tokens, H_tokens, W_tokens),
            start=(cap_ori_len + cap_padding_len + 1, 0, 0),
            device=device,
        ).flatten(0, 2)
        image_padding_pos_ids = (
            create_coordinate_grid(
                size=(1, 1, 1),
                start=(0, 0, 0),
                device=device,
            )
            .flatten(0, 2)
            .repeat(image_padding_len, 1)
        )
        image_padded_pos_ids = torch.cat(
            [image_ori_pos_ids, image_padding_pos_ids], dim=0
        )
        cap_freqs_cis = rotary_emb(cap_padded_pos_ids)
        x_freqs_cis = rotary_emb(image_padded_pos_ids)
        return (cap_freqs_cis, x_freqs_cis)

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {
            "freqs_cis": self.get_freqs_cis(
                batch.prompt_embeds[0],
                batch.width,
                batch.height,
                device,
                rotary_emb,
                batch,
            ),
        }


def zimage_omni_preprocess_text(prompt: str):
    # TODO: single image only
    #
    # elif num_condition_images > 0:
    #  ....

    return "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"


def zimage_omni_postprocess_text(
    outputs: BaseEncoderOutput, _text_inputs
) -> torch.Tensor:
    """
    # TODO: debug note, remove later
    Basic mode:
        single text embeds
        which assert batch_size == 1
        torch.tensor
    Omni mode:
        Omni mode text embedding require a fregment pattern
        which break a single text into fregments(split by <|vision_start|>).
        number of fregments is num_image + 2
        mask flatten embeds and reconstruct batch into
        List[List[torch.Tensor]]
    Returns:
        (torch.tensor | List[List[torch.Tensor]])
            torch.tensor: single batch embed
            List[List[torch.Tensor]]: single batch fregs of embeds.
            where len(d) == batch size. len(d[bid]) == num_image + 2
    """

    prompt_list_lengths = getattr(_text_inputs, "prompt_list_lengths", None)
    if prompt_list_lengths is None:
        # Basic mode
        device = outputs.hidden_states[-2].device
        prompt_mask = _text_inputs.attention_mask.to(device).bool()
        embeds = outputs.hidden_states[-2][0][prompt_mask[0]]
    else:
        # Omni mode
        # from flatten to batching
        # List[List[torch.Tensor]]

        device = outputs.hidden_states[-2].device
        embeddings_list = []
        start_idx = 0
        for i in range(len(prompt_list_lengths)):
            batch_embeddings = []
            end_idx = start_idx + prompt_list_lengths[i]
            prompt_embeds = outputs.hidden_states[-2]
            prompt_masks = _text_inputs.attention_mask.to(device).bool()
            for j in range(start_idx, end_idx):
                batch_embeddings.append(prompt_embeds[j][prompt_masks[j]])
            embeddings_list.append(batch_embeddings)
            start_idx = end_idx
        # TODO: hard code debug.
        embeds = embeddings_list

    return embeds


@dataclass
class ZImageOmniPipelineConfig(ZImagePipelineConfig):
    preprocess_text_funcs: tuple[Callable, ...] = field(
        default_factory=lambda: (zimage_omni_preprocess_text,)
    )

    postprocess_text_funcs: tuple[Callable, ...] = field(
        default_factory=lambda: (zimage_omni_postprocess_text,)
    )

    def tokenize_prompt(self, prompts, tokenizer, tok_kwargs) -> dict:
        """
        template was inject in preprocess, no apply_chat_template now.
        """

        # TODO: 2d list for omni mode
        # where
        # dim0 = batch size
        # dim1 = sequence-item len
        if isinstance(prompts, str):
            prompts = [[prompts]]
        elif isinstance(prompts, list) and isinstance(prompts[0], str):
            prompts = [prompts]
        elif (
            isinstance(prompts, list)
            and isinstance(prompts[0], list)
            and isinstance(prompts[0][0], str)
        ):
            pass
        else:
            raise NotImplementedError

        # all batch flattened
        flattened_prompt = []
        prompt_list_lengths = []

        # do flatten and record metadata
        for i in range(len(prompts)):
            # record freg numbers
            prompt_list_lengths.append(len(prompts[i]))
            # NOTE: all batch flattened
            flattened_prompt.extend(prompts[i])

        inputs = tokenizer(
            flattened_prompt,
            padding="max_length",
            # TODO: review ?
            max_length=512,  # TODO (yhyang201): set max length according to config
            truncation=True,
            return_tensors="pt",
        )

        # TODO: hack
        # prompt_list_lengths is used to reconstruct flattened_prompt into batching
        inputs.prompt_list_lengths = prompt_list_lengths

        return inputs

    # TODO: hack
    # pos and freq is online compute
    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        pos_cond_kwargs = {}
        if (
            batch.condition_siglip_embeds is not None
            and batch.condition_latents is not None
        ):
            assert len(batch.condition_siglip_embeds) == 1, "Single batch only for now."
            assert len(batch.condition_latents) == 1, "Single batch only for now."

            current_batch_size = len(batch.condition_latents)

            condition_latents_model_input = batch.condition_latents
            # Create noise mask: 0 for condition images (clean), 1 for target image (noisy)
            image_noise_mask = [
                [0] * len(condition_latents_model_input[i]) + [1]
                for i in range(current_batch_size)
            ]

            pos_cond_kwargs["siglip_feats"] = batch.condition_siglip_embeds
            pos_cond_kwargs["image_noise_mask"] = image_noise_mask

        return pos_cond_kwargs
