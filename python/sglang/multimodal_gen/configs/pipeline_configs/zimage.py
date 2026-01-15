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
    """
    Simply return, delay process in tokenize_prompt
        since we need a `self` object to take metadatas
    """
    return prompt


def zimage_omni_postprocess_text(
    outputs: BaseEncoderOutput, _text_inputs
) -> torch.Tensor:
    """
    assert batch_size == 1

    Basic mode:
        single text embeds
        torch.tensor
    Omni mode:
        Omni mode break a prompt_str into list of slice.
        which will be concat together cat([torch.tensor, ...])

    Returns:
        torch.tensor: a flatten tensor for single batch and concat all split prompts.
    """

    prompt_list_lengths = getattr(_text_inputs, "prompt_list_lengths", None)
    if prompt_list_lengths is None:
        # Basic mode
        device = outputs.hidden_states[-2].device
        prompt_mask = _text_inputs.attention_mask.to(device).bool()
        embeds = outputs.hidden_states[-2][0][prompt_mask[0]]
        raise NotImplementedError("useless, which equals to else case below")
    else:
        assert len(prompt_list_lengths) == 1, "Single batch only."

        device = outputs.hidden_states[-2].device
        embeddings_list = []
        start_idx = 0
        batch_embeddings = []
        end_idx = start_idx + prompt_list_lengths[0]  # single batch
        prompt_embeds = outputs.hidden_states[-2]
        prompt_masks = _text_inputs.attention_mask.to(device).bool()
        for j in range(start_idx, end_idx):
            batch_embeddings.append(prompt_embeds[j][prompt_masks[j]])
        embeddings_list.append(batch_embeddings)
        # assert single batch
        embeds = torch.cat(embeddings_list[0], dim=0)

    return embeds


@dataclass
class ZImageOmniPipelineConfig(ZImagePipelineConfig):
    preprocess_text_funcs: tuple[Callable, ...] = field(
        default_factory=lambda: (zimage_omni_preprocess_text,)
    )

    postprocess_text_funcs: tuple[Callable, ...] = field(
        default_factory=lambda: (zimage_omni_postprocess_text,)
    )

    # TODO: review
    # ugly hack
    # pos token_lens, neg token_lens, pos token_lens, neg token_lens
    # token_lens[0] for pos token_lens
    # token_lens[1] for neg token_lens
    # maybe bug in serving case
    token_lens = []

    def _apply_zimage_omni_template(
        self, prompts: list[str], num_condition_images: int
    ):
        """
        Args:
            prompts (list[str]): 1d list of strings
        Returns
            processed_text_list (list[list[str]]): 2d list of strings
                a single prompt_str was break into list of strings, split by <|vision_start|>
        """
        processed_text_list: list[list[str]] = []
        for prompt_str in prompts:
            if num_condition_images == 0:
                prompt_str = [
                    "<|im_start|>user\n"
                    + prompt_str
                    + "<|im_end|>\n<|im_start|>assistant\n"
                ]
                processed_text_list.append(prompt_str)
            else:
                prompt_list = ["<|im_start|>user\n<|vision_start|>"]
                prompt_list += ["<|vision_end|><|vision_start|>"] * (
                    num_condition_images - 1
                )
                prompt_list += [
                    "<|vision_end|>"
                    + prompt_str
                    + "<|im_end|>\n<|im_start|>assistant\n<|vision_start|>"
                ]
                prompt_list += ["<|vision_end|><|im_end|>"]
                processed_text_list.append(prompt_list)
        return processed_text_list

    def tokenize_prompt(self, prompts, tokenizer, tok_kwargs) -> dict:
        """
        template was inject in preprocess, no apply_chat_template now.
        """
        prompts = self._apply_zimage_omni_template(
            prompts, tok_kwargs.get("num_condition_images", 0)
        )
        assert (
            isinstance(prompts, list)
            and isinstance(prompts[0], list)
            and isinstance(prompts[0][0], str)
        ), "Process type mismatch."

        # all batch flattened
        flattened_prompt = []
        prompt_list_lengths = []

        # do flatten and record metadata
        for i in range(len(prompts)):
            # record freg numbers
            prompt_list_lengths.append(len(prompts[i]))
            # all batch flattened prompts
            flattened_prompt.extend(prompts[i])

        inputs = tokenizer(
            flattened_prompt,
            padding="max_length",
            max_length=512,  # TODO (yhyang201): set max length according to config
            truncation=True,
            return_tensors="pt",
        )

        # TODO: hack
        # prompt_list_lengths is used to reconstruct flattened_prompt into batching
        inputs.prompt_list_lengths = prompt_list_lengths
        token_lens = inputs.attention_mask.sum(dim=-1).tolist()

        self.token_lens.append(token_lens)

        return inputs

    # TODO: hack
    # pos and freq is online compute
    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        pos_cond_kwargs = {
            "condition_latents": batch.condition_latents,
            "token_lens": self.token_lens[0],
        }
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

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        neg_cond_kwargs = {
            "condition_latents": batch.negative_condition_latents,
            "token_lens": self.token_lens[1],
        }
        if (
            batch.negative_condition_siglip_embeds is not None
            and batch.negative_condition_latents is not None
        ):
            assert (
                len(batch.negative_condition_siglip_embeds) == 1
            ), "Single batch only for now."
            assert (
                len(batch.negative_condition_latents) == 1
            ), "Single batch only for now."

            current_batch_size = len(batch.negative_condition_latents)

            condition_latents_model_input = batch.negative_condition_latents
            # Create noise mask: 0 for condition images (clean), 1 for target image (noisy)
            image_noise_mask = [
                [0] * len(condition_latents_model_input[i]) + [1]
                for i in range(current_batch_size)
            ]

            neg_cond_kwargs["siglip_feats"] = batch.negative_condition_siglip_embeds
            neg_cond_kwargs["image_noise_mask"] = image_noise_mask

        return neg_cond_kwargs
