# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
import math
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

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
)
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_parallel_rank,
    get_sp_world_size,
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

    SEQ_LEN_MULTIPLE: int = 32
    PATCH_SIZE: int = 2
    F_PATCH_SIZE: int = 1

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

    @staticmethod
    def _ceil_to_multiple(x: int, m: int) -> int:
        if m <= 0:
            return x
        return int(math.ceil(x / m) * m)

    def _build_zimage_sp_plan(self, batch) -> dict:
        """Build a minimal SP plan on batch for zimage (spatial sharding + cap sharding)."""
        sp_size = get_sp_world_size()
        rank = get_sp_parallel_rank()

        raw_latent_shape = getattr(batch, "raw_latent_shape", None)
        if raw_latent_shape is not None and len(raw_latent_shape) >= 5:
            H = int(raw_latent_shape[3])
            W = int(raw_latent_shape[4])
        else:
            H = int(
                batch.height // self.vae_config.arch_config.spatial_compression_ratio
            )
            W = int(
                batch.width // self.vae_config.arch_config.spatial_compression_ratio
            )

        # Rule: shard along the larger spatial dimension (W/H), implemented via optional H/W transpose.
        # Choose the larger of H and W for sharding, so H_eff = max(H, W).
        swap_hw = W > H
        H_eff = W if swap_hw else H
        W_eff = H if swap_hw else W

        # ZImage uses PATCH_SIZE=2 for spatial patchify; shard in token space and convert back to latent rows.
        H_tok = H_eff // self.PATCH_SIZE
        W_tok = W_eff // self.PATCH_SIZE
        H_tok_pad = self._ceil_to_multiple(H_tok, sp_size)
        H_tok_local = H_tok_pad // sp_size
        h0_tok = rank * H_tok_local

        # Cap/text sharding: avoid duplicating cap tokens across ranks.
        cap_len = (
            int(batch.prompt_embeds[0].size(0))
            if getattr(batch, "prompt_embeds", None)
            else 0
        )
        cap_total = self._ceil_to_multiple(cap_len, self.SEQ_LEN_MULTIPLE * sp_size)
        cap_local = cap_total // sp_size
        cap_start = rank * cap_local

        plan = {
            "sp_size": sp_size,
            "rank": rank,
            "swap_hw": swap_hw,
            "H": H,
            "W": W,
            "H_eff": H_eff,
            "W_eff": W_eff,
            "H_tok": H_tok,
            "W_tok": W_tok,
            "H_tok_pad": H_tok_pad,
            "H_tok_local": H_tok_local,
            "h0_tok": h0_tok,
            "cap_total": cap_total,
            "cap_local": cap_local,
            "cap_start": cap_start,
        }
        batch._zimage_sp_plan = plan
        return plan

    def _get_zimage_sp_plan(self, batch) -> dict:
        plan = getattr(batch, "_zimage_sp_plan", None)
        sp_size = get_sp_world_size()
        if plan is None or plan.get("sp_size") != sp_size:
            plan = self._build_zimage_sp_plan(batch)
        return plan

    def _shard_cap(self, cap: torch.Tensor, plan: dict) -> torch.Tensor:
        """cap: [L, D] -> [cap_local, D], padded by repeating last token."""
        if plan["sp_size"] <= 1:
            return cap
        # print(f"cap shape: {cap.shape}")  # [L, 2560] for zimage-turbo
        L = cap.size(0)
        cap_total = plan["cap_total"]
        if cap_total > L:
            cap = torch.cat([cap, cap[-1:].repeat(cap_total - L, 1)], dim=0)
        start = plan["cap_start"]
        local = plan["cap_local"]
        return cap[start : start + local]

    def get_pos_prompt_embeds(self, batch):
        # Keep ZImage model signature: encoder_hidden_states is List[Tensor]
        if get_sp_world_size() <= 1:
            return batch.prompt_embeds
        plan = self._get_zimage_sp_plan(batch)
        return [self._shard_cap(batch.prompt_embeds[0], plan)]

    def shard_latents_for_sp(self, batch, latents):
        sp_size = get_sp_world_size()
        if sp_size <= 1 or latents.dim() != 5:
            return latents, False

        plan = self._get_zimage_sp_plan(batch)

        # Layout: [B, C, T, H, W]. Always shard on dim=3 by optionally swapping H/W.
        if plan["swap_hw"]:
            latents = latents.transpose(3, 4).contiguous()

        # Pad on effective-H so that H_tok is divisible by sp.
        H_eff = latents.size(3)

        H_tok = H_eff // self.PATCH_SIZE
        pad_tok = plan["H_tok_pad"] - H_tok
        pad_lat = pad_tok * self.PATCH_SIZE
        if pad_lat > 0:
            pad = latents[:, :, :, -1:, :].repeat(1, 1, 1, pad_lat, 1)
            latents = torch.cat([latents, pad], dim=3)
        h0 = plan["h0_tok"] * self.PATCH_SIZE
        h1 = (plan["h0_tok"] + plan["H_tok_local"]) * self.PATCH_SIZE
        latents = latents[:, :, :, h0:h1, :]

        batch._zimage_sp_swap_hw = plan["swap_hw"]
        return latents, True

    def gather_latents_for_sp(self, latents):
        # Gather on effective-H dim=3 (matches shard_latents_for_sp); swap-back is handled in post_denoising_loop.
        latents = latents.contiguous()
        if get_sp_world_size() <= 1 or latents.dim() != 5:
            return latents
        return sequence_model_parallel_all_gather(latents, dim=3)

    def post_denoising_loop(self, latents, batch):
        # Restore swapped H/W and crop padded spatial dims before final reshape.
        if latents.dim() == 5 and getattr(batch, "_zimage_sp_swap_hw", False):
            latents = latents.transpose(3, 4).contiguous()
        raw_latent_shape = getattr(batch, "raw_latent_shape", None)
        if raw_latent_shape is not None and latents.dim() == 5:
            latents = latents[:, :, :, : raw_latent_shape[3], : raw_latent_shape[4]]

        bs, channels, num_frames, height, width = latents.shape
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

        sp_size = get_sp_world_size()
        if sp_size > 1:
            # SP path: build local-only freqs_cis matching local cap/x.
            plan = self._get_zimage_sp_plan(batch)

            # cap (local)
            cap_pos_ids = create_coordinate_grid(
                size=(plan["cap_local"], 1, 1),
                start=(1 + plan["cap_start"], 0, 0),
                device=device,
            ).flatten(0, 2)
            cap_freqs_cis = rotary_emb(cap_pos_ids)

            # image (local, effective H-shard). Use cap_total for a stable offset across ranks/passes.
            F_tokens = 1
            H_tokens_local = plan["H_tok_local"]
            W_tokens = plan["W_tok"]
            img_pos_ids = create_coordinate_grid(
                size=(F_tokens, H_tokens_local, W_tokens),
                start=(plan["cap_total"] + 1, plan["h0_tok"], 0),
                device=device,
            ).flatten(0, 2)
            img_pad_len = (-img_pos_ids.shape[0]) % self.SEQ_LEN_MULTIPLE
            if img_pad_len:
                pad_ids = create_coordinate_grid(
                    size=(1, 1, 1), start=(0, 0, 0), device=device
                ).flatten(0, 2)
                img_pos_ids = torch.cat(
                    [img_pos_ids, pad_ids.repeat(img_pad_len, 1)], dim=0
                )
            x_freqs_cis = rotary_emb(img_pos_ids)
            return (cap_freqs_cis, x_freqs_cis)

        cap_ori_len = prompt_embeds.size(0)
        cap_padding_len = (-cap_ori_len) % self.SEQ_LEN_MULTIPLE
        cap_padded_pos_ids = create_coordinate_grid(
            size=(cap_ori_len + cap_padding_len, 1, 1),
            start=(1, 0, 0),
            device=device,
        ).flatten(0, 2)

        C = self.dit_config.num_channels_latents
        F = 1
        H = height // self.vae_config.arch_config.spatial_compression_ratio
        W = width // self.vae_config.arch_config.spatial_compression_ratio

        pH, pW = self.PATCH_SIZE, self.PATCH_SIZE
        pF = self.F_PATCH_SIZE
        F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW
        image_ori_len = F_tokens * H_tokens * W_tokens
        image_padding_len = (-image_ori_len) % self.SEQ_LEN_MULTIPLE

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

        # prompt_list_lengths is used to reconstruct flattened_prompt into batching
        inputs.prompt_list_lengths = prompt_list_lengths
        token_lens = inputs.attention_mask.sum(dim=-1).tolist()

        self.token_lens.append(token_lens)
        self.token_lens = self.token_lens[-2:]

        return inputs

    def get_freqs_cis(
        self,
        prompt_embeds: List[torch.Tensor],
        images: List[torch.Tensor],
        siglips: List[torch.Tensor],
        device,
        rotary_emb,
    ):
        def create_coordinate_grid(size, start=None, device=None):
            if start is None:
                start = (0 for _ in size)

            axes = [
                torch.arange(x0, x0 + span, dtype=torch.int32, device=device)
                for x0, span in zip(start, size)
            ]
            grids = torch.meshgrid(axes, indexing="ij")
            return torch.stack(grids, dim=-1)

        def _get_pos_ids(
            ori_len: int,
            pos_grid_size: Tuple,
            pos_start: Tuple,
            device: torch.device,
        ):
            pad_len = (-ori_len) % SEQ_MULTI_OF
            total_len = ori_len + pad_len

            # Pos IDs
            ori_pos_ids = create_coordinate_grid(
                size=pos_grid_size, start=pos_start, device=device
            ).flatten(0, 2)
            if pad_len > 0:
                pad_pos_ids = (
                    create_coordinate_grid(
                        size=(1, 1, 1), start=(0, 0, 0), device=device
                    )
                    .flatten(0, 2)
                    .repeat(pad_len, 1)
                )
                pos_ids = torch.cat([ori_pos_ids, pad_pos_ids], dim=0)
            else:
                pos_ids = ori_pos_ids

            return pos_ids

        # TODO: assert batch size == 1

        # TODO: hard code....
        PATCH_SIZE = 2
        F_PATCH_SIZE = 1
        SEQ_MULTI_OF = 32

        # cap_start_pos for cap pos ids
        cap_cu_len = 1
        # cap_end_pos + 0 for image pos ids
        # cap_end_pos + 1 for image siglip ids
        cap_end_pos = []

        image_size = []

        cap_pos_ids_list = []
        image_pos_ids_list = []
        siglip_pos_ids_list = []

        for cap_item in prompt_embeds:
            cap_padded_pos_ids = _get_pos_ids(
                len(cap_item),
                (len(cap_item) + (-len(cap_item)) % SEQ_MULTI_OF, 1, 1),
                (cap_cu_len, 0, 0),
                device,
            )
            cap_cu_len += len(cap_item)
            cap_end_pos.append(cap_cu_len)
            cap_cu_len += 2  # for image vae and siglip tokens
            cap_pos_ids_list.append(cap_padded_pos_ids)

        for j, image in enumerate(images):
            if image is not None:
                pH, pW, pF = PATCH_SIZE, PATCH_SIZE, F_PATCH_SIZE
                C, F, H, W = image.size()
                F_t, H_t, W_t = F // pF, H // pH, W // pW
                image = image.view(C, F_t, pF, H_t, pH, W_t, pW)
                image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(
                    F_t * H_t * W_t, pF * pH * pW * C
                )
                image_pos = _get_pos_ids(
                    F_t * H_t * W_t, (F_t, H_t, W_t), (cap_end_pos[j], 0, 0), device
                )
                image_size.append((F, H, W))
            else:
                image_len = SEQ_MULTI_OF
                image_pos = (
                    create_coordinate_grid((1, 1, 1), (0, 0, 0), device)
                    .flatten(0, 2)
                    .repeat(image_len, 1)
                )
                image_size.append(None)

            image_pos_ids_list.append(image_pos)

        for j, sig_item in enumerate(siglips if siglips is not None else []):
            if sig_item is not None:
                sig_H, sig_W, sig_C = sig_item.size()
                sig_flat = sig_item.permute(2, 0, 1).reshape(sig_H * sig_W, sig_C)
                sig_pos = _get_pos_ids(
                    len(sig_flat),
                    (1, sig_H, sig_W),
                    (cap_end_pos[j] + 1, 0, 0),
                    device,
                )
                # Scale position IDs to match x resolution
                if image_size[j] is not None:
                    sig_pos = sig_pos.float()
                    sig_pos[..., 1] = (
                        sig_pos[..., 1] / max(sig_H - 1, 1) * (image_size[j][1] - 1)
                    )
                    sig_pos[..., 2] = (
                        sig_pos[..., 2] / max(sig_W - 1, 1) * (image_size[j][2] - 1)
                    )
                    sig_pos = sig_pos.to(torch.int32)
            else:
                sig_len = SEQ_MULTI_OF
                sig_pos = (
                    create_coordinate_grid((1, 1, 1), (0, 0, 0), device)
                    .flatten(0, 2)
                    .repeat(sig_len, 1)
                )
            siglip_pos_ids_list.append(sig_pos)

        cap_freqs_cis = rotary_emb(torch.cat(cap_pos_ids_list, dim=0))
        x_freqs_cis = rotary_emb(torch.cat(image_pos_ids_list, dim=0))
        siglip_freqs_cis = (
            rotary_emb(torch.cat(siglip_pos_ids_list, dim=0))
            if len(siglip_pos_ids_list) != 0
            else None
        )
        return (cap_freqs_cis, x_freqs_cis, siglip_freqs_cis)

    # TODO: hack
    # pos and freq is online compute
    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        # TODO: hard code bsz1
        current_batch_size = 1
        if (
            batch.condition_siglip_embeds is not None
            and batch.condition_latents is not None
        ):
            assert len(batch.condition_siglip_embeds) == 1, "Single batch only for now."
            assert len(batch.condition_latents) == 1, "Single batch only for now."

            condition_latents_model_input = batch.condition_latents
            # Create noise mask: 0 for condition images (clean), 1 for target image (noisy)
            image_noise_mask = [
                [0] * len(condition_latents_model_input[i]) + [1]
                for i in range(current_batch_size)
            ]

            pos_cond_kwargs = {
                "condition_latents": batch.condition_latents,
                "token_lens": self.token_lens[0],
                "siglip_feats": batch.condition_siglip_embeds,
                "image_noise_mask": image_noise_mask,
            }
        else:
            image_noise_mask = [[1] for i in range(current_batch_size)]
            pos_cond_kwargs = {
                # NOTE: always omni mode in omni pipeline
                # which condition_latents is not None, [[]] instead
                "condition_latents": [[] for i in range(current_batch_size)],
                "token_lens": self.token_lens[0],
                "siglip_feats": None,
                "image_noise_mask": image_noise_mask,
            }

        encoder_hidden_states = [
            list(
                batch.prompt_embeds[0].split_with_sizes(
                    pos_cond_kwargs["token_lens"], dim=0
                )
            )
        ]
        hidden_states = [
            pos_cond_kwargs["condition_latents"][i] + [batch.latents[i]]
            for i in range(current_batch_size)
        ]
        freqs_cis = self.get_freqs_cis(
            prompt_embeds=encoder_hidden_states[0],
            images=hidden_states[0],
            siglips=(
                pos_cond_kwargs["siglip_feats"][0]
                if pos_cond_kwargs["siglip_feats"] is not None
                else None
            ),
            device=device,
            rotary_emb=rotary_emb,
        )

        pos_cond_kwargs["freqs_cis"] = freqs_cis

        return pos_cond_kwargs

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        # TODO: hard code bsz1
        current_batch_size = 1
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

            condition_latents_model_input = batch.negative_condition_latents
            # Create noise mask: 0 for condition images (clean), 1 for target image (noisy)
            image_noise_mask = [
                [0] * len(condition_latents_model_input[i]) + [1]
                for i in range(current_batch_size)
            ]

            neg_cond_kwargs = {
                "condition_latents": batch.negative_condition_latents,
                "token_lens": self.token_lens[1],
                "siglip_feats": batch.negative_condition_siglip_embeds,
                "image_noise_mask": image_noise_mask,
            }
        else:
            image_noise_mask = [[1] for i in range(current_batch_size)]
            # NOTE: always omni mode in omni pipeline
            # which condition_latents is not None, [[]] instead
            neg_cond_kwargs = {
                "condition_latents": [[] for i in range(current_batch_size)],
                "token_lens": self.token_lens[1],
                "siglip_feats": None,
                "image_noise_mask": image_noise_mask,
            }

        encoder_hidden_states = [
            list(
                batch.negative_prompt_embeds[0].split_with_sizes(
                    neg_cond_kwargs["token_lens"], dim=0
                )
            )
        ]
        hidden_states = [
            neg_cond_kwargs["condition_latents"][i] + [batch.latents[i]]
            for i in range(current_batch_size)
        ]
        freqs_cis = self.get_freqs_cis(
            prompt_embeds=encoder_hidden_states[0],
            images=hidden_states[0],
            siglips=(
                neg_cond_kwargs["siglip_feats"][0]
                if neg_cond_kwargs["siglip_feats"] is not None
                else None
            ),
            device=device,
            rotary_emb=rotary_emb,
        )

        neg_cond_kwargs["freqs_cis"] = freqs_cis

        return neg_cond_kwargs
