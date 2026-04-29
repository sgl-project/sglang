# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Prompt encoding stages for diffusion pipelines.

This module contains implementations of prompt encoding stages for diffusion pipelines.
"""

import inspect
from dataclasses import dataclass
from typing import Any

import torch

from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class TextEncodingFingerprint:
    prompt: Any
    negative_prompt: Any
    do_classifier_free_guidance: bool
    prompt_template: Any
    max_sequence_length: int | None


class TextEncodingStage(PipelineStage):
    """
    Stage for encoding text prompts into embeddings for diffusion models.

    This stage handles the encoding of text prompts into the embedding space
    expected by the diffusion model.
    """

    deduplicated_output_fields = (
        "prompt_embeds",
        "negative_prompt_embeds",
        "prompt_attention_mask",
        "negative_attention_mask",
        "pooled_embeds",
        "neg_pooled_embeds",
        "clip_embedding_pos",
        "clip_embedding_neg",
        "is_prompt_processed",
    )

    def __init__(self, text_encoders, tokenizers) -> None:
        """
        Initialize the prompt encoding stage.

        """
        super().__init__()
        self.tokenizers = tokenizers
        self.text_encoders = text_encoders

    @torch.no_grad()
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Encode the prompt into text encoder hidden states.
        """
        assert len(self.tokenizers) == len(self.text_encoders)
        assert len(self.text_encoders) == len(
            server_args.pipeline_config.text_encoder_configs
        )

        # Encode positive prompt with all available encoders
        assert batch.prompt is not None
        prompt_text: str | list[str] = batch.prompt

        all_indices: list[int] = list(range(len(self.text_encoders)))

        prompt_embeds_list, prompt_masks_list, pooler_embeds_list = self.encode_text(
            prompt_text,
            server_args,
            encoder_index=all_indices,
            return_attention_mask=True,
        )

        for pe in prompt_embeds_list:
            batch.prompt_embeds.append(pe)

        for pe in pooler_embeds_list:
            batch.pooled_embeds.append(pe)

        if batch.prompt_attention_mask is None:
            batch.prompt_attention_mask = []
            for am in prompt_masks_list:
                batch.prompt_attention_mask.append(am)

        # Encode negative prompt if CFG is enabled
        if batch.do_classifier_free_guidance:
            assert isinstance(batch.negative_prompt, str)
            neg_embeds_list, neg_masks_list, neg_pooler_embeds_list = self.encode_text(
                batch.negative_prompt,
                server_args,
                encoder_index=all_indices,
                return_attention_mask=True,
            )

            assert batch.negative_prompt_embeds is not None

            for ne in neg_embeds_list:
                batch.negative_prompt_embeds.append(ne)

            for pe in neg_pooler_embeds_list:
                batch.neg_pooled_embeds.append(pe)
            if batch.negative_attention_mask is None:
                batch.negative_attention_mask = []
                for nm in neg_masks_list:
                    batch.negative_attention_mask.append(nm)

        return batch

    def build_dedup_fingerprint(
        self, batch: Req, server_args: ServerArgs
    ) -> TextEncodingFingerprint:
        return TextEncodingFingerprint(
            prompt=self.freeze_for_dedup(batch.prompt),
            negative_prompt=self.freeze_for_dedup(batch.negative_prompt),
            do_classifier_free_guidance=bool(batch.do_classifier_free_guidance),
            prompt_template=self.freeze_for_dedup(batch.prompt_template),
            max_sequence_length=batch.max_sequence_length,
        )

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify text encoding stage inputs."""
        result = VerificationResult()
        result.add_check("prompt", batch.prompt, V.string_or_list_strings)
        result.add_check(
            "negative_prompt",
            batch.negative_prompt,
            lambda x: not batch.do_classifier_free_guidance or V.string_not_none(x),
        )
        result.add_check(
            "do_classifier_free_guidance",
            batch.do_classifier_free_guidance,
            V.bool_value,
        )
        result.add_check("prompt_embeds", batch.prompt_embeds, V.is_list)
        result.add_check(
            "negative_prompt_embeds", batch.negative_prompt_embeds, V.none_or_list
        )
        return result

    def prepare_tokenizer_kwargs(self, tokenizer_kwargs, **kwargs):
        tok_kwargs = tokenizer_kwargs | kwargs

        return tok_kwargs

    def _forward_text_encoder(self, text_encoder, encoder_forward_kwargs):
        if not getattr(text_encoder, "uses_sglang_forward_context", True):
            return text_encoder(**encoder_forward_kwargs)

        with set_forward_context(current_timestep=0, attn_metadata=None):
            return text_encoder(**encoder_forward_kwargs)

    @torch.no_grad()
    def encode_text(
        self,
        text: str | list[str],
        server_args: ServerArgs,
        encoder_index: int | list[int] | None = None,
        return_attention_mask: bool = False,
        return_type: str = "list",  # one of: "list", "dict", "stack"
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        max_length: int | None = None,
        truncation: bool | None = None,
        padding: bool | str | None = None,
        return_overflowing_tokens=None,
        return_length=None,
    ):
        """
        Encode plain text using selected text encoder(s) and return embeddings.

        Args:
            text: A single string or a list of strings to encode.
            server_args: The inference arguments providing pipeline config,
                including tokenizer and encoder settings, preprocess and postprocess
                functions.
            encoder_index: Encoder selector by index. Accepts an int or list of ints.
            return_attention_mask: If True, also return attention masks for each
                selected encoder.
            return_type: "list" (default) returns a list aligned with selection;
                "dict" returns a dict keyed by encoder index as a string; "stack" stacks along a
                new first dimension (requires matching shapes).
            device: Optional device override for inputs; defaults to local torch device.
            dtype: Optional dtype to cast returned embeddings to.
            max_length: Optional per-call tokenizer override.
            truncation: Optional per-call tokenizer override.
            padding: Optional per-call tokenizer override.

        Returns:
            Depending on return_type and return_attention_mask:
            - list: List[Tensor] or (List[Tensor], List[Tensor])
            - dict: Dict[str, Tensor] or (Dict[str, Tensor], Dict[str, Tensor])
            - stack: Tensor of shape [num_encoders, ...] or a tuple with stacked
              attention masks
        """

        assert len(self.tokenizers) == len(self.text_encoders)
        assert len(self.text_encoders) == len(
            server_args.pipeline_config.text_encoder_configs
        )

        # Resolve selection into indices
        encoder_cfgs = server_args.pipeline_config.text_encoder_configs
        if encoder_index is None:
            indices: list[int] = [0]
        elif isinstance(encoder_index, int):
            indices = [encoder_index]
        else:
            indices = list(encoder_index)
        # validate range
        num_encoders = len(self.text_encoders)
        for idx in indices:
            if idx < 0 or idx >= num_encoders:
                raise IndexError(
                    f"encoder index {idx} out of range [0, {num_encoders - 1}]"
                )

        # Validate indices are within range
        num_encoders = len(self.text_encoders)

        # Normalize input to list[str]
        assert isinstance(text, str | list)
        if isinstance(text, str):
            texts: list[str] = [text]
        else:
            texts = text

        embeds_list: list[torch.Tensor] = []
        pooled_embeds_list: list[torch.Tensor] = []

        attn_masks_list: list[torch.Tensor] = []

        preprocess_funcs = server_args.pipeline_config.preprocess_text_funcs
        postprocess_funcs = server_args.pipeline_config.postprocess_text_funcs
        text_encoder_extra_args = server_args.pipeline_config.text_encoder_extra_args
        encoder_cfgs = server_args.pipeline_config.text_encoder_configs

        if return_type not in ("list", "dict", "stack"):
            raise ValueError(
                f"Invalid return_type '{return_type}'. Expected one of: 'list', 'dict', 'stack'"
            )

        target_device = device if device is not None else get_local_torch_device()

        for i in indices:
            tokenizer = self.tokenizers[i]
            text_encoder = self.text_encoders[i]
            encoder_config = encoder_cfgs[i]
            preprocess_func = preprocess_funcs[i]
            postprocess_func = postprocess_funcs[i]
            text_encoder_extra_arg = (
                text_encoder_extra_args[i]
                if i < len(text_encoder_extra_args) and text_encoder_extra_args[i]
                else {}
            )

            if preprocess_func is not None:
                processed_text_list: list[str] = [
                    preprocess_func(prompt_str) for prompt_str in texts
                ]
            else:
                processed_text_list = texts

            # Prepare tokenizer args
            tok_kwargs = self.prepare_tokenizer_kwargs(
                encoder_config.tokenizer_kwargs,
                **text_encoder_extra_arg,
            )

            text_inputs: dict = server_args.pipeline_config.tokenize_prompt(
                processed_text_list, tokenizer, tok_kwargs
            ).to(target_device)

            input_ids = text_inputs["input_ids"]
            attention_mask = (
                server_args.pipeline_config.get_text_encoder_attention_mask(
                    text_inputs, i
                )
            )
            encoder_forward_kwargs = {
                "input_ids": input_ids,
                "output_hidden_states": True,
            }
            if attention_mask is not None:
                encoder_forward_kwargs["attention_mask"] = attention_mask
            if "use_cache" in inspect.signature(text_encoder.forward).parameters:
                encoder_forward_kwargs["use_cache"] = False
            outputs: BaseEncoderOutput = self._forward_text_encoder(
                text_encoder, encoder_forward_kwargs
            )
            postprocess_sig = inspect.signature(postprocess_func)

            postprocess_kwargs = {}
            if "pipeline_config" in postprocess_sig.parameters:
                # required by models like LTX
                postprocess_kwargs["pipeline_config"] = server_args.pipeline_config
            if "return_attention_mask" in postprocess_sig.parameters:
                postprocess_kwargs["return_attention_mask"] = return_attention_mask
            prompt_embeds = postprocess_func(outputs, text_inputs, **postprocess_kwargs)
            has_postprocessed_attention_mask = False
            postprocessed_attention_mask = None
            if isinstance(prompt_embeds, tuple):
                prompt_embeds, postprocessed_attention_mask = prompt_embeds
                has_postprocessed_attention_mask = True
            if dtype is not None:
                prompt_embeds = prompt_embeds.to(device=target_device, dtype=dtype)
            else:
                prompt_embeds = prompt_embeds.to(device=target_device)

            embeds_list.append(prompt_embeds)

            pooled_output = server_args.pipeline_config.get_text_encoder_pooler_output(
                outputs, i
            )
            if pooled_output is not None:
                pooled_embeds_list.append(pooled_output.to(device=target_device))

            if return_attention_mask:
                if has_postprocessed_attention_mask:
                    mask_to_store = (
                        postprocessed_attention_mask.to(device=target_device)
                        if postprocessed_attention_mask is not None
                        else None
                    )
                elif attention_mask is not None:
                    mask_to_store = attention_mask.to(device=target_device)
                else:
                    mask_to_store = torch.ones(
                        input_ids.shape[:2], device=target_device
                    )
                attn_masks_list.append(mask_to_store)

        # Shape results according to return_type
        if return_type == "list":
            if return_attention_mask:
                return embeds_list, attn_masks_list, pooled_embeds_list
            return embeds_list, pooled_embeds_list

        if return_type == "dict":
            key_strs = [str(i) for i in indices]
            embeds_dict = {k: v for k, v in zip(key_strs, embeds_list, strict=False)}
            if return_attention_mask:
                attn_dict = {
                    k: v for k, v in zip(key_strs, attn_masks_list, strict=False)
                }
                return embeds_dict, attn_dict
            return embeds_dict

        # return_type == "stack"
        # Validate shapes are compatible
        base_shape = list(embeds_list[0].shape)
        for t in embeds_list[1:]:
            if list(t.shape) != base_shape:
                raise ValueError(
                    f"Cannot stack embeddings with differing shapes: {[list(t.shape) for t in embeds_list]}"
                )
        stacked_embeds = torch.stack(embeds_list, dim=0)
        if return_attention_mask:
            base_mask_shape = list(attn_masks_list[0].shape)
            for m in attn_masks_list[1:]:
                if list(m.shape) != base_mask_shape:
                    raise ValueError(
                        f"Cannot stack attention masks with differing shapes: {[list(m.shape) for m in attn_masks_list]}"
                    )
            stacked_masks = torch.stack(attn_masks_list, dim=0)
            return stacked_embeds, stacked_masks
        return stacked_embeds

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify text encoding stage outputs."""
        result = VerificationResult()
        result.add_check(
            "prompt_embeds", batch.prompt_embeds, V.list_of_tensors_min_dims(2)
        )
        result.add_check(
            "negative_prompt_embeds",
            batch.negative_prompt_embeds,
            lambda x: not batch.do_classifier_free_guidance
            or V.list_of_tensors_with_min_dims(x, 2),
        )
        if batch.debug:
            logger.debug(f"{batch.prompt_embeds=}")
            logger.debug(f"{batch.negative_prompt_embeds=}")
        return result
