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
from sglang.multimodal_gen.configs.pipeline_configs.base import TextConditioningOutput
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.component_manager import ComponentUse
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


def stack_tensors(name: str, tensors: list[torch.Tensor]) -> torch.Tensor:
    base_shape = list(tensors[0].shape)
    for tensor in tensors[1:]:
        if list(tensor.shape) != base_shape:
            raise ValueError(
                f"Cannot stack {name} with differing shapes: {[list(t.shape) for t in tensors]}"
            )
    return torch.stack(tensors, dim=0)


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
        "prompt_embeds_mask",
        "negative_prompt_embeds_mask",
        "prompt_seq_lens",
        "negative_prompt_seq_lens",
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
        self._negative_text_cache_key = None
        self._negative_text_cache_value = None

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        return [
            ComponentUse(
                stage_name=stage_name,
                component_name="text_encoder" if i == 0 else f"text_encoder_{i + 1}",
                preferred_ready_after_request=i == 0,
            )
            for i in range(len(self.text_encoders))
        ]

    def get_or_compute_negative_text_embedding(
        self, batch: Req, server_args: ServerArgs, all_indices: list[int]
    ):
        negative_cache_key = self._build_negative_text_cache_key(
            batch, server_args, all_indices
        )
        use_negative_cache = not batch.is_warmup
        cached_negative = None
        if use_negative_cache:
            cached_negative = (
                self._negative_text_cache_value
                if self._negative_text_cache_key == negative_cache_key
                else None
            )
        if cached_negative is None:
            (
                neg_embeds_list,
                neg_masks_list,
                neg_pooler_embeds_list,
                neg_embeds_masks_list,
                neg_seq_lens_list,
            ) = self.encode_text(
                batch.negative_prompt,
                server_args,
                encoder_index=all_indices,
                return_attention_mask=True,
            )

            if use_negative_cache:
                self._negative_text_cache_key = negative_cache_key
                self._negative_text_cache_value = (
                    tuple(neg_embeds_list),
                    tuple(neg_masks_list),
                    tuple(neg_pooler_embeds_list),
                    tuple(neg_embeds_masks_list),
                    tuple(neg_seq_lens_list),
                )
        else:
            (
                neg_embeds_list,
                neg_masks_list,
                neg_pooler_embeds_list,
                neg_embeds_masks_list,
                neg_seq_lens_list,
            ) = cached_negative
        return (
            neg_embeds_list,
            neg_masks_list,
            neg_pooler_embeds_list,
            neg_embeds_masks_list,
            neg_seq_lens_list,
        )

    def _build_negative_text_cache_key(
        self, batch: Req, server_args: ServerArgs, encoder_indices: list[int]
    ):
        # Negative text encoding changes when the template or max length changes,
        # even if the visible negative prompt string is the same.
        return (
            server_args.pipeline_class_name,
            tuple(encoder_indices),
            self.freeze_for_dedup(batch.negative_prompt),
            self.freeze_for_dedup(batch.prompt_template),
            batch.max_sequence_length,
        )

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

        # Get max_sequence_length from batch if available
        max_seq_length = getattr(batch, "max_sequence_length", None)

        (
            prompt_embeds_list,
            prompt_masks_list,
            pooler_embeds_list,
            prompt_embeds_masks_list,
            prompt_seq_lens_list,
        ) = self.encode_text(
            prompt_text,
            server_args,
            encoder_index=all_indices,
            return_attention_mask=True,
            max_length=max_seq_length,
        )

        for pe in prompt_embeds_list:
            batch.prompt_embeds.append(pe)

        for pe in pooler_embeds_list:
            batch.pooled_embeds.append(pe)

        if batch.prompt_attention_mask is None:
            batch.prompt_attention_mask = []
            for am in prompt_masks_list:
                batch.prompt_attention_mask.append(am)

        batch.prompt_embeds_mask = []
        batch.prompt_seq_lens = []
        for mask in prompt_embeds_masks_list:
            batch.prompt_embeds_mask.append(mask)
        for seq_lens in prompt_seq_lens_list:
            batch.prompt_seq_lens.append(seq_lens)

        # Encode negative prompt if CFG is enabled
        if batch.do_classifier_free_guidance:
            assert isinstance(batch.negative_prompt, str)
            (
                neg_embeds_list,
                neg_masks_list,
                neg_pooler_embeds_list,
                neg_embeds_masks_list,
                neg_seq_lens_list,
            ) = self.get_or_compute_negative_text_embedding(
                batch, server_args, all_indices
            )

            assert batch.negative_prompt_embeds is not None

            # A single negative prompt can be shared across positive prompts.
            target_batch_sizes = [pe.shape[0] for pe in prompt_embeds_list]

            def align_negative_batch_dim(
                tensor: torch.Tensor, target_batch: int, name: str
            ) -> torch.Tensor:
                if tensor.shape[0] == target_batch:
                    return tensor
                if tensor.shape[0] == 1 and target_batch > 1:
                    return tensor.expand(target_batch, *tensor.shape[1:])
                raise ValueError(
                    f"{name} batch dimension mismatch: got {tensor.shape[0]}, expected 1 or {target_batch}"
                )

            def align_negative_seq_lens(
                seq_lens: list[int], target_batch: int, name: str
            ) -> list[int]:
                if len(seq_lens) == target_batch:
                    return [int(x) for x in seq_lens]
                if len(seq_lens) == 1 and target_batch > 1:
                    return [int(seq_lens[0])] * target_batch
                raise ValueError(
                    f"{name} batch dimension mismatch: got {len(seq_lens)}, expected 1 or {target_batch}"
                )

            for idx, ne in enumerate(neg_embeds_list):
                target_batch = target_batch_sizes[min(idx, len(target_batch_sizes) - 1)]
                ne = align_negative_batch_dim(
                    ne, target_batch, "negative_prompt_embeds"
                )
                batch.negative_prompt_embeds.append(ne)

            for idx, pe in enumerate(neg_pooler_embeds_list):
                target_batch = target_batch_sizes[min(idx, len(target_batch_sizes) - 1)]
                pe = align_negative_batch_dim(
                    pe, target_batch, "negative_pooled_embeds"
                )
                batch.neg_pooled_embeds.append(pe)
            if batch.negative_attention_mask is None:
                batch.negative_attention_mask = []
                for idx, nm in enumerate(neg_masks_list):
                    target_batch = target_batch_sizes[
                        min(idx, len(target_batch_sizes) - 1)
                    ]
                    nm = align_negative_batch_dim(
                        nm, target_batch, "negative_attention_mask"
                    )
                    batch.negative_attention_mask.append(nm)

            batch.negative_prompt_embeds_mask = []
            batch.negative_prompt_seq_lens = []
            for idx, nm in enumerate(neg_embeds_masks_list):
                target_batch = target_batch_sizes[min(idx, len(target_batch_sizes) - 1)]
                nm = align_negative_batch_dim(
                    nm, target_batch, "negative_prompt_embeds_mask"
                )
                batch.negative_prompt_embeds_mask.append(nm)
            for idx, seq_lens in enumerate(neg_seq_lens_list):
                target_batch = target_batch_sizes[min(idx, len(target_batch_sizes) - 1)]
                batch.negative_prompt_seq_lens.append(
                    align_negative_seq_lens(
                        seq_lens, target_batch, "negative_prompt_seq_lens"
                    )
                )

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

    def _manage_text_encoder_use(self, encoder_index: int) -> None:
        manager = self._component_residency_manager
        if manager is None:
            return
        component_name = (
            "text_encoder"
            if encoder_index == 0
            else f"text_encoder_{encoder_index + 1}"
        )
        use = self._declared_component_use(component_name=component_name)
        # TODO: Keep this begin-only interval until manager supports explicit
        # declared-use interval grouping. Wrapping each encoder call separately
        # can offload between positive and negative prompt encoding.
        manager.before_use(use)

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
            - list: (embeds, pooler_outputs) or
              (embeds, attention_masks, pooler_outputs, embeds_masks, seq_lens)
            - dict: Dict[str, Tensor] or (Dict[str, Tensor], Dict[str, Tensor])
            - stack: Tensor of shape [num_encoders, ...] or a tuple with stacked
              attention masks

            `embeds_masks` and `seq_lens` are aligned with postprocessed
            embeddings for variable-length text conditioning.
        """

        assert len(self.tokenizers) == len(self.text_encoders)
        assert len(self.text_encoders) == len(
            server_args.pipeline_config.text_encoder_configs
        )

        # Resolve selection into indices
        if encoder_index is None:
            indices: list[int] = [0]
        elif isinstance(encoder_index, int):
            indices = [encoder_index]
        else:
            indices = list(encoder_index)

        # Validate indices are within range
        num_encoders = len(self.text_encoders)
        for idx in indices:
            if idx < 0 or idx >= num_encoders:
                raise IndexError(
                    f"encoder index {idx} out of range [0, {num_encoders - 1}]"
                )

        # Normalize input to list[str]
        assert isinstance(text, str | list)
        if isinstance(text, str):
            texts: list[str] = [text]
        else:
            texts = text

        embeds_list: list[torch.Tensor] = []
        pooled_embeds_list: list[torch.Tensor] = []

        attn_masks_list: list[torch.Tensor | None] = []
        embeds_masks_list: list[torch.Tensor] = []
        seq_lens_list: list[list[int]] = []

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
            # Pass max_length to tokenizer if specified in the request. Flux v1 encoder 0
            # is CLIP with a fixed 77-token context; overriding breaks tokenization.
            is_flux_v1 = server_args.pipeline_config.is_flux_v1()
            if max_length is not None and not (is_flux_v1 and i == 0):
                tok_kwargs["max_length"] = max_length

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
            self._manage_text_encoder_use(i)
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
            postprocess_result = postprocess_func(
                outputs, text_inputs, **postprocess_kwargs
            )
            prompt_embeds_mask = None
            prompt_seq_lens = None
            if isinstance(postprocess_result, TextConditioningOutput):
                prompt_embeds = postprocess_result.prompt_embeds
                prompt_embeds_mask = postprocess_result.prompt_embeds_mask
                prompt_seq_lens = postprocess_result.prompt_seq_lens
            elif isinstance(postprocess_result, tuple):
                if len(postprocess_result) != 2:
                    raise ValueError(
                        "Text postprocess tuple output must be (prompt_embeds, prompt_embeds_mask)"
                    )
                prompt_embeds, prompt_embeds_mask = postprocess_result
            else:
                prompt_embeds = postprocess_result

            if dtype is not None:
                prompt_embeds = prompt_embeds.to(device=target_device, dtype=dtype)
            else:
                prompt_embeds = prompt_embeds.to(device=target_device)

            if prompt_embeds_mask is not None:
                prompt_embeds_mask = prompt_embeds_mask.to(
                    device=target_device, dtype=torch.bool
                )

            embeds_list.append(prompt_embeds)

            pooled_output = server_args.pipeline_config.get_text_encoder_pooler_output(
                outputs, i
            )
            if pooled_output is not None:
                pooled_embeds_list.append(pooled_output.to(device=target_device))

            if return_attention_mask:
                if prompt_embeds_mask is not None:
                    mask_to_store = prompt_embeds_mask.to(
                        device=target_device,
                        dtype=(
                            attention_mask.dtype
                            if attention_mask is not None
                            else torch.long
                        ),
                    )
                elif attention_mask is not None and list(attention_mask.shape) == list(
                    prompt_embeds.shape[:2]
                ):
                    mask_to_store = attention_mask.to(device=target_device)
                else:
                    mask_to_store = torch.ones(
                        prompt_embeds.shape[:2],
                        device=target_device,
                        dtype=(
                            attention_mask.dtype
                            if attention_mask is not None
                            else torch.long
                        ),
                    )
                attn_masks_list.append(mask_to_store)

                embeds_mask = prompt_embeds_mask
                if embeds_mask is None:
                    embeds_mask = (
                        server_args.pipeline_config.build_text_conditioning_mask(
                            text_inputs,
                            attention_mask,
                            prompt_embeds,
                            i,
                        )
                    )
                embeds_masks_list.append(embeds_mask)
                if prompt_seq_lens is not None:
                    seq_lens_list.append([int(x) for x in prompt_seq_lens])
                elif embeds_mask is not None:
                    seq_lens_list.append(
                        server_args.pipeline_config.seq_lens_from_text_conditioning_mask(
                            embeds_mask
                        )
                    )
                elif prompt_embeds.ndim == 2:
                    seq_lens_list.append([int(prompt_embeds.shape[0])])
                else:
                    seq_lens_list.append(
                        [int(prompt_embeds.shape[1])] * int(prompt_embeds.shape[0])
                    )

        # Shape results according to return_type
        if return_type == "list":
            if return_attention_mask:
                return (
                    embeds_list,
                    attn_masks_list,
                    pooled_embeds_list,
                    embeds_masks_list,
                    seq_lens_list,
                )
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
        stacked_embeds = stack_tensors("embeddings", embeds_list)
        if return_attention_mask:
            stackable_masks = [
                (
                    mask
                    if mask is not None
                    else torch.ones(
                        embed.shape[:2], device=embed.device, dtype=torch.long
                    )
                )
                for embed, mask in zip(embeds_list, attn_masks_list, strict=True)
            ]
            stacked_masks = stack_tensors("attention masks", stackable_masks)
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
