from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Union

from sglang.srt.environ import envs
from sglang.srt.managers.embed_types import PositionalEmbeds  # noqa: F401
from sglang.srt.managers.io_struct import (
    EmbeddingReqInput,
    GenerateReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem

logger = logging.getLogger(__name__)
from typing import Dict, List, Optional

from sglang.srt.managers.tokenizer_manager_components.multimodal_processor_owner import (
    MultimodalProcessor,
)
from sglang.srt.managers.tokenizer_manager_components.raw_tokenizer_wrapper import (
    RawTokenizerWrapper,
)
from sglang.srt.managers.tokenizer_manager_components.request_state import ReqState
from sglang.srt.managers.tokenizer_manager_components.request_validator import (
    RequestValidator,
)
from sglang.srt.managers.tokenizer_manager_components.tokenized_request_builder import (
    TokenizedRequestBuilder,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class RequestPreparerConfig:
    skip_tokenizer_init: bool
    enable_dp_attention: bool
    enable_tokenizer_batch_encode: bool
    is_generation: bool
    disable_radix_cache: bool
    is_multimodal: bool
    architectures: List[str]
    max_req_input_len: Optional[int]
    language_only: bool
    encoder_transfer_backend: str


@dataclass(frozen=True, slots=True, kw_only=True)
class RequestPreparer:
    raw_tokenizer_wrapper: RawTokenizerWrapper
    multimodal_processor: MultimodalProcessor
    request_validator: RequestValidator
    tokenized_request_builder: TokenizedRequestBuilder
    rid_to_state: Dict[str, ReqState]
    config: RequestPreparerConfig

    async def _tokenize_one_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
    ):
        """Tokenize one request."""
        # Tokenize
        input_embeds = None
        input_text = obj.text
        token_type_ids = None
        is_cross_encoder_request = (
            isinstance(obj, EmbeddingReqInput) and obj.is_cross_encoder_request
        )
        if obj.input_embeds is not None:
            if not self.config.disable_radix_cache:
                raise ValueError(
                    "input_embeds is provided while disable_radix_cache is False. "
                    "Please add `--disable-radix-cache` when you launch the server "
                    "if you want to use input_embeds as inputs."
                )
            input_embeds = obj.input_embeds
            input_ids = obj.input_ids
        elif obj.input_ids is not None:
            input_ids = obj.input_ids
        else:
            if self.tokenizer is None:
                raise ValueError(
                    "The engine initialized with skip_tokenizer_init=True cannot "
                    "accept text prompts. Please provide input_ids or re-initialize "
                    "the engine with skip_tokenizer_init=False."
                )

            # For audio-only requests (e.g., Whisper), text may be empty.
            # The multimodal processor will provide input_ids later.
            if not input_text and self.mm_processor and obj.contains_mm_input():
                # Use empty placeholder - multimodal processor will override
                input_ids = []
            else:
                input_ids, token_type_ids = (
                    await self.raw_tokenizer_wrapper._tokenize_texts(
                        input_text, is_cross_encoder_request
                    )
                )

        contains_mm_input = obj.contains_mm_input()
        is_mossvl = "MossVLForConditionalGeneration" in self.config.architectures
        should_run_mm_processor = self.mm_processor is not None and (
            contains_mm_input or is_mossvl
        )

        if should_run_mm_processor:
            if obj.image_data is not None and not isinstance(obj.image_data, list):
                obj.image_data = [obj.image_data]
            if obj.video_data is not None and not isinstance(obj.video_data, list):
                obj.video_data = [obj.video_data]
            if obj.audio_data is not None and not isinstance(obj.audio_data, list):
                obj.audio_data = [obj.audio_data]
            if contains_mm_input:
                self.request_validator._validate_mm_limits(obj)

            mm_inputs = None

            if (
                not self.config.language_only
                or self.config.encoder_transfer_backend
                in ["zmq_to_tokenizer", "mooncake"]
            ):
                if self.config.language_only:
                    mm_inputs = (
                        await self.multimodal_processor.mm_receiver.recv_mm_data(
                            request_obj=obj,
                            mm_processor=self.mm_processor,
                            prompt=(input_text or input_ids),
                            need_wait_for_mm_inputs=obj.need_wait_for_mm_inputs,
                        )
                    )
                if mm_inputs is None:
                    mm_inputs = await self.mm_processor.process_mm_data_async(
                        image_data=obj.image_data,
                        audio_data=obj.audio_data,
                        input_text=(input_text or input_ids),
                        request_obj=obj,
                        max_req_input_len=self.config.max_req_input_len,
                    )
            elif (
                self.config.language_only
                and self.config.encoder_transfer_backend == "zmq_to_scheduler"
                and not obj.need_wait_for_mm_inputs
            ):
                # In language_only mode with zmq_to_scheduler, if we didn't dispatch
                # to encoder (e.g., only one image), process locally like non-language_only mode
                mm_inputs = await self.mm_processor.process_mm_data_async(
                    image_data=obj.image_data,
                    audio_data=obj.audio_data,
                    input_text=(input_text or input_ids),
                    request_obj=obj,
                    max_req_input_len=self.config.max_req_input_len,
                )

            if mm_inputs and mm_inputs.input_ids is not None:
                input_ids = mm_inputs.input_ids
            if mm_inputs and mm_inputs.token_type_ids is not None:
                token_type_ids = mm_inputs.token_type_ids
                if not isinstance(token_type_ids, list):
                    token_type_ids = token_type_ids.flatten().tolist()
            if (
                envs.SGLANG_MM_PRECOMPUTE_HASH.get()
                and mm_inputs
                and mm_inputs.mm_items
            ):
                for item in mm_inputs.mm_items:
                    if isinstance(item, MultimodalDataItem):
                        item.set_pad_value()
        else:
            mm_inputs = None

        self.request_validator.validate_one(obj=obj, input_ids=input_ids)
        tokenized_obj = self.tokenized_request_builder.build(
            obj,
            input_text,
            input_ids,
            input_embeds,
            mm_inputs,
            token_type_ids,
        )
        tokenized_obj.time_stats = self.rid_to_state[obj.rid].time_stats
        self.rid_to_state[obj.rid].time_stats.set_tokenize_finish_time()
        return tokenized_obj

    async def _batch_tokenize_and_process(
        self,
        batch_size: int,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
    ) -> List[Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput]]:
        """Handle batch tokenization for text inputs only."""
        logger.debug(f"Starting batch tokenization for {batch_size} text requests")

        # If batch does not have text nothing to tokenize
        # so lets construct the return object
        if not self._batch_has_text(batch_size, obj):
            # All requests already have input_ids, no need to tokenize
            return [await self._tokenize_one_request(obj[i]) for i in range(batch_size)]

        self.request_validator.validate_batch_tokenization_constraints(
            batch_size=batch_size, obj=obj
        )

        # Collect requests and texts
        requests = [obj[i] for i in range(batch_size)]
        texts = [req.text for req in requests]

        # Check if any request is a cross-encoder request
        is_cross_encoder_request = any(
            isinstance(req, EmbeddingReqInput) and req.is_cross_encoder_request
            for req in requests
        )

        # Batch tokenize all texts using unified method
        input_ids_list, token_type_ids_list = (
            await self.raw_tokenizer_wrapper._tokenize_texts(
                texts, is_cross_encoder_request
            )
        )

        # Process all requests
        tokenized_objs = []
        for i, req in enumerate(requests):
            self.request_validator.validate_one(obj=obj[i], input_ids=input_ids_list[i])
            token_type_ids = (
                token_type_ids_list[i] if token_type_ids_list is not None else None
            )
            tokenized_obj = self.tokenized_request_builder.build(
                req,
                req.text,
                input_ids_list[i],
                None,
                None,
                token_type_ids,
            )
            tokenized_obj.time_stats = self.rid_to_state[req.rid].time_stats
            self.rid_to_state[req.rid].time_stats.set_tokenize_finish_time()
            tokenized_objs.append(tokenized_obj)
        logger.debug(f"Completed batch processing for {batch_size} requests")
        return tokenized_objs

    def _batch_has_text(
        self,
        batch_size: int,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
    ) -> bool:
        """Check if any request in the batch contains text input."""
        for i in range(batch_size):
            if obj[i].text:
                return True
            elif self.config.is_generation and obj[i].contains_mm_input():
                return True

        return False

    def _should_use_batch_tokenization(self, batch_size, requests) -> bool:
        """Return True if we should run the tokenizer in batch mode.

        Current policy:
        - Respect explicit server flag `enable_tokenizer_batch_encode`.
        - Or, if no request has text or multimodal input (all use pre-tokenized input_ids or input_embeds), batch the requests without tokenization.
        - Batch tokenization does not support DP attention yet, and it will make everything goes to the first rank currently
        """
        return batch_size > 0 and (
            self.config.enable_tokenizer_batch_encode
            or (
                (not self.config.enable_dp_attention)
                and (not self._batch_has_text(batch_size, requests))
            )
        )
