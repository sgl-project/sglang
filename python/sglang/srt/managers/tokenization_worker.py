import asyncio
import logging
import threading
from typing import Dict, List, Optional, Union
import os
import zmq
import zmq.asyncio

from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    EmbeddingReqInput,
    TokenizedGenerateReqInput,
    TokenizedEmbeddingReqInput,
)
from sglang.srt.hf_transformers_utils import get_tokenizer, get_processor
from sglang.srt.managers.multimodal_processor import get_mm_processor, get_dummy_processor,import_processors
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, PortArgs
from sglang.srt.utils import get_zmq_socket
from sglang.utils import TypeBasedDispatcher

from sglang.srt.managers.io_struct import SessionParams
from sglang.srt.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)

class TokenizationWorker:
    """A worker class that handles tokenization tasks asynchronously."""

    def __init__(self, server_args: ServerArgs, port_args: PortArgs):
        self.server_args = server_args
        self.max_req_input_len = None
        # TODO re-apply ModelConfig PR
        self.model_config = ModelConfig(
            server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
            context_length=server_args.context_length,
            model_override_args=server_args.json_model_override_args,
            is_embedding=server_args.is_embedding,
            enable_multimodal=server_args.enable_multimodal,
            dtype=server_args.dtype,
            quantization=server_args.quantization,
        )
        self.is_generation = self.model_config.is_generation
        self.is_image_gen = self.model_config.is_image_gen
        self.context_len = self.model_config.context_len
        self.image_token_id = self.model_config.image_token_id

        if self.model_config.is_multimodal:
            import_processors()
            _processor = get_processor(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
                use_fast=not server_args.disable_fast_image_processor,
            )

            # We want to parallelize the image pre-processing so we create an executor for it
            # We create mm_processor for any skip_tokenizer_init to make sure we still encode
            # images even with skip_tokenizer_init=False.
            self.mm_processor = get_mm_processor(
                self.model_config.hf_config, server_args, _processor
            )

            if server_args.skip_tokenizer_init:
                self.tokenizer = self.processor = None
            else:
                self.processor = _processor
                self.tokenizer = self.processor.tokenizer
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
        else:
            self.mm_processor = get_dummy_processor()

            if server_args.skip_tokenizer_init:
                self.tokenizer = self.processor = None
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
        # Init inter-process communication
        context = zmq.asyncio.Context(2)

        self.send_to_tokenizer = get_zmq_socket(
            context, zmq.PUSH, port_args.tokenizer_worker_output_ipc_name, False
        )

        self.recv_from_tokenizer = get_zmq_socket(
            context, zmq.PULL, port_args.tokenizer_worker_ipc_name, False
        )
        # GenerateReqInput, EmbeddingReqInput
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (GenerateReqInput, self.handle_req_input),
                (EmbeddingReqInput, self.handle_req_input),
            ]
        )

    async def event_loop(self):
        """The event loop that handles requests"""
        while True:
            recv_obj = await self.recv_from_tokenizer.recv_pyobj()

            if isinstance(recv_obj, GenerateReqInput) or isinstance(recv_obj, EmbeddingReqInput):
                # Create task for async tokenization
                recv_obj.normalize_batch_and_arguments()
                
                if recv_obj.batch_size > 1:
                    task = asyncio.create_task(self._batch_tokenize_and_process(recv_obj.batch_size, recv_obj))
                elif recv_obj.batch_size == 1:
                    task = asyncio.create_task(self._tokenize_one_request(recv_obj))
                else:
                    raise ValueError(f"Invalid batch size: {recv_obj.batch_size}")
                tokenized_obj = await task
                await self.send_to_tokenizer.send_pyobj(tokenized_obj)
            else:
                raise ValueError(f"Invalid request type: {type(recv_obj)}")

    def handle_req_input(self, recv_obj):
        tokenized_obj = self._tokenize_one_request(recv_obj)
        return tokenized_obj

    async def _tokenize_one_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
    ):
        """Tokenize one request."""
        # Tokenize
        input_embeds = None
        input_text = obj.text
        if obj.input_embeds is not None:
            if not self.server_args.disable_radix_cache:
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
            input_ids = self.tokenizer.encode(input_text)

        image_inputs: Dict = await self.mm_processor.process_mm_data_async(
            obj.image_data, input_text or input_ids, obj, self.max_req_input_len
        )
        if image_inputs and "input_ids" in image_inputs:
            input_ids = image_inputs["input_ids"]

        self._validate_token_len(obj, input_ids)
        return self._create_tokenized_object(
            obj, input_text, input_ids, input_embeds, image_inputs
        )

    def _validate_token_len(
        self, obj: Union[GenerateReqInput, EmbeddingReqInput], input_ids: List[int]
    ) -> None:
        """Validates that the input token count and the requested token count doesn't exceed the model's context length."""

        input_token_num = len(input_ids) if input_ids is not None else 0
        # Check if input alone exceeds context length
        if input_token_num >= self.context_len:
            raise ValueError(
                f"The input ({input_token_num} tokens) is longer than the "
                f"model's context length ({self.context_len} tokens)."
            )

        # Check total tokens (input + max_new_tokens)
        max_new_tokens = obj.sampling_params.get("max_new_tokens")
        if (
            max_new_tokens is not None
            and (max_new_tokens + input_token_num) >= self.context_len
        ):
            total_tokens = max_new_tokens + input_token_num
            error_msg = (
                f"Requested token count exceeds the model's maximum context length "
                f"of {self.context_len} tokens. You requested a total of {total_tokens} "
                f"tokens: {input_token_num} tokens from the input messages and "
                f"{max_new_tokens} tokens for the completion. Please reduce the number "
                f"of tokens in the input messages or the completion to fit within the limit."
            )
            raise ValueError(error_msg)

    def _create_tokenized_object(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        input_text: str,
        input_ids: List[int],
        input_embeds: Optional[Union[List[float], None]] = None,
        image_inputs: Optional[Dict] = None,
    ) -> Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput]:
        """Create a tokenized request object from common parameters."""

        if self.is_generation:
            return_logprob = obj.return_logprob
            logprob_start_len = obj.logprob_start_len
            top_logprobs_num = obj.top_logprobs_num
            token_ids_logprob = obj.token_ids_logprob
            session_params = (
                SessionParams(**obj.session_params) if obj.session_params else None
            )

        sampling_params = SamplingParams(**obj.sampling_params)
        sampling_params.normalize(self.tokenizer)
        sampling_params.verify()

        # Build return object
        if isinstance(obj, GenerateReqInput):
            tokenized_obj = TokenizedGenerateReqInput(
                obj.rid,
                input_text,
                input_ids,
                image_inputs,
                sampling_params,
                return_logprob,
                logprob_start_len,
                top_logprobs_num,
                token_ids_logprob,
                obj.stream,
                bootstrap_host=obj.bootstrap_host,
                bootstrap_room=obj.bootstrap_room,
                lora_path=obj.lora_path,
                input_embeds=input_embeds,
                session_params=session_params,
                custom_logit_processor=obj.custom_logit_processor,
                return_hidden_states=obj.return_hidden_states,
            )
        elif isinstance(obj, EmbeddingReqInput):
            tokenized_obj = TokenizedEmbeddingReqInput(
                obj.rid,
                input_text,
                input_ids,
                image_inputs,
                sampling_params,
            )

        return tokenized_obj

    async def _batch_tokenize_and_process(
        self, batch_size: int, obj: Union[GenerateReqInput, EmbeddingReqInput]
    ) -> List[Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput]]:
        """Handle batch tokenization for text inputs only."""
        logger.debug(f"Starting batch tokenization for {batch_size} text requests")

        # Collect requests and texts
        requests = [obj[i] for i in range(batch_size)]
        texts = [req.text for req in requests]

        # Batch tokenize all texts
        encoded = self.tokenizer(texts)
        input_ids_list = encoded["input_ids"]

        # Process all requests
        tokenized_objs = []
        for i, req in enumerate(requests):
            self._validate_token_len(obj[i], input_ids_list[i])
            tokenized_objs.append(
                self._create_tokenized_object(
                    req, req.text, input_ids_list[i], None, None
                )
            )
        logger.debug(f"Completed batch processing for {batch_size} requests")
        return tokenized_objs

    def _validate_batch_tokenization_constraints(
        self, batch_size: int, obj: Union[GenerateReqInput, EmbeddingReqInput]
    ) -> None:
        """Validate constraints for batch tokenization processing."""
        for i in range(batch_size):
            if self.is_generation and obj[i].image_data:
                raise ValueError(
                    "For image input processing do not set `enable_tokenizer_batch_encode`."
                )
            if obj[i].input_ids is not None:
                raise ValueError(
                    "Batch tokenization is not needed for pre-tokenized input_ids. Do not set `enable_tokenizer_batch_encode`."
                )
            if obj[i].input_embeds is not None:
                raise ValueError(
                    "Batch tokenization is not needed for input_embeds. Do not set `enable_tokenizer_batch_encode`."
                )
