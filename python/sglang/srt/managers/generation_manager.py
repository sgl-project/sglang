import asyncio
import dataclasses
import os
import time
from typing import Optional, List, Any, Union, Dict

from sglang.srt.hf_transformers_utils import get_processor, get_tokenizer
from sglang.srt.managers.image_processor import get_dummy_image_processor, get_image_processor
from sglang.srt.managers.io_struct import GenerateReqInput, EmbeddingReqInput, SessionParams, TokenizedGenerateReqInput, \
    TokenizedEmbeddingReqInput, BatchStrOut, BatchEmbeddingOut, BatchTokenIDOut
from sglang.srt.metrics.collector import TokenizerMetricsCollector
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs


class GenerationManager:
    pass


class GenerationConverter:
    """Preprocessors and postprocessors for generation"""

    def __init__(
        self,
        server_args: ServerArgs,
    ):
        self.server_args = server_args
        self.model_config = _compute_model_config(server_args)

        # Create image processor placeholder
        self.image_processor = get_dummy_image_processor()

        # Create tokenizer
        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if self.model_config.is_multimodal:
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
                self.tokenizer = self.processor.tokenizer
                os.environ["TOKENIZERS_PARALLELISM"] = "false"

                # We want to parallelize the image pre-processing so we create an executor for it
                self.image_processor = get_image_processor(
                    self.model_config.hf_config, server_args, self.processor
                )
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )

    async def tokenize_request(
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

        if self.is_generation:
            # TODO: also support getting embeddings for multimodal models
            image_inputs: Dict = await self.image_processor.process_images_async(
                obj.image_data, input_text or input_ids, obj, self.max_req_input_len
            )
            if image_inputs and "input_ids" in image_inputs:
                input_ids = image_inputs["input_ids"]
            return_logprob = obj.return_logprob
            logprob_start_len = obj.logprob_start_len
            top_logprobs_num = obj.top_logprobs_num
            session_params = (
                SessionParams(**obj.session_params) if obj.session_params else None
            )

        input_token_num = len(input_ids) if input_ids is not None else 0
        if input_token_num >= self.context_len:
            raise ValueError(
                f"The input ({input_token_num} tokens) is longer than the "
                f"model's context length ({self.context_len} tokens)."
            )

        if (
            obj.sampling_params.get("max_new_tokens") is not None
            and obj.sampling_params.get("max_new_tokens") + input_token_num
            >= self.context_len
        ):
            raise ValueError(
                f"Requested token count exceeds the model's maximum context length "
                f"of {self.context_len} tokens. You requested a total of "
                f"{obj.sampling_params.get('max_new_tokens') + input_token_num} "
                f"tokens: {input_token_num} tokens from the input messages and "
                f"{obj.sampling_params.get('max_new_tokens')} tokens for the "
                f"completion. Please reduce the number of tokens in the input "
                f"messages or the completion to fit within the limit."
            )

        # Parse sampling parameters
        sampling_params = SamplingParams(**obj.sampling_params)
        sampling_params.normalize(self.tokenizer)
        sampling_params.verify()

        # Build return object
        if isinstance(obj, GenerateReqInput):
            return TokenizedGenerateReqInput(
                obj.rid,
                input_text,
                input_ids,
                image_inputs,
                sampling_params,
                return_logprob,
                logprob_start_len,
                top_logprobs_num,
                obj.stream,
                lora_path=obj.lora_path,
                input_embeds=input_embeds,
                session_params=session_params,
                custom_logit_processor=obj.custom_logit_processor,
            )
        elif isinstance(obj, EmbeddingReqInput):
            return TokenizedEmbeddingReqInput(
                obj.rid,
                input_text,
                input_ids,
                sampling_params,
            )
        else:
            raise NotImplementedError

    def tokenize_requests(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
    ) -> List[Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput]]:
        objs = [obj[i] for i in range(obj.batch_size)]
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            asyncio.gather(*(self.tokenize_request(obj) for obj in objs))
        )

    def postprocess_response(
        self,
        recv_obj: Union[BatchStrOut, BatchEmbeddingOut, BatchTokenIDOut],
        index: int,
        req_obj: Union[GenerateReqInput, EmbeddingReqInput],
    ) -> Dict[str, Any]:
        meta_info = self._compute_meta_info(index, recv_obj, req_obj)

        if isinstance(recv_obj, BatchStrOut):
            return {
                "text": recv_obj.output_strs[i],
                "meta_info": meta_info,
            }
        elif isinstance(recv_obj, BatchTokenIDOut):
            return {
                "token_ids": recv_obj.output_ids[i],
                "meta_info": meta_info,
            }
        elif isinstance(recv_obj, BatchEmbeddingOut):
            return {
                "embedding": recv_obj.embeddings[i],
                "meta_info": meta_info,
            }
        else:
            raise NotImplementedError

    def _compute_meta_info(self, index, recv_obj, req_obj):
        meta_info = {
            "id": rid,
            "finish_reason": recv_obj.finished_reasons[i],
            "prompt_tokens": recv_obj.prompt_tokens[i],
        }

        if getattr(state.obj, "return_logprob", False):
            self.convert_logprob_style(
                meta_info,
                state.obj.top_logprobs_num,
                state.obj.return_text_in_logprobs,
                recv_obj,
                i,
            )

        if not isinstance(recv_obj, BatchEmbeddingOut):
            meta_info.update(
                {
                    "completion_tokens": recv_obj.completion_tokens[i],
                    "cached_tokens": recv_obj.cached_tokens[i],
                }
            )

        return meta_info

    def convert_logprob_style(
        self,
        meta_info: dict,
        top_logprobs_num: int,
        return_text_in_logprobs: bool,
        recv_obj: BatchStrOut,
        recv_obj_index: int,
    ):
        meta_info["input_token_logprobs"] = self.detokenize_logprob_tokens(
            recv_obj.input_token_logprobs_val[recv_obj_index],
            recv_obj.input_token_logprobs_idx[recv_obj_index],
            return_text_in_logprobs,
        )
        meta_info["output_token_logprobs"] = self.detokenize_logprob_tokens(
            recv_obj.output_token_logprobs_val[recv_obj_index],
            recv_obj.output_token_logprobs_idx[recv_obj_index],
            return_text_in_logprobs,
        )

        if top_logprobs_num > 0:
            meta_info["input_top_logprobs"] = self.detokenize_top_logprobs_tokens(
                recv_obj.input_top_logprobs_val[recv_obj_index],
                recv_obj.input_top_logprobs_idx[recv_obj_index],
                return_text_in_logprobs,
            )
            meta_info["output_top_logprobs"] = self.detokenize_top_logprobs_tokens(
                recv_obj.output_top_logprobs_val[recv_obj_index],
                recv_obj.output_top_logprobs_idx[recv_obj_index],
                return_text_in_logprobs,
            )

    def detokenize_logprob_tokens(
        self,
        token_logprobs_val: List[float],
        token_logprobs_idx: List[int],
        decode_to_text: bool,
    ):
        if not decode_to_text:
            return [
                (logprob, token_id, None)
                for logprob, token_id in zip(token_logprobs_val, token_logprobs_idx)
            ]
        else:
            assert self.tokenizer is not None
            token_texts = self.tokenizer.batch_decode(token_logprobs_idx)
            return list(zip(token_logprobs_val, token_logprobs_idx, token_texts))

    def detokenize_top_logprobs_tokens(
        self,
        token_logprobs_val: List[float],
        token_logprobs_idx: List[int],
        decode_to_text: bool,
    ):
        # TODO: The current implementation only batches the detokenization for top-k tokens per single position.
        # We should batch all top-k tokens in all positions.
        ret = []
        for i in range(len(token_logprobs_val)):
            if token_logprobs_val[i]:
                ret.append(
                    self.detokenize_logprob_tokens(
                        token_logprobs_val[i], token_logprobs_idx[i], decode_to_text
                    )
                )
            else:
                ret.append(None)
        return ret


class _MetricManager:
    def __init__(self, server_args: ServerArgs):
        self.metrics_collector = TokenizerMetricsCollector(
            labels={
                "model_name": server_args.served_model_name,
                # TODO: Add lora name/path in the future,
            },
        )

    def handle_batch_output_metrics(
        self,
        recv_obj,
        i: int,
        state: "_MetricReqState",
        finished: bool,
        stream: Optional[bool],
    ):
        completion_tokens = (
            recv_obj.completion_tokens[i]
            if getattr(recv_obj, "completion_tokens", None)
            else 0
        )

        if state.first_token_time is None:
            state.first_token_time = time.time()
            self.metrics_collector.observe_time_to_first_token(
                state.first_token_time - state.created_time
            )
        else:
            if completion_tokens >= 2:
                # Compute time_per_output_token for the streaming case
                self.metrics_collector.observe_time_per_output_token(
                    (time.time() - state.first_token_time) / (completion_tokens - 1)
                )

        if finished:
            self.metrics_collector.observe_one_finished_request(
                recv_obj.prompt_tokens[i], completion_tokens
            )
            self.metrics_collector.observe_e2e_request_latency(
                time.time() - state.created_time
            )
            # Compute time_per_output_token for the non-streaming case
            if stream is not None and not stream and completion_tokens >= 1:
                self.metrics_collector.observe_time_per_output_token(
                    (time.time() - state.created_time) / completion_tokens
                )


@dataclasses.dataclass
class _ReqState:
    """Store the state a request."""

    out_list: List
    finished: bool
    event: asyncio.Event
    obj: Any

    # For streaming output
    last_output_offset: int = 0


@dataclasses.dataclass
class _MetricReqState:
    created_time: float
    first_token_time: Optional[float] = None
