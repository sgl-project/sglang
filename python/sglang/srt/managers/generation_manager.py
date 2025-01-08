import asyncio
import copy
import dataclasses
import logging
import os
import time
from typing import Dict, List, Union, Any, Callable
from typing import Optional

import fastapi
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.hf_transformers_utils import get_processor, get_tokenizer
from sglang.srt.managers.image_processor import (
    get_dummy_image_processor,
    get_image_processor,
)
from sglang.srt.managers.io_struct import (
    AbortReq,
)
from sglang.srt.managers.io_struct import (
    BatchEmbeddingOut,
    BatchStrOut,
    BatchTokenIDOut,
)
from sglang.srt.managers.io_struct import (
    EmbeddingReqInput,
    GenerateReqInput,
    SessionParams,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.metrics.collector import TokenizerMetricsCollector
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    dataclass_to_string_truncated,
)

logger = logging.getLogger(__name__)


class GenerationManager:
    def __init__(
        self,
        server_args: ServerArgs,
        on_request: Callable,
    ):
        self.server_args = server_args
        self.on_request = on_request

        self.model_config = ModelConfig(
            server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
            context_length=server_args.context_length,
            model_override_args=server_args.json_model_override_args,
            is_embedding=server_args.is_embedding,
            dtype=server_args.dtype,
            quantization=server_args.quantization,
        )

        self.is_generation = self.model_config.is_generation
        self.context_len = self.model_config.context_len
        self.image_token_id = self.model_config.image_token_id

        self._generation_converter = GenerationConverter(
            server_args=server_args,
        )

        self.rid_to_state: Dict[str, _ReqState] = {}

        # Metrics
        if server_args.enable_metrics:
            self._metric_manager = _MetricManager()

    async def generate(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        created_time = time.time()

        if isinstance(obj, EmbeddingReqInput) and self.is_generation:
            raise ValueError(
                "This model does not appear to be an embedding model by default. "
                "Please add `--is-embedding` when launching the server or try another model."
            )

        obj.normalize_batch_and_arguments()

        if self.server_args.log_requests:
            logger.info(f"Receive: obj={dataclass_to_string_truncated(obj)}")

        is_single = obj.is_single
        if is_single:
            tokenized_obj = await self._generation_converter.tokenize_request(obj)
            self._send_one_request(obj, tokenized_obj, created_time)
            async for response in self._wait_one_response(obj, request):
                yield response
        else:
            async for response in self._handle_batch_request(
                obj, request, created_time
            ):
                yield response

    def _send_one_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        tokenized_obj: Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput],
        created_time: Optional[float] = None,
    ):
        event = asyncio.Event()
        state = _ReqState([], False, event, obj, metric=_MetricReqState(created_time=created_time))
        self.rid_to_state[obj.rid] = state
        self.on_request(tokenized_obj)

    async def _wait_one_response(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        """Wait for the response of one request."""
        state = self.rid_to_state[obj.rid]

        while True:
            try:
                await asyncio.wait_for(state.event.wait(), timeout=4)
            except asyncio.TimeoutError:
                if request is not None and await request.is_disconnected():
                    self.abort_request(obj.rid)
                    raise ValueError(f"Abort request {obj.rid}")
                continue

            out = state.out_list[-1]

            state.out_list = []
            if state.finished:
                if self.server_args.log_requests:
                    msg = f"Finish: obj={dataclass_to_string_truncated(obj)}, out={dataclass_to_string_truncated(out)}"
                    logger.info(msg)
                del self.rid_to_state[obj.rid]
                yield out
                break

            state.event.clear()

            if obj.stream:
                yield out
            else:
                if request is not None and await request.is_disconnected():
                    self.abort_request(obj.rid)
                    raise ValueError(f"Abort request {obj.rid}")

    async def _handle_batch_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
        created_time: Optional[float] = None,
    ):
        batch_size = obj.batch_size

        generators = []
        rids = []
        if getattr(obj, "parallel_sample_num", 1) == 1:
            # Send all requests
            for i in range(batch_size):
                tmp_obj = obj[i]
                tokenized_obj = await self._generation_converter.tokenize_request(tmp_obj)
                self._send_one_request(tmp_obj, tokenized_obj, created_time)
                generators.append(self._wait_one_response(tmp_obj, request))
                rids.append(tmp_obj.rid)
        else:
            # FIXME: When using batch and parallel_sample_num together, the perf is not optimal.
            if batch_size > 128:
                logger.warning(
                    "Sending a single large batch with parallel sampling (n > 1) has not been well optimized. "
                    "The performance might be better if you just duplicate the requests n times or use "
                    "many threads to send them one by one with parallel sampling (n > 1)."
                )

            # Tokenize all requests
            objs = [obj[i] for i in range(batch_size)]
            tokenized_objs = await asyncio.gather(
                *(self._generation_converter.tokenize_request(obj) for obj in objs)
            )

            # Cache the common prefix for parallel sampling
            for i in range(batch_size):
                tmp_obj = copy.copy(objs[i])
                tokenized_obj = copy.copy(tokenized_objs[i])
                tokenized_obj.rid = tmp_obj.regenerate_rid()
                tokenized_obj.sampling_params = copy.copy(tokenized_obj.sampling_params)
                tokenized_obj.sampling_params.max_new_tokens = 0
                tokenized_obj.stream = False
                self._send_one_request(tmp_obj, tokenized_obj, created_time)
                await self._wait_one_response(tmp_obj, request).__anext__()

            # Expand requests, assign new rids for them, and send them
            for i in range(batch_size):
                for _ in range(obj.parallel_sample_num):
                    tmp_obj = copy.copy(objs[i])
                    tokenized_obj = copy.copy(tokenized_objs[i])
                    tokenized_obj.rid = tmp_obj.regenerate_rid()
                    self._send_one_request(tmp_obj, tokenized_obj, created_time)
                    generators.append(self._wait_one_response(tmp_obj, request))
                    rids.append(tmp_obj.rid)

        # Wait for all requests
        is_stream = hasattr(obj, "stream") and obj.stream
        if not is_stream:
            outputs = await asyncio.gather(*(gen.__anext__() for gen in generators))
            yield outputs
        else:
            rid_to_index = {rid: i for i, rid in enumerate(rids)}
            task_map = {asyncio.create_task(gen.__anext__()): gen for gen in generators}
            while task_map:
                done, _ = await asyncio.wait(
                    task_map.keys(), return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    gen = task_map.pop(task)
                    try:
                        result = task.result()
                        result["index"] = rid_to_index[result["meta_info"]["id"]]
                        yield result
                        new_task = asyncio.create_task(gen.__anext__())
                        task_map[new_task] = gen
                    except StopAsyncIteration:
                        pass

    def handle_batch_output(
        self, recv_obj: Union[BatchStrOut, BatchEmbeddingOut, BatchTokenIDOut]
    ):
        for index, rid in enumerate(recv_obj.rids):
            state = self.rid_to_state.get(rid, None)
            if state is None:
                continue

            out_dict = self._generation_converter.postprocess_response(recv_obj, index, rid, state.obj)

            state.out_list.append(out_dict)
            state.finished = recv_obj.finished_reasons[index] is not None
            state.event.set()

            if self._metric_manager:
                self._metric_manager.handle_batch_output_metrics(recv_obj, index, state.metric, finished=state.finished,
                                                                 stream=state.obj.stream)

    def abort_request(self, rid: str):
        if rid not in self.rid_to_state:
            return
        del self.rid_to_state[rid]
        req = AbortReq(rid)
        self.on_request(req)


@dataclasses.dataclass
class _ReqState:
    """Store the state a request."""

    out_list: List
    finished: bool
    event: asyncio.Event
    obj: Any

    metric: '_MetricReqState'

    # For streaming output
    last_output_offset: int = 0


class GenerationConverter:
    """Preprocessors and postprocessors for generation"""

    def __init__(
        self,
        server_args: ServerArgs,
        model_config: ModelConfig,
    ):
        self.server_args = server_args

        # Create image processor placeholder
        self.image_processor = get_dummy_image_processor()

        # Create tokenizer
        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if model_config.is_multimodal:
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                )
                self.tokenizer = self.processor.tokenizer
                os.environ["TOKENIZERS_PARALLELISM"] = "false"

                # We want to parallelize the image pre-processing so we create an executor for it
                self.image_processor = get_image_processor(
                    model_config.hf_config, server_args, self.processor
                )
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                )

    async def tokenize_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
    ) -> Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput]:
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
        elif obj.input_ids is None:
            input_ids = self.tokenizer.encode(input_text)
        else:
            input_ids = obj.input_ids

        if self.is_generation:
            # TODO: also support getting embeddings for multimodal models
            image_inputs: Dict = await self.image_processor.process_images_async(
                obj.image_data, input_text or input_ids, obj
            )
            if image_inputs and "input_ids" in image_inputs:
                input_ids = image_inputs["input_ids"]
            return_logprob = obj.return_logprob
            logprob_start_len = obj.logprob_start_len
            top_logprobs_num = obj.top_logprobs_num
            session_params = (
                SessionParams(**obj.session_params) if obj.session_params else None
            )

        if obj.input_ids is not None and len(input_ids) >= self.context_len:
            raise ValueError(
                f"The input ({len(input_ids)} tokens) is longer than the "
                f"model's context length ({self.context_len} tokens)."
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

    def postprocess_response(
        self,
        recv_obj: Union[BatchStrOut, BatchEmbeddingOut, BatchTokenIDOut],
        index: int,
        rid: str,
        req_obj: Union[GenerateReqInput, EmbeddingReqInput],
    ) -> Dict[str, Any]:
        meta_info = self._compute_meta_info(index, recv_obj, req_obj, rid)

        if isinstance(recv_obj, BatchStrOut):
            out_dict = {
                "text": recv_obj.output_strs[index],
                "meta_info": meta_info,
            }
            if self.server_args.return_token_ids:
                out_dict.update(
                    {
                        "input_ids": recv_obj.origin_input_ids[index],
                        "output_ids": recv_obj.output_ids[index],
                    }
                )
            return out_dict

        elif isinstance(recv_obj, BatchTokenIDOut):
            return {
                "token_ids": recv_obj.output_ids[index],
                "meta_info": meta_info,
            }

        elif isinstance(recv_obj, BatchEmbeddingOut):
            return {
                "embedding": recv_obj.embeddings[index],
                "meta_info": meta_info,
            }

        else:
            raise NotImplementedError

    def _compute_meta_info(self, index, recv_obj, req_obj, rid):
        meta_info = {
            "id": rid,
            "finish_reason": recv_obj.finished_reasons[index],
            "prompt_tokens": recv_obj.prompt_tokens[index],
        }
        if getattr(req_obj, "return_logprob", False):
            self._convert_logprob_style(
                meta_info,
                req_obj.top_logprobs_num,
                req_obj.return_text_in_logprobs,
                recv_obj,
                index,
            )
        if not isinstance(recv_obj, BatchEmbeddingOut):
            meta_info.update(
                {
                    "completion_tokens": recv_obj.completion_tokens[index],
                    "cached_tokens": recv_obj.cached_tokens[index],
                }
            )
        return meta_info

    def _convert_logprob_style(
        self,
        meta_info: dict,
        top_logprobs_num: int,
        return_text_in_logprobs: bool,
        recv_obj: BatchStrOut,
        recv_obj_index: int,
    ):
        meta_info["input_token_logprobs"] = self._detokenize_logprob_tokens(
            recv_obj.input_token_logprobs_val[recv_obj_index],
            recv_obj.input_token_logprobs_idx[recv_obj_index],
            return_text_in_logprobs,
        )
        meta_info["output_token_logprobs"] = self._detokenize_logprob_tokens(
            recv_obj.output_token_logprobs_val[recv_obj_index],
            recv_obj.output_token_logprobs_idx[recv_obj_index],
            return_text_in_logprobs,
        )
        meta_info["normalized_prompt_logprob"] = recv_obj.normalized_prompt_logprob[
            recv_obj_index
        ]

        if top_logprobs_num > 0:
            meta_info["input_top_logprobs"] = self._detokenize_top_logprobs_tokens(
                recv_obj.input_top_logprobs_val[recv_obj_index],
                recv_obj.input_top_logprobs_idx[recv_obj_index],
                return_text_in_logprobs,
            )
            meta_info["output_top_logprobs"] = self._detokenize_top_logprobs_tokens(
                recv_obj.output_top_logprobs_val[recv_obj_index],
                recv_obj.output_top_logprobs_idx[recv_obj_index],
                return_text_in_logprobs,
            )

    def _detokenize_logprob_tokens(
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

    def _detokenize_top_logprobs_tokens(
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
                    self._detokenize_logprob_tokens(
                        token_logprobs_val[i], token_logprobs_idx[i], decode_to_text
                    )
                )
            else:
                ret.append(None)
        return ret


class _MetricManager:
    def __init__(self):
        self._metrics_collector = TokenizerMetricsCollector(
            labels={
                "model_name": self.server_args.served_model_name,
                # TODO: Add lora name/path in the future,
            },
        )

    def handle_batch_output_metrics(
        self,
        recv_obj,
        i: int,
        state: '_MetricReqState',
        finished: bool,
        stream: bool,
    ):
        completion_tokens = (
            recv_obj.completion_tokens[i] if recv_obj.completion_tokens else 0
        )
        if state.first_token_time is None:
            state.first_token_time = time.time()
            self._metrics_collector.observe_time_to_first_token(
                state.first_token_time - state.created_time
            )
        else:
            if completion_tokens >= 2:
                # Compute time_per_output_token for the streaming case
                self._metrics_collector.observe_time_per_output_token(
                    (time.time() - state.first_token_time)
                    / (completion_tokens - 1)
                )
        if finished:
            self._metrics_collector.inc_prompt_tokens(recv_obj.prompt_tokens[i])
            self._metrics_collector.inc_generation_tokens(completion_tokens)
            self._metrics_collector.observe_e2e_request_latency(
                time.time() - state.created_time
            )
            # Compute time_per_output_token for the non-streaming case
            if not stream and completion_tokens >= 1:
                self._metrics_collector.observe_time_per_output_token(
                    (time.time() - state.created_time) / completion_tokens
                )


@dataclasses.dataclass
class _MetricReqState:
    created_time: float
    first_token_time: Optional[float] = None
