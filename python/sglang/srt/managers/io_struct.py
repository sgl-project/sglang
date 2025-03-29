# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
The definition of objects transfered between different
processes (TokenizerManager, DetokenizerManager, Controller).
"""

import copy
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from sglang.srt.managers.schedule_batch import BaseFinishReason
from sglang.srt.sampling.sampling_params import SamplingParams


@dataclass
class SessionParams:
    id: Optional[str] = None
    rid: Optional[str] = None
    offset: Optional[int] = None
    replace: Optional[bool] = None


@dataclass
class GenerateReqInput:
    # The input prompt. It can be a single prompt or a batch of prompts.
    text: Optional[Union[List[str], str]] = None
    # The token ids for text; one can specify either text or input_ids
    input_ids: Optional[Union[List[List[int]], List[int]]] = None
    # The embeddings for input_ids; one can specify either text or input_ids or input_embeds.
    input_embeds: Optional[Union[List[List[List[float]]], List[List[float]]]] = None
    # The image input. It can be a file name, a url, or base64 encoded string.
    # See also python/sglang/srt/utils.py:load_image.
    image_data: Optional[Union[List[str], str]] = None
    # The audio input. Like image data, tt can be a file name, a url, or base64 encoded string.
    audio_data: Optional[Union[List[str], str]] = None
    # The sampling_params. See descriptions below.
    sampling_params: Optional[Union[List[Dict], Dict]] = None
    # The request id.
    rid: Optional[Union[List[str], str]] = None
    # Whether to return logprobs.
    return_logprob: Optional[Union[List[bool], bool]] = None
    # If return logprobs, the start location in the prompt for returning logprobs.
    # By default, this value is "-1", which means it will only return logprobs for output tokens.
    logprob_start_len: Optional[Union[List[int], int]] = None
    # If return logprobs, the number of top logprobs to return at each position.
    top_logprobs_num: Optional[Union[List[int], int]] = None
    # If return logprobs, the token ids to return logprob for.
    token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None
    # Whether to detokenize tokens in text in the returned logprobs.
    return_text_in_logprobs: bool = False
    # Whether to stream output.
    stream: bool = False
    # Whether to log metrics for this request (e.g. health_generate calls do not log metrics)
    log_metrics: bool = True

    # The modalities of the image data [image, multi-images, video]
    modalities: Optional[List[str]] = None
    # LoRA related
    lora_path: Optional[Union[List[Optional[str]], Optional[str]]] = None

    # Session info for continual prompting
    session_params: Optional[Union[List[Dict], Dict]] = None

    # Custom logit processor for advanced sampling control. Must be a serialized instance
    # of `CustomLogitProcessor` in python/sglang/srt/sampling/custom_logit_processor.py
    # Use the processor's `to_str()` method to generate the serialized string.
    custom_logit_processor: Optional[Union[List[Optional[str]], str]] = None

    # Whether to return hidden states
    return_hidden_states: bool = False

    def normalize_batch_and_arguments(self):
        if (
            self.text is None and self.input_ids is None and self.input_embeds is None
        ) or (
            self.text is not None
            and self.input_ids is not None
            and self.input_embeds is not None
        ):
            raise ValueError(
                "Either text, input_ids or input_embeds should be provided."
            )

        # Derive the batch size
        if self.text is not None:
            if isinstance(self.text, str):
                self.is_single = True
                self.batch_size = 1
            else:
                self.is_single = False
                self.batch_size = len(self.text)
            self.input_embeds = None
        elif self.input_ids is not None:
            if len(self.input_ids) == 0:
                raise ValueError("input_ids cannot be empty.")
            if isinstance(self.input_ids[0], int):
                self.is_single = True
                self.batch_size = 1
            else:
                self.is_single = False
                self.batch_size = len(self.input_ids)
            self.input_embeds = None
        else:
            if isinstance(self.input_embeds[0][0], float):
                self.is_single = True
                self.batch_size = 1
            else:
                self.batch_size = len(self.input_embeds)

        # Handle parallel sampling
        # When parallel sampling is used, we always treat the input as a batch.
        if self.sampling_params is None:
            self.parallel_sample_num = 1
        elif isinstance(self.sampling_params, dict):
            self.parallel_sample_num = self.sampling_params.get("n", 1)
        else:  # isinstance(self.sampling_params, list):
            self.parallel_sample_num = self.sampling_params[0].get("n", 1)
            assert all(
                self.parallel_sample_num == sampling_params.get("n", 1)
                for sampling_params in self.sampling_params
            ), "The parallel_sample_num should be the same for all samples in sample params."

        if self.parallel_sample_num > 1 and self.is_single:
            self.is_single = False
            if self.text is not None:
                self.text = [self.text]
            if self.input_ids is not None:
                self.input_ids = [self.input_ids]

        # Fill in default arguments
        if self.is_single:
            if self.sampling_params is None:
                self.sampling_params = {}
            if self.rid is None:
                self.rid = uuid.uuid4().hex
            if self.return_logprob is None:
                self.return_logprob = False
            if self.logprob_start_len is None:
                self.logprob_start_len = -1
            if self.top_logprobs_num is None:
                self.top_logprobs_num = 0
            if not self.token_ids_logprob:  # covers both None and []
                self.token_ids_logprob = None
        else:
            if self.parallel_sample_num == 1:
                num = self.batch_size
            else:
                # Expand parallel_sample_num
                num = self.batch_size * self.parallel_sample_num

            if not self.image_data:
                self.image_data = [None] * num
            elif not isinstance(self.image_data, list):
                self.image_data = [self.image_data] * num
            elif isinstance(self.image_data, list):
                pass

            if self.audio_data is None:
                self.audio_data = [None] * num
            elif not isinstance(self.audio_data, list):
                self.audio_data = [self.audio_data] * num
            elif isinstance(self.audio_data, list):
                pass

            if self.sampling_params is None:
                self.sampling_params = [{}] * num
            elif not isinstance(self.sampling_params, list):
                self.sampling_params = [self.sampling_params] * num

            if self.rid is None:
                self.rid = [uuid.uuid4().hex for _ in range(num)]
            else:
                assert isinstance(self.rid, list), "The rid should be a list."

            if self.return_logprob is None:
                self.return_logprob = [False] * num
            elif not isinstance(self.return_logprob, list):
                self.return_logprob = [self.return_logprob] * num
            else:
                assert self.parallel_sample_num == 1

            if self.logprob_start_len is None:
                self.logprob_start_len = [-1] * num
            elif not isinstance(self.logprob_start_len, list):
                self.logprob_start_len = [self.logprob_start_len] * num
            else:
                assert self.parallel_sample_num == 1

            if self.top_logprobs_num is None:
                self.top_logprobs_num = [0] * num
            elif not isinstance(self.top_logprobs_num, list):
                self.top_logprobs_num = [self.top_logprobs_num] * num
            else:
                assert self.parallel_sample_num == 1

            if not self.token_ids_logprob:  # covers both None and []
                self.token_ids_logprob = [None] * num
            elif not isinstance(self.token_ids_logprob, list):
                self.token_ids_logprob = [[self.token_ids_logprob] for _ in range(num)]
            elif not isinstance(self.token_ids_logprob[0], list):
                self.token_ids_logprob = [
                    copy.deepcopy(self.token_ids_logprob) for _ in range(num)
                ]
            else:
                assert self.parallel_sample_num == 1

            if self.custom_logit_processor is None:
                self.custom_logit_processor = [None] * num
            elif not isinstance(self.custom_logit_processor, list):
                self.custom_logit_processor = [self.custom_logit_processor] * num
            else:
                assert self.parallel_sample_num == 1

        # Other checks
        if self.session_params is not None:
            assert isinstance(self.session_params, dict) or isinstance(
                self.session_params[0], dict
            )

    def regenerate_rid(self):
        self.rid = uuid.uuid4().hex
        return self.rid

    def __getitem__(self, i):
        return GenerateReqInput(
            text=self.text[i] if self.text is not None else None,
            input_ids=self.input_ids[i] if self.input_ids is not None else None,
            image_data=self.image_data[i],
            audio_data=self.audio_data[i],
            sampling_params=self.sampling_params[i],
            rid=self.rid[i],
            return_logprob=self.return_logprob[i],
            logprob_start_len=self.logprob_start_len[i],
            top_logprobs_num=self.top_logprobs_num[i],
            token_ids_logprob=self.token_ids_logprob[i],
            return_text_in_logprobs=self.return_text_in_logprobs,
            stream=self.stream,
            log_metrics=self.log_metrics,
            modalities=self.modalities[i] if self.modalities else None,
            lora_path=self.lora_path[i] if self.lora_path is not None else None,
            custom_logit_processor=(
                self.custom_logit_processor[i]
                if self.custom_logit_processor is not None
                else None
            ),
            return_hidden_states=self.return_hidden_states,
        )


@dataclass
class TokenizedGenerateReqInput:
    # The request id
    rid: str
    # The input text
    input_text: str
    # The input token ids
    input_ids: List[int]
    # The multimodal inputs
    mm_inputs: dict
    # The sampling parameters
    sampling_params: SamplingParams
    # Whether to return the logprobs
    return_logprob: bool
    # If return logprobs, the start location in the prompt for returning logprobs.
    logprob_start_len: int
    # If return logprobs, the number of top logprobs to return at each position.
    top_logprobs_num: int
    # If return logprobs, the token id to return logprob for
    token_ids_logprob: List[int]
    # Whether to stream output
    stream: bool

    # LoRA related
    lora_path: Optional[str] = None  # None means just use the base model
    # The input embeds
    input_embeds: Optional[Union[List[List[List[float]]], List[List[float]]]] = None

    # Session info for continual prompting
    session_params: Optional[SessionParams] = None

    # Custom logit processor for advanced sampling control. Must be a serialized instance
    # of `CustomLogitProcessor` in python/sglang/srt/sampling/custom_logit_processor.py
    # Use the processor's `to_str()` method to generate the serialized string.
    custom_logit_processor: Optional[str] = None

    # Whether to return hidden states
    return_hidden_states: bool = False


@dataclass
class EmbeddingReqInput:
    # The input prompt. It can be a single prompt or a batch of prompts.
    text: Optional[Union[List[str], str]] = None
    # The image input. It can be a file name, a url, or base64 encoded string.
    image_data: Optional[Union[List[str], str]] = None
    # The token ids for text; one can either specify text or input_ids.
    input_ids: Optional[Union[List[List[int]], List[int]]] = None
    # The request id.
    rid: Optional[Union[List[str], str]] = None
    # Dummy sampling params for compatibility
    sampling_params: Union[List[Dict], Dict] = None
    # Dummy input embeds for compatibility
    input_embeds: Optional[Union[List[List[List[float]]], List[List[float]]]] = None
    # Whether to log metrics for this request (e.g. health_generate calls do not log metrics)
    log_metrics: bool = True
    # The modalities of the image data [image, multi-images, video]
    modalities: Optional[List[str]] = None

    def normalize_batch_and_arguments(self):
        # at least one of text, input_ids, or image should be provided
        if self.text is None and self.input_ids is None and self.image_data is None:
            raise ValueError(
                "At least one of text, input_ids, or image should be provided"
            )

        # text and input_ids cannot be provided at the same time
        if self.text is not None and self.input_ids is not None:
            raise ValueError("text and input_ids cannot be provided at the same time")

        # Derive the batch size
        self.batch_size = 0
        self.is_single = True

        # check the batch size of text
        if self.text is not None:
            if isinstance(self.text, list):
                self.batch_size += len(self.text)
            else:
                self.batch_size += 1

        # check the batch size of input_ids
        if self.input_ids is not None:
            if isinstance(self.input_ids[0], list):
                self.batch_size += len(self.input_ids)
            else:
                self.batch_size += 1

        if self.batch_size > 1:
            self.is_single = False

        # Fill in default arguments
        if self.is_single:
            if self.rid is None:
                self.rid = uuid.uuid4().hex
            if self.sampling_params is None:
                self.sampling_params = {}
            self.sampling_params["max_new_tokens"] = 0
        else:
            if self.rid is None:
                self.rid = [uuid.uuid4().hex for _ in range(self.batch_size)]
            else:
                assert isinstance(self.rid, list), "The rid should be a list."

            if self.sampling_params is None:
                self.sampling_params = [{}] * self.batch_size
            for i in range(self.batch_size):
                self.sampling_params[i]["max_new_tokens"] = 0

    def regenerate_rid(self):
        self.rid = uuid.uuid4().hex
        return self.rid

    def __getitem__(self, i):
        return EmbeddingReqInput(
            text=self.text[i] if self.text is not None else None,
            input_ids=self.input_ids[i] if self.input_ids is not None else None,
            image_data=self.image_data[i] if self.image_data is not None else None,
            sampling_params=self.sampling_params[i],
            rid=self.rid[i],
        )


@dataclass
class TokenizedEmbeddingReqInput:
    # The request id
    rid: str
    # The input text
    input_text: str
    # The input token ids
    input_ids: List[int]
    # The image inputs
    image_inputs: dict
    # Dummy sampling params for compatibility
    sampling_params: SamplingParams


@dataclass
class BatchTokenIDOut:
    # The request id
    rids: List[str]
    # The finish reason
    finished_reasons: List[BaseFinishReason]
    # For incremental decoding
    decoded_texts: List[str]
    decode_ids: List[int]
    read_offsets: List[int]
    # Only used when `--skip-tokenizer-init` is on
    output_ids: Optional[List[int]]
    # Detokenization configs
    skip_special_tokens: List[bool]
    spaces_between_special_tokens: List[bool]
    no_stop_trim: List[bool]

    # Token counts
    prompt_tokens: List[int]
    completion_tokens: List[int]
    cached_tokens: List[int]
    spec_verify_ct: List[int]

    # Logprobs
    input_token_logprobs_val: List[float]
    input_token_logprobs_idx: List[int]
    output_token_logprobs_val: List[float]
    output_token_logprobs_idx: List[int]
    input_top_logprobs_val: List[List]
    input_top_logprobs_idx: List[List]
    output_top_logprobs_val: List[List]
    output_top_logprobs_idx: List[List]
    input_token_ids_logprobs_val: List[List]
    input_token_ids_logprobs_idx: List[List]
    output_token_ids_logprobs_val: List[List]
    output_token_ids_logprobs_idx: List[List]

    # Hidden states
    output_hidden_states: List[List[float]]


@dataclass
class BatchMultimodalDecodeReq:
    # The request id
    rids: List[str]
    finished_reasons: List[BaseFinishReason]

    # Token counts
    prompt_tokens: List[int]
    completion_tokens: List[int]
    cached_tokens: List[int]


@dataclass
class BatchStrOut:
    # The request id
    rids: List[str]
    # The finish reason
    finished_reasons: List[dict]
    # The output decoded strings
    output_strs: List[str]
    # The token ids
    output_ids: Optional[List[int]]

    # Token counts
    prompt_tokens: List[int]
    completion_tokens: List[int]
    cached_tokens: List[int]
    spec_verify_ct: List[int]

    # Logprobs
    input_token_logprobs_val: List[float]
    input_token_logprobs_idx: List[int]
    output_token_logprobs_val: List[float]
    output_token_logprobs_idx: List[int]
    input_top_logprobs_val: List[List]
    input_top_logprobs_idx: List[List]
    output_top_logprobs_val: List[List]
    output_top_logprobs_idx: List[List]
    input_token_ids_logprobs_val: List[List]
    input_token_ids_logprobs_idx: List[List]
    output_token_ids_logprobs_val: List[List]
    output_token_ids_logprobs_idx: List[List]

    # Hidden states
    output_hidden_states: List[List[float]]


@dataclass
class BatchMultimodalOut:
    # The request id
    rids: List[str]
    # The finish reason
    finished_reasons: List[dict]
    # The outputs
    outputs: List[List[Dict]]

    # Token counts
    prompt_tokens: List[int]
    completion_tokens: List[int]
    cached_tokens: List[int]


@dataclass
class BatchEmbeddingOut:
    # The request id
    rids: List[str]
    # The finish reason
    finished_reasons: List[BaseFinishReason]
    # The output embedding
    embeddings: List[List[float]]
    # Token counts
    prompt_tokens: List[int]
    cached_tokens: List[int]


@dataclass
class FlushCacheReq:
    pass


@dataclass
class UpdateWeightFromDiskReqInput:
    # The model path with the new weights
    model_path: str
    # The format to load the weights
    load_format: Optional[str] = None


@dataclass
class UpdateWeightFromDiskReqOutput:
    success: bool
    message: str
    # Number of paused requests during weight sync.
    num_paused_requests: Optional[int] = 0


@dataclass
class UpdateWeightsFromDistributedReqInput:
    name: str
    dtype: str
    shape: List[int]


@dataclass
class UpdateWeightsFromDistributedReqOutput:
    success: bool
    message: str


@dataclass
class UpdateWeightsFromTensorReqInput:
    # List containing one serialized Dict[str, torch.Tensor] per TP worker
    serialized_named_tensors: List[bytes]
    load_format: Optional[str]
    flush_cache: bool


@dataclass
class UpdateWeightsFromTensorReqOutput:
    success: bool
    message: str


@dataclass
class InitWeightsUpdateGroupReqInput:
    # The master address
    master_address: str
    # The master port
    master_port: int
    # The rank offset
    rank_offset: int
    # The world size
    world_size: int
    # The group name
    group_name: str = "weight_update_group"
    # The backend
    backend: str = "nccl"


@dataclass
class InitWeightsUpdateGroupReqOutput:
    success: bool
    message: str


@dataclass
class GetWeightsByNameReqInput:
    name: str
    truncate_size: int = 100


@dataclass
class GetWeightsByNameReqOutput:
    parameter: list


@dataclass
class ReleaseMemoryOccupationReqInput:
    pass


@dataclass
class ReleaseMemoryOccupationReqOutput:
    pass


@dataclass
class ResumeMemoryOccupationReqInput:
    pass


@dataclass
class ResumeMemoryOccupationReqOutput:
    pass


@dataclass
class AbortReq:
    # The request id
    rid: str


@dataclass
class GetInternalStateReq:
    pass


@dataclass
class GetInternalStateReqOutput:
    internal_state: Dict[Any, Any]


@dataclass
class SetInternalStateReq:
    server_args: Dict[str, Any]


@dataclass
class SetInternalStateReqOutput:
    updated: bool
    server_args: Dict[str, Any]


@dataclass
class ProfileReqInput:
    # The output directory
    output_dir: Optional[str] = None
    # If set, it profile as many as this number of steps.
    # If it is set, profiling is automatically stopped after this step, and
    # the caller doesn't need to run stop_profile.
    num_steps: Optional[int] = None
    activities: Optional[List[Literal["CPU", "GPU", "MEM", "CUDA_PROFILER"]]] = None


class ProfileReqType(Enum):
    START_PROFILE = 1
    STOP_PROFILE = 2


class ExpertDistributionReq(Enum):
    START_RECORD = 1
    STOP_RECORD = 2
    DUMP_RECORD = 3


@dataclass
class ExpertDistributionReqOutput:
    pass


@dataclass
class ProfileReq:
    type: ProfileReqType
    output_dir: Optional[str] = None
    num_steps: Optional[int] = None
    activities: Optional[List[str]] = None
    with_stack: Optional[bool] = None
    record_shapes: Optional[bool] = None


@dataclass
class ProfileReqOutput:
    success: bool
    message: str


@dataclass
class ConfigureLoggingReq:
    log_requests: Optional[bool] = None
    log_requests_level: Optional[int] = None
    dump_requests_folder: Optional[str] = None
    dump_requests_threshold: Optional[int] = None


@dataclass
class OpenSessionReqInput:
    capacity_of_str_len: int
    session_id: Optional[str] = None


@dataclass
class CloseSessionReqInput:
    session_id: str


@dataclass
class OpenSessionReqOutput:
    session_id: Optional[str]
    success: bool


@dataclass
class HealthCheckOutput:
    pass


@dataclass
class Function:
    description: Optional[str] = None
    name: Optional[str] = None
    parameters: Optional[object] = None


@dataclass
class Tool:
    function: Function
    type: Optional[str] = "function"


@dataclass
class ParseFunctionCallReq:
    text: str  # The text to parse.
    tools: List[Tool] = field(
        default_factory=list
    )  # A list of available function tools (name, parameters, etc.).
    tool_call_parser: Optional[str] = (
        None  # Specify the parser type, e.g. 'llama3', 'qwen25', or 'mistral'. If not specified, tries all.
    )


@dataclass
class SeparateReasoningReqInput:
    text: str  # The text to parse.
    reasoning_parser: str  # Specify the parser type, e.g., "deepseek-r1".


@dataclass
class VertexGenerateReqInput:
    instances: List[dict]
    parameters: Optional[dict] = None


@dataclass
class RpcReqInput:
    method: str
    parameters: Optional[Dict] = None


@dataclass
class RpcReqOutput:
    success: bool
    message: str
