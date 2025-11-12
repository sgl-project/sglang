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
The definition of objects transferred between different
processes (TokenizerManager, DetokenizerManager, Scheduler).
"""

import copy
import uuid
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.managers.schedule_batch import BaseFinishReason
from sglang.srt.multimodal.mm_utils import has_valid_data
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils import ImageData

# Handle serialization of Image for pydantic
if TYPE_CHECKING:
    from PIL.Image import Image
else:
    Image = Any


@dataclass
class BaseReq(ABC):
    rid: Optional[Union[str, List[str]]] = field(default=None, kw_only=True)
    http_worker_ipc: Optional[str] = field(default=None, kw_only=True)

    def regenerate_rid(self):
        """Generate a new request ID and return it."""
        if isinstance(self.rid, list):
            self.rid = [uuid.uuid4().hex for _ in range(len(self.rid))]
        else:
            self.rid = uuid.uuid4().hex
        return self.rid


@dataclass
class BaseBatchReq(ABC):
    rids: Optional[List[str]] = field(default=None, kw_only=True)
    http_worker_ipcs: Optional[List[str]] = field(default=None, kw_only=True)

    def regenerate_rids(self):
        """Generate new request IDs and return them."""
        self.rids = [uuid.uuid4().hex for _ in range(len(self.rids))]
        return self.rids


@dataclass
class RequestTimingMetricsMixin:
    """
    Mixin class containing common request-level timing metrics.

    This class consolidates the timing metrics that are shared across all batch output types
    to avoid code duplication and ensure consistency.
    """

    # Queue duration: time spent waiting in queue before request is scheduled.
    queue_time: Optional[List[Optional[float]]]

    # Forward entry time: timestamp when the request enters the forward pass stage.
    # This corresponds to `forward_entry_time` in TimeStats.
    # In different modes:
    #   - Unified/PD-colocate: timestamp when forward computation begins (covers prefill + decode)
    #   - Prefill instance (P): timestamp when prefill forward pass begins
    #   - Decode instance (D): timestamp when decode forward pass begins
    # Note: This is NOT the same as prefill_start_time. There may be a delay between
    # forward_entry_time and prefill_start_time (see prefill_launch_delay).
    forward_entry_time: Optional[List[Optional[float]]]

    # Prefill launch delay: time spent waiting between forward entry and prefill start.
    # Calculated as: prefill_start_time - forward_entry_time
    # This represents the delay between when the request enters the forward stage
    # and when prefill computation actually begins.
    prefill_launch_delay: Optional[List[Optional[float]]]

    # Prefill launch latency: time spent during prefill kernel launch.
    # Calculated as: prefill_end_time_host - prefill_start_time_host
    prefill_launch_latency: Optional[List[Optional[float]]]


@dataclass
class SpeculativeDecodingMetricsMixin:
    """
    Mixin class containing speculative decoding metrics.

    This class consolidates speculative decoding metrics that are shared across
    batch output types that support speculative decoding to avoid code duplication.
    """

    # Verify count: number of verification forward passes
    spec_verify_ct: List[int]

    # Accepted tokens: Number of accepted tokens during speculative decoding
    spec_accepted_tokens: List[int]


# Parameters for a session
@dataclass
class SessionParams:
    id: Optional[str] = None
    rid: Optional[str] = None
    offset: Optional[int] = None
    replace: Optional[bool] = None
    drop_previous_output: Optional[bool] = None


# Type definitions for multimodal input data
# Individual data item types for each modality
ImageDataInputItem = Union[Image, str, ImageData, Dict]
AudioDataInputItem = Union[str, Dict]
VideoDataInputItem = Union[str, Dict]
# Union type for any multimodal data item
MultimodalDataInputItem = Union[
    ImageDataInputItem, VideoDataInputItem, AudioDataInputItem
]
# Format types supporting single items, lists, or nested lists for batch processing
MultimodalDataInputFormat = Union[
    List[List[MultimodalDataInputItem]],
    List[MultimodalDataInputItem],
    MultimodalDataInputItem,
]


@dataclass
class GenerateReqInput(BaseReq):
    # The input prompt. It can be a single prompt or a batch of prompts.
    text: Optional[Union[List[str], str]] = None
    # The token ids for text; one can specify either text or input_ids
    input_ids: Optional[Union[List[List[int]], List[int]]] = None
    # The embeddings for input_ids; one can specify either text or input_ids or input_embeds.
    input_embeds: Optional[Union[List[List[List[float]]], List[List[float]]]] = None
    # The image input. It can be an image instance, file name, URL, or base64 encoded string.
    # Can be formatted as:
    # - Single image for a single request
    # - List of images (one per request in a batch)
    # - List of lists of images (multiple images per request)
    # See also python/sglang/srt/utils.py:load_image for more details.
    image_data: Optional[MultimodalDataInputFormat] = None
    # The video input. Like image data, it can be a file name, a url, or base64 encoded string.
    video_data: Optional[MultimodalDataInputFormat] = None
    # The audio input. Like image data, it can be a file name, a url, or base64 encoded string.
    audio_data: Optional[MultimodalDataInputFormat] = None
    # The sampling_params. See descriptions below.
    sampling_params: Optional[Union[List[Dict], Dict]] = None
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
    # Whether to return hidden states
    return_hidden_states: Union[List[bool], bool] = False

    # The modalities of the image data [image, multi-images, video]
    modalities: Optional[List[str]] = None
    # Session info for continual prompting
    session_params: Optional[Union[List[Dict], Dict]] = None

    # The path to the LoRA adaptors
    lora_path: Optional[Union[List[Optional[str]], Optional[str]]] = None
    # The uid of LoRA adaptors, should be initialized by tokenizer manager
    lora_id: Optional[Union[List[Optional[str]], Optional[str]]] = None

    # Custom logit processor for advanced sampling control. Must be a serialized instance
    # of `CustomLogitProcessor` in python/sglang/srt/sampling/custom_logit_processor.py
    # Use the processor's `to_str()` method to generate the serialized string.
    custom_logit_processor: Optional[Union[List[Optional[str]], str]] = None

    # For disaggregated inference
    bootstrap_host: Optional[Union[List[str], str]] = None
    bootstrap_port: Optional[Union[List[Optional[int]], int]] = None
    bootstrap_room: Optional[Union[List[int], int]] = None
    bootstrap_pair_key: Optional[Union[List[str], str]] = None

    # Validation step duration
    validation_time: Optional[float] = None

    # For data parallel rank routing
    data_parallel_rank: Optional[int] = None

    # For background responses (OpenAI responses API)
    background: bool = False

    # Conversation id used for tracking requests
    conversation_id: Optional[str] = None

    # Priority for the request
    priority: Optional[int] = None

    # Extra key for classifying the request (e.g. cache_salt)
    extra_key: Optional[Union[List[str], str]] = None

    # Whether to disallow logging for this request (e.g. due to ZDR)
    no_logs: bool = False

    # For custom metric labels
    custom_labels: Optional[Dict[str, str]] = None

    # (Internal) Whether to return bytes for image generation
    return_bytes: bool = False

    # Whether to return entropy
    return_entropy: bool = False

    def contains_mm_input(self) -> bool:
        return (
            has_valid_data(self.image_data)
            or has_valid_data(self.video_data)
            or has_valid_data(self.audio_data)
        )

    def normalize_batch_and_arguments(self):
        """
        Normalize the batch size and arguments for the request.

        This method resolves various input formats and ensures all parameters
        are properly formatted as either single values or batches depending on the input.
        It also handles parallel sampling expansion and sets default values for
        unspecified parameters.

        Raises:
            ValueError: If inputs are not properly specified (e.g., none or all of
                       text, input_ids, input_embeds are provided)
        """
        self._validate_inputs()
        self._determine_batch_size()
        self._handle_parallel_sampling()

        if self.is_single:
            self._normalize_single_inputs()
        else:
            self._normalize_batch_inputs()

    def _validate_inputs(self):
        """Validate that the input configuration is valid."""
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

    def _determine_batch_size(self):
        """Determine if this is a single example or a batch and the batch size."""
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
                self.is_single = False
                self.batch_size = len(self.input_embeds)

    def _handle_parallel_sampling(self):
        """Handle parallel sampling parameters and adjust batch size if needed."""
        # Determine parallel sample count
        if self.sampling_params is None:
            self.parallel_sample_num = 1
            return
        elif isinstance(self.sampling_params, dict):
            self.parallel_sample_num = self.sampling_params.get("n", 1)
        else:  # isinstance(self.sampling_params, list):
            self.parallel_sample_num = self.sampling_params[0].get("n", 1)
            for sampling_params in self.sampling_params:
                if self.parallel_sample_num != sampling_params.get("n", 1):
                    raise ValueError(
                        "The parallel_sample_num should be the same for all samples in sample params."
                    )

        # If using parallel sampling with a single example, convert to batch
        if self.parallel_sample_num > 1 and self.is_single:
            self.is_single = False
            if self.text is not None:
                self.text = [self.text]
            if self.input_ids is not None:
                self.input_ids = [self.input_ids]
            if self.input_embeds is not None:
                self.input_embeds = [self.input_embeds]

    def _normalize_single_inputs(self):
        """Normalize inputs for a single example."""
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

    def _normalize_batch_inputs(self):
        """Normalize inputs for a batch of examples, including parallel sampling expansion."""
        # Calculate expanded batch size
        if self.parallel_sample_num == 1:
            num = self.batch_size
        else:
            # Expand parallel_sample_num
            num = self.batch_size * self.parallel_sample_num

        # Expand input based on type
        self._expand_inputs(num)
        self._normalize_rid(num)
        self._normalize_lora_paths(num)
        self._normalize_image_data(num)
        self._normalize_video_data(num)
        self._normalize_audio_data(num)
        self._normalize_sampling_params(num)
        self._normalize_logprob_params(num)
        self._normalize_custom_logit_processor(num)
        self._normalize_bootstrap_params(num)

    def _expand_inputs(self, num):
        """Expand the main inputs (text, input_ids, input_embeds) for parallel sampling."""
        if self.text is not None:
            if not isinstance(self.text, list):
                raise ValueError("Text should be a list for batch processing.")
            self.text = self.text * self.parallel_sample_num
        elif self.input_ids is not None:
            if not isinstance(self.input_ids, list) or not isinstance(
                self.input_ids[0], list
            ):
                raise ValueError(
                    "input_ids should be a list of lists for batch processing."
                )
            self.input_ids = self.input_ids * self.parallel_sample_num
        elif self.input_embeds is not None:
            if not isinstance(self.input_embeds, list):
                raise ValueError("input_embeds should be a list for batch processing.")
            self.input_embeds = self.input_embeds * self.parallel_sample_num

    def _normalize_lora_paths(self, num):
        """Normalize LoRA paths for batch processing."""
        if self.lora_path is not None:
            if isinstance(self.lora_path, str):
                self.lora_path = [self.lora_path] * num
            elif isinstance(self.lora_path, list):
                self.lora_path = self.lora_path * self.parallel_sample_num
            else:
                raise ValueError("lora_path should be a list or a string.")

    def _normalize_image_data(self, num):
        """Normalize image data for batch processing."""
        if self.image_data is None:
            self.image_data = [None] * num
        elif not isinstance(self.image_data, list):
            # Single image, convert to list of single-image lists
            self.image_data = [[self.image_data]] * num
            self.modalities = ["image"] * num
        elif isinstance(self.image_data, list):
            # Handle empty list case - treat as no images
            if len(self.image_data) == 0:
                self.image_data = [None] * num
                return

            if len(self.image_data) != self.batch_size:
                raise ValueError(
                    "The length of image_data should be equal to the batch size."
                )

            self.modalities = []
            if len(self.image_data) > 0 and isinstance(self.image_data[0], list):
                # Already a list of lists, keep as is
                for i in range(len(self.image_data)):
                    if self.image_data[i] is None or self.image_data[i] == [None]:
                        self.modalities.append(None)
                    elif len(self.image_data[i]) == 1:
                        self.modalities.append("image")
                    elif len(self.image_data[i]) > 1:
                        self.modalities.append("multi-images")
                    else:
                        # Ensure len(self.modalities) == len(self.image_data)
                        self.modalities.append(None)
                # Expand parallel_sample_num
                self.image_data = self.image_data * self.parallel_sample_num
                self.modalities = self.modalities * self.parallel_sample_num
            else:
                # List of images for a batch, wrap each in a list
                wrapped_images = [[img] for img in self.image_data]
                # Expand for parallel sampling
                self.image_data = wrapped_images * self.parallel_sample_num
                self.modalities = ["image"] * num

    def _normalize_video_data(self, num):
        """Normalize video data for batch processing."""
        if self.video_data is None:
            self.video_data = [None] * num
        elif not isinstance(self.video_data, list):
            self.video_data = [self.video_data] * num
        elif isinstance(self.video_data, list):
            self.video_data = self.video_data * self.parallel_sample_num

    def _normalize_audio_data(self, num):
        """Normalize audio data for batch processing."""
        if self.audio_data is None:
            self.audio_data = [None] * num
        elif not isinstance(self.audio_data, list):
            self.audio_data = [self.audio_data] * num
        elif isinstance(self.audio_data, list):
            self.audio_data = self.audio_data * self.parallel_sample_num

    def _normalize_sampling_params(self, num):
        """Normalize sampling parameters for batch processing."""
        if self.sampling_params is None:
            self.sampling_params = [{}] * num
        elif isinstance(self.sampling_params, dict):
            self.sampling_params = [self.sampling_params] * num
        else:  # Already a list
            self.sampling_params = self.sampling_params * self.parallel_sample_num

    def _normalize_rid(self, num):
        """Normalize request IDs for batch processing."""
        if self.rid is None:
            self.rid = [uuid.uuid4().hex for _ in range(num)]
        elif isinstance(self.rid, str):
            new_rids = [f"{self.rid}_{i}" for i in range(num)]
            self.rid = new_rids
        elif isinstance(self.rid, list):
            # Note: the length of rid shall be the same as the batch_size,
            # as the rid would be expanded for parallel sampling in tokenizer_manager
            if len(self.rid) != self.batch_size:
                raise ValueError(
                    "The specified rids length mismatch with the batch_size for batch processing."
                )
        else:
            raise ValueError("The rid should be a string or a list of strings.")

    def _normalize_logprob_params(self, num):
        """Normalize logprob-related parameters for batch processing."""

        # Helper function to normalize a parameter
        def normalize_param(param, default_value, param_name):
            if param is None:
                return [default_value] * num
            elif not isinstance(param, list):
                return [param] * num
            else:
                if self.parallel_sample_num > 1:
                    raise ValueError(
                        f"Cannot use list {param_name} with parallel_sample_num > 1"
                    )
                return param

        # Normalize each logprob parameter
        self.return_logprob = normalize_param(
            self.return_logprob, False, "return_logprob"
        )
        self.logprob_start_len = normalize_param(
            self.logprob_start_len, -1, "logprob_start_len"
        )
        self.top_logprobs_num = normalize_param(
            self.top_logprobs_num, 0, "top_logprobs_num"
        )

        # Handle token_ids_logprob specially due to its nested structure
        if not self.token_ids_logprob:  # covers both None and []
            self.token_ids_logprob = [None] * num
        elif not isinstance(self.token_ids_logprob, list):
            self.token_ids_logprob = [[self.token_ids_logprob] for _ in range(num)]
        elif not isinstance(self.token_ids_logprob[0], list):
            self.token_ids_logprob = [
                copy.deepcopy(self.token_ids_logprob) for _ in range(num)
            ]
        elif self.parallel_sample_num > 1:
            raise ValueError(
                "Cannot use list token_ids_logprob with parallel_sample_num > 1"
            )

    def _normalize_custom_logit_processor(self, num):
        """Normalize custom logit processor for batch processing."""
        if self.custom_logit_processor is None:
            self.custom_logit_processor = [None] * num
        elif not isinstance(self.custom_logit_processor, list):
            self.custom_logit_processor = [self.custom_logit_processor] * num
        elif self.parallel_sample_num > 1:
            raise ValueError(
                "Cannot use list custom_logit_processor with parallel_sample_num > 1"
            )

    def _normalize_bootstrap_params(self, num):
        """Normalize bootstrap parameters for batch processing."""
        # Normalize bootstrap_host
        if self.bootstrap_host is None:
            self.bootstrap_host = [None] * num
        elif not isinstance(self.bootstrap_host, list):
            self.bootstrap_host = [self.bootstrap_host] * num
        elif isinstance(self.bootstrap_host, list):
            self.bootstrap_host = self.bootstrap_host * self.parallel_sample_num

        # Normalize bootstrap_port
        if self.bootstrap_port is None:
            self.bootstrap_port = [None] * num
        elif not isinstance(self.bootstrap_port, list):
            self.bootstrap_port = [self.bootstrap_port] * num
        elif isinstance(self.bootstrap_port, list):
            self.bootstrap_port = self.bootstrap_port * self.parallel_sample_num

        # Normalize bootstrap_room
        if self.bootstrap_room is None:
            self.bootstrap_room = [None] * num
        elif not isinstance(self.bootstrap_room, list):
            self.bootstrap_room = [self.bootstrap_room + i for i in range(num)]
        elif isinstance(self.bootstrap_room, list):
            self.bootstrap_room = self.bootstrap_room * self.parallel_sample_num

        # Normalize bootstrap_pair_key
        if self.bootstrap_pair_key is None:
            self.bootstrap_pair_key = [None] * num
        elif not isinstance(self.bootstrap_pair_key, list):
            self.bootstrap_pair_key = [self.bootstrap_pair_key] * num
        elif isinstance(self.bootstrap_pair_key, list):
            self.bootstrap_pair_key = self.bootstrap_pair_key * self.parallel_sample_num

    def _validate_session_params(self):
        """Validate that session parameters are properly formatted."""
        if self.session_params is not None:
            if not isinstance(self.session_params, dict) and not isinstance(
                self.session_params[0], dict
            ):
                raise ValueError("Session params must be a dict or a list of dicts.")

    def __getitem__(self, i):
        return GenerateReqInput(
            text=self.text[i] if self.text is not None else None,
            input_ids=self.input_ids[i] if self.input_ids is not None else None,
            input_embeds=(
                self.input_embeds[i] if self.input_embeds is not None else None
            ),
            image_data=self.image_data[i],
            video_data=self.video_data[i],
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
            return_hidden_states=(
                self.return_hidden_states[i]
                if isinstance(self.return_hidden_states, list)
                else self.return_hidden_states
            ),
            modalities=self.modalities[i] if self.modalities else None,
            session_params=self.session_params,
            lora_path=self.lora_path[i] if self.lora_path is not None else None,
            lora_id=self.lora_id[i] if self.lora_id is not None else None,
            custom_logit_processor=(
                self.custom_logit_processor[i]
                if self.custom_logit_processor is not None
                else None
            ),
            # if `__getitem__` is called, the bootstrap_host, bootstrap_port, bootstrap_room must be a list
            bootstrap_host=(
                self.bootstrap_host[i] if self.bootstrap_host is not None else None
            ),
            bootstrap_port=(
                self.bootstrap_port[i] if self.bootstrap_port is not None else None
            ),
            bootstrap_room=(
                self.bootstrap_room[i] if self.bootstrap_room is not None else None
            ),
            bootstrap_pair_key=(
                self.bootstrap_pair_key[i]
                if self.bootstrap_pair_key is not None
                else None
            ),
            validation_time=self.validation_time,
            data_parallel_rank=(
                self.data_parallel_rank if self.data_parallel_rank is not None else None
            ),
            conversation_id=self.conversation_id,
            priority=self.priority,
            extra_key=self.extra_key,
            no_logs=self.no_logs,
            custom_labels=self.custom_labels,
            return_bytes=self.return_bytes,
            return_entropy=self.return_entropy,
            http_worker_ipc=self.http_worker_ipc,
        )


@dataclass
class TokenizedGenerateReqInput(BaseReq):
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

    # Whether to return hidden states
    return_hidden_states: bool = False

    # The input embeds
    input_embeds: Optional[Union[List[List[List[float]]], List[List[float]]]] = None

    # Session info for continual prompting
    session_params: Optional[SessionParams] = None

    # LoRA related
    lora_id: Optional[str] = None  # None means just use the base model

    # Custom logit processor for advanced sampling control. Must be a serialized instance
    # of `CustomLogitProcessor` in python/sglang/srt/sampling/custom_logit_processor.py
    # Use the processor's `to_str()` method to generate the serialized string.
    custom_logit_processor: Optional[str] = None

    # For disaggregated inference
    bootstrap_host: Optional[str] = None
    bootstrap_port: Optional[int] = None
    bootstrap_room: Optional[int] = None
    bootstrap_pair_key: Optional[str] = None

    # For data parallel rank routing
    data_parallel_rank: Optional[int] = None

    # Priority for the request
    priority: Optional[int] = None

    # Extra key for classifying the request (e.g. cache_salt)
    extra_key: Optional[str] = None

    # Whether to disallow logging for this request (e.g. due to ZDR)
    no_logs: bool = False

    # tracing context
    trace_context: Optional[Dict] = None

    # (Internal) Whether to return bytes for image generation
    return_bytes: bool = False

    # Whether to return entropy
    return_entropy: bool = False


@dataclass
class BatchTokenizedGenerateReqInput(BaseBatchReq):
    # The batch of tokenized requests
    batch: List[TokenizedGenerateReqInput]

    def __len__(self):
        return len(self.batch)

    def __getitem__(self, i):
        return self.batch[i]

    def __iter__(self):
        return iter(self.batch)


@dataclass
class EmbeddingReqInput(BaseReq):
    # The input prompt. It can be a single prompt or a batch of prompts.
    text: Optional[Union[List[List[str]], List[str], str]] = None
    # The image input. It can be an image instance, file name, URL, or base64 encoded string.
    # Can be formatted as:
    # - Single image for a single request
    # - List of images (one per request in a batch)
    # - List of lists of images (multiple images per request)
    # See also python/sglang/srt/utils.py:load_image for more details.
    image_data: Optional[MultimodalDataInputFormat] = None
    # The video input. Like image data, it can be a file name, a url, or base64 encoded string.
    video_data: Optional[MultimodalDataInputFormat] = None
    # The audio input. Like image data, it can be a file name, a url, or base64 encoded string.
    audio_data: Optional[MultimodalDataInputFormat] = None
    # The token ids for text; one can either specify text or input_ids.
    input_ids: Optional[Union[List[List[int]], List[int]]] = None
    # Dummy sampling params for compatibility
    sampling_params: Optional[Union[List[Dict], Dict]] = None
    # Dummy input embeds for compatibility
    input_embeds: Optional[Union[List[List[List[float]]], List[List[float]]]] = None
    # Whether to log metrics for this request (e.g. health_generate calls do not log metrics)
    log_metrics: bool = True
    # The modalities of the image data [image, multi-images, video]
    modalities: Optional[List[str]] = None
    # Validation step duration
    validation_time: Optional[float] = None
    # For cross-encoder requests
    is_cross_encoder_request: bool = False
    # Priority for the request
    priority: Optional[int] = None

    # For background responses (OpenAI responses API)
    background: bool = False

    # tracing context
    trace_context: Optional[Dict] = None

    # The number of dimensions the resulting output embeddings should have. It is applicable for Matryoshka Embeddings.
    dimensions: Optional[int] = None

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
                self.is_single = False
            else:
                self.batch_size += 1

        # check the batch size of input_ids
        if self.input_ids is not None:
            if isinstance(self.input_ids[0], list):
                self.batch_size += len(self.input_ids)
                self.is_single = False
            else:
                self.batch_size += 1

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
            elif isinstance(self.sampling_params, dict):
                self.sampling_params = [self.sampling_params] * self.batch_size
            for i in range(self.batch_size):
                self.sampling_params[i]["max_new_tokens"] = 0

    def contains_mm_input(self) -> bool:
        return (
            has_valid_data(self.image_data)
            or has_valid_data(self.video_data)
            or has_valid_data(self.audio_data)
        )

    def __getitem__(self, i):
        if self.is_cross_encoder_request:
            return EmbeddingReqInput(
                text=[self.text[i]] if self.text is not None else None,
                sampling_params=self.sampling_params[i],
                rid=self.rid[i],
                is_cross_encoder_request=True,
                http_worker_ipc=self.http_worker_ipc,
            )

        return EmbeddingReqInput(
            text=self.text[i] if self.text is not None else None,
            input_ids=self.input_ids[i] if self.input_ids is not None else None,
            image_data=self.image_data[i] if self.image_data is not None else None,
            audio_data=self.audio_data[i] if self.audio_data is not None else None,
            video_data=self.video_data[i] if self.video_data is not None else None,
            sampling_params=self.sampling_params[i],
            rid=self.rid[i],
            validation_time=self.validation_time,
            dimensions=self.dimensions,
            http_worker_ipc=self.http_worker_ipc,
        )


@dataclass
class TokenizedEmbeddingReqInput(BaseReq):
    # The input text
    input_text: str
    # The input token ids
    input_ids: List[int]
    # The image inputs
    image_inputs: dict
    # The token type ids
    token_type_ids: List[int]
    # Dummy sampling params for compatibility
    sampling_params: SamplingParams
    # For data parallel rank routing
    data_parallel_rank: Optional[int] = None
    # Priority for the request
    priority: Optional[int] = None
    # The number of dimensions the resulting output embeddings should have. It is applicable for Matryoshka Embeddings.
    dimensions: Optional[int] = None


@dataclass
class BatchTokenizedEmbeddingReqInput(BaseBatchReq):
    # The batch of tokenized embedding requests
    batch: List[TokenizedEmbeddingReqInput]

    def __len__(self):
        return len(self.batch)

    def __getitem__(self, i):
        return self.batch[i]

    def __iter__(self):
        return iter(self.batch)


@dataclass
class BatchTokenIDOutput(
    BaseBatchReq, RequestTimingMetricsMixin, SpeculativeDecodingMetricsMixin
):
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
    output_token_entropy_val: List[float]

    # Hidden states
    output_hidden_states: List[List[float]]

    # The information of placeholder tokens (e.g., image token)
    # idx is the index of the token in the prompt after expansion.
    # val is the length of padded tokens after expansion.
    placeholder_tokens_idx: List[Optional[List[int]]]
    placeholder_tokens_val: List[Optional[List[int]]]

    # Number of times each request was retracted.
    retraction_counts: List[int]

    # The trainer step id. Used to know which step's weights are used for sampling.
    token_steps: List[List[int]] = None


@dataclass
class BatchMultimodalDecodeReq(BaseBatchReq):
    decoded_ids: List[int]
    input_token_logprobs_val: List[float]
    input_token_logprobs_idx: List[int]
    output_token_logprobs_val: List[float]
    output_token_logprobs_idx: List[int]
    read_offsets: List[int]
    skip_special_tokens: List[bool]
    spaces_between_special_tokens: List[bool]
    image_resolutions: List[List[int]]
    resize_image_resolutions: List[List[int]]

    finished_reasons: List[BaseFinishReason]

    # Token counts
    prompt_tokens: List[int]
    completion_tokens: List[int]
    cached_tokens: List[int]

    # The information of placeholder tokens (e.g., image token)
    # idx is the index of the token in the prompt after expansion.
    # val is the length of padded tokens after expansion.
    placeholder_tokens_idx: List[Optional[List[int]]]
    placeholder_tokens_val: List[Optional[List[int]]]

    return_bytes: List[bool]

    # The trainer step id. Used to know which step's weights are used for sampling.
    token_steps: List[List[int]] = None


@dataclass
class BatchStrOutput(
    BaseBatchReq, RequestTimingMetricsMixin, SpeculativeDecodingMetricsMixin
):
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
    output_token_entropy_val: List[float]

    # Hidden states
    output_hidden_states: List[List[float]]

    # The information of placeholder tokens (e.g., image token)
    # idx is the index of the token in the prompt after expansion.
    # val is the length of padded tokens after expansion.
    placeholder_tokens_idx: List[Optional[List[int]]]
    placeholder_tokens_val: List[Optional[List[int]]]

    # Number of times each request was retracted.
    retraction_counts: List[int]

    # The trainer step id. Used to know which step's weights are used for sampling.
    token_steps: List[List[int]] = None


@dataclass
class BatchMultimodalOutput(BaseBatchReq):
    # The finish reason
    finished_reasons: List[dict]
    decoded_ids: List[List[int]]
    # The outputs
    outputs: Union[List[str | bytes], List[List[Dict]]]

    # probability values for input tokens and output tokens
    input_token_logprobs_val: List[List[float]]
    input_token_logprobs_idx: List[List[int]]
    output_token_logprobs_val: List[List[float]]
    output_token_logprobs_idx: List[List[int]]

    # Token counts
    prompt_tokens: List[int]
    completion_tokens: List[int]
    cached_tokens: List[int]

    placeholder_tokens_idx: List[Optional[List[int]]]
    placeholder_tokens_val: List[Optional[List[int]]]

    return_bytes: List[bool]


@dataclass
class BatchEmbeddingOutput(BaseBatchReq, RequestTimingMetricsMixin):
    # The finish reason
    finished_reasons: List[BaseFinishReason]
    # The output embedding
    embeddings: Union[List[List[float]], List[Dict[int, float]]]
    # Token counts
    prompt_tokens: List[int]
    cached_tokens: List[int]
    # Placeholder token info
    placeholder_tokens_idx: List[Optional[List[int]]]
    placeholder_tokens_val: List[Optional[List[int]]]

    # Number of times each request was retracted.
    retraction_counts: List[int]


@dataclass
class ClearHiCacheReqInput(BaseReq):
    pass


@dataclass
class ClearHiCacheReqOutput(BaseReq):
    success: bool


@dataclass
class FlushCacheReqInput(BaseReq):
    pass


@dataclass
class FlushCacheReqOutput(BaseReq):
    success: bool


@dataclass
class UpdateWeightFromDiskReqInput(BaseReq):
    # The model path with the new weights
    model_path: str
    # The format to load the weights
    load_format: Optional[str] = None
    # Whether to abort all requests before updating weights
    abort_all_requests: bool = False
    # Optional: Update weight version along with weights
    weight_version: Optional[str] = None
    # Whether to update weights asynchronously
    is_async: bool = False
    # Whether to empty torch cache
    torch_empty_cache: bool = False
    # Whether to keep the scheduler paused after weight update
    keep_pause: bool = False
    # Whether to recapture cuda graph after weight udpdate
    recapture_cuda_graph: bool = False
    # The trainer step id. Used to know which step's weights are used for sampling.
    token_step: int = 0


@dataclass
class UpdateWeightFromDiskReqOutput(BaseReq):
    success: bool
    message: str
    # Number of paused requests during weight sync.
    num_paused_requests: Optional[int] = 0


@dataclass
class UpdateWeightsFromDistributedReqInput(BaseReq):
    names: List[str]
    dtypes: List[str]
    shapes: List[List[int]]
    # The group name
    group_name: str = "weight_update_group"
    # Whether to flush the cache after updating weights
    flush_cache: bool = True
    # Whether to abort all requests before updating weights
    abort_all_requests: bool = False
    # Optional: Update weight version along with weights
    weight_version: Optional[str] = None


@dataclass
class UpdateWeightsFromDistributedReqOutput(BaseReq):
    success: bool
    message: str


@dataclass
class UpdateWeightsFromTensorReqInput(BaseReq):
    """Update model weights from tensor input.

    - Tensors are serialized for transmission
    - Data is structured in JSON for easy transmission over HTTP
    """

    serialized_named_tensors: List[Union[str, bytes]]
    # Optional format specification for loading
    load_format: Optional[str] = None
    # Whether to flush the cache after updating weights
    flush_cache: bool = True
    # Whether to abort all requests before updating weights
    abort_all_requests: bool = False
    # Optional: Update weight version along with weights
    weight_version: Optional[str] = None


@dataclass
class UpdateWeightsFromTensorReqOutput(BaseReq):
    success: bool
    message: str


@dataclass
class InitWeightsSendGroupForRemoteInstanceReqInput(BaseReq):
    # The master address
    master_address: str
    # The ports for each rank's communication group
    ports: str
    # The rank in the communication group
    group_rank: int
    # The world size
    world_size: int
    # The group name
    group_name: str = "weight_send_group"
    # The backend
    backend: str = "nccl"


# Now UpdateWeightsFromIPCReqInput and UpdateWeightsFromIPCReqOutput
# are only used by Checkpoint Engine (https://github.com/MoonshotAI/checkpoint-engine)
@dataclass
class UpdateWeightsFromIPCReqInput(BaseReq):
    # ZMQ socket paths for each device UUID
    zmq_handles: Dict[str, str]
    # Whether to flush cache after weight update
    flush_cache: bool = True
    # Optional: Update weight version along with weights
    weight_version: Optional[str] = None


@dataclass
class UpdateWeightsFromIPCReqOutput(BaseReq):
    success: bool
    message: str


@dataclass
class InitWeightsSendGroupForRemoteInstanceReqOutput(BaseReq):
    success: bool
    message: str


@dataclass
class SendWeightsToRemoteInstanceReqInput(BaseReq):
    # The master address
    master_address: str
    # The ports for each rank's communication group
    ports: str
    # The group name
    group_name: str = "weight_send_group"


@dataclass
class SendWeightsToRemoteInstanceReqOutput(BaseReq):
    success: bool
    message: str


@dataclass
class InitWeightsUpdateGroupReqInput(BaseReq):
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
class InitWeightsUpdateGroupReqOutput(BaseReq):
    success: bool
    message: str


@dataclass
class DestroyWeightsUpdateGroupReqInput(BaseReq):
    group_name: str = "weight_update_group"


@dataclass
class DestroyWeightsUpdateGroupReqOutput(BaseReq):
    success: bool
    message: str


@dataclass
class UpdateWeightVersionReqInput(BaseReq):
    # The new weight version
    new_version: str
    # Whether to abort all running requests before updating
    abort_all_requests: bool = True


@dataclass
class GetWeightsByNameReqInput(BaseReq):
    name: str
    truncate_size: int = 100


@dataclass
class GetWeightsByNameReqOutput(BaseReq):
    parameter: list


@dataclass
class ReleaseMemoryOccupationReqInput(BaseReq):
    # Optional tags to identify the memory region, which is primarily used for RL
    # Currently we only support `weights` and `kv_cache`
    tags: Optional[List[str]] = None


@dataclass
class ReleaseMemoryOccupationReqOutput(BaseReq):
    pass


@dataclass
class ResumeMemoryOccupationReqInput(BaseReq):
    # Optional tags to identify the memory region, which is primarily used for RL
    # Currently we only support `weights` and `kv_cache`
    tags: Optional[List[str]] = None


@dataclass
class ResumeMemoryOccupationReqOutput(BaseReq):
    pass


@dataclass
class SlowDownReqInput(BaseReq):
    forward_sleep_time: Optional[float]


@dataclass
class SlowDownReqOutput(BaseReq):
    pass


@dataclass
class AbortReq(BaseReq):
    # Whether to abort all requests
    abort_all: bool = False
    # The finished reason data
    finished_reason: Optional[Dict[str, Any]] = None
    abort_message: Optional[str] = None

    def __post_init__(self):
        # FIXME: This is a hack to keep the same with the old code
        if self.rid is None:
            self.rid = ""


@dataclass
class GetInternalStateReq(BaseReq):
    pass


@dataclass
class GetInternalStateReqOutput(BaseReq):
    internal_state: Dict[Any, Any]


@dataclass
class SetInternalStateReq(BaseReq):
    server_args: Dict[str, Any]


@dataclass
class SetInternalStateReqOutput(BaseReq):
    updated: bool
    server_args: Dict[str, Any]


@dataclass
class ProfileReqInput(BaseReq):
    # The output directory
    output_dir: Optional[str] = None
    # If set, it profile as many as this number of steps.
    # If it is set, profiling is automatically stopped after this step, and
    # the caller doesn't need to run stop_profile.
    start_step: Optional[int] = None
    num_steps: Optional[int] = None
    activities: Optional[List[str]] = None
    profile_by_stage: bool = False
    with_stack: Optional[bool] = None
    record_shapes: Optional[bool] = None
    # Merge profiles from all ranks into a single trace
    merge_profiles: bool = False


class ProfileReqType(Enum):
    START_PROFILE = 1
    STOP_PROFILE = 2


@dataclass
class ProfileReq(BaseReq):
    type: ProfileReqType
    output_dir: Optional[str] = None
    start_step: Optional[int] = None
    num_steps: Optional[int] = None
    activities: Optional[List[str]] = None
    profile_by_stage: bool = False
    with_stack: Optional[bool] = None
    record_shapes: Optional[bool] = None
    profile_id: Optional[str] = None
    # Merge profiles from all ranks into a single trace
    merge_profiles: bool = False


@dataclass
class ProfileReqOutput(BaseReq):
    success: bool
    message: str


@dataclass
class FreezeGCReq(BaseReq):
    pass


@dataclass
class ConfigureLoggingReq(BaseReq):
    log_requests: Optional[bool] = None
    log_requests_level: Optional[int] = None
    dump_requests_folder: Optional[str] = None
    dump_requests_threshold: Optional[int] = None
    crash_dump_folder: Optional[str] = None


@dataclass
class OpenSessionReqInput(BaseReq):
    capacity_of_str_len: int
    session_id: Optional[str] = None


@dataclass
class CloseSessionReqInput(BaseReq):
    session_id: str


@dataclass
class OpenSessionReqOutput(BaseReq):
    session_id: Optional[str]
    success: bool


@dataclass
class HealthCheckOutput(BaseReq):
    pass


class ExpertDistributionReqType(Enum):
    START_RECORD = 1
    STOP_RECORD = 2
    DUMP_RECORD = 3


@dataclass
class ExpertDistributionReq(BaseReq):
    action: ExpertDistributionReqType


@dataclass
class ExpertDistributionReqOutput(BaseReq):
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
class ParseFunctionCallReq(BaseReq):
    text: str  # The text to parse.
    tools: List[Tool] = field(
        default_factory=list
    )  # A list of available function tools (name, parameters, etc.).
    tool_call_parser: Optional[str] = (
        None  # Specify the parser type, e.g. 'llama3', 'qwen25', or 'mistral'. If not specified, tries all.
    )


@dataclass
class SeparateReasoningReqInput(BaseReq):
    text: str  # The text to parse.
    reasoning_parser: str  # Specify the parser type, e.g., "deepseek-r1".


@dataclass
class VertexGenerateReqInput(BaseReq):
    instances: List[dict]
    parameters: Optional[dict] = None


@dataclass
class RpcReqInput(BaseReq):
    method: str
    parameters: Optional[Dict] = None


@dataclass
class RpcReqOutput(BaseReq):
    success: bool
    message: str


@dataclass
class LoadLoRAAdapterReqInput(BaseReq):
    # The name of the lora module to newly loaded.
    lora_name: str
    # The path of loading.
    lora_path: str
    # Whether to pin the LoRA adapter in memory.
    pinned: bool = False
    # The unique identifier for the LoRA adapter, which automatically generated in the `TokenizerManager`.
    lora_id: Optional[str] = None

    def to_ref(self) -> LoRARef:
        return LoRARef(
            lora_id=self.lora_id,
            lora_name=self.lora_name,
            lora_path=self.lora_path,
            pinned=self.pinned,
        )


@dataclass
class UnloadLoRAAdapterReqInput(BaseReq):
    # The name of lora module to unload.
    lora_name: str
    # The unique identifier for the LoRA adapter, which automatically generated in the `TokenizerManager`.
    lora_id: Optional[str] = None

    def to_ref(self) -> LoRARef:
        return LoRARef(
            lora_id=self.lora_id,
            lora_name=self.lora_name,
        )


@dataclass
class LoRAUpdateOutput(BaseReq):
    success: bool
    error_message: Optional[str] = None
    loaded_adapters: Optional[Dict[str, LoRARef]] = None


LoadLoRAAdapterReqOutput = UnloadLoRAAdapterReqOutput = LoRAUpdateOutput


class BlockReqType(Enum):
    BLOCK = 1
    UNBLOCK = 2


@dataclass
class BlockReqInput(BaseReq):
    type: BlockReqType


@dataclass
class GetLoadReqInput(BaseReq):
    pass


@dataclass
class GetLoadReqOutput(BaseReq):
    dp_rank: int
    num_reqs: int
    num_waiting_reqs: int
    num_tokens: int


@dataclass
class WatchLoadUpdateReq(BaseReq):
    loads: List[GetLoadReqOutput]


@dataclass
class SetInjectDumpMetadataReqInput(BaseReq):
    dump_metadata: Dict[str, Any]


@dataclass
class SetInjectDumpMetadataReqOutput(BaseReq):
    success: bool


@dataclass
class LazyDumpTensorsReqInput(BaseReq):
    pass


@dataclass
class LazyDumpTensorsReqOutput(BaseReq):
    success: bool


def _check_all_req_types():
    """A helper function to check all request types are defined in this file."""
    import inspect
    import sys

    all_classes = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    for class_type in all_classes:
        # check its name
        name = class_type[0]
        is_io_struct = (
            name.endswith("Req") or name.endswith("Input") or name.endswith("Output")
        )
        is_base_req = issubclass(class_type[1], BaseReq) or issubclass(
            class_type[1], BaseBatchReq
        )
        if is_io_struct and not is_base_req:
            raise ValueError(f"{name} is not a subclass of BaseReq or BaseBatchReq.")
        if is_base_req and not is_io_struct:
            raise ValueError(
                f"{name} is a subclass of BaseReq but not follow the naming convention."
            )


_check_all_req_types()
