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

Keep this file focused on IPC struct definitions so it stays concise. Put
normalizers, helper utilities, and future non-struct logic in the owning module
instead, such as sglang.srt.utils.common.
"""

from __future__ import annotations

import copy
import logging
import pickle
import uuid
from array import array
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Type,
    Union,
)

import msgspec
import numpy as np
import torch
import zmq
import zmq.asyncio
from pydantic import PlainValidator

from sglang.srt.environ import envs
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.managers.embed_types import PositionalEmbeds
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.mm_utils import has_valid_data
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils import ImageData, VideoData
from sglang.srt.utils.field_validators import validate_optional_list_i64_1d_2d
from sglang.srt.utils.msgspec_utils import (
    Base64Bytes,
    msgspec_struct_pydantic_core_schema,
)

# Handle serialization of Image for pydantic
if TYPE_CHECKING:
    from PIL.Image import Image
else:
    Image = Any

logger = logging.getLogger(__name__)


class BaseReq(msgspec.Struct, tag=True, kw_only=True, array_like=True):
    """Base for single-request IPC payloads."""

    rid: Optional[str] = None
    http_worker_ipc: Optional[str] = None

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return msgspec_struct_pydantic_core_schema(cls, handler)


class BaseBatchReq(msgspec.Struct, tag=True, kw_only=True, array_like=True):
    """Base for batched IPC payloads."""

    rids: Optional[List[str]] = None
    # Used by batch messages whose items are parallel arrays, such as scheduler
    # outputs. Tokenized input batches store routing on batch[i].http_worker_ipc
    # because the scheduler unpacks them into single-request handlers.
    http_worker_ipcs: Optional[List[Optional[str]]] = None

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return msgspec_struct_pydantic_core_schema(cls, handler)


class PickleWrapper(msgspec.Struct, tag=True, array_like=True):
    """Wraps an arbitrary Python object as pickle-serialized bytes for msgpack IPC.

    In msgpack mode, fields that carry opaque or non-msgspec-typed payloads
    (e.g. multimodal inputs, time stats, customized info) are stored as
    PickleWrapper so the outer struct can still be msgpack-encoded.  In pickle
    mode (_USE_PICKLE_IPC=True), wrap_as_pickle / unwrap_from_pickle are no-ops
    and this class is not used on the wire.
    """

    data: bytes


# Parameters for a session
class SessionParams(msgspec.Struct, kw_only=True, array_like=True):
    # The session identifier. Used by the scheduler to look up or create the
    # Session object that groups all requests in a multi-turn conversation.
    id: Optional[str] = None
    # A request identifier *within* the session. In non-streaming sessions the
    # session maintains a tree of request nodes keyed by rid; this field selects
    # which node to continue from (append) or replace. When None the default
    # branch point is used (latest node for streaming, all nodes cleared on
    # replace).
    rid: Optional[str] = None
    # Token-level insertion point. When set, the new request's tokens are
    # spliced into the accumulated context at this position instead of being
    # appended at the end (i.e. ``context[:offset] + new_tokens``).
    offset: Optional[int] = None
    # When True, the request node identified by ``rid`` (or all nodes if
    # ``rid`` is None) is aborted and its children are cleared before the new
    # request is inserted. Not supported in streaming sessions.
    replace: Optional[bool] = None
    # When True, the previous request's generated output tokens are excluded
    # from the accumulated context so the new turn sees only the original input.
    # Not supported in streaming sessions.
    drop_previous_output: Optional[bool] = None


# Type definitions for multimodal input data
# Individual data item types for each modality
ImageDataInputItem = Union[str, bytes, Dict[str, Any], ImageData, Image]
AudioDataInputItem = Union[str, bytes, Dict[str, Any]]
VideoDataInputItem = Union[str, bytes, Dict[str, Any], VideoData]
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
class GenerateReqInput:
    # Request ID(s). If omitted, generated during normalization. For batch
    # requests, a string is expanded to per-item IDs using it as a prefix.
    rid: Optional[Union[str, List[str]]] = field(default=None, kw_only=True)
    # Stable identity shared by requests in the same session. Unlike
    # session_params, this does not alter or reconstruct the prompt.
    session_id: Optional[str] = field(default=None, kw_only=True)
    # The input prompt. It can be a single prompt or a batch of prompts.
    text: Optional[Union[List[str], str]] = None
    # The token ids for text.
    # Use C-loop validator to replace Pydantic per-element type check for efficiency.
    input_ids: Annotated[
        Optional[Union[List[List[int]], List[int]]],
        PlainValidator(validate_optional_list_i64_1d_2d),
    ] = None
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
    # Optional per-image hashes the caller has already computed (hex strings).
    # Single request: one hash per image. Batch request: either one hash per
    # request when each request has one image, or one list of hashes per request.
    # When supplied, each MultimodalDataItem's
    # `hash` is initialised from this list and `set_pad_value` skips the
    # internal `hash_feature()` recompute, so the resulting `pad_value` is
    # deterministic from the caller's hash. Intended for external KV routers
    # that compute their own per-image hash for routing decisions and need
    # sglang's prefix-cache key to align. When unset, behavior is unchanged
    # (sglang hashes the processor feature tensor).
    mm_hashes: Optional[Union[List[str], List[List[str]]]] = None
    # Whether to extract and process audio from video inputs.
    use_audio_in_video: bool = False
    # The sampling_params. See descriptions below.
    sampling_params: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None
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
    # Whether to return captured routed experts
    return_routed_experts: bool = False
    # Absolute start position for returned routings; response covers
    # `[routed_experts_start_len, seqlen - 1)`. Must be in [0, prompt_tokens].
    # 0 = full sequence.
    routed_experts_start_len: int = 0
    return_indexer_topk: bool = False

    # The modalities of the image data [image, multi-images, video]
    modalities: Optional[List[str]] = None
    # Session info for continual prompting
    session_params: Optional[Dict[str, Any]] = None

    # The path to the LoRA adaptors
    lora_path: Optional[Union[List[Optional[str]], str]] = None
    # The uid of LoRA adaptors, should be initialized by tokenizer manager
    lora_id: Optional[Union[List[Optional[str]], str]] = None

    # Custom logit processor for advanced sampling control. Must be a serialized instance
    # of `CustomLogitProcessor` in python/sglang/srt/sampling/custom_logit_processor.py
    # Use the processor's `to_str()` method to generate the serialized string.
    custom_logit_processor: Optional[Union[List[Optional[str]], str]] = None
    # Embedding overrides to place at specific token positions.
    # Runtime type: Optional[Union[PositionalEmbeds, List[Optional[PositionalEmbeds]]]]
    # Typed as Any to avoid Pydantic/FastAPI schema errors (PositionalEmbeds contains torch.Tensor).
    positional_embed_overrides: Any = None

    # For disaggregated inference
    bootstrap_host: Optional[Union[List[Optional[str]], str]] = None
    bootstrap_port: Optional[Union[List[Optional[int]], int]] = None
    bootstrap_room: Optional[Union[List[Optional[int]], int]] = None
    bootstrap_pair_key: Optional[Union[List[Optional[str]], str]] = None
    decode_tp_size: Optional[Union[List[Optional[int]], int]] = None

    # For DP routing — external router assigns a specific DP worker
    routed_dp_rank: Optional[int] = None
    # For PD disagg — hint telling decode which prefill DP worker has the KV cache
    disagg_prefill_dp_rank: Optional[int] = None
    # Routing key for routing-key schedule policy
    routing_key: Optional[str] = None
    # Conversation id used for tracking requests
    conversation_id: Optional[str] = None
    # Internal IPC endpoint of the HTTP/tokenizer worker that owns this request.
    # Used to route outputs back in multi-tokenizer mode.
    http_worker_ipc: Optional[str] = field(default=None, kw_only=True)

    # For background responses (OpenAI responses API)
    background: bool = False
    # Require reasoning for the request (hybrid reasoning model only)
    require_reasoning: bool = False

    # Priority for the request
    priority: Optional[int] = None
    # Extra cache key for classifying the request (e.g. cache_salt)
    extra_key: Optional[Union[List[str], str]] = None

    # Whether to disallow logging for this request (e.g. due to ZDR)
    no_logs: bool = False
    # For custom metric labels
    custom_labels: Optional[Dict[str, str]] = None

    # (Internal) Whether to return bytes for image generation
    return_bytes: bool = False
    # Whether to return entropy
    return_entropy: bool = False
    # Whether to return prompt token IDs without computing logprobs
    return_prompt_token_ids: bool = False

    # Propagates trace context via Engine.generate/async_generate
    external_trace_header: Optional[Dict[str, Any]] = None
    received_time: Optional[float] = None

    # For EPD-disaggregated inference
    need_wait_for_mm_inputs: Optional[bool] = None
    num_items_assigned: Optional[Dict[Modality, List[int]]] = None
    mm_data_mooncake: Optional[List[Any]] = None
    # Snapshot of encoder URLs at the time tokenizer-side computed
    # ``num_items_assigned``.
    encoder_urls: Optional[List[str]] = None

    # Multimodal tiling controls (extensions)
    max_dynamic_patch: Optional[int] = None
    min_dynamic_patch: Optional[int] = None
    image_max_dynamic_patch: Optional[int] = None
    video_max_dynamic_patch: Optional[int] = None

    # For Unlimited-OCR
    images_config: Optional[dict] = None

    # Pre-computed delimiter indices for multi-item scoring.
    # Batch-level: List[List[int]] (one per request). After __getitem__: List[int].
    multi_item_delimiter_indices: Optional[Union[List[List[int]], List[int]]] = None

    def regenerate_rid(self):
        """Generate a new request ID and return it."""
        if isinstance(self.rid, list):
            self.rid = [uuid.uuid4().hex for _ in range(len(self.rid))]
        else:
            self.rid = uuid.uuid4().hex
        return self.rid

    def _validate_rid_uniqueness(self):
        """Validate that request IDs within a batch are unique."""
        if isinstance(self.rid, list) and len(set(self.rid)) != len(self.rid):
            counts = Counter(self.rid)
            duplicates = [rid for rid, count in counts.items() if count > 1]
            raise ValueError(
                f"Duplicate request IDs detected within the request: {duplicates}"
            )

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
        if self.session_id is not None and self.session_params is not None:
            raise ValueError("session_id and session_params cannot both be set.")
        self._handle_parallel_sampling()

        if self.is_single:
            self._normalize_single_inputs()
        else:
            self._normalize_batch_inputs()

        self._validate_rid_uniqueness()

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
        self._normalize_extra_key(num)
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

    def _normalize_extra_key(self, num):
        """Normalize extra_key for batch processing."""
        if self.extra_key is None:
            return
        if isinstance(self.extra_key, str):
            self.extra_key = [self.extra_key] * num
        elif isinstance(self.extra_key, list):
            if len(self.extra_key) != self.batch_size:
                raise ValueError(
                    "The length of extra_key should be equal to the batch size."
                )
            self.extra_key = self.extra_key * self.parallel_sample_num
        else:
            raise ValueError("extra_key should be a list or a string.")

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

        # Normalize decode_tp_size
        if self.decode_tp_size is None:
            self.decode_tp_size = [None] * num
        elif not isinstance(self.decode_tp_size, list):
            self.decode_tp_size = [self.decode_tp_size] * num
        elif isinstance(self.decode_tp_size, list):
            self.decode_tp_size = self.decode_tp_size * self.parallel_sample_num

    def _get_positional_embed_overrides_item(
        self, i: int
    ) -> Optional[PositionalEmbeds]:
        """Extract the i-th item from positional_embed_overrides."""
        if self.positional_embed_overrides is None:
            return None
        if isinstance(self.positional_embed_overrides, PositionalEmbeds):
            return self.positional_embed_overrides
        return self.positional_embed_overrides[i]

    def __getitem__(self, i):
        # Cache sub-objects so that repeated obj[i] calls return the same instance.
        # This avoids subtle bugs where different call sites get divergent objects.
        cache = self.__dict__.setdefault("_sub_obj_cache", {})
        if i in cache:
            return cache[i]
        sub = GenerateReqInput(
            rid=self.rid[i],
            session_id=self.session_id,
            text=self.text[i] if self.text is not None else None,
            input_ids=self.input_ids[i] if self.input_ids is not None else None,
            input_embeds=(
                self.input_embeds[i] if self.input_embeds is not None else None
            ),
            image_data=self.image_data[i],
            video_data=self.video_data[i],
            audio_data=self.audio_data[i],
            sampling_params=self.sampling_params[i],
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
            return_routed_experts=self.return_routed_experts,
            routed_experts_start_len=self.routed_experts_start_len,
            return_indexer_topk=self.return_indexer_topk,
            modalities=self.modalities[i] if self.modalities else None,
            session_params=self.session_params,
            lora_path=self.lora_path[i] if self.lora_path is not None else None,
            lora_id=self.lora_id[i] if self.lora_id is not None else None,
            custom_logit_processor=(
                self.custom_logit_processor[i]
                if self.custom_logit_processor is not None
                else None
            ),
            positional_embed_overrides=self._get_positional_embed_overrides_item(i),
            # If `__getitem__` is called, these bootstrap fields must be lists.
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
            decode_tp_size=(
                self.decode_tp_size[i] if self.decode_tp_size is not None else None
            ),
            routed_dp_rank=self.routed_dp_rank,
            disagg_prefill_dp_rank=self.disagg_prefill_dp_rank,
            conversation_id=self.conversation_id,
            http_worker_ipc=self.http_worker_ipc,
            priority=self.priority,
            extra_key=self.extra_key[i] if self.extra_key is not None else None,
            no_logs=self.no_logs,
            custom_labels=self.custom_labels,
            return_bytes=self.return_bytes,
            return_entropy=self.return_entropy,
            return_prompt_token_ids=self.return_prompt_token_ids,
            external_trace_header=self.external_trace_header,
            received_time=self.received_time,
            multi_item_delimiter_indices=(
                self.multi_item_delimiter_indices[i]
                if self.multi_item_delimiter_indices is not None
                else None
            ),
        )
        cache[i] = sub
        return sub


class TokenizedGenerateReqInput(BaseReq, kw_only=True):
    input_text: Optional[Union[str, List[Union[str, List[str]]]]]
    # The input token ids
    input_ids: Optional[array]  # Optional[array[int]]
    # The input embeds
    input_embeds: Optional[List[List[float]]]
    # The multimodal inputs
    mm_inputs: Optional[PickleWrapper]  # Pickled Optional[MultimodalProcessorOutput]
    token_type_ids: Optional[List[int]]
    # The sampling parameters
    sampling_params: SamplingParams
    # Whether to return the logprobs
    return_logprob: bool
    # If return logprobs, the start location in the prompt for returning logprobs.
    logprob_start_len: int
    # If return logprobs, the number of top logprobs to return at each position.
    top_logprobs_num: int
    # If return logprobs, the token id to return logprob for
    token_ids_logprob: Optional[List[int]]
    # Whether to stream output
    stream: bool

    # Whether to return hidden states
    return_hidden_states: bool = False

    # Whether to return captured routed experts
    return_routed_experts: bool = False
    # See GenerateReqInput.routed_experts_start_len.
    routed_experts_start_len: int = 0
    return_indexer_topk: bool = False

    # Session info for continual prompting
    session_id: Optional[str] = None
    session_params: Optional[SessionParams] = None

    # LoRA related
    lora_id: Optional[str] = None  # None means just use the base model

    # Custom logit processor for advanced sampling control. Must be a serialized instance
    # of `CustomLogitProcessor` in python/sglang/srt/sampling/custom_logit_processor.py
    # Use the processor's `to_str()` method to generate the serialized string.
    custom_logit_processor: Optional[str] = None
    # Embedding overrides to place at specific token positions.
    positional_embed_overrides: Optional[PositionalEmbeds] = None

    # For disaggregated inference
    bootstrap_host: Optional[str] = None
    bootstrap_port: Optional[int] = None
    bootstrap_room: Optional[int] = None
    bootstrap_pair_key: Optional[str] = None
    decode_tp_size: Optional[int] = None

    # For DP routing
    routed_dp_rank: Optional[int] = None
    # For PD disagg — hint telling decode which prefill DP worker has the KV cache
    disagg_prefill_dp_rank: Optional[int] = None

    # Routing key for routing-key schedule policy
    routing_key: Optional[str] = None
    # Require reasoning for the request (hybrid reasoning model only)
    require_reasoning: bool = False

    # Priority for the request
    priority: Optional[int] = None

    # Extra cache key for classifying the request (e.g. cache_salt)
    extra_key: Optional[str] = None

    # Whether to disallow logging for this request (e.g. due to ZDR)
    no_logs: bool = False

    # (Internal) Whether to return bytes for image generation
    return_bytes: bool = False
    # Whether to return entropy
    return_entropy: bool = False

    need_wait_for_mm_inputs: Optional[bool] = None
    num_items_assigned: Optional[Dict[Modality, List[int]]] = None
    # Pickled Optional[List[{"url": MultimodalDataInputItem, "modality": Modality}]]
    # from MMReceiverBase._extract_url_data. "url" is ImageData.url,
    # dict["url"] when present, or the original raw multimodal item.
    mm_data_mooncake: Optional[PickleWrapper] = None
    # Encoder URL snapshot frozen at tokenizer-side dispatch time so that
    # encoder_idx assignments stay consistent in the scheduler subprocess.
    # Internal IPC only.
    encoder_urls: Optional[List[str]] = None

    # Pre-computed delimiter indices for multi-item scoring
    multi_item_delimiter_indices: Optional[List[int]] = None

    # For observability
    # Pickled Optional[Union[APIServerReqTimeStats, DPControllerReqTimeStats]]
    time_stats: Optional[PickleWrapper] = None

    def wrap_pickle_fields(self):
        self.mm_inputs = wrap_as_pickle(self.mm_inputs)
        self.mm_data_mooncake = wrap_as_pickle(self.mm_data_mooncake)
        self.time_stats = wrap_as_pickle(self.time_stats)

    def unwrap_pickle_fields(self):
        self.mm_inputs = unwrap_from_pickle(self.mm_inputs)
        self.mm_data_mooncake = unwrap_from_pickle(self.mm_data_mooncake)
        self.time_stats = unwrap_from_pickle(self.time_stats)


class BatchTokenizedGenerateReqInput(BaseBatchReq, kw_only=True):
    # The batch of tokenized requests
    # Routing for request i is batch[i].http_worker_ipc, not http_worker_ipcs[i].
    batch: List[TokenizedGenerateReqInput]

    def __len__(self):
        return len(self.batch)

    def __getitem__(self, i):
        return self.batch[i]

    def __iter__(self):
        return iter(self.batch)


@dataclass
class EmbeddingReqInput:
    # Request ID(s). If omitted, generated during normalization. For batch
    # requests, a string is expanded to per-item IDs using it as a prefix.
    rid: Optional[Union[str, List[str]]] = field(default=None, kw_only=True)
    # The input prompt. It can be a single prompt or a batch of prompts.
    text: Optional[Union[List[List[str]], List[str], str]] = None
    # The token ids for text; one can either specify text or input_ids.
    input_ids: Optional[Union[List[List[int]], List[int]]] = None
    # Dummy input embeds for compatibility
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
    # Placeholder token ID used to locate embedding override positions in input token IDs.
    embed_override_token_id: Optional[int] = None
    # Unresolved embedding overrides: per-input list of tensors.
    # Position resolution happens in the tokenizer manager after tokenization.
    # Shape: [num_inputs][num_replacements] where each entry is a torch.Tensor of [hidden_size].
    # Per-input entry may be None when only some inputs in a batch need overrides.
    # Runtime type: Optional[List[Optional[List[torch.Tensor]]]]
    # Typed as Any to avoid Pydantic/FastAPI schema errors (contains torch.Tensor).
    embed_overrides: Any = None
    # Dummy sampling params for compatibility
    sampling_params: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None
    # Whether to log metrics for this request (e.g. health_generate calls do not log metrics)
    log_metrics: bool = True
    # The modalities of the image data [image, multi-images, video]
    modalities: Optional[List[str]] = None
    # For cross-encoder requests
    is_cross_encoder_request: bool = False
    # The path to the LoRA adaptors
    lora_path: Optional[Union[List[Optional[str]], str]] = None
    # The uid of LoRA adaptors, should be initialized by tokenizer manager
    lora_id: Optional[Union[List[Optional[str]], str]] = None
    # Resolved embedding overrides with positions (set by tokenizer manager or score mixin).
    # Runtime type: Optional[Union[PositionalEmbeds, List[Optional[PositionalEmbeds]]]]
    positional_embed_overrides: Any = None
    # Routing key for routing-key schedule policy
    routing_key: Optional[str] = None
    # Internal IPC endpoint of the HTTP/tokenizer worker that owns this request.
    # Used to route outputs back in multi-tokenizer mode.
    http_worker_ipc: Optional[str] = field(default=None, kw_only=True)

    # For background responses (OpenAI responses API)
    background: bool = False

    # Priority for the request
    priority: Optional[int] = None

    # The number of dimensions the resulting output embeddings should have. It is applicable for Matryoshka Embeddings.
    dimensions: Optional[int] = None
    # Whether to return pooled hidden states (pre-head transformer output)
    return_pooled_hidden_states: bool = False
    # Whether to return prompt token IDs without computing logprobs
    return_prompt_token_ids: bool = False

    # Propagates trace context via Engine.encode/async_encode
    external_trace_header: Optional[Dict[str, Any]] = None
    received_time: Optional[float] = None

    # Pre-computed delimiter indices for multi-item scoring.
    # Batch-level: List[List[int]] (one per request). After __getitem__: List[int].
    multi_item_delimiter_indices: Optional[Union[List[List[int]], List[int]]] = None

    def regenerate_rid(self):
        """Generate a new request ID and return it."""
        if isinstance(self.rid, list):
            self.rid = [uuid.uuid4().hex for _ in range(len(self.rid))]
        else:
            self.rid = uuid.uuid4().hex
        return self.rid

    def _validate_rid_uniqueness(self):
        """Validate that request IDs within a batch are unique."""
        if isinstance(self.rid, list) and len(set(self.rid)) != len(self.rid):
            counts = Counter(self.rid)
            duplicates = [rid for rid, count in counts.items() if count > 1]
            raise ValueError(
                f"Duplicate request IDs detected within the request: {duplicates}"
            )

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

            self._normalize_lora_paths(self.batch_size)

        self._validate_rid_uniqueness()

    def _normalize_lora_paths(self, num):
        """Normalize LoRA paths for batch processing."""
        if self.lora_path is not None:
            if isinstance(self.lora_path, str):
                self.lora_path = [self.lora_path] * num
            elif isinstance(self.lora_path, list):
                if len(self.lora_path) != num:
                    raise ValueError(
                        f"lora_path list length ({len(self.lora_path)}) must match batch size ({num})"
                    )
            else:
                raise ValueError("lora_path should be a list or a string.")

    def contains_mm_input(self) -> bool:
        return (
            has_valid_data(self.image_data)
            or has_valid_data(self.video_data)
            or has_valid_data(self.audio_data)
        )

    def _get_positional_embed_overrides_item(
        self, i: int
    ) -> Optional[PositionalEmbeds]:
        """Extract the i-th item from positional_embed_overrides."""
        if self.positional_embed_overrides is None:
            return None
        if isinstance(self.positional_embed_overrides, PositionalEmbeds):
            return self.positional_embed_overrides
        return self.positional_embed_overrides[i]

    def __getitem__(self, i):
        # Cache sub-objects so that repeated obj[i] calls return the same instance.
        cache = self.__dict__.setdefault("_sub_obj_cache", {})
        if i in cache:
            return cache[i]

        if self.is_cross_encoder_request:
            sub = EmbeddingReqInput(
                rid=self.rid[i],
                text=[self.text[i]] if self.text is not None else None,
                sampling_params=self.sampling_params[i],
                is_cross_encoder_request=True,
                lora_path=self.lora_path[i] if self.lora_path is not None else None,
                lora_id=self.lora_id[i] if self.lora_id is not None else None,
                positional_embed_overrides=self._get_positional_embed_overrides_item(i),
                http_worker_ipc=self.http_worker_ipc,
                return_pooled_hidden_states=self.return_pooled_hidden_states,
                return_prompt_token_ids=self.return_prompt_token_ids,
                multi_item_delimiter_indices=(
                    self.multi_item_delimiter_indices[i]
                    if self.multi_item_delimiter_indices is not None
                    else None
                ),
            )
        else:
            sub = EmbeddingReqInput(
                rid=self.rid[i],
                text=self.text[i] if self.text is not None else None,
                input_ids=self.input_ids[i] if self.input_ids is not None else None,
                image_data=self.image_data[i] if self.image_data is not None else None,
                video_data=self.video_data[i] if self.video_data is not None else None,
                audio_data=self.audio_data[i] if self.audio_data is not None else None,
                embed_override_token_id=self.embed_override_token_id,
                embed_overrides=(
                    self.embed_overrides[i]
                    if self.embed_overrides is not None
                    else None
                ),
                sampling_params=self.sampling_params[i],
                lora_path=self.lora_path[i] if self.lora_path is not None else None,
                lora_id=self.lora_id[i] if self.lora_id is not None else None,
                positional_embed_overrides=self._get_positional_embed_overrides_item(i),
                http_worker_ipc=self.http_worker_ipc,
                dimensions=self.dimensions,
                return_pooled_hidden_states=self.return_pooled_hidden_states,
                return_prompt_token_ids=self.return_prompt_token_ids,
                external_trace_header=self.external_trace_header,
                received_time=self.received_time,
                multi_item_delimiter_indices=(
                    self.multi_item_delimiter_indices[i]
                    if self.multi_item_delimiter_indices is not None
                    else None
                ),
            )
        cache[i] = sub
        return sub


class TokenizedEmbeddingReqInput(BaseReq, kw_only=True):
    input_text: Optional[Union[str, List[Union[str, List[str]]]]]
    # The input token ids
    input_ids: Optional[array]  # array[int]
    # The multimodal inputs
    mm_inputs: Optional[PickleWrapper]  # Pickled Optional[MultimodalProcessorOutput]
    # The token type ids
    token_type_ids: Optional[List[int]]
    # Dummy sampling params for compatibility
    sampling_params: SamplingParams
    # LoRA related
    lora_id: Optional[str] = None  # None means just use the base model
    # Embedding overrides to place at specific token positions.
    positional_embed_overrides: Optional[PositionalEmbeds] = None
    # For DP routing
    routed_dp_rank: Optional[int] = None
    # Priority for the request
    priority: Optional[int] = None
    # The number of dimensions the resulting output embeddings should have. It is applicable for Matryoshka Embeddings.
    dimensions: Optional[int] = None
    # Whether to return pooled hidden states (pre-head transformer output)
    return_pooled_hidden_states: bool = False
    # Pre-computed delimiter indices for multi-item scoring
    multi_item_delimiter_indices: Optional[List[int]] = None

    # For observability
    # Pickled Optional[Union[APIServerReqTimeStats, DPControllerReqTimeStats]]
    time_stats: Optional[PickleWrapper] = None

    def wrap_pickle_fields(self):
        self.mm_inputs = wrap_as_pickle(self.mm_inputs)
        self.time_stats = wrap_as_pickle(self.time_stats)

    def unwrap_pickle_fields(self):
        self.mm_inputs = unwrap_from_pickle(self.mm_inputs)
        self.time_stats = unwrap_from_pickle(self.time_stats)


class BatchTokenizedEmbeddingReqInput(BaseBatchReq, kw_only=True):
    # The batch of tokenized embedding requests
    # Routing for request i is batch[i].http_worker_ipc, not http_worker_ipcs[i].
    batch: List[TokenizedEmbeddingReqInput]

    def __len__(self):
        return len(self.batch)

    def __getitem__(self, i):
        return self.batch[i]

    def __iter__(self):
        return iter(self.batch)


TokenLogprobValues = Optional[List[Optional[List[Optional[float]]]]]
TokenLogprobIndices = Optional[List[Optional[List[Optional[int]]]]]
TopLogprobValues = Optional[List[Optional[List[Optional[List[float]]]]]]
TopLogprobIndices = Optional[List[Optional[List[Optional[List[int]]]]]]
TokenIdsLogprobValues = Optional[List[Optional[List[Optional[List[float]]]]]]
TokenIdsLogprobIndices = Optional[List[Optional[List[Optional[List[int]]]]]]
HiddenStateChunk = List[Optional[Union[float, List[float]]]]
OutputHiddenStates = Optional[List[Optional[List[HiddenStateChunk]]]]
CachedTokensDetails = Dict[str, Union[int, str]]
# Serialized form of BaseFinishReason.to_json() — all values are primitives.
FinishReasonDict = Dict[str, Optional[Union[str, int, List[int]]]]


class BatchTokenIDOutput(BaseBatchReq, kw_only=True):
    # The finish reason
    finished_reasons: List[Optional[FinishReasonDict]]
    # For incremental decoding
    decoded_texts: List[str]
    decode_ids: List[array]  # List[array[int]]
    read_offsets: List[int]
    # Only used when `--skip-tokenizer-init` is on
    output_ids: Optional[List[array]]  # Optional[List[array[int]]]
    # Detokenization configs
    skip_special_tokens: List[bool]
    spaces_between_special_tokens: List[bool]
    no_stop_trim: List[bool]

    # Token counts
    prompt_tokens: List[int]
    reasoning_tokens: List[int]
    completion_tokens: List[int]
    cached_tokens: List[int]

    # Logprobs
    input_token_logprobs_val: TokenLogprobValues
    input_token_logprobs_idx: TokenLogprobIndices
    output_token_logprobs_val: TokenLogprobValues
    output_token_logprobs_idx: TokenLogprobIndices
    input_top_logprobs_val: TopLogprobValues
    input_top_logprobs_idx: TopLogprobIndices
    output_top_logprobs_val: TopLogprobValues
    output_top_logprobs_idx: TopLogprobIndices
    input_token_ids_logprobs_val: TokenIdsLogprobValues
    input_token_ids_logprobs_idx: TokenIdsLogprobIndices
    output_token_ids_logprobs_val: TokenIdsLogprobValues
    output_token_ids_logprobs_idx: TokenIdsLogprobIndices
    output_token_entropy_val: Optional[List[Optional[float]]]

    # Hidden states
    output_hidden_states: OutputHiddenStates

    # Per-request routed experts (input + output tokens), shape
    # (token, layer, top_k). DetokenizerManager encodes to base64 into
    # BatchStrOutput; on the skip_tokenizer_init path the scheduler sends this
    # straight to TokenizerManager, which encodes on demand.
    routed_experts: Optional[List[Optional[torch.Tensor]]]

    indexer_topk: Optional[List[Optional[torch.Tensor]]]

    # The information of placeholder tokens (e.g., image token)
    # idx is the index of the token in the prompt after expansion.
    # val is the length of padded tokens after expansion.
    placeholder_tokens_idx: Optional[List[Optional[List[int]]]]
    placeholder_tokens_val: Optional[List[Optional[List[int]]]]

    # Number of times each request was retracted.
    retraction_counts: Optional[List[int]] = None

    # The trainer step id. Used to know which step's weights are used for sampling.
    token_steps: Optional[List[List[int]]] = None

    # Customized info
    customized_info: Optional[PickleWrapper] = None
    # Detailed breakdown of cached tokens by source (device/host/storage)
    cached_tokens_details: Optional[List[Optional[CachedTokensDetails]]] = None
    # DP rank of the scheduler that processed each request
    dp_ranks: Optional[List[Optional[int]]] = None

    # For observability
    # Pickled Optional[List[SchedulerReqTimeStats]]
    time_stats: Optional[PickleWrapper] = None

    # Multimodal prompt token counts (image/audio/video). None when not applicable.
    image_tokens: Optional[List[int]] = None
    audio_tokens: Optional[List[int]] = None
    video_tokens: Optional[List[int]] = None

    # Verify count: number of verification forward passes
    spec_verify_ct: Optional[List[int]] = None
    # Accepted drafts
    spec_num_correct_drafts: Optional[List[int]] = None
    # Acceptance histogram
    spec_correct_drafts_histogram: Optional[List[List[int]]] = None


class BatchStrOutput(BaseBatchReq, kw_only=True):
    # The finish reason
    finished_reasons: List[Optional[FinishReasonDict]]
    # The output decoded strings
    output_strs: List[str]
    # The token ids
    output_ids: Optional[List[array]]

    # Token counts
    prompt_tokens: List[int]
    completion_tokens: List[int]
    reasoning_tokens: List[int]
    cached_tokens: List[int]

    # Logprobs
    input_token_logprobs_val: TokenLogprobValues
    input_token_logprobs_idx: TokenLogprobIndices
    output_token_logprobs_val: TokenLogprobValues
    output_token_logprobs_idx: TokenLogprobIndices
    input_top_logprobs_val: TopLogprobValues
    input_top_logprobs_idx: TopLogprobIndices
    output_top_logprobs_val: TopLogprobValues
    output_top_logprobs_idx: TopLogprobIndices
    input_token_ids_logprobs_val: TokenIdsLogprobValues
    input_token_ids_logprobs_idx: TokenIdsLogprobIndices
    output_token_ids_logprobs_val: TokenIdsLogprobValues
    output_token_ids_logprobs_idx: TokenIdsLogprobIndices
    output_token_entropy_val: Optional[List[Optional[float]]]

    # Hidden states
    output_hidden_states: OutputHiddenStates

    # Per-request routed experts, base64-encoded by DetokenizerManager off the
    # tokenizer hot path. Underlying tensor shape is (token, layer, top_k);
    # see BatchTokenIDOutput.routed_experts.
    routed_experts: Optional[List[Optional[str]]]

    indexer_topk: Optional[List[Optional[str]]]

    # The information of placeholder tokens (e.g., image token)
    # idx is the index of the token in the prompt after expansion.
    # val is the length of padded tokens after expansion.
    placeholder_tokens_idx: Optional[List[Optional[List[int]]]]
    placeholder_tokens_val: Optional[List[Optional[List[int]]]]

    # Number of times each request was retracted.
    retraction_counts: Optional[List[int]] = None

    # The trainer step id. Used to know which step's weights are used for sampling.
    token_steps: Optional[List[List[int]]] = None

    # Customized info
    customized_info: Optional[PickleWrapper] = None
    # Detailed breakdown of cached tokens by source (device/host/storage)
    cached_tokens_details: Optional[List[Optional[CachedTokensDetails]]] = None
    # DP rank of the scheduler that processed each request
    dp_ranks: Optional[List[Optional[int]]] = None

    # For observability
    # Pickled Optional[List[SchedulerReqTimeStats]]
    time_stats: Optional[PickleWrapper] = None

    # Multimodal prompt token counts (image/audio/video). None when not applicable.
    image_tokens: Optional[List[int]] = None
    audio_tokens: Optional[List[int]] = None
    video_tokens: Optional[List[int]] = None

    # Verify count: number of verification forward passes
    spec_verify_ct: Optional[List[int]] = None
    # Accepted drafts
    spec_num_correct_drafts: Optional[List[int]] = None
    # Acceptance histogram
    spec_correct_drafts_histogram: Optional[List[List[int]]] = None


class BatchEmbeddingOutput(BaseBatchReq, kw_only=True):
    # The finish reason
    finished_reasons: List[Optional[FinishReasonDict]]
    # The output embedding
    embeddings: List[Union[List[Union[float, List[float]]], Dict[int, float], float]]
    # Token counts
    prompt_tokens: List[int]
    cached_tokens: List[int]
    # Placeholder token info
    placeholder_tokens_idx: Optional[List[Optional[List[int]]]]
    placeholder_tokens_val: Optional[List[Optional[List[int]]]]

    # Number of times each request was retracted.
    retraction_counts: Optional[List[int]] = None
    # Detailed breakdown of cached tokens by source (device/host/storage)
    cached_tokens_details: Optional[List[Optional[CachedTokensDetails]]] = None

    # For observability
    # Pickled Optional[List[SchedulerReqTimeStats]]
    time_stats: Optional[PickleWrapper] = None

    # Optional pooled hidden states (pre-head transformer output).
    # Two IPC formats, disambiguated by len vs len(rids):
    #   Stacked:     [stacked_tensor(N, ...)] — len 1, reduces pickle overhead
    #   Non-stacked: [t0, t1, ..., tN]       — len N, when shapes differ or None entries exist
    pooled_hidden_states: Optional[List[Optional[torch.Tensor]]] = None


class ClearHiCacheReqInput(BaseReq, kw_only=True):
    pass


class ClearHiCacheReqOutput(BaseReq, kw_only=True):
    success: bool


class FlushCacheReqInput(BaseReq, kw_only=True):
    timeout_s: Optional[float] = None


class FlushCacheReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str = ""


class AddExternalCorpusReqInput(BaseReq, kw_only=True):
    corpus_id: Optional[str] = None
    file_path: Optional[str] = None
    documents: Optional[List[str]] = None
    token_chunks: Optional[List[List[int]]] = None


class AddExternalCorpusReqOutput(BaseReq, kw_only=True):
    success: bool
    corpus_id: str = ""
    message: str = ""
    loaded_token_count: int = 0


class RemoveExternalCorpusReqInput(BaseReq, kw_only=True):
    corpus_id: str


class RemoveExternalCorpusReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str = ""


class ListExternalCorporaReqInput(BaseReq, kw_only=True):
    pass


class ListExternalCorporaReqOutput(BaseReq, kw_only=True):
    success: bool
    corpus_token_counts: Dict[str, int] = msgspec.field(default_factory=dict)
    message: str = ""


class AttachHiCacheStorageReqInput(BaseReq, kw_only=True):
    """Dynamically attach (enable) HiCache storage backend at runtime.

    Note: `hicache_storage_backend_extra_config_json` is a JSON string. It may contain both:
    - backend-specific configs (e.g., mooncake master address)
    - prefetch-related knobs (prefetch_threshold, prefetch_timeout_*, hicache_storage_pass_prefix_keys)
    """

    hicache_storage_backend: str
    hicache_storage_backend_extra_config_json: Optional[str] = None
    hicache_storage_prefetch_policy: Optional[str] = None
    hicache_write_policy: Optional[str] = None


class AttachHiCacheStorageReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str = ""


class DetachHiCacheStorageReqInput(BaseReq, kw_only=True):
    """Dynamically detach (disable) HiCache storage backend at runtime."""

    pass


class DetachHiCacheStorageReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str = ""


class PauseGenerationReqInput(BaseReq, kw_only=True):
    """
    Note that the PauseGenerationRequests is only supported in SGLang Server.
    abort: Abort and return all requests currently being processed.

    in_place: Pause the scheduler's event_loop from performing inference;
            only non-inference requests (e.g., control commands) will be handled.
            The requests in the engine will be paused and stay in the event_loop,
            then continue generation after continue_generation with the old kv cache.
            Note: In 'inplace' mode, flush_cache will fail if there are any requests
            in the running_batch.

    retract: Pause the scheduler's event loop from performing inference;
            only non-inference requests will be handled, and all currently running
            requests will be retracted back to the waiting_queue.
            Note: The KV cache can be flushed in this mode and will be automatically
            recomputed after continue_generation.
    """

    mode: Literal["abort", "retract", "in_place"] = "abort"


class ContinueGenerationReqInput(BaseReq, kw_only=True):
    # Call torch.cuda.empty_cache() before un-pausing. Returns blocks
    # cached by the PyTorch allocator (left over from transient allocs
    # during post-weight-update processing) back to the driver before
    # inference resumes, with no race against active streams. Set to
    # False to skip the empty_cache call.
    torch_empty_cache: bool = True


class TokenizerWorkerRegistrationReq(BaseReq, kw_only=True):
    """Sent by each TokenizerWorker on startup to register its IPC name with the router."""

    worker_ipc_name: str


class PauseContinueBroadcastReq(BaseReq, kw_only=True):
    """Broadcast from router to all workers to set is_pause state."""

    is_pause: bool


class UpdateWeightFromDiskReqInput(BaseReq, kw_only=True):
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
    # Whether to call torch.cuda.empty_cache() during flush
    torch_empty_cache: bool = False
    # Whether to keep the scheduler paused after weight update
    keep_pause: bool = False
    # Whether to recapture cuda graph after weight update
    recapture_cuda_graph: bool = False
    # The trainer step id. Used to know which step's weights are used for sampling.
    token_step: int = 0
    # Whether to flush the cache after updating weights
    flush_cache: bool = True
    # Tensor metadata
    manifest: Optional[Dict[str, Any]] = None


class UpdateWeightFromDiskReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str
    # Number of paused requests during weight sync.
    num_paused_requests: int = 0


class UpdateWeightsFromDistributedReqInput(BaseReq, kw_only=True):
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
    # Optional format specification for loading
    load_format: Optional[str] = None
    # Whether to call torch.cuda.empty_cache() during flush
    torch_empty_cache: bool = False


class UpdateWeightsFromDistributedReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str


class UpdateWeightsFromTensorReqInput(BaseReq, kw_only=True):
    """Internal IPC request for updating model weights from serialized tensors."""

    # Serialized named tensors, normalized to raw MultiprocessingSerializer
    # bytes before scheduler IPC. Python Engine callers construct this field
    # with bytes directly. FastAPI HTTP callers send base64 strings because JSON
    # has no bytes type; the Annotated Base64Bytes marker is used only by the
    # msgspec-to-Pydantic schema for the HTTP protocol to decode those strings.
    serialized_named_tensors: Annotated[List[bytes], Base64Bytes()]
    # Optional format specification for loading
    load_format: Optional[str] = None
    # Whether to flush the cache after updating weights
    flush_cache: bool = True
    # Whether to abort all requests before updating weights
    abort_all_requests: bool = False
    # Optional: Update weight version along with weights
    weight_version: Optional[str] = None
    # Optional: Determine whether to disable updating the draft model
    disable_draft_model: Optional[bool] = None
    # Whether to call torch.cuda.empty_cache() during flush
    torch_empty_cache: bool = False


class UpdateWeightsFromTensorReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str


class InitWeightsSendGroupForRemoteInstanceReqInput(BaseReq, kw_only=True):
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
class UpdateWeightsFromIPCReqInput(BaseReq, kw_only=True):
    # ZMQ socket paths for each device UUID
    zmq_handles: Dict[str, str]
    # Whether to flush cache after weight update
    flush_cache: bool = True
    # Optional: Update weight version along with weights
    weight_version: Optional[str] = None
    # Whether to call torch.cuda.empty_cache() during flush
    torch_empty_cache: bool = False


class UpdateWeightsFromIPCReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str


class InitWeightsSendGroupForRemoteInstanceReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str


class SendWeightsToRemoteInstanceReqInput(BaseReq, kw_only=True):
    # The master address
    master_address: str
    # The ports for each rank's communication group
    ports: str
    # The group name
    group_name: str = "weight_send_group"


class SendWeightsToRemoteInstanceReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str


class UpdateExpertBackupReq(BaseReq, kw_only=True):
    pass


class BackupDramReq(BaseReq, kw_only=True):
    rank: int
    weight_pointer_map: Dict[str, Any]
    session_id: str
    buffer_size: int


class InitWeightsUpdateGroupReqInput(BaseReq, kw_only=True):
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


class InitWeightsUpdateGroupReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str


class DestroyWeightsUpdateGroupReqInput(BaseReq, kw_only=True):
    group_name: str = "weight_update_group"


class DestroyWeightsUpdateGroupReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str


class UpdateWeightVersionReqInput(BaseReq, kw_only=True):
    # The new weight version
    new_version: str
    # Whether to abort all running requests before updating
    abort_all_requests: bool = True


class GetWeightsByNameReqInput(BaseReq, kw_only=True):
    name: str
    truncate_size: int = 100


class GetWeightsByNameReqOutput(BaseReq, kw_only=True):
    parameter: Optional[List[Any]]


class ReleaseMemoryOccupationReqInput(BaseReq, kw_only=True):
    # Optional tags to identify the memory region, which is primarily used for RL
    # Currently we only support `weights` and `kv_cache`
    tags: Optional[List[str]] = None


class ReleaseMemoryOccupationReqOutput(BaseReq, kw_only=True):
    pass


class ResumeMemoryOccupationReqInput(BaseReq, kw_only=True):
    # Optional tags to identify the memory region, which is primarily used for RL
    # Currently we only support `weights` and `kv_cache`
    tags: Optional[List[str]] = None


class ResumeMemoryOccupationReqOutput(BaseReq, kw_only=True):
    pass


class CheckWeightsReqInput(BaseReq, kw_only=True):
    action: str = "checksum"
    allow_quant_error: bool = False


class CheckWeightsReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str
    payload: Optional[Dict[str, Any]] = None


class SlowDownReqInput(BaseReq, kw_only=True):
    forward_sleep_time: Optional[float]


class SlowDownReqOutput(BaseReq, kw_only=True):
    pass


class AbortReq(BaseReq, kw_only=True):
    # Whether to abort all requests
    abort_all: bool = False
    # The finished reason data (from BaseFinishReason.to_json())
    finished_reason: Optional[FinishReasonDict] = None
    abort_message: Optional[str] = None

    def __post_init__(self):
        # FIXME: This is a hack to keep the same with the old code
        if self.rid is None:
            self.rid = ""


class ActiveRanksOutput(BaseReq, kw_only=True):
    status: List[bool]


class GetInternalStateReq(BaseReq, kw_only=True):
    pass


class GetInternalStateReqOutput(BaseReq, kw_only=True):
    internal_state: Dict[str, Any]


class SetInternalStateReq(BaseReq, kw_only=True):
    server_args: Dict[str, Any]


class SetInternalStateReqOutput(BaseReq, kw_only=True):
    updated: bool
    server_args: Dict[str, Any]


class ProfileReqType(Enum):
    START_PROFILE = 1
    STOP_PROFILE = 2


class ProfileReq(BaseReq, kw_only=True):
    req_type: ProfileReqType = ProfileReqType.START_PROFILE
    # The output directory
    output_dir: Optional[str] = None
    # Specify the steps to start the profiling
    start_step: Optional[int] = None
    # If set, it profile as many as this number of steps.
    # If it is set, profiling is automatically stopped after this step, and
    # the caller doesn't need to run stop_profile.
    num_steps: Optional[int] = None
    # The activities to record. The choices are ["CPU", "GPU", "MEM", "RPD"]
    activities: Optional[List[str]] = None
    # Whether profile by stages (e.g., prefill and decode) separately
    profile_by_stage: bool = False
    # Whether to record source information (file and line number) for the ops.
    with_stack: Optional[bool] = None
    # Whether to save information about operator’s input shapes.
    record_shapes: Optional[bool] = None
    profile_id: Optional[str] = None
    # Merge profiles from all ranks into a single trace
    merge_profiles: bool = False
    # The prefix of the profile filenames
    profile_prefix: Optional[str] = None
    # Only profile these stages and ignore others
    profile_stages: Optional[List[str]] = None
    # Whether to enable shape discovery (Triton / FlashInfer kernel metadata)
    shape_discovery: bool = False


class ProfileReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str


class FreezeGCReq(BaseReq, kw_only=True):
    pass


class ShutdownReq(BaseReq, kw_only=True):
    # Broadcast across TP ranks via the normal recv path, so all ranks break
    # the scheduler loop on the same iteration.
    pass


class ConfigureLoggingReq(BaseReq, kw_only=True):
    log_requests: Optional[bool] = None
    log_requests_level: Optional[int] = None
    log_requests_format: Optional[str] = None
    log_level: Optional[str] = None
    dump_requests_folder: Optional[str] = None
    dump_requests_threshold: Optional[int] = None
    crash_dump_folder: Optional[str] = None
    dump_requests_exclude_meta_keys: Optional[List[str]] = None


class OpenSessionReqInput(BaseReq, kw_only=True):
    capacity_of_str_len: int
    session_id: Optional[str] = None
    streaming: Optional[bool] = None
    timeout: Optional[float] = None


class CloseSessionReqInput(BaseReq, kw_only=True):
    session_id: str


class OpenSessionReqOutput(BaseReq, kw_only=True):
    session_id: Optional[str]
    success: bool


class HealthCheckOutput(BaseReq, kw_only=True):
    pass


class ExpertDistributionReqType(Enum):
    START_RECORD = 1
    STOP_RECORD = 2
    DUMP_RECORD = 3


class ExpertDistributionReq(BaseReq, kw_only=True):
    action: ExpertDistributionReqType


class ExpertDistributionReqOutput(BaseReq, kw_only=True):
    pass


class Function(msgspec.Struct, kw_only=True, array_like=True):
    description: Optional[str] = None
    name: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return msgspec_struct_pydantic_core_schema(cls, handler)


class Tool(msgspec.Struct, kw_only=True, array_like=True):
    function: Function
    type: str = "function"

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return msgspec_struct_pydantic_core_schema(cls, handler)


class ParseFunctionCallReq(BaseReq, kw_only=True):
    text: str  # The text to parse.
    tools: List[Tool] = msgspec.field(
        default_factory=list
    )  # A list of available function tools (name, parameters, etc.).
    tool_call_parser: Optional[str] = (
        None  # Specify the parser type, e.g. 'llama3', 'qwen25', or 'mistral'. If not specified, tries all.
    )


class SeparateReasoningReqInput(BaseReq, kw_only=True):
    text: str  # The text to parse.
    reasoning_parser: str  # Specify the parser type, e.g., "deepseek-r1".
    return_blocks: bool = False  # If True, also return segmented reasoning blocks.


class VertexGenerateReqInput(BaseReq, kw_only=True):
    instances: List[Dict[str, Any]]
    parameters: Optional[Dict[str, Any]] = None


class RpcReqInput(BaseReq, kw_only=True):
    method: str
    parameters: Optional[Dict[str, Any]] = None


class RpcReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str


class LoadLoRAAdapterReqInput(BaseReq, kw_only=True):
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


class UnloadLoRAAdapterReqInput(BaseReq, kw_only=True):
    # The name of lora module to unload.
    lora_name: str
    # The unique identifier for the LoRA adapter, which automatically generated in the `TokenizerManager`.
    lora_id: Optional[str] = None

    def to_ref(self) -> LoRARef:
        return LoRARef(
            lora_id=self.lora_id,
            lora_name=self.lora_name,
        )


class LoadLoRAAdapterFromTensorsReqInput(BaseReq, kw_only=True):
    lora_name: str
    config_dict: Dict[str, Any]
    serialized_tensors: str
    pinned: bool = False
    added_tokens_config: Optional[Dict[str, Any]] = None
    lora_id: Optional[str] = None
    load_format: Optional[str] = None

    def to_ref(self) -> LoRARef:
        return LoRARef(
            lora_id=self.lora_id,
            lora_name=self.lora_name,
            lora_path="__tensor__",
            pinned=self.pinned,
        )


class LoRAUpdateOutput(BaseReq, kw_only=True):
    success: bool
    error_message: Optional[str] = None
    loaded_adapters: Optional[Dict[str, Union[str, LoRARef]]] = None


LoadLoRAAdapterReqOutput = UnloadLoRAAdapterReqOutput = (
    LoadLoRAAdapterFromTensorsReqOutput
) = LoRAUpdateOutput


class BlockReqType(Enum):
    BLOCK = 1
    UNBLOCK = 2


class BlockReqInput(BaseReq, kw_only=True):
    req_type: BlockReqType


class SetInjectDumpMetadataReqInput(BaseReq, kw_only=True):
    dump_metadata: Dict[str, Any]


class SetInjectDumpMetadataReqOutput(BaseReq, kw_only=True):
    success: bool


class LazyDumpTensorsReqInput(BaseReq, kw_only=True):
    pass


class LazyDumpTensorsReqOutput(BaseReq, kw_only=True):
    success: bool


class DumperControlReqInput(BaseReq, kw_only=True):
    method: str
    body: Dict[str, Any]


class DumperControlReqOutput(BaseReq, kw_only=True):
    success: bool
    response: List[Dict[str, Any]]
    error: str = ""


# The following request types are either defined in other files,
# or not subclasses of BaseReq/BaseBatchReq, so we skip the check for them.
_IGNORE_REQ_TYPES_CHECK = (
    GenerateReqInput.__name__,
    EmbeddingReqInput.__name__,
)


def _check_all_req_types():
    """A helper function to check all request types are defined in this file."""
    import inspect
    import sys

    all_classes = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    for class_type in all_classes:
        # check its name
        name = class_type[0]
        if name in _IGNORE_REQ_TYPES_CHECK:
            continue
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

# IPC struct types whose fields still use opaque annotations (Any, Dict[str, Any],
# List[Any], etc.) instead of precise types. Keep these on explicit pickle
# transport until their field schemas are tightened, and keep the registry
# explicit so opaque usage can be audited and gradually narrowed.
# NOTE: GenerateReqInput and EmbeddingReqInput are standalone (not BaseReq/
# BaseBatchReq subclasses) and are tracked separately.
_REQ_TYPES_WITH_OPAQUE_FIELDS: tuple[Type[msgspec.Struct], ...] = (
    UpdateWeightFromDiskReqInput,  # manifest: Optional[Dict[str, Any]]
    BackupDramReq,  # weight_pointer_map: Dict[str, Any]
    GetWeightsByNameReqOutput,  # parameter: Optional[List[Any]]
    CheckWeightsReqOutput,  # payload: Optional[Dict[str, Any]]
    GetInternalStateReqOutput,  # internal_state: Dict[str, Any]
    SetInternalStateReq,  # server_args: Dict[str, Any]
    SetInternalStateReqOutput,  # server_args: Dict[str, Any]
    VertexGenerateReqInput,  # instances, parameters: Dict[str, Any]
    RpcReqInput,  # parameters: Optional[Dict[str, Any]]
    LoadLoRAAdapterFromTensorsReqInput,  # config_dict, added_tokens_config: Dict[str, Any]
    SetInjectDumpMetadataReqInput,  # dump_metadata: Dict[str, Any]
    DumperControlReqInput,  # body: Dict[str, Any]
    DumperControlReqOutput,  # response: List[Dict[str, Any]]
)


def wrap_as_pickle(obj: object) -> object:
    if obj is None:
        return None
    if _USE_PICKLE_IPC:
        return obj
    return PickleWrapper(pickle.dumps(obj))


def unwrap_from_pickle(obj: Optional[object]) -> Optional[object]:
    if obj is None:
        return None
    if _USE_PICKLE_IPC:
        return obj
    assert isinstance(obj, PickleWrapper)
    return pickle.loads(obj.data)


def enc_hook(obj: Any) -> Any:
    if isinstance(obj, array):
        return (obj.typecode, obj.tobytes())
    elif isinstance(obj, torch.Tensor):
        tensor_dtype = str(obj.dtype).removeprefix("torch.")
        raw_data = (
            obj.cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        )
        return (obj.shape, tensor_dtype, raw_data)
    elif isinstance(obj, np.ndarray):
        raw_data = np.ascontiguousarray(obj).reshape(-1).view(np.uint8).data
        return (obj.shape, obj.dtype.str, raw_data)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        raise TypeError(
            f"Cannot msgpack encode object of type {type(obj)} with enc_hook. "
            "Use an explicit PickleWrapper field via wrap_as_pickle(...) for "
            "arbitrary payloads, or add a dedicated enc_hook/dec_hook branch "
            "for this transport type."
        )


def dec_hook(tp: Type, obj: Any) -> Any:
    if tp is array:
        typecode, raw_data = obj
        res = array(typecode)
        res.frombytes(raw_data)
        return res
    elif tp is torch.Tensor:
        shape, dtype, data = obj
        tensor_dtype = getattr(torch, dtype)
        if len(data) == 0:
            return torch.empty(shape, dtype=tensor_dtype)
        return torch.frombuffer(bytearray(data), dtype=tensor_dtype).reshape(shape)
    elif tp is np.ndarray:
        shape, dtype, data = obj
        return np.frombuffer(data, dtype=np.dtype(dtype)).copy().reshape(shape)
    else:
        raise TypeError(
            f"Cannot msgpack decode object of type {type(obj)} as {tp} with "
            "dec_hook. Use an explicit PickleWrapper field via wrap_as_pickle(...) "
            "and unwrap_from_pickle(...) for arbitrary payloads, or add a "
            "dedicated enc_hook/dec_hook branch for this transport type."
        )


_struct_types = tuple(
    cls
    for cls in BaseReq.__subclasses__()
    + BaseBatchReq.__subclasses__()
    + [PickleWrapper]
)
# Primitive types that msgpack can serialize directly without PickleWrapper.
# Do not include str here: msgspec rejects a Union containing both str and bytes
# as multiple str-like arms. Top-level strings use PickleWrapper; string fields
# inside typed structs are still decoded by their struct schemas.
_primitive_types = (int, float, bool, bytes)
_all_types = _struct_types + _primitive_types

_msgpack_encoder = msgspec.msgpack.Encoder(enc_hook=enc_hook)
_msgpack_decoder = msgspec.msgpack.Decoder(Union[_all_types], dec_hook=dec_hook)
_USE_PICKLE_IPC = envs.SGLANG_USE_PICKLE_IPC.get()


def hook_custom_types(*new_types: Type):
    global _msgpack_decoder, _all_types
    _all_types = tuple(dict.fromkeys(_all_types + new_types))
    _msgpack_decoder = msgspec.msgpack.Decoder(Union[_all_types], dec_hook=dec_hook)


def _maybe_wrap_pickle(obj: Any) -> Any:
    if isinstance(obj, _REQ_TYPES_WITH_OPAQUE_FIELDS):
        if envs.SGLANG_LOG_PICKLE_IPC_OBJECTS.get():
            logger.info(f"Object of type {type(obj)} is wrapped via PickleWrapper.")
        return PickleWrapper(pickle.dumps(obj))

    if isinstance(obj, (msgspec.Struct, *_primitive_types)):
        return obj

    raise TypeError(
        f"Cannot serialize object of type {type(obj)} over msgpack IPC. "
        "Add a precise msgspec-compatible type, use an explicit PickleWrapper "
        "field for the opaque payload, or add the struct to "
        "_REQ_TYPES_WITH_OPAQUE_FIELDS with an audit comment."
    )


def _maybe_unwrap_pickle(obj: Any) -> Any:
    if isinstance(obj, PickleWrapper):
        obj = pickle.loads(obj.data)
        if envs.SGLANG_LOG_PICKLE_IPC_OBJECTS.get():
            logger.info(f"Object of type {type(obj)} is unwrapped from PickleWrapper.")
        return obj

    return obj


def msgpack_encode(obj: Any) -> bytes:
    return _msgpack_encoder.encode(_maybe_wrap_pickle(obj))


def msgpack_decode(data: bytes) -> Any:
    return _maybe_unwrap_pickle(_msgpack_decoder.decode(data))


def sock_send(socket: zmq.Socket, obj: Any, flags: int = 0) -> None:
    if _USE_PICKLE_IPC:
        socket.send_pyobj(obj, flags=flags, protocol=pickle.HIGHEST_PROTOCOL)
        return

    socket.send(msgpack_encode(obj), flags=flags)


def sock_recv(socket: zmq.Socket, flags: int = 0) -> Any:
    if _USE_PICKLE_IPC:
        return socket.recv_pyobj(flags=flags)

    data = socket.recv(flags=flags)
    return msgpack_decode(data)


async def async_sock_send(socket: zmq.asyncio.Socket, obj: Any, flags: int = 0) -> None:
    if _USE_PICKLE_IPC:
        await socket.send_pyobj(obj, flags=flags, protocol=pickle.HIGHEST_PROTOCOL)
        return

    await socket.send(msgpack_encode(obj), flags=flags)


async def async_sock_recv(socket: zmq.asyncio.Socket, flags: int = 0) -> Any:
    if _USE_PICKLE_IPC:
        return await socket.recv_pyobj(flags=flags)

    data = await socket.recv(flags=flags)
    return msgpack_decode(data)
