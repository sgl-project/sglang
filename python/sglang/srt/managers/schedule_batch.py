from __future__ import annotations

import enum

from sglang.srt.model_executor.forward_batch_info import ForwardBatch

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
Store information about requests and batches.

The following is the flow of data structures for a batch:

ScheduleBatch -> ModelWorkerBatch -> ForwardBatch

- ScheduleBatch is managed by `scheduler.py::Scheduler`.
  It contains high-level scheduling data. Most of the data is on the CPU.
- ModelWorkerBatch is managed by `tp_worker.py::TpModelWorker`.
  It is a subset of `ScheduleBatch` that only contains data related to the model forward on GPU.
  It will be transformed from CPU scheduler to GPU model runner.
- ForwardBatch is managed by `model_runner.py::ModelRunner`.
  It contains low-level tensor data. Most of the data consists of GPU tensors.

TODO(lmzheng): ModelWorkerBatch seems a bit redundant and we consider removing it in the future.
"""

import copy
import dataclasses
import logging
import re
import time
from enum import Enum, auto
from http import HTTPStatus
from itertools import chain
from typing import TYPE_CHECKING, Any, List, Optional, Set, Tuple, Union

import numpy as np
import torch

from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.disaggregation.base import BaseKVSender
from sglang.srt.disaggregation.decode_schedule_batch_mixin import (
    ScheduleBatchDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_rank
from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocator import (
    BaseTokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.chunk_cache import SWAChunkCache
from sglang.srt.mem_cache.common import (
    alloc_for_decode,
    alloc_for_extend,
    evict_from_tree_cache,
)
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache
from sglang.srt.metrics.collector import SchedulerMetricsCollector, TimeStats
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, get_global_server_args
from sglang.srt.utils import flatten_nested_list
from sglang.srt.utils.common import is_npu
from sglang.srt.utils.cuda_ipc_transport_utils import CudaIpcTensorTransportProxy

_is_npu = is_npu()

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.speculative.eagle_info import EagleDraftInput
    from sglang.srt.speculative.spec_info import SpecInput, SpeculativeAlgorithm

INIT_INCREMENTAL_DETOKENIZATION_OFFSET = 5


logger = logging.getLogger(__name__)


class BaseFinishReason:
    def __init__(self, is_error: bool = False):
        self.is_error = is_error

    def to_json(self):
        raise NotImplementedError()


class FINISH_MATCHED_TOKEN(BaseFinishReason):
    def __init__(self, matched: Union[int, List[int]]):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


class FINISH_MATCHED_STR(BaseFinishReason):
    def __init__(self, matched: str):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


class FINISHED_MATCHED_REGEX(BaseFinishReason):
    def __init__(self, matched: str):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


class FINISH_LENGTH(BaseFinishReason):
    def __init__(self, length: int):
        super().__init__()
        self.length = length

    def to_json(self):
        return {
            "type": "length",  # to match OpenAI API's return value
            "length": self.length,
        }


class FINISH_ABORT(BaseFinishReason):
    def __init__(self, message=None, status_code=None, err_type=None):
        super().__init__(is_error=True)
        self.message = message or "Aborted"
        self.status_code = status_code
        self.err_type = err_type

    def to_json(self):
        return {
            "type": "abort",
            "message": self.message,
            "status_code": self.status_code,
            "err_type": self.err_type,
        }


class Modality(Enum):
    IMAGE = auto()
    MULTI_IMAGES = auto()
    VIDEO = auto()
    AUDIO = auto()

    @staticmethod
    def from_str(modality_str: str):
        try:
            return Modality[modality_str.upper()]
        except KeyError:
            raise ValueError(
                f"Invalid modality string: {modality_str}. Valid modalities are: {[m.name for m in Modality]}"
            )

    @staticmethod
    def all():
        return [Modality.IMAGE, Modality.VIDEO, Modality.AUDIO]


@dataclasses.dataclass
class MultimodalDataItem:
    """
    One MultimodalDataItem contains all inputs for one modality.
    For example, if there are 3 images and 1 audio inputs, there will be 2 MultimodalDataItem.
    One for images and one for audio.

    We put the common fields first and the model-specific fields in model_specific_data.
    """

    modality: Modality
    hash: int = None
    pad_value: int = None
    offsets: Optional[list] = None

    # the raw features returned by processor, e.g. pixel_values or audio_features
    feature: Union[torch.Tensor, np.ndarray] = None
    # the precomputed embeddings, passed as final encoder embeddings
    # One and only one of the feature and precomputed_embeddings will be empty
    precomputed_embeddings: Optional[Union[torch.Tensor, np.ndarray]] = None

    # Model-specific data stored in a dictionary
    model_specific_data: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __getattr__(self, name: str):
        if (
            "model_specific_data" in self.__dict__
            and name in self.__dict__["model_specific_data"]
        ):
            return self.__dict__["model_specific_data"][name]
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    def __setitem__(self, key: str, value: Any):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self.model_specific_data[key] = value

    def set(self, key: str, value: Any):
        self.__setitem__(key, value)

    @staticmethod
    def is_empty_list(l):
        if l is None:
            return True
        return len([item for item in flatten_nested_list(l) if item is not None]) == 0

    def set_pad_value(self):
        """
        Set the pad value after first hashing the data
        """
        from sglang.srt.managers.mm_utils import hash_feature

        if self.hash is None:
            if self.feature is not None:
                hashed_feature = self.feature
            else:
                hashed_feature = self.precomputed_embeddings
            self.hash = hash_feature(hashed_feature)
        assert self.hash is not None
        self.pad_value = self.hash % (1 << 30)

    def is_modality(self, modality: Modality) -> bool:
        return self.modality == modality

    def is_audio(self):
        return self.modality == Modality.AUDIO

    def is_image(self):
        return self.modality in [Modality.IMAGE, Modality.MULTI_IMAGES]

    def is_video(self):
        return self.modality == Modality.VIDEO

    def is_valid(self) -> bool:
        return self.is_image() or self.is_video() or self.is_audio()

    def validate(self):
        ...
        # TODO

    @staticmethod
    def from_dict(obj: dict):
        kwargs = dict(obj)
        modality = kwargs.pop("modality")
        if isinstance(modality, str):
            modality = Modality[modality]
        ret = MultimodalDataItem(modality=modality, **kwargs)
        ret.validate()
        return ret

    def merge(self, other):
        self.feature += other.feature
        self.offsets += other.offsets
        self.hash = hash((self.hash, other.hash))
        self.set_pad_value()


@dataclasses.dataclass
class MultimodalInputs:
    """The multimodal data related inputs."""

    # items of data
    mm_items: List[MultimodalDataItem]
    image_pad_len: Optional[list] = None
    num_image_tokens: Optional[int] = None

    # image
    im_token_id: Optional[int] = None
    im_start_id: Optional[int] = None
    im_end_id: Optional[int] = None
    slice_start_id: Optional[int] = None
    slice_end_id: Optional[int] = None

    # video
    video_token_id: Optional[int] = None

    # audio
    audio_token_id: Optional[int] = None
    audio_start_id: Optional[int] = None
    audio_end_id: Optional[int] = None

    # QWen2-VL related
    mrope_positions: Optional[torch.Tensor] = None
    mrope_position_delta: Optional[torch.Tensor] = None

    @staticmethod
    def from_dict(obj: dict):
        ret = MultimodalInputs(
            mm_items=obj["mm_items"],
        )

        assert isinstance(ret.mm_items, list)
        ret.mm_items = [item for item in ret.mm_items if item.is_valid()]
        for item in ret.mm_items:
            item.set_pad_value()

        optional_args = [
            "mrope_positions",
            "mrope_position_delta",
            "im_token_id",
            "im_start_id",
            "im_end_id",
            "video_token_id",
            "slice_start_id",
            "slice_end_id",
            "audio_start_id",
            "audio_end_id",
            "audio_token_id",
        ]
        for arg in optional_args:
            if arg in obj:
                setattr(ret, arg, obj[arg])

        return ret

    def contains_image_inputs(self) -> bool:
        return any(item.is_image() for item in self.mm_items)

    def contains_video_inputs(self) -> bool:
        return any(item.is_video() for item in self.mm_items)

    def contains_audio_inputs(self) -> bool:
        return any(item.is_audio() for item in self.mm_items)

    def contains_mm_input(self) -> bool:
        return any(True for item in self.mm_items if item.is_valid())

    def merge(self, other: MultimodalInputs):
        """
        merge image inputs when requests are being merged
        """

        # args needed to be merged
        optional_args = [
            "mm_items",
            "image_pad_len",
        ]
        for arg in optional_args:
            self_arg = getattr(self, arg, None)
            if self_arg is not None:
                setattr(self, arg, self_arg + getattr(other, arg))

        mrope_positions = self.mrope_positions
        if mrope_positions is not None:
            if other.mrope_positions is None:
                self.mrope_positions = mrope_positions
            else:
                self.mrope_positions = torch.cat(
                    [self.mrope_positions, other.mrope_positions], dim=1
                )

        mrope_position_delta = self.mrope_position_delta
        if mrope_position_delta is not None:
            if other.mrope_position_delta is None:
                self.mrope_position_delta = mrope_position_delta
            else:
                self.mrope_position_delta = torch.cat(
                    [self.mrope_position_delta, other.mrope_position_delta], dim=0
                )

        for key, val in other.__dict__.items():
            if "_id" in key:
                # set token_ids
                if getattr(self, key, None) is None:
                    setattr(self, key, getattr(other, key, None))
        # other args would be kept intact


class RequestStage(str, enum.Enum):
    # Tokenizer
    TOKENIZE = "tokenize"
    TOKENIZER_DISPATCH = "dispatch"

    # DP controller
    DC_DISPATCH = "dc_dispatch"

    # common/non-disaggregation
    PREFILL_WAITING = "prefill_waiting"
    REQUEST_PROCESS = "request_process"
    DECODE_LOOP = "decode_loop"
    PREFILL_FORWARD = "prefill_forward"
    PREFILL_CHUNKED_FORWARD = "chunked_prefill"

    # disaggregation prefill
    PREFILL_PREPARE = "prefill_prepare"
    PREFILL_BOOTSTRAP = "prefill_bootstrap"
    PREFILL_TRANSFER_KV_CACHE = "prefill_transfer_kv_cache"

    # disaggregation decode
    DECODE_PREPARE = "decode_prepare"
    DECODE_BOOTSTRAP = "decode_bootstrap"
    DECODE_WAITING = "decode_waiting"
    DECODE_TRANSFERRED = "decode_transferred"
    DECODE_FAKE_OUTPUT = "fake_output"
    DECODE_QUICK_FINISH = "quick_finish"


class Req:
    """The input and output status of a request."""

    def __init__(
        self,
        rid: str,
        origin_input_text: str,
        origin_input_ids: List[int],
        sampling_params: SamplingParams,
        return_logprob: bool = False,
        top_logprobs_num: int = 0,
        token_ids_logprob: List[int] = None,
        stream: bool = False,
        origin_input_ids_unpadded: Optional[Tuple[int]] = None,
        lora_id: Optional[str] = None,
        input_embeds: Optional[List[List[float]]] = None,
        token_type_ids: List[int] = None,
        session_id: Optional[str] = None,
        custom_logit_processor: Optional[str] = None,
        return_hidden_states: bool = False,
        eos_token_ids: Optional[Set[int]] = None,
        bootstrap_host: Optional[str] = None,
        bootstrap_port: Optional[int] = None,
        bootstrap_room: Optional[int] = None,
        disagg_mode: Optional[DisaggregationMode] = None,
        data_parallel_rank: Optional[int] = None,
        vocab_size: Optional[int] = None,
        priority: Optional[int] = None,
        metrics_collector: Optional[SchedulerMetricsCollector] = None,
        extra_key: Optional[str] = None,
        dimensions: Optional[int] = None,
        http_worker_ipc: Optional[str] = None,
    ):
        # Input and output info
        self.rid = rid
        self.origin_input_text = origin_input_text
        self.origin_input_ids_unpadded = (
            origin_input_ids_unpadded
            if origin_input_ids_unpadded
            else origin_input_ids  # Before image padding
        )
        self.origin_input_ids = origin_input_ids
        # Each decode stage's output ids
        self.output_ids = []
        # fill_ids = origin_input_ids + output_ids. Updated if chunked.
        self.fill_ids = []
        self.session_id = session_id
        self.input_embeds = input_embeds

        # for corss-endoder model
        self.token_type_ids = token_type_ids

        # The length of KV that have been removed in local attention chunked prefill
        self.evicted_seqlen_local = 0

        # For multi-http worker
        self.http_worker_ipc = http_worker_ipc

        # Sampling info
        if isinstance(sampling_params.custom_params, dict):
            sampling_params = copy.copy(sampling_params)
            sampling_params.custom_params = sampling_params.custom_params | {
                "__req__": self
            }
        self.sampling_params = sampling_params
        self.custom_logit_processor = custom_logit_processor
        self.return_hidden_states = return_hidden_states

        # extra key for classifying the request (e.g. cache_salt)
        if lora_id is not None:
            extra_key = (
                extra_key or ""
            ) + lora_id  # lora_id is concatenated to the extra key

        self.extra_key = extra_key
        self.lora_id = lora_id

        # Memory pool info
        self.req_pool_idx: Optional[int] = None
        self.mamba_pool_idx: Optional[torch.Tensor] = None  # shape (1)

        # Check finish
        self.tokenizer = None
        self.finished_reason: Optional[BaseFinishReason] = None
        # finished position (in output_ids), used when checking stop conditions with speculative decoding
        self.finished_len = None
        # Whether this request has finished output
        self.finished_output = None
        # If we want to abort the request in the middle of the event loop,
        # set to_finish instead of directly setting finished_reason.
        # Note: We should never set finished_reason in the middle, the req will get filtered and never respond
        self.to_finish: Optional[BaseFinishReason] = None
        self.stream = stream
        self.eos_token_ids = eos_token_ids
        self.vocab_size = vocab_size
        self.priority = priority

        # For incremental decoding
        # ----- | --------- read_ids -------|
        # ----- |   surr_ids  |
        # xxxxx | xxxxxxxxxxx | xxxxxxxxxxx |
        # ----- ^ ----------- ^ ----------- ^
        # ----- 1 ----------- 2 ----------- 3
        # 1: surr_offset
        # 2: read_offset
        # 3: last token
        self.surr_offset = None  # Surrounding offset to defeat the cleanup algorithm
        self.read_offset = None
        self.decoded_text = ""

        # For multimodal inputs
        self.multimodal_inputs: Optional[MultimodalInputs] = None

        # Prefix info
        # The indices to kv cache for the shared prefix.
        self.prefix_indices: torch.Tensor = torch.empty((0,), dtype=torch.int64)
        # Number of tokens to run prefill.
        self.extend_input_len = 0
        # The relative logprob_start_len in an extend batch
        self.extend_logprob_start_len = 0
        self.last_node: Any = None
        self.last_host_node: Any = None
        self.host_hit_length = 0
        # The node to lock until for swa radix tree lock ref
        self.swa_uuid_for_lock: Optional[int] = None
        # The prefix length of the last prefix matching
        self.last_matched_prefix_len: int = 0

        # Whether or not if it is chunked. It increments whenever
        # it is chunked, and decrement whenever chunked request is
        # processed.
        self.is_chunked = 0

        # For retraction
        self.is_retracted = False

        # Incremental streamining
        self.send_token_offset: int = 0
        self.send_decode_id_offset: int = 0
        # TODO (Byron): send_output_token_logprobs_offset and send_decode_id_offset can be different in disaggregation mode
        # because the decode server does not have the first output token logprobs
        self.send_output_token_logprobs_offset: int = 0

        # Logprobs (arguments)
        self.return_logprob = return_logprob
        # Start index to compute logprob from.
        self.logprob_start_len = 0
        self.top_logprobs_num = top_logprobs_num
        self.token_ids_logprob = token_ids_logprob
        self.temp_scaled_logprobs = False
        self.top_p_normalized_logprobs = False

        # Logprobs (return values)
        # True means the input logprob has been already sent to detokenizer.
        self.input_logprob_sent: bool = False
        self.input_token_logprobs_val: Optional[List[float]] = None
        self.input_token_logprobs_idx: Optional[List[int]] = None
        self.input_top_logprobs_val: Optional[List[float]] = None
        self.input_top_logprobs_idx: Optional[List[int]] = None
        self.input_token_ids_logprobs_val: Optional[List[float]] = None
        self.input_token_ids_logprobs_idx: Optional[List[int]] = None
        # Temporary holder to store input_token_logprobs.
        self.input_token_logprobs: Optional[List[Tuple[int]]] = None
        self.temp_input_top_logprobs_val: Optional[List[torch.Tensor]] = None
        self.temp_input_top_logprobs_idx: Optional[List[int]] = None
        self.temp_input_token_ids_logprobs_val: Optional[List[float]] = None
        self.temp_input_token_ids_logprobs_idx: Optional[List[int]] = None

        if return_logprob:
            # shape: (bs, 1)
            self.output_token_logprobs_val = []
            self.output_token_logprobs_idx = []
            # shape: (bs, k)
            self.output_top_logprobs_val = []
            self.output_top_logprobs_idx = []
            # Can contain either lists or GPU tensors (delayed copy optimization for prefill-only scoring)
            self.output_token_ids_logprobs_val: List[
                Union[List[float], torch.Tensor]
            ] = []
            self.output_token_ids_logprobs_idx = []
        else:
            self.output_token_logprobs_val = self.output_token_logprobs_idx = (
                self.output_top_logprobs_val
            ) = self.output_top_logprobs_idx = self.output_token_ids_logprobs_val = (
                self.output_token_ids_logprobs_idx
            ) = None
        self.hidden_states: List[List[float]] = []
        self.hidden_states_tensor = None  # Note: use tensor instead of list to transfer hidden_states when PD + MTP
        self.output_topk_p = None
        self.output_topk_index = None

        # Embedding (return values)
        self.embedding = None

        # Constrained decoding
        self.grammar: Optional[BaseGrammarObject] = None
        self.grammar_wait_ct = 0

        # The number of cached tokens that were already cached in the KV cache
        self.cached_tokens = 0
        self.already_computed = 0

        # The number of verification forward passes in the speculative decoding.
        # This is used to compute the average acceptance length per request.
        self.spec_verify_ct = 0

        # The number of accepted tokens in speculative decoding for this request.
        # This is used to compute the acceptance rate and average acceptance length per request.
        self.spec_accepted_tokens = 0

        # The number of times this request has been retracted / preempted.
        self.retraction_count = 0

        # For metrics
        self.metrics_collector = metrics_collector
        self.time_stats: TimeStats = TimeStats(disagg_mode=disagg_mode)
        self.has_log_time_stats: bool = False
        self.last_tic = time.monotonic()

        # For disaggregation
        self.bootstrap_host: str = bootstrap_host
        self.bootstrap_port: Optional[int] = bootstrap_port
        self.bootstrap_room: Optional[int] = bootstrap_room
        self.disagg_kv_sender: Optional[BaseKVSender] = None

        # For data parallel rank routing
        self.data_parallel_rank: Optional[int] = data_parallel_rank

        # the start index of the sent kv cache
        # We want to send it chunk by chunk for chunked prefill.
        # After every chunk forward, we do the following:
        # kv_send(req.input_ids[req.start_send_idx:len(req.fill_ids)])
        # start_send_idx = len(req.fill_ids)
        self.start_send_idx: int = 0

        # For overlap schedule, we delay the kv transfer until `process_batch_result_disagg_prefill` rather than `process_prefill_chunk` in non-overlap
        # This is because kv is not ready in `process_prefill_chunk`.
        # We use `tmp_end_idx` to store the end index of the kv cache to send.
        self.tmp_end_idx: int = -1
        self.metadata_buffer_index: int = -1

        # For Matryoshka embeddings
        self.dimensions = dimensions

    @property
    def seqlen(self):
        return len(self.origin_input_ids) + len(self.output_ids)

    @property
    def is_prefill_only(self) -> bool:
        """Check if this request is prefill-only (no token generation needed)."""
        # NOTE: when spec is enabled, prefill_only optimizations are disabled

        spec_alg = get_global_server_args().speculative_algorithm
        return self.sampling_params.max_new_tokens == 0 and spec_alg is None

    @property
    def output_ids_through_stop(self) -> List[int]:
        """Get the output ids through the stop condition. Stop position is included."""
        if self.finished_len is not None:
            return self.output_ids[: self.finished_len]
        return self.output_ids

    def add_latency(self, stage: RequestStage):
        if self.metrics_collector is None:
            return

        now = time.monotonic()
        self.metrics_collector.observe_per_stage_req_latency(
            stage.value, now - self.last_tic
        )
        self.last_tic = now

    def extend_image_inputs(self, image_inputs):
        if self.multimodal_inputs is None:
            self.multimodal_inputs = image_inputs
        else:
            self.multimodal_inputs.merge(image_inputs)

    def finished(self) -> bool:
        # Whether request reached finished condition
        return self.finished_reason is not None

    def init_next_round_input(self, tree_cache: Optional[BasePrefixCache] = None):
        self.fill_ids = self.origin_input_ids + self.output_ids
        input_len = len(self.fill_ids)
        # NOTE: the matched length is at most 1 less than the input length to enable logprob computation
        max_prefix_len = input_len - 1
        if self.return_logprob:
            max_prefix_len = min(max_prefix_len, self.logprob_start_len)
        max_prefix_len = max(max_prefix_len, 0)
        token_ids = self.fill_ids[:max_prefix_len]

        if tree_cache is not None:
            (
                self.prefix_indices,
                self.last_node,
                self.last_host_node,
                self.host_hit_length,
            ) = tree_cache.match_prefix(
                key=RadixKey(token_ids=token_ids, extra_key=self.extra_key),
                **(
                    {"req": self, "cow_mamba": True}
                    if isinstance(tree_cache, MambaRadixCache)
                    else {}
                ),
            )
            self.last_matched_prefix_len = len(self.prefix_indices)
        self.extend_input_len = len(self.fill_ids) - len(self.prefix_indices)

    # Based on https://github.com/vllm-project/vllm/blob/7a64d24aad69e4d2548aa0bf528d9fe63428ab01/vllm/transformers_utils/detokenizer.py#L194-L313
    def init_incremental_detokenize(self):
        first_iter = self.surr_offset is None or self.read_offset is None

        output_ids = self.output_ids_through_stop

        if first_iter:
            self.read_offset = len(self.origin_input_ids_unpadded)
            self.surr_offset = max(
                self.read_offset - INIT_INCREMENTAL_DETOKENIZATION_OFFSET, 0
            )
            self.surr_and_decode_ids = (
                self.origin_input_ids_unpadded[self.surr_offset :] + output_ids
            )
            self.cur_decode_ids_len = len(output_ids)
        else:
            self.surr_and_decode_ids.extend(output_ids[self.cur_decode_ids_len :])
            self.cur_decode_ids_len = len(output_ids)

        return self.surr_and_decode_ids, self.read_offset - self.surr_offset

    def tail_str(self) -> str:
        # Check stop strings and stop regex patterns together
        if (
            len(self.sampling_params.stop_strs) > 0
            or len(self.sampling_params.stop_regex_strs) > 0
        ):
            max_len_tail_str = max(
                self.sampling_params.stop_str_max_len + 1,
                self.sampling_params.stop_regex_max_len + 1,
            )

        tail_len = min((max_len_tail_str + 1), len(self.output_ids))
        return self.tokenizer.decode(self.output_ids[-tail_len:])

    def check_match_stop_str_prefix(self) -> bool:
        """
        Check if the suffix of tail_str overlaps with any stop_str prefix
        """
        if not self.sampling_params.stop_strs:
            return False

        tail_str = self.tail_str()

        # Early return if tail_str is empty
        if not tail_str:
            return False

        for stop_str in self.sampling_params.stop_strs:
            if not stop_str:
                continue
            # Check if stop_str is contained in tail_str (fastest check first)
            if stop_str in tail_str:
                return True

            # Check if tail_str suffix matches stop_str prefix
            # Only check if stop_str is not empty, it's for stream output
            min_len = min(len(tail_str), len(stop_str))
            for i in range(1, min_len + 1):
                if tail_str[-i:] == stop_str[:i]:
                    return True

        return False

    def _check_token_based_finish(self, new_accepted_tokens: List[int]) -> bool:
        if self.sampling_params.ignore_eos:
            return False

        # Check stop token ids
        matched_eos = False

        for i, token_id in enumerate(new_accepted_tokens):
            if self.sampling_params.stop_token_ids:
                matched_eos |= token_id in self.sampling_params.stop_token_ids
            if self.eos_token_ids:
                matched_eos |= token_id in self.eos_token_ids
            if self.tokenizer is not None:
                matched_eos |= token_id == self.tokenizer.eos_token_id
                if self.tokenizer.additional_stop_token_ids:
                    matched_eos |= token_id in self.tokenizer.additional_stop_token_ids
            if matched_eos:
                self.finished_reason = FINISH_MATCHED_TOKEN(matched=token_id)
                matched_pos = len(self.output_ids) - len(new_accepted_tokens) + i
                self.finished_len = matched_pos + 1
                return True

        return False

    def _check_str_based_finish(self):
        if (
            len(self.sampling_params.stop_strs) > 0
            or len(self.sampling_params.stop_regex_strs) > 0
        ):
            tail_str = self.tail_str()

            # Check stop strings
            if len(self.sampling_params.stop_strs) > 0:
                for stop_str in self.sampling_params.stop_strs:
                    if stop_str in tail_str or stop_str in self.decoded_text:
                        self.finished_reason = FINISH_MATCHED_STR(matched=stop_str)
                        return True

            # Check stop regex
            if len(self.sampling_params.stop_regex_strs) > 0:
                for stop_regex_str in self.sampling_params.stop_regex_strs:
                    if re.search(stop_regex_str, tail_str):
                        self.finished_reason = FINISHED_MATCHED_REGEX(
                            matched=stop_regex_str
                        )
                        return True

        return False

    def _check_vocab_boundary_finish(self, new_accepted_tokens: List[int] = None):
        for i, token_id in enumerate(new_accepted_tokens):
            if token_id > self.vocab_size or token_id < 0:
                offset = len(self.output_ids) - len(new_accepted_tokens) + i
                if self.sampling_params.stop_token_ids:
                    self.output_ids[offset] = next(
                        iter(self.sampling_params.stop_token_ids)
                    )
                if self.eos_token_ids:
                    self.output_ids[offset] = next(iter(self.eos_token_ids))
                self.finished_reason = FINISH_MATCHED_STR(matched="NaN happened")
                self.finished_len = offset + 1
                return True

        return False

    def check_finished(self, new_accepted_len: int = 1):
        if self.finished():
            return

        if self.to_finish:
            self.finished_reason = self.to_finish
            self.to_finish = None
            return

        if len(self.output_ids) >= self.sampling_params.max_new_tokens:
            self.finished_reason = FINISH_LENGTH(
                length=self.sampling_params.max_new_tokens
            )
            self.finished_len = self.sampling_params.max_new_tokens
            return

        if self.grammar is not None:
            if self.grammar.is_terminated():
                self.finished_reason = FINISH_MATCHED_TOKEN(matched=self.output_ids[-1])
                return

        new_accepted_tokens = self.output_ids[-new_accepted_len:]

        if self._check_token_based_finish(new_accepted_tokens):
            return

        if self._check_vocab_boundary_finish(new_accepted_tokens):
            return

        if self._check_str_based_finish():
            return

    def reset_for_retract(self):
        # Increment retraction count before resetting other state. We should not reset this
        # since we are tracking the total number of retractions for each request.
        self.retraction_count += 1

        self.prefix_indices = torch.empty((0,), dtype=torch.int64)
        self.last_node = None
        self.swa_uuid_for_lock = None
        self.extend_input_len = 0
        self.is_retracted = True
        self.input_token_logprobs = None
        self.temp_input_top_logprobs_val = None
        self.temp_input_top_logprobs_idx = None
        self.extend_logprob_start_len = 0
        self.is_chunked = 0
        self.mamba_pool_idx = None
        self.already_computed = 0

    def offload_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):
        token_indices = req_to_token_pool.req_to_token[
            self.req_pool_idx, : self.seqlen - 1
        ]
        self.kv_cache_cpu = token_to_kv_pool_allocator.get_cpu_copy(token_indices)

    def load_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):
        token_indices = req_to_token_pool.req_to_token[
            self.req_pool_idx, : self.seqlen - 1
        ]
        token_to_kv_pool_allocator.load_cpu_copy(self.kv_cache_cpu, token_indices)
        del self.kv_cache_cpu

    def log_time_stats(self):
        # If overlap schedule, we schedule one decode batch ahead so this gets called twice.
        if self.has_log_time_stats is True:
            return

        if self.bootstrap_room is not None:
            prefix = f"Req Time Stats(rid={self.rid}, bootstrap_room={self.bootstrap_room}, input len={len(self.origin_input_ids)}, output len={len(self.output_ids)}, type={self.time_stats.disagg_mode_str()})"
        else:
            prefix = f"Req Time Stats(rid={self.rid}, input len={len(self.origin_input_ids)}, output len={len(self.output_ids)}, type={self.time_stats.disagg_mode_str()})"
        logger.info(f"{prefix}: {self.time_stats.convert_to_duration()}")
        self.has_log_time_stats = True

    def set_finish_with_abort(self, error_msg: str):
        if get_tensor_model_parallel_rank() == 0:
            logger.error(f"{error_msg}, {self.rid=}")
        self.multimodal_inputs = None
        self.grammar = None
        self.origin_input_ids = [0]  # set it to one token to skip the long prefill
        self.return_logprob = False
        self.to_finish = FINISH_ABORT(
            error_msg, HTTPStatus.BAD_REQUEST, "BadRequestError"
        )

    def __repr__(self):
        return (
            f"Req(rid={self.rid}, "
            f"input_ids={self.origin_input_ids}, output_ids={self.output_ids}, "
            f"{self.grammar=}, "
            f"{self.sampling_params=})"
        )


@dataclasses.dataclass
class ScheduleBatch(ScheduleBatchDisaggregationDecodeMixin):
    """Store all information of a batch on the scheduler."""

    # Request, memory pool, and cache
    reqs: List[Req]
    req_to_token_pool: ReqToTokenPool = None
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator = None
    tree_cache: BasePrefixCache = None
    is_hybrid: bool = False

    # Batch configs
    model_config: ModelConfig = None
    forward_mode: ForwardMode = None
    enable_overlap: bool = False
    # Tell whether the current running batch is full so that we can skip
    # the check of whether to prefill new requests.
    # This is an optimization to reduce the overhead of the prefill check.
    batch_is_full: bool = False

    # For chunked prefill in PP
    chunked_req: Optional[Req] = None

    # Sampling info
    sampling_info: SamplingBatchInfo = None

    # Batched arguments to model runner
    input_ids: torch.Tensor = None  # shape: [b], int64
    input_embeds: torch.Tensor = None  # shape: [b, hidden_size], float32
    token_type_ids: torch.Tensor = None  # shape: [b], int64
    req_pool_indices: torch.Tensor = None  # shape: [b], int64
    seq_lens: torch.Tensor = None  # shape: [b], int64
    seq_lens_cpu: torch.Tensor = None  # shape: [b], int64
    # The output locations of the KV cache
    out_cache_loc: torch.Tensor = None  # shape: [b], int64
    output_ids: torch.Tensor = None  # shape: [b], int64

    # For multimodal inputs
    multimodal_inputs: Optional[List] = None

    # The sum of all sequence lengths
    seq_lens_sum: int = None
    # The original sequence lengths, Qwen-1M related
    orig_seq_lens: torch.Tensor = None  # shape: [b], int32

    # For DP attention
    global_num_tokens: Optional[List[int]] = None
    global_num_tokens_for_logprob: Optional[List[int]] = None
    is_extend_in_batch: bool = False
    can_run_dp_cuda_graph: bool = False
    tbo_split_seq_index: Optional[int] = None
    global_forward_mode: Optional[ForwardMode] = None

    # For processing logprobs
    return_logprob: bool = False
    top_logprobs_nums: Optional[List[int]] = None
    token_ids_logprobs: Optional[List[List[int]]] = None

    # For logits and logprob post processing
    temp_scaled_logprobs: bool = False
    top_p_normalized_logprobs: bool = False

    # For extend and mixed chunekd prefill
    prefix_lens: List[int] = None
    extend_lens: List[int] = None
    extend_num_tokens: Optional[int] = None
    decoding_reqs: List[Req] = None
    extend_logprob_start_lens: List[int] = None
    # It comes empty list if logprob is not required.
    extend_input_logprob_token_ids: Optional[torch.Tensor] = None

    # For encoder-decoder architectures
    encoder_cached: Optional[List[bool]] = None
    encoder_lens: Optional[torch.Tensor] = None
    encoder_lens_cpu: Optional[List[int]] = None
    encoder_out_cache_loc: Optional[torch.Tensor] = None

    # For matryoshka embeddings
    dimensions: Optional[list[int]] = None

    # For split prefill
    split_index: int = 0
    split_prefill_finished: bool = False
    split_forward_count: int = 1
    split_forward_batch: ForwardBatch = None
    seq_lens_cpu_cache: torch.Tensor = None

    # Stream
    has_stream: bool = False

    # Has grammar
    has_grammar: bool = False

    # Device
    if not _is_npu:
        device: str = "cuda"
    else:
        device: str = "npu"

    # Speculative decoding
    spec_algorithm: SpeculativeAlgorithm = None
    # spec_info: Optional[SpecInput] = None
    spec_info: Optional[SpecInput] = None

    # Whether to return hidden states
    return_hidden_states: bool = False

    # Whether this batch is prefill-only (no token generation needed)
    is_prefill_only: bool = False

    # hicache pointer for synchronizing data loading from CPU to GPU
    hicache_consumer_index: int = -1

    @classmethod
    def init_new(
        cls,
        reqs: List[Req],
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tree_cache: BasePrefixCache,
        model_config: ModelConfig,
        enable_overlap: bool,
        spec_algorithm: SpeculativeAlgorithm,
        chunked_req: Optional[Req] = None,
    ):
        return_logprob = any(req.return_logprob for req in reqs)

        is_hybrid = False
        if isinstance(token_to_kv_pool_allocator, SWATokenToKVPoolAllocator):
            assert (
                tree_cache is None
                or isinstance(tree_cache, SWARadixCache)
                or isinstance(tree_cache, SWAChunkCache)
            ), "SWARadixCache or SWAChunkCache is required for SWATokenToKVPoolAllocator"
            is_hybrid = True

        return cls(
            reqs=reqs,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            is_hybrid=is_hybrid,
            model_config=model_config,
            enable_overlap=enable_overlap,
            return_logprob=return_logprob,
            has_stream=any(req.stream for req in reqs),
            has_grammar=any(req.grammar for req in reqs),
            device=req_to_token_pool.device,
            spec_algorithm=spec_algorithm,
            return_hidden_states=any(req.return_hidden_states for req in reqs),
            is_prefill_only=all(req.is_prefill_only for req in reqs),
            chunked_req=chunked_req,
        )

    def batch_size(self):
        return len(self.reqs)

    def is_empty(self):
        return len(self.reqs) == 0

    def prepare_encoder_info_extend(self, input_ids: List[int], seq_lens: List[int]):
        self.encoder_lens_cpu = []
        self.encoder_cached = []

        for req in self.reqs:
            im = req.multimodal_inputs
            if im is None or im.num_image_tokens is None:
                # No image input
                self.encoder_lens_cpu.append(0)
                self.encoder_cached.append(True)
            else:
                self.encoder_lens_cpu.append(im.num_image_tokens)
                self.encoder_cached.append(
                    self.forward_mode.is_decode()
                    or len(req.prefix_indices) >= im.num_image_tokens
                )

        self.encoder_lens = torch.tensor(self.encoder_lens_cpu, dtype=torch.int64).to(
            self.device, non_blocking=True
        )

        # Strip encoder infos
        pt = 0
        decoder_out_cache_loc = []
        encoder_out_cache_loc = []
        for i, req in enumerate(self.reqs):
            encoder_len = self.encoder_lens_cpu[i]
            seq_lens[i] -= encoder_len

            if len(req.prefix_indices) < encoder_len:
                # NOTE: the encoder part should be considered as a whole
                assert len(req.prefix_indices) == 0
                input_ids[i] = input_ids[i][encoder_len:]
                encoder_out_cache_loc.append(self.out_cache_loc[pt : pt + encoder_len])
                decoder_out_cache_loc.append(
                    self.out_cache_loc[pt + encoder_len : pt + req.extend_input_len]
                )
                self.extend_lens[i] -= encoder_len
                self.extend_num_tokens -= encoder_len
            else:
                decoder_out_cache_loc.append(
                    self.out_cache_loc[pt : pt + req.extend_input_len]
                )
                self.prefix_lens[i] -= encoder_len

            pt += req.extend_input_len

        # Reassign
        self.input_ids = torch.tensor(sum(input_ids, []), dtype=torch.int64).to(
            self.device, non_blocking=True
        )
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int64).to(
            self.device, non_blocking=True
        )
        self.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)

        if not decoder_out_cache_loc:
            self.out_cache_loc = torch.zeros(0, dtype=torch.int64).to(
                self.device, non_blocking=True
            )
        else:
            self.out_cache_loc = torch.cat(decoder_out_cache_loc)

        if not encoder_out_cache_loc:
            self.encoder_out_cache_loc = torch.zeros(0, dtype=torch.int64).to(
                self.device, non_blocking=True
            )
        else:
            self.encoder_out_cache_loc = torch.cat(encoder_out_cache_loc)

        assert (
            len(self.out_cache_loc) == self.extend_num_tokens
        ), f"Expected {len(self.out_cache_loc)}, got {self.extend_num_tokens}"

    def prepare_for_extend(self):
        self.forward_mode = ForwardMode.EXTEND

        # Init tensors
        reqs = self.reqs
        input_ids = [r.fill_ids[len(r.prefix_indices) :] for r in reqs]
        extend_num_tokens = sum(len(ids) for ids in input_ids)
        seq_lens = [len(r.fill_ids) for r in reqs]
        orig_seq_lens = [max(len(r.fill_ids), len(r.origin_input_ids)) for r in reqs]
        prefix_lens = [len(r.prefix_indices) for r in reqs]
        extend_lens = [r.extend_input_len for r in reqs]

        # For matryoshka embeddings
        if self.model_config.is_matryoshka and any(
            r.dimensions is not None for r in reqs
        ):
            self.dimensions = [
                r.dimensions if r.dimensions else self.model_config.hidden_size
                for r in reqs
            ]

        token_type_ids = [
            r.token_type_ids for r in reqs if r.token_type_ids is not None
        ]

        input_ids_tensor = torch.tensor(
            list(chain.from_iterable(input_ids)), dtype=torch.int64
        ).to(self.device, non_blocking=True)
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int64).to(
            self.device, non_blocking=True
        )
        seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        orig_seq_lens_tensor = torch.tensor(orig_seq_lens, dtype=torch.int32).to(
            self.device, non_blocking=True
        )

        token_type_ids_tensor = None
        if len(token_type_ids) > 0:
            token_type_ids_tensor = torch.tensor(
                sum(token_type_ids, []), dtype=torch.int64
            ).to(self.device, non_blocking=True)

        # Set batch fields needed by alloc_for_extend
        self.prefix_lens = prefix_lens
        self.extend_lens = extend_lens
        self.seq_lens = seq_lens_tensor
        self.seq_lens_cpu = seq_lens_cpu
        self.extend_num_tokens = extend_num_tokens

        # Allocate memory
        out_cache_loc, req_pool_indices_tensor, req_pool_indices = alloc_for_extend(
            self
        )

        # Set fields
        input_embeds = []
        extend_input_logprob_token_ids = []
        multimodal_inputs = []

        for i, (req, seq_len, pre_len) in enumerate(zip(reqs, seq_lens, prefix_lens)):
            req.req_pool_idx = req_pool_indices[i]
            assert seq_len - pre_len == req.extend_input_len

            # If input_embeds are available, store them
            if req.input_embeds is not None:
                # If req.input_embeds is already a list, append its content directly
                input_embeds.extend(req.input_embeds)  # Use extend to avoid nesting

            multimodal_inputs.append(req.multimodal_inputs)

            req.cached_tokens += pre_len - req.already_computed
            req.already_computed = seq_len
            req.is_retracted = False

            # Compute the relative logprob_start_len in an extend batch
            #
            # Key variables:
            # - logprob_start_len: Absolute position in full sequence where logprob computation begins
            # - extend_logprob_start_len: Relative position within current extend batch where logprob computation begins
            # - extend_input_len: Number of tokens that need to be processed in this extend batch
            #   (= len(fill_ids) - len(prefix_indices), where fill_ids = origin_input_ids + output_ids
            #    and prefix_indices are the cached/shared prefix tokens)
            #
            if req.logprob_start_len >= pre_len:
                # Optimization for prefill-only requests: When we only need logprobs at
                # positions beyond the input sequence (to score next-token likelihood), skip all
                # input logprob computation during prefill since no generation will occur.
                if self.is_prefill_only and req.logprob_start_len == len(
                    req.origin_input_ids
                ):
                    # Skip ALL input logprobs: set extend_logprob_start_len = extend_input_len
                    req.extend_logprob_start_len = req.extend_input_len
                else:
                    # Convert absolute logprob_start_len to relative extend_logprob_start_len
                    #
                    # Example: origin_input_ids=[1,2,3,4,5] (5 tokens, positions 0-4), logprob_start_len=3
                    # Regular logic: min(3-0, 5, 5-1) = min(3,5,4) = 3
                    # This means: "compute logprobs from position 3 onwards in extend batch"
                    req.extend_logprob_start_len = min(
                        req.logprob_start_len - pre_len,
                        req.extend_input_len,
                        req.seqlen - 1,
                    )
            else:
                # logprob_start_len is before the current extend batch, so start from beginning
                req.extend_logprob_start_len = 0

            if self.return_logprob:
                # Find input logprob token ids.
                # First, find a global index within origin_input_ids and slide it by 1
                # to compute input logprobs. It is because you need the next token
                # to compute input logprobs. E.g., (chunk size 2)
                #
                # input_logprobs = [1, 2, 3, 4]
                # fill_ids = [1, 2]
                # extend_input_logprob_token_id = [2, 3]
                #
                # Note that it can also overflow. In this case, we pad it with 0.
                # input_logprobs = [1, 2, 3, 4]
                # fill_ids = [3, 4]
                # extend_input_logprob_token_id = [4, 0]
                global_start_idx, global_end_idx = (
                    len(req.prefix_indices),
                    len(req.fill_ids),
                )
                # Apply logprob_start_len
                if global_start_idx < req.logprob_start_len:
                    global_start_idx = req.logprob_start_len

                logprob_token_ids = req.origin_input_ids[
                    global_start_idx + 1 : global_end_idx + 1
                ]
                extend_input_logprob_token_ids.extend(logprob_token_ids)

                # We will need req.extend_input_len - req.extend_logprob_start_len number of
                # tokens, and logprob_token_ids is for input logprob, so pad the rest of them by 0.
                extend_input_logprob_token_ids.extend(
                    [0]
                    * (
                        req.extend_input_len
                        - req.extend_logprob_start_len
                        - len(logprob_token_ids)
                    )
                )

        if self.return_logprob:
            extend_input_logprob_token_ids = torch.tensor(
                extend_input_logprob_token_ids
            )
        else:
            extend_input_logprob_token_ids = None

        self.input_ids = input_ids_tensor
        self.req_pool_indices = req_pool_indices_tensor
        self.orig_seq_lens = orig_seq_lens_tensor
        self.out_cache_loc = out_cache_loc
        self.input_embeds = (
            torch.tensor(input_embeds).to(self.device, non_blocking=True)
            if input_embeds
            else None
        )
        for mm_input in multimodal_inputs:
            if mm_input is None:
                continue
            for mm_item in mm_input.mm_items:
                pixel_values = getattr(mm_item, "feature", None)
                if isinstance(pixel_values, torch.Tensor):
                    mm_item.feature = pixel_values.to(self.device, non_blocking=True)
                elif isinstance(pixel_values, CudaIpcTensorTransportProxy):
                    mm_item.feature = pixel_values.reconstruct_on_target_device(
                        torch.cuda.current_device()
                    )
        self.multimodal_inputs = multimodal_inputs
        self.token_type_ids = token_type_ids_tensor
        self.seq_lens_sum = sum(seq_lens)

        if self.return_logprob:
            self.top_logprobs_nums = [r.top_logprobs_num for r in reqs]
            self.token_ids_logprobs = [r.token_ids_logprob for r in reqs]

        self.extend_logprob_start_lens = [r.extend_logprob_start_len for r in reqs]
        self.extend_input_logprob_token_ids = extend_input_logprob_token_ids

        if self.model_config.is_encoder_decoder:
            self.prepare_encoder_info_extend(input_ids, seq_lens)

        # Build sampling info
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
        )

    def prepare_for_split_prefill(self):
        self.prepare_for_extend()
        # For split prefill, we need to set the forward mode to SPLIT_PREFILL
        self.forward_mode = ForwardMode.SPLIT_PREFILL

    def mix_with_running(self, running_batch: "ScheduleBatch"):
        self.forward_mode = ForwardMode.MIXED
        running_bs = running_batch.batch_size()

        for req in running_batch.reqs:
            req.fill_ids = req.origin_input_ids + req.output_ids
            req.extend_input_len = 1

        input_ids = torch.cat([self.input_ids, running_batch.input_ids])
        out_cache_loc = torch.cat([self.out_cache_loc, running_batch.out_cache_loc])

        self.merge_batch(running_batch)
        self.input_ids = input_ids
        self.out_cache_loc = out_cache_loc

        # For overlap scheduler, the output_ids has one step delay
        delta = 0 if self.enable_overlap else -1

        # NOTE: prefix_indices is what has been cached, but we don't cache each decode step
        self.prefix_lens.extend(
            [
                len(r.origin_input_ids) + len(r.output_ids) + delta
                for r in running_batch.reqs
            ]
        )
        self.extend_lens.extend([1] * running_bs)
        self.extend_num_tokens += running_bs
        # TODO (lianmin): Revisit this. It should be seq_len - 1
        self.extend_logprob_start_lens.extend([0] * running_bs)
        self.is_prefill_only = False

    def new_page_count_next_decode(self, selected_indices: Optional[List[int]] = None):
        page_size = self.token_to_kv_pool_allocator.page_size
        requests = (
            self.reqs
            if selected_indices is None
            else [self.reqs[i] for i in selected_indices]
        )
        if page_size == 1:
            return len(requests)
        # In the decoding phase, the length of a request's KV cache should be
        # the total length of the request minus 1
        return (
            sum(1 for req in requests if req.seqlen % page_size == 0)
            if self.enable_overlap
            else sum(1 for req in requests if (req.seqlen - 1) % page_size == 0)
        )

    def check_decode_mem(
        self, buf_multiplier=1, selected_indices: Optional[List[int]] = None
    ):
        num_tokens = (
            self.new_page_count_next_decode(selected_indices)
            * buf_multiplier
            * self.token_to_kv_pool_allocator.page_size
        )

        evict_from_tree_cache(self.tree_cache, num_tokens)
        return self._is_available_size_sufficient(num_tokens)

    def retract_decode(
        self, server_args: ServerArgs
    ) -> Tuple[List[Req], float, List[Req]]:
        """Retract the decoding requests when there is not enough memory."""
        sorted_indices = list(range(len(self.reqs)))

        # TODO(lsyin): improve retraction policy for radix cache
        # For spec decoding, filter_batch API can only filter
        # requests from the back, so we can only retract from the back.
        # TODO(sang): Clean up finish path and support better retract
        # policy.
        if not server_args.speculative_algorithm:
            sorted_indices.sort(
                key=lambda i: (
                    len(self.reqs[i].output_ids),
                    -len(self.reqs[i].origin_input_ids),
                ),
                reverse=True,
            )

        retracted_reqs = []
        first_iter = True
        while first_iter or (
            not self.check_decode_mem(selected_indices=sorted_indices)
        ):
            if len(sorted_indices) == 1:
                # Corner case: only one request left
                if self.is_hybrid:
                    full_available_size = (
                        self.token_to_kv_pool_allocator.full_available_size()
                    )
                    swa_available_size = (
                        self.token_to_kv_pool_allocator.swa_available_size()
                    )
                    assert (
                        full_available_size > 0 and swa_available_size > 0
                    ), f"No space left for only one request in SWA mode {full_available_size=}, {swa_available_size=}"
                else:
                    assert (
                        self.token_to_kv_pool_allocator.available_size() > 0
                    ), f"No space left for only one request, {self.token_to_kv_pool_allocator.available_size()=}"
                break

            first_iter = False
            idx = sorted_indices.pop()
            req = self.reqs[idx]
            retracted_reqs.append(req)
            # release memory and don't insert into the tree because we need the space instantly
            self.release_req(idx, len(sorted_indices), server_args)

            if len(retracted_reqs) == 0:
                # Corner case: only one request left
                raise ValueError(
                    "Failed to retract any request. No space left for only one request."
                )

        self.filter_batch(keep_indices=sorted_indices)

        # Reqs in batch are filtered
        total_decoded_tokens = sum(len(r.output_ids) for r in self.reqs)
        total_max_new_tokens = sum(r.sampling_params.max_new_tokens for r in self.reqs)

        new_estimate_ratio = (
            total_decoded_tokens
            + envs.SGLANG_RETRACT_DECODE_STEPS.get() * len(self.reqs)
        ) / (
            total_max_new_tokens + 1
        )  # avoid zero division
        new_estimate_ratio = min(1.0, new_estimate_ratio)

        return retracted_reqs, new_estimate_ratio, []

    def release_req(self, idx: int, remaing_req_count: int, server_args: ServerArgs):
        req = self.reqs[idx]

        if server_args.disaggregation_mode == "decode":
            req.offload_kv_cache(
                self.req_to_token_pool, self.token_to_kv_pool_allocator
            )
        # TODO (csy): for preempted requests, we may want to insert into the tree
        self.tree_cache.cache_finished_req(req, is_insert=False)
        # NOTE(lsyin): we should use the newly evictable memory instantly.
        num_tokens = remaing_req_count * envs.SGLANG_RETRACT_DECODE_STEPS.get()
        evict_from_tree_cache(self.tree_cache, num_tokens)

        req.reset_for_retract()

    def prepare_encoder_info_decode(self):
        # Reset the encoder cached status
        self.encoder_cached = [True] * len(self.reqs)

    def prepare_for_idle(self):
        self.forward_mode = ForwardMode.IDLE
        self.input_ids = torch.empty(0, dtype=torch.int64, device=self.device)
        self.seq_lens = torch.empty(0, dtype=torch.int64, device=self.device)
        self.seq_lens_cpu = torch.empty(0, dtype=torch.int64)
        self.orig_seq_lens = torch.empty(0, dtype=torch.int32, device=self.device)
        self.out_cache_loc = torch.empty(0, dtype=torch.int64, device=self.device)
        self.req_pool_indices = torch.empty(0, dtype=torch.int32, device=self.device)
        self.seq_lens_sum = 0
        self.extend_num_tokens = 0
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
        )

    @property
    def is_v2_eagle(self):
        # FIXME: finally deprecate is_v2_eagle
        return self.enable_overlap and self.spec_algorithm.is_eagle()

    def prepare_for_decode(self):
        self.forward_mode = ForwardMode.DECODE
        bs = len(self.reqs)

        if self.is_v2_eagle:
            # TODO(spec-v2): all v2 spec should go through this path
            draft_input: EagleDraftInput = self.spec_info
            draft_input.prepare_for_decode(self)

        if not self.spec_algorithm.is_none():
            # if spec decoding is used, the decode batch is prepared inside
            # `forward_batch_speculative_generation` after running draft models.
            return

        if self.sampling_info.penalizer_orchestrator.is_required:
            if self.enable_overlap:
                # TODO: this can be slow, optimize this.
                delayed_output_ids = torch.tensor(
                    [
                        (
                            req.output_ids[-1]
                            if len(req.output_ids)
                            else req.origin_input_ids[-1]
                        )
                        for req in self.reqs
                    ],
                    dtype=torch.int64,
                    device=self.device,
                )
                self.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                    delayed_output_ids
                )
            else:
                self.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                    self.output_ids.to(torch.int64)
                )

        # Update fields
        self.input_ids = self.output_ids
        self.output_ids = None

        if self.model_config.is_encoder_decoder:
            self.prepare_encoder_info_decode()

        # Allocate memory
        self.out_cache_loc = alloc_for_decode(self, token_per_req=1)

        # Update seq_lens after allocation
        if self.enable_overlap:
            # Do not use in-place operations in the overlap mode
            self.seq_lens = self.seq_lens + 1
            self.seq_lens_cpu = self.seq_lens_cpu + 1
            self.orig_seq_lens = self.orig_seq_lens + 1
        else:
            # A faster in-place version
            self.seq_lens.add_(1)
            self.seq_lens_cpu.add_(1)
            self.orig_seq_lens.add_(1)
        self.seq_lens_sum += bs

    def maybe_wait_verify_done(self):
        if self.is_v2_eagle:
            draft_input: EagleDraftInput = self.spec_info
            if draft_input.verify_done is not None:
                draft_input.verify_done.synchronize()

    def filter_batch(
        self,
        chunked_req_to_exclude: Optional[Union[Req, List[Req]]] = None,
        keep_indices: Optional[List[int]] = None,
    ):
        # FIXME(lsyin): used here to get the correct seq_lens
        # The batch has been launched but we need it verified to get correct next batch info
        self.maybe_wait_verify_done()

        if keep_indices is None:
            if isinstance(chunked_req_to_exclude, Req):
                chunked_req_to_exclude = [chunked_req_to_exclude]
            elif chunked_req_to_exclude is None:
                chunked_req_to_exclude = []
            keep_indices = [
                i
                for i in range(len(self.reqs))
                if not self.reqs[i].finished()
                and self.reqs[i] not in chunked_req_to_exclude
            ]

        if keep_indices is None or len(keep_indices) == 0:
            # Filter out all requests
            self.reqs = []
            return

        if len(keep_indices) == len(self.reqs):
            # No need to filter
            return

        keep_indices_device = torch.tensor(keep_indices, dtype=torch.int64).to(
            self.device, non_blocking=True
        )

        if self.model_config.is_encoder_decoder:
            self.encoder_lens = self.encoder_lens[keep_indices_device]
            self.encoder_lens_cpu = [self.encoder_lens_cpu[i] for i in keep_indices]

        self.reqs = [self.reqs[i] for i in keep_indices]
        if self.multimodal_inputs is not None:
            self.multimodal_inputs = [self.multimodal_inputs[i] for i in keep_indices]
        self.req_pool_indices = self.req_pool_indices[keep_indices_device]
        self.seq_lens = self.seq_lens[keep_indices_device]
        self.seq_lens_cpu = self.seq_lens_cpu[keep_indices]
        self.orig_seq_lens = self.orig_seq_lens[keep_indices_device]
        self.out_cache_loc = None
        self.seq_lens_sum = self.seq_lens.sum().item()
        self.output_ids = self.output_ids[keep_indices_device]
        self.return_logprob = any(req.return_logprob for req in self.reqs)
        if self.return_logprob:
            self.top_logprobs_nums = [self.top_logprobs_nums[i] for i in keep_indices]
            self.token_ids_logprobs = [self.token_ids_logprobs[i] for i in keep_indices]
        else:
            self.top_logprobs_nums = None
            self.token_ids_logprobs = None

        self.has_stream = any(req.stream for req in self.reqs)
        self.has_grammar = any(req.grammar for req in self.reqs)

        self.sampling_info.filter_batch(keep_indices, keep_indices_device)
        if self.spec_info:
            if chunked_req_to_exclude is not None and len(chunked_req_to_exclude) > 0:
                has_been_filtered = False
            else:
                has_been_filtered = True
            self.spec_info.filter_batch(
                new_indices=keep_indices_device,
                has_been_filtered=has_been_filtered,
            )

    def merge_batch(self, other: "ScheduleBatch"):
        # NOTE: in v2 eagle mode, we do not need wait verify here because
        # 1) current batch is always prefill, whose seq_lens and allocate_lens are not a future
        # 2) other batch is always decode, which is finished in previous step

        # Penalizer orchestrator must be merged before Batch.reqs is merged. This is because
        # orchestrator.merge() depends on Batch.reqs during preparation of each penalizers, so it
        # needs to be called with pre-merged Batch.reqs.
        self.sampling_info.merge_batch(other.sampling_info)

        # Encoder-decoder infos
        if self.model_config.is_encoder_decoder:
            self.encoder_lens = torch.cat([self.encoder_lens, other.encoder_lens])
            self.encoder_lens_cpu.extend(other.encoder_lens_cpu)
        self.req_pool_indices = torch.cat(
            [self.req_pool_indices, other.req_pool_indices]
        )
        self.seq_lens = torch.cat([self.seq_lens, other.seq_lens])
        self.seq_lens_cpu = torch.cat([self.seq_lens_cpu, other.seq_lens_cpu])
        self.orig_seq_lens = torch.cat([self.orig_seq_lens, other.orig_seq_lens])
        self.out_cache_loc = None
        self.seq_lens_sum += other.seq_lens_sum
        if self.output_ids is not None:
            self.output_ids = torch.cat([self.output_ids, other.output_ids])
        if self.return_logprob and other.return_logprob:
            self.top_logprobs_nums.extend(other.top_logprobs_nums)
            self.token_ids_logprobs.extend(other.token_ids_logprobs)
        elif self.return_logprob:
            self.top_logprobs_nums.extend([0] * len(other.reqs))
            self.token_ids_logprobs.extend([None] * len(other.reqs))
        elif other.return_logprob:
            self.top_logprobs_nums = [0] * len(self.reqs) + other.top_logprobs_nums
            self.token_ids_logprobs = [None] * len(self.reqs) + other.token_ids_logprobs
        self.reqs.extend(other.reqs)
        if self.multimodal_inputs is not None:
            self.multimodal_inputs.extend(other.multimodal_inputs)

        self.return_logprob |= other.return_logprob
        self.has_stream |= other.has_stream
        self.has_grammar |= other.has_grammar
        self.return_hidden_states |= other.return_hidden_states

        if self.spec_info:
            self.spec_info.merge_batch(other.spec_info)

    def get_model_worker_batch(
        self, seq_lens_cpu_cache: Optional[torch.Tensor] = None
    ) -> ModelWorkerBatch:
        if self.forward_mode.is_decode_or_idle():
            extend_seq_lens = extend_prefix_lens = extend_logprob_start_lens = None
        else:
            extend_seq_lens = self.extend_lens
            extend_prefix_lens = self.prefix_lens
            extend_logprob_start_lens = self.extend_logprob_start_lens

        if self.sampling_info:
            if self.has_grammar:
                self.sampling_info.grammars = [req.grammar for req in self.reqs]
            else:
                self.sampling_info.grammars = None

        seq_lens_cpu = (
            seq_lens_cpu_cache if seq_lens_cpu_cache is not None else self.seq_lens_cpu
        )

        return ModelWorkerBatch(
            forward_mode=self.forward_mode,
            input_ids=self.input_ids,
            req_pool_indices=self.req_pool_indices,
            seq_lens=self.seq_lens,
            orig_seq_lens=self.orig_seq_lens,
            out_cache_loc=self.out_cache_loc,
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_sum=self.seq_lens_sum,
            return_logprob=self.return_logprob,
            top_logprobs_nums=self.top_logprobs_nums,
            token_ids_logprobs=self.token_ids_logprobs,
            global_num_tokens=self.global_num_tokens,
            global_num_tokens_for_logprob=self.global_num_tokens_for_logprob,
            is_extend_in_batch=self.is_extend_in_batch,
            can_run_dp_cuda_graph=self.can_run_dp_cuda_graph,
            tbo_split_seq_index=self.tbo_split_seq_index,
            global_forward_mode=self.global_forward_mode,
            extend_num_tokens=self.extend_num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_logprob_start_lens=extend_logprob_start_lens,
            multimodal_inputs=self.multimodal_inputs,
            encoder_cached=self.encoder_cached,
            encoder_lens=self.encoder_lens,
            encoder_lens_cpu=self.encoder_lens_cpu,
            encoder_out_cache_loc=self.encoder_out_cache_loc,
            lora_ids=[req.lora_id for req in self.reqs],
            sampling_info=self.sampling_info,
            input_embeds=self.input_embeds,
            token_type_ids=self.token_type_ids,
            spec_algorithm=self.spec_algorithm,
            spec_info=self.spec_info,
            hicache_consumer_index=self.hicache_consumer_index,
            capture_hidden_mode=(
                CaptureHiddenMode.FULL
                if self.return_hidden_states
                else (
                    getattr(
                        self.spec_info, "capture_hidden_mode", CaptureHiddenMode.NULL
                    )
                    if self.spec_info
                    else CaptureHiddenMode.NULL
                )
            ),
            extend_input_logprob_token_ids=self.extend_input_logprob_token_ids,
            is_prefill_only=self.is_prefill_only,
            dimensions=self.dimensions,
        )

    def copy(self):
        # Only contain fields that will be used by process_batch_result
        return ScheduleBatch(
            reqs=self.reqs,
            req_to_token_pool=self.req_to_token_pool,
            req_pool_indices=self.req_pool_indices,
            model_config=self.model_config,
            forward_mode=self.forward_mode,
            out_cache_loc=self.out_cache_loc,
            return_logprob=self.return_logprob,
            decoding_reqs=self.decoding_reqs,
            spec_algorithm=self.spec_algorithm,
            global_num_tokens=self.global_num_tokens,
            global_num_tokens_for_logprob=self.global_num_tokens_for_logprob,
            can_run_dp_cuda_graph=self.can_run_dp_cuda_graph,
            is_extend_in_batch=self.is_extend_in_batch,
            is_prefill_only=self.is_prefill_only,
            seq_lens_cpu=self.seq_lens_cpu,
            enable_overlap=self.enable_overlap,
        )

    def _is_available_size_sufficient(self, num_tokens: int) -> bool:
        if self.is_hybrid:
            return (
                self.token_to_kv_pool_allocator.full_available_size() >= num_tokens
                and self.token_to_kv_pool_allocator.swa_available_size() >= num_tokens
            )
        else:
            return self.token_to_kv_pool_allocator.available_size() >= num_tokens

    def __str__(self):
        return (
            f"ScheduleBatch(forward_mode={self.forward_mode.name if self.forward_mode else 'None'}, "
            f"#req={(len(self.reqs))})"
        )


@dataclasses.dataclass
class ModelWorkerBatch:
    # The forward mode
    forward_mode: ForwardMode
    # The input ids
    input_ids: torch.Tensor
    # The indices of requests in the req_to_token_pool
    req_pool_indices: torch.Tensor
    # The sequence length
    seq_lens: torch.Tensor
    # The indices of output tokens in the token_to_kv_pool_allocator
    out_cache_loc: torch.Tensor
    # The sequence length tensor on CPU
    seq_lens_cpu: Optional[torch.Tensor]
    seq_lens_sum: int

    # For logprob
    return_logprob: bool
    top_logprobs_nums: Optional[List[int]]
    token_ids_logprobs: Optional[List[List[int]]]

    # For DP attention
    global_num_tokens: Optional[List[int]]
    global_num_tokens_for_logprob: Optional[List[int]]
    is_extend_in_batch: bool
    can_run_dp_cuda_graph: bool
    tbo_split_seq_index: Optional[int]
    global_forward_mode: Optional[ForwardMode]

    # For extend
    extend_num_tokens: Optional[int]
    extend_seq_lens: Optional[List[int]]
    extend_prefix_lens: Optional[List[int]]
    extend_logprob_start_lens: Optional[List[int]]
    extend_input_logprob_token_ids: Optional[torch.Tensor]

    # For multimodal
    multimodal_inputs: Optional[List[MultimodalInputs]]

    # For encoder-decoder
    encoder_cached: Optional[List[bool]]
    encoder_lens: Optional[torch.Tensor]
    encoder_lens_cpu: Optional[List[int]]
    encoder_out_cache_loc: Optional[torch.Tensor]

    # For LoRA
    lora_ids: Optional[List[str]]

    # Sampling info
    sampling_info: SamplingBatchInfo

    # The original sequence lengths, Qwen-1M related
    orig_seq_lens: Optional[torch.Tensor] = None

    # The input Embeds
    input_embeds: Optional[torch.Tensor] = None

    # For corss-encoder model
    token_type_ids: Optional[torch.Tensor] = None

    # Speculative decoding
    spec_algorithm: SpeculativeAlgorithm = None

    spec_info: Optional[SpecInput] = None

    # If set, the output of the batch contains the hidden states of the run.
    capture_hidden_mode: CaptureHiddenMode = None
    hicache_consumer_index: int = -1

    # For matryoshka embeddings
    dimensions: Optional[list[int]] = None

    # Whether this batch is prefill-only (no token generation needed)
    is_prefill_only: bool = False
