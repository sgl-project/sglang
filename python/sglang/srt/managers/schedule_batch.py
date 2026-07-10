from __future__ import annotations

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils.common import (
    Range,
    ceil_align,
    flatten_arrays_to_pinned_cpu,
    is_pin_memory_available,
)

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

ScheduleBatch -> ForwardBatch

- ScheduleBatch is managed by `scheduler.py::Scheduler`.
  It contains high-level scheduling data. Most of the data is on the CPU.
- ForwardBatch is managed by `model_runner.py::ModelRunner`.
  It contains low-level tensor data. Most of the data consists of GPU tensors.
  It is constructed directly from a ScheduleBatch by `ForwardBatch.init_new`.
"""

import copy
import dataclasses
import logging
import re
import sys
from array import array
from concurrent.futures import Future
from enum import Enum, auto
from functools import lru_cache
from http import HTTPStatus
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
)

import msgspec
import numpy as np
import torch

from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.disaggregation.base import BaseKVSender
from sglang.srt.disaggregation.decode_schedule_batch_mixin import (
    ScheduleBatchDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST, DisaggregationMode
from sglang.srt.dllm.mixin.req import ReqDllmMixin
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
    maybe_evict_dsv4_state,
)
from sglang.srt.managers.embed_types import PositionalEmbeds
from sglang.srt.managers.scheduler_components.new_token_ratio_tracker import (
    NewTokenRatioTracker,
)
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    EvictParams,
    MatchPrefixParams,
    zero_match_result,
)
from sglang.srt.mem_cache.common import (
    alloc_for_decode,
    alloc_for_extend,
    evict_from_tree_cache,
    free_swa_out_of_window_slots,
    get_alloc_reserve_per_decode,
    release_kv_cache,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.observability.metrics_collector import (
    DPCooperationInfo,
    SchedulerMetricsCollector,
)
from sglang.srt.observability.req_time_stats import (
    APIServerReqTimeStats,
    DPControllerReqTimeStats,
    SchedulerReqTimeStats,
)
from sglang.srt.runtime_context import get_parallel, get_server_args
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import flatten_nested_list
from sglang.srt.utils.cuda_ipc_transport_utils import CudaIpcTensorTransportProxy

if TYPE_CHECKING:
    from typing import Any, Dict

    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
    from sglang.srt.managers.scheduler_components.metrics_reporter import PrefillStats
    from sglang.srt.session.session_controller import Session
    from sglang.srt.speculative.spec_info import SpecInput, SpeculativeAlgorithm

INIT_INCREMENTAL_DETOKENIZATION_OFFSET = 5

# Constant used as the base offset for MM (multimodal) pad values.
# This ensures pad_values don't overlap with valid text token IDs.
MM_PAD_SHIFT_VALUE = 1_000_000

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def sanity_check_mm_pad_shift_value(vocab_size: int) -> None:
    if vocab_size > MM_PAD_SHIFT_VALUE:
        raise ValueError(
            f"Model vocab_size ({vocab_size}) exceeds MM_PAD_SHIFT_VALUE ({MM_PAD_SHIFT_VALUE}). "
            f"MM pad_values may overlap with valid token IDs. "
            f"Please increase MM_PAD_SHIFT_VALUE in schedule_batch.py."
        )


def _compute_pad_value(hash: int) -> int:
    """Compute pad value from hash."""
    return MM_PAD_SHIFT_VALUE + (hash % (1 << 30))


class BaseFinishReason:
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
        super().__init__()
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


class MultimodalInputFormat(Enum):
    NORMAL = auto()
    PROCESSOR_OUTPUT = auto()
    PRECOMPUTED_EMBEDDING = auto()


@dataclasses.dataclass
class MultimodalDataItem:
    """
    One MultimodalDataItem represents a single multimodal input (one image, one video, or one audio).
    For example, if there are 3 images and 1 audio, there will be 4 MultimodalDataItems.

    Each item has its own hash and pad_value, enabling per-image RadixAttention caching.

    We put the common fields first and the model-specific fields in model_specific_data.
    """

    modality: Modality
    hash: int = None
    pad_value: int = None
    offsets: Optional[list] = None

    format: MultimodalInputFormat = MultimodalInputFormat.NORMAL

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
        if self.pad_value is not None:
            return

        from sglang.srt.managers.mm_utils import hash_feature

        if envs.SGLANG_MM_SKIP_COMPUTE_HASH.get():
            import uuid

            self.hash = uuid.uuid4().int
            self.pad_value = _compute_pad_value(self.hash)
            return
        if self.hash is None:
            if self.feature is not None:
                hashed_feature = self.feature
            else:
                hashed_feature = self.precomputed_embeddings
            self.hash = hash_feature(hashed_feature)
        assert self.hash is not None
        self.pad_value = _compute_pad_value(self.hash)

    def is_modality(self, modality: Modality) -> bool:
        return self.modality == modality

    def is_audio(self):
        return self.modality == Modality.AUDIO

    def is_image(self):
        return self.modality == Modality.IMAGE

    def is_video(self):
        return self.modality == Modality.VIDEO

    def is_valid(self) -> bool:
        return self.is_image() or self.is_video() or self.is_audio()

    def validate(self):
        ...
        # TODO

    def is_precomputed_embedding(self):
        return self.format == MultimodalInputFormat.PRECOMPUTED_EMBEDDING

    @staticmethod
    def from_dict(obj: dict):
        kwargs = dict(obj)
        modality = kwargs.pop("modality")
        if isinstance(modality, str):
            modality = Modality[modality]
        ret = MultimodalDataItem(modality=modality, **kwargs)
        ret.validate()
        return ret

    def has_cuda_ipc_proxy(self):
        return (
            isinstance(self.feature, CudaIpcTensorTransportProxy)
            or isinstance(self.precomputed_embeddings, CudaIpcTensorTransportProxy)
            or any(
                isinstance(value, CudaIpcTensorTransportProxy)
                for value in self.model_specific_data.values()
            )
        )

    def reconstruct(self, target_device: int):
        """materialize cuda ipc proxy tensors in-place on target_device"""
        if isinstance(self.feature, CudaIpcTensorTransportProxy):
            self.feature = self.feature.reconstruct_on_target_device(target_device)
        if isinstance(self.precomputed_embeddings, CudaIpcTensorTransportProxy):
            self.precomputed_embeddings = (
                self.precomputed_embeddings.reconstruct_on_target_device(target_device)
            )
        for extra_key in self.model_specific_data:
            if isinstance(
                self.model_specific_data[extra_key], CudaIpcTensorTransportProxy
            ):
                extra_data = self.model_specific_data[
                    extra_key
                ].reconstruct_on_target_device(target_device)
                self.model_specific_data[extra_key] = extra_data


@dataclasses.dataclass
class MultimodalProcessorOutput:
    """Raw output from multimodal processors before scheduler-side preparation (pad, hash).

    This is the typed replacement for the dict previously returned by
    ``BaseMultimodalProcessor.process_mm_data_async``.  Preprocessed inputs may
    already carry ``pad_value`` and ``hash`` to avoid hashing the same tensor once
    per scheduler TP rank.
    """

    mm_items: List[MultimodalDataItem]
    input_ids: Optional[List[int]] = None
    padded_input_ids: Optional[List[int]] = None

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

    # Moss-VL related
    vision_position_ids: Optional[torch.Tensor] = None
    media_nums_per_sample: Optional[List[int]] = None
    visible_frame_counts: Optional[torch.Tensor] = None

    # for transformers-compatibility
    token_type_ids: Optional[torch.Tensor] = None

    @staticmethod
    def from_dict(d: dict) -> MultimodalProcessorOutput:
        return MultimodalProcessorOutput(
            mm_items=d["mm_items"],
            input_ids=d.get("input_ids"),
            padded_input_ids=d.get("padded_input_ids"),
            im_token_id=d.get("im_token_id"),
            im_start_id=d.get("im_start_id"),
            im_end_id=d.get("im_end_id"),
            slice_start_id=d.get("slice_start_id"),
            slice_end_id=d.get("slice_end_id"),
            video_token_id=d.get("video_token_id"),
            audio_token_id=d.get("audio_token_id"),
            audio_start_id=d.get("audio_start_id"),
            audio_end_id=d.get("audio_end_id"),
            mrope_positions=d.get("mrope_positions"),
            mrope_position_delta=d.get("mrope_position_delta"),
            vision_position_ids=d.get("vision_position_ids"),
            media_nums_per_sample=d.get("media_nums_per_sample"),
            visible_frame_counts=d.get("visible_frame_counts"),
        )

    @staticmethod
    def build_padded_input_ids(input_ids, mm_items: List[MultimodalDataItem]):
        """pad the input_ids with mm_items if it's not already padded"""
        if input_ids is None or not mm_items:
            return None

        for item in mm_items:
            if item.pad_value is None or item.offsets is None:
                return None

        if isinstance(input_ids, torch.Tensor):
            padded_input_ids = input_ids.flatten().tolist()
        else:
            padded_input_ids = list(input_ids)

        for item in mm_items:
            for start, end in item.offsets:
                padded_input_ids[start : end + 1] = [item.pad_value] * (end - start + 1)
        return padded_input_ids


@dataclasses.dataclass
class MultimodalInputs:
    """The multimodal data related inputs."""

    # items of data
    mm_items: List[MultimodalDataItem]
    padded_input_ids: Optional[List[int]] = None
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
    mrope_position_delta_repeated_cache: Optional[torch.Tensor] = None

    # Moss-VL related
    vision_position_ids: Optional[torch.Tensor] = None
    media_nums_per_sample: Optional[List[int]] = None
    visible_frame_counts: Optional[torch.Tensor] = None

    def release_features(self):
        """Release feature tensors to free GPU memory."""
        for item in self.mm_items:
            item.feature = None

    @staticmethod
    def from_processor_output(obj: MultimodalProcessorOutput):
        mm_items = obj.mm_items
        assert isinstance(mm_items, list)
        mm_items = [item for item in mm_items if item.is_valid()]

        # try reconstructing from cuda-ipc
        reconstruct_device = None
        for mm_item in mm_items:
            if mm_item.has_cuda_ipc_proxy():
                if reconstruct_device is None:
                    reconstruct_device = torch.cuda.current_device()
                mm_item.reconstruct(reconstruct_device)

        if envs.SGLANG_MM_BUFFER_SIZE_MB.get() > 0:
            # Multi-modal feature hashing optimization:
            # When SGLANG_MM_BUFFER_SIZE_MB > 0, we temporarily move feature tensors to GPU
            # for faster hash computation, while avoiding OOM issues.
            from sglang.srt.managers.mm_utils import (
                init_feature_buffer,
                is_feature_buffer_initialized,
                reset_buffer_offset,
                try_add_to_buffer,
            )

            device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
            if not is_feature_buffer_initialized():
                init_feature_buffer(device)
            reset_buffer_offset()
            for item in mm_items:
                if item.feature is not None:
                    if isinstance(item.feature, torch.Tensor):
                        item.feature = try_add_to_buffer(item.feature)

        for item in mm_items:
            item.set_pad_value()

        if envs.SGLANG_MM_BUFFER_SIZE_MB.get() > 0:
            for item in mm_items:
                if item.feature is not None:
                    item.feature = item.feature.to("cpu", non_blocking=True)

        mm_inputs = MultimodalInputs(
            mm_items=mm_items,
            padded_input_ids=obj.padded_input_ids,
        )
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
            "vision_position_ids",
            "media_nums_per_sample",
            "visible_frame_counts",
        ]
        for arg in optional_args:
            val = getattr(obj, arg, None)
            if val is not None:
                setattr(mm_inputs, arg, val)

        return mm_inputs

    def contains_image_inputs(self) -> bool:
        return any(item.is_image() for item in self.mm_items)

    def contains_video_inputs(self) -> bool:
        return any(item.is_video() for item in self.mm_items)

    def contains_audio_inputs(self) -> bool:
        return any(item.is_audio() for item in self.mm_items)

    def contains_mm_input(self) -> bool:
        return any(True for item in self.mm_items if item.is_valid())

    def compute_mm_token_counts(self) -> Tuple[int, int, int]:
        """Count prompt tokens consumed by each modality (image, audio, video).

        A modality's token count is the total span covered by its items'
        offsets. Returns a (image_tokens, audio_tokens, video_tokens) tuple.
        """
        image_tokens = audio_tokens = video_tokens = 0
        for item in self.mm_items:
            if not item.offsets:
                continue
            num_tokens = sum(end - start + 1 for start, end in item.offsets)
            if item.is_image():
                image_tokens += num_tokens
            elif item.is_audio():
                audio_tokens += num_tokens
            elif item.is_video():
                video_tokens += num_tokens
        return image_tokens, audio_tokens, video_tokens

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


@dataclasses.dataclass(slots=True, kw_only=True)
class ReqLogprob:
    top_logprobs_num: int
    token_ids_logprob: Optional[List[int]]
    input_token_logprobs_val: Optional[List[float]] = None
    input_token_logprobs_idx: Optional[List[int]] = None
    input_top_logprobs_val: Optional[List[List[float]]] = None
    input_top_logprobs_idx: Optional[List[List[int]]] = None
    input_token_ids_logprobs_val: Optional[List[List[float]]] = None
    input_token_ids_logprobs_idx: Optional[List[List[int]]] = None
    output_token_logprobs_val: Optional[list] = None
    output_token_logprobs_idx: Optional[list] = None
    output_top_logprobs_val: Optional[list] = None
    output_top_logprobs_idx: Optional[list] = None
    # Can contain either lists or GPU tensors (delayed copy optimization for prefill-only scoring)
    output_token_ids_logprobs_val: Optional[List[Union[List[float], torch.Tensor]]] = (
        None
    )
    output_token_ids_logprobs_idx: Optional[list] = None


class Req(ReqDllmMixin):
    """The input and output status of a request."""

    def __init__(
        self,
        rid: str,
        origin_input_text: str,
        origin_input_ids: array[int],
        sampling_params: SamplingParams,
        return_logprob: bool = False,
        top_logprobs_num: int = 0,
        dllm_config: Optional[DllmConfig] = None,
        token_ids_logprob: List[int] = None,
        stream: bool = False,
        origin_input_ids_unpadded: Optional[array[int]] = None,
        lora_id: Optional[str] = None,
        input_embeds: Optional[List[List[float]]] = None,
        positional_embed_overrides: Optional[PositionalEmbeds] = None,
        token_type_ids: List[int] = None,
        session: Optional[Session] = None,
        custom_logit_processor: Optional[str] = None,
        require_reasoning: bool = False,
        return_hidden_states: bool = False,
        return_routed_experts: bool = False,
        routed_experts_start_len: int = 0,
        return_indexer_topk: bool = False,
        eos_token_ids: Optional[Set[int]] = None,
        bootstrap_host: Optional[str] = None,
        bootstrap_port: Optional[int] = None,
        bootstrap_room: Optional[int] = None,
        disagg_mode: Optional[DisaggregationMode] = None,
        routed_dp_rank: Optional[int] = None,
        disagg_prefill_dp_rank: Optional[int] = None,
        vocab_size: Optional[int] = None,
        priority: Optional[int] = None,
        metrics_collector: Optional[SchedulerMetricsCollector] = None,
        extra_key: Optional[str] = None,
        routing_key: Optional[str] = None,
        dimensions: Optional[int] = None,
        http_worker_ipc: Optional[str] = None,
        time_stats: Optional[
            Union[APIServerReqTimeStats, DPControllerReqTimeStats]
        ] = None,
        return_pooled_hidden_states: bool = False,
        multi_item_delimiter_indices: Optional[List[int]] = None,
        session_id: Optional[str] = None,
    ):
        # Input and output info
        self.rid = rid
        self.origin_input_ids = origin_input_ids
        self.origin_input_ids_unpadded = (
            origin_input_ids_unpadded
            if origin_input_ids_unpadded
            else self.origin_input_ids
        )  # Before image padding
        # Each decode stage's output ids. Append-only by contract:
        # _refresh_fill_ids infers how many output tokens are already in
        # full_untruncated_fill_ids from lengths alone, so in-place rewrites
        # that preserve length would silently corrupt fill_ids.
        self.output_ids = array("q")
        # Full untruncated sequence: origin + output (+ DLLM mask block).
        # Kept in sync by _refresh_fill_ids; admission only updates
        # extend_range, never mutates this array's length.
        self.full_untruncated_fill_ids = array("q")
        self.extend_range: Optional[Range] = None
        self.dllm_initialized: bool = False

        self.session = session
        self.session_id = session_id
        self.input_embeds = input_embeds
        self.positional_embed_overrides = positional_embed_overrides
        self.multi_item_delimiter_indices = multi_item_delimiter_indices

        # For req-level memory management
        self.kv_committed_len = 0
        self.kv_allocated_len = 0
        self.kv_committed_freed = False
        self.kv_overallocated_freed = False

        # for cross-encoder model
        self.token_type_ids = token_type_ids

        # The length of KV that have been removed in swa cache.
        # SWA KV cache eviction behavior differs by cache type:
        # - Radix cache: KV in range [cache_protected_len, swa_evicted_seqlen) is freed manually in
        #   `ScheduleBatch.maybe_evict_swa`; KV in range [0, cache_protected_len) is freed during radix cache eviction.
        # - Chunk cache: KV in range [0, swa_evicted_seqlen) is freed manually in `ScheduleBatch.maybe_evict_swa`.
        self.swa_evicted_seqlen = 0
        # Tokens in [0, swa_evict_floor) are protected from SWA window eviction.
        # This is used by prefill-aware SWA models such as Unlimited-OCR to keep prompt/image KV visible during decode.
        self.swa_evict_floor: int = 0

        # The index of the extend / decode batch
        self.extend_batch_idx = 0
        self.decode_batch_idx = 0

        # For multi-http worker
        self.http_worker_ipc = http_worker_ipc

        # Require reasoning for the request
        self.require_reasoning = require_reasoning

        # State indicating whether the reasoning phase has finished (only meaningful when require_reasoning is True)
        self._is_reasoning_over = False
        self.reasoning_tokens = 0

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
        self.routing_key = routing_key

        # Memory pool info
        self.req_pool_idx: Optional[int] = None
        self.mamba_pool_idx: Optional[torch.Tensor] = None  # shape (1)
        self.mamba_ping_pong_track_buffer: Optional[torch.Tensor] = None  # shape (2)
        self.mamba_next_track_idx: Optional[int] = None  # 0 or 1
        self.mamba_last_track_seqlen: Optional[int] = (
            None  # seq len of the last cached mamba state
        )
        # the branching point seqlen to track mamba state. If set, given by prefix match,
        # it will be the tracked seqlen in the ping pong buffer for the right prefill pass.
        self.mamba_branching_seqlen: Optional[int] = None
        # Deferred COW: source mamba pool index from radix cache node (copy on forward stream)
        self.mamba_cow_src_index: Optional[torch.Tensor] = None
        # Deferred clear: newly allocated mamba slot needs zeroing on forward stream
        self.mamba_needs_clear: bool = False
        # Lazy extra buffer: skip radix cache insert when prealloc failed at
        # boundary — the forward overwrites the only slot, corrupting the state.
        self.mamba_lazy_is_insert: bool = True

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
        # Pre-computed multimodal prompt token counts; populated on the prefill
        # node and transferred to decode via the metadata buffer in disagg (PD) mode.
        self.mm_image_tokens: int = 0
        self.mm_audio_tokens: int = 0
        self.mm_video_tokens: int = 0

        # Prefix info
        # The indices to kv cache for the shared prefix.
        self.prefix_indices: torch.Tensor = torch.empty((0,), dtype=torch.int64)
        # TODO(ispobock): rename to last_device_node
        self.last_node: Any = None
        self.last_host_node: Any = None
        self.best_match_node: Any = None
        # Per-component host hit lengths split off from host_hit_length:
        self.host_hit_length = 0
        self.swa_host_hit_length = 0
        self.mamba_host_hit_length = 0
        # Total cached prefix length (on-device prefix_indices + host_hit_length),
        # capped at the max allowed prefix. Set during prefix matching at schedule
        # time and used to estimate uncached tokens / sort by longest prefix for
        # load reporting.
        self.num_matched_prefix_tokens = 0
        # Tokens loaded from storage backend (L3) during prefetch for this request
        self.storage_hit_length = 0
        # The node to lock until for swa radix tree lock ref
        self.swa_uuid_for_lock: Optional[int] = None
        # Whether the prefill-time SWA tree lock has been released early
        self.swa_prefix_lock_released: bool = False
        # The prefix length that is inserted into the tree cache
        self.cache_protected_len: int = 0

        # Whether or not if it is chunked. It increments whenever
        # it is chunked, and decrement whenever chunked request is
        # processed.
        self.inflight_middle_chunks = 0

        # For retraction
        self.is_retracted = False
        # Indicates if the req has ever been retracted.
        self.retracted_stain = False

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
        self.logprob = ReqLogprob(
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
        )

        # Logprobs (return values)
        # True means the input logprob has been already sent to detokenizer.
        self.input_logprob_sent: bool = False
        # Temporary holder to store input_token_logprobs.
        self.input_token_logprobs: Optional[List[Tuple[int]]] = None
        self.temp_input_top_logprobs_val: Optional[List[torch.Tensor]] = None
        self.temp_input_top_logprobs_idx: Optional[List[int]] = None
        self.temp_input_token_ids_logprobs_val: Optional[List[float]] = None
        self.temp_input_token_ids_logprobs_idx: Optional[List[int]] = None

        if return_logprob:
            # shape: (bs, 1)
            self.logprob.output_token_logprobs_val = []
            self.logprob.output_token_logprobs_idx = []
            # shape: (bs, k)
            self.logprob.output_top_logprobs_val = []
            self.logprob.output_top_logprobs_idx = []
            # Can contain either lists or GPU tensors (delayed copy optimization for prefill-only scoring)
            self.logprob.output_token_ids_logprobs_val = []
            self.logprob.output_token_ids_logprobs_idx = []
        self.hidden_states: List[List[float]] = []
        self.hidden_states_tensor = None  # Note: use tensor instead of list to transfer hidden_states when PD + MTP
        self.output_topk_p = None
        self.output_topk_index = None

        # capture routed experts
        self.return_routed_experts = return_routed_experts
        self.routed_experts_start_len = routed_experts_start_len
        self.routed_experts: Optional[torch.Tensor] = (
            None  # cpu tensor: shape (seqlen, topk)
        )

        self.return_indexer_topk = return_indexer_topk
        self.indexer_topk: Optional[torch.Tensor] = (
            None  # cpu tensor: shape (seqlen, num_indexer_layers, index_topk)
        )
        # Customized info
        self.customized_info: Optional[Dict[str, List[Any]]] = None

        # Embedding (return values)
        self.embedding = None

        # Constrained decoding
        self.grammar_key: Optional[Tuple[str, str]] = None
        self.grammar: Optional[Union[BaseGrammarObject, Future[BaseGrammarObject]]] = (
            None
        )
        self.grammar_wait_ct = 0

        # The number of cached tokens that were already cached in the KV cache
        self.cached_tokens = 0
        self.already_computed = 0

        # Detailed breakdown of cached tokens by source (for HiCache)
        self.cached_tokens_device = 0  # Tokens from device cache (GPU)
        self.cached_tokens_host = 0  # Tokens from host cache (CPU memory)
        self.cached_tokens_storage = 0  # Tokens from L3 storage backend
        self._cache_breakdown_computed = (
            False  # Track if breakdown was already computed
        )

        # Per-request count of verification forward passes.
        self.spec_verify_ct = 0

        # Per-request count of accepted draft tokens (excludes the bonus token).
        self.spec_num_correct_drafts = 0

        # Acceptance histogram for speculative decoding.
        # List index = number of accepted tokens in a step, List value = count of steps with that many accepted tokens.
        # Example: histogram[0] = 5 means 5 steps with 0 accepted tokens, histogram[3] = 10 means 10 steps with 3 accepted tokens.
        self.spec_correct_drafts_histogram: List[int] = []

        # The number of times this request has been retracted / preempted.
        self.retraction_count = 0
        self.retraction_mb_id = None

        # For observability
        self.metrics_collector = metrics_collector
        if time_stats is not None:
            self.time_stats = SchedulerReqTimeStats.new_from_obj(time_stats)
        else:
            self.time_stats = SchedulerReqTimeStats(disagg_mode=disagg_mode)
        self.time_stats.set_metrics_collector(metrics_collector)
        self.time_stats.set_scheduler_recv_time()
        self.has_log_time_stats: bool = False

        # For disaggregation
        self.bootstrap_host: str = bootstrap_host
        self.bootstrap_port: Optional[int] = bootstrap_port
        self.bootstrap_room: Optional[int] = bootstrap_room
        # Decode-local: the already-emitted boundary token to replay when a
        # retracted request is rebootstrapped. Set in pause_generation(retract)
        # and consumed in the decode transfer commit; never plumbed to prefill.
        self.pd_rebootstrap_forced_output_id: Optional[int] = None
        self.skip_radix_cache_insert = bootstrap_host == FAKE_BOOTSTRAP_HOST
        self.disagg_kv_sender: Optional[BaseKVSender] = None

        self.routed_dp_rank: Optional[int] = routed_dp_rank
        self.disagg_prefill_dp_rank: Optional[int] = disagg_prefill_dp_rank

        # the start index of the sent kv cache
        # We want to send it chunk by chunk for chunked prefill.
        # After every chunk forward, we do the following:
        # kv_send(req.input_ids[req.start_send_idx:req.extend_range.end])
        # start_send_idx = req.extend_range.end
        self.start_send_idx: int = 0

        # For overlap schedule, we delay the kv transfer until `process_batch_result_disagg_prefill` rather than `process_prefill_chunk` in non-overlap
        # This is because kv is not ready in `process_prefill_chunk`.
        # We use `tmp_end_idx` to store the end index of the kv cache to send.
        self.tmp_end_idx: int = -1
        self.metadata_buffer_index: int = -1
        # Used in overlap sequence to signal that an optimistic request should
        # abort chunking. Set in create_sender, consumed in process_batch_result.
        self.pending_bootstrap = False

        # For Matryoshka embeddings
        self.dimensions = dimensions

        # Whether to return pooled hidden states (pre-head transformer output)
        self.return_pooled_hidden_states = return_pooled_hidden_states
        self.pooled_hidden_state = None

        # For diffusion LLM
        self.init_diffusion_llm(dllm_config)

        # For hisparse
        self.hisparse_staging = False

    @property
    def seqlen(self) -> int:
        """Get the current sequence length of the request."""
        return len(self.origin_input_ids) + len(self.output_ids)

    @property
    def is_prefill_only(self) -> bool:
        """Check if this request is prefill-only (no token generation needed)."""
        # NOTE: when spec is enabled, prefill_only optimizations are disabled

        spec_alg = get_server_args().speculative_algorithm
        return self.sampling_params.max_new_tokens == 0 and spec_alg is None

    @property
    def output_ids_through_stop(self) -> array[int]:
        """Get the output ids through the stop condition. Stop position is included."""
        if self.finished_len is not None:
            return self.output_ids[: self.finished_len]
        return self.output_ids

    def needs_host_load_back(self) -> bool:
        """Whether any cache layer has a host hit that needs L2 H2D load_back."""
        return (
            self.host_hit_length > 0
            or self.swa_host_hit_length > 0
            or self.mamba_host_hit_length > 0
        )

    def _cache_commit_len(self) -> int:
        # Report only the prompt prefix so thinking + answer fall into the
        # overallocated range and are reclaimed by release_kv_cache. #22373.
        if get_server_args().strip_thinking_cache and self.reasoning_tokens > 0:
            return min(self.kv_committed_len, len(self.origin_input_ids))
        return self.kv_committed_len

    def pop_committed_kv_cache(self) -> int:
        """Return the length of committed KV cache and mark them as freed."""
        assert (
            not self.kv_committed_freed
        ), f"Committed KV cache already freed ({self.kv_committed_len=})"
        self.kv_committed_freed = True
        return self._cache_commit_len()

    def pop_overallocated_kv_cache(self) -> Tuple[int, int]:
        """Return the range of over-allocated KV cache and mark them as freed."""

        # NOTE: This function is called when there is over-allocation of KV cache.
        # Over-allocation: we allocate more KV cache than the committed length.
        # e.g., speculative decoding may allocate more KV cache than actually used.
        assert (
            not self.kv_overallocated_freed
        ), f"Overallocated KV cache already freed, {self.kv_committed_len=}, {self.kv_allocated_len=}"
        self.kv_overallocated_freed = True
        return self._cache_commit_len(), self.kv_allocated_len

    def update_spec_correct_drafts_histogram(self, num_correct_drafts: int):
        """Update the speculative decoding acceptance histogram.

        Args:
            num_correct_drafts: Number of correct draft tokens (no bonus) in this step.
        """
        if len(self.spec_correct_drafts_histogram) <= num_correct_drafts:
            self.spec_correct_drafts_histogram.extend(
                [0] * (num_correct_drafts - len(self.spec_correct_drafts_histogram) + 1)
            )
        self.spec_correct_drafts_histogram[num_correct_drafts] += 1

    def extend_image_inputs(self, image_inputs):
        if self.multimodal_inputs is None:
            self.multimodal_inputs = image_inputs
        else:
            self.multimodal_inputs.merge(image_inputs)

    def finished(self) -> bool:
        # Whether request reached finished condition
        return self.finished_reason is not None

    def set_extend_range(self, start: int, end: int) -> None:
        self.extend_range = Range(start, end)

    def get_fill_ids(self) -> array:
        return self.full_untruncated_fill_ids[: self.extend_range.end]

    def _refresh_fill_ids(self) -> None:
        """Keep full_untruncated_fill_ids == origin_input_ids + output_ids by
        appending only the new output tokens.

        Falls back to a full rebuild when the in-place append is invalid:
        - aliasing: scheduler_pp_mixin assigns full_untruncated_fill_ids =
          origin_input_ids directly, so extending in place would write output
          tokens into the origin;
        - lengths disagree: fresh req (array still empty), retraction
          (output_ids reset to empty), or set_finish_with_abort (origin
          replaced by a 1-token stub).
        """
        n_have_output = len(self.full_untruncated_fill_ids) - len(self.origin_input_ids)
        if (
            self.full_untruncated_fill_ids is not self.origin_input_ids
            and 0 <= n_have_output <= len(self.output_ids)
        ):
            self.full_untruncated_fill_ids.extend(self.output_ids[n_have_output:])
        else:
            self.full_untruncated_fill_ids = self.origin_input_ids + self.output_ids

    def init_next_round_input(
        self,
        tree_cache: Optional[BasePrefixCache] = None,
        cow_mamba: Optional[bool] = None,
    ):
        if self.is_dllm():
            self._init_fill_ids_for_dllm()
            self.determine_dllm_phase()
        else:
            self._refresh_fill_ids()

        input_len = len(self.full_untruncated_fill_ids)

        # Streaming sessions reuse committed KV from the session slot, so
        # custom logprob_start_len is not supported — override to -1.
        if (
            self.session is not None
            and self.session.streaming
            and self.return_logprob
            and self.logprob_start_len >= 0
        ):
            logger.warning(
                "logprob_start_len=%d is not supported for streaming sessions "
                "and will be ignored (rid=%s). Only new-token logprobs are returned.",
                self.logprob_start_len,
                self.rid,
            )
            self.logprob_start_len = -1

        # Pass the full array with a raw-token cap (limit) instead of slicing,
        # avoiding an O(context) copy per prefill-batch build.
        token_ids_to_match = self.full_untruncated_fill_ids
        key_limit: Optional[int] = self._compute_max_prefix_len(input_len)

        # SWA lives in a per-request ring that's not content-stable and is never
        # stored in the radix tree, so a reused prefix carries stale SWA. Cap the
        # match by the trailing sliding window so it gets re-prefilled, rewriting
        # this request's SWA ring. No-op for other layouts.
        if tree_cache is not None:
            reprefill_tail = tree_cache.swa_reprefill_tail_tokens()
            if reprefill_tail:
                capped = max(0, input_len - reprefill_tail)
                key_limit = capped if key_limit is None else min(key_limit, capped)

        # Disable prefix caching when embed overrides are present: same token IDs
        # with different override vectors must not share cached KV values.
        if self.positional_embed_overrides is not None:
            token_ids_to_match = array("q")
            key_limit = None

        if tree_cache is not None:
            if cow_mamba is None:
                cow_mamba = tree_cache.supports_mamba()
            # unified_kv SWA lives in a per-request ring that is not content-stable
            # and never cached in the radix tree, so a reused prefix carries stale
            # SWA. Cap the match by the trailing sliding window so it is re-prefilled
            # into this request's ring. No-op for other layouts (returns 0).
            reprefill_tail = tree_cache.swa_reprefill_tail_tokens()
            if reprefill_tail:
                capped = max(0, input_len - reprefill_tail)
                key_limit = capped if key_limit is None else min(key_limit, capped)
            match_result = tree_cache.match_prefix(
                MatchPrefixParams(
                    key=RadixKey(
                        token_ids=token_ids_to_match,
                        extra_key=self.extra_key,
                        limit=key_limit,
                    ),
                    req=self,
                    cow_mamba=cow_mamba,
                )
            )
            if envs.SGLANG_RADIX_FORCE_MISS.get():
                match_result = zero_match_result(tree_cache, match_result)
            (
                self.prefix_indices,
                self.last_node,
                self.last_host_node,
                self.best_match_node,
                self.host_hit_length,
                self.swa_host_hit_length,
                self.mamba_host_hit_length,
                self.mamba_branching_seqlen,
            ) = (
                match_result.device_indices,
                match_result.last_device_node,
                match_result.last_host_node,
                match_result.best_match_node,
                match_result.host_hit_length,
                match_result.swa_host_hit_length,
                match_result.mamba_host_hit_length,
                match_result.mamba_branching_seqlen,
            )
            if match_result.cache_protected_len is not None:
                self.cache_protected_len = match_result.cache_protected_len
            else:
                self.cache_protected_len = len(self.prefix_indices)

            if self.is_dllm():
                self._update_block_offset_for_dllm()

        if (
            self.is_retracted
            and self.multimodal_inputs is not None
            and self.multimodal_inputs.mrope_positions is not None
        ):
            from sglang.srt.managers.mm_utils import (
                extend_mrope_positions_for_retracted_request,
            )

            self.multimodal_inputs.mrope_positions = (
                extend_mrope_positions_for_retracted_request(
                    self.multimodal_inputs.mrope_positions, len(self.output_ids)
                )
            )

    def _compute_max_prefix_len(self, input_len: int) -> int:
        # NOTE: the matched length is at most 1 less than the input length to enable logprob computation
        max_prefix_len = input_len - 1
        if self.return_logprob and self.logprob_start_len >= 0:
            max_prefix_len = min(max_prefix_len, self.logprob_start_len)
        return max(max_prefix_len, 0)

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

    def _stop_match_tail_len(self, new_accepted_len: int) -> int:
        max_len_tail_str = max(
            self.sampling_params.stop_str_max_len + 1,
            self.sampling_params.stop_regex_max_len + 1,
        )
        # Cover all newly accepted tokens so an early stop string is not missed
        # when speculative decoding accepts multiple tokens per step.
        return min(
            max_len_tail_str + max(new_accepted_len - 1, 0), len(self.output_ids)
        )

    def tail_str(self, new_accepted_len: int = 1) -> str:
        # Check stop strings and stop regex patterns together
        if (
            len(self.sampling_params.stop_strs) == 0
            and len(self.sampling_params.stop_regex_strs) == 0
        ):
            return ""

        tail_len = self._stop_match_tail_len(new_accepted_len)
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

    def _locate_str_stop_finished_len(
        self,
        new_accepted_len: int,
        *,
        stop_str: Optional[str] = None,
        stop_regex: Optional[str] = None,
    ) -> int:
        """Map a matched stop string/regex to output_ids length (stop included)."""

        def matched(text: str) -> bool:
            if stop_str is not None:
                return stop_str in text
            return re.search(stop_regex, text) is not None

        tail_len = self._stop_match_tail_len(new_accepted_len)
        start = len(self.output_ids) - tail_len
        token_window = self.output_ids[start:]

        # Old prefixes were checked in the previous step.
        for token_count in range(
            max(1, len(token_window) - new_accepted_len + 1), len(token_window)
        ):
            if matched(self.tokenizer.decode(token_window[:token_count])):
                return start + token_count

        # The full tail window is already known to match by the caller.
        return len(self.output_ids)

    def _check_str_based_finish(self, new_accepted_len: int = 1):
        if (
            len(self.sampling_params.stop_strs) > 0
            or len(self.sampling_params.stop_regex_strs) > 0
        ):
            tail_str = self.tail_str(new_accepted_len)

            # Check stop strings
            if len(self.sampling_params.stop_strs) > 0:
                for stop_str in self.sampling_params.stop_strs:
                    stop_str_in_tail = stop_str in tail_str
                    if stop_str_in_tail or stop_str in self.decoded_text:
                        self.finished_reason = FINISH_MATCHED_STR(matched=stop_str)
                        if stop_str_in_tail:
                            self.finished_len = self._locate_str_stop_finished_len(
                                new_accepted_len, stop_str=stop_str
                            )
                        return True

            # Check stop regex
            if len(self.sampling_params.stop_regex_strs) > 0:
                for stop_regex_str in self.sampling_params.stop_regex_strs:
                    if re.search(stop_regex_str, tail_str):
                        self.finished_reason = FINISHED_MATCHED_REGEX(
                            matched=stop_regex_str
                        )
                        self.finished_len = self._locate_str_stop_finished_len(
                            new_accepted_len, stop_regex=stop_regex_str
                        )
                        return True

        return False

    def _check_vocab_boundary_finish(self, new_accepted_tokens: List[int] = None):
        for i, token_id in enumerate(new_accepted_tokens):
            if token_id >= self.vocab_size or token_id < 0:
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

    def update_finish_state(self, new_accepted_len: int = 1):
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

        # Sanitize out-of-range / NaN token ids before any decode.
        if self._check_vocab_boundary_finish(new_accepted_tokens):
            return

        # Stop string beats EOS/stop-token matched in the same step (speculative
        # decoding can accept >1 token): token-based would trim only the last
        # token and leak the stop string.
        if self._check_str_based_finish(new_accepted_len):
            return

        if self._check_token_based_finish(new_accepted_tokens):
            return

    def reset_for_retract(self):
        # Increment retraction count before resetting other state. We should not reset this
        # since we are tracking the total number of retractions for each request.
        self.retraction_count += 1

        self.prefix_indices = torch.empty((0,), dtype=torch.int64)
        self.routed_experts = None
        self.indexer_topk = None
        self.last_node = None
        self.cache_protected_len = 0
        self.num_matched_prefix_tokens = 0
        self.swa_uuid_for_lock = None
        self.swa_prefix_lock_released = False
        self.extend_range = None
        self.dllm_initialized = False
        self.is_retracted = True
        self.retracted_stain = True
        self.input_token_logprobs = None
        self.temp_input_top_logprobs_val = None
        self.temp_input_top_logprobs_idx = None
        self.inflight_middle_chunks = 0
        self.mamba_pool_idx = None
        self.mamba_ping_pong_track_buffer = None
        self.mamba_next_track_idx = None
        self.mamba_last_track_seqlen = None
        self.mamba_branching_seqlen = None
        self.mamba_cow_src_index = None
        self.mamba_needs_clear = False
        self.already_computed = 0
        self.kv_allocated_len = 0
        self.kv_committed_len = 0
        self.kv_committed_freed = False
        self.kv_overallocated_freed = False
        self.swa_evicted_seqlen = 0
        self.extend_batch_idx = 0
        self.decode_batch_idx = 0

        # When using input_embeds, we cannot easily mix the original input embeddings
        # with the newly generated output token IDs during re-prefill of retracted request.
        # output_ids will have no use, but will lead to wrong size cache indexes.
        # Therefore, we discard the generated output_ids and restart prefill and generation
        # to ensure shape consistency in KV cache.
        if self.input_embeds is not None:
            self.output_ids = array("q")

    def offload_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):
        token_indices = req_to_token_pool.req_to_token[
            self.req_pool_idx, : self.seqlen - 1
        ]
        # Copies over both the kv cache and mamba state if available
        self.kv_cache_cpu = token_to_kv_pool_allocator.get_cpu_copy(
            token_indices, mamba_indices=self.mamba_pool_idx
        )

    def load_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):
        token_indices = req_to_token_pool.req_to_token[
            self.req_pool_idx, : self.seqlen - 1
        ]
        # Loads both the kv cache and mamba state if exists
        token_to_kv_pool_allocator.load_cpu_copy(
            self.kv_cache_cpu, token_indices, mamba_indices=self.mamba_pool_idx
        )
        del self.kv_cache_cpu

    def build_rebootstrap_payload(self) -> dict:
        """Build the prefill ``/generate`` payload that asks the original prefill
        worker to recompute this request's prefix KV under the current weights
        (PD true-retraction rebootstrap).

        ``input_ids`` are coerced to plain ``int`` so the payload is always
        JSON-serializable even when ``origin_input_ids``/``output_ids`` hold
        numpy scalars. The sampling-param allow-list forces ``max_new_tokens=1``
        and drops stop/grammar/min_new_tokens so the recompute only re-derives
        the prefix KV and samples a single handoff token. The already-emitted
        boundary token is replayed on the *decode* side (the transfer commit
        overrides the sampled handoff with it), so it is intentionally not sent
        to the prefill here.
        """
        # TODO: multi-modal requests are not supported here. The payload only
        # carries token ``input_ids`` and drops any image/audio/video inputs, so
        # the rebootstrap recompute would not reproduce the original prefix KV
        # for multi-modal requests. Add multi-modal support before enabling it.
        sp = self.sampling_params
        return {
            "input_ids": [int(x) for x in self.origin_input_ids]
            + [int(x) for x in self.output_ids],
            "sampling_params": {
                "max_new_tokens": 1,
                "temperature": sp.temperature,
                "top_p": sp.top_p,
                "top_k": sp.top_k,
                "min_p": sp.min_p,
                "frequency_penalty": sp.frequency_penalty,
                "presence_penalty": sp.presence_penalty,
                "repetition_penalty": sp.repetition_penalty,
                "ignore_eos": sp.ignore_eos,
                "skip_special_tokens": sp.skip_special_tokens,
                "spaces_between_special_tokens": sp.spaces_between_special_tokens,
                "no_stop_trim": sp.no_stop_trim,
            },
            "return_logprob": False,
            "stream": False,
            "rid": self.rid,
            "bootstrap_host": self.bootstrap_host,
            "bootstrap_port": self.bootstrap_port,
            "bootstrap_room": self.bootstrap_room,
            "priority": self.priority,
            "extra_key": self.extra_key,
            "routing_key": self.routing_key,
            "disagg_prefill_dp_rank": self.disagg_prefill_dp_rank,
        }

    def log_time_stats(self):
        # If overlap schedule, we schedule one decode batch ahead so this gets called twice.
        if self.has_log_time_stats:
            return

        bootstrap_info = (
            f", bootstrap_room={self.bootstrap_room}"
            if self.bootstrap_room is not None
            else ""
        )
        prefix = (
            f"ReqTimeStats("
            f"rid={self.rid}{bootstrap_info}, "
            f"input_len={len(self.origin_input_ids)}, "
            f"cached_input_len={self.cached_tokens}, "
            f"output_len={len(self.output_ids)}, "
            f"type={self.time_stats.disagg_mode_str()})"
        )
        logger.info(f"{prefix}: {self.time_stats.convert_to_duration()}")
        self.has_log_time_stats = True

    def set_finish_with_abort(self, error_msg: str):
        if get_parallel().tp_rank == 0:
            logger.error(f"{error_msg}, {self.rid=}")
        self.multimodal_inputs = None
        self.grammar = None
        self.origin_input_ids = array(
            "q", [0]
        )  # set it to one token to skip the long prefill
        self.return_logprob = False
        self.logprob_start_len = -1
        self.to_finish = FINISH_ABORT(
            error_msg, HTTPStatus.BAD_REQUEST, "BadRequestError"
        )

    def update_reasoning_tokens(self, token_id, think_end_id):
        if self._is_reasoning_over:
            return

        if not isinstance(token_id, list):
            token_id = [token_id]

        try:
            end_pos = token_id.index(think_end_id)
            self.reasoning_tokens += end_pos + 1
            self._is_reasoning_over = True
        except ValueError:
            self.reasoning_tokens += len(token_id)

    def __repr__(self):
        return (
            f"Req(rid={self.rid}, "
            f"input_ids={self.origin_input_ids}, output_ids={self.output_ids}, "
            f"{self.grammar=}, "
            f"{self.sampling_params=})"
        )


class _MambaRadixCacheV2TrackEntry(NamedTuple):
    track_mask: bool
    track_index: int
    track_seqlen: int


def set_mamba_track_indices_from_reqs(batch):
    """Build mamba_track_indices from req objects (authoritative source)."""
    req_to_token_pool = batch.req_to_token_pool
    all_buffers = req_to_token_pool.req_index_to_mamba_ping_pong_track_buffer_mapping[
        batch.req_pool_indices
    ]  # (bs, ping_pong_size), int64, on device
    idx = (
        torch.tensor(
            [req.mamba_next_track_idx for req in batch.reqs],
            dtype=torch.int64,
            pin_memory=True,
        )
        .unsqueeze(1)
        .to(device=all_buffers.device, non_blocking=True)
    )
    batch.mamba_track_indices = (
        torch.gather(all_buffers, 1, idx).squeeze(1).to(torch.int64)
    )


def release_req(
    *,
    req: Req,
    remaing_req_count: int,
    server_args: ServerArgs,
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    tree_cache: BasePrefixCache,
    hisparse_coordinator: Optional[HiSparseCoordinator],
    offload_kv: bool = True,
) -> None:
    if hisparse_coordinator is not None and not req.finished():
        hisparse_coordinator.retract_req(req)

    # In decode disaggregation the retracted KV is offloaded to host so it can be
    # restored later without recompute (see resume_retracted_reqs/load_kv_cache).
    # Callers that will recompute the KV instead (PD true-retraction rebootstrap)
    # pass offload_kv=False to skip the wasteful device->host copy.
    if server_args.disaggregation_mode == "decode" and offload_kv:
        req.offload_kv_cache(req_to_token_pool, token_to_kv_pool_allocator)
    # TODO (csy): for preempted requests, we may want to insert into the tree
    release_kv_cache(req, tree_cache, is_insert=False)
    # NOTE(lsyin): we should use the newly evictable memory instantly.
    num_tokens = remaing_req_count * envs.SGLANG_RETRACT_DECODE_STEPS.get()
    evict_from_tree_cache(tree_cache, num_tokens)

    req.reset_for_retract()


def retract_all(
    *,
    reqs: List[Req],
    server_args: ServerArgs,
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    tree_cache: BasePrefixCache,
    hisparse_coordinator: Optional[HiSparseCoordinator],
    offload_kv: bool = True,
) -> List[Req]:
    retracted_reqs = reqs
    for idx in range(len(reqs)):
        release_req(
            req=reqs[idx],
            remaing_req_count=len(reqs) - idx,
            server_args=server_args,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            hisparse_coordinator=hisparse_coordinator,
            offload_kv=offload_kv,
        )
    return retracted_reqs


def compute_extend_logprob_start_len(
    *,
    logprob_start_len: int,
    prefix_len: int,
    extend_len: int,
    full_untruncated_fill_len: int,
) -> int:
    # Key variables:
    # - logprob_start_len: Absolute position in full sequence where logprob computation begins
    # - extend_logprob_start_len: Relative position within current extend batch where logprob computation begins
    # - extend_input_len: Number of tokens that need to be processed in this extend batch
    if logprob_start_len == -1:
        resolved_start = full_untruncated_fill_len
    else:
        # logprob_start_len should be at least the length of the prefix indices
        resolved_start = max(logprob_start_len, prefix_len)
    return min(resolved_start - prefix_len, extend_len)


def _compute_chunked_req_next_prompt_token(
    chunked_req: Optional[Req],
    vocab_size: int,
) -> Optional[int]:
    """Return the next real prompt token after the fill boundary, skipping
    multimodal placeholder (hash) tokens that lie outside the model vocab."""
    if chunked_req is None:
        return None
    fill_len = chunked_req.extend_range.end
    origin_ids = chunked_req.origin_input_ids
    if fill_len >= len(origin_ids):
        return None
    if origin_ids[fill_len] < vocab_size:
        return int(origin_ids[fill_len])
    return None


@dataclasses.dataclass
class ScheduleBatch(ScheduleBatchDisaggregationDecodeMixin):
    """Store all information of a batch on the scheduler."""

    # === Core: request list (ForwardBatch derives lora_ids / rids / grammars / positions from it) ===
    reqs: List[Req]

    # === Global config and shared resources (engine-lifetime; identical across batches) ===
    # Memory pool and cache
    req_to_token_pool: ReqToTokenPool = None
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator = None
    tree_cache: BasePrefixCache = None

    # Batch configs
    model_config: ModelConfig = None
    enable_overlap: bool = False

    # Device
    device: str = "cuda"

    # HiSparse (engine-level coordinator ref, same across batches)
    hisparse_coordinator: Optional[HiSparseCoordinator] = None

    # === Batch-variant scheduler state (per-batch; not read by ForwardBatch) ===
    # Tell whether the current running batch is full so that we can skip
    # the check of whether to prefill new requests.
    # This is an optimization to reduce the overhead of the prefill check.
    batch_is_full: bool = False

    # For chunked prefill in PP
    chunked_req: Optional[Req] = None
    chunked_req_next_prompt_token: Optional[int] = None
    contains_last_prefill_chunk: bool = True

    # For DP attention
    inner_idle_batch: Optional[ScheduleBatch] = None
    # Decode requests carried alongside a chunked-prefill batch
    decoding_reqs: List[Req] = None

    # For split prefill
    split_index: int = 0
    split_prefill_finished: bool = False
    split_forward_count: int = 1
    split_forward_batch: ForwardBatch = None

    # CPU mirror of req_pool_indices; schedule-path only (used in overlap_utils,
    # not read by ForwardBatch), stale in spec draft window
    req_pool_indices_cpu: torch.Tensor = None  # shape: [b], int64

    # Forward-pass metrics
    fpm_start_time: float = 0.0

    # hicache pointer for synchronizing data loading from CPU to GPU
    hicache_consumer_index: int = -1

    # Metrics
    dp_cooperation_info: Optional[DPCooperationInfo] = None
    prefill_stats: Optional[PrefillStats] = None
    forward_iter: Optional[int] = None

    # === GPU tensors crossing to ForwardBatch (clone targets for stream isolation) ===
    # Batched arguments to model runner
    input_ids: torch.Tensor = None  # shape: [b], int64
    # Staging consumed by resolve_forward_inputs (prefill H2D / mixed gather).
    prefill_input_ids_cpu: Optional[torch.Tensor] = None
    mix_running_indices: Optional[torch.Tensor] = None
    input_embeds: torch.Tensor = None  # shape: [b, hidden_size], float32

    # Token replacement embeddings and absolute positions (optional).
    replace_embeds: Optional[torch.Tensor] = None
    replace_positions: Optional[torch.Tensor] = None

    # Read by ForwardBatch ngram embedding init
    ne_token_table: torch.Tensor = None
    # Mask marking chunked (not-yet-finished) prefill requests whose sampled
    # pseudo next-token must NOT be written into the ngram token table.
    ne_skip_token_table_update: torch.Tensor = None

    req_pool_indices: torch.Tensor = None  # shape: [b], int64
    seq_lens: torch.Tensor = None  # shape: [b], int64

    # The original sequence lengths, Qwen-1M related
    orig_seq_lens: torch.Tensor = None  # shape: [b], int32

    # The output locations of the KV cache
    out_cache_loc: torch.Tensor = None  # shape: [b], int64
    # DSV4-NPU: per-pool slot bundle from DSV4NPUTokenToKVPoolAllocator (None
    # elsewhere); c4/c128 state lens ride on ``batch.dsv4_state_lens``.
    out_cache_loc_dsv4: Optional[Any] = None

    # For hybrid GDN prefix cache
    mamba_track_indices: torch.Tensor = None  # shape: [b], int64
    mamba_track_mask: torch.Tensor = None  # shape: [b], bool
    mamba_track_seqlens: torch.Tensor = None  # shape: [b], int64
    # Deferred mamba init ops: COW pairs and clear indices (performed on forward stream)
    mamba_cow_src_indices: torch.Tensor = None
    mamba_cow_dst_indices: torch.Tensor = None
    mamba_clear_indices: torch.Tensor = None

    # Encoder-decoder device tensors (host fields in the host metadata group)
    encoder_lens: Optional[torch.Tensor] = None
    encoder_out_cache_loc: Optional[torch.Tensor] = None

    # It comes empty list if logprob is not required.
    extend_input_logprob_token_ids: Optional[torch.Tensor] = None

    # === Config / flags crossing to ForwardBatch (by-value) ===
    forward_mode: ForwardMode = None
    global_forward_mode: Optional[ForwardMode] = None

    # For DP attention
    is_extend_in_batch: bool = False
    can_run_dp_cuda_graph: bool = False
    can_run_dp_breakable_cuda_graph: bool = False
    tbo_split_seq_index: Optional[int] = None

    # For processing logprobs
    return_logprob: bool = False

    # Whether this batch is prefill-only (no token generation needed)
    is_prefill_only: bool = False

    # Speculative decoding
    spec_algorithm: SpeculativeAlgorithm = None

    # Whether to return hidden states
    return_hidden_states: bool = False

    # Has grammar
    has_grammar: bool = False

    # The sum of all sequence lengths
    seq_lens_sum: int = None
    extend_num_tokens: Optional[int] = None

    # Diffusion LLM
    dllm_config: Optional[DllmConfig] = None

    # === Host metadata crossing to ForwardBatch (CPU lists / mirrors) ===
    seq_lens_cpu: torch.Tensor = None  # shape: [b], int64

    # For multimodal inputs
    multimodal_inputs: Optional[List] = None

    # For processing logprobs
    top_logprobs_nums: Optional[List[int]] = None
    token_ids_logprobs: Optional[List[List[int]]] = None

    # For encoder-decoder architectures
    encoder_cached: Optional[List[bool]] = None
    encoder_lens_cpu: Optional[List[int]] = None

    # For extend and mixed chunekd prefill
    prefix_lens: List[int] = None
    extend_lens: List[int] = None
    extend_logprob_start_lens: List[int] = None

    # For DP attention
    global_num_tokens: Optional[List[int]] = None
    global_num_tokens_for_logprob: Optional[List[int]] = None

    # === Compound crossing to ForwardBatch (carry their own device tensors) ===
    # Sampling info
    sampling_info: SamplingBatchInfo = None

    # Speculative decoding
    # spec_info: Optional[SpecInput] = None
    spec_info: Optional[SpecInput] = None

    # === One-shot per-forward overrides; init_new consumes and resets ===
    capture_hidden_mode: Optional[CaptureHiddenMode] = None
    return_hidden_states_before_norm: bool = False

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
        dllm_config: Optional[DllmConfig] = None,
    ):
        return_logprob = any(req.return_logprob for req in reqs)

        batch = cls(
            reqs=reqs,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=model_config,
            enable_overlap=enable_overlap,
            return_logprob=return_logprob,
            has_grammar=any(req.grammar for req in reqs),
            device=req_to_token_pool.device,
            spec_algorithm=spec_algorithm,
            return_hidden_states=any(req.return_hidden_states for req in reqs),
            is_prefill_only=all(req.is_prefill_only for req in reqs),
            chunked_req=chunked_req,
            chunked_req_next_prompt_token=_compute_chunked_req_next_prompt_token(
                chunked_req,
                model_config.vocab_size,
            ),
            dllm_config=dllm_config,
        )
        return batch

    def batch_size(self):
        return len(self.reqs)

    def is_empty(self):
        return len(self.reqs) == 0

    def is_dllm(self):
        return self.dllm_config is not None

    def prepare_encoder_info_extend(
        self, input_ids: List[array[int]], seq_lens: List[int]
    ):
        _pin = is_pin_memory_available(self.device)
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

        self.encoder_lens = torch.tensor(
            self.encoder_lens_cpu, dtype=torch.int64, pin_memory=_pin
        ).to(self.device, non_blocking=True)

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
                    self.out_cache_loc[pt + encoder_len : pt + req.extend_range.length]
                )
                self.extend_lens[i] -= encoder_len
                self.extend_num_tokens -= encoder_len
            else:
                decoder_out_cache_loc.append(
                    self.out_cache_loc[pt : pt + req.extend_range.length]
                )
                self.prefix_lens[i] -= encoder_len

            pt += req.extend_range.length

        # Reassign: ED stripping rebuilds prefill_input_ids_cpu (CPU pinned);
        # resolve_forward_inputs will H2D this on forward stream. self.input_ids
        # stays None.
        self.prefill_input_ids_cpu = flatten_arrays_to_pinned_cpu(input_ids, _pin)
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, pin_memory=_pin).to(
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

        if self.extend_input_logprob_token_ids is not None:
            new_token_ids_parts = []
            offset = 0
            for i, req in enumerate(self.reqs):
                encoder_len = self.encoder_lens_cpu[i]
                old_start_len = self.extend_logprob_start_lens[i]
                old_contribution = req.extend_range.length - old_start_len

                if len(req.prefix_indices) < encoder_len:
                    tokens_to_strip = max(0, encoder_len - old_start_len)
                    new_token_ids_parts.append(
                        self.extend_input_logprob_token_ids[
                            offset + tokens_to_strip : offset + old_contribution
                        ]
                    )
                    self.extend_logprob_start_lens[i] = max(
                        0, old_start_len - encoder_len
                    )
                else:
                    new_token_ids_parts.append(
                        self.extend_input_logprob_token_ids[
                            offset : offset + old_contribution
                        ]
                    )

                offset += old_contribution

            if new_token_ids_parts:
                self.extend_input_logprob_token_ids = torch.cat(new_token_ids_parts)
            else:
                self.extend_input_logprob_token_ids = None

        for i, req in enumerate(self.reqs):
            encoder_len = self.encoder_lens_cpu[i]
            if encoder_len == 0:
                continue
            if len(req.prefix_indices) < encoder_len:
                assert len(req.prefix_indices) == 0
                req.extend_range = req.extend_range._replace(
                    start=req.extend_range.start + encoder_len
                )
            req.logprob_start_len = max(req.logprob_start_len, encoder_len)

    def prepare_for_extend(self):
        self.forward_mode = ForwardMode.EXTEND

        if self.is_dllm():
            # For DLLM, we use a separate forward mode
            self.forward_mode = ForwardMode.DLLM_EXTEND

        # Init tensors
        reqs = self.reqs
        input_ids = [r.get_fill_ids()[len(r.prefix_indices) :] for r in reqs]
        extend_num_tokens = sum(len(ids) for ids in input_ids)
        seq_lens = [r.extend_range.end for r in reqs]
        orig_seq_lens = [max(r.extend_range.end, len(r.origin_input_ids)) for r in reqs]
        prefix_lens = [len(r.prefix_indices) for r in reqs]
        extend_lens = [r.extend_range.length for r in reqs]
        extend_logprob_start_lens = [
            compute_extend_logprob_start_len(
                logprob_start_len=r.logprob_start_len,
                prefix_len=prefix_lens[i],
                extend_len=extend_lens[i],
                full_untruncated_fill_len=len(r.full_untruncated_fill_ids),
            )
            for i, r in enumerate(reqs)
        ]

        _pin = is_pin_memory_available(self.device)
        # Stay on pinned CPU; H2D is deferred to forward stream via
        # resolve_forward_inputs.
        pinned_input_ids = flatten_arrays_to_pinned_cpu(input_ids, _pin)
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int64, pin_memory=_pin).to(
            self.device, non_blocking=True
        )
        seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        orig_seq_lens_tensor = torch.tensor(
            orig_seq_lens, dtype=torch.int32, pin_memory=_pin
        ).to(self.device, non_blocking=True)

        # Set batch fields needed by alloc_for_extend
        self.prefix_lens = prefix_lens
        self.extend_lens = extend_lens
        self.seq_lens = seq_lens_tensor
        self.seq_lens_cpu = seq_lens_cpu
        self.extend_num_tokens = extend_num_tokens

        # Allocate memory
        out_cache_loc, req_pool_indices_tensor, req_pool_indices_cpu = alloc_for_extend(
            self
        )

        # Set fields
        input_embeds = []
        all_replace_embeds: List[torch.Tensor] = []
        all_replace_positions: List[int] = []
        has_replace_embeds = False
        input_id_pointer = 0
        input_id_lens = [len(input_id) for input_id in input_ids]
        extend_input_logprob_token_ids = []
        multimodal_inputs = []
        mamba_track_mask_cpu = []
        mamba_track_indices_cpu = []
        mamba_track_seqlens_cpu = []

        for i, (req, seq_len, pre_len) in enumerate(zip(reqs, seq_lens, prefix_lens)):
            assert seq_len - pre_len == req.extend_range.length

            req.extend_batch_idx += 1

            # update req-level memory management fields
            req.kv_committed_len = seq_len
            req.kv_allocated_len = seq_len

            # If input_embeds are available, store them
            if req.input_embeds is not None:
                # Slice to match extend_input_len — PrefillAdder truncates
                # fill_len/extend_input_len on chunk overflow but not input_embeds.
                input_embeds.extend(
                    req.input_embeds[pre_len : pre_len + req.extend_range.length]
                )

            if req.positional_embed_overrides is not None:
                # Override positions are absolute in the full sequence.
                # Convert to extend-tensor coordinates by subtracting pre_len,
                # then skip any that fall within the cached prefix.
                embeds_to_add = []
                for embed_idx, pos in enumerate(
                    req.positional_embed_overrides.positions
                ):
                    extend_pos = pos - pre_len
                    if extend_pos < 0 or extend_pos >= req.extend_range.length:
                        continue  # Outside current extend chunk, skip
                    embeds_to_add.append((embed_idx, input_id_pointer + extend_pos))
                if embeds_to_add:
                    has_replace_embeds = True
                    indices, positions = zip(*embeds_to_add)
                    all_replace_embeds.append(
                        req.positional_embed_overrides.embeds[list(indices)]
                    )
                    all_replace_positions.extend(positions)
            input_id_pointer += input_id_lens[i]

            multimodal_inputs.append(req.multimodal_inputs)

            # Only calculate cached_tokens once. Once retracted, the 'retracted_stain'
            # flag will always True
            if not req.retracted_stain:
                new_cached = pre_len - req.already_computed
                req.cached_tokens += new_cached

                # Calculate detailed breakdown of cached tokens by source (for HiCache)
                # Only compute once on FIRST chunk - subsequent chunks in chunked prefill
                # would incorrectly count previously computed tokens as cache hits.
                if not req._cache_breakdown_computed:
                    # At this point, prefix_indices has been extended with host data
                    # via init_load_back in schedule_policy, so:
                    # - len(prefix_indices) = device_original + host_loaded
                    # - host_hit_length = total tokens from host cache (including storage-prefetched)
                    # - storage_hit_length = tokens loaded from storage backend (L3 hits)
                    # - device_portion = len(prefix_indices) - host_hit_length
                    #
                    # Storage hits are now tracked via scheduler after prefetch completes.
                    # storage_hit_length is set by scheduler.pop_prefetch_loaded_tokens()
                    host_total = req.host_hit_length
                    # Clamp storage to host_total to handle edge cases
                    storage_portion = min(host_total, req.storage_hit_length)
                    host_portion = host_total - storage_portion
                    device_portion = max(0, len(req.prefix_indices) - host_total)

                    req.cached_tokens_device = device_portion
                    req.cached_tokens_host = host_portion
                    req.cached_tokens_storage = storage_portion
                    req._cache_breakdown_computed = True

                req.already_computed = seq_len
            req.is_retracted = False

            if get_server_args().enable_mamba_extra_buffer():
                track_entry = self._mamba_radix_cache_v2_req_prepare_for_extend(req)
                mamba_track_mask_cpu.append(track_entry.track_mask)
                mamba_track_indices_cpu.append(track_entry.track_index)
                mamba_track_seqlens_cpu.append(track_entry.track_seqlen)

            if self.return_logprob:
                # Find input logprob token ids.
                # First, find a global index within origin_input_ids and slide it by 1
                # to compute input logprobs. It is because you need the next token
                # to compute input logprobs. E.g., (chunk size 2)
                #
                # input_logprobs = [1, 2, 3, 4]
                # get_fill_ids() = [1, 2]
                # extend_input_logprob_token_id = [2, 3]
                #
                # Note that it can also overflow. In this case, we pad it with 0.
                # input_logprobs = [1, 2, 3, 4]
                # get_fill_ids() = [3, 4]
                # extend_input_logprob_token_id = [4, 0]
                global_start_idx, global_end_idx = (
                    len(req.prefix_indices),
                    req.extend_range.end,
                )
                if req.logprob_start_len == -1:
                    logprob_start_len = len(req.origin_input_ids)
                else:
                    logprob_start_len = req.logprob_start_len
                # Apply logprob_start_len
                if global_start_idx < logprob_start_len:
                    global_start_idx = logprob_start_len

                logprob_token_ids = req.origin_input_ids[
                    global_start_idx + 1 : global_end_idx + 1
                ]
                extend_input_logprob_token_ids.extend(logprob_token_ids)

                # We will need req.extend_range.length - extend_logprob_start_lens[i] number of
                # tokens, and logprob_token_ids is for input logprob, so pad the rest of them by 0.
                extend_input_logprob_token_ids.extend(
                    [0]
                    * (
                        req.extend_range.length
                        - extend_logprob_start_lens[i]
                        - len(logprob_token_ids)
                    )
                )

        if self.return_logprob:
            extend_input_logprob_token_ids = torch.tensor(
                extend_input_logprob_token_ids
            )
            # Clamp placeholder or out-of-range token IDs (e.g., multimodal hashes)
            # so they stay within the vocab boundary before being sent to GPU.
            extend_input_logprob_token_ids.clamp_(0, self.model_config.vocab_size - 1)
        else:
            extend_input_logprob_token_ids = None

        if has_replace_embeds:
            replace_embeds_tensor = torch.cat(all_replace_embeds, dim=0).to(
                self.device, non_blocking=True
            )
            replace_positions_tensor = torch.tensor(
                all_replace_positions, dtype=torch.long, device=self.device
            )
        else:
            replace_embeds_tensor = None
            replace_positions_tensor = None

        self.input_ids = None
        self.prefill_input_ids_cpu = pinned_input_ids
        self.req_pool_indices = req_pool_indices_tensor
        self.req_pool_indices_cpu = req_pool_indices_cpu
        self.orig_seq_lens = orig_seq_lens_tensor
        self.out_cache_loc = out_cache_loc
        self.input_embeds = (
            torch.tensor(input_embeds, pin_memory=_pin).to(
                self.device, non_blocking=True
            )
            if input_embeds
            else None
        )
        self.replace_embeds = replace_embeds_tensor
        self.replace_positions = replace_positions_tensor
        for mm_input in multimodal_inputs:
            if mm_input is None:
                continue
            if isinstance(mm_input.vision_position_ids, torch.Tensor):
                mm_input.vision_position_ids = mm_input.vision_position_ids.to(
                    self.device, non_blocking=True
                )
            if isinstance(mm_input.visible_frame_counts, torch.Tensor):
                mm_input.visible_frame_counts = mm_input.visible_frame_counts.to(
                    self.device, non_blocking=True
                )
        self.multimodal_inputs = multimodal_inputs
        self.seq_lens_sum = sum(seq_lens)

        if self.return_logprob:
            self.top_logprobs_nums = [r.logprob.top_logprobs_num for r in reqs]
            self.token_ids_logprobs = [r.logprob.token_ids_logprob for r in reqs]

        self.extend_logprob_start_lens = extend_logprob_start_lens
        self.extend_input_logprob_token_ids = extend_input_logprob_token_ids

        if get_server_args().enable_mamba_extra_buffer():
            self.mamba_track_indices = torch.tensor(
                mamba_track_indices_cpu,
                dtype=torch.int64,
                device=self.device,
            )
            self.mamba_track_mask = torch.tensor(
                mamba_track_mask_cpu,
                dtype=torch.bool,
                device=self.device,
            )
            self.mamba_track_seqlens = torch.tensor(
                mamba_track_seqlens_cpu,
                dtype=torch.int64,
                device=self.device,
            )

        # Collect mamba init info for deferred ops on forward stream
        if any(req.mamba_pool_idx is not None for req in reqs):
            self._collect_deferred_mamba_cow_and_clear(reqs)

        if self.model_config.is_encoder_decoder:
            self.prepare_encoder_info_extend(input_ids, seq_lens)

        # Build sampling info
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
        )

    def _mamba_radix_cache_v2_req_prepare_for_extend(
        self,
        req: Req,
    ) -> _MambaRadixCacheV2TrackEntry:
        mamba_cache_chunk_size = get_server_args().mamba_cache_chunk_size

        def _force_track_h(i: int) -> int:
            assert i % mamba_cache_chunk_size == 0
            # There are 3 cases for mamba_track_seqlen passed to mamba_track_seqlens_cpu:
            # 1) aligned with mamba_cache_chunk_size-> retrieve from last_recurrent_state
            #    a) is the last position -> retrieve from last_recurrent_state
            #    b) is NOT the last position -> retrieve from h
            # 2) unaligned with mamba_cache_chunk_size -> retrieve from h
            # Currently, the math calculation only supports case 1a and 2. So for 1b, we need to add 1
            # to force the math calculation to retrieve the correct mamba state from h.
            return i + 1

        mask = req.extend_range.length >= mamba_cache_chunk_size
        track_index = req.mamba_ping_pong_track_buffer[req.mamba_next_track_idx].item()
        mamba_track_seqlen = -1
        if mask:
            # mamba_track_seqlen is used to calculate the indices to track in
            # hybrid_linear_attn_backend's _init_track_ssm_indices. Due to the
            # fact that the ssm state between aligned and non-aligned are retrieved differently,
            # if 1) last pos and 2) is aligned, then retrieved from the last_recurrent_state,
            # otherwise retrieved from h (i.e. unaligned).
            # We need to pass the non-aligned seqlen to the calculation. Even though
            # we pass in mamba_track_seqlen, the actual tracked seqlen is mamba_last_track_seqlen.
            mamba_track_seqlen = len(req.prefix_indices) + req.extend_range.length

            # mamba_track_seqlen_aligned/mamba_last_track_seqlen is actual tracked seqlen. Used to pass to
            # mamba radix cache to track which seqlen this mamba state should store at.
            mamba_track_seqlen_aligned = (
                len(req.prefix_indices)
                + (req.extend_range.length // mamba_cache_chunk_size)
                * mamba_cache_chunk_size
            )

            # mamba_track_fla_chunk_aligned is the aligned seqlen based on mamba_cache_chunk_size
            # If mamba_track_fla_chunk_aligned != mamba_track_seqlen_aligned, which can be true when
            # page_size > mamba_cache_chunk_size, we need to force the math calculation to retrieve the correct mamba state from h
            # by _force_track_h()
            mamba_track_fla_chunk_aligned = (
                len(req.prefix_indices)
                + (req.extend_range.length // mamba_cache_chunk_size)
                * mamba_cache_chunk_size
            )
            if mamba_track_fla_chunk_aligned != mamba_track_seqlen_aligned:
                # We want to track mamba_track_seqlen_aligned, and it's not the last position,
                # so we need to add 1 to the seqlen to retrieve the correct mamba state from h.
                mamba_track_seqlen = _force_track_h(mamba_track_seqlen_aligned)

            # In lazy mode, skip the swap — the second ping-pong slot is not
            # allocated yet; it will be allocated on demand at the track boundary
            # in mamba_lazy_prealloc_at_boundary during prepare_for_decode.
            if not get_server_args().enable_mamba_extra_buffer_lazy():
                req.mamba_next_track_idx = (
                    self.req_to_token_pool.get_mamba_ping_pong_other_idx(
                        req.mamba_next_track_idx
                    )
                )
            if req.mamba_branching_seqlen is not None:
                # track branching point in this forward if the branching point
                # is within the current extend batch.
                branching_seqlen_aligned_mask = (
                    req.mamba_branching_seqlen - len(req.prefix_indices)
                ) % mamba_cache_chunk_size == 0
                if (
                    req.mamba_branching_seqlen > len(req.prefix_indices)
                    and req.mamba_branching_seqlen < mamba_track_seqlen
                    and branching_seqlen_aligned_mask
                ):
                    # We want to track mamba_track_seqlen_aligned, and it's not the last position,
                    # so we need to add 1 to the seqlen to retrieve the correct mamba state from h.
                    # See _force_track_h() for more details.
                    mamba_track_seqlen = _force_track_h(req.mamba_branching_seqlen)
                    mamba_track_seqlen_aligned = req.mamba_branching_seqlen
            req.mamba_last_track_seqlen = mamba_track_seqlen_aligned

        return _MambaRadixCacheV2TrackEntry(
            track_mask=mask,
            track_index=track_index,
            track_seqlen=mamba_track_seqlen,
        )

    def _collect_deferred_mamba_cow_and_clear(self, reqs):
        """Collect deferred COW/clear info from requests."""
        cow_src_tensors = []
        cow_dst_tensors = []
        clear_tensors = []
        for req in reqs:
            if req.mamba_cow_src_index is not None:
                cow_src_tensors.append(req.mamba_cow_src_index)
                cow_dst_tensors.append(req.mamba_pool_idx.unsqueeze(0))
                req.mamba_cow_src_index = None
                req.mamba_needs_clear = False
            elif req.mamba_needs_clear:
                clear_tensors.append(req.mamba_pool_idx.unsqueeze(0))
                req.mamba_needs_clear = False
        self.mamba_cow_src_indices = (
            torch.cat(cow_src_tensors) if cow_src_tensors else None
        )
        self.mamba_cow_dst_indices = (
            torch.cat(cow_dst_tensors) if cow_dst_tensors else None
        )
        self.mamba_clear_indices = torch.cat(clear_tensors) if clear_tensors else None

    def prepare_for_split_prefill(self):
        self.prepare_for_extend()
        # For split prefill, we need to set the forward mode to SPLIT_PREFILL
        self.forward_mode = ForwardMode.SPLIT_PREFILL

    def mix_with_running(self, running_batch: ScheduleBatch):
        self.forward_mode = ForwardMode.MIXED
        running_bs = running_batch.batch_size()

        for req in running_batch.reqs:
            req._refresh_fill_ids()
            full_len = len(req.full_untruncated_fill_ids)
            req.set_extend_range(full_len - 1, full_len)

        # Decode tokens of the running portion live in future_map.output_tokens_buf.
        self.input_ids = None
        self.mix_running_indices = running_batch.req_pool_indices
        out_cache_loc = torch.cat([self.out_cache_loc, running_batch.out_cache_loc])

        self.merge_batch(running_batch)
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

    def new_tokens_required_next_decode(
        self, selected_indices: Optional[List[int]] = None
    ):
        page_size = self.token_to_kv_pool_allocator.page_size
        requests = (
            self.reqs
            if selected_indices is None
            else [self.reqs[i] for i in selected_indices]
        )

        if self.spec_algorithm.is_none():
            new_pages = sum(1 for r in requests if r.kv_committed_len % page_size == 0)
            return new_pages * page_size

        return self._new_tokens_required_next_decode_spec_v2(requests, page_size)

    def _new_tokens_required_next_decode_spec_v2(self, requests, page_size):
        """Tight estimate matching eagle_utils.eagle_prepare_for_decode allocation."""
        reserve = get_alloc_reserve_per_decode()
        total = 0
        for r in requests:
            x = max(0, r.kv_committed_len + reserve - r.kv_allocated_len)
            cur = r.kv_allocated_len
            nxt = cur + x
            total += ceil_align(nxt, page_size) - ceil_align(cur, page_size)
        return total

    def check_decode_mem(self, selected_indices: Optional[List[int]] = None):
        num_tokens = self.new_tokens_required_next_decode(selected_indices)
        evict_from_tree_cache(self.tree_cache, num_tokens)
        return self.token_to_kv_pool_allocator.available_size() >= num_tokens

    def retract_all(self, server_args: ServerArgs, offload_kv: bool = True):
        retracted_reqs = retract_all(
            reqs=self.reqs,
            server_args=server_args,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            hisparse_coordinator=self.hisparse_coordinator,
            offload_kv=offload_kv,
        )
        self.reqs = []
        return retracted_reqs

    def retract_decode(
        self, server_args: ServerArgs
    ) -> Tuple[List[Req], float, List[Req]]:
        """Retract the decoding requests when there is not enough memory."""
        sorted_indices = self._get_decode_retraction_order(
            self.reqs,
            server_args,
            allow_policy_sort=(
                self.spec_algorithm is None or self.spec_algorithm.is_none()
            ),
        )

        retracted_reqs = []
        first_iter = True
        while first_iter or (
            not self.check_decode_mem(selected_indices=sorted_indices)
        ):
            if len(sorted_indices) == 1:
                # Always keep at least one request
                break

            first_iter = False
            idx = sorted_indices.pop()
            req = self.reqs[idx]
            retracted_reqs.append(req)
            # release memory and don't insert into the tree because we need the space instantly
            self.release_req(idx, len(sorted_indices), server_args)

        reqs_to_abort: List[Req] = []
        if len(sorted_indices) <= 1 and not self.check_decode_mem(
            selected_indices=sorted_indices
        ):
            # Even the last remaining request cannot fit in memory.
            # Instead of crashing the scheduler, gracefully abort it.
            last_idx = sorted_indices.pop()
            last_req = self.reqs[last_idx]
            last_req.to_finish = FINISH_ABORT(
                "Out of memory even after retracting all other requests "
                "in the decode batch. Aborting the last request.",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            reqs_to_abort.append(last_req)
            self.release_req(last_idx, 0, server_args)
            logger.warning(
                "retract_decode: aborted last request %s due to OOM", last_req.rid
            )

        self.filter_batch(keep_indices=sorted_indices)

        # Reqs in batch are filtered
        new_estimate_ratio = (
            NewTokenRatioTracker.estimate_new_token_ratio_after_retract(self.reqs)
        )

        return retracted_reqs, new_estimate_ratio, reqs_to_abort

    @staticmethod
    def _get_decode_retraction_order(
        reqs: List[Req], server_args: ServerArgs, *, allow_policy_sort: bool
    ) -> List[int]:
        """Return indices ordered from most-preferred to least-preferred to keep.

        The retraction loop pops from the end of this list, so the least-preferred
        request is retracted first.
        """
        sorted_indices = list(range(len(reqs)))

        # TODO(lsyin): improve retraction policy for radix cache
        # For spec decoding, filter_batch API can only filter requests from the
        # back, so we can only retract from the back.
        # TODO(sang): Clean up finish path and support better retract policy.
        if not allow_policy_sort:
            return sorted_indices

        def length_key(req: Req) -> Tuple[int, int]:
            return (len(req.output_ids), -len(req.origin_input_ids))

        if server_args.retraction_policy == "priority":
            priority_sign = 1 if server_args.schedule_low_priority_values_first else -1

            def retraction_key(req: Req) -> Tuple[int, int, int]:
                priority = req.priority
                if priority is None:
                    priority = (
                        sys.maxsize
                        if server_args.schedule_low_priority_values_first
                        else -sys.maxsize - 1
                    )
                return (priority * (-priority_sign), *length_key(req))

            sorted_indices.sort(
                key=lambda i: retraction_key(reqs[i]),
                reverse=True,
            )
            return sorted_indices

        sorted_indices.sort(
            key=lambda i: length_key(reqs[i]),
            reverse=True,
        )
        return sorted_indices

    def release_req(self, idx: int, remaing_req_count: int, server_args: ServerArgs):
        release_req(
            req=self.reqs[idx],
            remaing_req_count=remaing_req_count,
            server_args=server_args,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            hisparse_coordinator=self.hisparse_coordinator,
        )

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
        self.req_pool_indices = torch.empty(0, dtype=torch.int64, device=self.device)
        self.req_pool_indices_cpu = torch.empty(0, dtype=torch.int64)
        self.seq_lens_sum = 0
        self.extend_num_tokens = 0
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
        )

    def mamba_lazy_prealloc_at_boundary(self, mamba_track_interval: int):
        """Allocate a temporary second ping-pong slot for reqs at a track boundary.

        In lazy mode each request normally holds only 1 ping-pong slot.
        When seq_len hits a track interval boundary, we allocate the
        second slot so the forward pass can write the new tracked state
        there. The old slot is freed after the forward in
        mamba_lazy_post_decode_at_boundary.
        """
        pool = self.req_to_token_pool
        for i, req in enumerate(self.reqs):
            buf = req.mamba_ping_pong_track_buffer
            assert buf is not None
            # Skip reqs not at a track boundary
            if self.seq_lens_cpu[i].item() % mamba_track_interval != 0:
                continue
            other_idx = 1 - req.mamba_next_track_idx
            if buf[other_idx].item() != -1:
                # With overlap the previous forward's post-processing
                # (which frees this slot) hasn't run yet. Skip.
                continue
            if envs.SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL.get():
                new_slot = None
            else:
                new_slot = pool.mamba_allocator.alloc(1)
                if new_slot is None:
                    self.tree_cache.evict(EvictParams(num_tokens=0, mamba_num=1))
                    new_slot = pool.mamba_allocator.alloc(1)
            if new_slot is not None:
                pool.set_mamba_ping_pong_slot(req, other_idx, new_slot[0])
                req.mamba_next_track_idx = other_idx

    def cumulate_penalty_output_tokens(self):
        # Under overlap batch.input_ids is just a placeholder here -- the
        # real token is relayed via future_map and resolved at forward
        # entry. So take the last output token from Req directly
        # (origin_input_ids[-1] on the first decode, before any output).
        last_tokens = [
            req.output_ids[-1] if len(req.output_ids) else req.origin_input_ids[-1]
            for req in self.reqs
        ]
        # Non-blocking H2D so this per-step copy doesn't sync behind the forward.
        # pin_memory (matching the prefill-path tensors) keeps the copy async;
        # is_pin_memory_available falls back to pageable on unsupported devices.
        latest_output_ids = torch.tensor(
            last_tokens,
            dtype=torch.int64,
            pin_memory=is_pin_memory_available(self.device),
        ).to(self.device, non_blocking=True)
        self.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
            latest_output_ids
        )

    def prepare_for_decode(self):
        self.forward_mode = ForwardMode.DECODE
        # Decode embeds the last output token via embed_tokens; clear the stale
        # prefill-time tensor so it doesn't leak into ForwardBatch.
        self.input_embeds = None

        # Clear context parallel metadata - CP is only for prefill, not decode
        if hasattr(self, "attn_cp_metadata") and self.attn_cp_metadata is not None:
            self.attn_cp_metadata = None

        if not self.spec_algorithm.is_none():
            # Spec decoding owns decode preparation (allocation, seq-lens bookkeeping).
            from sglang.srt.speculative.spec_utils import spec_prepare_for_decode

            spec_prepare_for_decode(self)
            return

        if self.sampling_info.penalizer_orchestrator.is_required:
            self.cumulate_penalty_output_tokens()

        # input_ids is set at end of previous run_batch (placeholder for
        # overlap; next_token_ids cast for non-overlap).

        if self.model_config.is_encoder_decoder:
            self.prepare_encoder_info_decode()

        # Allocate memory (DSV4-NPU c{4,128}_state alloc lens are computed inside
        # the allocator, triggered from mem_cache/common.py.)
        self.out_cache_loc = alloc_for_decode(self, token_per_req=1)

        # Update req-level memory management fields
        for req in self.reqs:
            req.decode_batch_idx += 1
            req.kv_committed_len += 1
            req.kv_allocated_len += 1

        # New-tensor avoids racing model_worker_batch refs queued for
        # overlap forward.
        self.seq_lens = self.seq_lens + 1
        self.seq_lens_cpu = self.seq_lens_cpu + 1
        self.orig_seq_lens = self.orig_seq_lens + 1
        # Sum is recomputed lazily by ForwardBatch.init_new.
        self.seq_lens_sum = None

        if self.hisparse_coordinator is not None:
            self.hisparse_coordinator.map_last_loc_to_buffer(
                self.seq_lens,
                self.out_cache_loc,
                self.req_pool_indices,
                self.seq_lens_cpu,
                self.req_pool_indices_cpu,
            )

        if get_server_args().enable_mamba_extra_buffer():
            mamba_track_interval = get_server_args().mamba_track_interval

            if len(self.reqs) == 0:
                self.mamba_track_indices = torch.empty(
                    (0,), dtype=torch.int64, device=self.device
                )
            else:
                if get_server_args().enable_mamba_extra_buffer_lazy():
                    self.mamba_lazy_prealloc_at_boundary(mamba_track_interval)
                set_mamba_track_indices_from_reqs(self)

            # async H2D
            self.mamba_track_mask = (
                (self.seq_lens_cpu % mamba_track_interval == 0)
                .pin_memory()
                .to(device=self.device, non_blocking=True)
            )

    def filter_batch(
        self,
        chunked_req_to_exclude: Optional[Union[Req, List[Req]]] = None,
        keep_indices: Optional[List[int]] = None,
    ):
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
            # Filter out all requests. Stale tensors are left as-is: is_empty()
            # keys off reqs, so callers drop the batch before a forward reads them.
            self.reqs = []
            return

        if len(keep_indices) == len(self.reqs):
            # No need to filter
            return

        keep_indices_device = torch.tensor(
            keep_indices,
            dtype=torch.int64,
            pin_memory=is_pin_memory_available(self.device),
        ).to(self.device, non_blocking=True)

        if self.model_config.is_encoder_decoder:
            self.encoder_lens = self.encoder_lens[keep_indices_device]
            self.encoder_lens_cpu = [self.encoder_lens_cpu[i] for i in keep_indices]

        self.reqs = [self.reqs[i] for i in keep_indices]
        if self.multimodal_inputs is not None:
            self.multimodal_inputs = [self.multimodal_inputs[i] for i in keep_indices]
        self.req_pool_indices = self.req_pool_indices[keep_indices_device]
        self.req_pool_indices_cpu = self.req_pool_indices_cpu[keep_indices]
        self.seq_lens = self.seq_lens[keep_indices_device]
        self.orig_seq_lens = self.orig_seq_lens[keep_indices_device]
        self.out_cache_loc = None
        # Sum is recomputed lazily by ForwardBatch.init_new.
        self.seq_lens_sum = None

        if self.input_ids is not None:
            self.input_ids = self.input_ids[keep_indices_device]
        # Optional under no-verify-sync; resolve_seq_lens repopulates before forward.
        if self.seq_lens_cpu is not None:
            self.seq_lens_cpu = self.seq_lens_cpu[keep_indices]

        self.mamba_track_indices = None
        self.mamba_track_mask = None
        self.mamba_track_seqlens = None
        self.mamba_cow_src_indices = None
        self.mamba_cow_dst_indices = None
        self.mamba_clear_indices = None
        self.return_logprob = any(req.return_logprob for req in self.reqs)
        if self.return_logprob:
            self.top_logprobs_nums = [self.top_logprobs_nums[i] for i in keep_indices]
            self.token_ids_logprobs = [self.token_ids_logprobs[i] for i in keep_indices]
        else:
            self.top_logprobs_nums = None
            self.token_ids_logprobs = None

        self.has_grammar = any(req.grammar for req in self.reqs)

        self.sampling_info.filter_batch(keep_indices, keep_indices_device)
        if self.spec_info:
            self.spec_info.filter_batch(
                new_indices=keep_indices_device,
                has_been_filtered=False,
            )

    def merge_batch(self, other: ScheduleBatch):
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
        self.req_pool_indices_cpu = torch.cat(
            [self.req_pool_indices_cpu, other.req_pool_indices_cpu]
        )
        self.seq_lens = torch.cat([self.seq_lens, other.seq_lens])
        self.orig_seq_lens = torch.cat([self.orig_seq_lens, other.orig_seq_lens])
        self.out_cache_loc = None
        # Sum is recomputed lazily by ForwardBatch.init_new.
        self.seq_lens_sum = None
        # Cat only when both sides hold a real token tensor; otherwise drop to
        # None and let resolve_forward_inputs rebuild from the merged
        # req_pool_indices. Mismatch arises e.g. with spec_v1, which keeps its
        # tensor while a relay-staged side is None -- there the worker rebuilds.
        if self.input_ids is not None and other.input_ids is not None:
            self.input_ids = torch.cat([self.input_ids, other.input_ids])
        else:
            self.input_ids = None
        # Optional under no-verify-sync; drop the mirror if either side absent.
        if self.seq_lens_cpu is None or other.seq_lens_cpu is None:
            self.seq_lens_cpu = None
        else:
            self.seq_lens_cpu = torch.cat([self.seq_lens_cpu, other.seq_lens_cpu])
        self.mamba_track_indices = None
        self.mamba_track_mask = None
        self.mamba_track_seqlens = None
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
        self.has_grammar |= other.has_grammar
        self.return_hidden_states |= other.return_hidden_states
        self.is_prefill_only = self.is_prefill_only and other.is_prefill_only

        if self.spec_info:
            self.spec_info.merge_batch(other.spec_info)

    def copy(self):
        # Only contain fields that will be used by process_batch_result.
        # Shallow-copy the reqs list so that in-place mutations (filter_batch,
        # merge_batch) on the original don't corrupt this snapshot.
        return ScheduleBatch(
            reqs=self.reqs[:],
            # Per-request extend/prefix lens, snapshotted (sliced like reqs) so the
            # deferred prefill-stats report reads them after the original batch has
            # moved on. prepare_for_extend sets these; mix_with_running mutates them
            # in place. None for decode batches (no extend), which the reader skips.
            extend_lens=self.extend_lens[:] if self.extend_lens is not None else None,
            prefix_lens=self.prefix_lens[:] if self.prefix_lens is not None else None,
            req_to_token_pool=self.req_to_token_pool,
            req_pool_indices=self.req_pool_indices,
            model_config=self.model_config,
            forward_mode=self.forward_mode,
            out_cache_loc=self.out_cache_loc,
            return_logprob=self.return_logprob,
            decoding_reqs=self.decoding_reqs,
            spec_algorithm=self.spec_algorithm,
            spec_info=self.spec_info,
            global_num_tokens=self.global_num_tokens,
            global_num_tokens_for_logprob=self.global_num_tokens_for_logprob,
            can_run_dp_cuda_graph=self.can_run_dp_cuda_graph,
            can_run_dp_breakable_cuda_graph=self.can_run_dp_breakable_cuda_graph,
            is_extend_in_batch=self.is_extend_in_batch,
            is_prefill_only=self.is_prefill_only,
            seq_lens_cpu=self.seq_lens_cpu,
            enable_overlap=self.enable_overlap,
            mamba_track_indices=self.mamba_track_indices,
            mamba_track_mask=self.mamba_track_mask,
            mamba_track_seqlens=self.mamba_track_seqlens,
            dp_cooperation_info=self.dp_cooperation_info,
            prefill_stats=self.prefill_stats,
            fpm_start_time=self.fpm_start_time,
            forward_iter=self.forward_iter,
        )

    def maybe_evict_swa(self):
        if self.tree_cache.supports_swa():
            sliding_window_size = self.tree_cache.sliding_window_size
            server_args = get_server_args()

            release_leaf_lock = (
                envs.SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW.get()
                and hasattr(self.tree_cache, "dec_swa_lock_only")
            )

            eviction_interval = max(1, envs.SGLANG_SWA_EVICTION_INTERVAL.get())
            swa_maintenance_step = (self.forward_iter or 0) % eviction_interval == 0
            for idx, req in enumerate(self.reqs):
                if self.forward_mode.is_decode():
                    # We set evict_swa condition here with two reasons:
                    # 1. In overlap scheduler, we cannot evict swa when req.decode_batch_idx == 0 since the prev extend batch is still running.
                    # 2. Evict swa every eviction_interval iterations to reduce the overhead.
                    if swa_maintenance_step and req.decode_batch_idx >= 1:
                        self._evict_swa(req, req.seqlen - 1)

                    # DSV4-NPU only (no-op elsewhere): the small paged compress-state
                    # pool must drain every decode step, independent of SWA cadence.
                    maybe_evict_dsv4_state(self, req, req.seqlen - 1)

                    # Once the decode position has moved past the sliding window,
                    # the SWA portion of the prefill-time tree lock is no longer
                    # needed by this request. Convert it from protected to
                    # evictable so SWA LRU can reclaim it under pressure.
                    if (
                        release_leaf_lock
                        and not req.swa_prefix_lock_released
                        and req.swa_uuid_for_lock is not None
                        and req.last_node is not None
                        and req.decode_batch_idx >= sliding_window_size
                    ):
                        self.tree_cache.dec_swa_lock_only(
                            req.last_node, req.swa_uuid_for_lock
                        )
                        req.swa_prefix_lock_released = True
                elif self.forward_mode.is_extend() and self.tree_cache.is_chunk_cache():
                    pre_len = self.prefix_lens[idx]
                    if self.enable_overlap:
                        # In chunked prefill case, when the second extend batch is scheduling, the first extend batch is still running, so we cannot evict swa tokens
                        if req.extend_batch_idx < 2:
                            continue
                        else:
                            pre_len = (
                                pre_len - server_args.chunked_prefill_size
                                if server_args.chunked_prefill_size > 0
                                else pre_len
                            )
                            self._evict_swa(req, pre_len)
                    else:
                        self._evict_swa(req, pre_len)

    def _evict_swa(self, req: Req, pre_len: int):
        assert self.tree_cache.supports_swa(), "prefix cache must support swa"
        free_swa_out_of_window_slots(
            req,
            pre_len,
            sliding_window_size=self.tree_cache.sliding_window_size,
            page_size=self.tree_cache.page_size,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            is_chunk_cache=self.tree_cache.is_chunk_cache(),
        )

    def __str__(self):
        return (
            f"ScheduleBatch(forward_mode={self.forward_mode.name if self.forward_mode else 'None'}, "
            f"#req={(len(self.reqs))})"
        )


class NextBatchPlan(msgspec.Struct):
    batch_to_run: Optional[ScheduleBatch]
    running_batch: ScheduleBatch
