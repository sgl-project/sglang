from __future__ import annotations

import hashlib
from enum import Enum, auto

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
"""

import copy
import dataclasses
import logging
import threading
from typing import TYPE_CHECKING, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import triton
import triton.language as tl

from sglang.global_config import global_config
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.disaggregation.base import BaseKVSender
from sglang.srt.disaggregation.decode import ScheduleBatchDisaggregationDecodeMixin
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool, TokenToKVPoolAllocator
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import flatten_nested_list, get_compiler_backend, is_hpu

_is_hpu = is_hpu()
if _is_hpu:
    from sglang.srt.hpu_utils import HPUBlockMetadata, create_hpu_block_metadata

if TYPE_CHECKING:
    from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

INIT_INCREMENTAL_DETOKENIZATION_OFFSET = 5

# Put some global args for easy access
global_server_args_dict = {
    "attention_backend": ServerArgs.attention_backend,
    "sampling_backend": ServerArgs.sampling_backend,
    "triton_attention_reduce_in_fp32": ServerArgs.triton_attention_reduce_in_fp32,
    "torchao_config": ServerArgs.torchao_config,
    "enable_nan_detection": ServerArgs.enable_nan_detection,
    "enable_dp_attention": ServerArgs.enable_dp_attention,
    "enable_ep_moe": ServerArgs.enable_ep_moe,
    "enable_deepep_moe": ServerArgs.enable_deepep_moe,
    "deepep_mode": ServerArgs.deepep_mode,
    "device": ServerArgs.device,
    "speculative_accept_threshold_single": ServerArgs.speculative_accept_threshold_single,
    "speculative_accept_threshold_acc": ServerArgs.speculative_accept_threshold_acc,
    "disable_radix_cache": ServerArgs.disable_radix_cache,
    "flashinfer_mla_disable_ragged": ServerArgs.flashinfer_mla_disable_ragged,
    "moe_dense_tp_size": ServerArgs.moe_dense_tp_size,
    "chunked_prefill_size": ServerArgs.chunked_prefill_size,
    "n_share_experts_fusion": ServerArgs.n_share_experts_fusion,
    "disable_chunked_prefix_cache": ServerArgs.disable_chunked_prefix_cache,
}

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
    def __init__(self, message="Unknown error", status_code=None, err_type=None):
        super().__init__(is_error=True)
        self.message = message
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


@dataclasses.dataclass
class MultimodalDataItem:
    """
    A single multimodal data, from a single image/video/audio or others
    """

    modality: Modality

    hash: int = None
    pad_value: int = None

    aspect_ratio_id: Optional[List[torch.Tensor]] = None
    aspect_ratio_mask: Optional[List[torch.Tensor]] = None

    image_sizes: Tuple[int, int] = None
    image_offsets: Optional[list] = None

    # the real data, pixel_values or audio_features
    # data: Union[List[torch.Tensor], List[np.array]]
    pixel_values: Union[torch.Tensor, np.array] = None
    image_grid_thws: Union[torch.Tensor, np.array] = None
    video_grid_thws: Union[torch.Tensor, np.array] = None

    image_emb_mask: Optional[torch.Tensor] = None
    image_spatial_crop: Optional[torch.Tensor] = None
    second_per_grid_ts: Optional[List[torch.Tensor]] = None

    # [num_images, (n, w, h)]
    tgt_size: Tuple[int, int] = None

    audio_features: Union[torch.Tensor, np.array] = None
    audio_feature_lens: Optional[List[torch.Tensor]] = None

    @staticmethod
    def is_empty_list(l):
        if l is None:
            return True
        return len([item for item in flatten_nested_list(l) if item is not None]) == 0

    def set_pad_value(self):
        """
        Set the pad value after first hashing the data
        """

        def data_hash(data) -> int:
            hash_bytes = hashlib.sha256(data).digest()[:8]
            return int.from_bytes(hash_bytes, byteorder="big", signed=False)

        def tensor_hash(tensor_list) -> int:
            """
            hash a tensor or a tensor list
            """
            tensor = tensor_list
            if isinstance(tensor_list, list):
                tensor_list = flatten_nested_list(tensor_list)
                tensor_list = [
                    x.flatten() if isinstance(x, torch.Tensor) else x
                    for x in tensor_list
                ]
                tensor = torch.concat(tensor_list)

            tensor = tensor.detach().contiguous()

            if tensor.dtype == torch.bfloat16:
                # memoryview() doesn't support PyTorch's BFloat16 dtype
                tensor = tensor.float()

            assert isinstance(tensor, torch.Tensor)
            if tensor.is_cuda:
                # TODO: improve this
                tensor_cpu = tensor.cpu()
            else:
                tensor_cpu = tensor

            mv = memoryview(tensor_cpu.numpy())
            return data_hash(mv.tobytes())

        def hash_feature(f):
            if isinstance(f, list):
                if isinstance(f[0], torch.Tensor):
                    return tensor_hash(f)
                return data_hash(tuple(flatten_nested_list(f)))
            elif isinstance(f, np.ndarray):
                arr = np.ascontiguousarray(f)
                arr_bytes = arr.tobytes()
                return data_hash(arr_bytes)
            elif isinstance(f, torch.Tensor):
                return tensor_hash([f])
            return data_hash(f)

        if self.is_audio():
            self.hash = hash_feature(self.audio_features)
        else:
            self.hash = hash_feature(self.pixel_values)

        assert self.hash is not None
        self.pad_value = self.hash % (1 << 30)

    def is_audio(self):
        return (
            self.modality == Modality.AUDIO
        ) and not MultimodalDataItem.is_empty_list(self.audio_features)

    def is_image(self):
        return (
            self.modality == Modality.IMAGE or self.modality == Modality.MULTI_IMAGES
        ) and not MultimodalDataItem.is_empty_list(self.pixel_values)

    def is_video(self):
        return (
            self.modality == Modality.VIDEO
        ) and not MultimodalDataItem.is_empty_list(self.pixel_values)

    def is_valid(self) -> bool:
        return self.is_image() or self.is_video() or self.is_audio()

    def validate(self):
        ...
        # TODO


@dataclasses.dataclass
class MultimodalInputs:
    """The multimodal data related inputs."""

    # items of data
    mm_items: List[MultimodalDataItem]
    image_pad_len: Optional[list] = None
    num_image_tokens: Optional[int] = None

    # QWen2-VL related
    mrope_positions: Optional[torch.Tensor] = None
    mrope_position_delta: Optional[torch.Tensor] = None

    # image
    im_token_id: Optional[int] = None
    im_start_id: Optional[int] = None
    im_end_id: Optional[int] = None
    slice_start_id: Optional[int] = None
    slice_end_id: Optional[int] = None

    # video
    video_token_id: Optional[int] = None

    # audio
    audio_start_id: Optional[torch.Tensor] = None
    audio_end_id: Optional[torch.Tensor] = None

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
            "slice_start_id",
            "slice_end_id",
            "audio_start_id",
            "audio_end_id",
        ]
        for arg in optional_args:
            if arg in obj:
                setattr(ret, arg, obj[arg])

        return ret

    def contains_image_inputs(self) -> bool:
        """ """
        return any(item.is_image() for item in self.mm_items)

    def contains_audio_inputs(self) -> bool:
        """ """
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


class Req:
    """The input and output status of a request."""

    def __init__(
        self,
        rid: str,
        origin_input_text: str,
        origin_input_ids: Tuple[int],
        sampling_params: SamplingParams,
        return_logprob: bool = False,
        top_logprobs_num: int = 0,
        token_ids_logprob: List[int] = None,
        stream: bool = False,
        origin_input_ids_unpadded: Optional[Tuple[int]] = None,
        lora_path: Optional[str] = None,
        input_embeds: Optional[List[List[float]]] = None,
        session_id: Optional[str] = None,
        custom_logit_processor: Optional[str] = None,
        return_hidden_states: bool = False,
        eos_token_ids: Optional[Set[int]] = None,
        bootstrap_host: Optional[str] = None,
        bootstrap_port: Optional[int] = None,
        bootstrap_room: Optional[int] = None,
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
        self.fill_ids = None
        self.session_id = session_id
        self.input_embeds = input_embeds

        # Sampling info
        if isinstance(sampling_params.custom_params, dict):
            sampling_params = copy.copy(sampling_params)
            sampling_params.custom_params = sampling_params.custom_params | {
                "__req__": self
            }
        self.sampling_params = sampling_params
        self.custom_logit_processor = custom_logit_processor
        self.return_hidden_states = return_hidden_states

        # Memory pool info
        self.req_pool_idx: Optional[int] = None

        # Check finish
        self.tokenizer = None
        self.finished_reason = None
        # If we want to abort the request in the middle of the event loop, set this to true
        # Note: We should never set finished_reason in the middle, the req will get filtered and never respond
        self.to_abort = False
        # This carries the error message for `.to_abort` and will be attached to the finished_reason at the end of the event loop
        self.to_abort_message: str = "Unknown error"
        self.stream = stream
        self.eos_token_ids = eos_token_ids

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
        self.prefix_indices = []
        # Number of tokens to run prefill.
        self.extend_input_len = 0
        # The relative logprob_start_len in an extend batch
        self.extend_logprob_start_len = 0
        self.last_node = None
        self.last_node_global = None

        # Whether or not if it is chunked. It increments whenever
        # it is chunked, and decrement whenever chunked request is
        # processed.
        self.is_chunked = 0

        # For retraction
        self.is_retracted = False

        # Logprobs (arguments)
        self.return_logprob = return_logprob
        # Start index to compute logprob from.
        self.logprob_start_len = 0
        self.top_logprobs_num = top_logprobs_num
        self.token_ids_logprob = token_ids_logprob
        self.temp_scaled_logprobs = False
        self.top_p_normalized_logprobs = False

        # Latency Breakdown
        self.queue_time_start = None
        self.queue_time_end = None

        # Logprobs (return values)
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
            self.output_token_logprobs_val = []
            self.output_token_logprobs_idx = []
            self.output_top_logprobs_val = []
            self.output_top_logprobs_idx = []
            self.output_token_ids_logprobs_val = []
            self.output_token_ids_logprobs_idx = []
        else:
            self.output_token_logprobs_val = self.output_token_logprobs_idx = (
                self.output_top_logprobs_val
            ) = self.output_top_logprobs_idx = self.output_token_ids_logprobs_val = (
                self.output_token_ids_logprobs_idx
            ) = None
        self.hidden_states: List[List[float]] = []

        # Embedding (return values)
        self.embedding = None

        # Constrained decoding
        self.grammar: Optional[BaseGrammarObject] = None

        # The number of cached tokens that were already cached in the KV cache
        self.cached_tokens = 0
        self.already_computed = 0

        # The number of verification forward passes in the speculative decoding.
        # This is used to compute the average acceptance length per request.
        self.spec_verify_ct = 0
        self.lora_path = lora_path

        # For disaggregation
        self.bootstrap_host: str = bootstrap_host
        self.bootstrap_port: Optional[int] = bootstrap_port
        self.bootstrap_room: Optional[int] = bootstrap_room
        self.disagg_kv_sender: Optional[BaseKVSender] = None

        # used for warmup because we don't have a pair yet when init
        self.skip_kv_transfer: bool = False
        # the start index of the sent kv cache
        # We want to send it chunk by chunk for chunked prefill.
        # After every chunk forward, we do the following:
        # kv_send(req.input_ids[req.start_send_idx:len(req.fill_ids)])
        # start_send_idx = len(req.fill_ids)
        self.start_send_idx: int = 0

        self.metadata_buffer_index: int = -1
        # The first output_id transferred from prefill instance.
        self.transferred_output_id: Optional[int] = None

        # For overlap schedule, we delay the kv transfer until `process_batch_result_disagg_prefill` rather than `process_prefill_chunk` in non-overlap
        # This is because kv is not ready in `process_prefill_chunk`.
        # We use `tmp_end_idx` to store the end index of the kv cache to send.
        self.tmp_end_idx: int = -1

    @property
    def seqlen(self):
        return len(self.origin_input_ids) + len(self.output_ids)

    def extend_image_inputs(self, image_inputs):
        if self.multimodal_inputs is None:
            self.multimodal_inputs = image_inputs
        else:
            self.multimodal_inputs.merge(image_inputs)

    def finished(self) -> bool:
        # Whether request reached finished condition
        return self.finished_reason is not None

    def init_next_round_input(
        self,
        tree_cache: Optional[BasePrefixCache] = None,
        enable_hierarchical_cache=False,
    ):
        self.fill_ids = self.origin_input_ids + self.output_ids
        if tree_cache is not None:
            # tree cache is None if the prefix is not computed with tree cache.
            if enable_hierarchical_cache:
                self.prefix_indices, self.last_node, self.last_node_global = (
                    tree_cache.match_prefix(
                        key=self.adjust_max_prefix_ids(), include_evicted=True
                    )
                )
            else:
                self.prefix_indices, self.last_node = tree_cache.match_prefix(
                    rid=self.rid, key=self.adjust_max_prefix_ids()
                )
        elif enable_hierarchical_cache:
            # in case last_node is evicted during scheduling, we need to update the prefix_indices
            while self.last_node.evicted:
                self.prefix_indices = self.prefix_indices[
                    : -len(self.last_node.host_value)
                ]
                self.last_node = self.last_node.parent

        self.extend_input_len = len(self.fill_ids) - len(self.prefix_indices)

    def adjust_max_prefix_ids(self):
        self.fill_ids = self.origin_input_ids + self.output_ids
        input_len = len(self.fill_ids)

        # FIXME: To work around some bugs in logprob computation, we need to ensure each
        # request has at least one token. Later, we can relax this requirement and use `input_len`.
        max_prefix_len = input_len - 1

        if self.sampling_params.max_new_tokens > 0:
            # Need at least one token to compute logits
            max_prefix_len = min(max_prefix_len, input_len - 1)

        if self.return_logprob:
            max_prefix_len = min(max_prefix_len, self.logprob_start_len)

        max_prefix_len = max(max_prefix_len, 0)
        return self.fill_ids[:max_prefix_len]

    # Based on https://github.com/vllm-project/vllm/blob/7a64d24aad69e4d2548aa0bf528d9fe63428ab01/vllm/transformers_utils/detokenizer.py#L194-L313
    def init_incremental_detokenize(self):
        first_iter = self.surr_offset is None or self.read_offset is None

        if first_iter:
            self.read_offset = len(self.origin_input_ids_unpadded)
            self.surr_offset = max(
                self.read_offset - INIT_INCREMENTAL_DETOKENIZATION_OFFSET, 0
            )

        all_ids = self.origin_input_ids_unpadded + self.output_ids
        return all_ids[self.surr_offset :], self.read_offset - self.surr_offset

    def check_finished(self):
        if self.finished():
            return

        if self.to_abort:
            self.finished_reason = FINISH_ABORT(
                message=self.to_abort_message,
            )
            return

        if len(self.output_ids) >= self.sampling_params.max_new_tokens:
            self.finished_reason = FINISH_LENGTH(
                length=self.sampling_params.max_new_tokens
            )
            return

        last_token_id = self.output_ids[-1]

        if not self.sampling_params.ignore_eos:
            matched_eos = False

            # Check stop token ids
            if self.sampling_params.stop_token_ids:
                matched_eos = last_token_id in self.sampling_params.stop_token_ids
            if self.eos_token_ids:
                matched_eos |= last_token_id in self.eos_token_ids
            if self.tokenizer is not None:
                matched_eos |= last_token_id == self.tokenizer.eos_token_id
                if self.tokenizer.additional_stop_token_ids:
                    matched_eos |= (
                        last_token_id in self.tokenizer.additional_stop_token_ids
                    )
            if matched_eos:
                self.finished_reason = FINISH_MATCHED_TOKEN(matched=last_token_id)
                return

        # Check stop strings
        if len(self.sampling_params.stop_strs) > 0:
            tail_str = self.tokenizer.decode(
                self.output_ids[-(self.sampling_params.stop_str_max_len + 1) :]
            )

            for stop_str in self.sampling_params.stop_strs:
                if stop_str in tail_str or stop_str in self.decoded_text:
                    self.finished_reason = FINISH_MATCHED_STR(matched=stop_str)
                    return

    def reset_for_retract(self):
        self.prefix_indices = []
        self.last_node = None
        self.extend_input_len = 0
        self.is_retracted = True
        self.input_token_logprobs = None
        self.temp_input_top_logprobs_val = None
        self.temp_input_top_logprobs_idx = None
        self.extend_logprob_start_len = 0
        self.is_chunked = 0
        self.req_pool_idx = None
        self.already_computed = 0

    def __repr__(self):
        return (
            f"Req(rid={self.rid}, "
            f"input_ids={self.origin_input_ids}, output_ids={self.output_ids})"
        )


bid = 0


@dataclasses.dataclass
class ScheduleBatch(ScheduleBatchDisaggregationDecodeMixin):
    """Store all information of a batch on the scheduler."""

    # Request, memory pool, and cache
    reqs: List[Req]
    req_to_token_pool: ReqToTokenPool = None
    token_to_kv_pool_allocator: TokenToKVPoolAllocator = None
    tree_cache: BasePrefixCache = None

    # Batch configs
    model_config: ModelConfig = None
    forward_mode: ForwardMode = None
    enable_overlap: bool = False
    # Tell whether the current running batch is full so that we can skip
    # the check of whether to prefill new requests.
    # This is an optimization to reduce the overhead of the prefill check.
    batch_is_full: bool = False

    # Events
    launch_done: Optional[threading.Event] = None

    # Sampling info
    sampling_info: SamplingBatchInfo = None
    next_batch_sampling_info: SamplingBatchInfo = None

    # Batched arguments to model runner
    input_ids: torch.Tensor = None  # shape: [b], int64
    input_embeds: torch.Tensor = None  # shape: [b, hidden_size], float32
    req_pool_indices: torch.Tensor = None  # shape: [b], int64
    seq_lens: torch.Tensor = None  # shape: [b], int64
    # The output locations of the KV cache
    out_cache_loc: torch.Tensor = None  # shape: [b], int64
    output_ids: torch.Tensor = None  # shape: [b], int64

    # The sum of all sequence lengths
    seq_lens_sum: int = None

    # For DP attention
    global_num_tokens: Optional[List[int]] = None
    global_num_tokens_for_logprob: Optional[List[int]] = None
    can_run_dp_cuda_graph: bool = False

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
    extend_num_tokens: int = None
    decoding_reqs: List[Req] = None
    extend_logprob_start_lens: List[int] = None
    # It comes empty list if logprob is not required.
    extend_input_logprob_token_ids: Optional[torch.Tensor] = None

    # For encoder-decoder architectures
    encoder_cached: Optional[List[bool]] = None
    encoder_lens: Optional[torch.Tensor] = None
    encoder_lens_cpu: Optional[List[int]] = None
    encoder_out_cache_loc: Optional[torch.Tensor] = None

    # Stream
    has_stream: bool = False

    # Has grammar
    has_grammar: bool = False

    # Device
    device: str = "cuda"

    # Speculative decoding
    spec_algorithm: SpeculativeAlgorithm = None
    spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]] = None

    # Enable custom logit processor
    enable_custom_logit_processor: bool = False

    # Whether to return hidden states
    return_hidden_states: bool = False

    @classmethod
    def init_new(
        cls,
        reqs: List[Req],
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator,
        tree_cache: BasePrefixCache,
        model_config: ModelConfig,
        enable_overlap: bool,
        spec_algorithm: SpeculativeAlgorithm,
        enable_custom_logit_processor: bool,
    ):
        return_logprob = any(req.return_logprob for req in reqs)

        return cls(
            reqs=reqs,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=model_config,
            enable_overlap=enable_overlap,
            return_logprob=return_logprob,
            has_stream=any(req.stream for req in reqs),
            has_grammar=any(req.grammar for req in reqs),
            device=req_to_token_pool.device,
            spec_algorithm=spec_algorithm,
            enable_custom_logit_processor=enable_custom_logit_processor,
            return_hidden_states=any(req.return_hidden_states for req in reqs),
        )

    def batch_size(self):
        return len(self.reqs)

    def is_empty(self):
        return len(self.reqs) == 0

    def alloc_req_slots(self, num_reqs: int):
        req_pool_indices = self.req_to_token_pool.alloc(num_reqs)
        if req_pool_indices is None:
            raise RuntimeError(
                "alloc_req_slots runs out of memory. "
                "Please set a smaller number for `--max-running-requests`. "
                f"{self.req_to_token_pool.available_size()=}, "
                f"{num_reqs=}, "
            )
        return req_pool_indices

    def alloc_token_slots(self, num_tokens: int, backup_state: bool = False):
        if self.token_to_kv_pool_allocator.available_size() < num_tokens:
            if self.tree_cache is not None:
                self.tree_cache.evict(num_tokens)

        if backup_state:
            state = self.token_to_kv_pool_allocator.backup_state()

        out_cache_loc = self.token_to_kv_pool_allocator.alloc(num_tokens)
        if out_cache_loc is None:
            phase_str = "Prefill" if self.forward_mode.is_extend() else "Decode"
            error_msg = (
                f"{phase_str} out of memory. Try to lower your batch size.\n"
                f"Try to allocate {num_tokens} tokens.\n"
                f"Avaliable tokens: {self.token_to_kv_pool_allocator.available_size() + self.tree_cache.evictable_size()}\n"
            )
            logger.error(error_msg)
            if self.tree_cache is not None:
                self.tree_cache.pretty_print()
            raise RuntimeError(error_msg)

        if backup_state:
            return out_cache_loc, state
        else:
            return out_cache_loc

    def alloc_paged_token_slots_extend(
        self,
        prefix_lens: torch.Tensor,
        seq_lens: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        backup_state: bool = False,
    ):
        if (
            self.token_to_kv_pool_allocator.available_size()
            < extend_num_tokens
            + len(seq_lens) * self.token_to_kv_pool_allocator.page_size
        ):
            if self.tree_cache is not None:
                self.tree_cache.evict(
                    extend_num_tokens
                    + len(seq_lens) * self.token_to_kv_pool_allocator.page_size,
                )

        if backup_state:
            state = self.token_to_kv_pool_allocator.backup_state()

        out_cache_loc = self.token_to_kv_pool_allocator.alloc_extend(
            prefix_lens, seq_lens, last_loc, extend_num_tokens
        )
        if out_cache_loc is None:
            error_msg = (
                f"Prefill out of memory. Try to lower your batch size.\n"
                f"Try to allocate {extend_num_tokens} tokens.\n"
                f"Avaliable tokens: {self.token_to_kv_pool_allocator.available_size() + self.tree_cache.evictable_size()}\n"
                f"{self.token_to_kv_pool_allocator.available_size()=}\n"
                f"{self.tree_cache.evictable_size()=}\n"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        if backup_state:
            return out_cache_loc, state
        else:
            return out_cache_loc

    def alloc_paged_token_slots_decode(
        self,
        seq_lens: torch.Tensor,
        last_loc: torch.Tensor,
        backup_state: bool = False,
    ):
        if self.tree_cache is not None:
            if (
                self.token_to_kv_pool_allocator.available_size()
                < len(seq_lens) * self.token_to_kv_pool_allocator.page_size
            ):
                self.tree_cache.evict(
                    len(seq_lens) * self.token_to_kv_pool_allocator.page_size,
                )

        if backup_state:
            state = self.token_to_kv_pool_allocator.backup_state()

        out_cache_loc = self.token_to_kv_pool_allocator.alloc_decode(seq_lens, last_loc)
        if out_cache_loc is None:
            error_msg = (
                f"Decode out of memory. Try to lower your batch size.\n"
                f"Try to allocate {len(seq_lens)} tokens.\n"
                f"Avaliable tokens: {self.token_to_kv_pool_allocator.available_size() + self.tree_cache.evictable_size()}\n"
                f"{self.token_to_kv_pool_allocator.available_size()=}\n"
                f"{self.tree_cache.evictable_size()=}\n"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        if backup_state:
            return out_cache_loc, state
        else:
            return out_cache_loc

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

        assert len(self.out_cache_loc) == self.extend_num_tokens

    def prepare_for_extend(self):
        self.forward_mode = ForwardMode.EXTEND

        # Allocate req slots
        bs = len(self.reqs)
        req_pool_indices = self.alloc_req_slots(bs)

        # Init tensors
        reqs = self.reqs
        input_ids = [r.fill_ids[len(r.prefix_indices) :] for r in reqs]
        extend_num_tokens = sum(len(ids) for ids in input_ids)
        seq_lens = [len(r.fill_ids) for r in reqs]
        prefix_lens = [len(r.prefix_indices) for r in reqs]
        extend_lens = [r.extend_input_len for r in reqs]

        req_pool_indices_tensor = torch.tensor(req_pool_indices, dtype=torch.int64).to(
            self.device, non_blocking=True
        )
        input_ids_tensor = torch.tensor(sum(input_ids, []), dtype=torch.int64).to(
            self.device, non_blocking=True
        )
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int64).to(
            self.device, non_blocking=True
        )
        prefix_lens_tensor = torch.tensor(
            prefix_lens, dtype=torch.int64, device=self.device
        )
        extend_lens_tensor = seq_lens_tensor - prefix_lens_tensor

        # Copy prefix and do some basic check
        input_embeds = []
        extend_input_logprob_token_ids = []

        for i, (req, seq_len, pre_len) in enumerate(zip(reqs, seq_lens, prefix_lens)):
            req.req_pool_idx = req_pool_indices[i]
            assert seq_len - pre_len == req.extend_input_len

            if pre_len > 0:
                self.req_to_token_pool.write(
                    (req.req_pool_idx, slice(0, pre_len)), req.prefix_indices
                )

            # If input_embeds are available, store them
            if req.input_embeds is not None:
                # If req.input_embeds is already a list, append its content directly
                input_embeds.extend(req.input_embeds)  # Use extend to avoid nesting

            req.cached_tokens += pre_len - req.already_computed
            req.already_computed = seq_len
            req.is_retracted = False

            # Compute the relative logprob_start_len in an extend batch
            if req.logprob_start_len >= pre_len:
                req.extend_logprob_start_len = min(
                    req.logprob_start_len - pre_len,
                    req.extend_input_len,
                    req.seqlen - 1,
                )
            else:
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

        # Allocate memory
        if self.token_to_kv_pool_allocator.page_size == 1:
            out_cache_loc = self.alloc_token_slots(extend_num_tokens)
        else:
            last_loc = get_last_loc(
                self.req_to_token_pool.req_to_token,
                req_pool_indices_tensor,
                prefix_lens_tensor,
            )
            out_cache_loc = self.alloc_paged_token_slots_extend(
                prefix_lens_tensor, seq_lens_tensor, last_loc, extend_num_tokens
            )

        # Set fields
        self.input_ids = input_ids_tensor
        self.req_pool_indices = req_pool_indices_tensor
        self.seq_lens = seq_lens_tensor
        self.out_cache_loc = out_cache_loc
        self.input_embeds = (
            torch.tensor(input_embeds).to(self.device, non_blocking=True)
            if input_embeds
            else None
        )
        self.seq_lens_sum = sum(seq_lens)

        if self.return_logprob:
            self.top_logprobs_nums = [r.top_logprobs_num for r in reqs]
            self.token_ids_logprobs = [r.token_ids_logprob for r in reqs]

        self.extend_logprob_start_lens = [r.extend_logprob_start_len for r in reqs]
        self.extend_num_tokens = extend_num_tokens
        self.prefix_lens = prefix_lens
        self.extend_lens = extend_lens
        self.extend_input_logprob_token_ids = extend_input_logprob_token_ids

        # Write to req_to_token_pool
        if global_server_args_dict["attention_backend"] not in [
            "torch_native",
            "hpu_attn_backend",
        ]:
            # TODO: some tensors can be reused for ForwardBatchInfo (e.g., extend_lens, cumsum_start)

            write_req_to_token_pool_triton[(bs,)](
                self.req_to_token_pool.req_to_token,
                req_pool_indices_tensor,
                prefix_lens_tensor,
                seq_lens_tensor,
                extend_lens_tensor,
                out_cache_loc,
                self.req_to_token_pool.req_to_token.shape[1],
            )
        else:
            pt = 0
            for i in range(bs):
                self.req_to_token_pool.write(
                    (req_pool_indices[i], slice(prefix_lens[i], seq_lens[i])),
                    out_cache_loc[pt : pt + extend_lens[i]],
                )
                pt += extend_lens[i]

        if self.model_config.is_encoder_decoder:
            self.prepare_encoder_info_extend(input_ids, seq_lens)

        # Build sampling info
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
        )

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

    def new_page_count_next_decode(self):
        page_size = self.token_to_kv_pool_allocator.page_size
        if page_size == 1:
            return len(self.reqs)
        return sum(1 for req in self.reqs if req.seqlen % page_size == 0)

    def check_decode_mem(self, buf_multiplier=1):
        tokens_required = (
            self.new_page_count_next_decode()
            * buf_multiplier
            * self.token_to_kv_pool_allocator.page_size
        )

        if self.token_to_kv_pool_allocator.available_size() >= tokens_required:
            return True

        self.tree_cache.evict(tokens_required)

        return self.token_to_kv_pool_allocator.available_size() >= tokens_required

    def retract_decode(self, server_args: ServerArgs):
        """Retract the decoding requests when there is not enough memory."""
        sorted_indices = [i for i in range(len(self.reqs))]

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

        def get_required_tokens(num_reqs: int):
            headroom_for_spec_decode = 0
            if server_args.speculative_algorithm:
                headroom_for_spec_decode += (
                    num_reqs
                    * server_args.speculative_eagle_topk
                    * server_args.speculative_num_steps
                    + num_reqs * server_args.speculative_num_draft_tokens
                )
            return (
                num_reqs * global_config.retract_decode_steps + headroom_for_spec_decode
            )

        retracted_reqs = []
        seq_lens_cpu = self.seq_lens.cpu().numpy()
        first_iter = True
        while (
            self.token_to_kv_pool_allocator.available_size()
            < get_required_tokens(len(sorted_indices))
            or first_iter
        ):
            if len(sorted_indices) == 1:
                # Corner case: only one request left
                assert (
                    self.token_to_kv_pool_allocator.available_size() > 0
                ), "No space left for only one request"
                break

            first_iter = False
            idx = sorted_indices.pop()
            req = self.reqs[idx]
            retracted_reqs.append(req)

            if isinstance(self.tree_cache, ChunkCache):
                # ChunkCache does not have eviction
                token_indices = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, : seq_lens_cpu[idx]
                ]
                self.token_to_kv_pool_allocator.free(token_indices)
                self.req_to_token_pool.free(req.req_pool_idx)
            else:
                # TODO: apply more fine-grained retraction
                last_uncached_pos = (
                    len(req.prefix_indices) // server_args.page_size
                ) * server_args.page_size
                token_indices = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, last_uncached_pos : seq_lens_cpu[idx]
                ]
                self.token_to_kv_pool_allocator.free(token_indices)
                self.req_to_token_pool.free(req.req_pool_idx)

                # release the last node
                self.tree_cache.dec_lock_ref(req.last_node)

                # NOTE(lsyin): we should use the newly evictable memory instantly.
                residual_size = (
                    len(sorted_indices) * global_config.retract_decode_steps
                    - self.token_to_kv_pool_allocator.available_size()
                )
                residual_size = max(0, residual_size)
                self.tree_cache.evict(residual_size)

            req.reset_for_retract()

        self.filter_batch(keep_indices=sorted_indices)

        # Reqs in batch are filtered
        total_decoded_tokens = sum(len(r.output_ids) for r in self.reqs)
        total_max_new_tokens = sum(r.sampling_params.max_new_tokens for r in self.reqs)

        new_estimate_ratio = (
            total_decoded_tokens + global_config.retract_decode_steps * len(self.reqs)
        ) / total_max_new_tokens
        new_estimate_ratio = min(1.0, new_estimate_ratio)

        return retracted_reqs, new_estimate_ratio

    def prepare_encoder_info_decode(self):
        # Reset the encoder cached status
        self.encoder_cached = [True] * len(self.reqs)

    def prepare_for_idle(self):
        self.forward_mode = ForwardMode.IDLE
        self.input_ids = torch.empty(0, dtype=torch.int64, device=self.device)
        self.seq_lens = torch.empty(0, dtype=torch.int64, device=self.device)
        self.out_cache_loc = torch.empty(0, dtype=torch.int64, device=self.device)
        self.req_pool_indices = torch.empty(0, dtype=torch.int32, device=self.device)
        self.seq_lens_sum = 0
        self.extend_num_tokens = 0
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
        )

    def prepare_for_decode(self):
        self.forward_mode = ForwardMode.DECODE
        bs = len(self.reqs)

        if self.spec_algorithm.is_eagle():
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
            locs = self.encoder_lens + self.seq_lens
            self.prepare_encoder_info_decode()
        else:
            locs = self.seq_lens.clone()

        if self.enable_overlap:
            # Do not use in-place operations in the overlap mode
            self.seq_lens = self.seq_lens + 1
        else:
            # A faster in-place version
            self.seq_lens.add_(1)
        self.seq_lens_sum += bs

        # Allocate memory
        if self.token_to_kv_pool_allocator.page_size == 1:
            self.out_cache_loc = self.alloc_token_slots(bs)
        else:
            last_loc = self.req_to_token_pool.req_to_token[
                self.req_pool_indices, self.seq_lens - 2
            ]
            self.out_cache_loc = self.alloc_paged_token_slots_decode(
                self.seq_lens, last_loc
            )

        self.req_to_token_pool.write(
            (self.req_pool_indices, locs), self.out_cache_loc.to(torch.int32)
        )

    def filter_batch(
        self,
        chunked_req_to_exclude: Optional[Req] = None,
        keep_indices: Optional[List[int]] = None,
    ):
        if keep_indices is None:
            keep_indices = [
                i
                for i in range(len(self.reqs))
                if not self.reqs[i].finished()
                and self.reqs[i] is not chunked_req_to_exclude
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
        self.req_pool_indices = self.req_pool_indices[keep_indices_device]
        self.seq_lens = self.seq_lens[keep_indices_device]
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
            self.spec_info.filter_batch(keep_indices_device)

    def merge_batch(self, other: "ScheduleBatch"):
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

        self.return_logprob |= other.return_logprob
        self.has_stream |= other.has_stream
        self.has_grammar |= other.has_grammar
        self.return_hidden_states |= other.return_hidden_states

        if self.spec_info:
            self.spec_info.merge_batch(other.spec_info)

    def get_model_worker_batch(self) -> ModelWorkerBatch:
        if self.forward_mode.is_decode_or_idle():
            extend_seq_lens = extend_prefix_lens = extend_logprob_start_lens = None
        else:
            extend_seq_lens = self.extend_lens
            extend_prefix_lens = self.prefix_lens
            extend_logprob_start_lens = self.extend_logprob_start_lens

        # Create seq_lens_cpu when needed
        if (
            (
                global_server_args_dict["use_mla_backend"]
                and global_server_args_dict["attention_backend"] == "flashinfer"
            )
            or global_server_args_dict["attention_backend"] == "flashmla"
            or global_server_args_dict["attention_backend"] == "fa3"
        ):
            seq_lens_cpu = self.seq_lens.cpu()
        else:
            seq_lens_cpu = None

        if self.sampling_info:
            if self.has_grammar:
                self.sampling_info.grammars = [req.grammar for req in self.reqs]
            else:
                self.sampling_info.grammars = None

        global bid
        bid += 1
        worker_batch = ModelWorkerBatch(
            bid=bid,
            forward_mode=self.forward_mode,
            input_ids=self.input_ids,
            req_pool_indices=self.req_pool_indices,
            seq_lens=self.seq_lens,
            out_cache_loc=self.out_cache_loc,
            seq_lens_sum=self.seq_lens_sum,
            return_logprob=self.return_logprob,
            top_logprobs_nums=self.top_logprobs_nums,
            token_ids_logprobs=self.token_ids_logprobs,
            global_num_tokens=self.global_num_tokens,
            global_num_tokens_for_logprob=self.global_num_tokens_for_logprob,
            can_run_dp_cuda_graph=self.can_run_dp_cuda_graph,
            seq_lens_cpu=seq_lens_cpu,
            extend_num_tokens=self.extend_num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_logprob_start_lens=extend_logprob_start_lens,
            multimodal_inputs=[r.multimodal_inputs for r in self.reqs],
            encoder_cached=self.encoder_cached,
            encoder_lens=self.encoder_lens,
            encoder_lens_cpu=self.encoder_lens_cpu,
            encoder_out_cache_loc=self.encoder_out_cache_loc,
            lora_paths=[req.lora_path for req in self.reqs],
            sampling_info=self.sampling_info,
            input_embeds=self.input_embeds,
            spec_algorithm=self.spec_algorithm,
            spec_info=self.spec_info,
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
            launch_done=self.launch_done,
        )

        if _is_hpu:
            worker_batch.hpu_metadata = create_hpu_block_metadata(
                worker_batch,
                self.token_to_kv_pool_allocator.page_size,
                self.req_to_token_pool,
                disable_prefix_cache=global_server_args_dict["disable_radix_cache"],
            )

        return worker_batch

    def copy(self):
        # Only contain fields that will be used by process_batch_result
        return ScheduleBatch(
            reqs=self.reqs,
            model_config=self.model_config,
            forward_mode=self.forward_mode,
            out_cache_loc=self.out_cache_loc,
            return_logprob=self.return_logprob,
            decoding_reqs=self.decoding_reqs,
            spec_algorithm=self.spec_algorithm,
            enable_custom_logit_processor=self.enable_custom_logit_processor,
        )

    def __str__(self):
        return (
            f"ScheduleBatch(forward_mode={self.forward_mode.name}, "
            f"#req={(len(self.reqs))})"
        )


@dataclasses.dataclass
class ModelWorkerBatch:
    # The batch id
    bid: int
    # The forward mode
    forward_mode: ForwardMode
    # The input ids
    input_ids: torch.Tensor
    # The indices of requests in the req_to_token_pool
    req_pool_indices: torch.Tensor
    # The sequence length
    seq_lens: torch.Tensor
    seq_lens_cpu: Optional[torch.Tensor]
    # The indices of output tokens in the token_to_kv_pool_allocator
    out_cache_loc: torch.Tensor

    # The sum of all sequence lengths
    seq_lens_sum: int

    # For logprob
    return_logprob: bool
    top_logprobs_nums: Optional[List[int]]
    token_ids_logprobs: Optional[List[List[int]]]

    # For DP attention
    global_num_tokens: Optional[List[int]]
    global_num_tokens_for_logprob: Optional[List[int]]
    can_run_dp_cuda_graph: bool

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
    lora_paths: Optional[List[str]]

    # Sampling info
    sampling_info: SamplingBatchInfo

    # The input Embeds
    input_embeds: Optional[torch.tensor] = None

    # Speculative decoding
    spec_algorithm: SpeculativeAlgorithm = None
    spec_info: Optional[Union[EagleVerifyInput, EagleDraftInput]] = None
    # If set, the output of the batch contains the hidden states of the run.
    capture_hidden_mode: CaptureHiddenMode = None

    # Overlap event
    launch_done: Optional[threading.Event] = None
    hpu_metadata: Optional[HPUBlockMetadata] = None


@triton.jit
def write_req_to_token_pool_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices,
    pre_lens,
    seq_lens,
    extend_lens,
    out_cache_loc,
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(0)

    req_pool_index = tl.load(req_pool_indices + pid)
    pre_len = tl.load(pre_lens + pid)
    seq_len = tl.load(seq_lens + pid)

    # NOTE: This can be slow for large bs
    cumsum_start = tl.cast(0, tl.int64)
    for i in range(pid):
        cumsum_start += tl.load(extend_lens + i)

    num_loop = tl.cdiv(seq_len - pre_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < (seq_len - pre_len)
        value = tl.load(out_cache_loc + cumsum_start + offset, mask=mask)
        tl.store(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + offset
            + pre_len,
            value,
            mask=mask,
        )


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_hpu)
def get_last_loc(req_to_token, req_pool_indices_tensor, prefix_lens_tensor):
    return torch.where(
        prefix_lens_tensor > 0,
        req_to_token[req_pool_indices_tensor, prefix_lens_tensor - 1],
        torch.full_like(prefix_lens_tensor, -1),
    )
