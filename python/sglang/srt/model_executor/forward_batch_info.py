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
Store information about a forward batch.

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

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto
from functools import total_ordering
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
import triton
import triton.language as tl

from sglang.srt.distributed.parallel_state import get_moe_expert_parallel_world_size
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    get_attention_dp_rank,
    get_attention_tp_size,
    set_dp_buffer_len,
    set_is_extend_in_batch,
)
from sglang.srt.utils import get_compiler_backend, is_npu, support_triton

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.schedule_batch import ModelWorkerBatch, MultimodalInputs
    from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
    from sglang.srt.speculative.spec_info import SpecInput, SpeculativeAlgorithm

_is_npu = is_npu()


class ForwardMode(IntEnum):
    # Extend a sequence. The KV cache of the beginning part of the sequence is already computed (e.g., system prompt).
    # It is also called "prefill" in common terminology.
    EXTEND = auto()
    # Decode one token.
    DECODE = auto()
    # Contains both EXTEND and DECODE when doing chunked prefill.
    MIXED = auto()
    # No sequence to forward. For data parallel attention, some workers will be IDLE if no sequence are allocated.
    IDLE = auto()

    # Used in speculative decoding: verify a batch in the target model.
    TARGET_VERIFY = auto()
    # Used in speculative decoding: extend a batch in the draft model.
    DRAFT_EXTEND = auto()

    DRAFT_EXTEND_V2 = auto()

    # Split Prefill for PD multiplexing
    SPLIT_PREFILL = auto()

    def is_prefill(self):
        return self.is_extend()

    def is_extend(self, include_draft_extend_v2: bool = False):
        return (
            self == ForwardMode.EXTEND
            or self == ForwardMode.MIXED
            or self == ForwardMode.DRAFT_EXTEND
            or (include_draft_extend_v2 and self == ForwardMode.DRAFT_EXTEND_V2)
            or self == ForwardMode.TARGET_VERIFY
            or self == ForwardMode.SPLIT_PREFILL
        )

    def is_decode(self):
        return self == ForwardMode.DECODE

    def is_mixed(self):
        return self == ForwardMode.MIXED

    def is_idle(self):
        return self == ForwardMode.IDLE

    def is_decode_or_idle(self):
        return self == ForwardMode.DECODE or self == ForwardMode.IDLE

    def is_target_verify(self):
        return self == ForwardMode.TARGET_VERIFY

    def is_draft_extend(self, include_v2: bool = False):
        return self == ForwardMode.DRAFT_EXTEND or (
            include_v2 and self == ForwardMode.DRAFT_EXTEND_V2
        )

    def is_draft_extend_v2(self):
        # For fixed shape logits output in v2 eagle worker
        return self == ForwardMode.DRAFT_EXTEND_V2

    def is_extend_or_draft_extend_or_mixed(self, include_draft_extend_v2: bool = False):
        return (
            self == ForwardMode.EXTEND
            or self == ForwardMode.DRAFT_EXTEND
            or self == ForwardMode.MIXED
            or self == ForwardMode.SPLIT_PREFILL
            or (include_draft_extend_v2 and self == ForwardMode.DRAFT_EXTEND_V2)
        )

    def is_cuda_graph(self):
        return (
            self == ForwardMode.DECODE
            or self == ForwardMode.TARGET_VERIFY
            or self == ForwardMode.IDLE
        )

    def is_cpu_graph(self):
        return self == ForwardMode.DECODE

    def is_split_prefill(self):
        return self == ForwardMode.SPLIT_PREFILL


@total_ordering
class CaptureHiddenMode(IntEnum):
    # Do not capture anything.
    NULL = 0
    # Capture a hidden state of the last token.
    LAST = 1
    # Capture hidden states of all tokens.
    FULL = 2

    def need_capture(self):
        return self != CaptureHiddenMode.NULL

    def is_full(self):
        return self == CaptureHiddenMode.FULL

    def is_last(self):
        return self == CaptureHiddenMode.LAST

    def __lt__(self, other):
        return self.value < other.value


@dataclass
class ForwardBatch:
    """Store all inputs of a forward pass."""

    # The forward mode
    forward_mode: ForwardMode
    # The batch size
    batch_size: int
    # The input ids
    input_ids: torch.Tensor
    # The indices of requests in the req_to_token_pool
    req_pool_indices: torch.Tensor
    # The sequence length
    seq_lens: torch.Tensor
    # The indices of output tokens in the token_to_kv_pool
    out_cache_loc: torch.Tensor

    # The sum of all sequence lengths
    seq_lens_sum: int

    # The original sequence length without being chunked. Qwen-1M related.
    orig_seq_lens: Optional[torch.Tensor] = None

    # Optional seq_lens on cpu
    seq_lens_cpu: Optional[torch.Tensor] = None

    # For logprob
    return_logprob: bool = False
    top_logprobs_nums: Optional[List[int]] = None
    token_ids_logprobs: Optional[List[List[int]]] = None

    # For logits and logprobs post processing
    next_token_logits_buffer: torch.Tensor = None
    temp_scaled_logprobs: bool = False
    temperature: torch.Tensor = None
    top_p_normalized_logprobs: bool = False
    top_p: torch.Tensor = None

    # Position information
    positions: torch.Tensor = None

    # For extend
    extend_num_tokens: Optional[int] = None
    extend_seq_lens: Optional[torch.Tensor] = None
    extend_prefix_lens: Optional[torch.Tensor] = None
    extend_start_loc: Optional[torch.Tensor] = None
    extend_prefix_lens_cpu: Optional[List[int]] = None
    extend_seq_lens_cpu: Optional[List[int]] = None
    extend_logprob_start_lens_cpu: Optional[List[int]] = None
    extend_input_logprob_token_ids_gpu: Optional[torch.Tensor] = None

    # For split prefill
    # intermediate values for split prefill
    hidden_states: torch.Tensor = None
    residual: torch.Tensor = None
    model_specific_states: Dict[str, any] = None
    # current split index of layer
    split_index: int = 0

    # For MLA chunked prefix cache used in chunked prefill
    # Tell attention backend whether the kv cache needs to be attended in current pass
    attn_attend_prefix_cache: Optional[bool] = None
    # Number of prefix cache chunks
    num_prefix_chunks: Optional[int] = None
    # Index of current chunk, used by attention backend
    prefix_chunk_idx: Optional[int] = None
    # Maximum number of tokens in each chunk per sequence. Computed from maximum chunk capacity
    prefix_chunk_len: Optional[int] = None
    # Start positions of prefix cache for each chunk, (num_prefix_chunks, batch_size)
    prefix_chunk_starts: Optional[torch.Tensor] = None
    # Lengths of prefix cache for each chunk, (num_prefix_chunks, batch_size)
    prefix_chunk_seq_lens: Optional[torch.Tensor] = None
    # Accumulated lengths of prefix cache for each chunk, (num_prefix_chunks, batch_size + 1)
    prefix_chunk_cu_seq_lens: Optional[torch.Tensor] = None
    # Max lengths of prefix cache for each chunk, (num_prefix_chunks,)
    prefix_chunk_max_seq_lens: Optional[List[int]] = None
    # Number of tokens in each prefix cache chunk, (num_prefix_chunks,)
    prefix_chunk_num_tokens: Optional[List[int]] = None
    # KV Indices for each chunk
    prefix_chunk_kv_indices: Optional[List[torch.Tensor]] = None
    # For MLA chunked prefix cache used in chunked prefill
    # Tell attention backend whether lse needs to be returned
    mha_return_lse: Optional[bool] = None
    mha_one_shot_kv_indices: Optional[torch.Tensor] = None
    mha_one_shot: Optional[bool] = None

    # For multimodal
    mm_inputs: Optional[List[MultimodalInputs]] = None

    # Encoder-decoder
    encoder_cached: Optional[List[bool]] = None
    encoder_lens: Optional[torch.Tensor] = None
    encoder_lens_cpu: Optional[List[int]] = None
    encoder_out_cache_loc: Optional[torch.Tensor] = None

    # For LoRA
    lora_ids: Optional[List[str]] = None

    # For input embeddings
    input_embeds: Optional[torch.Tensor] = None

    # For cross-encoder model
    token_type_ids: Optional[torch.Tensor] = None

    # Sampling info
    sampling_info: SamplingBatchInfo = None

    # Attention backend
    req_to_token_pool: ReqToTokenPool = None
    token_to_kv_pool: KVCache = None
    attn_backend: AttentionBackend = None

    # For DP attention
    global_num_tokens_cpu: Optional[List[int]] = None
    global_num_tokens_gpu: Optional[torch.Tensor] = None
    # Has to be None when cuda graph is captured.
    global_num_tokens_for_logprob_cpu: Optional[List[int]] = None
    global_num_tokens_for_logprob_gpu: Optional[torch.Tensor] = None
    # The padding mode for DP attention
    dp_padding_mode: Optional[DpPaddingMode] = None
    # for extend, local start pos and num tokens is different in logits processor
    # this will be computed in get_dp_local_info
    # this will be recomputed in LogitsMetadata.from_forward_batch
    dp_local_start_pos: Optional[torch.Tensor] = None  # cached info at runtime
    dp_local_num_tokens: Optional[torch.Tensor] = None  # cached info at runtime
    global_dp_buffer_len: Optional[int] = None
    is_extend_in_batch: bool = False
    can_run_dp_cuda_graph: bool = False
    global_forward_mode: Optional[ForwardMode] = None

    # Whether this batch is prefill-only (no token generation needed)
    is_prefill_only: bool = False

    # Speculative decoding
    spec_info: Optional[SpecInput] = None
    spec_algorithm: SpeculativeAlgorithm = None
    capture_hidden_mode: CaptureHiddenMode = None

    # For padding
    padded_static_len: int = -1  # -1 if not padded
    num_token_non_padded: Optional[torch.Tensor] = None  # scalar tensor
    num_token_non_padded_cpu: int = None

    # For Qwen2-VL
    mrope_positions: torch.Tensor = None

    # For two-batch overlap
    tbo_split_seq_index: Optional[int] = None
    tbo_parent_token_range: Optional[Tuple[int, int]] = None
    tbo_children: Optional[List[ForwardBatch]] = None

    # For matryoshka embeddings
    dimensions: Optional[list[int]] = None

    @classmethod
    def init_new(
        cls,
        batch: ModelWorkerBatch,
        model_runner: ModelRunner,
    ):
        from sglang.srt.two_batch_overlap import TboForwardBatchPreparer

        ret = cls(
            forward_mode=batch.forward_mode,
            batch_size=len(batch.seq_lens),
            input_ids=batch.input_ids,
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            out_cache_loc=batch.out_cache_loc,
            mm_inputs=batch.multimodal_inputs,
            encoder_cached=batch.encoder_cached,
            encoder_lens=batch.encoder_lens,
            encoder_lens_cpu=batch.encoder_lens_cpu,
            encoder_out_cache_loc=batch.encoder_out_cache_loc,
            seq_lens_sum=batch.seq_lens_sum,
            seq_lens_cpu=batch.seq_lens_cpu,
            orig_seq_lens=batch.orig_seq_lens,
            return_logprob=batch.return_logprob,
            top_logprobs_nums=batch.top_logprobs_nums,
            token_ids_logprobs=batch.token_ids_logprobs,
            is_extend_in_batch=batch.is_extend_in_batch,
            can_run_dp_cuda_graph=batch.can_run_dp_cuda_graph,
            global_forward_mode=batch.global_forward_mode,
            is_prefill_only=batch.is_prefill_only,
            lora_ids=batch.lora_ids,
            sampling_info=batch.sampling_info,
            req_to_token_pool=model_runner.req_to_token_pool,
            token_to_kv_pool=model_runner.token_to_kv_pool,
            attn_backend=model_runner.attn_backend,
            spec_algorithm=batch.spec_algorithm,
            spec_info=batch.spec_info,
            capture_hidden_mode=batch.capture_hidden_mode,
            input_embeds=batch.input_embeds,
            token_type_ids=batch.token_type_ids,
            tbo_split_seq_index=batch.tbo_split_seq_index,
            dimensions=batch.dimensions,
        )
        device = model_runner.device

        if batch.extend_input_logprob_token_ids is not None:
            ret.extend_input_logprob_token_ids_gpu = (
                batch.extend_input_logprob_token_ids.to(device, non_blocking=True)
            )

        if enable_num_token_non_padded(model_runner.server_args):
            ret.num_token_non_padded = torch.tensor(
                len(batch.input_ids), dtype=torch.int32
            ).to(device, non_blocking=True)
        ret.num_token_non_padded_cpu = len(batch.input_ids)

        # For MLP sync
        if batch.global_num_tokens is not None:
            assert batch.global_num_tokens_for_logprob is not None

            # process global_num_tokens and global_num_tokens_for_logprob
            if batch.spec_info is not None:
                spec_info: SpecInput = batch.spec_info
                global_num_tokens, global_num_tokens_for_logprob = (
                    spec_info.get_spec_adjusted_global_num_tokens(batch)
                )
            else:
                global_num_tokens = batch.global_num_tokens
                global_num_tokens_for_logprob = batch.global_num_tokens_for_logprob

            ret.global_num_tokens_cpu = global_num_tokens
            ret.global_num_tokens_gpu = torch.tensor(
                global_num_tokens, dtype=torch.int64
            ).to(device, non_blocking=True)

            ret.global_num_tokens_for_logprob_cpu = global_num_tokens_for_logprob
            ret.global_num_tokens_for_logprob_gpu = torch.tensor(
                global_num_tokens_for_logprob, dtype=torch.int64
            ).to(device, non_blocking=True)

        if ret.forward_mode.is_idle():
            ret.positions = torch.empty((0,), dtype=torch.int64, device=device)
            TboForwardBatchPreparer.prepare(
                ret, is_draft_worker=model_runner.is_draft_worker
            )
            return ret

        # Override the positions with spec_info
        if (
            ret.spec_info is not None
            and getattr(ret.spec_info, "positions", None) is not None
        ):
            ret.positions = ret.spec_info.positions

        # Init position information
        if ret.forward_mode.is_decode() or ret.forward_mode.is_target_verify():
            if ret.positions is None:
                ret.positions = clamp_position(batch.seq_lens)
        else:
            assert isinstance(batch.extend_seq_lens, list)
            assert isinstance(batch.extend_prefix_lens, list)
            ret.extend_seq_lens = torch.tensor(
                batch.extend_seq_lens, dtype=torch.int32
            ).to(device, non_blocking=True)
            ret.extend_prefix_lens = torch.tensor(
                batch.extend_prefix_lens, dtype=torch.int32
            ).to(device, non_blocking=True)
            ret.extend_num_tokens = batch.extend_num_tokens
            positions, ret.extend_start_loc = compute_position(
                model_runner.server_args.attention_backend,
                ret.extend_prefix_lens,
                ret.extend_seq_lens,
                ret.extend_num_tokens,
            )
            if ret.positions is None:
                ret.positions = positions
            ret.extend_prefix_lens_cpu = batch.extend_prefix_lens
            ret.extend_seq_lens_cpu = batch.extend_seq_lens
            ret.extend_logprob_start_lens_cpu = batch.extend_logprob_start_lens

        if model_runner.model_is_mrope:
            if (
                ret.spec_info is not None
                and getattr(ret.spec_info, "positions", None) is not None
            ):
                ret._compute_spec_mrope_positions(model_runner, batch)
            else:
                ret._compute_mrope_positions(model_runner, batch)

        # Init lora information
        if model_runner.server_args.enable_lora:
            model_runner.lora_manager.prepare_lora_batch(ret)

        TboForwardBatchPreparer.prepare(
            ret, is_draft_worker=model_runner.is_draft_worker
        )

        return ret

    def merge_mm_inputs(self) -> Optional[MultimodalInputs]:
        """
        Merge all multimodal inputs in the batch into a single MultiModalInputs object.

        Returns:
            if none, current batch contains no multimodal input

        """
        if not self.mm_inputs or all(x is None for x in self.mm_inputs):
            return None
        # Filter out None values
        valid_inputs = [x for x in self.mm_inputs if x is not None]

        # TODO: is it expensive?
        # a workaround to avoid importing `MultimodalInputs`
        merged = valid_inputs[0].__class__(mm_items=[])

        # Merge remaining inputs
        for mm_input in valid_inputs:
            merged.merge(mm_input)

        return merged

    def contains_image_inputs(self) -> bool:
        if self.mm_inputs is None:
            return False
        return any(
            mm_input is not None and mm_input.contains_image_inputs()
            for mm_input in self.mm_inputs
        )

    def contains_audio_inputs(self) -> bool:
        if self.mm_inputs is None:
            return False
        return any(
            mm_input is not None and mm_input.contains_audio_inputs()
            for mm_input in self.mm_inputs
        )

    def contains_video_inputs(self) -> bool:
        if self.mm_inputs is None:
            return False
        return any(
            mm_input is not None and mm_input.contains_video_inputs()
            for mm_input in self.mm_inputs
        )

    def contains_mm_inputs(self) -> bool:
        return (
            self.contains_audio_inputs()
            or self.contains_video_inputs()
            or self.contains_image_inputs()
        )

    def _compute_spec_mrope_positions(
        self, model_runner: ModelRunner, batch: ModelWorkerBatch
    ):
        # TODO support batched deltas
        batch_size = self.seq_lens.shape[0]
        device = model_runner.device
        mm_inputs = batch.multimodal_inputs

        if batch.forward_mode.is_draft_extend():  # draft_extend_after_decode
            mrope_deltas = []
            extend_lens = []
            for batch_idx in range(batch_size):
                extend_seq_len = batch.extend_seq_lens[batch_idx]
                extend_lens.append(extend_seq_len)
                mrope_delta = (
                    torch.zeros(1, dtype=torch.int64)
                    if mm_inputs[batch_idx] is None
                    else mm_inputs[batch_idx].mrope_position_delta.squeeze(0)
                )
                mrope_deltas.append(mrope_delta.to(device=device))
            position_chunks = torch.split(batch.spec_info.positions, extend_lens)
            mrope_positions_list = [
                pos_chunk + delta
                for pos_chunk, delta in zip(position_chunks, mrope_deltas)
            ]
            next_input_positions = (
                torch.cat(mrope_positions_list, dim=0).unsqueeze(0).repeat(3, 1)
            )

        else:  # target_verify or draft_decode
            seq_positions = batch.spec_info.positions.view(batch_size, -1)
            mrope_deltas = [
                (
                    torch.tensor([0], dtype=torch.int64)
                    if mm_inputs[i] is None
                    else mm_inputs[i].mrope_position_delta.squeeze(0)
                )
                for i in range(batch_size)
            ]
            mrope_delta_tensor = torch.stack(mrope_deltas, dim=0).to(device=device)
            next_input_positions = (
                (seq_positions + mrope_delta_tensor).flatten().unsqueeze(0).repeat(3, 1)
            )

        self.mrope_positions = next_input_positions

    def _expand_mrope_from_input(
        self,
        mm_input: MultimodalInputs,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if mm_input.mrope_position_delta.device.type != device:
            # transfer mrope_position_delta to device when the first running,
            # avoiding successvie host-to-device data transfer
            mm_input.mrope_position_delta = mm_input.mrope_position_delta.to(
                device, non_blocking=True
            )

        mrope_position_deltas = mm_input.mrope_position_delta.flatten()
        mrope_positions = (
            (mrope_position_deltas + seq_len - 1).unsqueeze(0).repeat(3, 1)
        )
        return mrope_positions

    def _compute_mrope_positions(
        self, model_runner: ModelRunner, batch: ModelWorkerBatch
    ):
        # batch_size * [3 * seq_len]
        batch_size = self.seq_lens.shape[0]
        mrope_positions_list = [[]] * batch_size
        for batch_idx in range(batch_size):
            mm_input = batch.multimodal_inputs[batch_idx]
            if self.forward_mode.is_decode():
                # 3 * N
                if mm_input is None:
                    mrope_positions_list[batch_idx] = torch.full(
                        (3, 1),
                        self.seq_lens[batch_idx] - 1,
                        dtype=torch.int64,
                        device=model_runner.device,
                    )
                else:
                    mrope_positions = self._expand_mrope_from_input(
                        mm_input, self.seq_lens[batch_idx], model_runner.device
                    )
                    mrope_positions_list[batch_idx] = mrope_positions
            elif self.forward_mode.is_extend():
                extend_seq_len, extend_prefix_len = (
                    batch.extend_seq_lens[batch_idx],
                    batch.extend_prefix_lens[batch_idx],
                )
                if mm_input is None:
                    # text only
                    mrope_positions = torch.tensor(
                        [
                            [
                                pos
                                for pos in range(
                                    extend_prefix_len,
                                    extend_prefix_len + extend_seq_len,
                                )
                            ]
                        ]
                        * 3
                    )
                else:
                    mrope_positions = mm_input.mrope_positions[
                        :,
                        extend_prefix_len : extend_prefix_len + extend_seq_len,
                    ]
                    if mrope_positions.numel() == 0:
                        mrope_positions = self._expand_mrope_from_input(
                            mm_input, self.seq_lens[batch_idx], model_runner.device
                        )
                mrope_positions_list[batch_idx] = mrope_positions

        self.mrope_positions = torch.cat(
            [pos.to(device=model_runner.device) for pos in mrope_positions_list],
            dim=1,
        ).to(dtype=torch.int64, device=model_runner.device)

    def get_max_chunk_capacity(self):
        # Maximum number of tokens in each chunk
        # TODO: Should be changed to a better value, maybe passed through server args
        return 128 * 1024

    def set_prefix_chunk_idx(self, idx: int):
        self.prefix_chunk_idx = idx

    def set_attn_attend_prefix_cache(self, attn_attend_prefix_cache: bool):
        self.attn_attend_prefix_cache = attn_attend_prefix_cache

    def prepare_chunked_kv_indices(self, device: torch.device):
        self.prefix_chunk_kv_indices = []
        for idx in range(self.num_prefix_chunks):
            chunk_starts = self.prefix_chunk_starts[idx]
            chunk_seq_lens = self.prefix_chunk_seq_lens[idx]
            chunk_cu_seq_lens = self.prefix_chunk_cu_seq_lens[idx]
            num_chunk_tokens = self.prefix_chunk_num_tokens[idx]

            chunk_kv_indices = torch.empty(
                num_chunk_tokens, dtype=torch.int32, device=device
            )

            create_chunked_prefix_cache_kv_indices[(self.batch_size,)](
                self.req_to_token_pool.req_to_token,
                self.req_pool_indices,
                chunk_starts,
                chunk_seq_lens,
                chunk_cu_seq_lens,
                chunk_kv_indices,
                self.req_to_token_pool.req_to_token.shape[1],
            )
            self.prefix_chunk_kv_indices.append(chunk_kv_indices)

    def _pad_tensor_to_size(self, tensor: torch.Tensor, size: int, *, value: int = 0):
        if value == 0:
            return torch.cat(
                [tensor, tensor.new_zeros(size - tensor.shape[0], *tensor.shape[1:])],
                dim=0,
            )
        else:
            return torch.cat(
                [
                    tensor,
                    tensor.new_full((size - tensor.shape[0], *tensor.shape[1:]), value),
                ],
                dim=0,
            )

    def prepare_mlp_sync_batch(self, model_runner: ModelRunner):
        assert self.global_num_tokens_cpu is not None
        assert self.global_num_tokens_for_logprob_cpu is not None

        global_num_tokens = self.global_num_tokens_cpu
        sync_group_size = len(global_num_tokens)
        attn_tp_size = get_attention_tp_size()

        for i in range(sync_group_size):
            # make sure that the padded length is divisible by attn_tp_size because we may need reduce-scatter across attn_tp dim.
            # there is no reduce-scatter in LM logprob, so we do not need to adjust the padded length for logprob
            global_num_tokens[i] = (
                (global_num_tokens[i] - 1) // attn_tp_size + 1
            ) * attn_tp_size

        dp_padding_mode = DpPaddingMode.get_dp_padding_mode(
            self.is_extend_in_batch, global_num_tokens
        )
        self.dp_padding_mode = dp_padding_mode

        if dp_padding_mode.is_max_len():
            # when DP gather mode is all gather, we will use
            # all_gather_into_tensor to gather hidden states, where transferred
            # tokens should be padded to the same length. We will also use
            # reduce-scatter instead of all-reduce after MLP.
            max_num_tokens = max(global_num_tokens)
            global_num_tokens = [max_num_tokens] * sync_group_size
            buffer_len = max_num_tokens * sync_group_size
        else:
            buffer_len = sum(global_num_tokens)

        if len(global_num_tokens) > 1:
            num_tokens = global_num_tokens[get_attention_dp_rank()]
        else:
            num_tokens = global_num_tokens[0]

        self.global_dp_buffer_len = buffer_len
        set_dp_buffer_len(
            buffer_len, num_tokens, dp_padding_mode.is_max_len(), global_num_tokens
        )
        set_is_extend_in_batch(self.is_extend_in_batch)

        bs = self.batch_size

        if self.forward_mode.is_decode():
            if self.is_extend_in_batch and dp_padding_mode.is_max_len():
                setattr(self, "_original_forward_mode", self.forward_mode)
                self.forward_mode = ForwardMode.EXTEND
                self.extend_num_tokens = bs
                self.extend_seq_lens = torch.full_like(self.seq_lens, 1)
                self.extend_prefix_lens = self.seq_lens - 1
                self.extend_start_loc = torch.arange(
                    bs, dtype=torch.int32, device=self.seq_lens.device
                )
                self.extend_prefix_lens_cpu = self.extend_prefix_lens.cpu()
                self.extend_seq_lens_cpu = self.extend_seq_lens.cpu()
                self.extend_logprob_start_lens_cpu = self.extend_prefix_lens_cpu
            else:
                setattr(self, "_original_batch_size", self.batch_size)
                if self.spec_info is not None:
                    bs = self.batch_size = (
                        num_tokens // self.spec_info.num_tokens_per_batch
                    )
                else:
                    bs = self.batch_size = num_tokens

        # padding
        self.input_ids = self._pad_tensor_to_size(self.input_ids, num_tokens)
        self.req_pool_indices = self._pad_tensor_to_size(self.req_pool_indices, bs)

        seq_len_fill_value = (
            model_runner.attn_backend.get_cuda_graph_seq_len_fill_value()
        )
        self.seq_lens_sum = self.seq_lens_sum + seq_len_fill_value * (
            bs - self.seq_lens.shape[0]
        )
        self.seq_lens = self._pad_tensor_to_size(
            self.seq_lens, bs, value=seq_len_fill_value
        )
        if self.seq_lens_cpu is not None:
            self.seq_lens_cpu = self._pad_tensor_to_size(
                self.seq_lens_cpu, bs, value=seq_len_fill_value
            )

        self.out_cache_loc = self._pad_tensor_to_size(self.out_cache_loc, num_tokens)
        if self.encoder_lens is not None:
            self.encoder_lens = self._pad_tensor_to_size(self.encoder_lens, bs)
        self.positions = self._pad_tensor_to_size(self.positions, num_tokens)
        self.global_num_tokens_cpu = global_num_tokens
        global_num_tokens_pinned = torch.tensor(global_num_tokens, pin_memory=True)
        self.global_num_tokens_gpu.copy_(global_num_tokens_pinned, non_blocking=True)

        if self.mrope_positions is not None:
            self.mrope_positions = self._pad_tensor_to_size(self.mrope_positions, bs)

        # TODO: check if we need to pad other tensors
        if self.extend_seq_lens is not None:
            self.extend_seq_lens = self._pad_tensor_to_size(self.extend_seq_lens, bs)

        if self.spec_info is not None and self.spec_info.is_draft_input():
            # FIXME(lsyin): remove this isinstance logic
            spec_info = self.spec_info
            self.output_cache_loc_backup = self.out_cache_loc
            self.hidden_states_backup = spec_info.hidden_states
            if spec_info.topk_p is not None:
                spec_info.topk_p = self._pad_tensor_to_size(spec_info.topk_p, bs)
            if spec_info.topk_index is not None:
                spec_info.topk_index = self._pad_tensor_to_size(
                    spec_info.topk_index, bs
                )
            if spec_info.accept_length is not None:
                spec_info.accept_length = self._pad_tensor_to_size(
                    spec_info.accept_length, bs
                )
            spec_info.hidden_states = self._pad_tensor_to_size(
                spec_info.hidden_states, num_tokens
            )

    def post_forward_mlp_sync_batch(self, logits_output: LogitsProcessorOutput):

        self.forward_mode = getattr(self, "_original_forward_mode", self.forward_mode)
        self.batch_size = getattr(self, "_original_batch_size", self.batch_size)
        bs = self.batch_size

        if self.spec_info is not None:
            if self.forward_mode.is_decode():  # draft
                num_tokens = self.hidden_states_backup.shape[0]
                self.positions = self.positions[:num_tokens]
                self.seq_lens = self.seq_lens[:bs]
                self.req_pool_indices = self.req_pool_indices[:bs]
                if self.seq_lens_cpu is not None:
                    self.seq_lens_cpu = self.seq_lens_cpu[:bs]
                logits_output.next_token_logits = logits_output.next_token_logits[
                    :num_tokens
                ]
                logits_output.hidden_states = logits_output.hidden_states[:num_tokens]
            elif self.forward_mode.is_target_verify():  # verify
                num_tokens = bs * self.spec_info.draft_token_num
                logits_output.next_token_logits = logits_output.next_token_logits[
                    :num_tokens
                ]
                logits_output.hidden_states = logits_output.hidden_states[:num_tokens]
            elif self.forward_mode.is_draft_extend():  # draft extend
                self.spec_info.accept_length = self.spec_info.accept_length[:bs]
                logits_output.next_token_logits = logits_output.next_token_logits[:bs]
                logits_output.hidden_states = logits_output.hidden_states[:bs]
            elif self.forward_mode.is_extend() or self.forward_mode.is_idle():
                logits_output.next_token_logits = logits_output.next_token_logits[:bs]
                logits_output.hidden_states = logits_output.hidden_states[:bs]

            if hasattr(self, "hidden_states_backup"):
                self.spec_info.hidden_states = self.hidden_states_backup
            if hasattr(self, "output_cache_loc_backup"):
                self.out_cache_loc = self.output_cache_loc_backup

        elif self.forward_mode.is_decode() or self.forward_mode.is_idle():
            logits_output.next_token_logits = logits_output.next_token_logits[:bs]
            if logits_output.hidden_states is not None:
                logits_output.hidden_states = logits_output.hidden_states[:bs]
        elif self.forward_mode.is_extend():
            num_tokens = self.seq_lens_sum
            logits_output.next_token_logits = logits_output.next_token_logits[
                :num_tokens
            ]
            if logits_output.hidden_states is not None:
                logits_output.hidden_states = logits_output.hidden_states[:num_tokens]

    # Here we suppose the length of each chunk is equal
    # For example, if we have 4 sequences with prefix length [256, 512, 768, 1024], prefix_chunk_len = 256
    # num_prefix_chunks = cdiv(1024, 256) = 4
    # prefix_chunk_starts = [[0, 0, 0, 0], [256, 256, 256, 256], [512, 512, 512, 512], [768, 768, 768, 768]]
    # prefix_chunk_ends = [[256, 256, 256, 256], [256, 512, 512, 512], [256, 512, 768, 768], [256, 512, 768, 1024]]
    # prefix_chunk_seq_lens = [[256, 256, 256, 256], [0, 256, 256, 256], [0, 0, 256, 256], [0, 0, 0, 256]]
    # TODO: Implement a better way to allocate chunk lengths that uses memory spaces more efficiently.
    def get_prefix_chunk_seq_lens(
        self, prefix_lens: torch.Tensor, num_prefix_chunks: int, prefix_chunk_len: int
    ):
        device = prefix_lens.device
        prefix_chunk_starts = (
            torch.arange(num_prefix_chunks, device=device, dtype=torch.int32)
            .unsqueeze(1)
            .expand(-1, self.batch_size)
            * prefix_chunk_len
        )
        prefix_chunk_ends = torch.min(
            prefix_lens.unsqueeze(0),
            prefix_chunk_starts + prefix_chunk_len,
        ).to(torch.int32)

        prefix_chunk_seq_lens = (
            (prefix_chunk_ends - prefix_chunk_starts).clamp(min=0).to(torch.int32)
        )

        return prefix_chunk_starts, prefix_chunk_seq_lens

    # Called before each attention module if using chunked kv cache for prefill
    # Some of the codes are adapted from https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/common.py
    def prepare_chunked_prefix_cache_info(self, device: torch.device):

        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        assert isinstance(
            self.token_to_kv_pool, MLATokenToKVPool
        ), "Currently chunked prefix cache can only be used by Deepseek models"

        if not any(self.extend_prefix_lens_cpu):
            self.num_prefix_chunks = 0
            return

        if self.prefix_chunk_len is not None:
            # Chunked kv cache info already prepared by prior modules
            return

        self.prefix_chunk_idx = -1

        # chunk_capacity is the maximum number of tokens in each chunk
        chunk_capacity = self.get_max_chunk_capacity()
        self.prefix_chunk_len = chunk_capacity // self.batch_size

        self.num_prefix_chunks = (
            max(self.extend_prefix_lens_cpu) + self.prefix_chunk_len - 1
        ) // self.prefix_chunk_len

        # Here we compute chunk lens twice to avoid stream sync, once on gpu and once on cpu.
        prefix_chunk_starts_cuda, prefix_chunk_seq_lens_cuda = (
            self.get_prefix_chunk_seq_lens(
                self.extend_prefix_lens,
                self.num_prefix_chunks,
                self.prefix_chunk_len,
            )
        )
        _, prefix_chunk_seq_lens_cpu = self.get_prefix_chunk_seq_lens(
            torch.tensor(self.extend_prefix_lens_cpu),
            self.num_prefix_chunks,
            self.prefix_chunk_len,
        )
        self.prefix_chunk_starts = prefix_chunk_starts_cuda
        self.prefix_chunk_seq_lens = prefix_chunk_seq_lens_cuda

        # Metadata for attention backend
        self.prefix_chunk_cu_seq_lens = torch.zeros(
            self.num_prefix_chunks,
            self.batch_size + 1,
            device=device,
            dtype=torch.int32,
        )
        self.prefix_chunk_cu_seq_lens[:, 1:] = prefix_chunk_seq_lens_cuda.cumsum(
            dim=1
        ).to(torch.int32)
        self.prefix_chunk_max_seq_lens = prefix_chunk_seq_lens_cpu.max(
            dim=1
        ).values.tolist()

        self.prefix_chunk_num_tokens = prefix_chunk_seq_lens_cpu.sum(dim=1).tolist()
        assert max(self.prefix_chunk_num_tokens) <= self.get_max_chunk_capacity()

        # Precompute the kv indices for each chunk
        self.prepare_chunked_kv_indices(device)

    @property
    def can_run_tbo(self):
        return self.tbo_split_seq_index is not None

    def fetch_mha_one_shot_kv_indices(self):
        if self.mha_one_shot_kv_indices is not None:
            return self.mha_one_shot_kv_indices
        batch_size = self.batch_size
        paged_kernel_lens_sum = sum(self.seq_lens_cpu)
        kv_indices = torch.empty(
            paged_kernel_lens_sum,
            dtype=torch.int32,
            device=self.req_pool_indices.device,
        )
        kv_indptr = torch.zeros(
            batch_size + 1,
            dtype=torch.int32,
            device=self.req_pool_indices.device,
        )
        kv_indptr[1:] = torch.cumsum(self.seq_lens, dim=0)
        create_flashinfer_kv_indices_triton[(self.batch_size,)](
            self.req_to_token_pool.req_to_token,
            self.req_pool_indices,
            self.seq_lens,
            kv_indptr,
            None,
            kv_indices,
            self.req_to_token_pool.req_to_token.shape[1],
        )
        self.mha_one_shot_kv_indices = kv_indices
        return kv_indices


def enable_num_token_non_padded(server_args):
    return get_moe_expert_parallel_world_size() > 1


class PPProxyTensors:
    # adapted from https://github.com/vllm-project/vllm/blob/d14e98d924724b284dc5eaf8070d935e214e50c0/vllm/sequence.py#L1103
    tensors: Dict[str, torch.Tensor]

    def __init__(self, tensors):
        # manually define this function, so that
        # Dynamo knows `IntermediateTensors()` comes from this file.
        # Otherwise, dataclass will generate this function by evaluating
        # a string, and we will lose the information about the source file.
        self.tensors = tensors

    def __getitem__(self, key: Union[str, slice]):
        if isinstance(key, str):
            return self.tensors[key]
        elif isinstance(key, slice):
            return self.__class__({k: v[key] for k, v in self.tensors.items()})

    def __setitem__(self, key: str, value: torch.Tensor):
        self.tensors[key] = value

    def __len__(self):
        return len(self.tensors)

    def __eq__(self, other: object):
        return isinstance(other, self.__class__) and self

    def __repr__(self) -> str:
        return f"PPProxyTensors(tensors={self.tensors})"


def compute_position(
    attn_backend: str,
    extend_prefix_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    extend_seq_lens_sum: int,
):
    if support_triton(attn_backend):
        positions, extend_start_loc = compute_position_triton(
            extend_prefix_lens,
            extend_seq_lens,
            extend_seq_lens_sum,
        )
    else:
        positions, extend_start_loc = compute_position_torch(
            extend_prefix_lens, extend_seq_lens
        )
    return positions, extend_start_loc


def compute_position_triton(
    extend_prefix_lens: torch.Tensor, extend_seq_lens: torch.Tensor, extend_seq_lens_sum
):
    """Compute positions. It is a fused version of `compute_position_torch`."""
    batch_size = extend_seq_lens.shape[0]
    has_prefix = extend_prefix_lens.shape[0] == batch_size

    positions = torch.empty(
        extend_seq_lens_sum, dtype=torch.int64, device=extend_seq_lens.device
    )
    extend_start_loc = torch.empty(
        batch_size, dtype=torch.int32, device=extend_seq_lens.device
    )

    # Launch kernel
    compute_position_kernel[(batch_size,)](
        positions,
        extend_start_loc,
        extend_prefix_lens,
        extend_seq_lens,
        has_prefix,
    )

    return positions, extend_start_loc


@triton.jit
def compute_position_kernel(
    positions,
    extend_start_loc,
    extend_prefix_lens,
    extend_seq_lens,
    has_prefix: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(0).to(tl.int64)

    prefix_len = tl.load(extend_prefix_lens + pid) if has_prefix else 0
    seq_len = tl.load(extend_seq_lens + pid)

    # NOTE: This can be slow for large bs
    cumsum_start = tl.cast(0, tl.int64)
    for i in range(pid):
        cumsum_start += tl.load(extend_seq_lens + i)

    num_loop = tl.cdiv(seq_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        tl.store(
            positions + cumsum_start + offset,
            prefix_len + offset,
            mask=offset < seq_len,
        )
    tl.store(extend_start_loc + pid, cumsum_start)


def compute_position_torch(
    extend_prefix_lens: torch.Tensor, extend_seq_lens: torch.Tensor
):
    positions = torch.cat(
        [
            torch.arange(
                prefix_len, prefix_len + extend_len, device=extend_prefix_lens.device
            )
            for prefix_len, extend_len in zip(extend_prefix_lens, extend_seq_lens)
        ],
        axis=0,
    )
    extend_start_loc = torch.zeros_like(extend_seq_lens)
    extend_start_loc[1:] = torch.cumsum(extend_seq_lens[:-1], dim=0)
    return positions.to(torch.int64), extend_start_loc


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def clamp_position(seq_lens):
    return torch.clamp((seq_lens - 1), min=0).to(torch.int64)


@triton.jit
def create_chunked_prefix_cache_kv_indices(
    req_to_token_ptr,  # (max_batch, max_context_len,)
    req_pool_indices_ptr,  # (batch_size,)
    chunk_start_idx_ptr,  # (batch_size,)
    chunk_seq_lens_ptr,  # (batch_size,)
    chunk_cu_seq_lens_ptr,  # (batch_size + 1,)
    chunk_kv_indices_ptr,  # (num_chunk_tokens,)
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    # find the req pool idx, this is for batch to token
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    chunk_kv_indices_offset = tl.load(chunk_cu_seq_lens_ptr + pid)

    # get the token positions of current chunk
    chunk_start_pos = tl.load(chunk_start_idx_ptr + pid).to(tl.int32)
    chunk_seq_len = tl.load(chunk_seq_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(chunk_seq_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < chunk_seq_len
        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + chunk_start_pos
            + offset,
            mask=mask,
        )
        tl.store(
            chunk_kv_indices_ptr + chunk_kv_indices_offset + offset, data, mask=mask
        )
