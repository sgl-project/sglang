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

ScheduleBatch -> ForwardBatch

- ScheduleBatch is managed by `scheduler.py::Scheduler`.
  It contains high-level scheduling data. Most of the data is on the CPU.
- ForwardBatch is managed by `model_runner.py::ModelRunner`.
  It contains low-level tensor data. Most of the data consists of GPU tensors.
  It is constructed directly from a ScheduleBatch by `ForwardBatch.init_new`.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from enum import IntEnum, auto
from functools import total_ordering
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union

import torch

from sglang.kernels.ops.attention.position import compute_position_triton
from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.environ import envs
from sglang.srt.kv_canary.req_to_expected_token_ids_manager import (
    compute_req_all_ids_info,
)
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    set_dp_buffer_len,
    set_is_extend_in_batch,
    world_dp_gather_enabled,
)
from sglang.srt.model_executor.forward_batch_deepseek_mha_mixin import (
    ForwardBatchDeepSeekMHAMixin,
)
from sglang.srt.runtime_context import get_parallel, get_server_args
from sglang.srt.utils import (
    is_cuda,
    is_hip,
    is_npu,
    support_triton,
)
from sglang.srt.utils.common import ceil_align, is_pin_memory_available

if TYPE_CHECKING:
    from sglang.srt.layers.dcp.metadata import DecodeContextParallelMetadata
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.layers.utils.cp_utils import ContextParallelMetadata
    from sglang.srt.managers.schedule_batch import MultimodalInputs, ScheduleBatch
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
    from sglang.srt.speculative.spec_info import SpecInput, SpeculativeAlgorithm

# Warn-once flag for the deprecated skip_attn_backend_init kwarg; see
# ForwardBatch.apply_deprecated_skip_attn_backend_init.
_skip_attn_backend_init_warned = False

_is_npu = is_npu()


def _elastic_should_preserve_local_token_counts(
    *,
    model_runner: ModelRunner,
    dp_padding_mode: DpPaddingMode,
    global_num_tokens: List[int],
) -> bool:
    if not getattr(model_runner, "enable_elastic_ep", False):
        return False
    if not world_dp_gather_enabled():
        return False
    if not dp_padding_mode.is_max_len():
        return False
    if len(global_num_tokens) <= 1:
        return False

    uneven_token_count = len(set(global_num_tokens)) > 1
    return uneven_token_count


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
    DRAFT_EXTEND_V2 = auto()

    # Used in disaggregated decode worker
    # Represent a batch of requests having their KV cache ready to start decoding
    PREBUILT = auto()

    # Split Prefill for PD multiplexing
    SPLIT_PREFILL = auto()

    # Used in dLLM
    DLLM_EXTEND = auto()

    def is_prefill(self, include_draft_extend_v2: bool = False):
        return self.is_extend(include_draft_extend_v2=include_draft_extend_v2)

    def is_extend(self, include_draft_extend_v2: bool = False):
        return (
            self == ForwardMode.EXTEND
            or self == ForwardMode.MIXED
            or (include_draft_extend_v2 and self == ForwardMode.DRAFT_EXTEND_V2)
            or self == ForwardMode.TARGET_VERIFY
            or self == ForwardMode.SPLIT_PREFILL
            or self == ForwardMode.DLLM_EXTEND
        )

    def is_context_parallel_extend(self, include_draft_extend_v2: bool = False):
        return (
            self == ForwardMode.EXTEND
            or self == ForwardMode.MIXED
            or (
                self == ForwardMode.DRAFT_EXTEND_V2
                if include_draft_extend_v2
                else False
            )
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

    def is_draft_extend_v2(self):
        # For fixed shape logits output in eagle v2 worker
        return self == ForwardMode.DRAFT_EXTEND_V2

    def is_extend_or_draft_extend_or_mixed(self, include_draft_extend_v2: bool = False):
        return (
            self == ForwardMode.EXTEND
            or self == ForwardMode.MIXED
            or self == ForwardMode.SPLIT_PREFILL
            or (include_draft_extend_v2 and self == ForwardMode.DRAFT_EXTEND_V2)
        )

    def is_cuda_graph(self):
        return (
            self == ForwardMode.DECODE
            or self == ForwardMode.TARGET_VERIFY
            or self == ForwardMode.IDLE
            or self == ForwardMode.DLLM_EXTEND
        )

    def is_cpu_graph(self):
        return self == ForwardMode.DECODE

    def is_split_prefill(self):
        return self == ForwardMode.SPLIT_PREFILL

    def is_extend_without_speculative(self):
        return self.is_extend() and not self.is_target_verify()

    def is_prebuilt(self):
        return self == ForwardMode.PREBUILT

    def is_dllm_extend(self):
        return self == ForwardMode.DLLM_EXTEND


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


def compute_local_num_token_non_padded(
    global_num_token_non_padded: torch.Tensor,
    num_tokens_per_dp: int,
) -> torch.Tensor:
    """Compute local non-padded token count for this attention-TP rank.

    Converts a global count (across all TP ranks) to a local count for this rank.
    The "global" scope is within the current DP rank; DP is handled via num_tokens_per_dp.
    """
    attn_tp_rank = get_parallel().attn_tp_rank
    attn_tp_size = get_parallel().attn_tp_size
    tokens_per_rank = num_tokens_per_dp // attn_tp_size

    return torch.clamp(
        global_num_token_non_padded - tokens_per_rank * attn_tp_rank,
        0,
        tokens_per_rank,
    )


@dataclass
class DSV4OutCacheLoc:
    """Per-forward-pass KV cache allocation for DeepSeek-V4 on NPU.

    Bundles slot indices for full/SWA pools, the two compressed-KV pools
    (c4/c128), and the two compressed-state pools (c4_state/c128_state).
    Populated by the NPU V4 allocator (DSV4NPUTokenToKVPoolAllocator) when
    the model is DeepSeek-V4 on NPU; left as ``None`` on ForwardBatch
    otherwise. CUDA's DSV4 path doesn't construct this bundle (state is
    derived via translate_kv_loc_to_compress_state_loc there).

    All fields are token-level slot ids in their respective pools (NOT page
    ids). Attention backends convert to page ids via ``// page_size`` when
    constructing PA_ND block tables.

    State fields default to ``None`` so the bundle is constructible from
    paths that allocate KV but not state (or vice versa); the NPU allocator
    fills all six on real alloc, CUDA paths leave state ones None and use
    the ring-hash translation instead.
    """

    out_full_loc: torch.Tensor
    out_swa_loc: torch.Tensor
    out_c4_loc: torch.Tensor
    out_c128_loc: torch.Tensor
    out_c4_state_loc: Optional[torch.Tensor] = None
    out_c128_state_loc: Optional[torch.Tensor] = None


@dataclass
class DSV4StateLens:
    """Per-extend/decode c4/c128 compress-state pool allocation lens (DSV4-NPU).

    Built by ``ScheduleBatch._compute_dsv4_state_lens_{extend,decode}`` and
    threaded through ``mem_cache/common.py`` to
    ``DSV4NPUTokenToKVPoolAllocator.alloc_{extend,decode}``, which consumes:

      * ``c{4,128}_prefix_lens`` / ``..._cpu`` — per-req prev cumulative
        state-slot count (the paged allocator's ``prefix`` contract).
      * ``c{4,128}_seq_lens`` / ``..._cpu`` — per-req new cumulative count.
      * ``c{4,128}_extend_num_tokens`` — total new state slots this step.

    Replaces the 10 loose ``c{4,128}_state_*`` kwargs the allocator used to
    take: scheduler only produces this object, common only forwards it, the
    allocator only consumes it.
    """

    c4_prefix_lens: torch.Tensor
    c4_prefix_lens_cpu: torch.Tensor
    c4_seq_lens: torch.Tensor
    c4_seq_lens_cpu: torch.Tensor
    c4_extend_num_tokens: int
    c128_prefix_lens: torch.Tensor
    c128_prefix_lens_cpu: torch.Tensor
    c128_seq_lens: torch.Tensor
    c128_seq_lens_cpu: torch.Tensor
    c128_extend_num_tokens: int


@dataclass
class NgramEmbeddingInfo:
    """Ngram embedding state for LongCat models."""

    token_table: torch.Tensor
    column_starts: torch.Tensor
    req_lens: torch.Tensor
    out_column_starts: torch.Tensor
    out_req_lens: torch.Tensor
    # Mask marking chunked (not-yet-finished) prefill requests whose sampled
    # pseudo next-token must NOT be written into the token table.
    skip_token_table_update: Optional[torch.Tensor] = None

    @classmethod
    def create(
        cls,
        token_table: torch.Tensor,
        batch_size: int,
        device: torch.device,
        column_starts=None,
        req_lens=None,
        skip_token_table_update=None,
    ) -> NgramEmbeddingInfo:
        info = cls(
            token_table=token_table,
            column_starts=torch.empty(batch_size, dtype=torch.int32, device=device),
            req_lens=torch.empty(batch_size, dtype=torch.int32, device=device),
            out_column_starts=torch.empty(batch_size, dtype=torch.int32, device=device),
            out_req_lens=torch.empty(batch_size, dtype=torch.int32, device=device),
            skip_token_table_update=skip_token_table_update,
        )
        if column_starts is not None:
            info.column_starts[:] = column_starts
        if req_lens is not None:
            info.req_lens[:] = req_lens
        return info

    def slice(self, bs: int) -> NgramEmbeddingInfo:
        return NgramEmbeddingInfo(
            token_table=self.token_table,
            column_starts=self.column_starts[:bs],
            req_lens=self.req_lens[:bs],
            out_column_starts=self.out_column_starts[:bs],
            out_req_lens=self.out_req_lens[:bs],
            skip_token_table_update=(
                self.skip_token_table_update[:bs]
                if self.skip_token_table_update is not None
                else None
            ),
        )


@dataclass
class ForwardBatch(ForwardBatchDeepSeekMHAMixin):
    """Store all inputs of a forward pass."""

    # === Required core inputs (no default; input_ids / req_pool_indices / seq_lens / out_cache_loc are borrowed from ScheduleBatch) ===
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

    # === Borrowed from ScheduleBatch: GPU tensors (cross-stream; clone targets for stream isolation) ===
    # FIXME(lsyin): these are currently aliased by reference from ScheduleBatch. Once
    # they are cloned/relayed into FB-owned copies at the boundary, move them out of
    # "Borrowed" into a dedicated "Forward-resolved snapshot" group.
    # The original sequence length without being chunked. Qwen-1M related.
    orig_seq_lens: Optional[torch.Tensor] = None

    # DSV4-NPU only: per-pool slot bundle from DSV4NPUTokenToKVPoolAllocator,
    # consumed by the Ascend backend for PA_ND block tables. None elsewhere.
    out_cache_loc_dsv4: Optional[DSV4OutCacheLoc] = None
    # The indices to track mamba state with
    mamba_track_indices: Optional[torch.Tensor] = None  # shape: [b], int64
    # The mask to track mamba state if needed
    mamba_track_mask: Optional[torch.Tensor] = None  # shape: [b], bool
    # The seqlens to track mamba state if masked, prefill only.
    mamba_track_seqlens: Optional[torch.Tensor] = None  # shape: [b], int64
    # Deferred mamba init ops: COW pairs and clear indices (performed on forward stream)
    mamba_cow_src_indices: Optional[torch.Tensor] = None
    mamba_cow_dst_indices: Optional[torch.Tensor] = None
    mamba_clear_indices: Optional[torch.Tensor] = None

    # For input embeddings
    input_embeds: Optional[torch.Tensor] = None
    # For token embedding overrides (sparse replacement at specific positions)
    replace_embeds: Optional[torch.Tensor] = None
    replace_positions: Optional[torch.Tensor] = None

    # For cross-encoder model
    token_type_ids: Optional[torch.Tensor] = None

    # Encoder-decoder device tensors
    encoder_lens: Optional[torch.Tensor] = None
    encoder_out_cache_loc: Optional[torch.Tensor] = None

    # === Borrowed from ScheduleBatch: config / flags (by-value) ===
    # For logprob
    return_logprob: bool = False
    # Whether this batch is prefill-only (no token generation needed)
    is_prefill_only: bool = False
    spec_algorithm: SpeculativeAlgorithm = None
    # For matryoshka embeddings
    dimensions: Optional[list[int]] = None
    # Whether to return pooled hidden states (pre-head transformer output)
    return_pooled_hidden_states: bool = False

    # For DP attention
    is_extend_in_batch: bool = False
    can_run_dp_cuda_graph: bool = False
    can_run_dp_breakable_cuda_graph: bool = False
    global_forward_mode: Optional[ForwardMode] = None

    # For two-batch overlap
    tbo_split_seq_index: Optional[int] = None

    # === Borrowed from ScheduleBatch: host metadata (CPU lists / mirrors) ===
    # Optional seq_lens on cpu (CPU mirror of seq_lens)
    seq_lens_cpu: Optional[torch.Tensor] = None

    # For logprob
    top_logprobs_nums: Optional[List[int]] = None
    token_ids_logprobs: Optional[List[List[int]]] = None

    # For multimodal
    mm_inputs: Optional[List[MultimodalInputs]] = None

    # Encoder-decoder host fields
    encoder_cached: Optional[List[bool]] = None
    encoder_lens_cpu: Optional[List[int]] = None

    # Pre-computed delimiter indices for multi-item scoring (CPU tensors, one per request)
    multi_item_delimiter_indices: Optional[List[torch.Tensor]] = None

    # === Borrowed from ScheduleBatch: compound (carry their own device tensors) ===
    # Sampling info
    sampling_info: SamplingBatchInfo = None
    # Speculative decoding
    spec_info: Optional[SpecInput] = None

    # === Derived from ScheduleBatch.reqs ===
    # For LoRA
    lora_ids: Optional[List[str]] = None
    # For dumper: request IDs for cross-step sequence tracking
    rids: Optional[List[str]] = None

    # === Per-forward overrides passed explicitly to init_new ===
    capture_hidden_mode: CaptureHiddenMode = None
    # For hidden states before normal
    return_hidden_states_before_norm: bool = False

    # Gate for reusing the first MTP draft step's indexer topk across steps;
    # the carried topk lives on spec_info (see EagleDraftInput.dsa_topk_indices).
    reuse_dsa_topk_indices: Optional[bool] = False

    minimax_m3_precached_sparse_layers: Optional[Set[int]] = None

    # === Forward-derived (built in init_new on the forward stream; FB-owned) ===
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

    # For DP attention (MLP sync sizes)
    original_global_num_tokens_cpu: Optional[List[int]] = None
    _original_batch_size: Optional[int] = None
    _original_forward_mode: Optional[ForwardMode] = None
    global_num_tokens_cpu: Optional[List[int]] = None
    global_num_tokens_gpu: Optional[torch.Tensor] = None
    # Has to be None when cuda graph is captured.
    global_num_tokens_for_logprob_cpu: Optional[List[int]] = None
    global_num_tokens_for_logprob_gpu: Optional[torch.Tensor] = None

    # For padding
    num_token_non_padded: Optional[torch.Tensor] = None  # scalar tensor
    num_token_non_padded_cpu: int = None

    # === Runtime-filled (set during the forward pass / cuda graph / managers; not at construction) ===
    # For logits and logprobs post processing
    next_token_logits_buffer: torch.Tensor = None
    temperature: torch.Tensor = None
    top_p: torch.Tensor = None

    # For split prefill
    # intermediate values for split prefill
    hidden_states: torch.Tensor = None
    residual: torch.Tensor = None
    model_specific_states: Dict[str, any] = None
    # current split index of layer
    split_index: int = 0

    # For multimodal
    mm_input_embeds: Optional[torch.Tensor] = None

    # Encoder-decoder cross-attention mask
    cross_attention_custom_mask: Optional[torch.Tensor] = None

    # For DP attention (padding / local info)
    dp_padding_mode: Optional[DpPaddingMode] = None
    # for extend, local start pos and num tokens is different in logits processor
    # this will be computed in get_dp_local_info
    # this will be recomputed in LogitsMetadata.from_forward_batch
    dp_local_start_pos: Optional[torch.Tensor] = None  # cached info at runtime
    dp_local_num_tokens: Optional[torch.Tensor] = None  # cached info at runtime
    global_dp_buffer_len: Optional[int] = None

    # For Qwen2-VL
    mrope_positions: torch.Tensor = None

    # For two-batch overlap
    tbo_parent_token_range: Optional[Tuple[int, int]] = None
    tbo_padded_len: Optional[int] = None
    tbo_children: Optional[List[ForwardBatch]] = None

    attn_cp_metadata: Optional[ContextParallelMetadata] = None

    # For decode context parallel.
    # NOTE: DecodeContextParallelMetadata is imported under TYPE_CHECKING only (see the
    # import block above) — available for annotations but NOT bound at runtime in this
    # module. Import it from sglang.srt.layers.dcp.metadata if a runtime use is added.
    attn_dcp_metadata: Optional[DecodeContextParallelMetadata] = None

    # Decode context parallel KV write mask.
    dcp_kv_mask: Optional[torch.Tensor] = None

    # For ngram embedding
    ngram_embedding_info: Optional[NgramEmbeddingInfo] = None

    # For dumper: int-hashed request / bootstrap-room IDs (derived from rids)
    rids_int: Optional[torch.Tensor] = None
    bootstrap_room_ids_int: Optional[torch.Tensor] = None

    # kv-canary token-id validator snapshot
    req_all_ids_flat: Optional[torch.Tensor] = None
    req_all_ids_lens: Optional[torch.Tensor] = None

    # Attention planning state. True iff attention metadata for this batch has
    # already been planned outside ModelRunner.forward (multi-step draft
    # pre-plan, plan-stream load_batch, hand-built spec batches), so the
    # forward path must not plan again. Only such pre-planners may set this —
    # ModelRunner / graph runners never mark after their own planning. The
    # marker is only valid for the planning regime (backend set) it was set
    # under; a fresh batch from init_new always starts unplanned.
    forward_metadata_ready: bool = False
    # Shapes the batch had when it was marked (plan record). Lets the
    # judgment predicate detect staleness when DP padding
    # (prepare_mlp_sync_batch) reshapes the batch after pre-planning.
    # Deliberately plain ints — no planner object ref on ForwardBatch
    # (runtime refs were removed from this dataclass on purpose).
    forward_metadata_planned_bs: Optional[int] = None
    forward_metadata_planned_num_tokens: Optional[int] = None
    # Whether the forward path may re-plan this batch when its shapes no
    # longer match the plan record. Only mark sites where the forward
    # path's own init_forward_metadata is equivalent to the pre-plan
    # (same backend object, no special context) may opt in; multi-step
    # wrapper plans and view-context plans must keep this False — a
    # forward-path re-plan would clobber their metadata.
    forward_metadata_replan_equivalent: bool = False

    def mark_forward_metadata_ready(self, replan_equivalent: bool = False):
        """Record that attention metadata was pre-planned for this batch.

        Call right next to the out-of-forward planning action
        (e.g. ``draft_attn_backend.init_forward_metadata(fb)`` or
        ``graph_runner.load_batch(fb)``). Records the batch shapes so
        staleness is detectable; pass ``replan_equivalent=True`` only when
        a forward-path re-plan is equivalent to the pre-plan (see field
        docs).
        """
        self.forward_metadata_ready = True
        self.forward_metadata_planned_bs = self.batch_size
        self.forward_metadata_planned_num_tokens = (
            self.input_ids.shape[0] if self.input_ids is not None else 0
        )
        self.forward_metadata_replan_equivalent = replan_equivalent

    def needs_forward_metadata_init(self) -> bool:
        """Single judgment point for whether the forward path must plan.

        A marked batch is treated as stale — and re-planned — when its
        shapes no longer match the plan record AND the mark site declared
        the re-plan safe (replan_equivalent). This runs after
        prepare_mlp_sync_batch in _forward_raw, so the re-plan sees the
        padded (final) shapes. Sites that cannot opt in (multi-step
        wrapper plans etc.) keep today's behavior: marked stays skipped,
        backends' defensive checks remain the backstop.
        """
        if not self.forward_metadata_ready:
            return True
        if not self.forward_metadata_replan_equivalent:
            return False
        num_tokens = self.input_ids.shape[0] if self.input_ids is not None else 0
        return (
            self.batch_size != self.forward_metadata_planned_bs
            or num_tokens != self.forward_metadata_planned_num_tokens
        )

    def apply_deprecated_skip_attn_backend_init(
        self, skip_attn_backend_init: Optional[bool]
    ) -> None:
        """Map the deprecated ``skip_attn_backend_init`` kwarg onto the marker.

        Mapped, not ignored: callers passing True relied on planning being
        skipped — ignoring the flag would silently re-plan and corrupt
        pre-planned multi-step draft metadata. Warns once per process (a
        module flag, not the warnings filter, so the hot decode loop never
        pays warnings.warn per forward).
        """
        if skip_attn_backend_init is None:
            return
        global _skip_attn_backend_init_warned
        if not _skip_attn_backend_init_warned:
            _skip_attn_backend_init_warned = True
            warnings.warn(
                "skip_attn_backend_init is deprecated and will be removed; "
                "pre-planners should call "
                "ForwardBatch.mark_forward_metadata_ready() after planning "
                "instead. The flag is mapped onto the marker for now.",
                DeprecationWarning,
                stacklevel=3,
            )
        if skip_attn_backend_init:
            self.mark_forward_metadata_ready()

    @classmethod
    def init_new(
        cls,
        batch: ScheduleBatch,
        model_runner: ModelRunner,
        *,
        capture_hidden_mode: Optional[CaptureHiddenMode] = None,
        return_hidden_states_before_norm: bool,
    ):
        # init_new must not mutate the input ScheduleBatch; per-forward
        # overrides go through explicit keyword arguments.

        # capture_hidden_mode=None means no override: derive from
        # SB.return_hidden_states / spec_info.capture_hidden_mode.
        if capture_hidden_mode is None:
            if batch.return_hidden_states:
                capture_hidden_mode = CaptureHiddenMode.FULL
            elif batch.spec_info is not None:
                capture_hidden_mode = getattr(
                    batch.spec_info, "capture_hidden_mode", CaptureHiddenMode.NULL
                )
            else:
                capture_hidden_mode = CaptureHiddenMode.NULL

        # extend-mode-only fields are None on decode/idle
        if batch.forward_mode.is_decode_or_idle():
            extend_seq_lens = extend_prefix_lens = extend_logprob_start_lens = None
        else:
            extend_seq_lens = batch.extend_lens
            extend_prefix_lens = batch.prefix_lens
            extend_logprob_start_lens = batch.extend_logprob_start_lens

        # Mirror the grammars-population behavior previously done in
        # ScheduleBatch.get_model_worker_batch.
        if batch.sampling_info is not None:
            if batch.has_grammar:
                batch.sampling_info.grammars = [req.grammar for req in batch.reqs]
            else:
                batch.sampling_info.grammars = None

        # ScheduleBatch.sampling_info is already swapped to the forward-only
        # copy by Scheduler.run_batch under overlap mode (see save/restore
        # block there). Use it directly.
        seq_lens_cpu = batch.seq_lens_cpu

        # TODO(seq-lens-removal): the whole ScheduleBatch seq_lens family
        # (incl. seq_lens_sum) is slated for removal in favor of kv-committed
        # lengths, so this init_new-time backfill onto the ScheduleBatch is
        # tolerated for now despite the init_new-must-not-mutate-SB rule.
        if batch.seq_lens_sum is None and seq_lens_cpu is not None:
            batch.seq_lens_sum = int(seq_lens_cpu.sum())

        ret = cls(
            # Required core inputs
            forward_mode=batch.forward_mode,
            batch_size=len(batch.seq_lens),
            input_ids=batch.input_ids,
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            out_cache_loc=batch.out_cache_loc,
            seq_lens_sum=batch.seq_lens_sum,
            # Inputs aliased by reference from ScheduleBatch
            seq_lens_cpu=seq_lens_cpu,
            orig_seq_lens=batch.orig_seq_lens,
            out_cache_loc_dsv4=batch.out_cache_loc_dsv4,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_track_mask=batch.mamba_track_mask,
            mamba_track_seqlens=batch.mamba_track_seqlens,
            mamba_cow_src_indices=batch.mamba_cow_src_indices,
            mamba_cow_dst_indices=batch.mamba_cow_dst_indices,
            mamba_clear_indices=batch.mamba_clear_indices,
            encoder_lens=batch.encoder_lens,
            encoder_out_cache_loc=batch.encoder_out_cache_loc,
            input_embeds=batch.input_embeds,
            replace_embeds=batch.replace_embeds,
            replace_positions=batch.replace_positions,
            # Scalar config / flags
            return_logprob=batch.return_logprob,
            is_extend_in_batch=batch.is_extend_in_batch,
            can_run_dp_cuda_graph=batch.can_run_dp_cuda_graph,
            can_run_dp_breakable_cuda_graph=batch.can_run_dp_breakable_cuda_graph,
            global_forward_mode=batch.global_forward_mode,
            is_prefill_only=batch.is_prefill_only,
            spec_algorithm=batch.spec_algorithm,
            capture_hidden_mode=capture_hidden_mode,
            return_hidden_states_before_norm=return_hidden_states_before_norm,
            tbo_split_seq_index=batch.tbo_split_seq_index,
            # Host-side metadata
            top_logprobs_nums=batch.top_logprobs_nums,
            token_ids_logprobs=batch.token_ids_logprobs,
            mm_inputs=batch.multimodal_inputs,
            encoder_cached=batch.encoder_cached,
            encoder_lens_cpu=batch.encoder_lens_cpu,
            lora_ids=[req.lora_id for req in batch.reqs],
            rids=[req.rid for req in batch.reqs],
            # Compound (carry their own device tensors)
            sampling_info=batch.sampling_info,
            spec_info=batch.spec_info,
        )

        ret._maybe_init_non_generation_fields(batch)

        device = model_runner.device

        if envs.SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE.get():
            hashed = _hash_rids_to_tensor(
                rids=[req.rid for req in batch.reqs],
                device=device,
            )
            bootstrap_room_ids = _bootstrap_rooms_to_tensor(
                bootstrap_rooms=[req.bootstrap_room for req in batch.reqs],
                device=device,
            )
            batch.sampling_info.rids_int = hashed
            batch.sampling_info.bootstrap_room_ids_int = bootstrap_room_ids
            ret.rids_int = hashed
            ret.bootstrap_room_ids_int = bootstrap_room_ids

        if envs.SGLANG_KV_CANARY_ENABLE_VERIFY_TOKEN_ASSERT.get():
            ret.req_all_ids_flat, ret.req_all_ids_lens = compute_req_all_ids_info(
                batch.reqs
            )

        if batch.extend_input_logprob_token_ids is not None:
            ret.extend_input_logprob_token_ids_gpu = (
                batch.extend_input_logprob_token_ids.to(device, non_blocking=True)
            )

        num_tokens = len(batch.input_ids) if batch.input_ids is not None else 0
        if enable_num_token_non_padded():
            ret.num_token_non_padded = torch.tensor(num_tokens, dtype=torch.int32).to(
                device, non_blocking=True
            )
        ret.num_token_non_padded_cpu = num_tokens

        # For MLP sync
        if batch.global_num_tokens is not None:
            assert batch.global_num_tokens_for_logprob is not None

            # process global_num_tokens and global_num_tokens_for_logprob
            if batch.spec_info is not None:
                from sglang.srt.speculative.spec_info import (
                    spec_scale_global_num_tokens,
                )

                global_num_tokens, global_num_tokens_for_logprob = (
                    spec_scale_global_num_tokens(
                        batch.spec_info,
                        batch.global_num_tokens,
                        batch.global_num_tokens_for_logprob,
                    )
                )
            else:
                global_num_tokens = batch.global_num_tokens
                global_num_tokens_for_logprob = batch.global_num_tokens_for_logprob

            ret.original_global_num_tokens_cpu = batch.global_num_tokens
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
            return ret

        # Override the positions with diffusion LLM or spec_info
        if batch.dllm_config is not None:
            block_size = batch.dllm_config.block_size
            # Use int64 for AMD rotary embedding kernel compatibility
            positions_dtype = torch.int64 if is_hip() or _is_npu else torch.int32
            ret.positions = torch.tensor(
                [
                    i
                    for block_offset in (req.dllm_block_offset for req in batch.reqs)
                    for i in range(block_offset, block_offset + block_size)
                ],
                dtype=positions_dtype,
            ).to(device, non_blocking=True)
        elif (
            ret.spec_info is not None
            and getattr(ret.spec_info, "positions", None) is not None
        ):
            ret.positions = ret.spec_info.positions

        # Init position information
        if ret.forward_mode.is_decode() or ret.forward_mode.is_target_verify():
            if ret.positions is None:
                ret.positions = clamp_position(batch.seq_lens)
        else:
            if isinstance(extend_seq_lens, list):
                # Main path: H2D from host lists; populate *_cpu mirrors.
                assert isinstance(extend_prefix_lens, list)
                ret.extend_seq_lens = torch.tensor(
                    extend_seq_lens, dtype=torch.int32
                ).to(device, non_blocking=True)
                ret.extend_prefix_lens = torch.tensor(
                    extend_prefix_lens, dtype=torch.int32
                ).to(device, non_blocking=True)
                ret.extend_prefix_lens_cpu = extend_prefix_lens
                ret.extend_seq_lens_cpu = extend_seq_lens
            else:
                # gpu_only: device tensors handed in directly; leave *_cpu unset.
                assert isinstance(extend_seq_lens, torch.Tensor)
                ret.extend_seq_lens = extend_seq_lens
                ret.extend_prefix_lens = extend_prefix_lens
            ret.extend_num_tokens = batch.extend_num_tokens
            positions, ret.extend_start_loc = compute_position(
                model_runner.server_args.attention_backend,
                ret.extend_prefix_lens,
                ret.extend_seq_lens,
                ret.extend_num_tokens,
            )
            if ret.positions is None:
                ret.positions = positions
            ret.extend_logprob_start_lens_cpu = extend_logprob_start_lens

        if model_runner.ngram_embedding_manager.enabled:
            ret._init_ngram_embedding_info(batch, device)

        if model_runner.model_config.model_is_mrope:
            if (
                ret.spec_info is not None
                and getattr(ret.spec_info, "positions", None) is not None
            ):
                ret.compute_spec_mrope_positions(model_runner, batch)
            elif ret.forward_mode.is_draft_extend_v2():
                # Draft-extend tokens are uniform text continuation; reuse the
                # spec mrope path with the input-consistent `ret.positions` rather
                # than the per-request rebuild (which mis-sizes mm requests).
                ret.compute_spec_mrope_positions(
                    model_runner, batch, seq_positions=ret.positions
                )
            else:
                ret._compute_mrope_positions(model_runner, batch)

        # Init lora information
        if model_runner.server_args.enable_lora:
            # In the non-LoRA overlap loading case, we fetch LoRA adapters into the memory pool
            # as a batch, right before running the batch
            if not model_runner.server_args.enable_lora_overlap_loading:
                model_runner.lora_manager.fetch_new_loras(set(ret.lora_ids))

            model_runner.lora_manager.prepare_lora_batch(ret)

        if (
            getattr(model_runner, "dcp_size", 1) > 1
            and ret.out_cache_loc is not None
            and is_hip()
        ):
            ret.dcp_kv_mask = (
                ret.positions % model_runner.dcp_size == model_runner.dcp_rank
            )

        return ret

    def _maybe_init_non_generation_fields(self, batch: ScheduleBatch):
        """Derive non-generation (max_new_tokens==0) forward fields from reqs.

        token_type_ids gates on presence, not is_prefill_only: a missing
        tensor makes bert/roberta silently fall back to zeros.
        """
        if self.is_prefill_only:
            if batch.model_config.is_matryoshka and any(
                r.dimensions is not None for r in batch.reqs
            ):
                self.dimensions = [
                    r.dimensions if r.dimensions else batch.model_config.hidden_size
                    for r in batch.reqs
                ]

            self.return_pooled_hidden_states = any(
                r.return_pooled_hidden_states for r in batch.reqs
            )

            # --enable-mis: every request must carry delimiter indices (the score
            # endpoint always produces MIS-structured requests; consumers index
            # without None-checking).
            if get_server_args().enable_mis and any(
                r.multi_item_delimiter_indices is not None for r in batch.reqs
            ):
                assert all(
                    r.multi_item_delimiter_indices is not None for r in batch.reqs
                ), "MIS batch must have delimiter indices on every request"
                self.multi_item_delimiter_indices = [
                    torch.tensor(r.multi_item_delimiter_indices, dtype=torch.int64)
                    for r in batch.reqs
                ]

        token_type_ids = [
            r.token_type_ids for r in batch.reqs if r.token_type_ids is not None
        ]
        if token_type_ids:
            self.token_type_ids = torch.tensor(
                sum(token_type_ids, []),
                dtype=torch.int64,
                pin_memory=is_pin_memory_available(batch.device),
            ).to(batch.device, non_blocking=True)

    def adjust_num_token_non_padded_for_attn_tp(self, server_args) -> None:
        """Make num_token_non_padded local to this attention-TP rank."""
        from sglang.srt.utils.common import require_mlp_tp_gather

        dp_rank = get_parallel().attn_dp_rank
        assert self.global_num_tokens_cpu is not None

        if require_mlp_tp_gather(server_args):
            num_tokens_per_dp = self.global_num_tokens_cpu[dp_rank]
        else:
            num_tokens_per_dp = self.global_num_tokens_cpu[0]

        if num_tokens_per_dp == 0:
            # init_new creates both mirrors from the same host-side token count.
            # Keep the already-zero device scalar instead of launching scalar
            # elementwise kernels on an idle ROCm rank.
            assert self.num_token_non_padded_cpu == 0
            return

        self.num_token_non_padded = compute_local_num_token_non_padded(
            global_num_token_non_padded=self.num_token_non_padded,
            num_tokens_per_dp=num_tokens_per_dp,
        )

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

    def _init_ngram_embedding_info(self, batch: ScheduleBatch, device: torch.device):
        if self.forward_mode.is_decode():
            column_starts, req_lens = self.seq_lens - 1, 1
        else:
            column_starts, req_lens = self.extend_prefix_lens, self.extend_seq_lens
        self.ngram_embedding_info = NgramEmbeddingInfo.create(
            batch.ne_token_table,
            self.batch_size,
            device,
            column_starts=column_starts,
            req_lens=req_lens,
            skip_token_table_update=batch.ne_skip_token_table_update,
        )

    def compute_spec_mrope_positions(
        self, model_runner: ModelRunner, batch: ScheduleBatch, seq_positions=None
    ):
        # TODO support batched deltas
        batch_size = self.seq_lens.shape[0]
        device = model_runner.device
        mm_inputs = batch.multimodal_inputs

        # target_verify / draft_decode read spec_info.positions; draft_extend
        # passes its own positions (uniform num_draft_tokens per request).
        if seq_positions is None:
            seq_positions = batch.spec_info.positions
        seq_positions = seq_positions.view(batch_size, -1)
        # Split text-only and mixed batches here because SpecV2 text-only batches can avoid an extra D2H.
        if all(mm_input is None for mm_input in mm_inputs):
            mrope_delta_tensor = torch.zeros(
                (batch_size, 1), dtype=torch.int64, device=device
            )
        else:
            mrope_deltas = [
                (
                    torch.zeros(1, dtype=torch.int64)
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
    ) -> torch.Tensor:
        # Some generation models precompute decode positions for future tokens.
        # For example, GLM-Image needs 2D spatial MRoPE positions instead of
        # sequential delta-based positions.
        # This is needed for image generation models (e.g. GlmImage) where
        # decode tokens require 2D spatial MRoPE positions, not sequential.
        if (
            mm_input.mrope_positions is not None
            and mm_input.mrope_positions.shape[1] >= seq_len
        ):
            pos = mm_input.mrope_positions[:, seq_len - 1 : seq_len]
            return pos

        # doing below compute on cpu to avoid frequent small kernels
        if mm_input.mrope_position_delta_repeated_cache is None:
            mm_input.mrope_position_delta_repeated_cache = (
                (mm_input.mrope_position_delta - 1).flatten().unsqueeze(0).repeat(3, 1)
            )
        mrope_positions = mm_input.mrope_position_delta_repeated_cache + seq_len
        return mrope_positions

    def _compute_mrope_positions(self, model_runner: ModelRunner, batch: ScheduleBatch):
        # batch_size * [3 * seq_len]
        batch_size = self.seq_lens_cpu.shape[0]
        mrope_positions_list = [[]] * batch_size
        rl_on_policy_target = get_server_args().rl_on_policy_target
        for batch_idx in range(batch_size):
            mm_input = batch.multimodal_inputs[batch_idx]
            if self.forward_mode.is_decode():
                # 3 * N
                if mm_input is None or rl_on_policy_target is not None:
                    mrope_positions_list[batch_idx] = torch.full(
                        (3, 1),
                        self.seq_lens_cpu[batch_idx] - 1,
                        dtype=torch.int64,
                    )
                else:
                    mrope_positions = self._expand_mrope_from_input(
                        mm_input, self.seq_lens_cpu[batch_idx]
                    )
                    mrope_positions_list[batch_idx] = mrope_positions
            elif self.forward_mode.is_extend(include_draft_extend_v2=True):
                extend_seq_len, extend_prefix_len = (
                    batch.extend_lens[batch_idx],
                    batch.prefix_lens[batch_idx],
                )
                if mm_input is None or rl_on_policy_target is not None:
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
                            mm_input, self.seq_lens_cpu[batch_idx]
                        )
                mrope_positions_list[batch_idx] = mrope_positions

        self.mrope_positions = torch.cat(
            [pos for pos in mrope_positions_list],
            dim=1,
        ).to(dtype=torch.int64, device=model_runner.device, non_blocking=True)

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
        from sglang.srt.batch_overlap.two_batch_overlap import TboForwardBatchPreparer

        # Local imports: module-level CP helper imports here are circular (#27014).
        from sglang.srt.layers.cp.padding import get_cp_padding_align_size
        from sglang.srt.layers.cp.utils import enable_cp_v2

        assert self.global_num_tokens_cpu is not None
        assert self.global_num_tokens_for_logprob_cpu is not None

        self._original_batch_size = self.batch_size
        global_num_tokens = list(self.global_num_tokens_cpu)
        sync_group_size = len(global_num_tokens)
        attn_tp_size = get_parallel().attn_tp_size

        for i in range(sync_group_size):
            # make sure that the padded length is divisible by attn_tp_size because we may need reduce-scatter across attn_tp dim.
            # there is no reduce-scatter in LM logprob, so we do not need to adjust the padded length for logprob
            global_num_tokens[i] = ceil_align(global_num_tokens[i], attn_tp_size)

        # make sure that each rank has the same number of tokens to do collective communication.
        # Zigzag (in-seq-split) CP pads to 2 * attn_cp_size for load balance; other CP modes
        # pad to attn_cp_size; CP off pads nothing (extra padding breaks EAGLE/MTP draft
        # prefill with NaN draft logits, see #23269).
        # FIXME(kpham-sgl): revisit so draft prefill-extend tolerates padded dummy tokens.
        if not enable_cp_v2():
            cp_align_size = get_cp_padding_align_size()
            for i in range(sync_group_size):
                global_num_tokens[i] = ceil_align(global_num_tokens[i], cp_align_size)

        dp_padding_mode = DpPaddingMode.get_dp_padding_mode(
            self.is_extend_in_batch, global_num_tokens
        )
        if _elastic_should_preserve_local_token_counts(
            model_runner=model_runner,
            dp_padding_mode=dp_padding_mode,
            global_num_tokens=global_num_tokens,
        ):
            # Joined ranks require real token counts instead of MAX_LEN padding.
            dp_padding_mode = DpPaddingMode.SUM_LEN
        # Prefill breakable CUDA graph requires every DP rank to run the SAME
        # captured shape. Under SUM_LEN each rank pads to its own local token
        # count and can select a different capture bucket, so the in-graph DP
        # collectives (all_gather / reduce_scatter) mismatch across ranks and
        # corrupt the output. Force MAX_LEN so every rank pads to the global
        # max and picks the same bucket (mirrors the decode cuda graph
        # contract, which always runs MAX_LEN).
        #
        # Only force MAX_LEN when the batch fits a captured breakable prefill
        # graph; larger prefills fall back to eager and keep the
        # memory-efficient SUM_LEN. global_num_tokens is identical across ranks
        # (all-gathered), so the decision is consistent cluster-wide.
        prefill_cg = model_runner.server_args.cuda_graph_config.prefill
        if (
            self.can_run_dp_breakable_cuda_graph
            and self.is_extend_in_batch
            and prefill_cg.bs
            and max(global_num_tokens) <= max(prefill_cg.bs)
        ):
            dp_padding_mode = DpPaddingMode.MAX_LEN
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
            num_tokens = global_num_tokens[get_parallel().attn_dp_rank]
        else:
            num_tokens = global_num_tokens[0]

        self.global_dp_buffer_len = buffer_len
        set_dp_buffer_len(
            buffer_len,
            num_tokens,
            dp_padding_mode.is_max_len(),
            global_num_tokens,
        )
        set_is_extend_in_batch(self.is_extend_in_batch)

        bs = self.batch_size

        if (
            self.forward_mode.is_decode()
            or self.forward_mode.is_target_verify()
            or self.forward_mode.is_draft_extend_v2()
            or self.forward_mode.is_idle()
        ):
            # Mamba-hybrid families need the fabricated-row idle conversion
            # below; this includes their MTP draft workers, whose mamba-less
            # "*E" pattern makes mambaish_config return None.
            hybrid_ssm = mambaish_config(model_runner.model_config) is not None or (
                model_runner.is_draft_worker
                and getattr(
                    model_runner.model_config.hf_config,
                    "mtp_hybrid_override_pattern",
                    None,
                )
                is not None
            )
            if (
                hybrid_ssm
                and self.spec_info is not None
                and not self.spec_info.is_draft_input()
            ):
                if self.forward_mode.is_idle():
                    self._original_forward_mode = self.forward_mode
                    self.forward_mode = ForwardMode.TARGET_VERIFY
                # Invert the spec_scale_global_num_tokens scaling.
                bs = self.batch_size = num_tokens // self.spec_info.num_tokens_per_req
            elif self.is_extend_in_batch and dp_padding_mode.is_max_len():
                self._original_forward_mode = self.forward_mode
                self.forward_mode = ForwardMode.EXTEND
                # Fabricate a single dummy request covering num_tokens for an
                # empty (idle) rank. Hybrid-SSM families always take this path;
                # non-hybrid ranks reach it once MAX_LEN is forced for the
                # prefill breakable CUDA graph (idle + prefill), which needs
                # every DP rank to run the same captured shape. The `else`
                # branch handles decode rows padded to a 1-token extend.
                if hybrid_ssm or self.seq_lens.shape[0] == 0:
                    dev = self.seq_lens.device
                    assert (
                        self.seq_lens.shape[0] == 0
                    ), "extend-idle conversion expects an empty rank"
                    self.extend_num_tokens = num_tokens
                    self.extend_seq_lens = torch.tensor(
                        [num_tokens], dtype=torch.int32, device=dev
                    )
                    self.extend_prefix_lens = torch.zeros(
                        1, dtype=self.seq_lens.dtype, device=dev
                    )
                    self.extend_start_loc = torch.zeros(
                        1, dtype=torch.int32, device=dev
                    )
                    self.seq_lens = torch.tensor(
                        [num_tokens], dtype=self.seq_lens.dtype, device=dev
                    )
                    # orig_seq_lens is not padded by _pad_inputs_to_size, so
                    # fabricate it to match the dummy request (the breakable
                    # prefill CUDA graph runner reads it).
                    self.orig_seq_lens = torch.tensor(
                        [num_tokens], dtype=self.orig_seq_lens.dtype, device=dev
                    )
                    self.seq_lens_sum = int(num_tokens)
                    if self.seq_lens_cpu is not None:
                        self.seq_lens_cpu = torch.tensor(
                            [num_tokens], dtype=self.seq_lens.dtype
                        )
                    self.extend_prefix_lens_cpu = [0]
                    self.extend_seq_lens_cpu = [int(num_tokens)]
                    self.extend_logprob_start_lens_cpu = [0]
                    bs = self.batch_size = 1
                    # Count the dummy tokens as real, else MoE topk/all-to-all
                    # treats this rank as empty and starves later layers.
                    # (num_token_non_padded is None unless moe_ep_size > 1.)
                    if self.num_token_non_padded is not None:
                        self.num_token_non_padded.fill_(num_tokens)
                    self.num_token_non_padded_cpu = num_tokens
                else:
                    self.extend_num_tokens = bs
                    self.extend_seq_lens = torch.full_like(self.seq_lens, 1)
                    self.extend_prefix_lens = self.seq_lens - 1
                    self.extend_start_loc = torch.arange(
                        bs, dtype=torch.int32, device=self.seq_lens.device
                    )
                    self.extend_prefix_lens_cpu = self.extend_prefix_lens.cpu().tolist()
                    self.extend_seq_lens_cpu = self.extend_seq_lens.cpu().tolist()
                    self.extend_logprob_start_lens_cpu = self.extend_prefix_lens_cpu
            else:
                if self.spec_info is not None:
                    # Invert the spec_scale_global_num_tokens scaling.
                    bs = self.batch_size = (
                        num_tokens // self.spec_info.num_tokens_per_req
                    )
                else:
                    bs = self.batch_size = num_tokens
        elif self.forward_mode.is_extend():
            self.extend_num_tokens = num_tokens

        # padding
        self._pad_inputs_to_size(model_runner, num_tokens, bs)
        self.global_num_tokens_cpu = global_num_tokens
        global_num_tokens_pinned = torch.tensor(global_num_tokens, pin_memory=True)
        self.global_num_tokens_gpu.copy_(global_num_tokens_pinned, non_blocking=True)

        TboForwardBatchPreparer.prepare(
            batch=self, is_draft_worker=model_runner.is_draft_worker
        )
        # TODO: The following is added to make sure sub-batch input_ids are padded
        # to the multiple of attn_tp_size. It can likely be removed after this
        # function is refactored and merged into the Scheduler.
        if self.tbo_children:
            for child in self.tbo_children:
                child._pad_inputs_to_size(
                    model_runner, child.tbo_padded_len, child.batch_size
                )

    def _pad_inputs_to_size(self, model_runner: ModelRunner, num_tokens, bs):
        # padding
        self.input_ids = self._pad_tensor_to_size(self.input_ids, num_tokens)
        self.req_pool_indices = self._pad_tensor_to_size(self.req_pool_indices, bs)
        if self.lora_ids is not None:
            self.lora_ids.extend((bs - len(self.lora_ids)) * [None])

        seq_len_fill_value = (
            model_runner.attn_backend.get_cuda_graph_seq_len_fill_value()
        )
        # Keep gpu_only batches sync-free: leave seq_lens_sum None and let the
        # attention backend over-allocate from an upper bound (see #26738).
        if self.seq_lens_sum is not None:
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
        if self.mamba_track_indices is not None:
            self.mamba_track_indices = self._pad_tensor_to_size(
                self.mamba_track_indices, bs
            )
        if self.mamba_track_mask is not None:
            self.mamba_track_mask = self._pad_tensor_to_size(self.mamba_track_mask, bs)
        if self.mamba_track_seqlens is not None:
            self.mamba_track_seqlens = self._pad_tensor_to_size(
                self.mamba_track_seqlens, bs
            )

        if self.mrope_positions is not None:
            self.mrope_positions = torch.cat(
                [
                    self.mrope_positions,
                    self.mrope_positions.new_zeros(
                        3, num_tokens - self.mrope_positions.shape[1]
                    ),
                ],
                dim=1,
            )

        # TODO: check if we need to pad other tensors
        if self.extend_seq_lens is not None:
            self.extend_seq_lens = self._pad_tensor_to_size(self.extend_seq_lens, bs)

        if self.rids_int is not None:
            self.rids_int = self._pad_tensor_to_size(self.rids_int, bs)
            if self.sampling_info is not None:
                self.sampling_info.rids_int = self.rids_int
        if self.bootstrap_room_ids_int is not None:
            self.bootstrap_room_ids_int = self._pad_tensor_to_size(
                self.bootstrap_room_ids_int, bs, value=-1
            )
            if self.sampling_info is not None:
                self.sampling_info.bootstrap_room_ids_int = self.bootstrap_room_ids_int

        if self.spec_info is not None and self.spec_info.is_draft_input():
            spec_info = self.spec_info
            self.output_cache_loc_backup = self.out_cache_loc
            self.hidden_states_backup = spec_info.hidden_states
            # spec_info is EagleDraftInput | EagleDraftExtendInput; each carries
            # a disjoint subset of the fields below, so getattr-guard each one.
            if getattr(spec_info, "topk_p", None) is not None:
                spec_info.topk_p = self._pad_tensor_to_size(spec_info.topk_p, bs)
            if getattr(spec_info, "topk_index", None) is not None:
                spec_info.topk_index = self._pad_tensor_to_size(
                    spec_info.topk_index, bs
                )
            if getattr(spec_info, "draft_probs", None) is not None:
                spec_info.draft_probs = self._pad_tensor_to_size(
                    spec_info.draft_probs, bs
                )
            if getattr(spec_info, "num_correct_drafts", None) is not None:
                spec_info.num_correct_drafts = self._pad_tensor_to_size(
                    spec_info.num_correct_drafts, bs
                )
                spec_info.num_accept_tokens = self._pad_tensor_to_size(
                    spec_info.num_accept_tokens, bs
                )
            spec_info.hidden_states = self._pad_tensor_to_size(
                spec_info.hidden_states, num_tokens
            )

    def prepare_attn_tp_scatter_input(self, model_runner: ModelRunner):
        from sglang.srt.layers.communicator import get_attn_tp_context

        attn_tp_context = get_attn_tp_context()
        input_scattered = attn_tp_context.use_input_scattered(self)
        if not input_scattered:
            return
        assert self.forward_mode.is_extend()
        tokens = self.input_ids.shape[0]
        rank_size = get_parallel().tp_size
        tokens_padded = (tokens + rank_size - 1) // rank_size * rank_size
        self._pad_inputs_to_size(model_runner, tokens_padded, self.batch_size)

    def post_forward_mlp_sync_batch(self, logits_output: LogitsProcessorOutput):
        if self._original_forward_mode is not None:
            self.forward_mode = self._original_forward_mode
        if self._original_batch_size is not None:
            self.batch_size = self._original_batch_size
        bs = self.batch_size

        if self.spec_info is not None:
            if self.forward_mode.is_decode():  # draft
                num_tokens = self.hidden_states_backup.shape[0]
                self.positions = self.positions[:num_tokens]
                self.seq_lens = self.seq_lens[:bs]
                self.req_pool_indices = self.req_pool_indices[:bs]
                if self.seq_lens_cpu is not None:
                    self.seq_lens_cpu = self.seq_lens_cpu[:bs]
                if logits_output.next_token_logits is not None:
                    logits_output.next_token_logits = logits_output.next_token_logits[
                        :num_tokens
                    ]
                logits_output.hidden_states = logits_output.hidden_states[:num_tokens]
            elif self.forward_mode.is_target_verify():  # verify
                num_tokens = bs * self.spec_info.draft_token_num
                if logits_output.next_token_logits is not None:
                    logits_output.next_token_logits = logits_output.next_token_logits[
                        :num_tokens
                    ]
                logits_output.hidden_states = logits_output.hidden_states[:num_tokens]
            elif self.forward_mode.is_draft_extend_v2():  # draft extend_v2
                bs = bs * self.spec_info.num_tokens_per_req
                if logits_output.next_token_logits is not None:
                    logits_output.next_token_logits = logits_output.next_token_logits[
                        :bs
                    ]
                logits_output.hidden_states = logits_output.hidden_states[:bs]
            elif self.forward_mode.is_extend() or self.forward_mode.is_idle():
                if logits_output.next_token_logits is not None:
                    logits_output.next_token_logits = logits_output.next_token_logits[
                        :bs
                    ]
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

    @property
    def can_run_tbo(self):
        return self.tbo_split_seq_index is not None


def enable_num_token_non_padded():
    return get_parallel().moe_ep_size > 1


def build_inner_fb_view(
    forward_batch: ForwardBatch,
    *,
    bs: int,
    forward_mode: ForwardMode,
    encoder_lens: Optional[torch.Tensor] = None,
):
    """Build a ForwardBatch-like view for MultiStep draft wrapper dispatch.

    MultiStep draft wrappers (FlashInferMultiStepDraftBackend,
    AiterMultiStepDraftBackend, TritonMultiStepDraftBackend, etc.) need
    to dispatch to per-step inner backends'
    :py:meth:`AttentionBackend.init_forward_metadata_out_graph` with an
    overridden ``forward_mode`` (typically pinned to ``DECODE``) and
    sometimes overridden ``encoder_lens``. The result is a thin
    namespace mirroring just the fields backend init reads, avoiding
    the cost of allocating a real ``ForwardBatch``.

    ``actual_forward_mode`` carries the original runtime
    ``forward_batch.forward_mode`` (e.g., spec-decode draft) so backends
    that check it for IDLE substitution (DSV4) see the unaltered value.
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        batch_size=bs,
        forward_mode=forward_mode,
        actual_forward_mode=forward_batch.forward_mode,
        input_ids=getattr(forward_batch, "input_ids", None),
        positions=getattr(forward_batch, "positions", None),
        req_pool_indices=forward_batch.req_pool_indices,
        seq_lens=forward_batch.seq_lens,
        seq_lens_sum=forward_batch.seq_lens_sum,
        seq_lens_cpu=forward_batch.seq_lens_cpu,
        encoder_lens=encoder_lens,
        out_cache_loc=getattr(forward_batch, "out_cache_loc", None),
        out_cache_loc_dsv4=getattr(forward_batch, "out_cache_loc_dsv4", None),
        spec_info=forward_batch.spec_info,
    )


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


def _clamp_position_native(seq_lens):
    return torch.clamp((seq_lens - 1), min=0).to(torch.int64)


if is_cuda() or is_hip():
    from sglang.jit_kernel.clamp_position import clamp_position_cuda

    clamp_position = clamp_position_cuda
else:
    clamp_position = _clamp_position_native


def _hash_rids_to_tensor(*, rids: List[str], device: torch.device) -> torch.Tensor:
    values: List[int] = [_stable_hash_str_to_i64(rid) for rid in rids]
    return torch.tensor(values, dtype=torch.int64, device=device)


def _bootstrap_rooms_to_tensor(
    *, bootstrap_rooms: List[Optional[int]], device: torch.device
) -> torch.Tensor:
    values: List[int] = [room if room is not None else -1 for room in bootstrap_rooms]
    return torch.tensor(values, dtype=torch.int64, device=device)


def _stable_hash_str_to_i64(rid: str) -> int:
    digest = hashlib.blake2b(rid.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=True)
