# Copyright 2023-2026 SGLang Team
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

"""Zigzag context parallel strategy shell.

For ``cp_size = 4``, each sequence is split into ``2 * cp_size`` blocks. Each
rank owns one early block and one late block:

    cp0: block0, block7
    cp1: block1, block6
    cp2: block2, block5
    cp3: block3, block4

After all-gather, the blocks are reranged back to their original order:

    block0 | block7 | block1 | block6 | block2 | block5 | block3 | block4
      -> block0 | block1 | block2 | block3 | block4 | block5 | block6 | block7
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from itertools import accumulate
from typing import Any, List, Optional

import torch
import torch.nn.functional as F

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.cp.base import (
    BaseContextParallelMetadata,
    ContextParallelStrategy,
    ContextParallelStrategyKind,
    CPAttentionBackendKind,
)
from sglang.srt.layers.cp.padding import pad_local_rows
from sglang.srt.layers.dp_attention import (
    is_allocation_symmetric,
)
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sglang.srt.runtime_context import get_device, get_parallel


@dataclass
class _RankMajorAllGather:
    send_buffer: torch.Tensor
    output_buffer: torch.Tensor
    per_rank_token: tuple[int, ...]
    max_len: int


@dataclass
class _PendingMLAKVMaterialization:
    stream: torch.cuda.Stream
    group: Any
    inputs: tuple[torch.Tensor, ...]
    gather: _RankMajorAllGather
    reverse_split_len: tuple[int, ...]
    cp_reverse_index: tuple[int, ...]
    kv_lora_rank: int
    producer_event: Optional[torch.cuda.Event] = None
    launched: bool = False


@dataclass
class ZigzagContextParallelMetadata(BaseContextParallelMetadata):
    # Layout lists have length bs * cp_segment_num (= bs * 2 * cp_size).
    split_list: Optional[List[int]] = None
    zigzag_index: Optional[List[int]] = None
    cp_reverse_index: Optional[List[int]] = None
    reverse_split_len: Optional[List[int]] = None

    # Per-rank aggregate lists have length cp_size.
    per_rank_actual_token: Optional[List[int]] = None
    max_rank_len: Optional[List[int]] = None
    per_rank_logical_token: Optional[List[int]] = None

    # Per-sequence FlashAttention tensors (shape [bs] or [bs + 1]).
    kv_len_prev_tensor: Optional[Any] = None
    kv_len_next_tensor: Optional[Any] = None
    cu_seqlens_kv_prev_tensor: Optional[Any] = None
    cu_seqlens_kv_next_tensor: Optional[Any] = None
    actual_seq_q_prev_tensor: Optional[Any] = None
    actual_seq_q_next_tensor: Optional[Any] = None
    cu_seqlens_q_prev_tensor: Optional[Any] = None
    cu_seqlens_q_next_tensor: Optional[Any] = None

    # Combined prev-then-next TRT-LLM geometry (shape [2 * bs] or [2 * bs + 1]).
    actual_seq_q_combined_tensor: Optional[Any] = None
    kv_len_combined_tensor: Optional[Any] = None
    cu_seqlens_q_combined_tensor: Optional[Any] = None
    cu_seqlens_kv_combined_tensor: Optional[Any] = None

    # Scalars derived from the per-sequence lists above.
    total_q_prev_tokens: int = 0
    total_q_next_tokens: int = 0
    max_seqlen_q_prev: int = 0
    max_seqlen_q_next: int = 0
    max_seqlen_q_combined: int = 0

    # Per-sequence CPU lists, useful for indexers and diagnostics.
    kv_len_prev_list: Optional[List[int]] = None
    kv_len_next_list: Optional[List[int]] = None
    actual_seq_q_prev_list: Optional[List[int]] = None
    actual_seq_q_next_list: Optional[List[int]] = None

    # A launch belongs to this forward and layer, never to the next batch.
    pending_mla_kv_materializations: dict[int, _PendingMLAKVMaterialization] = field(
        default_factory=dict, repr=False
    )


ContextParallelMetadata = ZigzagContextParallelMetadata


class ZigzagCPStrategy(ContextParallelStrategy):
    name = "zigzag"
    kind = ContextParallelStrategyKind.ZIGZAG

    def can_apply(self, num_tokens: int, forward_batch) -> bool:
        if self.cp_size <= 1 or num_tokens < self.cp_size * 2:
            return False
        forward_mode = getattr(forward_batch, "forward_mode", None)
        if forward_mode is not None and not forward_mode.is_context_parallel_extend():
            return False

        extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if extend_lens is None:
            return True
        return all(int(length) >= self.cp_size * 2 for length in extend_lens)

    def build_metadata(
        self,
        num_tokens: int,
        seqs_len: Optional[List[int]],
        extend_seqs_len: Optional[List[int]] = None,
    ) -> ZigzagContextParallelMetadata:
        if extend_seqs_len is None:
            extend_seqs_len = seqs_len or [num_tokens]
        extend_seqs_len = [int(x) for x in extend_seqs_len]

        pad_len = int(num_tokens) - sum(extend_seqs_len)
        if pad_len > 0:
            extend_seqs_len[-1] += pad_len
            if seqs_len is not None and len(seqs_len) == len(extend_seqs_len):
                seqs_len = list(seqs_len)
                seqs_len[-1] += pad_len

        bs = len(extend_seqs_len)
        cp_segment_num = self.cp_size * 2
        if seqs_len is not None and len(seqs_len) == bs:
            prefix_offsets = [
                max(int(seqs_len[i]) - extend_seqs_len[i], 0) for i in range(bs)
            ]
        else:
            prefix_offsets = [0] * bs

        # TODO: move these per-request layout/index computations to a Triton
        # kernel if Python-side metadata construction becomes a bottleneck.
        per_seq_block_sizes: List[List[int]] = []
        split_list: List[int] = []
        for length in extend_seqs_len:
            base = length // cp_segment_num
            rem = length % cp_segment_num
            block_sizes = [
                base + 1 if block_id < rem else base
                for block_id in range(cp_segment_num)
            ]
            per_seq_block_sizes.append(block_sizes)
            split_list.extend(block_sizes)

        per_rank_actual_token = []
        for rank in range(self.cp_size):
            per_rank_actual_token.append(
                sum(
                    block_sizes[rank] + block_sizes[cp_segment_num - 1 - rank]
                    for block_sizes in per_seq_block_sizes
                )
            )
        max_rank_len = [max(per_rank_actual_token)] * self.cp_size

        cp_rank = self.cp_rank
        zigzag_index = list(
            range(cp_rank, cp_rank + bs * cp_segment_num, cp_segment_num)
        ) + list(
            range(
                cp_segment_num - cp_rank - 1,
                bs * cp_segment_num,
                cp_segment_num,
            )
        )

        cp_reverse_index: List[int] = []
        for batch_id in range(bs):
            cp_reverse_index.extend(
                list(range(batch_id, cp_segment_num * bs, 2 * bs))
                + list(
                    range(
                        (cp_segment_num - 1) * bs + batch_id,
                        0,
                        -2 * bs,
                    )
                )
            )

        reverse_split_len: List[int] = []
        for rank in range(self.cp_size):
            for batch_id in range(bs):
                reverse_split_len.append(per_seq_block_sizes[batch_id][rank])
            for batch_id in range(bs):
                reverse_split_len.append(
                    per_seq_block_sizes[batch_id][cp_segment_num - 1 - rank]
                )

        kv_len_prev_list: List[int] = []
        kv_len_next_list: List[int] = []
        actual_seq_q_prev_list: List[int] = []
        actual_seq_q_next_list: List[int] = []
        for batch_id, block_sizes in enumerate(per_seq_block_sizes):
            kv_len_prev_list.append(
                prefix_offsets[batch_id] + sum(block_sizes[: cp_rank + 1])
            )
            kv_len_next_list.append(
                prefix_offsets[batch_id] + sum(block_sizes[: cp_segment_num - cp_rank])
            )
            actual_seq_q_prev_list.append(block_sizes[cp_rank])
            actual_seq_q_next_list.append(block_sizes[cp_segment_num - cp_rank - 1])

        try:
            device = torch.device(get_device().device)
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cu_prev = [0] + list(accumulate(actual_seq_q_prev_list))
        cu_next = [0] + list(accumulate(actual_seq_q_next_list))
        cu_kv_prev = [0] + list(accumulate(kv_len_prev_list))
        cu_kv_next = [0] + list(accumulate(kv_len_next_list))
        actual_seq_q_combined_list = actual_seq_q_prev_list + actual_seq_q_next_list
        kv_len_combined_list = kv_len_prev_list + kv_len_next_list
        cu_q_combined = [0] + list(accumulate(actual_seq_q_combined_list))
        cu_kv_combined = [0] + list(accumulate(kv_len_combined_list))

        total_seq_lens = sum(extend_seqs_len)
        assert len(split_list) == bs * cp_segment_num
        assert sum(split_list) == total_seq_lens
        assert len(zigzag_index) == 2 * bs
        assert len(cp_reverse_index) == bs * cp_segment_num
        assert sorted(cp_reverse_index) == list(range(bs * cp_segment_num))
        assert sum(per_rank_actual_token) == total_seq_lens

        return ZigzagContextParallelMetadata(
            split_list=split_list,
            zigzag_index=zigzag_index,
            cp_reverse_index=cp_reverse_index,
            reverse_split_len=reverse_split_len,
            per_rank_actual_token=per_rank_actual_token,
            max_rank_len=max_rank_len,
            kv_len_prev_tensor=torch.tensor(
                kv_len_prev_list, device=device, dtype=torch.int32
            ),
            kv_len_next_tensor=torch.tensor(
                kv_len_next_list, device=device, dtype=torch.int32
            ),
            cu_seqlens_kv_prev_tensor=torch.tensor(
                cu_kv_prev, device=device, dtype=torch.int32
            ),
            cu_seqlens_kv_next_tensor=torch.tensor(
                cu_kv_next, device=device, dtype=torch.int32
            ),
            actual_seq_q_prev_tensor=torch.tensor(
                actual_seq_q_prev_list, device=device, dtype=torch.int32
            ),
            actual_seq_q_next_tensor=torch.tensor(
                actual_seq_q_next_list, device=device, dtype=torch.int32
            ),
            cu_seqlens_q_prev_tensor=torch.tensor(
                cu_prev, device=device, dtype=torch.int32
            ),
            cu_seqlens_q_next_tensor=torch.tensor(
                cu_next, device=device, dtype=torch.int32
            ),
            actual_seq_q_combined_tensor=torch.tensor(
                actual_seq_q_combined_list, device=device, dtype=torch.int32
            ),
            kv_len_combined_tensor=torch.tensor(
                kv_len_combined_list, device=device, dtype=torch.int32
            ),
            cu_seqlens_q_combined_tensor=torch.tensor(
                cu_q_combined, device=device, dtype=torch.int32
            ),
            cu_seqlens_kv_combined_tensor=torch.tensor(
                cu_kv_combined, device=device, dtype=torch.int32
            ),
            total_q_prev_tokens=cu_prev[-1],
            total_q_next_tokens=cu_next[-1],
            max_seqlen_q_prev=(
                max(actual_seq_q_prev_list) if actual_seq_q_prev_list else 0
            ),
            max_seqlen_q_next=(
                max(actual_seq_q_next_list) if actual_seq_q_next_list else 0
            ),
            max_seqlen_q_combined=(
                max(actual_seq_q_combined_list) if actual_seq_q_combined_list else 0
            ),
            kv_len_prev_list=kv_len_prev_list,
            kv_len_next_list=kv_len_next_list,
            actual_seq_q_prev_list=actual_seq_q_prev_list,
            actual_seq_q_next_list=actual_seq_q_next_list,
            total_seq_lens=total_seq_lens,
            bs=bs,
        )

    def shard_hidden_states(self, x: Any, forward_batch) -> Any:
        metadata = forward_batch.attn_cp_metadata
        x = x[: metadata.total_seq_lens]
        chunks = torch.split(x, metadata.split_list, dim=0)
        local_x = torch.cat([chunks[i] for i in metadata.zigzag_index], dim=0)
        return pad_local_rows(local_x, metadata, dim=0)

    def shard_position_ids(self, positions: Any, forward_batch) -> Any:
        metadata = forward_batch.attn_cp_metadata
        positions = positions[..., : metadata.total_seq_lens]
        chunks = torch.split(positions, metadata.split_list, dim=-1)
        local_positions = torch.cat([chunks[i] for i in metadata.zigzag_index], dim=-1)
        return pad_local_rows(local_positions, metadata, dim=-1)

    def gather_hidden_states(
        self, x: Any, forward_batch, stream: Optional[Any] = None
    ) -> Any:
        gathered = self._all_gather_reorganized(x, forward_batch)
        chunks = torch.split(
            gathered, forward_batch.attn_cp_metadata.reverse_split_len, dim=0
        )
        return torch.cat(
            [chunks[i] for i in forward_batch.attn_cp_metadata.cp_reverse_index], dim=0
        )

    def gather_kv_cache(
        self, x: Any, forward_batch, stream: Optional[Any] = None
    ) -> Any:
        gathered = self._all_gather_reorganized(x, forward_batch)
        chunks = torch.split(
            gathered, forward_batch.attn_cp_metadata.reverse_split_len, dim=0
        )
        return torch.cat(
            [chunks[i] for i in forward_batch.attn_cp_metadata.cp_reverse_index], dim=0
        )

    def get_supported_attention_backend(self):
        return [
            CPAttentionBackendKind.FLASH_ATTENTION,
            CPAttentionBackendKind.TRTLLM_MHA,
        ]

    def run_attention(
        self,
        q: Any,
        forward_batch,
        device: Any,
        attn_fn,
        attention_backend: CPAttentionBackendKind = CPAttentionBackendKind.FLASH_ATTENTION,
        single_launch: bool = False,
    ) -> Any:
        assert (
            attention_backend in self.get_supported_attention_backend()
        ), f"{self.name} CP does not support {attention_backend=}"

        meta = forward_batch.attn_cp_metadata
        q_prev = q[: meta.total_q_prev_tokens]
        logical_tokens = meta.total_q_prev_tokens + meta.total_q_next_tokens
        q_next = q[meta.total_q_prev_tokens : logical_tokens]

        prev_kwargs = {}
        next_kwargs = {}
        if (
            single_launch
            and attention_backend == CPAttentionBackendKind.FLASH_ATTENTION
        ):
            # One varlen launch over [q_prev | q_next]: the combined metadata
            # lists every request's prev block first, then every next block,
            # each with its own kv length; the closure maps `combined=True`
            # to per-sequence K base offsets (request-major, repeated).
            result = attn_fn(
                q[:logical_tokens],
                meta.cu_seqlens_q_combined_tensor,
                meta.kv_len_combined_tensor,
                meta.max_seqlen_q_combined,
                combined=True,
            )
        elif attention_backend == CPAttentionBackendKind.TRTLLM_MHA:
            result = attn_fn(
                q[:logical_tokens],
                meta.cu_seqlens_q_combined_tensor,
                meta.kv_len_combined_tensor,
                meta.max_seqlen_q_combined,
                cu_seqlens_kv=meta.cu_seqlens_kv_combined_tensor,
                use_zigzag_page_table=True,
            )
        else:
            result_prev = attn_fn(
                q_prev,
                meta.cu_seqlens_q_prev_tensor,
                meta.kv_len_prev_tensor,
                meta.max_seqlen_q_prev,
                **prev_kwargs,
            )
            result_next = attn_fn(
                q_next,
                meta.cu_seqlens_q_next_tensor,
                meta.kv_len_next_tensor,
                meta.max_seqlen_q_next,
                **next_kwargs,
            )
            result = torch.cat([result_prev, result_next], dim=0)

        pad_size = q.shape[0] - logical_tokens
        assert pad_size >= 0
        if pad_size > 0:
            result = torch.cat(
                [result, result.new_zeros(pad_size, *result.shape[1:])], dim=0
            )
        return result

    def materialize_full_kv(
        self,
        forward_batch,
        layer: Any = None,
        k: Any = None,
        v: Any = None,
        swa_loc: Optional[Any] = None,
    ) -> Any:
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        if swa_loc is not None:
            swa_loc = swa_loc[: cache_loc.shape[0]]
        k_dim = k.shape[-1]
        v_dim = v.shape[-1]
        kv_cache = torch.cat([k, v], dim=-1).contiguous()
        key_cache_full, value_cache_full = self.gather_kv_cache(
            kv_cache, forward_batch
        ).split([k_dim, v_dim], dim=-1)
        key_cache_full = key_cache_full.contiguous()
        value_cache_full = value_cache_full.contiguous()
        get_token_to_kv_pool().set_kv_buffer(
            layer,
            KVWriteLoc(cache_loc, swa_loc),
            key_cache_full,
            value_cache_full,
            layer.k_scale,
            layer.v_scale,
        )

    def start_mla_kv_materialization(
        self,
        forward_batch,
        layer: Any,
        k_nope: torch.Tensor,
        k_rope: torch.Tensor,
        *,
        producer_event: Optional[torch.cuda.Event] = None,
        prepare_only: bool = False,
    ) -> bool:
        """Prepare KV transfer buffers and optionally launch the collective."""
        if not k_nope.is_cuda or torch.cuda.is_current_stream_capturing():
            return False
        group = get_parallel().attn_cp_group
        pynccl = group.pynccl_comm
        if pynccl is None or not pynccl.available:
            return False
        pending = forward_batch.attn_cp_metadata.pending_mla_kv_materializations
        assert layer.layer_id not in pending, "MLA KV materialization already pending"
        stream = getattr(self, "_mla_kv_stream", None)
        if stream is None:
            stream = self._mla_kv_stream = torch.cuda.Stream(device=k_nope.device)
        current_stream = torch.cuda.current_stream()
        if producer_event is None:
            stream.wait_stream(current_stream)
        else:
            # Q may already be queued on current_stream. Only the earlier
            # KV producer must precede this stream's packing and transfer.
            stream.wait_event(producer_event)
        inputs = (k_nope, k_rope, forward_batch.out_cache_loc)
        # Keep producer allocations alive on the communication stream, including
        # on exceptions before the backend can consume the completion event.
        for tensor in inputs:
            tensor.record_stream(stream)
        # Keep partially prepared buffers alive if packing raises after
        # enqueueing work. The exception handler joins before these locals die.
        prepared = None
        pending_kv = None
        try:
            with torch.cuda.stream(stream):
                latent = torch.cat([k_nope, k_rope], dim=-1).contiguous()
                prepared = self._prepare_all_gather_rank_major(latent, forward_batch)
                meta = forward_batch.attn_cp_metadata
                pending_kv = _PendingMLAKVMaterialization(
                    stream=stream,
                    group=group,
                    inputs=inputs,
                    gather=prepared,
                    reverse_split_len=tuple(meta.reverse_split_len),
                    cp_reverse_index=tuple(meta.cp_reverse_index),
                    kv_lora_rank=k_nope.shape[-1],
                    producer_event=producer_event,
                )
        except BaseException:
            current_stream.wait_stream(stream)
            raise
        pending[layer.layer_id] = pending_kv
        if not prepare_only:
            return self.launch_mla_kv_materialization(forward_batch, layer)
        return True

    def launch_mla_kv_materialization(self, forward_batch, layer: Any) -> bool:
        """Submit only NCCL after independent Q kernels are already queued."""
        pending_by_layer = (
            forward_batch.attn_cp_metadata.pending_mla_kv_materializations
        )
        pending = pending_by_layer.get(layer.layer_id)
        if pending is None:
            return False
        if pending.launched:
            return True
        try:
            with (
                torch.cuda.stream(pending.stream),
                pending.group.pynccl_comm.change_state(enable=True),
            ):
                pending.group.all_gather_into_tensor(
                    pending.gather.output_buffer, pending.gather.send_buffer
                )
            pending.launched = True
        except BaseException:
            pending_by_layer.pop(layer.layer_id, None)
            # Retain all staging allocations until any partially submitted
            # collective is ordered before subsequent work on the caller.
            torch.cuda.current_stream().wait_stream(pending.stream)
            raise
        return True

    def finish_mla_kv_materialization(self, forward_batch, layer: Any) -> bool:
        """Queue gather consumers after Q was submitted, then join cache readiness."""
        pending = forward_batch.attn_cp_metadata.pending_mla_kv_materializations.pop(
            layer.layer_id, None
        )
        if pending is None:
            return False
        current_stream = torch.cuda.current_stream()
        if not pending.launched:
            # Query preparation may fail before launch. Drain only packing;
            # never read the uninitialized gather output or write it to cache.
            # A normal backend call can then use synchronous materialization.
            current_stream.wait_stream(pending.stream)
            return False
        try:
            # The original producer wait and this stream's own ordering already
            # protect the gathered data. Waiting on current_stream here would
            # serialize this tail behind the query work we want to overlap.
            with torch.cuda.stream(pending.stream):
                gathered = self._compact_all_gather_rows(pending.gather)
                chunks = torch.split(gathered, pending.reverse_split_len, dim=0)
                latent_full = torch.cat(
                    [chunks[index] for index in pending.cp_reverse_index], dim=0
                )
                get_token_to_kv_pool().set_mla_kv_buffer(
                    layer,
                    pending.inputs[2],
                    latent_full[..., : pending.kv_lora_rank],
                    latent_full[..., pending.kv_lora_rank :],
                )
                event = torch.cuda.Event()
                event.record(pending.stream)
            current_stream.wait_event(event)
        except BaseException:
            # The local pending object retains producers and staging buffers
            # until the queued operations are ordered before subsequent work.
            current_stream.wait_stream(pending.stream)
            raise
        return True

    def materialize_full_mla_kv(
        self, forward_batch, layer: Any, k_nope: Any, k_rope: Any
    ) -> None:
        # The normal backend call consumes an early launch instead of gathering
        # or writing twice. Without a launch this retains the synchronous path.
        if self.finish_mla_kv_materialization(forward_batch, layer):
            return
        kv_lora_rank = k_nope.shape[-1]
        latent = torch.cat([k_nope, k_rope], dim=-1).contiguous()
        latent_full = self.gather_kv_cache(latent, forward_batch)
        get_token_to_kv_pool().set_mla_kv_buffer(
            layer,
            forward_batch.out_cache_loc,
            latent_full[..., :kv_lora_rank],
            latent_full[..., kv_lora_rank:],
        )

    def _all_gather_reorganized(self, x: torch.Tensor, forward_batch):
        prepared = self._prepare_all_gather_rank_major(x, forward_batch)
        get_parallel().attn_cp_group.all_gather_into_tensor(
            prepared.output_buffer, prepared.send_buffer
        )
        return self._compact_all_gather_rows(prepared)

    def _prepare_all_gather_rank_major(
        self, x: torch.Tensor, forward_batch
    ) -> _RankMajorAllGather:
        meta = forward_batch.attn_cp_metadata
        per_rank_token = meta.per_rank_logical_token or meta.per_rank_actual_token
        max_len = max(per_rank_token)
        if per_rank_token == meta.per_rank_actual_token:
            local_len = x.shape[0]
        else:
            local_len = per_rank_token[self.cp_rank]
        assert x.shape[0] >= local_len
        x = x[:local_len]
        pad_size = max_len - x.shape[0]
        if pad_size > 0:
            padding = [0, 0] * (x.ndim - 1) + [0, pad_size]
            x = F.pad(x, padding, mode="constant", value=0)

        group = get_parallel().attn_cp_group
        ctx = (
            use_symmetric_memory(group, disabled=not is_allocation_symmetric())
            if x.is_cuda
            else nullcontext()
        )
        with ctx:
            gathered = torch.empty(
                max_len * self.cp_size,
                *x.shape[1:],
                device=x.device,
                dtype=x.dtype,
            )
        return _RankMajorAllGather(x, gathered, tuple(per_rank_token), max_len)

    @staticmethod
    def _compact_all_gather_rows(prepared: _RankMajorAllGather) -> torch.Tensor:
        gathered = prepared.output_buffer
        per_rank_token = prepared.per_rank_token
        max_len = prepared.max_len

        # Balanced ranks need no padding removal. Keep the collective output
        # for the caller's zigzag reorder instead of copying every row twice.
        if all(per_rank_len == max_len for per_rank_len in per_rank_token):
            return gathered

        chunks = torch.split(gathered, [max_len] * len(per_rank_token), dim=0)
        return torch.cat(
            [
                chunks[rank][:per_rank_len]
                for rank, per_rank_len in enumerate(per_rank_token)
            ],
            dim=0,
        )
