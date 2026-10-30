# Copyright 2026 SGLang Team
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
"""Shared-context decode attention for wide beam batches."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner


class BeamAttentionBucket(msgspec.Struct):
    row_indices: torch.Tensor
    context_slots: torch.Tensor
    context_cu_seqlens: torch.Tensor
    decode_slots: torch.Tensor
    beam_width: int
    decode_lens: torch.Tensor


class BeamAttentionMetadata(msgspec.Struct):
    buckets: list[BeamAttentionBucket]
    num_rows: int


def enabled() -> bool:
    return envs.SGLANG_ENABLE_BEAM_SHARED_CONTEXT_ATTENTION.get()


def _positive_int(field) -> int:
    value = field.get()
    if value <= 0:
        raise ValueError(f"{field.name} must be positive, got {value}")
    return value


def graph_beam_width() -> int:
    return _positive_int(envs.SGLANG_BEAM_SHARED_CONTEXT_WIDTH)


def graph_max_context() -> int:
    return _positive_int(envs.SGLANG_BEAM_SHARED_CONTEXT_GRAPH_MAX_CONTEXT)


def graph_max_decode() -> int:
    return _positive_int(envs.SGLANG_BEAM_SHARED_CONTEXT_GRAPH_MAX_DECODE)


def supports_shared_context(model_runner) -> bool:
    from sglang.srt.configs.model_config import AttentionArch

    return (
        model_runner.decode_attention_backend_str == "flashinfer"
        and model_runner.sliding_window_size is None
        and not model_runner.model_config.is_encoder_decoder
        and model_runner.model_config.attention_arch == AttentionArch.MHA
        and model_runner.kv_cache_dtype in (torch.float16, torch.bfloat16)
    )


def _metadata_tensor(values, dtype, device):
    if torch.device(device).type == "cuda":
        return torch.tensor(values, dtype=dtype, pin_memory=True).to(
            device, non_blocking=True
        )
    return torch.tensor(values, dtype=dtype, device=device)


def graph_compatible(metadata, num_rows: int, captured_rows: int) -> bool:
    if (
        metadata is None
        or num_rows != captured_rows
        or metadata.num_rows != num_rows
        or len(metadata.buckets) != 1
    ):
        return False
    bucket = metadata.buckets[0]
    if bucket.row_indices.ndim != 2 or bucket.decode_slots.ndim != 2:
        return False
    groups = bucket.row_indices.shape[0]
    return (
        bucket.beam_width == graph_beam_width()
        and bucket.row_indices.shape == (groups, bucket.beam_width)
        and groups * bucket.beam_width == num_rows
        and bucket.context_slots.numel() <= groups * graph_max_context()
        and bucket.context_cu_seqlens.shape == (groups + 1,)
        and bucket.decode_lens.shape == (groups,)
        and bucket.decode_slots.shape[0] == groups
        and bucket.decode_slots.shape[1] <= graph_max_decode() * bucket.beam_width
    )


def build_graph_capture_metadata(num_rows: int, device):
    width = graph_beam_width()
    if not enabled() or num_rows < width or num_rows % width:
        return None
    groups = num_rows // width
    rows = [
        [group]
        + list(range(groups + group * (width - 1), groups + (group + 1) * (width - 1)))
        for group in range(groups)
    ]
    return BeamAttentionMetadata(
        num_rows=num_rows,
        buckets=[
            BeamAttentionBucket(
                row_indices=_metadata_tensor(rows, torch.int32, device),
                context_slots=torch.zeros(
                    groups * graph_max_context(), dtype=torch.int32, device=device
                ),
                context_cu_seqlens=torch.arange(
                    groups + 1, dtype=torch.int32, device=device
                ),
                decode_slots=torch.zeros(
                    (groups, width * graph_max_decode()),
                    dtype=torch.int32,
                    device=device,
                ),
                beam_width=width,
                decode_lens=torch.full(
                    (groups,), graph_max_decode(), dtype=torch.int32, device=device
                ),
            )
        ],
    )


def stage_graph_metadata(target, source) -> None:
    if not graph_compatible(source, source.num_rows, target.num_rows):
        raise ValueError("Shared-context metadata does not fit the captured graph")
    if len(target.buckets) != 1:
        raise ValueError("Captured shared-context metadata must have one width")
    destination, current = target.buckets[0], source.buckets[0]
    if (
        destination.beam_width != current.beam_width
        or destination.row_indices.shape != current.row_indices.shape
        or current.context_slots.numel() > destination.context_slots.numel()
        or destination.context_cu_seqlens.shape != current.context_cu_seqlens.shape
        or destination.decode_lens.shape != current.decode_lens.shape
        or current.decode_slots.shape[0] != destination.decode_slots.shape[0]
        or current.decode_slots.shape[1] > destination.decode_slots.shape[1]
    ):
        raise ValueError("Shared-context metadata exceeds captured buffers")
    destination.context_slots[current.context_slots.numel() :].zero_()
    destination.decode_slots[:, current.decode_slots.shape[1] :].zero_()
    destination.row_indices.copy_(current.row_indices)
    destination.context_slots[: current.context_slots.numel()].copy_(
        current.context_slots
    )
    destination.context_cu_seqlens.copy_(current.context_cu_seqlens)
    destination.decode_slots[:, : current.decode_slots.shape[1]].copy_(
        current.decode_slots
    )
    destination.decode_lens.copy_(current.decode_lens)


def build_metadata(
    batch: ScheduleBatch,
    model_runner: ModelRunner,
) -> Optional[BeamAttentionMetadata]:
    if batch.beam_tail is None or not batch.forward_mode.is_decode() or not enabled():
        return None
    if not supports_shared_context(model_runner):
        return None
    entries = batch.beam_tail.entries
    if not entries or batch.seq_lens_cpu is None:
        return None

    widths = [entry.end - entry.start + 1 for entry in entries]
    if any(width <= 1 for width in widths):
        return None
    if sum(widths) != len(batch.seq_lens):
        return None

    num_base_rows = batch.beam_tail.num_base_rows
    req_to_token = model_runner.req_to_token_pool.req_to_token
    entries_by_width: dict[int, list] = {}
    for entry, beam_width in zip(entries, widths, strict=True):
        entries_by_width.setdefault(beam_width, []).append(entry)

    buckets = []
    for beam_width, bucket_entries in entries_by_width.items():
        group_rows = _metadata_tensor(
            [
                [entry.leader_idx]
                + list(
                    range(
                        num_base_rows + entry.start,
                        num_base_rows + entry.end,
                    )
                )
                for entry in bucket_entries
            ],
            dtype=torch.int64,
            device=batch.device,
        )
        req_rows = batch.req_pool_indices[group_rows]

        context_parts = []
        decode_parts = []
        prompt_lens = []
        decode_lens = []
        for group_index, entry in enumerate(bucket_entries):
            prompt_len = int(entry.group.prompt_len)
            seq_len = int(batch.seq_lens_cpu[entry.leader_idx])
            decode_len = seq_len - prompt_len
            if decode_len <= 0:
                return None
            group_req_rows = req_rows[group_index]
            context_parts.append(
                req_to_token[group_req_rows[:1], :prompt_len].reshape(-1)
            )
            decode_parts.append(
                req_to_token[
                    group_req_rows,
                    prompt_len:seq_len,
                ]
                .transpose(0, 1)
                .reshape(-1)
            )
            prompt_lens.append(prompt_len)
            decode_lens.append(decode_len)

        max_decode_len = max(decode_lens)
        padded_decode_parts = []
        for decode_part in decode_parts:
            padding = max_decode_len * beam_width - decode_part.numel()
            if padding:
                decode_part = torch.cat(
                    [
                        decode_part,
                        torch.zeros(
                            padding,
                            dtype=decode_part.dtype,
                            device=decode_part.device,
                        ),
                    ]
                )
            padded_decode_parts.append(decode_part)

        buckets.append(
            BeamAttentionBucket(
                row_indices=group_rows.to(dtype=torch.int32),
                context_slots=torch.cat(context_parts),
                context_cu_seqlens=_metadata_tensor(
                    [0, *itertools.accumulate(prompt_lens)],
                    dtype=torch.int32,
                    device=batch.device,
                ),
                decode_slots=torch.stack(padded_decode_parts),
                beam_width=beam_width,
                decode_lens=_metadata_tensor(
                    decode_lens,
                    dtype=torch.int32,
                    device=batch.device,
                ),
            )
        )

    return BeamAttentionMetadata(buckets=buckets, num_rows=len(batch.seq_lens))


def forward(
    q: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    token_to_kv_pool: MHATokenToKVPool,
) -> Optional[torch.Tensor]:
    metadata = forward_batch.beam_attention_metadata
    if metadata is None:
        return None
    if layer.logit_cap > 0:
        raise ValueError(
            "Shared-context beam attention does not support attention logit caps; "
            "disable SGLANG_ENABLE_BEAM_SHARED_CONTEXT_ATTENTION for this model."
        )

    k_buffer, v_buffer = token_to_kv_pool.get_kv_buffer(layer.layer_id)
    if k_buffer.ndim != 3 or v_buffer.ndim != 3:
        raise RuntimeError("Shared-context attention requires 3D MHA K/V cache buffers")

    q_heads = layer.tp_q_head_num
    head_dim = layer.head_dim
    flat_q = q.contiguous().view(metadata.num_rows, q_heads, head_dim)
    output = torch.empty_like(flat_q)
    from sglang.kernels.ops.attention.beam_shared_context import (
        shared_context_attention,
    )

    for bucket in metadata.buckets:
        shared_context_attention(
            flat_q,
            k_buffer,
            v_buffer,
            bucket.row_indices,
            bucket.context_slots,
            bucket.context_cu_seqlens,
            bucket.decode_slots,
            bucket.decode_lens,
            output=output,
            softmax_scale=layer.scaling,
        )
    return output.view(-1, q_heads * head_dim)
