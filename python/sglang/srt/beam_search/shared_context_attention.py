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
import logging
import weakref
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.kv.router import KVLayerRouter
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


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


_KERNELS = ("default", "split_kv", "stream_k")


def selected_kernel() -> str:
    kernel = envs.SGLANG_BEAM_SHARED_CONTEXT_KERNEL.get()
    if kernel not in _KERNELS:
        raise ValueError(
            f"SGLANG_BEAM_SHARED_CONTEXT_KERNEL is {kernel!r}, not one of {_KERNELS}"
        )
    return kernel


def _attention_kernel(kernel: str):
    if kernel == "split_kv":
        from sglang.kernels.ops.attention.beam_shared_context_split_kv import (
            shared_context_attention_split_kv,
        )

        return shared_context_attention_split_kv
    if kernel == "stream_k":
        from sglang.kernels.ops.attention.beam_shared_context_stream_k import (
            shared_context_attention_stream_k,
        )

        return shared_context_attention_stream_k
    from sglang.kernels.ops.attention.beam_shared_context import (
        shared_context_attention,
    )

    return shared_context_attention


def _unsupported_reason(model_runner: ModelRunner) -> Optional[str]:
    from sglang.srt.configs.model_config import AttentionArch
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.runtime_context import get_exec

    if model_runner.decode_attention_backend_str != "flashinfer":
        return "the decode attention backend is not FlashInfer"
    if model_runner.sliding_window_size is not None:
        return "sliding-window attention is not supported"
    if model_runner.model_config.is_encoder_decoder:
        return "encoder-decoder models are not supported"
    if model_runner.model_config.attention_arch != AttentionArch.MHA:
        return "only MHA attention is supported"
    if model_runner.kv_cache_dtype not in (torch.float16, torch.bfloat16):
        return "the KV cache dtype must be float16 or bfloat16"
    if get_exec().overlap.enable_two_batch_overlap:
        return "two-batch overlap is enabled"
    for module in model_runner.model.modules():
        if isinstance(module, RadixAttention):
            if module.logit_cap > 0:
                return "attention logit caps are not supported"
            if module.v_head_dim != module.head_dim:
                return "v_head_dim differs from head_dim"
    return None


_support: weakref.WeakKeyDictionary[ModelRunner, bool] = weakref.WeakKeyDictionary()


def supports_shared_context(model_runner: ModelRunner) -> bool:
    """Evaluated once per runner; warns once when the env flag is on but the
    setup falls back to ordinary attention."""
    supported = _support.get(model_runner)
    if supported is None:
        reason = _unsupported_reason(model_runner)
        if reason is not None:
            logger.warning(
                "SGLANG_ENABLE_BEAM_SHARED_CONTEXT_ATTENTION is set but "
                "shared-context beam attention is unsupported (%s); "
                "using ordinary attention.",
                reason,
            )
        supported = _support[model_runner] = reason is None
    return supported


def _metadata_tensor(values, dtype, device):
    if torch.device(device).type == "cuda":
        return torch.tensor(values, dtype=dtype, pin_memory=True).to(
            device, non_blocking=True
        )
    return torch.tensor(values, dtype=dtype, device=device)


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
    kv_layers: KVLayerRouter,
) -> Optional[torch.Tensor]:
    metadata = forward_batch.beam_attention_metadata
    if metadata is None:
        return None
    assert layer.logit_cap == 0

    k_buffer, v_buffer = kv_layers.read(layer.layer_id).kv()
    if k_buffer.ndim != 3 or v_buffer.ndim != 3:
        raise RuntimeError("Shared-context attention requires 3D MHA K/V cache buffers")

    q_heads = layer.tp_q_head_num
    head_dim = layer.head_dim
    flat_q = q.contiguous().view(metadata.num_rows, q_heads, head_dim)
    output = torch.empty_like(flat_q)
    attention = _attention_kernel(selected_kernel())
    for bucket in metadata.buckets:
        attention(
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
