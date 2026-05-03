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
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional, Tuple

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.frozen_kv_mtp_info import (
    FrozenKVMTPContext,
    FrozenKVMTPDraftInput,
)
from sglang.srt.speculative.spec_utils import fast_topk


def _resolve_frozen_kv_mtp_backend_type(worker) -> str:
    server_args = worker.server_args
    return (
        server_args.speculative_draft_attention_backend
        or server_args.decode_attention_backend
        or server_args.attention_backend
    )


def create_frozen_kv_mtp_draft_backend(worker):
    if worker.topk == 1:
        return worker.draft_model_runner.attn_backend

    backend_type = _resolve_frozen_kv_mtp_backend_type(worker)
    backend_map = {
        "triton": _create_triton_frozen_kv_mtp_backend,
    }
    if backend_type not in backend_map:
        raise ValueError(
            "Frozen-KV MTP topk > 1 currently supports only the triton attention "
            f"backend, got {backend_type}."
        )
    return backend_map[backend_type](worker)


def _create_triton_frozen_kv_mtp_backend(worker):
    from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

    max_bs = worker.req_to_token_pool.size * worker.topk
    kv_indptr_buf = torch.zeros(
        (max_bs + 1,), dtype=torch.int32, device=worker.draft_model_runner.device
    )
    return TritonAttnBackend(
        worker.draft_model_runner,
        skip_prefill=True,
        kv_indptr_buf=kv_indptr_buf,
    )


def build_and_bind_frozen_kv_mtp_context(worker) -> Optional[FrozenKVMTPContext]:
    draft_model = worker.draft_model_runner.model
    if not hasattr(draft_model, "build_frozen_kv_mtp_context") or not hasattr(
        draft_model, "bind_frozen_kv_context"
    ):
        return None

    ctx = draft_model.build_frozen_kv_mtp_context(
        target_model=worker.target_worker.model_runner.model,
        target_token_to_kv_pool=worker.target_worker.model_runner.token_to_kv_pool,
    )
    draft_model.bind_frozen_kv_context(ctx)
    return ctx


@contextmanager
def frozen_kv_target_view(forward_batch: ForwardBatch, kv_context: FrozenKVMTPContext):
    """Build attention metadata against committed target-prefix geometry."""
    if kv_context is None:
        raise RuntimeError(
            "Frozen-KV MTP target view called before the model was bound; "
            "bind the frozen KV context first."
        )
    saved_spec_info = forward_batch.spec_info
    saved_kv_pool = forward_batch.token_to_kv_pool
    forward_batch.spec_info = None
    forward_batch.token_to_kv_pool = kv_context.target_token_to_kv_pool
    try:
        yield
    finally:
        forward_batch.spec_info = saved_spec_info
        forward_batch.token_to_kv_pool = saved_kv_pool


@contextmanager
def target_kv_pool_view(forward_batch: ForwardBatch, kv_context: FrozenKVMTPContext):
    if kv_context is None:
        raise RuntimeError(
            "Frozen-KV MTP target KV pool view called before the model was bound; "
            "bind the frozen KV context first."
        )
    saved_kv_pool = forward_batch.token_to_kv_pool
    forward_batch.token_to_kv_pool = kv_context.target_token_to_kv_pool
    try:
        yield
    finally:
        forward_batch.token_to_kv_pool = saved_kv_pool


def set_frozen_kv_positions(forward_batch: ForwardBatch, topk: int) -> None:
    """Rope phase = last written target slot, not advanced per draft step."""
    seq_lens = forward_batch.seq_lens
    positions = torch.clamp(seq_lens - 1, min=0).to(torch.int64)
    if (
        topk > 1
        and forward_batch.positions is not None
        and forward_batch.positions.numel() == positions.numel() * topk
    ):
        positions = positions.repeat_interleave(topk, dim=0)
    if forward_batch.positions is None:
        forward_batch.positions = positions
    else:
        if forward_batch.positions.shape == positions.shape:
            forward_batch.positions.copy_(positions)
        else:
            forward_batch.positions = positions


def expand_for_topk_draft(forward_batch: ForwardBatch, topk: int) -> None:
    """Repeat committed-prefix metadata for the active ``B * topk`` frontier."""
    if topk == 1 or forward_batch.batch_size == 0:
        return

    if forward_batch.batch_size != forward_batch.seq_lens.shape[0]:
        raise RuntimeError(
            "Frozen-KV MTP topk expansion expects an unexpanded forward "
            "batch where batch_size == len(seq_lens)."
        )

    forward_batch.batch_size *= topk
    forward_batch.req_pool_indices = forward_batch.req_pool_indices.repeat_interleave(
        topk, dim=0
    )
    forward_batch.seq_lens = forward_batch.seq_lens.repeat_interleave(topk, dim=0)
    if forward_batch.seq_lens_cpu is not None:
        forward_batch.seq_lens_cpu = forward_batch.seq_lens_cpu.repeat_interleave(
            topk, dim=0
        )
        forward_batch.seq_lens_sum = forward_batch.seq_lens_cpu.sum().item()
    else:
        forward_batch.seq_lens_sum = torch.sum(forward_batch.seq_lens).item()

    positions = torch.clamp(forward_batch.seq_lens - 1, min=0).to(torch.int64)
    forward_batch.positions = positions
    forward_batch.num_token_non_padded_cpu = positions.numel()
    if forward_batch.num_token_non_padded is not None:
        forward_batch.num_token_non_padded.fill_(positions.numel())
    if (
        forward_batch.mrope_positions is not None
        and forward_batch.mrope_positions.shape[-1] * topk == positions.numel()
    ):
        forward_batch.mrope_positions = forward_batch.mrope_positions.repeat_interleave(
            topk, dim=-1
        )


def position_for_batch(batch: ScheduleBatch) -> torch.Tensor:
    return torch.clamp(batch.seq_lens - 1, min=0).to(torch.int64)


def recurrent_hidden_size(worker) -> int:
    return int(worker.draft_model_runner.model.backbone_hidden_size)


def init_frozen_kv_metadata(worker, forward_batch: ForwardBatch) -> None:
    if forward_batch.forward_mode.is_idle():
        return
    if forward_batch.seq_lens_cpu is not None:
        forward_batch.seq_lens_sum = forward_batch.seq_lens_cpu.sum().item()
    else:
        forward_batch.seq_lens_sum = torch.sum(forward_batch.seq_lens).item()
    with frozen_kv_target_view(forward_batch, worker.kv_context):
        worker.draft_attn_backend.init_forward_metadata(forward_batch)
    forward_batch.attn_backend = worker.draft_attn_backend


def init_frozen_kv_metadata_capture_cuda_graph(
    worker, forward_batch: ForwardBatch
) -> None:
    with frozen_kv_target_view(forward_batch, worker.kv_context):
        worker.draft_attn_backend.init_forward_metadata_capture_cuda_graph(
            forward_batch.batch_size,
            forward_batch.positions.numel(),
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            encoder_lens=None,
            forward_mode=ForwardMode.DECODE,
            spec_info=None,
        )
    forward_batch.attn_backend = worker.draft_attn_backend


def init_frozen_kv_metadata_replay_cuda_graph(
    worker, forward_batch: ForwardBatch, bs: int, seq_lens_sum: int
) -> None:
    with frozen_kv_target_view(forward_batch, worker.kv_context):
        worker.draft_attn_backend.init_forward_metadata_replay_cuda_graph(
            bs,
            forward_batch.req_pool_indices[:bs],
            forward_batch.seq_lens[:bs],
            seq_lens_sum,
            encoder_lens=None,
            forward_mode=ForwardMode.DECODE,
            spec_info=None,
            seq_lens_cpu=(
                forward_batch.seq_lens_cpu[:bs]
                if forward_batch.seq_lens_cpu is not None
                else None
            ),
        )
    forward_batch.attn_backend = worker.draft_attn_backend


def select_last_extend_hidden(
    batch: ScheduleBatch, hidden_states: torch.Tensor
) -> torch.Tensor:
    if hidden_states.shape[0] == batch.batch_size():
        return hidden_states
    lens = torch.tensor(batch.extend_lens, device=hidden_states.device)
    last_indices = torch.cumsum(lens, dim=0) - 1
    return hidden_states[last_indices.to(torch.long)]


def select_last_verified_seed(
    draft_input: FrozenKVMTPDraftInput,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if draft_input.num_accepted_tokens is None:
        return draft_input.verified_id, draft_input.hidden_states

    counts = draft_input.num_accepted_tokens.to(torch.long)
    last_indices = torch.cumsum(counts, dim=0) - 1
    return (
        draft_input.verified_id[last_indices],
        draft_input.hidden_states[last_indices],
    )


def capture_for_decode(
    logits_output: LogitsProcessorOutput, draft_input: FrozenKVMTPDraftInput, topk: int
) -> None:
    probs = torch.softmax(logits_output.next_token_logits, dim=-1)
    draft_input.topk_p, draft_input.topk_index = fast_topk(probs, topk, dim=-1)
    draft_input.hidden_states = logits_output.hidden_states
