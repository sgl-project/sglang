"""Opt-in runtime trace for diffusion-LLM vocab-state lifecycles."""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

import torch

from sglang.srt.dllm.tp_local_vocab_state import VocabState
from sglang.srt.environ import envs

_TRACE_LOCK = threading.Lock()


def tensor_nbytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def vocab_state_nbytes(state: VocabState | None) -> int:
    if state is None:
        return 0
    return (
        tensor_nbytes(state.max_values)
        + tensor_nbytes(state.argmax_ids)
        + tensor_nbytes(state.logsumexp)
        + tensor_nbytes(state.max_probs)
    )


def compact_tp_gather_bytes(*, rows: int, tp_size: int, packed: bool) -> int:
    rows = int(rows)
    tp_size = int(tp_size)
    if rows <= 0 or tp_size <= 1:
        return 0
    if packed:
        return rows * tp_size * 3 * torch.tensor([], dtype=torch.float32).element_size()
    return rows * tp_size * (
        2 * torch.tensor([], dtype=torch.float32).element_size()
        + torch.tensor([], dtype=torch.int64).element_size()
    )


def emit_consumer_state_trace(record: dict[str, Any]) -> None:
    path = envs.SGLANG_CONSUMER_STATE_TRACE_JSONL.get()
    if not path:
        return

    event = {
        "schema_version": 1,
        "timestamp_unix_s": time.time(),
        "framework": "sglang",
        **record,
    }
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    line = json.dumps(event, sort_keys=True, separators=(",", ":"))
    with _TRACE_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def emit_full_vocab_trace(
    *,
    component: str,
    full_logits: torch.Tensor | None,
    vocab_size: int,
    tp_size: int,
    rank: int,
    consumer_contract: str,
    fallback_reason: str | None = None,
) -> None:
    rows = int(full_logits.shape[0]) if full_logits is not None else 0
    emit_consumer_state_trace(
        {
            "component": component,
            "path": "full_vocab_materialized",
            "consumer_contract": consumer_contract,
            "rank": int(rank),
            "tp_size": int(tp_size),
            "rows": rows,
            "vocab_size": int(vocab_size),
            "local_vocab_size": int(full_logits.shape[-1])
            if full_logits is not None and full_logits.ndim > 0
            else 0,
            "full_vocab_materialized_bytes": tensor_nbytes(full_logits),
            "full_vocab_reread_bytes": 0,
            "compact_state_bytes": 0,
            "tp_gather_bytes": tensor_nbytes(full_logits) if int(tp_size) > 1 else 0,
            "fallback_reason": fallback_reason,
            "exact_replay_status": "runtime_metadata_only",
        }
    )


def emit_compact_vocab_state_trace(
    *,
    component: str,
    local_logits: torch.Tensor,
    state: VocabState,
    vocab_size: int,
    valid_vocab_size: int,
    tp_size: int,
    rank: int,
    packed_gather: bool,
    consumer_contract: str,
) -> None:
    rows = int(local_logits.shape[0])
    dtype_bytes = int(local_logits.element_size())
    avoidable_full_vocab_bytes = rows * int(vocab_size) * dtype_bytes
    emit_consumer_state_trace(
        {
            "component": component,
            "path": "consumer_sufficient_compact",
            "consumer_contract": consumer_contract,
            "rank": int(rank),
            "tp_size": int(tp_size),
            "rows": rows,
            "vocab_size": int(vocab_size),
            "local_vocab_size": int(valid_vocab_size),
            "local_vocab_materialized_bytes": tensor_nbytes(local_logits),
            "full_vocab_materialized_bytes": 0,
            "full_vocab_reread_bytes": 0,
            "avoidable_full_vocab_materialized_bytes": avoidable_full_vocab_bytes,
            "compact_state_bytes": vocab_state_nbytes(state),
            "tp_gather_bytes": compact_tp_gather_bytes(
                rows=rows, tp_size=tp_size, packed=packed_gather
            ),
            "fallback_reason": None,
            "exact_replay_status": "runtime_metadata_only",
        }
    )
