"""Row-level TARGET_VERIFY vs EXTEND dump. Off unless SGLANG_DEBUG_DSV41_ALIGN_DUMP=1.

Compares the scored token: TARGET_VERIFY row 0 vs EXTEND last real row.
Writes one small JSON per matching forward (T<=32). Host-syncs; do not enable
under a captured CUDA graph (hooks would miss Engram/MoE on replay anyway).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_step = 0
_cur: Optional[dict[str, Any]] = None
_NO_ID = 4484


def enabled() -> bool:
    return bool(envs.SGLANG_DEBUG_DSV41_ALIGN_DUMP.get())


def _capturing() -> bool:
    try:
        return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _rank() -> int:
    try:
        from sglang.srt.runtime_context import get_parallel

        return int(get_parallel().tp_group.rank_in_group)
    except Exception:
        return int(os.environ.get("RANK", "0"))


def _fp(t: Optional[torch.Tensor]) -> Optional[dict[str, Any]]:
    if t is None:
        return None
    x = t.detach().reshape(-1).float()
    n = int(x.numel())
    if n == 0:
        return {"n": 0}
    # Avoid a giant .tolist(); a few probes plus reductions catch ulp vs route flips.
    head = min(8, n)
    return {
        "n": n,
        "sum": float(x.sum().item()),
        "sumabs": float(x.abs().sum().item()),
        "absmax": float(x.abs().max().item()),
        "head": [float(v) for v in x[:head].cpu().tolist()],
    }


def _ints(t: Optional[torch.Tensor], cap: int = 64) -> Optional[list[int]]:
    if t is None:
        return None
    x = t.detach().reshape(-1).long()
    n = int(x.numel())
    vals = [int(v) for v in x[: min(cap, n)].cpu().tolist()]
    return vals


def scored_row(forward_batch, num_tokens: int) -> int:
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is not None and (
        callable(getattr(mode, "is_target_verify", None))
        and mode.is_target_verify()
        or callable(getattr(mode, "is_decode", None))
        and mode.is_decode()
    ):
        return 0
    n = getattr(forward_batch, "num_token_non_padded_cpu", None)
    if n is None:
        n = num_tokens
    n = int(n)
    if n <= 0:
        return 0
    return min(n, num_tokens) - 1


def _should_dump(forward_batch, num_tokens: int) -> bool:
    if not enabled() or _capturing() or num_tokens <= 0 or num_tokens > 32:
        return False
    if _rank() != 0:
        return False
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is not None and callable(getattr(mode, "is_target_verify", None)):
        if mode.is_target_verify():
            return True
    return 20 <= num_tokens <= 32


def begin_forward(
    forward_batch,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    hash_ids: Optional[torch.Tensor] = None,
    hasher=None,
) -> None:
    global _cur
    _cur = None
    if hidden_states is None:
        return
    t = int(hidden_states.shape[0])
    if not _should_dump(forward_batch, t):
        return
    mode = getattr(forward_batch, "forward_mode", None)
    mode_name = getattr(mode, "name", str(mode))
    row = scored_row(forward_batch, t)
    ids = input_ids.detach().reshape(-1)
    pos = positions.detach().reshape(-1)
    token_id = int(ids[row].item()) if row < ids.numel() else None
    position = int(pos[row].item()) if row < pos.numel() else None
    # Glass-windows scored token is No at pos 24. Skip warmup junk that is not that
    # and not a T=6 verify (those we always keep).
    if token_id != _NO_ID:
        return
    hist = None
    if hasher is not None and getattr(hasher, "history", None) is not None:
        slots = getattr(forward_batch, "req_pool_indices", None)
        if slots is not None and slots.numel():
            slot = int(slots.reshape(-1)[0].item())
            hist = _ints(hasher.history[slot], cap=int(hasher.history.shape[-1]))
    _cur = {
        "rank": _rank(),
        "mode": mode_name,
        "t": t,
        "row": row,
        "token_id": token_id,
        "position": position,
        "input_ids_row_window": _ints(ids[max(0, row - 3) : row + 4], cap=8),
        "engram_history": hist,
        "hash_ids_row": _ints(None if hash_ids is None else hash_ids[row], cap=64),
        "embed": _fp(hidden_states[row]),
        "layers": {},
    }


def record_hidden(name: str, hidden: Optional[torch.Tensor]) -> None:
    if _cur is None or hidden is None:
        return
    row = int(_cur["row"])
    if row >= hidden.shape[0]:
        return
    _cur[name] = _fp(hidden[row])


def record_layer(layer_id: int, key: str, hidden: Optional[torch.Tensor]) -> None:
    if _cur is None or hidden is None:
        return
    row = int(_cur["row"])
    if row >= hidden.shape[0]:
        return
    lid = str(int(layer_id))
    slot = _cur["layers"].setdefault(lid, {})
    slot[key] = _fp(hidden[row])


def record_moe_row(layer_id: int, hidden: Optional[torch.Tensor], topk_output: Any) -> None:
    if _cur is None:
        return
    row = int(_cur["row"])
    lid = str(int(layer_id))
    slot = _cur["layers"].setdefault(lid, {})
    if hidden is not None and row < hidden.shape[0]:
        slot["moe_in"] = _fp(hidden[row])
    ids = getattr(topk_output, "topk_ids", None)
    w = getattr(topk_output, "topk_weights", None)
    logits = getattr(topk_output, "router_logits", None)
    if ids is not None and row < ids.shape[0]:
        slot["moe_topk_ids"] = _ints(ids[row], cap=32)
    if w is not None and row < w.shape[0]:
        slot["moe_topk_w"] = _fp(w[row])
    if logits is not None and row < logits.shape[0]:
        slot["moe_gate"] = _fp(logits[row])
        g = logits[row].detach().float()
        if g.numel():
            topv, topi = torch.topk(g, k=min(8, g.numel()))
            slot["moe_gate_top8_ids"] = [int(i) for i in topi.cpu().tolist()]
            slot["moe_gate_top8"] = [float(v) for v in topv.cpu().tolist()]


def flush() -> None:
    global _cur, _step
    if _cur is None:
        return
    payload = _cur
    _cur = None
    out_dir = envs.SGLANG_DEBUG_DSV41_ALIGN_DUMP_DIR.get() or "dsv41-align-dump"
    os.makedirs(out_dir, exist_ok=True)
    rank = payload.get("rank", _rank())
    path = os.path.join(
        out_dir,
        f"rank{rank}_step{_step}_{payload.get('mode')}_t{payload.get('t')}_"
        f"id{payload.get('token_id')}_pos{payload.get('position')}.json",
    )
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)
    logger.warning("dsv41 align dump wrote %s", path)
    _step += 1
