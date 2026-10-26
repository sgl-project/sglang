"""Geometry of one NPU EAGLE draft-decode step.

The draft writes ``topk`` KV slots per request per step (the tree level), so two
quantities depend on the tree width and not just on the step counter:

    live KV window at step i  =  prefix + (i + 1) * topk
    queries per request       =  topk

Both used to be hard-coded to the ``topk == 1`` shape (``step + 1`` and
``[1..bs]``). With ``topk == 1`` these helpers return exactly the old values, so
the production chain path is unchanged; with a wider tree the draft stops
reading a window that only covers the first level.

Kept torch-free on purpose: importable and testable on any host.
"""

from __future__ import annotations

__all__ = ["draft_step_kv_offset", "draft_step_query_lens"]


def draft_step_kv_offset(step: int, topk: int, num_steps: int) -> int:
    """KV slots live at draft step ``step``, counted from the prefix.

    Draft slots are laid out branch-major: branch ``t`` owns the contiguous
    positions ``prefix + t * num_steps + [0, num_steps)``, so at step ``s`` the
    farthest written slot is ``prefix + (topk - 1) * num_steps + s`` and the
    window has to reach it inclusively. For ``topk == 1`` this is the historical
    ``step + 1``.
    """
    step = int(step)
    topk = int(topk)
    num_steps = int(num_steps)
    if step < 0:
        raise ValueError(f"draft step must be >= 0, got {step}")
    if topk < 1:
        raise ValueError(f"topk must be >= 1, got {topk}")
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    return (topk - 1) * num_steps + step + 1


def draft_step_query_lens(num_requests: int, topk: int) -> list[int]:
    """TND query split for one draft step: ``[topk, 2*topk, ..., bs*topk]``.

    ``num_requests`` is the batch size (``forward_batch.seq_lens.numel()``), not
    the token count: the step runs ``topk`` query rows per request.
    """
    num_requests = int(num_requests)
    topk = int(topk)
    if num_requests < 0:
        raise ValueError(f"num_requests must be >= 0, got {num_requests}")
    if topk < 1:
        raise ValueError(f"topk must be >= 1, got {topk}")
    return [(i + 1) * topk for i in range(num_requests)]
