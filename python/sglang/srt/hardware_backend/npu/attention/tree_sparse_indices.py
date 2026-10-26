"""Tree-aware sparse_indices for DSA attention on Ascend.

GLM-5.2 runs DeepSeek Sparse Attention on every layer: instead of a mask,
npu_sparse_flash_attention takes sparse_indices[T, K], an explicit list of KV
positions each query may read. A tree is therefore expressed in that list:

    row = [ ancestors | prefix top-* (score order) | -1 padding ]

Four properties the SFA kernel forces:

* ancestors first, because sparse_mode=3 truncates the row tail when the causal
  budget is short;
* no sorting, for the same reason: sorting pushes large positions into the tail;
* no repeated position: the kernel has no uniqueness check and reads the list
  verbatim, so a repeat double-counts in the softmax and eats the budget;
* -1 only in the tail, since the scan stops at the first one.

Pure torch (no torch_npu), so the geometry stays unit-testable on CPU.
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = [
    "ancestor_width_for",
    "gather_tree_blocks",
    "draft_ancestors",
    "draft_step_ancestor_positions",
    "merge_tree_sparse_indices",
    "DraftTreeSparseState",
    "DRAFT_TREE_ANCESTORS_ATTR",
    "DRAFT_TREE_PREFIX_ATTR",
]


# ---------------------------------------------------------------------------
# Verify tree: one row per query token of the draft block.
# ---------------------------------------------------------------------------


def ancestor_width_for(num_steps: int, num_verify_tokens: Optional[int] = None) -> int:
    """Slots reserved for "self + ancestors" in each sparse-index row.

    A node at depth ``d`` needs ``d + 1`` entries; the deepest node of an EAGLE
    tree built with ``num_steps`` draft steps sits at depth ``num_steps``.
    """
    width = int(num_steps) + 1
    if num_verify_tokens is not None:
        width = min(width, int(num_verify_tokens))
    if width <= 0:
        raise ValueError(f"invalid ancestor width {width}")
    return width


def gather_tree_blocks(
    full_mask: torch.Tensor,
    prefix_lens: torch.Tensor,
    num_verify_tokens: int,
) -> torch.Tensor:
    """Slice the draft-block part of the flat ``custom_mask`` into ``[bs, N, N]``.

    ``custom_mask`` is the FLAT buffer produced by
    ``build_tree_kernel_efficient``: for request ``i`` it holds ``N`` rows of
    width ``P_i + N``; columns ``[0, P_i)`` are the (always visible) prefix and
    columns ``[P_i, P_i + N)`` are the draft block, where ``True`` means
    "visible" (self + ancestors).

    Returns a bool tensor with the same convention (``True`` = visible).
    """
    n = int(num_verify_tokens)
    if n <= 0:
        raise ValueError(f"num_verify_tokens must be positive, got {n}")
    if prefix_lens.dim() != 1:
        raise ValueError(f"prefix_lens must be 1-D, got {prefix_lens.shape}")
    p = prefix_lens.to(torch.int64)
    bs = p.numel()
    flat = full_mask.reshape(-1)

    expected = int((p + n).sum().item()) * n
    if flat.numel() < expected:
        raise ValueError(
            f"custom_mask too small: need {expected} elements for N={n}, "
            f"prefix_lens={p.tolist()}, got {flat.numel()}"
        )

    device = flat.device
    p = p.to(device)
    widths = p + n
    offsets = torch.cumsum(widths * n, dim=0) - widths * n
    rows = torch.arange(n, device=device)
    # index[b, r, c] = offset[b] + r * (P_b + N) + P_b + c
    index = (
        offsets.view(bs, 1, 1)
        + rows.view(1, n, 1) * widths.view(bs, 1, 1)
        + p.view(bs, 1, 1)
        + rows.view(1, 1, n)
    )
    return flat[index].to(torch.bool)


def draft_ancestors(allowed: torch.Tensor, ancestor_width: int) -> torch.Tensor:
    """Per row, the visible draft nodes in ascending node order, ``-1`` padded.

    Args:
        allowed: ``[bs, N, N]`` bool, "self + ancestors" relation (``True`` =
            visible), as returned by :func:`gather_tree_blocks`.
        ancestor_width: reserved width ``A``; clamped to ``N``.

    Returns:
        ``[bs, N, A]`` int32 request-local node ids, ascending, with ``-1`` in
        the unused tail slots.

    Ascending order is used instead of a score ranking on purpose: a row's
    visible set is exactly ``{self} | ancestors``, whose size is ``depth + 1``
    and therefore never exceeds ``A = num_steps + 1``. There is nothing to
    choose between, so ordering by position removes both the scoring matmul and
    any tie-breaking risk.
    """
    if allowed.dim() != 3:
        raise ValueError(f"expected 3-D allowed mask, got {tuple(allowed.shape)}")

    num_nodes = allowed.shape[1]
    if allowed.shape[2] != num_nodes:
        raise ValueError(
            f"allowed mask must be square in the draft dims, got "
            f"{tuple(allowed.shape)}"
        )
    width = min(int(ancestor_width), num_nodes)
    if width <= 0:
        raise ValueError(f"invalid ancestor width {width}")

    allowed = allowed.to(torch.bool)
    order = torch.arange(num_nodes, device=allowed.device)
    visible = allowed.sum(dim=-1)
    if bool((visible > width).any()):
        raise ValueError(
            f"a row sees {int(visible.max())} draft nodes but only {width} slots "
            f"are reserved (num_steps too small for N={num_nodes}); dropping an "
            "ancestor would silently change the attention"
        )
    # Allowed nodes keep their own index; hidden ones become the sentinel, so a
    # stable ascending sort puts the visible ids first, in ascending order.
    key = torch.where(allowed, order, torch.full_like(order, num_nodes))
    picked = torch.sort(key, dim=-1, stable=True).values[..., :width]
    return torch.where(picked < num_nodes, picked, -1).to(torch.int32)


def merge_tree_sparse_indices(
    prefix_indices: torch.Tensor,
    draft_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    num_verify_tokens: int,
    topk: int,
) -> torch.Tensor:
    """Concatenate ancestors and the prefix top-k into one sparse-index row.

    The prefix block is shifted left per row by that row's ancestor count, so a
    shallow node simply carries more prefix entries and the row tail stays
    ``-1``:

        a_r = (draft_ids[r] >= 0).sum()
        row = [ ancestors(a_r) | prefix(0 .. K - a_r) | -1 (slack) ]

    Args:
        prefix_indices: ``[bs*N, W]`` absolute KV positions (or ``-1`` padding
            produced by the indexer when the prefix is shorter), score order.
        draft_ids: ``[bs*N, A]`` request-local node ids from
            :func:`draft_ancestors`, ``-1`` padded.
        seq_lens: ``[bs]`` prefix length per request (excluding draft tokens).
        num_verify_tokens: ``N``, rows per request.
        topk: ``K``, the sparse width the attention op expects.

    Returns:
        ``[bs*N, K]`` int32, **ancestors first** then the prefix block, with the
        prefix block's own order preserved (score order), no sorting, and
        ``-1`` only in the tail. No position appears twice.

    Raises:
        AssertionError: width/row-count mismatch.
        ValueError: an ancestor outside its request's draft block, or a prefix
            index that is not ``< P_i`` (and not ``-1``).
    """
    n = int(num_verify_tokens)
    topk = int(topk)
    bs = int(seq_lens.numel())
    rows = bs * n

    assert prefix_indices.dim() == 2 and draft_ids.dim() == 2, (
        f"prefix_indices/draft_ids must be 2-D, got "
        f"{tuple(prefix_indices.shape)} / {tuple(draft_ids.shape)}"
    )
    assert prefix_indices.shape[0] == rows, (
        f"prefix_indices has {prefix_indices.shape[0]} rows, expected bs*N={rows}"
    )
    assert draft_ids.shape[0] == rows, (
        f"draft_ids has {draft_ids.shape[0]} rows, expected bs*N={rows}"
    )
    assert prefix_indices.shape[1] + draft_ids.shape[1] == topk, (
        f"reserved width mismatch: prefix {prefix_indices.shape[1]} + "
        f"ancestors {draft_ids.shape[1]} != topk {topk}"
    )

    device = prefix_indices.device
    prefix_width = int(prefix_indices.shape[1])
    prefix_indices = prefix_indices.to(torch.int64).to(device)
    draft_ids = draft_ids.to(torch.int64).to(device)

    valid = draft_ids >= 0
    if draft_ids.numel() and bool((draft_ids[valid] >= n).any()):
        raise ValueError(
            f"draft node ids must be in [0, {n}) or -1, got "
            f"max {int(draft_ids.max())}"
        )
    if bool((valid.sum(dim=-1) == 0).any()):
        raise ValueError("every row needs at least its own node id")

    prefix_lens = seq_lens.to(device=device, dtype=torch.int64).repeat_interleave(n)
    prefix_lens = prefix_lens.view(-1, 1)
    absolute = prefix_lens + draft_ids.clamp(min=0)

    if prefix_width:
        pref = prefix_indices
        bad = (pref >= 0) & (pref >= prefix_lens)
        if bool(bad.any()):
            raise ValueError(
                f"prefix index {int(pref[bad].max())} is not < its prefix length"
            )
    if bool((absolute < prefix_lens).any()) or bool(
        (absolute >= prefix_lens + n).any()
    ):
        raise ValueError(
            "draft ancestor index outside its request's draft block "
            f"[P, P + {n})"
        )

    # Per-row shift: a_r ancestors occupy the front, the prefix follows.
    count = valid.sum(dim=-1, keepdim=True)  # [rows, 1]
    slots = torch.arange(topk, device=device).view(1, -1)
    rel = slots - count  # >= 0 marks the prefix region

    anc_col = slots.clamp(max=draft_ids.shape[1] - 1).expand(rows, -1)
    anc_val = torch.gather(absolute, 1, anc_col)
    if prefix_width:
        pref_col = rel.clamp(0, prefix_width - 1)
        pref_val = torch.gather(prefix_indices, 1, pref_col)
    else:
        pref_val = torch.full((rows, topk), -1, dtype=torch.int64, device=device)

    neg_one = torch.full((), -1, dtype=torch.int64, device=device)
    merged = torch.where(
        rel < 0, anc_val, torch.where(rel < prefix_width, pref_val, neg_one)
    )
    return merged.to(torch.int32)


# ---------------------------------------------------------------------------
# Draft tree: one row per candidate of a single draft step.
# ---------------------------------------------------------------------------


def draft_step_ancestor_positions(
    paths: Optional[torch.Tensor],
    parent_slot: Optional[torch.Tensor],
    prefix_len: torch.Tensor,
    step: int,
    topk: int,
    num_steps: int,
    check: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ancestor paths and absolute positions of one draft-decode step.

    Draft-stage counterpart of the verify tree: the draft keeps a fixed width,
    so the tree is ``topk`` nodes per level, and level ``d`` occupies KV
    positions ``prefix + d * topk + slot``. A step-``i`` candidate therefore sees
    one node per earlier level (its own chain) plus itself: ``i + 1`` entries.
    Its parent chain terminates inside the prefix (the already-committed token
    the first level hangs off), which the prefix top-k covers, not this list.

    Args:
        paths: ``[bs, topk, step - 1]`` request-local slot ids of each previous
            level's ancestors, as returned by the previous call. ``None`` at
            step 0.
        parent_slot: ``[bs, topk]`` slot (``0 <= s < topk``) of this level's
            parent within the previous level. ``None`` at step 0.
        prefix_len: ``[bs]`` committed KV length, **including** the parent token
            the first level hangs off.
        step: 0-based draft step.
        topk: tree width (the previous ``speculative_topk``).
        check: validate device values (parent range, prefix sign). Off while the
            draft is under graph capture, where a host read is illegal; the
            structural checks (shape, step, topk) always run.

    Returns:
        ``(paths, positions)`` int32: ``paths`` is ``[bs, topk, step]`` (one
        ancestor slot per level ``0 .. step - 1``), ``positions`` is
        ``[bs, topk, step + 1]`` (those ancestors followed by self). Positions
        strictly ascend and no slot repeats inside a row.

    Raises:
        ValueError: bad ``step``/``topk``, a parent outside ``[0, topk)``, a
            negative prefix length, or shape/level mismatches.
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
    if prefix_len.dim() != 1:
        raise ValueError(f"prefix_len must be 1-D, got {prefix_len.shape}")
    if check and bool((prefix_len < 0).any()):
        raise ValueError("prefix_len must be non-negative")

    bs = int(prefix_len.numel())
    device = prefix_len.device
    prefix = prefix_len.to(torch.int64).to(device).view(bs, 1)
    slots = torch.arange(topk, device=device, dtype=torch.int64).view(1, topk)

    if step == 0:
        if parent_slot is not None and int(parent_slot.numel()):
            raise ValueError("step 0 has no parent level")
        if paths is not None and int(paths.numel()):
            raise ValueError("step 0 has no earlier level")
        self_pos = prefix + slots * num_steps  # branch-major, level 0
        return (
            torch.empty((bs, topk, 0), dtype=torch.int32, device=device),
            self_pos.unsqueeze(-1).to(torch.int32),
        )

    if parent_slot is None:
        raise ValueError(f"step {step} needs parent_slot")
    if tuple(parent_slot.shape) != (bs, topk):
        raise ValueError(
            f"parent_slot must be [bs={bs}, topk={topk}], got "
            f"{tuple(parent_slot.shape)}"
        )
    if paths is None or tuple(paths.shape) != (bs, topk, step - 1):
        raise ValueError(
            f"paths must be [bs={bs}, topk={topk}, step-1={step - 1}], got "
            f"{None if paths is None else tuple(paths.shape)}"
        )

    parent = parent_slot.to(torch.int64).to(device)
    if check and (bool((parent < 0).any()) or bool((parent >= topk).any())):
        raise ValueError(
            f"parent_slot must be in [0, {topk}), got "
            f"[{int(parent.min())}, {int(parent.max())}]"
        )

    paths = paths.to(torch.int64).to(device)
    # One level down: this row's ancestors = its parent's ancestors + the parent.
    gathered = torch.gather(
        paths, 1, parent.unsqueeze(-1).expand(bs, topk, step - 1)
    )
    new_paths = torch.cat([gathered, parent.unsqueeze(-1)], dim=-1)

    levels = torch.arange(step, device=device, dtype=torch.int64).view(1, 1, step)
    # Branch-major slots: branch t owns [prefix + t*num_steps, +num_steps).
    abs_ancestors = prefix.unsqueeze(-1) + new_paths * num_steps + levels
    self_pos = prefix + slots * num_steps + step
    positions = torch.cat([abs_ancestors, self_pos.unsqueeze(-1)], dim=-1)
    return new_paths.to(torch.int32), positions.to(torch.int32)


# Batch attributes the draft loop fills in and the NPU DSA indexer reads back.
# Same idiom as `forward_batch.npu_indexer_sin_cos_cache`: platform-local state
# that never enters a shared dataclass.
DRAFT_TREE_ANCESTORS_ATTR = "npu_draft_tree_ancestors"
DRAFT_TREE_PREFIX_ATTR = "npu_draft_tree_prefix_len"


def _parent_slots_from_parents_list(
    parents: torch.Tensor,
    step: int,
    topk: int,
    check: bool = True,
) -> torch.Tensor:
    """Recover one draft level's parent slot from ``select_top_k_tokens``.

    ``_select_top_k_tokens_later`` scores the ``topk * topk`` (parent slot,
    child) pairs and returns their flat index shifted by
    ``topk**2 * (step - 1) + topk``; ``draft_forward`` keeps that row in
    ``parents_list[step]``. Removing the shift recovers the flat score index,
    and ``// topk`` is the parent's slot in the previous level (``% topk`` picks
    the child). Inverting it here keeps the shared ``spec_utils`` untouched.

    Args:
        parents: ``[bs, topk]`` row of ``parents_list[step]`` (``step >= 1``).
        step: draft step whose parents are described (``1 <= step``).
        topk: tree width.
        check: validate the recovered range (host read; off under capture).

    Returns:
        ``[bs, topk]`` int32 parent slots in ``[0, topk)``.
    """
    step = int(step)
    topk = int(topk)
    if topk < 1:
        raise ValueError(f"topk must be >= 1, got {topk}")
    if step < 1:
        raise ValueError(f"parent slots need step >= 1, got {step}")
    if parents.dim() != 2:
        raise ValueError(f"parents must be 2-D, got {tuple(parents.shape)}")
    if int(parents.shape[1]) != topk:
        raise ValueError(
            f"parents must have topk={topk} columns, got {tuple(parents.shape)}"
        )

    offset = topk * topk * (step - 1) + topk
    flat = parents.to(torch.int64) - offset
    if check and (bool((flat < 0).any()) or bool((flat >= topk * topk).any())):
        raise ValueError(
            f"parent row out of range after removing offset {offset}: "
            f"[{int(flat.min())}, {int(flat.max())}]"
        )
    return (flat // topk).to(torch.int32)


class DraftTreeSparseState:
    """Per-step ancestor bookkeeping for the DSA draft tree (Ascend only).

    One instance per draft worker. ``begin_step`` is called from the shared
    draft loop through a three-line, attribute-guarded hook and leaves this
    step's absolute ancestor positions on ``forward_batch`` for the NPU DSA
    indexer. Nothing here runs on other platforms: the hook only fires when this
    object is installed, and it is installed only for a DSA draft with
    ``topk > 1``.

    All tensor work is device-side so the sequence survives graph capture; the
    only host reads are the optional value checks, which switch themselves off
    while a stream capture is active.
    """

    def __init__(self, topk: int, num_steps: int, check: bool = True) -> None:
        topk = int(topk)
        num_steps = int(num_steps)
        if topk < 1:
            raise ValueError(f"topk must be >= 1, got {topk}")
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        self.topk = topk
        self.num_steps = num_steps
        self.check = bool(check)
        self._paths = None

    def reset(self) -> None:
        """Forget the previous iteration (step 0 resets implicitly)."""
        self._paths = None

    def _validating(self) -> bool:
        # A host read inside a capture raises, so skip the optional checks there.
        if not self.check:
            return False
        probe = getattr(
            getattr(torch, "npu", None), "is_current_stream_capturing", None
        )
        if probe is None:
            return True
        try:
            return not bool(probe())
        except Exception:
            return True

    def begin_step(
        self,
        step: int,
        forward_batch,
        parents_list,
    ) -> torch.Tensor:
        """Compute this step's ancestor positions and stash them on the batch."""
        step = int(step)
        if not 0 <= step < self.num_steps:
            raise ValueError(f"draft step {step} outside [0, {self.num_steps})")
        if not parents_list or len(parents_list) <= step:
            raise ValueError(
                f"parents_list has {len(parents_list)} rows, need index {step}"
            )

        bs = int(parents_list[step].shape[0])
        validate = self._validating()
        if step == 0:
            parent_slot = None
        else:
            parent_slot = _parent_slots_from_parents_list(
                parents_list[step], step, self.topk, check=validate
            )

        seq_lens = getattr(forward_batch, "seq_lens", None)
        if seq_lens is None:
            raise ValueError("draft tree sparse needs forward_batch.seq_lens")
        if int(seq_lens.numel()) < bs:
            raise ValueError(
                f"seq_lens has {int(seq_lens.numel())} entries for {bs} requests"
            )
        prefix_len = seq_lens[:bs]

        self._paths, positions = draft_step_ancestor_positions(
            self._paths,
            parent_slot,
            prefix_len,
            step,
            self.topk,
            self.num_steps,
            check=validate,
        )
        setattr(forward_batch, DRAFT_TREE_ANCESTORS_ATTR, positions)
        setattr(forward_batch, DRAFT_TREE_PREFIX_ATTR, prefix_len)
        return positions
