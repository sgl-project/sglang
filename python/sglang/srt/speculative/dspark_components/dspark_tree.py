"""Backend-neutral core for the DSpark speculative-decoding draft tree.

This module is the single-file port of the reference ``tree`` package (shape,
scoring, layout, builder, acceptance, output, metadata, sparse, commit,
config, round). It is backend-neutral and GPU-portable: every function is pure
torch (plus ``dataclasses``/``typing`` and the stdlib) with no ``torch_npu`` /
``sgl_kernel_npu`` / ACL / vendor-kernel dependency, so it can be unit-tested
on CPU and later dispatched onto any device by the worker/verify integration.

Device tensors drive the fast path; the only device->host sync points are the
host readbacks (``.item()``, scalar ``int(...)``/``bool(...)`` conversions and
per-request Python loops), each annotated with ``# HOST-SYNC:``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

# ---------------------------------------------------------------------------
# shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TreeShape:
    """Draft depth, non-root node budget and per-level branching factor.

    ``depth`` is the DSpark draft block size (gamma); ``budget`` counts
    non-root nodes; ``topk`` bounds each retained level. Node 0 is the
    accepted root. This mirrors vllm-ascend's TreeShape contract.
    """

    depth: int
    budget: int
    topk: int
    dsv4_dspark: bool = True

    def __post_init__(self) -> None:
        if self.depth < 1 or self.budget < self.depth or self.topk < 1:
            raise ValueError(
                "Tree dimensions require depth >= 1, budget >= depth and topk >= 1"
            )

    @property
    def target_query_len(self) -> int:
        """Maximum target verify rows per request, including the root."""
        if self.topk == 1:
            return 1 + self.depth
        return 1 + self.budget

    def validate_selected_nodes(self, count: int) -> int:
        if not 0 <= count < self.target_query_len:
            raise ValueError(
                f"Selected tree nodes {count} exceed capacity "
                f"{self.target_query_len - 1}"
            )
        return count


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------


@dataclass
class TreeLayout:
    """Flattened draft tree for a batch of parallel-draft logits.

    Node ids include the root (root=0, draft nodes 1..N). ``tokens``,
    ``depths`` and ``parents`` are slot-indexed for non-root nodes.
    ``first_child`` and ``next_sibling`` are node-id-indexed and include the
    root. ``visibility[i, j]`` means non-root node i+1 may see non-root node
    j+1; seeing the root is implicit.
    """

    tokens: torch.Tensor
    depths: torch.Tensor
    parents: torch.Tensor
    num_nodes: torch.Tensor
    visibility: torch.Tensor
    first_child: torch.Tensor
    next_sibling: torch.Tensor


def _empty_layout(batch_size: int, budget: int, device) -> TreeLayout:
    return TreeLayout(
        tokens=torch.full((batch_size, budget), -1, dtype=torch.long, device=device),
        depths=torch.zeros((batch_size, budget), dtype=torch.long, device=device),
        parents=torch.full((batch_size, budget), -1, dtype=torch.long, device=device),
        num_nodes=torch.zeros(batch_size, dtype=torch.int32, device=device),
        visibility=torch.zeros(
            (batch_size, budget, budget), dtype=torch.bool, device=device
        ),
        first_child=torch.full(
            (batch_size, budget + 1), -1, dtype=torch.long, device=device
        ),
        next_sibling=torch.full(
            (batch_size, budget + 1), -1, dtype=torch.long, device=device
        ),
    )


def finalize_tree_layout_torch(
    out: TreeLayout,
    tokens: torch.Tensor,
    depths: torch.Tensor,
    parent_ids: torch.Tensor,
    num_nodes: int,
) -> TreeLayout:
    """Fill ``out`` with selected nodes and derive sibling links + visibility."""
    if num_nodes < 0 or num_nodes > out.tokens.shape[1]:
        raise ValueError(f"num_nodes {num_nodes} outside budget {out.tokens.shape[1]}")
    num_reqs = tokens.shape[0]
    out.tokens.fill_(-1)
    out.depths.zero_()
    out.parents.fill_(-1)
    out.visibility.zero_()
    out.first_child.fill_(-1)
    out.next_sibling.fill_(-1)

    if num_nodes:
        out.tokens[:, :num_nodes] = tokens[:, :num_nodes].to(out.tokens.dtype)
        out.depths[:, :num_nodes] = depths[:, :num_nodes].to(out.depths.dtype)
        out.parents[:, :num_nodes] = parent_ids[:, :num_nodes].to(out.parents.dtype)
    out.num_nodes[:] = num_nodes

    rows = torch.arange(num_reqs, device=parent_ids.device)
    for slot in range(num_nodes):
        node_id = slot + 1
        parent_id = parent_ids[:, slot].long()
        # first_child holds the most recently inserted child; next_sibling
        # chains back to the previously inserted one.
        previous_first = out.first_child.gather(1, parent_id.unsqueeze(1)).squeeze(1)
        out.next_sibling[:, node_id] = previous_first
        out.first_child[rows, parent_id] = node_id

        if slot > 0:
            src_idx = (parent_id - 1).clamp(min=0)
            parent_vis = out.visibility[:, :num_nodes, :slot].gather(
                1,
                src_idx.view(-1, 1, 1).expand(-1, 1, slot),
            ).squeeze(1)
            out.visibility[:, slot, :slot] = torch.where(
                (parent_id > 0).unsqueeze(-1),
                parent_vis,
                out.visibility[:, slot, :slot],
            )
        out.visibility[:, slot, slot] = True

    return out




# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def select_candidates(
    logits: torch.Tensor, width: int, *, keep_greedy: bool = False
):
    """Top-``width`` full-vocabulary edges with normalized log probabilities."""
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    scores, ids = log_probs.topk(width, dim=-1)
    greedy_child = None
    if keep_greedy:
        greedy = logits.argmax(-1)
        present = (ids == greedy.unsqueeze(-1)).any(-1)
        ids[..., -1] = torch.where(present, ids[..., -1], greedy)
        scores[..., -1] = log_probs.gather(-1, ids[..., -1:])[..., 0]
        greedy_child = (ids == greedy.unsqueeze(-1)).long().argmax(-1)
    return scores, ids, greedy_child


@dataclass
class CandidateLayer:
    scores: torch.Tensor
    token_ids: torch.Tensor
    greedy_child: torch.Tensor | None = None


class IndependentScorer:
    """Use per-depth base logits as-is (no Markov correction)."""

    name = "independent"
    depth_penalty = 0.0

    def __init__(self, topk: int):
        self.topk = int(topk)

    def score(
        self,
        base_logits: torch.Tensor,
        frontier_tokens: torch.Tensor,
        width: int,
        *,
        keep_greedy: bool = False,
    ) -> CandidateLayer:
        logits = base_logits.unsqueeze(1)
        scores, ids, greedy = select_candidates(logits, width, keep_greedy=keep_greedy)
        parents = frontier_tokens.shape[1]
        scores = scores.expand(-1, parents, -1)
        ids = ids.expand(-1, parents, -1)
        if greedy is not None:
            greedy = greedy.expand(-1, parents)
        return CandidateLayer(scores, ids, greedy)


class DSparkMarkovScorer:
    """Apply the DSpark Markov bias per frontier token, then select top-k."""

    name = "markov"
    depth_penalty = 0.0

    def __init__(self, head, topk: int):
        if getattr(head, "vocab_shard", None) is not None:
            raise NotImplementedError(
                "DSpark tree mode does not support TP-sharded markov_w2 "
                "(SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD=1); unset it or use topk=1."
            )
        self.head = head
        self.topk = int(topk)

    def score(
        self,
        base_logits: torch.Tensor,
        frontier_tokens: torch.Tensor,
        width: int,
        *,
        keep_greedy: bool = False,
    ) -> CandidateLayer:
        bias, state = self.head.step_bias(
            frontier_tokens, hidden_states=None, state=None
        )
        if state is not None:
            raise NotImplementedError(
                "DSpark tree mode supports stateless markov heads only; "
                "the head returned state (Gated/RNN markov)."
            )
        if bias.shape[-1] != base_logits.shape[-1]:
            raise ValueError(
                "markov bias width != base logits width; "
                "this usually means a TP-sharded vocab was left enabled."
            )
        logits = base_logits.unsqueeze(1) + bias
        scores, ids, greedy = select_candidates(logits, width, keep_greedy=keep_greedy)
        return CandidateLayer(scores, ids, greedy)


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------


@dataclass
class TreeBuildState:
    tokens: torch.Tensor  # [r, capacity] non-root node tokens
    depths: torch.Tensor  # [r, capacity] non-root node depths
    parents: torch.Tensor  # [r, capacity + 1] node-id indexed, root at 0
    path_scores: torch.Tensor  # [r, capacity + 1] cumulative log prob
    width: int
    num_nodes: int = 0


class TreeBuilder:
    """Build a tree of at most ``budget`` non-root nodes.

    At each draft depth the builder expands every frontier node by up to
    ``topk`` candidates and keeps the best ``width = min(topk, vocab, budget)``
    candidates globally. After all depths it keeps the best ``budget`` nodes by
    cumulative score. Because a child's cumulative score can never exceed its
    parent's, the kept set is ancestor-closed.
    """

    def __init__(self, budget: int, topk: int, depth: int, scorer=None):
        if budget < 1 or topk < 1 or depth < 1:
            raise ValueError("Tree budget, topk and depth must be positive")
        if budget < depth:
            raise ValueError("Tree budget must be >= draft depth")
        self.budget = int(budget)
        self.topk = int(topk)
        self.depth = int(depth)
        self.scorer = scorer if scorer is not None else IndependentScorer(topk)
        self.num_selected_nodes = 0

    def build(
        self,
        base_logits: torch.Tensor,
        root_token_ids: torch.Tensor,
        *,
        out: TreeLayout | None = None,
    ) -> TreeLayout:
        if base_logits.dim() != 3:
            raise ValueError("base_logits must be [batch, depth, vocab]")
        r, depth, vocab = base_logits.shape
        if depth != self.depth:
            raise ValueError(f"base_logits depth {depth} != builder depth {self.depth}")
        if root_token_ids.shape[0] != r:
            raise ValueError("root_token_ids batch mismatch")
        width = min(self.topk, vocab, self.budget)
        capacity = width * depth
        if self.budget > capacity:
            raise ValueError(
                f"tree budget {self.budget} exceeds builder capacity "
                f"topk * depth = {capacity} for topk={self.topk}, "
                f"depth={depth}, vocab={vocab}; increase topk/vocab or lower budget"
            )
        device = base_logits.device

        state = TreeBuildState(
            tokens=torch.full((r, capacity), -1, dtype=torch.long, device=device),
            depths=torch.zeros((r, capacity), dtype=torch.long, device=device),
            parents=torch.full(
                (r, capacity + 1), -1, dtype=torch.long, device=device
            ),
            path_scores=torch.zeros((r, capacity + 1), dtype=torch.float32, device=device),
            width=width,
        )
        state.parents[:, 0] = 0
        frontier = torch.zeros((r, 1), dtype=torch.long, device=device)

        for depth_slot in range(depth):
            if depth_slot == 0:
                frontier_tokens = root_token_ids.long().unsqueeze(1)
            else:
                frontier_tokens = state.tokens.gather(1, frontier - 1)
            layer = self.scorer.score(
                base_logits[:, depth_slot], frontier_tokens, width
            )
            parent_scores = state.path_scores.gather(1, frontier)
            flat = (parent_scores.unsqueeze(-1) + layer.scores).reshape(r, -1)
            selected = flat.argsort(dim=-1, descending=True, stable=True)[:, :width]

            start = state.num_nodes
            node_ids = torch.arange(
                start + 1, start + width + 1, device=device, dtype=torch.long
            )
            parent_slots = selected // width
            state.tokens[:, start : start + width] = layer.token_ids.reshape(
                r, -1
            ).gather(1, selected)
            state.depths[:, start : start + width] = depth_slot + 1
            state.parents[:, start + 1 : start + width + 1] = frontier.gather(
                1, parent_slots
            )
            state.path_scores[:, start + 1 : start + width + 1] = flat.gather(
                1, selected
            )
            state.num_nodes += width
            frontier = node_ids.unsqueeze(0).expand(r, -1)

        keep_count = min(self.budget, state.num_nodes)
        priority = state.path_scores[:, 1 : state.num_nodes + 1]
        self.num_selected_nodes = keep_count
        if out is None:
            out = _empty_layout(r, keep_count, device)
        if priority.device.type == "npu":
            try:
                from .dspark_tree_triton import (
                    finalize_tree_layout_triton,
                    prune_tree_layout_triton,
                )
            except ImportError:
                pass
            else:
                prune_tree_layout_triton(
                    priority,
                    state.tokens,
                    state.depths,
                    state.parents,
                    out.tokens,
                    out.depths,
                    out.parents,
                    keep_count,
                )
                out.num_nodes[:] = keep_count
                finalize_tree_layout_triton(
                    out.parents,
                    out.first_child,
                    out.next_sibling,
                    out.visibility,
                    keep_count,
                )
                return out
        kept = priority.argsort(dim=-1, descending=True, stable=True)[
            :, :keep_count
        ]
        kept_ids = (kept + 1).sort(dim=-1).values
        old_to_new = torch.zeros(
            (r, state.num_nodes + 1), dtype=torch.long, device=device
        )
        old_to_new.scatter_(
            1,
            kept_ids,
            torch.arange(1, keep_count + 1, device=device)
            .unsqueeze(0)
            .expand(r, -1),
        )
        new_parents = old_to_new.gather(1, state.parents.gather(1, kept_ids))
        new_tokens = state.tokens.gather(1, kept_ids - 1)
        new_depths = state.depths.gather(1, kept_ids - 1)

        return finalize_tree_layout_torch(
            out, new_tokens, new_depths, new_parents, keep_count
        )


# ---------------------------------------------------------------------------
# acceptance
# ---------------------------------------------------------------------------


@dataclass
class TreeAcceptance:
    path_node_ids: torch.Tensor  # [r, budget], -1 padded
    num_sampled: torch.Tensor  # [r], includes bonus
    num_accepted: torch.Tensor  # [r], draft nodes only
    endpoint_node: torch.Tensor  # [r], last accepted node or 0
    bonus: torch.Tensor  # [r], target token after the endpoint
    active: torch.Tensor  # [r] bool


def greedy_tree_accept(
    tree,
    target_predict: torch.Tensor,
    *,
    active: torch.Tensor | None = None,
) -> TreeAcceptance:
    """Walk the draft tree against per-node target argmax tokens.

    ``target_predict`` is [batch, budget + 1]; column 0 is the root's target
    argmax and column i+1 is node i's target argmax.
    """
    if tree.tokens.dim() != 2:
        raise ValueError("tree.tokens must be [batch, budget]")
    batch, budget = tree.tokens.shape
    if target_predict.shape != (batch, budget + 1):
        raise ValueError(
            f"target_predict must be [{batch}, {budget + 1}], "
            f"got {tuple(target_predict.shape)}"
        )
    device = tree.tokens.device
    rows = torch.arange(batch, device=device)
    if active is None:
        active = torch.ones(batch, dtype=torch.bool, device=device)
    alive = active.to(torch.bool).clone()
    current = torch.zeros(batch, dtype=torch.long, device=device)
    path = torch.full((batch, budget), -1, dtype=torch.long, device=device)
    num_accepted = torch.zeros(batch, dtype=torch.int32, device=device)
    num_sampled = active.to(torch.int32).clone()

    for step in range(budget):
        target = target_predict[rows, current.clamp(0, budget)]
        child = tree.first_child[rows, current.clamp(0, budget)].long()
        found = torch.full_like(current, -1)
        for _ in range(budget):
            linked = alive & (found < 0) & (child > 0) & (child <= budget)
            token = tree.tokens[rows, (child - 1).clamp(0, budget - 1)]
            match = linked & (token == target)
            found = torch.where(match, child, found)
            nxt = tree.next_sibling[rows, child.clamp(0, budget)].long()
            child = torch.where(linked & ~match, nxt, child)
        matched = alive & (found > 0)
        path[:, step] = torch.where(matched, found, -1)
        current = torch.where(matched, found, current)
        num_accepted += matched.to(torch.int32)
        num_sampled += matched.to(torch.int32)
        alive = matched

    bonus = target_predict[rows, current.clamp(0, budget)]
    bonus = torch.where(active, bonus, torch.zeros_like(bonus))
    endpoint = torch.where(active, current, torch.zeros_like(current))
    return TreeAcceptance(
        path_node_ids=path,
        num_sampled=num_sampled,
        num_accepted=num_accepted,
        endpoint_node=endpoint,
        bonus=bonus,
        active=active.to(torch.bool),
    )


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


@dataclass
class TreeAcceptOuts:
    out_tokens: torch.Tensor  # [r, query_width]
    accept_index: torch.Tensor  # [r, query_width], flat verify-row indices
    commit_lens: torch.Tensor  # [r], root + accepted drafts
    new_seq_lens: torch.Tensor  # [r], prefix + commit_lens
    num_accepted: torch.Tensor  # [r]


def build_tree_accept_outs(
    verify_ids: torch.Tensor,
    path_node_ids: torch.Tensor,
    bonus: torch.Tensor,
    prefix_lens: torch.Tensor,
) -> TreeAcceptOuts:
    """Gather accepted node tokens in depth order and append the bonus token.

    ``verify_ids`` is [r, 1 + budget] with column 0 the accepted root/anchor.
    ``path_node_ids`` is [r, budget] with node ids in 1..budget or -1.
    """
    if verify_ids.dim() != 2 or path_node_ids.dim() != 2:
        raise ValueError("verify_ids and path_node_ids must be 2-D")
    batch, query_width = verify_ids.shape
    if path_node_ids.shape[0] != batch:
        raise ValueError("path_node_ids batch mismatch")
    if bonus.shape[0] != batch or prefix_lens.shape[0] != batch:
        raise ValueError("bonus/prefix_lens batch mismatch")
    if query_width != path_node_ids.shape[1] + 1:
        raise ValueError(
            "verify_ids width must be 1 + budget "
            f"(got {query_width} vs budget {path_node_ids.shape[1]})"
        )

    device = verify_ids.device
    rows = torch.arange(batch, device=device)
    valid = path_node_ids >= 0
    num_accepted = valid.sum(dim=1).to(torch.int32)

    out_tokens = torch.zeros((batch, query_width), dtype=verify_ids.dtype, device=device)
    accept_index = torch.full(
        (batch, query_width), -1, dtype=torch.long, device=device
    )
    depth_count = path_node_ids.shape[1]
    if depth_count:
        nodes = path_node_ids.clamp(min=0).long()
        gathered_tokens = verify_ids.gather(1, nodes)
        out_tokens[:, :depth_count] = torch.where(
            valid, gathered_tokens, out_tokens[:, :depth_count]
        )
        accept_index[:, :depth_count] = torch.where(
            valid,
            rows.unsqueeze(1) * query_width + nodes,
            accept_index[:, :depth_count],
        )

    out_tokens[rows, num_accepted.long()] = bonus
    commit_lens = num_accepted + 1
    new_seq_lens = prefix_lens.to(torch.int64) + commit_lens.to(torch.int64)
    return TreeAcceptOuts(
        out_tokens=out_tokens,
        accept_index=accept_index,
        commit_lens=commit_lens,
        new_seq_lens=new_seq_lens,
        num_accepted=num_accepted,
    )


def gather_accepted_rows(tensor: torch.Tensor, accept_index: torch.Tensor) -> torch.Tensor:
    """Gather each row's accepted path into the leading columns.

    ``tensor`` is [bs, query_width, ...]; ``accept_index`` is [bs, query_width]
    of flat verify-row indices (row * query_width + node) or -1. Padding columns
    are clamped to column 0 and are dropped by the commit-length gate.
    """
    if tensor.dim() < 2 or tensor.shape[:2] != accept_index.shape:
        raise ValueError(
            f"tensor leading shape {tuple(tensor.shape[:2])} != "
            f"accept_index {tuple(accept_index.shape)}"
        )
    query_width = accept_index.shape[1]
    local = torch.where(
        accept_index >= 0, accept_index % query_width, accept_index.new_zeros(())
    ).to(torch.long)
    rows = torch.arange(accept_index.shape[0], device=accept_index.device).unsqueeze(1)
    return tensor[rows, local]


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------


@dataclass
class TreeMetadata:
    positions: torch.Tensor  # [T] logical position per query node
    node_seqused_kv: torch.Tensor  # [T] visible KV length per node
    verify_ids: torch.Tensor  # [T] query token ids (anchor + nodes)
    tree_prefix: torch.Tensor  # [T] per-node prefix base
    tree_ancestors: torch.Tensor  # [T, A] root-first node ids, -1 padded
    node_req_ids: torch.Tensor  # [T] request index per node
    node_cu_seqlens_q: torch.Tensor  # [T + 1]
    request_cu_seqlens_q: torch.Tensor  # [bs + 1]
    request_seqused_kv: torch.Tensor  # [bs]
    query_width: int
    num_nodes: int


def _build_ancestor_table(layout, max_depth: int) -> torch.Tensor:
    """Build ``[batch, query_width, max_depth + 1]`` root-first node ids.

    The parent walk is done with ``max_depth`` device-side gathers instead of
    one device readback per request/node.  Padded nodes keep the existing
    ``[0, node_id]`` behavior: a missing parent stops the chain at the root.
    """
    batch, budget = layout.tokens.shape
    query_width = budget + 1
    max_len = max_depth + 1
    device = layout.parents.device
    if budget == 0:
        return torch.zeros((batch, query_width, 1), dtype=torch.long, device=device)

    node_ids = torch.arange(query_width, dtype=torch.long, device=device).expand(
        batch, -1
    )
    chain = torch.zeros(
        (batch, query_width, max_len), dtype=torch.long, device=device
    )
    chain[:, :, 0] = node_ids
    chain_len = (node_ids > 0).long() + 1
    active = node_ids > 0
    cur = node_ids
    parent_index = (cur - 1).clamp(min=0, max=budget - 1)

    for step in range(1, max_len):
        parent_raw = layout.parents.long().gather(1, parent_index)
        parent = torch.where(
            parent_raw > 0, parent_raw, torch.zeros_like(parent_raw)
        )
        chain[:, :, step] = torch.where(
            cur > 0, parent, torch.zeros_like(parent)
        )
        positive = active & (parent_raw > 0)
        chain_len = chain_len + positive.long()
        cur = torch.where(positive, parent, cur)
        active = positive
        parent_index = (cur - 1).clamp(min=0, max=budget - 1)

    if bool((chain_len > max_len).any()):
        raise ValueError(
            f"tree depth exceeds ancestor slots ({max_len}); "
            "max_depth underestimates the layout"
        )

    reversed_chain = chain.flip(-1)
    cols = torch.arange(max_len, dtype=torch.long, device=device)
    gather_idx = (max_len - chain_len.unsqueeze(-1) + cols).clamp(
        min=0, max=max_len - 1
    )
    ancestors = reversed_chain.gather(-1, gather_idx)
    return torch.where(
        cols < chain_len.unsqueeze(-1),
        ancestors,
        torch.full_like(ancestors, -1),
    )


def build_tree_metadata(
    layout,
    *,
    prefix_lens: torch.Tensor,
    anchor_tokens: torch.Tensor,
    num_nodes: int | None = None,
) -> TreeMetadata:
    """Build per-node verify geometry for a finalized TreeLayout."""
    batch, budget = layout.tokens.shape
    if prefix_lens.shape[0] != batch or anchor_tokens.shape[0] != batch:
        raise ValueError("prefix_lens/anchor_tokens batch mismatch")
    device = layout.tokens.device
    query_width = budget + 1
    if num_nodes is None:
        num_nodes = int(layout.num_nodes.max().item()) if batch else 0  # HOST-SYNC: max readback
    max_depth = int(layout.depths.max().item()) if budget else 0  # HOST-SYNC: max readback

    depths = torch.cat(
        (
            torch.zeros((batch, 1), dtype=torch.long, device=device),
            layout.depths.long(),
        ),
        dim=1,
    )
    verify_ids_2d = torch.cat(
        (anchor_tokens.long().unsqueeze(1), layout.tokens.long()), dim=1
    )
    prefix_2d = prefix_lens.long().unsqueeze(1).expand(batch, query_width)
    positions_2d = prefix_2d + depths
    seqused_2d = positions_2d + 1

    ancestors = _build_ancestor_table(layout, max_depth)

    node_req_ids = torch.arange(batch, device=device).repeat_interleave(query_width)
    node_cu = torch.arange(
        batch * query_width + 1, dtype=torch.int32, device=device
    )
    request_cu = (
        torch.arange(batch + 1, dtype=torch.int32, device=device) * query_width
    )
    request_seqused = prefix_lens.to(torch.int32) + query_width
    return TreeMetadata(
        positions=positions_2d.reshape(-1),
        node_seqused_kv=seqused_2d.reshape(-1).to(torch.int32),
        verify_ids=verify_ids_2d.reshape(-1),
        tree_prefix=prefix_2d.reshape(-1).to(torch.int32),
        tree_ancestors=ancestors.reshape(batch * query_width, max_depth + 1),
        node_req_ids=node_req_ids.to(torch.int32),
        node_cu_seqlens_q=node_cu,
        request_cu_seqlens_q=request_cu,
        request_seqused_kv=request_seqused,
        query_width=query_width,
        num_nodes=num_nodes,
    )




# ---------------------------------------------------------------------------
# sparse
# ---------------------------------------------------------------------------


def _align_up(value: int, alignment: int = 128) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def build_tree_ori_sparse_indices(
    layout,
    *,
    prefix_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    window: int,
    budget: int | None = None,
    full_to_swa_mapping: torch.Tensor | None = None,
    draft_slots: torch.Tensor | None = None,
    ancestors: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return [T, 1, W] int32 physical slots, W 128-aligned.

    ``block_table`` indexes the full KV pool. When ``full_to_swa_mapping`` is
    supplied, each full-pool slot is translated to its SWA slot; the attention
    op reads ``ori_kv`` from the SWA pool. ``draft_slots`` supplies this
    round's verify write slots for positions at/after the prefix; those rows
    are not visible in ``req_to_token`` yet.
    """
    batch, layout_budget = layout.tokens.shape
    budget = layout_budget if budget is None else int(budget)
    if block_size < 1 or window < 1:
        raise ValueError("block_size and window must be positive")
    if budget > layout_budget:
        raise ValueError("sparse budget exceeds layout budget")
    if prefix_lens.shape[0] != batch:
        raise ValueError("prefix_lens batch mismatch")
    if block_table.shape[0] != batch:
        raise ValueError("block_table batch mismatch")
    query_width = budget + 1
    width = _align_up(window + budget, 128)
    device = prefix_lens.device
    if ancestors is None:
        max_depth = int(layout.depths.max().item()) if layout_budget else 0
        ancestors = _build_ancestor_table(layout, max_depth)
    else:
        ancestors = ancestors.to(device=device, dtype=torch.long)
        if ancestors.dim() != 3 or ancestors.shape[:2] != (batch, query_width):
            raise ValueError(
                "ancestors must be [batch, query_width, max_depth + 1]"
            )

    prefix = prefix_lens.to(torch.long)
    depths = torch.cat(
        (
            torch.zeros((batch, 1), dtype=torch.long, device=device),
            layout.depths[:, :budget].long(),
        ),
        dim=1,
    )
    start = (prefix.unsqueeze(1) + depths - window).clamp(min=0)
    hist_positions = start.unsqueeze(-1) + torch.arange(
        width, dtype=torch.long, device=device
    )
    hist_valid = hist_positions < prefix.unsqueeze(1).unsqueeze(1)
    hist_len = (prefix.unsqueeze(1) - start).clamp(min=0, max=width)

    table = block_table.long().unsqueeze(1).expand(-1, query_width, -1)
    hist_page_idx = (hist_positions // block_size).clamp(
        min=0, max=block_table.shape[1] - 1
    )
    hist_page = table.gather(2, hist_page_idx)
    hist_full = hist_page * block_size + (hist_positions % block_size)
    hist_swa = hist_full
    if full_to_swa_mapping is not None:
        mapping = full_to_swa_mapping.long()
        hist_swa = mapping[
            hist_swa.clamp(min=-1, max=mapping.numel() - 1)
        ]

    slot_values = torch.where(
        hist_valid,
        hist_swa,
        torch.full_like(hist_swa, -1),
    ).to(torch.int32)
    slots = slot_values

    ancestor_ids = ancestors.to(torch.long)
    ancestor_valid = ancestor_ids >= 0
    ancestor_node = ancestor_ids.clamp(min=0)
    ancestor_pos = prefix.unsqueeze(1).unsqueeze(1) + ancestor_node
    if draft_slots is not None:
        if bool((ancestor_ids >= draft_slots.shape[1]).any()):
            raise ValueError(
                f"draft slot column out of range {draft_slots.shape[1]}"
            )
        draft = draft_slots.long().unsqueeze(1).expand(-1, query_width, -1)
        ancestor_full = draft.gather(2, ancestor_node)
    else:
        ancestor_page_idx = (ancestor_pos // block_size).clamp(
            min=0, max=block_table.shape[1] - 1
        )
        ancestor_page = table.gather(2, ancestor_page_idx)
        ancestor_full = ancestor_page * block_size + (
            ancestor_pos % block_size
        )
    if full_to_swa_mapping is not None:
        ancestor_full = full_to_swa_mapping.long()[
            ancestor_full.clamp(min=-1, max=full_to_swa_mapping.numel() - 1)
        ]

    ancestor_cols = hist_len.unsqueeze(-1) + torch.arange(
        ancestors.shape[-1], dtype=torch.long, device=device
    )
    if bool((ancestor_valid & (ancestor_cols >= width)).any()):
        raise ValueError("tree row sees more positions than available slots")
    slots = slots.scatter(
        2,
        ancestor_cols.clamp(max=width - 1),
        torch.where(
            ancestor_valid,
            ancestor_full,
            torch.full_like(ancestor_full, -1),
        ).to(torch.int32),
    )
    return slots.reshape(batch * query_width, 1, width)


# ---------------------------------------------------------------------------
# commit
# ---------------------------------------------------------------------------


@dataclass
class TreeCommitPlan:
    source_rows: torch.Tensor  # [r, max_rows] flat verify rows to move
    dest_positions: torch.Tensor  # [r, max_rows] logical history positions
    dest_slots: torch.Tensor  # [r, max_rows] physical cache slots
    commit_lens: torch.Tensor  # [r]
    new_seq_lens: torch.Tensor  # [r]
    state_source: torch.Tensor | None = None
    state_dest: torch.Tensor | None = None


def plan_tree_commit(
    *,
    path_node_ids: torch.Tensor,
    verify_cache_loc: torch.Tensor,
    prefix_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
) -> TreeCommitPlan:
    if path_node_ids.dim() != 2 or verify_cache_loc.dim() != 2:
        raise ValueError("path_node_ids and verify_cache_loc must be 2-D")
    batch = path_node_ids.shape[0]
    if verify_cache_loc.shape[0] != batch:
        raise ValueError("verify_cache_loc batch mismatch")
    if prefix_lens.shape[0] != batch or block_table.shape[0] != batch:
        raise ValueError("prefix_lens/block_table batch mismatch")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    device = path_node_ids.device
    valid = path_node_ids >= 0
    num_accepted = valid.sum(dim=1).to(torch.int64)
    max_rows = path_node_ids.shape[1] + 1

    source_rows = torch.full((batch, max_rows), -1, dtype=torch.long, device=device)
    dest_positions = torch.full((batch, max_rows), -1, dtype=torch.long, device=device)
    dest_slots = torch.full((batch, max_rows), -1, dtype=torch.long, device=device)

    source_rows[:, 0] = verify_cache_loc[:, 0]
    dest_positions[:, 0] = prefix_lens.long()
    # The target verify already wrote the root/path rows into this window.
    # Use those columns as destinations instead of re-reading req_to_token,
    # whose tree-depth columns may still be -1.
    dest_slots[:, 0] = verify_cache_loc[:, 0]

    depth_count = max_rows - 1
    if depth_count:
        depth_offsets = torch.arange(
            depth_count, dtype=torch.long, device=device
        ).unsqueeze(0)
        depth_valid = depth_offsets < num_accepted.unsqueeze(1)
        path_nodes = path_node_ids[:, :depth_count].clamp(min=0).long()
        source_rows[:, 1:] = torch.where(
            depth_valid,
            verify_cache_loc.gather(1, path_nodes),
            source_rows[:, 1:],
        )
        dest_positions[:, 1:] = torch.where(
            depth_valid,
            prefix_lens.long().unsqueeze(1) + depth_offsets + 1,
            dest_positions[:, 1:],
        )
        dest_slots[:, 1:] = torch.where(
            depth_valid,
            verify_cache_loc[:, 1:max_rows],
            dest_slots[:, 1:],
        )

    commit_lens = num_accepted + 1
    new_seq_lens = prefix_lens.to(torch.int64) + commit_lens
    return TreeCommitPlan(
        source_rows=source_rows,
        dest_positions=dest_positions,
        dest_slots=dest_slots,
        commit_lens=commit_lens,
        new_seq_lens=new_seq_lens,
    )






# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TreeConfig:
    enabled: bool
    depth: int
    budget: int
    topk: int
    verify_width: int


@dataclass(frozen=True)
class DSparkWidths:
    """Draft depth and target-verify width are two different quantities.

    ``gamma`` is the draft horizon (``--speculative-dspark-block-size``).
    ``verify_width`` is the number of target-verify rows per request; it is
    ``1 + budget`` for a tree and ``gamma + 1`` for a chain.
    """

    gamma: int
    verify_width: int
    tree: TreeConfig | None


def resolve_dspark_widths(
    *,
    explicit_block_size: int | None,
    checkpoint_gamma: int | None,
    num_draft_tokens: int | None,
    tree_enable: bool,
    tree_topk: int | None,
    tree_budget: int | None,
    ragged_mode: str = "static",
    default_gamma: int = 7,
) -> DSparkWidths:
    """Resolve gamma and the target-verify width together.

    A tree with ``topk > 1`` may have ``budget > gamma``; its verify width is
    then ``1 + budget`` while the draft model still proposes ``gamma`` tokens.
    A chain (``topk == 1``) keeps ``budget == gamma`` and ``gamma + 1`` rows.
    Non-tree DSpark keeps ``gamma + 1`` rows.
    """
    gamma = explicit_block_size if explicit_block_size is not None else checkpoint_gamma
    if tree_enable:
        if gamma is None:
            raise ValueError(
                "DSpark tree mode requires a known draft depth (gamma): set "
                "--speculative-dspark-block-size or use a draft checkpoint with "
                "a block_size."
            )
        tree = resolve_tree_config(
            enable=True,
            topk=tree_topk,
            budget=tree_budget,
            depth=int(gamma),
            ragged_mode=ragged_mode,
        )
        assert tree is not None
        verify_width = tree.verify_width
    else:
        if gamma is None:
            if num_draft_tokens is not None:
                gamma = int(num_draft_tokens) - 1
            else:
                gamma = int(default_gamma)
        gamma = int(gamma)
        if gamma < 1:
            raise ValueError(
                "DSpark gamma must be >= 1, got "
                f"{gamma} (num_draft_tokens={num_draft_tokens})."
            )
        tree = None
        verify_width = gamma + 1

    if num_draft_tokens is not None and int(num_draft_tokens) != verify_width:
        width_src = (
            f"1 + tree budget (={verify_width})"
            if tree is not None and tree.topk > 1
            else f"gamma + 1 (={verify_width})"
        )
        raise ValueError(
            "DSpark speculative_num_draft_tokens must equal "
            f"{width_src}, but got {num_draft_tokens}."
        )

    return DSparkWidths(gamma=int(gamma), verify_width=int(verify_width), tree=tree)


def resolve_draft_query_width(*, gamma: int, sample_from_anchor: bool) -> int:
    """Draft-block query rows: anchor sampling uses gamma; otherwise gamma + 1."""
    return int(gamma) if sample_from_anchor else int(gamma) + 1


def resolve_tree_config(
    *,
    enable: bool,
    topk: int | None,
    budget: int | None,
    depth: int,
    ragged_mode: str = "static",
) -> TreeConfig | None:
    """Resolve and validate the tree knobs. Returns None when disabled.

    ``topk`` is the per-level branching factor; ``budget`` counts non-root
    nodes. topk=1 is the chain: it requires budget == depth and does not use
    the tree attention path. For topk>1, budget may exceed depth, up to the
    builder capacity ``topk * depth``; the target-verify width is ``1 + budget``.
    """
    if not enable:
        return None
    if budget is None:
        raise ValueError("DSpark tree mode requires --speculative-dspark-tree-budget")
    if topk is None:
        topk = 2
    topk = int(topk)
    budget = int(budget)
    depth = int(depth)
    if topk < 1:
        raise ValueError(f"tree topk must be >= 1, got {topk}")
    if depth < 1:
        raise ValueError(f"tree depth must be >= 1, got {depth}")
    if budget < depth:
        raise ValueError(f"tree budget {budget} must be >= draft depth {depth}")
    if topk == 1 and budget != depth:
        raise ValueError(
            f"topk=1 is the chain and requires budget == depth ({depth}), got {budget}"
        )
    if topk > 1 and budget > topk * depth:
        raise ValueError(
            f"tree budget {budget} exceeds builder capacity topk * depth = "
            f"{topk * depth}; increase --speculative-eagle-topk or lower the budget"
        )
    if ragged_mode != "static":
        raise ValueError(
            f"DSpark tree requires SGLANG_RAGGED_VERIFY_MODE=static, got {ragged_mode!r}"
        )
    verify_width = 1 + (depth if topk == 1 else budget)
    return TreeConfig(
        enabled=True,
        depth=depth,
        budget=budget,
        topk=topk,
        verify_width=verify_width,
    )


def resolve_tree_for_server(server_args, *, gamma: int, ragged_mode: str = "static"):
    """Resolve tree knobs from a ServerArgs-like object (getattr-safe)."""
    return resolve_tree_config(
        enable=bool(getattr(server_args, "speculative_dspark_tree_enable", False)),
        topk=getattr(server_args, "speculative_eagle_topk", None),
        budget=getattr(server_args, "speculative_dspark_tree_budget", None),
        depth=gamma,
        ragged_mode=ragged_mode,
    )


# ---------------------------------------------------------------------------
# round
# ---------------------------------------------------------------------------


@dataclass
class TreePrePlan:
    layout: TreeLayout
    metadata: TreeMetadata
    ori_sparse_indices: torch.Tensor
    max_prefix: int | None = None
    node_records: list = field(default_factory=list)
    tree_node_kv: dict = field(default_factory=dict)
    tree_node_scale: dict = field(default_factory=dict)


@dataclass
class TreePostPlan:
    acceptance: TreeAcceptance
    outputs: TreeAcceptOuts
    commit: TreeCommitPlan




def plan_tree_pre(
    *,
    base_logits: torch.Tensor,
    root_token_ids: torch.Tensor,
    prefix_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    window: int,
    config: TreeConfig,
    scorer=None,
    full_to_swa_mapping: torch.Tensor | None = None,
    draft_slots: torch.Tensor | None = None,
    max_prefix: int | None = None,
) -> TreePrePlan:
    """Build topology, per-node verify geometry and sparse ori slots.

    Shapes:
      base_logits     [bs, depth, vocab]
      root_token_ids  [bs]
      prefix_lens     [bs]
    """
    if not config.enabled:
        raise ValueError("plan_tree_pre requires an enabled TreeConfig")
    scorer = scorer if scorer is not None else IndependentScorer(config.topk)
    builder = TreeBuilder(
        budget=config.budget, topk=config.topk, depth=config.depth, scorer=scorer
    )
    layout = builder.build(base_logits, root_token_ids)
    metadata = build_tree_metadata(
        layout,
        prefix_lens=prefix_lens,
        anchor_tokens=root_token_ids,
        num_nodes=builder.num_selected_nodes,
    )
    if metadata.query_width != config.verify_width:
        raise ValueError(
            f"resolved verify width {config.verify_width} != "
            f"layout width {metadata.query_width}"
        )
    sparse = build_tree_ori_sparse_indices(
        layout,
        prefix_lens=prefix_lens,
        block_table=block_table,
        block_size=block_size,
        window=window,
        budget=config.budget,
        full_to_swa_mapping=full_to_swa_mapping,
        draft_slots=draft_slots,
        ancestors=metadata.tree_ancestors.reshape(
            layout.tokens.shape[0], metadata.query_width, -1
        ),
    )
    return TreePrePlan(
        layout=layout,
        metadata=metadata,
        ori_sparse_indices=sparse,
        max_prefix=max_prefix,
    )


def plan_tree_post(
    *,
    pre: TreePrePlan,
    target_predict: torch.Tensor,
    verify_cache_loc: torch.Tensor,
    prefix_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
) -> TreePostPlan:
    """Walk the tree against target argmax tokens and build output/commit plans."""
    batch = pre.layout.tokens.shape[0]
    if target_predict.shape != (batch, pre.metadata.query_width):
        raise ValueError(
            f"target_predict must be [{batch}, {pre.metadata.query_width}], "
            f"got {tuple(target_predict.shape)}"
        )
    acceptance = greedy_tree_accept(pre.layout, target_predict)
    outputs = build_tree_accept_outs(
        verify_ids=pre.metadata.verify_ids.reshape(batch, -1),
        path_node_ids=acceptance.path_node_ids,
        bonus=acceptance.bonus,
        prefix_lens=prefix_lens,
    )
    commit = plan_tree_commit(
        path_node_ids=acceptance.path_node_ids,
        verify_cache_loc=verify_cache_loc,
        prefix_lens=prefix_lens,
        block_table=block_table,
        block_size=block_size,
    )
    return TreePostPlan(acceptance=acceptance, outputs=outputs, commit=commit)




__all__ = [
    # shape
    "TreeShape",
    # layout
    "TreeLayout",
    "finalize_tree_layout_torch",
    # scoring
    "select_candidates",
    "CandidateLayer",
    "IndependentScorer",
    "DSparkMarkovScorer",
    # builder
    "TreeBuilder",
    "TreeBuildState",
    # acceptance
    "TreeAcceptance",
    "greedy_tree_accept",
    # output
    "TreeAcceptOuts",
    "build_tree_accept_outs",
    "gather_accepted_rows",
    # metadata
    "TreeMetadata",
    "build_tree_metadata",
    # sparse
    "build_tree_ori_sparse_indices",
    # commit
    "TreeCommitPlan",
    "plan_tree_commit",
    # config
    "TreeConfig",
    "DSparkWidths",
    "resolve_tree_config",
    "resolve_dspark_widths",
    "resolve_draft_query_width",
    "resolve_tree_for_server",
    # round
    "TreePrePlan",
    "TreePostPlan",
    "plan_tree_pre",
    "plan_tree_post",
]
