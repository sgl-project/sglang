from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Sequence

import numpy as np


@dataclass(eq=False)
class _TrieNode:
    children: Dict[int, "_TrieNode"] = field(default_factory=dict)
    fail: "_TrieNode | None" = None
    token: int = -1
    freq: int = 0
    depth: int = 0
    parent: "_TrieNode | None" = None


class _DraftTree:
    """Fixed-K proposal tree grown in place like NGRAM_LOGITS padding fill."""

    def __init__(self, budget: int):
        self.budget = int(budget)
        self.tokens = np.zeros(self.budget, dtype=np.int64)
        self.mask = np.zeros((self.budget, self.budget), dtype=np.bool_)
        self.children: dict[tuple[int, int], int] = {}
        self.root_indices: dict[int, int] = {}
        self.active = 0

    def add(self, parent: int, token: int) -> int | None:
        """Insert ``token`` under ``parent``, reusing an existing edge for free."""

        token = int(token)
        if parent < 0:
            idx = self.root_indices.get(token)
            if idx is not None:
                return idx
            if self.active >= self.budget:
                return None
            idx = self.active
            self.tokens[idx] = token
            self.mask[idx, idx] = True
            self.root_indices[token] = idx
            self.active += 1
            return idx

        key = (parent, token)
        idx = self.children.get(key)
        if idx is not None:
            return idx
        if self.active >= self.budget:
            return None
        idx = self.active
        self.tokens[idx] = token
        self.mask[idx, :] = self.mask[parent, :]
        self.mask[idx, idx] = True
        self.children[key] = idx
        self.active += 1
        return idx

    def pad_with_zeros(self) -> None:
        """Fill leftover slots with dummy token-0 children of the draft root."""

        while self.active < self.budget:
            idx = self.active
            self.tokens[idx] = 0
            self.mask[idx, :] = False
            self.mask[idx, 0] = True
            self.mask[idx, idx] = True
            self.active += 1


class RacerAutomaton:
    """Per-request RACER Automaton with NGRAM_LOGITS-style Logits Tree fill.

    Retrieval Tree selection and AC fail transitions keep the original RACER
    semantics. After retrieval occupies the draft trie, leftover fixed-K
    slots are filled in place from copy-logit TokenBin successors. Existing
    retrieval edges are reused for free. A node with no TokenBin outgoing
    edge grows one token-0 placeholder; any residual capacity becomes dummy
    token-0 root-children so TARGET_VERIFY always sees exactly K nodes.
    """

    def __init__(
        self,
        *,
        ngram: int = 10,
        topk: int = 9,
        max_nodes: int = 10_000,
        min_depth: int = 2,
    ):
        self.ngram = max(1, int(ngram))
        self.topk = max(1, int(topk))
        self.max_nodes = max(1, int(max_nodes))
        self.min_depth = max(1, int(min_depth))
        self.root = _TrieNode()
        self.root.fail = self.root
        self._node_count = 0
        self._history: list[int] = []
        self._token_bin: dict[int, list[int]] = {}
        self._cur_state = self.root
        self._last_real_count = 0

    def reset(self) -> None:
        self.root = _TrieNode()
        self.root.fail = self.root
        self._node_count = 0
        self._history.clear()
        self._token_bin.clear()
        self._cur_state = self.root
        self._last_real_count = 0

    def _new_child(self, parent: _TrieNode, token: int) -> _TrieNode | None:
        if self._node_count >= self.max_nodes:
            return None
        child = _TrieNode(
            token=int(token),
            depth=parent.depth + 1,
            parent=parent,
            fail=self.root,
        )
        parent.children[int(token)] = child
        self._node_count += 1
        return child

    def insert(self, pattern: Sequence[int]) -> None:
        if not pattern:
            return
        node = self.root
        node.freq += 1
        for token in pattern:
            token = int(token)
            child = node.children.get(token)
            if child is None:
                child = self._new_child(node, token)
                if child is None:
                    return
            node = child
            node.freq += 1

    def _build_fail_links(self) -> None:
        self.root.fail = self.root
        q: deque[_TrieNode] = deque()
        for child in self.root.children.values():
            child.fail = self.root
            q.append(child)

        while q:
            cur = q.popleft()
            for token, child in cur.children.items():
                f = cur.fail if cur.fail is not None else self.root
                while f is not self.root and token not in f.children:
                    f = f.fail if f.fail is not None else self.root
                if token in f.children and f.children[token] is not child:
                    child.fail = f.children[token]
                else:
                    child.fail = self.root
                q.append(child)

    def _transition_from(self, state: _TrieNode, token: int) -> _TrieNode:
        token = int(token)
        node = state
        while node is not self.root and token not in node.children:
            node = node.fail if node.fail is not None else self.root
        return node.children.get(token, self.root)

    def _recompute_state(self, tokens: Sequence[int]) -> None:
        state = self.root
        for token in tokens:
            state = self._transition_from(state, int(token))
        self._cur_state = state

    def sync_history(self, tokens: Sequence[int]) -> None:
        """Synchronize the AC state to context immediately before next_token.

        Original RACER inserts all prompt n-grams, then after every accepted step
        inserts the newly exposed n-grams and advances the AC state.  Rebuilding
        fail links after incremental insertions gives the same retrieval semantics
        while keeping this first SGLang port simple and deterministic.
        """

        tokens = [int(x) for x in tokens]
        common = 0
        limit = min(len(tokens), len(self._history))
        while common < limit and tokens[common] == self._history[common]:
            common += 1

        if common != len(self._history):
            # Request context was rewound/replaced. Reconstruct exactly from it.
            token_bin = self._token_bin
            self.root = _TrieNode()
            self.root.fail = self.root
            self._node_count = 0
            self._history = []
            self._cur_state = self.root
            self._token_bin = token_bin

        old_len = len(self._history)
        changed = False
        if len(tokens) > old_len:
            self._history.extend(tokens[old_len:])
            first_end = max(self.ngram, old_len + 1)
            for end in range(first_end, len(self._history) + 1):
                self.insert(self._history[end - self.ngram : end])
                changed = True

        if changed or old_len == 0:
            self._build_fail_links()
        self._recompute_state(self._history)

    def update_logits(
        self, tokens: Sequence[int], topk_ids: Sequence[Sequence[int]]
    ) -> None:
        """Refresh RACER's copy-logit top-k adjacency.

        Paper correspondence: Sec. 3.1 (copy-logit) and Sec. 3.3 (top-k
        adjacency matrix). Conceptually each observed token stores

            A[token_id] -> TopK_k(next-token logits).

        The original implementation materializes A as a vocabulary-sized
        matrix. This port stores only observed rows in ``_token_bin``. Repeated
        token ids overwrite the row so later observations provide the current
        copy-logit distribution.
        """

        for token, row in zip(tokens, topk_ids):
            self._token_bin[int(token)] = [int(x) for x in row[: self.topk]]

    def _token_bin_row(self, token: int) -> list[int]:
        return self._token_bin.get(int(token), [0] * self.topk)

    def _token_bin_successors(self, token: int, breadth: int) -> Sequence[int]:
        row = self._token_bin.get(int(token))
        if not row:
            # No copy-logit adjacency: grow one token-0 placeholder so the
            # leftover budget still materializes unique nodes instead of
            # stalling before K.
            return (0,)
        return row[: max(1, min(int(breadth), len(row)))]

    def _fill_logits_tree(self, tree: _DraftTree, root_idx: int, root_token: int) -> None:
        """Expand leftover slots from TokenBin with RACER Sec. 3.1 Eq. (3).

        The root keeps its breadth for the highest-ranked child; deeper nodes
        start at half their parent's breadth; later siblings keep halving,
        clamped to one. An existing retrieval edge is reused and does not
        consume a slot.
        """

        queue = deque([(root_idx, int(root_token), int(self.topk), 0)])
        expanded: set[int] = set()

        while queue and tree.active < tree.budget:
            node_idx, token, breadth, depth = queue.popleft()
            if node_idx in expanded:
                continue
            expanded.add(node_idx)

            successors = self._token_bin_successors(token, breadth)
            if not successors:
                continue

            next_breadth = breadth if depth == 0 else max(1, breadth >> 1)
            for child_token in successors:
                child_breadth = max(1, next_breadth)
                next_breadth = max(1, next_breadth >> 1)
                child_idx = tree.add(node_idx, int(child_token))
                if child_idx is None:
                    break
                queue.append(
                    (child_idx, int(child_token), child_breadth, depth + 1)
                )

    def _collect_borders(self, next_token: int) -> list[_TrieNode]:
        """Collect AC border states used by the Retrieval Tree (Sec. 3.2)."""

        borders: list[_TrieNode] = []
        u = self._cur_state
        state_updated = False

        # RACER considers matched border states with depth >= 2 before pooling
        # their continuation sub-tries for retrieval expansion.
        while u is not self.root:
            v = u.children.get(int(next_token))
            if v is not None:
                if v.depth >= self.min_depth:
                    borders.append(v)
                if not state_updated:
                    self._cur_state = v
                    state_updated = True
            u = u.fail if u.fail is not None else self.root

        v = self.root.children.get(int(next_token))
        if v is not None:
            if v.depth >= self.min_depth:
                borders.append(v)
            if not state_updated:
                self._cur_state = v
                state_updated = True

        if not state_updated:
            self._cur_state = self.root
        return borders

    def _select_retrieval_nodes(
        self, borders: Sequence[_TrieNode], budget: int
    ) -> list[tuple[_TrieNode, _TrieNode]]:
        """Select globally frequent continuation states across AC borders.

        This corresponds to Sec. 3.2: continuations from all matched border
        sub-tries are pooled, ranked by empirical frequency, and the strongest
        states are retained. Leftover fixed-K slots are filled afterwards by
        the Logits Tree.
        """

        scored: list[tuple[int, int, int, _TrieNode, _TrieNode]] = []
        serial = 0
        for border in borders:
            q = deque([border])
            while q:
                node = q.popleft()
                # Higher freq wins; for equal freq shallower depth wins.
                scored.append((-node.freq, node.depth, serial, node, border))
                serial += 1
                q.extend(node.children.values())

        scored.sort(key=lambda x: (x[0], x[1], x[2]))
        return [(node, start) for _, _, _, node, start in scored[:budget]]

    @staticmethod
    def _selected_paths(
        selected: Sequence[tuple[_TrieNode, _TrieNode]],
    ) -> list[list[int]]:
        selected_set = {(id(node), id(start)) for node, start in selected}
        paths: list[list[int]] = []

        for node, start in selected:
            is_leaf = True
            for child in node.children.values():
                if (id(child), id(start)) in selected_set:
                    is_leaf = False
                    break
            if not is_leaf:
                continue

            rev: list[int] = []
            cur = node
            while cur is not start:
                rev.append(cur.token)
                assert cur.parent is not None
                cur = cur.parent
            rev.append(start.token)
            paths.append(list(reversed(rev)))

        return paths

    def _record_proposal_shape(self, **kwargs) -> None:
        return None

    def retrieve(
        self, root_token: int, max_num_draft: int
    ) -> tuple[list[int], list[list[bool]]]:
        """Build a fixed-K proposal: Retrieval Tree first, then Logits Tree fill.

        Retrieval nodes are materialized first and never replaced. Logits Tree
        BFS then walks from the current token, reusing any retrieval edge that
        TokenBin also wants, and only consuming leftover slots for new nodes.
        A node with no TokenBin outgoing edge grows one token-0 placeholder
        child. Residual capacity, if any, is dummy token-0 children of the
        draft root so TARGET_VERIFY always sees exactly K nodes.
        """

        budget = int(max_num_draft)
        if budget <= 0:
            raise ValueError("max_num_draft must be greater than 0")
        root_token = int(root_token)

        borders = self._collect_borders(root_token)
        selected = self._select_retrieval_nodes(borders, budget)

        tree = _DraftTree(budget)
        for path in self._selected_paths(selected):
            parent = -1
            for token in path:
                idx = tree.add(parent, token)
                if idx is None:
                    break
                parent = idx

        retrieval_unique_nodes = tree.active
        if tree.active == 0 or root_token not in tree.root_indices:
            tree.add(-1, root_token)

        root_idx = tree.root_indices.get(root_token)
        if root_idx is None:
            raise RuntimeError("RACER draft tree is missing the current root token")

        after_seed = tree.active
        self._fill_logits_tree(tree, root_idx, root_token)
        nodes_before_padding = tree.active
        tree.pad_with_zeros()
        self._last_real_count = nodes_before_padding

        self._record_proposal_shape(
            borders=len(borders),
            retrieval_selected=len(selected),
            retrieval_unique_nodes=retrieval_unique_nodes,
            logits_fill_nodes=max(0, nodes_before_padding - after_seed),
            nodes_before_padding=nodes_before_padding,
            padding_nodes=max(0, budget - nodes_before_padding),
        )

        if tree.active != budget:
            raise RuntimeError(
                f"RACER fixed-K proposal invariant failed: {tree.active=}, {budget=}"
            )
        return tree.tokens.tolist(), tree.mask.tolist()
