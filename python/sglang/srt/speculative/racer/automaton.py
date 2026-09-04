from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import logging
from typing import Dict, Sequence

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class _TrieNode:
    children: Dict[int, "_TrieNode"] = field(default_factory=dict)
    fail: "_TrieNode | None" = None
    token: int = -1
    freq: int = 0
    depth: int = 0
    parent: "_TrieNode | None" = None


class _CandidateTrie:
    """Small merge trie used only for one RACER proposal tree."""

    def __init__(self):
        self.root = _TrieNode()
        self.node_count = 0

    def insert(self, path: Sequence[int], budget: int) -> None:
        node = self.root
        for token in path:
            token = int(token)
            nxt = node.children.get(token)
            if nxt is None:
                if self.node_count >= budget:
                    return
                nxt = _TrieNode(
                    token=token,
                    depth=node.depth + 1,
                    parent=node,
                )
                node.children[token] = nxt
                self.node_count += 1
            node = nxt

    def flatten(self, budget: int) -> tuple[list[int], list[list[bool]]]:
        tokens: list[int] = []
        parents: list[int] = []
        q = deque((child, -1) for child in self.root.children.values())
        while q and len(tokens) < budget:
            node, parent_idx = q.popleft()
            idx = len(tokens)
            tokens.append(node.token)
            parents.append(parent_idx)
            for child in node.children.values():
                q.append((child, idx))

        n = len(tokens)
        mask = [[False] * n for _ in range(n)]
        for i in range(n):
            cur = i
            while cur >= 0:
                mask[i][cur] = True
                cur = parents[cur]
        return tokens, mask


class RacerAutomaton:
    """Pure-Python port of the original RACER C++ Automaton/TokenBin semantics.

    The serving worker keeps one instance per request.  The implementation is
    intentionally fidelity-first: retrieval-tree selection, AC fail transitions,
    TokenBin breadth propagation, and the split of the K-node budget match the
    original C++ implementation.  A final padding path is only a defensive guard
    for SGLang's fixed-K verify ABI and should not be used in the normal path.
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
        self._warned_padding = False

    def reset(self) -> None:
        self.root = _TrieNode()
        self.root.fail = self.root
        self._node_count = 0
        self._history.clear()
        self._token_bin.clear()
        self._cur_state = self.root
        self._warned_padding = False

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
        # Original TokenBin is keyed only by token id; latest observation wins.
        for token, row in zip(tokens, topk_ids):
            values = [int(x) for x in row[: self.topk]]
            if len(values) < self.topk:
                values.extend([0] * (self.topk - len(values)))
            self._token_bin[int(token)] = values

    def _token_bin_row(self, token: int) -> list[int]:
        # C++ TokenBin preallocates the whole matrix with zeros.  Missing entries
        # therefore behave as a top-k row of zero token ids.
        return self._token_bin.get(int(token), [0] * self.topk)

    def _token_bin_retrieve(
        self, next_token: int, max_num_draft: int, is_chain: bool = False
    ) -> list[list[int]]:
        """Port of TokenBin::retrieve from the original RACER C++ code."""

        if max_num_draft <= 0:
            return []

        # (token, parent_position, breadth, depth)
        q: list[tuple[int, int, int, int]] = [
            (int(next_token), -1, 1 if is_chain else self.topk, 0)
        ]
        remaining = int(max_num_draft) - 1
        head = 0
        candidates: list[list[int]] = []

        while head < len(q):
            token, pos_parent, breadth, depth = q[head]
            pos_u = head
            head += 1

            if remaining > 0 and breadth > 0:
                row = self._token_bin_row(token)
                # Exact C++ rule:
                #   depth == 1 -> preserve current breadth for first child
                #   otherwise  -> halve before the first child
                next_breadth = breadth if depth == 1 else (breadth >> 1)
                next_depth = depth + 1
                added = 0
                for i in range(min(breadth, self.topk)):
                    if remaining <= 0:
                        break
                    child = int(row[i])
                    q.append(
                        (
                            child,
                            pos_u,
                            max(1, next_breadth),
                            next_depth,
                        )
                    )
                    next_breadth >>= 1
                    remaining -= 1
                    added += 1
                if added:
                    continue

            # Leaf: recover root -> leaf candidate from parent indices.
            candidate: list[int] = []
            cur = pos_u
            while cur >= 0:
                candidate.append(q[cur][0])
                cur = q[cur][1]
            candidates.append(list(reversed(candidate)))

        return candidates

    def _collect_borders(self, next_token: int) -> list[_TrieNode]:
        borders: list[_TrieNode] = []
        u = self._cur_state
        state_updated = False

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
        """Equivalent final selection to the C++ frequency/depth top-k heap."""

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

    def _defensive_pad(self, trie: _CandidateTrie, budget: int, root_token: int) -> None:
        if trie.node_count >= budget:
            return
        if not self._warned_padding:
            logger.warning(
                "RACER proposal produced %d/%d nodes; applying defensive fixed-K padding. "
                "Normal RetrievalTree+TokenBin generation should already fill the budget.",
                trie.node_count,
                budget,
            )
            self._warned_padding = True

        # Follow the TokenBin top-1 continuation so padding remains a valid token
        # path. This is a last-resort ABI guard, not part of normal RACER proposal.
        path = [int(root_token)]
        token = int(root_token)
        guard = 0
        while trie.node_count < budget and guard < budget * 4:
            row = self._token_bin_row(token)
            token = int(row[0])
            path.append(token)
            before = trie.node_count
            trie.insert(path, budget)
            if trie.node_count == before:
                # Duplicate path: extend it once more on the same continuation.
                guard += 1
                continue
            guard += 1

        # Extremely defensive fallback for a pathological self-looping TokenBin.
        # Valid token 0 mirrors an uninitialized C++ TokenBin row.
        while trie.node_count < budget:
            path.append(0)
            trie.insert(path, budget)

    def retrieve(
        self, root_token: int, max_num_draft: int
    ) -> tuple[list[int], list[list[bool]]]:
        """Port of Automaton::retrieve for the default non-chain RACER mode."""

        budget = int(max_num_draft)
        if budget <= 0:
            raise ValueError("max_num_draft must be greater than 0")
        root_token = int(root_token)

        borders = self._collect_borders(root_token)
        selected = self._select_retrieval_nodes(borders, budget)

        candidate_trie = _CandidateTrie()
        for path in self._selected_paths(selected):
            candidate_trie.insert(path, budget)

        # Original RACER spends the remainder of the K-node budget on TokenBin.
        remaining = budget - len(selected)
        for path in self._token_bin_retrieve(root_token, remaining, is_chain=False):
            candidate_trie.insert(path, budget)

        # Original fallback when only the implicit trie root exists.
        if candidate_trie.node_count == 0:
            candidate_trie.insert([root_token], budget)

        # SGLang's current NGRAM verify ABI requires exactly K flattened nodes.
        # This should be unreachable in the normal RACER path, but keep a guard
        # for duplicate merging / capacity edge cases instead of crashing serving.
        self._defensive_pad(candidate_trie, budget, root_token)

        tokens, mask = candidate_trie.flatten(budget)
        if len(tokens) != budget:
            raise RuntimeError(
                f"RACER fixed-K proposal invariant failed: {len(tokens)=}, {budget=}"
            )
        return tokens, mask
