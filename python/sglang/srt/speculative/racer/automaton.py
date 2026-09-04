from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Sequence


@dataclass
class _TrieNode:
    children: Dict[int, "_TrieNode"] = field(default_factory=dict)
    token: int = -1
    freq: int = 0
    depth: int = 0
    parent: "_TrieNode | None" = None


class _CandidateTrie:
    def __init__(self):
        self.root = _TrieNode()
        self.node_count = 0

    def insert(self, path: Sequence[int], budget: int) -> None:
        node = self.root
        for token in path:
            nxt = node.children.get(int(token))
            if nxt is None:
                if self.node_count >= budget:
                    return
                nxt = _TrieNode(token=int(token), depth=node.depth + 1, parent=node)
                node.children[int(token)] = nxt
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
    """Build RACER retrieval and copy-logit proposal trees without extra builds."""

    def __init__(self, *, ngram: int = 10, topk: int = 9, max_nodes: int = 10_000, min_depth: int = 2):
        self.ngram = max(1, int(ngram))
        self.topk = max(1, int(topk))
        self.max_nodes = max(1, int(max_nodes))
        self.min_depth = max(1, int(min_depth))
        self.root = _TrieNode()
        self._node_count = 0
        self._history: list[int] = []
        self._token_bin: dict[int, list[int]] = {}

    def reset(self) -> None:
        self.root = _TrieNode()
        self._node_count = 0
        self._history.clear()
        self._token_bin.clear()

    def _new_child(self, parent: _TrieNode, token: int) -> _TrieNode | None:
        if self._node_count >= self.max_nodes:
            return None
        child = _TrieNode(token=int(token), depth=parent.depth + 1, parent=parent)
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

    def sync_history(self, tokens: Sequence[int]) -> None:
        tokens = [int(x) for x in tokens]
        common = 0
        limit = min(len(tokens), len(self._history))
        while common < limit and tokens[common] == self._history[common]:
            common += 1
        if common != len(self._history):
            self.root = _TrieNode()
            self._node_count = 0
            self._history = []

        old_len = len(self._history)
        if len(tokens) <= old_len:
            return
        self._history.extend(tokens[old_len:])
        first_end = max(self.ngram, old_len + 1)
        for end in range(first_end, len(self._history) + 1):
            self.insert(self._history[end - self.ngram : end])

    def update_logits(self, tokens: Sequence[int], topk_ids: Sequence[Sequence[int]]) -> None:
        for token, row in zip(tokens, topk_ids):
            self._token_bin[int(token)] = [int(x) for x in row[: self.topk]]

    def _lookup(self, pattern: Sequence[int]) -> _TrieNode | None:
        node = self.root
        for token in pattern:
            node = node.children.get(int(token))
            if node is None:
                return None
        return node

    def _history_paths(self, root_token: int, budget: int) -> list[tuple[int, list[int]]]:
        if budget <= 1 or not self._history:
            return []
        borders: list[_TrieNode] = []
        max_suffix = min(self.ngram - 1, len(self._history))
        for suffix_len in range(1, max_suffix + 1):
            node = self._lookup(self._history[-suffix_len:] + [int(root_token)])
            if node is not None and node.depth >= self.min_depth:
                borders.append(node)

        scored: list[tuple[int, int, list[int]]] = []
        for border in borders:
            q = deque([(border, [int(root_token)])])
            while q:
                node, path = q.popleft()
                scored.append((node.freq, len(path), path))
                for child in node.children.values():
                    q.append((child, path + [child.token]))
        scored.sort(key=lambda x: (-x[0], x[1], x[2]))

        out: list[tuple[int, list[int]]] = []
        seen = set()
        for freq, _, path in scored:
            key = tuple(path)
            if key in seen:
                continue
            seen.add(key)
            out.append((freq, path))
            if len(out) >= budget:
                break
        return out

    def _logit_paths(self, root_token: int, budget: int) -> list[list[int]]:
        q = deque([(int(root_token), [int(root_token)], self.topk, 0)])
        paths: list[list[int]] = []
        expanded = 0
        while q and expanded < budget:
            token, path, breadth, depth = q.popleft()
            children = self._token_bin.get(token)
            if not children:
                paths.append(path)
                continue
            next_breadth = breadth if depth == 0 else max(1, breadth >> 1)
            added = 0
            for i, child in enumerate(children[:breadth]):
                if expanded >= budget:
                    break
                q.append((int(child), path + [int(child)], max(1, next_breadth >> i), depth + 1))
                expanded += 1
                added += 1
            if added == 0:
                paths.append(path)
        paths.extend(path for _, path, _, _ in q)
        return paths or [[int(root_token)]]

    def retrieve(self, root_token: int, max_num_draft: int) -> tuple[list[int], list[list[bool]]]:
        budget = max(1, int(max_num_draft))
        root_token = int(root_token)
        trie = _CandidateTrie()
        for _, path in self._history_paths(root_token, budget):
            trie.insert(path, budget)
        for path in self._logit_paths(root_token, budget):
            trie.insert(path, budget)
        if trie.node_count == 0:
            trie.insert([root_token], budget)

        path = [root_token]
        token = root_token
        while trie.node_count < budget:
            row = self._token_bin.get(token)
            token = int(row[0]) if row else 0
            path = path + [token]
            trie.insert(path, budget)
        return trie.flatten(budget)
