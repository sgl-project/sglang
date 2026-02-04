from __future__ import annotations

from typing import Dict, List, Optional, Tuple


def _key_match(key0: Tuple[int, ...], key1: Tuple[int, ...]) -> int:
    i = 0
    for k0, k1 in zip(key0, key1):
        if k0 != k1:
            break
        i += 1
    return i


class AdmissionNode:
    def __init__(self):
        self.children: Dict[object, "AdmissionNode"] = {}
        self.parent: Optional["AdmissionNode"] = None
        self.key: Tuple[int, ...] = tuple()
        self.value: List[int] = []
        self.prefix_len = 0
        self.extra_key: Optional[str] = None
        self.insert_count = 0
        self.hit_count = 0
        self.success_count = 0
        self.last_access_time = 0
        self.score = 0.0
        self.last_score_time = 0


class MarconiAdmissionTree:
    def __init__(
        self,
        *,
        policy: str = "taxonomy",
        min_hits: int = 2,
        min_success_ratio: float = 0.1,
        decay: float = 0.995,
        score_threshold: float = 1.0,
        max_nodes: Optional[int] = None,
        max_tokens: Optional[int] = None,
        prune_interval: int = 200,
    ):
        self.root_node = AdmissionNode()
        if policy not in ("taxonomy", "thresholded"):
            raise ValueError(f"Unknown Marconi admission policy: {policy}")
        self.policy = policy
        self.min_hits = min_hits
        self.min_success_ratio = min_success_ratio
        self.decay = decay
        self.score_threshold = score_threshold
        self.max_nodes = max_nodes
        self.max_tokens = max_tokens
        self.prune_interval = prune_interval
        self.logical_ts = 0
        self.num_nodes = 1
        self.num_tokens = 0
        self.insertions_since_prune = 0

    def _tick(self) -> int:
        self.logical_ts += 1
        return self.logical_ts

    def _update_score(self, node: AdmissionNode, ts: int, add: float = 0.0) -> None:
        if node.last_score_time != ts:
            delta = ts - node.last_score_time
            if delta > 0:
                node.score *= self.decay**delta
            node.last_score_time = ts
        if add:
            node.score += add

    def _eligible(self, node: Optional[AdmissionNode]) -> bool:
        if node is None:
            return False
        if self.policy == "taxonomy":
            return True
        if node.hit_count < self.min_hits:
            return False
        success_ratio = (
            node.success_count / node.hit_count if node.hit_count > 0 else 0.0
        )
        if success_ratio < self.min_success_ratio:
            return False
        if node.score < self.score_threshold:
            return False
        return True

    def match_prefix(
        self, token_ids: List[int], extra_key: Optional[str]
    ) -> Tuple[int, bool]:
        node = self.root_node
        key = tuple(token_ids)
        matched_len = 0
        candidate_node: Optional[AdmissionNode] = None
        candidate_len = 0
        ts = self._tick()
        while key:
            child_key = self._child_key(extra_key, key)
            child = node.children.get(child_key)
            if child is None:
                break
            prefix_len = _key_match(child.key, key)
            matched_len += prefix_len
            child.last_access_time = ts
            child.hit_count += 1
            self._update_score(child, ts, add=1.0)
            if prefix_len < len(child.key):
                # Speculative insertion would create an intermediate branch here.
                if self._eligible(child):
                    candidate_node = child
                    candidate_len = matched_len
                break
            if len(child.children) >= 2 and self._eligible(child):
                candidate_node = child
                candidate_len = child.prefix_len
            key = key[prefix_len:]
            node = child
        branchoff_required = candidate_node is not None and candidate_len > 0
        return candidate_len if branchoff_required else matched_len, branchoff_required

    def insert(self, token_ids: List[int], extra_key: Optional[str]) -> None:
        node = self.root_node
        key = tuple(token_ids)
        value = list(token_ids)
        ts = self._tick()
        while key:
            child_key = self._child_key(extra_key, key)
            child = node.children.get(child_key)
            if child is None:
                new_node = AdmissionNode()
                new_node.parent = node
                new_node.key = key
                new_node.value = value
                new_node.prefix_len = node.prefix_len + len(value)
                new_node.extra_key = extra_key
                new_node.insert_count = 1
                new_node.last_access_time = ts
                self._update_score(new_node, ts, add=1.0)
                node.children[child_key] = new_node
                self.num_nodes += 1
                self.num_tokens += len(new_node.value)
                self.insertions_since_prune += 1
                self._maybe_prune()
                return
            prefix_len = _key_match(child.key, key)
            if prefix_len == len(child.key):
                if prefix_len == len(key):
                    child.insert_count += 1
                    child.last_access_time = ts
                    self._update_score(child, ts, add=1.0)
                    self.insertions_since_prune += 1
                    self._maybe_prune()
                    return
                node = child
                key = key[prefix_len:]
                value = value[prefix_len:]
                child.insert_count += 1
                child.last_access_time = ts
                self._update_score(child, ts, add=1.0)
                continue
            node = self._split_node(child, prefix_len, extra_key)
            key = key[prefix_len:]
            value = value[prefix_len:]
        self.insertions_since_prune += 1
        self._maybe_prune()

    def _split_node(
        self, child: AdmissionNode, split_len: int, extra_key: Optional[str]
    ) -> AdmissionNode:
        new_node = AdmissionNode()
        new_node.parent = child.parent
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len]
        new_node.prefix_len = new_node.parent.prefix_len + len(new_node.value)
        new_node.extra_key = extra_key
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        child.prefix_len = new_node.prefix_len + len(child.value)
        new_node.children[self._child_key(extra_key, child.key)] = child
        new_node.parent.children[self._child_key(extra_key, new_node.key)] = new_node
        self.num_nodes += 1
        return new_node

    def record_cache_hit(
        self,
        token_ids: List[int],
        hit_len: int,
        extra_key: Optional[str],
        weight: float = 0.5,
    ) -> None:
        if hit_len <= 0:
            return
        node = self.root_node
        key = tuple(token_ids[:hit_len])
        ts = self._tick()
        while key:
            child_key = self._child_key(extra_key, key)
            child = node.children.get(child_key)
            if child is None:
                break
            prefix_len = _key_match(child.key, key)
            if prefix_len < len(child.key):
                break
            child.success_count += 1
            child.last_access_time = ts
            self._update_score(child, ts, add=weight)
            key = key[prefix_len:]
            node = child

    def _maybe_prune(self) -> None:
        if self.insertions_since_prune < self.prune_interval:
            return
        self.insertions_since_prune = 0
        if self.max_nodes is None and self.max_tokens is None:
            return
        self._prune()

    def _prune(self) -> None:
        leaves = self._collect_leaves()
        if not leaves:
            return
        leaves.sort(key=lambda n: n.last_access_time)
        idx = 0
        while (
            (self.max_nodes is not None and self.num_nodes > self.max_nodes)
            or (self.max_tokens is not None and self.num_tokens > self.max_tokens)
        ) and idx < len(leaves):
            node = leaves[idx]
            idx += 1
            if node == self.root_node or node.children:
                continue
            self._delete_leaf(node)

    def _delete_leaf(self, node: AdmissionNode) -> None:
        if node.parent is None:
            return
        node.parent.children.pop(self._child_key(node.extra_key, node.key), None)
        self.num_nodes -= 1
        self.num_tokens -= len(node.value)

    def _collect_leaves(self) -> List[AdmissionNode]:
        ret_list = []
        stack = [self.root_node]
        while stack:
            cur_node = stack.pop()
            if len(cur_node.children) == 0:
                ret_list.append(cur_node)
            else:
                stack.extend(cur_node.children.values())
        return ret_list

    @staticmethod
    def _child_key(extra_key: Optional[str], key: Tuple[int, ...]):
        token = key[0]
        return token if extra_key is None else (extra_key, token)
