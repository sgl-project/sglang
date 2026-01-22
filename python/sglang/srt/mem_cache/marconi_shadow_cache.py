from __future__ import annotations

import copy
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Tuple

from sglang.srt.mem_cache.marconi_config import MarconiModelStats
from sglang.srt.mem_cache.marconi_utils import (
    get_attn_flops,
    get_kv_cache_size_bytes,
    get_mamba1_flops,
    get_mlp_flops,
    normalize,
)


@dataclass
class ShadowNode:
    key: Tuple[int, ...]
    value: List[int]
    extra_key: Optional[str] = None
    parent: Optional["ShadowNode"] = None
    children: Optional[dict[object, "ShadowNode"]] = None
    last_access_time: int = 0
    prefix_len: int = 0

    def __post_init__(self):
        if self.children is None:
            self.children = {}


class MarconiShadowCache:
    def __init__(
        self,
        model_stats: MarconiModelStats,
        capacity_bytes: int,
        eff_weight: float,
        tuning_interval: int,
    ):
        self.model_stats = model_stats
        self.capacity_bytes = capacity_bytes
        self.eff_weight = eff_weight
        self.tuning_interval = tuning_interval
        self.root_node = ShadowNode(key=tuple(), value=[])
        self.logical_ts = 0
        self.request_history: List[Tuple[bool, int, int]] = []

    def clone(self) -> "MarconiShadowCache":
        return copy.deepcopy(self)

    def _tick(self) -> int:
        self.logical_ts += 1
        return self.logical_ts

    def _child_key(self, extra_key: Optional[str], key: Tuple[int, ...]):
        token = key[0]
        return token if extra_key is None else (extra_key, token)

    def _key_match(self, key0: Tuple[int, ...], key1: Tuple[int, ...]) -> int:
        i = 0
        for k0, k1 in zip(key0, key1):
            if k0 != k1:
                break
            i += 1
        return i

    def match_prefix(
        self,
        input_token_ids: List[int],
        extra_key: Optional[str] = None,
        record_request: bool = True,
    ):
        self._tick()
        prefix_token_ids: List[List[int]] = []
        nodes_accessed = [self.root_node]
        prefix_len = self._match_prefix_helper(
            self.root_node,
            tuple(input_token_ids),
            prefix_token_ids,
            nodes_accessed,
            extra_key,
        )
        prefix_token_ids = [t for part in prefix_token_ids for t in part]
        num_tokens_skipped = len(prefix_token_ids)
        branchoff_required = prefix_len > num_tokens_skipped

        if record_request:
            cache_hit = num_tokens_skipped > 0
            self.request_history.append(
                (cache_hit, len(input_token_ids), num_tokens_skipped)
            )
        if len(nodes_accessed) > 1:
            nodes_accessed[-1].last_access_time = self.logical_ts
        return prefix_token_ids, branchoff_required, prefix_len

    def insert(self, token_ids: List[int], extra_key: Optional[str] = None) -> None:
        _, branchoff_required, prefix_len = self.match_prefix(
            token_ids, extra_key=extra_key, record_request=False
        )
        num_extra_tokens = len(token_ids) - prefix_len
        num_extra_mamba_states = 2 if branchoff_required else 1
        bytes_needed = (
            num_extra_mamba_states
            * self.model_stats.num_mamba_layers
            * self.model_stats.mamba_state_size_bytes
            + self.model_stats.num_attn_layers
            * get_kv_cache_size_bytes(
                num_extra_tokens,
                self.model_stats.model_dim,
                self.model_stats.kv_cache_dtype_size,
            )
        )
        if self.get_tree_size() + bytes_needed > self.capacity_bytes:
            bytes_to_remove = self.get_tree_size() + bytes_needed - self.capacity_bytes
            self.evict(bytes_to_remove)
        self._insert_helper(
            node=self.root_node,
            key=tuple(token_ids),
            value=token_ids,
            extra_key=extra_key,
        )

    def get_tree_size(self) -> int:
        num_cached_mamba_states, num_cached_kv_tokens = self._count_cached_tokens(
            self.root_node
        )
        mamba_state_size = (
            self.model_stats.num_mamba_layers
            * num_cached_mamba_states
            * self.model_stats.mamba_state_size_bytes
        )
        attn_state_size = self.model_stats.num_attn_layers * get_kv_cache_size_bytes(
            num_cached_kv_tokens,
            self.model_stats.model_dim,
            self.model_stats.kv_cache_dtype_size,
        )
        return mamba_state_size + attn_state_size

    def get_cache_stats(self, last_n: Optional[int] = None):
        if last_n is None:
            request_history = self.request_history
        else:
            request_history = self.request_history[-last_n:]
        num_requests_recorded = len(request_history)
        if num_requests_recorded == 0:
            return 0.0, 0.0, 0.0
        num_total_tokens = sum(x[1] for x in request_history)
        num_tokens_saved = sum(x[2] for x in request_history)

        total_mamba_flop_savings = self.model_stats.num_mamba_layers * get_mamba1_flops(
            num_tokens_saved,
            self.model_stats.model_dim,
            self.model_stats.ssm_state_size,
        )
        total_attn_flop_savings = sum(
            self.model_stats.num_attn_layers
            * get_attn_flops(x[2], self.model_stats.model_dim)
            for x in request_history
        )
        total_mlp_flop_savings = sum(
            self.model_stats.num_mlp_layers
            * get_mlp_flops(x[2], self.model_stats.model_dim)
            for x in request_history
        )
        total_flops_savings = (
            total_mamba_flop_savings + total_attn_flop_savings + total_mlp_flop_savings
        )

        request_hit_rate = sum(x[0] for x in request_history) / num_requests_recorded
        token_hit_rate = (
            num_tokens_saved / num_total_tokens if num_total_tokens else 0.0
        )
        return request_hit_rate, token_hit_rate, total_flops_savings

    def evict(self, bytes_to_remove: int) -> None:
        bytes_evicted = 0
        while bytes_evicted < bytes_to_remove:
            candidates = self._collect_leaf_and_single_child_nodes(self.root_node)
            if not candidates:
                break
            node_to_evict = self._select_node(candidates)
            if node_to_evict is None:
                break
            if len(node_to_evict.children) == 0:
                bytes_evicted += (
                    self.model_stats.num_mamba_layers
                    * self.model_stats.mamba_state_size_bytes
                    + self.model_stats.num_attn_layers
                    * get_kv_cache_size_bytes(
                        len(node_to_evict.value),
                        self.model_stats.model_dim,
                        self.model_stats.kv_cache_dtype_size,
                    )
                )
                self._delete_leaf(node_to_evict)
            else:
                bytes_evicted += (
                    self.model_stats.num_mamba_layers
                    * self.model_stats.mamba_state_size_bytes
                )
                self._evict_intermediate_node(node_to_evict)

    def _select_node(self, nodes: List[ShadowNode]) -> Optional[ShadowNode]:
        utilities = self._compute_utilities(nodes)
        if not utilities:
            return None
        min_idx = utilities.index(min(utilities))
        return nodes[min_idx]

    def _compute_utilities(self, nodes: List[ShadowNode]) -> List[float]:
        current_ts = self._tick()
        timestamps = [node.last_access_time for node in nodes]
        recency_scores = []
        for ts in timestamps:
            delta = current_ts - ts
            recency_scores.append(1.0 / delta if delta > 0 else 0.0)
        recency_scores = normalize(recency_scores)

        efficiency_scores = []
        for node in nodes:
            seqlen_total = node.prefix_len
            seqlen_child = len(node.value)
            seqlen_parent = max(seqlen_total - seqlen_child, 0)

            flops_savings_mamba = self.model_stats.num_mamba_layers * get_mamba1_flops(
                seqlen_child,
                self.model_stats.model_dim,
                self.model_stats.ssm_state_size,
            )
            flops_savings_attn = self.model_stats.num_attn_layers * (
                get_attn_flops(seqlen_total, self.model_stats.model_dim)
                - get_attn_flops(seqlen_parent, self.model_stats.model_dim)
            )
            flops_savings_mlp = self.model_stats.num_mlp_layers * (
                get_mlp_flops(seqlen_total, self.model_stats.model_dim)
                - get_mlp_flops(seqlen_parent, self.model_stats.model_dim)
            )
            total_flops_savings = (
                flops_savings_mamba + flops_savings_attn + flops_savings_mlp
            )
            total_memory = (
                self.model_stats.num_mamba_layers
                * self.model_stats.mamba_state_size_bytes
                + self.model_stats.num_attn_layers
                * get_kv_cache_size_bytes(
                    seqlen_child,
                    self.model_stats.model_dim,
                    self.model_stats.kv_cache_dtype_size,
                )
            )
            if total_memory > 0:
                efficiency_scores.append(total_flops_savings / total_memory)
            else:
                efficiency_scores.append(0.0)
        efficiency_scores = normalize(efficiency_scores)
        return [
            self.eff_weight * eff + recency
            for eff, recency in zip(efficiency_scores, recency_scores)
        ]

    def _match_prefix_helper(
        self,
        node: ShadowNode,
        key: Tuple[int, ...],
        value: List[List[int]],
        nodes_accessed: List[ShadowNode],
        extra_key: Optional[str],
    ) -> int:
        if len(key) == 0:
            return 0
        child_key = self._child_key(extra_key, key)
        if child_key in node.children:
            child = node.children[child_key]
            prefix_len = self._key_match(child.key, key)
            if prefix_len < len(child.key):
                return prefix_len + self._match_prefix_helper(
                    child, key[prefix_len:], value, nodes_accessed, extra_key
                )
            value.append(child.value)
            nodes_accessed.append(child)
            return prefix_len + self._match_prefix_helper(
                child, key[prefix_len:], value, nodes_accessed, extra_key
            )
        return 0

    def _insert_helper(
        self,
        node: ShadowNode,
        key: Tuple[int, ...],
        value: List[int],
        extra_key: Optional[str],
    ) -> int:
        if len(key) == 0:
            return 0

        child_key = self._child_key(extra_key, key)
        if child_key in node.children:
            child = node.children[child_key]
            prefix_len = self._key_match(child.key, key)
            if prefix_len == len(child.key):
                if prefix_len == len(key):
                    return prefix_len
                key = key[prefix_len:]
                value = value[prefix_len:]
                return prefix_len + self._insert_helper(child, key, value, extra_key)

            new_node = self._split_node(child.key, child, prefix_len, extra_key)
            return prefix_len + self._insert_helper(
                new_node, key[prefix_len:], value[prefix_len:], extra_key
            )

        if len(key):
            new_node = ShadowNode(
                key=key, value=value, parent=node, extra_key=extra_key
            )
            new_node.prefix_len = node.prefix_len + len(value)
            node.children[self._child_key(extra_key, key)] = new_node
        return 0

    def _split_node(
        self,
        key: Tuple[int, ...],
        child: ShadowNode,
        split_len: int,
        extra_key: Optional[str],
    ) -> ShadowNode:
        new_node = ShadowNode(
            key=child.key[:split_len],
            value=child.value[:split_len],
            parent=child.parent,
            extra_key=extra_key,
        )
        new_node.prefix_len = new_node.parent.prefix_len + len(new_node.value)
        new_node.children = {self._child_key(extra_key, key[split_len:]): child}
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        child.prefix_len = new_node.prefix_len + len(child.value)
        new_node.parent.children[self._child_key(extra_key, new_node.key)] = new_node
        return new_node

    def _delete_leaf(self, node: ShadowNode) -> None:
        if node.parent is None:
            return
        node.parent.children.pop(self._child_key(node.extra_key, node.key), None)

    def _evict_intermediate_node(self, node: ShadowNode) -> None:
        if len(node.children) != 1 or node.parent is None:
            return
        child = next(iter(node.children.values()))
        new_node = ShadowNode(
            key=tuple(node.value + child.value),
            value=node.value + child.value,
            parent=node.parent,
            extra_key=node.extra_key,
        )
        new_node.prefix_len = node.parent.prefix_len + len(new_node.value)
        new_node.children = child.children
        for grandchild in new_node.children.values():
            grandchild.parent = new_node
        node.parent.children[self._child_key(new_node.extra_key, new_node.key)] = (
            new_node
        )

    def _collect_leaf_and_single_child_nodes(
        self, node: ShadowNode
    ) -> List[ShadowNode]:
        ret_list = []
        stack = [node]
        while stack:
            cur_node = stack.pop()
            if len(cur_node.children) <= 1 and cur_node.value:
                ret_list.append(cur_node)
            stack.extend(cur_node.children.values())
        return ret_list

    def _count_cached_tokens(self, node: ShadowNode) -> Tuple[int, int]:
        if node.value:
            total_mamba_states = 1
            total_kv_tokens = len(node.value)
        else:
            total_mamba_states, total_kv_tokens = 0, 0
        for child in node.children.values():
            child_mamba, child_kv = self._count_cached_tokens(child)
            total_mamba_states += child_mamba
            total_kv_tokens += child_kv
        return total_mamba_states, total_kv_tokens


class MarconiConfigTuner:
    def __init__(self, weights: Optional[List[float]] = None):
        self.tree_snapshot: Optional[MarconiShadowCache] = None
        self.num_tunings = 0
        self.executor: Optional[ProcessPoolExecutor] = None
        self.weights = weights or [
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5,
            1.6,
            1.7,
            1.8,
            1.9,
            2.0,
        ]

    def _ensure_executor(self) -> None:
        if self.executor is None:
            ctx = mp.get_context("spawn")
            self.executor = ProcessPoolExecutor(max_workers=1, mp_context=ctx)

    def tune(
        self,
        request_history_windowed: List[Tuple[Optional[str], List[int], List[int]]],
        tuning_interval: int,
    ) -> Optional[float]:
        _, best_eff_weight = self._tune_with_snapshot(
            self.tree_snapshot,
            self.weights,
            request_history_windowed,
            tuning_interval,
            0,
        )
        if best_eff_weight is None:
            return None
        self.num_tunings += 1
        return best_eff_weight

    def submit(
        self,
        tree_snapshot: Optional[MarconiShadowCache],
        request_history_windowed: List[Tuple[Optional[str], List[int], List[int]]],
        tuning_interval: int,
        generation: int,
    ):
        self._ensure_executor()
        return self.executor.submit(
            self._tune_with_snapshot,
            tree_snapshot,
            list(self.weights),
            request_history_windowed,
            tuning_interval,
            generation,
        )

    @staticmethod
    def _tune_with_snapshot(
        tree_snapshot: Optional[MarconiShadowCache],
        weights: List[float],
        request_history_windowed: List[Tuple[Optional[str], List[int], List[int]]],
        tuning_interval: int,
        generation: int,
    ) -> Tuple[int, Optional[float]]:
        if tree_snapshot is None:
            return generation, None
        results = {}
        for weight in weights:
            shadow_cache = tree_snapshot.clone()
            shadow_cache.eff_weight = weight
            for extra_key, input_ids, output_ids in request_history_windowed:
                shadow_cache.match_prefix(
                    input_ids, extra_key=extra_key, record_request=True
                )
                shadow_cache.insert(input_ids + output_ids, extra_key=extra_key)
            _, _, total_flops_saved = shadow_cache.get_cache_stats(
                last_n=tuning_interval
            )
            results[weight] = total_flops_saved
        if not results:
            return generation, None
        best_eff_weight = max(results, key=results.get)
        return generation, best_eff_weight
