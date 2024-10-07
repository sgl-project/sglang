"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Request scheduler policy"""

import os
import random
from collections import defaultdict
from contextlib import contextmanager
from enum import Enum, auto
from typing import Dict, List, Optional

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.radix_cache import TreeNode

# Clip the estimation of max_new_tokens for the request whose max_new_tokens is very large.
# This can prevent the server from being too conservative.
# Note that this only clips the estimation in the scheduler but does not change the stop
# condition. The request can still generate tokens until it hits the unclipped max_new_tokens.
CLIP_MAX_NEW_TOKENS = int(os.environ.get("SGLANG_CLIP_MAX_NEW_TOKENS", "4096"))


class SchedulePolicy:
    def __init__(self, policy: str, tree_cache: BasePrefixCache):
        if tree_cache.disable and policy in ["lpm", "dfs-weight"]:
            # LPM and DFS-weight is meaningless when the tree cache is disabled.
            policy = "fcfs"

        self.policy = policy
        self.tree_cache = tree_cache

    def calc_priority(self, waiting_queue: List[Req]):
        # Compute matched prefix length
        prefix_computed = False
        if self.policy in ["lpm", "dfs-weight"]:
            for r in waiting_queue:
                # NOTE: the prefix_indices must always be aligned with last_node
                r.prefix_indices, r.last_node = self.tree_cache.match_prefix(
                    rid=r.rid, key=r.adjust_max_prefix_ids()
                )
            prefix_computed = True

        if self.policy == "lpm":
            # Longest Prefix Match
            waiting_queue.sort(key=lambda x: -len(x.prefix_indices))
        elif self.policy == "fcfs":
            # first come first serve
            pass
        elif self.policy == "lof":
            # longest output first
            waiting_queue.sort(key=lambda x: -x.sampling_params.max_new_tokens)
        elif self.policy == "random":
            random.shuffle(waiting_queue)
        elif self.policy == "dfs-weight":
            last_node_to_reqs = defaultdict(list)
            for req in waiting_queue:
                last_node_to_reqs[req.last_node].append(req)

            node_to_weight = defaultdict(int)
            for node in last_node_to_reqs:
                node_to_weight[node] = len(last_node_to_reqs[node])
            self.calc_weight(self.tree_cache.root_node, node_to_weight)

            waiting_queue.clear()
            self.get_dfs_priority(
                self.tree_cache.root_node,
                node_to_weight,
                last_node_to_reqs,
                waiting_queue,
            )
        else:
            raise ValueError(f"Unknown schedule_policy: {self.policy}")

        return prefix_computed

    def calc_weight(self, cur_node: TreeNode, node_to_weight: Dict):
        for child in cur_node.children.values():
            self.calc_weight(child, node_to_weight)
            node_to_weight[cur_node] += node_to_weight[child]

    def get_dfs_priority(
        self,
        cur_node: TreeNode,
        node_to_priority: Dict,
        last_node_to_reqs: Dict,
        q: List,
    ):
        childs = [child for child in cur_node.children.values()]
        childs.sort(key=lambda x: -node_to_priority[x])
        for child in childs:
            self.get_dfs_priority(child, node_to_priority, last_node_to_reqs, q)
        q.extend(last_node_to_reqs[cur_node])


class AddReqResult(Enum):
    CONTINUE = auto()  # Continue to add requests
    NO_TOKEN = auto()  # No token left
    OTHER = auto()  # Other reasons to stop adding requests


class PrefillAdder:
    def __init__(
        self,
        tree_cache: BasePrefixCache,
        running_batch: ScheduleBatch,
        new_token_ratio: float,
        rem_total_tokens: int,
        rem_input_tokens: int,
        rem_chunk_tokens: Optional[int],
        mixed_with_decode_tokens: int = 0,
    ):
        self.tree_cache = tree_cache
        self.running_batch = running_batch
        self.new_token_ratio = new_token_ratio
        self.rem_total_tokens = rem_total_tokens - mixed_with_decode_tokens
        self.rem_input_tokens = rem_input_tokens - mixed_with_decode_tokens
        self.rem_chunk_tokens = rem_chunk_tokens
        if self.rem_chunk_tokens is not None:
            self.rem_chunk_tokens -= mixed_with_decode_tokens

        self.cur_rem_tokens = rem_total_tokens - mixed_with_decode_tokens

        self.req_states = None
        self.can_run_list = []
        self.new_inflight_req = None
        self.log_hit_tokens = 0
        self.log_input_tokens = 0

        if running_batch is not None:
            # Pre-remove the tokens which will be occupied by the running requests
            self.rem_total_tokens -= sum(
                [
                    min(
                        (r.sampling_params.max_new_tokens - len(r.output_ids)),
                        CLIP_MAX_NEW_TOKENS,
                    )
                    * self.new_token_ratio
                    for r in running_batch.reqs
                ]
            )

    def budget_state(self):
        if self.rem_total_tokens <= 0 or self.cur_rem_tokens <= 0:
            return AddReqResult.NO_TOKEN

        if self.rem_input_tokens <= 0 or (
            self.rem_chunk_tokens is not None and self.rem_chunk_tokens <= 0
        ):
            return AddReqResult.OTHER

        return AddReqResult.CONTINUE

    def _prefill_one_req(
        self, prefix_len: int, extend_input_len: int, max_new_tokens: int
    ):
        self.rem_total_tokens -= extend_input_len + max_new_tokens
        self.cur_rem_tokens -= extend_input_len
        self.rem_input_tokens -= extend_input_len
        if self.rem_chunk_tokens is not None:
            self.rem_chunk_tokens -= extend_input_len

        self.log_hit_tokens += prefix_len
        self.log_input_tokens += extend_input_len

    def add_inflight_req(self, req: Req):
        truncated = req.extend_input_len > self.rem_chunk_tokens
        req.extend_input_len = min(req.extend_input_len, self.rem_chunk_tokens)
        req.fill_ids = req.fill_ids[: len(req.prefix_indices) + req.extend_input_len]
        self.can_run_list.append(req)

        self._prefill_one_req(
            len(req.prefix_indices),
            req.extend_input_len,
            (
                min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)
                if not truncated
                else 0
            ),
        )

        # Return if chunked prefill not finished
        return req if truncated else None

    @contextmanager
    def _lock_node(self, last_node: TreeNode):
        try:
            delta = self.tree_cache.inc_lock_ref(last_node)
            self.rem_total_tokens += delta
            yield None
        finally:
            delta = self.tree_cache.dec_lock_ref(last_node)
            self.rem_total_tokens += delta

    def add_one_req_ignore_eos(self, req: Req):
        def add_req_state(r, insert_sort=False):
            new_token_ratio = (
                1.0 if r.sampling_params.ignore_eos else self.new_token_ratio
            )
            tokens_left = r.sampling_params.max_new_tokens * new_token_ratio - len(
                r.output_ids
            )
            tokens_occupied = len(r.origin_input_ids) + len(r.output_ids)

            if tokens_left > 0:
                if not insert_sort:
                    self.req_states.append((tokens_left, tokens_occupied))
                else:
                    i = 0
                    for i in range(len(self.req_states)):
                        if tokens_left <= self.req_states[i][0]:
                            break
                    self.req_states.insert(i, (tokens_left, tokens_occupied))

        if self.req_states is None:
            self.req_states = []
            add_req_state(req)
            if self.running_batch is not None:
                for r in self.running_batch.reqs:
                    add_req_state(r)
            for r in self.can_run_list:
                add_req_state(r)
            self.req_states.sort(key=lambda x: x[0])
        else:
            add_req_state(req, insert_sort=True)

        cur_rem_tokens = self.cur_rem_tokens - len(req.origin_input_ids)
        tokens_freed = 0
        for i, (tokens_left, tokens_occupied) in enumerate(self.req_states):
            decode_steps = (
                self.req_states[i + 1][0]
                if i + 1 < len(self.req_states)
                else tokens_left
            )
            bs = len(self.req_states) - i
            if cur_rem_tokens + tokens_freed - decode_steps * bs <= 0:
                return AddReqResult.NO_TOKEN
            tokens_freed += tokens_occupied

        if (
            self.rem_chunk_tokens is None
            or req.extend_input_len <= self.rem_chunk_tokens
        ):
            self.can_run_list.append(req)
            self._prefill_one_req(
                0,
                req.extend_input_len,
                min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS),
            )
        else:
            # Chunked prefill
            trunc_len = self.rem_chunk_tokens
            req.extend_input_len = trunc_len
            req.fill_ids = req.fill_ids[:trunc_len]
            self.can_run_list.append(req)
            self.new_inflight_req = req
            self._prefill_one_req(0, trunc_len, 0)

        return self.budget_state()

    def add_one_req(self, req: Req):
        if req.sampling_params.ignore_eos and self.tree_cache.disable:
            return self.add_one_req_ignore_eos(req)

        total_tokens = req.extend_input_len + min(
            req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS
        )
        input_tokens = req.extend_input_len
        prefix_len = len(req.prefix_indices)

        if total_tokens >= self.rem_total_tokens:
            return AddReqResult.NO_TOKEN

        if input_tokens > self.rem_input_tokens and len(self.can_run_list) != 0:
            return AddReqResult.OTHER

        with self._lock_node(req.last_node):
            if total_tokens > self.rem_total_tokens:
                return AddReqResult.NO_TOKEN

            if (
                self.rem_chunk_tokens is None
                or input_tokens <= self.rem_chunk_tokens
                or (req.return_logprob and req.normalized_prompt_logprob is None)
            ):
                # Non-chunked prefill
                self.can_run_list.append(req)
                self.tree_cache.inc_lock_ref(req.last_node)
                self._prefill_one_req(
                    prefix_len,
                    input_tokens,
                    min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS),
                )
            else:
                # Chunked prefill
                trunc_len = self.rem_chunk_tokens
                if trunc_len == 0:
                    return AddReqResult.OTHER

                req.extend_input_len = trunc_len
                req.fill_ids = req.fill_ids[: len(req.prefix_indices) + trunc_len]
                self.can_run_list.append(req)
                self.new_inflight_req = req
                self.tree_cache.inc_lock_ref(req.last_node)
                self._prefill_one_req(prefix_len, trunc_len, 0)

        return self.budget_state()
