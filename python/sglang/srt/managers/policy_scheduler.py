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

"""Request policy scheduler"""

import random
from collections import defaultdict


class PolicyScheduler:
    def __init__(
        self,
        policy,
        max_running_seqs,
        max_prefill_num_tokens,
        max_total_num_tokens,
        tree_cache,
    ):
        if tree_cache.disable and policy == "lpm":
            # LMP is meaningless when the tree cache is disabled.
            policy = "fcfs"

        self.policy = policy
        self.max_running_seqs = max_running_seqs
        self.max_prefill_num_tokens = max_prefill_num_tokens
        self.max_total_num_tokens = max_total_num_tokens
        self.tree_cache = tree_cache

    def get_priority_queue(self, waiting_queue):
        if self.policy == "lpm":
            # longest prefix match
            waiting_queue.sort(key=lambda x: -len(x.prefix_indices))
            return waiting_queue
        elif self.policy == "fcfs":
            # first come first serve
            return waiting_queue
        elif self.policy == "lof":
            # longest output first
            waiting_queue.sort(key=lambda x: -x.sampling_params.max_new_tokens)
            return waiting_queue
        elif self.policy == "random":
            random.shuffle(waiting_queue)
            return waiting_queue
        elif self.policy == "dfs-weight":
            last_node_to_reqs = defaultdict(list)
            for req in waiting_queue:
                last_node_to_reqs[req.last_node].append(req)

            node_to_weight = defaultdict(int)
            for node in last_node_to_reqs:
                node_to_weight[node] = len(last_node_to_reqs[node])
            self.calc_weight(self.tree_cache.root_node, node_to_weight)

            q = []
            self.get_dfs_priority(
                self.tree_cache.root_node, node_to_weight, last_node_to_reqs, q
            )
            assert len(q) == len(waiting_queue)
            return q
        else:
            raise ValueError(f"Unknown schedule_policy: {self.policy}")

    def calc_weight(self, cur_node, node_to_weight):
        for child in cur_node.children.values():
            self.calc_weight(child, node_to_weight)
            node_to_weight[cur_node] += node_to_weight[child]

    def get_dfs_priority(self, cur_node, node_to_priority, last_node_to_reqs, q):
        childs = [child for child in cur_node.children.values()]
        childs.sort(key=lambda x: -node_to_priority[x])
        for child in childs:
            self.get_dfs_priority(child, node_to_priority, last_node_to_reqs, q)
        q.extend(last_node_to_reqs[cur_node])
