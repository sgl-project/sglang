"""Request scheduler heuristic."""

import random
from collections import defaultdict


class ScheduleHeuristic:
    def __init__(
        self,
        schedule_heuristic,
        max_running_seqs,
        max_prefill_num_tokens,
        max_total_num_tokens,
        tree_cache,
    ):
        self.schedule_heuristic = schedule_heuristic
        self.max_running_seqs = max_running_seqs
        self.max_prefill_num_tokens = max_prefill_num_tokens
        self.max_total_num_tokens = max_total_num_tokens
        self.tree_cache = tree_cache

    def get_priority_queue(self, forward_queue):
        if self.schedule_heuristic == "lpm":
            # longest prefix match
            forward_queue.sort(key=lambda x: -len(x.prefix_indices))
            return forward_queue
        elif self.schedule_heuristic == "random":
            random.shuffle(forward_queue)
            return forward_queue
        elif self.schedule_heuristic == "fcfs":
            return forward_queue
        elif self.schedule_heuristic == "dfs-weight":
            last_node_to_reqs = defaultdict(list)
            for req in forward_queue:
                last_node_to_reqs[req.last_node].append(req)

            node_to_weight = defaultdict(int)
            for node in last_node_to_reqs:
                node_to_weight[node] = len(last_node_to_reqs[node])
            self.calc_weight(self.tree_cache.root_node, node_to_weight)

            q = []
            self.get_dfs_priority(
                self.tree_cache.root_node, node_to_weight, last_node_to_reqs, q
            )
            assert len(q) == len(forward_queue)
            return q
        else:
            raise ValueError(f"Unknown schedule_heuristic: {self.schedule_heuristic}")

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
