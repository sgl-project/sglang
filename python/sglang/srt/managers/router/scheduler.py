import random
from collections import defaultdict


class Scheduler:
    def __init__(
        self,
        schedule_heuristic,
        max_running_seq,
        max_prefill_num_token,
        max_total_num_token,
        tree_cache,
    ):
        self.schedule_heuristic = schedule_heuristic
        self.max_running_seq = max_running_seq
        self.max_prefill_num_token = max_prefill_num_token
        self.max_total_num_token = max_total_num_token
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
        elif self.schedule_heuristic == "weight":
            last_node_to_reqs = defaultdict(list)
            for req in forward_queue:
                last_node_to_reqs[req.last_node].append(req)
            for node in last_node_to_reqs:
                last_node_to_reqs[node].sort(key=lambda x: -len(x.prefix_indices))

            node_to_weight = defaultdict(int)
            self._calc_weight_recursive(
                self.tree_cache.root_node, last_node_to_reqs, node_to_weight
            )

            tmp_queue = []
            self._get_weight_priority_recursive(
                self.tree_cache.root_node, node_to_weight, last_node_to_reqs, tmp_queue
            )
            assert len(tmp_queue) == len(forward_queue)
            return tmp_queue
        else:
            raise ValueError(f"Unknown schedule_heuristic: {self.schedule_heuristic}")

    def _calc_weight_recursive(self, cur_node, last_node_to_reqs, node_to_weight):
        node_to_weight[cur_node] = 1
        if cur_node in last_node_to_reqs:
            node_to_weight[cur_node] += len(last_node_to_reqs[cur_node])
        for child in cur_node.children.values():
            self._calc_weight_recursive(child, last_node_to_reqs, node_to_weight)
            node_to_weight[cur_node] += node_to_weight[child]

    def _get_weight_priority_recursive(
        self, cur_node, node_to_wight, last_node_to_reqs, tmp_queue
    ):
        visit_list = [child for child in cur_node.children.values()]
        visit_list.sort(key=lambda x: -node_to_wight[x])
        # for node in visit_list:
        #     print(f"{node_to_wight[node]} {len(node.value) if node.value is not None else 0}")
        for child in visit_list:
            self._get_weight_priority_recursive(
                child, node_to_wight, last_node_to_reqs, tmp_queue
            )
        tmp_queue.extend(last_node_to_reqs[cur_node])
