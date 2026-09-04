from __future__ import annotations

from bisect import bisect_left
from typing import Sequence

from sglang.srt.speculative.racer.automaton import RacerAutomaton, _CandidateTrie


class BinaryRefillRacerAutomaton(RacerAutomaton):
    """RACER with a fixed-K, unique-aware TokenBin refill for SGLang.

    The first RetrievalTree/TokenBin split is identical to the original RACER
    C++ implementation. If candidate-trie merging leaves fewer than K unique
    nodes, extend the same TokenBin BFS trace and find the smallest raw TokenBin
    budget whose merged trie reaches K.

    The refill search is intentionally cheap: TokenBin's raw BFS is generated
    once (up to 4K nodes), cumulative unique-node counts are recorded while the
    trace is merged, and ``bisect_left`` finds the minimal sufficient raw budget.
    This preserves the binary-search semantics without replaying TokenBin and
    rebuilding the candidate trie for every probe.
    """

    def _token_bin_raw_trace(
        self, next_token: int, max_num_draft: int
    ) -> list[tuple[int, int, int, int]]:
        """Return TokenBin BFS nodes in the exact raw-budget order.

        Each tuple is ``(token, parent_position, breadth, depth)``. The first B
        entries are exactly the raw TokenBin tree that the original C++
        ``TokenBin::retrieve(next_token, B)`` explores. Inserting every root-to-
        node path from that prefix yields the same candidate trie as inserting
        only its recovered leaf paths, because internal nodes are prefixes of
        those leaves.
        """

        if max_num_draft <= 0:
            return []

        q: list[tuple[int, int, int, int]] = [
            (int(next_token), -1, self.topk, 0)
        ]
        remaining = int(max_num_draft) - 1
        head = 0

        while head < len(q):
            token, _, breadth, depth = q[head]
            pos_u = head
            head += 1

            if remaining <= 0 or breadth <= 0:
                continue

            row = self._token_bin_row(token)
            next_breadth = breadth if depth == 1 else (breadth >> 1)
            next_depth = depth + 1
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

        return q

    @staticmethod
    def _trace_path(
        trace: Sequence[tuple[int, int, int, int]], index: int
    ) -> list[int]:
        path: list[int] = []
        cur = int(index)
        while cur >= 0:
            token, parent, _, _ = trace[cur]
            path.append(int(token))
            cur = int(parent)
        path.reverse()
        return path

    @staticmethod
    def _leaf_count(
        trace: Sequence[tuple[int, int, int, int]], prefix_len: int
    ) -> int:
        """Number of leaf paths returned by TokenBin for a raw prefix."""

        n = min(max(0, int(prefix_len)), len(trace))
        if n == 0:
            return 0
        is_leaf = [True] * n
        for i in range(1, n):
            parent = int(trace[i][1])
            if 0 <= parent < n:
                is_leaf[parent] = False
        return sum(is_leaf)

    def retrieve(
        self, root_token: int, max_num_draft: int
    ) -> tuple[list[int], list[list[bool]]]:
        budget = int(max_num_draft)
        if budget <= 0:
            raise ValueError("max_num_draft must be greater than 0")
        root_token = int(root_token)

        borders = self._collect_borders(root_token)
        selected = self._select_retrieval_nodes(borders, budget)
        retrieval_paths = self._selected_paths(selected)

        candidate_trie = _CandidateTrie()
        for path in retrieval_paths:
            candidate_trie.insert(path, budget)
        retrieval_unique_nodes = candidate_trie.node_count
        merge_holes = max(0, len(selected) - retrieval_unique_nodes)

        # Exact original C++ accounting before candidate-trie merging.
        original_tokenbin_budget = max(0, budget - len(selected))
        max_trace_budget = max(4 * budget, original_tokenbin_budget)
        trace = self._token_bin_raw_trace(root_token, max_trace_budget)
        original_tokenbin_paths = self._leaf_count(
            trace, original_tokenbin_budget
        )

        cumulative_unique_nodes: list[int] = []
        nodes_after_original_tokenbin = retrieval_unique_nodes
        reached_full_at: int | None = None

        # Merge one TokenBin trace only once. Counts are monotone in raw budget.
        for raw_index in range(len(trace)):
            candidate_trie.insert(
                self._trace_path(trace, raw_index),
                budget,
            )
            raw_budget = raw_index + 1
            cumulative_unique_nodes.append(candidate_trie.node_count)

            if raw_budget == original_tokenbin_budget:
                nodes_after_original_tokenbin = candidate_trie.node_count

            if candidate_trie.node_count >= budget:
                reached_full_at = raw_budget
                # If K was reached before the original TokenBin allowance was
                # exhausted, the original proposal would also be full. We still
                # know its node count is K because candidate capacity is K.
                if raw_budget <= original_tokenbin_budget:
                    nodes_after_original_tokenbin = budget
                break

        if original_tokenbin_budget == 0:
            nodes_after_original_tokenbin = retrieval_unique_nodes
        elif (
            reached_full_at is None
            and original_tokenbin_budget > len(cumulative_unique_nodes)
        ):
            nodes_after_original_tokenbin = candidate_trie.node_count

        refill_used = nodes_after_original_tokenbin < budget
        refill_budget = original_tokenbin_budget
        refill_probes = 0

        if refill_used:
            refill_probes = 1  # one shared raw trace; no TokenBin replay probes
            if cumulative_unique_nodes and cumulative_unique_nodes[-1] >= budget:
                # Lower bound on the monotone cumulative unique-node counts.
                refill_budget = bisect_left(
                    cumulative_unique_nodes, budget
                ) + 1
            else:
                # Even 4K raw nodes cannot produce K unique candidates, usually
                # because the request is cold and many TokenBin rows are still
                # zero-initialized. Keep the best merged trie and let the
                # defensive fixed-K padding handle the residual gap.
                refill_budget = len(cumulative_unique_nodes)

        nodes_after_refill = candidate_trie.node_count
        refill_unique_added = max(
            0, nodes_after_refill - nodes_after_original_tokenbin
        )

        if candidate_trie.node_count == 0:
            candidate_trie.insert([root_token], budget)

        nodes_before_padding = candidate_trie.node_count
        padding_nodes = max(0, budget - nodes_before_padding)

        self._record_proposal_shape(
            borders=len(borders),
            retrieval_selected=len(selected),
            retrieval_unique_nodes=retrieval_unique_nodes,
            merge_holes=merge_holes,
            tokenbin_budget=original_tokenbin_budget,
            tokenbin_paths=original_tokenbin_paths,
            nodes_after_original_tokenbin=nodes_after_original_tokenbin,
            refill_used=int(refill_used),
            refill_budget=refill_budget,
            refill_probes=refill_probes,
            nodes_after_refill=nodes_after_refill,
            refill_unique_added=refill_unique_added,
            nodes_before_padding=nodes_before_padding,
            padding_nodes=padding_nodes,
        )

        self._defensive_pad(candidate_trie, budget, root_token)

        tokens, mask = candidate_trie.flatten(budget)
        if len(tokens) != budget:
            raise RuntimeError(
                f"RACER fixed-K proposal invariant failed: {len(tokens)=}, {budget=}"
            )
        return tokens, mask
