from __future__ import annotations

from typing import Sequence

from sglang.srt.speculative.racer.automaton import RacerAutomaton, _CandidateTrie


class BinaryRefillRacerAutomaton(RacerAutomaton):
    """RACER with a fixed-K, unique-aware TokenBin refill for SGLang.

    The first RetrievalTree/TokenBin split is identical to the original RACER
    C++ implementation.  If candidate-trie merging leaves fewer than K unique
    nodes, replay TokenBin from the same next-token root with a larger raw node
    budget.  Since a larger TokenBin budget only extends previously recovered
    paths, the merged unique-node count is monotone, so we can binary-search the
    smallest raw budget that reaches K.
    """

    def _build_candidate_trie(
        self,
        retrieval_paths: Sequence[Sequence[int]],
        root_token: int,
        tokenbin_budget: int,
        budget: int,
    ) -> tuple[_CandidateTrie, int]:
        trie = _CandidateTrie()
        for path in retrieval_paths:
            trie.insert(path, budget)

        tokenbin_paths = self._token_bin_retrieve(
            root_token, tokenbin_budget, is_chain=False
        )
        for path in tokenbin_paths:
            trie.insert(path, budget)
        return trie, len(tokenbin_paths)

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

        retrieval_trie = _CandidateTrie()
        for path in retrieval_paths:
            retrieval_trie.insert(path, budget)
        retrieval_unique_nodes = retrieval_trie.node_count
        merge_holes = max(0, len(selected) - retrieval_unique_nodes)

        # Pass 1: exact C++ accounting before candidate-trie merging.
        original_tokenbin_budget = budget - len(selected)
        candidate_trie, original_tokenbin_paths = self._build_candidate_trie(
            retrieval_paths,
            root_token,
            original_tokenbin_budget,
            budget,
        )
        nodes_after_original_tokenbin = candidate_trie.node_count

        refill_used = candidate_trie.node_count < budget
        refill_budget = original_tokenbin_budget
        refill_probes = 0

        if refill_used:
            # Find a sufficient upper bound first. K is tiny (typically 16-64),
            # so replaying TokenBin a handful of times is negligible compared
            # with target verification.
            lo = original_tokenbin_budget + 1
            hi = max(lo, budget)
            max_hi = max(4 * budget, hi)

            hi_trie, _ = self._build_candidate_trie(
                retrieval_paths, root_token, hi, budget
            )
            refill_probes += 1

            while hi_trie.node_count < budget and hi < max_hi:
                lo = hi + 1
                hi = min(max_hi, max(hi + 1, hi * 2))
                hi_trie, _ = self._build_candidate_trie(
                    retrieval_paths, root_token, hi, budget
                )
                refill_probes += 1

            if hi_trie.node_count >= budget:
                # Lower-bound binary search for the smallest raw TokenBin
                # budget whose merged candidate trie reaches K unique nodes.
                left = original_tokenbin_budget + 1
                right = hi
                while left < right:
                    mid = (left + right) // 2
                    mid_trie, _ = self._build_candidate_trie(
                        retrieval_paths, root_token, mid, budget
                    )
                    refill_probes += 1
                    if mid_trie.node_count >= budget:
                        right = mid
                    else:
                        left = mid + 1

                refill_budget = left
                candidate_trie, _ = self._build_candidate_trie(
                    retrieval_paths, root_token, refill_budget, budget
                )
                refill_probes += 1
            else:
                # Even 4K raw TokenBin nodes were insufficient, typically due
                # to very low-diversity or zero-initialized rows.  Keep the best
                # trie and let the existing defensive pad handle the remainder.
                refill_budget = hi
                candidate_trie = hi_trie

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
