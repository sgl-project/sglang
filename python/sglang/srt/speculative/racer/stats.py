from __future__ import annotations

from dataclasses import asdict, dataclass

from sglang.srt.speculative.racer.automaton import RacerAutomaton


@dataclass
class RacerProposalStats:
    borders: int = 0
    retrieval_selected: int = 0
    tokenbin_budget: int = 0
    tokenbin_paths: int = 0
    nodes_before_padding: int = 0
    padding_nodes: int = 0
    proposal_ms: float = 0.0

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


class InstrumentedRacerAutomaton(RacerAutomaton):
    """RACER automaton with proposal-shape instrumentation only.

    All proposal decisions remain in RacerAutomaton.  The overrides below only
    observe inputs/outputs of the original helpers, so enabling statistics does
    not change Retrieval Tree or TokenBin semantics.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_proposal_stats = RacerProposalStats()

    def retrieve(self, root_token: int, max_num_draft: int):
        self.last_proposal_stats = RacerProposalStats()
        return super().retrieve(root_token, max_num_draft)

    def _collect_borders(self, next_token: int):
        borders = super()._collect_borders(next_token)
        self.last_proposal_stats.borders = len(borders)
        return borders

    def _select_retrieval_nodes(self, borders, budget: int):
        selected = super()._select_retrieval_nodes(borders, budget)
        self.last_proposal_stats.retrieval_selected = len(selected)
        return selected

    def _token_bin_retrieve(
        self, next_token: int, max_num_draft: int, is_chain: bool = False
    ):
        self.last_proposal_stats.tokenbin_budget = max(0, int(max_num_draft))
        paths = super()._token_bin_retrieve(next_token, max_num_draft, is_chain)
        self.last_proposal_stats.tokenbin_paths = len(paths)
        return paths

    def _defensive_pad(self, trie, budget: int, root_token: int) -> None:
        before = int(trie.node_count)
        self.last_proposal_stats.nodes_before_padding = before
        self.last_proposal_stats.padding_nodes = max(0, int(budget) - before)
        super()._defensive_pad(trie, budget, root_token)
