from __future__ import annotations

from dataclasses import asdict, dataclass

from sglang.srt.speculative.racer.automaton import RacerAutomaton


@dataclass
class RacerProposalStats:
    borders: int = 0
    retrieval_selected: int = 0
    retrieval_unique_nodes: int = 0
    merge_holes: int = 0
    tokenbin_budget: int = 0
    tokenbin_paths: int = 0
    nodes_after_original_tokenbin: int = 0
    refill_used: int = 0
    refill_budget: int = 0
    refill_probes: int = 0
    nodes_after_refill: int = 0
    refill_unique_added: int = 0
    nodes_before_padding: int = 0
    padding_nodes: int = 0
    proposal_ms: float = 0.0

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


class InstrumentedRacerAutomaton(RacerAutomaton):
    """RACER automaton with proposal-shape instrumentation only."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_proposal_stats = RacerProposalStats()

    def retrieve(self, root_token: int, max_num_draft: int):
        self.last_proposal_stats = RacerProposalStats()
        return super().retrieve(root_token, max_num_draft)

    def _record_proposal_shape(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self.last_proposal_stats, key):
                setattr(self.last_proposal_stats, key, value)
