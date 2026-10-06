from __future__ import annotations

from dataclasses import dataclass, field

from sglang.srt.debug_utils.comparator.aligner.token_aligner.smart.types import (
    SeqId,
    TokenAlignerGlobalAux,
    TokenAlignerSeqInfo,
    TokenAlignerSeqsInfo,
    TokenAlignerStepAux,
    TokenLocator,
)


@dataclass
class _SeqInfoAccumulator:
    """Mutable accumulator for building TokenAlignerSeqInfo without per-step validation."""

    input_ids: list[int] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    token_index_in_step: list[int] = field(default_factory=list)

    def extend(
        self,
        *,
        input_ids: list[int],
        positions: list[int],
        steps: list[int],
        token_index_in_step: list[int],
    ) -> None:
        self.input_ids.extend(input_ids)
        self.positions.extend(positions)
        self.steps.extend(steps)
        self.token_index_in_step.extend(token_index_in_step)

    def build(self) -> TokenAlignerSeqInfo:
        return TokenAlignerSeqInfo(
            input_ids=self.input_ids,
            positions=self.positions,
            locator=TokenLocator(
                steps=self.steps,
                token_index_in_step=self.token_index_in_step,
            ),
        )


def build_seqs_info(global_aux: TokenAlignerGlobalAux) -> TokenAlignerSeqsInfo:
    """Build sequence info for one side from its auxiliary tensors."""
    return TokenAlignerSeqsInfo(
        sequences=_build_token_aligner_seq_infos(global_aux),
        layout=global_aux.layout,
    )


def _build_token_aligner_seq_infos(
    global_aux: TokenAlignerGlobalAux,
) -> dict[SeqId, TokenAlignerSeqInfo]:
    """Build token index for any framework/layout using seq_ids for identity tracking."""
    accum: dict[SeqId, _SeqInfoAccumulator] = {}

    for step in sorted(global_aux.step_auxs.keys()):
        aux: TokenAlignerStepAux = global_aux.step_auxs[step]

        offset: int = 0
        for seq_index, seq_len in enumerate(aux.seq_lens):
            seq_id: SeqId = aux.seq_ids[seq_index]

            if seq_id not in accum:
                accum[seq_id] = _SeqInfoAccumulator()

            accum[seq_id].extend(
                input_ids=aux.input_ids[offset : offset + seq_len],
                positions=aux.positions[offset : offset + seq_len],
                steps=[step] * seq_len,
                token_index_in_step=list(range(offset, offset + seq_len)),
            )

            offset += seq_len

    return {seq_id: acc.build() for seq_id, acc in accum.items()}
