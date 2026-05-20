from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import polars as pl

from sglang.srt.debug_utils.comparator.aligner.token_aligner.concat_steps.thd_seq_lens_loader import (
    load_thd_seq_lens_only,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.smart.aux_loader import (
    has_aux_tensors,
    load_and_normalize_aux,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.smart.planner import (
    compute_token_aligner_plan,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.smart.seq_info_builder import (
    build_seqs_info,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.smart.types import (
    TokenAlignerGlobalAux,
    TokenAlignerPlan,
    TokenAlignerSeqsInfo,
)
from sglang.srt.debug_utils.comparator.log_sink import log_sink
from sglang.srt.debug_utils.comparator.output_types import InfoLog
from sglang.srt.debug_utils.comparator.utils import Pair

_NONE_THD: Pair[Optional[dict[int, list[int]]]] = Pair(x=None, y=None)


TokenAlignerMode = Literal["concat_steps", "smart"]


@dataclass(frozen=True)
class TokenAlignerResult:
    """Result of token aligner computation, bundling mode + plan with THD metadata."""

    mode: Optional[TokenAlignerMode]
    plan: Optional[TokenAlignerPlan]
    thd_seq_lens_by_step_pair: Pair[Optional[dict[int, list[int]]]]


def compute_maybe_token_aligner_result(
    *,
    dir_pair: Pair[Path],
    dfs: Pair[pl.DataFrame],
    token_aligner_mode: Optional[TokenAlignerMode],
) -> TokenAlignerResult:
    if token_aligner_mode is None:
        return TokenAlignerResult(
            mode=None, plan=None, thd_seq_lens_by_step_pair=_NONE_THD
        )

    if token_aligner_mode == "concat_steps":
        thd_pair: Pair[Optional[dict[int, list[int]]]] = _load_thd_seq_lens_pair(
            dir_pair=dir_pair, dfs=dfs
        )
        return TokenAlignerResult(
            mode="concat_steps", plan=None, thd_seq_lens_by_step_pair=thd_pair
        )
    elif token_aligner_mode == "smart":
        if not (has_aux_tensors(dfs.x) and has_aux_tensors(dfs.y)):
            log_sink.add(
                InfoLog(
                    category="aux_tensors_missing",
                    message="Aux tensors missing, skipping token alignment",
                )
            )
            return TokenAlignerResult(
                mode=None, plan=None, thd_seq_lens_by_step_pair=_NONE_THD
            )

        return _build_smart_result(dir_pair=dir_pair, dfs=dfs)
    else:
        raise NotImplementedError(f"Unknown {token_aligner_mode=}")


def _build_smart_result(
    *,
    dir_pair: Pair[Path],
    dfs: Pair[pl.DataFrame],
) -> TokenAlignerResult:
    """Load aux tensors, build token indices, and compute the alignment plan."""
    aux_pair: Pair[Optional[TokenAlignerGlobalAux]] = Pair(
        x=load_and_normalize_aux(dump_path=dir_pair.x, df=dfs.x),
        y=load_and_normalize_aux(dump_path=dir_pair.y, df=dfs.y),
    )

    thd_seq_lens_by_step_pair: Pair[Optional[dict[int, list[int]]]] = aux_pair.map(
        lambda aux: aux.thd_seq_lens_by_step if aux is not None else None
    )

    if aux_pair.x is None or aux_pair.y is None:
        log_sink.add(
            InfoLog(
                category="framework_detection_failed",
                message="Framework detection failed, skipping token alignment",
            )
        )
        return TokenAlignerResult(
            mode=None,
            plan=None,
            thd_seq_lens_by_step_pair=thd_seq_lens_by_step_pair,
        )

    global_aux: Pair[TokenAlignerGlobalAux] = Pair(x=aux_pair.x, y=aux_pair.y)

    seqs_info: Pair[TokenAlignerSeqsInfo] = global_aux.map(build_seqs_info)

    plan: Optional[TokenAlignerPlan] = compute_token_aligner_plan(
        seqs_info_pair=seqs_info
    )
    return TokenAlignerResult(
        mode="smart",
        plan=plan,
        thd_seq_lens_by_step_pair=thd_seq_lens_by_step_pair,
    )


def _load_thd_seq_lens_pair(
    *,
    dir_pair: Pair[Path],
    dfs: Pair[pl.DataFrame],
) -> Pair[Optional[dict[int, list[int]]]]:
    """Load only thd_seq_lens for each side (lightweight, no full aux loading)."""
    return Pair(
        x=load_thd_seq_lens_only(dump_path=dir_pair.x, df=dfs.x),
        y=load_thd_seq_lens_only(dump_path=dir_pair.y, df=dfs.y),
    )
