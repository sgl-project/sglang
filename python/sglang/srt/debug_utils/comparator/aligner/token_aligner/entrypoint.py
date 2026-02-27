from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import polars as pl

from sglang.srt.debug_utils.comparator.aligner.token_aligner.aux_loader import (
    has_aux_tensors,
    load_and_normalize_aux,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.planner import (
    compute_token_aligner_plan,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.seq_info_builder import (
    build_seqs_info,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
    TokenAlignerGlobalAux,
    TokenAlignerPlan,
    TokenAlignerSeqsInfo,
)
from sglang.srt.debug_utils.comparator.output_types import GeneralWarning
from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.srt.debug_utils.comparator.warning_sink import warning_sink


@dataclass(frozen=True)
class TokenAlignerResult:
    """Result of token aligner computation, bundling the plan with THD metadata."""

    plan: Optional[TokenAlignerPlan]
    thd_seq_lens_by_step_pair: Pair[Optional[dict[int, list[int]]]]


def compute_maybe_token_aligner_result(
    args: argparse.Namespace,
    dfs: Pair[pl.DataFrame],
) -> TokenAlignerResult:
    if args.grouping == "logical":
        if not (has_aux_tensors(dfs.x) and has_aux_tensors(dfs.y)):
            warning_sink.add(
                GeneralWarning(
                    category="aux_tensors_missing",
                    message="Aux tensors missing, skipping token alignment",
                )
            )
            return TokenAlignerResult(
                plan=None, thd_seq_lens_by_step_pair=Pair(x=None, y=None)
            )

        return _build_token_aligner_result(args=args, dfs=dfs)

    return TokenAlignerResult(plan=None, thd_seq_lens_by_step_pair=Pair(x=None, y=None))


def _build_token_aligner_result(
    *,
    args: argparse.Namespace,
    dfs: Pair[pl.DataFrame],
) -> TokenAlignerResult:
    """Load aux tensors, build token indices, and compute the alignment plan."""
    dump_paths: Pair[Path] = Pair(x=Path(args.baseline_path), y=Path(args.target_path))

    baseline_aux: Optional[TokenAlignerGlobalAux] = load_and_normalize_aux(
        dump_path=dump_paths.x, df=dfs.x
    )
    target_aux: Optional[TokenAlignerGlobalAux] = load_and_normalize_aux(
        dump_path=dump_paths.y, df=dfs.y
    )

    thd_seq_lens_by_step_pair: Pair[Optional[dict[int, list[int]]]] = Pair(
        x=baseline_aux.thd_seq_lens_by_step if baseline_aux is not None else None,
        y=target_aux.thd_seq_lens_by_step if target_aux is not None else None,
    )

    if baseline_aux is None or target_aux is None:
        warning_sink.add(
            GeneralWarning(
                category="framework_detection_failed",
                message="Framework detection failed, skipping token alignment",
            )
        )
        return TokenAlignerResult(
            plan=None, thd_seq_lens_by_step_pair=thd_seq_lens_by_step_pair
        )

    global_aux: Pair[TokenAlignerGlobalAux] = Pair(
        x=baseline_aux,
        y=target_aux,
    )

    seqs_info: Pair[TokenAlignerSeqsInfo] = global_aux.map(build_seqs_info)

    plan: Optional[TokenAlignerPlan] = compute_token_aligner_plan(
        seqs_info_pair=seqs_info
    )
    return TokenAlignerResult(
        plan=plan, thd_seq_lens_by_step_pair=thd_seq_lens_by_step_pair
    )
