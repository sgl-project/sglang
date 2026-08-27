"""Minimal demo: run the comparator on synthetic data and print its output.

This is NOT a correctness test suite.
The sole purpose is to let a new user run ``pytest -s test_e2e_demo.py``
and immediately see what comparator text output looks like (passed, failed,
skipped in one shot).  Correctness is verified via the JSONL report file.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import pytest
import torch

import sglang.srt.debug_utils.dumper as _dumper_module
from sglang.srt.debug_utils.comparator.entrypoint import parse_args, run
from sglang.srt.debug_utils.comparator.output_types import (
    AnyRecord,
    ComparisonErrorRecord,
    SummaryRecord,
    parse_record_json,
)
from sglang.srt.debug_utils.dumper import DumperConfig, _Dumper
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)

_EXP_NAME = "demo_exp"


# This file has exactly ONE test. All demo scenarios go here — do not add separate tests.
def test_demo(tmp_path: Path) -> None:
    """Passed + failed + skipped + sharded + errored in a single demo file."""
    torch.manual_seed(0)
    good_tensor = torch.randn(4, 8)
    sharded_full = torch.randn(2, 8, 16)

    baseline_dir = tmp_path / "baseline"
    target_dir = tmp_path / "target"
    baseline_dir.mkdir()
    target_dir.mkdir()

    # Step 1: simple tensors (single rank, no parallelism)
    _dump_single(baseline_dir, name="my_good_tensor", tensor=good_tensor)
    _dump_single(baseline_dir, name="my_bad_tensor", tensor=torch.randn(4, 8))

    _dump_single(
        target_dir, name="my_good_tensor", tensor=good_tensor + torch.randn(4, 8) * 1e-5
    )
    _dump_single(target_dir, name="my_bad_tensor", tensor=torch.randn(4, 8) * 100)
    _dump_single(target_dir, name="my_orphan_tensor", tensor=torch.randn(4, 8))

    # Step 2: sharded tensor (BSHD) — baseline: TP=2 on h, target: CP=2 zigzag + SP=2 on s
    sharded_target = sharded_full + torch.randn_like(sharded_full) * 1e-5
    _dump_tp_sharded(
        baseline_dir, name="my_sharded_tensor", full_tensor=sharded_full, tp_size=2
    )
    _dump_cp_zigzag_sp_sharded(
        target_dir,
        name="my_sharded_tensor",
        full_tensor=sharded_target,
        cp_size=2,
        sp_size=2,
    )

    # Step 3: bad dims — target says h[cp] but parallel_info has tp → undeclared axis error
    bad_dims_tensor = torch.randn(2, 8, 16)
    for tp_rank, shard in enumerate(bad_dims_tensor.chunk(2, dim=-1)):
        _dump_rank(
            baseline_dir,
            rank=tp_rank,
            name="my_bad_dims_tensor",
            tensor=shard,
            dims="b s h[tp]",
            parallel_info={"tp_rank": tp_rank, "tp_size": 2},
        )
        _dump_rank(
            target_dir,
            rank=tp_rank,
            name="my_bad_dims_tensor",
            tensor=shard,
            dims="b s h[cp]",
            parallel_info={"tp_rank": tp_rank, "tp_size": 2},
        )

    baseline_exp = baseline_dir / _EXP_NAME
    target_exp = target_dir / _EXP_NAME

    # Step 4: run normal, then verbose
    for verbosity in ("normal", "verbose"):
        report_path = tmp_path / f"report_{verbosity}.jsonl"
        _run(
            baseline_exp,
            target_exp,
            report_path=report_path,
            output_format="text",
            verbosity=verbosity,
        )
        _assert_summary(report_path, passed=2, failed=1, skipped=1, errored=1)

    # Step 5: verify error record content
    records = _read_report(tmp_path / "report_verbose.jsonl")
    errors = [r for r in records if isinstance(r, ComparisonErrorRecord)]
    assert len(errors) == 1
    assert "tp" in errors[0].exception_message
    assert "--override-dims" in errors[0].traceback_str


# ── Helpers ──────────────────────────────────────────────────────────


def _assert_summary(
    report_path: Path, *, passed: int, failed: int, skipped: int, errored: int = 0
) -> None:
    records = _read_report(report_path)
    summary = next(r for r in records if isinstance(r, SummaryRecord))
    assert summary.passed == passed
    assert summary.failed == failed
    assert summary.skipped == skipped
    assert summary.errored == errored


def _dump_single(directory: Path, *, name: str, tensor: torch.Tensor) -> None:
    _dump_rank(directory, rank=0, name=name, tensor=tensor)


def _dump_tp_sharded(
    directory: Path,
    *,
    name: str,
    full_tensor: torch.Tensor,
    tp_size: int,
) -> None:
    """Dump TP-sharded tensor: dims="b s h[tp]", shard along last dim."""
    shards = list(full_tensor.chunk(tp_size, dim=-1))
    for tp_rank, shard in enumerate(shards):
        _dump_rank(
            directory,
            rank=tp_rank,
            name=name,
            tensor=shard,
            dims="b s h[tp]",
            parallel_info={"tp_rank": tp_rank, "tp_size": tp_size},
        )


def _dump_cp_zigzag_sp_sharded(
    directory: Path,
    *,
    name: str,
    full_tensor: torch.Tensor,
    cp_size: int,
    sp_size: int,
) -> None:
    """Dump CP-zigzag+SP sharded tensor: dims="b s[cp:zigzag,sp] h", shard seq dim."""
    seq_dim = 1
    num_chunks = cp_size * 2
    natural_chunks = list(full_tensor.chunk(num_chunks, dim=seq_dim))

    zigzag_order: List[int] = []
    for i in range(cp_size):
        zigzag_order.append(i)
        zigzag_order.append(num_chunks - 1 - i)

    zigzagged = torch.cat([natural_chunks[idx] for idx in zigzag_order], dim=seq_dim)
    cp_chunks = list(zigzagged.chunk(cp_size, dim=seq_dim))

    rank = 0
    for cp_rank in range(cp_size):
        sp_chunks = list(cp_chunks[cp_rank].chunk(sp_size, dim=seq_dim))
        for sp_rank in range(sp_size):
            _dump_rank(
                directory,
                rank=rank,
                name=name,
                tensor=sp_chunks[sp_rank],
                dims="b s[cp:zigzag,sp] h",
                parallel_info={
                    "cp_rank": cp_rank,
                    "cp_size": cp_size,
                    "sp_rank": sp_rank,
                    "sp_size": sp_size,
                },
            )
            rank += 1


def _dump_rank(
    directory: Path,
    *,
    rank: int,
    name: str,
    tensor: torch.Tensor,
    dims: Optional[str] = None,
    parallel_info: Optional[Dict[str, int]] = None,
) -> None:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_dumper_module, "_get_rank", lambda: rank)
        dumper = _Dumper(
            config=DumperConfig(enable=True, dir=str(directory), exp_name=_EXP_NAME)
        )
        static_meta: Dict[str, object] = {"world_rank": rank, "world_size": 1}
        if parallel_info is not None:
            static_meta["sglang_parallel_info"] = parallel_info
        dumper.__dict__["_static_meta"] = static_meta
        dumper.dump(name, tensor, dims=dims)
        dumper.step()


def _run(
    baseline_path: Path,
    target_path: Path,
    *,
    report_path: Path,
    output_format: str = "text",
    verbosity: str = "normal",
) -> int:
    argv = [
        "--baseline-path",
        str(baseline_path),
        "--target-path",
        str(target_path),
        "--output-format",
        output_format,
        "--verbosity",
        verbosity,
        "--preset",
        "sglang_dev",
        "--report-path",
        str(report_path),
    ]
    print(
        f"\n  $ python -m sglang.srt.debug_utils.comparator {' '.join(argv)}\n",
        flush=True,
    )
    return run(parse_args(argv))


def _read_report(report_path: Path) -> List[AnyRecord]:
    return [
        parse_record_json(line) for line in report_path.read_text().strip().splitlines()
    ]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-v"]))
