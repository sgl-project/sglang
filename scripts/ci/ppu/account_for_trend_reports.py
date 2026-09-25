#!/usr/bin/env python3
"""Count the reports that reached this job against the entries that ran.

Every step between a measurement and its row is best effort on purpose. The
upload an entry ends with tolerates a reset on the github egress because the
numbers have already left the job over its annotations and the on-NAS record
outlives the artifact, so a lost upload must not turn a passing measurement red.
The download this job begins with tolerates a missing artifact for the same
reason.

What none of them did was say so. Run 34548547824 filed seven reports for ten
entries with every step green, and nothing anywhere in it named the three that
were lost or the paths their copies still sat at. A gap that is tolerated but
never counted is indistinguishable from no gap at all.

This counts, and judges nothing. A report can be missing because an upload was
reset, which leaves a copy to back-fill from, or because the entry died before it
measured anything, which leaves nothing -- and from here those two look the same.
The entry that lost an upload it did produce says so itself, in its own warning,
with the path to read. Making the arithmetic visible is what gets that warning
looked for at all.

Reads:
  INCOMING_DIR   directory the run's artifacts were downloaded into
  NEEDS_JSON     toJSON(needs) -- one result per measuring job

Usage: python3 scripts/ci/ppu/account_for_trend_reports.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from trend_io import contract, files, read_rows

# A job that never started contributes no report and is no gap. Anything that
# reached a conclusion was in a position to measure, which is what "should have
# reported" can mean without knowing how far into the suite each one got.
CONCLUDED = frozenset({"success", "failure"})


def main() -> None:
    incoming = Path(os.environ.get("INCOMING_DIR", "incoming"))
    # The same thing stage_trend_rows.py reads, so that the two cannot disagree
    # about what a report is: one trend.jsonl per measuring job's artifact.
    arrived = files(incoming, suffix="trend.jsonl")
    errors = files(incoming, suffix="trend-error.json")
    measured = unmeasured = 0
    identities = set()
    reports = set()
    for path in arrived:
        rows = read_rows(path)
        report_identity = []
        for row, _ in rows:
            identity = (
                *contract.series_key(row),
                row["generated_at"],
                row["provenance"].get("github_run_id"),
                row["provenance"].get("github_run_attempt"),
            )
            report_identity.append(identity)
            if identity not in identities:
                measured += row["status"] == "measured"
                unmeasured += row["status"] != "measured"
                identities.add(identity)
        reports.add(tuple(sorted(report_identity)))

    raw = os.environ.get("NEEDS_JSON", "")
    try:
        needs = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        # Not fatal, and not silent: without the results there is nothing to
        # compare against, but what arrived is still worth printing.
        print("::warning::NEEDS_JSON did not parse; reporting arrivals only")
        needs = {}

    concluded = sorted(
        key for key, value in needs.items() if (value or {}).get("result") in CONCLUDED
    )

    print(
        f"concluded_jobs={len(concluded)} reports={len(reports)} "
        f"measured={measured} unmeasured={unmeasured} conversion_errors={len(errors)}"
    )
    print("观测对账，不代表调度覆盖；matrix展开数量可能多于job数量")

    if not needs:
        return

    missing = len(concluded) - len(reports)
    print(f"report_gap={max(missing, 0)}（相对已结束job数的下界）")
    if missing == 0:
        print("报告数与已结束job数相同，不能据此证明逐配置齐全")
        return

    # A warning rather than a failure, for the reason the uploads themselves are
    # best effort: this job measured nothing, and failing it would neither
    # recover a report nor keep the rows that did arrive from being worth filing.
    if missing > 0:
        print(
            f"::warning::{missing} of {len(concluded)} entries that ran "
            "contributed no report to the series; an entry whose upload was reset "
            "names its own NAS path in a warning of its own, and that copy is "
            "what a back-fill reads from"
        )
    else:
        print(
            f"::warning::{len(arrived)} reports arrived from {len(concluded)} "
            "entries that ran; the download pattern matched something beyond this "
            "run's own measurements"
        )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, TypeError):
        print("::error::趋势对账输入无效")
        sys.exit(1)
