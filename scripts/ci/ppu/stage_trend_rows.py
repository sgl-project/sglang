#!/usr/bin/env python3
"""Stage this run's trend rows into the data branch's working tree.

Reads every ``trend.jsonl`` under ``INCOMING_DIR`` -- one per measuring job's
artifact -- and writes the rows under ``data/<test_id>/``, one file per
(test_id, run, attempt).

The line text is copied through unchanged rather than re-serialised. The kit
already emitted each row with sorted keys and no spare whitespace, so passing the
bytes along keeps what lands on the branch identical to what the run measured,
and leaves this script with no way to alter a number.

A row that does not parse, or whose ``test_id`` would not be a plain directory
name, stops the run before anything is filed: a series is only as good as the
worst line anyone ever appended to it, and a bad line is far cheaper to reject
here than to find months later.

Reads:
  INCOMING_DIR         directory the run's artifacts were downloaded into
  GITHUB_RUN_ID        \\
  GITHUB_RUN_ATTEMPT   /  what makes a filename unique within a series

Usage: python3 scripts/ci/ppu/stage_trend_rows.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from trend_io import contract, files, no_links, read_rows


def fail(message: str) -> None:
    print(f"::error::{message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    incoming = Path(os.environ.get("INCOMING_DIR", "incoming"))
    run_id = os.environ.get("GITHUB_RUN_ID", "unknown")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "1")

    stage(incoming, Path("data"), run_id=run_id, attempt=attempt)


def stage(incoming, data, *, run_id, attempt):
    contract.require(
        contract.positive_id(run_id) and contract.positive_id(attempt), "无效发布身份"
    )
    contract.require(not files(incoming, suffix="trend-error.json"), "报告趋势转换失败")
    sources = files(incoming, suffix="trend.jsonl")
    if not sources:
        print("::warning::no trend.jsonl in this run's artifacts, nothing to file")
        return

    staged = {}
    identities = {}
    for source in sources:
        for row, line in read_rows(source):
            prov = row["provenance"]
            original_run = prov.get("github_run_id") or run_id
            original_attempt = prov.get("github_run_attempt") or attempt
            contract.require(original_run == run_id, "不接受其他run的报告")
            contract.require(
                contract.positive_id(original_attempt)
                and int(original_attempt) <= int(attempt),
                "无效原始attempt",
            )
            identity = (*contract.series_key(row), original_run, original_attempt)
            if identity in identities:
                contract.require(identities[identity] == line, "同身份内容冲突")
                continue
            identities[identity] = line
            day = str(contract.utc(row["generated_at"]).date())
            target = (
                data / row["test_id"] / f"{day}-{original_run}-{original_attempt}.jsonl"
            )
            staged.setdefault(target, []).append((identity, line))

    # 先验证所有目标和冲突，再落盘，避免部分校验失败却留下可提交文件。
    prepared = {}
    for target, entries in sorted(staged.items()):
        no_links(target)
        content = "".join(line + "\n" for _, line in sorted(entries))
        if target.exists():
            # 历史发布器保留报告顺序；同一组原行顺序不同不应触发覆盖。
            existing = sorted(
                line for _, line in read_rows(target, allow_legacy_perf=True)
            )
            contract.require(
                existing == sorted(line for _, line in entries), "禁止覆盖原始行"
            )
        else:
            prepared[target] = content
    for target, content in prepared.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    print(f"归档新增文件 {len(prepared)}，收到报告 {len(sources)}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, TypeError, KeyError):
        fail("趋势输入校验或落盘失败；未发布数据")
