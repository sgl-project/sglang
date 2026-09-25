#!/usr/bin/env python3
"""只渲染Markdown快照；不使用输入URL，不创建Pages。"""

import argparse
import html
import json
from pathlib import Path

from trend_io import contract

REPOSITORY = "https://github.com/flytiger-eco/sglang-for-sail"


def escape(value):
    text = "n/a" if value is None else str(value)
    text = html.escape(text, quote=True)
    for char in "\\`*_[]|":
        text = text.replace(char, "&#" + str(ord(char)) + ";")
    return text.replace("\r", " ").replace("\n", " ")


def _number(value):
    return "n/a" if value is None else format(value, ".6g")


def _run(observation):
    run = (observation or {}).get("run_id")
    return (
        f"[{run}]({REPOSITORY}/actions/runs/{run})"
        if contract.positive_id(run)
        else "unknown"
    )


def render(summary):
    lines = [
        "# PPU nightly 趋势快照",
        "",
        f"UTC 日期：{escape(summary['as_of_date'])}；窗口：{escape(summary['window_start'])} 至 {escape(summary['window_end'])}（右开）。",
        "",
        "| 类别 / 配置 / 测量 / 完整摘要 | 指标 | 最新观测 / 原因 | 最近有效值 / UTC时间 | p50 / p90 / MAD / n | 偏离 | 观测覆盖 | 质量失败 | 最后观测 / run |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for series in summary["series"]:
        key = series["key"]
        identity = " / ".join(
            escape(key[field])
            for field in (
                "schema_version",
                "test_id",
                "measurement_id",
                "config_digest",
            )
        )
        latest = series["latest_observation"] or {}
        state = (
            escape(latest.get("status", "no_observation"))
            + " / "
            + escape(latest.get("reason_code"))
        )
        metrics = series["metrics"] or {
            "unknown": {
                "stats": {"n": 0, "status": "insufficient_samples"},
                "latest_measured": None,
                "deviation": None,
                "observed_days": 0,
            }
        }
        for name, metric in sorted(metrics.items()):
            stats = metric["stats"]
            measured = metric["latest_measured"] or {}
            description = " / ".join(
                _number(stats.get(x)) for x in ("p50", "p90", "mad", "n")
            )
            if stats["status"] == "insufficient_samples":
                description += " (insufficient_samples)"
            cells = [
                identity,
                escape(name),
                state,
                _number(measured.get("value"))
                + " / "
                + escape(measured.get("generated_at")),
                description,
                _number(metric["deviation"]),
                f"{metric['observed_days']}/14",
                str(series["quality_failed"]),
                escape(latest.get("generated_at")) + " / " + _run(latest),
            ]
            lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        *["- " + escape(item) for item in summary["limitations"]],
        "",
        "data tree：" + escape(summary["input_data_tree"]),
        "工具 SHA：" + escape(summary["derivation_revision"]),
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        render(json.loads(args.input.read_text(encoding="utf-8"))), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
