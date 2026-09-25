#!/usr/bin/env python3
"""从不可变data树确定性派生14日快照，不读取当前时钟。"""

import argparse
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from statistics import median

from trend_io import archived_rows, contract

KEY_FIELDS = ("schema_version", "test_id", "measurement_id", "config_digest")
LIMITATIONS = [
    "当日为UTC快照，可能尚未结束。",
    "覆盖为有效测量UTC日期数/14，不是调度完成率；从未有报告的配置无法推断。",
    "缺报告不能区分未安排、执行失败与上传丢失；未启用回归阈值。",
    "缺少provenance时为unknown；Accuracy外部数据集未保证完整内容指纹。",
    "Answer仅代表既有L0/L1判定，不是全面语义正确率。",
]


def _order(record):
    return (
        record["time"],
        int(record["run"]),
        int(record["attempt"]),
        contract.series_key(record["row"]),
    )


def _deduplicate(records):
    identities, copies, result = {}, set(), []
    # attempt小者先处理，相同原报告的旧perf重发不能伪装为一次新的测量。
    for record in sorted(records, key=lambda r: (int(r["attempt"]), _order(r))):
        key = contract.series_key(record["row"])
        identity = (*key, record["run"], record["attempt"])
        old = identities.get(identity)
        contract.require(old is None or old == record["content"], "同测量身份内容冲突")
        if old is not None:
            continue
        identities[identity] = record["content"]
        copy_key = (*key, record["run"], record["content"])
        legacy = record["row"]["schema_version"] == contract.PERF and not record["row"][
            "provenance"
        ].get("github_run_attempt")
        if legacy and copy_key in copies:
            continue
        copies.add(copy_key)
        result.append(record)
    return sorted(result, key=_order)


def _observation(record):
    if record is None:
        return None
    row = record["row"]
    source = row["provenance"].get("source_revision")
    reason = row.get("reason_code")
    # 历史perf自由文本也不原样扩散到新视图。
    if reason is not None and reason not in contract.MISSING_REASONS:
        reason = "unmeasured"
    return {
        "generated_at": record["time"].isoformat(),
        "run_id": record["run"],
        "attempt": record["attempt"],
        "status": row["status"],
        "reason_code": reason,
        "quality_verdict": (
            row.get("quality_verdict")
            if row["schema_version"] != contract.PERF
            else None
        ),
        "source_revision": (
            source
            if isinstance(source, str) and contract.SHA1.fullmatch(source)
            else None
        ),
    }


def _stats(values):
    n = len(values)
    if n < 3:
        return {"status": "insufficient_samples", "n": n}
    values = sorted(values)
    center = median(values)
    position = (n - 1) * 0.9
    lower = int(position)
    p90 = values[lower] + (values[min(lower + 1, n - 1)] - values[lower]) * (
        position - lower
    )
    return {
        "status": "available",
        "n": n,
        "p50": center,
        "p90": p90,
        "mad": median(abs(value - center) for value in values),
    }


def _metric(records, name):
    measured = [
        r
        for r in records
        if r["row"]["status"] == "measured" and name in r["row"]["metrics"]
    ]
    stats = _stats([r["row"]["metrics"][name] for r in measured[-16:]])
    latest = measured[-1] if measured else None
    value = latest["row"]["metrics"][name] if latest else None
    deviation = None
    if stats.get("p50") and records and records[-1] is latest:
        deviation = (value - stats["p50"]) / stats["p50"]
    return {
        "latest_measured": {**_observation(latest), "value": value} if latest else None,
        "stats": stats,
        "deviation": deviation,
        "observed_days": len({r["time"].date() for r in measured}),
        "coverage_denominator": 14,
    }


def _series(key, history, *, start, end):
    # 先考虑回放上界之前的attempt，再按run选最新；最后应用窗口下界。
    latest_attempts = {}
    for record in history:
        run = record["run"]
        if run not in latest_attempts or int(record["attempt"]) > int(
            latest_attempts[run]["attempt"]
        ):
            latest_attempts[run] = record
    records = sorted(
        (r for r in latest_attempts.values() if start <= r["time"] < end), key=_order
    )
    names = sorted({name for r in history for name in (r["row"]["metrics"] or {})})
    if not names:
        names = (
            ["score"]
            if key[0] == contract.ACCURACY
            else (
                ["pass_rate" if key[2] == "suite" else "passed"]
                if key[0] == contract.ANSWER
                else []
            )
        )
    metric_names = {
        r["row"].get("metric_name")
        for r in history
        if key[0] == contract.ACCURACY and r["row"].get("metric_name") is not None
    }
    contract.require(len(metric_names) <= 1, "同序列metric口径冲突")
    return {
        "key": dict(zip(KEY_FIELDS, key)),
        "metric_name": next(iter(metric_names), None),
        "n_measured": sum(r["row"]["status"] == "measured" for r in records),
        "n_unmeasured": sum(r["row"]["status"] != "measured" for r in records),
        "quality_failed": sum(
            key[0] != contract.PERF and r["row"].get("quality_verdict") == "failed"
            for r in records
        ),
        "latest_observation": _observation(records[-1] if records else None),
        "metrics": {name: _metric(records, name) for name in names},
    }


def derive(data, *, as_of_date, input_data_tree, derivation_revision):
    anchor = date.fromisoformat(as_of_date)
    contract.require(anchor.isoformat() == as_of_date, "无效UTC日期")
    contract.require(
        contract.SHA1.fullmatch(derivation_revision) is not None,
        "工具版本必须是完整SHA",
    )
    contract.require(
        contract.SHA1.fullmatch(input_data_tree) is not None, "输入必须是data tree OID"
    )
    midnight = datetime.combine(anchor, time.min, tzinfo=timezone.utc)
    start, end = midnight - timedelta(days=13), midnight + timedelta(days=1)
    records = _deduplicate(archived_rows(Path(data)))
    histories = defaultdict(list)
    for record in records:
        if record["time"] < end:
            histories[contract.series_key(record["row"])].append(record)
    series, ledger = [], []
    for key, history in sorted(histories.items()):
        components = (
            {contract.canonical(r["row"].get("config_components")) for r in history}
            if key[0] == contract.ANSWER
            else {"null"}
        )
        contract.require(len(components) == 1, "同摘要组成项冲突")
        series.append(_series(key, history, start=start, end=end))
        ledger.append(
            {
                "key": dict(zip(KEY_FIELDS, key)),
                "config_components": (
                    history[0]["row"].get("config_components")
                    if key[0] == contract.ANSWER
                    else None
                ),
                "first_observation": _observation(history[0]),
                "last_observation": _observation(history[-1]),
            }
        )
    return {
        "schema_version": "ppu-trend-summary/v1",
        "as_of_date": as_of_date,
        "generated_at": midnight.isoformat(),
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "window_days": 14,
        "input_data_tree": input_data_tree,
        "derivation_revision": derivation_revision,
        "series": series,
        "digest_history": ledger,
        "limitations": LIMITATIONS,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--as-of-date", required=True)
    parser.add_argument("--input-data-tree", required=True)
    parser.add_argument("--derivation-revision", required=True)
    args = parser.parse_args()
    result = derive(
        args.data,
        as_of_date=args.as_of_date,
        input_data_tree=args.input_data_tree,
        derivation_revision=args.derivation_revision,
    )
    args.output.write_text(contract.canonical(result) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
