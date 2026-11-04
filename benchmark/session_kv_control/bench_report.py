"""All-sample summaries, ready subsets and complete-pair confidence intervals."""
import itertools
import json
import math
from pathlib import Path
import statistics

from bench_statistics import bootstrap_mean_ci, stable_seed

NEGATIVES = {"unused", "missing", "cancel", "route_change"}


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values)-1)*p
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi]-values[lo])*(pos-lo)


def distribution(values):
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            **{f"p{round(p*100)}": percentile(values, p) for p in (.5, .95, .99)}}


def summarize(records):
    trials = [x for x in records if x.get("kind") == "trial"]
    summary = {"attempts": len(trials), "valid_attempts": sum(t.get("valid", False) for t in trials),
               "failures": [{"trial": t["id"], "group": t["group"], "reasons": t.get("invalid_reasons", [])}
                            for t in trials if not t.get("valid")], "groups": {}, "paired": {}, "incremental_paired": {}}
    key = lambda t: json.dumps([t["prefix_tokens"], t["tool_wait_s"], t["lead_s"], t["group"]])
    for label, grouped in itertools.groupby(sorted(trials, key=key), key=key):
        rows = list(grouped)
        resumed = [x for x in rows if x.get("resume")]
        valid = [x for x in resumed if x.get("valid")]
        ready = [x for x in resumed if x.get("prefetch_ready_before_resume", x.get("prefetch_ready_by_resume"))]
        late = [x for x in resumed if x.get("background", {}).get("prefetch") and x not in ready]
        summary["groups"][label] = {
            "attempts": len(rows), "valid": len(valid), "completed_resumes": len(resumed),
            "resumed_ttft_ms": distribution([x["resume"]["ttft_ms"] for x in resumed]),
            "resumed_latency_ms": distribution([x["resume"]["latency_ms"] for x in resumed]),
            "complete_cycle_ms": distribution([x["complete_cycle_ms"] for x in resumed if "complete_cycle_ms" in x]),
            "validated_resumed_ttft_ms": distribution([x["resume"]["ttft_ms"] for x in valid]),
            "prefetch_ready_before_resume": {"n": len(ready), "ttft_ms": distribution([x["resume"]["ttft_ms"] for x in ready]),
                                              "selection": "post-treatment descriptive subset; all-sample results remain primary"},
            "prefetch_not_ready_before_resume": {"n": len(late), "ttft_ms": distribution([x["resume"]["ttft_ms"] for x in late])},
            "source_computation_ms": distribution([x["source"]["latency_ms"] for x in valid]),
            "repeated_prefill_tokens": distribution([x["repeated_prefill_tokens"] for x in valid]),
            "tier_hit_tokens": {tier: sum(x["resume_cache"][tier] for x in valid) for tier in ("device", "cpu", "storage")},
            "prefetch_ready_by_resume": sum(x.get("prefetch_ready_before_resume", x.get("prefetch_ready_by_resume", False)) for x in rows),
            "unused_prepared_tokens": sum(x.get("unused_prepared_tokens") or 0 for x in rows),
            "unknown_prefetch_usage_trials": sum(x.get("prefetch_usage_ambiguous", False) for x in rows),
            "background_throughput_requests_per_s": distribution([x["background_load"]["completion_cohort_requests_per_s"]
                for x in rows if x.get("background_load")]),
            "background_failures": sum(x.get("background_load", {}).get("total_failures", 0) for x in rows),
            "background_in_window_failures": sum(x.get("background_load", {}).get("in_window_failures", 0) for x in rows),
            "background_drained_after_cutoff": sum(x.get("background_load", {}).get("drained_after_cutoff", 0) for x in rows),
        }
    pairs = {}
    for t in trials:
        pairs.setdefault(t["pair"], {})[t["group"]] = t
    deltas = {}
    for pair, members in pairs.items():
        base = members.get("on_demand")
        if not base:
            continue
        for group, candidate in members.items():
            if group == "on_demand" or group in NEGATIVES:
                continue
            label = json.dumps([candidate["prefix_tokens"], candidate["tool_wait_s"], candidate["lead_s"], group])
            block = deltas.setdefault(label, {"eligible_pairs": 0, "valid_pairs": 0, "completed_pairs": 0,
                                             "ttft_delta_ms": [], "cycle_delta_ms": [], "validated_ttft_delta_ms": []})
            block["eligible_pairs"] += 1
            if base.get("resume") and candidate.get("resume"):
                block["completed_pairs"] += 1
                block["ttft_delta_ms"].append(candidate["resume"]["ttft_ms"]-base["resume"]["ttft_ms"])
                if "complete_cycle_ms" in candidate and "complete_cycle_ms" in base:
                    block["cycle_delta_ms"].append(candidate["complete_cycle_ms"]-base["complete_cycle_ms"])
                if base.get("valid") and candidate.get("valid"):
                    block["valid_pairs"] += 1
                    block["validated_ttft_delta_ms"].append(candidate["resume"]["ttft_ms"]-base["resume"]["ttft_ms"])
    for label, block in deltas.items():
        summary["paired"][label] = {**block, "ttft_delta_ms": distribution(block["ttft_delta_ms"]),
                                    "cycle_delta_ms": distribution(block["cycle_delta_ms"]),
                                    "validated_ttft_delta_ms": distribution(block["validated_ttft_delta_ms"]),
                                    "paired_mean_ttft_bootstrap": bootstrap_mean_ci(block["ttft_delta_ms"], seed=stable_seed(label)),
                                    "paired_mean_cycle_bootstrap": bootstrap_mean_ci(block["cycle_delta_ms"], seed=stable_seed(label+"cycle"))}
    for base_group, candidate_group in (("demote", "pin_demote"), ("pin_demote", "pin_demote_prefetch")):
        blocks = {}
        for members in pairs.values():
            base, candidate = members.get(base_group), members.get(candidate_group)
            if not base or not candidate:
                continue
            label = json.dumps([candidate["prefix_tokens"], candidate["tool_wait_s"], candidate["lead_s"],
                                candidate_group+" − "+base_group])
            block = blocks.setdefault(label, {"eligible_pairs": 0, "completed_pairs": 0, "valid_pairs": 0,
                                               "ttft": [], "cycle": []})
            block["eligible_pairs"] += 1
            if base.get("resume") and candidate.get("resume"):
                block["completed_pairs"] += 1
                block["valid_pairs"] += int(bool(base.get("valid") and candidate.get("valid")))
                block["ttft"].append(candidate["resume"]["ttft_ms"]-base["resume"]["ttft_ms"])
                if "complete_cycle_ms" in base and "complete_cycle_ms" in candidate:
                    block["cycle"].append(candidate["complete_cycle_ms"]-base["complete_cycle_ms"])
        for label, block in blocks.items():
            summary["incremental_paired"][label] = {
                **{k: v for k, v in block.items() if k not in ("ttft", "cycle")},
                "ttft_delta_ms": distribution(block["ttft"]), "cycle_delta_ms": distribution(block["cycle"]),
                "paired_mean_ttft_bootstrap": bootstrap_mean_ci(block["ttft"], seed=stable_seed(label))}
    summary["physical_io_bytes"] = None
    summary["notes"] = ["negative deltas mean lower latency", "primary latency distributions include every completed resume, including failed-control trials",
                        "validated_* distributions are separate and must be read with attempts and failures; invalid trials cannot establish cache-path correctness",
                        "physical I/O deduplication and host peak require owner telemetry; not inferred from token counts",
                        "quantiles use linear interpolation; small samples do not establish tail performance"]
    return summary


def write_summary(path, records):
    summary = summarize(records)
    Path(path, "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n")
    lines = ["# Session KV 实验结果", "", f"尝试 {summary['attempts']} 条；有效 {summary['valid_attempts']} 条。",
             "", "TTFT 从恢复请求发出计时，到首个实际输出 token；完整周期包含源计算、发布、工具等待和恢复请求。",
             "", "表内延迟包含全部完成的恢复请求，包括控制失败的尝试；valid 列只表示路径与输出校验通过数量。无效路径不可用于证明缓存收益。",
             "", "| tokens / wait / lead / group | attempts | completed | valid | TTFT p50 ms | p95 ms | cycle p50 ms |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    fmt = lambda x: "—" if x is None else f"{x:.3f}"
    for label, group in summary["groups"].items():
        lines.append(f"| {label} | {group['attempts']} | {group['completed_resumes']} | {group['valid']} | {fmt(group['resumed_ttft_ms']['p50'])} | {fmt(group['resumed_ttft_ms']['p95'])} | {fmt(group['complete_cycle_ms']['p50'])} |")
    lines += ["", "| paired group | eligible pairs | completed pairs | valid pairs | candidate − on-demand TTFT p50 ms | cycle p50 ms |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for label, pair in summary["paired"].items():
        lines.append(f"| {label} | {pair['eligible_pairs']} | {pair['completed_pairs']} | {pair['valid_pairs']} | {fmt(pair['ttft_delta_ms']['p50'])} | {fmt(pair['cycle_delta_ms']['p50'])} |")
    if summary["paired"]:
        lines += ["", "| paired group | N | mean TTFT delta ms | paired bootstrap 95% CI ms |",
                  "| --- | ---: | ---: | --- |"]
        for label, pair in summary["paired"].items():
            ci = pair["paired_mean_ttft_bootstrap"]
            interval = "—" if ci["ci"] is None else f"[{fmt(ci['ci'][0])}, {fmt(ci['ci'][1])}]"
            lines.append(f"| {label} | {ci['n_pairs']} | {fmt(ci['mean'])} | {interval} |")
        lines += ["", "bootstrap 对完整配对差值采样；小样本、相关时序与状态变化会限制区间解释。ready 子集只描述已完成准备的样本，全部样本仍保留。"]
    lines += ["", "| group | ready-before-resume N / mean TTFT ms | not-ready N / mean TTFT ms | ambiguous usage N |",
              "| --- | --- | --- | ---: |"]
    for label, group in summary["groups"].items():
        ready, late = group["prefetch_ready_before_resume"], group["prefetch_not_ready_before_resume"]
        lines.append(f"| {label} | {ready['n']} / {fmt(ready['ttft_ms']['mean'])} | {late['n']} / {fmt(late['ttft_ms']['mean'])} | {group['unknown_prefetch_usage_trials']} |")
    if summary["incremental_paired"]:
        lines += ["", "| incremental pair | eligible | completed | valid | TTFT delta p50 ms | cycle delta p50 ms |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for label, pair in summary["incremental_paired"].items():
            lines.append(f"| {label} | {pair['eligible_pairs']} | {pair['completed_pairs']} | {pair['valid_pairs']} | {fmt(pair['ttft_delta_ms']['p50'])} | {fmt(pair['cycle_delta_ms']['p50'])} |")
    if summary["failures"]:
        lines += ["", "无效记录：", ""]
        lines += [f"- {x['trial']} ({x['group']}): {'; '.join(x['reasons'])}" for x in summary["failures"]]
    lines += ["", "原始请求、响应、控制状态、范围与错误均在 raw.jsonl。未观测到的物理 I/O、host 峰值和服务端 CPU 保留为空值。", ""]
    Path(path, "summary.md").write_text("\n".join(lines))
    return summary
