#!/usr/bin/env python3
"""Read-only attribution of an existing run and optional Engine B log."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from bench_statistics import bootstrap_mean_ci, quantile, stable_seed


def analyze(directory, engine_log=None):
    root = Path(directory)
    raw = [json.loads(x) for x in (root/"raw.jsonl").read_text().splitlines()]
    trials = [x for x in raw if x["kind"] == "trial"]
    manifest = json.loads((root/"manifest.json").read_text())
    log_bytes = Path(engine_log).read_bytes() if engine_log else b""
    log_lines = log_bytes.decode(errors="replace").splitlines()
    pairs = {}
    for trial in trials:
        pairs.setdefault(trial["pair"], {})[trial["group"]] = trial
    evidence = {"source": str(root.resolve()), "source_raw_sha256": hashlib.sha256((root/"raw.jsonl").read_bytes()).hexdigest(),
                "attempts": len(trials), "invalid": [t["id"] for t in trials if not t.get("valid")], "lengths": {}, "trials": [],
                "engine_log": {"file": str(Path(engine_log).resolve()) if engine_log else None,
                               "sha256": hashlib.sha256(log_bytes).hexdigest(), "lines": len(log_lines)}}
    for length in sorted({x["prefix_tokens"] for x in trials}):
        rows = [x for x in trials if x["prefix_tokens"] == length]
        base = [x for x in rows if x["group"] == "on_demand"]
        pre = [x for x in rows if x["group"] == "prefetch"]
        ready = [x for x in pre if x.get("prefetch_ready_by_resume")]
        late = [x for x in pre if not x.get("prefetch_ready_by_resume")]
        deltas = [x["prefetch"]["resume"]["ttft_ms"]-x["on_demand"]["resume"]["ttft_ms"]
                  for x in pairs.values() if "prefetch" in x and "on_demand" in x and x["prefetch"]["prefix_tokens"] == length]
        cycle = [x["prefetch"]["complete_cycle_ms"]-x["on_demand"]["complete_cycle_ms"]
                 for x in pairs.values() if "prefetch" in x and "on_demand" in x and x["prefetch"]["prefix_tokens"] == length]
        elapsed = [x["background"]["prefetch"]["elapsed_ms"] for x in pre]
        evidence["lengths"][str(length)] = {"baseline_n": len(base), "prefetch_n": len(pre), "ready_before_resume_n": len(ready),
            "baseline_mean_ttft_ms": statistics.mean(x["resume"]["ttft_ms"] for x in base),
            "prefetch_mean_ttft_ms": statistics.mean(x["resume"]["ttft_ms"] for x in pre),
            "ready_subset_mean_ttft_ms": statistics.mean(x["resume"]["ttft_ms"] for x in ready) if ready else None,
            "late_subset_mean_ttft_ms": statistics.mean(x["resume"]["ttft_ms"] for x in late) if late else None,
            "prefetch_completion_ms": {"min": min(elapsed), "max": max(elapsed), "mean": statistics.mean(elapsed), "p95": quantile(elapsed, .95)},
            "paired_ttft_delta": bootstrap_mean_ci(deltas, seed=stable_seed(str(length))),
            "paired_cycle_delta": bootstrap_mean_ci(cycle, seed=stable_seed(str(length)+"cycle")),
            "logical_full_kv_mib": length*147456/2**20,
            "duplicate_on_demand_registration_n": 0, "repeated_prefill_tokens": sum(x.get("repeated_prefill_tokens", 0) for x in rows)}
        for trial in pre:
            controls = [x for x in raw if x.get("trial") == trial["id"] and x["kind"] == "http" and x.get("path") == "/v1/kv/control"]
            submit = next(x for x in controls if x.get("request", {}).get("kind") == "submit" and x["request"]["envelope"]["actions"][0]["action_type"] == "sglang.prefetch")
            queries = [x for x in controls if x.get("request", {}).get("kind") == "query" and x["request"]["message_id"] == submit["request"]["envelope"]["message_id"]]
            rid = trial["resume"]["meta_info"]["id"]
            ordinary = [(i+1, line) for i, line in enumerate(log_lines) if f"req={rid} " in line and "HiCache prefetch success" in line]
            if ordinary:
                evidence["lengths"][str(length)]["duplicate_on_demand_registration_n"] += 1
            native = []
            if ordinary:
                index = ordinary[0][0]-2
                while index >= 0 and "/flush_cache" not in log_lines[index]:
                    if "HiCache prefetch success req=native-kv:" in log_lines[index] and f"completed={length} " in log_lines[index]:
                        native = [(index+1, log_lines[index])]
                        break
                    index -= 1
            trial_ev = {"id": trial["id"], "prefix_tokens": length, "ready_before_resume": trial["prefetch_ready_by_resume"],
                "ttft_ms": trial["resume"]["ttft_ms"], "prefetch_elapsed_ms": trial["background"]["prefetch"]["elapsed_ms"],
                "acceptance_lead_ms": trial.get("control_acceptance_lead_ms"),
                "query_n": len(queries), "query_max_rpc_ms": max([x["elapsed_ms"] for x in queries] or [0]),
                "submit_rpc_ms": submit["elapsed_ms"], "ordinary_prefetch_log": ordinary, "preceding_native_log": native,
                "resume_cache": trial["resume_cache"], "attribution": "confirmed_ready_host_reuse" if trial["prefetch_ready_by_resume"] else "ambiguous_concurrent_control_and_on_demand",
                "unused_prepared_tokens": 0 if trial["prefetch_ready_by_resume"] else None,
                "physical_duplicate_io_bytes": None}
            evidence["trials"].append(trial_ev)
    return evidence, manifest


def write_report(directory, output, engine_log):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    snapshot = None
    if engine_log:
        snapshot = out/"p2-engine-b-evidence.log"
        if Path(engine_log).resolve() != snapshot.resolve():
            snapshot.write_bytes(Path(engine_log).read_bytes())
    evidence, manifest = analyze(directory, snapshot)
    args = manifest.get("args", {})
    if (evidence["attempts"] != 36 or args.get("tool_waits") != [1.0]
            or args.get("lead_times") != [.5] or args.get("output_tokens") != 8
            or args.get("repetitions") != 6):
        raise ValueError("This narrative is specific to the frozen P2 matrix (36 trials, 1s wait, .5s lead, 8 outputs, six repetitions)")
    if {k: v["duplicate_on_demand_registration_n"] for k, v in evidence["lengths"].items()} != {"2048": 0, "8192": 4, "16384": 6}:
        raise ValueError("Required original Engine B log evidence is missing; use the saved p2-engine-b-evidence.log snapshot")
    evidence["engine_log"]["original_source"] = str(Path(engine_log).resolve()) if engine_log else None
    (out/"p2-matrix-analysis.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False)+"\n")
    fmt = lambda x: "—" if x is None else f"{x:.1f}"
    lines = ["# P2 双 Engine matrix 归因", "", "0.5 秒提前量在 2K 前缀上覆盖了控制准备耗时；8K 只有 2/6 次在恢复前就绪，16K 为 0/6。16K 全部恢复都进入第二次普通按需流程，平均 TTFT 增加约 1.03 秒。固定提前量无法覆盖本轮长前缀的准备耗时。", "",
        f"输入：`{evidence['source']}`；raw SHA256 `{evidence['source_raw_sha256']}`。日志快照为 `p2-engine-b-evidence.log`，SHA256 `{evidence['engine_log']['sha256']}`。尝试 {evidence['attempts']} 条，路径/输出校验无效 {len(evidence['invalid'])} 条；所有试验保留。每种长度 6 个随机配对，工具等待 1 秒、提前量 0.5 秒、恢复输出 8 tokens。输入为合成 ledger，不能当作真实应用轨迹。", "",
        "| prefix | paired N | on-demand mean TTFT ms | prefetch mean ms | ready before resume | control completion mean / range ms | paired mean delta / 95% CI ms |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |"]
    for length, block in evidence["lengths"].items():
        ci = block["paired_ttft_delta"]
        completion = block["prefetch_completion_ms"]
        lines.append(f"| {length} | {ci['n_pairs']} | {fmt(block['baseline_mean_ttft_ms'])} | {fmt(block['prefetch_mean_ttft_ms'])} | {block['ready_before_resume_n']}/{block['prefetch_n']} | {fmt(completion['mean'])} / {fmt(completion['min'])}–{fmt(completion['max'])} | {fmt(ci['mean'])} [{fmt(ci['ci'][0])}, {fmt(ci['ci'][1])}] |")
    lines += ["", "差值为 PREFETCH − on-demand。区间对完整配对差值做 4,000 次 percentile bootstrap，seed 写入 JSON。N=6 很小，CI 仅描述本批配对变化；时序、缓存和共用 Mooncake 负载使样本独立性有限。ready 子集是运行后条件选择，不能替代全样本结论。", "",
        "| prefix | ready subset N / mean TTFT ms | late subset N / mean TTFT ms | second ordinary registration | repeated prefix prefill | logical FULL KV MiB |",
        "| --- | --- | --- | ---: | ---: | ---: |"]
    for length, block in evidence["lengths"].items():
        lines.append(f"| {length} | {block['ready_before_resume_n']} / {fmt(block['ready_subset_mean_ttft_ms'])} | {block['prefetch_n']-block['ready_before_resume_n']} / {fmt(block['late_subset_mean_ttft_ms'])} | {block['duplicate_on_demand_registration_n']} | {block['repeated_prefill_tokens']} | {fmt(block['logical_full_kv_mib'])} |")
    lines += ["", "逐条观测：", "", "| trial | ready | prefetch elapsed ms | resume TTFT ms | max query RPC ms | second ordinary log line |",
              "| --- | --- | ---: | ---: | ---: | --- |"]
    for trial in evidence["trials"]:
        links = ", ".join(str(x[0]) for x in trial["ordinary_prefetch_log"]) or "—"
        lines.append(f"| {trial['id']} | {trial['ready_before_resume']} | {fmt(trial['prefetch_elapsed_ms'])} | {fmt(trial['ttft_ms'])} | {fmt(trial['query_max_rpc_ms'])} | {links} |")
    lines += ["", "Engine B 日志中，来不及的试验先出现 `native-kv:* completed=N matched=0 loaded=N`，随后正式请求 ID 出现 `completed=N matched=N loaded=0`。8K 有 4 条，16K 有 6 条。第二条日志确认普通请求完成了额外的按需恢复/登记流程；它没有证明第二次网络 GET 的物理字节数。若没有 storage job ID、对象读取计数和字节遥测，重复物理 I/O 仍属待核验。日志行号与正式请求 ID 已写入 JSON。", "",
        "例如 16K r0，原始日志中的 native 读取在 12:12:40 完成，正式请求的额外流程到 12:12:42 才登记完成，恢复 TTFT 为 2931.8 ms。所有组只 prefill 新增的 21-token 工具后缀，分层反馈覆盖完整前缀，重复前缀 prefill tokens=0。这里不能把额外流程称为整段前缀重新计算。", "",
        "控制完成 elapsed 包含 submit、轮询、登记和 ack；query 每 20ms 轮询，单个 query RPC 的最长延迟为逐条表中数值。16K 的 1.12–1.86s 准备耗时远超 0.5s 提前量，单次 query 阻塞不足以解释全部耗时。按需流程与 native 控制同时在途时，host 反馈只能确认最终复用位置，无法从它独立归因传输获益。这 10 条来不及的准备应标为 ambiguous，不能计为 unused。", "",
        "完整周期的配对 mean delta / 95% bootstrap CI：", ""]
    for length, block in evidence["lengths"].items():
        ci = block["paired_cycle_delta"]
        lines.append(f"- {length} tokens：{fmt(ci['mean'])} ms [{fmt(ci['ci'][0])}, {fmt(ci['ci'][1])}]，N={ci['n_pairs']}。")
    lines += ["", "本轮 lead 只测试 0.5s，无法从单个提前量识别最佳值。用于后续选择的候选实验是 2K 的 0/0.25/0.5s、8K 的 0.25/0.5/1/1.5s、16K 的 0.5/1/1.5/2/3s，同时把工具等待扩大到能容纳对应 lead 的区间。每个 wait/lead 单元继续保留配对基线，不人为增加 storage 延迟。较大 lead 会提高等待期间的 L2 占用，应同时记录自然驱逐、host 压力和 unused。", "",
        "参考策略可使用真实完成结果的 client-observed 完成时间 EMA，加保守倍数作为所需提前量。未完成时遭遇正式按需恢复的样本另存 contended EMA，不污染空闲准备成本。KV 字节可以按模型配置估算：Qwen3-8B 为 2 × 36 layers × 8 KV heads × 128 dim × 2 bytes = 147456 bytes/token；表内是逻辑 FULL KV 的估计，物理共享/压缩/重试字节仍留空。", "",
        "本轮没有缺失、取消、路由变化或无后续请求的负例，manifest 的 negative 参数为空。该批数据无法给这些情形提供性能结论；harness 的负例仍保留，未从后续计划中删除。host 峰值、物理 I/O、后台吞吐没有遥测，不能以 token 数推算为已测结果。", ""]
    (out/"p2-matrix-analysis.md").write_text("\n".join(lines))
    return evidence


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory")
    p.add_argument("--engine-log", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    write_report(args.directory, args.output, args.engine_log)
