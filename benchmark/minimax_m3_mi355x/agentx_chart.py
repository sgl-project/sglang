"""Throughput-vs-interactivity chart for the AgentX replay: SGLang (M3-perf) vs ATOM, one point per concurrency."""
import csv, json, re, sys, glob, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_PNG, OUT_CSV = sys.argv[1], sys.argv[2]

# ATOM published (recipes/MiniMax-M3-Agentic-InferenceX.md @ 47a81f9): tok/s/chip (total), intvty_p90, MI355X, forced acceptance
ATOM = {1: (4845, 256.5), 2: (5050, 257.6), 8: (14047, 198.8), 10: (16756, 188.6), 15: (22025, 163.1),
        20: (32870, 121.2), 24: (39680, 106.9), 32: (42476, 58.0)}

# SGLang result dirs per series and concurrency (SemiAnalysis client, ATOM flags, MI350X)
SGLANG = {
    "SGLang · real acceptance": {1: "aiperf_ladder_real_SAclient_c1", 8: "aiperf_ladder_real_SAclient_c8", 24: "aiperf_v8_full_SAclient_c24", 32: "aiperf_v8_full_SAclient_c32"},
    "SGLang · forced acceptance (ATOM parity)": {1: "aiperf_ladder_lossy_SAclient_c1", 8: "aiperf_ladder_lossy_SAclient_c8", 24: "aiperf_v13_lossy_SAclient_c24", 32: "aiperf_v13_lossy_SAclient_c32"},
}

def load_point(d):
    p = f"/scratch/results/{d}/profile_export.jsonl"
    try: recs = [json.loads(l) for l in open(p)]
    except FileNotFoundError: return None
    ok = [r for r in recs if r.get("error") is None and r["metadata"]["benchmark_phase"] == "profiling"]
    if not ok: return None
    def val(r, k):
        v = r["metrics"].get(k); return v.get("value") if isinstance(v, dict) else v
    try: j = json.load(open(f"/scratch/results/{d}/profile_export_aiperf.json"))
    except FileNotFoundError: return None
    total = j["total_token_throughput"]["avg"] / 4
    itl = [val(r, "inter_token_latency") for r in ok if val(r, "inter_token_latency")]
    itl_p90 = sorted(itl)[int(0.9 * len(itl)) - 1]
    itl_p50 = statistics.median(itl)
    isl = [val(r, "input_sequence_length") for r in ok]; osl = [val(r, "output_sequence_length") for r in ok]
    t0 = min(r["metadata"]["request_start_ns"] for r in ok); t1 = max(r["metadata"]["request_end_ns"] for r in ok)
    window = (sum(isl) + sum(osl)) / max((t1 - t0) / 1e9, 1) / 4
    return dict(total_per_gpu=total, window_per_gpu=window, intvty_p90=1000 / itl_p90, intvty_p50=1000 / itl_p50)

rows = []
fig, ax = plt.subplots(figsize=(12, 6.2), dpi=150)
colors = {"SGLang · real acceptance": "#1f77b4", "SGLang · forced acceptance (ATOM parity)": "#ff7f0e"}
for name, dirs in SGLANG.items():
    pts = []
    for c, d in sorted(dirs.items()):
        m = load_point(d)
        if m is None: continue
        pts.append((c, m["intvty_p90"], m["window_per_gpu"]))
        rows.append([name, c, round(m["total_per_gpu"]), round(m["window_per_gpu"]), round(m["intvty_p90"], 1), round(m["intvty_p50"], 1)])
    if pts:
        ax.plot([p[1] for p in pts], [p[2] for p in pts], "-o", color=colors[name], lw=2.2, ms=8, label=name)
        for c, x, y in pts: ax.annotate(f"c={c}", (x, y), textcoords="offset points", xytext=(6, 8), fontsize=9, color=colors[name])
apts = sorted(ATOM.items())
ax.plot([v[1] for _, v in apts], [v[0] for _, v in apts], "-o", color="#d62728", lw=2.2, ms=8, label="ATOM · published (MI355X, forced acceptance)")
for c, (y, x) in apts:
    if c in (1, 8, 24, 32): ax.annotate(f"c={c}", (x, y), textcoords="offset points", xytext=(6, -14), fontsize=9, color="#d62728")
    rows.append(["ATOM published", c, y, "", x, ""])
ax.set_xlabel("interactivity — tok/s per user (p90) →")
ax.set_ylabel("throughput — total tok/s per GPU ↑")
ax.set_title("MiniMax-M3 · AIPerf inferencex-agentx-mvp · TP4 · MXFP4 · EAGLE3", loc="left", fontsize=11)
ax.grid(alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(bottom=0)
ax.legend(loc="upper right", fontsize=9, frameon=False)
fig.text(0.01, 0.01, "SGLang: kevin-mii/sglang M3-perf, MI350X, SemiAnalysis AIPerf fork with ATOM's client flags, 3600 s (c=24/32) and 1800 s (c=1/8). "
         "SGLang points are the 3600 s window rate (AIPerf reported: 34,720 / 35,935 real, 38,196 / 40,598 forced). ATOM: recipe table (tok/s/chip, intvty_p90).", fontsize=7.5, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT_PNG)
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["series", "concurrency", "total_tok_s_per_gpu", "window_tok_s_per_gpu", "interactivity_p90", "interactivity_p50"]); w.writerows(rows)
print("\n".join(",".join(map(str, r)) for r in rows))
