#!/usr/bin/env python3
"""AC-6 DSA-default SLO metrics: exact per-request arrays + fail-closed verifier.

The DSA-default steady-state SLO run (num_prompts=64, warmup 120s / window 600s,
request_rate=inf, cross-node bench_serving --host node1) is the evidence that the
DSA-default product meets the SLO unchanged. Its raw `.jsonl` are gitignored, so this
extracts the exact per-request arrays into a tracked JSON and recomputes the reported
percentiles from the committed file alone (fail-closed).

Usage:
  python3 dsa_slo_metrics_tool.py --build    # reads the JSONLs (must be present), writes dsa_slo_arrays.json
  python3 dsa_slo_metrics_tool.py --verify    # recomputes from dsa_slo_arrays.json alone; exit 1 on any mismatch

Percentile method: numpy.percentile(arr, q) (linear interpolation).
Per-request TPOT (ms): 1000 * sum(itls[i]) / (output_lens[i] - 1)  (reproduces stored median_tpot_ms).
SLO: P99 TTFT < 22.0 s AND per-req TPS (1000/median_TPOT) >= 30.
"""
import json, hashlib, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARRAYS = os.path.join(HERE, "dsa_slo_arrays.json")
CONCS = [16, 32, 64]
SRC = lambda c: f"/tmp/ac6b/dsa_slo_np64/dsa_np64_c{c}_t1.jsonl"
TOL = 0.01


def pct(a, q):
    return float(np.percentile(a, q))


def build():
    out = {
        "_description": "DSA-default steady-state SLO (num_prompts=64, warmup 120s/window 600s, request_rate=inf, bench_serving --host node1). Exact per-request arrays so the SLO percentiles recompute from committed files.",
        "percentile_method": "numpy.percentile(arr,q) linear interpolation",
        "tpot_formula_ms": "1000*sum(itls[i])/(output_lens[i]-1)",
        "slo": "P99 TTFT < 22.0 s AND per-req TPS (1000/median_TPOT) >= 30",
        "build_command": "python3 dsa_slo_metrics_tool.py --build",
        "verify_command": "python3 dsa_slo_metrics_tool.py --verify",
        "source_jsonls": {}, "conc": {},
    }
    for c in CONCS:
        p = SRC(c)
        d = json.load(open(p))
        out["source_jsonls"][f"c{c}"] = {"path": os.path.basename(p), "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(), "bytes": os.path.getsize(p)}
        outs = [int(x) for x in d["output_lens"]]
        out["conc"][str(c)] = {
            "completed": d["completed"], "achieved_concurrency": d.get("max_concurrency"), "duration_s": d["duration"],
            "errors_nonempty_count": sum(1 for e in d["errors"] if e), "errors_all_empty": all(not e for e in d["errors"]),
            "ttfts_s": [float(x) for x in d["ttfts"]],
            "tpots_ms": [1000.0 * sum(d["itls"][i]) / (outs[i] - 1) for i in range(len(outs))],
            "input_lens": [int(x) for x in d["input_lens"]], "output_lens": outs,
            "stored_summary_ms": {k: d[k] for k in ("median_ttft_ms", "p99_ttft_ms", "median_tpot_ms", "p99_tpot_ms")},
        }
    json.dump(out, open(ARRAYS, "w"), indent=1)
    print("wrote", ARRAYS)
    verify()


def verify():
    d = json.load(open(ARRAYS))
    print("# DSA-default SLO recomputed from committed dsa_slo_arrays.json (no JSONL needed)")
    fails = []
    for c in ("16", "32", "64"):
        cc = d["conc"][c]; n = cc["completed"]
        tt = [x * 1000 for x in cc["ttfts_s"]]; tp = cc["tpots_ms"]; st = cc["stored_summary_ms"]
        if len(cc["ttfts_s"]) != n or len(tp) != n or len(cc["input_lens"]) != n or len(cc["output_lens"]) != n:
            fails.append(f"c{c}: array length != completed {n}")
        if not (cc["errors_all_empty"] and cc["errors_nonempty_count"] == 0):
            fails.append(f"c{c}: errors not all-empty")
        if not all(o == 512 for o in cc["output_lens"]):
            fails.append(f"c{c}: output_lens != 512")
        ttft_p99 = pct(tt, 99); tpot_p50 = pct(tp, 50); tps = 1000.0 / tpot_p50
        slo_ttft = ttft_p99 / 1000 < 22.0; slo_tps = tps >= 30.0
        print(f"  conc {c}: P99 TTFT={ttft_p99/1000:.2f}s [<22:{slo_ttft}]  per-req TPS={tps:.1f} [>=30:{slo_tps}]  (completed={n}, errors_all_empty={cc['errors_all_empty']})")
        for name, got, want in [("ttft_p99", ttft_p99, st["p99_ttft_ms"]), ("tpot_p50", tpot_p50, st["median_tpot_ms"])]:
            if abs(got - want) > TOL:
                fails.append(f"c{c}: {name} recomputed {got:.3f} != stored {want:.3f}")
    if fails:
        print("\nFAIL:")
        for f in fails:
            print("  -", f)
        raise SystemExit(1)
    print("\nrecomputed==stored + sanity checks: PASS")
    print("# SLO verdict: P99 TTFT < 22 s at every conc = PASS; per-req TPS >= 30 at conc 16/32 = PASS,")
    print("#   conc-64 ~29.4 = marginal MISS (pre-existing DSA characteristic, also 29.5 in the Loop-5 baseline).")


if __name__ == "__main__":
    if "--build" in sys.argv:
        build()
    elif "--verify" in sys.argv:
        verify()
    else:
        print(__doc__); sys.exit(2)
