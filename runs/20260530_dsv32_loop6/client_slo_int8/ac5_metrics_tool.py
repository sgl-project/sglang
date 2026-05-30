#!/usr/bin/env python3
"""AC-5 metrics tool: build an exact per-request metrics artifact from the
benchmark JSONLs, and verify (recompute) the reported percentiles from the
committed artifact alone (no JSONL needed).

The raw benchmark JSONLs (development/results/double_sparsity_gsp_isl4096_osl512_c{16,32,64}_t1.jsonl)
are gitignored (*.jsonl). This tool extracts the exact per-request arrays the
client-SLO report depends on into a tracked JSON so the numbers are
independently recomputable from committed files.

Usage:
  python3 ac5_metrics_tool.py --build    # reads the JSONLs (must be present), writes ac5_metrics_arrays.json
  python3 ac5_metrics_tool.py --verify   # reads ac5_metrics_arrays.json only, recomputes + prints percentiles

Percentile method: numpy.percentile(arr, q) (linear interpolation). This
reproduces the sglang bench_serving stored summary fields exactly.
Per-request TPOT (ms): 1000 * sum(itls[i]) / (output_lens[i] - 1)  -- sglang's
formula; reproduces stored median_tpot_ms exactly. ITL percentiles are over the
flattened per-token list; the full per-token array is recomputable from the
checksummed source JSONL (--build), and its summary is stored here.
"""
import json, hashlib, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARRAYS = os.path.join(HERE, "ac5_metrics_arrays.json")
CONCS = [16, 32, 64]
# source JSONLs live in the repo's results dir relative to this file
SRC = lambda c: os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "development", "results",
    f"double_sparsity_gsp_isl4096_osl512_c{c}_t1.jsonl"))


def pct(arr, q):
    import numpy as np
    return float(np.percentile(arr, q))


def build():
    out = {
        "_description": "Exact per-request AC-5 metrics extracted from the gitignored benchmark JSONLs so the client-SLO percentiles recompute from committed files.",
        "percentile_method": "numpy.percentile(arr, q) (linear interpolation); reproduces sglang bench_serving stored summary fields exactly.",
        "tpot_formula_ms": "1000 * sum(itls[i]) / (output_lens[i] - 1)  (sglang formula; reproduces stored median_tpot_ms exactly)",
        "ttft_units": "seconds", "tpot_units": "ms", "itl_units": "ms",
        "build_command": "python3 ac5_metrics_tool.py --build",
        "verify_command": "python3 ac5_metrics_tool.py --verify  (recomputes from this file alone, no JSONL)",
        "source_jsonls": {}, "conc": {},
    }
    for c in CONCS:
        p = SRC(c)
        d = json.load(open(p))
        sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
        out["source_jsonls"][f"c{c}"] = {"path": os.path.relpath(p, HERE), "sha256": sha, "bytes": os.path.getsize(p)}
        ttfts_s = [float(x) for x in d["ttfts"]]
        outs = [int(x) for x in d["output_lens"]]
        tpots_ms = [1000.0 * sum(d["itls"][i]) / (outs[i] - 1) for i in range(len(outs))]
        flat_itl_ms = [v * 1000.0 for x in d["itls"] for v in x]
        out["conc"][str(c)] = {
            "completed": d["completed"], "achieved_concurrency": d.get("max_concurrency"),
            "duration_s": d["duration"], "seed": None,
            "errors_nonempty_count": sum(1 for e in d["errors"] if e),
            "errors_all_empty": all(not e for e in d["errors"]),
            "ttfts_s": ttfts_s, "tpots_ms": tpots_ms,
            "input_lens": [int(x) for x in d["input_lens"]], "output_lens": outs,
            "itl_all_tokens": {
                "count": len(flat_itl_ms),
                "recompute": "flatten itls from the checksummed source JSONL; summary below matches np.percentile of that flat array",
                "median_ms": pct(flat_itl_ms, 50), "p95_ms": pct(flat_itl_ms, 95), "p99_ms": pct(flat_itl_ms, 99),
            },
            "stored_summary_ms": {k: d[k] for k in ("median_ttft_ms", "p99_ttft_ms", "median_tpot_ms", "p99_tpot_ms", "median_itl_ms", "p95_itl_ms", "p99_itl_ms")},
        }
    json.dump(out, open(ARRAYS, "w"), indent=1)
    print("wrote", ARRAYS)
    verify()


def verify():
    d = json.load(open(ARRAYS))
    print("# AC-5 metrics recomputed from ac5_metrics_arrays.json (committed file; no JSONL needed)")
    print(f"# percentile_method: {d['percentile_method']}")
    ok = True
    for c in ("16", "32", "64"):
        cc = d["conc"][c]
        tt = [x * 1000 for x in cc["ttfts_s"]]
        tp = cc["tpots_ms"]
        st = cc["stored_summary_ms"]
        ttft_p50, ttft_p99 = pct(tt, 50), pct(tt, 99)
        tpot_p50, tpot_p99 = pct(tp, 50), pct(tp, 99)
        tps = 1000.0 / tpot_p50
        isl = cc["input_lens"]
        m = lambda a, q: pct(a, q)
        print(f"\n== conc {c} (completed={cc['completed']}, errors_all_empty={cc['errors_all_empty']}) ==")
        print(f"  TTFT s: p50={ttft_p50/1000:.3f} p90={m(tt,90)/1000:.3f} p99={ttft_p99/1000:.3f} min={min(tt)/1000:.3f} max={max(tt)/1000:.3f}  [SLO <22.0]")
        print(f"  TPOT ms: p50={tpot_p50:.3f} p99={tpot_p99:.3f}  => per-req TPS=1000/medTPOT={tps:.2f}  [SLO >=30]")
        print(f"  ISL: p50={m(isl,50):.0f} p99={m(isl,99):.0f} (nominal 4096)  OSL: all={cc['output_lens'][0]}")
        # validation against stored
        checks = [("ttft_p50", ttft_p50, st["median_ttft_ms"]), ("ttft_p99", ttft_p99, st["p99_ttft_ms"]),
                  ("tpot_p50", tpot_p50, st["median_tpot_ms"]), ("tpot_p99", tpot_p99, st["p99_tpot_ms"])]
        for name, got, want in checks:
            if abs(got - want) > 0.01:
                ok = False
                print(f"  MISMATCH {name}: recomputed {got:.4f} != stored {want:.4f}")
        print(f"  recomputed == stored summary (ttft/tpot p50,p99): {'PASS' if all(abs(g-w)<=0.01 for _,g,w in checks) else 'FAIL'}")
    print("\nALL recomputed==stored:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    if "--build" in sys.argv:
        build()
    elif "--verify" in sys.argv:
        verify()
    else:
        print(__doc__)
        sys.exit(2)
