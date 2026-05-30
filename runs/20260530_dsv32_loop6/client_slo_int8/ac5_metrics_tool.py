#!/usr/bin/env python3
"""AC-5 metrics tool: build exact per-request metrics artifacts from the
benchmark JSONLs, and verify (recompute) the reported percentiles from the
committed artifacts alone (no JSONL needed).

The raw benchmark JSONLs (development/results/double_sparsity_gsp_isl4096_osl512_c{16,32,64}_t1.jsonl)
are gitignored (*.jsonl). This tool extracts the exact per-request arrays the
client-SLO report depends on into tracked JSON so the numbers are
independently recomputable from committed files.

Committed artifacts:
  - ac5_metrics_arrays.json  : per conc, exact per-request ttfts (s), tpots (ms),
                               input_lens, output_lens, errors-all-empty, source
                               JSONL SHA256, and the stored summary for cross-check.
  - ac5_itl_flat_ms.json     : per conc, the exact flattened per-token ITL (ms,
                               sorted ascending) so median/p95/p99 ITL recompute
                               from committed data.

Usage:
  python3 ac5_metrics_tool.py --build    # reads the JSONLs (must be present), writes both artifacts
  python3 ac5_metrics_tool.py --verify   # reads the committed artifacts only, recomputes + ASSERTS (exit 1 on any mismatch)

Percentile method: numpy.percentile(arr, q) (linear interpolation). This
reproduces the sglang bench_serving stored summary fields exactly.
Per-request TPOT (ms): 1000 * sum(itls[i]) / (output_lens[i] - 1)  -- sglang's
formula; reproduces stored median_tpot_ms exactly. ITL percentiles are over the
flattened per-token list (committed in ac5_itl_flat_ms.json).
"""
import json, hashlib, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARRAYS = os.path.join(HERE, "ac5_metrics_arrays.json")
ITL = os.path.join(HERE, "ac5_itl_flat_ms.json")
CONCS = [16, 32, 64]
SRC = lambda c: os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "development", "results",
    f"double_sparsity_gsp_isl4096_osl512_c{c}_t1.jsonl"))
TOL = 0.01  # ms / ms-equivalent tolerance for recomputed == stored


def pct(arr, q):
    return float(np.percentile(arr, q))


def build():
    out = {
        "_description": "Exact per-request AC-5 metrics extracted from the gitignored benchmark JSONLs so the client-SLO percentiles recompute from committed files.",
        "percentile_method": "numpy.percentile(arr, q) (linear interpolation); reproduces sglang bench_serving stored summary fields exactly.",
        "tpot_formula_ms": "1000 * sum(itls[i]) / (output_lens[i] - 1)  (sglang formula; reproduces stored median_tpot_ms exactly)",
        "itl_source": "ac5_itl_flat_ms.json (flattened per-token ITL in ms, sorted ascending; reproduces stored median/p95/p99 itl exactly)",
        "ttft_units": "seconds", "tpot_units": "ms", "itl_units": "ms",
        "build_command": "python3 ac5_metrics_tool.py --build",
        "verify_command": "python3 ac5_metrics_tool.py --verify  (recomputes TTFT/TPOT/TPS/ITL from committed files; exits 1 on any mismatch)",
        "source_jsonls": {}, "conc": {},
    }
    itl_out = {"_units": "ms", "_note": "flattened per-token inter-token latencies, sorted ascending; ITL percentiles = numpy.percentile of these.", "conc": {}}
    for c in CONCS:
        p = SRC(c)
        d = json.load(open(p))
        sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
        out["source_jsonls"][f"c{c}"] = {"path": os.path.relpath(p, HERE), "sha256": sha, "bytes": os.path.getsize(p)}
        outs = [int(x) for x in d["output_lens"]]
        ttfts_s = [float(x) for x in d["ttfts"]]
        tpots_ms = [1000.0 * sum(d["itls"][i]) / (outs[i] - 1) for i in range(len(outs))]
        flat_itl_ms = sorted(round(v * 1000.0, 4) for x in d["itls"] for v in x)
        itl_out["conc"][str(c)] = flat_itl_ms
        out["conc"][str(c)] = {
            "completed": d["completed"], "achieved_concurrency": d.get("max_concurrency"),
            "duration_s": d["duration"],
            "errors_nonempty_count": sum(1 for e in d["errors"] if e),
            "errors_all_empty": all(not e for e in d["errors"]),
            "ttfts_s": ttfts_s, "tpots_ms": tpots_ms,
            "input_lens": [int(x) for x in d["input_lens"]], "output_lens": outs,
            "itl_flat_count": len(flat_itl_ms),
            "stored_summary_ms": {k: d[k] for k in ("median_ttft_ms", "p99_ttft_ms", "median_tpot_ms", "p99_tpot_ms", "median_itl_ms", "p95_itl_ms", "p99_itl_ms")},
        }
    json.dump(out, open(ARRAYS, "w"), indent=1)
    json.dump(itl_out, open(ITL, "w"))
    print("wrote", ARRAYS, "and", ITL)


def verify():
    d = json.load(open(ARRAYS))
    itl = json.load(open(ITL))
    print("# AC-5 metrics recomputed from committed files (ac5_metrics_arrays.json + ac5_itl_flat_ms.json; no JSONL needed)")
    print(f"# percentile_method: {d['percentile_method']}")
    fails = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)
            print("  FAIL:", msg)

    for c in ("16", "32", "64"):
        cc = d["conc"][c]
        tt = [x * 1000 for x in cc["ttfts_s"]]
        tp = cc["tpots_ms"]
        st = cc["stored_summary_ms"]
        flat = itl["conc"][c]
        n = cc["completed"]
        print(f"\n== conc {c} (completed={n}, errors_all_empty={cc['errors_all_empty']}) ==")
        # sanity checks
        check(len(cc["ttfts_s"]) == n, f"c{c}: len(ttfts) {len(cc['ttfts_s'])} != completed {n}")
        check(len(tp) == n, f"c{c}: len(tpots) {len(tp)} != completed {n}")
        check(len(cc["input_lens"]) == n and len(cc["output_lens"]) == n, f"c{c}: input/output length array != completed {n}")
        check(cc["errors_all_empty"] is True and cc["errors_nonempty_count"] == 0, f"c{c}: errors not all-empty")
        check(all(o == 512 for o in cc["output_lens"]), f"c{c}: not all output_lens == 512")
        check(len(flat) == cc["itl_flat_count"], f"c{c}: itl flat count {len(flat)} != recorded {cc['itl_flat_count']}")
        # recomputed == stored
        ttft_p50, ttft_p99 = pct(tt, 50), pct(tt, 99)
        tpot_p50, tpot_p99 = pct(tp, 50), pct(tp, 99)
        itl_p50, itl_p95, itl_p99 = pct(flat, 50), pct(flat, 95), pct(flat, 99)
        tps = 1000.0 / tpot_p50
        print(f"  TTFT s: p50={ttft_p50/1000:.3f} p90={pct(tt,90)/1000:.3f} p99={ttft_p99/1000:.3f} min={min(tt)/1000:.3f} max={max(tt)/1000:.3f}  [SLO <22.0]")
        print(f"  TPOT ms: p50={tpot_p50:.3f} p99={tpot_p99:.3f}  => per-req TPS=1000/medTPOT={tps:.2f}  [SLO >=30]")
        print(f"  ITL ms: p50={itl_p50:.3f} p95={itl_p95:.3f} p99={itl_p99:.3f}  (over {len(flat)} tokens)")
        print(f"  ISL: p50={pct(cc['input_lens'],50):.0f} p99={pct(cc['input_lens'],99):.0f} (nominal 4096)")
        for name, got, want in [("ttft_p50", ttft_p50, st["median_ttft_ms"]), ("ttft_p99", ttft_p99, st["p99_ttft_ms"]),
                                ("tpot_p50", tpot_p50, st["median_tpot_ms"]), ("tpot_p99", tpot_p99, st["p99_tpot_ms"]),
                                ("itl_p50", itl_p50, st["median_itl_ms"]), ("itl_p95", itl_p95, st["p95_itl_ms"]), ("itl_p99", itl_p99, st["p99_itl_ms"])]:
            check(abs(got - want) <= TOL, f"c{c}: {name} recomputed {got:.4f} != stored {want:.4f}")
        print(f"  recomputed==stored (ttft/tpot/itl p50,p95,p99): {'PASS' if not fails or not any(c in f for f in fails) else 'see FAIL above'}")

    if fails:
        print(f"\nALL recomputed==stored: FAIL ({len(fails)} mismatch(es))")
        raise SystemExit(1)
    print("\nALL recomputed==stored + sanity checks: PASS")


if __name__ == "__main__":
    if "--build" in sys.argv:
        build()
        verify()
    elif "--verify" in sys.argv:
        verify()
    else:
        print(__doc__)
        sys.exit(2)
