#!/usr/bin/env python3
"""AC-5 full-context client-SLO metrics + fail-closed verifier (Round 20).

Builds `ac5_fullctx_arrays.json` from the AC-5 full-context bench JSONLs (DS int8 /
mem_fraction_static=0.7 / radix-on / context_len=163840 / TP=8, conc 16/32/64, steady-state
warmup 120 / window 300) as an EXACT source, then recomputes the published numbers from the
committed JSON alone — failing closed on any mismatch or degenerate (empty-latency) data.

Per-request metrics (the consumer is the SLO check, not benchmark_compare):
  - achieved concurrency = JSONL ``concurrency`` (effective).
  - per-request gen TPS p50 = percentile50( output_lens[i] / sum(itls[i]) ).
  - P99 TTFT (s) = ``p99_ttft_ms`` / 1000   (cross-checked against percentile99(ttfts)).
Fail-closed sanity (catches the R18 empty-stream class the R19 bench fix now refuses):
  64-hex sha per source, completed>0, output_len all 512, errors empty, len(ttfts)==len(itls)==
  len(output_lens)==completed, and EVERY ttft>0 + EVERY request has >=1 ITL (no empty-latency rows).

Usage:
  python3 ac5_fullctx_metrics_tool.py --build   # reads /tmp/ac5r20/results/*.jsonl, writes arrays json
  python3 ac5_fullctx_metrics_tool.py --verify  # recompute from committed json + fail-closed
"""
import glob, hashlib, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARRAYS = os.path.join(HERE, "ac5_fullctx_arrays.json")
SRC_GLOB = "/tmp/ac5r20/results/double_sparsity_gsp_isl4096_osl512_c{c}_t*.jsonl"
CONCS = [16, 32, 64]
TOL = 0.01
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def _pct(vals, p):
    s = sorted(vals)
    if not s:
        return None
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k); hi = min(lo + 1, len(s) - 1); f = k - lo
    return float(s[lo] * (1 - f) + s[hi] * f)


def _per_req_tps(d):
    ol, itls = d["output_lens"], d["itls"]
    out = []
    for i in range(min(len(ol), len(itls))):
        s = sum(v for v in itls[i] if isinstance(v, (int, float)))
        if ol[i] and s > 0:
            out.append(float(ol[i]) / s)
    return out


def build():
    out = {"_purpose": "AC-5 full-context (DS int8/mem0.7/radix-on/context_len=163840/TP8) measured "
                       "client-SLO arrays; raw *.jsonl gitignored (sha256 below). Recomputes P99 TTFT + "
                       "per-req TPS p50 + achieved conc; fail-closed on empty-latency.",
           "operating_point": {}, "conc": {}}
    for c in CONCS:
        files = sorted(glob.glob(SRC_GLOB.format(c=c)))
        trials = []
        for f in files:
            d = json.load(open(f))
            trials.append({
                "trial": os.path.basename(f).split("_t")[-1].split(".")[0],
                "concurrency": d["concurrency"],
                "max_concurrency": d.get("max_concurrency"),
                "completed": d["completed"],
                "p99_ttft_ms": d["p99_ttft_ms"],
                "ttfts_s": [float(x) for x in d["ttfts"]],
                "per_req_gen_tps": _per_req_tps(d),
                "errors_nonempty": sum(1 for e in d.get("errors", []) if e),
                "output_len_all_512": all(o == 512 for o in d["output_lens"]),
                "min_ttft_s": min(d["ttfts"]) if d["ttfts"] else None,
                "every_req_has_itl": all(len(x) > 0 for x in d["itls"]) if d["itls"] else False,
                "duration_s": d["duration"],
                "sha256": hashlib.sha256(open(f, "rb").read()).hexdigest(),
            })
        out["conc"][str(c)] = trials
    json.dump(out, open(ARRAYS, "w"), indent=1)
    print("wrote", ARRAYS)
    verify()


def verify():
    d = json.load(open(ARRAYS))
    fails = []

    def chk(cond, msg):
        if not cond:
            fails.append(msg); print("  FAIL:", msg)

    print("# AC-5 full-context recomputed from committed arrays (fail-closed)")
    for c in CONCS:
        trials = d["conc"].get(str(c), [])
        chk(len(trials) >= 1, f"c{c}: no trials")
        for t in trials:
            tag = f"c{c} t{t.get('trial')}"
            chk(isinstance(t.get("sha256"), str) and bool(_SHA_RE.match(t["sha256"])), f"{tag}: sha not 64-hex")
            chk(t["completed"] > 0, f"{tag}: completed not > 0")
            chk(t["errors_nonempty"] == 0, f"{tag}: errors>0")
            chk(t["output_len_all_512"], f"{tag}: output_len!=512")
            chk(len(t["ttfts_s"]) == t["completed"], f"{tag}: ttfts len != completed")
            chk(len(t["per_req_gen_tps"]) == t["completed"], f"{tag}: per_req_gen_tps len != completed")
            # the R18 empty-latency class must be impossible here:
            chk(all(x > 0 for x in t["ttfts_s"]), f"{tag}: a ttft==0 (empty-latency row)")
            chk(t["every_req_has_itl"], f"{tag}: a request had zero ITLs (empty-latency row)")
            ttft_p99 = _pct(t["ttfts_s"], 99)
            tps_p50 = _pct(t["per_req_gen_tps"], 50)
            chk(abs(ttft_p99 - t["p99_ttft_ms"] / 1000.0) <= max(TOL, 0.02 * ttft_p99),
                f"{tag}: recomputed p99 ttft {ttft_p99:.3f}s != stored {t['p99_ttft_ms']/1000:.3f}s")
            print(f"  c{c} t{t['trial']}: achieved={t['concurrency']:.2f} completed={t['completed']} "
                  f"P99_TTFT={t['p99_ttft_ms']/1000:.2f}s per_req_TPS_p50={tps_p50:.1f} "
                  f"(<22s={t['p99_ttft_ms']/1000<22.0}, >=30TPS={tps_p50>=30.0})")
    if fails:
        print(f"\nFAIL ({len(fails)})"); raise SystemExit(1)
    print("\nPASS: arrays recompute + fail-closed sanity (no empty-latency rows).")


if __name__ == "__main__":
    if "--build" in sys.argv:
        build()
    elif "--verify" in sys.argv:
        verify()
    else:
        print(__doc__); sys.exit(2)
