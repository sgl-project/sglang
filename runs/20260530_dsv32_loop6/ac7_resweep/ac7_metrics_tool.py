#!/usr/bin/env python3
"""AC-7 exact-recomputable metrics + fail-closed verifier.

Rebuilds `ac7_resweep_metrics.json` from the raw AC-7 sweep JSONLs (gitignored) as an
EXACT source of truth, and recomputes the `ac11_resweep.md` comparator rows from the
committed JSON alone — failing closed on any mismatch.

The comparator (`benchmark_compare.py --ac11`) per concurrency, median over trials:
  - achieved concurrency = the JSONL ``concurrency`` field (effective, NOT ``max_concurrency``).
  - per-request gen TPS p50 = percentile_50( output_lens[i] / sum(itls[i]) ) over requests.
  - P99 TTFT (s)          = stored ``p99_ttft_ms`` / 1000.
Percentile/median methods replicate benchmark_compare.py (_percentile linear-interp, _median).

Usage:
  python3 ac7_metrics_tool.py --build    # reads the raw JSONLs (must be present), writes ac7_resweep_metrics.json
  python3 ac7_metrics_tool.py --verify   # recompute comparator rows from ac7_resweep_metrics.json + assert == ac11_resweep.md; exit 1 on mismatch
"""
import json, glob, hashlib, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARRAYS = os.path.join(HERE, "ac7_resweep_metrics.json")
RESWEEP_MD = os.path.join(HERE, "ac11_resweep.md")
SRC_GLOB = "/tmp/ac7/results/{mode}_gsp_isl4096_osl512_c{c}_t*.jsonl"
MODES = {"DS": "double_sparsity", "DSA": "native_nsa"}
CONCS = [16, 32, 64]
TOL = 0.05  # comparator prints 3 decimals; allow rounding slack


def _percentile(values, pct):
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k); hi = min(lo + 1, len(s) - 1); frac = k - lo
    return float(s[lo] * (1.0 - frac) + s[hi] * frac)


def _median(values):
    nums = sorted(float(v) for v in values if v is not None)
    if not nums:
        return None
    n = len(nums); mid = n // 2
    return float(nums[mid]) if n % 2 else float((nums[mid - 1] + nums[mid]) / 2.0)


def _per_req_gen_tps(d):
    ol, itls = d.get("output_lens"), d.get("itls")
    out = []
    for i in range(min(len(ol), len(itls))):
        if ol[i] and isinstance(itls[i], list):
            s = sum(v for v in itls[i] if isinstance(v, (int, float)))
            if s > 0:
                out.append(float(ol[i]) / s)
    return out


def build():
    out = {"_description": "AC-7 exact per-trial metrics for the 3-trial DS+DSA lifted-point re-sweep. "
                           "Recomputes the ac11_resweep.md comparator rows; raw *.jsonl are gitignored (full SHA256 below).",
           "comparator_formulas": {"achieved": "JSONL 'concurrency' (effective)",
                                   "tps_p50": "percentile50(output_lens[i]/sum(itls[i]))",
                                   "ttft_p99_s": "p99_ttft_ms/1000", "median": "over trials"},
           "conc": {}}
    for c in CONCS:
        out["conc"][str(c)] = {}
        for side, mode in MODES.items():
            trials = []
            for f in sorted(glob.glob(SRC_GLOB.format(mode=mode, c=c))):
                d = json.load(open(f))
                gen = _per_req_gen_tps(d)
                trials.append({
                    "trial": os.path.basename(f).split("_t")[-1].split(".")[0],
                    "concurrency": d["concurrency"],
                    "max_concurrency": d.get("max_concurrency"),
                    "p99_ttft_ms": d["p99_ttft_ms"],
                    "ttfts_s": [float(x) for x in d["ttfts"]],
                    "per_req_gen_tps": gen,
                    "completed": d["completed"],
                    "errors_nonempty": sum(1 for e in d["errors"] if e),
                    "output_len_all_512": all(o == 512 for o in d["output_lens"]),
                    "input_len_median": sorted(d["input_lens"])[len(d["input_lens"]) // 2],
                    "duration_s": d["duration"],
                    "sha256": hashlib.sha256(open(f, "rb").read()).hexdigest(),
                })
            out["conc"][str(c)][side] = trials
    json.dump(out, open(ARRAYS, "w"))
    print("wrote", ARRAYS, "(%d bytes)" % os.path.getsize(ARRAYS))
    verify()


def _parse_reswep_md():
    """Parse the comparator gate table: {conc: {DSA_tps, DS_tps, DSA_ttft, DS_ttft}} and the achieved table."""
    txt = open(RESWEEP_MD).read()
    gates, ach = {}, {}
    for m in re.finditer(r"^\|\s*(16|32|64)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*[\d.]+\s*\|\s*\w+\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", txt, re.M):
        c = int(m.group(1)); gates[c] = {"dsa_tps": float(m.group(2)), "ds_tps": float(m.group(3)),
                                         "dsa_ttft": float(m.group(4)), "ds_ttft": float(m.group(5))}
    for m in re.finditer(r"^\|\s*(16|32|64)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*\d+%\s*\|", txt, re.M):
        c = int(m.group(1)); ach[c] = {"dsa_ach": float(m.group(2)), "ds_ach": float(m.group(3))}
    return gates, ach


def verify():
    d = json.load(open(ARRAYS))
    gates, ach = _parse_reswep_md()
    print("# AC-7 recomputed from ac7_resweep_metrics.json (committed) vs ac11_resweep.md")
    fails = []

    def chk(cond, msg):
        if not cond:
            fails.append(msg); print("  FAIL:", msg)

    for c in CONCS:
        for side in ("DS", "DSA"):
            tr = d["conc"][str(c)][side]
            chk(len(tr) == 3, f"c{c} {side}: {len(tr)} trials (need 3)")
            for t in tr:
                chk(t["errors_nonempty"] == 0, f"c{c} {side} t{t['trial']}: errors>0")
                chk(t["output_len_all_512"], f"c{c} {side} t{t['trial']}: output_len!=512")
                chk(len(t["ttfts_s"]) == t["completed"], f"c{c} {side} t{t['trial']}: ttfts len != completed")
        ds = d["conc"][str(c)]["DS"]; dsa = d["conc"][str(c)]["DSA"]
        r = {
            "ds_ach": _median([t["concurrency"] for t in ds]), "dsa_ach": _median([t["concurrency"] for t in dsa]),
            "ds_tps": _median([_percentile(t["per_req_gen_tps"], 50) for t in ds]),
            "dsa_tps": _median([_percentile(t["per_req_gen_tps"], 50) for t in dsa]),
            "ds_ttft": _median([t["p99_ttft_ms"] / 1000 for t in ds]),
            "dsa_ttft": _median([t["p99_ttft_ms"] / 1000 for t in dsa]),
        }
        print(f"  conc {c}: DS achieved={r['ds_ach']:.3f} (md {ach[c]['ds_ach']})  DS tps={r['ds_tps']:.3f} (md {gates[c]['ds_tps']})  DS ttft={r['ds_ttft']:.3f}s (md {gates[c]['ds_ttft']})")
        chk(abs(r["ds_ach"] - ach[c]["ds_ach"]) <= TOL, f"c{c}: DS achieved {r['ds_ach']:.3f} != md {ach[c]['ds_ach']}")
        chk(abs(r["dsa_ach"] - ach[c]["dsa_ach"]) <= TOL, f"c{c}: DSA achieved {r['dsa_ach']:.3f} != md {ach[c]['dsa_ach']}")
        chk(abs(r["ds_tps"] - gates[c]["ds_tps"]) <= TOL, f"c{c}: DS tps {r['ds_tps']:.3f} != md {gates[c]['ds_tps']}")
        chk(abs(r["dsa_tps"] - gates[c]["dsa_tps"]) <= TOL, f"c{c}: DSA tps {r['dsa_tps']:.3f} != md {gates[c]['dsa_tps']}")
        chk(abs(r["ds_ttft"] - gates[c]["ds_ttft"]) <= TOL, f"c{c}: DS ttft {r['ds_ttft']:.3f} != md {gates[c]['ds_ttft']}")
        chk(abs(r["dsa_ttft"] - gates[c]["dsa_ttft"]) <= TOL, f"c{c}: DSA ttft {r['dsa_ttft']:.3f} != md {gates[c]['dsa_ttft']}")
    if fails:
        print(f"\nFAIL ({len(fails)} mismatch[es])"); raise SystemExit(1)
    print("\nrecomputed == ac11_resweep.md (achieved/TPS/TTFT, DS+DSA, all conc) + sanity: PASS")


if __name__ == "__main__":
    if "--build" in sys.argv:
        build()
    elif "--verify" in sys.argv:
        verify()
    else:
        print(__doc__); sys.exit(2)
