#!/usr/bin/env python3
"""AC-5 conc-16 strict-decode verifier (fail-closed).

Recomputes per-request decode TPS = median(gen throughput)/batch from the committed
closed-batch samples in `ctx8192_decode_curve.json` and asserts the AC-5 conc-16
strict-decode result at the bounded-context client-SLO operating point:
  - conc-16 per-req TPS >= 30.0  (the strict decode bar — MUST hold)
  - conc-32/64 per-req TPS < 30.0 (characterized structural decode-batch ceiling)
  - monotonic non-increase of per-req TPS with batch (sanity)
Exits 1 on any mismatch / missing data; exit 0 + PASS on clean data.

Usage: python3 ctx8192_decode_metrics_tool.py --verify
"""
import json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "ctx8192_decode_curve.json")
STRICT = 30.0


def verify():
    d = json.load(open(SRC))
    op = d.get("operating_point", {})
    fails = []

    def chk(cond, msg):
        if not cond:
            fails.append(msg); print("  FAIL:", msg)

    # operating point must be the bounded-context client-SLO point
    chk(op.get("signature_dtype") == "int8", "signature_dtype != int8")
    chk(op.get("mem_fraction_static") == 0.7, "mem_fraction_static != 0.7")
    chk(op.get("disable_radix_cache") is False, "radix not on")
    chk(op.get("context_length") == 8192, "context_length != 8192")
    chk(op.get("tp_size") == 8, "tp_size != 8")

    per_req = {}
    for b_str, e in d.get("batches", {}).items():
        b = int(b_str)
        samples = e.get("gen_tps_samples", [])
        chk(len(samples) >= 3, f"batch {b}: <3 gen-tps samples")
        for g in samples:
            chk(isinstance(g, (int, float)) and g > 0, f"batch {b}: bad sample {g}")
        med = statistics.median(samples) if samples else None
        chk(med is not None and abs(med - e["gen_tps_median"]) <= 1e-6,
            f"batch {b}: median {med} != stored {e.get('gen_tps_median')}")
        recomputed = round(med / b, 2) if med else None
        chk(recomputed == e["per_req_tps"],
            f"batch {b}: per_req_tps recomputed {recomputed} != stored {e.get('per_req_tps')}")
        per_req[b] = recomputed
        print(f"  batch {b}: median_gen={med}  per_req_tps={recomputed}")

    # the strict gate: conc-16 >= 30
    chk(16 in per_req and per_req[16] >= STRICT,
        f"conc-16 per_req_tps {per_req.get(16)} < strict {STRICT}")
    # characterized: conc 32/64 below 30 (structural)
    for b in (32, 64):
        if b in per_req:
            chk(per_req[b] < STRICT, f"conc-{b} per_req_tps {per_req[b]} unexpectedly >= {STRICT}")
    # monotonic non-increase with batch (decode-batch -> TPS tradeoff)
    bs = sorted(per_req)
    for i in range(1, len(bs)):
        chk(per_req[bs[i]] <= per_req[bs[i - 1]] + 1e-6,
            f"per_req_tps not monotone: batch {bs[i-1]}={per_req[bs[i-1]]} < batch {bs[i]}={per_req[bs[i]]}")

    if fails:
        print(f"\nFAIL ({len(fails)} issue[s])"); raise SystemExit(1)
    print(f"\nPASS: conc-16 per-req decode TPS = {per_req[16]} >= {STRICT} (strict decode bar MET); "
          f"conc-32/64 = {per_req.get(32)}/{per_req.get(64)} < {STRICT} (characterized structural ceiling).")


if __name__ == "__main__":
    if "--verify" in sys.argv:
        verify()
    else:
        print(__doc__); sys.exit(2)
