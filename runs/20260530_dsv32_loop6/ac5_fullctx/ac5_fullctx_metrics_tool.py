#!/usr/bin/env python3
"""AC-5 full-context client-SLO EXACT metrics + fail-closed verifier (Round 21 rebuild).

R20's verifier stored only a DERIVED ``per_req_gen_tps`` array and re-checked it, so mutating that
array to 100.0 still "passed" the strict TPS axis. This rebuild commits the EXACT per-request source
(ttfts, per-request ITL sums + output/input lens + errors) and the verifier RECOMPUTES P99 TTFT and
per-request TPS p50 from those raw arrays (never a stored derived metric), matching the stored headline
at published precision, and validates the operating point from all three `.meta.json` sidecars. It
fails closed on any mismatch / empty-latency row / wrong operating point (demonstrated by tamper tests).

Per-request TPS p50 = percentile50( output_lens[i] / itl_sum_s[i] ).  P99 TTFT (s) = percentile99(ttfts_s).

Usage:
  python3 ac5_fullctx_metrics_tool.py --build    # reads /tmp/ac5r20/results/*.jsonl + sidecars; writes arrays json
  python3 ac5_fullctx_metrics_tool.py --verify   # recompute from committed files alone; exit 1 on any mismatch
"""
import glob, hashlib, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARRAYS = os.path.join(HERE, "ac5_fullctx_arrays.json")
SRC_GLOB = "/tmp/ac5r20/results/double_sparsity_gsp_isl4096_osl512_c{c}_t1.jsonl"
CONCS = [16, 32, 64]
TOL = 0.005  # arrays recompute exactly; allow rounding only
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
SIDECARS = {16: "meta_c16.json", 32: "meta_c32.json", 64: "meta_c64.json"}


def _pct(vals, p):
    s = sorted(vals)
    if not s:
        return None
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k); hi = min(lo + 1, len(s) - 1); f = k - lo
    return float(s[lo] * (1 - f) + s[hi] * f)


def build():
    out = {"_purpose": "AC-5 full-context (DS int8/mem0.7/radix-on/full context/TP8) EXACT per-request source. "
                       "Verifier recomputes P99 TTFT + per-req TPS p50 from raw arrays (no stored derived metric); "
                       "raw *.jsonl gitignored (64-hex sha below). Methodology: np64 steady-state warmup120/window300 "
                       "(cold-flood lesson) -- pending owner approval vs the literal NUM_PROMPTS=320.",
           # The AC-5 workload identity the verifier asserts on EVERY sidecar (fail-closed). num_prompts/
           # warmup/window record the np64-steady-state methodology (pending owner approval); they are
           # asserted for consistency so a sidecar tampered to NUM_PROMPTS=320 fails.
           "expected_workload": {"mode": "double_sparsity", "isl_total_tokens": 4096, "osl_tokens": 512,
                                 "num_prompts": 64, "warmup_seconds": 120.0, "measurement_window_seconds": 300.0,
                                 "chunked_prefill_size": 8192, "max_total_num_tokens": 396096},
           "conc": {}}
    for c in CONCS:
        f = sorted(glob.glob(SRC_GLOB.format(c=c)))[0]
        d = json.load(open(f))
        itls = d["itls"]; ol = d["output_lens"]
        itl_sum = [round(sum(v for v in row if isinstance(v, (int, float))), 6) for row in itls]
        tps = [ol[i] / itl_sum[i] for i in range(len(ol)) if itl_sum[i] > 0]
        out["conc"][str(c)] = {
            "completed": d["completed"],
            "achieved": d["concurrency"],
            "max_concurrency": d.get("max_concurrency"),
            "stored_p99_ttft_ms": d["p99_ttft_ms"],
            "stored_tps_p50": round(_pct(tps, 50), 4),
            # aggregate means are sensitive to EVERY element (unlike a robust
            # percentile/median) so a single-element tamper of ttfts/itl_sum/
            # output_lens shifts them and the verifier catches it.
            "stored_ttft_mean_s": round(sum(d["ttfts"]) / len(d["ttfts"]), 6),
            "stored_tps_mean": round(sum(tps) / len(tps), 6),
            "ttfts_s": [round(float(x), 6) for x in d["ttfts"]],
            "itl_sum_s": itl_sum,
            "output_lens": ol,
            "input_lens": d["input_lens"],
            "errors_empty": [(not e) for e in d.get("errors", [])],
            "gen_nonempty_count": sum(1 for t in d.get("generated_texts", []) if t),
            "duration_s": d["duration"],
            "sha256": hashlib.sha256(open(f, "rb").read()).hexdigest(),
            "sidecar": SIDECARS[c],
        }
    json.dump(out, open(ARRAYS, "w"))
    print("wrote", ARRAYS, "(%d bytes)" % os.path.getsize(ARRAYS))
    verify()


def _sidecar_args(path):
    d = json.load(open(path))
    return d.get("server_args", d)


def verify():
    d = json.load(open(ARRAYS))
    fails = []

    def chk(cond, msg):
        if not cond:
            fails.append(msg); print("  FAIL:", msg)

    print("# AC-5 full-context recomputed from committed exact arrays + sidecars (fail-closed)")
    for c in CONCS:
        e = d["conc"].get(str(c))
        chk(e is not None, f"c{c}: missing")
        if e is None:
            continue
        tag = f"c{c}"
        n = e["completed"]
        chk(isinstance(e.get("sha256"), str) and bool(_SHA_RE.match(e["sha256"])), f"{tag}: sha not 64-hex")
        for k in ("ttfts_s", "itl_sum_s", "output_lens", "input_lens", "errors_empty"):
            chk(len(e[k]) == n, f"{tag}: len({k})={len(e[k])} != completed {n}")
        chk(n > 0, f"{tag}: completed not > 0")
        chk(all(o == 512 for o in e["output_lens"]), f"{tag}: an output_len != 512")
        chk(all(e["errors_empty"]), f"{tag}: a non-empty error present")
        chk(all(t > 0 for t in e["ttfts_s"]), f"{tag}: a ttft <= 0 (empty-latency row)")
        chk(all(s > 0 for s in e["itl_sum_s"]), f"{tag}: an itl_sum <= 0 (empty-latency row)")
        chk(e["gen_nonempty_count"] == n, f"{tag}: gen_nonempty_count {e['gen_nonempty_count']} != completed {n}")
        # recompute the published headline from the raw arrays (independent of any stored metric)
        ttft_p99 = _pct(e["ttfts_s"], 99)
        tps = [e["output_lens"][i] / e["itl_sum_s"][i] for i in range(n) if e["itl_sum_s"][i] > 0]
        tps_p50 = _pct(tps, 50)
        chk(abs(ttft_p99 - e["stored_p99_ttft_ms"] / 1000.0) <= max(TOL, 0.005 * ttft_p99),
            f"{tag}: recomputed P99 TTFT {ttft_p99:.3f}s != stored {e['stored_p99_ttft_ms']/1000:.3f}s")
        chk(abs(tps_p50 - e["stored_tps_p50"]) <= max(TOL, 0.005 * tps_p50),
            f"{tag}: recomputed TPS p50 {tps_p50:.3f} != stored {e['stored_tps_p50']:.3f}")
        # aggregate-mean integrity: catches a single-element tamper a robust
        # percentile would miss (one outlier does not move the median).
        ttft_mean = sum(e["ttfts_s"]) / n
        tps_mean = sum(tps) / len(tps)
        chk(abs(ttft_mean - e["stored_ttft_mean_s"]) <= max(1e-4, 1e-4 * ttft_mean),
            f"{tag}: recomputed mean TTFT {ttft_mean:.5f}s != stored {e['stored_ttft_mean_s']:.5f}s")
        chk(abs(tps_mean - e["stored_tps_mean"]) <= max(1e-4, 1e-4 * tps_mean),
            f"{tag}: recomputed mean TPS {tps_mean:.5f} != stored {e['stored_tps_mean']:.5f}")
        # operating-point invariants from the committed sidecar
        sc = os.path.join(HERE, e["sidecar"])
        chk(os.path.exists(sc), f"{tag}: sidecar {e['sidecar']} missing")
        if os.path.exists(sc):
            meta = json.load(open(sc))
            a = meta.get("server_args", meta)
            ew = d["expected_workload"]
            # workload identity (top-level sidecar fields) — fail-closed so a sidecar tampered to a
            # different mode / conc / ISL / OSL / num_prompts is rejected (Codex R21 fail-open gap).
            chk(meta.get("mode") == ew["mode"], f"{tag}: sidecar mode {meta.get('mode')} != {ew['mode']}")
            chk(meta.get("concurrency") == c, f"{tag}: sidecar concurrency {meta.get('concurrency')} != key {c}")
            chk(meta.get("isl_total_tokens") == ew["isl_total_tokens"], f"{tag}: sidecar ISL != 4096")
            chk(meta.get("osl_tokens") == ew["osl_tokens"], f"{tag}: sidecar OSL != 512")
            chk(meta.get("num_prompts") == ew["num_prompts"], f"{tag}: sidecar num_prompts {meta.get('num_prompts')} != recorded {ew['num_prompts']} (methodology)")
            chk(meta.get("warmup_seconds") == ew["warmup_seconds"], f"{tag}: sidecar warmup_seconds != {ew['warmup_seconds']}")
            chk(meta.get("measurement_window_seconds") == ew["measurement_window_seconds"], f"{tag}: sidecar window != {ew['measurement_window_seconds']}")
            chk(a.get("max_total_num_tokens") == ew["max_total_num_tokens"], f"{tag}: server_args max_total_num_tokens {a.get('max_total_num_tokens')} != {ew['max_total_num_tokens']}")
            chk(a.get("chunked_prefill_size") == ew["chunked_prefill_size"], f"{tag}: server_args chunked_prefill_size != {ew['chunked_prefill_size']}")
            chk(a.get("enable_double_sparsity") is True, f"{tag}: sidecar enable_double_sparsity != True")
            chk('"signature_dtype": "int8"' in json.dumps(a.get("double_sparsity_config", "")) or
                a.get("double_sparsity_config", {}).get("signature_dtype") == "int8" if isinstance(a.get("double_sparsity_config"), dict) else
                "int8" in str(a.get("double_sparsity_config")), f"{tag}: sidecar signature_dtype != int8")
            chk(a.get("mem_fraction_static") == 0.7, f"{tag}: sidecar mem_fraction_static != 0.7")
            chk(a.get("disable_radix_cache") is False, f"{tag}: sidecar disable_radix_cache != False")
            chk(bool(a.get("double_sparsity_radix_fixture_artifact")), f"{tag}: sidecar no radix fixture artifact")
            chk(a.get("context_length") is None, f"{tag}: sidecar context_length not null (not full context)")
            chk(a.get("tp_size") == 8, f"{tag}: sidecar tp_size != 8")
            chk(a.get("enable_request_time_stats_logging") is True, f"{tag}: sidecar request-time-stats off")
        print(f"  c{c}: achieved={e['achieved']:.2f} completed={n}  P99_TTFT={ttft_p99:.2f}s "
              f"per_req_TPS_p50={tps_p50:.1f}  (<22s={ttft_p99<22.0}, >=30TPS={tps_p50>=30.0})")
    if fails:
        print(f"\nFAIL ({len(fails)} issue[s])"); raise SystemExit(1)
    print("\nPASS: P99 TTFT + per-req TPS recompute from raw arrays == stored headline; operating point "
          "verified from all sidecars; no empty-latency rows.")


if __name__ == "__main__":
    if "--build" in sys.argv:
        build()
    elif "--verify" in sys.argv:
        verify()
    else:
        print(__doc__); sys.exit(2)
