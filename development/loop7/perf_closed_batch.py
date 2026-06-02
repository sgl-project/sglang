"""Loop-7 closed-batch decode-TPS probe (AC-6 perf guardrail).

The trustworthy pure-decode-TPS method (per the loop's bench lessons): fire ``conc``
concurrent ``/generate`` requests with a SHORT prompt + ``ignore_eos`` + a fixed
output length, so the server runs a steady closed decode batch (no new arrivals,
``#queue-req: 0``). Per-request decode TPS = output_tokens / e2e (prefill is
negligible for a short prompt, so e2e ~= decode time). This avoids the GSP
window-mode harness that can fabricate throughput from empty streams.

Usage:
    DS_BASE_URL=http://127.0.0.1:30000 python development/loop7/perf_closed_batch.py \
        --conc 16 --osl 256 --label "DS-hybrid graph" --out development/loop7/perf_x.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
import urllib.request


def _one(base_url: str, osl: int):
    body = json.dumps(
        {
            "text": "The capital of France is",
            "sampling_params": {
                "max_new_tokens": osl,
                "temperature": 0.0,
                "ignore_eos": True,
            },
        }
    ).encode()
    t0 = time.time()
    req = urllib.request.Request(
        base_url + "/generate", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    e2e = time.time() - t0
    ct = int(d.get("meta_info", {}).get("completion_tokens", osl))
    return e2e, ct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conc", type=int, required=True)
    ap.add_argument("--osl", type=int, default=256)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base = os.environ.get("DS_BASE_URL", "http://127.0.0.1:30000")

    _one(base, 16)  # warmup (capture/JIT)

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.conc) as ex:
        futs = [ex.submit(_one, base, args.osl) for _ in range(args.conc)]
        res = [f.result() for f in futs]
    wall = time.time() - t0

    e2es = [e for e, _ in res]
    cts = [c for _, c in res]
    per_req_tps = [c / e for c, e in zip(cts, e2es) if e > 0]
    out = {
        "label": args.label,
        "conc": args.conc,
        "osl": args.osl,
        "completed": len(res),
        "per_req_decode_tps_mean": round(sum(per_req_tps) / len(per_req_tps), 2),
        "per_req_decode_tps_min": round(min(per_req_tps), 2),
        "system_throughput_tok_s": round(sum(cts) / wall, 1),
        "mean_e2e_s": round(sum(e2es) / len(e2es), 2),
        "total_out_tokens": sum(cts),
        "wall_s": round(wall, 1),
    }
    print(json.dumps(out), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
