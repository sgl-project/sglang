#!/usr/bin/env python3
"""Closed-batch pure-decode profiler for the AC-5 remediation.

Fires N concurrent /generate requests (a fixed ~4096-token prompt, ignore_eos so
each decodes exactly --max-new-tokens steps) with NO new arrivals, so once all N
finish prefill the server runs a CLEAN decode batch of N with no prefill interleave.
The server decode-batch log's steady "gen throughput (token/s)" / N is then the pure
per-request decode TPS at batch N — isolating real decode cost from the WARMUP=0
cold-flood prefill-interleave the AC-5 directional run was contaminated by.

Client-side it also records each request's wall time + completion_tokens as a
cross-check. Writes <outdir>/closed_batch_b<N>.json.

Usage:
  python3 closed_batch_decode.py --host 127.0.0.1 --port 30000 \
      --prompt /tmp/ac5r17/prompt_4096.json --batch 16 --max-new-tokens 256 \
      --outdir /sgl-workspace/sglang/runs/20260530_dsv32_loop6/ac5_decode_profile --tag DS
"""
import argparse, json, os, time, threading, urllib.request, urllib.error

def one(base, prompt, m, results, i):
    payload = {"text": prompt, "sampling_params": {
        "max_new_tokens": m, "temperature": 0.0, "ignore_eos": True}}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(base + "/generate", data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1200) as r:
            resp = json.loads(r.read().decode())
        dt = time.time() - t0
        meta = resp.get("meta_info", {})
        results[i] = {"ok": True, "latency_s": dt,
                      "prompt_tokens": meta.get("prompt_tokens"),
                      "completion_tokens": meta.get("completion_tokens"),
                      "finish_reason": meta.get("finish_reason")}
    except Exception as e:  # noqa: BLE001
        results[i] = {"ok": False, "error": f"{type(e).__name__}: {e}",
                      "latency_s": time.time() - t0}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default="30000")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", default="DS")
    a = ap.parse_args()
    base = f"http://{a.host}:{a.port}"
    prompt = json.load(open(a.prompt))["prompt"]
    os.makedirs(a.outdir, exist_ok=True)
    N = a.batch
    results = [None] * N
    t_wall0 = time.time()
    ths = [threading.Thread(target=one, args=(base, prompt, a.max_new_tokens, results, i))
           for i in range(N)]
    for t in ths: t.start()
    for t in ths: t.join()
    wall = time.time() - t_wall0
    ok = [r for r in results if r and r["ok"]]
    # client-side aggregate (includes prefill in latency; server gen-throughput is the pure-decode signal)
    comp = [r["completion_tokens"] for r in ok if r["completion_tokens"]]
    lat = [r["latency_s"] for r in ok]
    out = {
        "tag": a.tag, "batch": N, "max_new_tokens": a.max_new_tokens,
        "wall_s": round(wall, 3), "completed": len(ok),
        "prompt_tokens": (ok[0]["prompt_tokens"] if ok else None),
        "completion_tokens_each": comp[:N],
        "client_latency_s_median": round(sorted(lat)[len(lat)//2], 3) if lat else None,
        "note": "PURE per-request decode TPS = server 'Decode batch #running-req:%d gen throughput'/%d "
                "(read from the server log over the steady window); client latency includes prefill." % (N, N),
        "per_request": results,
    }
    path = os.path.join(a.outdir, f"closed_batch_b{N}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("tag","batch","max_new_tokens","wall_s","completed",
                                          "prompt_tokens","client_latency_s_median")}, indent=1))
    print("wrote", path)

if __name__ == "__main__":
    main()
