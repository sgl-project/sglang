#!/usr/bin/env python3
"""AC-8 64K servability probe driver (reproducible).

Sends the committed ~70K-token probe (development/loop6/probe_64k.json) to the raw
/generate endpoint of the lifted DS int8 / mem-0.7 / radix-on server and records,
as durable tracked JSON, whether the prompt is now ADMITTED (HTTP 200) instead of
the Loop-5 mem-0.6 "HTTP 400 Input length (69970) exceeds the maximum allowed (53050)".

A server rejection is a RECORDABLE result (status + body), never an uncaught error —
so a still-too-big prompt yields a characterized ceiling, not a crash.

Usage:
  python3 ac8_probe.py --host 127.0.0.1 --port 30000 \
      --probe /sgl-workspace/sglang/development/loop6/probe_64k.json \
      --outdir /sgl-workspace/sglang/runs/20260530_dsv32_loop6/ac8_servability
"""
import argparse, json, hashlib, os, sys, time, urllib.request, urllib.error

def _get(url, timeout):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, {"_http_error_body": e.read().decode(errors="replace")}
    except Exception as e:  # noqa: BLE001 - record, never raise
        return None, {"_error": f"{type(e).__name__}: {e}"}

def _post(url, payload, timeout):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode()), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, {"_http_error_body": e.read().decode(errors="replace")}, time.time() - t0
    except Exception as e:  # noqa: BLE001 - record, never raise
        return None, {"_error": f"{type(e).__name__}: {e}"}, time.time() - t0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default="30000")
    ap.add_argument("--probe", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    base = f"http://{a.host}:{a.port}"
    os.makedirs(a.outdir, exist_ok=True)

    probe = json.load(open(a.probe))
    text = probe["text"]
    text_sha = hashlib.sha256(text.encode()).hexdigest()
    assert text_sha == probe["text_sha256"], "probe text sha mismatch vs committed payload"

    # 1) server alive + operating point BEFORE
    st_b, info_b = _get(f"{base}/get_server_info", timeout=30)
    json.dump(info_b, open(os.path.join(a.outdir, "get_server_info_before.json"), "w"), indent=1)

    # 2) the probe: raw /generate (the same admission path that 400'd at mem-0.6)
    params = probe.get("generate_params", {"max_new_tokens": 16, "temperature": 0.0})
    payload = {"text": text, "sampling_params": {
        "max_new_tokens": params.get("max_new_tokens", 16),
        "temperature": params.get("temperature", 0.0)}}
    status, resp, secs = _post(f"{base}/generate", payload, timeout=900)

    # 3) server alive + operating point AFTER
    st_a, info_a = _get(f"{base}/get_server_info", timeout=30)
    json.dump(info_a, open(os.path.join(a.outdir, "get_server_info_after.json"), "w"), indent=1)

    meta = resp.get("meta_info", {}) if isinstance(resp, dict) else {}
    out_text = resp.get("text", "") if isinstance(resp, dict) else ""
    mtt = None
    for src in (info_a, info_b):
        if isinstance(src, dict):
            mtt = src.get("max_total_num_tokens", mtt)
    result = {
        "_purpose": "AC-8 ~70K-token /generate servability/admission probe at lifted DS int8/mem-0.7/radix-on.",
        "endpoint": "/generate (raw)",
        "probe_file": os.path.basename(a.probe),
        "probe_text_sha256": text_sha,
        "probe_n_tokens_local_estimate": probe.get("n_tokens_local_estimate"),
        "http_status": status,
        "admitted_http_200": status == 200,
        "server_alive_before": st_b == 200,
        "server_alive_after": st_a == 200,
        "served_max_total_num_tokens": mtt,
        "prompt_tokens_reported": meta.get("prompt_tokens"),
        "completion_tokens_reported": meta.get("completion_tokens"),
        "finish_reason": meta.get("finish_reason"),
        "latency_s": round(secs, 3),
        "output_text_snippet": out_text[:400],
        "loop5_mem06": {"pool_tokens": 53056, "rejected_prompt_tokens": 69970,
                        "http_status": 400,
                        "reason": "Input length (69970) exceeds the maximum allowed (53050)"},
        "response_raw_if_error": (resp if status != 200 else None),
    }
    json.dump(result, open(os.path.join(a.outdir, "ac8_probe_response.json"), "w"), indent=1)
    print(json.dumps({k: result[k] for k in (
        "http_status", "admitted_http_200", "served_max_total_num_tokens",
        "prompt_tokens_reported", "completion_tokens_reported", "finish_reason",
        "latency_s", "server_alive_before", "server_alive_after")}, indent=1))
    print("output snippet:", repr(out_text[:160]))

if __name__ == "__main__":
    main()
