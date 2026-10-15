"""Measure independent real-text requests against a dedicated SGLang server.

Flush prefix cache before every request. Model/kernel warmup is separate from
request-owned DoSA prior state; every measured request uses a unique RID.
"""
import argparse
import json
from pathlib import Path
import statistics
import time
import urllib.request
import uuid


def measure(url, text_path, output, loaded_n=5):
    if loaded_n < 2:
        raise ValueError("Use at least two loaded requests to report dispersion")
    text_path, output = Path(text_path), Path(output)
    text = text_path.read_bytes().decode("utf-8")
    run_id = uuid.uuid4().hex
    state = dict(status="started", url=url, text_path=str(text_path),
                 text_bytes=len(text.encode()), loaded_n=loaded_n, requests=[],
                 sampling_params={"max_new_tokens": 2}, request_retries=0,
                 accounting="Each unique request pays its prior. Cache flush and client payload encoding are outside TTFT; server tokenization and prefill are inside.")
    # Never overwrite an earlier attempt or silently resubmit its requests.
    with output.open("x", encoding="utf-8") as stream:
        json.dump(state, stream, indent=2)

    def save():
        temporary = output.with_name(output.name + ".pending-" + run_id)
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(output)

    def request(path, body=None):
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
        return urllib.request.Request(url.rstrip("/") + path, data=data,
                                      headers={"Content-Type": "application/json"})

    try:
        with urllib.request.urlopen(request("/get_server_info"), timeout=30) as response:
            info = json.load(response)
        fields = ("tp_size", "nnodes", "chunked_prefill_size", "mem_fraction_static",
                  "page_size", "dsa_topk_backend", "max_total_num_tokens", "stream_interval")
        state["server"] = {key: info.get(key) for key in fields}
        save()
        token_count = None
        for index in range(loaded_n + 1):
            with urllib.request.urlopen(request("/flush_cache?timeout=10", {}), timeout=30) as response:
                response.read()
            rid = f"dosa-ttft-{run_id}-{index}"
            row = dict(rid=rid, kind="warmup" if index == 0 else "loaded",
                       status="sent", events=[], done=False, ttft_s=None)
            state["requests"].append(row)
            save()
            payload = request("/generate", dict(text=text, rid=rid, stream=True,
                                                sampling_params=state["sampling_params"]))
            start = time.perf_counter()
            with urllib.request.urlopen(payload, timeout=1800) as response:
                for line in response:
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        row["done"] = True
                        break
                    event = json.loads(data)
                    elapsed = time.perf_counter() - start
                    row["events"].append(dict(elapsed_s=elapsed, data=event))
                    if row["ttft_s"] is None and event.get("meta_info", {}).get("completion_tokens", 0) > 0:
                        row["ttft_s"] = elapsed
            row["e2e_s"] = time.perf_counter() - start
            if not row["done"] or row["ttft_s"] is None:
                raise RuntimeError("Stream did not finish with an actual output token")
            metadata = row["events"][-1]["data"]["meta_info"]
            if metadata.get("cached_tokens") != 0 or metadata.get("num_retractions") != 0:
                raise RuntimeError("Request reused prefix cache or retracted; keep raw, do not retry")
            if not 1 <= metadata["completion_tokens"] <= 2:
                raise RuntimeError("Unexpected output count; natural one-token EOS is allowed")
            if token_count is None:
                token_count = metadata["prompt_tokens"]
            if metadata["prompt_tokens"] != token_count:
                raise RuntimeError("Normal tokenizer input length changed across requests")
            row["status"] = "complete"
            save()
        samples = [row["ttft_s"] for row in state["requests"] if row["kind"] == "loaded"]
        state.update(status="complete", prompt_tokens=token_count, loaded_samples_s=samples,
                     loaded_median_s=statistics.median(samples), loaded_SD_s=statistics.stdev(samples))
        save()
        return state
    except Exception as error:
        state.update(status="failed_preserved", error=str(error))
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--text", type=Path, required=True, help="Real UTF-8 text; normal server tokenizer")
    parser.add_argument("--output", type=Path, required=True, help="New JSON file; all samples and failures retained")
    parser.add_argument("--loaded", type=int, default=5)
    args = parser.parse_args()
    result = measure(args.url, args.text, args.output, args.loaded)
    print(json.dumps({key: result[key] for key in
                      ("prompt_tokens", "loaded_n", "loaded_median_s", "loaded_SD_s", "loaded_samples_s")}, indent=2))


if __name__ == "__main__":
    main()
