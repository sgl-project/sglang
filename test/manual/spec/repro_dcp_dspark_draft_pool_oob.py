#!/usr/bin/env python3
"""Repro: DCP + DSPARK long-sequence crash (draft KV pool virtual-loc OOB).

Mechanism
---------
Under DCP the shared allocator hands out *virtual* cache locs in
``[0, max_total_num_tokens * dcp_size)`` (kv_cache_configurator builds
``PagedTokenToKVPoolAllocator(max_total * dcp_size, page * dcp_size)``).
The target MLA pool translates them (``loc // dcp_size`` + owner mask in the
Triton write kernel and the DCP page-table build). The DSPARK draft worker
shares that allocator but its dense-GQA KV pool is sized only
``max_total_num_tokens`` rows and consumes the raw virtual locs for both
writes (``set_kv_buffer`` / ``set_kv_buffer_prefix_valid``) and reads
(trtllm_mha page table: ``req_to_token // page_size``, no ``// dcp_size``).
The draft is self-consistent below the boundary, so short outputs are fine;
the first loc past ``max_total`` starts unchecked OOB scatters that corrupt
neighboring GPU memory, detonating later as an ATen indexing device assert
(``vectorized_gather_kernel`` / ``indexSelectSmallIndex: srcIndex <
srcSelectDimSize``). Observed in the wild ~68 min into AIME26 x16 on
TP8+DCP8+DSPARK (accept length also drifts down as a growing fraction of
context pages falls past the boundary).

Repro (compresses the 68-min failure to ~5 min)
-----------------------------------------------
Launch Kimi-K3 with the DCP+DSPARK composition and a small token budget so
the allocator frontier crosses ``max_total`` quickly, e.g.:

    export SGLANG_RAGGED_VERIFY_MODE=static
    export SGLANG_PREP_IN_CUDA_GRAPH=1
    python -m sglang.launch_server \
      --model-path <Kimi-K3> --tp 8 --dcp-size 8 --dcp-comm-backend a2a \
      --dcp-replicate-q-proj --trust-remote-code --kv-cache-dtype fp8_e4m3 \
      --moe-runner-backend flashinfer_mxfp4 --bf16-gemm-backend cutedsl \
      --mem-fraction-static 0.85 --max-total-tokens 65536 \
      --cuda-graph-max-bs-decode 128 --enable-symm-mem \
      --attention-backend tokenspeed_mla --skip-server-warmup \
      --speculative-algorithm DSPARK \
      --speculative-draft-model-path <dspark-draft> \
      --speculative-draft-attention-backend trtllm_mha \
      --speculative-dspark-block-size 7

then run this script:

    python test/manual/spec/repro_dcp_dspark_draft_pool_oob.py \
      --base http://127.0.0.1:30001 --concurrency 16 --max-new 49152

Unfixed, every rank dies with the device assert at exactly
``#full token == max_total_num_tokens`` (65,536 here; 16 reqs x 4,096 tok).
Any draft checkpoint works — a random-weight draft reproduces it; the bug is
in loc plumbing, not draft quality. Fixed (draft pool sized
``max_total * dcp_size``), the same workload runs past the boundary.
"""
import argparse
import concurrent.futures as cf
import json
import time
import urllib.request


def one_request(base, idx, max_new, prompt_len):
    prompt = "9 " * prompt_len + f"request {idx}: please think. "
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_new,
            "ignore_eos": True,
            "temperature": 1.0,
            "top_p": 1.0,
        },
        "stream": False,
    }
    req = urllib.request.Request(
        f"{base}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=86400) as r:
            data = json.loads(r.read())
        meta = data.get("meta_info", {})
        return idx, "OK", meta.get("completion_tokens"), time.time() - t0
    except Exception as e:  # noqa: BLE001
        return idx, f"ERR:{type(e).__name__}:{str(e)[:120]}", None, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:30001")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=49152)
    ap.add_argument("--prompt-len", type=int, default=64)
    args = ap.parse_args()

    print(f"launching {args.concurrency} x {args.max_new}-token generations", flush=True)
    t0 = time.time()
    fails = 0
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [
            ex.submit(one_request, args.base, i, args.max_new, args.prompt_len)
            for i in range(args.concurrency)
        ]
        for f in cf.as_completed(futs):
            idx, status, toks, dt = f.result()
            print(
                f"[{time.time()-t0:8.1f}s] req{idx:02d} {status} tokens={toks} dt={dt:.0f}s",
                flush=True,
            )
            if status != "OK":
                fails += 1
    print(f"DONE fails={fails}", flush=True)
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
