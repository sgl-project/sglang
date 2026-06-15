"""
3-phase write/read latency benchmark for SharedMLA vs. standard hierarchical cache.

SharedMLA shares one host (L2) KV-cache slab across the TP ranks of a node
instead of keeping one identical copy per rank. This script measures whether
reading KV from the shared slab (instead of a per-rank local copy) costs
anything on the two hot paths, at *equal effective L2 capacity* (so cache-hit
behaviour is identical and only the host-pool layout differs).

Each round, against a freshly flushed cache:

  Phase 1 (WRITE): send N fresh ~16k-token prompts. Each is a cold prefill plus
                   an L1->L2 backup write. We time the per-request latency.
  Phase 2 (EVICT): send a burst of filler prompts to push the warm set out of
                   the GPU (L1) pool, leaving it resident only in L2. Not timed.
  Phase 3 (READ):  re-send the same N prompts. They now miss L1 but hit L2 and
                   are served via load_back. We time the per-request latency.

The warm set stays within the L2 pool (it is not evicted from L2), so phase 3 is
a guaranteed L2 hit -- you can confirm this in the server log: the phase-3
prefill batches report `#cached-token` equal to the prompt length.

Run it once against a standard-hicache server and once against a SharedMLA
server with the *same* `--hicache-size`, then compare. The launch commands below
are exactly the ones used for the numbers in the PR (DeepSeek-V3.1, TP8):

  # A: standard hierarchical cache
  python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3.1 \
      --tp 8 --trust-remote-code --port 30001 \
      --mem-fraction-static 0.88 --max-total-tokens 18000 --context-length 16000 \
      --max-running-requests 32 --chunked-prefill-size 8192 \
      --enable-hierarchical-cache --hicache-write-policy write_through \
      --hicache-size 40 &
  python benchmark/hicache/bench_shared_mla.py --tag A_standard --rounds 3 --prompt-words 2400

  # B: SharedMLA (add --enable-shared-mla; everything else identical)
  python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3.1 \
      --tp 8 --trust-remote-code --port 30001 \
      --mem-fraction-static 0.88 --max-total-tokens 18000 --context-length 16000 \
      --max-running-requests 32 --chunked-prefill-size 8192 \
      --enable-hierarchical-cache --enable-shared-mla --hicache-write-policy write_through \
      --hicache-size 40 &
  python benchmark/hicache/bench_shared_mla.py --tag B_sharedmla --rounds 3 --prompt-words 2400

The GPU pool (`--max-total-tokens 18000`) holds a single 16k-token prompt while
staying small enough that phase 2 fills L1 fast; the larger `--hicache-size 40`
keeps the warm set in L2 across the round.
"""

import argparse
import time

import requests


def percentiles(latencies):
    s = sorted(latencies)
    n = len(s)
    return {
        "n": n,
        "avg": sum(s) / n,
        "p50": s[n // 2],
        "p90": s[int(n * 0.9)],
    }


def median(xs):
    return sorted(xs)[len(xs) // 2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag", type=str, default="run", help="Label for the output rows."
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30001)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=30,
        help="Warm-set size N (timed in phases 1 and 3).",
    )
    parser.add_argument(
        "--prompt-words",
        type=int,
        default=1800,
        help="Words per warm prompt; ~1800 tokenizes to ~10k tokens.",
    )
    parser.add_argument(
        "--filler-prompts",
        type=int,
        default=40,
        help="Phase-2 burst size to evict the warm set from L1.",
    )
    parser.add_argument("--filler-words", type=int, default=200)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1,
        help="Keep at 1 so each request's latency is dominated by "
        "prefill / load_back, i.e. the path we measure.",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/generate"
    flush_url = f"http://{args.host}:{args.port}/flush_cache"

    def generate(prompt):
        start = time.time()
        try:
            r = requests.post(
                url,
                json={
                    "text": prompt,
                    "sampling_params": {
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": 0,
                    },
                },
                timeout=args.timeout,
            )
            return time.time() - start, r.status_code == 200
        except Exception:
            return time.time() - start, False

    # Round/index in the token stream so every prompt is unique and deterministic;
    # reusing the same (round, i) in phase 3 reproduces the phase-1 prompt exactly.
    def warm_prompt(rd, i):
        return " ".join("W%dr%dt%d" % (rd, i, j) for j in range(args.prompt_words))

    def filler_prompt(rd, i):
        return " ".join("F%dr%dz%d" % (rd, i, j) for j in range(args.filler_words))

    def flush():
        requests.post(flush_url)
        time.sleep(2)

    print(
        "==== %s : %d rounds, N=%d, prompt_words=%d, filler=%d ===="
        % (
            args.tag,
            args.rounds,
            args.num_prompts,
            args.prompt_words,
            args.filler_prompts,
        ),
        flush=True,
    )

    write_p50s, read_p50s = [], []
    for rd in range(args.rounds):
        flush()

        # Phase 1: WRITE (cold prefill + L2 backup).
        write_lat, fails = [], 0
        for i in range(args.num_prompts):
            dt, ok = generate(warm_prompt(rd, i))
            write_lat.append(dt)
            fails += 0 if ok else 1

        # Phase 2: evict the warm set from L1 (not timed).
        for i in range(args.filler_prompts):
            generate(filler_prompt(rd, i))

        # Phase 3: READ (L1 miss, L2 hit via load_back).
        read_lat = [generate(warm_prompt(rd, i))[0] for i in range(args.num_prompts)]

        w, r = percentiles(write_lat), percentiles(read_lat)
        write_p50s.append(w["p50"])
        read_p50s.append(r["p50"])
        print(
            "[%s][round %d] fail=%d WRITE p50=%.3f avg=%.3f p90=%.3f | "
            "READ p50=%.3f avg=%.3f p90=%.3f"
            % (
                args.tag,
                rd,
                fails,
                w["p50"],
                w["avg"],
                w["p90"],
                r["p50"],
                r["avg"],
                r["p90"],
            ),
            flush=True,
        )

    print(
        "[%s] SUMMARY write_p50_median=%.3f read_p50_median=%.3f over %d rounds"
        % (args.tag, median(write_p50s), median(read_p50s), args.rounds),
        flush=True,
    )


if __name__ == "__main__":
    main()
