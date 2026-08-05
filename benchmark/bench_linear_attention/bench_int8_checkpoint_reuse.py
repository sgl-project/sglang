"""Benchmark: int8 linear-attention checkpoint pool — prefix-reuse capacity & latency.

Drives a *running* SGLang server that serves a linear-attention (KDA / GDN) hybrid
model, and measures how prefix reuse — and the probe-phase prefill latency that
depends on it — holds up as the number of DISTINCT cached prefixes grows.

The active bf16 mamba state pool caches one state per distinct prefix and is sized
to the running set; once the number of distinct cached prefixes exceeds it, reuse
collapses (states get evicted and recomputed). With ``--enable-int8-mamba-checkpoint``
the radix-cached states live in a separate int8 pool holding ~2x more slots at ~the
same memory, so the collapse knee moves out ~2x and the probe-phase prefill stays a
cheap cache hit well past the bf16 pool size.

Method, per K in ``--num-prefixes``:
  flush -> WARM: send K distinct ~P-token prefixes once (populate the cache)
        -> PROBE: re-send each prefix with a short different suffix; read
           meta_info.cached_tokens (= reused prefix length) and time each request.
  reuse_frac = sum(cached) / sum(prefix_tokens);  probe throughput = K / wall_time.

Run the server twice and compare (same flags, toggle int8):

  python -m sglang.launch_server --model-path <gdn-or-kda-hybrid> --tp 4 \
      --trust-remote-code --mamba-scheduler-strategy extra_buffer \
      --max-mamba-cache-size 256 [--enable-int8-mamba-checkpoint] --port 30000

  python benchmark/bench_linear_attention/bench_int8_checkpoint_reuse.py \
      --port 30000 --prefix-tokens 1000 --num-prefixes 128 384 640 --parallel 8

NOTE: prefix-tokens must cross the mamba cache chunk granularity (typically ~512),
otherwise nothing is cacheable and reuse is 0 by construction (not a regression).
Use ``--mamba-scheduler-strategy extra_buffer`` on the server: ``no_buffer`` only
snapshots state at the full-sequence leaf, so a divergent-suffix probe never reuses.
"""

import argparse
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import requests

# A small word pool so each prefix is distinct but realistic English text.
_VOCAB = (
    "time year people way day man thing woman life child world school state family "
    "student group country problem hand part place case week company system program "
    "question work government number night point home water room mother area money "
    "story fact month lot right study book eye job word business issue side kind head "
    "house service friend father power hour game line end member law car city community "
    "name president team minute idea body information back parent face level office door "
    "health person art war history party result change morning reason research girl guy "
    "moment air teacher force education foot boy age policy process music market sense "
    "nation plan college interest death course someone experience behavior career goal"
).split()


def make_prefix(i: int, n_words: int) -> str:
    rng = random.Random(1000 + i)
    head = f"Document {i} unique tag {i * 7919 % 100000}. "
    return head + " ".join(rng.choice(_VOCAB) for _ in range(n_words))


def make_suffix(i: int, salt: int, n_words: int) -> str:
    rng = random.Random(salt * i + salt)
    return " ".join(rng.choice(_VOCAB) for _ in range(n_words))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument(
        "--num-prefixes",
        type=int,
        nargs="+",
        default=[128, 384, 640],
        help="distinct-prefix counts (K) to sweep",
    )
    ap.add_argument(
        "--prefix-tokens",
        type=int,
        default=1000,
        help="approx prefix length in words (must cross the ~512 chunk granularity)",
    )
    ap.add_argument("--suffix-tokens", type=int, default=8)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    gen_url = base + "/generate"

    def send(prompt):
        t0 = time.perf_counter()
        try:
            r = requests.post(
                gen_url,
                headers={"Content-Type": "application/json"},
                json={
                    "text": prompt,
                    "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
                },
                timeout=args.timeout,
            )
            dt = time.perf_counter() - t0
            mi = r.json().get("meta_info", {})
            return mi.get("prompt_tokens"), mi.get("cached_tokens"), dt, None
        except Exception as e:  # noqa: BLE001
            return None, None, time.perf_counter() - t0, str(e)[:80]

    def flush():
        try:
            requests.post(base + "/flush_cache", timeout=60)
            time.sleep(1.5)
        except Exception:
            pass

    print(
        f"server={base} prefix_tokens~{args.prefix_tokens} suffix_tokens={args.suffix_tokens} "
        f"parallel={args.parallel} K_sweep={args.num_prefixes}"
    )
    header = (
        f"{'K':>6} {'reuse_frac':>11} {'probe_p50_ms':>13} {'probe_p90_ms':>13} "
        f"{'probe_thru_rps':>15} {'errors':>7}"
    )
    print(header)
    print("-" * len(header))

    for K in args.num_prefixes:
        flush()
        prefixes = [make_prefix(i, args.prefix_tokens) for i in range(K)]
        warm = [
            p + " " + make_suffix(i, 7, args.suffix_tokens)
            for i, p in enumerate(prefixes)
        ]
        probe = [
            p + " " + make_suffix(i, 13, args.suffix_tokens)
            for i, p in enumerate(prefixes)
        ]

        with ThreadPoolExecutor(args.parallel) as ex:
            list(ex.map(send, warm))  # WARM: populate the cache
            t0 = time.perf_counter()
            results = list(ex.map(send, probe))  # PROBE: measured
            wall = time.perf_counter() - t0

        ok = [(pt, ct, dt) for (pt, ct, dt, err) in results if err is None and pt]
        errs = [r for r in results if r[3] is not None]
        if not ok:
            print(f"{K:>6}  ALL-ERR e.g. {errs[:1]}")
            continue
        sum_prompt = sum(pt for pt, _, _ in ok)
        sum_cached = sum((ct or 0) for _, ct, _ in ok)
        lat_ms = sorted(dt * 1000.0 for _, _, dt in ok)
        p50 = statistics.median(lat_ms)
        p90 = lat_ms[min(len(lat_ms) - 1, int(0.9 * len(lat_ms)))]
        thru = len(ok) / wall if wall > 0 else 0.0
        print(
            f"{K:>6} {sum_cached / max(1, sum_prompt):>11.3f} {p50:>13.1f} "
            f"{p90:>13.1f} {thru:>15.1f} {len(errs):>7}"
        )


if __name__ == "__main__":
    main()
