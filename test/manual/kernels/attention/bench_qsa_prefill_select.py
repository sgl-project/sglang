"""QSA indexer prefill selection on gfx95, today's against the new one.

Every arm calls ``QSAIndexer.select_prefill_tokens`` itself on the same inputs. The
"today" arm runs it with ``qsa_indexer.is_gfx95_supported`` returning False, which takes the
branch every other platform takes (fast_topk on scaled scores); the new arm takes
the gfx95 branch (topk_v2 on unscaled scores). The "new_divided" arm takes the gfx95
branch with the scores still divided by sqrt(head_dim), which splits the difference
between the top-k and the division pass. Nothing else differs.

Each input is one request of Gaussian, rms-normed Q and K at the serving geometry (4
index heads, head_dim 128, compress ratio 4), chunked as the indexer chunks it.

Timing: a 512 MB write before every timed call (cold L2), CUDA events around one call
of the whole function (all row chunks: one layer), the arms interleaved with the
starting arm rotating every rep, and an A/A control (the new arm twice). The profiler
names the kernels of one call per arm: the new arm must launch topk_ragged_kernel and
neither a fast_topk kernel nor the division's multiply, the today arm fast_topk and the
multiply, new_divided topk_ragged_kernel and the multiply; the script exits 1 otherwise.

  python bench_qsa_prefill_select.py [--reps 31] [--cases served,no_prefix] [--out result.json]
"""

import argparse
import json
import statistics
import sys
import time
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.qsa import qsa_indexer
from sglang.srt.layers.attention.qsa.mqa import HAS_TILELANG
from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.utils import is_gfx95_supported

HEAD_DIM = 128
NUM_Q_HEADS = 4
RATIO = 4
BLOCK_TOPK = 512
TOKEN_TOPK = 2048
FLUSH_BYTES = 512 << 20

# name -> (new query rows, cached prefix tokens) of the one request
CASES = {
    "served": (16384, 83968),
    "no_prefix": (16384, 0),
    "chunk4096_on_96256": (4096, 96256),
    "odd_width": (16384, 83973),  # 25,089 compressed keys: not a multiple of 4
}
# arm -> kernel fragments one call must launch / must not launch; the division by
# sqrt(head_dim) runs as an elementwise MulFunctor (a multiply by the reciprocal)
EXPECTED_KERNELS = {
    "today": (("fast_topk_kernel", "MulFunctor"), ("topk_ragged_kernel",)),
    "new": (("topk_ragged_kernel",), ("fast_topk", "MulFunctor")),
    "new_repeat": (("topk_ragged_kernel",), ("fast_topk", "MulFunctor")),
    "new_divided": (("topk_ragged_kernel", "MulFunctor"), ("fast_topk",)),
}
ARMS = list(EXPECTED_KERNELS)

_original_gate = qsa_indexer.is_gfx95_supported
_original_scoring = qsa_indexer.qsa_mqa_prefill_relative
_flush_buffer = None


def flush_l2():
    global _flush_buffer
    if _flush_buffer is None:
        _flush_buffer = torch.empty(
            FLUSH_BYTES // 4, dtype=torch.float32, device="cuda"
        )
    _flush_buffer.fill_(1.0)


def rms_normed(x, generator):
    weight = 1.0 + 0.1 * torch.randn(x.shape[-1], generator=generator, device=x.device)
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * weight).bfloat16()


def make_case(rows, prefix, seed=11):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    seq_len = prefix + rows
    keys = seq_len // RATIO
    positions = torch.arange(prefix, seq_len, device="cuda")
    return SimpleNamespace(
        q=rms_normed(
            torch.randn(rows, NUM_Q_HEADS, HEAD_DIM, generator=gen, device="cuda"), gen
        ),
        k=rms_normed(torch.randn(keys, 1, HEAD_DIM, generator=gen, device="cuda"), gen),
        starts=torch.zeros(rows, dtype=torch.int32, device="cuda"),
        ends=torch.clamp((positions + 1) // RATIO, max=keys).int(),
        positions=positions,
        seq_lens=torch.full((rows,), seq_len, dtype=torch.int32, device="cuda"),
        max_row_keys=keys,
    )


def select(case):
    indexer = SimpleNamespace(
        token_topk=TOKEN_TOPK, compress_ratio=RATIO, block_topk=BLOCK_TOPK
    )
    return QSAIndexer.select_prefill_tokens(
        indexer,
        case.q,
        case.k,
        case.starts,
        case.ends,
        case.positions,
        case.seq_lens,
        max_row_keys=case.max_row_keys,
    )


def divided_scoring(q, k, row_starts, row_ends, width, score_scale=None):
    return _original_scoring(q, k, row_starts, row_ends, width, None)


def use_arm(arm):
    qsa_indexer.is_gfx95_supported = (
        (lambda: False) if arm == "today" else _original_gate
    )
    qsa_indexer.qsa_mqa_prefill_relative = (
        divided_scoring if arm == "new_divided" else _original_scoring
    )


def time_arms(case, reps, warmup=3):
    """Per arm: GPU time of one call in us (rep order) and the host time to issue it."""
    for _ in range(warmup):
        for arm in ARMS:
            use_arm(arm)
            flush_l2()
            select(case)
    torch.cuda.synchronize()
    events = {arm: [] for arm in ARMS}
    host = {arm: [] for arm in ARMS}
    for rep in range(reps):
        shift = rep % len(ARMS)
        for arm in ARMS[shift:] + ARMS[:shift]:
            use_arm(arm)
            flush_l2()
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            t0 = time.perf_counter()
            start.record()
            select(case)
            end.record()
            host[arm].append((time.perf_counter() - t0) * 1e6)
            events[arm].append((start, end))
    torch.cuda.synchronize()
    return {
        arm: {
            "us": [s.elapsed_time(e) * 1000.0 for s, e in events[arm]],
            "host_us": host[arm],
        }
        for arm in ARMS
    }


def kernel_names(case, arm):
    use_arm(arm)
    flush_l2()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        select(case)
        torch.cuda.synchronize()
    names = {}
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            names[event.name] = names.get(event.name, 0) + 1
    return names


def check_kernels(arm, names):
    must, must_not = EXPECTED_KERNELS[arm]
    missing = [f for f in must if not any(f in n for n in names)]
    forbidden = [f for f in must_not if any(f in n for n in names)]
    return {
        "ok": not missing and not forbidden,
        "missing": missing,
        "forbidden_found": forbidden,
    }


def summarize(samples):
    ordered = sorted(samples)
    n = len(ordered)
    return {
        "median": statistics.median(ordered),
        "p10": ordered[n // 10],
        "p90": ordered[(9 * n) // 10],
    }


def paired_ratio(numerator, denominator):
    return summarize([a / b for a, b in zip(numerator, denominator, strict=True)])


def differing_rows(case):
    """Rows whose selected token sets differ between today's arm and the new one: a
    tie at the k-th score, or two scores the division by sqrt(head_dim) merged there."""
    use_arm("today")
    today = select(case).sort(dim=1).values
    use_arm("new")
    new = select(case).sort(dim=1).values
    return int((today != new).any(dim=1).sum())


def run_case(name, reps):
    rows, prefix = CASES[name]
    case = make_case(rows, prefix)
    timing = time_arms(case, reps)
    kernels = {arm: kernel_names(case, arm) for arm in ARMS}
    gates = {arm: check_kernels(arm, kernels[arm]) for arm in ARMS}
    us = {arm: timing[arm]["us"] for arm in ARMS}
    return {
        "rows": rows,
        "prefix": prefix,
        "max_row_keys": case.max_row_keys,
        "layer_us": {arm: summarize(us[arm]) for arm in ARMS},
        "host_us_median": {
            arm: statistics.median(timing[arm]["host_us"]) for arm in ARMS
        },
        "new_over_today": paired_ratio(us["new"], us["today"]),
        "new_divided_over_today": paired_ratio(us["new_divided"], us["today"]),
        "new_over_new_divided": paired_ratio(us["new"], us["new_divided"]),
        "aa_new_over_repeat": paired_ratio(us["new"], us["new_repeat"]),
        "rows_with_different_token_sets": differing_rows(case),
        "kernels": kernels,
        "gates": gates,
    }


def print_case(name, result):
    print(
        f"{name}: {result['rows']} rows on a {result['prefix']} prefix, {result['max_row_keys']} keys"
    )
    for arm in ARMS:
        s = result["layer_us"][arm]
        print(
            f"  {arm:11s} {s['median']:9.1f} us  (p10-p90 {s['p10']:.1f}-{s['p90']:.1f})"
        )
    for label, key in (
        ("new / today", "new_over_today"),
        ("new_divided / today", "new_divided_over_today"),
        ("new / new_divided", "new_over_new_divided"),
        ("new / new (A/A)", "aa_new_over_repeat"),
    ):
        r = result[key]
        print(
            f"  {label:19s} {r['median']:.4f}  (p10-p90 {r['p10']:.4f}-{r['p90']:.4f})"
        )
    print(
        f"  rows with different token sets: {result['rows_with_different_token_sets']}"
    )
    for arm, gate in result["gates"].items():
        print(f"  kernels {arm}: {'ok' if gate['ok'] else gate}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reps", type=int, default=31)
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--out", default=None, help="write the results as JSON")
    args = parser.parse_args()
    if not (is_gfx95_supported() and HAS_TILELANG):
        sys.exit("needs a gfx95 GPU and TileLang")

    results = {}
    try:
        for name in args.cases.split(","):
            results[name] = run_case(name, args.reps)
            print_case(name, results[name])
    finally:
        use_arm("new")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1, sort_keys=True)
    failed = [
        f"{c}/{a}"
        for c, r in results.items()
        for a, g in r["gates"].items()
        if not g["ok"]
    ]
    if failed:
        sys.exit(f"kernel gates failed: {failed}")


if __name__ == "__main__":
    main()
