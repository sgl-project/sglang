"""QSA indexer block selection on gfx95, today's against the new one: prefill and decode.

Every arm calls ``QSAIndexer.select_prefill_tokens`` or ``select_decode_tokens`` itself on
the same inputs. The "today" arm runs it with ``qsa_indexer.is_gfx95_supported`` returning
False, which takes the branch every other platform takes (fast_topk); the new arm takes
the gfx95 branch (topk_v2). Nothing else differs, except in the arms below.

Prefill: each input is one request of Gaussian, rms-normed Q and K at the serving
geometry (4 index heads, head_dim 128, compress ratio 4), chunked as the indexer chunks
it. Today's arm selects on scores divided by sqrt(head_dim), the new arm on unscaled ones
(topk_v2 reads only their order); the "new_divided" arm takes the gfx95 branch with the
scores still divided, which splits the difference between the top-k and the division
pass. Timing is one call of the whole function (all row chunks: one layer).

Decode: each input is the draft rows of a target-verify step, 4 rows a request, over
random request contexts, padded with length-0 requests to the batch size a CUDA graph
was captured for. The scorer runs once per input into a static buffer; each arm is
``select_decode_tokens`` over it, captured in a CUDA graph. The "new_ragged" arm runs
the prefill's ragged topk_v2 entry point on the same rows instead of the paged one.
Timing is one call, one layer of a verify step (or the draft-extend step); a decode step
is STEP_CALLS calls.

Timing: a 512 MB write before every timed call (cold L2), CUDA events around the call
(decode: the scoring graph runs, untimed, before it, so the selection reads the scores
the scorer just wrote), the arms interleaved with the starting arm rotating every rep,
and an A/A control (the new arm twice). The profiler names the kernels of one call per
arm; each arm must launch its own top-k kernel and no other arm's, and the prefill's
today arm the division's multiply, the new arm not; the script exits 1 otherwise.

  python bench_qsa_select.py [--reps 31] [--cases served,decode_r60] [--out result.json]
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

PAGE_SIZE = 16
VERIFY_ROWS = 4
# A decode graph's page table: a 262,144-token context limit, compress ratio 4, 16-key pages.
TABLE_PAGES = 4096
MAX_MODEL_LEN = TABLE_PAGES * PAGE_SIZE
# A decode step selects once for each of Qwen3.8-Flash-Next's 12 full-attention layers in
# the verify pass, and once for its MTP layer in the draft-extend pass.
STEP_CALLS = 13
# name -> (requests, requests in the CUDA graph, lowest and highest context in tokens)
DECODE_CASES = {
    "decode_r2": (2, 2, 70_000, 130_000),
    "decode_r7": (7, 7, 4_000, 130_000),
    "decode_r45": (45, 48, 4_000, 252_000),
    "decode_r60": (60, 60, 4_000, 252_000),
    # every row fits k keys: both arms take their kernel's trivial path
    "decode_short_r60": (60, 60, 400, 2_000),
}
# "a|b" asks for a kernel with either fragment: topk_v2's paged entry point runs
# topk_main_kernel, or the split pair when few rows are wider than its split floor.
PAGED_V2 = "topk_main_kernel|topk_split_select"
EXPECTED_DECODE_KERNELS = {
    "today": (("fast_topk_kernel",), (PAGED_V2, "topk_ragged_kernel")),
    "new": ((PAGED_V2,), ("fast_topk", "topk_ragged_kernel")),
    "new_repeat": ((PAGED_V2,), ("fast_topk", "topk_ragged_kernel")),
    "new_ragged": (("topk_ragged_kernel",), ("fast_topk", PAGED_V2)),
}
DECODE_ARMS = list(EXPECTED_DECODE_KERNELS)

_original_gate = qsa_indexer.is_gfx95_supported
_original_scoring = qsa_indexer.qsa_mqa_prefill_relative
_original_decode_scoring = qsa_indexer.qsa_mqa_decode
_original_decode_topk = qsa_indexer._exact_decode_block_topk
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


def time_arms(arms, reps, prepare, call, warmup=3):
    """Per arm: GPU time of call(arm) in us (rep order) and the host time to issue it.
    prepare(arm) runs, untimed, before each call."""
    for _ in range(warmup):
        for arm in arms:
            prepare(arm)
            call(arm)
    torch.cuda.synchronize()
    events = {arm: [] for arm in arms}
    host = {arm: [] for arm in arms}
    for rep in range(reps):
        shift = rep % len(arms)
        for arm in arms[shift:] + arms[:shift]:
            prepare(arm)
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            t0 = time.perf_counter()
            start.record()
            call(arm)
            end.record()
            host[arm].append((time.perf_counter() - t0) * 1e6)
            events[arm].append((start, end))
    torch.cuda.synchronize()
    return {
        arm: {
            "us": [s.elapsed_time(e) * 1000.0 for s, e in events[arm]],
            "host_us": host[arm],
        }
        for arm in arms
    }


def kernel_names(call):
    """The device kernels one call() launches, by name, with their counts."""
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        call()
        torch.cuda.synchronize()
    names = {}
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            names[event.name] = names.get(event.name, 0) + 1
    return names


def check_kernels(expected, arm, names):
    def launched(fragment):
        return any(part in n for part in fragment.split("|") for n in names)

    must, must_not = expected[arm]
    missing = [f for f in must if not launched(f)]
    forbidden = [f for f in must_not if launched(f)]
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

    def prepare(arm):
        use_arm(arm)
        flush_l2()

    timing = time_arms(ARMS, reps, prepare, lambda arm: select(case))
    kernels = {}
    for arm in ARMS:
        prepare(arm)
        kernels[arm] = kernel_names(lambda: select(case))
    gates = {arm: check_kernels(EXPECTED_KERNELS, arm, kernels[arm]) for arm in ARMS}
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


def capture(call):
    """call() captured as a CUDA graph after two eager runs, as SGLang's graph runners
    do (they pay one-time setup before the capture); returns the graph and the tensor
    call() returned."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
        call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    return graph, output


def make_decode_case(requests, graph_requests, low, high, seed=11):
    """The draft rows of a target-verify step: request r's VERIFY_ROWS rows see its
    random context plus 1..VERIFY_ROWS tokens, back to back over the request's
    page-table row; the requests the graph pads to follow, with length 0."""
    generator = torch.Generator().manual_seed(seed)
    contexts = torch.randint(low, high + 1, (requests,), generator=generator).tolist()
    lengths, first_pages, num_pages, next_page = [], [], [], 1
    for context in contexts:
        row_lengths = [(context + j + 1) // RATIO for j in range(VERIFY_ROWS)]
        assert max(row_lengths) <= MAX_MODEL_LEN, context
        pages = -(-max(row_lengths) // PAGE_SIZE)
        lengths += row_lengths
        first_pages += [next_page] * VERIFY_ROWS
        num_pages += [pages] * VERIFY_ROWS
        next_page += pages
    rows = VERIFY_ROWS * graph_requests
    page_table = torch.zeros(rows, TABLE_PAGES, dtype=torch.int32)
    for row, (first, count) in enumerate(zip(first_pages, num_pages, strict=True)):
        page_table[row, :count] = torch.arange(first, first + count, dtype=torch.int32)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.zeros(rows, NUM_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    q[: len(lengths)] = rms_normed(
        torch.randn(len(lengths), NUM_Q_HEADS, HEAD_DIM, generator=gen, device="cuda"),
        gen,
    )
    k_cache = rms_normed(
        torch.randn(next_page, PAGE_SIZE, 1, HEAD_DIM, generator=gen, device="cuda"),
        gen,
    )
    return SimpleNamespace(
        q=q,
        k_cache=k_cache,
        page_table=page_table.cuda(),
        lengths=torch.tensor(
            lengths + [0] * (rows - len(lengths)), dtype=torch.int32, device="cuda"
        ),
        contexts=contexts,
    )


def use_decode_arm(arm):
    qsa_indexer.is_gfx95_supported = (
        (lambda: False) if arm == "today" else _original_gate
    )
    qsa_indexer._exact_decode_block_topk = (
        qsa_indexer._exact_block_topk if arm == "new_ragged" else _original_decode_topk
    )


def restore_decode_patches():
    qsa_indexer.is_gfx95_supported = _original_gate
    qsa_indexer.qsa_mqa_decode = _original_decode_scoring
    qsa_indexer._exact_decode_block_topk = _original_decode_topk


def select_decode(case):
    indexer = SimpleNamespace(
        token_topk=TOKEN_TOPK, compress_ratio=RATIO, block_topk=BLOCK_TOPK
    )
    return QSAIndexer.select_decode_tokens(
        indexer,
        case.q,
        case.k_cache,
        case.page_table,
        case.lengths,
        MAX_MODEL_LEN,
        case.lengths,
        case.lengths,
        defer_expansion=True,
    )


def rows_not_top_k(selected, logits, lengths):
    """Rows of more than BLOCK_TOPK blocks whose selection is not BLOCK_TOPK distinct
    blocks below the row's length holding a top BLOCK_TOPK of its scores (a tie at the
    cut may go either way), against torch.topk."""
    bad = 0
    for row in (lengths > BLOCK_TOPK).nonzero().flatten().tolist():
        length = int(lengths[row])
        blocks = selected[row].long()
        if (
            bool((blocks < 0).any() or (blocks >= length).any())
            or blocks.unique().numel() != BLOCK_TOPK
        ):
            bad += 1
            continue
        expected = torch.topk(logits[row, :length], BLOCK_TOPK).values.sort().values
        bad += not torch.equal(logits[row, blocks].sort().values, expected)
    return bad


def run_decode_case(name, reps):
    requests, graph_requests, low, high = DECODE_CASES[name]
    case = make_decode_case(requests, graph_requests, low, high)
    scoring, logits = capture(
        lambda: _original_decode_scoring(
            case.q,
            case.k_cache,
            case.page_table,
            case.lengths,
            MAX_MODEL_LEN,
            min_scored_len=BLOCK_TOPK,
        )
    )
    scoring.replay()
    # The arms select over the scorer's static output.
    qsa_indexer.qsa_mqa_decode = lambda *args, **kwargs: logits
    graphs = {}
    for arm in DECODE_ARMS:
        use_decode_arm(arm)
        graphs[arm] = capture(lambda: select_decode(case))

    def prepare(arm):
        flush_l2()
        scoring.replay()

    timing = time_arms(DECODE_ARMS, reps, prepare, lambda arm: graphs[arm][0].replay())
    kernels = {}
    for arm in DECODE_ARMS:
        use_decode_arm(arm)
        prepare(arm)
        kernels[arm] = kernel_names(lambda: select_decode(case))
    gates = {
        arm: check_kernels(EXPECTED_DECODE_KERNELS, arm, kernels[arm])
        for arm in DECODE_ARMS
    }
    prepare("today")
    for graph, _ in graphs.values():
        graph.replay()
    torch.cuda.synchronize()
    chosen = {arm: output.clone() for arm, (_, output) in graphs.items()}
    us = {arm: timing[arm]["us"] for arm in DECODE_ARMS}
    restore_decode_patches()
    return {
        "requests": requests,
        "graph_requests": graph_requests,
        "rows": case.lengths.numel(),
        "rows_longer_than_k": int((case.lengths > BLOCK_TOPK).sum()),
        "max_context": max(case.contexts),
        "mean_context": round(statistics.mean(case.contexts)),
        "call_us": {arm: summarize(us[arm]) for arm in DECODE_ARMS},
        "new_over_today": paired_ratio(us["new"], us["today"]),
        "new_ragged_over_new": paired_ratio(us["new_ragged"], us["new"]),
        "aa_new_over_repeat": paired_ratio(us["new"], us["new_repeat"]),
        "step_saving_ms": summarize(
            [
                STEP_CALLS * (today - new) / 1000.0
                for today, new in zip(us["today"], us["new"], strict=True)
            ]
        ),
        "rows_not_top_k": {
            arm: rows_not_top_k(chosen[arm], logits, case.lengths)
            for arm in ("today", "new")
        },
        "rows_with_different_block_sets": int(
            (chosen["today"].sort(dim=1).values != chosen["new"].sort(dim=1).values)
            .any(dim=1)
            .sum()
        ),
        "kernels": kernels,
        "gates": gates,
    }


def print_decode_case(name, result):
    print(
        f"{name}: {result['requests']} requests ({result['rows']} rows in the graph, "
        f"{result['rows_longer_than_k']} longer than k), mean context "
        f"{result['mean_context']}, max {result['max_context']} tokens"
    )
    for arm in DECODE_ARMS:
        s = result["call_us"][arm]
        print(
            f"  {arm:11s} {s['median']:9.1f} us  (p10-p90 {s['p10']:.1f}-{s['p90']:.1f})"
        )
    for label, key in (
        ("new / today", "new_over_today"),
        ("new_ragged / new", "new_ragged_over_new"),
        ("new / new (A/A)", "aa_new_over_repeat"),
    ):
        r = result[key]
        print(
            f"  {label:19s} {r['median']:.4f}  (p10-p90 {r['p10']:.4f}-{r['p90']:.4f})"
        )
    s = result["step_saving_ms"]
    print(
        f"  saving per decode step ({STEP_CALLS} calls): {s['median']:.3f} ms "
        f"(p10-p90 {s['p10']:.3f}-{s['p90']:.3f})"
    )
    print(f"  rows not a top-k: {result['rows_not_top_k']}")
    print(
        f"  rows with different block sets: {result['rows_with_different_block_sets']}"
    )
    for arm, gate in result["gates"].items():
        print(f"  kernels {arm}: {'ok' if gate['ok'] else gate}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reps", type=int, default=31)
    parser.add_argument("--cases", default=",".join([*CASES, *DECODE_CASES]))
    parser.add_argument("--out", default=None, help="write the results as JSON")
    args = parser.parse_args()
    if not (is_gfx95_supported() and HAS_TILELANG):
        sys.exit("needs a gfx95 GPU and TileLang")

    results = {}
    try:
        for name in args.cases.split(","):
            if name in DECODE_CASES:
                results[name] = run_decode_case(name, args.reps)
                print_decode_case(name, results[name])
            else:
                results[name] = run_case(name, args.reps)
                print_case(name, results[name])
    finally:
        use_arm("new")
        restore_decode_patches()
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
