"""Benchmark the decode-size FP8 activation quant fusion of Qwen3.8-Flash-Next (TP1) on gfx950.

Two call sites, each a chain of distinct layers (own inputs and weights) captured in one CUDA graph:
  gdn   gated RMSNorm (48 heads x 128, sigmoid gate) -> out_proj [2560, 6144]; 36 layers, those of a verify step
  attn  output gate (24 heads x 256)                 -> o_proj   [2560, 6144]; 12 layers

Arms (the call site as it is with the gate off, and with it on):
  separate  the producer writes BF16; apply_fp8_linear quantizes it per token, then the GEMM
  fused     the producer writes the per-token (fp8, scale); apply_fp8_linear takes it, then the GEMM
  repeat    separate again: the noise between two identical arms (A/A)

Each timing is CUDA events around one graph replay after a 512 MB write (cold L2), arms interleaved with a
rotating start; the table gives medians of the per-repetition ratios. The kernels of each arm are counted from
the profiler, and the arms' GEMM outputs must be bitwise equal or the run fails.

  python test/manual/kernels/bench_decode_fp8_quant_fuse.py --out bench.json
"""

import argparse
import json
import statistics
import tempfile

import torch
from torch.profiler import ProfilerActivity, profile

from sglang.kernels.ops.attention.fla.layernorm_gated import (
    _layer_norm_fwd,
    quant_heads_num_warps,
    rms_norm_gated,
)
from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul
from sglang.srt.layers.quantization.fp8_utils import (
    apply_fp8_linear,
    use_aiter_bpreshuffle_gemm,
)

HIDDEN = 2560
GDN_HEADS, GDN_DIM = 48, 128
ATTN_HEADS, ATTN_DIM = 24, 256
K = 6144  # GDN_HEADS * GDN_DIM == ATTN_HEADS * ATTN_DIM
LAYERS = {"gdn": 36, "attn": 12}
# verify rows of 12 and 44 requests of 4 tokens; the attention gate also runs on the 12 and 44 draft rows
TOKENS = {"gdn": (48, 176), "attn": (12, 44, 48, 176)}
SWEEP = (1, 4, 16, 22, 256, 384, 512, 1024, 2048)
ARMS = ("separate", "fused", "repeat")
FLUSH_BYTES = 512 * 2**20


def linear_weights(n, k, seed):
    """[K, N] weight and [N, 1] scale as QuarkW8A8Fp8.process_weights_after_loading leaves them."""
    from aiter.ops.shuffle import shuffle_weight

    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(n, k, device="cuda", generator=g) * 0.05).to(torch.float8_e4m3fn)
    s = torch.rand(n, 1, device="cuda", generator=g) * 0.01 + 1e-3
    return (shuffle_weight(w, (16, 16)) if use_aiter_bpreshuffle_gemm(n) else w).t(), s


def linear(x, weights):
    w, s = weights
    if isinstance(x, tuple):
        return apply_fp8_linear(
            x[0], w, s, input_scale=x[1], use_per_token_if_dynamic=True
        )
    return apply_fp8_linear(x, w, s, use_per_token_if_dynamic=True)


def gdn_layers(tokens, count):
    g = torch.Generator(device="cuda").manual_seed(0)

    def rand(*shape, scale):
        return (torch.randn(*shape, device="cuda", generator=g) * scale).bfloat16()

    return [
        (
            rand(tokens * GDN_HEADS, GDN_DIM, scale=0.05),
            rand(tokens * GDN_HEADS, GDN_DIM, scale=2.0),
            rand(GDN_DIM, scale=0.3) + 1,
            linear_weights(HIDDEN, K, i),
        )
        for i in range(count)
    ]


def attn_layers(tokens, count):
    g = torch.Generator(device="cuda").manual_seed(1)

    def rand(*shape, scale):
        return (torch.randn(*shape, device="cuda", generator=g) * scale).bfloat16()

    return [
        (
            rand(tokens, K, scale=0.5),
            rand(tokens, K, scale=2.0),
            linear_weights(HIDDEN, K, 100 + i),
        )
        for i in range(count)
    ]


def gdn_separate(x, z, w, weights):
    norm = rms_norm_gated(
        x=x,
        weight=w,
        bias=None,
        z=z,
        eps=1e-6,
        is_rms_norm=True,
        activation="sigmoid",
    )
    return linear(norm.view(-1, K), weights)


def gdn_fused(x, z, w, weights):
    pair, _, _ = _layer_norm_fwd(
        x,
        w,
        None,
        1e-6,
        z=z,
        is_rms_norm=True,
        activation="sigmoid",
        quant_heads=GDN_HEADS,
    )
    return linear(pair, weights)


def attn_separate(a, gate, weights):
    return linear(fused_sigmoid_mul(a, gate), weights)


def attn_fused(a, gate, weights):
    return linear(fused_sigmoid_mul(a, gate, quant=True), weights)


SITES = {
    "gdn": (gdn_layers, gdn_separate, gdn_fused),
    "attn": (attn_layers, attn_separate, attn_fused),
}

_flush = None


def flush():
    global _flush
    if _flush is None:
        _flush = torch.empty(FLUSH_BYTES, dtype=torch.uint8, device="cuda")
    _flush.fill_(1)


def capture(fn, layers):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn(*layers[0])
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = [fn(*layer) for layer in layers]
    return graph, outs


def kernel_counts(graph, replays=3):
    """Launches per kernel name in one replay. The profiler can drop events but not invent them, so the most
    seen over a few profiled replays."""
    counts = {}
    for _ in range(replays):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            graph.replay()
            torch.cuda.synchronize()
        seen = {}
        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            prof.export_chrome_trace(f.name)
            with open(f.name) as trace:
                events = json.load(trace)["traceEvents"]
        for e in events:
            if e.get("cat") == "kernel":
                seen[e["name"]] = seen.get(e["name"], 0) + 1
        counts = {
            k: max(counts.get(k, 0), seen.get(k, 0))
            for k in counts.keys() | seen.keys()
        }
    return counts


def bench_site(site, tokens, reps, warmup):
    make_layers, separate, fused = SITES[site]
    count = LAYERS[site]
    layers = make_layers(tokens, count)
    arm_fns = {"separate": separate, "fused": fused, "repeat": separate}
    graphs, outs = {}, {}
    for arm, fn in arm_fns.items():
        graphs[arm], outs[arm] = capture(fn, layers)
    for g in graphs.values():
        g.replay()
    torch.cuda.synchronize()
    for arm in ("fused", "repeat"):
        assert all(torch.equal(a, b) for a, b in zip(outs["separate"], outs[arm])), (
            f"{site} t={tokens}: the {arm} arm's GEMM output differs from the separate arm's"
        )
    times = {arm: [] for arm in ARMS}
    for rep in range(warmup + reps):
        for arm in ARMS[rep % len(ARMS) :] + ARMS[: rep % len(ARMS)]:
            flush()
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            graphs[arm].replay()
            end.record()
            torch.cuda.synchronize()
            if rep >= warmup:
                times[arm].append(start.elapsed_time(end) * 1000.0)
    ratio = lambda a, b: statistics.median(x / y for x, y in zip(times[a], times[b]))
    kernels = {arm: kernel_counts(graphs[arm]) for arm in ("separate", "fused")}
    quants = {
        arm: sum(n for k, n in c.items() if "per_token_group_quant" in k)
        for arm, c in kernels.items()
    }
    return {
        "layers": count,
        "us_per_call": {arm: statistics.median(t) / count for arm, t in times.items()},
        "fused_over_separate": ratio("fused", "separate"),
        "repeat_over_separate": ratio("repeat", "separate"),
        "quant_launches_per_replay": quants,
        "kernels_per_replay": kernels,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--sweep", action="store_true", help="also time more token counts"
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    results = {}
    for site, tokens in TOKENS.items():
        for t in tokens + (SWEEP if args.sweep else ()):
            if (
                site == "gdn"
                and quant_heads_num_warps(t * GDN_HEADS, GDN_DIM, GDN_HEADS, device)
                is None
            ):
                print(
                    f"{site} t={t}: below the floor, the norm and the quant stay two launches"
                )
                continue
            r = results[f"{site}/{t}"] = bench_site(site, t, args.reps, args.warmup)
            us = r["us_per_call"]
            print(
                f"{site:5s} t={t:5d}  separate {us['separate']:7.2f}  fused {us['fused']:7.2f} us/call"
                f"  fused/separate {r['fused_over_separate']:.4f}  A/A {r['repeat_over_separate']:.4f}"
                f"  quant launches per replay {r['quant_launches_per_replay']}",
                flush=True,
            )
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
