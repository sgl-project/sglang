"""Per-boundary time of the qwen4_exp HC boundary: stock GatedResidual (combine, then norm + mix) vs the fused seam.

Each arm runs BOUNDARIES boundaries with distinct weights inside one CUDA graph; the median replay is divided by
BOUNDARIES. inter = hc_gated_combine + mix-only seam (EP1: every boundary); intra = combine fused into the seam.

    python benchmark/kernels/bench_hc_seam.py --rows 16,32,64,128,193
"""

import argparse
import statistics

import torch

import sglang.srt.layers.hyperconnection as hcm
from sglang.kernels.ops.gemm.hc_seam import HC, HS, LOWRANK, hc_gated_combine, hc_seam
from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig

D = HC * HS
EPS = 1e-6


def _module(gen):
    cfg = HyperConnectionConfig(
        hc_count=HC,
        hidden_size=HS,
        params_dtype=torch.bfloat16,
        hc_lowrank=LOWRANK,
        rms_norm_eps=EPS,
        hc_per_branch_norm=True,
    )
    m = GatedResidual(cfg).cuda()
    with torch.no_grad():
        for p, std in (
            (m.hc_norm.weight, 0.1),
            (m.input_mix_weight_down.weight, 0.04),
            (m.input_mix_weight_up.weight, 0.1),
            (m.block_inject_weight.weight, 0.04),
        ):
            p.copy_(torch.randn(p.shape, generator=gen, device="cuda") * std)
    return m


def _graph_us(fn, replays):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(20):
        g.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(replays):
        a, b = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1e3)
    return statistics.median(times)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="16,32,64,128,193")
    ap.add_argument("--boundaries", type=int, default=24)
    ap.add_argument("--replays", type=int, default=200)
    args = ap.parse_args()
    rows_list = [int(x) for x in args.rows.split(",")]
    hcm._HC_SEAM = False  # the stock arm, whatever SGLANG_ENABLE_QWEN4_HC_SEAM says
    gen = torch.Generator(device="cuda").manual_seed(0)
    torch.set_default_dtype(torch.bfloat16)  # as the model loader builds the layers
    mods = [(_module(gen), _module(gen)) for _ in range(args.boundaries)]
    torch.set_default_dtype(torch.float32)

    def acts(rows):
        y = torch.randn(rows, HS, generator=gen, device="cuda").to(torch.bfloat16)
        r = (torch.randn(rows, D, generator=gen, device="cuda") * 3).to(torch.bfloat16)
        return y, r

    # Compile the stock mix as a server meets widths: a prefill size, then decode sizes large to small.
    for rows in [512] + sorted(rows_list, reverse=True):
        y, r = acts(rows)
        prev, cur = mods[0]
        cur.mix(prev.combine(y, prev.mix(r)[1]))

    print(
        f"{'rows':>5} {'stock us':>9} {'inter us':>9} {'intra us':>9} {'inter/stock':>12}"
    )
    for rows in rows_list:
        inputs = []
        for prev, cur in mods:
            y, r = acts(rows)
            _, res = prev.mix(r)
            gate = 2.0 * torch.sigmoid(
                res[1].float() @ prev.block_inject_weight.weight.float().t() / HC
            )
            inputs.append((y, res, gate.contiguous()))
        weights = [
            (
                cur.hc_norm.weight,
                cur.input_mix_weight_down.weight,
                cur.input_mix_weight_up.weight,
                cur.block_inject_weight.weight,
                EPS,
            )
            for _, cur in mods
        ]

        def stock():
            for (prev, cur), (y, res, _) in zip(mods, inputs):
                cur.mix(prev.combine(y, res))

        def inter():
            for w, (y, res, gate) in zip(weights, inputs):
                hc_seam(None, hc_gated_combine(y, res[0], gate), None, *w)

        def intra():
            for w, (y, res, gate) in zip(weights, inputs):
                hc_seam(y, res[0], gate, *w)

        t = [
            _graph_us(f, args.replays) / args.boundaries for f in (stock, inter, intra)
        ]
        print(f"{rows:>5} {t[0]:>9.2f} {t[1]:>9.2f} {t[2]:>9.2f} {t[1] / t[0]:>12.3f}")


if __name__ == "__main__":
    main()
