#!/usr/bin/env python3
"""Config sweep for fused_kda_conv_gating_verify_kernel at a given T.

The production config (BV=32, num_warps=4, num_stages=3) was tuned at T=4;
this sweeps BV x num_warps x num_stages at the current draft depth, with the
per-layer buffers rotated across NLAYER distinct instances (the real model
streams a different layer's state every call, so nothing stays hot in L2)
and all launches captured in one CUDA graph (in-graph timing is what the
decode step actually sees).

BV=128 doubles as the fusion-feasibility probe: one program then owns the
whole V row, which is the precondition for folding the gated RMSNorm into
the kernel epilogue. If BV=128 compiles without spilling and lands near the
front of the table, the norm fusion is worth building.

Correctness: every config's `o` and rolled conv_state are compared against
the production config's output on identical inputs. Reduction order changes
with num_warps, so ulp-level diffs are expected; anything > 1e-2 max-abs is
flagged FAIL.

Usage (per-TP4-rank shapes; read H/HV from the model config and divide by 4):

    python3 bench_kda_verify_sweep.py --H 4 --HV 8 --K 128 --V 128 --T 5
"""

import argparse
import itertools
import json

import torch
import triton

from sglang.kernels.ops.attention.fla.fused_kda_conv_recurrent_verify import (
    fused_kda_conv_gating_verify_kernel,
)

NLAYER = 35
WARMUP = 10
ITERS = 200


def build_layer(args, device, dtype, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    H, HV, K, V, T = args.H, args.HV, args.K, args.V, args.T
    dim = 2 * H * K + HV * V

    def r(*shape, dt=dtype):
        return torch.randn(*shape, generator=g, device=device, dtype=torch.float32).to(
            dt
        )

    layer = dict(
        mixed_qkv=r(T, dim) * 0.5,
        conv_weight=r(dim, 4) * 0.3,
        # [lines, dim, state_len] with dim-stride 1 (the kernel hardcodes the
        # dim stride to 1): a [1, 3, dim] contiguous buffer transposed. Do NOT
        # call .contiguous() after the transpose -- that rebuilds dim-stride 3
        # and scrambles the (channel, col) addressing into write collisions.
        conv_state=(r(1, 3, dim) * 0.5).transpose(1, 2),
        conv_state_indices=torch.zeros(1, dtype=torch.int32, device=device),
        inter_window=torch.zeros(1, T, dim, 3, device=device, dtype=dtype),
        inter_indices=torch.zeros(1, dtype=torch.int32, device=device),
        a=r(T, HV * K) * 0.5,
        b=r(T, HV) * 0.5,
        A_log=r(HV, dt=torch.float32) * 0.1,
        dt_bias=r(HV * K, dt=torch.float32) * 0.1,
        ssm=r(1, HV, V, K, dt=torch.float32) * 0.2,
        h0_idx=torch.zeros(1, dtype=torch.int32, device=device),
        inter_states=torch.zeros(
            1, T, HV, V, K, device=device, dtype=torch.float32
        ),
        o=torch.empty(T, HV, V, device=device, dtype=dtype),
    )
    return layer


def launch(layer, args, BV, num_warps, num_stages):
    H, HV, K, V, T = args.H, args.HV, args.K, args.V, args.T
    NV = triton.cdiv(V, BV)
    grid = (NV, 1 * HV)
    fused_kda_conv_gating_verify_kernel[grid](
        x=layer["mixed_qkv"],
        w=layer["conv_weight"],
        conv_bias=layer["conv_weight"],  # HAS_BIAS=False -> unused
        conv_state=layer["conv_state"],
        conv_state_indices=layer["conv_state_indices"],
        inter_conv_window=layer["inter_window"],
        inter_state_indices=layer["inter_indices"],
        a=layer["a"],
        b_gate=layer["b"],
        A_log=layer["A_log"],
        dt_bias=layer["dt_bias"],
        lower_bound=None,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        h0_source=layer["ssm"],
        h0_indices=layer["h0_idx"],
        inter_states=layer["inter_states"],
        o=layer["o"],
        scale=K**-0.5,
        cache_steps=T,
        stride_x_tok=layer["mixed_qkv"].stride(0),
        stride_w_dim=layer["conv_weight"].stride(0),
        stride_cs_line=layer["conv_state"].stride(0),
        stride_cs_tok=layer["conv_state"].stride(2),
        stride_iw_line=layer["inter_window"].stride(0),
        stride_iw_step=layer["inter_window"].stride(1),
        stride_iw_dim=layer["inter_window"].stride(2),
        stride_iw_win=layer["inter_window"].stride(3),
        stride_a_tok=layer["a"].stride(0),
        stride_b_tok=layer["b"].stride(0),
        T=T,
        W=4,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=K,
        BV=BV,
        HAS_BIAS=False,
        USE_QK_L2NORM_IN_KERNEL=True,
        USE_LOWER_BOUND=False,
        SAVE_INTERMEDIATE_WINDOW=True,
        CACHE_INTERMEDIATE_STATES=True,
        USE_GDC=False,  # clean per-kernel timing; production keeps PDL
        num_warps=num_warps,
        num_stages=num_stages,
    )


def snapshot(layers):
    return [
        (l["o"].clone(), l["conv_state"].clone(), l["inter_states"].clone())
        for l in layers
    ]


def restore_states(layers, saved_cs):
    for l, cs in zip(layers, saved_cs):
        l["conv_state"].copy_(cs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--H", type=int, required=True, help="q heads per rank")
    p.add_argument("--HV", type=int, required=True, help="v heads per rank")
    p.add_argument("--K", type=int, default=128)
    p.add_argument("--V", type=int, default=128)
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    torch.cuda.set_device(0)

    layers = [build_layer(args, device, dtype, seed=1000 + i) for i in range(NLAYER)]
    # conv_state is mutated by the kernel (rolled window); keep pristine copies
    # so every config starts from identical inputs.
    pristine_cs = [l["conv_state"].clone() for l in layers]

    REF = (32, 4, 3)
    sweep = [
        (bv, w, s)
        for bv, w, s in itertools.product([2, 4, 8, 16, 32, 64, 128], [2, 4, 8], [1, 2, 3])
        if not (bv == 128 and w == 2)  # hopeless register pressure, skip
    ]
    if REF not in sweep:
        sweep.insert(0, REF)
    # reference first so its outputs are captured for comparison
    sweep.sort(key=lambda c: c != REF)

    ref_out = None
    results = []
    for BV, w, s in sweep:
        restore_states(layers, pristine_cs)
        try:
            # eager warmup (JIT) + correctness pass
            for l in layers:
                launch(l, args, BV, w, s)
            torch.cuda.synchronize()
        except Exception as e:
            results.append(dict(BV=BV, warps=w, stages=s, status=f"COMPILE_FAIL {type(e).__name__}"))
            continue

        # correctness snapshot from a fresh state
        restore_states(layers, pristine_cs)
        for l in layers:
            launch(l, args, BV, w, s)
        torch.cuda.synchronize()
        outs = snapshot(layers)
        if (BV, w, s) == REF:
            ref_out = outs
            max_diff = 0.0
            bit_equal = True
        else:
            max_diff = 0.0
            bit_equal = True
            for (o, cs, ist), (ro, rcs, rist) in zip(outs, ref_out):
                max_diff = max(
                    max_diff,
                    (o.float() - ro.float()).abs().max().item(),
                    (cs.float() - rcs.float()).abs().max().item(),
                )
                if not (torch.equal(o, ro) and torch.equal(cs, rcs)):
                    bit_equal = False

        # in-graph timing: capture all 35 rotated launches into one graph
        restore_states(layers, pristine_cs)
        for _ in range(2):  # graph warmup on side stream happens inside capture
            for l in layers:
                launch(l, args, BV, w, s)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for l in layers:
                launch(l, args, BV, w, s)
        for _ in range(WARMUP):
            graph.replay()
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(ITERS):
            graph.replay()
        t1.record()
        torch.cuda.synchronize()
        us_per_layer = t0.elapsed_time(t1) * 1000.0 / ITERS / NLAYER
        status = "ok" if max_diff <= 1e-2 else "FAIL_NUMERICS"
        results.append(
            dict(
                BV=BV,
                warps=w,
                stages=s,
                us_per_layer=round(us_per_layer, 3),
                max_diff=float(max_diff),
                bit_equal_ref=bit_equal,
                status=status,
            )
        )

    results.sort(key=lambda r: r.get("us_per_layer", 1e9))
    if args.json:
        print(json.dumps(results, indent=1))
    else:
        print(f"shapes: H={args.H} HV={args.HV} K={args.K} V={args.V} T={args.T}, "
              f"{NLAYER} rotated layers, in-graph, ref={REF}")
        print(f"{'BV':>4} {'warps':>5} {'stages':>6} {'us/layer':>9} "
              f"{'max_diff':>10} {'bit==ref':>8}  status")
        for r in results:
            print(f"{r['BV']:>4} {r['warps']:>5} {r['stages']:>6} "
                  f"{r.get('us_per_layer', float('nan')):>9} "
                  f"{r.get('max_diff', float('nan')):>10.2e} "
                  f"{str(r.get('bit_equal_ref', '-')):>8}  {r['status']}")


if __name__ == "__main__":
    main()
