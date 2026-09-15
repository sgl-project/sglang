"""Sweep benchmark for the KDA chain-verify kernels (one layer, in-graph).

Compares the four target-verify variants the KDA backend can dispatch:

    unfused        causal_conv1d_update + recurrence, per-step ssm snapshots
    unfused+ring   causal_conv1d_update + recurrence, ReplaySSM CACHE_RING
    fused          fused_kda_conv_gating_verify, per-step ssm snapshots
    fused+ring     fused_kda_conv_gating_verify, ReplaySSM CACHE_RING

Timing replays a CUDA graph capturing GRAPH_BATCH calls, matching how the
production verify runs (in-graph; bare launches would drown these ~10us
kernels in launch overhead). Imports only sglang.kernels.*, so it runs on
boxes where the sglang.srt/test import chain is broken.

    PYTHONPATH=python python3 benchmark/kernels/bench_kda_verify_sweep.py
    ... --batch-sizes 1 4 16 64 --modes fused fused+ring
    ... --sweep-bv          # re-tune KDA_VERIFY_BLOCK_V per mode/batch
    ... --hv-heads 16       # GQA shape (HV != H)
"""

import argparse

import torch

import sglang.kernels.ops.attention.fla.fused_kda_conv_recurrent_verify as fused_mod
from sglang.kernels.ops.attention.fla.fused_kda_conv_recurrent_verify import (
    fused_kda_conv_gating_verify,
)
from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update

_DEVICE = "cuda"
_DTYPE = torch.bfloat16
_W = 4
# Ring length: power of two >= 2 * draft tokens (memory_pool.py invariant).
_RING_LEN = 16
_MODES = ("unfused", "unfused+ring", "fused", "fused+ring")
GRAPH_BATCH = 10


def make_inputs(B, T, H, HV, K, V, seed=0):
    torch.manual_seed(seed)
    dim = 2 * H * K + HV * V
    seq_len = B * T
    lines = slots = B + 1
    rnd = lambda *s, dt=_DTYPE: torch.randn(*s, device=_DEVICE, dtype=dt)
    return {
        "mixed": rnd(seq_len, dim) * 0.5,
        "w": rnd(dim, _W) * 0.3,
        "bias": rnd(dim) * 0.1,
        "a": rnd(seq_len, HV * K) * 0.5,
        "b": rnd(seq_len, HV),
        "A_log": rnd(HV, dt=torch.float32) * 0.5,
        "dt_bias": rnd(HV * K, dt=torch.float32) * 0.5,
        "conv_pool": rnd(lines, _W - 1, dim),
        "ssm": rnd(slots, HV, V, K, dt=torch.float32) * 0.2,
        "win_pool": torch.zeros(lines, T, _W - 1, dim, device=_DEVICE, dtype=_DTYPE),
        "inter_ssm": torch.zeros(
            lines, T, HV, V, K, device=_DEVICE, dtype=torch.float32
        ),
        "rawv": rnd(slots, HV, _RING_LEN, V),
        "rawk": rnd(slots, H, _RING_LEN, K),
        "g": rnd(slots, HV, _RING_LEN, K, dt=torch.float32),
        "beta": rnd(slots, HV, _RING_LEN, dt=torch.float32),
        "cache_indices": torch.arange(B, device=_DEVICE, dtype=torch.int32),
        "inter_indices": torch.arange(B, device=_DEVICE, dtype=torch.int32),
        "cu": torch.arange(0, B + 1, device=_DEVICE, dtype=torch.int32) * T,
    }


def _ring_kwargs(inp, on):
    return dict(
        cache_ring=on,
        replayssm_rawv=inp["rawv"] if on else None,
        replayssm_rawk=inp["rawk"] if on else None,
        replayssm_g=inp["g"] if on else None,
        replayssm_beta=inp["beta"] if on else None,
    )


def make_runner(mode, inp, B, T, H, HV, K, V, lower_bound=None):
    dim = 2 * H * K + HV * V
    seq_len = B * T
    ring = mode.endswith("+ring")
    scale = K**-0.5

    if mode.startswith("fused"):

        def fn():
            fused_kda_conv_gating_verify(
                mixed_qkv=inp["mixed"],
                conv_weight=inp["w"],
                conv_bias=inp["bias"],
                conv_state=inp["conv_pool"].transpose(-1, -2),
                conv_state_indices=inp["cache_indices"],
                intermediate_conv_window=inp["win_pool"].transpose(-1, -2),
                intermediate_state_indices=inp["inter_indices"],
                a=inp["a"],
                b=inp["b"],
                A_log=inp["A_log"],
                dt_bias=inp["dt_bias"],
                ssm_states=inp["ssm"],
                cache_indices=inp["cache_indices"],
                intermediate_states_buffer=None if ring else inp["inter_ssm"],
                scale=scale,
                T=T,
                num_q_heads=H,
                num_v_heads=HV,
                head_k_dim=K,
                head_v_dim=V,
                lower_bound=lower_bound,
                **_ring_kwargs(inp, ring),
            )

        return fn

    def fn():
        x3 = inp["mixed"].reshape(B, T, dim).transpose(1, 2)
        out3 = causal_conv1d_update(
            x3,
            inp["conv_pool"].transpose(-1, -2),
            inp["w"],
            inp["bias"],
            activation="silu",
            conv_state_indices=inp["cache_indices"],
            intermediate_conv_window=inp["win_pool"].transpose(-1, -2),
            intermediate_state_indices=inp["inter_indices"],
        )
        mixed_out = out3.transpose(1, 2).reshape(seq_len, dim)
        q, k, v = mixed_out.split([H * K, H * K, HV * V], dim=-1)
        fused_sigmoid_gating_delta_rule_update(
            A_log=inp["A_log"],
            a=inp["a"],
            dt_bias=inp["dt_bias"],
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q.unflatten(-1, (H, K)).unsqueeze(0),
            k=k.unflatten(-1, (H, K)).unsqueeze(0),
            v=v.unflatten(-1, (HV, V)).unsqueeze(0),
            b=inp["b"],
            initial_state_source=inp["ssm"],
            initial_state_indices=inp["cache_indices"],
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=inp["cu"],
            is_kda=True,
            disable_state_update=True,
            intermediate_states_buffer=None if ring else inp["inter_ssm"],
            intermediate_state_indices=None if ring else inp["inter_indices"],
            cache_steps=T,
            retrieve_parent_token=None,
            lower_bound=lower_bound,
            **_ring_kwargs(inp, ring),
        )

    return fn


def bench_graph(fn, iters=200):
    """us per call, timed as CUDA-graph replays of GRAPH_BATCH captured calls."""
    for _ in range(3):  # compile outside capture
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(GRAPH_BATCH):
            fn()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    graph.reset()
    return start.elapsed_time(end) * 1e3 / (iters * GRAPH_BATCH)


def check_ring_bitwise(B, T, H, HV, K, V, lower_bound=None):
    """One-shot guard: fused+ring must fill the same ring bytes as unfused+ring."""
    ref, fus = (make_inputs(B, T, H, HV, K, V, seed=7) for _ in range(2))
    make_runner("unfused+ring", ref, B, T, H, HV, K, V, lower_bound)()
    make_runner("fused+ring", fus, B, T, H, HV, K, V, lower_bound)()
    torch.cuda.synchronize()
    for name in ("rawv", "rawk", "g", "beta"):
        assert torch.equal(ref[name], fus[name]), f"ring mismatch: {name}"


def run_modes(args, label_extra=""):
    print(
        f"H={args.heads} HV={args.hv_heads} K={args.head_k_dim} V={args.head_v_dim} "
        f"T={args.draft_tokens} gate={'safe' if args.lower_bound is not None else 'std'} "
        f"BV={fused_mod.KDA_VERIFY_BLOCK_V}{label_extra}"
    )
    header = f"{'B':>4}  " + "".join(f"{m:>14}" for m in args.modes)
    print(header)
    for B in args.batch_sizes:
        times = []
        for mode in args.modes:
            inp = make_inputs(
                B,
                args.draft_tokens,
                args.heads,
                args.hv_heads,
                args.head_k_dim,
                args.head_v_dim,
            )
            fn = make_runner(
                mode,
                inp,
                B,
                args.draft_tokens,
                args.heads,
                args.hv_heads,
                args.head_k_dim,
                args.head_v_dim,
                args.lower_bound,
            )
            times.append(bench_graph(fn, iters=args.iters))
        row = f"{B:>4}  " + "".join(f"{t:>11.2f} us" for t in times)
        print(row)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, default=8)  # ling-v3 TP4 KDA shape
    parser.add_argument("--hv-heads", type=int, default=None)
    parser.add_argument("--head-k-dim", type=int, default=128)
    parser.add_argument("--head-v-dim", type=int, default=128)
    parser.add_argument("--draft-tokens", type=int, default=4)
    # ling-v3 runs the safe gate: --lower-bound -5.0 (kda_lower_bound).
    parser.add_argument("--lower-bound", type=float, default=None)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64]
    )
    parser.add_argument("--modes", nargs="+", default=list(_MODES), choices=_MODES)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--sweep-bv",
        action="store_true",
        help="re-run the fused modes across KDA_VERIFY_BLOCK_V candidates; "
        "BLOCK_V was tuned with snapshot writes on, so ring mode may move it",
    )
    parser.add_argument("--skip-check", action="store_true")
    args = parser.parse_args()
    if args.hv_heads is None:
        args.hv_heads = args.heads
    if args.draft_tokens * 2 > _RING_LEN:
        raise ValueError(f"--draft-tokens > {_RING_LEN // 2} exceeds the bench ring")

    if not args.skip_check:
        check_ring_bitwise(
            4,
            args.draft_tokens,
            args.heads,
            args.hv_heads,
            args.head_k_dim,
            args.head_v_dim,
            args.lower_bound,
        )
        print("ring bitwise check: OK\n")

    run_modes(args)

    if args.sweep_bv:
        args.modes = [m for m in args.modes if m.startswith("fused")] or [
            "fused",
            "fused+ring",
        ]
        default_bv = fused_mod.KDA_VERIFY_BLOCK_V
        try:
            for bv in (2, 4, 8, 16, 32):
                fused_mod.KDA_VERIFY_BLOCK_V = bv
                run_modes(args, label_extra=" (BV sweep)")
        finally:
            fused_mod.KDA_VERIFY_BLOCK_V = default_bv


if __name__ == "__main__":
    main()
