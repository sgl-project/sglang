"""Microbenchmark: buffered output-only GDN decode (ReplaySSM Part A) vs. the
existing packed GDN decode kernel.

Compares per-step decode latency of
``fused_recurrent_gated_delta_rule_packed_decode`` (writes the full recurrent
state S every step) against ``fused_recurrent_gdn_replayssm_decode`` at
L in {1, 8, 16} (writes the full state only every L steps) across batch sizes
{1, 16, 64, 256} for a realistic GDN config (HV=32, K=V=128).

The win is per-step HBM *state* traffic: the packed kernel reads + writes S
(~2 * num_slots * HV * V * K * 4 bytes / step for an fp32 state), while the
ReplaySSM kernel reads S every step but writes it only 1-in-L steps, plus a
small ring append (d:[HV,V], k:[H,K], g:[HV] per step). The amortized state
traffic ratio is reported per L.

Run::

    python -m sglang.srt.layers.attention.fla.bench_gdn_replayssm_decode

Requires a GPU (Triton).
"""

from __future__ import annotations

import argparse

import torch
import triton

from sglang.srt.layers.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
)
from sglang.srt.layers.attention.fla.fused_recurrent_linear_replayssm import (
    fused_recurrent_gdn_replayssm_decode,
)


def _make_static(B, H, HV, K, V, dtype, device):
    qk_dim = 2 * H * K
    v_dim = HV * V
    mixed_qkv = torch.randn(B, qk_dim + v_dim, device=device, dtype=dtype)
    a = torch.randn(B, HV, device=device, dtype=dtype) * 0.5
    b = torch.randn(B, HV, device=device, dtype=dtype)
    A_log = (torch.randn(HV, device=device, dtype=torch.float32) * 0.3).contiguous()
    dt_bias = (torch.randn(HV, device=device, dtype=torch.float32) * 0.1).contiguous()
    return mixed_qkv, a, b, A_log, dt_bias


def _state_bytes_per_step(B, HV, K, V, L, dtype):
    """Amortized per-step HBM *state* traffic (bytes), state in fp32.

    packed: read S + write S every step.
    replay: read S every step; write S once per L steps; append ring records
            (d:[HV,V] in `dtype`, k:[H,K] in `dtype` shared across HV//H, g:[HV]
            fp32) every step. We report the dominant fp32-state terms; ring
            appends are tiny by comparison and shown separately.
    """
    fp32 = 4
    state_elems = B * HV * V * K  # one record per active request slot
    packed = (state_elems * fp32) * 2  # read + write
    replay = (state_elems * fp32) * (1 + 1.0 / L)  # read every step + write 1/L
    return packed, replay


def _bench_cfg(B, H, HV, K, V, Ls, dtype, device, num_slots=None, warmup=25, rep=100):
    num_slots = num_slots or B
    mixed_qkv, a, b, A_log, dt_bias = _make_static(B, H, HV, K, V, dtype, device)
    scale = K**-0.5
    cache_indices = torch.arange(B, device=device, dtype=torch.int32)

    # packed decode
    state = torch.randn(num_slots, HV, V, K, device=device, dtype=torch.float32)
    out = mixed_qkv.new_empty(B, 1, HV, V)

    def run_packed():
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=state,
            out=out,
            ssm_state_indices=cache_indices,
            use_qk_l2norm_in_kernel=True,
        )

    t_packed = triton.testing.do_bench(run_packed, warmup=warmup, rep=rep)

    rows = []
    for L in Ls:
        rstate = torch.randn(num_slots, HV, V, K, device=device, dtype=torch.float32)
        d_cache = torch.zeros(num_slots, HV, L, V, device=device, dtype=dtype)
        k_cache = torch.zeros(num_slots, H, L, K, device=device, dtype=dtype)
        g_cache = torch.zeros(num_slots, HV, L, device=device, dtype=torch.float32)
        write_pos = torch.zeros(B, device=device, dtype=torch.int32)
        rout = mixed_qkv.new_empty(B, 1, HV, V)
        nk = 1 if L == 1 else 2

        def run_replay():
            fused_recurrent_gdn_replayssm_decode(
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=scale,
                initial_state=rstate,
                d_cache=d_cache,
                k_cache=k_cache,
                g_cache=g_cache,
                out=rout,
                ssm_state_indices=cache_indices,
                write_pos=write_pos,
                use_qk_l2norm_in_kernel=True,
                nk=nk,
            )

        t_replay = triton.testing.do_bench(run_replay, warmup=warmup, rep=rep)
        packed_bytes, replay_bytes = _state_bytes_per_step(B, HV, K, V, L, dtype)
        rows.append((L, t_replay, t_packed / t_replay, replay_bytes / packed_bytes))
    return t_packed, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hv", type=int, default=32, help="num value heads")
    parser.add_argument("--h", type=int, default=16, help="num key/query heads")
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--v", type=int, default=128)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16, 64, 256])
    parser.add_argument("--ls", type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA / Triton required for this microbenchmark.")
    device = "cuda"
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    print(
        f"GDN ReplaySSM decode microbench  HV={args.hv} H={args.h} "
        f"K={args.k} V={args.v} dtype={args.dtype}\n"
        "per-step latency (ms); speedup = packed/replay; "
        "state-traffic = replay/packed (lower is better)"
    )
    for B in args.batch_sizes:
        t_packed, rows = _bench_cfg(
            B, args.h, args.hv, args.k, args.v, args.ls, dtype, device
        )
        print(f"\nB={B:<4d}  packed={t_packed:.4f} ms")
        for L, t_replay, speedup, traffic_ratio in rows:
            print(
                f"    L={L:<3d} replay={t_replay:.4f} ms  "
                f"speedup={speedup:5.2f}x  "
                f"state-traffic={traffic_ratio:5.2f}x"
            )


if __name__ == "__main__":
    main()
