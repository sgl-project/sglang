"""Microbenchmark for ``commit_gdn_replayssm_fold_all_layers`` (defaults match
Qwen3.5-397B at TP4). The kernel is bound by the mandatory checkpoint
read+write (``bs * layers * HV * K * V * 4 B * 2``), so the reported GB/s
approximates achieved HBM bandwidth.

Run: ``python -m sglang.kernels.ops.attention.fla.bench_gdn_replayssm_fold``
"""

from __future__ import annotations

import argparse

import torch
import triton

from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_fold import (
    commit_gdn_replayssm_fold_all_layers,
)


def _make_pool(num_layers, num_slots, HV, H, K, V, RL, device):
    state = torch.randn(
        num_layers, num_slots, HV, K, V, device=device, dtype=torch.float32
    )
    rawv = torch.randn(
        num_layers, num_slots, HV, RL, V, device=device, dtype=torch.bfloat16
    )
    rawk = torch.randn(
        num_layers, num_slots, H, RL, K, device=device, dtype=torch.bfloat16
    )
    g_ring = (
        -torch.rand(num_layers, num_slots, HV, RL, device=device, dtype=torch.float32)
        * 0.5
    )
    beta = torch.rand(num_layers, num_slots, HV, RL, device=device, dtype=torch.float32)
    return state, rawv, rawk, g_ring, beta


def _bench_bs(*, pool, bs, num_slots, num_layers, HV, H, K, V, RL, device, with_track):
    state, rawv, rawk, g_ring, beta = pool
    gen = torch.Generator(device=device).manual_seed(bs)
    slots = torch.randperm(num_slots, device=device, generator=gen)[:bs].to(torch.int32)
    accept_lens = torch.randint(
        1, RL + 1, (bs,), device=device, dtype=torch.int32, generator=gen
    )
    if with_track:
        track_indices = torch.randperm(num_slots, device=device, generator=gen)[:bs].to(
            torch.int64
        )
        track_steps = torch.where(
            torch.rand(bs, device=device, generator=gen) < 0.25,
            accept_lens.to(torch.int64) - 1,
            torch.full((bs,), -1, dtype=torch.int64, device=device),
        )
    else:
        track_indices = None
        track_steps = None

    def run():
        commit_gdn_replayssm_fold_all_layers(
            checkpoint_state=state,
            rawv_cache=rawv,
            rawk_cache=rawk,
            g_cache=g_ring,
            beta_cache=beta,
            ssm_state_indices=slots,
            accept_lens=accept_lens,
            max_cache_len=RL,
            num_k_heads=H,
            mamba_track_indices=track_indices,
            mamba_steps_to_track=track_steps,
            null_block_id=-1,
        )

    ms = triton.testing.do_bench(run, warmup=25, rep=100)
    traffic_gb = bs * num_layers * HV * K * V * 4 * 2 / 1e9
    return ms * 1000, traffic_gb / (ms / 1000) if ms > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-layers", type=int, default=45)
    parser.add_argument("--num-slots", type=int, default=475)
    parser.add_argument("--hv", type=int, default=16)
    parser.add_argument("--h", type=int, default=4)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--v", type=int, default=128)
    parser.add_argument("--ring-len", type=int, default=4)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 128])
    args = parser.parse_args()

    device = "cuda"
    pool = _make_pool(
        args.num_layers,
        args.num_slots,
        args.hv,
        args.h,
        args.k,
        args.v,
        args.ring_len,
        device,
    )
    print(
        f"config: layers={args.num_layers} slots={args.num_slots} HV={args.hv} "
        f"H={args.h} K={args.k} V={args.v} ring_len={args.ring_len}"
    )
    print(f"{'bs':>4} {'track':>6} {'us/launch':>10} {'achieved GB/s':>14}")
    for bs in args.batch_sizes:
        for with_track in (False, True):
            us, gbs = _bench_bs(
                pool=pool,
                bs=bs,
                num_slots=args.num_slots,
                num_layers=args.num_layers,
                HV=args.hv,
                H=args.h,
                K=args.k,
                V=args.v,
                RL=args.ring_len,
                device=device,
                with_track=with_track,
            )
            print(f"{bs:>4} {str(with_track):>6} {us:>10.1f} {gbs:>14.0f}")


if __name__ == "__main__":
    main()
