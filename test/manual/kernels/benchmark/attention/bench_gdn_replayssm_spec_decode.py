"""Microbenchmark for one GDN ReplaySSM spec-decode step with a 16-bit
checkpoint (``--mamba-ssm-dtype bfloat16``): the verify of every GDN layer
(``gdn_replayssm_spec_decode``) and the commit that folds the accepted prefix
into the checkpoint (``commit_gdn_replayssm_circular``). Defaults match a GDN
model with 16 key and 48 value heads of 128 and 36 GDN layers at TP1, and MTP
with 4 draft tokens and the default ring of 16.

The pools are sized as a server with ``--max-running-requests 128`` allocates
them (640 state slots, a ring row per request plus one), and q/k/v are views
of one conv output, as in the backend. Sizes matter on AMD: Triton compiles a
pointer whose tensor storage is under 2 GiB with 32-bit offsets.

Fold-every-commit state: the history is empty at the verify, and the commit
folds every row, with track tensors (the radix cache's default; no row at a
track step) and without them. Both run in a CUDA graph; the commit restores
the request cursors before each replay, and that restore is timed alone and
subtracted. Step = verify + commit with track tensors.

Run: ``python test/manual/kernels/benchmark/attention/bench_gdn_replayssm_spec_decode.py``
"""

from __future__ import annotations

import argparse

import torch
import triton

from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import (
    commit_gdn_replayssm_circular,
    gdn_replayssm_spec_decode,
)


def _median_us(fn):
    # do_bench_cudagraph runs fn on a side stream that does not wait for the
    # kernels that wrote the inputs.
    torch.cuda.synchronize()
    return triton.testing.do_bench_cudagraph(fn, rep=100, return_mode="median") * 1e3


def _make_pool(args, device):
    NL, H, HV, K, V, L = args.num_layers, args.h, args.hv, args.k, args.v, args.ring_len
    ring = (NL, args.ring_rows)

    def bf16(*shape):
        return torch.empty(*shape, device=device, dtype=torch.bfloat16)

    return dict(
        state=bf16(NL, args.num_slots, HV, V, K).normal_(0, 0.1),
        d=bf16(*ring, HV, L, V).normal_(0, 0.1),
        rawv=bf16(*ring, HV, L, V).normal_(0, 1e-3),
        k_ring=bf16(*ring, H, L, K).normal_(0, 0.1),
        rawk=bf16(*ring, H, L, K).normal_(0, 1e-3),
        g=-torch.rand(*ring, HV, L, device=device) * 0.1,
    )


def _bench_bs(args, pool, bs, device):
    NL, H, HV, K, V = args.num_layers, args.h, args.hv, args.k, args.v
    L, D = args.ring_len, args.draft_tokens
    T = bs * D
    gen = torch.Generator(device=device).manual_seed(bs)

    def rand(*shape, scale=1.0, dtype=torch.bfloat16):
        return (torch.randn(*shape, device=device, generator=gen) * scale).to(dtype)

    def randint(lo, hi, size, dtype):
        return torch.randint(lo, hi, size, device=device, generator=gen, dtype=dtype)

    def pick(n):
        return torch.randperm(n, device=device, generator=gen)[:bs].to(torch.int32)

    state, d, rawv = pool["state"], pool["d"], pool["rawv"]
    k_ring, rawk, g = pool["k_ring"], pool["rawk"], pool["g"]
    slots, rows = pick(args.num_slots), pick(args.ring_rows)
    mixed_qkv = rand(T, 2 * H * K + HV * V)
    q, k, v = torch.split(mixed_qkv, [H * K, H * K, HV * V], dim=-1)
    q, k, v = q.view(T, H, K), k.view(T, H, K), v.view(T, HV, V)
    a, b = rand(T, HV), rand(T, HV)
    A_log, dt_bias = rand(HV, dtype=torch.float32), rand(HV, dtype=torch.float32)
    query_start_loc = torch.arange(0, T + 1, D, device=device, dtype=torch.int32)
    out = torch.empty(T, HV, V, device=device, dtype=torch.bfloat16)
    write_pos = torch.zeros(args.ring_rows, device=device, dtype=torch.int32)
    cache_base = randint(0, L, (args.ring_rows,), torch.int32)
    is_flush = torch.zeros(args.ring_rows, device=device, dtype=torch.int8)

    def verify():
        for layer in range(NL):
            gdn_replayssm_spec_decode(
                q=q,
                k=k,
                v=v,
                a=a,
                b=b,
                A_log=A_log,
                dt_bias=dt_bias,
                checkpoint_state=state[layer],
                d_cache=d[layer],
                k_cache=k_ring[layer],
                g_cache=g[layer],
                rawv_cache=rawv[layer],
                rawk_cache=rawk[layer],
                beta_cache=None,
                out=out,
                query_start_loc=query_start_loc,
                ssm_state_indices=slots,
                replay_indices=rows,
                write_pos=write_pos,
                cache_base=cache_base,
                is_flush=is_flush,
                max_cache_len=L,
                max_spec_len=D,
                null_block_id=-1,
                launch_mode="verify",
            )

    verify_us = _median_us(verify)

    accept = randint(1, D + 1, (bs,), torch.int32)
    cursors = (write_pos, cache_base, is_flush)
    committed = (write_pos.clone(), cache_base.clone(), torch.ones_like(is_flush))
    committed[0][rows.long()] = accept

    def restore():
        for dst, src in zip(cursors, committed):
            dst.copy_(src)

    no_track = torch.full((bs,), -1, device=device, dtype=torch.int64)
    track_slots = pick(args.num_slots)

    def commit(track):
        restore()
        commit_gdn_replayssm_circular(
            checkpoint_state=state,
            d_cache=d,
            k_cache=k_ring,
            g_cache=g,
            d_residual_cache=rawv,
            k_residual_cache=rawk,
            state_batch_indices=slots,
            replay_indices=rows,
            write_pos=write_pos,
            cache_base=cache_base,
            is_flush=is_flush,
            accept_lens=accept,
            mamba_track_indices=track_slots if track else None,
            mamba_steps_to_track=no_track if track else None,
            null_block_id=-1,
        )

    restore_us = _median_us(restore)
    commit_us = _median_us(lambda: commit(True)) - restore_us
    commit_no_track_us = _median_us(lambda: commit(False)) - restore_us
    return verify_us, commit_us, commit_no_track_us


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-layers", type=int, default=36)
    parser.add_argument("--h", type=int, default=16)
    parser.add_argument("--hv", type=int, default=48)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--v", type=int, default=128)
    parser.add_argument("--ring-len", type=int, default=16)
    parser.add_argument("--draft-tokens", type=int, default=4)
    parser.add_argument("--num-slots", type=int, default=640)
    parser.add_argument("--ring-rows", type=int, default=129)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 24, 32, 48, 64]
    )
    args = parser.parse_args()

    pool = _make_pool(args, "cuda")
    print(
        f"config: layers={args.num_layers} H={args.h} HV={args.hv} K={args.k} "
        f"V={args.v} ring_len={args.ring_len} draft_tokens={args.draft_tokens} "
        f"slots={args.num_slots} ring_rows={args.ring_rows}"
    )
    print(
        f"{'bs':>4} {'verify us':>10} {'commit us':>10} {'step us':>10} "
        f"{'commit us, no track tensors':>28}"
    )
    for bs in args.batch_sizes:
        verify_us, commit_us, commit_no_track_us = _bench_bs(args, pool, bs, "cuda")
        print(
            f"{bs:>4} {verify_us:>10.1f} {commit_us:>10.1f} "
            f"{verify_us + commit_us:>10.1f} {commit_no_track_us:>28.1f}"
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
