"""Real-distributed KDA CP check (manual, torchrun).

Runs the actual NCCL path end to end on W GPUs — no mocked collectives:
* chunk_kda with an active LinearAttnCPContext (state pre-scan -> real
  all_gather_into_tensor -> merge -> pool writeback), and
* exchange_kda_conv_halo (real all-gather of conv tails).

Every rank builds identical global inputs from a fixed seed, runs its own
shard through the real CP path, and compares against locally-computed
references: the sequential shard chain for the SSM output/state (the
chunked-prefill semantics CP must reproduce) and a direct full-stream window
reference for the conv halo.

Usage:
  PYTHONPATH=<repo>/python torchrun --nproc-per-node=2 \
      test/manual/kda_cp_dist_check.py
"""

import os
import sys

import torch
import torch.distributed as dist

from sglang.kernels.ops.attention.fla.chunk_delta_h_cp import (
    LinearAttnCPContext,
    build_cp_shard_layout,
)
from sglang.kernels.ops.attention.fla.kda import chunk_kda
from sglang.srt.layers.attention.linear.kda_cp import (
    build_kda_cp_prefill_metadata,
    exchange_kda_conv_halo,
)

H, D, NUM_SLOTS = 4, 128, 8
CONV_WINDOW = 3
CHAIN_TOL = 2e-3
DEVICE = "cuda"


def norm_ratio(actual, ref):
    ref = ref.float()
    return ((actual.float() - ref).norm() / ref.norm().clamp(min=1e-12)).item()


def make_inputs(total_tokens):
    shape = (1, total_tokens, H, D)
    return {
        "q": torch.randn(shape, dtype=torch.bfloat16, device=DEVICE),
        "k": torch.randn(shape, dtype=torch.bfloat16, device=DEVICE),
        "v": torch.randn(shape, dtype=torch.bfloat16, device=DEVICE) * 0.1,
        "g": (torch.randn(shape, dtype=torch.float32, device=DEVICE) * 0.5 - 2.0).to(
            torch.bfloat16
        ),
        "beta": torch.rand(1, total_tokens, H, dtype=torch.bfloat16, device=DEVICE)
        .float()
        .sigmoid(),
        "A_log": torch.randn(1, 1, H, 1, dtype=torch.float32, device=DEVICE) * 0.1,
        "dt_bias": torch.randn(H * D, dtype=torch.float32, device=DEVICE) * 0.1,
    }


def slice_rank_inputs(inputs, shard_ranges):
    sliced = {
        name: torch.cat(
            [inputs[name][:, lo:hi] for lo, hi in shard_ranges], dim=1
        ).contiguous()
        for name in ("q", "k", "v", "g", "beta")
    }
    sliced["A_log"] = inputs["A_log"]
    sliced["dt_bias"] = inputs["dt_bias"]
    return sliced


def run_chunk_kda(inputs, cu_seqlens, pool, slot_indices, cp_context=None):
    return chunk_kda(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"].clone(),
        g=inputs["g"],
        beta=inputs["beta"],
        initial_state=pool,
        initial_state_indices=slot_indices,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        cp_context=cp_context,
    )


def check_ssm_case(name, seq_lens, zero_seed, rank, world_size):
    torch.manual_seed(42)
    total_tokens = sum(seq_lens)
    inputs = make_inputs(total_tokens)
    seed_pool = torch.zeros(NUM_SLOTS, H, D, D, dtype=torch.float32, device=DEVICE)
    if zero_seed is False:
        seed_pool.normal_(std=0.05)
    slot_indices = torch.tensor(
        [3, 5, 1, 6][: len(seq_lens)], dtype=torch.int32, device=DEVICE
    )
    cu_vals = [0]
    for n in seq_lens:
        cu_vals.append(cu_vals[-1] + n)
    layouts = [
        build_cp_shard_layout(cu_vals, world_size, r) for r in range(world_size)
    ]

    # Local reference: the sequential shard chain (chunked-prefill semantics).
    ref_pool = seed_pool.clone()
    ref_shards = []
    for local_cu, ranges, seq_ids in layouts:
        cu = torch.tensor(local_cu, dtype=torch.int32, device=DEVICE)
        ref_shards.append(
            run_chunk_kda(
                slice_rank_inputs(inputs, ranges), cu, ref_pool, slot_indices[seq_ids]
            )
        )

    # Real CP run for THIS rank.
    local_cu, ranges, seq_ids = layouts[rank]
    ctx = LinearAttnCPContext(
        world_size=world_size,
        rank=rank,
        group=dist.group.WORLD,
        num_global_seqs=len(seq_lens),
        local_seq_ids=torch.tensor(seq_ids, dtype=torch.int32, device=DEVICE),
    )
    pool_cp = seed_pool.clone()
    o_cp = run_chunk_kda(
        slice_rank_inputs(inputs, ranges),
        torch.tensor(local_cu, dtype=torch.int32, device=DEVICE),
        pool_cp,
        slot_indices,
        cp_context=ctx,
    )

    o_ratio = norm_ratio(o_cp, ref_shards[rank])
    state_ratio = norm_ratio(pool_cp[slot_indices], ref_pool[slot_indices])
    ok = o_ratio < CHAIN_TOL and state_ratio < CHAIN_TOL
    print(
        f"[rank {rank}] [{'PASS' if ok else 'FAIL'}] {name}: "
        f"o_vs_chain={o_ratio:.2e} state_vs_chain={state_ratio:.2e}",
        flush=True,
    )
    return ok


def check_conv_halo(rank, world_size):
    torch.manual_seed(7)
    seq_lens = [37, 501]
    total = sum(seq_lens)
    dim = 24
    tokens = torch.randn(total, dim, dtype=torch.bfloat16, device=DEVICE)
    prior = torch.randn(
        len(seq_lens), CONV_WINDOW, dim, dtype=torch.bfloat16, device=DEVICE
    )
    has_prior = torch.tensor([True, False], device=DEVICE)

    meta = build_kda_cp_prefill_metadata(
        seq_lens, world_size=world_size, rank=rank, device=DEVICE
    )
    offsets = [0] * len(seq_lens)
    for j in range(rank):
        for n in range(len(seq_lens)):
            offsets[n] += meta.shard_lens[j][n]
    seq_starts = [0]
    for n in seq_lens[:-1]:
        seq_starts.append(seq_starts[-1] + n)
    conv_input = torch.cat(
        [
            tokens[
                seq_starts[n] + offsets[n] : seq_starts[n]
                + offsets[n]
                + meta.shard_lens[rank][n]
            ]
            for n in meta.local_seq_ids_list
        ],
        dim=0,
    )

    halo, halo_has_initial, global_tails = exchange_kda_conv_halo(
        conv_input=conv_input,
        metadata=meta,
        prior_conv_windows=prior,
        has_prior=has_prior,
        group=dist.group.WORLD,
    )

    ok = True
    for i, n in enumerate(meta.local_seq_ids_list):
        stream = tokens[seq_starts[n] : seq_starts[n] + seq_lens[n]]
        prior_or_zero = prior[n] if bool(has_prior[n]) else torch.zeros_like(prior[n])
        expect_halo = torch.cat([prior_or_zero, stream[: offsets[n]]], dim=0)[
            -CONV_WINDOW:
        ]
        expect_tail = torch.cat([prior_or_zero, stream], dim=0)[-CONV_WINDOW:]
        ok &= torch.equal(halo[i], expect_halo)
        ok &= torch.equal(global_tails[n], expect_tail)
    print(
        f"[rank {rank}] [{'PASS' if ok else 'FAIL'}] conv_halo (bitwise vs "
        "full-stream reference)",
        flush=True,
    )
    return ok


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl")

    results = [
        check_ssm_case("ssm_fresh", [1000, 704], True, rank, world_size),
        check_ssm_case("ssm_continuation", [831, 512], False, rank, world_size),
        check_conv_halo(rank, world_size),
    ]
    all_ok = torch.tensor([int(all(results))], device=DEVICE)
    dist.all_reduce(all_ok, op=dist.ReduceOp.MIN)
    if rank == 0:
        print("DIST_ALL_PASS" if int(all_ok.item()) == 1 else "DIST_SOME_FAILED",
              flush=True)
    dist.destroy_process_group()
    sys.exit(0 if int(all_ok.item()) == 1 else 1)


if __name__ == "__main__":
    main()
