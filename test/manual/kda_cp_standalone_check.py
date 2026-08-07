"""Standalone KDA context-parallel correctness check (manual, non-CI).

Mirrors test/registered/attention/test_kda_cp_state.py but imports ONLY the
kernels tree, so it runs on boxes where the full sglang.test / srt import
chain breaks on unrelated dependencies (xgrammar, etc.). Keep the two in sync
when the CP op surface changes.

References used (established by test/manual/kda_cp_diag2.py):
* sequential shard chain (run shards one by one through the pool slot —
  exactly chunked-prefill semantics, same per-shard chunk grid as CP): CP must
  reproduce this tightly; only the h0 delivery differs (affine merge chain vs
  sequential carry). Tolerance 2e-3.
* monolithic single run: differs from any re-chunked run by bf16 rounding
  (~4e-3 with well-scaled inputs). Loose cross-check at 1e-2.
Inputs use the well-scaled distribution of the existing KDA CI test —
unscaled randn gates/values are numerically adversarial for the output path
(0.17 off the fp32 naive truth even for a monolithic run, see diag2).

Usage: PYTHONPATH=<repo>/python python3 test/manual/kda_cp_standalone_check.py
"""

import sys
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.fla.chunk_delta_h_cp import (
    LinearAttnCPContext,
    build_cp_shard_layout,
)
from sglang.kernels.ops.attention.fla.kda import chunk_kda

H, D, NUM_SLOTS = 4, 128, 8
CHAIN_TOL = 2e-3
MONO_TOL = 1e-2
DEVICE = "cuda"


def norm_ratio(actual, ref):
    ref = ref.float()
    return ((actual.float() - ref).norm() / ref.norm().clamp(min=1e-12)).item()


class RecordReplayGather:
    def __init__(self, world_size):
        self.world_size = world_size
        self.recorded = {}
        self.replay = None
        self.current_rank = None

    def __call__(self, out, inp, group=None):
        if self.replay is None:
            self.recorded[self.current_rank] = inp.clone()
            out.zero_()
        else:
            out.copy_(self.replay)

    def build_replay(self):
        assert len(self.recorded) == self.world_size
        self.replay = torch.stack([self.recorded[r] for r in range(self.world_size)])


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


def make_seed_pool(zero):
    pool = torch.zeros(NUM_SLOTS, H, D, D, dtype=torch.float32, device=DEVICE)
    if zero is False:
        pool.normal_(std=0.05)
    return pool


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
    # chunk_kda writes its output into the v buffer (o=v aliasing) — clone.
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


def scatter_shards(inputs, layouts, o_shards, total_tokens):
    o_full = inputs["q"].new_empty(1, total_tokens, H, D)
    for r, (_, ranges, _ids) in enumerate(layouts):
        offset = 0
        for lo, hi in ranges:
            o_full[:, lo:hi] = o_shards[r][:, offset : offset + (hi - lo)]
            offset += hi - lo
    return o_full


def run_sequential_chain(inputs, layouts, seed_pool, slot_indices, total_tokens):
    """Chunked-prefill reference: run shard 0..W-1 in order through the pool."""
    pool = seed_pool.clone()
    o_shards = []
    for local_cu, ranges, seq_ids in layouts:
        cu = torch.tensor(local_cu, dtype=torch.int32, device=DEVICE)
        o_shards.append(
            run_chunk_kda(
                slice_rank_inputs(inputs, ranges), cu, pool, slot_indices[seq_ids]
            )
        )
    return scatter_shards(inputs, layouts, o_shards, total_tokens), pool


def run_cp_sim(inputs, layouts, seed_pool, slot_indices, world_size, total_tokens):
    rank_inputs = [slice_rank_inputs(inputs, ranges) for _, ranges, _ids in layouts]
    rank_cu = [
        torch.tensor(local_cu, dtype=torch.int32, device=DEVICE)
        for local_cu, _ranges, _ids in layouts
    ]
    num_global_seqs = len(slot_indices)
    gather = RecordReplayGather(world_size)
    o_shards, pools = None, None
    with patch("torch.distributed.all_gather_into_tensor", new=gather):
        for do_replay in (False, True):
            o_shards, pools = [], []
            for r in range(world_size):
                gather.current_rank = r
                ctx = LinearAttnCPContext(
                    world_size=world_size,
                    rank=r,
                    group=object(),
                    num_global_seqs=num_global_seqs,
                    local_seq_ids=torch.tensor(
                        layouts[r][2], dtype=torch.int32, device=DEVICE
                    ),
                )
                pool_r = seed_pool.clone()
                o_r = run_chunk_kda(
                    rank_inputs[r], rank_cu[r], pool_r, slot_indices, cp_context=ctx
                )
                o_shards.append(o_r)
                pools.append(pool_r)
            if do_replay is False:
                gather.build_replay()
    return scatter_shards(inputs, layouts, o_shards, total_tokens), pools


def check_case(name, seq_lens, world_size, zero_seed):
    torch.manual_seed(42)
    total_tokens = sum(seq_lens)
    inputs = make_inputs(total_tokens)
    seed_pool = make_seed_pool(zero=zero_seed)
    slot_indices = torch.tensor(
        [3, 5, 1, 6][: len(seq_lens)], dtype=torch.int32, device=DEVICE
    )
    cu_vals = [0]
    for n in seq_lens:
        cu_vals.append(cu_vals[-1] + n)
    cu = torch.tensor(cu_vals, dtype=torch.int32, device=DEVICE)
    layouts = [
        build_cp_shard_layout(cu_vals, world_size, r) for r in range(world_size)
    ]

    mono_pool = seed_pool.clone()
    o_mono = run_chunk_kda(inputs, cu, mono_pool, slot_indices)
    o_chain, chain_pool = run_sequential_chain(
        inputs, layouts, seed_pool, slot_indices, total_tokens
    )
    o_cp, cp_pools = run_cp_sim(
        inputs, layouts, seed_pool, slot_indices, world_size, total_tokens
    )

    chain_ratio = norm_ratio(o_cp, o_chain)
    mono_ratio = norm_ratio(o_cp, o_mono)
    state_chain = max(
        norm_ratio(p[slot_indices], chain_pool[slot_indices]) for p in cp_pools
    )
    state_mono = max(
        norm_ratio(p[slot_indices], mono_pool[slot_indices]) for p in cp_pools
    )
    untouched = torch.ones(NUM_SLOTS, dtype=torch.bool)
    untouched[slot_indices.cpu()] = False
    untouched_ok = all(
        torch.equal(p[untouched], seed_pool[untouched]) for p in cp_pools
    )
    ok = (
        chain_ratio < CHAIN_TOL
        and mono_ratio < MONO_TOL
        and state_chain < CHAIN_TOL
        and state_mono < MONO_TOL
        and untouched_ok
    )
    print(
        f"[{'PASS' if ok else 'FAIL'}] {name}: o_vs_chain={chain_ratio:.2e} "
        f"o_vs_mono={mono_ratio:.2e} state_vs_chain={state_chain:.2e} "
        f"state_vs_mono={state_mono:.2e} untouched_ok={untouched_ok}"
    )
    return ok


def check_cp1_passthrough():
    torch.manual_seed(42)
    total_tokens = 320
    inputs = make_inputs(total_tokens)
    seed_pool = make_seed_pool(zero=False)
    slot_indices = torch.tensor([2], dtype=torch.int32, device=DEVICE)
    cu = torch.tensor([0, total_tokens], dtype=torch.int32, device=DEVICE)

    ref_pool = seed_pool.clone()
    o_ref = run_chunk_kda(inputs, cu, ref_pool, slot_indices)
    cp_pool = seed_pool.clone()
    ctx = LinearAttnCPContext(world_size=1, rank=0, group=None)
    o_cp = run_chunk_kda(inputs, cu, cp_pool, slot_indices, cp_context=ctx)
    ok = torch.equal(o_cp, o_ref) and torch.equal(cp_pool, ref_pool)
    print(f"[{'PASS' if ok else 'FAIL'}] cp1_passthrough (bitwise)")
    return ok


def main():
    print(
        f"torch {torch.__version__} cuda_available={torch.cuda.is_available()} "
        f"device_cap={torch.cuda.get_device_capability(0)}"
    )
    results = [
        check_case("cp4_fresh_prefill", [1000, 704], 4, zero_seed=True),
        check_case("cp4_chunked_continuation", [831, 512], 4, zero_seed=False),
        check_case("cp8_empty_shards", [5, 640], 8, zero_seed=False),
        check_cp1_passthrough(),
    ]
    print("ALL_PASS" if all(results) else "SOME_FAILED")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
