"""Diagnostic for KDA CP: localize whether h0 handoff or output path is wrong.

Per rank r: compare (a) the h0 the CP pre-process produced vs the true state
at the shard boundary (obtained by running chunk_kda on the global prefix),
and (b) the per-rank output shard error vs the reference output slice.

Usage: PYTHONPATH=<repo>/python python3 test/manual/kda_cp_diag.py
"""

from unittest.mock import patch

import torch

import sglang.kernels.ops.attention.fla.kda as kda_mod
from sglang.kernels.ops.attention.fla.chunk_delta_h_cp import (
    LinearAttnCPContext,
    build_cp_shard_layout,
    chunk_gated_delta_rule_fwd_h_cp_pre_process,
)
from sglang.kernels.ops.attention.fla.kda import chunk_kda

H, D, NUM_SLOTS = 4, 128, 8
DEVICE = "cuda"
SEQ_LENS = [1000, 704]
W = 4


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
        self.replay = torch.stack([self.recorded[r] for r in range(self.world_size)])


def make_inputs(total_tokens):
    mk = lambda *s: torch.randn(*s, dtype=torch.bfloat16, device=DEVICE)
    return {
        "q": mk(1, total_tokens, H, D),
        "k": mk(1, total_tokens, H, D),
        "v": mk(1, total_tokens, H, D),
        "g": mk(1, total_tokens, H, D),
        "beta": mk(1, total_tokens, H).float().sigmoid(),
        "A_log": torch.randn(1, 1, H, 1, dtype=torch.float32, device=DEVICE),
        "dt_bias": torch.randn(H * D, dtype=torch.float32, device=DEVICE),
    }


def run_chunk_kda(
    inputs, token_ranges, cu_seqlens, pool, slot_indices, cp_context=None
):
    """Run chunk_kda on the concatenation of token_ranges slices."""
    sl = {
        name: torch.cat(
            [inputs[name][:, lo:hi] for lo, hi in token_ranges], dim=1
        ).contiguous()
        for name in ("q", "k", "v", "g", "beta")
    }
    return chunk_kda(
        q=sl["q"],
        k=sl["k"],
        v=sl["v"],
        g=sl["g"],
        beta=sl["beta"],
        initial_state=pool,
        initial_state_indices=slot_indices,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        cp_context=cp_context,
    )


def main():
    torch.manual_seed(42)
    total = sum(SEQ_LENS)
    inputs = make_inputs(total)
    seed_pool = torch.zeros(NUM_SLOTS, H, D, D, dtype=torch.float32, device=DEVICE)
    slot_indices = torch.tensor([3, 5], dtype=torch.int32, device=DEVICE)
    cu_vals = [0]
    for n in SEQ_LENS:
        cu_vals.append(cu_vals[-1] + n)
    cu = torch.tensor(cu_vals, dtype=torch.int32, device=DEVICE)

    # Reference full run
    ref_pool = seed_pool.clone()
    o_ref = run_chunk_kda(
        inputs, [(0, total)], cu, ref_pool, slot_indices
    )

    # True boundary states: run on each global prefix [seq_start, shard_lo)
    layouts = [build_cp_shard_layout(cu_vals, W, r) for r in range(W)]
    true_h0 = {}  # (rank, seq) -> [H, D, D]
    for r in range(W):
        _, ranges, _ids = layouts[r]
        for n, (lo, hi) in enumerate(ranges):
            seq_start = cu_vals[n]
            if lo == seq_start:
                true_h0[(r, n)] = seed_pool[slot_indices[n]].clone()
            else:
                p = seed_pool.clone()
                pref_cu = torch.tensor(
                    [0, lo - seq_start], dtype=torch.int32, device=DEVICE
                )
                run_chunk_kda(
                    inputs,
                    [(seq_start, lo)],
                    pref_cu,
                    p,
                    slot_indices[n : n + 1],
                )
                true_h0[(r, n)] = p[slot_indices[n]].clone()

    # CP sim, capturing the h0 handed to the main kernel per rank
    captured_h0 = {}
    real_pre = chunk_gated_delta_rule_fwd_h_cp_pre_process

    def capturing_pre(**kw):
        h0, idx = real_pre(**kw)
        captured_h0[kw["cp_context"].rank] = h0.clone() if h0 is not None else None
        return h0, idx

    gather = RecordReplayGather(W)
    o_shards = None
    with patch("torch.distributed.all_gather_into_tensor", new=gather), patch.object(
        kda_mod, "chunk_gated_delta_rule_fwd_h_cp_pre_process", new=capturing_pre
    ):
        for do_replay in (False, True):
            o_shards = []
            for r in range(W):
                gather.current_rank = r
                ctx = LinearAttnCPContext(world_size=W, rank=r, group=object())
                local_cu = torch.tensor(
                    layouts[r][0], dtype=torch.int32, device=DEVICE
                )
                pool_r = seed_pool.clone()
                o_r = run_chunk_kda(
                    inputs,
                    layouts[r][1],
                    local_cu,
                    pool_r,
                    slot_indices,
                    cp_context=ctx,
                )
                o_shards.append(o_r)
            if do_replay is False:
                gather.build_replay()

    # Report
    for r in range(W):
        _, ranges, _ids = layouts[r]
        for n, (lo, hi) in enumerate(ranges):
            h0_ratio = norm_ratio(captured_h0[r][n], true_h0[(r, n)])
            offset = sum(x[1] - x[0] for x in ranges[:n])
            o_slice = o_shards[r][:, offset : offset + (hi - lo)]
            o_ratio = norm_ratio(o_slice, o_ref[:, lo:hi])
            print(
                f"rank {r} seq {n} [{lo}:{hi}): h0_ratio={h0_ratio:.3e} "
                f"o_shard_ratio={o_ratio:.3e}"
            )

    # Bisect: plain (non-CP) chunked continuation of seq0 shard [250, 500)
    # seeded by the prefix state — does the BASE path already deviate from the
    # monolithic reference?
    lo, hi = layouts[1][1][0]
    seg_cu = torch.tensor([0, hi - lo], dtype=torch.int32, device=DEVICE)
    p = seed_pool.clone()
    run_chunk_kda(
        inputs, [(0, lo)], torch.tensor([0, lo], dtype=torch.int32, device=DEVICE),
        p, slot_indices[0:1],
    )
    o_cont = run_chunk_kda(
        inputs, [(lo, hi)], seg_cu, p, slot_indices[0:1]
    )
    print(
        f"plain continuation [{lo}:{hi}) via pool slot: "
        f"o_ratio={norm_ratio(o_cont, o_ref[:, lo:hi]):.3e}"
    )

    # Same continuation but with a scratch-style state tensor ([1, H, V, K]
    # fp32 + arange index) exactly like the CP hook hands the main kernel.
    p2 = seed_pool.clone()
    run_chunk_kda(
        inputs, [(0, lo)], torch.tensor([0, lo], dtype=torch.int32, device=DEVICE),
        p2, slot_indices[0:1],
    )
    scratch = p2[slot_indices[0:1]].clone().contiguous()
    o_scratch = run_chunk_kda(
        inputs, [(lo, hi)], seg_cu, scratch,
        torch.arange(1, dtype=torch.int32, device=DEVICE),
    )
    print(
        f"scratch-style continuation [{lo}:{hi}): "
        f"o_ratio={norm_ratio(o_scratch, o_ref[:, lo:hi]):.3e}"
    )


if __name__ == "__main__":
    main()
