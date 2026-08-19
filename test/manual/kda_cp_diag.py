"""Manual diagnostics for KDA CP numerics.

Two probes, selected by --mode (default: all):

* ``cp`` — fault localization for the CP state hand-off. Per rank r, compare
  (a) the h0 the CP pre-process produced vs the true state at the shard
  boundary (obtained by running chunk_kda on the global prefix), and (b) the
  per-rank output shard vs the reference output slice. Answers "is the
  hand-off wrong or the output path?" when the CI test goes red. Uses the
  well-scaled input distribution so output ratios are meaningful (see below).

* ``numerics`` — the input-conditioning study that established the test
  methodology (and the CI tolerances CHAIN_TOL/MONO_TOL): a naive per-token
  fp32 recurrence judges monolithic vs chunked continuation on wild
  (unscaled randn) and well-scaled inputs. Conclusion: wild inputs make ANY
  re-chunked run — and the monolithic run itself — deviate from fp32 ground
  truth at the 1e-1 level, while the scaled distribution keeps everything at
  the bf16 baseline. Large continuation-vs-monolithic gaps on unscaled
  random inputs are conditioning, not a CP bug.

Usage:
    PYTHONPATH=<repo>/python python3 test/manual/kda_cp_diag.py [--mode cp|numerics|all]
"""

import argparse
from unittest.mock import patch

import torch
import torch.nn.functional as F

import sglang.kernels.ops.attention.fla.kda as kda_mod
from sglang.kernels.ops.attention.fla.chunk_delta_h_cp import (
    LinearAttnCPContext,
    build_cp_shard_layout,
    chunk_gated_delta_rule_fwd_h_cp_pre_process,
)
from sglang.kernels.ops.attention.fla.kda import chunk_kda

H, D = 4, 128
DEVICE = "cuda"


def norm_ratio(actual, ref):
    ref = ref.float()
    return ((actual.float() - ref).norm() / ref.norm().clamp(min=1e-12)).item()


def make_inputs(total_tokens, wild):
    if wild:
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
    shape = (1, total_tokens, H, D)
    return {
        "q": torch.randn(shape, dtype=torch.bfloat16, device=DEVICE),
        "k": torch.randn(shape, dtype=torch.bfloat16, device=DEVICE),
        "v": (torch.randn(shape, dtype=torch.bfloat16, device=DEVICE) * 0.1),
        "g": (torch.randn(shape, dtype=torch.float32, device=DEVICE) * 0.5 - 2.0).to(
            torch.bfloat16
        ),
        "beta": torch.rand(1, total_tokens, H, dtype=torch.bfloat16, device=DEVICE)
        .float()
        .sigmoid(),
        "A_log": (torch.randn(1, 1, H, 1, dtype=torch.float32, device=DEVICE) * 0.1),
        "dt_bias": (torch.randn(H * D, dtype=torch.float32, device=DEVICE) * 0.1),
    }


def run_chunk_kda(
    inputs, token_ranges, cu_seqlens, pool, slot_indices, cp_context=None
):
    """Run chunk_kda on the concatenation of token_ranges slices.

    torch.cat builds fresh tensors, so chunk_kda's o=v aliasing cannot
    clobber the shared ``inputs`` dict across calls.
    """
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


class RecordReplayGather:
    """Single-GPU stand-in for all_gather_into_tensor: record pass, then replay."""

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


def run_cp_localization():
    """Per-rank h0 / output-shard localization of a CP hand-off fault."""
    torch.manual_seed(42)
    seq_lens = [1000, 704]
    world = 4
    num_slots = 8
    total = sum(seq_lens)
    inputs = make_inputs(total, wild=False)
    seed_pool = torch.zeros(num_slots, H, D, D, dtype=torch.float32, device=DEVICE)
    slot_indices = torch.tensor([3, 5], dtype=torch.int32, device=DEVICE)
    cu_vals = [0]
    for n in seq_lens:
        cu_vals.append(cu_vals[-1] + n)
    cu = torch.tensor(cu_vals, dtype=torch.int32, device=DEVICE)

    # Reference full run
    ref_pool = seed_pool.clone()
    o_ref = run_chunk_kda(inputs, [(0, total)], cu, ref_pool, slot_indices)

    # True boundary states: run on each global prefix [seq_start, shard_lo)
    layouts = [build_cp_shard_layout(cu_vals, world, r) for r in range(world)]
    true_h0 = {}  # (rank, seq) -> [H, D, D]
    for r in range(world):
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

    gather = RecordReplayGather(world)
    o_shards = None
    with patch("torch.distributed.all_gather_into_tensor", new=gather), patch.object(
        kda_mod, "chunk_gated_delta_rule_fwd_h_cp_pre_process", new=capturing_pre
    ):
        for do_replay in (False, True):
            o_shards = []
            for r in range(world):
                gather.current_rank = r
                ctx = LinearAttnCPContext(world_size=world, rank=r, group=object())
                local_cu = torch.tensor(layouts[r][0], dtype=torch.int32, device=DEVICE)
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

    for r in range(world):
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


def _naive_recurrence(inputs, total):
    """Per-token fp32 recurrence (mirrors TestKDAChunkExponentDomain naive)."""
    q = F.normalize(inputs["q"].float(), dim=-1)
    k = F.normalize(inputs["k"].float(), dim=-1)
    v = inputs["v"].float()
    beta = inputs["beta"].float()
    gate = -torch.exp(inputs["A_log"]) * F.softplus(
        inputs["g"].float() + inputs["dt_bias"].view(1, 1, H, D)
    )
    scale = D**-0.5
    out = torch.empty_like(v)
    state = torch.zeros(H, D, D, dtype=torch.float32, device=DEVICE)  # [H, V, K]
    for t in range(total):
        state = state * gate[0, t].exp().unsqueeze(-2)
        residual = v[0, t] - torch.einsum("hvk,hk->hv", state, k[0, t])
        state = state + torch.einsum(
            "hv,hk->hvk", residual * beta[0, t, :, None], k[0, t]
        )
        out[0, t] = torch.einsum("hvk,hk->hv", state, q[0, t]) * scale
    return out, state


def run_numerics_study():
    """Judge monolithic vs continuation against fp32 truth, wild vs scaled."""
    total, split = 500, 250
    slot = torch.tensor([0], dtype=torch.int32, device=DEVICE)

    for wild in (True, False):
        torch.manual_seed(42)
        inputs = make_inputs(total, wild=wild)
        cu_full = torch.tensor([0, total], dtype=torch.int32, device=DEVICE)
        cu_head = torch.tensor([0, split], dtype=torch.int32, device=DEVICE)
        cu_tail = torch.tensor([0, total - split], dtype=torch.int32, device=DEVICE)

        pool_mono = torch.zeros(1, H, D, D, dtype=torch.float32, device=DEVICE)
        o_mono = run_chunk_kda(inputs, [(0, total)], cu_full, pool_mono, slot)

        pool_cont = torch.zeros(1, H, D, D, dtype=torch.float32, device=DEVICE)
        run_chunk_kda(inputs, [(0, split)], cu_head, pool_cont, slot)
        o_cont = run_chunk_kda(inputs, [(split, total)], cu_tail, pool_cont, slot)

        tag = "wild" if wild else "scaled"
        print(
            f"[{tag}] continuation vs monolithic on [{split}:{total}): "
            f"o_ratio={norm_ratio(o_cont, o_mono[:, split:]):.3e} "
            f"final_state_ratio={norm_ratio(pool_cont, pool_mono):.3e}"
        )
        naive_o, naive_final = _naive_recurrence(inputs, total)
        print(
            f"[{tag}]   monolithic vs naive:  head [0:{split})="
            f"{norm_ratio(o_mono[:, :split], naive_o[:, :split]):.3e}  "
            f"tail [{split}:{total})="
            f"{norm_ratio(o_mono[:, split:], naive_o[:, split:]):.3e}"
        )
        print(
            f"[{tag}]   continuation vs naive: tail [{split}:{total})="
            f"{norm_ratio(o_cont, naive_o[:, split:]):.3e}"
        )
        print(
            f"[{tag}]   final state: mono vs naive="
            f"{norm_ratio(pool_mono[0], naive_final):.3e}  "
            f"cont vs naive={norm_ratio(pool_cont[0], naive_final):.3e}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cp", "numerics", "all"), default="all")
    args = parser.parse_args()
    if args.mode in ("cp", "all"):
        print("=== cp: per-rank h0 / output-shard localization ===")
        run_cp_localization()
    if args.mode in ("numerics", "all"):
        print("=== numerics: input-conditioning study ===")
        run_numerics_study()


if __name__ == "__main__":
    main()
