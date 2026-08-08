"""Diag round 2: is the 0.21 continuation-vs-monolithic gap a kernel bug or
input conditioning?

(a) Naive per-token fp32 recurrence as judge on the wild inputs: which of
    monolithic / continuation deviates from ground truth?
(b) Re-run the continuation comparison with the well-scaled input
    distribution used by the existing CI test (v*0.1, gate*0.5-2, A_log*0.1).

Usage: PYTHONPATH=<repo>/python python3 test/manual/kda_cp_diag2.py
"""

import torch
import torch.nn.functional as F

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


def run_chunk_kda(inputs, lo, hi, pool, slot_indices):
    cu = torch.tensor([0, hi - lo], dtype=torch.int32, device=DEVICE)
    return chunk_kda(
        q=inputs["q"][:, lo:hi].contiguous(),
        k=inputs["k"][:, lo:hi].contiguous(),
        v=inputs["v"][:, lo:hi].clone().contiguous(),
        g=inputs["g"][:, lo:hi].contiguous(),
        beta=inputs["beta"][:, lo:hi].contiguous(),
        initial_state=pool,
        initial_state_indices=slot_indices,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu,
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
    )


def naive(inputs, total):
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
    states_at = {}
    for t in range(total):
        state = state * gate[0, t].exp().unsqueeze(-2)
        residual = v[0, t] - torch.einsum("hvk,hk->hv", state, k[0, t])
        state = state + torch.einsum(
            "hv,hk->hvk", residual * beta[0, t, :, None], k[0, t]
        )
        out[0, t] = torch.einsum("hvk,hk->hv", state, q[0, t]) * scale
        states_at[t + 1] = state.clone()
    return out, states_at


def continuation_gap(inputs, total, split):
    slot = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    pool_mono = torch.zeros(1, H, D, D, dtype=torch.float32, device=DEVICE)
    o_mono = run_chunk_kda(inputs, 0, total, pool_mono, slot)

    pool_cont = torch.zeros(1, H, D, D, dtype=torch.float32, device=DEVICE)
    run_chunk_kda(inputs, 0, split, pool_cont, slot)
    o_cont = run_chunk_kda(inputs, split, total, pool_cont, slot)
    return o_mono, o_cont, pool_mono, pool_cont


def main():
    total, split = 500, 250

    for wild in (True, False):
        torch.manual_seed(42)
        inputs = make_inputs(total, wild=wild)
        o_mono, o_cont, pool_mono, pool_cont = continuation_gap(inputs, total, split)
        tag = "wild" if wild else "scaled"
        print(
            f"[{tag}] continuation vs monolithic on [{split}:{total}): "
            f"o_ratio={norm_ratio(o_cont, o_mono[:, split:]):.3e} "
            f"final_state_ratio={norm_ratio(pool_cont, pool_mono):.3e}"
        )
        naive_o, naive_states = naive(inputs, total)
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
            f"{norm_ratio(pool_mono[0], naive_states[total]):.3e}  "
            f"cont vs naive={norm_ratio(pool_cont[0], naive_states[total]):.3e}"
        )


if __name__ == "__main__":
    main()
