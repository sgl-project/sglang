"""Benchmark the fused KDA chain-verify kernel vs the unfused reference pair.

Reference = the sequence kda_backend.forward_extend runs on the MTP
target_verify chain path: reshape/transpose + causal_conv1d_update +
transpose/reshape + split/unflatten + fused_sigmoid_gating_delta_rule_update.

Measures eager latency and cuda-graph replay latency (the production verify
context), sweeping num_warps for the fused kernel and checking numerics per
setting.

Usage:
    python3 benchmark/kernels/bench_fused_kda_conv_recurrent_verify.py

Representative results (H20-3e, ling-v3 per-rank shapes B1 T4 H4 HV4 K128 V128):
    eager     : ref 108.1 us   fused 47.6 us   (2.27x)
    cuda-graph: ref  12.4 us   fused@warps4 9.7 us (1.28x); warps1 29.3 us
"""

import torch

from sglang.kernels.ops.attention.fla.fused_kda_conv_recurrent_verify import (
    fused_kda_conv_gating_verify,
)
from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import (
    causal_conv1d_update,
)

DEV = "cuda"
# ling-v3-ish per-rank shapes at TP4 (verify: B requests x T draft tokens).
B, T, H, HV, K, V, W = 1, 4, 4, 4, 128, 128, 4
DIM = 2 * H * K + HV * V
SEQ_LEN = B * T
LINES = SLOTS = 64

torch.manual_seed(0)
mixed = torch.randn(SEQ_LEN, DIM, device=DEV, dtype=torch.bfloat16) * 0.5
w = torch.randn(DIM, W, device=DEV, dtype=torch.bfloat16) * 0.3
bias = torch.randn(DIM, device=DEV, dtype=torch.bfloat16) * 0.1
a = torch.randn(SEQ_LEN, HV * K, device=DEV, dtype=torch.bfloat16) * 0.5
b = torch.randn(SEQ_LEN, HV, device=DEV, dtype=torch.bfloat16)
A_log = torch.randn(HV, device=DEV, dtype=torch.float32) * 0.5
dt_bias = torch.randn(HV * K, device=DEV, dtype=torch.float32) * 0.5
conv_pool0 = torch.randn(LINES, W - 1, DIM, device=DEV, dtype=torch.bfloat16)
ssm0 = torch.randn(SLOTS, HV, V, K, device=DEV, dtype=torch.float32) * 0.2
cache_indices = torch.arange(2, 2 + B, device=DEV, dtype=torch.int32)
inter_indices = torch.arange(B, device=DEV, dtype=torch.int32)
cu = torch.arange(0, B + 1, device=DEV, dtype=torch.int32) * T


def make_bufs():
    return (
        conv_pool0.clone(),
        ssm0.clone(),
        torch.zeros(LINES, T, W - 1, DIM, device=DEV, dtype=torch.bfloat16),
        torch.zeros(LINES, T, HV, V, K, device=DEV, dtype=torch.float32),
    )


def ref_step(conv_pool, ssm, win, ic):
    x3 = mixed.reshape(B, T, DIM).transpose(1, 2)
    out3 = causal_conv1d_update(
        x3,
        conv_pool.transpose(-1, -2),
        w,
        bias,
        activation="silu",
        conv_state_indices=cache_indices,
        intermediate_conv_window=win.transpose(-1, -2),
        intermediate_state_indices=inter_indices,
    )
    mixed_out = out3.transpose(1, 2).reshape(SEQ_LEN, DIM)
    q, k_, v_ = mixed_out.split([H * K, H * K, HV * V], dim=-1)
    q = q.unflatten(-1, (H, K)).unsqueeze(0)
    k_ = k_.unflatten(-1, (H, K)).unsqueeze(0)
    v_ = v_.unflatten(-1, (HV, V)).unsqueeze(0)
    return fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k_,
        v=v_,
        b=b,
        initial_state_source=ssm,
        initial_state_indices=cache_indices,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu,
        is_kda=True,
        disable_state_update=True,
        intermediate_states_buffer=ic,
        intermediate_state_indices=inter_indices,
        cache_steps=T,
        retrieve_parent_token=None,
        lower_bound=None,
    )


def fused_step(conv_pool, ssm, win, ic, num_warps):
    return fused_kda_conv_gating_verify(
        mixed_qkv=mixed,
        conv_weight=w,
        conv_bias=bias,
        conv_state=conv_pool.transpose(-1, -2),
        conv_state_indices=cache_indices,
        intermediate_conv_window=win.transpose(-1, -2),
        intermediate_state_indices=inter_indices,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        ssm_states=ssm,
        cache_indices=cache_indices,
        intermediate_states_buffer=ic,
        scale=K**-0.5,
        T=T,
        num_q_heads=H,
        num_v_heads=HV,
        head_k_dim=K,
        head_v_dim=V,
        lower_bound=None,
        num_warps=num_warps,
    )


def bench_eager(fn, iters=2000, warmup=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000  # us


def bench_graph(fn, iters=2000, warmup=200):
    g = torch.cuda.CUDAGraph()
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        fn()
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000


def main():
    print(f"shapes: B{B} T{T} H{H} HV{HV} K{K} V{V} dim={DIM}")

    # Reference outputs for numerics checks.
    cp, sm, wn, ic = make_bufs()
    o_ref = ref_step(cp, sm, wn, ic)
    ref_out = (o_ref.clone(), cp.clone(), wn.clone(), ic.clone())

    bufs = make_bufs()
    t_ref_eager = bench_eager(lambda: ref_step(*bufs))
    bufs = make_bufs()
    t_ref_graph = bench_graph(lambda: ref_step(*bufs))
    print(
        f"reference : eager {t_ref_eager:7.2f} us   cuda-graph {t_ref_graph:7.2f} us"
    )

    for nw in [1, 2, 4, 8]:
        cp, sm, wn, ic = make_bufs()
        o_f = fused_step(cp, sm, wn, ic, nw)
        parts = []
        for name, xr, xf in [
            ("o", o_ref, o_f),
            ("conv", ref_out[1], cp),
            ("win", ref_out[2], wn),
            ("ic", ref_out[3], ic),
        ]:
            if torch.equal(xr, xf):
                parts.append(f"{name}=EXACT")
            else:
                d = (xr.float() - xf.float()).abs().max().item()
                parts.append(f"{name}: max|diff|={d:.2e}")
        bufs = make_bufs()
        t_eager = bench_eager(lambda: fused_step(*bufs, nw))
        bufs = make_bufs()
        t_graph = bench_graph(lambda: fused_step(*bufs, nw))
        print(
            f"fused nw={nw}: eager {t_eager:7.2f} us ({t_ref_eager / t_eager:.2f}x)"
            f"   cuda-graph {t_graph:7.2f} us ({t_ref_graph / t_graph:.2f}x)"
            f"   numerics: [{', '.join(parts)}]"
        )


if __name__ == "__main__":
    main()
