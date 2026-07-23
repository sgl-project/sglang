"""Benchmark: KDA ReplaySSM verify ring-write, eager vs fused (CACHE_RING).

Two ways to fill the per-slot ReplaySSM ring during dspark verify:
  - eager: recurrent verify kernel + eager torch ring-write (gate/beta recompute +
    4 transposes + 4 ``index_put_`` scatters) -- ~8-12 extra kernels per KDA layer.
  - fused: recurrent verify kernel with ``CACHE_RING`` -- the pre-norm k / raw v /
    in-kernel gate / beta are stored inside the kernel loop, zero extra kernels.

Both paths are timed whole (verify + ring-write), not by subtracting a verify-only
baseline -- the ring-write delta is small next to the verify kernel, so a
difference-of-two-large-timings goes into the noise. do_bench with cuda graph
(the production decode path also runs under cuda graph) so the number reflects the
GPU work the fusion removes, consistent with the e2e win. Correctness is covered by
test/registered/kernels/test_kda_replayssm_ring_fused.py.

Usage: python bench_kda_verify_ringwrite.py
"""

import torch

from sglang.jit_kernel.benchmark import marker
from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)

HV = H = 32          # K3 KDA per TP8 rank
K = V = 128
L = 16               # ring length (linear_replayssm_cache_len)
GAMMA = 7            # dspark block size -> T = draft_token_num
LOWER_BOUND = -5.0   # K3 safe gate
RING_DTYPE = torch.bfloat16


def make_inputs(bs, device, dtype):
    torch.manual_seed(42)
    T = GAMMA
    num_slots = bs + 1
    return dict(
        T=T,
        q=torch.randn(bs, T, H, K, device=device, dtype=dtype),
        k=torch.randn(bs, T, H, K, device=device, dtype=dtype),
        v=torch.randn(bs, T, HV, V, device=device, dtype=dtype),
        a=torch.randn(bs, T, HV, K, device=device, dtype=dtype),
        b=torch.randn(bs, T, HV, device=device, dtype=dtype),
        A_log=torch.randn(HV, device=device, dtype=torch.float32),
        dt_bias=torch.randn(HV, K, device=device, dtype=torch.float32),
        slots=torch.arange(1, bs + 1, device=device, dtype=torch.int32),
        cu=torch.arange(0, (bs + 1) * T, T, device=device, dtype=torch.int32),
        h0=torch.zeros(num_slots, HV, V, K, device=device, dtype=torch.float32),
        inter=torch.zeros(num_slots, T, HV, V, K, device=device, dtype=torch.float32),
        rawv=torch.zeros(num_slots, HV, L, V, device=device, dtype=RING_DTYPE),
        rawk=torch.zeros(num_slots, H, L, K, device=device, dtype=RING_DTYPE),
        gring=torch.zeros(num_slots, HV, L, K, device=device, dtype=torch.float32),
        betar=torch.zeros(num_slots, HV, L, device=device, dtype=torch.float32),
    )


def _verify(inp, cache_ring):
    kw = {}
    if cache_ring:
        kw = dict(
            cache_ring=True, replayssm_rawv=inp["rawv"], replayssm_rawk=inp["rawk"],
            replayssm_g=inp["gring"], replayssm_beta=inp["betar"],
        )
    return fused_sigmoid_gating_delta_rule_update(
        A_log=inp["A_log"], a=inp["a"], dt_bias=inp["dt_bias"],
        softplus_beta=1.0, softplus_threshold=20.0,
        q=inp["q"], k=inp["k"], v=inp["v"], b=inp["b"],
        initial_state_source=inp["h0"], initial_state_indices=inp["slots"],
        cu_seqlens=inp["cu"], scale=K**-0.5, use_qk_l2norm_in_kernel=True,
        is_kda=True, lower_bound=LOWER_BOUND, disable_state_update=True,
        intermediate_states_buffer=inp["inter"], intermediate_state_indices=inp["slots"],
        cache_steps=inp["T"], **kw,
    )


def run_eager(inp):
    _verify(inp, cache_ring=False)
    # eager ring-write, mirroring kda_backend._forward_target_verify (pre-fusion).
    T = inp["T"]
    x = inp["a"].float() + inp["dt_bias"].view(1, 1, HV, K)
    exp_a_log = torch.exp(inp["A_log"]).view(1, 1, HV, 1)
    gk = LOWER_BOUND * torch.sigmoid(exp_a_log * x)
    beta = torch.sigmoid(inp["b"].float())
    s = inp["slots"].clamp(min=0).to(torch.long)
    inp["rawv"][s, :, :T] = inp["v"].transpose(1, 2).to(RING_DTYPE)
    inp["rawk"][s, :, :T] = inp["k"].transpose(1, 2).to(RING_DTYPE)
    inp["gring"][s, :, :T] = gk.transpose(1, 2)
    inp["betar"][s, :, :T] = beta.transpose(1, 2)


def run_fused(inp):
    _verify(inp, cache_ring=True)


@marker.parametrize("bs", [1, 4, 16, 32, 64, 128], [16])
@marker.benchmark("impl", ["eager", "fused"])
def benchmark(bs: int, impl: str):
    inp = make_inputs(bs, "cuda", torch.bfloat16)
    fn = (lambda: run_eager(inp)) if impl == "eager" else (lambda: run_fused(inp))
    return marker.do_bench(fn, use_cuda_graph=True, disable_log_bandwidth=True)


if __name__ == "__main__":
    benchmark.run()
