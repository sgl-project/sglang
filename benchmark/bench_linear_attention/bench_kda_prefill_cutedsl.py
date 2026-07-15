"""
Benchmark & Correctness: Triton KDA vs CuTeDSL KDA (prefill, SM100 Blackwell).

Compares:
  - Triton:  sglang's chunk_kda (FLA chunkwise gated delta rule, per-channel gate)
  - CuteDSL: kda_blackwell pipeline (fused Triton prologue -> kkt_inv_uw -> h -> o)

KDA differs from GDN by a PER-CHANNEL decay gate (g is [T, H, K], not scalar).
The cutedsl pipeline externalizes the per-channel decay into five pre-scaled
key/query tensors computed by a fused Triton prologue; the chunk metadata is
computed once and shared across layers in a real forward, so the benchmarked
cutedsl path precomputes it outside the timed region (the realistic ceiling).

Correctness is checked against the token-by-token fused_recurrent_kda ground
truth. Reports performance (ms, approx TFLOPS, TB/s, speedup).

Usage:
    python bench_kda_prefill_cutedsl.py                    # default sweep
    python bench_kda_prefill_cutedsl.py --mode bench       # benchmark only
    python bench_kda_prefill_cutedsl.py --mode correctness # correctness only
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.linear.kda_blackwell import prepare_metadata
from sglang.kernels.ops.attention.linear.kda_blackwell.kernel_h import (
    kda_h_cutedsl,
)
from sglang.kernels.ops.attention.linear.kda_blackwell.kernel_kkt_inv_uw import (
    kkt_inv_uw_cutedsl,
)
from sglang.kernels.ops.attention.linear.kda_blackwell.kernel_o import (
    kda_o_cutedsl,
)
from sglang.kernels.ops.attention.linear.kda_blackwell.prologue import (
    kda_prologue,
)
from sglang.srt.layers.attention.fla.kda import chunk_kda, fused_recurrent_kda

BT = 64  # chunk size

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), p=2, dim=-1)


def kda_flops(total_seq_len, num_heads, head_k, head_v):
    """Per-token-per-head: k@v outer (2*K*V) + q@state (2*K*V), plus the intra-chunk
    KKT (2*K*K averaged over the chunk). Approximate (ignores the inverse)."""
    return total_seq_len * num_heads * (4 * head_k * head_v + 2 * head_k * head_k)


def kda_bytes(total_seq_len, num_heads, head_k, head_v, num_seqs, dtype):
    elem = dtype.itemsize
    q_b = total_seq_len * num_heads * head_k * elem
    k_b = total_seq_len * num_heads * head_k * elem
    v_b = total_seq_len * num_heads * head_v * elem
    o_b = total_seq_len * num_heads * head_v * elem
    g_b = total_seq_len * num_heads * head_k * 4  # per-channel gate, fp32
    beta_b = total_seq_len * num_heads * 4
    state_b = 2 * num_seqs * num_heads * head_k * head_v * 4  # fp32 r/w
    return q_b + k_b + v_b + o_b + g_b + beta_b + state_b


# ---------------------------------------------------------------------------
# Input factory (single sequence per benchmark point, B=1)
# ---------------------------------------------------------------------------


def make_inputs(T, H, K, V, device, dtype, seed=42):
    torch.manual_seed(seed)
    q = _l2norm(torch.randn(1, T, H, K, device=device)).to(dtype)
    k = _l2norm(torch.randn(1, T, H, K, device=device)).to(dtype)
    v = torch.randn(1, T, H, V, device=device).to(dtype)
    # Mild per-channel gate (real Kimi-Linear regime; keeps exp() in fp32 range).
    A_log = torch.randn(H, device=device) * 0.5 - 1.5
    dt_bias = torch.randn(H, K, device=device) * 0.1
    g_raw = torch.randn(1, T, H, K, device=device)
    g_act = (
        -A_log.exp().view(1, 1, H, 1) * F.softplus(g_raw + dt_bias.view(1, 1, H, K))
    ).float()
    beta = torch.sigmoid(torch.randn(1, T, H, device=device)).float()
    return dict(q=q, k=k, v=v, g_act=g_act, beta=beta, T=T, H=H, K=K, V=V)


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def run_recurrent(inp, scale):
    """Token-by-token ground truth. Returns (o [T,H,V], state [1,H,V,K])."""
    cu = torch.tensor([0, inp["T"]], dtype=torch.int64, device=inp["q"].device)
    h0 = torch.zeros(1, inp["H"], inp["V"], inp["K"], device=inp["q"].device)
    o, state = fused_recurrent_kda(
        q=inp["q"],
        k=inp["k"],
        v=inp["v"],
        g=inp["g_act"],
        beta=inp["beta"],
        scale=scale,
        initial_state=h0,
        inplace_final_state=False,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=cu,
    )
    return o[0], state


def cutedsl_buffers(inp, num_sms, device):
    """Precompute metadata + preallocate (shared across layers in a real forward)."""
    T, H, K, V = inp["T"], inp["H"], inp["K"], inp["V"]
    cu = torch.tensor([0, T], dtype=torch.int32, device=device)
    ci, co, tc, total = prepare_metadata(cu)
    pad_t = total * BT
    return dict(
        cu=cu,
        ci=ci,
        co=co,
        tc=tc,
        total=total,
        num_sms=num_sms,
        h0=torch.zeros(1, H, V, K, device=device, dtype=torch.float32),
        U=torch.empty(pad_t, H, V, device=device, dtype=torch.bfloat16),
        W=torch.empty(pad_t, H, K, device=device, dtype=torch.bfloat16),
        V_new=torch.empty(pad_t, H, V, device=device, dtype=torch.bfloat16),
        h_chunks=torch.empty(total, H, V, K, device=device, dtype=torch.bfloat16),
        ht=torch.empty(1, H, V, K, device=device, dtype=torch.float32),
        o=torch.empty(T, H, V, device=device, dtype=torch.bfloat16),
    )


def run_cutedsl_pipeline(inp, buf, scale):
    """The fused-prologue + 3 cutedsl kernels (metadata precomputed in buf)."""
    q3, k3, v3 = inp["q"][0], inp["k"][0], inp["v"][0]
    g3, beta3 = inp["g_act"][0], inp["beta"][0].contiguous()
    KL, KR, KG, qg, qg2, g_cu = kda_prologue(
        q3, k3, g3, scale, buf["cu"], buf["ci"], buf["total"]
    )
    kkt_inv_uw_cutedsl(
        KL,
        KR,
        KG,
        v3,
        buf["U"],
        buf["W"],
        beta3,
        buf["cu"],
        buf["ci"],
        buf["tc"],
        num_sms=buf["num_sms"],
    )
    kda_h_cutedsl(
        KR,
        buf["U"],
        buf["W"],
        buf["V_new"],
        g_cu,
        buf["h_chunks"],
        buf["h0"],
        buf["ht"],
        buf["cu"],
        buf["co"],
    )
    kda_o_cutedsl(
        qg,
        qg2,
        KR,
        buf["V_new"],
        buf["h_chunks"],
        buf["o"],
        buf["cu"],
        buf["ci"],
        buf["tc"],
        num_sms=buf["num_sms"],
    )
    return buf["o"], buf["ht"]


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


def check_shape(T, H, K, V, device, dtype, num_sms):
    tag = f"T={T:>5} H={H:>2} K={K:>3} V={V:>3}"
    if K != 128 or V != 128:
        print(f"  [SKIP] {tag}  (cutedsl requires K=V=128)")
        return True

    scale = K**-0.5
    inp = make_inputs(T, H, K, V, device, dtype)
    o_ref, state_ref = run_recurrent(inp, scale)

    try:
        buf = cutedsl_buffers(inp, num_sms, device)
        o, ht = run_cutedsl_pipeline(inp, buf, scale)
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        print(f"  [SKIP] {tag}  (cutedsl error: {e})")
        return True

    finite = bool(torch.isfinite(o).all() and torch.isfinite(ht).all())
    o_err = (o.float() - o_ref.float()).abs().max().item()
    s_err = (ht.float() - state_ref.float()).abs().max().item()
    ok = finite and o_err < 1e-2 and s_err < 5e-2
    status = "PASS" if ok else "FAIL"
    print(
        f"  [{status}] {tag} | o_err {o_err:.2e}  state_err {s_err:.2e}  finite={finite}"
    )
    return ok


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def bench_shape(T, H, K, V, device, dtype, num_sms):
    import triton.testing

    if K != 128 or V != 128:
        print(f"  [SKIP] T={T} H={H} K={K} V={V} (cutedsl K=V=128 only)")
        return

    scale = K**-0.5
    inp = make_inputs(T, H, K, V, device, dtype)
    q, k, v = inp["q"], inp["k"], inp["v"]
    g_act, beta = inp["g_act"], inp["beta"]

    h0f = torch.zeros(1, H, K, V, device=device, dtype=torch.float32)
    idx = torch.zeros(1, dtype=torch.int32, device=device)

    def fn_triton():
        chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g_act,
            beta=beta,
            scale=scale,
            initial_state=h0f,
            initial_state_indices=idx,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=None,
            A_log=None,
            dt_bias=None,
            lower_bound=None,
        )

    buf = cutedsl_buffers(inp, num_sms, device)

    def fn_cutedsl():
        run_cutedsl_pipeline(inp, buf, scale)

    quantiles = [0.5, 0.2, 0.8]
    fn_triton()
    fn_cutedsl()
    torch.cuda.synchronize()

    ms_triton, _, _ = triton.testing.do_bench_cudagraph(fn_triton, quantiles=quantiles)
    ms_cutedsl, _, _ = triton.testing.do_bench_cudagraph(
        fn_cutedsl, quantiles=quantiles
    )

    flops = kda_flops(T, H, K, V)
    mem_bytes = kda_bytes(T, H, K, V, 1, dtype)
    speedup = ms_triton / ms_cutedsl if ms_cutedsl > 0 else float("inf")
    print(
        f"  {H:>3}  {T:>7} | "
        f"{ms_triton:>8.3f}  {flops / ms_triton / 1e9:>7.2f}  {mem_bytes / ms_triton / 1e9:>7.2f} | "
        f"{ms_cutedsl:>8.3f}  {flops / ms_cutedsl / 1e9:>7.2f}  {mem_bytes / ms_cutedsl / 1e9:>7.2f} | "
        f"{speedup:>7.2f}x"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_correctness(device, dtype, H, num_sms):
    print("=" * 72)
    print("Correctness: cutedsl pipeline vs fused_recurrent_kda (ground truth)")
    print("=" * 72)
    all_pass = True
    for T in (128, 192, 256, 512, 1024):
        if not check_shape(T, H, 128, 128, device, dtype, num_sms):
            all_pass = False
    print("\nALL PASSED." if all_pass else "\nSOME FAILED.")
    return all_pass


def run_benchmark(device, dtype, args, num_sms):
    print()
    print("=" * 92)
    print("Benchmark: Triton chunk_kda vs CuTeDSL pipeline  (do_bench_cudagraph)")
    print("=" * 92)
    print(f"  Device SMs={num_sms}, K=V=128, dtype={dtype}, metadata precomputed")
    print(
        f"  {'H':>3}  {'T':>7} | "
        f"{'tri(ms)':>8}  {'TFLOP':>7}  {'TB/s':>7} | "
        f"{'cute(ms)':>8}  {'TFLOP':>7}  {'TB/s':>7} | {'speedup':>8}"
    )
    print("  " + "-" * 84)
    for H in args.num_heads:
        for T in args.seq_lens:
            bench_shape(T, H, 128, 128, device, dtype, num_sms)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark & Correctness: Triton KDA vs CuTeDSL KDA (SM100)"
    )
    parser.add_argument(
        "--mode", choices=["all", "correctness", "bench"], default="all"
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--num-heads", type=int, nargs="+", default=[32])
    parser.add_argument(
        "--seq-lens", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192]
    )
    args = parser.parse_args()

    device = "cuda"
    dtype = getattr(torch, args.dtype)
    cap = torch.cuda.get_device_capability()
    print(f"Device: {torch.cuda.get_device_name()}  (SM {cap[0]}{cap[1]})")
    if cap[0] < 10:
        print("ERROR: CuTeDSL KDA prefill requires SM100+ (Blackwell). Exiting.")
        return 1
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count

    if args.mode in ("all", "correctness"):
        all_pass = run_correctness(device, dtype, args.num_heads[0], num_sms)
        if not all_pass and args.mode == "all":
            print("\nSkipping benchmark due to correctness failures.")
            return 1

    if args.mode in ("all", "bench"):
        run_benchmark(device, dtype, args, num_sms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
