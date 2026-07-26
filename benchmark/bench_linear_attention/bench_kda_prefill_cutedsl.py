"""
Benchmark & Correctness: Triton, CuTeDSL, and optional Helion KDA prefill.

Compares with --helion:
  - Triton: SGLang's full packed-varlen chunk_kda production operator
  - CuTeDSL core: kda_blackwell pipeline with preprocessing outside the timer
  - Helion: SGLang's full packed-varlen optional Helion production operator

KDA differs from GDN by a PER-CHANNEL decay gate (g is [T, H, K], not scalar).
Triton and Helion receive raw q/k and gate tensors, and include q/k L2
normalization and gate activation in the timed region just as the serving
adapter does. The cutedsl pipeline receives normalized q/k and an activated
gate because its current public entry point externalizes that preprocessing.
Its result is therefore a core-kernel ceiling, not an end-to-end comparison.
Packed-varlen metadata is precomputed for all paths, matching serving where it
is supplied by the scheduler. Without --helion, the script retains its original
preactivated Triton and CuTeDSL core comparison.

Correctness is checked against the token-by-token fused_recurrent_kda ground
truth. Reports performance (ms, approx TFLOPS, TB/s, speedup).

Usage:
    python bench_kda_prefill_cutedsl.py                    # default sweep
    python bench_kda_prefill_cutedsl.py --mode bench       # benchmark only
    python bench_kda_prefill_cutedsl.py --mode correctness # correctness only
    python bench_kda_prefill_cutedsl.py --helion           # include Helion
    python bench_kda_prefill_cutedsl.py --helion --newton-schulz
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.fla.kda import chunk_kda, fused_recurrent_kda
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

BT = 64  # chunk size

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_helion_chunk_kda(enabled):
    if not enabled:
        return None
    from sglang.kernels.ops.attention.helion.kda_prefill import chunk_kda

    return chunk_kda


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
    q_raw = torch.randn(1, T, H, K, device=device, dtype=dtype)
    k_raw = torch.randn(1, T, H, K, device=device, dtype=dtype)
    q = _l2norm(q_raw).to(dtype)
    k = _l2norm(k_raw).to(dtype)
    v = torch.randn(1, T, H, V, device=device, dtype=dtype)
    # Mild per-channel gate (real Kimi-Linear regime; keeps exp() in fp32 range).
    A_log = torch.randn(H, device=device) * 0.5 - 1.5
    dt_bias = torch.randn(H, K, device=device) * 0.1
    g_raw = torch.randn(1, T, H, K, device=device, dtype=dtype)
    g_act = (
        -A_log.exp().view(1, 1, H, 1)
        * F.softplus(g_raw.float() + dt_bias.view(1, 1, H, K))
    ).float()
    beta = torch.sigmoid(torch.randn(1, T, H, device=device)).float()
    return dict(
        q_raw=q_raw,
        k_raw=k_raw,
        q=q,
        k=k,
        v=v,
        g_raw=g_raw,
        g_act=g_act,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        T=T,
        H=H,
        K=K,
        V=V,
    )


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
        state_indices=torch.zeros(1, device=device, dtype=torch.int32),
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
        buf["state_indices"],
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


def check_shape(
    T,
    H,
    K,
    V,
    device,
    dtype,
    num_sms,
    helion_chunk_kda=None,
    newton_schulz=False,
):
    tag = f"T={T:>5} H={H:>2} K={K:>3} V={V:>3}"
    if K != 128 or V != 128:
        print(f"  [SKIP] {tag}  (cutedsl requires K=V=128)")
        return True

    scale = K**-0.5
    inp = make_inputs(T, H, K, V, device, dtype)
    o_ref, state_ref = run_recurrent(inp, scale)
    cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)

    try:
        buf = cutedsl_buffers(inp, num_sms, device)
        o, ht = run_cutedsl_pipeline(inp, buf, scale)
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        print(f"  [SKIP] {tag}  (cutedsl error: {e})")
        cutedsl_ok = True
    else:
        finite = bool(torch.isfinite(o).all() and torch.isfinite(ht).all())
        o_err = (o.float() - o_ref.float()).abs().max().item()
        s_err = (ht.float() - state_ref.float()).abs().max().item()
        cutedsl_ok = finite and o_err < 1e-2 and s_err < 5e-2
        status = "PASS" if cutedsl_ok else "FAIL"
        print(
            f"  [{status}] CuTeDSL {tag} | o_err {o_err:.2e}  "
            f"state_err {s_err:.2e}  finite={finite}"
        )

    helion_ok = True
    if helion_chunk_kda is not None:
        helion_v = inp["v"].clone()
        helion_state = torch.zeros(1, H, V, K, device=device, dtype=torch.float32)
        state_indices = torch.zeros(1, device=device, dtype=torch.int32)
        helion_o, _ = helion_chunk_kda(
            q=inp["q_raw"],
            k=inp["k_raw"],
            v=helion_v,
            g=inp["g_raw"],
            beta=inp["beta"],
            scale=scale,
            initial_state=helion_state,
            initial_state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
            A_log=inp["A_log"],
            dt_bias=inp["dt_bias"],
            lower_bound=None,
            output_intermediate_states=True,
            newton_schulz=newton_schulz,
        )
        torch.cuda.synchronize()
        finite = bool(
            torch.isfinite(helion_o).all() and torch.isfinite(helion_state).all()
        )
        o_err = (helion_o[0].float() - o_ref.float()).abs().max().item()
        s_err = (helion_state.float() - state_ref.float()).abs().max().item()
        helion_ok = finite and o_err < 1e-2 and s_err < 5e-2
        status = "PASS" if helion_ok else "FAIL"
        helion_name = "Helion-NS" if newton_schulz else "Helion"
        print(
            f"  [{status}] {helion_name:<9} {tag} | o_err {o_err:.2e}  "
            f"state_err {s_err:.2e}  finite={finite}"
        )

    return cutedsl_ok and helion_ok


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def bench_shape(
    T,
    H,
    K,
    V,
    device,
    dtype,
    num_sms,
    helion_chunk_kda=None,
    newton_schulz=False,
):
    import triton.testing

    if K != 128 or V != 128:
        print(f"  [SKIP] T={T} H={H} K={K} V={V} (cutedsl K=V=128 only)")
        return

    scale = K**-0.5
    inp = make_inputs(T, H, K, V, device, dtype)
    v, beta = inp["v"], inp["beta"]

    triton_v = v.clone()
    h0f = torch.zeros(1, H, V, K, device=device, dtype=torch.float32)
    idx = torch.zeros(1, dtype=torch.int32, device=device)
    buf = cutedsl_buffers(inp, num_sms, device)
    cu_seqlens = buf["cu"]

    if helion_chunk_kda is None:
        triton_q, triton_k, triton_g = inp["q"], inp["k"], inp["g_act"]
        triton_cu_seqlens = None
        triton_a_log = None
        triton_dt_bias = None
        triton_l2norm = False
    else:
        triton_q, triton_k, triton_g = (
            inp["q_raw"],
            inp["k_raw"],
            inp["g_raw"],
        )
        triton_cu_seqlens = cu_seqlens
        triton_a_log = inp["A_log"]
        triton_dt_bias = inp["dt_bias"]
        triton_l2norm = True

    def fn_triton():
        chunk_kda(
            q=triton_q,
            k=triton_k,
            v=triton_v,
            g=triton_g,
            beta=beta,
            scale=scale,
            initial_state=h0f,
            initial_state_indices=idx,
            use_qk_l2norm_in_kernel=triton_l2norm,
            cu_seqlens=triton_cu_seqlens,
            A_log=triton_a_log,
            dt_bias=triton_dt_bias,
            lower_bound=None,
        )

    def fn_cutedsl():
        run_cutedsl_pipeline(inp, buf, scale)

    if helion_chunk_kda is not None:
        helion_v = v.clone()
        helion_state = torch.zeros(1, H, V, K, device=device, dtype=torch.float32)

        def fn_helion():
            helion_chunk_kda(
                q=inp["q_raw"],
                k=inp["k_raw"],
                v=helion_v,
                g=inp["g_raw"],
                beta=beta,
                scale=scale,
                initial_state=helion_state,
                initial_state_indices=idx,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
                A_log=inp["A_log"],
                dt_bias=inp["dt_bias"],
                lower_bound=None,
                newton_schulz=newton_schulz,
            )

    quantiles = [0.5, 0.2, 0.8]
    fn_triton()
    fn_cutedsl()
    if helion_chunk_kda is not None:
        fn_helion()
    torch.cuda.synchronize()

    ms_triton, _, _ = triton.testing.do_bench_cudagraph(fn_triton, quantiles=quantiles)
    ms_cutedsl, _, _ = triton.testing.do_bench_cudagraph(
        fn_cutedsl, quantiles=quantiles
    )
    ms_helion = None
    if helion_chunk_kda is not None:
        ms_helion, _, _ = triton.testing.do_bench_cudagraph(
            fn_helion, quantiles=quantiles
        )

    flops = kda_flops(T, H, K, V)
    mem_bytes = kda_bytes(T, H, K, V, 1, dtype)
    speedup = ms_triton / ms_cutedsl if ms_cutedsl > 0 else float("inf")
    line = (
        f"  {H:>3}  {T:>7} | "
        f"{ms_triton:>8.3f}  {flops / ms_triton / 1e9:>7.2f}  {mem_bytes / ms_triton / 1e9:>7.2f} | "
        f"{ms_cutedsl:>8.3f}  {flops / ms_cutedsl / 1e9:>7.2f}  {mem_bytes / ms_cutedsl / 1e9:>7.2f} | "
        f"{speedup:>7.2f}x"
    )
    if ms_helion is not None:
        line += (
            f" | {ms_helion:>8.3f}  {flops / ms_helion / 1e9:>7.2f}  "
            f"{mem_bytes / ms_helion / 1e9:>7.2f} | "
            f"{ms_triton / ms_helion:>7.2f}x"
        )
    print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_correctness(
    device,
    dtype,
    H,
    num_sms,
    helion_chunk_kda=None,
    newton_schulz=False,
):
    print("=" * 72)
    print("Correctness: KDA prefill kernels vs fused_recurrent_kda (ground truth)")
    print("=" * 72)
    all_pass = True
    for T in (128, 192, 256, 512, 1024):
        if not check_shape(
            T,
            H,
            128,
            128,
            device,
            dtype,
            num_sms,
            helion_chunk_kda,
            newton_schulz,
        ):
            all_pass = False
    print("\nALL PASSED." if all_pass else "\nSOME FAILED.")
    return all_pass


def run_benchmark(
    device,
    dtype,
    args,
    num_sms,
    helion_chunk_kda=None,
    newton_schulz=False,
):
    print()
    print("=" * 92)
    if helion_chunk_kda is None:
        print("Benchmark: Triton chunk_kda vs CuTeDSL pipeline  (do_bench_cudagraph)")
    else:
        print("Benchmark: full KDA operators and CuTeDSL core  (do_bench_cudagraph)")
    print("=" * 92)
    if helion_chunk_kda is None:
        print(f"  Device SMs={num_sms}, K=V=128, dtype={dtype}, metadata precomputed")
        header = (
            f"  {'H':>3}  {'T':>7} | "
            f"{'tri(ms)':>8}  {'TFLOP':>7}  {'TB/s':>7} | "
            f"{'cute(ms)':>8}  {'TFLOP':>7}  {'TB/s':>7} | {'speedup':>8}"
        )
    else:
        print(
            f"  Device SMs={num_sms}, K=V=128, dtype={dtype}, "
            "packed-varlen metadata precomputed; "
            "CuTe excludes q/k + gate preprocessing"
        )
        header = (
            f"  {'H':>3}  {'T':>7} | "
            f"{'tri-full':>8}  {'TFLOP':>7}  {'TB/s':>7} | "
            f"{'cute-core':>9}  {'TFLOP':>7}  {'TB/s':>7} | {'tri/cute':>8}"
        )
        helion_label = "hel-ns-full" if newton_schulz else "hel-full"
        header += f" | {helion_label:>10}  {'TFLOP':>7}  {'TB/s':>7} | {'tri/hel':>8}"
    print(header)
    print("  " + "-" * (124 if helion_chunk_kda is not None else 84))
    for H in args.num_heads:
        for T in args.seq_lens:
            bench_shape(
                T,
                H,
                128,
                128,
                device,
                dtype,
                num_sms,
                helion_chunk_kda,
                newton_schulz,
            )


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
    parser.add_argument(
        "--helion",
        action="store_true",
        help="Include SGLang's optional Helion KDA prefill backend.",
    )
    parser.add_argument(
        "--newton-schulz",
        action="store_true",
        help="Use Helion's opt-in Newton-Schulz intra-chunk inverse.",
    )
    args = parser.parse_args()
    if args.newton_schulz and not args.helion:
        parser.error("--newton-schulz requires --helion")
    helion_chunk_kda = load_helion_chunk_kda(args.helion)

    device = "cuda"
    dtype = getattr(torch, args.dtype)
    cap = torch.cuda.get_device_capability()
    print(f"Device: {torch.cuda.get_device_name()}  (SM {cap[0]}{cap[1]})")
    if cap[0] < 10:
        print("ERROR: CuTeDSL KDA prefill requires SM100+ (Blackwell). Exiting.")
        return 1
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count

    if args.mode in ("all", "correctness"):
        all_pass = run_correctness(
            device,
            dtype,
            args.num_heads[0],
            num_sms,
            helion_chunk_kda,
            args.newton_schulz,
        )
        if not all_pass and args.mode == "all":
            print("\nSkipping benchmark due to correctness failures.")
            return 1

    if args.mode in ("all", "bench"):
        run_benchmark(
            device,
            dtype,
            args,
            num_sms,
            helion_chunk_kda,
            args.newton_schulz,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
