"""
Benchmark & Correctness: FlashInfer KDA (SM100) vs Triton KDA — decode & MTP verify.

Exercises the two real backend wrappers used by ``KDAKernelDispatcher``:
  - ``FlashInferKDAKernel`` — wraps ``flashinfer.kda_decode.recurrent_kda``
    (CuTe DSL, SM100/Blackwell only). Provides ``decode`` + ``target_verify``.
  - ``TritonKDAKernel``     — wraps ``fused_sigmoid_gating_delta_rule_update``
    (IS_KDA=True). Reference for both ``decode`` and ``target_verify``.

Two modes:
  - decode : single-token decode (T=1), in-place SSM update.
  - verify : MTP / speculative-decode ``target_verify`` over T=1+num_spec draft
             tokens per sequence, writing per-token states into the speculative
             ``intermediate_ssm`` scratch (the recurrent_kda adapter / the Triton
             intermediate_states_buffer path).

Reports correctness (output vs the Triton reference) and performance (us, speedup).
Requires an SM100 GPU + a FlashInfer build exposing ``recurrent_kda``; on other
GPUs the FlashInfer side is skipped and only the Triton path is timed.

Usage:
    python bench_kda_flashinfer_mtp.py                 # decode+verify, correctness+bench
    python bench_kda_flashinfer_mtp.py --mode bench --task verify
    python bench_kda_flashinfer_mtp.py --num-spec 7    # 8 draft tokens / verify step
    python bench_kda_flashinfer_mtp.py --task decode --helion
"""

import argparse

import torch

from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel


def _make_flashinfer_kernel():
    """Instantiate FlashInferKDAKernel, or None if unavailable (non-SM100)."""
    try:
        from sglang.srt.layers.attention.linear.kernels.kda_flashinfer import (
            FlashInferKDAKernel,
        )

        return FlashInferKDAKernel()
    except Exception as e:  # noqa: BLE001 - report and degrade gracefully
        print(f"  [skip flashinfer] {type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------
def make_decode_inputs(B, H, HV, K, V, pool_size, device, dtype, seed=42):
    torch.manual_seed(seed)
    q = torch.randn(1, B, H, K, device=device, dtype=dtype) * 0.5
    k = torch.randn(1, B, H, K, device=device, dtype=dtype) * 0.5
    v = torch.randn(1, B, HV, V, device=device, dtype=dtype) * 0.5
    a = torch.randn(B, HV * K, device=device, dtype=dtype) * 0.5 - 1.0  # raw per-K gate
    b = torch.randn(B, HV, device=device, dtype=dtype) * 0.5  # beta LOGIT
    A_log = torch.randn(HV, device=device, dtype=torch.float32) * 0.2
    dt_bias = torch.randn(HV * K, device=device, dtype=torch.float32) * 0.1
    ssm = torch.randn(pool_size, HV, V, K, device=device, dtype=dtype) * 0.01
    cache_indices = torch.arange(B, device=device, dtype=torch.int32)
    qsl = torch.arange(B + 1, device=device, dtype=torch.int32)
    return dict(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        a=a.contiguous(),
        b=b.contiguous(),
        A_log=A_log,
        dt_bias=dt_bias,
        ssm=ssm.contiguous(),
        cache_indices=cache_indices,
        qsl=qsl,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
    )


def make_verify_inputs(B, T, H, HV, K, V, pool_size, device, dtype, seed=42):
    torch.manual_seed(seed)
    seq = B * T
    q = torch.randn(1, seq, H, K, device=device, dtype=dtype) * 0.5
    k = torch.randn(1, seq, H, K, device=device, dtype=dtype) * 0.5
    v = torch.randn(1, seq, HV, V, device=device, dtype=dtype) * 0.5
    a = torch.randn(seq, HV * K, device=device, dtype=dtype) * 0.5 - 1.0
    b = torch.randn(seq, HV, device=device, dtype=dtype) * 0.5
    A_log = torch.randn(HV, device=device, dtype=torch.float32) * 0.2
    dt_bias = torch.randn(HV * K, device=device, dtype=torch.float32) * 0.1
    ssm = torch.randn(pool_size, HV, V, K, device=device, dtype=dtype) * 0.01
    cache_indices = torch.arange(B, device=device, dtype=torch.int32)
    qsl = torch.arange(0, seq + 1, T, device=device, dtype=torch.int32)
    # speculative intermediate_ssm scratch: [n_scratch, T, HV, V, K]; per-request row.
    intermediate_states = torch.zeros(B, T, HV, V, K, device=device, dtype=dtype)
    intermediate_indices = torch.arange(B, device=device, dtype=torch.int32)
    return dict(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        a=a.contiguous(),
        b=b.contiguous(),
        A_log=A_log,
        dt_bias=dt_bias,
        ssm=ssm.contiguous(),
        cache_indices=cache_indices,
        qsl=qsl,
        intermediate_states=intermediate_states.contiguous(),
        intermediate_indices=intermediate_indices,
        B=B,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        seq=seq,
    )


# ---------------------------------------------------------------------------
# Calls (fresh state clone each time so timing/correctness are independent)
# ---------------------------------------------------------------------------
def call_decode(kernel, inp, ssm):
    # `ssm` is the (mutable, updated in-place) committed-state buffer the caller owns
    # — cloned fresh for correctness, reused across timed iters (latency is unchanged
    # by accumulated state; cloning a ~100s-of-MB pool every call would dominate).
    out = kernel.decode(
        inp["q"],
        inp["k"],
        inp["v"],
        inp["a"],
        inp["b"],
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        ssm_states=ssm,
        cache_indices=inp["cache_indices"],
        query_start_loc=inp["qsl"],
    )
    return out.reshape(inp["B"], inp["HV"], inp["V"]).float()


def call_packed_triton(kernel, inp, ssm):
    out = kernel.packed_decode(
        inp["mixed_qkv"],
        inp["a"],
        inp["b"],
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        scale=inp["K"] ** -0.5,
        ssm_states=ssm,
        cache_indices=inp["cache_indices"],
        num_v_heads=inp["HV"],
        head_v_dim=inp["V"],
    )
    return out.reshape(inp["B"], inp["HV"], inp["V"]).float()


def call_helion(helion_decode, inp, ssm):
    out = inp["mixed_qkv"].new_empty(inp["B"], 1, inp["HV"], inp["V"])
    helion_decode(
        mixed_qkv=inp["mixed_qkv"],
        a=inp["a"],
        b=inp["b"],
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        scale=inp["K"] ** -0.5,
        initial_state=ssm,
        out=out,
        ssm_state_indices=inp["cache_indices"],
        use_qk_l2norm_in_kernel=True,
    )
    return out.reshape(inp["B"], inp["HV"], inp["V"]).float()


def call_verify(kernel, inp, ssm, intermediate_states):
    out = kernel.target_verify(
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        q=inp["q"],
        k=inp["k"],
        v=inp["v"],
        a=inp["a"],
        b=inp["b"],
        ssm_states=ssm,
        cache_indices=inp["cache_indices"],
        query_start_loc=inp["qsl"],
        intermediate_states_buffer=intermediate_states,
        intermediate_state_indices=inp["intermediate_indices"],
        cache_steps=inp["T"],
        retrieve_parent_token=None,
    )
    return out.reshape(inp["seq"], inp["HV"], inp["V"]).float()


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def _time(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def run_decode_all(fi, tri, helion_decode, device, dtype, args):
    width = 124
    print("=" * width)
    print(
        "KDA decode (T=1): generic Triton vs FlashInfer vs packed Triton vs "
        f"Helion  |  K={args.head_k} V={args.head_v} dtype={dtype}"
    )
    print("=" * width)
    print(
        f"  {'B':>6}  {'H':>3}  {'HV':>3} | {'generic(us)':>11} | "
        f"{'flashinfer(us)':>14} | {'packed(us)':>11} | {'helion(us)':>11} | "
        f"{'FI/H':>7} | {'packed/H':>9} | {'H max diff':>10}"
    )
    print("  " + "-" * 118)

    for B in args.batch_sizes:
        for H in args.num_q_heads:
            for HV in args.num_v_heads:
                if HV % H != 0:
                    continue
                K, V = args.head_k, args.head_v
                pool = max(args.pool_size, B + 16)
                inp = make_decode_inputs(B, H, HV, K, V, pool, device, dtype)
                inp["mixed_qkv"] = torch.cat(
                    (
                        inp["q"].reshape(B, H * K),
                        inp["k"].reshape(B, H * K),
                        inp["v"].reshape(B, HV * V),
                    ),
                    dim=-1,
                ).contiguous()

                generic_ref_state = inp["ssm"].clone()
                packed_ref_state = inp["ssm"].clone()
                helion_ref_state = inp["ssm"].clone()
                o_generic = call_decode(tri, inp, generic_ref_state)
                o_packed = call_packed_triton(tri, inp, packed_ref_state)
                o_helion = call_helion(helion_decode, inp, helion_ref_state)
                helion_diff = max(
                    (o_helion - o_packed).abs().max().item(),
                    (helion_ref_state - packed_ref_state).abs().max().item(),
                )
                if fi is not None:
                    fi_ref_state = inp["ssm"].clone()
                    o_fi = call_decode(fi, inp, fi_ref_state)
                    fi_diff = max(
                        (o_fi - o_generic).abs().max().item(),
                        (fi_ref_state - generic_ref_state).abs().max().item(),
                    )
                    if fi_diff > 2e-2:
                        raise AssertionError(
                            f"FlashInfer contract mismatch at B={B}: {fi_diff}"
                        )
                if helion_diff > 2e-2:
                    raise AssertionError(
                        f"Helion contract mismatch at B={B}: {helion_diff}"
                    )

                generic_state = inp["ssm"].clone()
                packed_state = inp["ssm"].clone()
                helion_state = inp["ssm"].clone()
                fi_state = inp["ssm"].clone()
                ms_generic = _time(lambda: call_decode(tri, inp, generic_state))
                ms_fi = (
                    _time(lambda: call_decode(fi, inp, fi_state))
                    if fi is not None
                    else float("nan")
                )
                ms_packed = _time(lambda: call_packed_triton(tri, inp, packed_state))
                ms_helion = _time(lambda: call_helion(helion_decode, inp, helion_state))

                fi_us = f"{ms_fi * 1000:>14.1f}" if fi is not None else f"{'skip':>14}"
                h_fi = f"{ms_fi / ms_helion:>6.2f}x" if fi is not None else f"{'-':>7}"
                print(
                    f"  {B:>6}  {H:>3}  {HV:>3} | "
                    f"{ms_generic * 1000:>11.1f} | {fi_us} | "
                    f"{ms_packed * 1000:>11.1f} | "
                    f"{ms_helion * 1000:>11.1f} | {h_fi} | "
                    f"{ms_packed / ms_helion:>8.2f}x | {helion_diff:>10.2e}"
                )


def run(task, fi, tri, device, dtype, args):
    if task == "decode" and args.helion_decode is not None:
        run_decode_all(
            fi,
            tri,
            args.helion_decode,
            device,
            dtype,
            args,
        )
        return
    is_verify = task == "verify"
    T = 1 + args.num_spec if is_verify else 1
    title = f"target_verify (MTP, T={T})" if is_verify else "decode (T=1)"
    print("=" * 92)
    print(
        f"KDA {title}: FlashInfer (SM100) vs Triton  |  K={args.head_k} V={args.head_v} dtype={dtype}"
    )
    print("=" * 92)
    hdr = "B" if not is_verify else "B(xT)"
    print(
        f"  {hdr:>6}  {'H':>3}  {'HV':>3} | {'triton(us)':>11} | "
        f"{'flashinfer(us)':>14} | {'speedup':>8} | {'out_max_diff':>12}"
    )
    print("  " + "-" * 86)

    for B in args.batch_sizes:
        for H in args.num_q_heads:
            for HV in args.num_v_heads:
                if HV % H != 0:
                    continue
                K, V = args.head_k, args.head_v
                pool = max(args.pool_size, B + 16)
                if is_verify:
                    inp = make_verify_inputs(B, T, H, HV, K, V, pool, device, dtype)
                    corr = lambda kern: call_verify(  # noqa: E731
                        kern,
                        inp,
                        inp["ssm"].clone(),
                        inp["intermediate_states"].clone(),
                    )
                    ssm_t, intermediate_states_t = (
                        inp["ssm"].clone(),
                        inp["intermediate_states"].clone(),
                    )

                    def timed(kern):
                        return call_verify(kern, inp, ssm_t, intermediate_states_t)

                else:
                    inp = make_decode_inputs(B, H, HV, K, V, pool, device, dtype)
                    corr = lambda kern: call_decode(
                        kern, inp, inp["ssm"].clone()
                    )  # noqa: E731
                    ssm_t = inp["ssm"].clone()

                    def timed(kern):
                        return call_decode(kern, inp, ssm_t)

                o_tri = corr(tri)
                diff = "n/a"
                if fi is not None:
                    o_fi = corr(fi)
                    diff = f"{(o_fi - o_tri).abs().max().item():.2e}"

                ms_tri = _time(lambda: timed(tri))
                ms_fi = _time(lambda: timed(fi)) if fi is not None else float("nan")
                speed = (
                    (ms_tri / ms_fi) if fi is not None and ms_fi > 0 else float("nan")
                )
                fi_us = f"{ms_fi * 1000:>14.1f}" if fi is not None else f"{'skip':>14}"
                sp = f"{speed:>7.2f}x" if fi is not None else f"{'-':>8}"
                print(
                    f"  {B:>6}  {H:>3}  {HV:>3} | {ms_tri * 1000:>11.1f} | "
                    f"{fi_us} | {sp} | {diff:>12}"
                )


def load_helion_decode(enabled):
    if not enabled:
        return None
    from sglang.kernels.ops.attention.helion.kda_decode import (
        helion_fused_recurrent_kda_packed_decode,
    )

    return helion_fused_recurrent_kda_packed_decode


def main():
    p = argparse.ArgumentParser(
        description="Benchmark FlashInfer vs Triton KDA decode/verify"
    )
    p.add_argument("--task", choices=["decode", "verify", "all"], default="all")
    p.add_argument(
        "--mode", choices=["all", "bench"], default="all"
    )  # correctness inlined
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--head-k", type=int, default=128)
    p.add_argument("--head-v", type=int, default=128)
    p.add_argument("--pool-size", type=int, default=512)
    p.add_argument(
        "--num-spec", type=int, default=7, help="draft tokens = 1 + num_spec"
    )
    p.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 32, 64, 128]
    )
    p.add_argument("--num-q-heads", type=int, nargs="+", default=[16])
    p.add_argument("--num-v-heads", type=int, nargs="+", default=[16])
    p.add_argument(
        "--helion",
        action="store_true",
        help="Include SGLang's optional Helion packed KDA backend.",
    )
    args = p.parse_args()
    args.helion_decode = load_helion_decode(args.helion)

    device, dtype = "cuda", getattr(torch, args.dtype)
    cap = torch.cuda.get_device_capability()
    print(f"Device: {torch.cuda.get_device_name()}  (SM {cap[0]}{cap[1]})")

    fi = _make_flashinfer_kernel()
    tri = TritonKDAKernel()

    tasks = ["decode", "verify"] if args.task == "all" else [args.task]
    for t in tasks:
        run(t, fi, tri, device, dtype, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
