#!/usr/bin/env python3
"""
Compare CuTe DSL K-last GDN verify kernel vs. Triton fused_recurrent reference.

This is a correctness-focused test that compares:
1) core attention output o: [B, T, HV, V]
2) intermediate state cache layout/content (after converting K-last <-> V-last)
"""

import argparse

import torch

from sglang.jit_kernel.cutedsl_gdn_verify import cutedsl_gdn_verify_k_last
from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.srt.layers.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_update,
)


def run_one(
    B: int,
    T: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    seed: int,
    use_qk_l2norm: bool = True,
) -> None:
    assert K == V, "This test targets Qwen3-Next verify config where K == V == 128"
    device = "cuda"

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Inputs (match server behavior: bf16 activations, fp32 SSM states)
    q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, T, HV, V, device=device, dtype=torch.bfloat16)
    a = torch.randn(B, T, HV, device=device, dtype=torch.bfloat16)
    b = torch.randn(B, T, HV, device=device, dtype=torch.bfloat16)
    A_log = torch.randn(HV, device=device, dtype=torch.float32)
    dt_bias = torch.randn(HV, device=device, dtype=torch.float32)

    pool_size = B + 1  # typical: size == max_running_requests, plus one padding slot
    h0_v_last = torch.randn(pool_size, HV, K, V, device=device, dtype=torch.float32).contiguous()
    h0_k_last = h0_v_last.transpose(-1, -2).contiguous()  # [pool, HV, V, K]

    # Indices: map each request to a pool slot (0..B-1), no padding for this test
    cache_indices = torch.arange(B, device=device, dtype=torch.int32)

    # Intermediate state buffers
    # - fused_recurrent writes V-last: [buffer, cache_steps, HV, K, V]
    # - cutedsl writes K-last: [buffer, cache_steps, HV, V, K]
    intermediate_v_last = torch.zeros(pool_size, T, HV, K, V, device=device, dtype=torch.float32)
    intermediate_k_last = torch.zeros(pool_size, T, HV, V, K, device=device, dtype=torch.float32)
    intermediate_state_indices = torch.arange(B, device=device, dtype=torch.int32)

    # Reference gating: fused_gdn_gating expects [tokens, HV]
    a_flat = a.reshape(B * T, HV)
    b_flat = b.reshape(B * T, HV)
    g_log, beta = fused_gdn_gating(A_log, a_flat, b_flat, dt_bias)
    g_log = g_log.squeeze(0).reshape(B, T, HV).contiguous()
    beta = beta.squeeze(0).reshape(B, T, HV).contiguous()

    # Reference output (V-last kernel)
    o_ref = fused_recurrent_gated_delta_rule_update(
        q=q,
        k=k,
        v=v,
        g=g_log,
        beta=beta,
        initial_state_source=h0_v_last,
        initial_state_indices=cache_indices,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
        disable_state_update=True,
        intermediate_states_buffer=intermediate_v_last,
        intermediate_state_indices=intermediate_state_indices,
        cache_steps=T,
    )

    # CuTe DSL output (K-last optimized)
    o_cute = cutedsl_gdn_verify_k_last(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        # NOTE: run_one intentionally does NOT pass g_log/beta (simulates "internal gating").
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=h0_k_last,
        initial_state_indices=cache_indices,
        intermediate_states_buffer=intermediate_k_last,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
        disable_state_update=True,
        cache_steps=T,
    )

    # Compare outputs
    o_ref_max = o_ref.float().abs().max().item()
    o_cute_max = o_cute.float().abs().max().item()
    out_max = (o_cute.float() - o_ref.float()).abs().max().item()
    print(f"o_ref max_abs: {o_ref_max:.6e}")
    print(f"o_cute max_abs: {o_cute_max:.6e}")
    print(f"output max_abs_diff: {out_max:.6e}")

    # Compare intermediate states (convert K-last -> V-last)
    intermediate_k_to_v = intermediate_k_last.transpose(-1, -2).contiguous()
    inter_max = (intermediate_k_to_v - intermediate_v_last).abs().max().item()
    print(f"intermediate max_abs_diff: {inter_max:.6e}")

    # Sanity: output should equal h @ q for a representative element (n=0, t=0, hv=0).
    # Compute q normalization the same way both kernels do (L2-norm + scale).
    scale = K ** -0.5
    q0 = q[0, 0, 0].float()
    if use_qk_l2norm:
        q0 = q0 / torch.sqrt((q0 * q0).sum() + 1e-6)
    q0 = q0 * scale

    o_ref0 = o_ref[0, 0, 0].float()
    o_cute0 = o_cute[0, 0, 0].float()
    o_from_v = (intermediate_v_last[0, 0, 0] * q0[:, None]).sum(dim=0).to(torch.bfloat16).float()
    o_from_k = (intermediate_k_last[0, 0, 0] * q0[None, :]).sum(dim=1).to(torch.bfloat16).float()

    print(f"sanity max_abs(o_ref - h@q): {(o_ref0 - o_from_v).abs().max().item():.6e}")
    print(f"sanity max_abs(o_cute - h@q): {(o_cute0 - o_from_k).abs().max().item():.6e}")

    # Check whether the reference kernel's output corresponds to PRE-update or POST-update state.
    # Define:
    #   pre_state_t = exp(g_t) * h_{t-1}
    #   post_state_t = h_t (stored in intermediate buffer at step t)
    # If o_ref matches pre_state_t @ q_t, then the CuTe kernel must compute output before state update.
    hv0 = 0
    i_h0 = hv0 // (HV // H)
    for t0 in range(min(T, 3)):
        qt = q[0, t0, i_h0].float()
        if use_qk_l2norm:
            qt = qt / torch.sqrt((qt * qt).sum() + 1e-6)
        qt = qt * scale

        g_t = g_log[0, t0, hv0]
        decay = torch.exp(g_t)
        if t0 == 0:
            h_prev = h0_v_last[0, hv0]
        else:
            h_prev = intermediate_v_last[0, t0 - 1, hv0]
        pre_state = h_prev * decay
        post_state = intermediate_v_last[0, t0, hv0]

        o_pre = (pre_state * qt[:, None]).sum(dim=0).to(torch.bfloat16).float()
        o_post = (post_state * qt[:, None]).sum(dim=0).to(torch.bfloat16).float()

        o_ref_t = o_ref[0, t0, hv0].float()
        diff_pre = (o_ref_t - o_pre).abs().max().item()
        diff_post = (o_ref_t - o_post).abs().max().item()
        print(f"t={t0} max_abs(o_ref - pre_state@q): {diff_pre:.6e}")
        print(f"t={t0} max_abs(o_ref - post_state@q): {diff_post:.6e}")

    # Infer q_norm from outputs via solve (h^T q = o) for (n=0, t=0, hv=0).
    # This helps confirm whether the fused_recurrent output is using a different q normalization.
    try:
        h_mat = intermediate_v_last[0, 0, hv0].float()  # [K, V]
        o_ref_vec = o_ref[0, 0, hv0].float()            # [V]
        o_cute_vec = o_cute[0, 0, hv0].float()          # [V]
        q_from_ref = torch.linalg.solve(h_mat.t(), o_ref_vec[:, None]).squeeze(-1)   # [K]
        q_from_cute = torch.linalg.solve(h_mat.t(), o_cute_vec[:, None]).squeeze(-1) # [K]

        q0 = q[0, 0, i_h0].float()
        if use_qk_l2norm:
            q0 = q0 / torch.sqrt((q0 * q0).sum() + 1e-6)
        q0 = q0 * scale

        def _cos(a, b):
            return (a @ b) / (torch.norm(a) * torch.norm(b) + 1e-12)

        print(f"cos(q_from_ref, q0): {_cos(q_from_ref, q0).item():.6f}")
        print(f"cos(q_from_cute, q0): {_cos(q_from_cute, q0).item():.6f}")
        print(f"cos(q_from_ref, q_from_cute): {_cos(q_from_ref, q_from_cute).item():.6f}")
        print(f"max_abs(q_from_ref - q0): {(q_from_ref - q0).abs().max().item():.6e}")
        print(f"max_abs(q_from_cute - q0): {(q_from_cute - q0).abs().max().item():.6e}")
    except Exception as e:
        print(f"solve check skipped: {type(e).__name__}: {e}")


@torch.no_grad()
def ablate_one(
    B: int,
    T: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    seed: int,
    *,
    use_qk_l2norm: bool = True,
    proj_dim: int = 256,
) -> dict:
    """Return per-variant precision metrics for a single random seed."""
    device = "cuda"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Inputs (match server behavior: bf16 activations, fp32 SSM states)
    q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, T, HV, V, device=device, dtype=torch.bfloat16)
    a = torch.randn(B, T, HV, device=device, dtype=torch.bfloat16)
    b = torch.randn(B, T, HV, device=device, dtype=torch.bfloat16)
    A_log = torch.randn(HV, device=device, dtype=torch.float32)
    dt_bias = torch.randn(HV, device=device, dtype=torch.float32)

    pool_size = B + 1
    h0_v_last = torch.randn(pool_size, HV, K, V, device=device, dtype=torch.float32).contiguous()
    h0_k_last = h0_v_last.transpose(-1, -2).contiguous()
    cache_indices = torch.arange(B, device=device, dtype=torch.int32)
    intermediate_state_indices = torch.arange(B, device=device, dtype=torch.int32)

    intermediate_v_last = torch.zeros(pool_size, T, HV, K, V, device=device, dtype=torch.float32)
    intermediate_k_last = torch.zeros(pool_size, T, HV, V, K, device=device, dtype=torch.float32)

    # Reference gating: fused_gdn_gating expects [tokens, HV]
    a_flat = a.reshape(B * T, HV)
    b_flat = b.reshape(B * T, HV)
    g_log, beta = fused_gdn_gating(A_log, a_flat, b_flat, dt_bias)
    g_log = g_log.squeeze(0).reshape(B, T, HV).contiguous()
    beta = beta.squeeze(0).reshape(B, T, HV).contiguous()
    g_decay = torch.exp(g_log).contiguous()

    # Reference output (Triton)
    o_ref = fused_recurrent_gated_delta_rule_update(
        q=q,
        k=k,
        v=v,
        g=g_log,
        beta=beta,
        initial_state_source=h0_v_last,
        initial_state_indices=cache_indices,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
        disable_state_update=True,
        intermediate_states_buffer=intermediate_v_last,
        intermediate_state_indices=intermediate_state_indices,
        cache_steps=T,
    )

    # Fixed random projection for "argmax flip rate" proxy.
    torch.manual_seed(0)
    proj = torch.randn(V, proj_dim, device=device, dtype=torch.float32)
    logits_ref = (o_ref.float().reshape(-1, V) @ proj).contiguous()
    arg_ref = torch.argmax(logits_ref, dim=-1)

    def _metrics(o_cute: torch.Tensor, intermediate_k: torch.Tensor) -> dict:
        out_max = (o_cute.float() - o_ref.float()).abs().max().item()
        inter_k_to_v = intermediate_k.transpose(-1, -2).contiguous()
        inter_max = (inter_k_to_v - intermediate_v_last).abs().max().item()
        logits_c = (o_cute.float().reshape(-1, V) @ proj).contiguous()
        arg_c = torch.argmax(logits_c, dim=-1)
        flips = (arg_c != arg_ref).sum().item()
        total = arg_ref.numel()
        return {
            "out_max_abs": out_max,
            "inter_max_abs": inter_max,
            "exact_bf16": bool(torch.equal(o_cute, o_ref)),
            "argmax_flip": int(flips),
            "argmax_total": int(total),
        }

    results = {"seed": seed, "ref_dtype": str(o_ref.dtype)}

    # Variant 1: CuTe internal gating (without external g_log/beta)
    # After cleanup, beta is always quantized and output is always bf16
    intermediate_k_1 = torch.zeros_like(intermediate_k_last)
    o_cute_internal = cutedsl_gdn_verify_k_last(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=h0_k_last,
        initial_state_indices=cache_indices,
        intermediate_states_buffer=intermediate_k_1,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
        disable_state_update=True,
        cache_steps=T,
    )
    results["cute_internal_gating"] = _metrics(o_cute_internal, intermediate_k_1)

    # Variant 2: CuTe with external (Triton) g_log/beta (current production path)
    intermediate_k_2 = torch.zeros_like(intermediate_k_last)
    o_cute_gbeta = cutedsl_gdn_verify_k_last(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        g_log_in=g_log,
        q=q,
        k=k,
        v=v,
        b=b,
        beta_in=beta,
        initial_state_source=h0_k_last,
        initial_state_indices=cache_indices,
        intermediate_states_buffer=intermediate_k_2,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
        disable_state_update=True,
        cache_steps=T,
    )
    results["cute_external_gbeta"] = _metrics(o_cute_gbeta, intermediate_k_2)

    # Variant 3: external g_log/beta + explicitly pass g_decay_in (should match variant 2)
    intermediate_k_3 = torch.zeros_like(intermediate_k_last)
    o_cute_gbeta_gdecay = cutedsl_gdn_verify_k_last(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        g_log_in=g_log,
        g_decay_in=g_decay,
        q=q,
        k=k,
        v=v,
        b=b,
        beta_in=beta,
        initial_state_source=h0_k_last,
        initial_state_indices=cache_indices,
        intermediate_states_buffer=intermediate_k_3,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
        disable_state_update=True,
        cache_steps=T,
    )
    results["cute_external_gbeta_gdecay_in"] = _metrics(o_cute_gbeta_gdecay, intermediate_k_3)

    return results


@torch.no_grad()
def run_ablation(
    B: int,
    T: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    start_seed: int,
    num_seeds: int,
    *,
    use_qk_l2norm: bool = True,
    proj_dim: int = 256,
) -> None:
    # After cleanup, the following ablation variants were removed (no longer configurable):
    # - cute_internal_gating_no_beta_quant (beta quant always on, critical for accuracy)
    # - cute_external_gbeta_out_fp16 (output always bf16, critical for accuracy)
    # - cute_external_gbeta_l2norm_inv_mul (always use div form)
    # - cute_external_gbeta_kernel_exp (always precompute g_decay in wrapper)
    variants = [
        "cute_internal_gating",
        "cute_external_gbeta",
        "cute_external_gbeta_gdecay_in",
    ]
    agg = {
        name: {"out_max_abs": [], "inter_max_abs": [], "exact": 0, "flip": 0, "total": 0}
        for name in variants
    }

    for i in range(num_seeds):
        seed = start_seed + i
        r = ablate_one(B, T, H, HV, K, V, seed, use_qk_l2norm=use_qk_l2norm, proj_dim=proj_dim)
        for name in variants:
            m = r[name]
            agg[name]["out_max_abs"].append(m["out_max_abs"])
            agg[name]["inter_max_abs"].append(m["inter_max_abs"])
            agg[name]["exact"] += int(m["exact_bf16"])
            agg[name]["flip"] += int(m["argmax_flip"])
            agg[name]["total"] += int(m["argmax_total"])

    print(
        f"ABLATE config: B={B},T={T},H={H},HV={HV},K=V={K}, act=bf16,state=fp32, "
        f"use_qk_l2norm={use_qk_l2norm}, seeds=[{start_seed},{start_seed+num_seeds-1}], proj_dim={proj_dim}"
    )
    print("Columns: variant | exact_rate | out_max_abs(max/mean) | inter_max_abs(max/mean) | argmax_flip_rate")
    for name in variants:
        out_list = agg[name]["out_max_abs"]
        inter_list = agg[name]["inter_max_abs"]
        exact_rate = agg[name]["exact"] / max(1, num_seeds)
        out_max = max(out_list) if out_list else float("nan")
        out_mean = sum(out_list) / len(out_list) if out_list else float("nan")
        inter_max = max(inter_list) if inter_list else float("nan")
        inter_mean = sum(inter_list) / len(inter_list) if inter_list else float("nan")
        flip_rate = agg[name]["flip"] / max(1, agg[name]["total"])
        print(
            f"{name:34s} | {exact_rate:9.2%} | "
            f"{out_max:12.4e}/{out_mean:12.4e} | {inter_max:12.4e}/{inter_mean:12.4e} | {flip_rate:12.4%}"
        )


@torch.no_grad()
def bench_one(
    B: int,
    T: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    seed: int,
    *,
    iters: int = 200,
    warmup: int = 30,
    use_qk_l2norm: bool = True,
):
    """Benchmark Triton verify vs CuTe verify for a single configuration."""
    assert K == V, "This benchmark targets Qwen3-Next verify config where K == V == 128"
    assert HV % H == 0, "HV must be divisible by H"
    device = "cuda"

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Inputs (match server behavior: bf16 activations, fp32 SSM states)
    q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, T, HV, V, device=device, dtype=torch.bfloat16)
    a = torch.randn(B, T, HV, device=device, dtype=torch.bfloat16)
    b = torch.randn(B, T, HV, device=device, dtype=torch.bfloat16)
    A_log = torch.randn(HV, device=device, dtype=torch.float32)
    dt_bias = torch.randn(HV, device=device, dtype=torch.float32)

    pool_size = B + 1  # typical: size == max_running_requests, plus one padding slot
    h0_v_last = torch.randn(pool_size, HV, K, V, device=device, dtype=torch.float32).contiguous()
    h0_k_last = h0_v_last.transpose(-1, -2).contiguous()  # [pool, HV, V, K]

    cache_indices = torch.arange(B, device=device, dtype=torch.int32)
    intermediate_state_indices = torch.arange(B, device=device, dtype=torch.int32)

    intermediate_v_last = torch.empty(pool_size, T, HV, K, V, device=device, dtype=torch.float32)
    intermediate_k_last = torch.empty(pool_size, T, HV, V, K, device=device, dtype=torch.float32)

    # Precompute gating (same as Triton path)
    a_flat = a.reshape(B * T, HV)
    b_flat = b.reshape(B * T, HV)
    g_log, beta = fused_gdn_gating(A_log, a_flat, b_flat, dt_bias)
    g_log = g_log.squeeze(0).reshape(B, T, HV).contiguous()
    beta = beta.squeeze(0).reshape(B, T, HV).contiguous()
    g_decay = torch.exp(g_log).contiguous()

    def _bench(fn):
        # Warmup (also triggers compilation)
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
        return start.elapsed_time(end) / iters

    def triton_verify():
        fused_recurrent_gated_delta_rule_update(
            q=q,
            k=k,
            v=v,
            g=g_log,
            beta=beta,
            initial_state_source=h0_v_last,
            initial_state_indices=cache_indices,
            use_qk_l2norm_in_kernel=use_qk_l2norm,
            disable_state_update=True,
            intermediate_states_buffer=intermediate_v_last,
            intermediate_state_indices=intermediate_state_indices,
            cache_steps=T,
        )

    def cute_verify_precomputed_decay():
        # Kernel-only path: pass g_decay_in so wrapper does not launch torch.exp every call.
        cutedsl_gdn_verify_k_last(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            g_log_in=g_log,
            g_decay_in=g_decay,
            q=q,
            k=k,
            v=v,
            b=b,
            beta_in=beta,
            initial_state_source=h0_k_last,
            initial_state_indices=cache_indices,
            intermediate_states_buffer=intermediate_k_last,
            use_qk_l2norm_in_kernel=use_qk_l2norm,
            disable_state_update=True,
            cache_steps=T,
        )

    def cute_verify_wrapper_exp():
        # Match current hybrid backend call pattern: pass g_log/beta, wrapper computes exp(g_log).
        cutedsl_gdn_verify_k_last(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            g_log_in=g_log,
            q=q,
            k=k,
            v=v,
            b=b,
            beta_in=beta,
            initial_state_source=h0_k_last,
            initial_state_indices=cache_indices,
            intermediate_states_buffer=intermediate_k_last,
            use_qk_l2norm_in_kernel=use_qk_l2norm,
            disable_state_update=True,
            cache_steps=T,
        )

    def cute_verify_internal_gating():
        # Simulate "revert accuracy changes": do NOT reuse fused_gdn_gating outputs.
        # Let the CuTe kernel compute g/beta internally.
        cutedsl_gdn_verify_k_last(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=h0_k_last,
            initial_state_indices=cache_indices,
            intermediate_states_buffer=intermediate_k_last,
            use_qk_l2norm_in_kernel=use_qk_l2norm,
            disable_state_update=True,
            cache_steps=T,
        )

    def gating_only():
        fused_gdn_gating(A_log, a_flat, b_flat, dt_bias)

    print(f"BENCH config: B={B}, T={T}, H={H}, HV={HV}, K=V={K}, act=bf16, state=fp32")
    ms_gating = _bench(gating_only)
    ms_triton = _bench(triton_verify)
    ms_cute_decay = _bench(cute_verify_precomputed_decay)
    ms_cute_wrapper = _bench(cute_verify_wrapper_exp)
    ms_cute_internal = _bench(cute_verify_internal_gating)
    print(f"gating (fused_gdn_gating) avg: {ms_gating:.4f} ms")
    print(f"Triton verify (fused_recurrent) avg: {ms_triton:.4f} ms")
    print(f"CuTe verify (cutedsl, g_decay_in precomputed) avg: {ms_cute_decay:.4f} ms")
    print(f"CuTe verify (cutedsl, wrapper exp(g_log)) avg: {ms_cute_wrapper:.4f} ms")
    print(f"CuTe verify (cutedsl, internal gating) avg: {ms_cute_internal:.4f} ms")

    # End-to-end verify-step cost comparison
    triton_total = ms_gating + ms_triton
    cute_total = ms_gating + ms_cute_wrapper
    print(f"Triton total (gating + verify): {triton_total:.4f} ms")
    print(f"CuTe total (gating + verify): {cute_total:.4f} ms")
    delta = ms_cute_internal - cute_total
    pct = (delta / cute_total) * 100
    print(f"Revert-accuracy (internal gating) vs current-CuTe total: {delta:+.4f} ms ({pct:+.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Defaults: Qwen3-Next (tp=2) verify config, and MTP 3/1/4 => draft_token_num(T)=4.
    parser.add_argument("--B", type=int, default=8)
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--HV", type=int, default=16)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--V", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-qk-l2norm", action="store_true")
    parser.add_argument("--bench", action="store_true", help="Benchmark CuTe vs Triton (no correctness prints)")
    parser.add_argument("--bench-iters", type=int, default=200)
    parser.add_argument("--bench-warmup", type=int, default=30)
    parser.add_argument("--ablate", action="store_true", help="Run precision ablation across kernel variants")
    parser.add_argument("--ablate-start-seed", type=int, default=0)
    parser.add_argument("--ablate-num-seeds", type=int, default=20)
    parser.add_argument("--ablate-proj-dim", type=int, default=256)
    args = parser.parse_args()
    if args.bench:
        bench_one(
            args.B,
            args.T,
            args.H,
            args.HV,
            args.K,
            args.V,
            args.seed,
            iters=args.bench_iters,
            warmup=args.bench_warmup,
            use_qk_l2norm=not args.no_qk_l2norm,
        )
    elif args.ablate:
        run_ablation(
            args.B,
            args.T,
            args.H,
            args.HV,
            args.K,
            args.V,
            args.ablate_start_seed,
            args.ablate_num_seeds,
            use_qk_l2norm=not args.no_qk_l2norm,
            proj_dim=args.ablate_proj_dim,
        )
    else:
        run_one(
            args.B,
            args.T,
            args.H,
            args.HV,
            args.K,
            args.V,
            args.seed,
            use_qk_l2norm=not args.no_qk_l2norm,
        )

