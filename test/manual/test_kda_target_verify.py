"""Kernel-level correctness test for KDA target_verify.

Verifies that:
1. target_verify output matches N sequential decode outputs (linear chain)
2. intermediate SSM states match per-step decode SSM states
3. SSM states are not modified in-place (disable_state_update=True)

Usage:
    python test/manual/test_kda_target_verify.py
"""

import torch


def test_kda_target_verify_equivalence():
    """Verify target_verify with linear chain (retrieve_parent_token=None)
    produces the same output as N sequential decode calls."""
    from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )

    B, HV, K, V = 2, 4, 64, 64
    N = 4  # draft tokens
    device = "cuda"
    dtype = torch.float32

    torch.manual_seed(42)
    # Input layout for target_verify: [req0_step0..stepN-1, req1_step0..stepN-1]
    # shape (1, B*N, HV, K)
    q = torch.randn(1, B * N, HV, K, dtype=dtype, device=device)
    k = torch.randn(1, B * N, HV, K, dtype=dtype, device=device)
    v = torch.randn(1, B * N, HV, V, dtype=dtype, device=device)
    # KDA a shape: (B*N, HV*K) — per-token, per-head-K gating
    a = torch.randn(B * N, HV * K, dtype=dtype, device=device)
    # KDA b shape: (1, B*N, HV) — per-token, per-head scalar beta
    # (from model: beta_proj output is (T, num_heads) then unsqueeze(0))
    b = torch.randn(1, B * N, HV, dtype=dtype, device=device)
    A_log = torch.randn(HV, dtype=torch.float32, device=device)
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device)

    num_slots = B + 2  # extra slots for padding
    ssm_states_base = torch.randn(num_slots, HV, K, V, dtype=dtype, device=device)
    cache_indices = torch.arange(B, dtype=torch.int32, device=device)
    # cu_seqlens for target_verify: [0, N, 2N, ..., B*N]
    query_start_loc = torch.arange(0, B * N + 1, N, dtype=torch.int32, device=device)

    # --- Path A: N sequential decode calls ---
    ssm_states_decode = ssm_states_base.clone()
    outputs_decode = []
    states_after_step = []

    for step in range(N):
        # For req i, step j is at position i*N + j in the flat layout
        step_indices = [i * N + step for i in range(B)]
        step_q = q[:, step_indices].contiguous()
        step_k = k[:, step_indices].contiguous()
        step_v = v[:, step_indices].contiguous()
        step_a = a[step_indices].contiguous()
        step_b = b[:, step_indices].contiguous()
        decode_qsl = torch.arange(0, B + 1, dtype=torch.int32, device=device)

        out = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=step_q,
            k=step_k,
            v=step_v,
            a=step_a,
            b=step_b,
            initial_state_source=ssm_states_decode,
            initial_state_indices=cache_indices,
            cu_seqlens=decode_qsl,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=True,
        )
        outputs_decode.append(out)
        # Save per-request states after this step
        states_after_step.append(ssm_states_decode[cache_indices].clone())

    # --- Path B: single target_verify call ---
    ssm_states_verify = ssm_states_base.clone()
    # intermediate buffer shape: (num_slots, N, HV, K, V)
    # This matches the actual per-layer shape from MambaPool.SpeculativeState
    intermediate_buffer = torch.zeros(
        num_slots, N, HV, K, V, dtype=dtype, device=device
    )
    intermediate_indices = torch.arange(B, dtype=torch.int32, device=device)

    out_verify = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        initial_state_source=ssm_states_verify,
        initial_state_indices=cache_indices,
        cu_seqlens=query_start_loc,
        use_qk_l2norm_in_kernel=True,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        is_kda=True,
        disable_state_update=True,
        intermediate_states_buffer=intermediate_buffer,
        intermediate_state_indices=intermediate_indices,
        cache_steps=N,
        retrieve_parent_token=None,
    )

    # --- Compare outputs ---
    # Sequential decode outputs per step: (1, B, HV, V)
    # Concatenate to (1, B*N, HV, V) matching target_verify layout
    out_decode_list = []
    for req_idx in range(B):
        for step in range(N):
            out_decode_list.append(outputs_decode[step][:, req_idx : req_idx + 1])
    out_decode_cat = torch.cat(out_decode_list, dim=1)  # (1, B*N, HV, V)

    max_diff = (out_verify - out_decode_cat).abs().max().item()
    mean_diff = (out_verify - out_decode_cat).abs().mean().item()
    print(f"Output max diff: {max_diff:.6e}, mean diff: {mean_diff:.6e}")
    assert max_diff < 1e-5, f"Output mismatch! max diff: {max_diff}"

    # --- Compare intermediate SSM states ---
    print("Intermediate state comparison:")
    for step in range(N):
        for req_idx in range(B):
            cached_state = intermediate_buffer[req_idx, step]  # (HV, K, V)
            decode_state = states_after_step[step][req_idx]  # (HV, K, V)
            state_diff = (cached_state - decode_state).abs().max().item()
            status = "OK" if state_diff < 1e-5 else "FAIL"
            print(f"  step={step} req={req_idx}: diff={state_diff:.6e} [{status}]")
            assert (
                state_diff < 1e-5
            ), f"Intermediate state mismatch at step={step}, req={req_idx}: {state_diff}"

    # --- Verify SSM states are NOT modified in-place ---
    ssm_unchanged_diff = (ssm_states_verify - ssm_states_base).abs().max().item()
    print(f"SSM state in-place change (should be 0): {ssm_unchanged_diff:.6e}")
    assert (
        ssm_unchanged_diff == 0.0
    ), f"target_verify modified ssm_states in-place! diff: {ssm_unchanged_diff}"

    print("\nPASSED: KDA target_verify matches sequential decode!")


def test_kda_target_verify_bf16():
    """Same test with bfloat16 to verify acceptable numerical tolerance."""
    from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )

    B, HV, K, V = 2, 4, 64, 64
    N = 4
    device = "cuda"
    dtype = torch.bfloat16

    torch.manual_seed(42)
    q = torch.randn(1, B * N, HV, K, dtype=dtype, device=device)
    k = torch.randn(1, B * N, HV, K, dtype=dtype, device=device)
    v = torch.randn(1, B * N, HV, V, dtype=dtype, device=device)
    a = torch.randn(B * N, HV * K, dtype=dtype, device=device)
    # KDA b shape: (1, B*N, HV)
    b = torch.randn(1, B * N, HV, dtype=dtype, device=device)
    A_log = torch.randn(HV, dtype=torch.float32, device=device)
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device)

    num_slots = B + 2
    ssm_states_base = torch.randn(num_slots, HV, K, V, dtype=dtype, device=device)
    cache_indices = torch.arange(B, dtype=torch.int32, device=device)
    query_start_loc = torch.arange(0, B * N + 1, N, dtype=torch.int32, device=device)

    # Sequential decode
    ssm_states_decode = ssm_states_base.clone()
    outputs_decode = []
    for step in range(N):
        step_indices = [i * N + step for i in range(B)]
        step_q = q[:, step_indices].contiguous()
        step_k = k[:, step_indices].contiguous()
        step_v = v[:, step_indices].contiguous()
        step_a = a[step_indices].contiguous()
        step_b = b[:, step_indices].contiguous()
        decode_qsl = torch.arange(0, B + 1, dtype=torch.int32, device=device)
        out = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=step_q,
            k=step_k,
            v=step_v,
            a=step_a,
            b=step_b,
            initial_state_source=ssm_states_decode,
            initial_state_indices=cache_indices,
            cu_seqlens=decode_qsl,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=True,
        )
        outputs_decode.append(out)

    # target_verify
    ssm_states_verify = ssm_states_base.clone()
    intermediate_buffer = torch.zeros(
        num_slots, N, HV, K, V, dtype=dtype, device=device
    )
    intermediate_indices = torch.arange(B, dtype=torch.int32, device=device)
    out_verify = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        initial_state_source=ssm_states_verify,
        initial_state_indices=cache_indices,
        cu_seqlens=query_start_loc,
        use_qk_l2norm_in_kernel=True,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        is_kda=True,
        disable_state_update=True,
        intermediate_states_buffer=intermediate_buffer,
        intermediate_state_indices=intermediate_indices,
        cache_steps=N,
        retrieve_parent_token=None,
    )

    out_decode_list = []
    for req_idx in range(B):
        for step in range(N):
            out_decode_list.append(outputs_decode[step][:, req_idx : req_idx + 1])
    out_decode_cat = torch.cat(out_decode_list, dim=1)

    max_diff = (out_verify - out_decode_cat).abs().max().item()
    mean_diff = (out_verify - out_decode_cat).abs().mean().item()
    print(f"\n[bf16] Output max diff: {max_diff:.6e}, mean diff: {mean_diff:.6e}")
    # bf16 tolerance: kernel accumulates in fp32 internally, so diff should still be small
    assert max_diff < 1e-3, f"[bf16] Output mismatch! max diff: {max_diff}"

    print("PASSED: KDA target_verify bf16 test!")


if __name__ == "__main__":
    test_kda_target_verify_equivalence()
    test_kda_target_verify_bf16()
