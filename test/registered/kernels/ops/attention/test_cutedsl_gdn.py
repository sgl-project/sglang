"""Tests for CuTe DSL fused sigmoid gating delta rule kernel (GDN)."""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

try:
    import cutlass  # noqa: F401

    from sglang.kernels.ops.attention import cutedsl_gdn, cutedsl_kda

    CUTEDSL_AVAILABLE = True
except ImportError:
    CUTEDSL_AVAILABLE = False
    cutedsl_gdn = cutedsl_kda = None

try:
    from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def run_triton_kernel(
    A_log, dt_bias, q, k, v, a, b, initial_state, indices, scale, cu_seqlens=None
):
    return fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=initial_state,
        initial_state_indices=indices,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )


@pytest.mark.skipif(not CUTEDSL_AVAILABLE, reason="CuTe DSL not available")
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton kernel not available")
@pytest.mark.parametrize("pool_view", ["4d", "1d"])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("B", [16, 128])
def test_cutedsl_gdn_precision(B: int, varlen: bool, pool_view: str):
    """CuTe DSL decode must read and update the Triton [pool, HV, V, K] state pool."""
    torch.manual_seed(2025)
    T, H, K, V, HV = 1, 16, 128, 128, 32
    scale = K**-0.5
    lead = (1, B) if varlen else (B, T)
    gate_shape = (B, HV) if varlen else (B, T, HV)

    A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.randn(HV, dtype=torch.bfloat16, device="cuda")
    pool_size = 2 * B + 1
    indices = torch.randperm(pool_size, device="cuda")[:B].to(torch.int32)
    indices[B // 2] = -1
    valid = indices >= 0
    cu_seqlens = (
        torch.arange(B + 1, dtype=torch.int32, device="cuda") if varlen else None
    )
    state_cutedsl = torch.randn(pool_size, HV, V, K, dtype=torch.float32, device="cuda")
    state_triton = state_cutedsl.clone()
    pool = state_cutedsl.view(-1) if pool_view == "1d" else state_cutedsl

    for _ in range(3):
        a = torch.randn(*gate_shape, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(*gate_shape, dtype=torch.bfloat16, device="cuda")
        q = torch.randn(*lead, H, K, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(*lead, H, K, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(*lead, HV, V, dtype=torch.bfloat16, device="cuda")

        out_cutedsl = cutedsl_gdn.cutedsl_fused_sigmoid_gating_delta_rule_update(
            A_log, dt_bias, q, k, v, a, b, pool, indices, cu_seqlens, scale=scale
        )
        out_triton = run_triton_kernel(
            A_log, dt_bias, q, k, v, a, b, state_triton, indices, scale, cu_seqlens
        )

        torch.testing.assert_close(
            out_cutedsl.reshape(B, HV, V)[valid].float(),
            out_triton.reshape(B, HV, V)[valid].float(),
            atol=1e-3,
            rtol=1e-2,
        )
        torch.testing.assert_close(state_cutedsl, state_triton, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not CUTEDSL_AVAILABLE, reason="CuTe DSL not available")
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton kernel not available")
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dt_bias_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("B", [16, 64])
def test_cutedsl_gdn_state_and_dt_bias_dtypes(
    B: int, varlen: bool, dt_bias_dtype: torch.dtype, state_dtype: torch.dtype
):
    """Support BF16/FP16 state pools and FP32/BF16 dt_bias without silent corruption."""
    torch.manual_seed(2026)
    T, H, K, V, HV = 1, 16, 128, 128, 32
    scale = K**-0.5
    lead = (1, B) if varlen else (B, T)
    gate_shape = (B, HV) if varlen else (B, T, HV)

    A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.randn(HV, dtype=dt_bias_dtype, device="cuda")
    pool_size = 2 * B + 1
    indices = torch.randperm(pool_size, device="cuda")[:B].to(torch.int32)
    indices[B // 2] = -1
    valid = indices >= 0
    cu_seqlens = (
        torch.arange(B + 1, dtype=torch.int32, device="cuda") if varlen else None
    )
    state_cutedsl = torch.randn(pool_size, HV, V, K, dtype=state_dtype, device="cuda")
    state_triton = state_cutedsl.clone()
    untouched_mask = torch.ones(pool_size, dtype=torch.bool, device="cuda")
    untouched_mask[indices[valid].long()] = False
    untouched_ref = state_cutedsl[untouched_mask].clone()

    for _ in range(2):
        a = torch.randn(*gate_shape, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(*gate_shape, dtype=torch.bfloat16, device="cuda")
        q = torch.randn(*lead, H, K, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(*lead, H, K, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(*lead, HV, V, dtype=torch.bfloat16, device="cuda")

        out_cutedsl = cutedsl_gdn.cutedsl_fused_sigmoid_gating_delta_rule_update(
            A_log,
            dt_bias,
            q,
            k,
            v,
            a,
            b,
            state_cutedsl,
            indices,
            cu_seqlens,
            scale=scale,
        )
        out_triton = run_triton_kernel(
            A_log, dt_bias, q, k, v, a, b, state_triton, indices, scale, cu_seqlens
        )

        assert torch.isfinite(out_cutedsl.reshape(B, HV, V)[valid]).all()
        assert torch.isfinite(state_cutedsl).all()
        # Untouched pool slots must remain bit-identical (guards against OOB writes).
        torch.testing.assert_close(
            state_cutedsl[untouched_mask], untouched_ref, atol=0, rtol=0
        )
        atol = 2e-2 if state_dtype == torch.bfloat16 else 5e-3
        rtol = 2e-2 if state_dtype == torch.bfloat16 else 1e-2
        torch.testing.assert_close(
            out_cutedsl.reshape(B, HV, V)[valid].float(),
            out_triton.reshape(B, HV, V)[valid].float(),
            atol=atol,
            rtol=rtol,
        )
        torch.testing.assert_close(
            state_cutedsl.float(), state_triton.float(), atol=atol, rtol=rtol
        )


@pytest.mark.skipif(not CUTEDSL_AVAILABLE, reason="CuTe DSL not available")
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton kernel not available")
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_cutedsl_gdn_strided_pool_and_split_ab(state_dtype: torch.dtype):
    """Envelope-strided state pools and non-contiguous split a/b must match Triton."""
    torch.manual_seed(2027)
    B, H, K, V, HV = 8, 16, 128, 128, 32
    scale = K**-0.5
    pool_size = 20
    extra_pad = 128  # extra elements per slot so stride(0) > HV * V * K

    raw_backing = torch.randn(
        pool_size, HV * V * K + extra_pad, dtype=state_dtype, device="cuda"
    )
    pad_ref = raw_backing[:, HV * V * K :].clone()
    state_cutedsl = raw_backing[:, : HV * V * K].view(pool_size, HV, V, K)
    assert state_cutedsl.stride(0) == HV * V * K + extra_pad
    state_triton = state_cutedsl.clone()

    A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.randn(HV, dtype=torch.float32, device="cuda")
    indices = torch.randperm(pool_size, device="cuda")[:B].to(torch.int32)
    cu_seqlens = torch.arange(B + 1, dtype=torch.int32, device="cuda")

    mixed_ba = torch.randn(B, 2 * HV, dtype=torch.bfloat16, device="cuda")
    b, a = mixed_ba.split([HV, HV], dim=-1)
    assert not a.is_contiguous() and not b.is_contiguous()
    q = torch.randn(1, B, H, K, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, B, H, K, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, B, HV, V, dtype=torch.bfloat16, device="cuda")

    out_cutedsl = cutedsl_gdn.cutedsl_fused_sigmoid_gating_delta_rule_update(
        A_log, dt_bias, q, k, v, a, b, state_cutedsl, indices, cu_seqlens, scale=scale
    )
    out_triton = run_triton_kernel(
        A_log, dt_bias, q, k, v, a, b, state_triton, indices, scale, cu_seqlens
    )

    torch.testing.assert_close(raw_backing[:, HV * V * K :], pad_ref, atol=0, rtol=0)
    atol = 2e-2 if state_dtype == torch.bfloat16 else 1e-3
    rtol = 2e-2 if state_dtype == torch.bfloat16 else 1e-2
    torch.testing.assert_close(
        out_cutedsl.float(), out_triton.float(), atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        state_cutedsl.float(), state_triton.float(), atol=atol, rtol=rtol
    )


@pytest.mark.skipif(not CUTEDSL_AVAILABLE, reason="CuTe DSL not available")
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton kernel not available")
def test_cutedsl_gdn_fp16_inputs_and_cuda_graph():
    """FP16 inputs/outputs and CUDA Graph capture/replay must match Triton (H=16, HV=48)."""
    torch.manual_seed(2028)
    B, H, K, V, HV = 16, 16, 128, 128, 48
    scale = K**-0.5
    pool_size = 2 * B + 2

    A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.randn(HV, dtype=torch.float32, device="cuda")
    indices = torch.randperm(pool_size, device="cuda")[:B].to(torch.int32)
    cu_seqlens = torch.arange(B + 1, dtype=torch.int32, device="cuda")

    q = torch.randn(1, B, H, K, dtype=torch.float16, device="cuda")
    k = torch.randn(1, B, H, K, dtype=torch.float16, device="cuda")
    v = torch.randn(1, B, HV, V, dtype=torch.float16, device="cuda")
    a = torch.randn(B, HV, dtype=torch.float16, device="cuda")
    b = torch.randn(B, HV, dtype=torch.float16, device="cuda")

    state_init = (
        torch.randn(pool_size, HV, V, K, dtype=torch.bfloat16, device="cuda") * 0.2
    )
    state_cutedsl = state_init.clone()
    state_triton = state_init.clone()

    # Warmup compile before CUDA Graph capture.
    out_warmup = cutedsl_gdn.cutedsl_fused_sigmoid_gating_delta_rule_update(
        A_log, dt_bias, q, k, v, a, b, state_cutedsl, indices, cu_seqlens, scale=scale
    )
    assert out_warmup.dtype == torch.float16
    state_cutedsl.copy_(state_init)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_graph = cutedsl_gdn.cutedsl_fused_sigmoid_gating_delta_rule_update(
            A_log,
            dt_bias,
            q,
            k,
            v,
            a,
            b,
            state_cutedsl,
            indices,
            cu_seqlens,
            scale=scale,
        )
    g.replay()
    torch.cuda.synchronize()

    out_triton = run_triton_kernel(
        A_log, dt_bias, q, k, v, a, b, state_triton, indices, scale, cu_seqlens
    )
    torch.testing.assert_close(
        out_graph.float(), out_triton.float(), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        state_cutedsl.float(), state_triton.float(), atol=2e-2, rtol=2e-2
    )


@pytest.mark.skipif(not CUTEDSL_AVAILABLE, reason="CuTe DSL not available")
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton kernel not available")
@pytest.mark.parametrize("indices_dtype", [torch.int32, torch.int64])
def test_cutedsl_gdn_indices_dtype(indices_dtype: torch.dtype):
    """Both int32 and int64 initial_state_indices must update exact target slots."""
    torch.manual_seed(2029)
    B, H, K, V, HV = 8, 16, 128, 128, 32
    scale = K**-0.5
    pool_size = 2 * B + 2

    A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.randn(HV, dtype=torch.float32, device="cuda")
    indices = torch.randperm(pool_size, device="cuda")[:B].to(indices_dtype)
    indices[B // 2] = -1
    valid = indices >= 0
    cu_seqlens = torch.arange(B + 1, dtype=torch.int32, device="cuda")

    q = torch.randn(1, B, H, K, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, B, H, K, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, B, HV, V, dtype=torch.bfloat16, device="cuda")
    a = torch.randn(B, HV, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(B, HV, dtype=torch.bfloat16, device="cuda")

    state_cutedsl = torch.randn(
        pool_size, HV, V, K, dtype=torch.bfloat16, device="cuda"
    )
    state_triton = state_cutedsl.clone()
    untouched = torch.ones(pool_size, dtype=torch.bool, device="cuda")
    untouched[indices[valid].long()] = False
    untouched_ref = state_cutedsl[untouched].clone()

    out_cutedsl = cutedsl_gdn.cutedsl_fused_sigmoid_gating_delta_rule_update(
        A_log, dt_bias, q, k, v, a, b, state_cutedsl, indices, cu_seqlens, scale=scale
    )
    out_triton = run_triton_kernel(
        A_log, dt_bias, q, k, v, a, b, state_triton, indices, scale, cu_seqlens
    )

    torch.testing.assert_close(state_cutedsl[untouched], untouched_ref, atol=0, rtol=0)
    torch.testing.assert_close(
        out_cutedsl.reshape(B, HV, V)[valid].float(),
        out_triton.reshape(B, HV, V)[valid].float(),
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(
        state_cutedsl.float(), state_triton.float(), atol=2e-2, rtol=2e-2
    )


@pytest.mark.skipif(not CUTEDSL_AVAILABLE, reason="CuTe DSL not available")
def test_cutedsl_gdn_rejects_unsupported_state_dtype():
    B, H, K, V, HV = 2, 16, 128, 128, 32
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="initial_state_source dtypes"):
        cutedsl_gdn.cutedsl_fused_sigmoid_gating_delta_rule_update(
            A_log=torch.zeros(HV, dtype=torch.float32, device="cuda"),
            dt_bias=torch.zeros(HV, **bf16),
            q=torch.zeros(B, 1, H, K, **bf16),
            k=torch.zeros(B, 1, H, K, **bf16),
            v=torch.zeros(B, 1, HV, V, **bf16),
            a=torch.zeros(B, 1, HV, **bf16),
            b=torch.zeros(B, 1, HV, **bf16),
            initial_state_source=torch.zeros(
                B, HV, V, K, dtype=torch.float64, device="cuda"
            ),
            initial_state_indices=torch.arange(B, dtype=torch.int32, device="cuda"),
        )


@pytest.mark.skipif(not CUTEDSL_AVAILABLE, reason="CuTe DSL not available")
@pytest.mark.parametrize("kernel", ["gdn", "kda"])
@pytest.mark.parametrize(
    "override",
    [
        {"scale": 1.0},
        {"use_qk_l2norm_in_kernel": False},
        {"softplus_beta": 2.0},
        {"softplus_threshold": 10.0},
    ],
)
def test_cutedsl_decode_rejects_uncompiled_constants(kernel: str, override: dict):
    """The decode kernels are compiled with fixed numerics; another value must raise
    instead of being silently ignored."""
    B, H, K, V, HV = 2, 16, 128, 128, 32
    gate_dim = (HV, K) if kernel == "kda" else (HV,)
    update = (
        cutedsl_kda.cutedsl_fused_sigmoid_gating_kda_update
        if kernel == "kda"
        else cutedsl_gdn.cutedsl_fused_sigmoid_gating_delta_rule_update
    )
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="only supports"):
        update(
            A_log=torch.zeros(HV, dtype=torch.float32, device="cuda"),
            dt_bias=torch.zeros(*gate_dim, **bf16),
            q=torch.zeros(B, 1, H, K, **bf16),
            k=torch.zeros(B, 1, H, K, **bf16),
            v=torch.zeros(B, 1, HV, V, **bf16),
            a=torch.zeros(B, 1, *gate_dim, **bf16),
            b=torch.zeros(B, 1, HV, **bf16),
            initial_state_source=torch.zeros(B, HV, V, K, device="cuda"),
            initial_state_indices=torch.arange(B, dtype=torch.int32, device="cuda"),
            **override,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
