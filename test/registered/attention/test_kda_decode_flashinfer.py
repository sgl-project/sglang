"""Correctness tests for the FlashInfer SM100 KDA decode + MTP backend.

Compares ``FlashInferKDAKernel`` with the Triton KDA reference for decode output,
state updates, and topk=1 target_verify checkpoints. ``recurrent_kda`` is
SM100-only and requires a FlashInfer build that exposes it.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

# SM100 single-GPU kernel-unit suite, same slot as the CuteDSL KDA prefill test.
# Disabled in public CI until the B200 runner image ships recurrent_kda.
register_cuda_ci(
    est_time=60,
    stage="base-b-kernel-unit",
    runner_config="4-gpu-b200",
    disabled="recurrent_kda (SM100 KDA decode) not guaranteed in public CI FlashInfer build",
)

if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10):
    pytest.skip(
        "FlashInfer KDA (recurrent_kda) requires CUDA SM10x (Blackwell).",
        allow_module_level=True,
    )

from sglang.srt.layers.attention.linear.kernels.kda_flashinfer import (  # noqa: E402
    FlashInferKDAKernel,
    _get_flashinfer_kda_kernel,
)
from sglang.srt.layers.attention.linear.kernels.kda_triton import (  # noqa: E402
    TritonKDAKernel,
)

_available, _ = _get_flashinfer_kda_kernel()
if not _available:
    pytest.skip(
        "FlashInfer build does not expose recurrent_kda (KDA decode).",
        allow_module_level=True,
    )

# KDA: head_k_dim == head_v_dim == 128; single q/v head group (HV == H) here.
H, HV, K, V = 16, 16, 128, 128


# ---------------------------------------------------------------------------
# Inputs (matched to the sglang KDA decode/verify contract: raw per-K gate `a`,
# beta logit `b`, SSM pool [N, HV, V, K], decode cu_seqlens = query_start_loc).
# ---------------------------------------------------------------------------
def _make_decode_inputs(batch_size, device="cuda", dtype=torch.bfloat16):
    B, pool = batch_size, batch_size + 16
    return dict(
        B=B,
        q=(torch.randn(1, B, H, K, device=device, dtype=dtype) * 0.5).contiguous(),
        k=(torch.randn(1, B, H, K, device=device, dtype=dtype) * 0.5).contiguous(),
        v=(torch.randn(1, B, HV, V, device=device, dtype=dtype) * 0.5).contiguous(),
        a=(torch.randn(B, HV * K, device=device, dtype=dtype) * 0.5 - 1.0).contiguous(),
        b=(torch.randn(B, HV, device=device, dtype=dtype) * 0.5).contiguous(),
        A_log=torch.randn(HV, device=device, dtype=torch.float32) * 0.2,
        dt_bias=torch.randn(HV * K, device=device, dtype=torch.float32) * 0.1,
        ssm=(
            torch.randn(pool, HV, V, K, device=device, dtype=dtype) * 0.01
        ).contiguous(),
        cache_indices=torch.arange(B, device=device, dtype=torch.int32),
        qsl=torch.arange(B + 1, device=device, dtype=torch.int32),
    )


def _make_verify_inputs(
    batch_size,
    cache_steps,
    allocated_steps=None,
    device="cuda",
    dtype=torch.bfloat16,
):
    B, T = batch_size, cache_steps
    S = allocated_steps or T
    assert S >= T
    seq, pool = B * T, B + 16
    return dict(
        B=B,
        T=T,
        allocated_steps=S,
        seq=seq,
        q=(torch.randn(1, seq, H, K, device=device, dtype=dtype) * 0.5).contiguous(),
        k=(torch.randn(1, seq, H, K, device=device, dtype=dtype) * 0.5).contiguous(),
        v=(torch.randn(1, seq, HV, V, device=device, dtype=dtype) * 0.5).contiguous(),
        a=(
            torch.randn(seq, HV * K, device=device, dtype=dtype) * 0.5 - 1.0
        ).contiguous(),
        b=(torch.randn(seq, HV, device=device, dtype=dtype) * 0.5).contiguous(),
        A_log=torch.randn(HV, device=device, dtype=torch.float32) * 0.2,
        dt_bias=torch.randn(HV * K, device=device, dtype=torch.float32) * 0.1,
        ssm=(
            torch.randn(pool, HV, V, K, device=device, dtype=dtype) * 0.01
        ).contiguous(),
        cache_indices=torch.arange(B, device=device, dtype=torch.int32),
        qsl=torch.arange(0, seq + 1, T, device=device, dtype=torch.int32),
        intermediate_states=torch.zeros(
            B, S, HV, V, K, device=device, dtype=dtype
        ).contiguous(),
        intermediate_indices=torch.arange(B, device=device, dtype=torch.int32),
    )


def _decode(kern, d, ssm):
    # `ssm` is updated in place (committed-pool decode step); pass a fresh clone.
    return kern.decode(
        d["q"],
        d["k"],
        d["v"],
        d["a"],
        d["b"],
        A_log=d["A_log"],
        dt_bias=d["dt_bias"],
        ssm_states=ssm,
        cache_indices=d["cache_indices"],
        query_start_loc=d["qsl"],
    ).reshape(d["B"], HV, V)


def _verify(kern, d, ssm, intermediate_states):
    return kern.target_verify(
        A_log=d["A_log"],
        dt_bias=d["dt_bias"],
        q=d["q"],
        k=d["k"],
        v=d["v"],
        a=d["a"],
        b=d["b"],
        ssm_states=ssm,
        cache_indices=d["cache_indices"],
        query_start_loc=d["qsl"],
        intermediate_states_buffer=intermediate_states,
        intermediate_state_indices=d["intermediate_indices"],
        cache_steps=d["T"],
        retrieve_parent_token=None,
    ).reshape(d["seq"], HV, V)


def _sequential_decode_states(kern, d):
    """Ground truth for verify checkpoints: single-token decode over each step."""
    B, T = d["B"], d["T"]
    st = d["ssm"].clone()  # committed pool [pool, HV, V, K], updated in place by decode
    ci = d["cache_indices"].long()
    qsl_dec = torch.arange(B + 1, device=st.device, dtype=torch.int32)
    ref = torch.zeros(B, T, HV, V, K, device=st.device, dtype=st.dtype)
    for t in range(T):
        pos = torch.arange(B, device=st.device) * T + t  # token t of each request
        kern.decode(
            d["q"][:, pos].contiguous(),
            d["k"][:, pos].contiguous(),
            d["v"][:, pos].contiguous(),
            d["a"][pos].contiguous(),
            d["b"][pos].contiguous(),
            A_log=d["A_log"],
            dt_bias=d["dt_bias"],
            ssm_states=st,
            cache_indices=d["cache_indices"],
            query_start_loc=qsl_dec,
        )
        ref[:, t] = st[ci]  # post-token-t state for each request
    return ref


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("batch_size", [1, 8, 64, 128])
def test_kda_decode_flashinfer_matches_triton(batch_size):
    """FlashInfer decode output + committed-pool state update match the Triton
    KDA decode reference."""
    torch.manual_seed(batch_size)
    d = _make_decode_inputs(batch_size)
    fi, tri = FlashInferKDAKernel(), TritonKDAKernel()

    st_ref = d["ssm"].clone()
    ref_out = _decode(tri, d, st_ref).float()
    st_fi = d["ssm"].clone()
    out = _decode(fi, d, st_fi).float()
    torch.cuda.synchronize()

    assert torch.isfinite(out).all(), "FlashInfer decode output has non-finite values"
    assert torch.isfinite(st_fi).all(), "FlashInfer decode state has non-finite values"

    o_err = (out - ref_out).abs()
    # bf16 recurrent step; B200 kernel-unit measured out max-abs-diff ~1e-4.
    assert o_err.max().item() < 1e-2, f"decode out max diff {o_err.max().item():.2e}"
    assert o_err.mean().item() < 1e-3, f"decode out mean diff {o_err.mean().item():.2e}"

    # Updated committed-pool slots (SSM state [HV, V, K]) must match too.
    idx = d["cache_indices"].long()
    s_err = (st_fi[idx].float() - st_ref[idx].float()).abs()
    assert s_err.max().item() < 1e-1, f"decode state max diff {s_err.max().item():.2e}"
    assert (
        s_err.mean().item() < 1e-2
    ), f"decode state mean diff {s_err.mean().item():.2e}"


@pytest.mark.parametrize("batch_size,num_spec", [(1, 7), (8, 7), (32, 3)])
def test_kda_target_verify_flashinfer_matches_triton(batch_size, num_spec):
    """FlashInfer MTP / target_verify (topk=1) per-draft-token output matches the
    Triton KDA verify reference over T = 1 + num_spec draft tokens per sequence."""
    torch.manual_seed(batch_size + num_spec)
    d = _make_verify_inputs(batch_size, 1 + num_spec)
    fi, tri = FlashInferKDAKernel(), TritonKDAKernel()

    ref_out = _verify(
        tri, d, d["ssm"].clone(), d["intermediate_states"].clone()
    ).float()
    out = _verify(fi, d, d["ssm"].clone(), d["intermediate_states"].clone()).float()
    torch.cuda.synchronize()

    assert torch.isfinite(out).all(), "FlashInfer verify output has non-finite values"
    o_err = (out - ref_out).abs()
    # B200 kernel-unit measured verify out max-abs-diff ~2e-4.
    assert o_err.max().item() < 1e-2, f"verify out max diff {o_err.max().item():.2e}"
    assert o_err.mean().item() < 1e-3, f"verify out mean diff {o_err.mean().item():.2e}"


@pytest.mark.parametrize(
    "batch_size,num_spec,extra_steps",
    [(1, 7, 0), (8, 7, 0), (32, 3, 2)],
)
def test_kda_target_verify_flashinfer_checkpoint_states(
    batch_size, num_spec, extra_steps
):
    """Checkpoint states must match true sequential decode states."""
    torch.manual_seed(1000 + batch_size + num_spec)
    cache_steps = 1 + num_spec
    d = _make_verify_inputs(
        batch_size,
        cache_steps,
        allocated_steps=cache_steps + extra_steps,
    )
    fi = FlashInferKDAKernel()

    ref_states = _sequential_decode_states(fi, d).float()

    intermediate_states = d["intermediate_states"].clone()
    _verify(
        fi, d, d["ssm"].clone(), intermediate_states
    )  # fills intermediate_states[n, t] in place
    torch.cuda.synchronize()

    got = intermediate_states[:, : d["T"]].float()  # [B, T, HV, V, K] checkpoint states
    assert torch.isfinite(got).all(), "verify checkpoint states have non-finite values"
    s_err = (got - ref_states).abs()
    # bf16 recurrent state; same tolerance as the decode committed-state check.
    assert (
        s_err.max().item() < 1e-1
    ), f"checkpoint state max diff {s_err.max().item():.2e}"
    assert (
        s_err.mean().item() < 1e-2
    ), f"checkpoint state mean diff {s_err.mean().item():.2e}"


def test_kda_target_verify_flashinfer_rejects_tree_spec():
    """Tree speculation (retrieve_parent_token != None) is unsupported (topk=1
    linear chain only) and must raise, not silently miscompute."""
    d = _make_verify_inputs(2, 4)
    parent = torch.zeros(d["seq"], device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="topk=1"):
        FlashInferKDAKernel().target_verify(
            A_log=d["A_log"],
            dt_bias=d["dt_bias"],
            q=d["q"],
            k=d["k"],
            v=d["v"],
            a=d["a"],
            b=d["b"],
            ssm_states=d["ssm"].clone(),
            cache_indices=d["cache_indices"],
            query_start_loc=d["qsl"],
            intermediate_states_buffer=d["intermediate_states"].clone(),
            intermediate_state_indices=d["intermediate_indices"],
            cache_steps=d["T"],
            retrieve_parent_token=parent,
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
