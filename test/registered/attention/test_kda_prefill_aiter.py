"""Correctness tests for the AITER KDA prefill backend (ROCm gfx950).

Validates ``AiterKDAKernel`` against the production Triton ``chunk_kda``
reference. AITER's FlashKDA path implements only the safe/bounded gate, so
these exercise ``lower_bound=-5``.

Requires ROCm gfx950 and an AITER that has ``chunk_kimi_delta_attn``. The
module skips otherwise. Mirrors ``test_kda_prefill_flashkda.py``.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, stage="stage-b", runner_config="1-gpu-small-amd")

from sglang.srt.layers.attention.linear.kernels.kda_aiter import (  # noqa: E402
    AiterKDAKernel,
)

if not AiterKDAKernel().supports_prefill:
    pytest.skip(
        f"AITER KDA prefill unavailable: {AiterKDAKernel().unavailable_reason}",
        allow_module_level=True,
    )

from sglang.kernels.ops.attention.fla.kda import chunk_kda  # noqa: E402

LOWER_BOUND = -5.0
H, K, V = 12, 128, 128  # AITER FlashKDA requires K == V == 128 and HV == H


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def _make_inputs(seq_lens, state_dtype=torch.float32, slots=None):
    n = len(seq_lens)
    slots = slots if slots is not None else n
    cu = torch.zeros(n + 1, device="cuda", dtype=torch.int32)
    cu[1:] = torch.tensor(seq_lens, device="cuda").cumsum(0)
    total = int(cu[-1].item())
    bf16 = dict(device="cuda", dtype=torch.bfloat16)
    return dict(
        cu=cu,
        idx=torch.arange(n, device="cuda", dtype=torch.int32),
        # Every sequence resumes, so the kernel reads the incoming state.
        prefix=torch.full((n,), 8, device="cuda", dtype=torch.int32),
        q=torch.randn(1, total, H, K, **bf16) * 0.5,
        k=torch.randn(1, total, H, K, **bf16) * 0.5,
        v=torch.randn(1, total, H, V, **bf16) * 0.5,
        g=torch.randn(1, total, H, K, **bf16) * 0.5,
        beta=torch.randn(1, total, H, device="cuda", dtype=torch.float32),
        A_log=torch.randn(1, 1, H, 1, device="cuda", dtype=torch.float32) * 0.5,
        dt_bias=torch.randn(H * K, device="cuda", dtype=torch.float32) * 0.1,
        pool=torch.randn(slots, H, V, K, device="cuda", dtype=state_dtype) * 0.1,
    )


def _chunk_kda_ref(d, lower_bound):
    """Triton reference. chunk_kda writes its output into v's storage and
    updates the state pool in place, so this helper passes its own copies."""
    state = d["pool"].clone()
    out = chunk_kda(
        q=d["q"].clone(),
        k=d["k"].clone(),
        v=d["v"].clone(),
        g=d["g"].clone(),
        beta=d["beta"].clone(),
        initial_state=state,
        initial_state_indices=d["idx"],
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=d["cu"],
        A_log=d["A_log"],
        dt_bias=d["dt_bias"],
        lower_bound=lower_bound,
        beta_is_raw=True,
    )
    return out, state[d["idx"]]


def _run_aiter(d, **overrides):
    state = d["pool"].clone()
    kwargs = dict(
        ssm_states=state,
        cache_indices=d["idx"],
        query_start_loc=d["cu"],
        A_log=d["A_log"],
        dt_bias=d["dt_bias"],
        lower_bound=LOWER_BOUND,
        extend_prefix_lens=d["prefix"],
        beta_is_raw=True,
    )
    kwargs.update(overrides)
    out = AiterKDAKernel().extend(
        d["q"].clone(),
        d["k"].clone(),
        d["v"].clone(),
        d["g"].clone(),
        d["beta"].clone(),
        **kwargs,
    )
    torch.cuda.synchronize()
    return out, state


@pytest.mark.parametrize("seq_lens", [[128], [128, 384, 512], [96] * 4])
def test_aiter_matches_triton_safe_gate(seq_lens):
    torch.manual_seed(len(seq_lens))
    d = _make_inputs(seq_lens)
    ref_out, ref_state = _chunk_kda_ref(d, LOWER_BOUND)

    out, state = _run_aiter(d)

    # forward_extend unpacks only when it asks for intermediate states. A
    # tuple here would make core_attn_out a tuple for every untracked batch.
    assert isinstance(out, torch.Tensor), "extend must return a bare tensor"
    assert torch.isfinite(out).all()
    assert torch.isfinite(state).all()
    assert _cos(ref_out, out) > 0.99, f"output cos too low: {_cos(ref_out, out):.4f}"
    assert _cos(ref_state, state[d["idx"]]) > 0.99


def test_aiter_paged_state_cache_honours_padded_slot_stride():
    """The fp32 tier addresses the pool in place. A padded stride(0) is the
    page-major envelope shape. The kernel must read it correctly, leave the
    untouched slots alone, and never pack the pool into a copy."""
    torch.manual_seed(0)
    d = _make_inputs([256, 128], slots=4)
    ref_out, ref_state = _chunk_kda_ref(d, LOWER_BOUND)

    inner = H * V * K
    slot_stride = inner + 64
    storage = torch.zeros(4 * slot_stride, device="cuda", dtype=torch.float32)
    padded = torch.as_strided(
        storage, size=(4, H, V, K), stride=(slot_stride, V * K, K, 1)
    )
    padded.copy_(d["pool"])
    untouched = padded[2:].clone()
    ptr_before = padded.data_ptr()

    out, _ = _run_aiter(d, ssm_states=padded)

    assert padded.data_ptr() == ptr_before, "paged pool must not be packed or copied"
    assert torch.equal(padded[2:], untouched), "untouched slots were modified"
    assert _cos(ref_out, out) > 0.99
    assert _cos(ref_state, padded[d["idx"]]) > 0.99


def test_aiter_accepts_a_bf16_state_pool():
    """A 2-byte pool cannot use the fp32-only paged cache, so the backend takes
    the initial_state gather/scatter tier instead of refusing to serve."""
    torch.manual_seed(0)
    d = _make_inputs([256], state_dtype=torch.bfloat16)
    ref_out, ref_state = _chunk_kda_ref(d, LOWER_BOUND)

    out, state = _run_aiter(d)

    assert state.dtype == torch.bfloat16
    assert _cos(ref_out, out) > 0.99
    assert _cos(ref_state, state[d["idx"]]) > 0.99


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"lower_bound": None}, id="unbounded_gate"),
        pytest.param({"is_spec_decode": True}, id="spec_decode"),
        pytest.param({"return_intermediate_states": True}, id="track_state"),
    ],
)
def test_aiter_falls_back_to_triton(overrides):
    """On the fused path each of these is either silently wrong or rejected.
    They must reach the Triton chunk_kda fallback."""
    torch.manual_seed(0)
    d = _make_inputs([256])
    ref_out, _ = _chunk_kda_ref(d, overrides.get("lower_bound", LOWER_BOUND))

    out, _ = _run_aiter(d, **overrides)
    out = out[0] if isinstance(out, tuple) else out

    # This runs the same Triton kernel as the reference, so the match is
    # near-exact. If AITER had run, the cos would be near 0.99 and this fails.
    assert _cos(ref_out, out) > 0.9999, (
        f"did not fall back to Triton: {_cos(ref_out, out):.5f}"
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
