"""CAKE KDA prefill through the exported prepared BF16 API (FP32 state pool).

The prepared export (``flashinfer.prepare_bf16_kda_prefill``) is selected by the
Cake kernel whenever the recurrent state pool is FP32. These tests compare it
against SGLang's Triton KDA prefill on the same FP32 pool.
"""

import os
from unittest.mock import patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120,
    stage="base-b",
    runner_config="4-gpu-b200",
    disabled="prepared BF16 KDA export is not in the pinned public FlashInfer build",
)

if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
    (10, 0),
    (10, 3),
):
    pytest.skip("CAKE KDA export requires SM100/SM103", allow_module_level=True)

from sglang.srt.layers.attention.linear.kernels import (  # noqa: E402
    kda_flashinfer as kda_flashinfer_module,
)
from sglang.srt.layers.attention.linear.kernels.kda_flashinfer import (  # noqa: E402
    CakeKDAKernel,
    _get_flashinfer_prepared_bf16_prefill,
)
from sglang.srt.layers.attention.linear.kernels.kda_triton import (  # noqa: E402
    TritonKDAKernel,
)

if not _get_flashinfer_prepared_bf16_prefill()[0]:
    pytest.skip(
        "installed FlashInfer lacks prepare_bf16_kda_prefill", allow_module_level=True
    )

K = V = 128
LOWER_BOUND = -5.0


def _make_inputs(seq_lens, num_heads, *, state_dtype=torch.float32):
    num_sequences = len(seq_lens)
    total_tokens = sum(seq_lens)
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    pool_size = num_sequences + 5
    cache_indices = torch.arange(
        pool_size - 1,
        pool_size - num_sequences - 1,
        -1,
        device="cuda",
        dtype=torch.int32,
    )
    make = lambda: torch.randn(  # noqa: E731
        1, total_tokens, num_heads, K, device="cuda", dtype=torch.bfloat16
    ).contiguous()
    return dict(
        q=make(),
        k=make(),
        v=make(),
        g=make(),
        # Fused-projection slice: raw BF16 logits with a wider token-row pitch.
        beta=torch.randn(1, total_tokens, 40, device="cuda", dtype=torch.bfloat16)[
            :, :, 8 : 8 + num_heads
        ],
        # Real models hand these over as nn.Parameters (requires_grad=True); the
        # adapter must detach them before calling the export.
        A_log=torch.nn.Parameter(
            (
                torch.randn(1, 1, num_heads, 1, device="cuda", dtype=torch.float32)
                * 0.2
            ).contiguous()
        ),
        dt_bias=torch.nn.Parameter(
            (
                torch.randn(num_heads * K, device="cuda", dtype=torch.float32) * 0.1
            ).contiguous()
        ),
        state=(
            torch.randn(pool_size, num_heads, V, K, device="cuda", dtype=state_dtype)
            * 0.01
        ).contiguous(),
        cache_indices=cache_indices,
        cu_seqlens=cu_seqlens,
    )


def _extend(kernel, data, state, seq_lens, lower_bound=LOWER_BOUND, **kwargs):
    if getattr(kernel, "supports_cake_route_telemetry", False):
        kwargs["layer_id"] = 7
    beta = data["beta"]
    if isinstance(kernel, TritonKDAKernel):
        beta = torch.sigmoid(beta)
    return kernel.extend(
        data["q"].clone(),
        data["k"].clone(),
        data["v"].clone(),
        data["g"].clone(),
        beta,
        ssm_states=state,
        cache_indices=data["cache_indices"],
        query_start_loc=data["cu_seqlens"],
        A_log=data["A_log"],
        dt_bias=data["dt_bias"],
        lower_bound=lower_bound,
        extend_seq_lens_cpu=seq_lens,
        **kwargs,
    )


def _assert_close(name, actual, expected, atol=2e-2, rtol=2e-2):
    diff = (actual.float() - expected.float()).abs()
    tol = atol + rtol * expected.float().abs()
    assert bool((diff <= tol).all()), (
        f"{name}: max abs diff {diff.max().item():.4g}, "
        f"{int((diff > tol).sum())} elements beyond tolerance"
    )


@pytest.mark.parametrize(
    "num_heads,seq_lens",
    [
        (8, [128]),  # Kimi-Linear-48B TP4 local heads
        (16, [96]),  # TP2 local heads
        (16, [64, 160]),
        (8, [17, 64, 65, 127, 128, 255]),
        (16, [8192]),
        (8, [32768]),
    ],
)
def test_kda_prefill_prepared_export_matches_triton(num_heads, seq_lens):
    torch.manual_seed(num_heads + sum(seq_lens))
    data = _make_inputs(seq_lens, num_heads)
    state_triton = data["state"].clone()
    state_cake = data["state"].clone()
    output_triton = _extend(TritonKDAKernel(), data, state_triton, seq_lens)
    with patch.object(
        CakeKDAKernel,
        "_extend_triton",
        side_effect=AssertionError("prepared export must not fall back to Triton"),
    ):
        output_cake = _extend(CakeKDAKernel(), data, state_cake, seq_lens)
    _assert_close("output", output_cake, output_triton)
    idx = data["cache_indices"].long()
    _assert_close(
        "final_state", state_cake[idx], state_triton[idx], atol=1e-2, rtol=1e-2
    )
    untouched = torch.ones(state_cake.shape[0], dtype=torch.bool, device="cuda")
    untouched[idx] = False
    assert torch.equal(state_cake[untouched], data["state"][untouched])


@pytest.mark.parametrize(
    "num_heads,seq_lens",
    [
        (16, [96]),
        (8, [17, 64, 65, 127, 128, 255]),
        (16, [8192]),
    ],
)
def test_kda_prefill_prepared_tf32_export_matches_triton(num_heads, seq_lens):
    """``--kda-cake-prefill-precision tf32`` runs the TF32 prepared export on
    the same FP32 state pool (bounded gate) and matches Triton within the BF16
    tolerance."""
    if kda_flashinfer_module._flashinfer_prepare_tf32_kda_prefill is None:
        pytest.skip("installed FlashInfer lacks prepare_tf32_kda_prefill")
    torch.manual_seed(1000 + num_heads + sum(seq_lens))
    data = _make_inputs(seq_lens, num_heads)
    state_triton = data["state"].clone()
    state_cake = data["state"].clone()
    output_triton = _extend(TritonKDAKernel(), data, state_triton, seq_lens)
    with (
        patch.dict(os.environ, {"SGLANG_KDA_CAKE_PREFILL_PRECISION": "tf32"}),
        patch.object(
            CakeKDAKernel,
            "_extend_triton",
            side_effect=AssertionError("prepared export must not fall back to Triton"),
        ),
    ):
        kernel = CakeKDAKernel()
        assert kernel.cake_prefill_precision == "tf32"
        output_cake = _extend(kernel, data, state_cake, seq_lens)
    _assert_close("output", output_cake, output_triton)
    idx = data["cache_indices"].long()
    _assert_close(
        "final_state", state_cake[idx], state_triton[idx], atol=1e-2, rtol=1e-2
    )


def test_kda_prefill_tf32_precision_hands_unbounded_gates_to_triton():
    """The TF32 export ships no unbounded-gate module without checkpoint rows,
    so ``--kda-cake-prefill-precision tf32`` runs the Triton prefill for an
    unbounded softplus gate (``lower_bound=None``, Kimi-Linear) instead of
    raising inside the exported call."""
    if kda_flashinfer_module._flashinfer_prepare_tf32_kda_prefill is None:
        pytest.skip("installed FlashInfer lacks prepare_tf32_kda_prefill")
    torch.manual_seed(1234)
    seq_lens = [96]
    data = _make_inputs(seq_lens, 16)
    state_triton = data["state"].clone()
    state_cake = data["state"].clone()
    output_triton = _extend(
        TritonKDAKernel(), data, state_triton, seq_lens, lower_bound=None
    )
    with patch.dict(os.environ, {"SGLANG_KDA_CAKE_PREFILL_PRECISION": "tf32"}):
        kernel = CakeKDAKernel()
        assert kernel.cake_prefill_precision == "tf32"
        with patch.object(
            CakeKDAKernel, "_extend_triton", wraps=kernel._extend_triton
        ) as triton_prefill:
            output_cake = _extend(kernel, data, state_cake, seq_lens, lower_bound=None)
    assert triton_prefill.call_count == 1
    _assert_close("output", output_cake, output_triton)
    idx = data["cache_indices"].long()
    _assert_close(
        "final_state", state_cake[idx], state_triton[idx], atol=1e-2, rtol=1e-2
    )


def _export_has_fp32_checkpoints(lower_bound) -> bool:
    """Whether the installed FlashInfer exports FP32 intermediate states here
    for the gate kind selected by ``lower_bound`` (None = unbounded softplus)."""
    try:
        from flashinfer.kda_prefill import kda_prefill_supports_fp32_checkpoints
    except ImportError:
        return False
    return bool(
        kda_prefill_supports_fp32_checkpoints(
            torch.device("cuda"), lower_bound=lower_bound
        )
    )


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


@torch.no_grad()
def _fp32_reference(data, seq, seq_lens, chunk_every=None):
    """Token-by-token FP32 KDA recurrence for one sequence of ``data`` (the same
    math as the kernels: L2-normalised q/k, softplus gate scaled by -exp(A_log),
    sigmoid beta). Returns (output [T, H, V], final state [H, V, K]); with
    ``chunk_every`` also the list of states at the start of every chunk."""
    start = sum(seq_lens[:seq])
    stop = start + seq_lens[seq]
    q = data["q"][0, start:stop].float()
    k = data["k"][0, start:stop].float()
    v = data["v"][0, start:stop].float()
    g_raw = data["g"][0, start:stop].float()
    beta = torch.sigmoid(data["beta"][0, start:stop].float())
    num_heads = q.shape[1]
    q = torch.nn.functional.normalize(q, dim=-1) * (K**-0.5)
    k = torch.nn.functional.normalize(k, dim=-1)
    a_log = data["A_log"].detach().float().reshape(num_heads)
    bias = data["dt_bias"].detach().float().reshape(num_heads, K)
    g = -torch.exp(a_log)[None, :, None] * torch.nn.functional.softplus(
        g_raw + bias[None]
    )
    state = data["state"][data["cache_indices"][seq].long()].float()
    s = state.transpose(-1, -2).clone()  # [H, K, V]
    out = torch.empty(q.shape[0], num_heads, V, device=q.device, dtype=torch.float32)
    chunks = [s.transpose(-1, -2).clone()]
    for t in range(q.shape[0]):
        s = s * torch.exp(g[t])[:, :, None]
        v_pred = torch.einsum("hk,hkv->hv", k[t], s)
        s = s + torch.einsum("hk,hv->hkv", k[t], beta[t][:, None] * (v[t] - v_pred))
        out[t] = torch.einsum("hk,hkv->hv", q[t], s)
        if chunk_every and (t + 1) % chunk_every == 0 and t + 1 < q.shape[0]:
            chunks.append(s.transpose(-1, -2).clone())
    if chunk_every:
        return out, s.transpose(-1, -2), chunks
    return out, s.transpose(-1, -2)


def test_kda_prefill_prepared_export_native_checkpoints():
    """Per-64-token-chunk states match Triton's intermediate states chunk by chunk.

    Checkpoint ``i`` of a sequence is the recurrent state at the start of its
    chunk ``i`` (checkpoint 0 is the initial state); the export stores them in
    BF16, so compare with a relative-L2 budget instead of elementwise atol.
    """
    torch.manual_seed(1234)
    seq_lens = [200, 130]
    data = _make_inputs(seq_lens, 16)
    checkpoint_source = torch.tensor([0, 3], device="cuda", dtype=torch.int64)
    checkpoint_cu_starts = torch.tensor([0, 4, 7], device="cuda", dtype=torch.int64)
    cp_kwargs = dict(
        return_intermediate_states=True,
        track_ssm_h_src=checkpoint_source,
        state_checkpoint_cu_starts=checkpoint_cu_starts,
        num_state_checkpoints=7,
        state_checkpoint_every_n_tokens=64,
    )
    state_ref = data["state"].clone()
    output_ref, h_ref = _extend(
        TritonKDAKernel(), data, state_ref, seq_lens, **cp_kwargs
    )
    state_cake = data["state"].clone()
    with patch.object(
        CakeKDAKernel,
        "_extend_triton",
        side_effect=AssertionError("native checkpoints must not fall back to Triton"),
    ):
        output, h = _extend(CakeKDAKernel(), data, state_cake, seq_lens, **cp_kwargs)
    assert output.shape == data["q"].shape
    assert h.shape == h_ref.shape == (1, 7, 16, V, K)
    assert h.dtype == state_cake.dtype
    assert torch.isfinite(h).all()
    _assert_close("output", output, output_ref)
    idx = data["cache_indices"].long()
    assert _rel_l2(state_cake[idx], state_ref[idx]) < 1e-2
    starts = checkpoint_cu_starts.tolist()
    fp32_rows = _export_has_fp32_checkpoints(LOWER_BOUND)
    for seq in range(len(seq_lens)):
        first = starts[seq]
        # Checkpoint 0 of every sequence is its initial state.  FP32 exports
        # return it exactly; BF16 exports return the BF16-rounded state exactly.
        initial = data["state"][idx[seq]]
        if fp32_rows:
            assert torch.equal(h[0, first], initial)
        else:
            assert torch.equal(h[0, first], initial.to(torch.bfloat16).to(h.dtype))
        assert _rel_l2(h[0, first], initial) < 1e-2
        for j in range(first + 1, starts[seq + 1]):
            err = _rel_l2(h[0, j], h_ref[0, j])
            assert err < 1e-2, f"sequence {seq} chunk {j - first}: rel L2 {err:.4g}"


@pytest.mark.parametrize(
    "num_heads,seq_lens",
    [(16, [64, 160]), (8, [17, 64, 65, 127, 128, 255]), (16, [8192])],
)
def test_kda_prefill_prepared_export_unbounded_gate_matches_triton(num_heads, seq_lens):
    """Kimi-Linear has no gate lower bound; the export serves the unbounded softplus
    gate by default (tile-anchored floored-prefix decay in the fused BF16 schedule)."""
    torch.manual_seed(99 + num_heads + sum(seq_lens))
    data = _make_inputs(seq_lens, num_heads)
    state_triton = data["state"].clone()
    state_cake = data["state"].clone()
    output_triton = _extend(
        TritonKDAKernel(), data, state_triton, seq_lens, lower_bound=None
    )
    with patch.object(
        CakeKDAKernel,
        "_extend_triton",
        side_effect=AssertionError("unbounded gate must not fall back to Triton"),
    ):
        output_cake = _extend(
            CakeKDAKernel(), data, state_cake, seq_lens, lower_bound=None
        )
    _assert_close("output", output_cake, output_triton)
    idx = data["cache_indices"].long()
    _assert_close(
        "final_state", state_cake[idx], state_triton[idx], atol=1e-2, rtol=1e-2
    )


def test_kda_prefill_policy_facade_keeps_bf16_pool_path():
    torch.manual_seed(7)
    data = _make_inputs([128], 8, state_dtype=torch.bfloat16)
    state = data["state"].clone()
    with patch.dict(os.environ, {"SGLANG_KDA_CAKE_PREFILL_API": "auto"}):
        kernel = CakeKDAKernel()
        assert not kernel._cake_prefill_uses_prepared_export(state)
    with patch.dict(os.environ, {"SGLANG_KDA_CAKE_PREFILL_API": "prepared"}):
        assert CakeKDAKernel()._cake_prefill_uses_prepared_export(state)
    with patch.dict(os.environ, {"SGLANG_KDA_CAKE_PREFILL_API": "facade"}):
        assert not CakeKDAKernel()._cake_prefill_uses_prepared_export(
            data["state"].float()
        )


def test_kda_prefill_unbounded_gate_extreme_decay_matches_fp32_reference():
    """Real Kimi-Linear activations drive per-token log2 gates far below -126 (the
    exp2 normal range). The fused schedule anchors each 16-token tile separately,
    so the per-sequence recurrent state and output must stay at the mild-gate
    error against an exact FP32 recurrence (~0.004 / 0.005 rel L2).

    Triton's ``chunk_kda`` is not the reference here: on these gates it drifts to
    0.012-0.019 (state) / 0.10-0.12 (output) rel L2 from the exact recurrence, so
    comparing the export against Triton would measure Triton's error."""
    torch.manual_seed(1542)
    seq_lens = [1542, 64]
    num_heads = 8
    data = _make_inputs(seq_lens, num_heads)
    # Strongly negative gate logits with a wide spread; A_log/dt_bias scale the
    # softplus so a handful of tokens per chunk decay by hundreds of log2 units.
    data["g"] = (data["g"].float() * 6.0 - 12.0).to(torch.bfloat16).contiguous()
    with torch.no_grad():
        data["A_log"].fill_(3.0)
    state_triton = data["state"].clone()
    state_cake = data["state"].clone()
    output_triton = _extend(
        TritonKDAKernel(), data, state_triton, seq_lens, lower_bound=None
    )
    with patch.object(
        CakeKDAKernel,
        "_extend_triton",
        side_effect=AssertionError("unbounded gate must not fall back to Triton"),
    ):
        output_cake = _extend(
            CakeKDAKernel(), data, state_cake, seq_lens, lower_bound=None
        )
    assert torch.isfinite(output_cake).all()
    idx = data["cache_indices"].long()
    for seq, length in enumerate(seq_lens):
        start = sum(seq_lens[:seq])
        out_ref, state_ref = _fp32_reference(data, seq, seq_lens)
        state_err = _rel_l2(state_cake[idx[seq]], state_ref)
        out_err = _rel_l2(output_cake[0, start : start + length], out_ref)
        triton_state_err = _rel_l2(state_triton[idx[seq]], state_ref)
        assert state_err < 1e-2, f"sequence {seq}: state rel L2 {state_err:.4g}"
        assert out_err < 1e-2, f"sequence {seq}: output rel L2 {out_err:.4g}"
        # Same order as (here: no worse than) the Triton prefill it replaces.
        assert state_err <= max(triton_state_err, 1e-2), (
            f"sequence {seq}: export {state_err:.4g} vs Triton {triton_state_err:.4g}"
        )


def test_kda_prefill_plan_cache_reuses_prepared_launch_bitwise():
    """Consecutive layers rebind one prepared launch; results equal an uncached kernel bit for bit."""
    torch.manual_seed(99)
    seq_lens = [64] * 16
    data = _make_inputs(seq_lens, 12)
    cp_kwargs = dict(
        return_intermediate_states=True,
        track_ssm_h_src=torch.arange(len(seq_lens), device="cuda", dtype=torch.int64),
        state_checkpoint_cu_starts=torch.arange(
            len(seq_lens) + 1, device="cuda", dtype=torch.int64
        ),
        num_state_checkpoints=len(seq_lens),
        state_checkpoint_every_n_tokens=64,
    )
    cached = CakeKDAKernel()
    uncached = CakeKDAKernel()
    uncached._kda_prefill_plan_cache_disabled = True
    # Two "layers": same shapes, different state pools (different addresses).
    pools = [data["state"].clone(), (data["state"] * 3.0).clone()]
    results = {}
    for name, kernel in (("cached", cached), ("uncached", uncached)):
        results[name] = []
        for layer_id, pool in enumerate(pools):
            state = pool.clone()
            output, h = kernel.extend(
                data["q"].clone(),
                data["k"].clone(),
                data["v"].clone(),
                data["g"].clone(),
                data["beta"],
                ssm_states=state,
                cache_indices=data["cache_indices"],
                query_start_loc=data["cu_seqlens"],
                A_log=data["A_log"],
                dt_bias=data["dt_bias"],
                lower_bound=None,
                extend_seq_lens_cpu=seq_lens,
                layer_id=layer_id,
                **cp_kwargs,
            )
            results[name].append((output.clone(), state.clone(), h.clone()))
    cache = cached._cake_prefill_plan_cache()
    assert cache is not None and cache.misses == 1 and cache.hits == 1
    for (o_c, s_c, h_c), (o_u, s_u, h_u) in zip(results["cached"], results["uncached"]):
        assert torch.equal(o_c, o_u)
        assert torch.equal(s_c, s_u)
        assert torch.equal(h_c, h_u)


def test_kda_prefill_fp32_checkpoints_track_fp32_reference_per_chunk():
    """With the FP32 export, per-chunk intermediate states stay within the FP32
    reference budget across a long sequence (no BF16 carrier drift)."""
    if not _export_has_fp32_checkpoints(None):
        pytest.skip("installed FlashInfer export writes BF16 checkpoint rows")
    torch.manual_seed(4321)
    seq_lens = [1024]
    data = _make_inputs(seq_lens, 12)
    n_cp = 16
    cp_kwargs = dict(
        return_intermediate_states=True,
        track_ssm_h_src=torch.zeros(1, device="cuda", dtype=torch.int64),
        state_checkpoint_cu_starts=torch.tensor(
            [0, n_cp], device="cuda", dtype=torch.int64
        ),
        num_state_checkpoints=n_cp,
        state_checkpoint_every_n_tokens=64,
    )
    state = data["state"].clone()
    output, h = _extend(
        CakeKDAKernel(), data, state, seq_lens, lower_bound=None, **cp_kwargs
    )
    assert h.dtype == torch.float32
    ref_out, ref_state, ref_chunks = _fp32_reference(data, 0, seq_lens, chunk_every=64)
    idx = data["cache_indices"].long()
    assert _rel_l2(state[idx][0], ref_state) < 1e-2
    errors = [_rel_l2(h[0, j], ref_chunks[j]) for j in range(1, n_cp)]
    assert max(errors) < 1e-2, errors
    # Exact carrier: the last checkpoint is no worse than the first.
    assert errors[-1] <= 2.0 * max(errors[0], 1e-3), errors
