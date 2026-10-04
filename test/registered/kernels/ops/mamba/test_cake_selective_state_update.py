"""Cake Mamba selective state update through sglang.kernels.

Checks that the registry resolves the explicit FlashInfer backend; that the
facade output and the in-place state pool are bitwise identical to
``flashinfer.mamba.selective_state_update(backend="cake")``; that the result
matches FlashInfer's own validated oracle for this kernel -- the non-Cake
``selective_state_update(backend="flashinfer")`` kernel on identical inputs at
``atol = rtol = 1e-2`` (``tests/mamba/test_cake_selective_state_update.py`` at
FlashInfer ``46340689a5ab``) -- and a pure-torch step within BF16 tolerance
(FP32 state at 1e-3); and that ``supports_selective_state_update`` agrees with
FlashInfer's own promotion decision (``try_cake_selective_state_update``
returned ``True``) on the promoted T=1 rows and refuses the un-promoted forms
(stochastic rounding, int32 indices). FlashInfer silently runs its non-Cake
kernel outside the promoted rows, so the predicate is the only Cake-ran signal
SGLang has.

Input conditioning: ``dt`` is a Mamba time step and must be positive. On the
BF16 rows the kernel applies ``softplus(dt + dt_bias)`` itself, so a Gaussian
``dt`` with the usual ``dt_bias in [-4, -3)`` is fine (FlashInfer's own
distribution). The ``stp_fp32_identity`` row is promoted only with
``dt_softplus=False``, i.e. the caller has already applied softplus; feeding
it the raw Gaussian ``dt`` makes ``exp(dt * A)`` reach ``e^14`` and the BF16
output of a cancelling 128-term sum is then not comparable to any oracle at
1e-2. That row therefore receives an already-positive step.

Skips with the reason when FlashInfer lacks the module or the GPU is outside
sm_100a / sm_103a. The first call per program compiles the source-built
program with nvcc (filelock-guarded); allow for that in ``est_time``.
"""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import mamba as cake_mamba
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.mamba.cake import cake_selective_state_update
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

OP = "mamba.selective_state_update"
DIM = DSTATE = 128
# FlashInfer's bound for Cake vs its non-Cake kernel on identical inputs
# (tests/mamba/test_cake_selective_state_update.py at 46340689a5ab).
FI_ATOL = FI_RTOL = 1e-2


def test_registry_resolves_flashinfer_backend():
    spec = select_kernel(OP, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.mamba:")


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(
        cake_mamba.FI_MODULE, cake_mamba.FI_SSU_JIT_MODULE
    ):
        pytest.skip(
            "installed FlashInfer lacks the Cake selective_state_update programs"
        )
    cc = torch.cuda.get_device_capability()
    if cc not in cake_mamba.ARCHS:
        pytest.skip(
            f"Cake selective_state_update is built for sm_100a/sm_103a, got {cc}"
        )


def _make(batch, nheads, ngroups, state_dtype, device, seed, *, dt_softplus, dim=DIM):
    """sglang's decode convention: per-head dt/A/D/dt_bias broadcast views.

    With ``dt_softplus`` the kernel positivises ``dt + dt_bias`` itself and the
    raw Gaussian step is used; without it the step is supplied already
    positive (``softplus`` of the same Gaussian) with a small positive bias.
    """
    torch.manual_seed(seed)
    slots = batch + 4
    state = (torch.randn(slots, nheads, dim, DSTATE, device=device) * 0.05).to(
        state_dtype
    )
    x = (torch.randn(batch, nheads, dim, device=device) * 0.1).bfloat16()
    dt_raw = torch.randn(batch, nheads, device=device)
    bias_raw = torch.rand(nheads, device=device) - 4.0
    if dt_softplus:
        dt_head, bias_head = dt_raw, bias_raw
    else:
        dt_head = F.softplus(dt_raw + bias_raw)
        bias_head = torch.rand(nheads, device=device) * 0.05
    dt = dt_head[:, :, None].expand(batch, nheads, dim)
    A = (-torch.rand(nheads, device=device) - 1.0)[:, None, None].expand(
        nheads, dim, DSTATE
    )
    B = (torch.randn(batch, ngroups, DSTATE, device=device) * 0.1).bfloat16()
    C = (torch.randn(batch, ngroups, DSTATE, device=device) * 0.1).bfloat16()
    D = torch.randn(nheads, device=device)[:, None].expand(nheads, dim)
    dt_bias = bias_head[:, None].expand(nheads, dim)
    indices = torch.randperm(slots, device=device)[:batch].to(torch.int64)
    out = torch.empty_like(x)
    return dict(
        state=state,
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D=D,
        dt_bias=dt_bias,
        idx=indices,
        out=out,
    )


def _reference(t, state_before, dt_softplus):
    nheads, ngroups = state_before.shape[1], t["B"].shape[1]
    s = state_before.index_select(0, t["idx"]).float()
    dt = t["dt"][:, :, 0].float() + t["dt_bias"][:, 0].float()
    if dt_softplus:
        dt = F.softplus(dt)
    A = t["A"][:, 0, 0].float()
    Bg = t["B"].float().repeat_interleave(nheads // ngroups, dim=1)
    Cg = t["C"].float().repeat_interleave(nheads // ngroups, dim=1)
    s = s * torch.exp(dt * A)[:, :, None, None] + dt[:, :, None, None] * (
        t["x"].float()[:, :, :, None] * Bg[:, :, None, :]
    )
    y = (
        torch.einsum("bhdn,bhn->bhd", s, Cg)
        + t["D"][:, 0].float()[None, :, None] * t["x"].float()
    )
    return y, s


def _flashinfer_reference(t, state_before, kwargs):
    """FlashInfer's non-Cake kernel on the identical inputs (FI's own oracle)."""
    from flashinfer.mamba import selective_state_update as fi_direct

    state = state_before.clone()
    out = fi_direct(
        state,
        t["x"],
        t["dt"],
        t["A"],
        t["B"],
        t["C"],
        t["D"],
        backend="flashinfer",
        **kwargs,
    )
    return out, state


def _assert_matches_flashinfer_oracle(t, state_before, kwargs):
    out_ref, state_ref = _flashinfer_reference(t, state_before, kwargs)
    torch.cuda.synchronize()
    torch.testing.assert_close(t["out"], out_ref, atol=FI_ATOL, rtol=FI_RTOL)
    torch.testing.assert_close(t["state"], state_ref, atol=FI_ATOL, rtol=FI_RTOL)


def _strict_cake(monkeypatch):
    """Record FlashInfer's own Cake promotion decision for the call."""
    import flashinfer.jit.mamba.cake_selective_state_update as fi_cake

    hits = []
    original = fi_cake.try_cake_selective_state_update

    def wrapper(**kwargs):
        hit = original(**kwargs)
        hits.append(hit)
        return hit

    monkeypatch.setattr(fi_cake, "try_cake_selective_state_update", wrapper)
    return hits


@pytest.mark.parametrize("nheads,ngroups", [(16, 2), (64, 8)])
def test_bf16_t1_row_matches_flashinfer_and_reference(nheads, ngroups, monkeypatch):
    _skip_unless_supported()
    device = torch.device("cuda")
    t = _make(
        16, nheads, ngroups, torch.bfloat16, device, seed=nheads, dt_softplus=True
    )
    kwargs = dict(dt_bias=t["dt_bias"], dt_softplus=True, state_batch_indices=t["idx"])
    assert cake_mamba.supports_selective_state_update(
        t["state"], t["x"], t["dt"], t["A"], t["B"], t["C"], t["D"], **kwargs
    )
    state_fi, state_ref = t["state"].clone(), t["state"].clone()
    hits = _strict_cake(monkeypatch)
    got = cake_selective_state_update(
        t["state"],
        t["x"],
        t["dt"],
        t["A"],
        t["B"],
        t["C"],
        t["D"],
        out=t["out"],
        **kwargs,
    )
    assert hits == [True], "promoted row fell back inside FlashInfer"
    assert got is t["out"]
    from flashinfer.mamba import selective_state_update as fi_direct

    out_fi = fi_direct(
        state_fi,
        t["x"],
        t["dt"],
        t["A"],
        t["B"],
        t["C"],
        t["D"],
        backend="cake",
        **kwargs,
    )
    torch.cuda.synchronize()
    assert torch.equal(t["out"], out_fi)
    assert torch.equal(t["state"], state_fi)
    _assert_matches_flashinfer_oracle(t, state_ref, kwargs)
    expected_out, expected_state = _reference(t, state_ref, dt_softplus=True)
    torch.testing.assert_close(
        t["state"].index_select(0, t["idx"]).float(),
        expected_state,
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(t["out"].float(), expected_out, atol=1e-2, rtol=1e-2)
    untouched = torch.ones(t["state"].shape[0], dtype=torch.bool, device=device)
    untouched[t["idx"]] = False
    assert torch.equal(t["state"][untouched], state_ref[untouched])


def test_fp32_identity_row_matches_flashinfer_and_reference(monkeypatch):
    _skip_unless_supported()
    device = torch.device("cuda")
    nheads, ngroups = 16, 2
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    batch = (8 * sms + nheads - 1) // nheads  # B * nheads >= 8 * SMs
    # The identity row is promoted without softplus: the caller supplies the
    # positive step (see the module docstring).
    t = _make(batch, nheads, ngroups, torch.float32, device, seed=99, dt_softplus=False)
    kwargs = dict(dt_bias=t["dt_bias"], dt_softplus=False, state_batch_indices=t["idx"])
    assert cake_mamba.supports_selective_state_update(
        t["state"], t["x"], t["dt"], t["A"], t["B"], t["C"], t["D"], **kwargs
    )
    state_fi, state_ref = t["state"].clone(), t["state"].clone()
    hits = _strict_cake(monkeypatch)
    cake_selective_state_update(
        t["state"],
        t["x"],
        t["dt"],
        t["A"],
        t["B"],
        t["C"],
        t["D"],
        out=t["out"],
        **kwargs,
    )
    assert hits == [True], "promoted row fell back inside FlashInfer"
    from flashinfer.mamba import selective_state_update as fi_direct

    out_fi = fi_direct(
        state_fi,
        t["x"],
        t["dt"],
        t["A"],
        t["B"],
        t["C"],
        t["D"],
        backend="cake",
        **kwargs,
    )
    torch.cuda.synchronize()
    assert torch.equal(t["out"], out_fi)
    assert torch.equal(t["state"], state_fi)
    _assert_matches_flashinfer_oracle(t, state_ref, kwargs)
    expected_out, expected_state = _reference(t, state_ref, dt_softplus=False)
    torch.testing.assert_close(
        t["state"].index_select(0, t["idx"]), expected_state, atol=1e-3, rtol=1e-3
    )
    torch.testing.assert_close(t["out"].float(), expected_out, atol=1e-2, rtol=1e-2)


def _make_hd64_raw(batch, nheads, ngroups, state_dtype, device, seed):
    """The Nemotron-H / granite decode call exactly as the hybrid backend
    issues it: BF16 ``dt``/``D``/``dt_bias`` broadcasts, int32 slot table with
    ``pad_slot_id=-1`` padding rows, and ``x``/``B``/``C`` as views into the
    fused ``xBC`` projection (padded batch stride)."""
    t = _make(batch, nheads, ngroups, state_dtype, device, seed, dt_softplus=True, dim=64)
    width = nheads * 64 + 2 * ngroups * 128
    xbc = (torch.randn(batch, width, device=device) * 0.1).bfloat16()
    t["x"] = xbc[:, : nheads * 64].view(batch, nheads, 64)
    t["B"] = xbc[:, nheads * 64 : nheads * 64 + ngroups * 128].view(batch, ngroups, 128)
    t["C"] = xbc[:, nheads * 64 + ngroups * 128 :].view(batch, ngroups, 128)
    # Convert the per-head bases, then re-expand: the engine's coefficients are
    # broadcast views (``dt.stride(1) == 1``, ``dt.stride(2) == 0``).
    t["dt"] = t["dt"][:, :, 0].bfloat16()[:, :, None].expand(batch, nheads, 64)
    t["D"] = t["D"][:, 0].bfloat16()[:, None].expand(nheads, 64)
    t["dt_bias"] = t["dt_bias"][:, 0].bfloat16()[:, None].expand(nheads, 64)
    t["idx"] = t["idx"].to(torch.int32)
    t["out"] = torch.empty(batch, nheads, 64, dtype=torch.bfloat16, device=device)
    return t


@pytest.mark.parametrize(
    "batch,nheads,ngroups,state_dtype",
    [(2, 128, 8, torch.bfloat16), (64, 128, 8, torch.bfloat16), (8, 128, 8, torch.float32)],
)
def test_hd64_raw_decode_row_matches_flashinfer_and_reference(
    batch, nheads, ngroups, state_dtype, monkeypatch
):
    """Row-owner (batch 2), paired TMA (batch 64) and FP32 programs on the raw
    engine ABI, with one padded slot (``pad_slot_id=-1``) that must produce a
    zero-state output and leave the state pool untouched."""
    _skip_unless_supported()
    device = torch.device("cuda")
    t = _make_hd64_raw(batch, nheads, ngroups, state_dtype, device, seed=batch)
    idx = t["idx"].clone()
    idx[-1] = -1
    kwargs = dict(
        dt_bias=t["dt_bias"], dt_softplus=True, state_batch_indices=idx, pad_slot_id=-1
    )
    args = (t["state"], t["x"], t["dt"], t["A"], t["B"], t["C"], t["D"])
    assert cake_mamba.supports_selective_state_update(*args, **kwargs)
    state_fi, state_ref = t["state"].clone(), t["state"].clone()
    hits = _strict_cake(monkeypatch)
    got = cake_selective_state_update(*args, out=t["out"], **kwargs)
    assert hits == [True], "promoted row fell back inside FlashInfer"
    assert got is t["out"]
    from flashinfer.mamba import selective_state_update as fi_direct

    out_fi = fi_direct(state_fi, *args[1:], backend="cake", **kwargs)
    torch.cuda.synchronize()
    assert torch.equal(t["out"], out_fi)
    assert torch.equal(t["state"], state_fi)
    _assert_matches_flashinfer_oracle(t, state_ref, kwargs)
    live = idx[:-1].to(torch.int64)
    expected_out, expected_state = _reference(
        dict(t, idx=live, x=t["x"][:-1], B=t["B"][:-1], C=t["C"][:-1], dt=t["dt"][:-1]),
        state_ref,
        dt_softplus=True,
    )
    tol = 1e-2
    torch.testing.assert_close(
        t["state"].index_select(0, live).float(), expected_state, atol=tol, rtol=tol
    )
    torch.testing.assert_close(t["out"][:-1].float(), expected_out, atol=tol, rtol=tol)
    # The padded row reads a zero state: output = dt * (x . B) . C + D * x.
    pad_out, _ = _reference(
        dict(t, idx=live[:1], x=t["x"][-1:], B=t["B"][-1:], C=t["C"][-1:], dt=t["dt"][-1:]),
        torch.zeros_like(state_ref),
        dt_softplus=True,
    )
    torch.testing.assert_close(t["out"][-1:].float(), pad_out, atol=tol, rtol=tol)
    untouched = torch.ones(t["state"].shape[0], dtype=torch.bool, device=device)
    untouched[live] = False
    assert torch.equal(t["state"][untouched], state_ref[untouched])


def test_hd64_raw_decode_captures_into_a_cuda_graph():
    _skip_unless_supported()
    device = torch.device("cuda")
    t = _make_hd64_raw(16, 128, 8, torch.bfloat16, device, seed=7)
    kwargs = dict(
        dt_bias=t["dt_bias"], dt_softplus=True, state_batch_indices=t["idx"], pad_slot_id=-1
    )
    args = (t["state"], t["x"], t["dt"], t["A"], t["B"], t["C"], t["D"])
    state_ref = t["state"].clone()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(2):  # warm the JIT path outside capture
            t["state"].copy_(state_ref)
            cake_selective_state_update(*args, out=t["out"], **kwargs)
        torch.cuda.synchronize()
        t["state"].copy_(state_ref)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            cake_selective_state_update(*args, out=t["out"], **kwargs)
    t["state"].copy_(state_ref)
    t["out"].fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    _assert_matches_flashinfer_oracle(t, state_ref, kwargs)


def test_supports_refuses_unpromoted_forms():
    _skip_unless_supported()
    device = torch.device("cuda")
    t = _make(4, 16, 2, torch.bfloat16, device, seed=5, dt_softplus=True)
    base = dict(dt_bias=t["dt_bias"], dt_softplus=True, state_batch_indices=t["idx"])
    args = (t["state"], t["x"], t["dt"], t["A"], t["B"], t["C"], t["D"])
    assert cake_mamba.supports_selective_state_update(*args, **base)
    # Stochastic rounding (sglang's FlashInferSSUBackend passes rand_seed).
    seed = torch.randint(0, 2**31, (1,), device=device)
    assert not cake_mamba.supports_selective_state_update(*args, rand_seed=seed, **base)
    # int32 slot indices are outside the legacy-typed 128x128 Cake rows (the
    # raw engine ABI is served by the headdim-64 row only).
    assert not cake_mamba.supports_selective_state_update(
        *args,
        dt_bias=t["dt_bias"],
        dt_softplus=True,
        state_batch_indices=t["idx"].to(torch.int32),
    )
    # The headdim-64 raw row refuses non-dense rows (a transposed x view).
    r = _make_hd64_raw(4, 128, 8, torch.bfloat16, device, seed=11)
    raw = dict(dt_bias=r["dt_bias"], dt_softplus=True, state_batch_indices=r["idx"], pad_slot_id=-1)
    raw_args = (r["state"], r["x"], r["dt"], r["A"], r["B"], r["C"], r["D"])
    assert cake_mamba.supports_selective_state_update(*raw_args, **raw)
    strided_x = r["x"].transpose(0, 1).contiguous().transpose(0, 1)
    assert not cake_mamba.supports_selective_state_update(
        r["state"], strided_x, *raw_args[2:], **raw
    )
    # A compact (non-broadcast) dt layout is rejected by Cake.
    assert not cake_mamba.supports_selective_state_update(
        t["state"], t["x"], t["dt"].contiguous(), t["A"], t["B"], t["C"], t["D"], **base
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
