"""Cake Mamba selective state update through sglang.kernels.

Checks that the registry resolves the explicit FlashInfer backend; that the
facade output and the in-place state pool are bitwise identical to
``flashinfer.mamba.selective_state_update(backend="cake")``; that the result
matches a pure-torch step within BF16 tolerance (FP32 state at 1e-3); and that
``supports_selective_state_update`` agrees with FlashInfer's own promotion
decision (``try_cake_selective_state_update`` returned ``True``) on the
promoted T=1 rows and refuses the un-promoted forms (stochastic rounding,
int32 indices). FlashInfer silently runs its non-Cake kernel outside the
promoted rows, so the predicate is the only Cake-ran signal SGLang has.

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


def _make(batch, nheads, ngroups, state_dtype, device, seed):
    """sglang's decode convention: per-head dt/A/D/dt_bias broadcast views."""
    torch.manual_seed(seed)
    slots = batch + 4
    state = (torch.randn(slots, nheads, DIM, DSTATE, device=device) * 0.05).to(
        state_dtype
    )
    x = (torch.randn(batch, nheads, DIM, device=device) * 0.1).bfloat16()
    dt = torch.randn(batch, nheads, device=device)[:, :, None].expand(
        batch, nheads, DIM
    )
    A = (-torch.rand(nheads, device=device) - 1.0)[:, None, None].expand(
        nheads, DIM, DSTATE
    )
    B = (torch.randn(batch, ngroups, DSTATE, device=device) * 0.1).bfloat16()
    C = (torch.randn(batch, ngroups, DSTATE, device=device) * 0.1).bfloat16()
    D = torch.randn(nheads, device=device)[:, None].expand(nheads, DIM)
    dt_bias = (torch.rand(nheads, device=device) - 4.0)[:, None].expand(nheads, DIM)
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
    t = _make(16, nheads, ngroups, torch.bfloat16, device, seed=nheads)
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
    t = _make(batch, nheads, ngroups, torch.float32, device, seed=99)
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
    expected_out, expected_state = _reference(t, state_ref, dt_softplus=False)
    torch.testing.assert_close(
        t["state"].index_select(0, t["idx"]), expected_state, atol=1e-3, rtol=1e-3
    )
    torch.testing.assert_close(t["out"].float(), expected_out, atol=1e-2, rtol=1e-2)


def test_supports_refuses_unpromoted_forms():
    _skip_unless_supported()
    device = torch.device("cuda")
    t = _make(4, 16, 2, torch.bfloat16, device, seed=5)
    base = dict(dt_bias=t["dt_bias"], dt_softplus=True, state_batch_indices=t["idx"])
    args = (t["state"], t["x"], t["dt"], t["A"], t["B"], t["C"], t["D"])
    assert cake_mamba.supports_selective_state_update(*args, **base)
    # Stochastic rounding (sglang's FlashInferSSUBackend passes rand_seed).
    seed = torch.randint(0, 2**31, (1,), device=device)
    assert not cake_mamba.supports_selective_state_update(*args, rand_seed=seed, **base)
    # int32 slot indices are outside the legacy-typed Cake rows.
    assert not cake_mamba.supports_selective_state_update(
        *args,
        dt_bias=t["dt_bias"],
        dt_softplus=True,
        state_batch_indices=t["idx"].to(torch.int32),
    )
    # A compact (non-broadcast) dt layout is rejected by Cake.
    assert not cake_mamba.supports_selective_state_update(
        t["state"], t["x"], t["dt"].contiguous(), t["A"], t["B"], t["C"], t["D"], **base
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
