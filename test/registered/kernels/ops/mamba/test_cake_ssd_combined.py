"""Cake Mamba2 SSD combined prefill through sglang.kernels.

Prepared-runner pattern: ``cake_ssd_combined(...)`` constructs the FlashInfer
``SSDCombined(backend="cake")`` runner; ``run()`` is launched, then launched
again with new activations in the same caller-owned ``out`` buffer. Checks
that the registry resolves the explicit FlashInfer backends; that the facade
output and final states are bitwise identical to constructing the FlashInfer
runner directly and to the functional ``cake_ssd_combined_fwd``; and that
output, final states and selective checkpoint rows (PR #35444's compact
checkpoints) match a pure-torch SSD recurrence within BF16 tolerance, in
batched mode and in packed-varlen mode (``seq_idx`` + chunk metadata).

The runner materializes strided inputs into graph-stable storage and owns its
workspaces, so one runner per stream; preparation (first ``run`` per shape)
must happen eagerly before any CUDA-graph capture. Skips with the reason when
FlashInfer lacks the module or the GPU is outside sm_100a / sm_103a.
"""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import mamba as cake_mamba
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.mamba.cake import cake_ssd_combined, cake_ssd_combined_fwd
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

CHUNK = cake_mamba.SSD_CHUNK_SIZE
HEADDIM = cake_mamba.SSD_HEADDIM
DSTATE = cake_mamba.SSD_DSTATE
NHEADS, NGROUPS = 16, 8


@pytest.mark.parametrize("op", ["mamba.ssd_combined", "mamba.ssd_combined_fwd"])
def test_registry_resolves_flashinfer_backend(op):
    spec = select_kernel(op, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.mamba:")


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(cake_mamba.FI_MODULE, cake_mamba.FI_SSD_MODULE):
        pytest.skip("installed FlashInfer lacks flashinfer.mamba.cake_ssd_combined")
    cc = torch.cuda.get_device_capability()
    if cc not in cake_mamba.ARCHS:
        pytest.skip(f"Cake SSDCombined is built for sm_100a/sm_103a, device is {cc}")


def _varlen_metadata(lengths, device):
    total = sum(lengths)
    seq_idx = torch.empty((1, total), dtype=torch.int32, device=device)
    start = 0
    for seq, n in enumerate(lengths):
        seq_idx[0, start : start + n] = seq
        start += n
    chunk_indices, chunk_offsets = [], []
    for chunk in range(total // CHUNK):
        values = seq_idx[0, chunk * CHUNK : (chunk + 1) * CHUNK]
        previous = torch.cat((values[:1] - 1, values[:-1]))
        for offset in (values != previous).nonzero(as_tuple=True)[0].tolist():
            chunk_indices.append(chunk)
            chunk_offsets.append(offset)
    return (
        seq_idx,
        torch.tensor(chunk_indices, dtype=torch.int32, device=device),
        torch.tensor(chunk_offsets, dtype=torch.int32, device=device),
    )


def _case(device, *, varlen, seed):
    torch.manual_seed(seed)
    batch, seqlen = (1, 256) if varlen else (2, 256)
    x = torch.randn(batch, seqlen, NHEADS, HEADDIM, device=device).bfloat16()
    dt = torch.randn(batch, seqlen, NHEADS, device=device)
    A = -torch.rand(NHEADS, device=device) - 1.0
    B = torch.randn(batch, seqlen, NGROUPS, DSTATE, device=device).bfloat16()
    C = torch.randn_like(B)
    D = torch.randn(NHEADS, device=device).bfloat16()
    z = torch.randn_like(x)
    dt_bias = torch.rand(NHEADS, device=device) - 4.0
    lengths = (96, 160) if varlen else (seqlen,) * batch
    initial_states = (
        torch.randn(len(lengths), NHEADS, HEADDIM, DSTATE, device=device) * 0.1
    ).bfloat16()
    if varlen:
        seq_idx, chunk_indices, chunk_offsets = _varlen_metadata(lengths, device)
    else:
        seq_idx = chunk_indices = chunk_offsets = None
    out = torch.empty(
        batch,
        NHEADS,
        HEADDIM,
        seqlen // CHUNK,
        CHUNK,
        device=device,
        dtype=torch.bfloat16,
    )
    return dict(
        tensors=(x, dt, A, B, C),
        lengths=lengths,
        run=dict(
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=True,
            dt_limit=(0.0, float("inf")),
            initial_states=initial_states,
            seq_idx=seq_idx,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            return_final_states=True,
        ),
        ctor=dict(
            chunk_size=CHUNK,
            nheads=NHEADS,
            headdim=HEADDIM,
            dstate=DSTATE,
            ngroups=NGROUPS,
            io_dtype=torch.bfloat16,
            state_dtype=torch.bfloat16,
            has_d=True,
            d_has_hdim=False,
            has_initial_states=True,
            has_varlen=varlen,
            has_z=True,
            seq_idx_dtype=torch.int32,
        ),
        out=out,
    )


def _reference(case, checkpoint_boundaries=None):
    """FP32 per-token SSD recurrence over packed sequences.

    Returns token-major output, final states and, when
    ``checkpoint_boundaries`` (exclusive token counts per sequence) is given,
    the state after that many tokens of each sequence.
    """
    x, dt, A, B, C = case["tensors"]
    run = case["run"]
    batch, seqlen = x.shape[:2]
    heads_per_group = NHEADS // NGROUPS
    dt_p = F.softplus(dt.float() + run["dt_bias"].float())
    out = torch.empty(batch, seqlen, NHEADS, HEADDIM, device=x.device)
    finals, checkpoints = [], []
    seq = 0
    for b in range(batch):
        start = 0
        for n in case["lengths"] if run["seq_idx"] is not None else (seqlen,):
            h = run["initial_states"][seq].float()  # [H, D, N]
            for t in range(start, start + n):
                xt, Bt, Ct = x[b, t].float(), B[b, t].float(), C[b, t].float()
                Bg = Bt.repeat_interleave(heads_per_group, dim=0)
                Cg = Ct.repeat_interleave(heads_per_group, dim=0)
                dA = torch.exp(dt_p[b, t] * A)
                h = h * dA[:, None, None] + dt_p[b, t][:, None, None] * (
                    xt[:, :, None] * Bg[:, None, :]
                )
                y = torch.einsum("hdn,hn->hd", h, Cg) + run["D"].float()[:, None] * xt
                out[b, t] = y * F.silu(run["z"][b, t].float())
                if checkpoint_boundaries is not None and t - start + 1 == int(
                    checkpoint_boundaries[seq]
                ):
                    checkpoints.append(h.clone())
            finals.append(h)
            start += n
            seq += 1
    return out, torch.stack(finals), checkpoints


def _assert_reference(case, out, final, checkpoint_states=None, boundaries=None):
    expected_out, expected_final, expected_cp = _reference(case, boundaries)
    torch.testing.assert_close(out.float(), expected_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(final.float(), expected_final, atol=1e-2, rtol=1e-2)
    if checkpoint_states is not None:
        torch.testing.assert_close(
            checkpoint_states.float(), torch.stack(expected_cp), atol=1e-2, rtol=1e-2
        )


@pytest.mark.parametrize("varlen", [False, True], ids=["batched", "varlen"])
def test_runner_matches_flashinfer_functional_and_reference(varlen):
    _skip_unless_supported()
    device = torch.device("cuda")
    case = _case(device, varlen=varlen, seed=int(varlen))
    x, dt, A, B, C = case["tensors"]
    assert cake_mamba.supports_ssd_combined(
        x,
        dt,
        A,
        B,
        C,
        D=case["run"]["D"],
        z=case["run"]["z"],
        dt_bias=case["run"]["dt_bias"],
        initial_states=case["run"]["initial_states"],
        seq_idx=case["run"]["seq_idx"],
        chunk_indices=case["run"]["chunk_indices"],
        chunk_offsets=case["run"]["chunk_offsets"],
        out=case["out"],
    )
    runner = cake_ssd_combined(**case["ctor"])
    out, final = runner.run(*case["tensors"], out=case["out"], **case["run"])
    assert out.untyped_storage().data_ptr() == case["out"].untyped_storage().data_ptr()
    from flashinfer.mamba import SSDCombined

    out_fi, final_fi = SSDCombined(**case["ctor"], backend="cake").run(
        *case["tensors"], **case["run"]
    )
    out_fn, final_fn = cake_ssd_combined_fwd(*case["tensors"], **case["run"])
    torch.cuda.synchronize()
    assert torch.equal(out, out_fi) and torch.equal(final, final_fi)
    assert torch.equal(out, out_fn) and torch.equal(final, final_fn)
    _assert_reference(case, out, final)

    # Second run: new activations in the same storage, same out buffer.
    x.copy_(torch.randn_like(x.float()).bfloat16())
    case["run"]["z"].copy_(torch.randn_like(x.float()).bfloat16())
    out2, final2 = runner.run(*case["tensors"], out=case["out"], **case["run"])
    torch.cuda.synchronize()
    _assert_reference(case, out2, final2)


@pytest.mark.parametrize("varlen", [False, True], ids=["batched", "varlen"])
def test_runner_writes_selective_checkpoint_states(varlen):
    _skip_unless_supported()
    device = torch.device("cuda")
    case = _case(device, varlen=varlen, seed=7 + int(varlen))
    num_seqs = len(case["lengths"])
    # Exclusive per-sequence boundaries (sequence-relative in batched mode,
    # absolute in the packed token axis for varlen), one chunk-unaligned.
    relative = [128, 72] if not varlen else [72, 96 + 128]
    boundaries = torch.tensor(relative, device=device, dtype=torch.int32)
    slots = torch.tensor([1, 0], device=device, dtype=torch.int32)
    checkpoint_states = torch.full(
        (num_seqs, NHEADS, HEADDIM, DSTATE), float("nan"), device=device
    ).bfloat16()
    x, dt, A, B, C = case["tensors"]
    assert cake_mamba.supports_ssd_combined(
        x,
        dt,
        A,
        B,
        C,
        D=case["run"]["D"],
        z=case["run"]["z"],
        dt_bias=case["run"]["dt_bias"],
        initial_states=case["run"]["initial_states"],
        seq_idx=case["run"]["seq_idx"],
        chunk_indices=case["run"]["chunk_indices"],
        chunk_offsets=case["run"]["chunk_offsets"],
        checkpoint_token_indices=boundaries,
        checkpoint_state_slots=slots,
        checkpoint_states=checkpoint_states,
    )
    runner = cake_ssd_combined(**case["ctor"])
    out, final = runner.run(
        *case["tensors"],
        checkpoint_token_indices=boundaries,
        checkpoint_state_slots=slots,
        checkpoint_states=checkpoint_states,
        **case["run"],
    )
    torch.cuda.synchronize()
    assert torch.isfinite(checkpoint_states).all()
    sequence_relative = [72, 128] if varlen else relative
    _assert_reference(
        case,
        out,
        final,
        checkpoint_states[slots.long()],
        boundaries=sequence_relative,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
