"""Cake Mamba2 SSD combined prefill through sglang.kernels.

Prepared-runner pattern: ``cake_ssd_combined(...)`` constructs the FlashInfer
``SSDCombined(backend="cake")`` runner; ``run()`` is launched, then launched
again with new activations in the same caller-owned ``out`` buffer. Checks
that the registry resolves the explicit FlashInfer backends; that the facade
output and final states are bitwise identical to constructing the FlashInfer
runner directly and to the functional ``cake_ssd_combined_fwd``; and that
output, final states and selective checkpoint rows (PR #35444's compact
checkpoints) match FlashInfer's own validated oracle for this kernel, in
batched mode and in packed-varlen mode (``seq_idx`` + chunk metadata).

Oracle and bound (FlashInfer ``tests/mamba/test_cake_ssd_combined.py`` at
``46340689a5ab``): the Cake route is validated against FlashInfer's CuTe SSD
backend (``SSDCombined(backend="cute")``) on identical inputs. Both backends
run the chunked SSD algorithm with the inter-chunk state carried in the state
dtype (BF16 here), which a per-token FP32 recurrence does not model; the bound
is FlashInfer's ``_assert_cute_parity``: ``rtol = 1e-2`` and ``atol =
max(1e-2, 5e-4 * amax(|reference|))`` per returned tensor ("cancellation ties
the error to the head's magnitude rather than the entry's"). Checkpoint states
are compared at ``atol = rtol = 1e-2`` against the CuTe final state of the
sequence prefix, as FlashInfer does.

Checkpoint contract (``flashinfer/mamba/ssd_combined.py`` docstring + the
generated kernels ``mamba_ssd_q_tmem_alias_*``): ``checkpoint_token_indices``
holds one *exclusive* token boundary per sequence -- sequence-relative in
batched mode, absolute in the packed token axis for varlen -- and the state
is captured only when that boundary is the end of a logical chunk
(``checkpoint_token == segment_end``): a multiple of ``chunk_size`` in
batched mode, or a boundary exposed through ``chunk_indices`` /
``chunk_offsets`` in varlen mode (the packed shape SGLang builds). Boundaries
anywhere else, negative entries and negative slots capture nothing and leave
the slot untouched.

The runner materializes strided inputs into graph-stable storage and owns its
workspaces, so one runner per stream; preparation (first ``run`` per shape)
must happen eagerly before any CUDA-graph capture. Skips with the reason when
FlashInfer lacks the module or the GPU is outside sm_100a / sm_103a.
"""

import sys

import pytest
import torch

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
# Unused checkpoint slot: must stay untouched (NaN-filled) after the launch.
UNTOUCHED_SLOT = 1


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


def _varlen_metadata(lengths, device, extra_boundaries=()):
    """``seq_idx`` plus the logical-chunk metadata of a packed batch.

    A logical chunk starts at every physical chunk start and at every
    sequence start; ``extra_boundaries`` (absolute packed token indices) add
    further logical boundaries, which is how a caller exposes a checkpoint
    inside a physical chunk.
    """
    total = sum(lengths)
    seq_idx = torch.empty((1, total), dtype=torch.int32, device=device)
    starts, start = [], 0
    for seq, n in enumerate(lengths):
        seq_idx[0, start : start + n] = seq
        starts.append(start)
        start += n
    boundaries = set(starts) | set(int(b) for b in extra_boundaries)
    chunk_indices, chunk_offsets = [], []
    for chunk in range(total // CHUNK):
        lo, hi = chunk * CHUNK, (chunk + 1) * CHUNK
        for offset in sorted({0} | {b - lo for b in boundaries if lo < b < hi}):
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


def _cute_reference(case):
    """FlashInfer's validated oracle for the Cake route: its CuTe SSD backend."""
    from flashinfer.mamba import SSDCombined

    return SSDCombined(**case["ctor"], backend="cute").run(
        *case["tensors"], **case["run"]
    )


def _assert_cute_parity(actual, expected):
    """FlashInfer's bound (tests/mamba/test_cake_ssd_combined.py, 46340689a5ab)."""
    for index in (0, 1):
        reference = expected[index]
        # Cancellation ties the error to the head's magnitude rather than the
        # entry's; the tensor max is a coarse bound on that.
        atol = max(1e-2, 5e-4 * reference.abs().amax().item())
        torch.testing.assert_close(actual[index], reference, atol=atol, rtol=1e-2)


def _cute_prefix_final_state(case, *, batch_index, start, length, sequence):
    """CuTe final state after ``length`` tokens of one sequence (batched run)."""
    from flashinfer.mamba import SSDCombined

    x, dt, A, B, C = case["tensors"]
    run = case["run"]
    sl = slice(start, start + length)
    prefix = {
        **run,
        "z": run["z"][batch_index : batch_index + 1, sl].contiguous(),
        "initial_states": run["initial_states"][sequence : sequence + 1].contiguous(),
        "seq_idx": None,
        "chunk_indices": None,
        "chunk_offsets": None,
    }
    ctor = {**case["ctor"], "has_varlen": False}
    _, final = SSDCombined(**ctor, backend="cute").run(
        x[batch_index : batch_index + 1, sl].contiguous(),
        dt[batch_index : batch_index + 1, sl].contiguous(),
        A,
        B[batch_index : batch_index + 1, sl].contiguous(),
        C[batch_index : batch_index + 1, sl].contiguous(),
        **prefix,
    )
    return final[0]


def _supports(case, **extra):
    x, dt, A, B, C = case["tensors"]
    return cake_mamba.supports_ssd_combined(
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
        chunk_indices=extra.pop("chunk_indices", case["run"]["chunk_indices"]),
        chunk_offsets=extra.pop("chunk_offsets", case["run"]["chunk_offsets"]),
        **extra,
    )


@pytest.mark.parametrize("varlen", [False, True], ids=["batched", "varlen"])
def test_runner_matches_flashinfer_functional_and_reference(varlen):
    _skip_unless_supported()
    device = torch.device("cuda")
    case = _case(device, varlen=varlen, seed=int(varlen))
    x, dt, A, B, C = case["tensors"]
    assert _supports(case, out=case["out"])
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
    assert torch.isfinite(out.float()).all() and torch.isfinite(final.float()).all()
    _assert_cute_parity((out, final), _cute_reference(case))

    # Second run: new activations in the same storage, same out buffer.
    x.copy_(torch.randn_like(x.float()).bfloat16())
    case["run"]["z"].copy_(torch.randn_like(x.float()).bfloat16())
    out2, final2 = runner.run(*case["tensors"], out=case["out"], **case["run"])
    torch.cuda.synchronize()
    _assert_cute_parity((out2, final2), _cute_reference(case))


@pytest.mark.parametrize("varlen", [False, True], ids=["batched", "varlen"])
def test_runner_writes_selective_checkpoint_states(varlen):
    _skip_unless_supported()
    device = torch.device("cuda")
    case = _case(device, varlen=varlen, seed=7 + int(varlen))
    lengths = case["lengths"]
    if varlen:
        # Packed [0, 96) + [96, 256). Sequence 0 checkpoints at its end (96,
        # a logical boundary because sequence 1 starts there); sequence 1
        # after 128 of its tokens (absolute 224), a boundary inside physical
        # chunk 1 that the caller exposes through the chunk metadata.
        boundaries = [96, 224]
        seq_idx, chunk_indices, chunk_offsets = _varlen_metadata(
            lengths, device, extra_boundaries=(224,)
        )
        assert torch.equal(
            chunk_indices.cpu(), torch.tensor([0, 0, 1, 1], dtype=torch.int32)
        )
        assert torch.equal(
            chunk_offsets.cpu(), torch.tensor([0, 96, 0, 96], dtype=torch.int32)
        )
        metadata = dict(chunk_indices=chunk_indices, chunk_offsets=chunk_offsets)
        prefix = [(0, 0, 96), (0, 96, 128)]  # (batch index, start, length)
    else:
        # Sequence-relative: sequence 0 after one chunk, sequence 1 at its end.
        boundaries = [128, 256]
        metadata = {}
        prefix = [(0, 0, 128), (1, 0, 256)]
    slots = [2, 0]
    token_indices = torch.tensor(boundaries, device=device, dtype=torch.int32)
    slot_indices = torch.tensor(slots, device=device, dtype=torch.int32)
    checkpoint_states = torch.full(
        (3, NHEADS, HEADDIM, DSTATE), float("nan"), device=device
    ).bfloat16()
    assert _supports(
        case,
        checkpoint_token_indices=token_indices,
        checkpoint_state_slots=slot_indices,
        checkpoint_states=checkpoint_states,
        **metadata,
    )
    runner = cake_ssd_combined(**case["ctor"])
    out, final = runner.run(
        *case["tensors"],
        checkpoint_token_indices=token_indices,
        checkpoint_state_slots=slot_indices,
        checkpoint_states=checkpoint_states,
        **{**case["run"], **metadata},
    )
    torch.cuda.synchronize()
    # The exposed logical boundary does not change the result.
    expected = _cute_reference(case)
    _assert_cute_parity((out, final), expected)
    # Written slots: CuTe final state of the sequence prefix (a checkpoint at
    # the sequence end is its final state); the unused slot stays untouched.
    for seq, (slot, (b, start, length)) in enumerate(zip(slots, prefix)):
        assert torch.isfinite(checkpoint_states[slot].float()).all()
        seq_end = sum(lengths[: seq + 1]) if varlen else lengths[seq]
        if start + length == seq_end:
            reference = expected[1][seq]
        else:
            reference = _cute_prefix_final_state(
                case, batch_index=b, start=start, length=length, sequence=seq
            )
        torch.testing.assert_close(
            checkpoint_states[slot], reference, atol=1e-2, rtol=1e-2
        )
    assert torch.isnan(checkpoint_states[UNTOUCHED_SLOT]).all()

    # A boundary that is not a logical chunk end (72) and a negative entry
    # capture nothing: the NaN fill survives in every slot.
    unaligned = torch.full_like(checkpoint_states, float("nan"))
    runner.run(
        *case["tensors"],
        checkpoint_token_indices=torch.tensor(
            [72, -1], device=device, dtype=torch.int32
        ),
        checkpoint_state_slots=slot_indices,
        checkpoint_states=unaligned,
        **{**case["run"], **metadata},
    )
    torch.cuda.synchronize()
    assert torch.isnan(unaligned).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
