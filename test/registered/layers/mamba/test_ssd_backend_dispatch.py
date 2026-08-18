from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.mamba.triton_ops import ssu_dispatch


class _FakeSSDCombined:
    def __init__(self) -> None:
        self.call = None

    def run(self, x, dt, A, B, C, **kwargs):
        self.call = (x, dt, A, B, C, kwargs)
        output = torch.arange(x.numel(), dtype=torch.float32).reshape(x.shape).to(x.dtype)
        final = torch.ones(
            (
                kwargs["initial_states"].shape[0],
                x.shape[2],
                x.shape[3],
                B.shape[3],
            ),
            dtype=kwargs["initial_states"].dtype,
        )
        return output, final


def _flashinfer_backend_without_imports():
    backend = object.__new__(ssu_dispatch.FlashInferSSDCombinedSSUBackend)
    backend._prefill_backend = "cute"
    backend._prefill_runners = {}
    backend._zero_initial_states = {}
    return backend


def test_existing_flashinfer_backend_preserves_triton_prefill():
    backend = object.__new__(ssu_dispatch.FlashInferSSUBackend)
    backend._prefill_backend = None
    observed = {}

    def triton_prefill(*args, **kwargs):
        observed["call"] = (args, kwargs)
        return "triton-prefill"

    backend._prefill_kernel = triton_prefill
    tensors = tuple(torch.empty(1) for _ in range(5))

    assert backend.chunk_scan_combined(*tensors, 128) == "triton-prefill"
    positional, keyword = observed["call"]
    assert len(positional) == 6 and positional[-1] == 128
    assert all(actual is expected for actual, expected in zip(positional, tensors))
    assert keyword["return_final_states"] is False


def test_flashinfer_prefill_pads_with_identity_and_copies_token_major(monkeypatch):
    backend = _flashinfer_backend_without_imports()
    runner = _FakeSSDCombined()
    monkeypatch.setattr(backend, "_get_prefill_runner", lambda **_: runner)

    seqlen, nheads, headdim, ngroups, dstate = 130, 2, 4, 1, 3
    x = torch.randn(1, seqlen, nheads, headdim, dtype=torch.bfloat16)
    dt = torch.randn(1, seqlen, nheads, dtype=torch.bfloat16)
    B = torch.randn(1, seqlen, ngroups, dstate, dtype=torch.bfloat16)
    C = torch.randn_like(B)
    seq_idx = torch.cat(
        (
            torch.zeros(65, dtype=torch.int32),
            torch.ones(65, dtype=torch.int32),
        )
    ).unsqueeze(0)
    chunk_indices = torch.tensor([0, 0, 1], dtype=torch.int32)
    chunk_offsets = torch.tensor([0, 65, 0], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 65, 130], dtype=torch.int32)
    out = torch.empty_like(x)

    intermediate, final = backend.chunk_scan_combined(
        x,
        dt,
        -torch.ones(nheads),
        B,
        C,
        128,
        D=torch.ones(nheads, dtype=torch.bfloat16),
        dt_bias=torch.zeros(nheads),
        seq_idx=seq_idx,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        cu_seqlens=cu_seqlens,
        dt_softplus=True,
        out=out,
        return_varlen_states=True,
        return_intermediate_states=True,
        state_dtype=torch.bfloat16,
    )

    padded_x, padded_dt, _, padded_B, padded_C, kwargs = runner.call
    assert padded_x.shape[1] == padded_dt.shape[1] == 256
    assert padded_B.shape[1] == padded_C.shape[1] == 256
    assert torch.count_nonzero(padded_x[:, seqlen:]) == 0
    assert torch.count_nonzero(padded_B[:, seqlen:]) == 0
    assert torch.count_nonzero(padded_C[:, seqlen:]) == 0
    assert torch.isneginf(padded_dt[:, seqlen:]).all()
    assert padded_dt.dtype == dt.dtype
    assert torch.equal(
        kwargs["seq_idx"][:, seqlen:],
        torch.ones((1, 256 - seqlen), dtype=torch.int32),
    )
    assert kwargs["initial_states"].shape == (2, nheads, headdim, dstate)
    assert torch.count_nonzero(kwargs["initial_states"]) == 0
    assert intermediate is None
    assert final.shape == (2, nheads, headdim, dstate)
    expected = torch.arange(padded_x.numel(), dtype=torch.float32).reshape(
        padded_x.shape
    )[:, :seqlen].to(out.dtype)
    assert torch.equal(out, expected)


def test_flashinfer_prefill_reuses_read_only_zero_initial_states():
    backend = _flashinfer_backend_without_imports()
    x = torch.empty(1, 128, 2, 4, dtype=torch.bfloat16)
    B = torch.empty(1, 128, 1, 3, dtype=torch.bfloat16)

    first = backend._get_zero_initial_states(
        x=x,
        B=B,
        num_sequences=2,
        state_dtype=torch.bfloat16,
    )
    second = backend._get_zero_initial_states(
        x=x,
        B=B,
        num_sequences=2,
        state_dtype=torch.bfloat16,
    )

    assert first is second
    assert first.shape == (2, 2, 4, 3)
    assert torch.count_nonzero(first) == 0


def test_flashinfer_prefill_refuses_non_sglang_return_contract():
    backend = _flashinfer_backend_without_imports()
    with pytest.raises(ValueError, match="only supports SGLang"):
        backend.chunk_scan_combined(
            torch.empty(1, 128, 1, 1),
            torch.empty(1, 128, 1),
            torch.empty(1),
            torch.empty(1, 128, 1, 1),
            torch.empty(1, 128, 1, 1),
            128,
        )


def test_flashinfer_prefill_refuses_non_identity_tail_limit():
    backend = _flashinfer_backend_without_imports()
    with pytest.raises(ValueError, match=r"dt_limit\[0\]=0"):
        backend.chunk_scan_combined(
            torch.empty(1, 128, 1, 1),
            torch.empty(1, 128, 1),
            torch.empty(1),
            torch.empty(1, 128, 1, 1),
            torch.empty(1, 128, 1, 1),
            128,
            cu_seqlens=torch.tensor([0, 128], dtype=torch.int32),
            seq_idx=torch.zeros(1, 128, dtype=torch.int32),
            chunk_indices=torch.zeros(1, dtype=torch.int32),
            chunk_offsets=torch.zeros(1, dtype=torch.int32),
            return_varlen_states=True,
            return_intermediate_states=True,
            dt_limit=(0.001, float("inf")),
        )


def test_flashinfer_prefill_requires_caller_owned_output():
    backend = _flashinfer_backend_without_imports()
    with pytest.raises(ValueError, match="caller-owned"):
        backend.chunk_scan_combined(
            torch.empty(1, 128, 1, 1),
            torch.empty(1, 128, 1),
            torch.empty(1),
            torch.empty(1, 128, 1, 1),
            torch.empty(1, 128, 1, 1),
            128,
            cu_seqlens=torch.tensor([0, 128], dtype=torch.int32),
            seq_idx=torch.zeros(1, 128, dtype=torch.int32),
            chunk_indices=torch.zeros(1, dtype=torch.int32),
            chunk_offsets=torch.zeros(1, dtype=torch.int32),
            return_varlen_states=True,
            return_intermediate_states=True,
        )


def test_cake_backend_is_registered_without_fallback():
    assert ssu_dispatch._BACKEND_REGISTRY["cake"] is ssu_dispatch.CakeSSUBackend
    assert (
        ssu_dispatch._BACKEND_REGISTRY["flashinfer_ssd"]
        is ssu_dispatch.FlashInferSSDCombinedSSUBackend
    )
    assert "cake" in ssu_dispatch._BACKEND_REGISTRY


def test_only_flashinfer_family_requires_unconditional_chunk_metadata(monkeypatch):
    for backend_cls, expected in (
        (ssu_dispatch.TritonSSUBackend, False),
        (ssu_dispatch.FlashInferSSUBackend, False),
        (ssu_dispatch.FlashInferSSDCombinedSSUBackend, True),
        (ssu_dispatch.CakeSSUBackend, True),
    ):
        monkeypatch.setattr(
            ssu_dispatch, "_mamba_ssu_backend", object.__new__(backend_cls)
        )
        assert ssu_dispatch.mamba_prefill_requires_chunk_metadata() is expected


def test_initialize_unknown_backend_is_fail_closed(monkeypatch):
    monkeypatch.setattr(ssu_dispatch, "_mamba_ssu_backend", None)
    args = SimpleNamespace(
        mamba_backend="not-a-backend",
        enable_mamba_cache_stochastic_rounding=False,
        mamba_cache_philox_rounds=0,
    )
    with pytest.raises(ValueError, match="Unknown mamba backend"):
        ssu_dispatch.initialize_mamba_selective_state_update_backend(args)
    assert ssu_dispatch._mamba_ssu_backend is None
