# SPDX-License-Identifier: Apache-2.0
"""Dispatch-level tests for the YuE2 platform/capability gates.

These prove which execution branch each platform selects by simulating CUDA,
ROCm, and missing/failing backends. They are dispatch proofs, not GPU
execution proofs; on-device validation (real ``sglang serve`` + audio
artifacts) is tracked separately in AIwork4me/YuE2-SGLang-ROCm.

Gate under test (all added by the ROCm compatibility patch):

* ``fastpath.deepselect_runnable`` — the loaded DeepSelect backend must be
  *runnable*, not merely importable (``is_deepselect_supported``).
* ``song._generate_song_tokens_step_graph`` — raises ``_StepGraphUnavailable``
  when the fused decode+sample graph's kernel cannot run.
* ``fa3._load_varlen`` — on HIP the shared FA op is skipped in favour of the
  native ATen varlen path (the proven Radeon route).
* ``batched.batched_ar_enabled`` — never constructs CUDA-only
  ``BatchedGraphSampler`` sessions on unsupported devices.

NVIDIA defaults must be preserved everywhere: with a runnable backend and no
environment overrides, the CUDA dispatch is bit-identical to upstream.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

from sglang.srt.models.yue2 import batched
from sglang.srt.models.yue2 import fa3 as fa3_module
from sglang.srt.models.yue2 import fastpath, song

SHARED_FA_MODULE = "sglang.kernels.ops.attention.flash_attention"


class FakeBackend:
    """Stands in for the in-tree sglang.kernels.ops.deep_select module."""

    def __init__(self, supported: bool | None, exc: Exception | None = None):
        self._supported = supported
        self._exc = exc
        self.probe_calls = 0

    def topk(self, *args, **kwargs):  # pragma: no cover - never called
        raise AssertionError("dispatch tests must not execute the kernel")

    def is_deepselect_supported(self, device=None) -> bool:
        self.probe_calls += 1
        if self._exc is not None:
            raise self._exc
        assert self._supported is not None
        return self._supported


@pytest.fixture()
def backend_caches():
    def reset(backend=False):
        fastpath._IN_TREE = backend
        fastpath._PIP = False

    reset()
    fa3_module._VARLEN = None
    fa3_module._LOADED = False
    yield reset
    reset()
    fa3_module._VARLEN = None
    fa3_module._LOADED = False


def inject_shared_varlen(monkeypatch, shared_op):
    module = types.ModuleType(SHARED_FA_MODULE)
    module.flash_attn_varlen_func = shared_op
    monkeypatch.setitem(sys.modules, SHARED_FA_MODULE, module)


# ---------------------------------------------------------------------------
# fastpath: capability-based fast-sampler selection
# ---------------------------------------------------------------------------


def test_deepselect_supported_preserves_fast_sampler(backend_caches):
    backend_caches(FakeBackend(supported=True))
    assert fastpath.fast_sampler_available() is True
    assert fastpath.deepselect_runnable() is True


def test_deepselect_unsupported_uses_reference_sampler(backend_caches):
    backend_caches(FakeBackend(supported=False))
    assert fastpath.deepselect_runnable() is False
    assert fastpath.fast_sampler_available() is False


def test_capability_probe_error_fails_safe(backend_caches):
    backend_caches(FakeBackend(supported=None, exc=RuntimeError("probe boom")))
    assert fastpath.deepselect_runnable() is False
    assert fastpath.fast_sampler_available() is False


def test_missing_backend_disables_fast_sampler(backend_caches):
    backend_caches()
    assert fastpath.deepselect_runnable() is False
    assert fastpath.fast_sampler_available() is False


def test_pip_backend_without_probe_keeps_original_check(backend_caches, monkeypatch):
    pip_backend = types.SimpleNamespace()  # no is_deepselect_supported attr
    fastpath._IN_TREE = False
    fastpath._PIP = pip_backend
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert fastpath.deepselect_runnable() is True
    assert fastpath.fast_sampler_available() is True


def test_fast_sampler_env_override_still_wins(backend_caches, monkeypatch):
    backend_caches(FakeBackend(supported=True))
    monkeypatch.setenv("SGLANG_YUE2_FAST_SAMPLER", "0")
    assert fastpath.fast_sampler_available() is False
    # the raw capability probe is independent of the env knob
    assert fastpath.deepselect_runnable() is True


# ---------------------------------------------------------------------------
# song: fused decode+sample step graph entry
# ---------------------------------------------------------------------------


def make_song_request(**overrides):
    from types import SimpleNamespace

    defaults = dict(cot="full", seed=7, abc=None)
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_step_graph_blocked_when_kernel_unsupported(backend_caches):
    backend_caches(FakeBackend(supported=False))
    session = types.SimpleNamespace(
        device=torch.device("cuda"), graph=None, step_mode=False
    )
    with pytest.raises(song._StepGraphUnavailable, match="unsupported"):
        song._generate_song_tokens_step_graph(
            None,
            session,
            None,
            make_song_request(),
            abc_sampling=None,
            semantic_sampling=None,
        )


def test_step_graph_blocked_when_backend_missing(backend_caches):
    # With no importable backend, ``_load_topk_backend()`` returns False and
    # the capability probe routes the request to the looped path.
    backend_caches()
    session = types.SimpleNamespace(
        device=torch.device("cuda"), graph=None, step_mode=False
    )
    with pytest.raises(song._StepGraphUnavailable):
        song._generate_song_tokens_step_graph(
            None,
            session,
            None,
            make_song_request(),
            abc_sampling=None,
            semantic_sampling=None,
        )


class StubGraphSampler:
    """Records construction; stands in for the real graph-resident sampler."""

    constructed = 0

    def __init__(self, vocab_size, device, generator):
        StubGraphSampler.constructed += 1


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="constructs a device Generator like upstream"
)
def test_step_graph_capability_gate_passes_on_supported_device(
    backend_caches, monkeypatch
):
    backend_caches(FakeBackend(supported=True))
    StubGraphSampler.constructed = 0
    monkeypatch.setattr(song, "GraphSampler", StubGraphSampler)
    fake_model = types.SimpleNamespace(config=types.SimpleNamespace(vocab_size=16))
    # A pooled session already captured without a sampler trips the next
    # upstream guard, proving execution got past the capability gate.
    session = types.SimpleNamespace(
        device=torch.device("cuda"), graph=object(), step_mode=False
    )
    with pytest.raises(song._StepGraphUnavailable, match="already captured"):
        song._generate_song_tokens_step_graph(
            fake_model,
            session,
            None,
            make_song_request(),
            abc_sampling=None,
            semantic_sampling=None,
        )
    assert StubGraphSampler.constructed == 1


def test_generate_song_tokens_falls_back_to_looped_when_unsupported(
    backend_caches, monkeypatch
):
    backend_caches(FakeBackend(supported=False))
    StubGraphSampler.constructed = 0
    monkeypatch.setattr(song, "GraphSampler", StubGraphSampler)
    monkeypatch.delenv("SGLANG_YUE2_STEP_GRAPH", raising=False)
    monkeypatch.setattr(
        song, "_generate_song_tokens_looped", lambda *a, **k: ("abc", "codec", {}, 0)
    )
    result = song.generate_song_tokens(
        model=None,
        session=types.SimpleNamespace(device=torch.device("cuda")),
        tokenizer=None,
        request=make_song_request(),
        abc_sampling=None,
        semantic_sampling=None,
    )
    assert result == ("abc", "codec", {}, 0)
    assert StubGraphSampler.constructed == 0  # gate tripped pre-construction


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="constructs a device Generator like upstream"
)
def test_generate_song_tokens_enters_step_graph_when_supported(
    backend_caches, monkeypatch
):
    backend_caches(FakeBackend(supported=True))
    StubGraphSampler.constructed = 0
    monkeypatch.setattr(song, "GraphSampler", StubGraphSampler)
    monkeypatch.delenv("SGLANG_YUE2_STEP_GRAPH", raising=False)
    monkeypatch.setattr(
        song, "_generate_song_tokens_looped", lambda *a, **k: ("abc", "codec", {}, 0)
    )
    fake_model = types.SimpleNamespace(config=types.SimpleNamespace(vocab_size=16))
    # The stub sampler + already-captured session end the step-graph attempt
    # with the upstream "already captured" fallback; construction count proves
    # the fused path was entered on a supported device.
    result = song.generate_song_tokens(
        model=fake_model,
        session=types.SimpleNamespace(
            device=torch.device("cuda"), graph=object(), step_mode=False
        ),
        tokenizer=None,
        request=make_song_request(),
        abc_sampling=None,
        semantic_sampling=None,
    )
    assert StubGraphSampler.constructed == 1
    assert result == ("abc", "codec", {}, 0)


def test_step_graph_env_override_still_forces_looped_on_cuda(
    backend_caches, monkeypatch
):
    backend_caches(FakeBackend(supported=True))
    monkeypatch.setenv("SGLANG_YUE2_STEP_GRAPH", "0")
    monkeypatch.setattr(
        song, "_generate_song_tokens_step_graph", lambda *a, **k: "step-graph"
    )  # pragma: no cover - must not run
    monkeypatch.setattr(song, "_generate_song_tokens_looped", lambda *a, **k: "looped")
    result = song.generate_song_tokens(
        model=None,
        session=types.SimpleNamespace(device=torch.device("cuda")),
        tokenizer=None,
        request=make_song_request(),
        abc_sampling=None,
        semantic_sampling=None,
    )
    assert result == "looped"


# ---------------------------------------------------------------------------
# fa3: HIP selects the native ATen varlen path; CUDA dispatch unchanged
# ---------------------------------------------------------------------------


def aten_varlen_spy(monkeypatch, out=None):
    calls = []

    def fake_aten_forward(*args, **kwargs):
        calls.append((args, kwargs))
        if out is not None:
            return (out,)  # aten returns a tuple; [0] is the output tensor
        return (torch.zeros(1),)

    monkeypatch.setattr(torch.ops.aten, "_flash_attention_forward", fake_aten_forward)
    return calls


def varlen_kwargs():
    q = k = v = torch.zeros(2, 1, 8)
    cu = torch.tensor([0, 2], dtype=torch.int32)
    seqused_k = torch.tensor([1, 2], dtype=torch.int32)
    return q, k, v, cu, cu, 2, 2, seqused_k


def test_hip_skips_shared_fa_op_and_uses_native_aten(backend_caches, monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.14")  # simulated ROCm
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setenv("SGLANG_YUE2_FA3", "1")
    inject_shared_varlen(
        monkeypatch,
        shared_op=lambda *a, **k: pytest.fail(
            "shared FA op must not be called on ROCm"
        ),
    )
    assert fa3_module.fa3_available() is False

    calls = aten_varlen_spy(monkeypatch, out=torch.full((2, 1, 8), 5.0))
    out = fa3_module.varlen_attention(*varlen_kwargs())
    assert torch.equal(out, torch.full((2, 1, 8), 5.0))
    assert len(calls) == 1
    assert calls[0][1]["seqused_k"] is not None  # seqused_k honoured


def test_cuda_shared_fa_op_dispatch_unchanged(backend_caches, monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)  # simulated CUDA
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setenv("SGLANG_YUE2_FA3", "1")
    calls = []

    def shared_op(*args, **kwargs):
        calls.append((args, kwargs))
        return "shared-op-output"

    inject_shared_varlen(monkeypatch, shared_op=shared_op)
    assert fa3_module.fa3_available() is True
    q, k, v, cu_q, cu_k, max_q, max_k, seqused_k = varlen_kwargs()
    out = fa3_module.varlen_attention(
        q, k, v, cu_q, cu_k, max_q, max_k, seqused_k=seqused_k
    )
    assert out == "shared-op-output"
    assert len(calls) == 1
    assert calls[0][1]["seqused_k"] is seqused_k
    assert calls[0][1]["causal"] is False
    assert calls[0][1]["ver"] == 3


def test_missing_shared_fa_op_falls_back_to_aten(backend_caches, monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setenv("SGLANG_YUE2_FA3", "1")
    monkeypatch.setitem(sys.modules, SHARED_FA_MODULE, None)  # ImportError
    assert fa3_module.fa3_available() is False
    calls = aten_varlen_spy(monkeypatch)
    fa3_module.varlen_attention(*varlen_kwargs())
    assert len(calls) == 1


def test_fa3_env_override_still_forces_aten(backend_caches, monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setenv("SGLANG_YUE2_FA3", "0")
    inject_shared_varlen(
        monkeypatch, shared_op=lambda *a, **k: pytest.fail("env override must win")
    )
    assert fa3_module.fa3_available() is False
    calls = aten_varlen_spy(monkeypatch)
    fa3_module.varlen_attention(*varlen_kwargs())
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# batched: never construct CUDA-only BatchedGraphSampler sessions
# ---------------------------------------------------------------------------


def test_batched_ar_disabled_when_kernel_unsupported(backend_caches, monkeypatch):
    backend_caches(FakeBackend(supported=False))
    monkeypatch.delenv("SGLANG_YUE2_BATCHED_AR", raising=False)
    assert batched.batched_ar_enabled() is False


def test_batched_ar_enabled_when_kernel_supported(backend_caches, monkeypatch):
    backend_caches(FakeBackend(supported=True))
    monkeypatch.delenv("SGLANG_YUE2_BATCHED_AR", raising=False)
    assert batched.batched_ar_enabled() is True


def test_batched_ar_env_override_still_wins(backend_caches, monkeypatch):
    backend_caches(FakeBackend(supported=True))
    monkeypatch.setenv("SGLANG_YUE2_BATCHED_AR", "0")
    assert batched.batched_ar_enabled() is False


def test_batched_ar_survives_fast_sampler_env_disable_on_cuda(
    backend_caches, monkeypatch
):
    # SGLANG_YUE2_FAST_SAMPLER=0 must not disable the batched path on devices
    # where the kernel itself is runnable (preserves upstream semantics).
    backend_caches(FakeBackend(supported=True))
    monkeypatch.setenv("SGLANG_YUE2_FAST_SAMPLER", "0")
    monkeypatch.delenv("SGLANG_YUE2_BATCHED_AR", raising=False)
    assert fastpath.fast_sampler_available() is False
    assert batched.batched_ar_enabled() is True


# ---------------------------------------------------------------------------
# Real in-tree backend wiring (runs wherever the suite's torch lives; on the
# W7900 gfx1100 validation host this is the real ROCm probe)
# ---------------------------------------------------------------------------


def test_real_in_tree_backend_wiring(backend_caches):
    deep_select = pytest.importorskip("sglang.kernels.ops.deep_select")
    backend_caches(deep_select)
    assert fastpath.deepselect_runnable() == deep_select.is_deepselect_supported()
