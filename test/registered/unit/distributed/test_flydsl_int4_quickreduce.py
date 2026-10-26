"""INT4 QuickReduce keeps HIP and adds FlyDSL for eligible bf16 payloads.

ROCM_QUICK_REDUCE_QUANTIZATION=INT4 always creates the HIP workspace. On
gfx950 with SGLANG_USE_AITER and an importable QuickAllReduceInt4, FlyDSL is
created beside it. A contiguous, 16-byte-aligned bf16 payload uses FlyDSL.
fp16, non-contiguous, and misaligned payloads use HIP. No model is launched.
"""

import sys
import types

import torch

import sglang.srt.distributed.device_communicators.quick_all_reduce as qr_mod
from sglang.srt.distributed.device_communicators.quick_all_reduce import (
    QuickAllReduce,
    QuickReduceRegime,
    _flydsl_int4_available,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

MB = 1024 * 1024


class _FakeFlyDSLInt4:
    instances = []
    fail_compile = False

    def __init__(self, group=None, device=None, rank=0, world_size=4):
        self.group = group
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.allreduce_calls = 0
        self.compile_calls = 0
        self.closed = False
        self.instances.append(self)

    def compile(self, inp, out):
        self.compile_calls += 1
        if self.fail_compile:
            raise RuntimeError("compile failed")

    def allreduce(self, inp, out):
        self.allreduce_calls += 1

    def close(self):
        self.closed = True


def _install_fake_aiter(monkeypatch, cls) -> None:
    """Make `from aiter.ops.flydsl import QuickAllReduceInt4` return `cls`."""
    aiter = types.ModuleType("aiter")
    aiter.__path__ = []
    ops_mod = types.ModuleType("aiter.ops")
    ops_mod.__path__ = []
    flydsl = types.ModuleType("aiter.ops.flydsl")
    flydsl.QuickAllReduceInt4 = cls
    aiter.ops = ops_mod
    ops_mod.flydsl = flydsl
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    monkeypatch.setitem(sys.modules, "aiter.ops", ops_mod)
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl", flydsl)


def _prepare_init(monkeypatch, level: str):
    """Stub the HIP library so init can run without a GPU."""
    monkeypatch.setenv("ROCM_QUICK_REDUCE_QUANTIZATION", level)
    monkeypatch.delenv("ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB", raising=False)
    monkeypatch.setattr(qr_mod.ops, "qr_max_size", lambda: 2 * 1024 * MB)
    monkeypatch.setattr(qr_mod.torch.cuda, "synchronize", lambda *args, **kwargs: None)
    hip_inits = []
    monkeypatch.setattr(
        qr_mod.ops, "init_custom_qr", lambda *args: hip_inits.append(args) or 11
    )
    monkeypatch.setattr(QuickAllReduce, "create_shared_buffer", lambda self: None)
    return hip_inits


def _make(level: QuickReduceRegime = QuickReduceRegime.INT4) -> QuickAllReduce:
    quick_reduce = QuickAllReduce.__new__(QuickAllReduce)
    # disabled, with no HIP handle: __del__ -> close() must not call qr_destroy.
    quick_reduce.disabled = True
    quick_reduce._ptr = 0
    quick_reduce._flydsl_int4 = None
    quick_reduce.qr_quant_level = level
    quick_reduce.use_fp16_kernels = 1
    quick_reduce.qr_max_size = 2 * 1024 * MB
    quick_reduce.world_size = 4
    quick_reduce.rank = 0
    quick_reduce.group = None
    quick_reduce.device = torch.device("cpu")
    return quick_reduce


def _release(quick_reduce: QuickAllReduce) -> None:
    flydsl = quick_reduce._flydsl_int4
    if flydsl is not None:
        flydsl.close()
        quick_reduce._flydsl_int4 = None
    quick_reduce._ptr = 0
    quick_reduce.disabled = True


def test_flydsl_requires_aiter_gfx950_and_the_symbol(monkeypatch):
    monkeypatch.delenv("SGLANG_USE_AITER", raising=False)
    assert _flydsl_int4_available() is False

    monkeypatch.setenv("SGLANG_USE_AITER", "1")
    monkeypatch.setattr("sglang.srt.utils.is_gfx95_supported", lambda: False)
    assert _flydsl_int4_available() is False

    monkeypatch.setattr("sglang.srt.utils.is_gfx95_supported", lambda: True)
    real_import = __import__

    def fail_flydsl(name, *args, **kwargs):
        if name == "aiter.ops.flydsl":
            raise ImportError("no QuickAllReduceInt4")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fail_flydsl)
    assert _flydsl_int4_available() is False


def test_int4_creates_hip_and_flydsl(monkeypatch):
    _FakeFlyDSLInt4.instances = []
    _FakeFlyDSLInt4.fail_compile = False
    _install_fake_aiter(monkeypatch, _FakeFlyDSLInt4)
    quick_reduce = _make()
    hip_inits = _prepare_init(monkeypatch, "INT4")
    monkeypatch.setattr(qr_mod, "_flydsl_int4_available", lambda: True)
    hip_calls = []
    monkeypatch.setattr(qr_mod.ops, "qr_all_reduce", lambda *args: hip_calls.append(args))

    quick_reduce.init_quick_all_reduce()
    bf16 = torch.empty(256, dtype=torch.bfloat16)
    fp16 = torch.empty(256, dtype=torch.float16)
    assert bf16.data_ptr() % 16 == 0
    quick_reduce.quick_all_reduce(bf16)
    quick_reduce.quick_all_reduce(fp16)
    strided = torch.empty(512, dtype=torch.bfloat16)[::2]
    quick_reduce.quick_all_reduce(strided)

    flydsl = _FakeFlyDSLInt4.instances[0]
    assert len(hip_inits) == 1
    assert quick_reduce._ptr == 11
    assert flydsl.compile_calls == 1
    assert flydsl.allreduce_calls == 1
    assert len(hip_calls) == 2
    assert hip_calls[0][1].dtype == torch.float16
    assert hip_calls[1][1].dtype == torch.bfloat16
    assert not hip_calls[1][1].is_contiguous()
    assert all(call[3] == QuickReduceRegime.INT4.value for call in hip_calls)
    _release(quick_reduce)
    assert flydsl.closed
    assert quick_reduce._flydsl_int4 is None


def test_misaligned_bf16_stays_on_hip(monkeypatch):
    quick_reduce = _make()
    fake = _FakeFlyDSLInt4()
    quick_reduce._flydsl_int4 = fake
    quick_reduce.disabled = False
    quick_reduce._ptr = 11
    hip_calls = []
    monkeypatch.setattr(qr_mod.ops, "qr_all_reduce", lambda *args: hip_calls.append(args))
    payload = torch.empty(256, dtype=torch.bfloat16)
    monkeypatch.setattr(payload, "data_ptr", lambda: 8)

    quick_reduce.quick_all_reduce(payload)

    assert fake.allreduce_calls == 0
    assert len(hip_calls) == 1
    _release(quick_reduce)


def test_int4_without_flydsl_creates_only_hip(monkeypatch):
    _FakeFlyDSLInt4.instances = []
    quick_reduce = _make()
    hip_inits = _prepare_init(monkeypatch, "INT4")
    monkeypatch.setattr(qr_mod, "_flydsl_int4_available", lambda: False)
    hip_calls = []
    monkeypatch.setattr(qr_mod.ops, "qr_all_reduce", lambda *args: hip_calls.append(args))

    quick_reduce.init_quick_all_reduce()
    payload = torch.empty(256, dtype=torch.bfloat16)
    quick_reduce.quick_all_reduce(payload)

    assert _FakeFlyDSLInt4.instances == []
    assert quick_reduce._flydsl_int4 is None
    assert len(hip_inits) == 1
    assert hip_calls[0][3] == QuickReduceRegime.INT4.value
    _release(quick_reduce)


def test_other_codecs_do_not_open_flydsl(monkeypatch):
    quick_reduce = _make(QuickReduceRegime.NONE)
    hip_inits = _prepare_init(monkeypatch, "INT8")
    monkeypatch.setattr(
        qr_mod,
        "_flydsl_int4_available",
        lambda: (_ for _ in ()).throw(AssertionError("FlyDSL availability checked")),
    )
    hip_calls = []
    monkeypatch.setattr(qr_mod.ops, "qr_all_reduce", lambda *args: hip_calls.append(args))

    quick_reduce.init_quick_all_reduce()
    quick_reduce.quick_all_reduce(torch.empty(256, dtype=torch.float16))

    assert quick_reduce._flydsl_int4 is None
    assert len(hip_inits) == 1
    assert hip_calls[0][3] == QuickReduceRegime.INT8.value
    _release(quick_reduce)


def test_flydsl_compile_failure_propagates(monkeypatch):
    _FakeFlyDSLInt4.instances = []
    _FakeFlyDSLInt4.fail_compile = True
    _install_fake_aiter(monkeypatch, _FakeFlyDSLInt4)
    quick_reduce = _make()
    _prepare_init(monkeypatch, "INT4")
    monkeypatch.setattr(qr_mod, "_flydsl_int4_available", lambda: True)

    try:
        try:
            quick_reduce.init_quick_all_reduce()
            raise AssertionError("compile failure was swallowed")
        except RuntimeError as exc:
            assert "compile failed" in str(exc)
        assert quick_reduce._ptr == 11
        assert quick_reduce.disabled
    finally:
        _FakeFlyDSLInt4.fail_compile = False
        _release(quick_reduce)
