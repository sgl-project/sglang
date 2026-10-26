# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType

import pytest
import torch

from sglang.srt.distributed.device_communicators import (
    quick_all_reduce as quick_all_reduce_module,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def _clear_quick_reduce_env(monkeypatch):
    # The ROCm image sets ROCM_QUICK_REDUCE_QUANTIZATION.
    for names in quick_all_reduce_module._QUICK_REDUCE_ENV_ALIASES.items():
        for name in names:
            monkeypatch.delenv(name, raising=False)


class _BundledQuickAllReduce:
    def __init__(self, **kwargs) -> None:
        pass


def _install_fake_backends(monkeypatch, aiter_cls, bundled_cls=_BundledQuickAllReduce):
    fake_module = ModuleType("aiter.dist.device_communicators.quick_all_reduce")
    fake_module.QuickAllReduce = aiter_cls
    monkeypatch.setitem(
        sys.modules, "aiter.dist.device_communicators.quick_all_reduce", fake_module
    )
    monkeypatch.setattr(quick_all_reduce_module, "QuickAllReduce", bundled_cls)
    monkeypatch.setattr(quick_all_reduce_module, "qr_rocm_arch_available", lambda: True)
    monkeypatch.setenv("SGLANG_USE_AITER", "1")


class _OutOfTreeQuickAllReduce:
    def __init__(self, **kwargs) -> None:
        self.closed = False

    def should_quick_allreduce(self, inp: torch.Tensor) -> bool:
        return inp.numel() == 4

    def quick_all_reduce(
        self, inp: torch.Tensor, *, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        result = inp + 1
        if out is not None:
            out.copy_(result)
            return out
        return result

    def close(self) -> None:
        self.closed = True


def test_adapter_does_not_require_out_of_tree_disabled_attribute():
    backend = _OutOfTreeQuickAllReduce()
    adapter = quick_all_reduce_module._QuickAllReduceAdapter(backend)
    inp = torch.zeros(4)

    assert not adapter.disabled
    assert adapter.should_quick_allreduce(inp)
    torch.testing.assert_close(adapter.quick_all_reduce(inp), torch.ones(4))

    out = torch.empty_like(inp)
    assert adapter.quick_all_reduce(inp, out=out) is out
    torch.testing.assert_close(out, torch.ones(4))

    adapter.close()
    assert backend.closed


def test_adapter_treats_missing_eligibility_method_as_ineligible():
    adapter = quick_all_reduce_module._QuickAllReduceAdapter(object())

    assert not adapter.disabled
    assert not adapter.should_quick_allreduce(torch.zeros(4))


def test_legacy_environment_aliases_do_not_override_aiter(monkeypatch):
    monkeypatch.setenv("ROCM_QUICK_REDUCE_QUANTIZATION", "INT4")
    monkeypatch.setenv("ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16", "0")
    monkeypatch.setenv("ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB", "128")
    monkeypatch.setenv("AITER_QUICK_REDUCE_MAX_SIZE_BYTES_MB", "256")

    quick_all_reduce_module._configure_aiter_quickreduce_env()

    assert (
        quick_all_reduce_module.os.environ["AITER_QUICK_REDUCE_QUANTIZATION"] == "INT4"
    )
    assert (
        quick_all_reduce_module.os.environ["AITER_QUICK_REDUCE_CAST_BF16_TO_FP16"]
        == "0"
    )
    assert (
        quick_all_reduce_module.os.environ["AITER_QUICK_REDUCE_MAX_SIZE_BYTES_MB"]
        == "256"
    )


def test_factory_selects_aiter_without_requiring_disabled(monkeypatch):
    _install_fake_backends(monkeypatch, _OutOfTreeQuickAllReduce)

    adapter = quick_all_reduce_module.create_quick_allreduce(
        group=object(), device="cpu"
    )

    assert isinstance(adapter, quick_all_reduce_module._QuickAllReduceAdapter)
    assert isinstance(adapter._communicator, _OutOfTreeQuickAllReduce)


def test_legacy_int8_keeps_bundled_codec(monkeypatch):
    _install_fake_backends(monkeypatch, _OutOfTreeQuickAllReduce)
    monkeypatch.setenv("ROCM_QUICK_REDUCE_QUANTIZATION", "INT8")

    adapter = quick_all_reduce_module.create_quick_allreduce(
        group=object(), device="cpu"
    )

    assert isinstance(adapter._communicator, _BundledQuickAllReduce)
    assert "AITER_QUICK_REDUCE_QUANTIZATION" not in quick_all_reduce_module.os.environ


@pytest.mark.parametrize(
    "codec_env", ["AITER_QUICK_REDUCE_QUANTIZATION", "ROCM_QUICK_REDUCE_QUANTIZATION"]
)
def test_fp8_request_uses_bundled_int8_codec(monkeypatch, codec_env):
    _install_fake_backends(monkeypatch, _OutOfTreeQuickAllReduce)
    monkeypatch.setenv(codec_env, "FP8")
    monkeypatch.setenv("AITER_QUICK_REDUCE_MAX_SIZE_BYTES_MB", "256")

    adapter = quick_all_reduce_module.create_quick_allreduce(
        group=object(), device="cpu"
    )

    environ = quick_all_reduce_module.os.environ
    assert isinstance(adapter._communicator, _BundledQuickAllReduce)
    assert environ["ROCM_QUICK_REDUCE_QUANTIZATION"] == "INT8"
    assert environ["ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB"] == "256"


def test_int4_request_keeps_bundled_codec(monkeypatch):
    _install_fake_backends(monkeypatch, _OutOfTreeQuickAllReduce)
    monkeypatch.setenv("ROCM_QUICK_REDUCE_QUANTIZATION", "INT8")
    monkeypatch.setenv("AITER_QUICK_REDUCE_QUANTIZATION", "INT4")

    adapter = quick_all_reduce_module.create_quick_allreduce(
        group=object(), device="cpu"
    )

    assert isinstance(adapter._communicator, _BundledQuickAllReduce)
    assert (
        quick_all_reduce_module.os.environ["ROCM_QUICK_REDUCE_QUANTIZATION"] == "INT4"
    )


def test_factory_falls_back_when_aiter_import_is_unavailable(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "aiter.dist.device_communicators.quick_all_reduce", None
    )
    monkeypatch.setattr(quick_all_reduce_module, "qr_rocm_arch_available", lambda: True)
    monkeypatch.setattr(
        quick_all_reduce_module, "QuickAllReduce", _OutOfTreeQuickAllReduce
    )
    monkeypatch.setenv("SGLANG_USE_AITER", "1")

    adapter = quick_all_reduce_module.create_quick_allreduce(
        group=object(), device="cpu"
    )

    assert isinstance(adapter, quick_all_reduce_module._QuickAllReduceAdapter)
    assert isinstance(adapter._communicator, _OutOfTreeQuickAllReduce)


@pytest.mark.parametrize("codec", ["INT8", "FP8", "INT4"])
@pytest.mark.parametrize("use_aiter", ["0", "1"])
def test_unsupported_arch_creates_no_communicator(monkeypatch, codec, use_aiter):
    class MustNotConstruct:
        def __init__(self, **kwargs):
            raise AssertionError("QuickReduce constructed on an unsupported arch")

    _install_fake_backends(monkeypatch, MustNotConstruct, MustNotConstruct)
    monkeypatch.setattr(
        quick_all_reduce_module, "qr_rocm_arch_available", lambda: False
    )
    monkeypatch.setenv("SGLANG_USE_AITER", use_aiter)
    monkeypatch.setenv("ROCM_QUICK_REDUCE_QUANTIZATION", codec)

    assert (
        quick_all_reduce_module.create_quick_allreduce(group=object(), device="cpu")
        is None
    )


def test_factory_propagates_aiter_constructor_failure(monkeypatch):
    class FailingQuickAllReduce:
        def __init__(self, **kwargs):
            raise RuntimeError("constructor failed")

    _install_fake_backends(monkeypatch, FailingQuickAllReduce)

    with pytest.raises(RuntimeError, match="constructor failed"):
        quick_all_reduce_module.create_quick_allreduce(group=object(), device="cpu")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
