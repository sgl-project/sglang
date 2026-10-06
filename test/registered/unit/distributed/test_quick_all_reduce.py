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
    monkeypatch.setenv("ROCM_QUICK_REDUCE_QUANTIZATION", "INT8")
    monkeypatch.setenv("ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16", "0")
    monkeypatch.setenv("ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB", "128")
    monkeypatch.setenv("AITER_QUICK_REDUCE_MAX_SIZE_BYTES_MB", "256")

    quick_all_reduce_module._configure_aiter_quickreduce_env()

    assert (
        quick_all_reduce_module.os.environ["AITER_QUICK_REDUCE_QUANTIZATION"] == "FP8"
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
    fake_module = ModuleType("aiter.dist.device_communicators.quick_all_reduce")
    fake_module.QuickAllReduce = _OutOfTreeQuickAllReduce
    monkeypatch.setitem(
        sys.modules, "aiter.dist.device_communicators.quick_all_reduce", fake_module
    )
    monkeypatch.setattr(quick_all_reduce_module, "qr_rocm_arch_available", lambda: True)
    monkeypatch.setenv("SGLANG_USE_AITER", "1")

    adapter = quick_all_reduce_module.create_quick_allreduce(
        group=object(), device="cpu"
    )

    assert isinstance(adapter, quick_all_reduce_module._QuickAllReduceAdapter)
    assert isinstance(adapter._communicator, _OutOfTreeQuickAllReduce)


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


def test_factory_propagates_aiter_constructor_failure(monkeypatch):
    class FailingQuickAllReduce:
        def __init__(self, **kwargs):
            raise RuntimeError("constructor failed")

    fake_module = ModuleType("aiter.dist.device_communicators.quick_all_reduce")
    fake_module.QuickAllReduce = FailingQuickAllReduce
    monkeypatch.setitem(
        sys.modules, "aiter.dist.device_communicators.quick_all_reduce", fake_module
    )
    monkeypatch.setattr(quick_all_reduce_module, "qr_rocm_arch_available", lambda: True)
    monkeypatch.setenv("SGLANG_USE_AITER", "1")

    with pytest.raises(RuntimeError, match="constructor failed"):
        quick_all_reduce_module.create_quick_allreduce(group=object(), device="cpu")
