import sys
from contextlib import nullcontext

import pytest
import torch

from sglang.srt.batch_invariant_ops import batch_invariant_ops as ops
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.fixture
def native_bmm(monkeypatch):
    was_enabled = ops.is_batch_invariant_mode_enabled()
    had_bmm = ops._original_torch_bmm is not None
    ops.disable_batch_invariant_mode()
    original = torch.bmm
    try:
        with monkeypatch.context() as patch:
            # Exercise real dispatcher registration with CPU-only replacements.
            patch.setattr(ops, "get_dispatch_device_backend", lambda: "CPU")
            patch.setattr(ops, "_is_npu", False)
            patch.setattr(ops, "mm_batch_invariant", lambda a, b: a + b)
            patch.setattr(ops, "bmm_batch_invariant", lambda a, b, *, out=None: a + b)
            patch.setattr(torch, "bmm", original)
            try:
                yield original
            finally:
                ops.disable_batch_invariant_mode()
    finally:
        if was_enabled:
            ops.enable_batch_invariant_mode(enable_bmm=had_bmm)


def _assert_mode(enabled, enable_bmm, native_bmm):
    assert ops.is_batch_invariant_mode_enabled() is enabled
    assert torch.bmm is (ops.bmm_batch_invariant if enable_bmm else native_bmm)
    a = torch.tensor([[2.0]], device="cpu")
    b = torch.tensor([[3.0]], device="cpu")
    assert torch.mm(a, b).item() == (5.0 if enabled else 6.0)
    for bmm in (torch.bmm, torch.ops.aten.bmm.default):
        assert bmm(a.unsqueeze(0), b.unsqueeze(0)).item() == (
            5.0 if enable_bmm else 6.0
        )


@pytest.mark.parametrize("initial_bmm", [None, False, True])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_scope_restores_dispatch(native_bmm, initial_bmm, enabled, raises):
    was_enabled = initial_bmm is not None
    if was_enabled:
        ops.enable_batch_invariant_mode(enable_bmm=initial_bmm)
    for _ in range(2):
        with (
            pytest.raises(RuntimeError, match="scope failed")
            if raises
            else nullcontext()
        ):
            with ops.set_batch_invariant_mode(enabled):
                _assert_mode(
                    enabled,
                    enabled and (initial_bmm if was_enabled else True),
                    native_bmm,
                )
                if raises:
                    raise RuntimeError("scope failed")
        _assert_mode(was_enabled, bool(initial_bmm), native_bmm)


@pytest.mark.parametrize("enable_bmm", [False, True])
@pytest.mark.parametrize("inner_enabled", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_nested_scope_preserves_outer_dispatch(
    native_bmm, enable_bmm, inner_enabled, raises
):
    ops.enable_batch_invariant_mode(enable_bmm=enable_bmm)
    with ops.set_batch_invariant_mode():
        with (
            pytest.raises(RuntimeError, match="scope failed")
            if raises
            else nullcontext()
        ):
            with ops.set_batch_invariant_mode(inner_enabled):
                _assert_mode(inner_enabled, inner_enabled and enable_bmm, native_bmm)
                if raises:
                    raise RuntimeError("scope failed")
        _assert_mode(True, enable_bmm, native_bmm)
    _assert_mode(True, enable_bmm, native_bmm)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
