"""XPU pinned host memory support for `sglang.srt.utils.is_pin_memory_available`.

Intel XPU supports pinned host memory (`torch.empty(..., pin_memory=True)` on a
CPU tensor is page-locked and `.is_pinned()` returns True), which sglang relies
on for async H2D staging on the hot path — e.g. `ScheduleBatch` input-id /
seq-len staging (`flatten_arrays_to_pinned_cpu`), the overlap scheduler's D2H
copies, and the host KV cache (`memory_pool_host`). All of those gate the
`pin_memory=` kwarg on `is_pin_memory_available(device)`; when it returns False
the copies silently fall back to pageable memory and lose the async overlap.

XPU has no dedicated in-tree platform class, so `current_platform` resolves to
the base `SRTPlatform`, whose `is_pin_memory_available` returns False. Without
the XPU short-circuit in `is_pin_memory_available`, every XPU run would report
"no pinned memory" and take the slow pageable path. This test pins the intended
semantics (mirroring CUDA): available for XPU targets, but not when the target
is explicitly CPU. If the short-circuit is dropped (regressing to the base
platform default), the `xpu`/`None` cases turn red.
"""

import pytest
import torch

from sglang.srt.utils import is_pin_memory_available, is_xpu
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=30, suite="stage-a-test-1-gpu-xpu")


@pytest.mark.skipif(not is_xpu(), reason="This test requires an Intel XPU device")
def test_is_pin_memory_available_on_xpu():
    """XPU (and a `None`/xpu-device target) reports pinned memory available;
    an explicit CPU target does not — matching CUDA semantics."""
    assert is_pin_memory_available() is True
    assert is_pin_memory_available("xpu") is True
    assert is_pin_memory_available(torch.device("xpu", 0)) is True

    # A CPU target has no accelerator to pin host memory to.
    assert is_pin_memory_available("cpu") is False
    assert is_pin_memory_available(torch.device("cpu")) is False


@pytest.mark.skipif(not is_xpu(), reason="This test requires an Intel XPU device")
def test_pinned_cpu_tensor_is_actually_pinned_on_xpu():
    """The reported capability is real: a CPU tensor allocated with the kwarg
    `is_pin_memory_available` returns is genuinely page-locked, so the async
    H2D copies that depend on it are valid."""
    pin = is_pin_memory_available("xpu")
    assert pin is True

    t = torch.empty(8, dtype=torch.int64, device="cpu", pin_memory=pin)
    assert t.is_pinned()

    # And it can be copied to the XPU device non_blocking (the actual use).
    d = t.to("xpu", non_blocking=True)
    torch.xpu.synchronize()
    assert d.device.type == "xpu"


if __name__ == "__main__":
    # Allow `python test_xpu_pin_memory.py` for local runs.
    test_is_pin_memory_available_on_xpu()
    test_pinned_cpu_tensor_is_actually_pinned_on_xpu()
    print("PASS")
