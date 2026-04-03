"""Test _ParamViewTracker: generic fix for stale parameter views in CPU offloader.

Background:
  SGLang's CPU offloader uses functional_call() to substitute parameters during
  forward.  However, functional_call only updates registered parameters/buffers,
  NOT plain tensor attributes that are views of those parameters.

  Models like Qwen3.5 create views at __init__ time:
      conv_weights = self.conv1d.weight.view(C, K)
      self.attn.conv_weights = conv_weights   # plain tensor, not Parameter

  When the offloader moves params to CPU and back, conv_weights still references
  the old CPU storage → NaN or "Expected weight.is_cuda() to be true".

Solution:
  _ParamViewTracker detects such views via storage().data_ptr() matching and
  rebuilds them from the device_state dict using as_strided() before each
  functional_call.

Run:
  SGLANG_DISABLE_CUDNN_CHECK=1 python test_offload_fix.py
"""

import os

os.environ["SGLANG_DISABLE_CUDNN_CHECK"] = "1"

import torch
import torch.nn as nn
from torch.func import functional_call

from sglang.srt.utils.offloader import _ParamViewTracker

# ---------------------------------------------------------------------------
# Helpers that mimic real model patterns
#
# Views must be created AFTER .cuda() so that view and param share GPU storage.
# In real models, params are loaded to GPU first, then views are created.
# ---------------------------------------------------------------------------


class FakeLinearAttn(nn.Module):
    """Mimics RadixLinearAttention: stores view(s) of conv1d weights."""

    def __init__(self, conv_weights):
        super().__init__()
        self.conv_weights = conv_weights

    def forward(self, x):
        if isinstance(self.conv_weights, tuple):
            total = sum(w.sum() for w in self.conv_weights)
        else:
            total = self.conv_weights.sum()
        return x + total


class FakeQwenDecoder(nn.Module):
    """Mimics Qwen3.5 GatedDeltaNet: single tensor view.

    Real pattern from sglang/srt/models/qwen3_5.py:
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)
        conv_weights = self.conv1d.weight.view(conv1d.weight.size(0), ...)
        self.attn = RadixLinearAttention(conv_weights=conv_weights, ...)
    """

    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.conv1d = nn.Linear(dim * kernel_size, dim, bias=True).cuda()
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0),
            self.conv1d.weight.size(2),
        )
        self.attn = FakeLinearAttn(conv_weights)

    def forward(self, x):
        return self.attn(x)


class FakeKimiDecoder(nn.Module):
    """Mimics KimiDeltaAttention: tuple of views.

    Real pattern:
        q_view = self.q_conv1d.weight.view(...)
        k_view = self.k_conv1d.weight.view(...)
        v_view = self.v_conv1d.weight.view(...)
        self.attn = RadixLinearAttention(conv_weights=(q_view, k_view, v_view), ...)
    """

    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.q_conv1d = nn.Linear(dim * kernel_size, dim, bias=True).cuda()
        self.k_conv1d = nn.Linear(dim * kernel_size, dim, bias=True).cuda()
        self.v_conv1d = nn.Linear(dim * kernel_size, dim, bias=True).cuda()

        self.q_conv1d.weight.data = self.q_conv1d.weight.data.unsqueeze(1)
        self.k_conv1d.weight.data = self.k_conv1d.weight.data.unsqueeze(1)
        self.v_conv1d.weight.data = self.v_conv1d.weight.data.unsqueeze(1)

        q_view = self.q_conv1d.weight.view(
            self.q_conv1d.weight.size(0), self.q_conv1d.weight.size(2)
        )
        k_view = self.k_conv1d.weight.view(
            self.k_conv1d.weight.size(0), self.k_conv1d.weight.size(2)
        )
        v_view = self.v_conv1d.weight.view(
            self.v_conv1d.weight.size(0), self.v_conv1d.weight.size(2)
        )

        self.attn = FakeLinearAttn((q_view, k_view, v_view))

    def forward(self, x):
        return self.attn(x)


class FakeTiedParam(nn.Module):
    """Mimics tied parameters (e.g., A_log in both GatedDeltaNet and child)."""

    def __init__(self, dim):
        super().__init__()
        self.A_log = nn.Parameter(torch.randn(dim)).cuda()
        self.child = nn.Module()
        self.child.A_log = self.A_log

    def forward(self, x):
        return x + self.A_log.sum() + self.child.A_log.sum()


# ===================================================================
# Part 1: Before/After demonstration
# Shows what breaks WITHOUT the fix and that it works WITH the fix.
# ===================================================================


def test_without_fix_stale_view_has_wrong_storage():
    """Demonstrate the bug: stale view references old storage, not device_state.

    After offloading params to CPU, conv_weights still references the OLD
    GPU storage.  When functional_call replaces the parameter with a NEW
    GPU copy from device_state, conv_weights still points to old data.
    """
    dim = 16
    module = FakeQwenDecoder(dim).cuda()

    # Record original data_ptr
    original_ptr = module.attn.conv_weights.data_ptr()

    # Offload params to CPU
    saved = {}
    for name, p in module.named_parameters():
        saved[name] = p.data
        p.data = p.data.cpu()

    # Build device_state (new GPU copies from CPU data)
    device_state = {
        k: v.to("cuda", non_blocking=True) for k, v in module.state_dict().items()
    }

    # WITHOUT refresh: conv_weights still points to old GPU storage
    # while device_state has a DIFFERENT GPU tensor for conv1d.weight
    new_param_ptr = device_state["conv1d.weight"].data_ptr()
    stale_view_ptr = module.attn.conv_weights.data_ptr()
    assert (
        stale_view_ptr != new_param_ptr
    ), "Bug: conv_weights should point to OLD storage, not device_state copy"
    assert (
        stale_view_ptr == original_ptr
    ), "conv_weights should still reference original pre-offload storage"

    # Restore
    for name, p in module.named_parameters():
        p.data = saved[name]
    print("PASS: without fix, stale view references old storage (confirmed bug)")


def test_with_fix_view_is_refreshed():
    """Demonstrate the fix: _ParamViewTracker refreshes the view to GPU."""
    dim = 16
    module = FakeQwenDecoder(dim).cuda()

    # Reference output on GPU
    x = torch.randn(1, 4, dim, device="cuda")
    with torch.no_grad():
        ref_out = module(x).clone()

    # Detect views BEFORE offloading
    tracker = _ParamViewTracker(module)
    assert tracker.has_views

    # Offload to CPU
    saved = {}
    for name, p in module.named_parameters():
        saved[name] = p.data
        p.data = p.data.cpu()

    # Build device_state and REFRESH views
    device_state = {
        k: v.to("cuda", non_blocking=True) for k, v in module.state_dict().items()
    }
    tracker.refresh(module, device_state)

    # conv_weights is now on CUDA
    assert (
        module.attn.conv_weights.device.type == "cuda"
    ), f"After refresh, conv_weights should be CUDA, got {module.attn.conv_weights.device}"

    # Forward produces correct result
    with torch.no_grad():
        out = functional_call(
            module, device_state, args=(x,), kwargs={}, tie_weights=False
        )
    assert torch.allclose(
        unwrap(out), unwrap(ref_out), atol=1e-5
    ), f"Output mismatch: max diff = {(unwrap(out) - unwrap(ref_out)).abs().max():.2e}"

    for name, p in module.named_parameters():
        p.data = saved[name]
    print("PASS: with fix, view is refreshed and output is correct")


# ===================================================================
# Part 2: Unit tests for _ParamViewTracker functionality
# ===================================================================


def test_detect_single_view():
    """Single tensor view (Qwen3.5 pattern) is detected."""
    module = FakeQwenDecoder(16).cuda()
    tracker = _ParamViewTracker(module)
    assert tracker.has_views
    print("PASS: detect single view")


def test_detect_tuple_views():
    """Tuple of views (Kimi pattern) is detected."""
    module = FakeKimiDecoder(16).cuda()
    tracker = _ParamViewTracker(module)
    assert tracker.has_views
    print("PASS: detect tuple views")


def test_no_false_positive():
    """Module without views should not be detected."""

    class PlainModule(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
            self.some_tensor = torch.randn(dim)

        def forward(self, x):
            return self.linear(x)

    module = PlainModule(16).cuda()
    tracker = _ParamViewTracker(module)
    assert not tracker.has_views
    print("PASS: no false positive")


def test_refresh_tuple_views():
    """Offload + refresh tuple of views (Kimi pattern)."""
    dim = 16
    module = FakeKimiDecoder(dim).cuda()

    x = torch.randn(1, 4, dim, device="cuda")
    with torch.no_grad():
        ref_out = module(x).clone()

    tracker = _ParamViewTracker(module)
    assert tracker.has_views

    saved = {n: p.data for n, p in module.named_parameters()}
    for p in module.parameters():
        p.data = p.data.cpu()

    device_state = {k: v.to("cuda") for k, v in module.state_dict().items()}
    tracker.refresh(module, device_state)

    for i, elem in enumerate(module.attn.conv_weights):
        assert (
            elem.device.type == "cuda"
        ), f"conv_weights[{i}] should be CUDA, got {elem.device}"

    with torch.no_grad():
        out = functional_call(
            module, device_state, args=(x,), kwargs={}, tie_weights=False
        )
    assert torch.allclose(unwrap(out), unwrap(ref_out), atol=1e-5)

    for n, p in module.named_parameters():
        p.data = saved[n]
    print("PASS: refresh tuple views + correct output")


def test_tied_weights_no_crash():
    """functional_call with tie_weights=False should handle tied params."""
    dim = 16
    module = FakeTiedParam(dim).cuda()

    saved = {n: p.data for n, p in module.named_parameters()}
    for p in module.parameters():
        p.data = p.data.cpu()

    device_state = {k: v.to("cuda") for k, v in module.state_dict().items()}
    x = torch.randn(2, dim, device="cuda")

    with torch.no_grad():
        out = functional_call(
            module, device_state, args=(x,), kwargs={}, tie_weights=False
        )
    assert torch.isfinite(unwrap(out)).all()

    for n, p in module.named_parameters():
        p.data = saved[n]
    print("PASS: tied weights + tie_weights=False")


# ---------------------------------------------------------------------------
# Util
# ---------------------------------------------------------------------------


def unwrap(x):
    return x[0] if isinstance(x, tuple) else x


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    print("=== Part 1: Before/After demonstration ===")
    test_without_fix_stale_view_has_wrong_storage()
    test_with_fix_view_is_refreshed()

    print("\n=== Part 2: Unit tests ===")
    test_detect_single_view()
    test_detect_tuple_views()
    test_no_false_positive()
    test_refresh_tuple_views()
    test_tied_weights_no_crash()

    print("\nAll tests passed!")
