"""Tests for OffloaderV1 with tied parameters and view aliases (see issue #23150).

Two failure modes caused the Qwen3-Next / Qwen3.5 CPU-offload regression:

1. **Tied parameters**: a single nn.Parameter is registered under both a parent
   and a child module (Qwen3GatedDeltaNet + RadixLinearAttention share
   ``A_log`` / ``dt_bias``). state_dict() then lists the same tensor under
   multiple keys, and functional_call(..., tie_weights=True) rejects it when
   the duplicate keys end up with distinct device-side tensors.

2. **Views of a parameter cached as a plain tensor attribute**: those same
   models stash ``conv1d.weight.view(...)`` on ``RadixLinearAttention.conv_weights``.
   Because that attribute is a plain Tensor (not an nn.Parameter) it is not in
   state_dict(); once the offloader rebinds ``conv1d.weight.data`` to pinned
   CPU memory, subsequent weight loading writes to the new CPU storage and the
   cached view is left pointing at uninitialised GPU memory. Without repair
   the forward pass silently produces garbage logits.

Both regressions are reproduced by handwritten minimal modules below.
"""

import unittest

import torch
import torch.nn as nn

from sglang.srt.utils.offloader import OffloaderV1
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, suite="stage-b-test-1-gpu-small")


class _TiedChild(nn.Module):
    def __init__(self, weight: nn.Parameter):
        super().__init__()
        # Assigning an nn.Parameter via setattr registers it again, so the
        # same tensor is now reachable under both parent.w and parent.child.w.
        self.w = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w


class _TiedParent(nn.Module):
    def __init__(self, dim: int = 4):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim, dim))
        self.child = _TiedChild(self.w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.child(x)


class _ViewChild(nn.Module):
    """Stores a plain-Tensor view of the parent's conv weight (non-Parameter).

    Mirrors `RadixLinearAttention.conv_weights`.
    """

    def __init__(self, view: torch.Tensor):
        super().__init__()
        # IMPORTANT: ``view`` is a plain Tensor (not a Parameter) so it is
        # *not* registered in state_dict()/parameters() — that is the whole
        # reason the offloader view-repair path is needed.
        self.view = view

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.view


class _ViewParent(nn.Module):
    """Parent with a conv-like weight and a sibling submodule that caches a
    view of that weight. Weight loading happens AFTER offloader wrapping.

    The parameter is created directly on CUDA so the cached view shares the
    same GPU storage — matching how the real hybrid models create
    ``conv1d.weight`` on the current device before viewing it.
    """

    def __init__(self, in_dim: int = 4, out_dim: int = 4):
        super().__init__()
        # Parameter starts as zeros on CUDA; the real values are written
        # *after* the offloader has pinned it, matching how SGLang loads
        # weights post-construction.
        self.weight = nn.Parameter(torch.zeros(out_dim, in_dim, device="cuda"))
        # Cached view (plain tensor attribute — same pattern as conv_weights).
        self.child = _ViewChild(self.weight.view(out_dim, in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.child(x)


class TestOffloaderV1TiedParams(CustomTestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for OffloaderV1 forward path")
        torch.manual_seed(0)

    def test_state_dict_lists_tied_paths(self):
        """Sanity check — confirms the tied-parameter pattern we rely on."""
        module = _TiedParent().cuda()
        entries = dict(module.state_dict(keep_vars=True))
        self.assertIn("w", entries)
        self.assertIn("child.w", entries)
        self.assertIs(entries["w"], entries["child.w"])

    def test_forward_matches_reference(self):
        """Wrapping a tied module with OffloaderV1 must not crash and must
        produce the same output as the unwrapped baseline."""
        reference = _TiedParent().cuda().eval()
        x = torch.randn(2, 4, device="cuda")
        with torch.no_grad():
            expected = reference(x)

        # Clone so the offloader's in-place CPU pinning does not disturb the
        # reference module.
        wrapped = _TiedParent().cuda().eval()
        wrapped.load_state_dict(reference.state_dict())

        # cpu_offload_max_bytes big enough to offload every parameter.
        offloader = OffloaderV1(cpu_offload_max_bytes=1 << 30)
        offloader.maybe_offload_to_cpu(wrapped)

        # Pinning moves .data to CPU and swaps forward.
        self.assertEqual(wrapped.w.device.type, "cpu")
        self.assertIsNot(wrapped.forward, _TiedParent.forward)

        with torch.no_grad():
            actual = wrapped(x)
        torch.testing.assert_close(actual, expected)

        # Second call exercises the forward-restore path.
        with torch.no_grad():
            actual2 = wrapped(x)
        torch.testing.assert_close(actual2, expected)


class TestOffloaderV1ViewAlias(CustomTestCase):
    """Regression coverage for the Qwen3.5 / Qwen3-Next conv_weights view."""

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for OffloaderV1 forward path")
        torch.manual_seed(0)

    def test_view_alias_is_refreshed(self):
        """A plain-tensor view of a Parameter must still yield the loaded
        values during the offloaded forward, even though the view itself is
        not in state_dict()."""
        module = _ViewParent().eval()
        # Sanity: the view indeed aliases the parameter's storage before we
        # touch anything.
        self.assertEqual(
            module.child.view.untyped_storage().data_ptr(),
            module.weight.untyped_storage().data_ptr(),
        )

        # Offload first (parameter goes to pinned CPU, cached view is
        # orphaned on the original GPU storage).
        offloader = OffloaderV1(cpu_offload_max_bytes=1 << 30)
        offloader.maybe_offload_to_cpu(module)
        self.assertEqual(module.weight.device.type, "cpu")

        # Then load the "real" weights, mimicking SGLang's load order.
        real = torch.randn(4, 4)
        module.weight.data.copy_(real)

        x = torch.randn(2, 4, device="cuda")
        with torch.no_grad():
            actual = module(x)
        expected = x @ real.to("cuda")
        torch.testing.assert_close(actual, expected)

        # A stale-view bug would silently return zeros (since the view
        # captures the pre-load storage); assert non-zero to catch that
        # regression crisply even if assert_close's tolerance ever drifts.
        self.assertGreater(actual.abs().sum().item(), 0.0)

    def test_view_attr_is_restored_after_forward(self):
        """The forward closure should leave the original view attribute in
        place after returning so the module remains inspectable."""
        module = _ViewParent().eval()
        offloader = OffloaderV1(cpu_offload_max_bytes=1 << 30)
        offloader.maybe_offload_to_cpu(module)
        module.weight.data.copy_(torch.randn(4, 4))

        original_view = module.child.view
        x = torch.randn(2, 4, device="cuda")
        with torch.no_grad():
            module(x)
        # After the closure returns, the attribute should point back at the
        # original view (still a valid Tensor, even if its storage is now
        # orphaned).
        self.assertIs(module.child.view, original_view)


if __name__ == "__main__":
    unittest.main()
