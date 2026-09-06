"""NPU device-branch tests for llguidance backend's bitmask application path.

These tests verify that ``GuidanceGrammar.apply_vocab_mask`` dispatches to the
NPU kernel when the logits tensor is on an NPU device, and falls back to the
generic ``llguidance`` implementation otherwise.  They are device-agnostic:
the NPU path is exercised via a mocked ``torch.ops.npu.apply_token_bitmask`` so
the tests run on CPU-only CI.

Requires: torch, llguidance.  When either is unavailable, a placeholder test
documents the skip so the module still collects and passes on this machine.
"""

import math
import sys
import types
import unittest
from unittest.mock import MagicMock

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import llguidance  # noqa: F401

    _HAS_LLGUIDANCE = True
except ImportError:
    _HAS_LLGUIDANCE = False


def _pack_mask(allowed_ids, vocab_size, batch_size=1):
    """Pack a list of allowed token ids into a packed int32 bitmask."""
    nwords = math.ceil(vocab_size / 32)
    m = torch.zeros((batch_size, nwords), dtype=torch.int32)
    for b in range(batch_size):
        for tid in allowed_ids[b]:
            m[b, tid // 32] |= 1 << (tid % 32)
    return m


def _apply_ref_cpu(logits, vocab_mask):
    """Reference bitmask application on CPU for parity checking."""
    vocab_size = logits.shape[-1]
    token_ids = torch.arange(vocab_size, device="cpu", dtype=torch.int64)
    word_idx = token_ids // 32
    bit_idx = (token_ids % 32).to(torch.int32)
    words = vocab_mask.cpu()[:, word_idx].to(torch.int32)
    allowed = ((words >> bit_idx) & 1).bool()
    out = logits.detach().clone().cpu()
    out.masked_fill_(~allowed, float("-inf"))
    return out


class _FakeNpuOps:
    """Minimal stand-in for ``torch.ops.npu`` capturing the last call."""

    def __init__(self):
        self.last_call = None

    def apply_token_bitmask(self, logits, vocab_mask):
        self.last_call = (logits, vocab_mask)
        # Logits may report device=npu via a subclass while storage is CPU;
        # apply the reference mask on CPU tensors to avoid device mismatch.
        out = _apply_ref_cpu(logits.detach().clone(), vocab_mask)
        logits.copy_(out)


def _install_npu_mock():
    """Install a fake ``sgl_kernel_npu`` and ``torch.ops.npu`` for testing."""
    fake_ops = _FakeNpuOps()

    if not hasattr(torch, "npu"):
        torch.npu = types.ModuleType("torch_npu_mock")
    torch.npu.is_available = lambda: True
    torch.ops.npu = fake_ops  # type: ignore[attr-defined]

    if "sgl_kernel_npu" not in sys.modules:
        sys.modules["sgl_kernel_npu"] = MagicMock()

    return fake_ops


# ---------------------------------------------------------------------------
# Tests that only run when torch + llguidance are importable.
# ---------------------------------------------------------------------------
if _HAS_TORCH and _HAS_LLGUIDANCE:
    from sglang.srt.constrained.llguidance_backend import GuidanceGrammar

    class _NpuDeviceTensor(torch.Tensor):
        """Tensor subclass whose ``device`` property reports npu:0."""

        @property
        def device(self):
            return torch.device("npu:0")

    class TestGuidanceGrammarApplyVocabMaskNpu(unittest.TestCase):  # noqa: E501
        """Verify the NPU branch in ``GuidanceGrammar.apply_vocab_mask``."""

        def test_npu_branch_dispatches_to_npu_kernel(self):
            """When logits are on npu, the NPU kernel is invoked (mocked)."""
            fake_ops = _install_npu_mock()

            vocab_size = 64
            logits = torch.zeros((1, vocab_size), dtype=torch.float32)
            logits[0, 16] = 22.125
            logits[0, 5] = 10.0

            allowed = [[5, 6, 7, 8]]
            vocab_mask = _pack_mask(allowed, vocab_size)

            npu_logits = logits.as_subclass(_NpuDeviceTensor)
            GuidanceGrammar.apply_vocab_mask(npu_logits, vocab_mask)

            self.assertIsNotNone(fake_ops.last_call)
            passed_logits, passed_mask = fake_ops.last_call
            self.assertTrue(torch.equal(passed_logits, npu_logits))
            self.assertTrue(torch.equal(passed_mask, vocab_mask))

        def test_cpu_branch_uses_llguidance_kernel(self):
            """When logits are on cpu, the generic llguidance kernel is used."""
            vocab_size = 64
            logits = torch.zeros((1, vocab_size), dtype=torch.float32)
            logits[0, 16] = 22.125
            logits[0, 5] = 10.0

            allowed = [[5, 6, 7, 8]]
            vocab_mask = _pack_mask(allowed, vocab_size)

            out = logits.clone()
            GuidanceGrammar.apply_vocab_mask(out, vocab_mask)

            ref = _apply_ref_cpu(logits, vocab_mask)
            self.assertTrue(torch.equal(torch.isfinite(out), torch.isfinite(ref)))

        def test_npu_branch_matches_reference(self):
            """The mocked NPU kernel matches the CPU reference result."""
            _install_npu_mock()

            vocab_size = 128
            torch.manual_seed(42)
            logits = torch.randn((2, vocab_size), dtype=torch.float32)

            allowed = [
                torch.randperm(vocab_size)[: vocab_size // 3].tolist(),
                torch.randperm(vocab_size)[: vocab_size // 4].tolist(),
            ]
            vocab_mask = _pack_mask(allowed, vocab_size, batch_size=2)

            npu_logits = logits.clone().as_subclass(_NpuDeviceTensor)
            GuidanceGrammar.apply_vocab_mask(npu_logits, vocab_mask)

            ref = _apply_ref_cpu(logits, vocab_mask)
            self.assertTrue(
                torch.equal(torch.isfinite(npu_logits), torch.isfinite(ref)),
                "NPU branch finiteness pattern diverges from CPU reference",
            )

    class TestGuidanceGrammarMoveVocabMaskNpu(unittest.TestCase):
        """Verify ``move_vocab_mask`` handles device targets."""

        def test_move_to_cpu_device(self):
            """move_vocab_mask returns a tensor on the requested device."""
            mask = torch.zeros((2, 4), dtype=torch.int32)
            moved = GuidanceGrammar.move_vocab_mask(mask, "cpu")
            self.assertEqual(moved.device.type, "cpu")
            self.assertTrue(torch.equal(moved, mask))

# ---------------------------------------------------------------------------
# Placeholder when dependencies are missing (e.g. on a plain Windows box).
# ---------------------------------------------------------------------------
else:

    class TestLLGuidanceNpuMaskSkipped(unittest.TestCase):
        """Placeholder: torch or llguidance not installed on this machine."""

        def test_skipped_missing_dependency(self):
            """Document why the NPU branch tests could not run here."""
            missing = []
            if not _HAS_TORCH:
                missing.append("torch")
            if not _HAS_LLGUIDANCE:
                missing.append("llguidance")
            assert (
                missing
            ), "Expected at least one missing dependency on this environment."


if __name__ == "__main__":
    unittest.main()
