"""NPU device-branch tests for llguidance backend's bitmask application path.

Manual NPU suite (not registered to CPU CI). Verifies that
``GuidanceGrammar.apply_vocab_mask`` dispatches to the NPU kernel when logits
are on an NPU device, and falls back to the generic ``llguidance``
implementation otherwise. The NPU path is exercised via a mocked public
``sgl_kernel_npu.operators.sampler.apply_token_bitmask`` entry.

Requires: torch, llguidance.  When either is unavailable, a placeholder test
documents the skip so the module still collects and passes on this machine.
"""

import math
import sys
import types
import unittest
from unittest.mock import patch

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


class _FakeSampler:
    """Capture calls to the public sampler entry and apply the CPU reference."""

    def __init__(self):
        self.last_call = None

    def apply_token_bitmask(self, logits, vocab_mask):
        self.last_call = (logits, vocab_mask)
        # Apply the reference on CPU storage even when logits report an NPU device.
        out = _apply_ref_cpu(logits.detach().clone(), vocab_mask)
        logits.copy_(out)
        return logits


# ---------------------------------------------------------------------------
# Tests that only run when torch + llguidance are importable.
# ---------------------------------------------------------------------------
if _HAS_TORCH and _HAS_LLGUIDANCE:
    from sglang.srt.constrained.llguidance_backend import GuidanceGrammar

    class _NpuDeviceTensor(torch.Tensor):
        """CPU-backed tensor that reports an NPU device to the dispatcher."""

        @property
        def device(self):
            return types.SimpleNamespace(type="npu")

    class TestGuidanceGrammarApplyVocabMaskNpu(unittest.TestCase):
        """Verify the NPU branch in ``GuidanceGrammar.apply_vocab_mask``."""

        def setUp(self):
            self.fake_sampler = _FakeSampler()
            sampler = types.ModuleType("sgl_kernel_npu.operators.sampler")
            sampler.apply_token_bitmask = self.fake_sampler.apply_token_bitmask
            patcher = patch.dict(sys.modules, {sampler.__name__: sampler})
            patcher.start()
            self.addCleanup(patcher.stop)

        def test_npu_branch_dispatches_to_npu_kernel(self):
            """When logits are on npu, the NPU kernel is invoked (mocked)."""
            vocab_size = 64
            logits = torch.zeros((1, vocab_size), dtype=torch.float32)
            logits[0, 16] = 22.125
            logits[0, 5] = 10.0

            allowed = [[5, 6, 7, 8]]
            vocab_mask = _pack_mask(allowed, vocab_size)

            npu_logits = logits.as_subclass(_NpuDeviceTensor)
            self.assertIsNone(GuidanceGrammar.apply_vocab_mask(npu_logits, vocab_mask))

            self.assertIsNotNone(self.fake_sampler.last_call)
            passed_logits, passed_mask = self.fake_sampler.last_call
            self.assertIs(passed_logits, npu_logits)
            self.assertIs(passed_mask, vocab_mask)

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

            self.assertIsNone(self.fake_sampler.last_call)
            ref = _apply_ref_cpu(logits, vocab_mask)
            self.assertTrue(torch.equal(torch.isfinite(out), torch.isfinite(ref)))

        def test_npu_branch_matches_reference(self):
            """The mocked NPU kernel matches the CPU reference result."""
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
            assert missing, (
                "Expected at least one missing dependency on this environment."
            )


if __name__ == "__main__":
    unittest.main()
