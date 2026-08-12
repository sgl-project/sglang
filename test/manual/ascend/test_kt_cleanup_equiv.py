# SPDX-License-Identifier: Apache-2.0
"""NPU-free regression locks for the KT clean-code pass (single-card 910C DSV4-Flash).

These tests pin the *pure-tensor* behaviour of the KT streaming-prefill helpers that the
clean-up either relies on (when deleting the diagnostic ``KT_DYN_FORCE_*`` branches) or
de-duplicates (when making ``_apply_dynamic_residency`` reuse ``_pick_resident_top`` /
``_set_resident_masks`` instead of re-implementing them inline). They run on CPU with no
NPU and no CANN — ``kt_stream_prefill`` is loaded straight from its file so importing it
pulls in no ``torch_npu`` (all NPU imports in that module are lazy, inside functions).

Run:  python -m pytest test/manual/ascend/test_kt_cleanup_equiv.py -q
  or:  python test/manual/ascend/test_kt_cleanup_equiv.py
"""
import importlib.util
import os
import pathlib
import types
import unittest

import torch


def _load_kt_stream_prefill():
    """Import kt_stream_prefill by file path (no sglang package chain, no NPU)."""
    here = pathlib.Path(__file__).resolve()
    rel = pathlib.Path("python/sglang/srt/layers/moe/kt_stream_prefill.py")
    for base in here.parents:
        cand = base / rel
        if cand.exists():
            spec = importlib.util.spec_from_file_location("kt_stream_prefill_under_test", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(f"could not locate {rel} above {here}")


class _FakeWrap:
    """Minimal stand-in for a KTEPWrapperMethod: the tensors _set_resident_masks writes."""

    def __init__(self, E, K):
        self.gpu_experts_mask = torch.zeros(E, dtype=torch.bool)
        self.logical_to_gpu_index = torch.full((E,), -1, dtype=torch.int64)
        # C++-side pinned bool mask lives under `.wrapper`; use a tiny namespace with its own tensor.
        self.wrapper = types.SimpleNamespace(gpu_experts_mask=torch.zeros(E, dtype=torch.bool))


class KTCleanupEquivTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The diagnostic force-set env vars are read at call time; clear them so we test the
        # production (default top-K) behaviour that the clean-up preserves.
        for k in ("KT_DYN_FORCE_SET", "KT_DYN_FORCE_PREFIX", "KT_DYN_SKIP_WEIGHTS"):
            os.environ.pop(k, None)
        cls.ksp = _load_kt_stream_prefill()

    def test_pick_resident_top_is_sorted_topk(self):
        """Production selection == top-K by activation, returned ascending. This is the invariant
        that must survive deleting the KT_DYN_FORCE_* diagnostic branches from _pick_resident_top."""
        torch.manual_seed(0)
        E = 256
        for K in (1, 8, 32, 64):
            for _ in range(20):
                counts = torch.randint(0, 1000, (E,), dtype=torch.int64)
                got = self.ksp._pick_resident_top(counts, K)
                ref = counts.topk(K).indices.sort().values
                self.assertTrue(torch.equal(got, ref), f"K={K}: {got} != {ref}")
                # ascending + within range + no dup
                self.assertTrue(torch.equal(got, got.sort().values))
                self.assertEqual(got.unique().numel(), K)
                self.assertTrue(int(got.min()) >= 0 and int(got.max()) < E)

    def test_set_resident_masks_structures(self):
        """_set_resident_masks writes exactly: mask True on the resident set, l2g = resident->slot
        (ascending), -1 elsewhere, and mirrors the mask to the C++-side pinned tensor."""
        torch.manual_seed(1)
        E, K = 256, 32
        for _ in range(20):
            top = torch.randperm(E)[:K].sort().values
            wrap = _FakeWrap(E, K)
            self.ksp._set_resident_masks(wrap, top, K, E)

            exp_mask = torch.zeros(E, dtype=torch.bool)
            exp_mask[top] = True
            exp_l2g = torch.full((E,), -1, dtype=torch.int64)
            exp_l2g[top] = torch.arange(K, dtype=torch.int64)

            self.assertTrue(torch.equal(wrap.gpu_experts_mask, exp_mask))
            self.assertTrue(torch.equal(wrap.logical_to_gpu_index, exp_l2g))
            self.assertTrue(torch.equal(wrap.wrapper.gpu_experts_mask, exp_mask))
            self.assertEqual(int(wrap.gpu_experts_mask.sum()), K)
            self.assertEqual(int((wrap.logical_to_gpu_index >= 0).sum()), K)

    def test_pick_then_set_roundtrip(self):
        """The two helpers compose the way both the inline-depool and W8A8 post-pass paths use them:
        pick top-K from counts, install as the resident set."""
        torch.manual_seed(2)
        E, K = 256, 32
        counts = torch.randint(0, 1000, (E,), dtype=torch.int64)
        top = self.ksp._pick_resident_top(counts, K)
        wrap = _FakeWrap(E, K)
        self.ksp._set_resident_masks(wrap, top.cpu(), K, E)
        # every resident logical id maps to a distinct slot in [0,K)
        slots = wrap.logical_to_gpu_index[top]
        self.assertTrue(torch.equal(slots.sort().values, torch.arange(K, dtype=torch.int64)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
