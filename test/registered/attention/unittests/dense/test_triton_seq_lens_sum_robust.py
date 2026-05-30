"""Regression test: Triton ``init_forward_metadata`` must size ``kv_indices``
from the actual GPU cumsum, not from a possibly-stale
``forward_batch.seq_lens_sum``.

Pre-#26665 Triton sized ``kv_indices`` as ``torch.empty(kv_indptr[-1], …)``
(the exact total derived from this call's cumsum). #26665's refactor switched
to ``torch.empty(forward_batch.seq_lens_sum, …)``, which is unsafe in three
scenarios:

1. ``needs_cpu_seq_lens=False`` (introduced by #26128): the scheduler may
   skip computing ``seq_lens_sum``, leaving it as ``None``. ``torch.empty(None
   …)`` raises ``TypeError`` and the request faults.
2. Stale cached value: spec-decode workers bump ``forward_batch.seq_lens``
   for a verify batch but don't always refresh ``seq_lens_sum``; the cached
   value reflects the pre-bump state and is smaller than the actual cumsum
   total.
3. Test/utility scaffolding that sets ``seq_lens`` directly without
   propagating to ``seq_lens_sum``.

When ``seq_lens_sum < sum(seq_lens)``, the allocated ``kv_indices`` buffer
is undersized; ``create_flashinfer_kv_indices_triton`` writes OOB, corrupts
neighboring memory, and a later ``_fwd_kernel`` invocation reads garbage
indices and faults with ``CUDBG_EXCEPTION_WARP_OUT_OF_RANGE_ADDRESS``.

That's the exact failure pattern hit by ``test_spec_ngram_extra.py`` on
``main`` post-#26665 — a GPU-only e2e failure that takes minutes to surface
and produces a coredump with only ``_fwd_kernel`` on the stack, with no
indication of which init wrote the bad metadata.

These tests pin the contract at the init layer (≈seconds) so the same
regression class can't sneak through the e2e gate next time.
"""

import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.dense_attention import (
    DenseAttentionCase,
    build_dense_attention_fixture,
)

register_cuda_ci(est_time=10, stage="base-a", runner_config="1-gpu-small")


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestTritonSeqLensSumRobust(CustomTestCase):

    def _build(self, *, forward_mode: ForwardMode, prefix_lens, extend_lens=None):
        case = DenseAttentionCase(
            name=f"seq_lens_sum_robust_{forward_mode.name}",
            backend="triton",
            forward_mode=forward_mode,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=prefix_lens,
            extend_lens=extend_lens,
        )
        try:
            return build_dense_attention_fixture(self, case)
        except (AssertionError, ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"triton backend unavailable: {exc}")

    # ----- positive: init must produce a kv_indices view consistent with kv_indptr -----

    def _assert_kv_indices_consistent_with_indptr(self, fixture):
        """``kv_indices.numel() >= kv_indptr[-1]`` post-init.

        ``create_flashinfer_kv_indices_triton`` writes ``[kv_indptr[i],
        kv_indptr[i+1])`` per sequence, so the kv_indices buffer must cover
        ``kv_indptr[-1]``. Pinning this catches the family of bugs where
        ``kv_indices`` is sized off a stale CPU sum.
        """
        fixture.backend.init_forward_metadata(fixture.forward_batch)
        meta = fixture.backend.forward_metadata
        self.assertIsNotNone(meta)
        kv_indptr = meta.kv_indptr
        kv_indices = meta.kv_indices
        self.assertIsNotNone(kv_indptr, "kv_indptr must be set after init")
        self.assertIsNotNone(kv_indices, "kv_indices must be set after init")
        total_from_indptr = int(kv_indptr[-1].item())
        self.assertGreaterEqual(
            kv_indices.numel(),
            total_from_indptr,
            f"kv_indices undersized: numel={kv_indices.numel()} but "
            f"kv_indptr[-1]={total_from_indptr}. "
            "create_flashinfer_kv_indices_triton would write OOB.",
        )

    def test_decode_consistent(self):
        fixture = self._build(forward_mode=ForwardMode.DECODE, prefix_lens=(7, 11, 13))
        self._assert_kv_indices_consistent_with_indptr(fixture)

    def test_extend_consistent(self):
        fixture = self._build(
            forward_mode=ForwardMode.EXTEND,
            prefix_lens=(0, 8, 16),
            extend_lens=(15, 8, 1),
        )
        self._assert_kv_indices_consistent_with_indptr(fixture)

    # ----- negative: init must not depend on forward_batch.seq_lens_sum being accurate -----

    def _force_stale_seq_lens_sum(self, forward_batch, stale_value):
        """Simulate the spec-v2 / spec-decode pattern where the cached sum is
        stale relative to the GPU ``seq_lens`` tensor (or unset entirely).
        """
        forward_batch.seq_lens_sum = stale_value

    def test_decode_robust_to_none_seq_lens_sum(self):
        """``forward_batch.seq_lens_sum = None`` (the ``needs_cpu_seq_lens=False``
        path from #26128) must not break ``init_forward_metadata`` —
        ``torch.empty(None, …)`` raises ``TypeError`` if a caller is naively
        relying on the cached sum to size buffers.
        """
        fixture = self._build(forward_mode=ForwardMode.DECODE, prefix_lens=(7, 11, 13))
        self._force_stale_seq_lens_sum(fixture.forward_batch, None)
        # Should not raise; kv_indices must be sized off the GPU cumsum total.
        fixture.backend.init_forward_metadata(fixture.forward_batch)
        self._assert_kv_indices_consistent_with_indptr_post_init(fixture)

    def test_decode_robust_to_understated_seq_lens_sum(self):
        """Caller cached ``seq_lens_sum`` before bumping ``seq_lens`` (the
        spec verify pattern). The init must size ``kv_indices`` off the GPU
        cumsum, not the stale cached value, otherwise OOB writes corrupt
        memory and a later kernel reads garbage indices."""
        fixture = self._build(forward_mode=ForwardMode.DECODE, prefix_lens=(7, 11, 13))
        # Pretend the cached sum reflects a smaller pre-bump state.
        actual_sum = int(fixture.forward_batch.seq_lens.sum().item())
        self._force_stale_seq_lens_sum(fixture.forward_batch, actual_sum // 2)
        fixture.backend.init_forward_metadata(fixture.forward_batch)
        self._assert_kv_indices_consistent_with_indptr_post_init(fixture)

    def test_extend_robust_to_stale_extend_prefix_lens_cpu(self):
        """For EXTEND mode, ``init_forward_metadata`` similarly used
        ``sum(extend_prefix_lens_cpu)`` to size ``kv_indices``. The fix sizes
        from the actual GPU cumsum total via the helper; this test pins the
        contract.
        """
        fixture = self._build(
            forward_mode=ForwardMode.EXTEND,
            prefix_lens=(0, 8, 16),
            extend_lens=(15, 8, 1),
        )
        # Corrupt the cached CPU list to simulate it being stale.
        fixture.forward_batch.extend_prefix_lens_cpu = [0, 0, 0]
        fixture.backend.init_forward_metadata(fixture.forward_batch)
        self._assert_kv_indices_consistent_with_indptr_post_init(fixture)

    def _assert_kv_indices_consistent_with_indptr_post_init(self, fixture):
        # Same as the helper above but expects init has already been called.
        meta = fixture.backend.forward_metadata
        self.assertIsNotNone(meta)
        kv_indptr = meta.kv_indptr
        kv_indices = meta.kv_indices
        self.assertIsNotNone(kv_indptr)
        self.assertIsNotNone(kv_indices)
        total_from_indptr = int(kv_indptr[-1].item())
        self.assertGreaterEqual(
            kv_indices.numel(),
            total_from_indptr,
            f"kv_indices undersized: numel={kv_indices.numel()} but "
            f"kv_indptr[-1]={total_from_indptr}.",
        )


if __name__ == "__main__":
    unittest.main()
