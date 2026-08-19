# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The unified-pool read translate must cover only the RAGGED PREFIX.

`kv_indices` is deliberately OVER-ALLOCATED on the Triton read path: when the
CPU token total is unavailable the caller sizes it to `bs * max_context_len`
with `torch.empty` and the fill kernel writes only `[0, kv_indptr[bs])`. The
attention kernel is indptr-bounded and never reads the tail, but a translate
applied to the WHOLE buffer gathers `virtual_to_physical[...]` over every
element — including uninitialized ids — and trips a device-side assert
(`IndexKernel.cu:111 ... index out of bounds`).

That is not theoretical, and recycled device memory makes it deterministic
rather than occasional: a previous forward's translated buffer leaves DENSE
ids in the tail, and a dense id is `kernel_page_multiplier` times larger than
any valid virtual id, so it lands far past the end of the mapping and the
gather asserts on the first batch that reaches the tail.

Speculative batches make it reachable even when the buffer is sized from the
host: a block drafter publishes `seq_lens_sum` as prefix + one verify block
while the GPU `seq_lens` that drive the ragged fill stay at the committed
prefix, so the host bound sits ABOVE the written frontier.

CPU-only — no GPU / Triton launch needed.

    python -m pytest test/registered/unit/layers/attention/test_triton_ragged_kv_translate.py -v
"""

import inspect
import unittest

import torch

from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Stands in for the allocator's virtual->physical mapping. Anything at or past
# this length is what the device gather asserts on.
_V2P_LEN = 64


def _backend(translate):
    """A bare backend carrying only what the helper under test touches."""
    be = object.__new__(TritonAttnBackend)
    be._translate_kv_loc = translate
    return be


def _strict_translate(kv_indices):
    """Mimics the device gather: raises on any id the mapping cannot hold.

    On GPU this is `virtual_to_physical[kv_indices]` and the failure is an
    async device-side assert; here it is a loud, synchronous IndexError.
    """
    bad = kv_indices[(kv_indices < 0) | (kv_indices >= _V2P_LEN)]
    if bad.numel():
        raise IndexError(f"index out of bounds: {bad.tolist()} (v2p len {_V2P_LEN})")
    # A stand-in for the real dense translate; the exact values do not matter,
    # only that untranslated ids never reach it.
    return kv_indices * 2


def _over_allocated(filled_ids, total):
    """A ragged buffer: `filled_ids` written, the tail left as recycled DENSE
    ids — exactly the state `torch.empty` hands back in the crashing run."""
    buf = torch.full((total,), 15_780_610, dtype=torch.int64)
    buf[: len(filled_ids)] = torch.tensor(filled_ids, dtype=torch.int64)
    return buf


class TestTritonRaggedKVTranslate(CustomTestCase):
    def test_host_bound_above_the_fill_does_not_touch_the_tail(self):
        """RED before the fix: the whole buffer was translated, so the
        uninitialized tail reached the gather and asserted. This is the shape
        that fails in practice — a speculative verify batch whose host bound
        (12) exceeds the ragged fill (5)."""
        filled = [1, 2, 3, 4, 5]
        kv_indices = _over_allocated(filled, total=12)
        kv_indptr = torch.tensor([0, len(filled)], dtype=torch.int64)

        out = _backend(_strict_translate)._translate_kv_indices_ragged(
            kv_indices, kv_indptr, bs=1, filled_len=len(filled)
        )

        # Only the written prefix comes back, translated.
        self.assertEqual(out.tolist(), [i * 2 for i in filled])

    def test_missing_host_bound_falls_back_to_the_indptr_frontier(self):
        """`filled_len=None` (no CPU mirror) must read the real frontier off
        `kv_indptr[bs]` rather than translating the allocated length."""
        filled = [7, 8, 9]
        # The gpu_only shape: sized to bs * max_context_len, 3 entries written.
        kv_indices = _over_allocated(filled, total=48)
        kv_indptr = torch.tensor([0, len(filled)], dtype=torch.int64)

        out = _backend(_strict_translate)._translate_kv_indices_ragged(
            kv_indices, kv_indptr, bs=1, filled_len=None
        )

        self.assertEqual(out.tolist(), [i * 2 for i in filled])

    def test_translation_off_returns_the_buffer_untouched(self):
        """Baseline pools and speculative DRAFT runners have no translate: the
        buffer must come back byte-identical, at its ALLOCATED length, because
        the ragged fill and the indptr the kernel reads assume that shape."""
        kv_indices = _over_allocated([1, 2], total=16)
        kv_indptr = torch.tensor([0, 2], dtype=torch.int64)

        out = _backend(None)._translate_kv_indices_ragged(
            kv_indices, kv_indptr, bs=1, filled_len=2
        )

        self.assertIs(out, kv_indices)

    def test_no_whole_buffer_translate_left_on_the_read_path(self):
        """Enforcement scan: a future edit that translates `kv_indices` whole
        would pass every test that does not exercise an over-allocated buffer,
        then kill a live speculative server. Keep the boundary explicit."""
        src = inspect.getsource(TritonAttnBackend)
        offenders = [
            line.strip()
            for line in src.splitlines()
            if "_translate_kv_loc(kv_indices)" in line
            and not line.strip().startswith("#")
        ]
        self.assertEqual(
            offenders,
            [],
            "the read path must go through _translate_kv_indices_ragged; "
            "kv_indices is over-allocated and its tail is uninitialized:\n  "
            + "\n  ".join(offenders),
        )


def _replay_backend(*, translate, kv_len=32, window=None, win_len=16):
    """A bare backend carrying what the replay zero hook touches: the
    PERSISTENT cuda-graph read buffers, poisoned with the previous replay's
    translated (kernel-facing) ids — the state a live server is always in
    after its first few replays."""
    be = object.__new__(TritonAttnBackend)
    be._translate_kv_loc = translate
    be.cuda_graph_kv_indices = torch.full((kv_len,), 15_780_610, dtype=torch.int64)
    be.sliding_window_size = window
    if window is not None:
        be.cuda_graph_window_kv_indices = torch.full(
            (win_len,), 15_780_610, dtype=torch.int64
        )
    return be


class _Batch:
    """The three fields the zero hook reads off a replay ForwardBatch."""

    def __init__(self, *, seq_lens_sum, seq_lens_cpu=None):
        self.seq_lens_sum = seq_lens_sum
        self.seq_lens_cpu = seq_lens_cpu


class TestReplayRefillZeroesTheTranslateRegion(CustomTestCase):
    """The replay refill translates [0, host bound) IN PLACE over persistent
    buffers, while the fill kernels write only the GPU-described ragged
    prefix. A speculative verify batch publishes host mirrors ABOVE that fill
    (prefix + one verify window vs the committed prefix), so without a
    pre-fill zero the tail re-translates the previous replay's already
    kernel-facing ids — out of bounds in the v2p gather, killing the server
    on the first short request that replays after a longer one finished.
    RED before the fix: the hook did not exist and nothing zeroed the tail."""

    def test_host_bound_region_is_zeroed_tail_left_alone(self):
        """The shape that fails in practice: fill will cover [0,5), the host
        bound says 13. Positions [5,13) must become the sink (0); positions
        [13:] are outside the translate region and must keep their bytes —
        this is a bounded memset, not a buffer wipe."""
        be = _replay_backend(translate=_strict_translate)
        be._zero_cuda_graph_kv_translate_region(_Batch(seq_lens_sum=13), bs=1)

        self.assertTrue(bool((be.cuda_graph_kv_indices[:13] == 0).all()))
        self.assertTrue(bool((be.cuda_graph_kv_indices[13:] == 15_780_610).all()))
        # ...and the refill's translate over the zeroed region now resolves
        # through the sink instead of asserting on the residue.
        be.cuda_graph_kv_indices[:5] = torch.arange(5)  # the fill's live prefix
        translated = _strict_translate(be.cuda_graph_kv_indices[:13])
        self.assertTrue(bool((translated[5:] == 0).all()))

    def test_window_region_zeroed_by_the_clamped_host_lengths(self):
        """The SWA window twin: its translate bound is the window-clamped sum
        of the host lengths, so the zero must cover exactly that."""
        be = _replay_backend(translate=_strict_translate, window=4)
        batch = _Batch(
            seq_lens_sum=8, seq_lens_cpu=torch.tensor([5, 3], dtype=torch.int64)
        )
        be._zero_cuda_graph_kv_translate_region(batch, bs=2)

        n_win = 4 + 3  # clamp(5, max=4) + clamp(3, max=4)
        self.assertTrue(bool((be.cuda_graph_window_kv_indices[:n_win] == 0).all()))
        self.assertTrue(
            bool((be.cuda_graph_window_kv_indices[n_win:] == 15_780_610).all())
        )

    def test_no_host_mirror_means_nothing_to_zero(self):
        """Without the host mirror the refill bounds its translate by the
        indptr — which IS the fill — so there is no gap and the hook must not
        touch the buffers (zeroing by a stale bound would be wrong)."""
        be = _replay_backend(translate=_strict_translate)
        be._zero_cuda_graph_kv_translate_region(_Batch(seq_lens_sum=None), bs=1)

        self.assertTrue(bool((be.cuda_graph_kv_indices == 15_780_610).all()))

    def test_translate_off_is_a_no_op(self):
        """Baseline pools and speculative draft runners have no translate and
        no id-space gap; their buffers must not be written."""
        be = _replay_backend(translate=None)
        be._zero_cuda_graph_kv_translate_region(_Batch(seq_lens_sum=13), bs=1)

        self.assertTrue(bool((be.cuda_graph_kv_indices == 15_780_610).all()))

    def test_zero_runs_before_every_fill(self):
        """Call-order pin: the zero is correct ONLY before the fill kernels
        write the live prefix (after, it would erase them). Both the capture
        and replay arms go through one entry, so the entry must invoke the
        hook before _apply_cuda_graph_metadata."""
        src = inspect.getsource(TritonAttnBackend.init_forward_metadata_out_graph)
        zero_at = src.find("_zero_cuda_graph_kv_translate_region")
        fill_at = src.find("_apply_cuda_graph_metadata")
        self.assertGreater(zero_at, 0, "the pre-fill zero hook is not called")
        self.assertGreater(fill_at, 0)
        self.assertLess(
            zero_at,
            fill_at,
            "the translate-region zero must precede the metadata fill",
        )


if __name__ == "__main__":
    unittest.main()
