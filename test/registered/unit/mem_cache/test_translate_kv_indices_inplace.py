"""The fused, GPU-bounded,
in-place ``translate_kv_indices_inplace`` Triton kernel.

This kernel CAPTURES the read-path v2p translate into the decode cuda-graph:
the build (`create_flashinfer_kv_indices_triton`) writes VIRTUAL ids into
``cuda_graph_kv_indices`` / ``cuda_graph_window_kv_indices`` eagerly in
replay-prep, and this kernel — recorded at the front of the captured graph —
rewrites that buffer in place to PHYSICAL ids, reading the valid extent
on-device from ``kv_indptr[bs]`` (no ``.item()``).

These tests prove the kernel:
  - matches a reference virtual->physical translate over the valid prefix
    ``[0, kv_indptr[bs])`` at ``page_size = 1`` and ``page_size > 1``;
  - leaves the stale tail ``[kv_indptr[bs], N)`` BYTE-UNTOUCHED (the
    masked-out region is never written — no over-translation);
  - clamps tombstoned (``v2p == -1``) entries to physical slot 0;
  - bounds the work by ``kv_indptr[bs]`` read on-device (varying the bound
    changes exactly which prefix is translated);
  - is a no-op for an empty buffer / zero extent;
  - exercises the grid-stride loop (num_programs < num_active_blocks).

Skipped on CPU — Triton requires a GPU.

    python -m pytest test/registered/unit/mem_cache/test_translate_kv_indices_inplace.py -v
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

_HAS_CUDA = torch.cuda.is_available()

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


def _reference_translate(
    kv_indices: torch.Tensor,
    v2p: torch.Tensor,
    total: int,
    page_size: int,
) -> torch.Tensor:
    """Oracle: the expected buffer after an in-place translate of the first
    ``total`` entries, with the tail left untouched. Mirrors
    ``MultiEndedAllocator.translate_kv_loc`` semantics (clamp_min(0))."""
    out = kv_indices.clone()
    if total == 0:
        return out
    prefix = kv_indices[:total]
    if page_size == 1:
        phys = v2p[prefix]
    else:
        page = prefix // page_size
        off = prefix % page_size
        phys = v2p[page] * page_size + off
    phys = torch.clamp_min(phys, 0)
    out[:total] = phys
    return out


@unittest.skipUnless(_HAS_CUDA, "Triton kernels require CUDA")
class TestTranslateKVIndicesInplace(unittest.TestCase):
    def _make_v2p(self, num_virtual_pages: int, *, tombstones=()):
        """Build a virtual_to_physical table (int64), sized
        ``num_virtual_pages + 1`` with a trailing ``-1`` sentinel. Maps virtual
        page p -> physical page (p + offset) so phys != virt (catches a no-op
        kernel). Selected pages are tombstoned to -1."""
        v2p = torch.empty(num_virtual_pages + 1, dtype=torch.int64, device="cuda")
        # arbitrary but deterministic bijection-ish map, kept in-range
        phys = (
            torch.arange(num_virtual_pages, device="cuda") * 7 + 3
        ) % num_virtual_pages
        v2p[:num_virtual_pages] = phys
        v2p[num_virtual_pages] = -1  # trailing sentinel
        for t in tombstones:
            v2p[t] = -1
        return v2p

    def _run(self, kv_indices, v2p, total, page_size, *, num_programs=1024, block=512):
        from sglang.srt.mem_cache.triton_ops.virtual_slot import (
            translate_kv_indices_inplace,
        )

        n_entries = kv_indices.numel()
        # kv_indptr buffer; element [bs] holds the valid extent. Use bs=3 to
        # confirm the kernel indexes the right entry (not [0] or [-1]).
        bs = 3
        kv_indptr = torch.zeros(bs + 4, dtype=torch.int32, device="cuda")
        kv_indptr[bs] = total
        buf = kv_indices.clone()
        translate_kv_indices_inplace(
            buf, v2p, kv_indptr, bs, page_size, block=block, num_programs=num_programs
        )
        return buf

    def test_ps1_parity_and_tail_untouched(self):
        page_size = 1
        num_virtual = 4096
        total = 1000
        n_entries = 4096
        v2p = self._make_v2p(num_virtual, tombstones=())
        # prefix: valid virtual token ids; tail: sentinel to detect any write.
        kv = torch.empty(n_entries, dtype=torch.int64, device="cuda")
        kv[:total] = torch.randint(0, num_virtual, (total,), device="cuda")
        kv[total:] = 123456789  # sentinel; must remain after the kernel
        got = self._run(kv, v2p, total, page_size)
        exp = _reference_translate(kv, v2p, total, page_size)
        self.assertTrue(torch.equal(got[:total], exp[:total]), "prefix mismatch")
        self.assertTrue(
            torch.equal(got[total:], kv[total:]),
            "masked tail was modified (over-translation)",
        )

    def test_ps_gt1_parity(self):
        page_size = 64
        num_virtual_pages = 256
        total = 1500
        n_entries = num_virtual_pages * page_size
        v2p = self._make_v2p(num_virtual_pages)
        kv = torch.empty(n_entries, dtype=torch.int64, device="cuda")
        # virtual token ids = page*ps + offset, in range
        kv[:total] = torch.randint(
            0, num_virtual_pages * page_size, (total,), device="cuda"
        )
        kv[total:] = 999999
        got = self._run(kv, v2p, total, page_size)
        exp = _reference_translate(kv, v2p, total, page_size)
        self.assertTrue(torch.equal(got[:total], exp[:total]), "prefix mismatch")
        self.assertTrue(torch.equal(got[total:], kv[total:]), "tail modified")

    def test_tombstone_clamped_to_zero(self):
        page_size = 1
        num_virtual = 512
        total = 64
        v2p = self._make_v2p(num_virtual, tombstones=(7, 42, 100))
        kv = torch.empty(256, dtype=torch.int64, device="cuda")
        # force some prefix entries onto tombstoned virtuals
        kv[:total] = torch.randint(0, num_virtual, (total,), device="cuda")
        kv[0] = 7
        kv[1] = 42
        kv[2] = 100
        kv[total:] = -55  # tail sentinel (also negative — must stay untouched)
        got = self._run(kv, v2p, total, page_size)
        # tombstoned virtuals -> physical 0 (clamp)
        self.assertEqual(int(got[0].item()), 0)
        self.assertEqual(int(got[1].item()), 0)
        self.assertEqual(int(got[2].item()), 0)
        # tail untouched
        self.assertTrue(torch.equal(got[total:], kv[total:]))
        # full parity vs oracle (oracle also clamps)
        exp = _reference_translate(kv, v2p, total, page_size)
        self.assertTrue(torch.equal(got[:total], exp[:total]))

    def test_bound_read_on_device(self):
        """Changing kv_indptr[bs] changes exactly which prefix is translated."""
        page_size = 1
        num_virtual = 1024
        n_entries = 2048
        v2p = self._make_v2p(num_virtual)
        base = torch.randint(0, num_virtual, (n_entries,), device="cuda").to(
            torch.int64
        )
        for total in (0, 1, 511, 512, 513, 2048):
            kv = base.clone()
            got = self._run(kv, v2p, total, page_size)
            exp = _reference_translate(kv, v2p, total, page_size)
            self.assertTrue(
                torch.equal(got, exp),
                f"bound={total}: translated prefix/tail boundary wrong",
            )

    def test_grid_stride_loop(self):
        """Small num_programs forces the GPU-side grid-stride loop to cover
        many more active blocks than there are programs."""
        page_size = 1
        num_virtual = 8192
        total = 8000
        v2p = self._make_v2p(num_virtual)
        kv = torch.randint(0, num_virtual, (8192,), device="cuda").to(torch.int64)
        # num_programs=4, block=128 -> ~63 active blocks across 4 programs
        got = self._run(kv, v2p, total, page_size, num_programs=4, block=128)
        exp = _reference_translate(kv, v2p, total, page_size)
        self.assertTrue(torch.equal(got, exp))

    def test_empty_and_zero_extent_noops(self):
        page_size = 1
        v2p = self._make_v2p(256)
        # empty buffer -> no-op (no launch)
        empty = torch.empty(0, dtype=torch.int64, device="cuda")
        self._run(empty, v2p, 0, page_size)  # must not raise
        # zero extent -> buffer unchanged
        kv = torch.randint(0, 256, (128,), device="cuda").to(torch.int64)
        got = self._run(kv, v2p, 0, page_size)
        self.assertTrue(torch.equal(got, kv))


if __name__ == "__main__":
    unittest.main()
