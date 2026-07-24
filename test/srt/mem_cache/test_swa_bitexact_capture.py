"""Component-level tests for the window-granular strict SWA HiCache capture.

Exercises the real methods with a minimal fake host pool and small CPU bf16
tensors — no GPU / full unified pool needed.

Covers:
  * capture geometry + byte identity: stride == page, one window [B-win, B) per
    page boundary keyed (rid, B); the page's first half is never captured;
    per-request offset + per-layer indexing correct.
  * binding: a node's SWA host_value is the single window at its end boundary
    (len == win), not the node's full value; falls back when the window is
    missing.
  * restore: LOAD_BACK maps only the window's (last n_tokens) full indices to
    the restored SWA slots.
"""

import types
import unittest

import torch

from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
    DeepseekV4HipRadixBackend,
)
from sglang.srt.layers.attention.dsv4.compress_hip import CompressorHip
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    BASE_COMPONENT_TYPE,
    ComponentType,
)

SWA = ComponentType.SWA


FULL = BASE_COMPONENT_TYPE


class _FakeHostPool:
    def __init__(
        self, *, win, head_dim, num_pages, layers, elem=2, device_buffers=None
    ):
        # `win` here is the host-page row count == device ring size (slot_page_size).
        self.slot_page_size = win
        self.item_bytes = win * head_dim * elem
        self.layer_num = layers
        self.data_refs = [
            torch.zeros(num_pages, self.item_bytes, dtype=torch.uint8)
            for _ in range(layers)
        ]
        # Only decode-source capture consults device_buffers; prefill leaves None.
        self.device_buffers = device_buffers
        self._capture_staging = {}
        self._num_pages = num_pages
        self._next_page = 0
        self.freed = []
        # H2 capture-done handshake probe: decode-source capture must record the
        # completion event AFTER its D2H copies so cross-stream consumers can
        # gate on it. Count calls; gpu_device None keeps the real event helper a
        # no-op if ever routed through the real pool.
        self.gpu_device = None
        self._capture_done_calls = 0

    def record_capture_done(self, stream=None):
        self._capture_done_calls += 1

    def alloc(self, n):
        assert n == self.slot_page_size
        if self._next_page >= self._num_pages:
            return None
        p = self._next_page
        self._next_page += 1
        start = p * self.slot_page_size
        return torch.arange(start, start + n, dtype=torch.int64)

    def free(self, idx):
        self.freed.append(idx)


class _FakePool:
    def __init__(self, host_pool, win, start_layer=0, ring=None):
        # `win` is the sliding window; `ring` (>= win) is the device ring size,
        # which carries speculative slack (ring = window + spec_extra). Default
        # ring == win reproduces the no-speculation geometry.
        self._swa_host_pool = host_pool
        self.unified_swa_ring_size = ring if ring is not None else win
        self.unified_swa_window = win
        self.start_layer = start_layer


def _fake_fb(ext, seqs, rids):
    return types.SimpleNamespace(
        forward_mode=types.SimpleNamespace(is_extend=lambda: True),
        extend_seq_lens_cpu=ext,
        seq_lens_cpu=torch.tensor(seqs),
        req_pool_indices=torch.tensor(rids),
    )


def _cd(value=None, host_value=None):
    return types.SimpleNamespace(value=value, host_value=host_value)


class TestSwaCaptureGeometry(unittest.TestCase):
    """Stride == page, one [B-win, B) window per boundary, first half never
    captured, per-request offset + per-layer bytes correct."""

    def _run(self):
        win, page, head_dim, layers = 2, 4, 4, 3
        host = _FakeHostPool(win=win, head_dim=head_dim, num_pages=8, layers=layers)
        pool = _FakePool(host, win)
        be = types.SimpleNamespace(token_to_kv_pool=pool, page_size=page)
        # batch: req0 = positions [0,8) (cs=0,e=8); req1 = [8,12) (cs=8,e=4)
        ext = [8, 4]
        seqs = [8, 12]
        rids = [5, 7]
        fb = _fake_fb(ext, seqs, rids)
        total_rows = sum(ext)
        # distinct bf16 bytes per (layer, row): base value differs per layer.
        kv_by_layer = [
            (
                torch.arange(total_rows * head_dim, dtype=torch.float32).reshape(
                    total_rows, head_dim
                )
                + L * 10000.0
            ).to(torch.bfloat16)
            for L in range(layers)
        ]
        for L in range(layers):
            DeepseekV4HipRadixBackend.capture_swa_windows(be, L, kv_by_layer[L], fb)
        return win, head_dim, layers, host, kv_by_layer

    def test_keys_are_second_half_windows_only(self):
        win, head_dim, layers, host, kv = self._run()
        # boundaries: req0 -> 4,8 ; req1 -> 12. windows [2,4),[6,8),[10,12).
        self.assertEqual(set(host._capture_staging.keys()), {(5, 4), (5, 8), (7, 12)})
        # exactly 3 host pages allocated (no first-half tiles).
        self.assertEqual(host._next_page, 3)

    def test_bytes_match_flat_kv_window_per_layer(self):
        win, head_dim, layers, host, kv = self._run()
        # (key -> (page_row, batch_row_start)) ; batch rows are absolute in kv.
        expected = {
            (5, 4): (0, 2),  # window [2,4)
            (5, 8): (1, 6),  # window [6,8)
            (7, 12): (2, 10),  # window [10,12)
        }
        for key, (page_row, r0) in expected.items():
            self.assertIn(key, host._capture_staging)
            got_page = int(host._capture_staging[key][0].item()) // win
            self.assertEqual(got_page, page_row)
            for L in range(layers):
                want = kv[L][r0 : r0 + win].contiguous().view(torch.uint8).reshape(-1)
                got = host.data_refs[L][page_row]
                self.assertTrue(
                    torch.equal(got, want),
                    f"layer {L} key {key}: bytes differ",
                )

    def test_flag_off_is_noop(self):
        # No _swa_host_pool => capture short-circuits.
        pool = types.SimpleNamespace(_swa_host_pool=None)
        be = types.SimpleNamespace(token_to_kv_pool=pool, page_size=4)
        fb = _fake_fb([4], [4], [0])
        # Must not raise / touch anything.
        DeepseekV4HipRadixBackend.capture_swa_windows(
            be, 0, torch.zeros(4, 4, dtype=torch.bfloat16), fb
        )


def _fake_fb_decode(seqs, rids):
    return types.SimpleNamespace(
        forward_mode=types.SimpleNamespace(
            is_decode_or_idle=lambda: True, is_extend=lambda: False
        ),
        seq_lens_cpu=torch.tensor(seqs),
        req_pool_indices=torch.tensor(rids),
        batch_size=len(seqs),
    )


class TestSwaSpecRingSlackCapture(unittest.TestCase):
    """Speculative decoding grows the device SWA ring by `spec_extra` slack rows
    (ring_size = window + spec_extra), so ``page % ring`` is no longer 0 and the
    sliding window is stored ROTATED inside the ring block (token p at row
    p % ring), not in token order. Capture must reproduce that ring layout so
    the positional H->D restore lands byte-exact."""

    @staticmethod
    def _sim_ring_block(kv_layer, B, ring):
        # Ground-truth device ring block after writing tokens [0, B): row p%ring
        # holds the most recent token p < B with that residue (== the trailing
        # `ring` tokens placed positionally by p % ring). Independent of the
        # implementation's roll, this is what attention reads via slot*ring+p%ring.
        head_dim = kv_layer.shape[1]
        block = torch.zeros(ring, head_dim, dtype=kv_layer.dtype)
        for p in range(B):
            block[p % ring] = kv_layer[p]
        return block.contiguous().view(torch.uint8).reshape(-1)

    def _kv(self, rows, head_dim, layers):
        return [
            (
                torch.arange(rows * head_dim, dtype=torch.float32).reshape(
                    rows, head_dim
                )
                + L * 10000.0
            ).to(torch.bfloat16)
            for L in range(layers)
        ]

    def test_prefill_capture_matches_ring_layout(self):
        # window=3, spec_extra=2 => ring=5; page=6 => page % ring == 1 (nonzero),
        # boundaries B=6 (h=1) and B=12 (h=2) both exercise a nonzero roll.
        window, ring, page, head_dim, layers = 3, 5, 6, 4, 2
        host = _FakeHostPool(win=ring, head_dim=head_dim, num_pages=8, layers=layers)
        pool = _FakePool(host, window, ring=ring)
        be = types.SimpleNamespace(token_to_kv_pool=pool, page_size=page)
        ext, seqs, rids = [12], [12], [0]
        fb = _fake_fb(ext, seqs, rids)
        kv_by_layer = self._kv(sum(ext), head_dim, layers)
        for L in range(layers):
            DeepseekV4HipRadixBackend.capture_swa_windows(be, L, kv_by_layer[L], fb)
        self.assertEqual(set(host._capture_staging.keys()), {(0, 6), (0, 12)})
        for rid, B in [(0, 6), (0, 12)]:
            page_row = int(host._capture_staging[(rid, B)][0].item()) // ring
            for L in range(layers):
                want = self._sim_ring_block(kv_by_layer[L], B, ring)
                got = host.data_refs[L][page_row]
                self.assertTrue(
                    torch.equal(got, want),
                    f"prefill ring-layout mismatch layer={L} B={B}",
                )

    def test_decode_capture_copies_full_ring_block(self):
        # Decode reads the live per-request ring block straight off the device
        # and copies all `ring` rows positionally -- byte-identical to prefill's
        # reconstruction and to what restore writes back.
        window, ring, page, head_dim, layers = 3, 5, 6, 4, 2
        num_slots = 4
        device_buffers = [
            (
                torch.arange(num_slots * ring * head_dim, dtype=torch.float32).reshape(
                    num_slots, ring, head_dim
                )
                + L * 10000.0
            ).to(torch.bfloat16)
            for L in range(layers)
        ]
        host = _FakeHostPool(
            win=ring,
            head_dim=head_dim,
            num_pages=8,
            layers=layers,
            device_buffers=device_buffers,
        )
        pool = _FakePool(host, window, ring=ring)
        be = types.SimpleNamespace(token_to_kv_pool=pool, page_size=page)
        rid = 2
        fb = _fake_fb_decode([12], [rid])  # total=12, 12 % page(6) == 0 boundary
        DeepseekV4HipRadixBackend.capture_swa_windows_decode(be, fb)
        self.assertIn((rid, 12), host._capture_staging)
        page_row = int(host._capture_staging[(rid, 12)][0].item()) // ring
        for L in range(layers):
            want = device_buffers[L][rid].contiguous().view(torch.uint8).reshape(-1)
            got = host.data_refs[L][page_row]
            self.assertTrue(
                torch.equal(got, want), f"decode ring-block mismatch layer={L}"
            )

    def test_no_spec_is_identity_token_order(self):
        # ring == window, page % ring == 0: roll degenerates to identity, so the
        # captured page is the token-order window (legacy behavior preserved).
        window = ring = 2
        page, head_dim, layers = 4, 4, 2
        host = _FakeHostPool(win=ring, head_dim=head_dim, num_pages=8, layers=layers)
        pool = _FakePool(host, window, ring=ring)
        be = types.SimpleNamespace(token_to_kv_pool=pool, page_size=page)
        fb = _fake_fb([8], [8], [0])
        kv = self._kv(8, head_dim, layers)
        for L in range(layers):
            DeepseekV4HipRadixBackend.capture_swa_windows(be, L, kv[L], fb)
        for B, r0 in [(4, 2), (8, 6)]:
            page_row = int(host._capture_staging[(0, B)][0].item()) // ring
            for L in range(layers):
                want = kv[L][r0 : r0 + ring].contiguous().view(torch.uint8).reshape(-1)
                self.assertTrue(torch.equal(host.data_refs[L][page_row], want))


class TestSwaStrideCapture(unittest.TestCase):
    """Task A2: capture keeps one window every ``stride`` pages PLUS the true
    sequence tail page (last prefill chunk, detected via
    ``orig_seq_lens == cumulative seq_len``). Intermediate chunk-tops are NOT
    captured, so coarse strides genuinely shrink the offload set."""

    def _capture(
        self,
        *,
        stride,
        ext,
        seqs,
        orig,
        rids,
        page=4,
        win=2,
        head_dim=4,
        layers=2,
        host=None,
    ):
        if host is None:
            host = _FakeHostPool(
                win=win, head_dim=head_dim, num_pages=64, layers=layers
            )
        pool = _FakePool(host, win)
        pool._swa_offload_page_stride = stride
        be = types.SimpleNamespace(token_to_kv_pool=pool, page_size=page)
        fb = types.SimpleNamespace(
            forward_mode=types.SimpleNamespace(is_extend=lambda: True),
            extend_seq_lens_cpu=ext,
            seq_lens_cpu=torch.tensor(seqs),
            req_pool_indices=torch.tensor(rids),
            orig_seq_lens=torch.tensor(orig),
        )
        total_rows = sum(ext)
        for L in range(layers):
            kv = (
                torch.arange(total_rows * head_dim, dtype=torch.float32).reshape(
                    total_rows, head_dim
                )
                + L * 1000.0
            ).to(torch.bfloat16)
            DeepseekV4HipRadixBackend.capture_swa_windows(be, L, kv, fb)
        return host

    def test_stride1_matches_legacy_every_page(self):
        host = self._capture(stride=1, ext=[16], seqs=[16], orig=[16], rids=[5])
        self.assertEqual(
            set(host._capture_staging.keys()),
            {(5, 4), (5, 8), (5, 12), (5, 16)},
        )

    def test_stride2_keeps_strided_pages_and_tail(self):
        # page_idx 2,4 -> B=8,16 kept; tail B==16 already among them.
        host = self._capture(stride=2, ext=[16], seqs=[16], orig=[16], rids=[5])
        self.assertEqual(set(host._capture_staging.keys()), {(5, 8), (5, 16)})

    def test_tail_forced_even_when_not_stride_multiple(self):
        # boundary=24 (page_idx 6); stride=4 keeps only page_idx 4 (B=16);
        # the tail forces B=24 even though 6 % 4 != 0.
        host = self._capture(stride=4, ext=[24], seqs=[24], orig=[24], rids=[5])
        self.assertEqual(set(host._capture_staging.keys()), {(5, 16), (5, 24)})

    def test_intermediate_chunk_top_not_captured(self):
        # A length-24 prompt processed in two 12-token chunks, stride=4. The
        # first chunk's top (B=12) is NOT a stride boundary and NOT the last
        # chunk -> nothing captured. This is the crux: coarse stride must not
        # degrade into "one window per chunk".
        host = _FakeHostPool(win=2, head_dim=4, num_pages=64, layers=2)
        self._capture(stride=4, ext=[12], seqs=[12], orig=[24], rids=[5], host=host)
        self.assertEqual(
            set(host._capture_staging.keys()),
            set(),
            "intermediate chunk-top must not be captured for coarse stride",
        )
        # last chunk (orig == cumulative == 24): stride boundary B=16 + tail B=24.
        self._capture(stride=4, ext=[12], seqs=[24], orig=[24], rids=[5], host=host)
        self.assertEqual(set(host._capture_staging.keys()), {(5, 16), (5, 24)})

    def test_tail_in_non_last_chunk_is_captured(self):
        # Corner case (A2 follow-up): the TRUE sequence tail page boundary can
        # land in a NON-last chunk when the final chunk is a sub-page remainder.
        # orig=10, page=4 -> tail_B=8 is reached at the top of chunk 1 (tokens
        # 0..8); chunk 2 is only tokens 8..10 (< page), whose loop starts above
        # tail_B and never revisits it. With stride=4, page_idx 2 (B=8) is not a
        # stride multiple, so the old (is_last_chunk and B==boundary) gate
        # misses the tail entirely. Matching tail_B directly must capture it.
        host = _FakeHostPool(win=2, head_dim=4, num_pages=64, layers=2)
        self._capture(stride=4, ext=[8], seqs=[8], orig=[10], rids=[5], host=host)
        self._capture(stride=4, ext=[2], seqs=[10], orig=[10], rids=[5], host=host)
        self.assertIn(
            (5, 8),
            host._capture_staging.keys(),
            "true sequence tail page must be captured even in a non-last chunk",
        )

    def test_straddling_prefix_boundary_skipped(self):
        # Prefix-cache reuse can leave a NON-page-aligned flat-KV start `cs`.
        # page=4, win=2: a chunk [cs=7, 12) has its first page boundary at B=8,
        # whose trailing window [6, 8) reaches back to token 6 -- before cs=7 and
        # thus NOT present in this chunk's flat `kv`. That boundary must be
        # SKIPPED (not asserted/crashed); a later reuse recomputes it. The true
        # tail B=12 (fully inside the chunk) is still captured.
        host = self._capture(stride=1, ext=[5], seqs=[12], orig=[12], rids=[5])
        self.assertEqual(
            set(host._capture_staging.keys()),
            {(5, 12)},
            "straddling boundary B=8 must be skipped; only tail B=12 captured",
        )


class TestSwaHostSizingByStride(unittest.TestCase):
    """Task A3: SWA host pool is sized as ceil(full_host_pages / stride) + tail,
    floored at the device ring."""

    def _pages(self, stride, full_host_pages=8000, device_ring_pages=64):
        import types

        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as A

        sa = types.SimpleNamespace(hicache_swa_offload_page_stride=stride)
        return A._swa_host_num_pages(
            server_args=sa,
            full_host_pages=full_host_pages,
            device_ring_pages=device_ring_pages,
            page_bytes=131072,
            page_size=256,
        )

    def test_stride1_covers_all_full_pages(self):
        # per-page: at least ceil(8000/1) windows (bounded below by device ring).
        self.assertGreaterEqual(self._pages(1), 8000)

    def test_stride8_is_roughly_one_eighth(self):
        p = self._pages(8)
        self.assertGreaterEqual(p, 8000 // 8)  # ~1000
        self.assertLess(p, 8000 // 8 + 200)  # + small tail allowance

    def test_floored_at_device_ring(self):
        # tiny full pool + huge stride -> still at least the device ring.
        self.assertGreaterEqual(self._pages(1024, full_host_pages=10), 64)


class TestSwaStrideArg(unittest.TestCase):
    """Task A1: --hicache-swa-offload-page-stride parses and defaults to 1.

    ServerArgs.__post_init__ eagerly resolves the model config (network I/O for
    a bogus model path), so we validate the dataclass field default and the CLI
    wiring directly instead of constructing a full ServerArgs.
    """

    def test_default_is_one(self):
        from sglang.srt.server_args import ServerArgs

        self.assertEqual(
            ServerArgs.__dataclass_fields__["hicache_swa_offload_page_stride"].default,
            1,
        )

    def test_cli_default_is_one(self):
        import argparse

        from sglang.srt.server_args import ServerArgs

        p = argparse.ArgumentParser()
        ServerArgs.add_cli_args(p)
        ns = p.parse_args(["--model-path", "x"])
        self.assertEqual(ns.hicache_swa_offload_page_stride, 1)

    def test_parses_custom_value(self):
        import argparse

        from sglang.srt.server_args import ServerArgs

        p = argparse.ArgumentParser()
        ServerArgs.add_cli_args(p)
        ns = p.parse_args(
            ["--model-path", "x", "--hicache-swa-offload-page-stride", "8"]
        )
        self.assertEqual(ns.hicache_swa_offload_page_stride, 8)


class _FakeDecodePool:
    def __init__(self, host_pool, win):
        self._swa_host_pool = host_pool
        self.unified_swa_ring_size = win
        self.unified_swa_window = win
        self.start_layer = 0


class TestSwaDecodeCapture(unittest.TestCase):
    """P.2 decode-source capture: at a page boundary crossed DURING decode,
    snapshot the per-request ring block into a host window page, byte-identical
    to the prefill capture layout (completes the dual-source model)."""

    WIN = 2
    HEAD_DIM = 4
    LAYERS = 2
    PAGE = 4  # page % ring(WIN) == 0
    NSLOTS = 8

    def _host(self):
        hp = _FakeHostPool(
            win=self.WIN, head_dim=self.HEAD_DIM, num_pages=64, layers=self.LAYERS
        )
        hp.layer_num = self.LAYERS
        # device SWA ring: one [win, head_dim] bf16 block per request slot/layer
        hp.device_buffers = [
            torch.zeros(self.NSLOTS, self.WIN, self.HEAD_DIM, dtype=torch.bfloat16)
            for _ in range(self.LAYERS)
        ]
        hp._capture_crc = {}
        return hp

    def _fill_ring(self, hp, rid):
        for li in range(self.LAYERS):
            block = (
                torch.arange(self.WIN * self.HEAD_DIM).reshape(self.WIN, self.HEAD_DIM)
                + li * 100
                + rid
            ).to(torch.bfloat16)
            hp.device_buffers[li][rid] = block

    def _fb(self, seqs, rids):
        return types.SimpleNamespace(
            forward_mode=types.SimpleNamespace(
                is_decode_or_idle=lambda: True, is_extend=lambda: False
            ),
            seq_lens_cpu=torch.tensor(seqs),
            req_pool_indices=torch.tensor(rids),
        )

    def _run(self, hp, seqs, rids, stride=1):
        pool = _FakeDecodePool(hp, self.WIN)
        pool._swa_offload_page_stride = stride
        be = types.SimpleNamespace(token_to_kv_pool=pool, page_size=self.PAGE)
        DeepseekV4HipRadixBackend.capture_swa_windows_decode(be, self._fb(seqs, rids))

    def test_boundary_snapshots_ring_block_byte_identical(self):
        hp = self._host()
        rid = 5
        self._fill_ring(hp, rid)
        # post-step seq_len 8 is a page multiple -> boundary B=8 captured
        self._run(hp, seqs=[8], rids=[rid])
        self.assertIn((rid, 8), hp._capture_staging)
        page_row = int(hp._capture_staging[(rid, 8)][0].item()) // hp.slot_page_size
        for li in range(self.LAYERS):
            expect = (
                hp.device_buffers[li][rid].contiguous().view(torch.uint8).reshape(-1)
            )
            got = hp.data_refs[li][page_row]
            self.assertTrue(
                bool((got == expect).all()),
                f"layer {li} host page bytes != ring block bytes",
            )

    def test_non_boundary_seqlen_not_captured(self):
        hp = self._host()
        self._fill_ring(hp, 5)
        self._run(hp, seqs=[7], rids=[5])  # 7 % 4 != 0
        self.assertEqual(hp._capture_staging, {})

    def test_zero_seqlen_ignored(self):
        hp = self._host()
        self._run(hp, seqs=[0], rids=[3])
        self.assertEqual(hp._capture_staging, {})

    def test_already_staged_key_not_overwritten(self):
        hp = self._host()
        self._fill_ring(hp, 5)
        # pretend prefill already produced this window
        sentinel = torch.arange(hp.slot_page_size, dtype=torch.int64)
        hp._capture_staging[(5, 8)] = sentinel
        next_before = hp._next_page
        self._run(hp, seqs=[8], rids=[5])
        self.assertIs(hp._capture_staging[(5, 8)], sentinel)  # untouched
        self.assertEqual(hp._next_page, next_before)  # no new host alloc

    def test_multi_request_only_boundary_reqs_captured(self):
        hp = self._host()
        for r in (2, 4, 6):
            self._fill_ring(hp, r)
        # r=2 at boundary(8), r=4 not(5), r=6 at boundary(4)
        self._run(hp, seqs=[8, 5, 4], rids=[2, 4, 6])
        self.assertIn((2, 8), hp._capture_staging)
        self.assertIn((6, 4), hp._capture_staging)
        self.assertNotIn((4, 5), hp._capture_staging)

    def test_stride_gate_skips_non_aligned_boundaries(self):
        hp = self._host()
        for r in (1, 2):
            self._fill_ring(hp, r)
        # PAGE=4, stride=2 -> keep boundaries where (B//page) % 2 == 0:
        # B=8 -> 8//4=2, 2%2==0 keep; B=4 -> 4//4=1, 1%2==1 skip.
        self._run(hp, seqs=[8], rids=[1], stride=2)
        self._run(hp, seqs=[4], rids=[2], stride=2)
        self.assertIn((1, 8), hp._capture_staging)
        self.assertNotIn((2, 4), hp._capture_staging)

    def test_stride_one_keeps_every_boundary(self):
        hp = self._host()
        for r in (1, 2):
            self._fill_ring(hp, r)
        self._run(hp, seqs=[4], rids=[1], stride=1)
        self._run(hp, seqs=[8], rids=[2], stride=1)
        self.assertIn((1, 4), hp._capture_staging)
        self.assertIn((2, 8), hp._capture_staging)

    def test_padded_entries_beyond_batch_size_ignored(self):
        # cuda-graph decode pads seq_lens/req_pool_indices to a static size; rows
        # >= batch_size are padding and must not stage a window even if they land
        # on a page boundary.
        hp = self._host()
        for r in (1, 2):
            self._fill_ring(hp, r)
        pool = _FakeDecodePool(hp, self.WIN)
        pool._swa_offload_page_stride = 1
        be = types.SimpleNamespace(token_to_kv_pool=pool, page_size=self.PAGE)
        fb = self._fb(seqs=[5, 8], rids=[1, 2])  # row0 non-boundary, row1 boundary
        fb.batch_size = 1  # only row0 is a real request
        DeepseekV4HipRadixBackend.capture_swa_windows_decode(be, fb)
        self.assertEqual(hp._capture_staging, {})

    def test_no_host_pool_is_noop(self):
        pool = types.SimpleNamespace(_swa_host_pool=None)
        be = types.SimpleNamespace(token_to_kv_pool=pool, page_size=self.PAGE)
        # must not raise
        DeepseekV4HipRadixBackend.capture_swa_windows_decode(be, self._fb([8], [5]))

    def test_boundary_records_capture_done_event(self):
        # H2 (overlap safety): after staging a decode window with a non_blocking
        # D2H on the compute/forward stream, capture MUST record the
        # capture-completion event so a later cross-stream consumer (restore /
        # L3 backup / device landing) can gate on it via wait_capture_done().
        # Without this record the wait is a silent no-op and correctness would
        # rely on the consumer happening to run on the same stream.
        hp = self._host()
        self._fill_ring(hp, 5)
        self._run(hp, seqs=[8], rids=[5])
        self.assertIn((5, 8), hp._capture_staging)
        self.assertEqual(
            hp._capture_done_calls, 1, "capture-done event not recorded after D2H"
        )

    def test_no_boundary_does_not_record_capture_done(self):
        # No window captured => no D2H enqueued => nothing to gate: do not create
        # /record an event on every no-op decode step (the common case).
        hp = self._host()
        self._fill_ring(hp, 5)
        self._run(hp, seqs=[7], rids=[5])  # 7 % 4 != 0, no boundary
        self.assertEqual(hp._capture_staging, {})
        self.assertEqual(hp._capture_done_calls, 0)

    def test_capture_done_recorded_once_per_step(self):
        # Multiple requests crossing a boundary in the same step stage several
        # pages but the completion event is recorded ONCE, after all copies.
        hp = self._host()
        for r in (2, 6):
            self._fill_ring(hp, r)
        self._run(hp, seqs=[8, 4], rids=[2, 6])  # both on boundaries
        self.assertIn((2, 8), hp._capture_staging)
        self.assertIn((6, 4), hp._capture_staging)
        self.assertEqual(hp._capture_done_calls, 1)


class TestHostPoolUpperBoundGuard(unittest.TestCase):
    """Task B3: startup budget guard for the SWA offload host pool.

    Two tiers, no silent clamp: projection in [slow, hard) only warns and
    proceeds (the "fake hang"/slow-pinning tier); projection >= hard fails fast.
    The hard ceiling is DRAM-derived (available_dram * 0.9 / ranks_per_node), so
    it fires only when the pool would physically exhaust host DRAM (real OOM),
    and the message suggests only --hicache-swa-offload-page-stride. Scope is the
    SWA host pool only -- FP4 c4/c128 was just the trigger that surfaced this,
    not the root cause, so it is intentionally out of scope here."""

    def _check(self, **kw):
        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as A

        defaults = dict(
            slow_gb=64.0,
            hard_gb=128.0,
            full_host_pages=1_000_000,
            stride=1,
            page_bytes=1 << 20,
        )
        defaults.update(kw)
        return A._check_swa_host_pool_upper_bound(**defaults)

    def test_oversized_swa_host_pool_raises_with_knob_suggestion(self):
        with self.assertRaises(ValueError) as ctx:
            self._check(swa_gb=512.0)
        msg = str(ctx.exception)
        # stride is the only shrink knob; the offending size is surfaced.
        self.assertIn("--hicache-swa-offload-page-stride", msg)
        # --max-total-tokens is deliberately NOT suggested (wrong lever).
        self.assertNotIn("--max-total-tokens", msg)
        self.assertIn("512", msg)

    def test_slow_launch_warns_but_does_not_raise(self):
        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as A

        with self.assertLogs(A.logger, level="WARNING") as cm:
            self._check(swa_gb=96.0)
        self.assertTrue(any("SWA host" in m for m in cm.output))

    def test_under_slow_threshold_is_silent_noop(self):
        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as A

        with self.assertNoLogs(A.logger, level="WARNING"):
            self._check(swa_gb=1.0)

    def test_hard_message_includes_dram_context_when_provided(self):
        with self.assertRaises(ValueError) as ctx:
            self._check(swa_gb=512.0, hard_gb=300.0, avail_gb=3000.0, ranks_per_node=8)
        msg = str(ctx.exception)
        # aggregate footprint across ranks and the available DRAM are surfaced.
        self.assertIn("8 rank", msg)
        self.assertIn("3000 GB available", msg)
        self.assertIn("OOM", msg)


class TestSwaHostHardLimitFromDram(unittest.TestCase):
    """Task B3 (DRAM-aware): the hard fail-fast ceiling is derived from real
    host DRAM -- available_dram * fraction / ranks_per_node -- not a fixed
    constant, so it fires only on a genuine host-OOM projection."""

    def _mod(self):
        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as A

        return A

    def test_hard_limit_is_available_times_fraction_over_ranks(self):
        import types
        from unittest import mock

        A = self._mod()
        sa = types.SimpleNamespace(tp_size=8, nnodes=1)
        vm = types.SimpleNamespace(available=3000 * 1e9)  # 3000 GB available
        with mock.patch("psutil.virtual_memory", return_value=vm):
            hard_gb, avail_gb, ranks = A._swa_host_hard_limit_gb(sa)
        self.assertEqual(ranks, 8)  # tp_size // nnodes
        self.assertAlmostEqual(avail_gb, 3000.0, places=3)
        # 3000 * 0.9 / 8 = 337.5 GB/rank
        self.assertAlmostEqual(hard_gb, 3000.0 * 0.9 / 8, places=3)

    def test_ranks_per_node_uses_tp_over_nnodes(self):
        import types
        from unittest import mock

        A = self._mod()
        sa = types.SimpleNamespace(tp_size=16, nnodes=2)
        vm = types.SimpleNamespace(available=2000 * 1e9)
        with mock.patch("psutil.virtual_memory", return_value=vm):
            hard_gb, _, ranks = A._swa_host_hard_limit_gb(sa)
        self.assertEqual(ranks, 8)  # 16 // 2
        self.assertAlmostEqual(hard_gb, 2000.0 * 0.9 / 8, places=3)

    def test_sizing_fails_fast_when_pool_would_exhaust_dram(self):
        # per-page (stride=1), DSv4-Pro geometry (~299 GB/rank) on a SMALLER
        # 2 TB node: aggregate pin across 8 ranks (~2.4 TB) blows past 90% of
        # available DRAM (1.8 TB) -> genuine host OOM -> fail fast.
        import types
        from unittest import mock

        A = self._mod()
        sa = types.SimpleNamespace(
            hicache_swa_offload_page_stride=1, tp_size=8, nnodes=1
        )
        vm = types.SimpleNamespace(available=2000 * 1e9)
        with mock.patch("psutil.virtual_memory", return_value=vm):
            with self.assertRaises(ValueError) as ctx:
                A._swa_host_num_pages(
                    server_args=sa,
                    full_host_pages=32_000,
                    device_ring_pages=1024,
                    page_bytes=9_054_720,  # ~9.05 MB/page (DSv4-Pro SWA window)
                    page_size=256,
                )
        self.assertIn("--hicache-swa-offload-page-stride", str(ctx.exception))

    def test_per_page_on_large_node_warns_not_fatal(self):
        # Same ~299 GB/rank per-page projection, but on the real 3 TB node the
        # aggregate (~2.4 TB) stays under 90% of available (2.7 TB), so the hard
        # tier does NOT fire -- it only warns (slow pinning), never raises. This
        # pins the intended semantics: the hard tier is a real-OOM guard, not a
        # slow-launch guard.
        import types
        from unittest import mock

        A = self._mod()
        sa = types.SimpleNamespace(
            hicache_swa_offload_page_stride=1, tp_size=8, nnodes=1
        )
        vm = types.SimpleNamespace(available=3000 * 1e9)
        with mock.patch("psutil.virtual_memory", return_value=vm):
            with self.assertLogs(A.logger, level="WARNING") as cm:
                pages = A._swa_host_num_pages(
                    server_args=sa,
                    full_host_pages=32_000,
                    device_ring_pages=1024,
                    page_bytes=9_054_720,
                    page_size=256,
                )
        self.assertGreater(pages, 0)
        self.assertTrue(any("SWA host pool" in m for m in cm.output))

    def test_sizing_ok_when_stride_shrinks_pool_within_dram(self):
        import types
        from unittest import mock

        A = self._mod()
        sa = types.SimpleNamespace(
            hicache_swa_offload_page_stride=64, tp_size=8, nnodes=1
        )
        vm = types.SimpleNamespace(available=3000 * 1e9)
        with mock.patch("psutil.virtual_memory", return_value=vm):
            pages = A._swa_host_num_pages(
                server_args=sa,
                full_host_pages=32_000,
                device_ring_pages=1024,
                page_bytes=9_054_720,
                page_size=256,
            )
        # stride=64 keeps the pool well under the DRAM ceiling -> no raise.
        self.assertGreater(pages, 0)


class _IKey:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


class _INode:
    def __init__(self, keylen):
        self.key = _IKey(keylen)
        self.component_data = {SWA: _cd(value=None, host_value=None)}


class _FakeHostLRU:
    def __init__(self):
        self._nodes = set()

    def in_list(self, node):
        return id(node) in self._nodes

    def insert_mru(self, node):
        self._nodes.add(id(node))

    def remove_node(self, node):
        self._nodes.discard(id(node))


# ---------------------------------------------------------------------------
# SWA-capture completion-event handshake (hazard H2), merged from
# test_swa_capture_done.py. The capture D2H runs on the compute/forward stream
# (same-stream-ordered vs the ring write/overwrite, hazard H1); a CROSS-stream
# consumer (restore H2D, L3 write-through, device-landing check) must order
# strictly after the page is fully written via record_capture_done /
# wait_capture_done.
# ---------------------------------------------------------------------------


def _bare_pool(gpu_device):
    """A DeepSeekV4PagedHostPool with only the attributes the capture-done event
    facility touches, avoiding the heavy host-memory allocation of __init__."""
    p = object.__new__(DeepSeekV4PagedHostPool)
    p.gpu_device = gpu_device
    p._capture_done_event = None
    return p


class TestCaptureDoneEventCPU(unittest.TestCase):
    def test_cpu_device_record_is_noop(self):
        p = _bare_pool("cpu")
        p.record_capture_done()
        self.assertIsNone(p._capture_done_event)
        p.wait_capture_done()  # must not raise

    def test_none_device_is_noop(self):
        p = _bare_pool(None)
        p.record_capture_done()
        self.assertIsNone(p._capture_done_event)
        p.wait_capture_done()


@unittest.skipUnless(torch.cuda.is_available(), "needs GPU")
class TestCaptureDoneEventGPU(unittest.TestCase):
    """The capture keeps its D2H on the compute/forward stream (no side stream);
    the completion event is created lazily and gates a cross-stream consumer."""

    def test_records_event_and_orders_cross_stream_consumer(self):
        p = _bare_pool(torch.device("cuda", 0))
        self.assertIsNone(p._capture_done_event)
        # emulate capture: non_blocking D2H on the current stream, then record.
        src = torch.ones(8, device="cuda")
        dst = torch.empty(8, device="cpu", pin_memory=True)
        dst.copy_(src, non_blocking=True)
        p.record_capture_done()
        self.assertIsInstance(p._capture_done_event, torch.cuda.Event)
        # a consumer on a DIFFERENT stream is ordered strictly after the D2H.
        consumer = torch.cuda.Stream()
        p.wait_capture_done(consumer)
        consumer.synchronize()
        self.assertTrue(bool((dst == 1).all().item()))

    def test_event_is_reused_across_calls(self):
        p = _bare_pool(torch.device("cuda", 0))
        p.record_capture_done()
        first = p._capture_done_event
        p.record_capture_done()
        self.assertIs(p._capture_done_event, first)

    def test_wait_before_any_capture_is_noop(self):
        p = _bare_pool(torch.device("cuda", 0))
        # no capture recorded yet -> no event -> wait is a no-op, must not raise.
        p.wait_capture_done()
        self.assertIsNone(p._capture_done_event)


# ---------------------------------------------------------------------------
# Strict SWA-HiCache geometry regression + FlexKV coupling contract, merged from
# test_swa_state_geometry_fixes.py.
#   * off0=0 packing decouples the host state tile from the spec-padded SWA ring,
#     so state capture stays byte-exact even when swa_ring % ring_size != 0 and
#     page % swa_ring != 0 (spec-decode geometry: window + spec_extra = 131).
#   * a reuse window reaching back into the overlap prefix (B - win < cs) is
#     captured when the prefix is present (buf_lo >= 0) and never crashes.
#   * get_swa_state_coupling_infos exposes (swa_page_size, ring_size) per sidecar
#     state pool in the same order / filtering as get_state_buf_infos.
# ---------------------------------------------------------------------------

_CAPTURE = CompressorHip._capture_compress_state_windows


def _state_host_pool(*, ring_size, slot_bytes, num_pages=16):
    item_bytes = ring_size * slot_bytes
    host_buf = torch.zeros((num_pages, item_bytes), dtype=torch.uint8)
    counter = {"n": 0}

    def alloc(need):
        assert need == ring_size
        p = counter["n"]
        counter["n"] += 1
        if p >= num_pages:
            return None
        return torch.arange(p * ring_size, p * ring_size + ring_size)

    return types.SimpleNamespace(
        slot_page_size=ring_size,
        item_bytes=item_bytes,
        data_refs=[host_buf],
        _capture_staging={},
        _capture_state_crc={},
        alloc=alloc,
    )


def _state_backend(host_pool, *, page, swa_ring):
    pool = types.SimpleNamespace(
        unified_swa_ring_size=swa_ring,
        _c4_state_layer_index={0: 0},
        _c4_state_host_pool=host_pool,
    )
    return types.SimpleNamespace(token_to_kv_pool=pool, page_size=page)


def _state_self(ratio=4):
    return types.SimpleNamespace(ratio=ratio, is_in_indexer=False, layer_id=0)


class TestGeometryDecoupleEagle(unittest.TestCase):
    def test_capture_nondividing_swa_ring_byte_exact(self):
        # EAGLE: window=128, spec_extra=3 -> swa_ring=131; page=256; ring_size=8.
        # 131 % 8 = 3 and 256 % 131 = 125 -> the old geometry asserts crashed here.
        ring_size, ratio, last_dim, page, swa_ring = 8, 4, 16, 256, 131
        slot_bytes = last_dim * torch.tensor([], dtype=torch.bfloat16).element_size()
        hp = _state_host_pool(ring_size=ring_size, slot_bytes=slot_bytes)
        be = _state_backend(hp, page=page, swa_ring=swa_ring)
        ext = 512
        buf = types.SimpleNamespace(
            kv_score=torch.randint(0, 255, (ext, last_dim), dtype=torch.int32).to(
                torch.bfloat16
            )
        )
        _CAPTURE(
            _state_self(ratio),
            kv_and_score_buffer=buf,
            valid_kv_len=ext,
            prefix_len=0,
            extend_len=ext,
            rid=7,
            backend=be,
        )
        self.assertEqual(set(hp._capture_staging), {(7, 256), (7, 512)})
        for B in (256, 512):
            hidx = hp._capture_staging[(7, B)]
            row = int(hidx[0].item()) // ring_size
            got = hp.data_refs[0][row][0 : ratio * slot_bytes]
            want = (
                buf.kv_score[B - ratio : B].contiguous().view(torch.uint8).reshape(-1)
            )
            self.assertTrue(torch.equal(got, want))


class TestReuseIntoOverlapPrefix(unittest.TestCase):
    def test_window_reaching_into_prefix_is_captured(self):
        # A tiny chunk crossing a page boundary: B - win < cs but buf_lo >= 0
        # (prefix present in [pre|new]) -> capture, never crash (old B-win<cs bug).
        ring_size, ratio, last_dim, page = 8, 4, 16, 8
        slot_bytes = last_dim * torch.tensor([], dtype=torch.bfloat16).element_size()
        hp = _state_host_pool(ring_size=ring_size, slot_bytes=slot_bytes)
        be = _state_backend(hp, page=page, swa_ring=128)
        cs, ext = 7, 2
        pre = cs % ratio + ratio  # overlap prefix length
        valid = pre + ext  # state_buf == [pre | new]
        buf = types.SimpleNamespace(
            kv_score=torch.randint(0, 255, (valid, last_dim), dtype=torch.int32).to(
                torch.bfloat16
            )
        )
        _CAPTURE(
            _state_self(ratio),
            kv_and_score_buffer=buf,
            valid_kv_len=valid,
            prefix_len=cs,
            extend_len=ext,
            rid=7,
            backend=be,
        )
        self.assertIn((7, 8), hp._capture_staging)  # B=8, B-win=4 < cs=7
        hidx = hp._capture_staging[(7, 8)]
        row = int(hidx[0].item()) // ring_size
        buf_lo = (valid - ext) + (8 - ratio - cs)
        got = hp.data_refs[0][row][0 : ratio * slot_bytes]
        want = (
            buf.kv_score[buf_lo : buf_lo + ratio]
            .contiguous()
            .view(torch.uint8)
            .reshape(-1)
        )
        self.assertTrue(torch.equal(got, want))


class TestFlexKVCouplingContract(unittest.TestCase):
    def test_coupling_infos_order_and_filter(self):
        def pool(ratio, swa_page_size, ring_size):
            return types.SimpleNamespace(
                ratio=ratio, swa_page_size=swa_page_size, ring_size=ring_size
            )

        fake = types.SimpleNamespace(
            compress_state_pools=[pool(4, 256, 8), None, pool(128, 256, 1)],
            indexer_compress_state_pools=[pool(4, 512, 8)],
        )
        infos = DeepSeekV4TokenToKVPool.get_swa_state_coupling_infos(fake)
        # None and ratio==128 filtered; attn pools then indexer, order preserved.
        self.assertEqual(infos, [(256, 8), (512, 8)])


if __name__ == "__main__":
    unittest.main()
