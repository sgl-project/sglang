"""Unit test for strict bit-exact c4 / c4-indexer compress-state capture+restore.

RED spec for "SWA window riding capture of c4/indexer state" (revived from fc18383181 /
b61ce6268a, removed by ab94e91021). Exercises the riding capture/restore helpers
with faked collaborators and real CPU tensors -- no GPU / model required:

  * capture strides by page; one host tile per (rid, B) page boundary
  * the [B-ratio, B) window is packed at tile start (off0=0 contract)
  * captured bytes are byte-identical to the kv_and_score_buffer window
  * capture -> restore round-trip lands byte-exact on the reusing request's ring
  * WITHOUT restore, the reuse-boundary read is stale (the dirty read this fixes)
  * flag OFF (no host pool wired) is a no-op

Until the riding helpers are re-added (Task S.x), the imports below fail and the
whole module errors out at collection time -- that is the intended RED.

Run:
  PYTHONPATH=<worktree>/python python -m pytest \
      test/srt/mem_cache/test_swa_bitexact_compress_state.py -q
"""

import threading
import types
import unittest

import torch

from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
    DeepseekV4HipRadixBackend,
)
from sglang.srt.layers.attention.dsv4.compress_hip import CompressorHip
from sglang.srt.mem_cache import memory_pool_host as MPH
from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool
from sglang.srt.mem_cache.unified_cache_components import swa_component as SC

_CAPTURE = CompressorHip._capture_compress_state_windows


_CAPTURE_DECODE = DeepseekV4HipRadixBackend.capture_compress_state_windows_decode


_RESTORE = SC._restore_state_windows


_TRANSLATE = CompressStatePool.translate_from_swa_loc_to_state_loc


def _fake_host_pool(*, ring_size, slot_bytes, num_pages):
    item_bytes = ring_size * slot_bytes
    host_buf = torch.zeros((num_pages, item_bytes), dtype=torch.uint8)
    counter = {"n": 0}

    def alloc(need):
        assert need == ring_size
        page = counter["n"]
        counter["n"] += 1
        if page >= num_pages:
            return None
        return torch.arange(page * ring_size, page * ring_size + ring_size)

    return types.SimpleNamespace(
        slot_page_size=ring_size,
        item_bytes=item_bytes,
        data_refs=[host_buf],
        _capture_staging={},
        _capture_state_crc={},
        alloc=alloc,
    )


def _fake_backend(host_pool, *, page=256, swa_ring=128, is_indexer=False):
    attr = "_c4_indexer_state_host_pool" if is_indexer else "_c4_state_host_pool"
    pool = types.SimpleNamespace(
        unified_swa_ring_size=swa_ring,
        _c4_state_layer_index={0: 0},
    )
    setattr(pool, attr, host_pool)
    return types.SimpleNamespace(token_to_kv_pool=pool, page_size=page)


def _fake_self(*, ratio=4, is_indexer=False, layer_id=0):
    return types.SimpleNamespace(
        ratio=ratio, is_in_indexer=is_indexer, layer_id=layer_id
    )


class TestCaptureRestoreRoundTrip(unittest.TestCase):
    """End-to-end (real capture + real restore + real translate): the state
    restored onto the device ring at the reuse boundary is byte-identical to the
    compressor buffer window captured at prefill, independent of the actual SWA
    slot offsets used by the reusing request."""

    def test_roundtrip_byte_exact(self):
        page, swa_ring, ring_size, ratio = 256, 128, 8, 4
        last_dim, dtype = 16, torch.bfloat16
        slot_bytes = last_dim * torch.tensor([], dtype=dtype).element_size()

        # --- capture (original request, rid=7, prefill [0, 256)) ---
        hp = _fake_host_pool(ring_size=ring_size, slot_bytes=slot_bytes, num_pages=8)
        backend = _fake_backend(hp, page=page, swa_ring=swa_ring)
        valid_kv_len = 256
        buf = types.SimpleNamespace(
            kv_score=torch.randint(
                0, 255, (valid_kv_len, last_dim), dtype=torch.int32
            ).to(dtype)
        )
        _CAPTURE(
            _fake_self(),
            kv_and_score_buffer=buf,
            valid_kv_len=valid_kv_len,
            prefix_len=0,
            extend_len=256,
            rid=7,
            backend=backend,
        )
        host_indices = hp._capture_staging[(7, 256)]

        # --- restore (reusing request): its window [128, 256) got restored SWA
        # slots swa_chunk; the last `ratio` are [B-ratio, B). Use a swa_base that
        # is a multiple of swa_ring (as real SWA pages are) but different from
        # capture, to prove offset independence. ---
        swa_base = 256
        swa_chunk = torch.arange(swa_base, swa_base + swa_ring, dtype=torch.int64)

        dev = torch.zeros((64, last_dim), dtype=dtype)
        fake_sp = types.SimpleNamespace(
            ring_size=ring_size,
            swa_page_size=swa_ring,
            ratio=ratio,
            kv_score_buffer=types.SimpleNamespace(kv_score=dev),
        )
        fake_sp.translate_from_swa_loc_to_state_loc = types.MethodType(
            _TRANSLATE, fake_sp
        )
        node = types.SimpleNamespace(
            _c4_state_host_value=host_indices,
            _c4_indexer_state_host_value=None,
        )
        restorer = types.SimpleNamespace(
            _c4_state_layer_index={0: 0},
            _c4_state_host_pool=hp,
            _c4_indexer_state_host_pool=None,
            _compress_state_pools=[fake_sp],
            _indexer_compress_state_pools=None,
        )
        _RESTORE(restorer, node, swa_chunk)

        # device ring slots for [B-ratio, B) == captured buffer window
        state_locs = _TRANSLATE(fake_sp, swa_chunk[-ratio:])
        got = dev[state_locs]
        want = buf.kv_score[valid_kv_len - ratio : valid_kv_len]
        self.assertTrue(torch.equal(got, want))

    def test_restore_noop_without_host_value(self):
        dev = torch.zeros((16, 8), dtype=torch.bfloat16)
        fake_sp = types.SimpleNamespace(
            ring_size=8,
            swa_page_size=128,
            ratio=4,
            kv_score_buffer=types.SimpleNamespace(kv_score=dev),
        )
        node = types.SimpleNamespace(
            _c4_state_host_value=None, _c4_indexer_state_host_value=None
        )
        restorer = types.SimpleNamespace(
            _c4_state_layer_index={0: 0},
            _c4_state_host_pool=types.SimpleNamespace(slot_page_size=8, data_refs=[]),
            _c4_indexer_state_host_pool=None,
            _compress_state_pools=[fake_sp],
            _indexer_compress_state_pools=None,
        )
        _RESTORE(restorer, node, torch.arange(128))  # must not raise / touch dev
        self.assertTrue(torch.equal(dev, torch.zeros_like(dev)))

    def test_active_rides_empty_when_unwired(self):
        restorer = types.SimpleNamespace(
            _c4_state_layer_index=None,
            _c4_state_host_pool=None,
            _c4_indexer_state_host_pool=None,
            _compress_state_pools=None,
            _indexer_compress_state_pools=None,
        )
        self.assertEqual(SC._state_rides(restorer), [])


class TestDirtyReadWithoutRestore(unittest.TestCase):
    """Reproduce the strict-mode dirty read.

    On cross-request reuse the compressor's ``compress_extend_paged`` reads the
    c4 / c4-indexer overlap ``pre_kv_state`` at the reuse page boundary
    (``compute_state_len_indices(L, 4) == [L-4, L)``, because c4 is
    overlap/CSA). The SWA-ring restore repopulates the attention window but does
    NOT touch the compress-state ring, so without the riding state restore the
    boundary read returns whatever a prior request left in the ring slot -- a
    dirty read. The non-strict path masks this via tail-reprefill recompute; the
    strict path cannot reprefill, so it must restore the true state.

    This asserts BOTH halves: (a) no-restore -> stale (dirty read reproduced),
    (b) riding restore -> byte-exact truth (the fix)."""

    def test_no_dirty_read_after_reuse(self):
        page, swa_ring, ring_size, ratio = 256, 128, 8, 4
        last_dim, dtype = 16, torch.bfloat16
        slot_bytes = last_dim * torch.tensor([], dtype=dtype).element_size()

        # request A prefill [0, 256): capture its [252, 256) overlap state.
        hp = _fake_host_pool(ring_size=ring_size, slot_bytes=slot_bytes, num_pages=8)
        backend = _fake_backend(hp, page=page, swa_ring=swa_ring)
        buf = types.SimpleNamespace(
            kv_score=torch.randint(0, 255, (256, last_dim), dtype=torch.int32).to(dtype)
        )
        _CAPTURE(
            _fake_self(),
            kv_and_score_buffer=buf,
            valid_kv_len=256,
            prefix_len=0,
            extend_len=256,
            rid=7,
            backend=backend,
        )
        truth = buf.kv_score[256 - ratio : 256]

        # reusing request B: its device state ring is pre-loaded with a DIFFERENT
        # request's leftovers (the stale bytes a dirty read would surface).
        dev = torch.full((64, last_dim), 123, dtype=dtype)
        fake_sp = types.SimpleNamespace(
            ring_size=ring_size,
            swa_page_size=swa_ring,
            ratio=ratio,
            kv_score_buffer=types.SimpleNamespace(kv_score=dev),
        )
        fake_sp.translate_from_swa_loc_to_state_loc = types.MethodType(
            _TRANSLATE, fake_sp
        )
        # B's restored SWA window slots (multiple of swa_ring, != A's).
        swa_chunk = torch.arange(384, 384 + swa_ring, dtype=torch.int64)
        state_locs = _TRANSLATE(fake_sp, swa_chunk[-ratio:])

        # (a) SWA restored but state NOT restored -> dirty read.
        self.assertFalse(
            torch.equal(dev[state_locs], truth),
            "precondition: reuse ring should be stale before state restore",
        )

        # (b) riding state restore -> boundary read now matches true value.
        node = types.SimpleNamespace(
            _c4_state_host_value=hp._capture_staging[(7, 256)],
            _c4_indexer_state_host_value=None,
        )
        restorer = types.SimpleNamespace(
            _c4_state_layer_index={0: 0},
            _c4_state_host_pool=hp,
            _c4_indexer_state_host_pool=None,
            _compress_state_pools=[fake_sp],
            _indexer_compress_state_pools=None,
        )
        _RESTORE(restorer, node, swa_chunk)
        self.assertTrue(
            torch.equal(dev[state_locs], truth),
            "riding restore must land the true [B-ratio, B) window (no dirty read)",
        )


class TestDecodeSourceCapture(unittest.TestCase):
    """Decode-source capture (Task B2): the boundary group state crossed DURING
    decode is snapshot from the device state ring into a host tile keyed (rid, B)
    that is byte-interchangeable with a prefill-captured one, so riding restore
    lands it bit-exact -- no dirty read for decode-region reuse."""

    def _mode(self):
        return types.SimpleNamespace(is_decode_or_idle=lambda: True)

    def test_decode_capture_then_restore_byte_exact(self):
        page, swa_ring, ring_size, ratio = 256, 128, 8, 4
        last_dim, dtype = 16, torch.bfloat16
        slot_bytes = last_dim * torch.tensor([], dtype=dtype).element_size()

        # device state ring big enough for req_pool_idx r=7 boundary group.
        r, B = 7, 512
        dev = torch.zeros((128, last_dim), dtype=dtype)
        fake_sp = types.SimpleNamespace(
            ring_size=ring_size,
            swa_page_size=swa_ring,
            ratio=ratio,
            kv_score_buffer=types.SimpleNamespace(kv_score=dev),
        )
        fake_sp.translate_from_swa_loc_to_state_loc = types.MethodType(
            _TRANSLATE, fake_sp
        )
        # seed the boundary-group device rows with a known window.
        swa_loc = r * swa_ring + (torch.arange(B - ratio, B) % swa_ring)
        state_locs = _TRANSLATE(fake_sp, swa_loc)
        truth = torch.randint(0, 255, (ratio, last_dim), dtype=torch.int32).to(dtype)
        dev[state_locs] = truth

        hp = _fake_host_pool(ring_size=ring_size, slot_bytes=slot_bytes, num_pages=8)
        pool = types.SimpleNamespace(
            _c4_state_layer_index={0: 0},
            _c4_state_host_pool=hp,
            _c4_indexer_state_host_pool=None,
            compress_state_pools=[fake_sp],
            indexer_compress_state_pools=None,
            unified_swa_ring_size=swa_ring,
        )
        backend = types.SimpleNamespace(token_to_kv_pool=pool, page_size=page)
        fb = types.SimpleNamespace(
            forward_mode=self._mode(),
            seq_lens_cpu=torch.tensor([B], dtype=torch.int64),
            req_pool_indices=torch.tensor([r], dtype=torch.int64),
            batch_size=1,
        )
        _CAPTURE_DECODE(backend, fb)

        # one tile keyed (r, B); window packed at tile start [0, ratio).
        self.assertEqual(set(hp._capture_staging), {(r, B)})
        hidx = hp._capture_staging[(r, B)]
        page_row = int(hidx[0].item()) // ring_size
        off0 = 0  # off0=0 contract
        got = hp.data_refs[0][page_row][off0 * slot_bytes : (off0 + ratio) * slot_bytes]
        want = truth.contiguous().view(torch.uint8).reshape(-1)
        self.assertTrue(torch.equal(got, want))

        # riding restore into a DIFFERENT reusing request lands the truth.
        dev2 = torch.full((128, last_dim), 99, dtype=dtype)
        sp2 = types.SimpleNamespace(
            ring_size=ring_size,
            swa_page_size=swa_ring,
            ratio=ratio,
            kv_score_buffer=types.SimpleNamespace(kv_score=dev2),
        )
        sp2.translate_from_swa_loc_to_state_loc = types.MethodType(_TRANSLATE, sp2)
        r2 = 3
        swa_chunk = torch.arange(
            r2 * swa_ring, r2 * swa_ring + swa_ring, dtype=torch.int64
        )
        node = types.SimpleNamespace(
            _c4_state_host_value=hidx, _c4_indexer_state_host_value=None
        )
        restorer = types.SimpleNamespace(
            _c4_state_layer_index={0: 0},
            _c4_state_host_pool=hp,
            _c4_indexer_state_host_pool=None,
            _compress_state_pools=[sp2],
            _indexer_compress_state_pools=None,
        )
        _RESTORE(restorer, node, swa_chunk)
        locs2 = _TRANSLATE(sp2, swa_chunk[-ratio:])
        self.assertTrue(torch.equal(dev2[locs2], truth))

    def test_decode_no_boundary_no_capture(self):
        page, swa_ring, ring_size = 256, 128, 8
        last_dim, dtype = 16, torch.bfloat16
        slot_bytes = last_dim * torch.tensor([], dtype=dtype).element_size()
        dev = torch.zeros((128, last_dim), dtype=dtype)
        fake_sp = types.SimpleNamespace(
            ring_size=ring_size,
            swa_page_size=swa_ring,
            ratio=4,
            kv_score_buffer=types.SimpleNamespace(kv_score=dev),
        )
        fake_sp.translate_from_swa_loc_to_state_loc = types.MethodType(
            _TRANSLATE, fake_sp
        )
        hp = _fake_host_pool(ring_size=ring_size, slot_bytes=slot_bytes, num_pages=8)
        pool = types.SimpleNamespace(
            _c4_state_layer_index={0: 0},
            _c4_state_host_pool=hp,
            _c4_indexer_state_host_pool=None,
            compress_state_pools=[fake_sp],
            indexer_compress_state_pools=None,
            unified_swa_ring_size=swa_ring,
        )
        backend = types.SimpleNamespace(token_to_kv_pool=pool, page_size=page)
        fb = types.SimpleNamespace(
            forward_mode=self._mode(),
            seq_lens_cpu=torch.tensor([300], dtype=torch.int64),  # 300 % 256 != 0
            req_pool_indices=torch.tensor([7], dtype=torch.int64),
            batch_size=1,
        )
        _CAPTURE_DECODE(backend, fb)
        self.assertEqual(hp._capture_staging, {})

    def test_decode_flag_off_noop(self):
        pool = types.SimpleNamespace(_c4_state_layer_index=None)
        backend = types.SimpleNamespace(token_to_kv_pool=pool, page_size=256)
        fb = types.SimpleNamespace(
            forward_mode=self._mode(),
            seq_lens_cpu=torch.tensor([512]),
            req_pool_indices=torch.tensor([0]),
            batch_size=1,
        )
        _CAPTURE_DECODE(backend, fb)  # must not raise


class TestStateRestoreChecksum(unittest.TestCase):
    """Flag-on double-ended check (SGLANG_SWA_DBG_CHECKSUM): the full
    capture -> bind -> promote -> restore path runs the host round-trip CRC +
    device-landing assertions in _dbg_verify_state_restore. Positive: a clean
    round-trip passes. Negative: corrupting the bound host tile between capture
    and restore MUST raise -- proving the check has teeth (guards c4-state riding
    against silent host-tile aliasing / page_row / off0 regressions)."""

    def setUp(self):
        import os

        self._prev_env = os.environ.get("SGLANG_SWA_DBG_CHECKSUM")
        os.environ["SGLANG_SWA_DBG_CHECKSUM"] = "1"
        self._prev_flag = SC._SWA_DBG_CHECKSUM
        SC._SWA_DBG_CHECKSUM = True

    def tearDown(self):
        import os

        SC._SWA_DBG_CHECKSUM = self._prev_flag
        if self._prev_env is None:
            os.environ.pop("SGLANG_SWA_DBG_CHECKSUM", None)
        else:
            os.environ["SGLANG_SWA_DBG_CHECKSUM"] = self._prev_env

    def _capture_bind_promote(self):
        page, swa_ring, ring_size, ratio = 256, 128, 8, 4
        last_dim, dtype = 16, torch.bfloat16
        slot_bytes = last_dim * torch.tensor([], dtype=dtype).element_size()
        hp = _fake_host_pool(ring_size=ring_size, slot_bytes=slot_bytes, num_pages=8)
        backend = _fake_backend(hp, page=page, swa_ring=swa_ring)
        valid_kv_len = 256
        buf = types.SimpleNamespace(
            kv_score=torch.randint(
                0, 255, (valid_kv_len, last_dim), dtype=torch.int32
            ).to(dtype)
        )
        _CAPTURE(
            _fake_self(),
            kv_and_score_buffer=buf,
            valid_kv_len=valid_kv_len,
            prefix_len=0,
            extend_len=256,
            rid=7,
            backend=backend,
        )
        # CRC populated at capture (flag on)
        self.assertTrue(any(k[:2] == (7, 256) for k in hp._capture_state_crc))
        dev = torch.zeros((64, last_dim), dtype=dtype)
        fake_sp = types.SimpleNamespace(
            ring_size=ring_size,
            swa_page_size=swa_ring,
            ratio=ratio,
            kv_score_buffer=types.SimpleNamespace(kv_score=dev),
        )
        fake_sp.translate_from_swa_loc_to_state_loc = types.MethodType(
            _TRANSLATE, fake_sp
        )
        node = types.SimpleNamespace(
            _c4_state_pending_host=None,
            _c4_state_host_value=None,
            _c4_indexer_state_pending_host=None,
            _c4_indexer_state_host_value=None,
        )
        restorer = types.SimpleNamespace(
            _c4_state_layer_index={0: 0},
            _c4_state_host_pool=hp,
            _c4_indexer_state_host_pool=None,
            _compress_state_pools=[fake_sp],
            _indexer_compress_state_pools=None,
        )
        self.assertTrue(SC._bind_state_rides(restorer, node, 7, 256))
        # bind stashed the CRC onto the node (keyed by host_value_attr)
        self.assertIsNotNone(getattr(node, "_c4_state_host_value_crc", None))
        SC._promote_state_pending(restorer, node)
        self.assertIsNotNone(node._c4_state_host_value)
        swa_chunk = torch.arange(256, 256 + swa_ring, dtype=torch.int64)
        return dict(
            restorer=restorer,
            node=node,
            swa_chunk=swa_chunk,
            hp=hp,
            buf=buf,
            dev=dev,
            fake_sp=fake_sp,
            ratio=ratio,
            valid_kv_len=valid_kv_len,
            ring_size=ring_size,
        )

    def test_roundtrip_checksum_passes(self):
        s = self._capture_bind_promote()
        # host round-trip + device-landing asserts run inside; must not raise
        _RESTORE(s["restorer"], s["node"], s["swa_chunk"])
        state_locs = _TRANSLATE(s["fake_sp"], s["swa_chunk"][-s["ratio"] :])
        self.assertTrue(
            torch.equal(
                s["dev"][state_locs],
                s["buf"].kv_score[s["valid_kv_len"] - s["ratio"] : s["valid_kv_len"]],
            )
        )

    def test_corrupted_host_tile_raises(self):
        s = self._capture_bind_promote()
        node, hp, ring_size, ratio = s["node"], s["hp"], s["ring_size"], s["ratio"]
        page_row = int(node._c4_state_host_value[0].item()) // ring_size
        off0 = 0  # off0=0 contract: window packed at tile start
        slot_bytes = hp.item_bytes // ring_size
        tile = hp.data_refs[0][page_row]
        # single-byte flip: CRC delta = 255 - 2*b (odd, always nonzero)
        tile[off0 * slot_bytes] = tile[off0 * slot_bytes].item() ^ 0xFF
        with self.assertRaises(AssertionError):
            _RESTORE(s["restorer"], s["node"], s["swa_chunk"])


def _build_state_pool(
    *, layers=2, ring=8, slot_bytes=4, durable_pages=16, slack=8, seed=1
):
    """Build a minimal unbound DeepSeekV4PagedHostPool on CPU for the promote/
    wiring tests below. The full durable/staging alloc/free/clear/promote
    behaviour is covered by TestSwaStateRoute1Durable in
    test_swa_bitexact_compress_state.py (not re-run here)."""
    p = object.__new__(MPH.DeepSeekV4PagedHostPool)
    p.layout = "layer_first"
    p.slot_page_size = ring
    p.layer_num = layers
    p.item_bytes = ring * slot_bytes
    p.dtype = torch.uint8
    p.size = (durable_pages + slack) * ring
    p._durable_reserve_slots = durable_pages * ring
    p.lock = threading.RLock()
    g = torch.Generator().manual_seed(seed)
    p.kv_buffer = [
        torch.randint(
            0,
            256,
            (durable_pages + slack, ring * slot_bytes),
            dtype=torch.uint8,
            generator=g,
        )
        for _ in range(layers)
    ]
    p.data_refs = p.kv_buffer
    p.clear()
    return p


class TestSwaStatePromoteWiring(unittest.TestCase):
    """Route1 promote/commit wiring in swa_component: _promote_state_pending moves
    the staged tile into the SWA window's durable row and records durable
    host_value; _attach_state_durable_row points a reused carrier at it."""

    def _component(self, state_pool, *, swa_ring=128):
        swa_hp = types.SimpleNamespace(slot_page_size=swa_ring)
        return types.SimpleNamespace(
            component_type="SWA",
            _swa_kv_pool_host=swa_hp,
            _c4_state_layer_index={0: 0},
            _c4_state_host_pool=state_pool,
            _compress_state_pools=[object()],
            _c4_indexer_state_host_pool=None,
            _indexer_compress_state_pools=None,
        )

    def _node(self, *, swa_row, swa_ring=128):
        base = swa_row * swa_ring
        cd = types.SimpleNamespace(host_value=torch.arange(base, base + swa_ring))
        return types.SimpleNamespace(
            component_data={"SWA": cd},
            _c4_state_pending_host=None,
            _c4_state_host_value=None,
        )

    def _state_pool(self, **kw):
        return _build_state_pool(**kw)

    def test_promote_moves_to_durable_row(self):
        ring, swa_ring, swa_row = 8, 128, 5
        p = self._state_pool(ring=ring, slot_bytes=4, durable_pages=16, slack=8, seed=9)
        comp = self._component(p, swa_ring=swa_ring)
        node = self._node(swa_row=swa_row, swa_ring=swa_ring)
        staged = p.alloc(ring)
        staged_row = int(staged[0].item()) // ring
        want = p.data_refs[0][staged_row].clone()
        node._c4_state_pending_host = staged

        SC._promote_state_pending(comp, node)

        # durable host_value addresses row == swa_row
        self.assertIsNotNone(node._c4_state_host_value)
        self.assertEqual(int(node._c4_state_host_value[0].item()) // ring, swa_row)
        self.assertIsNone(node._c4_state_pending_host)
        self.assertTrue(torch.equal(p.data_refs[0][swa_row], want))

    def test_attach_state_durable_row_points_carrier(self):
        ring, swa_ring, swa_row = 8, 128, 7
        p = self._state_pool(ring=ring, durable_pages=16, slack=8)
        comp = self._component(p, swa_ring=swa_ring)
        node = types.SimpleNamespace(_c4_state_host_value=None)
        swa_slice = torch.arange(swa_row * swa_ring, swa_row * swa_ring + swa_ring)
        SC._attach_state_durable_row(comp, node, swa_slice)
        self.assertEqual(int(node._c4_state_host_value[0].item()) // ring, swa_row)

    def test_attach_noop_without_durable_reserve(self):
        # _attach_state_durable_row activates for any state ride whose pool
        # reserves a durable region (that row is addressed by the SWA window row
        # on L3 read). It is a no-op only when there is no such ride -- here a
        # pool with reserve == 0, so there is no durable row to point the reused
        # carrier at.
        ring, swa_ring = 8, 128
        p = self._state_pool(ring=ring, durable_pages=16, slack=8)
        p._durable_reserve_slots = 0  # no durable region
        comp = self._component(p, swa_ring=swa_ring)
        node = types.SimpleNamespace(_c4_state_host_value=None)
        SC._attach_state_durable_row(comp, node, torch.arange(0, swa_ring))
        self.assertIsNone(node._c4_state_host_value)

    def test_promote_drops_binding_when_no_swa_row(self):
        # #3 robustness: durable region reserved but the coupled SWA window row is unknown
        # (host_value not attached -> _node_swa_page_row returns None). We must
        # NEVER adopt the transient slack page as a durable host_value (the
        # allocator can hand that slot to another in-flight capture -> stale /
        # dirty read on restore). Drop the binding instead (recompute on reuse).
        ring, swa_ring = 8, 128
        p = self._state_pool(ring=ring, slot_bytes=4, durable_pages=16, slack=8, seed=3)
        comp = self._component(p, swa_ring=swa_ring)
        cd = types.SimpleNamespace(host_value=None)  # SWA row unknown
        node = types.SimpleNamespace(
            component_data={"SWA": cd},
            _c4_state_pending_host=p.alloc(ring),
            _c4_state_host_value=None,
        )
        avail_before = p.available_size()
        SC._promote_state_pending(comp, node)
        self.assertIsNone(node._c4_state_host_value)  # not adopted
        self.assertIsNone(node._c4_state_pending_host)  # dropped
        self.assertEqual(p.available_size(), avail_before + ring)  # staged freed


class TestReuseValidatorStateGate(unittest.TestCase):
    """Regression fix (step 2): the strict REUSE validator treats a node whose
    durable c4/indexer state host value is MISSING exactly like a node missing
    its SWA host window -- it resets the running window length so the reuse
    boundary clamps to the nearest page that has BOTH a window and its state,
    instead of the old bind-time behavior that dropped state-less windows (which
    zeroed partial-prefix reuse). Gating is scoped to the strict reuse path and
    is a no-op when state riding is unwired or strict is off."""

    def _self(self, *, strict=True, wired=True):
        hp = types.SimpleNamespace(_capture_staging={})
        return types.SimpleNamespace(
            sliding_window_size=8,
            component_type="SWA",
            _strict_bit_exact=strict,
            _swa_kv_pool_host=object(),  # not None -> not swa_device_only_hicache
            cache=types.SimpleNamespace(cache_controller=None),
            _c4_state_layer_index={0: 0} if wired else None,
            _c4_state_host_pool=hp if wired else None,
            _compress_state_pools=[object()] if wired else None,
            # indexer ride unwired -> only the c4 state host_value is gated
            _c4_indexer_state_host_pool=None,
            _indexer_compress_state_pools=None,
        )

    def _node(self, *, host_value, state_value, key_len=4, device=False):
        cd = types.SimpleNamespace(
            value=(torch.arange(key_len) if device else None),
            host_value=host_value,
        )
        return types.SimpleNamespace(
            component_data={"SWA": cd},
            key=list(range(key_len)),
            _c4_state_host_value=state_value,
            _c4_indexer_state_host_value=None,
        )

    def _validator(self, **kw):
        return SC.SWAComponent.create_match_validator(self._self(**kw))

    def test_state_missing_node_rejected(self):
        # host-backed SWA window present but its c4 state is gone -> reject
        v = self._validator()
        node = self._node(host_value=torch.arange(8), state_value=None)
        self.assertFalse(v(node))

    def test_state_backed_node_accepted(self):
        v = self._validator()
        node = self._node(host_value=torch.arange(8), state_value=torch.arange(8))
        self.assertTrue(v(node))  # accumulates from inf

    def test_reset_clamps_to_nearest_window_plus_state(self):
        # walk: [ok] -> [state MISSING] (reset) -> [ok, small] -> [ok, small].
        # The reset means reuse re-accumulates from the state-less boundary; the
        # boundary only re-qualifies once a full sliding window of window+state
        # backed pages follows -- i.e. reuse clamps to the nearest such page.
        v = self._validator()
        ok1 = self._node(host_value=torch.arange(8), state_value=torch.arange(8))
        miss = self._node(host_value=torch.arange(8), state_value=None)
        ok2 = self._node(host_value=torch.arange(8), state_value=torch.arange(8))
        ok3 = self._node(host_value=torch.arange(8), state_value=torch.arange(8))
        self.assertTrue(v(ok1))  # inf -> True
        self.assertFalse(v(miss))  # gate -> reset len=0, reject
        self.assertFalse(v(ok2))  # len=4 < 8 -> still below window (reset held)
        self.assertTrue(v(ok3))  # len=8 >= 8 -> boundary re-qualifies

    def test_no_gate_when_unwired(self):
        # state riding not wired -> validator must not gate on state (unchanged)
        v = self._validator(wired=False)
        node = self._node(host_value=torch.arange(8), state_value=None)
        self.assertTrue(v(node))

    def test_no_gate_when_not_strict(self):
        v = self._validator(strict=False)
        node = self._node(host_value=torch.arange(8), state_value=None)
        self.assertTrue(v(node))

    def test_device_only_match_ignores_state(self):
        # cache_unfinished_req's self-match (match_device_only) must report the
        # request's own host-backed node as a boundary even without state.
        v = SC.SWAComponent.create_match_validator(self._self(), match_device_only=True)
        node = self._node(host_value=torch.arange(8), state_value=None)
        self.assertTrue(v(node))

    def test_missing_swa_host_still_rejected(self):
        # the pre-existing I2' gate (no SWA host copy) is preserved
        v = self._validator()
        node = self._node(host_value=None, state_value=torch.arange(8), device=True)
        self.assertFalse(v(node))


class TestBindStateRidesAtomicRollback(unittest.TestCase):
    """Decoupled co-lifetime (step 1): callers offload the SWA window regardless
    of _bind_state_rides' return, but the helper must still be ATOMIC -- on a
    partial miss (one ride present, the other missing) it rolls back the popped
    tile and sets NO pending ref, so a window is never bound to a half-claimed
    (leaked) state pair. Its return only informs the (now purely observational)
    caller; the window itself is kept and excluded from reuse by the validator."""

    def _ride_pool(self, staging):
        freed = []
        hp = types.SimpleNamespace(
            _capture_staging=staging,
            _capture_state_crc=None,
            free=lambda v: freed.append(v),
        )
        return hp, freed

    def _comp(self, c4_staging, idx_staging):
        c4_hp, c4_freed = self._ride_pool(c4_staging)
        idx_hp, idx_freed = self._ride_pool(idx_staging)
        comp = types.SimpleNamespace(
            _c4_state_layer_index={0: 0},
            _c4_state_host_pool=c4_hp,
            _compress_state_pools=[object()],
            _c4_indexer_state_host_pool=idx_hp,
            _indexer_compress_state_pools=[object()],
        )
        return comp, c4_freed, idx_freed

    def _node(self):
        return types.SimpleNamespace(
            _c4_state_pending_host=None,
            _c4_indexer_state_pending_host=None,
        )

    def test_partial_miss_rolls_back_and_returns_false(self):
        # c4 tile present, indexer tile MISSING -> False; popped c4 tile freed;
        # no pending ref set on the node; staging entry consumed on the pop.
        c4_staging = {(7, 256): torch.arange(8)}
        idx_staging = {}
        comp, c4_freed, idx_freed = self._comp(c4_staging, idx_staging)
        node = self._node()
        ok = SC._bind_state_rides(comp, node, 7, 256)
        self.assertFalse(ok)
        self.assertIsNone(node._c4_state_pending_host)
        self.assertIsNone(node._c4_indexer_state_pending_host)
        self.assertEqual(len(c4_freed), 1)  # popped c4 tile rolled back (freed)
        self.assertEqual(len(idx_freed), 0)
        self.assertNotIn((7, 256), c4_staging)  # popped during the attempt

    def test_both_present_binds_and_returns_true(self):
        c4_staging = {(7, 256): torch.arange(8)}
        idx_staging = {(7, 256): torch.arange(8, 16)}
        comp, c4_freed, idx_freed = self._comp(c4_staging, idx_staging)
        node = self._node()
        ok = SC._bind_state_rides(comp, node, 7, 256)
        self.assertTrue(ok)
        self.assertIsNotNone(node._c4_state_pending_host)
        self.assertIsNotNone(node._c4_indexer_state_pending_host)
        self.assertEqual(len(c4_freed), 0)
        self.assertEqual(len(idx_freed), 0)


if __name__ == "__main__":
    unittest.main()
