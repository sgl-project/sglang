"""Acceptance regression tests for the two strict SWA-HiCache geometry fixes
and the FlexKV coupling contract. CPU-only -- no GPU / model.

  * off0=0 packing decouples the host state tile from the spec-padded SWA ring,
    so state capture stays byte-exact even when swa_ring % ring_size != 0 and
    page % swa_ring != 0 (the EAGLE crash geometry: window + spec_extra = 131).
  * a reuse window reaching back into the overlap prefix (B - win < cs) is
    captured when the prefix is present (buf_lo >= 0) and never crashes.
  * get_swa_state_coupling_infos exposes (swa_page_size, ring_size) per sidecar
    state pool in the same order / filtering as get_state_buf_infos.
"""

import types
import unittest

import torch

from sglang.srt.layers.attention.dsv4.compress_hip import CompressorHip
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

_CAPTURE = CompressorHip._capture_compress_state_windows


def _host_pool(*, ring_size, slot_bytes, num_pages=16):
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


def _backend(host_pool, *, page, swa_ring):
    pool = types.SimpleNamespace(
        unified_swa_ring_size=swa_ring,
        _c4_state_layer_index={0: 0},
        _c4_state_host_pool=host_pool,
    )
    return types.SimpleNamespace(token_to_kv_pool=pool, page_size=page)


def _self(ratio=4):
    return types.SimpleNamespace(ratio=ratio, is_in_indexer=False, layer_id=0)


class TestGeometryDecoupleEagle(unittest.TestCase):
    def test_capture_nondividing_swa_ring_byte_exact(self):
        # EAGLE: window=128, spec_extra=3 -> swa_ring=131; page=256; ring_size=8.
        # 131 % 8 = 3 and 256 % 131 = 125 -> the old geometry asserts crashed here.
        ring_size, ratio, last_dim, page, swa_ring = 8, 4, 16, 256, 131
        slot_bytes = last_dim * torch.tensor([], dtype=torch.bfloat16).element_size()
        hp = _host_pool(ring_size=ring_size, slot_bytes=slot_bytes)
        be = _backend(hp, page=page, swa_ring=swa_ring)
        ext = 512
        buf = types.SimpleNamespace(
            kv_score=torch.randint(0, 255, (ext, last_dim), dtype=torch.int32).to(
                torch.bfloat16
            )
        )
        _CAPTURE(
            _self(ratio),
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
        hp = _host_pool(ring_size=ring_size, slot_bytes=slot_bytes)
        be = _backend(hp, page=page, swa_ring=128)
        cs, ext = 7, 2
        pre = cs % ratio + ratio  # overlap prefix length
        valid = pre + ext  # state_buf == [pre | new]
        buf = types.SimpleNamespace(
            kv_score=torch.randint(0, 255, (valid, last_dim), dtype=torch.int32).to(
                torch.bfloat16
            )
        )
        _CAPTURE(
            _self(ratio),
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
