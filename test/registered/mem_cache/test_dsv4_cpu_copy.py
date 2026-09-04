"""Tests for DeepSeek-V4 host offload of a retracted request's KV + state.

CPU-only. ``_StubDSV4Pool`` supplies the buffers that
``DeepSeekV4TokenToKVPool.iter_kv_regions`` reads, so the region wiring, the
stride derivation and the full save/load round trip are covered without a GPU or
a published runtime config. The device path is covered end to end by the PD
decode retract tests.

    python -m pytest test/registered/mem_cache/test_dsv4_cpu_copy.py -v
"""

import unittest

import torch

from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4TokenToKVPool,
    HiSparseC4DevicePool,
)
from sglang.srt.mem_cache.kv_region import RequestCtx
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PAGE_SIZE = 256
LAYER_NUM = 4
COMPRESSION_RATIOS = [4, 4, 128, 128]
ROW_BYTES = 16
C4_RING_SIZE = 8
C128_RING_SIZE = 128
NUM_REQ_SLOTS = 8
SWA_PAGES = 16


class _StubKVPool:
    def __init__(self, *, num_rows: int, page_size: int, layer_num: int):
        self.page_size = page_size
        self.kv_buffer = [
            torch.zeros(num_rows, ROW_BYTES, dtype=torch.uint8)
            for _ in range(layer_num)
        ]


class _StubIndexerPool:
    def __init__(self, *, num_rows: int, page_size: int, layer_num: int):
        self.page_size = page_size
        self.index_k_with_scale_buffer = [
            torch.zeros(num_rows, ROW_BYTES, dtype=torch.uint8)
            for _ in range(layer_num)
        ]


class _StubKVAndScore:
    def __init__(self, kv_score: torch.Tensor):
        self.kv_score = kv_score


class _StubStatePool:
    def __init__(self, *, ratio: int, ring_size: int, swa_page_size: int):
        self.ratio = ratio
        self.ring_size = ring_size
        self.swa_page_size = swa_page_size
        # Mirrors the real sizing: C4 state is indexed by SWA page * ring, C128
        # state gives every request slot a whole ring.
        num_rows = (
            SWA_PAGES * ring_size if ratio == 4 else NUM_REQ_SLOTS * ring_size
        ) + ring_size
        self.kv_score_buffer = _StubKVAndScore(
            torch.zeros(num_rows, 8, dtype=torch.float32)
        )


class _StubDSV4Pool(DeepSeekV4TokenToKVPool):
    """Attribute-level fixture for the offload region wiring.

    Deliberately skips ``DeepSeekV4TokenToKVPool.__init__``: the real buffers
    need a GPU and ``get_ring_size`` reads the published spec namespace, while
    ``iter_kv_regions`` only reads the attributes set here. This is also the
    shape a sibling pool takes -- it overrides buffers and inherits
    ``get_cpu_copy`` / ``load_cpu_copy`` unchanged.
    """

    def __init__(self, *, mapping: torch.Tensor, with_mtp_pending: bool = False):
        self._unified_kv = False
        self.page_size = PAGE_SIZE
        self.swa_page_size = PAGE_SIZE
        self.compression_ratios = COMPRESSION_RATIOS
        self.full_to_swa_index_mapping = mapping

        self.swa_kv_pool = _StubKVPool(
            num_rows=SWA_PAGES, page_size=PAGE_SIZE, layer_num=LAYER_NUM
        )
        self.c4_kv_pool = _StubKVPool(num_rows=16, page_size=64, layer_num=2)
        self.c128_kv_pool = _StubKVPool(num_rows=16, page_size=2, layer_num=2)
        self.c4_indexer_kv_pool = _StubIndexerPool(
            num_rows=16, page_size=64, layer_num=2
        )

        self.compress_state_pools = [
            _StubStatePool(
                ratio=ratio,
                ring_size=C4_RING_SIZE if ratio == 4 else C128_RING_SIZE,
                swa_page_size=PAGE_SIZE,
            )
            for ratio in COMPRESSION_RATIOS
        ]
        self.indexer_compress_state_pools = [
            (
                _StubStatePool(
                    ratio=4,
                    ring_size=C4_RING_SIZE,
                    swa_page_size=PAGE_SIZE,
                )
                if ratio == 4
                else None
            )
            for ratio in COMPRESSION_RATIOS
        ]
        self.online_c128_mtp_pending_seq_lens = (
            torch.zeros(NUM_REQ_SLOTS, dtype=torch.int64) if with_mtp_pending else None
        )

    def get_ring_size(self, compress_ratio: int) -> int:
        return C4_RING_SIZE if compress_ratio == 4 else C128_RING_SIZE


def _mapping(*, spans) -> torch.Tensor:
    """Build a full->swa mapping where each ``(token_start, n, swa_start)`` span
    is mapped and everything else stays on the reserved dummy slot 0."""
    mapping = torch.zeros(4096, dtype=torch.int64)
    for token_start, n, swa_start in spans:
        mapping[token_start : token_start + n] = torch.arange(
            swa_start, swa_start + n, dtype=torch.int64
        )
    return mapping


def _indices(start: int, seq_len: int) -> torch.Tensor:
    return torch.arange(start, start + seq_len, dtype=torch.int64)


SEQ_LEN = 600
SWA_TAIL = 512
# Save side: tokens [0, 600), SWA tail mapped to swa pages 3 and 4.
SAVE_START = 0
SAVE_SPAN = (SEQ_LEN - SWA_TAIL, SWA_TAIL, 3 * PAGE_SIZE)
# Load side: tokens [768, 1368), SWA tail mapped to swa pages 7 and 8.
LOAD_START = 768
LOAD_SPAN = (LOAD_START + SEQ_LEN - SWA_TAIL, SWA_TAIL, 7 * PAGE_SIZE)


class TestDSV4RegionList(unittest.TestCase):
    def setUp(self):
        self.pool = _StubDSV4Pool(mapping=_mapping(spans=[SAVE_SPAN]))

    def test_covers_every_per_request_buffer(self):
        names = [region.name for region in self.pool.iter_kv_regions()]
        self.assertEqual(
            names,
            [
                "swa_kv",
                "c4_kv",
                "c128_kv",
                "c4_indexer_kv",
                "c4_state",
                "c4_indexer_state",
                "c128_state",
            ],
        )

    def test_mtp_pending_region_appears_only_when_allocated(self):
        pool = _StubDSV4Pool(mapping=_mapping(spans=[SAVE_SPAN]), with_mtp_pending=True)
        self.assertIn(
            "online_c128_mtp_pending_seq_lens",
            [region.name for region in pool.iter_kv_regions()],
        )

    def test_compressed_strides_collapse_to_the_model_page(self):
        strides = {
            region.name: region.addressing.stride
            for region in self.pool.iter_kv_regions()
            if region.name in ("c4_kv", "c128_kv", "c4_indexer_kv")
        }
        self.assertEqual(strides, {"c4_kv": 256, "c128_kv": 256, "c4_indexer_kv": 256})

    def test_c128_state_resets_before_load(self):
        region = next(r for r in self.pool.iter_kv_regions() if r.name == "c128_state")
        self.assertEqual(region.reset_before_load, self.pool.clear_c128_req_state)

    def test_stride_must_divide_the_model_page(self):
        self.pool.c4_kv_pool.page_size = 63
        with self.assertRaisesRegex(AssertionError, "must divide the model"):
            self.pool.iter_kv_regions()

    def test_unified_kv_is_not_offloadable(self):
        self.pool._unified_kv = True
        with self.assertRaisesRegex(NotImplementedError, "unified_kv"):
            self.pool.iter_kv_regions()

    def test_hisparse_is_not_offloadable(self):
        self.pool.c4_kv_pool = object.__new__(HiSparseC4DevicePool)
        with self.assertRaisesRegex(NotImplementedError, "HiSparse"):
            self.pool.iter_kv_regions()

    def test_req_pool_idx_is_required(self):
        with self.assertRaisesRegex(AssertionError, "needs req_pool_idx"):
            self.pool.get_cpu_copy(_indices(SAVE_START, SEQ_LEN))


class TestDSV4RoundTrip(unittest.TestCase):
    def setUp(self):
        self.pool = _StubDSV4Pool(mapping=_mapping(spans=[SAVE_SPAN, LOAD_SPAN]))
        generator = torch.Generator().manual_seed(0)
        for region in self.pool.iter_kv_regions():
            for tensor in region.tensors:
                if tensor.dtype == torch.uint8:
                    tensor.copy_(
                        torch.randint(
                            1, 255, tensor.shape, generator=generator, dtype=torch.uint8
                        )
                    )
                else:
                    tensor.copy_(
                        torch.randn(tensor.shape, generator=generator).to(tensor.dtype)
                    )

    def _snapshot(self, ctx_start: int, req_pool_idx: int):
        ctx = RequestCtx(
            token_indices=_indices(ctx_start, SEQ_LEN), req_pool_idx=req_pool_idx
        )
        return {
            region.name: [
                tensor[region.addressing.save_plan(ctx)[0]].clone()
                for tensor in region.tensors
            ]
            for region in self.pool.iter_kv_regions()
        }

    def test_rows_survive_new_indices_and_new_req_slot(self):
        expected = self._snapshot(SAVE_START, req_pool_idx=1)
        host = self.pool.get_cpu_copy(_indices(SAVE_START, SEQ_LEN), req_pool_idx=1)

        for region in self.pool.iter_kv_regions():
            for tensor in region.tensors:
                tensor.zero_()

        self.pool.load_cpu_copy(host, _indices(LOAD_START, SEQ_LEN), req_pool_idx=3)

        landed = self._snapshot(LOAD_START, req_pool_idx=3)
        for name, saved_layers in expected.items():
            for layer_id, saved in enumerate(saved_layers):
                with self.subTest(region=name, layer=layer_id):
                    torch.testing.assert_close(landed[name][layer_id], saved)

    def test_row_counts_match_the_geometry(self):
        host = self.pool.get_cpu_copy(_indices(SAVE_START, SEQ_LEN), req_pool_idx=1)
        # 600 tokens over a 256-token row -> 3 rows; the 512-token SWA tail -> 2
        # pages; each SWA page carries one ring block of C4 state; offline C128
        # state moves a whole 128-row block of the request's ring.
        self.assertEqual(host["c4_kv"][0][0].shape[0], 3)
        self.assertEqual(host["c128_kv"][0][0].shape[0], 3)
        self.assertEqual(host["c4_indexer_kv"][0][0].shape[0], 3)
        self.assertEqual(host["swa_kv"][0][0].shape[0], 2)
        self.assertEqual(host["c4_state"][0][0].shape[0], 2 * C4_RING_SIZE)
        self.assertEqual(host["c128_state"][0][0].shape[0], C128_RING_SIZE)

    def test_block_aligned_sequence_skips_c128_state(self):
        host = self.pool.get_cpu_copy(_indices(0, 512), req_pool_idx=1)
        self.assertIsNone(host["c128_state"])


if __name__ == "__main__":
    unittest.main()
