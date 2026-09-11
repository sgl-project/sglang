"""Serving the same context under DCP costs 1/c of the latent KV and the same indexer.

[Test Category] Memory
[Test Target] NPUMLATokenToKVPool allocation geometry under decode context parallelism

This is P2's fourth exit criterion, and it is the only one that fails loudly when
the geometry is merely *wasteful* rather than wrong. The other tests here check
that locations land where they should; this one checks that the pool costs what
it is supposed to.

The comparison is at a fixed **served context** S, which is the thing an operator
actually holds constant -- not at a fixed per-rank pool size:

    dcp_size 1   size = S      index_buf_size = S
    dcp_size c   size = S / c  index_buf_size = S

so the latent KV falls to 1/c because it is sharded, while the indexer stays flat
because it is replicated and every rank still has to address all S positions.

Getting the page size right and the capacity wrong is the failure this catches:
it still allocates correctly-shaped memory, every read and write still lands in
the right place, and the pool simply costs c times what it should.

One NPU, no weights, no server.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=90, suite="full-1-npu-a3")

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
INDEX_HEAD_DIM = 128
PAGE_SIZE = 128
LAYER_NUM = 4
# The served context, chosen so S / c stays page-aligned for every c below.
SERVED_CONTEXT = 8192
DEVICE = "npu:0"

BYTES_PER_ELEM = 2  # bfloat16


def _build(*, size, index_buf_size, **overrides):
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool

    kwargs = dict(
        size=size,
        page_size=PAGE_SIZE,
        dtype=torch.bfloat16,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        layer_num=LAYER_NUM,
        device=DEVICE,
        enable_memory_saver=False,
        index_head_dim=INDEX_HEAD_DIM,
        start_layer=0,
        end_layer=LAYER_NUM,
        index_buf_size=index_buf_size,
    )
    kwargs.update(overrides)
    return NPUMLATokenToKVPool(**kwargs)


def _latent_bytes(pool) -> int:
    return pool.k_buffer.nbytes + pool.v_buffer.nbytes


def _index_bytes(pool) -> int:
    buffer = pool.index_k_buffer
    if isinstance(buffer, torch.Tensor):
        return buffer.nbytes
    return sum(layer.nbytes for layer in buffer)


def _measure(dcp_size, **overrides):
    """Bytes for serving SERVED_CONTEXT tokens at this DCP width."""
    pool = _build(
        size=SERVED_CONTEXT // dcp_size,
        index_buf_size=SERVED_CONTEXT,
        **overrides,
    )
    latent, index = _latent_bytes(pool), _index_bytes(pool)
    del pool
    torch.npu.empty_cache()
    return latent, index


class TestNpuMlaDcpPoolBytes(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        import torch_npu  # noqa: F401

        torch.npu.set_device(0)

    def tearDown(self):
        torch.npu.empty_cache()

    def test_latent_kv_falls_to_one_over_c(self):
        base_latent, _ = _measure(1)

        for dcp_size in (2, 4, 8):
            with self.subTest(dcp_size=dcp_size):
                latent, _ = _measure(dcp_size)
                # Not exactly 1/c: every buffer carries one padded page, so the
                # shrunk pool keeps a page the divided figure does not.
                pages = SERVED_CONTEXT // dcp_size // PAGE_SIZE + 1
                per_page = (
                    PAGE_SIZE * (KV_LORA_RANK + QK_ROPE_HEAD_DIM) * BYTES_PER_ELEM
                )
                self.assertEqual(latent, LAYER_NUM * pages * per_page)
                # And it really is close to 1/c, not merely self-consistent.
                self.assertAlmostEqual(latent / base_latent, 1 / dcp_size, delta=0.02)

    def test_the_indexer_stays_flat(self):
        """The replicated half. If this shrinks with c, the indexer has been
        given the sharded treatment and will run off the end of its buffer on
        the first position it does not own."""
        _, base_index = _measure(1)

        for dcp_size in (2, 4, 8):
            with self.subTest(dcp_size=dcp_size):
                _, index = _measure(dcp_size)
                self.assertEqual(index, base_index)

    def test_the_saving_is_the_whole_point(self):
        """State the headline directly, so a regression reads as a number rather
        than as a shape mismatch somewhere else."""
        base_latent, base_index = _measure(1)
        latent, index = _measure(8)

        saved = (base_latent + base_index) - (latent + index)
        self.assertGreater(saved, 0)
        # The saving is entirely latent-KV; the indexer contributes nothing.
        self.assertEqual(saved, base_latent - latent)
        self.assertEqual(index, base_index)

    def test_it_holds_with_the_indexer_elision_on(self):
        """Elision and DCP are independent axes and must compose: eliding layers
        must not change the 1/c latent behaviour, and sharding must not
        resurrect elided index-K rows."""
        # indexer_layer_ids, not the old skip_topk_layers bool mask: this pool
        # now compacts index-K to the layers that own an Indexer.
        live_ids = [i for i in range(LAYER_NUM) if i % 2 == 0]
        live = len(live_ids)

        base_latent, base_index = _measure(1, indexer_layer_ids=live_ids)
        latent, index = _measure(4, indexer_layer_ids=live_ids)

        self.assertEqual(index, base_index)
        self.assertAlmostEqual(latent / base_latent, 1 / 4, delta=0.02)

        # And the elision itself is still worth what it was: only live layers
        # hold index-K rows.
        pages = SERVED_CONTEXT // PAGE_SIZE + 1
        per_page = PAGE_SIZE * INDEX_HEAD_DIM * BYTES_PER_ELEM
        self.assertEqual(index, live * pages * per_page)


if __name__ == "__main__":
    unittest.main()
