"""Pool geometry under DCP: the latent KV shards, the indexer does not.

[Test Category] Memory
[Test Target] NPUMLATokenToKVPool allocation geometry and index_buf_size under
              decode context parallelism

This is P2's fourth exit criterion, and it is the only one that fails loudly when
the geometry is merely *wasteful* rather than wrong. The other tests in this
directory check that locations land where they should; this one checks that the
pool costs what it is supposed to, and that the widened indexer is addressable to
its last row.

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

register_npu_ci(est_time=80, suite="full-1-npu-a3")

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

    def test_the_widened_indexer_reaches_its_last_global_position(self):
        """Flat bytes are necessary but not sufficient: a replicated indexer is
        addressed at a raw, untranslated loc, so the TOP of the widened range has
        to be writable and readable. This is the only case here that performs a
        real write, and it is what turns index_buf_size from an allocation size
        into a contract with set_index_k_buffer."""
        dcp_size = 4
        pool = _build(size=SERVED_CONTEXT // dcp_size, index_buf_size=SERVED_CONTEXT)

        last = SERVED_CONTEXT - 1
        loc = torch.tensor([last], dtype=torch.int32, device=DEVICE)
        value = torch.full(
            (1, INDEX_HEAD_DIM), 3.0, dtype=torch.bfloat16, device=DEVICE
        )
        pool.set_index_k_buffer(0, loc, value)

        buffer = pool.get_index_k_buffer(0).view(-1, 1, INDEX_HEAD_DIM)
        self.assertEqual(buffer[last, 0, 0].item(), 3.0)

    def test_it_composes_with_the_indexer_elision(self):
        """Elision and DCP are independent axes and must compose: eliding layers
        must not change the 1/c latent behaviour, sharding must not resurrect
        elided index-K rows, and widening must not add them back either.

        Uses indexer_layer_ids, not a skip_topk_layers bool mask: this pool
        compacts index-K to the layers that own an Indexer and addresses it
        through indexer_layer_id_to_slot.
        """
        live_ids = [i for i in range(LAYER_NUM) if i % 2 == 0]
        live = len(live_ids)

        base_latent, base_index = _measure(1, indexer_layer_ids=live_ids)
        latent, index = _measure(4, indexer_layer_ids=live_ids)

        self.assertEqual(index, base_index)
        self.assertAlmostEqual(latent / base_latent, 1 / 4, delta=0.02)

        # The elision is still worth what it was: only live layers hold rows,
        # and the compacted buffer has exactly one slot per live layer.
        pages = SERVED_CONTEXT // PAGE_SIZE + 1
        per_page = PAGE_SIZE * INDEX_HEAD_DIM * BYTES_PER_ELEM
        self.assertEqual(index, live * pages * per_page)

        pool = _build(
            size=SERVED_CONTEXT // 4,
            index_buf_size=SERVED_CONTEXT,
            indexer_layer_ids=live_ids,
        )
        self.assertEqual(pool.num_indexer_layers, live)
        self.assertEqual(pool.index_k_buffer.shape[0], live)
        self.assertEqual(sorted(pool.indexer_layer_id_to_slot), live_ids)

    def test_reported_bytes_follow_the_widened_indexer(self):
        """get_kv_size_bytes is what the launch log prints, so it has to see the
        widening -- otherwise the pool silently costs more than it reports, and
        the one number an operator reads to size a deployment is wrong."""
        narrow = _build(size=SERVED_CONTEXT, index_buf_size=SERVED_CONTEXT)
        narrow_bytes = narrow.get_kv_size_bytes()
        del narrow
        torch.npu.empty_cache()

        wide = _build(size=SERVED_CONTEXT, index_buf_size=SERVED_CONTEXT * 2)
        index_bytes_per_page = PAGE_SIZE * 1 * INDEX_HEAD_DIM * BYTES_PER_ELEM

        expected_growth = (
            LAYER_NUM * (SERVED_CONTEXT // PAGE_SIZE) * index_bytes_per_page
        )
        self.assertEqual(wide.get_kv_size_bytes() - narrow_bytes, expected_growth)


if __name__ == "__main__":
    unittest.main()
