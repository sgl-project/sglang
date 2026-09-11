"""The index-K buffer's extent is a parameter, not a function of the pool size.

[Test Category] Memory
[Test Target] NPUMLATokenToKVPool(index_buf_size=...)

Under decode context parallelism the latent KV is sharded across ranks while the
LightningIndexer is replicated, so the two buffers stop having the same number of
token rows: the latent KV keeps ``max_total`` and translates writes into it, and
the indexer spans ``max_total * dcp_size`` and is written at a raw ``loc``. The
CUDA DSA pool has always had this seam (``DSATokenToKVPool(index_buf_size=...)``
feeding ``IndexKeyCache``); this pool derived both counts from ``self.size``.

These tests pin the seam itself, not DCP. Nothing passes a widened value yet, so
the first two are the no-op proof -- that adding the parameter changed no
existing geometry -- and the rest establish what widening it does and does not
touch, before any write-path translation depends on it.

One NPU, no weights, no server.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=60, suite="full-1-npu-a3")

# GLM-5.2's MLA dimensions, which is the shape this seam exists for.
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
INDEX_HEAD_DIM = 128
PAGE_SIZE = 128
LAYER_NUM = 8
SIZE = 4096
DEVICE = "npu:0"


def _build(**overrides):
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool

    kwargs = dict(
        size=SIZE,
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
    )
    kwargs.update(overrides)
    return NPUMLATokenToKVPool(**kwargs)


def _index_pages(pool, layer_idx: int) -> int:
    return pool.index_k_buffer[layer_idx].shape[0]


class TestNpuMlaIndexBufSize(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        import torch_npu  # noqa: F401

        torch.npu.set_device(0)
        cls.expected_pages = SIZE // PAGE_SIZE + 1

    def tearDown(self):
        torch.npu.empty_cache()

    def test_default_matches_the_latent_kv_extent(self):
        """Omitting the argument must reproduce the pre-seam geometry exactly."""
        pool = _build()

        self.assertEqual(pool.index_buf_size, SIZE)
        self.assertEqual(pool.k_buffer.shape[1], self.expected_pages)
        self.assertEqual(_index_pages(pool, 0), self.expected_pages)

    def test_passing_the_size_explicitly_is_the_same_object(self):
        """The no-op proof: naming the default changes nothing about the bytes."""
        implicit = _build()
        implicit_bytes = implicit.get_kv_size_bytes()
        del implicit
        torch.npu.empty_cache()

        explicit = _build(index_buf_size=SIZE)

        self.assertEqual(explicit.get_kv_size_bytes(), implicit_bytes)

    def test_widening_moves_the_indexer_and_nothing_else(self):
        """This is the property DCP needs: the two extents are independent."""
        narrow = _build()
        narrow_k_pages = narrow.k_buffer.shape[1]
        narrow_v_pages = narrow.v_buffer.shape[1]
        narrow_index_pages = _index_pages(narrow, 0)
        del narrow
        torch.npu.empty_cache()

        dcp_size = 4
        wide = _build(index_buf_size=SIZE * dcp_size)

        # The latent KV is sharded, so it must not grow.
        self.assertEqual(wide.k_buffer.shape[1], narrow_k_pages)
        self.assertEqual(wide.v_buffer.shape[1], narrow_v_pages)
        # The indexer is replicated, so it spans the whole virtual range. Not
        # exactly dcp_size x, because each buffer carries one padded page.
        self.assertEqual(_index_pages(wide, 0), SIZE * dcp_size // PAGE_SIZE + 1)
        self.assertGreater(_index_pages(wide, 0), narrow_index_pages * (dcp_size - 1))

    def test_widening_reaches_the_last_global_position(self):
        """A replicated indexer is addressed at raw loc, so the top of the
        widened range has to be writable -- that is the whole point of it."""
        dcp_size = 4
        wide = _build(index_buf_size=SIZE * dcp_size)

        last = SIZE * dcp_size - 1
        loc = torch.tensor([last], dtype=torch.int32, device=DEVICE)
        value = torch.full(
            (1, INDEX_HEAD_DIM), 3.0, dtype=torch.bfloat16, device=DEVICE
        )
        wide.set_index_k_buffer(0, loc, value)

        buffer = wide.get_index_k_buffer(0).view(-1, 1, INDEX_HEAD_DIM)
        self.assertEqual(buffer[last, 0, 0].item(), 3.0)

    def test_it_composes_with_the_skip_topk_elision(self):
        """Widening must not resurrect rows for layers that own no Indexer.

        The pool compacts index-K to the layers that own an Indexer and maps
        layer_id through ``indexer_layer_id_to_slot``; an elided layer is absent
        from that map rather than holding a zero-page tensor at its own index.
        This test previously passed a ``skip_topk_layers`` bool mask and asserted
        the zero-page form, which is the layout this pool no longer uses.
        """
        live = [i for i in range(LAYER_NUM) if i % 2 == 0]
        pool = _build(index_buf_size=SIZE * 4, indexer_layer_ids=live)

        # One slot per live layer, none for the elided ones -- that is the
        # elision, and widening must not add rows back.
        self.assertEqual(pool.num_indexer_layers, len(live))
        self.assertEqual(pool.index_k_buffer.shape[0], len(live))
        self.assertEqual(sorted(pool.indexer_layer_id_to_slot), live)

        for layer_id in range(LAYER_NUM):
            with self.subTest(layer=layer_id):
                if layer_id in pool.indexer_layer_id_to_slot:
                    slot = pool.indexer_layer_id_to_slot[layer_id]
                    self.assertEqual(
                        _index_pages(pool, slot), SIZE * 4 // PAGE_SIZE + 1
                    )
                else:
                    self.assertNotIn(layer_id, pool.indexer_layer_id_to_slot)

    def test_reported_bytes_follow_the_widened_indexer(self):
        """get_kv_size_bytes is what the launch log prints, so it has to see the
        widening -- otherwise the pool silently costs more than it reports."""
        narrow = _build()
        narrow_bytes = narrow.get_kv_size_bytes()
        del narrow
        torch.npu.empty_cache()

        wide = _build(index_buf_size=SIZE * 2)
        index_bytes_per_page = PAGE_SIZE * 1 * INDEX_HEAD_DIM * 2

        expected_growth = LAYER_NUM * (SIZE // PAGE_SIZE) * index_bytes_per_page
        self.assertEqual(wide.get_kv_size_bytes() - narrow_bytes, expected_growth)


if __name__ == "__main__":
    unittest.main()
