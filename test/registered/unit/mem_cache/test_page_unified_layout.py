"""The ``page_unified`` page layout and the unified L3 key scheme that names it.

Two halves of one thing: the layout fixes where a chunk's bytes sit inside a
page, and the key scheme names that same chunk as a layer-range x head-range
coordinate. They share a file because they have to agree on the grid.
"""

import unittest

import torch

from sglang.srt.mem_cache.hicache_key_scheme import (
    derive_namespace,
    namespace_digest,
    normalize_dtype,
    plan_unified_kv,
)
from sglang.srt.mem_cache.pool_host.page_unified import PageUnifiedLayout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _mha(**overrides):
    kwargs = dict(
        page_size=4,
        layer_num=6,
        head_num=4,
        head_group_num=2,
        head_dim=8,
        itemsize=2,
    )
    kwargs.update(overrides)
    return PageUnifiedLayout(**kwargs)


class TestPageUnifiedLayout(CustomTestCase):
    def test_page_dims_are_the_byte_order(self):
        """The buffer's C-contiguous strides must equal the declared strides.

        Everything else here -- and both transfer kernels -- computes offsets
        from the layout rather than from the tensor, so a disagreement between
        the two would move correct-looking bytes to wrong addresses.
        """
        layout = _mha()
        buffer = torch.empty(layout.page_dims(3), dtype=torch.bfloat16)
        strides = [s * buffer.element_size() for s in buffer.stride()]
        self.assertEqual(
            strides,
            [
                layout.bytes_per_page,
                layout.head_group_bytes,
                layout.layer_bytes,
                layout.page_size * layout.group_bytes,
                layout.group_bytes,
                layout.head_dim * layout.itemsize,
                layout.itemsize,
            ],
        )

    def test_chunks_tile_the_page_exactly_once(self):
        """The grid's chunks must partition a page: no gap, no overlap.

        A gap is KV that is never stored; an overlap is two keys naming the
        same bytes, which diverge as soon as one is evicted.
        """
        layout = _mha(layer_num=6, head_num=3, head_group_num=3)
        covered = bytearray(layout.bytes_per_page)
        for head_group in range(layout.head_group_num):
            for start in range(0, layout.layer_num, 2):
                offset, length = layout.chunk_span((start, start + 2), head_group)
                for byte in range(offset, offset + length):
                    covered[byte] += 1
        self.assertEqual(set(covered), {1})

    def test_ragged_layer_window_is_still_one_run(self):
        """A short final window is one run, which is the point of the axis order.

        DeepSeek-V3's 61 layers at layer_partition=30 leave a width-1 tail. If
        that tail were several runs the transport would have to switch
        descriptor forms mid-plan.
        """
        layout = _mha(layer_num=61, head_num=4, head_group_num=1)
        spans = [layout.chunk_span(r) for r in [(0, 30), (30, 60), (60, 61)]]
        self.assertEqual(
            spans,
            [
                (0, 30 * layout.layer_bytes),
                (30 * layout.layer_bytes, 30 * layout.layer_bytes),
                (60 * layout.layer_bytes, layout.layer_bytes),
            ],
        )
        self.assertEqual(sum(length for _, length in spans), layout.bytes_per_page)

    def test_head_group_count_does_not_change_page_size(self):
        """Cutting the head axis permutes a page; it must not resize it."""
        sizes = {
            PageUnifiedLayout(
                page_size=4,
                layer_num=6,
                head_num=8,
                head_group_num=groups,
                head_dim=16,
                itemsize=2,
            ).bytes_per_page
            for groups in (1, 2, 4, 8)
        }
        self.assertEqual(len(sizes), 1)

    def test_mla_page_has_no_head_or_component_axis(self):
        layout = PageUnifiedLayout(
            page_size=64,
            layer_num=61,
            head_num=1,
            head_group_num=1,
            head_dim=576,
            itemsize=2,
            is_mla=True,
        )
        self.assertEqual(layout.page_dims(7), (7, 61, 64, 576))
        self.assertEqual(layout.components, 1)
        self.assertEqual(layout.chunk_span((0, 61)), (0, layout.bytes_per_page))

    def test_unaligned_group_row_is_rejected(self):
        """The transfer kernels move 16-byte vectors and cannot express a tail."""
        with self.assertRaisesRegex(ValueError, "multiple of 16"):
            _mha(head_num=2, head_group_num=2, head_dim=7, itemsize=1)

    def test_head_group_must_divide_the_pools_heads(self):
        with self.assertRaisesRegex(ValueError, "must divide"):
            PageUnifiedLayout(
                page_size=4,
                layer_num=2,
                head_num=6,
                head_group_num=4,
                head_dim=16,
                itemsize=2,
            )

    def test_chunk_span_rejects_out_of_range_coordinates(self):
        layout = _mha()
        with self.assertRaisesRegex(ValueError, "layer range"):
            layout.chunk_span((0, layout.layer_num + 1))
        with self.assertRaisesRegex(ValueError, "layer range"):
            layout.chunk_span((3, 3))
        with self.assertRaisesRegex(ValueError, "head group"):
            layout.chunk_span((0, 1), layout.head_group_num)


# GQA-70B shape: 8 kv heads, 80 layers.
BASE = dict(
    model_id="meta-llama/Llama-3.1-70B",
    dtype="bfloat16",
    page_size=64,
    rank_replicated=False,
    attn_cp_size=1,
    start_layer=0,
    end_layer=80,
    is_final_stage=True,
)


def _plan(**overrides):
    kwargs = dict(BASE)
    kwargs.update(overrides)
    return plan_unified_kv(**kwargs)


class TestUnifiedKeyGrid(CustomTestCase):
    def test_tp_shards_of_one_grid_cover_each_other(self):
        """TP2's objects are exactly the union of TP4's, in order.

        This is the whole point of the scheme: a cache written by one topology
        must be readable by another. Nothing else in the stack checks it, and
        the failure mode is a silent 0% hit rate.
        """
        tp2 = [
            _plan(local_kv_heads=4, attn_tp_rank=r, attn_tp_size=2, head_group=2)
            for r in range(2)
        ]
        tp4 = [
            _plan(local_kv_heads=2, attn_tp_rank=r, attn_tp_size=4, head_group=2)
            for r in range(4)
        ]
        self.assertEqual(
            [s for p in tp2 for s in p.suffixes],
            [s for p in tp4 for s in p.suffixes],
        )

    def test_pp_stages_cover_the_unsplit_range(self):
        """Same property along the layer axis, including a ragged tail."""
        whole = _plan(
            local_kv_heads=8,
            attn_tp_rank=0,
            attn_tp_size=1,
            head_group=8,
            layer_partition=30,
            end_layer=61,
        )
        stage0 = _plan(
            local_kv_heads=8,
            attn_tp_rank=0,
            attn_tp_size=1,
            head_group=8,
            layer_partition=30,
            start_layer=0,
            end_layer=30,
            is_final_stage=False,
        )
        stage1 = _plan(
            local_kv_heads=8,
            attn_tp_rank=0,
            attn_tp_size=1,
            head_group=8,
            layer_partition=30,
            start_layer=30,
            end_layer=61,
        )
        self.assertEqual(whole.suffixes, stage0.suffixes + stage1.suffixes)
        # The 61st layer forms a short trailing chunk rather than being dropped.
        self.assertTrue(whole.suffixes[-1].endswith("_L60-61_H0"))

    def test_suffix_order_matches_the_range_cross_product(self):
        """Layer-major / head-minor, and the ranges pair with it positionally.

        The storage layer zips suffixes against layer_ranges x head_ranges to
        find each object's bytes; a reordering here writes every chunk under
        its neighbour's key.
        """
        plan = _plan(
            local_kv_heads=4,
            attn_tp_rank=0,
            attn_tp_size=2,
            head_group=2,
            layer_partition=40,
        )
        self.assertEqual(len(plan.layer_ranges), 2)
        self.assertEqual(plan.head_ranges, [(0, 2), (2, 4)])
        digest = namespace_digest(plan.namespace)
        self.assertEqual(
            plan.suffixes,
            [
                f"{digest}_L0-40_H0",
                f"{digest}_L0-40_H1",
                f"{digest}_L40-80_H0",
                f"{digest}_L40-80_H1",
            ],
        )

    def test_head_index_is_global_not_rank_local(self):
        """Rank 1's groups must not collide with rank 0's."""
        plans = [
            _plan(local_kv_heads=2, attn_tp_rank=r, attn_tp_size=4, head_group=2)
            for r in range(4)
        ]
        self.assertEqual(
            [p.suffixes[0].rsplit("_", 1)[1] for p in plans],
            ["H0", "H1", "H2", "H3"],
        )

    def test_local_ranges_are_pool_relative(self):
        """A PP stage indexes its own pool, so its ranges start at 0."""
        stage1 = _plan(
            local_kv_heads=8,
            attn_tp_rank=0,
            attn_tp_size=1,
            head_group=8,
            layer_partition=20,
            start_layer=40,
            end_layer=80,
        )
        self.assertEqual(stage1.layer_ranges, [(0, 20), (20, 40)])
        self.assertTrue(stage1.suffixes[0].endswith("_L40-60_H0"))


class TestNamespaceIdentity(CustomTestCase):
    def test_identity_fields_partition_the_keyspace(self):
        """Every identity field must change the digest.

        A field that does not reach the digest lets two deployments with
        different bytes collide on one key, which is corruption rather than a
        miss.
        """
        base = dict(
            model_id="m",
            dtype="bfloat16",
            page_size=64,
            rank_replicated=False,
            total_kv_heads=8,
            head_group=2,
        )
        digests = {namespace_digest(derive_namespace(**base))}
        for field, value in [
            ("model_id", "other"),
            ("dtype", "float16"),
            ("page_size", 32),
            ("total_kv_heads", 16),
            ("head_group", 4),
            ("layer_partition", 30),
            ("object_layout", "something-else"),
        ]:
            digests.add(namespace_digest(derive_namespace(**{**base, field: value})))
        self.assertEqual(len(digests), 8)

    def test_fp8_variants_do_not_share_a_keyspace(self):
        """Both store as uint8, so only the logical dtype separates them."""
        self.assertNotEqual(
            namespace_digest(
                _plan(
                    local_kv_heads=8,
                    attn_tp_rank=0,
                    attn_tp_size=1,
                    head_group=8,
                    dtype="float8_e4m3fn",
                ).namespace
            ),
            namespace_digest(
                _plan(
                    local_kv_heads=8,
                    attn_tp_rank=0,
                    attn_tp_size=1,
                    head_group=8,
                    dtype="float8_e5m2",
                ).namespace
            ),
        )

    def test_rank_replicated_drops_the_head_axis(self):
        plan = _plan(
            rank_replicated=True,
            local_kv_heads=0,
            attn_tp_rank=3,
            attn_tp_size=8,
        )
        self.assertEqual(plan.head_ranges, [(0, 1)])
        self.assertNotIn("_H", plan.suffixes[0])
        # Every rank derives the same keys, which is what makes one writer enough.
        other = _plan(
            rank_replicated=True, local_kv_heads=0, attn_tp_rank=0, attn_tp_size=8
        )
        self.assertEqual(plan.suffixes, other.suffixes)

    def test_normalize_dtype_strips_the_torch_prefix(self):
        self.assertEqual(normalize_dtype(torch.bfloat16), "bfloat16")


class TestUnifiedKeySchemeRejections(CustomTestCase):
    def test_head_group_must_tile_this_ranks_heads(self):
        with self.assertRaisesRegex(ValueError, "must divide this rank's"):
            _plan(local_kv_heads=3, attn_tp_rank=0, attn_tp_size=2, head_group=2)

    def test_misaligned_stage_start_is_rejected(self):
        """A stage starting off-grid would name chunks no reader can assemble."""
        with self.assertRaisesRegex(ValueError, "does not start on a multiple"):
            _plan(
                local_kv_heads=8,
                attn_tp_rank=0,
                attn_tp_size=1,
                head_group=8,
                layer_partition=30,
                start_layer=20,
                end_layer=50,
            )

    def test_non_final_stage_may_not_end_short(self):
        with self.assertRaisesRegex(ValueError, "only the FINAL pipeline stage"):
            _plan(
                local_kv_heads=8,
                attn_tp_rank=0,
                attn_tp_size=1,
                head_group=8,
                layer_partition=30,
                start_layer=0,
                end_layer=50,
                is_final_stage=False,
            )

    def test_one_kv_head_per_rank_needs_an_explicit_head_group(self):
        """1 head per rank is ambiguous: sharded, or replicated across ranks?

        Replication would have several ranks racing to write one key, so the
        operator has to attest which it is.
        """
        with self.assertRaises(NotImplementedError):
            _plan(local_kv_heads=1, attn_tp_rank=0, attn_tp_size=8)

    def test_attention_cp_is_refused(self):
        with self.assertRaises(NotImplementedError):
            _plan(
                local_kv_heads=8,
                attn_tp_rank=0,
                attn_tp_size=1,
                head_group=8,
                attn_cp_size=2,
            )

    def test_empty_model_id_is_refused(self):
        with self.assertRaisesRegex(ValueError, "non-empty model_id"):
            _plan(
                model_id="",
                local_kv_heads=8,
                attn_tp_rank=0,
                attn_tp_size=1,
                head_group=8,
            )


if __name__ == "__main__":
    unittest.main()
