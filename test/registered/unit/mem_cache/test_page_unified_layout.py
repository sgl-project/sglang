"""The ``page_unified`` page layout and the unified L3 key scheme that names it.

Two halves of one thing: the layout fixes where a chunk's bytes sit inside a
page, and the key scheme names that same chunk as a layer-range x head-range
coordinate. They share a file because they have to agree on the grid.
"""

import unittest

import torch

from sglang.srt.mem_cache.hicache_key_scheme import (
    DraftKVNamespace,
    derive_namespace,
    draft_namespace_digest,
    namespace_digest,
    normalize_dtype,
    plan_unified_draft_sidecar,
    plan_unified_kv,
    validate_unified_chunk_tiling,
    with_packed_draft,
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

    def test_attention_cp_preserves_head_shards_and_namespace(self):
        for rank_replicated in (False, True):
            for tp_rank in range(2):
                with self.subTest(rank_replicated=rank_replicated, tp_rank=tp_rank):
                    shard = dict(
                        rank_replicated=rank_replicated,
                        local_kv_heads=4,
                        attn_tp_rank=tp_rank,
                        attn_tp_size=2,
                        head_group=4,
                    )
                    self.assertEqual(
                        _plan(**shard, attn_cp_size=4),
                        _plan(**shard, attn_cp_size=1),
                    )

    def test_chunk_spans_address_each_suffixs_slice(self):
        """A per-chunk backend stores page base + offset under each suffix.

        So each span must be exactly the bytes of its (layer range, head group)
        slice of the page block, ragged tail included, or a chunk is stored
        under its neighbour's key.
        """
        cases = {
            "mha": (
                _plan(
                    local_kv_heads=4,
                    attn_tp_rank=1,
                    attn_tp_size=2,
                    head_group=2,
                    layer_partition=30,
                    end_layer=61,
                ),
                _mha(layer_num=61, head_group_num=2),
            ),
            "mla": (
                _plan(
                    rank_replicated=True,
                    local_kv_heads=0,
                    attn_tp_rank=1,
                    attn_tp_size=2,
                    layer_partition=30,
                    end_layer=61,
                ),
                PageUnifiedLayout(
                    page_size=4,
                    layer_num=61,
                    head_num=1,
                    head_group_num=1,
                    head_dim=16,
                    itemsize=2,
                    is_mla=True,
                ),
            ),
        }
        for name, (plan, layout) in cases.items():
            with self.subTest(name):
                page = torch.empty(layout.page_dims(2), dtype=torch.bfloat16)[1]
                spans = plan.chunk_spans(layout)
                self.assertEqual(len(spans), len(plan.suffixes))
                chunks = [
                    (layer_range, group)
                    for layer_range in plan.layer_ranges
                    for group in range(plan.head_group_num)
                ]
                for ((start, end), group), (offset, length) in zip(chunks, spans):
                    view = page[start:end] if layout.is_mla else page[group, start:end]
                    self.assertTrue(view.is_contiguous())
                    self.assertEqual(view.data_ptr() - page.data_ptr(), offset)
                    self.assertEqual(view.nbytes, length)

    def test_chunk_spans_refuse_a_pool_cut_differently(self):
        plan = _plan(local_kv_heads=4, attn_tp_rank=0, attn_tp_size=2, head_group=2)
        with self.assertRaisesRegex(ValueError, "head groups"):
            plan.chunk_spans(_mha(layer_num=80, head_group_num=1))


def _packed(plan, target_layer_num, **overrides):
    kwargs = dict(
        target_layer_num=target_layer_num,
        draft_layer_num=1,
        algorithm="EAGLE",
        draft_id="bundled",
        dtype="bfloat16",
        row_width=8,
        attn_tp_rank=0,
        attn_cp_size=1,
        is_final_stage=True,
    )
    kwargs.update(overrides)
    return with_packed_draft(plan, **kwargs)


class TestPackedDraftPlan(CustomTestCase):
    """Packed draft rows: chunks of their own, appended after the target's."""

    def _cases(self):
        # (name, target plan arguments, layout arguments, attention-TP rank)
        mla = dict(rank_replicated=True, local_kv_heads=0, end_layer=6)
        mla_layout = dict(head_num=1, head_group_num=1, head_dim=16, is_mla=True)
        for tp_size, tp_rank, head_group in ((1, 0, 4), (2, 1, 1), (1, 0, 1)):
            local_heads = 4 // tp_size
            yield (
                f"mha_tp{tp_size}_groups{local_heads // head_group}",
                dict(
                    local_kv_heads=local_heads,
                    attn_tp_rank=tp_rank,
                    attn_tp_size=tp_size,
                    head_group=head_group,
                    layer_partition=4,
                    end_layer=6,
                ),
                dict(head_num=local_heads, head_group_num=local_heads // head_group),
                tp_rank,
            )
        yield (
            "mla",
            dict(mla, attn_tp_rank=0, attn_tp_size=1, layer_partition=4),
            mla_layout,
            0,
        )
        yield (
            "mla_pp_final_stage",
            dict(mla, attn_tp_rank=1, attn_tp_size=2, start_layer=4, layer_partition=4),
            mla_layout,
            1,
        )

    def test_target_chunks_keep_their_names_and_draft_chunks_follow(self):
        for name, plan_kwargs, layout_kwargs, tp_rank in self._cases():
            for draft_layers in (1, 3):
                with self.subTest(name, draft_layers=draft_layers):
                    target = _plan(**plan_kwargs)
                    target_layers = plan_kwargs["end_layer"] - plan_kwargs.get(
                        "start_layer", 0
                    )
                    plan = _packed(
                        target,
                        target_layers,
                        draft_layer_num=draft_layers,
                        attn_tp_rank=tp_rank,
                    )
                    groups = target.head_group_num

                    # The target's part is the target plan, untouched.
                    self.assertEqual(plan.namespace, target.namespace)
                    self.assertEqual(
                        plan.suffixes[: len(target.suffixes)], target.suffixes
                    )
                    self.assertEqual(plan.layer_ranges[:-1], target.layer_ranges)
                    self.assertEqual(plan.head_ranges, target.head_ranges)

                    # One draft chunk per head group, under the draft digest
                    # and the group's GLOBAL index.
                    digest = draft_namespace_digest(plan.draft_namespace)
                    self.assertTrue(digest.startswith("ukd1-"))
                    if target.namespace.rank_replicated:
                        expected = [f"{digest}_D0-{draft_layers}"]
                    else:
                        expected = [
                            f"{digest}_D0-{draft_layers}_H{tp_rank * groups + i}"
                            for i in range(groups)
                        ]
                    self.assertEqual(plan.suffixes[len(target.suffixes) :], expected)
                    self.assertEqual(
                        plan.layer_ranges[-1],
                        (target_layers, target_layers + draft_layers),
                    )

                    # Every byte of the page block is named exactly once, and
                    # each draft chunk is the draft rows of its head group.
                    layout = _mha(
                        layer_num=target_layers + draft_layers, **layout_kwargs
                    )
                    spans = plan.chunk_spans(layout)
                    validate_unified_chunk_tiling(spans, layout.bytes_per_page)
                    page = torch.empty(layout.page_dims(2), dtype=torch.bfloat16)[1]
                    for group, (offset, length) in enumerate(spans[-groups:]):
                        rows = slice(target_layers, target_layers + draft_layers)
                        view = page[rows] if layout.is_mla else page[group, rows]
                        self.assertTrue(view.is_contiguous())
                        self.assertEqual(view.data_ptr() - page.data_ptr(), offset)
                        self.assertEqual(view.nbytes, length)

    def test_unsafe_packed_plans_are_refused(self):
        target = _plan(local_kv_heads=8, attn_tp_rank=0, attn_tp_size=1)
        cases = {
            "non_final_stage": (dict(is_final_stage=False), ValueError),
            "prefill_cp": (dict(attn_cp_size=2), NotImplementedError),
            "no_draft_id": (dict(draft_id=""), ValueError),
            "no_algorithm": (dict(algorithm=""), ValueError),
            "no_draft_layer": (dict(draft_layer_num=0), ValueError),
            "gap_before_the_draft_rows": (dict(target_layer_num=81), ValueError),
        }
        for name, (overrides, error) in cases.items():
            with self.subTest(name), self.assertRaises(error):
                _packed(target, **{"target_layer_num": 80, **overrides})
        with self.assertRaisesRegex(ValueError, "already names"):
            _packed(_packed(target, 80), 81)


class TestDraftNamespaceIdentity(CustomTestCase):
    BASE = dict(
        target="ukv1-0123456789abcdef",
        algorithm="EAGLE",
        draft_id="bundled",
        placement="packed",
        dtype="bfloat16",
        rank_replicated=False,
        total_kv_heads=8,
        head_group=2,
        row_width=128,
        layer_num=1,
        object_layout="page_unified/v1",
    )

    def test_every_field_partitions_the_draft_keyspace(self):
        base = draft_namespace_digest(DraftKVNamespace(**self.BASE))
        changes = dict(
            target="ukv1-fedcba9876543210",
            algorithm="DFLASH",
            draft_id="org/draft@main",
            placement="sidecar",
            # fp8 variants store as uint8 and must not share a keyspace.
            dtype="float8_e4m3fn",
            rank_replicated=True,
            total_kv_heads=16,
            head_group=4,
            row_width=64,
            layer_num=3,
            object_layout="page_unified/v2",
        )
        self.assertEqual(set(changes), set(self.BASE))
        digests = {base}
        for field, value in changes.items():
            with self.subTest(field):
                digest = draft_namespace_digest(
                    DraftKVNamespace(**{**self.BASE, field: value})
                )
                self.assertNotIn(digest, digests)
                digests.add(digest)

    def test_draft_digest_cannot_be_taken_for_a_target_digest(self):
        digest = draft_namespace_digest(DraftKVNamespace(**self.BASE))
        self.assertRegex(digest, r"^ukd1-[0-9a-f]{16}$")

    def test_plan_binds_the_draft_to_its_target_and_its_own_dtype(self):
        def plan(model_id="org/model", target_dtype="float8_e4m3fn", **overrides):
            target = _plan(
                model_id=model_id,
                dtype=target_dtype,
                local_kv_heads=8,
                attn_tp_rank=0,
                attn_tp_size=1,
            )
            return target, _packed(target, 80, **overrides)

        target, base = plan(dtype="float8_e4m3fn")
        # The same draft under another target is another draft keyspace.
        _, other_target = plan(model_id="org/other", dtype="float8_e4m3fn")
        # A draft KV dtype of its own: same target chunks, other draft chunk.
        _, other_dtype = plan(dtype="float8_e5m2")
        _, other_draft = plan(dtype="float8_e4m3fn", draft_id="org/draft@main")
        draft_keys = {
            p.suffixes[-1] for p in (base, other_target, other_dtype, other_draft)
        }
        self.assertEqual(len(draft_keys), 4)
        for same_target in (other_dtype, other_draft):
            self.assertEqual(same_target.suffixes[:-1], target.suffixes)
        self.assertEqual(
            base.draft_namespace.target, namespace_digest(target.namespace)
        )
        self.assertEqual(base.draft_namespace.dtype, "float8_e4m3fn")
        self.assertEqual(other_dtype.draft_namespace.dtype, "float8_e5m2")


class TestSidecarDraftPlan(CustomTestCase):
    """A draft KV pool of its own: all its layers of a head group are one chunk."""

    TARGET = dict(
        model_id="org/model",
        dtype="bfloat16",
        page_size=64,
        rank_replicated=False,
        total_kv_heads=8,
        head_group=2,
    )

    def _sidecar(self, **overrides):
        kwargs = dict(
            target=derive_namespace(**self.TARGET),
            algorithm="DFLASH",
            draft_id="org/dflash@main",
            dtype="bfloat16",
            rank_replicated=False,
            local_kv_heads=4,
            row_width=128,
            layer_num=5,
            attn_tp_rank=1,
            attn_tp_size=2,
            attn_cp_size=1,
            head_group=2,
        )
        kwargs.update(overrides)
        return plan_unified_draft_sidecar(**kwargs)

    def _layout(self, plan, *, heads=4, mla=False):
        return PageUnifiedLayout(
            page_size=64,
            layer_num=5,
            head_num=1 if mla else heads,
            head_group_num=plan.head_group_num,
            head_dim=128,
            itemsize=2,
            is_mla=mla,
        )

    def test_mha_draft_chunks_tile_the_draft_page(self):
        # 5 layers x 8 kv heads x 128 in bfloat16 at TP2 with 2 heads a chunk:
        # rank 1 owns head groups 2 and 3 of 4.
        plan = self._sidecar()
        digest = draft_namespace_digest(plan.draft_namespace)
        self.assertEqual(plan.suffixes, [f"{digest}_D0-5_H2", f"{digest}_D0-5_H3"])
        layout = self._layout(plan)
        self.assertEqual(layout.bytes_per_page, 655360)
        spans = plan.chunk_spans(layout)
        self.assertEqual(spans, [(0, 327680), (327680, 327680)])
        validate_unified_chunk_tiling(spans, layout.bytes_per_page)
        self.assertEqual(
            self._sidecar(attn_tp_rank=0).suffixes,
            [f"{digest}_D0-5_H0", f"{digest}_D0-5_H1"],
        )

    def test_tp_shards_of_the_draft_grid_cover_each_other(self):
        tp1 = self._sidecar(local_kv_heads=8, attn_tp_rank=0, attn_tp_size=1)
        tp2 = [self._sidecar(attn_tp_rank=rank) for rank in range(2)]
        tp4 = [
            self._sidecar(local_kv_heads=2, attn_tp_rank=rank, attn_tp_size=4)
            for rank in range(4)
        ]
        self.assertEqual(tp1.suffixes, [s for plan in tp2 for s in plan.suffixes])
        self.assertEqual(tp1.suffixes, [s for plan in tp4 for s in plan.suffixes])
        self.assertEqual(len(set(tp1.suffixes)), 4)

    def test_unset_head_partition_shares_only_within_one_tp_size(self):
        tp1 = self._sidecar(
            local_kv_heads=8, attn_tp_rank=0, attn_tp_size=1, head_group=None
        )
        tp2 = self._sidecar(head_group=None)
        self.assertEqual(tp1.draft_namespace.head_group, 8)
        self.assertEqual(tp2.draft_namespace.head_group, 4)
        self.assertFalse(set(tp1.suffixes) & set(tp2.suffixes))

    def test_replicated_draft_is_one_chunk_without_a_head_coordinate(self):
        plans = [
            self._sidecar(rank_replicated=True, local_kv_heads=1, attn_tp_rank=rank)
            for rank in range(2)
        ]
        self.assertEqual(plans[0].suffixes, plans[1].suffixes)
        (suffix,) = plans[0].suffixes
        self.assertTrue(suffix.endswith("_D0-5"))
        self.assertEqual(plans[0].draft_namespace.total_kv_heads, 0)
        self.assertEqual(plans[0].draft_namespace.head_group, 0)
        layout = self._layout(plans[0], mla=True)
        self.assertEqual(plans[0].chunk_spans(layout), [(0, layout.bytes_per_page)])

    def test_draft_is_bound_to_its_target_and_named_by_its_own_facts(self):
        base = self._sidecar()
        namespace = base.draft_namespace
        self.assertEqual(
            namespace,
            DraftKVNamespace(
                target=namespace_digest(derive_namespace(**self.TARGET)),
                algorithm="DFLASH",
                draft_id="org/dflash@main",
                placement="sidecar",
                dtype="bfloat16",
                rank_replicated=False,
                total_kv_heads=8,
                head_group=2,
                row_width=128,
                layer_num=5,
                object_layout="page-unified-v1",
            ),
        )
        changes = dict(
            target=derive_namespace(**{**self.TARGET, "model_id": "org/other"}),
            algorithm="DSPARK",
            draft_id="org/other-dflash@main",
            dtype="float8_e4m3fn",
            row_width=64,
            layer_num=3,
            head_group=4,
        )
        keys = {base.suffixes[0]}
        for field, value in changes.items():
            with self.subTest(field):
                suffix = self._sidecar(**{field: value}).suffixes[0]
                self.assertNotIn(suffix, keys)
                keys.add(suffix)
        # Never a packed draft's keyspace, whatever else agrees.
        packed = DraftKVNamespace(
            **{
                **{f: getattr(namespace, f) for f in namespace.__struct_fields__},
                "placement": "packed",
            }
        )
        self.assertNotEqual(
            draft_namespace_digest(packed), draft_namespace_digest(namespace)
        )

    def test_unsafe_sidecar_plans_are_refused(self):
        with self.assertRaisesRegex(NotImplementedError, "context parallelism"):
            self._sidecar(attn_cp_size=2)
        with self.assertRaisesRegex(NotImplementedError, "1 kv head per rank"):
            self._sidecar(local_kv_heads=1, head_group=None)
        for field in ("algorithm", "draft_id"):
            with self.subTest(field), self.assertRaisesRegex(ValueError, "draft id"):
                self._sidecar(**{field: ""})
        with self.assertRaisesRegex(ValueError, "must divide this rank's 4"):
            self._sidecar(head_group=3)
        with self.assertRaisesRegex(ValueError, "layer_num"):
            self._sidecar(layer_num=0)
        # A page cut into other head groups than the plan's cannot be named.
        plan = self._sidecar()
        with self.assertRaisesRegex(ValueError, "2 head groups.*holds 1"):
            plan.chunk_spans(
                PageUnifiedLayout(
                    page_size=64,
                    layer_num=5,
                    head_num=4,
                    head_group_num=1,
                    head_dim=128,
                    itemsize=2,
                )
            )


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
