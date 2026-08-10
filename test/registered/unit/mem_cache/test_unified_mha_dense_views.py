# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Dense MHA K/V views for the unified memory pool (uniform-row hybrid models).

Covers, CPU-only (pure torch — no GPU / Triton kernels):
  - `MHASubPoolSpec` dense surface: `is_uniform_row` is exactly
    `k_row_bytes == v_row_bytes`, and `dense_blocks_per_page == 2*L` refuses
    asymmetric specs;
  - `build_dense_mha_views` addressing: view_l[dense(t)] must land exactly at
    the page-major envelope byte offset the STRIDED builder assigns to the same
    (page, slot, layer, K|V) cell — the two builders are views over one truth;
  - K and V of one token share ONE dense id (per-layer origin shift does the
    disambiguation), with no aliasing across the 2*L overlapping views;
  - the missing-tail-pad and asymmetric-dims cases fail loud at construction;
  - the strided builder's retrofitted end-of-buffer guard fires on a short
    buffer instead of silently building an OOB view.

Addressing law under test (the derived property everything else builds on):

    dense(t) = (t // ps) * (ps * 2L) + t % ps
    K of layer l at block 2l, V at block 2l+1, blocks are ps rows of
    head_num*head_dim elements — offsets identical to
    MHASubPoolSpec.layer_k/v_offset_in_page when rows are uniform.

    python -m pytest test/registered/unit/mem_cache/test_unified_mha_dense_views.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.layout.page_major import (
    build_dense_mha_views,
    build_page_major_mha_views,
    mha_entry_bytes,
)
from sglang.srt.mem_cache.unified_memory_pool import (
    MHASubPoolSpec,
    MLASubPoolSpec,
    UnifiedDenseMHATokenToKVPool,
    UnifiedKVPool,
    UnifiedMHATokenToKVPool,
)

_DEV = "cpu"

# Small-but-nontrivial MHA geometry: L=2 layers, H=2 heads, D=4, so every byte
# offset is hand-checkable. blocks = 2L = 4 per page.
_L = 2
_H = 2
_D = 4
_ROW = _H * _D  # row elements
_DTYPE = torch.bfloat16
_ITEM = _DTYPE.itemsize
_BLOCKS = 2 * _L


def _mha_spec(head_dim=_D, v_head_dim=None, layer_num=_L, grow="down"):
    return MHASubPoolSpec(
        name="full",
        layer_num=layer_num,
        head_num=_H,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        store_dtype=_DTYPE,
        grow_direction=grow,
    )


def _dense(t, ps):
    return (t // ps) * (ps * _BLOCKS) + t % ps


def _make_raw(ps, num_pages, pad_pages=1):
    page_bytes = ps * _BLOCKS * _ROW * _ITEM
    raw = torch.zeros(
        (num_pages + pad_pages) * page_bytes, dtype=torch.uint8, device=_DEV
    )
    return raw


def _build_dense(raw, ps, num_pages, head_dim=_D, v_head_dim=_D, layer_num=_L):
    return build_dense_mha_views(
        raw,
        layer_num=layer_num,
        head_num=_H,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        store_dtype=_DTYPE,
        page_size=ps,
        num_pages=num_pages,
    )


class TestMHADenseSpecSurface(unittest.TestCase):
    def test_uniform_row_is_exactly_row_byte_equality(self):
        self.assertTrue(_mha_spec().is_uniform_row())
        # Asymmetric K/V (the MiMoV2 shape, scaled down) must NOT be uniform.
        self.assertFalse(_mha_spec(head_dim=6, v_head_dim=4).is_uniform_row())

    def test_dense_blocks_refuses_asymmetric(self):
        self.assertEqual(_mha_spec().dense_blocks_per_page(), _BLOCKS)
        with self.assertRaises(AssertionError):
            _mha_spec(head_dim=6, v_head_dim=4).dense_blocks_per_page()

    def test_spec_offsets_equal_dense_block_origins(self):
        """The spec's byte math and the view builder's origins are two
        independent derivations of the envelope; under uniform rows they must
        agree: layer_k_offset(l) == (2l)*ps*row, layer_v_offset(l) == (2l+1)*ps*row."""
        spec = _mha_spec()
        for ps in (1, 4):
            row = spec.k_row_bytes()
            for l in range(_L):
                self.assertEqual(spec.layer_k_offset_in_page(l, ps), (2 * l) * ps * row)
                self.assertEqual(
                    spec.layer_v_offset_in_page(l, ps), (2 * l + 1) * ps * row
                )

    def test_entry_bytes_matches_layout_helper(self):
        spec = _mha_spec()
        self.assertEqual(
            spec.entry_bytes(),
            mha_entry_bytes(
                layer_num=_L, head_num=_H, head_dim=_D, v_head_dim=_D, itemsize=_ITEM
            ),
        )


class TestDenseMHAViews(unittest.TestCase):
    def test_view_shapes_are_stock_mha(self):
        ps, num_pages = 4, 6
        k_views, v_views = _build_dense(_make_raw(ps, num_pages), ps, num_pages)
        n_dense = num_pages * _BLOCKS * ps
        self.assertEqual(len(k_views), _L)
        self.assertEqual(len(v_views), _L)
        for v in (*k_views, *v_views):
            # The stock MHATokenToKVPool per-layer signature: 3-D, packed rows.
            self.assertEqual(tuple(v.shape), (n_dense, _H, _D))
            self.assertEqual(v.stride(), (_ROW, _D, 1))

    def test_dense_addressing_matches_strided_builder(self):
        """Cross-readback: bytes written through the STRIDED views at
        (page, slot) must be read back through the DENSE views at dense(t),
        for both K and V of every layer — and vice versa. This pins that the
        two builders describe the same physical envelope."""
        for ps in (1, 4):
            num_pages = 5
            raw = _make_raw(ps, num_pages)
            sk, sv = build_page_major_mha_views(
                raw,
                layer_num=_L,
                head_num=_H,
                head_dim=_D,
                v_head_dim=_D,
                store_dtype=_DTYPE,
                page_size=ps,
                num_pages=num_pages,
            )
            dk, dv = _build_dense(raw, ps, num_pages)
            probes = [(0, 0, 0), (1, 1, ps - 1), (4, 0, ps // 2), (3, 1, 0)]
            # strided-write -> dense-read
            for p, l, s in probes:
                t = p * ps + s
                d = _dense(t, ps)
                sk[l][p, s] = float(p * 100 + l * 10 + s + 1)
                sv[l][p, s] = float(p * 100 + l * 10 + s + 2)
                self.assertTrue(
                    torch.all(dk[l][d] == float(p * 100 + l * 10 + s + 1)),
                    f"K (p={p}, l={l}, s={s}, ps={ps}) dense readback off-formula",
                )
                self.assertTrue(
                    torch.all(dv[l][d] == float(p * 100 + l * 10 + s + 2)),
                    f"V (p={p}, l={l}, s={s}, ps={ps}) dense readback off-formula",
                )
            # dense-write -> strided-read
            for p, l, s in probes:
                t = p * ps + s
                d = _dense(t, ps)
                dk[l][d] = float(p * 100 + l * 10 + s + 3)
                dv[l][d] = float(p * 100 + l * 10 + s + 4)
                self.assertTrue(
                    torch.all(sk[l][p, s] == float(p * 100 + l * 10 + s + 3))
                )
                self.assertTrue(
                    torch.all(sv[l][p, s] == float(p * 100 + l * 10 + s + 4))
                )

    def test_k_and_v_share_one_dense_id_without_aliasing(self):
        """One dense id, 2L distinct cells (K and V of every layer): writes
        through all 2L views at the SAME id must not clobber each other."""
        ps, num_pages = 4, 4
        dk, dv = _build_dense(_make_raw(ps, num_pages), ps, num_pages)
        t = 2 * ps + 1  # page 2, slot 1
        d = _dense(t, ps)
        for l in range(_L):
            dk[l][d] = float(2 * l + 1)
            dv[l][d] = float(2 * l + 2)
        for l in range(_L):
            self.assertTrue(torch.all(dk[l][d] == float(2 * l + 1)))
            self.assertTrue(torch.all(dv[l][d] == float(2 * l + 2)))

    def test_missing_tail_pad_fails_loud(self):
        ps, num_pages = 2, 4
        raw = _make_raw(ps, num_pages, pad_pages=0)
        with self.assertRaises(AssertionError):
            _build_dense(raw, ps, num_pages)

    def test_asymmetric_dims_rejected(self):
        ps, num_pages = 2, 4
        raw = _make_raw(ps, num_pages)
        with self.assertRaises(AssertionError):
            _build_dense(raw, ps, num_pages, head_dim=6, v_head_dim=4)

    def test_strided_builder_rejects_short_buffer(self):
        """The retrofitted end-of-buffer guard: a buffer one page short must
        fail at construction instead of silently building an OOB view."""
        ps, num_pages = 2, 4
        page_bytes = ps * _BLOCKS * _ROW * _ITEM
        raw = torch.zeros((num_pages - 1) * page_bytes, dtype=torch.uint8)
        with self.assertRaises(AssertionError):
            build_page_major_mha_views(
                raw,
                layer_num=_L,
                head_num=_H,
                head_dim=_D,
                v_head_dim=_D,
                store_dtype=_DTYPE,
                page_size=ps,
                num_pages=num_pages,
            )


# ---- pool-level dense mode ----

_N_FULL = 32  # full-attn token slots per pool in the fixtures below
_N_SWA = 16


def _swa_spec(grow="up", head_dim=_D, v_head_dim=None):
    return MHASubPoolSpec(
        name="swa",
        layer_num=_L,
        head_num=_H,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        store_dtype=_DTYPE,
        grow_direction=grow,
    )


def _make_pool(ps=1, dense_names=(), full_spec=None, tail_pad=None):
    full = full_spec if full_spec is not None else _mha_spec()
    swa = _swa_spec()
    total = full.entry_bytes() * _N_FULL + swa.entry_bytes() * _N_SWA
    if tail_pad is None:
        tail_pad = ps * max(full.entry_bytes(), swa.entry_bytes())
    return UnifiedKVPool(
        total_bytes=total,
        sub_pool_specs=[full, swa],
        device=_DEV,
        enable_memory_saver=False,
        page_size=ps,
        view_tail_pad_bytes=tail_pad,
        dense_mha_sub_pool_names=dense_names,
    )


class TestUnifiedKVPoolDenseDispatch(unittest.TestCase):
    def test_dense_names_produce_dense_views_others_stay_strided(self):
        pool = _make_pool(ps=4, dense_names=("full",))
        self.assertTrue(pool.is_dense_mha("full"))
        self.assertFalse(pool.is_dense_mha("swa"))
        dk, dv = pool.mha_views_for("full")
        sk, sv = pool.mha_views_for("swa")
        self.assertEqual(dk[0].dim(), 3)  # dense: stock 3-D per-layer shape
        self.assertEqual(dv[0].dim(), 3)
        self.assertEqual(sk[0].dim(), 4)  # strided: 4-D envelope views
        self.assertEqual(sv[0].dim(), 4)

    def test_dense_name_requires_uniform_row(self):
        asym = _mha_spec(head_dim=6, v_head_dim=4)
        with self.assertRaises(AssertionError):
            _make_pool(ps=1, dense_names=("full",), full_spec=asym)

    def test_dense_name_must_be_mha_sub_pool(self):
        full = MLASubPoolSpec(
            name="full",
            layer_num=_L,
            kv_lora_rank=6,
            qk_rope_head_dim=2,
            store_dtype=_DTYPE,
            grow_direction="down",
        )
        swa = _swa_spec()
        with self.assertRaises(AssertionError):
            UnifiedKVPool(
                total_bytes=full.entry_bytes() * _N_FULL + swa.entry_bytes() * _N_SWA,
                sub_pool_specs=[full, swa],
                device=_DEV,
                enable_memory_saver=False,
                page_size=1,
                view_tail_pad_bytes=full.entry_bytes(),
                dense_mha_sub_pool_names=("full",),
            )

    def test_missing_tail_pad_fails_at_pool_construction(self):
        """The dense views hang past the last envelope; a pool built without
        view_tail_pad_bytes must fail loudly at construction, not at first use."""
        with self.assertRaises(AssertionError):
            _make_pool(ps=2, dense_names=("full",), tail_pad=0)


def _layer(l):
    return SimpleNamespace(layer_id=l)


def _make_dense_and_strided_twins(ps):
    dense_kv = _make_pool(ps=ps, dense_names=("full",))
    strided_kv = _make_pool(ps=ps, dense_names=())
    dense_pool = UnifiedDenseMHATokenToKVPool(
        unified_buffer=dense_kv,
        sub_pool_name="full",
        page_size=ps,
        enable_alt_stream=False,
    )
    strided_pool = UnifiedMHATokenToKVPool(
        unified_buffer=strided_kv,
        sub_pool_name="full",
        page_size=ps,
        enable_alt_stream=False,
    )
    return dense_kv, strided_kv, dense_pool, strided_pool


class TestUnifiedDenseMHATokenToKVPool(unittest.TestCase):
    def test_size_is_dense_bound(self):
        """`size` drives BOTH the python OOB check and the store kernel's
        device-side size_limit; it must be the dense row bound, not slot count."""
        for ps in (1, 4):
            dense_kv, _, dense_pool, _ = _make_dense_and_strided_twins(ps)
            n_dense = (dense_kv.max_slots("full") // ps) * _BLOCKS * ps
            self.assertEqual(dense_pool.size, n_dense - ps)

    def test_stock_write_lands_on_envelope_truth(self):
        """Byte-identity: the DENSE pool's stock inherited set_kv_buffer at
        dense locs must produce exactly the bytes that direct writes through a
        strided TWIN pool's views produce at the same (page, slot, layer) cells.
        Pins the whole stock write path (loc -> view -> raw bytes) end-to-end."""
        for ps in (1, 4):
            dense_kv, strided_kv, dense_pool, strided_pool = (
                _make_dense_and_strided_twins(ps)
            )
            probes = [(1, 0), (2, ps - 1), (5, ps // 2)]
            for l in range(_L):
                toks = torch.tensor([p * ps + s for (p, s) in probes])
                dense_locs = (toks // ps) * (ps * _BLOCKS) + toks % ps
                k = torch.full((len(probes), _H, _D), float(l + 1), dtype=_DTYPE)
                v = torch.full((len(probes), _H, _D), float(l + 101), dtype=_DTYPE)
                dense_pool.set_kv_buffer(_layer(l), dense_locs, k, v)
                for p, s in probes:
                    strided_pool.k_buffer[l][p, s] = float(l + 1)
                    strided_pool.v_buffer[l][p, s] = float(l + 101)
            self.assertTrue(
                torch.equal(dense_kv._raw, strided_kv._raw),
                f"dense stock write diverged from envelope truth at ps={ps}",
            )

    def test_move_kv_cache_matches_strided_move(self):
        """Compaction hands PHYSICAL token runs; the dense pool's envelope-copy
        override must move exactly the bytes the strided pool's native move
        moves. Red if the override is lost (inherited move_kv_cache_native
        would apply physical ids to the dense row space)."""
        ps = 4
        dense_kv, strided_kv, dense_pool, strided_pool = _make_dense_and_strided_twins(
            ps
        )
        # Seed identical distinguishable content in both raw buffers.
        seed = (
            torch.arange(
                dense_kv._raw.numel() - dense_kv.view_tail_pad_bytes,
                dtype=torch.float32,
            )
            % 251
        )
        dense_kv._raw[: seed.numel()] = seed.to(torch.uint8)
        strided_kv._raw[: seed.numel()] = seed.to(torch.uint8)
        # Move pages 5,6 -> 2,3 as page-major token runs (the compaction shape).
        src_pages = torch.tensor([5, 6])
        tgt_pages = torch.tensor([2, 3])
        offs = torch.arange(ps)
        src_t = (src_pages[:, None] * ps + offs).reshape(-1)
        tgt_t = (tgt_pages[:, None] * ps + offs).reshape(-1)
        dense_pool.move_kv_cache(tgt_t, src_t)
        strided_pool.move_kv_cache(tgt_t, src_t)
        self.assertTrue(torch.equal(dense_kv._raw, strided_kv._raw))

    def test_requires_dense_built_pool(self):
        strided_kv = _make_pool(ps=1, dense_names=())
        with self.assertRaises(AssertionError):
            UnifiedDenseMHATokenToKVPool(
                unified_buffer=strided_kv,
                sub_pool_name="full",
                page_size=1,
                enable_alt_stream=False,
            )

    def test_transfer_entry_points_fail_loud(self):
        """PD / CPU-copy entry points would silently mis-index (or hit a
        missing-attr AttributeError) under both unified layouts; both classes
        must fail loudly on every one of them."""
        _, _, dense_pool, strided_pool = _make_dense_and_strided_twins(1)
        for pool in (dense_pool, strided_pool):
            with self.assertRaises(NotImplementedError):
                pool.get_contiguous_buf_infos()
            with self.assertRaises(NotImplementedError):
                pool.get_cpu_copy(torch.tensor([1]))
            with self.assertRaises(NotImplementedError):
                pool.load_cpu_copy(None, torch.tensor([1]))
            with self.assertRaises(NotImplementedError):
                pool.set_kv_buffer_prefix_valid()

    def test_hnd_env_cannot_hijack_layout(self):
        """SGLANG_USE_HND_KVCACHE=1 used to flip the inherited env-driven
        layout selector under the unified pools, silently mis-indexing the
        envelope; the pinned kv_cache_layout label must win for BOTH modes."""
        with envs.SGLANG_USE_HND_KVCACHE.override(True):
            _, _, dense_pool, strided_pool = _make_dense_and_strided_twins(1)
            self.assertFalse(dense_pool.use_hnd)
            self.assertFalse(strided_pool.use_hnd)
            self.assertEqual(dense_pool.kv_cache_layout, "page_major_dense")
            self.assertEqual(strided_pool.kv_cache_layout, "page_major_layer_major")


class TestFactoryDenseFlip(unittest.TestCase):
    """The real SWA factory flips dense on for uniform-row specs, the env
    escape hatch forces strided, and the rebind emits the matching
    kernel-facing id spaces — full-dense loc plus swa-dense rail, both derived
    from the STILL-VIRTUAL ids."""

    # _swa_factory geometry: L_full = L_swa = 2, uniform 8/8 dims, ps = 1.
    FULL_MULT = 4  # 2 * L_full
    SWA_MULT = 4  # 2 * L_swa

    def _bundle(self):
        # Self-contained tiny SWA-factory bundle (L_full = L_swa = 2, uniform
        # 8/8 dims, ps = 1) — small enough that dense views build on CPU.
        from sglang.srt.mem_cache.unified_memory_pool import init_unified_swa_pools

        return init_unified_swa_pools(
            device="cpu",
            kv_cache_dtype=torch.float16,
            head_num=2,
            head_dim=8,
            v_head_dim=8,
            swa_head_num=2,
            swa_head_dim=8,
            swa_v_head_dim=8,
            page_size=1,
            start_layer=0,
            end_layer=4,
            swa_attention_layer_ids=[1, 3],
            full_attention_layer_ids=[0, 2],
            full_max_total_num_tokens=64,
            swa_max_total_num_tokens=32,
            enable_memory_saver=False,
            need_sort=False,
        )

    def test_factory_builds_dense_with_multipliers(self):
        b = self._bundle()
        pool = b.unified_memory_pool
        self.assertTrue(pool.is_dense_mha("full"))
        self.assertTrue(pool.is_dense_mha("swa"))
        alloc = b.token_to_kv_pool_allocator
        self.assertEqual(alloc.kernel_page_multiplier, self.FULL_MULT)
        self.assertEqual(alloc.swa_kernel_page_multiplier, self.SWA_MULT)
        # Sub-pools are the dense class exposing stock 3-D per-layer views.
        self.assertEqual(b.token_to_kv_pool.full_kv_pool.k_buffer[0].dim(), 3)
        self.assertEqual(b.token_to_kv_pool.swa_kv_pool.k_buffer[0].dim(), 3)
        self.assertGreater(pool.view_tail_pad_bytes, 0)

    def test_env_escape_forces_strided(self):
        with envs.SGLANG_FORCE_STRIDED_UNIFIED_MHA.override(True):
            b = self._bundle()
        pool = b.unified_memory_pool
        self.assertFalse(pool.is_dense_mha("full"))
        self.assertFalse(pool.is_dense_mha("swa"))
        alloc = b.token_to_kv_pool_allocator
        self.assertEqual(alloc.kernel_page_multiplier, 1)
        self.assertEqual(alloc.swa_kernel_page_multiplier, 1)
        self.assertEqual(b.token_to_kv_pool.full_kv_pool.k_buffer[0].dim(), 4)

    def test_rebind_emits_dense_rails_from_virtual(self):
        """End-to-end over the real factory: apply_unified_kv_loc_rebind must
        rebind out_cache_loc to FULL-DENSE ids and fill the swa rail with
        SWA-DENSE ids — the swa formula over the VIRTUAL ids proves the
        order-critical rule (a rail computed after the rebind would gather
        v2p_swa at full-dense ids: garbage)."""
        from sglang.srt.model_executor.forward_batch_info import (
            apply_unified_kv_loc_rebind,
        )

        b = self._bundle()
        alloc = b.token_to_kv_pool_allocator
        v = alloc.alloc(4)
        self.assertIsNotNone(v)
        expected_full = alloc.full_v2p_page_table[v] * self.FULL_MULT  # ps=1
        expected_swa = (alloc.swa_v2p_page_table[v] * self.SWA_MULT).to(torch.int32)
        from sglang.srt.mem_cache.kv_index_source import KVIndexSource

        fb = SimpleNamespace(
            out_cache_loc=v.clone(),
            swa_out_cache_loc=None,
            out_cache_loc_is_physical=False,
            _kv_index_source=None,
        )
        # Real KVIndexSource over the real factory allocator/pool: this pins
        # the whole chain — capability probe, dense-first translate, swa-rail
        # ordering — not a faked translate surface.
        runner = SimpleNamespace(
            kv_index_source=KVIndexSource(
                req_to_token=torch.zeros((1, 8), dtype=torch.int64),
                token_to_kv_pool_allocator=alloc,
                token_to_kv_pool=b.token_to_kv_pool,
                page_size=1,
                device="cpu",
            )
        )
        apply_unified_kv_loc_rebind(fb, runner)
        self.assertTrue(fb.out_cache_loc_is_physical)
        self.assertTrue(torch.equal(fb.out_cache_loc, expected_full))
        self.assertTrue(torch.equal(fb.swa_out_cache_loc, expected_swa))


if __name__ == "__main__":
    unittest.main()
