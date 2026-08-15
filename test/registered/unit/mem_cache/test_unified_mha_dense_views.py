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
  - the missing-tail-pad and asymmetric-dims cases fail loud at construction.

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
    def test_asymmetric_rows_refused_at_construction(self):
        """The unified pool has ONE layout and the dense block array exists
        only for uniform rows, so an asymmetric-KV spec (the MiMoV2 shape,
        scaled down) cannot be built at all. ServerArgs screens such models out
        of --enable-unified-memory before we ever get here."""
        self.assertTrue(_mha_spec().is_uniform_row())
        with self.assertRaises(AssertionError):
            _mha_spec(head_dim=6, v_head_dim=4)

    def test_dense_blocks_is_k_and_v_per_layer(self):
        self.assertEqual(_mha_spec().dense_blocks_per_page(), _BLOCKS)

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


def _make_pool(ps=1, full_spec=None):
    full = full_spec if full_spec is not None else _mha_spec()
    swa = _swa_spec()
    total = full.entry_bytes() * _N_FULL + swa.entry_bytes() * _N_SWA
    return UnifiedKVPool(
        total_bytes=total,
        sub_pool_specs=[full, swa],
        device=_DEV,
        enable_memory_saver=False,
        page_size=ps,
    )


class TestUnifiedKVPoolDenseViews(unittest.TestCase):
    def test_every_mha_sub_pool_is_dense(self):
        """The unified pool has ONE MHA layout: both sub-pools come back as
        stock 3-D per-layer views, whatever their page size."""
        for ps in (1, 4):
            pool = _make_pool(ps=ps)
            for name in ("full", "swa"):
                k, v = pool.mha_views_for(name)
                self.assertEqual(k[0].dim(), 3, f"{name} K at ps={ps}")
                self.assertEqual(v[0].dim(), 3, f"{name} V at ps={ps}")
                self.assertTrue(k[0].is_contiguous())

    def test_tail_pad_is_derived_from_the_specs(self):
        """The dense views hang past the last page envelope, so the pool
        over-allocates one envelope of the widest sub-pool. Derived here, not
        passed in, so no construction site can under-allocate it."""
        for ps in (1, 4):
            kv = _make_pool(ps)
            full, swa = _mha_spec(), _swa_spec()
            self.assertEqual(
                kv.view_tail_pad_bytes,
                ps * max(full.entry_bytes(), swa.entry_bytes()),
                f"tail pad at ps={ps}",
            )
            self.assertEqual(
                kv._raw.numel(),
                full.entry_bytes() * _N_FULL
                + swa.entry_bytes() * _N_SWA
                + kv.view_tail_pad_bytes,
                "the pad extends the allocation only",
            )


def _layer(l):
    return SimpleNamespace(layer_id=l)


def _make_pool_and_kv(ps):
    kv = _make_pool(ps=ps)
    return kv, UnifiedMHATokenToKVPool(
        unified_buffer=kv,
        sub_pool_name="full",
        page_size=ps,
        enable_alt_stream=False,
    )


class TestUnifiedMHATokenToKVPool(unittest.TestCase):
    def test_size_is_dense_bound(self):
        """`size` drives BOTH the python OOB check and the store kernel's
        device-side size_limit; it must be the dense row bound, not slot count."""
        for ps in (1, 4):
            dense_kv, dense_pool = _make_pool_and_kv(ps)
            n_dense = (dense_kv.max_slots("full") // ps) * _BLOCKS * ps
            self.assertEqual(dense_pool.size, n_dense - ps)

    def test_stock_write_lands_on_envelope_truth(self):
        """Byte-identity: the pool's stock inherited `set_kv_buffer` at dense
        locs must produce exactly the bytes that direct writes through STRIDED
        views over the same envelope produce at the same (page, slot, layer)
        cells. The strided views are built here purely as the independent
        description of the envelope — pins the whole write path (loc -> view ->
        raw bytes) end to end."""
        for ps in (1, 4):
            kv, pool = _make_pool_and_kv(ps)
            # An independent view of the SAME sub-pool region, in the layout the
            # standalone page-major pool uses.
            sk, sv = build_page_major_mha_views(
                kv._raw,
                layer_num=_L,
                head_num=_H,
                head_dim=_D,
                v_head_dim=_D,
                store_dtype=_DTYPE,
                page_size=ps,
                num_pages=kv.max_slots("full") // ps,
                anchor_bytes=kv.anchor_bytes("full"),
            )
            probes = [(1, 0), (2, ps - 1), (5, ps // 2)]
            for l in range(_L):
                toks = torch.tensor([p * ps + s for (p, s) in probes])
                dense_locs = (toks // ps) * (ps * _BLOCKS) + toks % ps
                k = torch.full((len(probes), _H, _D), float(l + 1), dtype=_DTYPE)
                v = torch.full((len(probes), _H, _D), float(l + 101), dtype=_DTYPE)
                pool.set_kv_buffer(_layer(l), dense_locs, k, v)
                for p, s in probes:
                    self.assertTrue(
                        torch.all(sk[l][p, s] == float(l + 1)),
                        f"K (l={l}, p={p}, s={s}, ps={ps}) not at the envelope cell",
                    )
                    self.assertTrue(
                        torch.all(sv[l][p, s] == float(l + 101)),
                        f"V (l={l}, p={p}, s={s}, ps={ps}) not at the envelope cell",
                    )

    def test_move_kv_cache_relocates_whole_envelopes(self):
        """Compaction hands PHYSICAL token runs, not dense ids. The override
        must relocate exactly the page envelopes those runs name — red if it is
        lost, since the inherited per-layer move would apply physical ids to
        the dense row space."""
        ps = 4
        kv, pool = _make_pool_and_kv(ps)
        live = kv._raw.numel() - kv.view_tail_pad_bytes
        seed = (torch.arange(live, dtype=torch.float32) % 251).to(torch.uint8)
        kv._raw[:live] = seed
        page_bytes = ps * _mha_spec().entry_bytes()

        src_pages, tgt_pages = torch.tensor([5, 6]), torch.tensor([2, 3])
        offs = torch.arange(ps)
        run = lambda p: (p[:, None] * ps + offs).reshape(-1)
        pool.move_kv_cache(run(tgt_pages), run(src_pages))

        want = seed.clone()
        for sp, tp in zip(src_pages.tolist(), tgt_pages.tolist()):
            want[tp * page_bytes : (tp + 1) * page_bytes] = seed[
                sp * page_bytes : (sp + 1) * page_bytes
            ]
        self.assertTrue(
            torch.equal(kv._raw[:live], want),
            "envelope move did not relocate exactly the named pages",
        )

    def test_hnd_env_cannot_hijack_layout(self):
        """SGLANG_USE_HND_KVCACHE=1 used to flip the inherited env-driven
        layout selector, putting the pool in a mode whose code paths do not
        match its buffers (HND indexes 4-D; the dense views are 3-D). The
        pinned label must win."""
        with envs.SGLANG_USE_HND_KVCACHE.override(True):
            _, pool = _make_pool_and_kv(1)
            self.assertFalse(pool.use_hnd)
            self.assertEqual(pool.kv_cache_layout, "page_major_dense")


class TestFactoryDenseViews(unittest.TestCase):
    """The real SWA factory builds dense sub-pools and wires the matching
    kernel-facing multipliers into the composite allocator."""

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
        alloc = b.token_to_kv_pool_allocator
        self.assertEqual(alloc.kernel_page_multiplier, self.FULL_MULT)
        self.assertEqual(alloc.swa_kernel_page_multiplier, self.SWA_MULT)
        # Sub-pools are the dense class exposing stock 3-D per-layer views.
        self.assertEqual(b.token_to_kv_pool.full_kv_pool.k_buffer[0].dim(), 3)
        self.assertEqual(b.token_to_kv_pool.swa_kv_pool.k_buffer[0].dim(), 3)
        self.assertGreater(pool.view_tail_pad_bytes, 0)


if __name__ == "__main__":
    unittest.main()
