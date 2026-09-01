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
"""MHA K/V views for the unified memory pool (uniform-row hybrid models).

Covers, CPU-only (pure torch — no GPU / Triton kernels):
  - `build_mha_views` refuses an asymmetric-KV spec: its addressing
    assumes one uniform row width, so it is the boundary that checks;
  - `build_mha_views` addressing: view_l[kernel_id(t)] must land exactly at
    the page-major envelope byte offset the STRIDED builder assigns to the same
    (page, slot, layer, K|V) cell — the two builders are views over one truth;
  - K and V of one token share ONE kernel-facing id (per-layer origin shift does the
    disambiguation), with no aliasing across the 2*L overlapping views;
  - the missing-tail-pad and asymmetric-dims cases fail loud at construction.

Addressing law under test (the derived property everything else builds on):

    kernel_id(t) = (t // ps) * (ps * 2L) + t % ps
    K of layer l at block 2l, V at block 2l+1, blocks are ps rows of
    head_num*head_dim elements — offsets identical to
    MHASubPoolSpec.layer_k/v_offset_in_page when rows are uniform.

    python -m pytest test/registered/unit/mem_cache/test_unified_mha_views.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.layout.page_major import (
    build_mha_views,
    mha_entry_bytes,
)
from sglang.srt.mem_cache.unified_memory_pool import (
    MHASubPoolSpec,
    UnifiedKVPool,
    UnifiedMHATokenToKVPool,
)

_DEV = "cpu"
# `set_kv_buffer` dispatches on the PLATFORM (memory_pool._is_cuda, resolved at
# import), not on the tensors it is handed, so cases driving it must build on
# the platform's device. The rest of this file is byte arithmetic, so CPU.
_STORE_DEV = "cuda" if torch.cuda.is_available() else "cpu"

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


def _kernel_id(t, ps):
    return (t // ps) * (ps * _BLOCKS) + t % ps


def _make_raw(ps, num_pages, pad_pages=1):
    page_bytes = ps * _BLOCKS * _ROW * _ITEM
    raw = torch.zeros(
        (num_pages + pad_pages) * page_bytes, dtype=torch.uint8, device=_DEV
    )
    return raw


def _build_views(raw, ps, num_pages, head_dim=_D, v_head_dim=_D, layer_num=_L):
    return build_mha_views(
        raw,
        layer_num=layer_num,
        head_num=_H,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        store_dtype=_DTYPE,
        page_size=ps,
        num_pages=num_pages,
    )


def _reference_strided_views(raw, *, page_size, num_pages, anchor_bytes=0):
    """Independent 4-D strided description of the page-major envelope.

    This is the retired production strided builder, kept here as the oracle:
    per-layer ``(num_pages, page_size, head_num, head_dim)`` views addressed by
    ``(page, slot)``, so the builder's addressing can be cross-checked
    against a second, independently-derived description of the same bytes.
    """
    k_row_bytes = _ROW * _ITEM
    v_row_bytes = _ROW * _ITEM
    page_bytes = page_size * _L * (k_row_bytes + v_row_bytes)
    as_dtype_view = raw.view(_DTYPE)
    k_stride = (page_bytes // _ITEM, k_row_bytes // _ITEM, _D, 1)
    v_stride = (page_bytes // _ITEM, v_row_bytes // _ITEM, _D, 1)
    shape = (num_pages, page_size, _H, _D)
    k_views, v_views = [], []
    for layer in range(_L):
        k_base = anchor_bytes + layer * page_size * (k_row_bytes + v_row_bytes)
        v_base = k_base + page_size * k_row_bytes
        k_views.append(
            torch.as_strided(
                as_dtype_view,
                size=shape,
                stride=k_stride,
                storage_offset=k_base // _ITEM,
            )
        )
        v_views.append(
            torch.as_strided(
                as_dtype_view,
                size=shape,
                stride=v_stride,
                storage_offset=v_base // _ITEM,
            )
        )
    return k_views, v_views


class TestMHASpecSurface(unittest.TestCase):
    def test_asymmetric_rows_refused_by_the_view_builder(self):
        """The row-block array exists only for uniform rows, so the builder
        whose addressing depends on it is the one that refuses (the MiMoV2
        shape, scaled down). ServerArgs screens such models out of
        --enable-unified-memory long before we get here; this is the check for
        a caller that reaches the builder directly."""
        spec = _mha_spec()
        raw = torch.zeros(1 << 16, dtype=torch.uint8)
        with self.assertRaises(AssertionError):
            build_mha_views(
                raw,
                layer_num=spec.layer_num,
                head_num=spec.head_num,
                head_dim=6,
                v_head_dim=4,
                store_dtype=spec.store_dtype,
                page_size=1,
                num_pages=4,
            )

    def test_spec_offsets_equal_block_origins(self):
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


class TestMHAViews(unittest.TestCase):
    def test_view_shapes_are_stock_mha(self):
        ps, num_pages = 4, 6
        k_views, v_views = _build_views(_make_raw(ps, num_pages), ps, num_pages)
        n_rows = num_pages * _BLOCKS * ps
        self.assertEqual(len(k_views), _L)
        self.assertEqual(len(v_views), _L)
        for v in (*k_views, *v_views):
            # The stock MHATokenToKVPool per-layer signature: 3-D, packed rows.
            self.assertEqual(tuple(v.shape), (n_rows, _H, _D))
            self.assertEqual(v.stride(), (_ROW, _D, 1))

    def test_addressing_matches_strided_reference(self):
        """Cross-readback: bytes written through the reference STRIDED views at
        (page, slot) must be read back through the views at kernel_id(t),
        for both K and V of every layer — and vice versa. This pins that the
        view builder and the independent strided description agree on the
        same physical envelope."""
        for ps in (1, 4):
            num_pages = 5
            raw = _make_raw(ps, num_pages)
            sk, sv = _reference_strided_views(raw, page_size=ps, num_pages=num_pages)
            dk, dv = _build_views(raw, ps, num_pages)
            probes = [(0, 0, 0), (1, 1, ps - 1), (4, 0, ps // 2), (3, 1, 0)]
            # strided-write -> view-read
            for p, l, s in probes:
                t = p * ps + s
                d = _kernel_id(t, ps)
                sk[l][p, s] = float(p * 100 + l * 10 + s + 1)
                sv[l][p, s] = float(p * 100 + l * 10 + s + 2)
                self.assertTrue(
                    torch.all(dk[l][d] == float(p * 100 + l * 10 + s + 1)),
                    f"K (p={p}, l={l}, s={s}, ps={ps}) view readback off-formula",
                )
                self.assertTrue(
                    torch.all(dv[l][d] == float(p * 100 + l * 10 + s + 2)),
                    f"V (p={p}, l={l}, s={s}, ps={ps}) view readback off-formula",
                )
            # view-write -> strided-read
            for p, l, s in probes:
                t = p * ps + s
                d = _kernel_id(t, ps)
                dk[l][d] = float(p * 100 + l * 10 + s + 3)
                dv[l][d] = float(p * 100 + l * 10 + s + 4)
                self.assertTrue(
                    torch.all(sk[l][p, s] == float(p * 100 + l * 10 + s + 3))
                )
                self.assertTrue(
                    torch.all(sv[l][p, s] == float(p * 100 + l * 10 + s + 4))
                )

    def test_byte_addresses_match_envelope_formula(self):
        """The per-layer view's byte address for token ``t``, layer ``L`` must equal
        the hand-computed envelope formula: page origin + layer-block origin +
        slot offset. Independent of any view builder — this is the raw layout
        contract every envelope consumer (moves, sizing, transfer math) relies
        on."""
        k_row = _ROW * _ITEM
        v_row = _ROW * _ITEM
        for ps in (1, 4):
            num_pages = 5
            page_bytes = ps * _L * (k_row + v_row)
            dk, dv = _build_views(_make_raw(ps, num_pages), ps, num_pages)
            for t in (0, 1, ps, 3 * ps + (ps - 1), 4 * ps):
                d = _kernel_id(t, ps)
                for L in range(_L):
                    expected_k = (
                        (t // ps) * page_bytes
                        + L * ps * (k_row + v_row)
                        + (t % ps) * k_row
                    )
                    expected_v = (
                        (t // ps) * page_bytes
                        + L * ps * (k_row + v_row)
                        + ps * k_row
                        + (t % ps) * v_row
                    )
                    got_k = (dk[L].storage_offset() + d * dk[L].stride(0)) * _ITEM
                    got_v = (dv[L].storage_offset() + d * dv[L].stride(0)) * _ITEM
                    self.assertEqual(got_k, expected_k, f"K t={t} L={L} ps={ps}")
                    self.assertEqual(got_v, expected_v, f"V t={t} L={L} ps={ps}")

    def test_k_and_v_share_one_kernel_id_without_aliasing(self):
        """One kernel-facing id, 2L distinct cells (K and V of every layer): writes
        through all 2L views at the SAME id must not clobber each other."""
        ps, num_pages = 4, 4
        dk, dv = _build_views(_make_raw(ps, num_pages), ps, num_pages)
        t = 2 * ps + 1  # page 2, slot 1
        d = _kernel_id(t, ps)
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
            _build_views(raw, ps, num_pages)

    def test_asymmetric_dims_rejected(self):
        ps, num_pages = 2, 4
        raw = _make_raw(ps, num_pages)
        with self.assertRaises(AssertionError):
            _build_views(raw, ps, num_pages, head_dim=6, v_head_dim=4)


# ---- pool level ----

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


def _make_pool(ps=1, full_spec=None, device=_DEV):
    full = full_spec if full_spec is not None else _mha_spec()
    swa = _swa_spec()
    total = full.entry_bytes() * _N_FULL + swa.entry_bytes() * _N_SWA
    return UnifiedKVPool(
        total_bytes=total,
        sub_pool_specs=[full, swa],
        device=device,
        enable_memory_saver=False,
        page_size=ps,
    )


class TestUnifiedKVPoolViews(unittest.TestCase):
    def test_every_mha_sub_pool_is_per_layer_contiguous(self):
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
        """The per-layer views hang past the last page envelope, so the pool
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


def _make_pool_and_kv(ps, device=_DEV):
    kv = _make_pool(ps=ps, device=device)
    return kv, UnifiedMHATokenToKVPool(
        unified_buffer=kv,
        sub_pool_name="full",
        page_size=ps,
        enable_alt_stream=False,
    )


class TestUnifiedMHATokenToKVPool(unittest.TestCase):
    def test_size_is_view_row_bound(self):
        """`size` drives BOTH the python OOB check and the store kernel's
        device-side size_limit; it must be the view row bound, not slot count."""
        for ps in (1, 4):
            unified_kv, pool_under_test = _make_pool_and_kv(ps)
            n_rows = (unified_kv.max_slots("full") // ps) * _BLOCKS * ps
            self.assertEqual(pool_under_test.size, n_rows - ps)

    def test_stock_write_lands_on_envelope_truth(self):
        """Byte-identity: the pool's stock inherited `set_kv_buffer` at kernel-facing
        locs must produce exactly the bytes that direct writes through STRIDED
        views over the same envelope produce at the same (page, slot, layer)
        cells. The strided views are built here purely as the independent
        description of the envelope — pins the whole write path (loc -> view ->
        raw bytes) end to end."""
        for ps in (1, 4):
            kv, pool = _make_pool_and_kv(ps, device=_STORE_DEV)
            # An independent strided view of the SAME sub-pool region.
            sk, sv = _reference_strided_views(
                kv._raw,
                page_size=ps,
                num_pages=kv.max_slots("full") // ps,
                anchor_bytes=kv.anchor_bytes("full"),
            )
            probes = [(1, 0), (2, ps - 1), (5, ps // 2)]
            for l in range(_L):
                toks = torch.tensor(
                    [p * ps + s for (p, s) in probes], device=_STORE_DEV
                )
                kernel_locs = (toks // ps) * (ps * _BLOCKS) + toks % ps
                k = torch.full(
                    (len(probes), _H, _D),
                    float(l + 1),
                    dtype=_DTYPE,
                    device=_STORE_DEV,
                )
                v = torch.full(
                    (len(probes), _H, _D),
                    float(l + 101),
                    dtype=_DTYPE,
                    device=_STORE_DEV,
                )
                pool.set_kv_buffer(_layer(l), kernel_locs, k, v)
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
        """Compaction hands PHYSICAL token runs, not kernel-facing ids. The override
        must relocate exactly the page envelopes those runs name — red if it is
        lost, since the inherited per-layer move would apply physical ids to
        the row space."""
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

    def test_transfer_entry_points_fail_loud(self):
        """PD / CPU-copy entry points assume per-layer buffers indexed by TOKEN
        id; against the row space they would silently mis-index (or hit a
        missing-attr AttributeError). Every one of them must raise."""
        _, pool = _make_pool_and_kv(1)
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
        layout selector, putting the pool in a mode whose code paths do not
        match its buffers (HND indexes 4-D; the per-layer views are 3-D). The
        pinned label must win."""
        with envs.SGLANG_USE_HND_KVCACHE.override(True):
            _, pool = _make_pool_and_kv(1)
            self.assertFalse(pool.use_hnd)
            self.assertEqual(pool.kv_cache_layout, "page_major")


class TestFactoryViews(unittest.TestCase):
    """The real SWA factory builds the sub-pools and wires the matching
    kernel-facing multipliers into the composite allocator. End-to-end over
    that factory, the rebind must emit BOTH kernel-facing write locs."""

    # _swa_factory geometry: L_full = L_swa = 2, uniform 8/8 dims, ps = 1.
    FULL_MULT = 4  # 2 * L_full
    SWA_MULT = 4  # 2 * L_swa

    def _bundle(self):
        # Self-contained tiny SWA-factory bundle (L_full = L_swa = 2, uniform
        # 8/8 dims, ps = 1) — small enough that per-layer views build on CPU.
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

    def test_factory_wires_matching_multipliers(self):
        b = self._bundle()
        pool = b.unified_memory_pool
        alloc = b.token_to_kv_pool_allocator
        self.assertEqual(alloc.kernel_page_multiplier, self.FULL_MULT)
        self.assertEqual(alloc.swa_kernel_page_multiplier, self.SWA_MULT)
        # Sub-pools expose stock 3-D per-layer views.
        self.assertEqual(b.token_to_kv_pool.full_kv_pool.k_buffer[0].dim(), 3)
        self.assertEqual(b.token_to_kv_pool.swa_kv_pool.k_buffer[0].dim(), 3)
        self.assertGreater(pool.view_tail_pad_bytes, 0)

    def test_rebind_emits_kernel_facing_full_and_build_derives_swa(self):
        """End-to-end over the real factory: rebind_write_loc rebinds
        out_cache_loc to FULL-kernel-facing ids (phase 1), and the per-batch build
        derives the SWA write loc pointwise from those kernel-facing values
        (phase 2) — both checked against the formulas over the VIRTUAL
        ids."""
        from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator

        b = self._bundle()
        alloc = b.token_to_kv_pool_allocator
        v = alloc.alloc(4)
        self.assertIsNotNone(v)
        expected_full = alloc.full_v2p_page_table[v] * self.FULL_MULT  # ps=1
        expected_swa = alloc.swa_v2p_page_table[v] * self.SWA_MULT

        class _FB:
            pass

        fb = _FB()
        fb.out_cache_loc = v.clone()
        source = KVIndexTranslator(
            req_to_token=torch.zeros((2, 8), dtype=torch.int64),
            token_to_kv_pool_allocator=alloc,
            token_to_kv_pool=b.token_to_kv_pool,
            page_size=1,
            device="cpu",
        )
        self.assertTrue(source.is_translating)
        source.rebind_write_loc(fb)
        self.assertTrue(torch.equal(fb.out_cache_loc, expected_full))
        self.assertTrue(
            torch.equal(
                source.sliding_window_write_loc_for(fb.out_cache_loc), expected_swa
            )
        )


if __name__ == "__main__":
    unittest.main()
