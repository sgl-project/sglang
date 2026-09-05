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
"""MHA K/V views for the unified memory pool (token-major dense views).

Covers, CPU-only (pure torch — no GPU / Triton kernels):
  - `MHASubPoolSpec.layout()`: K and V of every layer sit at fixed offsets
    inside one slot entry, K and V rows may differ in width, alignment is
    enforced at construction;
  - `build_dense_views` addressing: view_l[t] for PHYSICAL token t must land
    exactly at the byte the envelope formula assigns to (page, slot, layer,
    K|V) — cross-checked against an independent 4-D (page, slot) description;
  - K and V of one token share ONE id (the per-layer origin shift does the
    disambiguation), with no aliasing across the 2L views;
  - a too-short buffer and misaligned rows fail loud at construction.

Addressing law under test:

    byte(t, l, K) = t * entry_bytes + l * (k_row + v_row)
    byte(t, l, V) = byte(t, l, K) + k_row
    t = page * page_size + slot      (the physical token id IS the kernel id)

    python -m pytest test/registered/unit/mem_cache/test_unified_mha_views.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.layout.page_major import (
    ENTRY_ALIGN_BYTES,
    build_dense_views,
    mha_entry_bytes,
    paged_view,
)
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
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
# offset is hand-checkable. One entry = 2 layers x (K row + V row) = 64 B.
_L = 2
_H = 2
_D = 4
_DTYPE = torch.bfloat16
_ITEM = _DTYPE.itemsize
_K_ROW = _H * _D * _ITEM
_V_ROW = _H * _D * _ITEM
_ENTRY = mha_entry_bytes(
    layer_num=_L, head_num=_H, head_dim=_D, v_head_dim=_D, itemsize=_ITEM
)
assert _ENTRY == _L * (_K_ROW + _V_ROW)  # no alignment pad at this geometry


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


def _make_raw(ps, num_pages, entry=_ENTRY, short=0):
    n = num_pages * ps * entry - short
    return torch.zeros(n, dtype=torch.uint8, device=_DEV)


def _build_views(raw, ps, num_pages, head_dim=_D, v_head_dim=None, layer_num=_L):
    layout = _mha_spec(head_dim, v_head_dim, layer_num).layout()
    kw = dict(layout=layout, page_size=ps, num_pages=num_pages)
    return (
        build_dense_views(raw, part=layout.part("k"), **kw),
        build_dense_views(raw, part=layout.part("v"), **kw),
    )


def _reference_paged_views(
    raw, *, page_size, num_pages, anchor_bytes=0, head_dim=_D, v_head_dim=_D
):
    """Independent 4-D description of the token-major envelope: per-layer
    ``(num_pages, page_size, head_num, head_dim)`` views addressed by
    ``(page, slot)``, so the flat builder's addressing can be cross-checked
    against a second, independently derived description of the same bytes."""
    k_row = _H * head_dim * _ITEM
    v_row = _H * v_head_dim * _ITEM
    entry = mha_entry_bytes(
        layer_num=_L,
        head_num=_H,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        itemsize=_ITEM,
    )
    as_dtype_view = raw.view(_DTYPE)
    k_views, v_views = [], []
    for layer in range(_L):
        k_base = anchor_bytes + layer * (k_row + v_row)
        v_base = k_base + k_row
        parts = ((k_base, head_dim, k_views), (v_base, v_head_dim, v_views))
        for base, dim, out in parts:
            out.append(
                torch.as_strided(
                    as_dtype_view,
                    size=(num_pages, page_size, _H, dim),
                    stride=(page_size * entry // _ITEM, entry // _ITEM, dim, 1),
                    storage_offset=base // _ITEM,
                )
            )
    return k_views, v_views


class TestMHASpecSurface(unittest.TestCase):
    def test_layout_parts_match_spec_offsets(self):
        """The spec's offset helpers and its layout descriptor are two
        derivations of the entry; they must agree, and the page size never
        enters (a page is page_size entries back to back)."""
        spec = _mha_spec()
        layout = spec.layout()
        self.assertEqual(layout.entry_bytes, spec.entry_bytes())
        for l in range(_L):
            self.assertEqual(spec.layer_k_offset_in_entry(l), l * (_K_ROW + _V_ROW))
            self.assertEqual(
                spec.layer_v_offset_in_entry(l), l * (_K_ROW + _V_ROW) + _K_ROW
            )
            self.assertEqual(
                layout.part("k").layer_offset_bytes(l), spec.layer_k_offset_in_entry(l)
            )
            self.assertEqual(
                layout.part("v").layer_offset_bytes(l), spec.layer_v_offset_in_entry(l)
            )

    def test_entry_bytes_matches_layout_helper_and_is_aligned(self):
        spec = _mha_spec()
        self.assertEqual(
            spec.entry_bytes(),
            mha_entry_bytes(
                layer_num=_L, head_num=_H, head_dim=_D, v_head_dim=_D, itemsize=_ITEM
            ),
        )
        # 48 B of rows round up to one 64 B entry; the parts still fit inside.
        padded = MHASubPoolSpec(
            name="full",
            layer_num=1,
            head_num=1,
            head_dim=16,
            v_head_dim=8,
            store_dtype=_DTYPE,
            grow_direction="down",
        )
        self.assertEqual(padded.entry_bytes(), ENTRY_ALIGN_BYTES * 2)
        self.assertEqual(padded.entry_bytes() % ENTRY_ALIGN_BYTES, 0)
        padded.layout()  # validates

    def test_asymmetric_rows_are_admitted(self):
        """K and V are two parts of one entry at their own offsets; they no
        longer have to be the same kind of row (the MiMoV2 shape, scaled)."""
        spec = _mha_spec(head_dim=8, v_head_dim=4)
        layout = spec.layout()
        self.assertEqual(layout.part("k").row_bytes(), _H * 8 * _ITEM)
        self.assertEqual(layout.part("v").row_bytes(), _H * 4 * _ITEM)
        self.assertEqual(layout.part("v").offset_bytes, _H * 8 * _ITEM)

    def test_misaligned_rows_fail_loud(self):
        """A row that is not a multiple of 16 B would break the vector stores
        every write kernel issues, so the layout refuses it at construction."""
        with self.assertRaises(AssertionError):
            _mha_spec(head_dim=6, v_head_dim=6).layout()


class TestMHAViews(unittest.TestCase):
    def test_view_shapes_and_strides(self):
        ps, num_pages = 4, 6
        k_views, v_views = _build_views(_make_raw(ps, num_pages), ps, num_pages)
        n_rows = num_pages * ps
        self.assertEqual(len(k_views), _L)
        self.assertEqual(len(v_views), _L)
        for v in (*k_views, *v_views):
            # The stock MHATokenToKVPool per-layer signature: 3-D, one row per
            # physical token, the slot stride being the whole entry.
            self.assertEqual(tuple(v.shape), (n_rows, _H, _D))
            self.assertEqual(v.stride(), (_ENTRY // _ITEM, _D, 1))
        for l in range(_L):
            self.assertEqual(
                (v_views[l].storage_offset() - k_views[l].storage_offset()) * _ITEM,
                _K_ROW,
            )

    def test_addressing_matches_paged_reference(self):
        """Cross-readback: bytes written through the reference (page, slot)
        views must be read back through the flat views at t = page*ps+slot,
        for both K and V of every layer — and vice versa."""
        for ps in (1, 4):
            num_pages = 5
            raw = _make_raw(ps, num_pages)
            sk, sv = _reference_paged_views(raw, page_size=ps, num_pages=num_pages)
            dk, dv = _build_views(raw, ps, num_pages)
            probes = [(0, 0, 0), (1, 1, ps - 1), (4, 0, ps // 2), (3, 1, 0)]
            for p, l, s in probes:
                t = p * ps + s
                sk[l][p, s] = float(p * 100 + l * 10 + s + 1)
                sv[l][p, s] = float(p * 100 + l * 10 + s + 2)
                self.assertTrue(
                    torch.all(dk[l][t] == float(p * 100 + l * 10 + s + 1)),
                    f"K (p={p}, l={l}, s={s}, ps={ps}) view readback off-formula",
                )
                self.assertTrue(
                    torch.all(dv[l][t] == float(p * 100 + l * 10 + s + 2)),
                    f"V (p={p}, l={l}, s={s}, ps={ps}) view readback off-formula",
                )
            for p, l, s in probes:
                t = p * ps + s
                dk[l][t] = float(p * 100 + l * 10 + s + 3)
                dv[l][t] = float(p * 100 + l * 10 + s + 4)
                want_k = float(p * 100 + l * 10 + s + 3)
                want_v = float(p * 100 + l * 10 + s + 4)
                self.assertTrue(torch.all(sk[l][p, s] == want_k))
                self.assertTrue(torch.all(sv[l][p, s] == want_v))

    def test_byte_addresses_match_envelope_formula(self):
        """The per-layer view's byte address for token ``t``, layer ``l`` must
        equal the hand-computed envelope formula. Independent of any view
        builder — this is the raw layout contract every envelope consumer
        (moves, sizing, transfer math) relies on."""
        for ps in (1, 4):
            num_pages = 5
            dk, dv = _build_views(_make_raw(ps, num_pages), ps, num_pages)
            for t in (0, 1, ps, 3 * ps + (ps - 1), 4 * ps):
                for l in range(_L):
                    expected_k = t * _ENTRY + l * (_K_ROW + _V_ROW)
                    expected_v = expected_k + _K_ROW
                    got_k = (dk[l].storage_offset() + t * dk[l].stride(0)) * _ITEM
                    got_v = (dv[l].storage_offset() + t * dv[l].stride(0)) * _ITEM
                    self.assertEqual(got_k, expected_k, f"K t={t} l={l} ps={ps}")
                    self.assertEqual(got_v, expected_v, f"V t={t} l={l} ps={ps}")

    def test_k_and_v_share_one_id_without_aliasing(self):
        """One id, 2L distinct cells (K and V of every layer): writes through
        all 2L views at the SAME id must not clobber each other."""
        ps, num_pages = 4, 4
        dk, dv = _build_views(_make_raw(ps, num_pages), ps, num_pages)
        t = 2 * ps + 1  # page 2, slot 1
        for l in range(_L):
            dk[l][t] = float(2 * l + 1)
            dv[l][t] = float(2 * l + 2)
        for l in range(_L):
            self.assertTrue(torch.all(dk[l][t] == float(2 * l + 1)))
            self.assertTrue(torch.all(dv[l][t] == float(2 * l + 2)))

    def test_asymmetric_views_address_their_own_rows(self):
        head_dim, v_head_dim = 8, 4
        k_row, v_row = _H * head_dim * _ITEM, _H * v_head_dim * _ITEM
        entry = mha_entry_bytes(
            layer_num=_L,
            head_num=_H,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            itemsize=_ITEM,
        )
        ps, num_pages = 4, 3
        raw = _make_raw(ps, num_pages, entry=entry)
        dk, dv = _build_views(raw, ps, num_pages, head_dim, v_head_dim)
        self.assertEqual(tuple(dk[0].shape[1:]), (_H, head_dim))
        self.assertEqual(tuple(dv[0].shape[1:]), (_H, v_head_dim))
        t = 2 * ps + 3
        for l in range(_L):
            dk[l][t] = float(l + 1)
            dv[l][t] = float(l + 11)
        flat = raw.view(_DTYPE)
        for l in range(_L):
            k0 = (t * entry + l * (k_row + v_row)) // _ITEM
            v0 = k0 + k_row // _ITEM
            self.assertTrue(torch.all(flat[k0 : k0 + k_row // _ITEM] == float(l + 1)))
            self.assertTrue(torch.all(flat[v0 : v0 + v_row // _ITEM] == float(l + 11)))

    def test_views_fill_the_buffer_exactly(self):
        """No tail pad: the last view's last byte is the buffer's last byte."""
        ps, num_pages = 2, 4
        raw = _make_raw(ps, num_pages)
        _, dv = _build_views(raw, ps, num_pages)
        last = dv[_L - 1]
        end = (last.storage_offset() + (last.shape[0] - 1) * last.stride(0)) * _ITEM
        self.assertEqual(end + _V_ROW, raw.numel())

    def test_short_buffer_fails_loud(self):
        ps, num_pages = 2, 4
        with self.assertRaises(AssertionError):
            _build_views(_make_raw(ps, num_pages, short=1), ps, num_pages)

    def test_paged_view_regroups_by_page(self):
        ps, num_pages = 4, 3
        dk, _ = _build_views(_make_raw(ps, num_pages), ps, num_pages)
        paged = paged_view(dk[1], ps)
        self.assertEqual(
            tuple(paged.stride()), (ps * _ENTRY // _ITEM, _ENTRY // _ITEM, _D, 1)
        )
        for t in range(num_pages * ps):
            self.assertEqual(paged[t // ps, t % ps].data_ptr(), dk[1][t].data_ptr())


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
    def test_every_mha_sub_pool_is_a_slot_strided_view(self):
        """The unified pool has ONE MHA layout: both sub-pools come back as
        3-D per-layer views whose slot stride is their own entry."""
        for ps in (1, 4):
            pool = _make_pool(ps=ps)
            for name, spec in (("full", _mha_spec()), ("swa", _swa_spec())):
                k, v = pool.mha_views_for(name)
                for t in (k[0], v[0], k[-1], v[-1]):
                    self.assertEqual(t.dim(), 3, f"{name} at ps={ps}")
                    self.assertEqual(t.stride(0) * _ITEM, spec.entry_bytes())

    def test_raw_is_exactly_the_budget(self):
        """The views end at the last slot, so the pool allocates exactly the
        byte budget: no tail pad, nothing to under-allocate."""
        for ps in (1, 4):
            kv = _make_pool(ps)
            self.assertEqual(
                kv._raw.numel(),
                _mha_spec().entry_bytes() * _N_FULL
                + _swa_spec().entry_bytes() * _N_SWA,
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
    def test_size_is_slot_bound(self):
        """`size` drives BOTH the python OOB check and the store kernel's
        device-side size_limit; `size + page_size` must be the view row count."""
        for ps in (1, 4):
            unified_kv, pool_under_test = _make_pool_and_kv(ps)
            n_rows = (unified_kv.max_slots("full") // ps) * ps
            self.assertEqual(pool_under_test.size, n_rows - ps)
            self.assertEqual(pool_under_test.k_buffer[0].shape[0], n_rows)

    def test_stock_write_lands_on_envelope_truth(self):
        """Byte-identity: the pool's stock inherited `set_kv_buffer` at physical
        token ids must produce exactly the bytes that direct writes through
        the reference (page, slot) views produce at the same cells — pins the
        whole write path (loc -> strided view -> raw bytes) end to end."""
        for ps in (1, 4):
            kv, pool = _make_pool_and_kv(ps, device=_STORE_DEV)
            sk, sv = _reference_paged_views(
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
                shape = (len(probes), _H, _D)
                k = torch.full(shape, float(l + 1), dtype=_DTYPE, device=_STORE_DEV)
                v = torch.full(shape, float(l + 101), dtype=_DTYPE, device=_STORE_DEV)
                pool.set_kv_buffer(_layer(l), KVWriteLoc(toks, id_space="kernel"), k, v)
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
        """Compaction hands PHYSICAL token runs. The override must relocate
        exactly the page envelopes those runs name."""
        ps = 4
        kv, pool = _make_pool_and_kv(ps)
        live = kv._raw.numel()
        seed = (torch.arange(live, dtype=torch.float32) % 251).to(torch.uint8)
        kv._raw[:] = seed
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
            torch.equal(kv._raw, want),
            "envelope move did not relocate exactly the named pages",
        )

    def test_transfer_entry_points_fail_loud(self):
        """PD / CPU-copy entry points assume per-layer contiguous buffers;
        against the strided views they would silently mis-index. Every one of
        them must raise."""
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
    """The real SWA factory builds the sub-pools and the composite allocator.
    End-to-end over that factory, kernel-facing ids are the physical ones and
    the rebind must emit BOTH write locs."""

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

    def test_factory_builds_per_layer_views(self):
        b = self._bundle()
        # Sub-pools expose stock 3-D per-layer views.
        self.assertEqual(b.token_to_kv_pool.full_kv_pool.k_buffer[0].dim(), 3)
        self.assertEqual(b.token_to_kv_pool.swa_kv_pool.k_buffer[0].dim(), 3)

    def test_rebind_emits_physical_full_and_build_derives_swa(self):
        """End-to-end over the real factory: rebind_write_loc rebinds
        out_cache_loc to FULL-side physical ids (phase 1), and the per-batch
        build derives the SWA write loc pointwise from those values (phase 2)
        — both checked against the v2p tables over the VIRTUAL ids."""
        from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator

        b = self._bundle()
        alloc = b.token_to_kv_pool_allocator
        v = alloc.alloc(4)
        self.assertIsNotNone(v)
        expected_full = alloc.full_v2p_page_table[v]  # ps=1
        expected_swa = alloc.swa_v2p_page_table[v]

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
