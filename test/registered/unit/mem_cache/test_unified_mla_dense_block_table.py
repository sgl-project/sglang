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
"""DENSE block table / kv_indices for the paged MLA backends under the unified
memory pool (Kimi-Linear).

`req_to_token` holds VIRTUAL token ids, while the per-layer MLA views are dense
(`build_dense_mla_views`). The paged MLA backends therefore need their page-level
block table filled with DENSE page ids:

    dense_page(virtual_page) = v2p[virtual_page] * layer_num

Three backend families reach that same formula by different routes:
  - `create_flashmla_kv_indices_triton` in-kernel via `v2p_ptr` / `PAGE_MULT`
    (trtllm_mla / cutedsl_mla / tokenspeed_mla);
  - the flashinfer_mla updaters, post-gathering `translate_kv_loc_dense` over the
    token-level kv_indices;
  - `normal_decode_set_metadata` in-kernel, for fa3's captured-decode page table.

Covered here:
  - kernel identity: `v2p_ptr=None, PAGE_MULT=1` is byte-identical to main;
  - kernel dense mapping against the python reference, for several page sizes,
    ragged sequence lengths and a non-identity v2p permutation;
  - padded block-table lanes never index the v2p table out of bounds;
  - the token-level dense translate the flashinfer updaters apply agrees with the
    page-level block table the trtllm path builds;
  - fa3's fused metadata kernels agree with the same reference, on both the
    page_size == 1 fast path (which is what Kimi-Linear takes: fa3 imposes no
    page-size constraint) and the general path.

    python -m pytest test/registered/unit/mem_cache/test_unified_mla_dense_block_table.py -v
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

_HAS_CUDA = torch.cuda.is_available()
_DEV = "cuda"
_LAYERS = 24  # K3 MLA full-attention layer count


def _fill_block_table(
    req_to_token, req_pool_indices, seq_lens, page_size, *, v2p, mult
):
    from sglang.kernels.ops.kvcache.kv_indices import (
        create_flashmla_kv_indices_triton,
        get_num_kv_index_blocks_flashmla,
    )

    bs = req_pool_indices.shape[0]
    max_blocks = (int(seq_lens.max().item()) + page_size - 1) // page_size
    out = torch.full((bs, max_blocks), -1, dtype=torch.int32, device=_DEV)
    grid = (bs, get_num_kv_index_blocks_flashmla(max_blocks, page_size))
    create_flashmla_kv_indices_triton[grid](
        req_to_token,
        req_pool_indices,
        seq_lens,
        None,
        out,
        req_to_token.stride(0),
        max_blocks,
        PAGED_SIZE=page_size,
        v2p_ptr=v2p,
        PAGE_MULT=mult,
    )
    return out


def _reference(req_to_token, req_pool_indices, seq_lens, page_size, *, v2p, mult):
    """Python reference: virtual token -> virtual page -> physical page -> dense."""
    bs = req_pool_indices.shape[0]
    max_blocks = (int(seq_lens.max().item()) + page_size - 1) // page_size
    ref = torch.full((bs, max_blocks), -1, dtype=torch.int64, device=_DEV)
    for r in range(bs):
        n_pages = (int(seq_lens[r].item()) + page_size - 1) // page_size
        row = req_to_token[int(req_pool_indices[r].item())]
        virt_pages = row[: n_pages * page_size : page_size] // page_size
        pages = v2p[virt_pages.long()] if v2p is not None else virt_pages.long()
        ref[r, :n_pages] = pages * mult
    return ref


@unittest.skipUnless(_HAS_CUDA, "requires CUDA")
class TestDenseBlockTable(unittest.TestCase):
    def _make_batch(self, page_size, bs=5, max_ctx=2048, n_pages=512):
        """Ragged batch with a non-identity virtual->physical page permutation."""
        g = torch.Generator(device="cpu").manual_seed(97 + page_size)
        seq_lens = torch.tensor(
            [page_size, page_size + 1, 3 * page_size, 7 * page_size - 3, 1],
            dtype=torch.int32,
            device=_DEV,
        )[:bs]
        req_to_token = torch.zeros((bs, max_ctx), dtype=torch.int32, device=_DEV)
        # Every request gets a distinct, non-monotonic run of virtual pages.
        virt_page_perm = torch.randperm(n_pages - 1, generator=g)[: bs * 8] + 1
        for r in range(bs):
            n = int(seq_lens[r].item())
            pages = virt_page_perm[r * 8 : r * 8 + 8].to(_DEV)
            toks = (
                pages[:, None] * page_size
                + torch.arange(page_size, device=_DEV)[None, :]
            ).reshape(-1)
            req_to_token[r, :n] = toks[:n].to(torch.int32)
        req_pool_indices = torch.arange(bs, dtype=torch.int32, device=_DEV)
        # Non-identity page-level v2p, with a tombstone that no request references.
        v2p = torch.randperm(n_pages, generator=g).to(_DEV).to(torch.int64)
        v2p[0] = 0  # page 0 is the reserved sink
        return req_to_token, req_pool_indices, seq_lens, v2p

    def test_identity_when_hooks_absent(self):
        """v2p_ptr=None / PAGE_MULT=1 must reproduce the pre-change behaviour."""
        for page_size in (1, 32, 64):
            rt, rpi, sl, _ = self._make_batch(page_size)
            got = _fill_block_table(rt, rpi, sl, page_size, v2p=None, mult=1)
            want = _reference(rt, rpi, sl, page_size, v2p=None, mult=1)
            self.assertTrue(
                torch.equal(got.long(), want), f"page_size={page_size}: {got} != {want}"
            )

    def test_dense_block_table_matches_reference(self):
        for page_size in (1, 32, 64):
            rt, rpi, sl, v2p = self._make_batch(page_size)
            got = _fill_block_table(rt, rpi, sl, page_size, v2p=v2p, mult=_LAYERS)
            want = _reference(rt, rpi, sl, page_size, v2p=v2p, mult=_LAYERS)
            self.assertTrue(
                torch.equal(got.long(), want),
                f"page_size={page_size}:\ngot ={got}\nwant={want}",
            )

    def test_single_full_attention_layer_still_maps_v2p(self):
        """A config with exactly ONE full-attention layer (e.g. a PP rank owning a
        single MLA layer) has `kernel_page_multiplier == 1`, but its req_to_token
        still holds VIRTUAL ids. The dense id collapses onto the physical id, so
        the v2p gather alone IS the whole translation -- it must not be skipped.

        Regression guard for detecting the unified pool via `multiplier > 1`:
        that predicate treats this config as a static pool and leaves the block
        table in virtual id space.
        """
        for page_size in (1, 64):
            rt, rpi, sl, v2p = self._make_batch(page_size)
            got = _fill_block_table(rt, rpi, sl, page_size, v2p=v2p, mult=1).long()
            want = _reference(rt, rpi, sl, page_size, v2p=v2p, mult=1)
            self.assertTrue(torch.equal(got, want), f"page_size={page_size}")
            # ... and the v2p permutation is non-trivial here, so a skipped
            # translation would be visibly different rather than accidentally equal.
            virtual = _reference(rt, rpi, sl, page_size, v2p=None, mult=1)
            self.assertFalse(
                torch.equal(want, virtual),
                "test batch degenerated: v2p is the identity on the pages used",
            )

    def test_padded_lanes_stay_untouched(self):
        """Lanes past a request's page count keep the -1 fill: the masked v2p load
        must not write a translated value (nor read out of bounds)."""
        page_size = 64
        rt, rpi, sl, v2p = self._make_batch(page_size)
        got = _fill_block_table(rt, rpi, sl, page_size, v2p=v2p, mult=_LAYERS)
        for r in range(got.shape[0]):
            n_pages = (int(sl[r].item()) + page_size - 1) // page_size
            self.assertTrue(
                torch.all(got[r, n_pages:] == -1),
                f"row {r} padded lanes were written: {got[r]}",
            )

    def test_agrees_with_token_level_dense_translate(self):
        """The flashinfer updaters translate TOKEN ids with
        `translate_kv_loc_dense`; the trtllm path builds PAGE ids in-kernel. Both
        must address the same dense page block."""
        page_size = 64
        rt, rpi, sl, v2p = self._make_batch(page_size)
        block_table = _fill_block_table(
            rt, rpi, sl, page_size, v2p=v2p, mult=_LAYERS
        ).long()
        for r in range(rt.shape[0]):
            n = int(sl[r].item())
            virt_tokens = rt[r, :n].long()
            # translate_kv_loc_dense's formula, applied to token ids.
            dense_tokens = (
                v2p[virt_tokens // page_size] * (page_size * _LAYERS)
                + virt_tokens % page_size
            )
            # The block-table entry scaled by page_size must be the dense id of
            # each page's first token.
            first_of_page = dense_tokens[::page_size]
            n_pages = (n + page_size - 1) // page_size
            self.assertTrue(
                torch.equal(block_table[r, :n_pages] * page_size, first_of_page),
                f"row {r}: block table and token translate disagree",
            )


@unittest.skipUnless(_HAS_CUDA, "requires CUDA")
class TestFa3MetadataDenseBlockTable(unittest.TestCase):
    """fa3 folds the unified remap into `normal_decode_set_metadata`, the fused
    gather that writes its captured-decode page table, so the kernel itself has
    to get the mapping right. Two kernels back it: a page_size == 1 / no-SWA fast
    path (what Kimi-Linear takes, since fa3 imposes no page-size constraint) and
    a general one.
    """

    def _run(self, page_size, *, v2p, mult, bs=5, max_ctx=2048):
        from sglang.kernels.ops.attention.metadata import normal_decode_set_metadata

        maker = TestDenseBlockTable._make_batch
        rt, rpi, sl, v2p_full = maker(self, page_size, bs=bs, max_ctx=max_ctx)
        v2p_arg = v2p_full if v2p else None

        max_pages = (max_ctx + page_size - 1) // page_size
        page_table = torch.zeros((bs, max_pages), dtype=torch.int32, device=_DEV)
        cache_seqlens = torch.zeros((bs,), dtype=torch.int32, device=_DEV)
        cu_seqlens_k = torch.zeros((bs + 1,), dtype=torch.int32, device=_DEV)
        strided = torch.arange(0, max_ctx, page_size, device=_DEV)
        max_seq_pages = (int(sl.max().item()) + page_size - 1) // page_size

        normal_decode_set_metadata(
            cache_seqlens,
            cu_seqlens_k,
            page_table,
            rt,
            rpi,
            strided,
            max_seq_pages,
            sl.to(torch.int64),
            0,
            page_size,
            v2p_page_table=v2p_arg,
            kernel_page_multiplier=mult,
        )
        torch.cuda.synchronize()
        want = _reference(rt, rpi, sl, page_size, v2p=v2p_arg, mult=mult)
        return page_table, want, sl

    def _assert_live_prefix(self, got, want, sl, page_size):
        """The kernel contract only (re)writes each row's live page prefix; the
        tail keeps stale values that consumers bound by cache_seqlens."""
        for r in range(got.shape[0]):
            n_pages = (int(sl[r].item()) + page_size - 1) // page_size
            self.assertTrue(
                torch.equal(got[r, :n_pages].long(), want[r, :n_pages]),
                f"row {r} (page_size={page_size}):\n"
                f"got ={got[r, :n_pages]}\nwant={want[r, :n_pages]}",
            )

    def test_identity_when_hooks_absent(self):
        """Static pool: no v2p, multiplier 1 -> byte-identical to pre-change."""
        for page_size in (1, 64):
            got, want, sl = self._run(page_size, v2p=False, mult=1)
            self._assert_live_prefix(got, want, sl, page_size)

    def test_dense_mapping_ps1_fast_path(self):
        got, want, sl = self._run(1, v2p=True, mult=_LAYERS)
        self._assert_live_prefix(got, want, sl, 1)

    def test_dense_mapping_general_path(self):
        got, want, sl = self._run(64, v2p=True, mult=_LAYERS)
        self._assert_live_prefix(got, want, sl, 64)

    def test_single_full_attention_layer(self):
        """multiplier 1 with a real v2p: the gather alone is the translation."""
        for page_size in (1, 64):
            got, want, sl = self._run(page_size, v2p=True, mult=1)
            self._assert_live_prefix(got, want, sl, page_size)
            virtual = _reference(
                *TestDenseBlockTable._make_batch(self, page_size)[:3],
                page_size,
                v2p=None,
                mult=1,
            )
            self.assertFalse(
                torch.equal(want, virtual),
                "test batch degenerated: v2p is the identity on the pages used",
            )

    def test_agrees_with_flashmla_block_table(self):
        """fa3 and trtllm_mla build the same table two different ways; a
        disagreement means one family is addressing the wrong pages."""
        for page_size in (1, 64):
            got, _, sl = self._run(page_size, v2p=True, mult=_LAYERS)
            rt, rpi, sl2, v2p = TestDenseBlockTable._make_batch(self, page_size)
            other = _fill_block_table(
                rt, rpi, sl2, page_size, v2p=v2p, mult=_LAYERS
            ).long()
            for r in range(got.shape[0]):
                n_pages = (int(sl[r].item()) + page_size - 1) // page_size
                self.assertTrue(
                    torch.equal(got[r, :n_pages].long(), other[r, :n_pages]),
                    f"fa3 and flashmla block tables disagree (row {r}, ps={page_size})",
                )


class TestUnifiedMLAHookDetection(unittest.TestCase):
    """`unified_mla_hooks` decides whether the paged MLA backends translate at
    all. Getting the predicate wrong is silent: the block table and KV write loc
    stay in virtual id space and address the wrong pages once virtual and
    physical diverge (e.g. after compaction)."""

    @staticmethod
    def _probe(**attrs):
        from sglang.srt.layers.attention.unified_mem_hooks import (
            unified_mla_hooks,
        )

        class _Alloc:
            pass

        alloc = _Alloc()
        for k, v in attrs.items():
            setattr(alloc, k, v)
        return unified_mla_hooks(alloc)

    def test_static_pool_disables_every_hook(self):
        """No v2p table -> statically-partitioned pool; req_to_token is already
        physical, so all hooks must stay off (byte-identical to pre-change)."""
        hooks = self._probe()
        self.assertFalse(hooks.enabled)
        self.assertIsNone(hooks.v2p_page_table)
        self.assertIsNone(hooks.translate_kv_loc_dense)
        self.assertEqual(hooks.kernel_page_multiplier, 1)

    def test_multi_layer_unified_pool(self):
        table = torch.arange(8)
        hooks = self._probe(
            full_v2p_page_table=table,
            translate_kv_loc_dense=lambda x, **kw: x,
            kernel_page_multiplier=_LAYERS,
        )
        self.assertTrue(hooks.enabled)
        self.assertIs(hooks.v2p_page_table, table)
        self.assertIsNotNone(hooks.translate_kv_loc_dense)
        self.assertEqual(hooks.kernel_page_multiplier, _LAYERS)

    def test_single_full_attention_layer_pool_is_still_unified(self):
        """REGRESSION: `kernel_page_multiplier == 1` does NOT mean static.

        A hybrid MLA config with exactly one full-attention layer (e.g. a
        pipeline-parallel rank owning a single MLA layer) has multiplier 1, yet
        its locs are still virtual. Detecting on `multiplier > 1` would disable
        the v2p gather here and corrupt reads/writes after compaction.
        """
        table = torch.arange(8)
        hooks = self._probe(
            full_v2p_page_table=table,
            translate_kv_loc_dense=lambda x, **kw: x,
            kernel_page_multiplier=1,
        )
        self.assertTrue(hooks.enabled, "single-layer unified pool read as static")
        self.assertIs(hooks.v2p_page_table, table)
        self.assertIsNotNone(hooks.translate_kv_loc_dense)
        # Multiplier stays 1: dense id == physical id, so the v2p gather alone is
        # the whole translation and PAGE_MULT must not scale it.
        self.assertEqual(hooks.kernel_page_multiplier, 1)


@unittest.skipUnless(_HAS_CUDA, "requires CUDA")
class TestInPlaceKvIndicesTranslate(unittest.TestCase):
    """The flashinfer decode updater must translate kv_indices IN PLACE.

    Under cuda-graph replay the `kv_indices` it is handed IS the capture-stable
    buffer the captured wrapper reads (`fast_decode_kwargs["kv_indices"]`), and
    `fast_mla_decode_plan` ignores its `kv_indices` argument -- so rebinding the
    local name to a fresh translated tensor leaves the graph reading VIRTUAL ids.
    These pin the write-back contract that fix relies on.
    """

    def _allocator(self, page_size=1, n_full_tokens=4096):
        from sglang.srt.mem_cache.multi_ended_allocator import MultiEndedAllocator
        from sglang.srt.mem_cache.unified_memory_pool import (
            MambaSubPoolSpec,
            MLASubPoolSpec,
            UnifiedKVPool,
        )

        full = MLASubPoolSpec(
            name="full",
            layer_num=_LAYERS,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            store_dtype=torch.bfloat16,
            grow_direction="down",
        )
        mamba = MambaSubPoolSpec(
            name="mamba",
            layer_num=2,
            conv_state_shapes=((8, 16),),
            conv_dtype=torch.bfloat16,
            temporal_state_shape=(4, 8, 8),
            temporal_dtype=torch.float32,
            grow_direction="up",
        )
        pool = UnifiedKVPool(
            total_bytes=full.entry_bytes() * n_full_tokens + mamba.entry_bytes() * 16,
            sub_pool_specs=[full, mamba],
            device=_DEV,
            enable_memory_saver=False,
            page_size=page_size,
            view_tail_pad_bytes=page_size * full.entry_bytes(),
        )

        class _Stub:
            def move_kv_cache(self, dst, src):
                pass

        full_alloc = MultiEndedAllocator(
            kvcache=_Stub(),
            unified_buffer=pool,
            sub_pool_name="full",
            device=_DEV,
            is_id_owner=True,
            page_size=page_size,
            kernel_page_multiplier=_LAYERS,
        )
        mamba_alloc = MultiEndedAllocator(
            kvcache=_Stub(),
            unified_buffer=pool,
            sub_pool_name="mamba",
            device=_DEV,
            is_id_owner=True,
        )
        full_alloc.bind_peer(mamba_alloc)
        mamba_alloc.bind_peer(full_alloc)
        return full_alloc

    def test_int32_buffer_prefix_translated_tail_untouched(self):
        """Mirrors the updater: an int32 capture-stable buffer holding VIRTUAL
        ids in [:n] gets the dense ids written back in place, narrowed to int32,
        with the stale tail left alone (it must never index the v2p table)."""
        alloc = self._allocator()
        virt = alloc.alloc(64)
        self.assertIsNotNone(virt)
        n = virt.numel()

        # Capture-stable int32 buffer: [:n] freshly filled with virtual ids by
        # create_flashinfer_kv_indices_triton, tail = stale junk from a bigger replay.
        buf = torch.full((n * 3,), 2**30, dtype=torch.int32, device=_DEV)
        buf[:n] = virt.to(torch.int32)
        tail_before = buf[n:].clone()

        valid = buf[:n]
        valid.copy_(alloc.translate_kv_loc_dense(valid))

        expected = alloc.translate_kv_loc_dense(virt)
        self.assertEqual(buf.dtype, torch.int32)
        self.assertTrue(
            torch.equal(buf[:n].long(), expected),
            "in-place translate did not land dense ids in the stable buffer",
        )
        self.assertTrue(
            torch.equal(buf[n:], tail_before),
            "stale tail was modified -- it can hold ids outside the v2p table",
        )

    def test_dense_ids_differ_from_virtual(self):
        """Guard the guard: if dense == virtual the in-place test proves nothing."""
        alloc = self._allocator()
        virt = alloc.alloc(64)
        self.assertIsNotNone(virt)
        self.assertFalse(
            torch.equal(alloc.translate_kv_loc_dense(virt), virt),
            "dense ids coincide with virtual ids; pick a different allocation",
        )


if __name__ == "__main__":
    unittest.main()
