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
"""Block table / kv_indices for the paged MLA backends under the unified
memory pool (Kimi-Linear).

`req_to_token` holds VIRTUAL token ids, while the per-layer MLA views are
contiguous (`build_mla_views`). The paged MLA backends therefore need their
page-level block table filled with kernel-facing page ids:

    kernel_page(virtual_page) = v2p[virtual_page] * layer_num

Since the read-path translator, ONE builder computes that formula for every
family — `build_index_table` (the canonical) — and the backends only
differ in how they consume it:
  - trtllm_mla / cutedsl_mla / tokenspeed_mla / flashmla: rows filled straight
    into their padded block tables (`KVIndexTranslator.fill_read_table`, prefix-only so
    the backends' own -1 / stale tail sentinels survive);
  - the flashinfer updaters: token ids reconstructed from the canonical by
    `create_flashinfer_kv_indices_triton[ENTRY_PAGE_SIZE=ps]`;
  - fa3's captured decode: `normal_decode_set_metadata` copies the canonical
    rows' live prefixes (src_is_read_table=True).

Covered here:
  - the static `create_flashmla_kv_indices_triton` (no id-space knowledge left)
    still matches the plain token//ps reference;
  - the canonical route against the python reference, for several page
    sizes, ragged sequence lengths and a non-identity v2p permutation;
  - lanes past a row's live prefix keep the backend's -1 sentinel (prefix-only
    discipline — the trtllm/flashmla tail contract);
  - the token-level kernel-facing translate the flashinfer updaters used to apply
    agrees with the canonical page table (page-affinity of the id space);
  - fa3's fused metadata kernels agree with the same reference, on both the
    page_size == 1 fast path (which is what Kimi-Linear takes: fa3 imposes no
    page-size constraint) and the general path.

    python -m pytest test/registered/unit/mem_cache/test_unified_mla_block_table.py -v
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
    """The unified route: canonical builder into a -1-filled block table
    (exactly what KVIndexTranslator.build_into does for trtllm_mla/flashmla)."""
    from sglang.kernels.ops.kvcache.kv_read_table import build_kv_read_table

    bs = req_pool_indices.shape[0]
    max_blocks = (int(seq_lens.max().item()) + page_size - 1) // page_size
    out = torch.full((bs, max_blocks), -1, dtype=torch.int32, device=_DEV)
    build_kv_read_table(
        req_to_token=req_to_token,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens.to(torch.int64),
        v2p=v2p,
        multiplier=mult,
        page_size=page_size,
        max_pages=max_blocks,
        out=out,
    )
    return out


def _fill_block_table_static(req_to_token, req_pool_indices, seq_lens, page_size):
    """The static-pool route: the stripped flashmla kernel, token//ps verbatim."""
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
    )
    return out


def _reference(req_to_token, req_pool_indices, seq_lens, page_size, *, v2p, mult):
    """Python reference: virtual token -> virtual page -> physical page -> kernel id."""
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
class TestBlockTable(unittest.TestCase):
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

    def test_static_kernel_matches_reference(self):
        """The stripped (id-space-free) flashmla kernel is byte-identical to the
        plain token//ps reference -- guards the v2p-arg removal itself."""
        for page_size in (1, 32, 64):
            rt, rpi, sl, _ = self._make_batch(page_size)
            got = _fill_block_table_static(rt, rpi, sl, page_size)
            want = _reference(rt, rpi, sl, page_size, v2p=None, mult=1)
            self.assertTrue(
                torch.equal(got.long(), want), f"page_size={page_size}: {got} != {want}"
            )

    def test_block_table_matches_reference(self):
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
        still holds VIRTUAL ids. The kernel-facing id collapses onto the physical id, so
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
        """Lanes past a request's page count keep the -1 fill: the prefix-only
        canonical build must never write a backend's tail sentinel (the
        trtllm/flashmla block-table contract)."""
        page_size = 64
        rt, rpi, sl, v2p = self._make_batch(page_size)
        got = _fill_block_table(rt, rpi, sl, page_size, v2p=v2p, mult=_LAYERS)
        for r in range(got.shape[0]):
            n_pages = (int(sl[r].item()) + page_size - 1) // page_size
            self.assertTrue(
                torch.all(got[r, n_pages:] == -1),
                f"row {r} padded lanes were written: {got[r]}",
            )

    def test_agrees_with_token_level_translate(self):
        """The flashinfer updaters translate TOKEN ids with
        `translate_kv_loc_for_kernel`; the trtllm path builds PAGE ids in-kernel. Both
        must address the same kernel-facing page block."""
        page_size = 64
        rt, rpi, sl, v2p = self._make_batch(page_size)
        block_table = _fill_block_table(
            rt, rpi, sl, page_size, v2p=v2p, mult=_LAYERS
        ).long()
        for r in range(rt.shape[0]):
            n = int(sl[r].item())
            virt_tokens = rt[r, :n].long()
            # translate_kv_loc_for_kernel's formula, applied to token ids.
            kernel_tokens = (
                v2p[virt_tokens // page_size] * (page_size * _LAYERS)
                + virt_tokens % page_size
            )
            # The block-table entry scaled by page_size must be the kernel-facing id of
            # each page's first token.
            first_of_page = kernel_tokens[::page_size]
            n_pages = (n + page_size - 1) // page_size
            self.assertTrue(
                torch.equal(block_table[r, :n_pages] * page_size, first_of_page),
                f"row {r}: block table and token translate disagree",
            )


@unittest.skipUnless(_HAS_CUDA, "requires CUDA")
class TestFa3MetadataBlockTable(unittest.TestCase):
    """fa3's captured-decode page table is written by `normal_decode_set_metadata`
    fed with the translator's read table kernel page table
    (src_is_read_table=True): the fused kernel copies the canonical
    rows' live prefixes into the capture-stable buffer. Pinned END-TO-END:
    build_kv_read_table -> wrapper -> page_table must equal the python
    reference of the kernel-facing formula, on both the page_size == 1 / no-SWA fast
    path (what Kimi-Linear takes) and the general kernel. The static call
    (no source flag) stays byte-identical to the pre-translator kernel.
    """

    def _run(self, page_size, *, v2p, mult, bs=5, max_ctx=2048):
        from sglang.kernels.ops.attention.metadata import normal_decode_set_metadata
        from sglang.kernels.ops.kvcache.kv_read_table import (
            build_kv_read_table,
        )

        maker = TestBlockTable._make_batch
        rt, rpi, sl, v2p_full = maker(self, page_size, bs=bs, max_ctx=max_ctx)

        max_pages = (max_ctx + page_size - 1) // page_size
        page_table = torch.zeros((bs, max_pages), dtype=torch.int32, device=_DEV)
        cache_seqlens = torch.zeros((bs,), dtype=torch.int32, device=_DEV)
        cu_seqlens_k = torch.zeros((bs + 1,), dtype=torch.int32, device=_DEV)
        max_seq_pages = (int(sl.max().item()) + page_size - 1) // page_size

        if v2p:
            # The translator's canonical, then the wrapper copies its rows.
            canonical = torch.zeros((bs, max_pages), dtype=torch.int32, device=_DEV)
            build_kv_read_table(
                req_to_token=rt,
                req_pool_indices=rpi,
                seq_lens=sl.to(torch.int64),
                v2p=v2p_full,
                multiplier=mult,
                page_size=page_size,
                max_pages=max_pages,
                out=canonical,
            )
            rows = torch.arange(bs, dtype=torch.int64, device=_DEV)
            normal_decode_set_metadata(
                cache_seqlens,
                cu_seqlens_k,
                page_table,
                canonical,
                rows,
                max_seq_pages,
                sl.to(torch.int64),
                0,
                page_size,
                None,
                None,
                src_is_read_table=True,
            )
        else:
            normal_decode_set_metadata(
                cache_seqlens,
                cu_seqlens_k,
                page_table,
                rt,
                rpi,
                max_seq_pages,
                sl.to(torch.int64),
                0,
                page_size,
                None,
                None,
            )
        torch.cuda.synchronize()
        want = _reference(
            rt, rpi, sl, page_size, v2p=(v2p_full if v2p else None), mult=mult
        )
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

    def test_translated_mapping_ps1_fast_path(self):
        got, want, sl = self._run(1, v2p=True, mult=_LAYERS)
        self._assert_live_prefix(got, want, sl, 1)

    def test_translated_mapping_general_path(self):
        got, want, sl = self._run(64, v2p=True, mult=_LAYERS)
        self._assert_live_prefix(got, want, sl, 64)

    def test_single_full_attention_layer(self):
        """multiplier 1 with a real v2p: the gather alone is the translation."""
        for page_size in (1, 64):
            got, want, sl = self._run(page_size, v2p=True, mult=1)
            self._assert_live_prefix(got, want, sl, page_size)
            virtual = _reference(
                *TestBlockTable._make_batch(self, page_size)[:3],
                page_size,
                v2p=None,
                mult=1,
            )
            self.assertFalse(
                torch.equal(want, virtual),
                "test batch degenerated: v2p is the identity on the pages used",
            )

    # (The old fa3<->flashmla agreement case is gone: both families now
    # consume the SAME canonical builder, so cross-family agreement holds by
    # construction and the per-family cases above cover the two consumers.)


if __name__ == "__main__":
    unittest.main()
