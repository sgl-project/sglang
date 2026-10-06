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
token-major (`build_dense_views`), indexed by the physical token id. The paged
MLA backends therefore need their page-level block table filled with physical
page ids:

    kernel_page(virtual_page) = v2p[virtual_page]

ONE builder computes that formula for every family -- the iteration plan's
read table (`KVLocPlan.read_table`) -- and the backends only differ in how
they consume it:
  - trtllm_mla / cutedsl_mla / tokenspeed_mla / flashmla: the plan's rows
    copied into their padded block tables (`KVIndexTranslator.copy_page_table`);
  - the flashinfer updaters: token ids reconstructed from the table by
    `create_flashinfer_kv_indices_triton[ENTRY_PAGE_SIZE=ps]`;
  - fa3's captured decode: the plan's rows copied into its captured page
    table.

Covered here:
  - the static `create_flashmla_kv_indices_triton` (no id-space knowledge left)
    still matches the plain token//ps reference;
  - the canonical route against the python reference, for several page
    sizes, ragged sequence lengths and a non-identity v2p permutation;
  - lanes past a row's live prefix keep the backend's -1 sentinel (prefix-only
    discipline — the trtllm/flashmla tail contract);
  - the token-level physical translate the flashinfer updaters used to apply
    agrees with the canonical page table (page-affinity of the id space);
  - fa3's fused metadata kernels agree with the same reference, on both the
    page_size == 1 fast path (which is what Kimi-Linear takes: fa3 imposes no
    page-size constraint) and the general path.

    python -m pytest test/registered/unit/mem_cache/test_unified_mla_block_table.py -v
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-small")

_HAS_CUDA = torch.cuda.is_available()
_DEV = "cuda"


def _fill_block_table(req_to_token, req_pool_indices, seq_lens, page_size, *, v2p):
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


def _reference(req_to_token, req_pool_indices, seq_lens, page_size, *, v2p):
    """Python reference: virtual token -> virtual page -> physical page -> kernel id."""
    bs = req_pool_indices.shape[0]
    max_blocks = (int(seq_lens.max().item()) + page_size - 1) // page_size
    ref = torch.full((bs, max_blocks), -1, dtype=torch.int64, device=_DEV)
    for r in range(bs):
        n_pages = (int(seq_lens[r].item()) + page_size - 1) // page_size
        row = req_to_token[int(req_pool_indices[r].item())]
        virt_pages = row[: n_pages * page_size : page_size] // page_size
        pages = v2p[virt_pages.long()] if v2p is not None else virt_pages.long()
        ref[r, :n_pages] = pages
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

    def test_seq_len_delta_matches_widened_lens(self):
        """The kernel's ``seq_len_delta`` equals building with ``seq_lens + k``
        (the two spellings of verify widening), and genuinely widens: the
        widened columns overwrite the -1 sentinel the plain build leaves."""
        from sglang.kernels.ops.kvcache.kv_read_table import build_kv_read_table

        for page_size in (1, 32):
            rt, rpi, sl, v2p = self._make_batch(page_size)
            delta = page_size + 1
            max_pages = int((sl.max().item() + delta + page_size - 1) // page_size)

            def fill(seq_lens, seq_len_delta):
                out = torch.full(
                    (rpi.shape[0], max_pages), -1, dtype=torch.int32, device=_DEV
                )
                build_kv_read_table(
                    req_to_token=rt,
                    req_pool_indices=rpi,
                    seq_lens=seq_lens.to(torch.int64),
                    v2p=v2p,
                    page_size=page_size,
                    max_pages=max_pages,
                    out=out,
                    seq_len_delta=seq_len_delta,
                )
                return out

            widened = fill(sl, delta)
            by_lens = fill(sl + delta, 0)
            plain = fill(sl, 0)
            self.assertTrue(torch.equal(widened, by_lens), f"ps={page_size}")
            self.assertFalse(torch.equal(widened, plain), f"ps={page_size}")

    def test_rows_stop_at_the_table_width(self):
        """A widened window longer than the table's row (a verify near the
        context limit) is cut at ``max_pages`` instead of running into the
        next row -- another request's page table -- or past the buffer. The
        CUDA kernel must agree with the CPU path, which clamps per row."""
        from sglang.kernels.ops.kvcache.kv_read_table import build_kv_read_table

        for page_size in (1, 32):
            rt, rpi, sl, v2p = self._make_batch(page_size)
            bs = rpi.shape[0]
            delta = 2 * page_size
            max_pages = 3
            n_pages = (sl + delta + page_size - 1) // page_size
            self.assertTrue(bool((n_pages > max_pages).any()), "no row overflows")

            def fill(device):
                # One guard row past the table catches a spill off the end.
                out = torch.full(
                    (bs + 1, max_pages), -1, dtype=torch.int32, device=device
                )
                build_kv_read_table(
                    req_to_token=rt.to(device),
                    req_pool_indices=rpi.to(device),
                    seq_lens=sl.to(device=device, dtype=torch.int64),
                    v2p=v2p.to(device),
                    page_size=page_size,
                    max_pages=max_pages,
                    out=out[:bs],
                    seq_len_delta=delta,
                )
                return out.cpu()

            gpu, cpu = fill(_DEV), fill("cpu")
            self.assertTrue(torch.all(gpu[bs] == -1), f"ps={page_size}: {gpu}")
            self.assertTrue(
                torch.equal(gpu, cpu), f"ps={page_size}:\ngpu={gpu}\ncpu={cpu}"
            )

    def test_zero_tail_fills_a_fresh_table(self):
        """With ``zero_tail`` a fresh (garbage) table comes back whole in one
        launch: each row's live prefix, the sink past it up to ``max_pages``,
        and nothing past ``max_pages``. The CUDA kernel must agree with the
        CPU path."""
        from sglang.kernels.ops.kvcache.kv_read_table import build_kv_read_table

        for page_size in (1, 32):
            rt, rpi, sl, v2p = self._make_batch(page_size)
            bs = rpi.shape[0]
            max_pages = int((sl.max().item() + page_size - 1) // page_size) + 2

            def fill(device):
                # One guard column past the table catches a spill off the row.
                out = torch.full(
                    (bs, max_pages + 1), 7, dtype=torch.int32, device=device
                )
                build_kv_read_table(
                    req_to_token=rt.to(device),
                    req_pool_indices=rpi.to(device),
                    seq_lens=sl.to(device=device, dtype=torch.int64),
                    v2p=v2p.to(device),
                    page_size=page_size,
                    max_pages=max_pages,
                    out=out,
                    zero_tail=True,
                )
                return out.cpu()

            gpu, cpu = fill(_DEV), fill("cpu")
            self.assertTrue(torch.equal(gpu, cpu), f"ps={page_size}")
            self.assertTrue(bool((gpu[:, max_pages] == 7).all()), "spilled")
            want = _reference(rt, rpi, sl, page_size, v2p=v2p).cpu().to(torch.int32)
            for b in range(bs):
                live = int((int(sl[b]) + page_size - 1) // page_size)
                self.assertTrue(torch.equal(gpu[b, :live], want[b, :live]))
                self.assertTrue(bool((gpu[b, live:max_pages] == 0).all()))

    def test_table_and_stream_in_one_launch(self):
        """One gather fills a fresh table -- live rows, the sink past them,
        padded rows all sink -- and the token stream of the same rows, each
        lane from its window start (one reading past its row's live pages),
        padded lanes reading the sink. The CUDA kernel agrees with the CPU
        path and with a table build plus a packed build of the same rows."""
        from sglang.kernels.ops.kvcache.kv_read_table import (
            build_kv_read_table,
            build_kv_read_table_and_stream,
            build_kv_read_table_packed,
        )

        for page_size in (1, 32, 64):
            rt, rpi, sl, v2p = self._make_batch(page_size)
            live = rpi.shape[0]
            rows = live + 2
            max_pages = int((sl.max().item() + page_size - 1) // page_size) + 2
            starts = torch.tensor([0, 1, page_size, 2, 0], dtype=torch.int64)[:live]
            lens = (sl.cpu().to(torch.int64) - starts).clamp(min=0)
            # The last lane reads a page past its live prefix.
            lens[-1] += page_size
            stream_lens = torch.cat([lens, torch.tensor([3, 1])])
            stream_starts = torch.cat([starts, torch.zeros(2, dtype=torch.int64)])
            indptr = torch.zeros(rows + 1, dtype=torch.int32)
            indptr[1:] = torch.cumsum(stream_lens, 0)
            total = int(indptr[-1])

            def fill(device):
                # Guard column / tail past the outputs catch a spill.
                table = torch.full(
                    (rows, max_pages + 1), 7, dtype=torch.int32, device=device
                )
                stream = torch.full((total + 4,), 7, dtype=torch.int32, device=device)
                build_kv_read_table_and_stream(
                    req_to_token=rt.to(device),
                    req_pool_indices=rpi.to(device),
                    seq_lens=sl.to(device=device, dtype=torch.int64),
                    v2p=v2p.to(device),
                    page_size=page_size,
                    max_pages=max_pages,
                    out=table,
                    stream_lens=stream_lens.to(device),
                    stream_indptr=indptr.to(device),
                    stream_out=stream,
                    stream_kv_start_idx=stream_starts.to(device),
                    seq_len_delta=1,
                )
                return table.cpu(), stream.cpu()

            (gt, gs), (ct, cs) = fill(_DEV), fill("cpu")
            self.assertTrue(torch.equal(gt, ct), f"ps={page_size} table")
            self.assertTrue(torch.equal(gs, cs), f"ps={page_size} stream")
            self.assertTrue(bool((gt[:, max_pages] == 7).all()), "table spilled")
            self.assertTrue(bool((gs[total:] == 7).all()), "stream spilled")
            self.assertTrue(bool((gt[live:, :max_pages] == 0).all()))

            want_table = torch.empty((live, max_pages), dtype=torch.int32, device=_DEV)
            build_kv_read_table(
                req_to_token=rt,
                req_pool_indices=rpi,
                seq_lens=sl.to(torch.int64),
                v2p=v2p,
                page_size=page_size,
                max_pages=max_pages,
                out=want_table,
                seq_len_delta=1,
                zero_tail=True,
            )
            self.assertTrue(torch.equal(gt[:live, :max_pages], want_table.cpu()))
            want_stream = torch.zeros(total, dtype=torch.int32, device=_DEV)
            build_kv_read_table_packed(
                req_to_token=rt,
                req_pool_indices=rpi,
                seq_lens=lens.to(_DEV),
                v2p=v2p,
                indptr=indptr.to(_DEV),
                page_size=page_size,
                max_tokens=total,
                out=want_stream,
                kv_start_idx=starts.to(_DEV),
            )
            n_live = int(indptr[live])
            self.assertTrue(torch.equal(gs[:n_live], want_stream[:n_live].cpu()))
            for b in range(live, rows):
                lo, hi = int(indptr[b]), int(indptr[b + 1])
                pos = torch.arange(hi - lo, dtype=torch.int32)
                self.assertTrue(torch.equal(gs[lo:hi], pos % page_size))

    def test_static_kernel_matches_reference(self):
        """The stripped (id-space-free) flashmla kernel is byte-identical to the
        plain token//ps reference -- guards the v2p-arg removal itself."""
        for page_size in (1, 32, 64):
            rt, rpi, sl, _ = self._make_batch(page_size)
            got = _fill_block_table_static(rt, rpi, sl, page_size)
            want = _reference(rt, rpi, sl, page_size, v2p=None)
            self.assertTrue(
                torch.equal(got.long(), want), f"page_size={page_size}: {got} != {want}"
            )

    def test_block_table_matches_reference(self):
        """The v2p gather alone is the whole translation, and it must not be
        skipped: a block table left in virtual id space differs from the
        reference here."""
        for page_size in (1, 32, 64):
            rt, rpi, sl, v2p = self._make_batch(page_size)
            got = _fill_block_table(rt, rpi, sl, page_size, v2p=v2p)
            want = _reference(rt, rpi, sl, page_size, v2p=v2p)
            self.assertTrue(
                torch.equal(got.long(), want),
                f"page_size={page_size}:\ngot ={got}\nwant={want}",
            )
            # The v2p permutation is non-trivial here, so a skipped translation
            # would be visibly different rather than accidentally equal.
            virtual = _reference(rt, rpi, sl, page_size, v2p=None)
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
        got = _fill_block_table(rt, rpi, sl, page_size, v2p=v2p)
        for r in range(got.shape[0]):
            n_pages = (int(sl[r].item()) + page_size - 1) // page_size
            self.assertTrue(
                torch.all(got[r, n_pages:] == -1),
                f"row {r} padded lanes were written: {got[r]}",
            )

    def test_agrees_with_token_level_translate(self):
        """The flashinfer updaters translate TOKEN ids with `translate_kv_loc`;
        the trtllm path builds PAGE ids in-kernel. Both must address the same
        physical page."""
        page_size = 64
        rt, rpi, sl, v2p = self._make_batch(page_size)
        block_table = _fill_block_table(rt, rpi, sl, page_size, v2p=v2p).long()
        for r in range(rt.shape[0]):
            n = int(sl[r].item())
            virt_tokens = rt[r, :n].long()
            # translate_kv_loc's formula, applied to token ids.
            kernel_tokens = (
                v2p[virt_tokens // page_size] * page_size + virt_tokens % page_size
            )
            # The block-table entry scaled by page_size must be the physical id of
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
    reference of the physical-id formula, on both the page_size == 1 / no-SWA fast
    path (what Kimi-Linear takes) and the general kernel. The static call
    (no source flag) stays byte-identical to the pre-translator kernel.
    """

    def _run(self, page_size, *, v2p, bs=5, max_ctx=2048):
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
            # Unified: the fused call does the prefix sum only, and the builder
            # writes the page table itself -- bounded by the `cache_seqlens`
            # that call just produced, which is what a reader bounds by too.
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
                skip_page_table=True,
            )
            build_kv_read_table(
                req_to_token=rt,
                req_pool_indices=rpi,
                seq_lens=cache_seqlens,
                v2p=v2p_full,
                page_size=page_size,
                max_pages=max_pages,
                out=page_table,
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
        want = _reference(rt, rpi, sl, page_size, v2p=(v2p_full if v2p else None))
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

    def _assert_translated(self, want, page_size):
        """The v2p permutation is non-trivial on the pages used, so a skipped
        translation would differ from `want` rather than match it by accident."""
        virtual = _reference(
            *TestBlockTable._make_batch(self, page_size)[:3],
            page_size,
            v2p=None,
        )
        self.assertFalse(
            torch.equal(want, virtual),
            "test batch degenerated: v2p is the identity on the pages used",
        )

    def test_identity_when_hooks_absent(self):
        """Static pool: no v2p -> byte-identical to pre-change."""
        for page_size in (1, 64):
            got, want, sl = self._run(page_size, v2p=False)
            self._assert_live_prefix(got, want, sl, page_size)

    def test_translated_mapping_ps1_fast_path(self):
        got, want, sl = self._run(1, v2p=True)
        self._assert_live_prefix(got, want, sl, 1)
        self._assert_translated(want, 1)

    def test_translated_mapping_general_path(self):
        got, want, sl = self._run(64, v2p=True)
        self._assert_live_prefix(got, want, sl, 64)
        self._assert_translated(want, 64)

    # (The old fa3<->flashmla agreement case is gone: both families now
    # consume the SAME canonical builder, so cross-family agreement holds by
    # construction and the per-family cases above cover the two consumers.)


if __name__ == "__main__":
    unittest.main()
