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

`create_flashmla_kv_indices_triton` does that in-kernel via `v2p_ptr` / `PAGE_MULT`
(trtllm_mla / cutedsl_mla / tokenspeed_mla), and the flashinfer_mla updaters do it
by post-gathering `translate_kv_loc_dense` over the token-level kv_indices.

Covered here:
  - kernel identity: `v2p_ptr=None, PAGE_MULT=1` is byte-identical to main;
  - kernel dense mapping against the python reference, for several page sizes,
    ragged sequence lengths and a non-identity v2p permutation;
  - padded block-table lanes never index the v2p table out of bounds;
  - the token-level dense translate the flashinfer updaters apply agrees with the
    page-level block table the trtllm path builds.

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


if __name__ == "__main__":
    unittest.main()
