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
"""A widened verify table must reach the kernel that reads past seq_lens.

A whole-sequence verify reads `[committed prefix | drafts]` back out of the
pool, so under the unified pool the translated page table a verify kernel
reads has to cover `seq_lens + num_draft_tokens` columns per row; a table
filled only to `seq_lens` silently leaves the draft tail stale.

These tests drive fa3's eager and captured verify builds from a verify plan
over a non-identity virtual->physical page map, and check the page table the
kernel reads: every row translated, and filled through the draft tail.

    python -m pytest test/registered/unit/layers/attention/test_verify_widening_is_wired.py -v
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
)
from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
from sglang.srt.mem_cache.kv_loc_plan import IdSpace, IdSpaceKind
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

_DEV = "cuda"
_NUM_DRAFT = 4
_MAX_CTX = 128


def _translator(req_to_token, page_size, n_pages):
    """A translating translator over a fixed, non-identity page map.

    The allocator wiring is irrelevant here; what matters is that the reads go
    through the real table builder and kernel with a v2p map that a skipped or
    truncated translation cannot accidentally satisfy."""
    g = torch.Generator(device="cpu").manual_seed(7 + page_size)
    v2p = torch.randperm(n_pages, generator=g).to(torch.int64)
    v2p[0] = 0  # page 0 is the reserved sink
    t = KVIndexTranslator.__new__(KVIndexTranslator)
    t.req_to_token = req_to_token
    t.page_size = page_size
    t.device = _DEV
    t.is_translating = True
    t.defer_read_translate = False
    t._capture_page_size = page_size
    t._full_v2p_table = v2p.to(_DEV)
    t._spaces = {
        IdSpaceKind.FULL: IdSpace(
            key=(IdSpaceKind.FULL, "test"),
            write=lambda ids: (
                t._full_v2p_table[ids // page_size] * page_size + ids % page_size
            ),
            read_v2p=t._full_v2p_table,
        )
    }
    t._rows = torch.arange(req_to_token.shape[0], dtype=torch.int64, device=_DEV)
    return t


def _batch(page_size, n_pages):
    """Three requests with distinct, non-monotonic virtual page runs; each row
    holds its prefix plus the draft window (verify writes the drafts first)."""
    g = torch.Generator(device="cpu").manual_seed(11 + page_size)
    live = torch.tensor([page_size + 1, 3 * page_size, 1], dtype=torch.int64)
    bs = live.numel()
    req_to_token = torch.zeros((bs, _MAX_CTX), dtype=torch.int32)
    pages = torch.randperm(n_pages - 1, generator=g)[: bs * 8] + 1
    for r in range(bs):
        run = pages[r * 8 : r * 8 + 8]
        toks = (run[:, None] * page_size + torch.arange(page_size)[None, :]).reshape(-1)
        n = int(live[r]) + _NUM_DRAFT
        req_to_token[r, :n] = toks[:n].to(torch.int32)
    return req_to_token.to(_DEV), live


def _expected(req_to_token, live, v2p, page_size):
    """Per row, the translated pages covering `live + num_draft_tokens`."""
    rows = []
    for r in range(live.numel()):
        n_pages = -(-(int(live[r]) + _NUM_DRAFT) // page_size)
        toks = req_to_token[r, torch.arange(n_pages, device=_DEV) * page_size].long()
        rows.append(v2p[toks // page_size].to(torch.int32).cpu())
    return rows


def _verify_plan(translator, req_to_token, live, host_lens=None):
    """The plan a verify forward builds for itself: it writes the draft window
    past `live`, so it reads that far."""
    bs = live.numel()
    rows = torch.arange(bs, device=_DEV)
    cols = live.to(_DEV)[:, None] + torch.arange(_NUM_DRAFT, device=_DEV)[None, :]
    return translator.own_plan(
        SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=SimpleNamespace(draft_token_num=_NUM_DRAFT),
            batch_size=bs,
            req_pool_indices=rows,
            seq_lens=live.to(_DEV),
            seq_lens_cpu=live if host_lens is None else host_lens,
            out_cache_loc=req_to_token[rows[:, None], cols].long().reshape(-1),
        )
    )


def _backend(translator, req_to_token, page_size):
    b = FlashAttentionBackend.__new__(FlashAttentionBackend)
    b.kv_index_translator = translator
    b.req_to_token = req_to_token
    b.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
    b.page_size = page_size
    b.max_context_len = _MAX_CTX
    b.max_num_pages = -(-_MAX_CTX // page_size)
    b.topk = 1
    b.speculative_num_draft_tokens = _NUM_DRAFT
    b.speculative_num_steps = 3
    b.speculative_step_id = 0
    b.has_swa = False
    b.use_sliding_window_kv_pool = False
    b.is_prefill_aware_swa = False
    b.token_to_kv_pool = None
    b._kv_shard_pool = None
    b._compute_scheduler_metadata = lambda *_: None
    b._maybe_init_local_attn_metadata = lambda *_: None
    return b


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestFa3VerifyReadsTheDraftTail(CustomTestCase):
    N_PAGES = 64

    def _check(self, page_table, req_to_token, live, translator, page_size):
        want = _expected(req_to_token, live, translator._full_v2p_table, page_size)
        got = page_table.cpu()
        for r, row in enumerate(want):
            self.assertTrue(
                torch.equal(got[r, : row.numel()], row),
                f"ps={page_size} row {r}: page table does not cover the draft "
                f"tail:\ngot ={got[r, : row.numel()]}\nwant={row}",
            )

    def test_captured_verify_build(self):
        """The captured build a cuda-graph replay runs."""
        for page_size in (1, 4):
            req_to_token, live = _batch(page_size, self.N_PAGES)
            translator = _translator(req_to_token, page_size, self.N_PAGES)
            b = _backend(translator, req_to_token, page_size)
            bs = live.numel()
            metadata = FlashAttentionMetadata()
            metadata.cache_seqlens_int32 = torch.zeros(
                bs, dtype=torch.int32, device=_DEV
            )
            metadata.cu_seqlens_k = torch.zeros(bs + 1, dtype=torch.int32, device=_DEV)
            metadata.page_table = torch.zeros(
                (bs, b.max_num_pages), dtype=torch.int32, device=_DEV
            )
            metadata.swa_page_table = None
            b.target_verify_metadata = {bs: metadata}
            b._apply_cuda_graph_metadata(
                _verify_plan(translator, req_to_token, live),
                bs,
                torch.arange(bs, device=_DEV),
                live.to(_DEV),
                None,
                ForwardMode.TARGET_VERIFY,
                SimpleNamespace(ragged_verify_layout=None),
                live,
            )
            self.assertTrue(
                torch.equal(
                    metadata.cache_seqlens_int32.cpu(), (live + _NUM_DRAFT).int()
                )
            )
            self._check(metadata.page_table, req_to_token, live, translator, page_size)

    def test_eager_verify_build(self):
        """The eager build, including a DSPARK-style batch whose host lens are
        already expanded by the window."""
        for page_size in (1, 4):
            req_to_token, live = _batch(page_size, self.N_PAGES)
            translator = _translator(req_to_token, page_size, self.N_PAGES)
            b = _backend(translator, req_to_token, page_size)
            bs = live.numel()
            for host_lens, spec_info in (
                (live, SimpleNamespace(ragged_verify_layout=None)),
                (
                    live + _NUM_DRAFT,
                    SimpleNamespace(ragged_verify_layout=None, live_seq_lens_cpu=live),
                ),
            ):
                fb = SimpleNamespace(
                    forward_mode=ForwardMode.TARGET_VERIFY,
                    batch_size=bs,
                    seq_lens=live.to(_DEV),
                    seq_lens_cpu=host_lens,
                    seq_lens_sum=int(host_lens.sum()),
                    req_pool_indices=torch.arange(bs, device=_DEV),
                    spec_info=spec_info,
                    encoder_lens=None,
                    out_cache_loc=None,
                    kv_loc_plan=_verify_plan(translator, req_to_token, live, host_lens),
                )
                b.init_forward_metadata(fb)
                self._check(
                    b.forward_metadata.page_table,
                    req_to_token,
                    live,
                    translator,
                    page_size,
                )


if __name__ == "__main__":
    unittest.main()
