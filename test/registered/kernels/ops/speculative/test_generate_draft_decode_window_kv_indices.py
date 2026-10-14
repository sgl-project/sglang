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
"""`generate_draft_decode_window_kv_indices`: the multi-step draft's per-step
sliding-window read rail.

A draft's window layer at step s attends the last `min(seq_len + s + 1,
window)` tokens of its chain: the committed prefix followed by the draft's own
uncommitted tokens, which live in the chain slots the full kernel emits, not
in req_to_token's live prefix. Pinned: the emitted ids are exactly that
window-clipped tail of ``prefix ++ chain``, translated through the read
source's v2p when it has one; the indptr is the clipped cumsum; TRANSLATE=False
emits the raw ids; and over the plan's physical page table the kernel emits
what the translated gather of the virtual rows does.

    python -m pytest test/registered/kernels/ops/speculative/test_generate_draft_decode_window_kv_indices.py -v
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_SENTINEL = -7


def _run_kernel(
    *,
    table,
    seq_lens,
    positions,
    num_steps,
    topk,
    page_size,
    window,
    v2p=None,
    entry_page_size=1,
):
    from sglang.kernels.ops.speculative.cache_locs import (
        generate_draft_decode_window_kv_indices,
    )
    from sglang.srt.utils import next_power_of_2

    num_seqs = seq_lens.numel()
    bs = num_seqs * topk
    width = bs * window
    out = torch.full((num_steps, width), _SENTINEL, dtype=torch.int64, device="cuda")
    indptr = torch.zeros((num_steps, bs + 1), dtype=torch.int32, device="cuda")
    generate_draft_decode_window_kv_indices[(num_steps, num_seqs, topk)](
        torch.arange(num_seqs, dtype=torch.int64, device="cuda"),
        table,
        seq_lens,
        out,
        indptr,
        positions,
        table.stride(0),
        width,
        indptr.shape[1],
        next_power_of_2(num_seqs),
        next_power_of_2(num_steps),
        next_power_of_2(bs),
        page_size,
        window,
        ENTRY_PAGE_SIZE=entry_page_size,
        v2p=v2p,
        TRANSLATE=v2p is not None,
    )
    return out, indptr


def _chain_slot(seq_len, topk_id, step, *, num_steps, topk, page_size):
    """Column of chain token ``step`` of branch ``topk_id`` in a req_to_token
    row, as `generate_draft_decode_kv_indices` lays the chain out."""
    if page_size == 1 or topk == 1:
        return seq_len + topk_id * num_steps + step
    last_page_len = seq_len % page_size
    num_new_pages_per_topk = (last_page_len + num_steps + page_size - 1) // page_size
    prefix_base = seq_len // page_size * page_size
    return (
        prefix_base
        + topk_id * num_new_pages_per_topk * page_size
        + last_page_len
        + step
    )


def _reference(*, rows, seq_lens, num_steps, topk, page_size, window, v2p):
    """Per step: the window-clipped tail of prefix ++ chain per branch."""
    steps = []
    for s in range(num_steps):
        iters = s + 1
        ids, indptr = [], [0]
        for b, seq_len in enumerate(seq_lens):
            for t in range(topk):
                total = seq_len + iters
                start = total - min(total, window)
                chain = [
                    rows[b][
                        _chain_slot(
                            seq_len,
                            t,
                            j,
                            num_steps=num_steps,
                            topk=topk,
                            page_size=page_size,
                        )
                    ]
                    for j in range(iters)
                ]
                tail = (list(rows[b][:seq_len]) + chain)[start:]
                if v2p is not None:
                    tail = [
                        max(v2p[i // page_size] * page_size + i % page_size, 0)
                        for i in tail
                    ]
                ids.extend(tail)
                indptr.append(len(ids))
        steps.append((ids, indptr))
    return steps


class TestGenerateDraftDecodeWindowKVIndices(CustomTestCase):
    def _check(self, *, page_size, topk, window, translate, num_steps=3):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        torch.manual_seed(3)
        num_seqs, max_context = 3, 64
        # Prefix ids and chain slots are all distinct so a misplaced read shows.
        req_to_token = torch.randperm(1 << 12, device="cuda")[: num_seqs * max_context]
        req_to_token = req_to_token.view(num_seqs, max_context).to(torch.int64)
        seq_lens = torch.tensor([5, 1, 9], dtype=torch.int64, device="cuda")
        positions = seq_lens.repeat_interleave(topk)
        v2p = None
        if translate:
            num_pages = (1 << 12) // page_size + 1
            perm = torch.randperm(num_pages, device="cuda")
            v2p = torch.empty(num_pages, dtype=torch.int64, device="cuda")
            v2p[perm] = torch.arange(num_pages, device="cuda")
        out, indptr = _run_kernel(
            table=req_to_token,
            seq_lens=seq_lens,
            positions=positions,
            num_steps=num_steps,
            topk=topk,
            page_size=page_size,
            window=window,
            v2p=v2p,
        )
        expected = _reference(
            rows=req_to_token.tolist(),
            seq_lens=seq_lens.tolist(),
            num_steps=num_steps,
            topk=topk,
            page_size=page_size,
            window=window,
            v2p=None if v2p is None else v2p.tolist(),
        )
        for s, (ids, ptr) in enumerate(expected):
            self.assertEqual(indptr[s].tolist(), ptr, f"step {s} indptr")
            self.assertEqual(out[s, : len(ids)].tolist(), ids, f"step {s} ids")
            # Nothing past the used prefix is touched.
            self.assertTrue(bool((out[s, len(ids) :] == _SENTINEL).all()), f"step {s}")

    def test_window_clips_the_prefix_and_keeps_the_chain(self):
        # window 4 < seq_len 5 + steps: only the tail of the prefix survives,
        # and by step 3 the window is chain-heavy.
        self._check(page_size=1, topk=1, window=4, translate=False)

    def test_window_wider_than_the_chain_emits_everything(self):
        self._check(page_size=1, topk=1, window=64, translate=False)

    def test_translate_uses_the_swa_side_page_table(self):
        self._check(page_size=1, topk=1, window=6, translate=True)
        self._check(page_size=2, topk=1, window=6, translate=True)

    def test_paged_chain_slots_with_topk(self):
        self._check(page_size=2, topk=2, window=5, translate=True)

    def test_the_plan_page_table_matches_the_translated_gather(self):
        """Once the iteration's sliding-window table is built, the rail reads
        physical pages from it (entry * ps + pos % ps) instead of translating
        the virtual rows; both must emit the same ids."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        for page_size, topk in ((1, 1), (2, 1), (2, 2)):
            torch.manual_seed(5)
            num_seqs, max_context, num_steps = 3, 64, 3
            num_pages = max_context // page_size
            # Rows of whole virtual pages in shuffled order over a shuffled v2p.
            v2p = torch.randperm(num_pages, device="cuda")
            offsets = torch.arange(page_size, device="cuda")
            req_to_token = torch.stack(
                [
                    (
                        torch.randperm(num_pages, device="cuda")[:, None] * page_size
                        + offsets
                    ).reshape(-1)
                    for _ in range(num_seqs)
                ]
            )
            table = v2p[req_to_token[:, ::page_size] // page_size].to(torch.int32)
            seq_lens = torch.tensor([5, 1, 9], dtype=torch.int64, device="cuda")
            common = dict(
                seq_lens=seq_lens,
                positions=seq_lens.repeat_interleave(topk),
                num_steps=num_steps,
                topk=topk,
                page_size=page_size,
                window=6,
            )
            gathered, gathered_indptr = _run_kernel(
                table=req_to_token, v2p=v2p, **common
            )
            out, out_indptr = _run_kernel(
                table=table, entry_page_size=page_size, **common
            )
            torch.testing.assert_close(out_indptr, gathered_indptr, rtol=0, atol=0)
            torch.testing.assert_close(out, gathered, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
