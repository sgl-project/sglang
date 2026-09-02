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
"""`generate_draft_decode_kv_indices` under the unified pool's translation.

BUG REGRESSION. The multi-step draft-decode kernel copied req_to_token values
verbatim into the per-step kv_indices. Under `--enable-unified-memory` those
values are VIRTUAL ids while the fused draft pool is indexed by draft-dense
ids, so with `speculative_num_steps > 1` every draft decode step read the
wrong KV rows — silently: bad drafts only depress acceptance, verify stays
correct. Pinned here: TRANSLATE=True must emit exactly
`translate_kv_loc_for_kernel`'s dense ids, and TRANSLATE=False must stay
byte-identical to the historical raw copy.

    python -m pytest test/registered/kernels/test_generate_draft_decode_kv_indices_translate.py -v
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

_SENTINEL = -7


def _run_kernel(
    *,
    req_to_token,
    seq_lens,
    positions,
    num_steps,
    topk,
    page_size,
    v2p,
    kv_mult,
    translate,
):
    from sglang.kernels.ops.speculative.cache_locs import (
        generate_draft_decode_kv_indices,
    )
    from sglang.srt.utils import next_power_of_2

    num_seqs = seq_lens.numel()
    bs = num_seqs * topk
    width = int((seq_lens.sum().item() + num_steps * bs) * 2 + 64)
    kv_indices = torch.full(
        (num_steps, width), _SENTINEL, dtype=torch.int64, device="cuda"
    )
    kv_indptr = torch.zeros((num_steps, bs + 1), dtype=torch.int32, device="cuda")
    generate_draft_decode_kv_indices[(num_steps, num_seqs, topk)](
        torch.arange(num_seqs, dtype=torch.int64, device="cuda"),
        req_to_token,
        seq_lens,
        kv_indices,
        kv_indptr,
        positions,
        v2p if translate else None,
        kv_mult if translate else 0,
        req_to_token.shape[1],
        width,
        kv_indptr.shape[1],
        next_power_of_2(num_seqs),
        next_power_of_2(num_steps),
        next_power_of_2(bs),
        page_size,
        TRANSLATE=translate,
    )
    return kv_indices, kv_indptr


class TestGenerateDraftDecodeKVIndicesTranslate(CustomTestCase):
    def _check(self, *, page_size, kv_mult, num_steps=3, topk=1):
        torch.manual_seed(7)
        num_seqs, max_context = 3, 64
        num_pages = max_context // page_size + 1
        # Virtual ids: a shuffled page layout so translation is not identity.
        perm = torch.randperm(num_pages, device="cuda")
        v2p = torch.empty(num_pages, dtype=torch.int64, device="cuda")
        v2p[perm] = torch.arange(num_pages, device="cuda")
        req_to_token = (
            torch.arange(max_context, dtype=torch.int64, device="cuda")
            .flip(0)
            .repeat(num_seqs, 1)
        )
        seq_lens = torch.tensor([5, 1, 9], dtype=torch.int64, device="cuda")[:num_seqs]
        positions = seq_lens.repeat_interleave(topk)

        common = dict(
            req_to_token=req_to_token,
            seq_lens=seq_lens,
            positions=positions,
            num_steps=num_steps,
            topk=topk,
            page_size=page_size,
            v2p=v2p,
            kv_mult=kv_mult,
        )
        raw, raw_indptr = _run_kernel(translate=False, **common)
        out, out_indptr = _run_kernel(translate=True, **common)

        torch.testing.assert_close(raw_indptr, out_indptr, rtol=0, atol=0)
        written = raw != _SENTINEL
        # Unwritten lanes stay untouched in both compilations.
        self.assertTrue(bool((out[~written] == _SENTINEL).all()))
        # Written lanes: translate_kv_loc_for_kernel's formula over the raw ids.
        virt = raw[written]
        expected = torch.clamp_min(
            v2p[virt // page_size] * (page_size * kv_mult) + virt % page_size, 0
        )
        torch.testing.assert_close(out[written], expected, rtol=0, atol=0)

    def test_page_size_one(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        self._check(page_size=1, kv_mult=6)

    def test_page_size_two(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        self._check(page_size=2, kv_mult=10)

    def test_translate_false_is_a_raw_copy(self):
        """The static-pool compilation must stay byte-identical to the
        historical kernel: the emitted indices ARE the req_to_token values."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        torch.manual_seed(11)
        num_seqs, max_context, page_size = 2, 32, 1
        req_to_token = torch.randint(
            0, 1 << 20, (num_seqs, max_context), dtype=torch.int64, device="cuda"
        )
        seq_lens = torch.tensor([4, 7], dtype=torch.int64, device="cuda")
        raw, _ = _run_kernel(
            req_to_token=req_to_token,
            seq_lens=seq_lens,
            positions=seq_lens,
            num_steps=2,
            topk=1,
            page_size=page_size,
            v2p=None,
            kv_mult=0,
            translate=False,
        )
        written = raw != _SENTINEL
        emitted = raw[written]
        self.assertTrue(bool(torch.isin(emitted, req_to_token.flatten()).all()))


if __name__ == "__main__":
    unittest.main()
