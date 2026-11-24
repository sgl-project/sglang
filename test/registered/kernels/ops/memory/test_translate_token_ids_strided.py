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
"""``translate_token_ids`` must walk `loc` and `out` by their own strides.
The SWA read path hands it a column slice of a capture-stable page table, whose
row stride is the whole buffer width; flat addressing reads the wrong ids there
without any error. Only this test enters the Triton kernel -- the CPU reference
never does.

    python -m pytest test/registered/kernels/ops/memory/test_translate_token_ids_strided.py -v
"""

import itertools
import unittest

import torch

from sglang.kernels.ops.memory.virtual_slot import translate_token_ids
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestTranslateTokenIdsStrided(unittest.TestCase):
    ROWS, COLS, PAD = 6, 13, 11

    def _v2p(self, num_pages):
        # Scrambled, with one tombstone so a missing clamp shows up.
        v2p = torch.tensor(
            [(5 * i + 2) % num_pages for i in range(num_pages)] + [-1],
            dtype=torch.int64,
            device="cuda",
        )
        v2p[3] = -1
        return v2p

    # Lanes that must resolve to the sink: padding, a tombstoned page, a loc
    # below the sentinel, a page past the table.
    SINK_LANES = (0, 7, 11, 12)

    def _locs(self, page_size, dtype=torch.int64):
        """Every lane kind the kernel tells apart, as `(locs, num_pages)`:
        non-zero in-page offsets, the -1 padding sink, a tombstoned page, a
        loc below the sentinel and a page past a `num_pages` table."""
        n = self.ROWS * self.COLS
        lane = torch.arange(n, dtype=torch.int64, device="cuda")
        locs = lane * page_size + lane % page_size
        locs[0] = -1  # the padding sink
        locs[7] = 4 * page_size - 1  # the last slot of the tombstoned page 3
        num_pages = int(locs.max()) // page_size + 4
        locs[11] = -(page_size + 1)  # below the sentinel
        locs[12] = (num_pages + 2) * page_size + 1  # past the table
        return locs.reshape(self.ROWS, self.COLS).to(dtype), num_pages

    def _run(self, loc, v2p, page_size, out=None):
        return translate_token_ids(loc=loc, v2p=v2p, page_size=page_size, out=out)

    def test_a_column_slice_reads_the_same_ids_as_its_contiguous_copy(self):
        for page_size, dtype in itertools.product((1, 64), (torch.int64, torch.int32)):
            with self.subTest(page_size=page_size, dtype=dtype):
                values, num_pages = self._locs(page_size, dtype)
                v2p = self._v2p(num_pages)
                backing = torch.full(
                    (self.ROWS, self.COLS + self.PAD),
                    -1,
                    dtype=dtype,
                    device="cuda",
                )
                view = backing[:, : self.COLS]
                view.copy_(values)
                self.assertFalse(view.is_contiguous())

                got = self._run(view, v2p, page_size)
                want = self._run(view.contiguous(), v2p, page_size)
                self.assertEqual(got.shape, view.shape)
                self.assertTrue(torch.equal(got, want), f"page_size={page_size}")

    def test_a_column_stride_above_one_is_honoured(self):
        page_size = 1
        values, num_pages = self._locs(page_size)
        v2p = self._v2p(num_pages)
        backing = torch.empty(
            (self.ROWS, self.COLS * 2), dtype=torch.int64, device="cuda"
        )
        view = backing.as_strided((self.ROWS, self.COLS), (self.COLS * 2, 2))
        view.copy_(values)
        self.assertTrue(
            torch.equal(
                self._run(view, v2p, page_size), self._run(values, v2p, page_size)
            )
        )

    def test_out_takes_its_own_row_stride_and_leaves_its_neighbours_alone(self):
        page_size = 64
        values, num_pages = self._locs(page_size)
        v2p = self._v2p(num_pages)
        # `out`'s row stride differs from `loc`'s, so one decomposition cannot
        # serve both; each side must use its own.
        loc_backing = torch.full(
            (self.ROWS, self.COLS + 3), -1, dtype=torch.int64, device="cuda"
        )
        loc = loc_backing[:, : self.COLS]
        loc.copy_(values)
        dst = torch.full(
            (self.ROWS, self.COLS + self.PAD), -9, dtype=torch.int64, device="cuda"
        )

        ret = self._run(loc, v2p, page_size, out=dst[:, : self.COLS])
        self.assertEqual(ret.data_ptr(), dst.data_ptr())
        self.assertTrue(
            torch.equal(dst[:, : self.COLS], self._run(values, v2p, page_size))
        )
        self.assertTrue(bool((dst[:, self.COLS :] == -9).all()))

    def test_the_cpu_reference_agrees_on_the_same_strided_input(self):
        for page_size, dtype in itertools.product((1, 64), (torch.int64, torch.int32)):
            with self.subTest(page_size=page_size, dtype=dtype):
                values, num_pages = self._locs(page_size, dtype)
                v2p = self._v2p(num_pages)
                backing = torch.full(
                    (self.ROWS, self.COLS + self.PAD), -1, dtype=dtype, device="cuda"
                )
                view = backing[:, : self.COLS]
                view.copy_(values)

                cuda_ids = self._run(view, v2p, page_size)
                cpu_backing = backing.cpu()
                cpu_ids = self._run(cpu_backing[:, : self.COLS], v2p.cpu(), page_size)
                self.assertTrue(torch.equal(cuda_ids.cpu(), cpu_ids))
                sink = cuda_ids.reshape(-1)[list(self.SINK_LANES)]
                self.assertTrue(bool((sink == 0).all()), f"sink lanes: {sink}")


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestTranslateTokenIdsOwnerRule(unittest.TestCase):
    """Under logical-page KV sharding (``shard_size > 1``) the kernel takes
    LOGICAL ids and applies the owner rule itself. The sharded allocator's own
    tests run on the CPU stage and reach only the torch reference."""

    # One rank's band: its first page past the sink, and its pages.
    FIRST_PAGE, PER_RANK = 3, 40
    # (page_size, shard_size, shard_rank, dtype)
    CONFIGS = (
        (1, 2, 1, torch.int64),
        (4, 2, 0, torch.int32),
        (64, 3, 0, torch.int32),
        (64, 8, 7, torch.int64),
    )

    def _band_tokens(self, loc, page_size, shard_size, shard_rank):
        """The owner rule spelled out: logical page ``l`` is rank ``l % N``'s
        band page ``l // N - 1 + FIRST_PAGE``; pages ``[0, N)`` are reserved.
        -1, the sink, for every other id."""
        tokens = loc.to(torch.int64)
        page = torch.div(tokens, page_size, rounding_mode="floor")
        band_page = torch.div(page, shard_size, rounding_mode="floor")
        band = (band_page - 1 + self.FIRST_PAGE) * page_size + tokens % page_size
        owned = (page >= shard_size) & (page % shard_size == shard_rank)
        return torch.where(owned, band, -1)

    def test_the_owner_rule_is_the_band_translation(self):
        generator = torch.Generator().manual_seed(0)
        for page_size, shard_size, shard_rank, dtype in self.CONFIGS:
            with self.subTest(page_size=page_size, shard_size=shard_size):
                v2p = torch.full(
                    (self.FIRST_PAGE + self.PER_RANK + 1,), -1, dtype=torch.int64
                )
                bound = torch.randperm(self.PER_RANK, generator=generator)[:30]
                v2p[bound + self.FIRST_PAGE] = (
                    torch.randperm(500, generator=generator)[:30] + 1
                )
                v2p = v2p.cuda()
                # One past the largest logical token id.
                top = shard_size * (self.PER_RANK + 1) * page_size
                ids = torch.cat(
                    (
                        torch.randint(
                            -3 * page_size,
                            top + 3 * page_size,
                            (1024,),
                            generator=generator,
                        ),
                        # Negative ids, the reserved pages, the largest id.
                        torch.arange(-2, shard_size * page_size + 2),
                        torch.tensor([top - 1, top, top + 1]),
                    )
                )
                loc = ids.to(dtype).cuda()
                shard = dict(
                    shard_size=shard_size,
                    shard_rank=shard_rank,
                    shard_first_page=self.FIRST_PAGE,
                )
                want = translate_token_ids(
                    loc=self._band_tokens(loc, page_size, shard_size, shard_rank),
                    v2p=v2p,
                    page_size=page_size,
                )
                got = translate_token_ids(
                    loc=loc, v2p=v2p, page_size=page_size, **shard
                )
                self.assertTrue(torch.equal(got, want))
                self.assertGreater(int((got > 0).sum()), 0)
                # A request row read by its page stride.
                view = loc[:1024].view(32, 32)[:, ::2]
                self.assertTrue(
                    torch.equal(
                        translate_token_ids(
                            loc=view, v2p=v2p, page_size=page_size, **shard
                        ),
                        want[:1024].view(32, 32)[:, ::2],
                    )
                )
                reference = translate_token_ids(
                    loc=loc.cpu(), v2p=v2p.cpu(), page_size=page_size, **shard
                )
                self.assertTrue(torch.equal(got.cpu(), reference))


if __name__ == "__main__":
    unittest.main()
