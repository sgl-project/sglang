"""DraftHeadPacker packs one decode rank's draft KV heads into registered staging
page by page and hands the Transport copies keyed by its staging buffer and the
decode's draft entry. A wrong head range or offset delivers another head's bytes
or another page; nothing fails, only speculative acceptance drops.

Expected bytes are sliced from the draft tensors by hand, never by calling the
packer, so a bug in it cannot make both sides agree.
"""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

import unittest

import numpy as np
import torch

from sglang.srt.disaggregation.mooncake.buffers import BufferKind, MooncakeBufferKey
from sglang.srt.disaggregation.mooncake.draft_head_packer import (
    DraftEntryJob,
    DraftHeadPacker,
)
from sglang.test.test_utils import CustomTestCase


class TestDraftHeadPackerPack(CustomTestCase):
    PAGE, HEADS, HEAD_DIM, SLOTS = 4, 4, 8, 32

    def test_copies_carry_each_jobs_heads_to_its_decode_pages(self):
        torch.manual_seed(0)
        tensors = [
            torch.randn(
                self.SLOTS, self.HEADS, self.HEAD_DIM, dtype=torch.bfloat16
            ).cuda()
            for _ in range(2)
        ]
        staging_keys = [MooncakeBufferKey(BufferKind.STAGING, i) for i in range(4)]
        packer = DraftHeadPacker(
            tensors=tensors,
            page_size=self.PAGE,
            window_pages=0,
            register=lambda regions: staging_keys[: len(regions)],
        )
        # A decode rank holding heads [1, 3) of entry 0 and [2, 3) of entry 1.
        dst_keys = [
            MooncakeBufferKey(BufferKind.KV, 5),
            MooncakeBufferKey(BufferKind.KV, 6),
        ]
        jobs = [
            DraftEntryJob(entry=0, dst_key=dst_keys[0], head_start=1, num_heads=2),
            DraftEntryJob(entry=1, dst_key=dst_keys[1], head_start=2, num_heads=1),
        ]
        src_pages = np.array([5, 1, 3])
        dst_pages = np.array([0, 7, 2])
        buffer = packer._free.get()

        copies = packer._pack(
            buffer=buffer, src_pages=src_pages, dst_pages=dst_pages, jobs=jobs
        )

        staged = buffer.data.cpu().numpy()
        self.assertEqual([c.dst for c in copies], dst_keys)
        for copy, job in zip(copies, jobs):
            self.assertEqual(copy.src, buffer.key)
            page_bytes = self.PAGE * job.num_heads * self.HEAD_DIM * 2
            np.testing.assert_array_equal(copy.ranges[:, 1], dst_pages * page_bytes)
            np.testing.assert_array_equal(copy.ranges[:, 2], page_bytes)
            for (src_offset, _, length), page in zip(copy.ranges, src_pages):
                rows = slice(page * self.PAGE, (page + 1) * self.PAGE)
                heads = slice(job.head_start, job.head_start + job.num_heads)
                expected = (
                    tensors[job.entry][rows, heads]
                    .contiguous()
                    .view(torch.uint8)
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
                np.testing.assert_array_equal(
                    staged[src_offset : src_offset + length], expected
                )


if __name__ == "__main__":
    unittest.main()
