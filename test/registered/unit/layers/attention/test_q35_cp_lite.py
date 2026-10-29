"""CP-lite refuses LL exchange polls narrower than their CTA: such a poll holds a word in two
warps, and its block-wide count could let one of them return the previous call's copy."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed._symmetric_memory as symm_mem

from sglang.srt.environ import envs
from sglang.srt.layers.attention import q35_cp_lite as cl
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


class TestLLPollWidths(CustomTestCase):
    def test_shipped_shapes_are_accepted(self):
        # TP8 / TEP8: cp 4 x 4 heads (256 words, or 128 in two gather chunks); TEP16: cp 8 x 2 heads
        for q_heads, chunks in ((4, 1), (4, 2), (2, 1)):
            cl._check_ll_polls(q_heads, chunks)

    def test_polls_narrower_than_the_cta_are_refused(self):
        # TP8 with SGLANG_PSM_CP_GATHER_CHUNKS=4, TEP16 in two chunks, one head a rank: 64 words each
        for q_heads, chunks in ((4, 4), (2, 2), (1, 1)):
            with self.subTest(q_heads=q_heads, chunks=chunks):
                with self.assertRaisesRegex(ValueError, "polling race"):
                    cl._check_ll_polls(q_heads, chunks)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_construction_checks_the_gather_width(self):
        # a real CpLite of TP8's group rank 0 (cp 4 x 4 heads); only the symmetric-memory rendezvous is stubbed
        def rendezvous(buf, group):
            return SimpleNamespace(rank=0, buffer_ptrs=[buf.data_ptr()] * 4)

        def build():
            return cl.CpLite(None, 0, 8, 4, 4, "cuda")

        with patch.object(symm_mem, "get_backend", lambda device: "NCCL"), patch.object(
            symm_mem, "empty", lambda shape, dtype, device: torch.empty(shape, dtype=dtype, device=device)
        ), patch.object(symm_mem, "rendezvous", rendezvous), patch.object(
            torch.distributed, "barrier", lambda group=None: None
        ):
            self.assertTrue(build().ll)
            with envs.SGLANG_TRTL18_CP_UNGATED.override(True):
                with envs.SGLANG_PSM_CP_GATHER_CHUNKS.override(2):
                    self.assertEqual(build().q_chunks, 2)
                with envs.SGLANG_PSM_CP_GATHER_CHUNKS.override(4):
                    with self.assertRaisesRegex(ValueError, "polling race"):
                        build()


if __name__ == "__main__":
    unittest.main()
