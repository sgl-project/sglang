import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
from sglang.srt.layers.attention.dsa.utils import should_use_dsa_fused_topk
from sglang.srt.runtime_context import get_context, get_exec
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "Test requires a GPU")
class TestDSATopKDeterministic(CustomTestCase):
    def setUp(self):
        self.enterContext(get_context().override_server_args())
        self.enterContext(
            get_exec().deterministic.override(enable_deterministic_inference=True)
        )

    def test_ties_are_stable_across_runs_and_batch_compositions(self):
        rows, width, topk = 8, 4096, 2048
        scores = torch.ones(rows, width, dtype=torch.float32, device="cuda")
        scores[::2, width // 2 :] = 1.001
        lengths = torch.full((rows,), width, dtype=torch.int32, device="cuda")

        full = DSATopKBackend.SGL_KERNEL.topk_func(scores, lengths, topk)
        for _ in range(3):
            torch.testing.assert_close(
                DSATopKBackend.SGL_KERNEL.topk_func(scores, lengths, topk),
                full,
                atol=0,
                rtol=0,
            )
        for start, size in ((0, 1), (3, 1), (2, 5)):
            part = DSATopKBackend.SGL_KERNEL.topk_func(
                scores[start : start + size].contiguous(),
                lengths[start : start + size],
                topk,
            )
            torch.testing.assert_close(part, full[start : start + size], atol=0, rtol=0)

        self.assertTrue(bool((full[:, 1:] > full[:, :-1]).all()))
        torch.testing.assert_close(
            full[1],
            torch.arange(topk, dtype=torch.int32, device="cuda"),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            full[0],
            torch.arange(width // 2, width, dtype=torch.int32, device="cuda"),
            atol=0,
            rtol=0,
        )

    def test_row_windows_and_padding_are_canonical(self):
        scores = torch.tensor(
            [
                [9.0, 8.0, 7.0, 6.0, 5.0, 4.0],
                [0.0, 5.0, 1.0, 5.0, 2.0, 5.0],
                [3.0, 1.0, 4.0, 1.0, 5.0, 9.0],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        lengths = torch.tensor([0, 3, 4], dtype=torch.int32, device="cuda")
        row_starts = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")

        expected = torch.tensor(
            [[-1, -1, -1, -1], [0, 1, 2, -1], [0, 1, 2, 3]],
            dtype=torch.int32,
            device="cuda",
        )
        for backend in DSATopKBackend:
            with self.subTest(backend=backend.value):
                actual = backend.topk_func(
                    scores, lengths, topk=4, row_starts=row_starts
                )
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_deterministic_inference_disables_fused_topk(self):
        with envs.SGLANG_DSA_FUSE_TOPK.override(True):
            self.assertTrue(should_use_dsa_fused_topk(False))
            self.assertFalse(should_use_dsa_fused_topk(False, deterministic=True))


if __name__ == "__main__":
    unittest.main()
