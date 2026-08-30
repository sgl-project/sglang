import unittest

import torch

from sglang.kernels.ops.speculative.spec_tree import pack_ngram_full_mask
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestNgramFullMask(CustomTestCase):
    def test_pack_variable_length_requests_into_static_buffer(self):
        num_draft_tokens = 3
        draft_tree_mask = torch.tensor(
            [
                [[1, 0, 0], [1, 1, 0], [1, 0, 1]],
                [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
            ],
            dtype=torch.bool,
            device="cuda",
        ).flatten()
        seq_lens = torch.tensor([2, 4], device="cuda")
        seq_lens_host = [2, 4]
        seq_lens_cumsum = torch.empty_like(seq_lens)
        max_context_len = 8
        output = torch.zeros(
            2 * num_draft_tokens * (max_context_len + num_draft_tokens),
            dtype=torch.bool,
            device="cuda",
        )

        pack_ngram_full_mask(
            draft_tree_mask=draft_tree_mask,
            seq_lens=seq_lens,
            num_draft_tokens=num_draft_tokens,
            max_seq_len=max_context_len,
            output=output,
            seq_lens_cumsum=seq_lens_cumsum,
        )

        expected = torch.cat(
            [
                torch.cat(
                    [
                        torch.ones(
                            (num_draft_tokens, seq_lens_host[i]),
                            dtype=torch.bool,
                            device="cuda",
                        ),
                        draft_tree_mask.view(2, num_draft_tokens, num_draft_tokens)[i],
                    ],
                    dim=1,
                ).flatten()
                for i in range(2)
            ]
        )
        self.assertTrue(torch.equal(output[: expected.numel()], expected))
        self.assertFalse(output[expected.numel() :].any().item())


if __name__ == "__main__":
    unittest.main()
