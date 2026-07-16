from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch

from sglang.kernels.ops.speculative.topk1 import draft_extend_topk1_postprocess
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestDraftExtendTopk1Triton(CustomTestCase):
    def test_matches_trace_contract(self):
        device = torch.device("cuda")
        logits = torch.randn((6, 154880), dtype=torch.float32, device=device)
        hidden_states = torch.randn((6, 6144), dtype=torch.bfloat16, device=device)
        dsa_topk_indices = torch.randint(
            0, 32000, (720, 2048), dtype=torch.int32, device=device
        )
        row_indices = torch.tensor([0, 5], dtype=torch.long, device=device)

        # Exercise tie-breaking both within and across the 8192-wide splits.
        logits[0, 11] = 1000.0
        logits[0, 17] = 1000.0
        logits[0, 8194] = 1000.0
        logits[5, 154879] = 1000.0
        expected_topk_index = torch.argmax(logits[row_indices], dim=-1, keepdim=True)
        expected_hidden_states = hidden_states[row_indices]

        for dsa_source in (dsa_topk_indices, None):
            with self.subTest(dsa=dsa_source is not None):
                topk_p, topk_index, selected_hidden_states, selected_dsa = (
                    draft_extend_topk1_postprocess(
                        logits,
                        row_indices,
                        hidden_states,
                        dsa_source,
                    )
                )

                torch.testing.assert_close(
                    topk_index, expected_topk_index, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    topk_p, torch.ones_like(topk_p), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    selected_hidden_states,
                    expected_hidden_states,
                    rtol=0,
                    atol=0,
                )
                if dsa_source is None:
                    self.assertIsNone(selected_dsa)
                else:
                    torch.testing.assert_close(
                        selected_dsa,
                        dsa_source[row_indices],
                        rtol=0,
                        atol=0,
                    )


if __name__ == "__main__":
    unittest.main()
