import unittest

import torch

from sglang.srt.speculative.eagle_utils import (
    build_tree_kernel_efficient,
    organize_draft_results,
)

device_type = getattr(torch.accelerator.current_accelerator(), "type", "cpu")


class TestBuildEagleTree(unittest.TestCase):
    """Unit tests for build_eagle_tree functionality."""

    def test_build_tree_kernel_efficient(self):
        """Test the build_tree_kernel_efficient function with known inputs and expected outputs."""
        verified_id = torch.tensor([29974, 13], device=device_type, dtype=torch.int32)
        score_list = [
            torch.tensor(
                [
                    [[7.1127e-01, 2.8292e-01, 2.2995e-03, 1.7357e-03]],
                    [[9.7476e-01, 2.2219e-02, 6.5031e-04, 1.3212e-04]],
                ],
                dtype=torch.float32,
                device=device_type,
            ),
            torch.tensor(
                [
                    [
                        [6.9142e-01, 1.2863e-02, 1.6873e-03, 1.1871e-03],
                        [2.4787e-01, 1.8818e-02, 1.4204e-02, 9.2235e-04],
                        [2.2971e-03, 1.6700e-06, 1.8737e-07, 8.3146e-08],
                        [1.2771e-03, 2.4374e-04, 1.7832e-04, 1.1947e-05],
                    ],
                    [
                        [8.4832e-02, 6.6068e-02, 5.8304e-02, 5.7851e-02],
                        [2.3616e-03, 1.1243e-03, 5.4368e-04, 2.7768e-04],
                        [2.5286e-04, 1.5578e-04, 2.8817e-05, 1.2888e-05],
                        [1.2834e-04, 2.5417e-06, 1.1279e-06, 1.6088e-08],
                    ],
                ],
                dtype=torch.float32,
                device=device_type,
            ),
            torch.tensor(
                [
                    [
                        [6.6438e-01, 2.6997e-02, 2.4236e-05, 4.0821e-06],
                        [2.4402e-01, 2.8409e-03, 5.0935e-04, 2.9022e-04],
                        [1.6178e-02, 2.0567e-03, 4.5892e-04, 3.0034e-05],
                        [1.3023e-02, 5.0497e-04, 3.6371e-04, 8.7750e-05],
                    ],
                    [
                        [2.3263e-02, 2.0054e-02, 9.3990e-03, 2.7783e-03],
                        [6.4156e-02, 5.5506e-04, 1.0429e-04, 9.7211e-05],
                        [4.9950e-02, 5.0630e-03, 9.0068e-04, 3.3656e-04],
                        [7.5817e-03, 8.5731e-04, 6.9972e-04, 6.0793e-04],
                    ],
                ],
                dtype=torch.float32,
                device=device_type,
            ),
            torch.tensor(
                [
                    [
                        [6.6420e-01, 1.0525e-04, 6.5864e-05, 1.2253e-06],
                        [1.3019e-01, 1.0461e-01, 5.2083e-03, 1.6777e-03],
                        [2.0103e-02, 6.7335e-03, 1.2625e-04, 1.0364e-05],
                        [1.5142e-02, 7.0819e-04, 9.6595e-05, 8.7951e-05],
                    ],
                    [
                        [5.8608e-02, 1.8840e-03, 7.8535e-04, 4.4400e-04],
                        [1.2185e-02, 2.0684e-03, 1.7418e-03, 1.4327e-03],
                        [6.2455e-03, 6.1487e-03, 2.6862e-03, 1.8034e-03],
                        [1.8590e-03, 1.6151e-03, 1.2481e-03, 3.6038e-04],
                    ],
                ],
                dtype=torch.float32,
                device=device_type,
            ),
        ]
        token_list = [
            torch.tensor(
                [[29896, 29906, 29900, 29945], [13, 2, 29871, 28956]],
                dtype=torch.int64,
                device=device_type,
            ),
            torch.tensor(
                [
                    [
                        29889,
                        29974,
                        29945,
                        29900,
                        29974,
                        29922,
                        29930,
                        29958,
                        29889,
                        29974,
                        29930,
                        29945,
                        29974,
                        29922,
                        29930,
                        29958,
                    ],
                    [
                        22550,
                        4136,
                        16492,
                        8439,
                        29871,
                        2,
                        3001,
                        13,
                        2,
                        13,
                        29906,
                        29946,
                        2,
                        13,
                        29871,
                        259,
                    ],
                ],
                device=device_type,
            ),
            torch.tensor(
                [
                    [
                        29946,
                        29945,
                        29953,
                        29906,
                        29896,
                        29945,
                        29900,
                        29906,
                        29896,
                        29945,
                        29906,
                        29953,
                        29896,
                        29945,
                        29906,
                        29946,
                    ],
                    [
                        29871,
                        2,
                        29901,
                        29889,
                        29871,
                        2,
                        395,
                        259,
                        29901,
                        29871,
                        2,
                        29889,
                        3001,
                        1234,
                        7146,
                        2186,
                    ],
                ],
                device=device_type,
            ),
            torch.tensor(
                [
                    [
                        29946,
                        29974,
                        29945,
                        29930,
                        29889,
                        29922,
                        29974,
                        29930,
                        29974,
                        29946,
                        29930,
                        29922,
                        29889,
                        29974,
                        29945,
                        29922,
                    ],
                    [
                        29941,
                        29906,
                        2,
                        29946,
                        29871,
                        450,
                        319,
                        14990,
                        29946,
                        29941,
                        2,
                        29906,
                        29871,
                        2,
                        3001,
                        13,
                    ],
                ],
                device=device_type,
            ),
        ]
        parents_list = [
            torch.tensor(
                [[-1, 0, 1, 2, 3], [-1, 0, 1, 2, 3]], dtype=torch.int64, device="cuda"
            ),
            torch.tensor(
                [[4, 8, 9, 10], [4, 5, 6, 7]], dtype=torch.int64, device="cuda"
            ),
            torch.tensor(
                [[20, 24, 21, 28], [24, 28, 20, 21]], dtype=torch.int64, device="cuda"
            ),
            torch.tensor(
                [[36, 40, 41, 44], [36, 40, 44, 45]], dtype=torch.int64, device="cuda"
            ),
        ]
        seq_lens = torch.tensor([5, 10], dtype=torch.int64, device="cuda")
        topk = 4
        depth = 4
        num_draft_token = 8

        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, num_draft_token
        )

        (
            tree_mask,
            position,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            verified_id=verified_id,
            parent_list=parent_list,
            top_scores_index=top_scores_index,
            draft_tokens=draft_tokens,
            seq_lens=seq_lens,
            seq_lens_sum=torch.sum(seq_lens).item(),
            topk=topk,
            spec_steps=depth,
            num_verify_tokens=num_draft_token,
        )

        # Verify expected outputs
        self.assertEqual(
            position.tolist(),
            [5, 6, 6, 7, 7, 8, 8, 9, 10, 11, 12, 12, 12, 12, 13, 14],
            "Position tensor does not match expected values",
        )
        self.assertEqual(
            retrieve_index.tolist(),
            [
                [0, 1, 2, 3, 4, 5, 6, 7],
                [8, 9, 10, 11, 12, 13, 14, 15],
            ],
            "Retrieve index tensor does not match expected values",
        )
        self.assertEqual(
            retrieve_next_token.tolist(),
            [
                [1, 3, 4, 5, 6, 7, -1, -1],
                [1, 2, -1, 6, -1, -1, 7, -1],
            ],
            "Retrieve next token tensor does not match expected values",
        )
        self.assertEqual(
            retrieve_next_sibling.tolist(),
            [
                [-1, 2, -1, -1, -1, -1, -1, -1],
                [-1, -1, 3, 4, 5, -1, -1, -1],
            ],
            "Retrieve next sibling tensor does not match expected values",
        )
        self.assertEqual(
            draft_tokens.tolist(),
            [
                29974,
                29896,
                29906,
                29889,
                29974,
                29946,
                29896,
                29946,
                13,
                13,
                22550,
                4136,
                16492,
                8439,
                29871,
                29941,
            ],
            "Draft tokens tensor does not match expected values",
        )


if __name__ == "__main__":
    unittest.main()
