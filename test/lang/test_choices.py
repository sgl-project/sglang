import unittest

import numpy as np

from sglang.lang.choices import (
    greedy_token_selection,
    token_length_normalized,
    unconditional_likelihood_normalized,
)
from sglang.test.test_utils import CustomTestCase

MOCK_CHOICES_INPUT_DATA = {
    "choices": [
        "organ",  # ["organ"]
        "organism",  # ["organ", "ism"]
        "antidisestablishmentarianism",  # ["ant", "id", "is", "est", "ablish", "ment", "arian", "ism"]
    ],
    "normalized_prompt_logprobs": [-0.1, -0.2, -0.05],
    "input_token_logprobs": [
        [[-0.1, 1, None]],
        [[-0.1, 1, None], [-0.3, 2, None]],
        [
            [-0.4, 3, None],
            [-0.25, 4, None],
            [-0.1, 5, None],
            [-0.01, 6, None],
            [-0.01, 7, None],
            [-0.01, 8, None],
            [-0.01, 9, None],
            [-0.01, 2, None],
        ],
    ],
    "output_token_logprobs": [
        [[-0.1, 10, None]],
        [[-0.1, 10, None]],
        [[-0.1, 10, None]],
    ],
    "unconditional_token_logprobs": [
        [[None, 1, None]],
        [[None, 1, None], [-1.4, 2, None]],
        [
            [None, 3, None],
            [-0.25, 4, None],
            [-0.1, 5, None],
            [-0.01, 6, None],
            [-0.01, 7, None],
            [-0.01, 8, None],
            [-0.01, 9, None],
            [-0.01, 2, None],
        ],
    ],
}


class TestChoices(CustomTestCase):

    def test_token_length_normalized(self):
        """Confirm 'antidisestablishmentarianism' is selected due to high confidences for
        its later tokens resulting in highest token length normalized prompt logprob."""
        decision = token_length_normalized(**MOCK_CHOICES_INPUT_DATA)
        assert decision.decision == "antidisestablishmentarianism"

    def test_greedy_token_selection(self):
        """Confirm 'organ' is selected due it having the joint highest initial token
        logprob, and a higher average logprob than organism's second token."""
        decision = greedy_token_selection(**MOCK_CHOICES_INPUT_DATA)
        assert decision.decision == "organ"
        assert np.allclose(
            decision.meta_info["greedy_logprob_matrix"],
            [
                [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1],
                [-0.1, -0.3, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2],
                [-0.4, -0.25, -0.1, -0.01, -0.01, -0.01, -0.01, -0.01],
            ],
            atol=0.01,
        )

    def test_unconditional_likelihood_normalized(self):
        """Confirm 'organism' is selected due to it having the highest average token logprob
        once normalized by the unconditional token logprobs."""
        decision = unconditional_likelihood_normalized(**MOCK_CHOICES_INPUT_DATA)
        assert decision.decision == "organism"
        assert np.allclose(
            decision.meta_info["normalized_unconditional_prompt_logprobs"],
            [-0.1, 0.5, -0.05],
            atol=0.01,
        )


if __name__ == "__main__":
    unittest.main()
