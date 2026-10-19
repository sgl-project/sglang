import unittest

import torch

from sglang.srt.layers.attention.llada2_cfg_flashinfer_backend import (
    _build_llada_image_custom_mask,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestLLaDA2CFGFlashInferMask(unittest.TestCase):
    def test_builds_per_request_text_query_block_masks(self):
        """Text tokens must not attend to query tokens, and requests stay separate."""
        flattened = _build_llada_image_custom_mask([1, 2], [3, 4], "cpu")

        first = flattened[:9].view(3, 3)
        second = flattened[9:].view(4, 4)
        torch.testing.assert_close(
            first,
            torch.tensor(
                [[True, False, False], [True, True, True], [True, True, True]]
            ),
        )
        torch.testing.assert_close(
            second,
            torch.tensor(
                [
                    [True, True, False, False],
                    [True, True, False, False],
                    [True, True, True, True],
                    [True, True, True, True],
                ]
            ),
        )

    def test_rejects_invalid_conditioning_spans(self):
        with self.assertRaisesRegex(RuntimeError, "metadata batch mismatch"):
            _build_llada_image_custom_mask([1], [2, 3], "cpu")
        with self.assertRaisesRegex(RuntimeError, "must leave query tokens"):
            _build_llada_image_custom_mask([2], [2], "cpu")


if __name__ == "__main__":
    unittest.main()
