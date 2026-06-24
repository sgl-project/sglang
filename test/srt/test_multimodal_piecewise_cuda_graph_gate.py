"""Unit test for the multimodal prefill piecewise-CUDA-graph opt-in gate.

ServerArgs disables prefill piecewise CUDA graph for every multimodal model. Some
multimodal archs (whose vision encoder runs eagerly outside the graph and whose LM
prefill captures cleanly) opt back in via
``multimodal_piecewise_cuda_graph_supported_model_archs``. This test pins that gate.
"""

import unittest

from sglang.srt.configs.model_config import (
    is_multimodal_piecewise_cuda_graph_supported,
    multimodal_piecewise_cuda_graph_supported_model_archs,
)


class TestMultimodalPiecewiseCudaGraphGate(unittest.TestCase):
    def test_cohere2_vision_opted_in(self):
        # Cohere2-Vision (command-a / aya-vision family) LM prefill captures cleanly
        # under piecewise CG; it must be opted back in.
        self.assertTrue(
            is_multimodal_piecewise_cuda_graph_supported(
                ["Cohere2VisionForConditionalGeneration"]
            )
        )

    def test_unlisted_multimodal_arch_stays_disabled(self):
        # An arch not on the allow-list keeps the default (disabled) behavior.
        self.assertFalse(
            is_multimodal_piecewise_cuda_graph_supported(
                ["SomeOtherVisionForConditionalGeneration"]
            )
        )
        self.assertFalse(is_multimodal_piecewise_cuda_graph_supported([]))

    def test_allow_list_entries_are_recognized(self):
        for arch in multimodal_piecewise_cuda_graph_supported_model_archs:
            self.assertTrue(is_multimodal_piecewise_cuda_graph_supported([arch]))

    def test_match_within_mixed_arch_list(self):
        self.assertTrue(
            is_multimodal_piecewise_cuda_graph_supported(
                ["OtherArch", "Cohere2VisionForConditionalGeneration"]
            )
        )


if __name__ == "__main__":
    unittest.main()
