# Copyright 2026 SGLang Team
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
"""
Test beam search output comparison between transformers and sglang.

This test compares the beam search results from HuggingFace transformers
and sglang to ensure they produce similar outputs.

Usage:
python3 test_beam_search_diff.py -v
"""

import unittest
from typing import List, Set

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sglang as sgl
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST

register_cuda_ci(est_time=60, suite="per-commit-1-gpu")
register_amd_ci(est_time=60, suite="per-commit-1-gpu-amd")


class TestBeamSearchDiff(unittest.TestCase):
    """Test beam search output comparison between transformers and sglang."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are shared across all tests."""
        cls.model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.prompt = "Hello SGLang"
        cls.max_new_tokens = 10
        cls.overlap_threshold = 0.8

    def _get_transformers_beam_sequences(
        self, prompt: str, beam_width: int, max_new_tokens: int
    ) -> List[str]:
        """Get beam search sequences from transformers."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path, dtype="auto", device_map="auto"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=beam_width,
                num_return_sequences=beam_width,
                do_sample=False,
            )

        sequences = []
        for seq_id in generated_ids:
            generated_tokens = seq_id[input_length:].cpu().tolist()
            # Decode tokens to text
            decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            sequences.append(decoded_text)

        del model
        torch.cuda.empty_cache()

        return sequences

    def _get_sglang_beam_sequences(
        self, prompt: str, beam_width: int, max_new_tokens: int
    ) -> List[str]:
        """Get beam search sequences from sglang."""
        engine = sgl.Engine(
            model_path=self.model_path,
            enable_beam_search=True,
            disable_radix_cache=True,
            disable_overlap_schedule=True,
            disable_cuda_graph=True,
        )

        sampling_params = {
            "max_new_tokens": max_new_tokens,
            "n": beam_width,
        }

        outputs = engine.generate(prompt, sampling_params=sampling_params)

        sequences = []

        if isinstance(outputs, list):
            for beam_result in outputs:
                sequences.append(beam_result["text"])
        else:
            raise ValueError(
                "Expected list output from sglang engine.generate(). "
                f"Got type: {type(outputs)}, Output: {outputs}"
            )

        engine.shutdown()

        return sequences

    def _calculate_sequence_overlap(
        self, sequences1: List[str], sequences2: List[str]
    ) -> float:
        """Calculate the overlap ratio between two sets of sequences.

        Returns: intersection / beam_width (expected number of sequences)
        """
        set1: Set[str] = set(sequences1)
        set2: Set[str] = set(sequences2)

        intersection = set1 & set2
        beam_width = len(sequences1)
        if beam_width == 0:
            return 0.0
        overlap_ratio = len(intersection) / beam_width
        return overlap_ratio

    def test_beam_search_different_widths(self):
        """Test beam search with different beam widths."""
        beam_widths = [2, 10]

        for beam_width in beam_widths:
            print(f"\n{'='*60}")
            print(f"Testing with beam width: {beam_width}")
            print(f"{'='*60}")

            transformers_sequences = self._get_transformers_beam_sequences(
                self.prompt, beam_width, self.max_new_tokens
            )
            sglang_sequences = self._get_sglang_beam_sequences(
                self.prompt, beam_width, self.max_new_tokens
            )

            overlap_ratio = self._calculate_sequence_overlap(
                transformers_sequences, sglang_sequences
            )

            print(f"\nOverlap ratio: {overlap_ratio:.2%}")

            self.assertGreaterEqual(
                overlap_ratio,
                self.overlap_threshold,
                f"Beam search overlap {overlap_ratio:.2%} is below threshold "
                f"{self.overlap_threshold:.2%} for beam width {beam_width}",
            )


if __name__ == "__main__":
    unittest.main()
