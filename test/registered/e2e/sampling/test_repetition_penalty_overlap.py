"""Repetition penalties must include the last output under overlap scheduling."""

import os
import unittest

import sglang as sgl
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=360, stage="base-b", runner_config="1-gpu-small")

# Keep the same checkpoint when overriding this with a local model directory.
MODEL_PATH = os.environ.get("TEST_MODEL_PATH", "Qwen/Qwen3-8B")
PROMPT = "Write the word hello ten times, separated by spaces."


class TestRepetitionPenaltyOverlap(CustomTestCase):
    def test_repetition_penalty_matches_non_overlap(self):
        """The first decode must penalize the token produced by prefill."""
        outputs = {}
        for overlap in (False, True):
            outputs[overlap] = {}
            engine = sgl.Engine(
                model_path=MODEL_PATH,
                tp_size=1,
                context_length=512,
                mem_fraction_static=0.7,
                max_total_tokens=128,
                max_running_requests=4,
                attention_backend="triton",
                sampling_backend="pytorch",
                cuda_graph_backend_decode="disabled",
                cuda_graph_backend_prefill="disabled",
                disable_overlap_schedule=not overlap,
            )
            try:
                for penalty in (1.0, 2.0):
                    result = engine.generate(
                        prompt=PROMPT,
                        sampling_params={
                            "temperature": 0,
                            "repetition_penalty": penalty,
                            "max_new_tokens": 2,
                            "ignore_eos": True,
                        },
                        return_logprob=True,
                    )
                    # Compare token IDs, not decoded text or logprob values.
                    token_ids = [
                        item[1] for item in result["meta_info"]["output_token_logprobs"]
                    ]
                    self.assertEqual(len(token_ids), 2)
                    outputs[overlap][penalty] = token_ids
            finally:
                # Release the first engine's GPU resources before starting the next.
                engine.shutdown()

        self.assertEqual(
            outputs[False][1.0],
            outputs[True][1.0],
            "The no-penalty control must agree across scheduling modes.",
        )
        self.assertNotEqual(
            outputs[False][1.0],
            outputs[False][2.0],
            "The prompt must exercise repetition penalty in the non-overlap baseline.",
        )
        self.assertEqual(
            outputs[False][2.0],
            outputs[True][2.0],
            "Overlap must apply repetition penalty to the latest generated token.",
        )


if __name__ == "__main__":
    unittest.main()
