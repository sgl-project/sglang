"""End-to-end parity test for post-capture startup weight loading."""

import unittest

import sglang as sgl
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    CustomTestCase,
)

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")


class TestStartupWeightLoad(CustomTestCase):
    @staticmethod
    def _generate(startup_weight_load_mode=None):
        kwargs = dict(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            dtype="bfloat16",
            random_seed=42,
            cuda_graph_max_bs_decode=1,
            max_total_tokens=256,
        )
        if startup_weight_load_mode is not None:
            kwargs["startup_weight_load_mode"] = startup_weight_load_mode

        engine = None
        try:
            engine = sgl.Engine(**kwargs)
            return engine.generate(
                "The capital of France is",
                sampling_params={
                    "temperature": 0,
                    "max_new_tokens": 8,
                    "ignore_eos": True,
                },
                return_logprob=True,
                logprob_start_len=0,
            )
        finally:
            if engine is not None:
                engine.shutdown()

    def test_overlap_matches_default_serial_startup(self):
        # Omitting the flag is intentional: it pins the merge-safe default path.
        serial = self._generate()
        overlap = self._generate("overlap")

        self.assertEqual(serial["output_ids"], overlap["output_ids"])
        self.assertEqual(serial["text"], overlap["text"])

        serial_logprobs = serial["meta_info"]["output_token_logprobs"]
        overlap_logprobs = overlap["meta_info"]["output_token_logprobs"]
        self.assertEqual(len(serial_logprobs), len(overlap_logprobs))
        self.assertGreater(len(serial_logprobs), 0)
        for index, (serial_item, overlap_item) in enumerate(
            zip(serial_logprobs, overlap_logprobs)
        ):
            self.assertEqual(
                serial_item[1],
                overlap_item[1],
                f"token id differs at output position {index}",
            )
            self.assertAlmostEqual(
                serial_item[0],
                overlap_item[0],
                delta=1e-5,
                msg=f"logprob differs at output position {index}",
            )


if __name__ == "__main__":
    unittest.main()
