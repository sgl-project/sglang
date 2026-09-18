from __future__ import annotations

import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.customized_info_sampler import (
    CUSTOMIZED_INFO_FIELD,
    CUSTOMIZED_INFO_SAMPLER_BACKEND,
    CustomizedInfoEngine,
    install_customized_info_sampler,
)
from sglang.test.mock_model.utils import MOCK_MODEL_PATH
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=26, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=120, stage="stage-b", runner_config="1-gpu-small-amd")


_INPUT_IDS = [464, 9345, 3958, 1752, 13]
_MAX_NEW_TOKENS = 17


class TestCustomizedInfoStreaming(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        install_customized_info_sampler()
        cls.engine = CustomizedInfoEngine(
            model_path=MOCK_MODEL_PATH,
            load_format="dummy",
            sampling_backend=CUSTOMIZED_INFO_SAMPLER_BACKEND,
            incremental_streaming_output=True,
            skip_tokenizer_init=True,
            disable_cuda_graph=True,
            disable_radix_cache=True,
            random_seed=0,
            log_level="error",
            mem_fraction_static=0.5,
            max_total_tokens=1024,
        )

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()

    def _sampling_params(self, *, stream_interval: int | None = None) -> dict:
        sampling_params = {
            "temperature": 0.0,
            "max_new_tokens": _MAX_NEW_TOKENS,
            "ignore_eos": True,
        }
        if stream_interval is not None:
            sampling_params["stream_interval"] = stream_interval
        return sampling_params

    def _generate(self, *, stream: bool, stream_interval: int | None = None):
        self.engine.flush_cache()
        # skip_tokenizer_init keeps this test focused on streaming output
        # handling; input_ids bypass tokenizer setup while the real Engine,
        # scheduler, and tokenizer-manager response path still run.
        return self.engine.generate(
            input_ids=_INPUT_IDS,
            sampling_params=self._sampling_params(stream_interval=stream_interval),
            stream=stream,
        )

    def _assert_customized_info_matches_output_ids(self, output: dict):
        # For streaming chunks this should compare per-chunk lists. For the
        # non-streaming final response it should compare fully accumulated
        # lists. Either failure means customized_info drifted from output_ids.
        self.assertIn("output_ids", output)
        self.assertIn("meta_info", output)
        self.assertIn(CUSTOMIZED_INFO_FIELD, output["meta_info"])
        self.assertEqual(
            output["meta_info"][CUSTOMIZED_INFO_FIELD], output["output_ids"]
        )

    def test_non_streaming_returns_accumulated_customized_info(self):
        output = self._generate(stream=False)

        self._assert_customized_info_matches_output_ids(output)
        self.assertEqual(len(output["output_ids"]), _MAX_NEW_TOKENS)

    def test_incremental_streaming_returns_chunk_customized_info(self):
        chunks = list(self._generate(stream=True, stream_interval=1))

        self.assertEqual(len(chunks), _MAX_NEW_TOKENS)
        output_ids = []
        for chunk in chunks:
            self._assert_customized_info_matches_output_ids(chunk)
            output_ids.extend(chunk["output_ids"])
        self.assertEqual(len(output_ids), _MAX_NEW_TOKENS)

    def test_incremental_streaming_interval_returns_chunk_customized_info(self):
        chunks = list(self._generate(stream=True, stream_interval=4))

        # stream_interval should coalesce multiple scheduler token events into
        # at least one multi-token Engine chunk while preserving per-chunk
        # customized_info alignment.
        self.assertGreater(len(chunks), 1)
        self.assertTrue(any(len(chunk["output_ids"]) > 1 for chunk in chunks))
        output_ids = []
        for chunk in chunks:
            self._assert_customized_info_matches_output_ids(chunk)
            output_ids.extend(chunk["output_ids"])
        self.assertEqual(len(output_ids), _MAX_NEW_TOKENS)


if __name__ == "__main__":
    unittest.main()
