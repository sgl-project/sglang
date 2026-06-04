from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, List

import torch

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.layers.sampler import Sampler, register_sampler_backend
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.mock_model.utils import MOCK_MODEL_PATH
from sglang.test.test_utils import CustomTestCase

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")


CUSTOMIZED_INFO_FIELD = "sampled_token_ids_copy"
CUSTOMIZED_INFO_SAMPLER_BACKEND = "customized_info_probe"
_INPUT_IDS = [464, 9345, 3958, 1752, 13]
_MAX_NEW_TOKENS = 17


class CustomizedInfoSampler(Sampler):
    """Sampler probe that mirrors every sampled token into customized_info.

    The scheduler already appends sampled token ids to each request's output_ids.
    By copying the same values into customized_info at the sampler boundary, the
    test can assert that customized_info is sliced and accumulated exactly like
    output_ids throughout the scheduler -> tokenizer manager -> Engine path.
    """

    def forward(
        self,
        logits_output: "LogitsProcessorOutput",
        sampling_info: "SamplingBatchInfo",
        return_logprob: bool,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[List[int]],
        positions: torch.Tensor,
    ) -> torch.Tensor:
        batch_next_token_ids = super().forward(
            logits_output,
            sampling_info,
            return_logprob,
            top_logprobs_nums,
            token_ids_logprobs,
            positions,
        )

        if logits_output.customized_info is None:
            logits_output.customized_info = {}
        logits_output.customized_info[CUSTOMIZED_INFO_FIELD] = (
            batch_next_token_ids.detach().cpu().tolist()
        )
        return batch_next_token_ids


def install_customized_info_sampler() -> None:
    # Register before ServerArgs validation in the parent and before sampler
    # construction in the scheduler subprocess.
    register_sampler_backend(
        CUSTOMIZED_INFO_SAMPLER_BACKEND,
        CustomizedInfoSampler,
    )


def run_scheduler_process_with_customized_info_sampler(*args, **kwargs):
    # Engine launches the scheduler in a subprocess. Install the sampler there
    # too so create_sampler() can resolve CUSTOMIZED_INFO_SAMPLER_BACKEND.
    install_customized_info_sampler()
    return run_scheduler_process(*args, **kwargs)


class _CustomizedInfoEngine(Engine):
    run_scheduler_process_func = staticmethod(
        run_scheduler_process_with_customized_info_sampler
    )


class TestCustomizedInfoStreaming(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        install_customized_info_sampler()
        cls.engine = _CustomizedInfoEngine(
            model_path=MOCK_MODEL_PATH,
            load_format="dummy",
            sampling_backend=CUSTOMIZED_INFO_SAMPLER_BACKEND,
            incremental_streaming_output=True,
            skip_tokenizer_init=True,
            disable_cuda_graph=True,
            disable_piecewise_cuda_graph=True,
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
