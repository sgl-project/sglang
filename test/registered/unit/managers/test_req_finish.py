import unittest
from array import array
from unittest.mock import patch

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.mixin.req import DllmReqPhase
from sglang.srt.managers.schedule_batch import (
    FINISH_LENGTH,
    FINISH_MATCHED_TOKEN,
    Req,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class TestReqFinish(CustomTestCase):
    def _req(self, rid: str):
        req = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=[1],
            sampling_params=SamplingParams(
                max_new_tokens=2,
                stop_token_ids=[7],
                stop_strs=[],
                stop_regex_strs=[],
            ),
            vocab_size=1000,
        )
        req.sampling_params.stop_strs = []
        req.sampling_params.stop_regex_strs = []
        req.output_ids = [5, 7]
        return req

    def test_update_finish_state_keeps_length_priority_by_default(self):
        req = self._req("length-first")

        req.update_finish_state(new_accepted_len=2)

        self.assertIsInstance(req.finished_reason, FINISH_LENGTH)
        self.assertEqual(req.finished_len, 2)

    def test_check_finished_can_check_stop_token_before_length(self):
        req = self._req("stop-token-first")

        req.check_finished_stop_before_length(new_accepted_len=2)

        self.assertIsInstance(req.finished_reason, FINISH_MATCHED_TOKEN)
        self.assertEqual(req.finished_len, 2)

    def test_check_finished_checks_full_block_stop_string_before_stop_token(self):
        req = self._req("stop-string-first")

        with (
            patch.object(req, "_check_vocab_boundary_finish", return_value=False),
            patch.object(
                req, "_check_str_based_finish", return_value=True
            ) as check_str,
            patch.object(req, "_check_token_based_finish") as check_token,
        ):
            req.check_finished_stop_before_length(new_accepted_len=2)

        check_str.assert_called_once_with(2)
        check_token.assert_not_called()


class TestDllmReqCompatibility(CustomTestCase):
    @staticmethod
    def _config(causal_context: bool):
        return DllmConfig(
            algorithm="FastDiffuser",
            algorithm_config={},
            block_size=32,
            mask_id=100,
            max_running_requests=1,
            max_steps=32,
            causal_context=causal_context,
        )

    def _req(self, causal_context: bool):
        return Req(
            rid=f"causal-{causal_context}",
            origin_input_text="",
            origin_input_ids=array("q", [1]),
            sampling_params=SamplingParams(max_new_tokens=2),
            dllm_config=self._config(causal_context),
        )

    def test_noncausal_short_prompt_keeps_decode_first_behavior(self):
        req = self._req(causal_context=False)
        self.assertEqual(req.dllm_phase, DllmReqPhase.INCOMING_DECODE)

        req._init_fill_ids_for_dllm()
        self.assertEqual(req.dllm_block_offset, 0)

    def test_causal_short_prompt_prefills_prompt_cache(self):
        req = self._req(causal_context=True)
        self.assertEqual(req.dllm_phase, DllmReqPhase.INCOMING_PREFILL)

        req._init_fill_ids_for_dllm()
        self.assertEqual(req.dllm_block_offset, len(req.origin_input_ids))


if __name__ == "__main__":
    unittest.main()
