import unittest

from sglang.srt.utils import is_sm100_supported, kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.basic_api_contract_kit import BasicAPIContractMixin
from sglang.test.kits.basic_decode_correctness_kit import BasicDecodeCorrectnessMixin
from sglang.test.kits.basic_scheduler_stress_kit import BasicSchedulerStressMixin
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.fwd_occupancy_kit import FwdOccupancyMixin
from sglang.test.kits.json_constrained_kit import JSONConstrainedMixin
from sglang.test.kits.spec_server_kits import SpecGrammarKit, SpecLogprobKit
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-large")

TARGET_MODEL = "Qwen/Qwen3-14B"
DRAFT_MODEL = "deepseek-ai/dspark_qwen3_14b_block7"

# trtllm_mha prefill requires SM100 (Blackwell); use the Hopper-native pair elsewhere.
if is_sm100_supported():
    ATTENTION_BACKEND = "trtllm_mha"
    DRAFT_ATTENTION_BACKEND = "fa4"
else:
    ATTENTION_BACKEND = "fa3"
    DRAFT_ATTENTION_BACKEND = "fa3"


class TestBasicSanityDSpark(
    BasicAPIContractMixin,
    BasicDecodeCorrectnessMixin,
    BasicSchedulerStressMixin,
    FwdOccupancyMixin,
    GSM8KMixin,
    JSONConstrainedMixin,
    SpecGrammarKit,
    SpecLogprobKit,
    CustomTestCase,
):
    served_model_name = TARGET_MODEL
    model = TARGET_MODEL

    fwd_occupancy_threshold = 60
    fwd_occupancy_max_new_tokens = 4096
    fwd_occupancy_acc_length_threshold: float = 2.0

    gsm8k_num_questions = 200
    gsm8k_accuracy_thres = 0.80
    gsm8k_accept_length_thres = 2.0

    attention_backend = ATTENTION_BACKEND
    draft_attention_backend = DRAFT_ATTENTION_BACKEND

    process = None

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            TARGET_MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                cls.attention_backend,
                "--speculative-draft-attention-backend",
                cls.draft_attention_backend,
                "--speculative-algorithm",
                "DSPARK",
                "--speculative-draft-model-path",
                DRAFT_MODEL,
                "--cuda-graph-max-bs-decode",
                "4",
                "--mem-fraction-static",
                "0.7",
                "--page-size",
                "1",
                "--enable-metrics",
                "--disable-piecewise-cuda-graph",
            ],
            env={
                "SGLANG_ENABLE_METRICS_DEVICE_TIMER": "1",
                "SGLANG_RAGGED_VERIFY_MODE": "compact",
            },
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            kill_process_tree(cls.process.pid)

    def test_sampling_mask(self):
        import requests

        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": ["The capital of France is"] * 2,
                "sampling_params": [
                    {
                        "temperature": 0.0,
                        "top_k": 1,
                        "max_new_tokens": 7,
                        "ignore_eos": True,
                    },
                    {
                        "temperature": 1.0,
                        "top_k": 10,
                        "top_p": 0.95,
                        "max_new_tokens": 7,
                        "ignore_eos": True,
                    },
                ],
                "return_sampling_mask": True,
            },
            timeout=60,
        )
        self.assertEqual(response.status_code, 200, response.text)
        outputs = response.json()
        self.assertEqual(len(outputs), 2)
        for request_idx, output in enumerate(outputs):
            output_ids = output["output_ids"]
            meta_info = output["meta_info"]
            masks = meta_info["output_token_sampling_mask"]
            logprobs = meta_info["output_token_sampling_logprobs"]
            self.assertEqual(len(output_ids), 7)
            self.assertEqual(len(masks), len(output_ids))
            self.assertEqual(len(logprobs), len(output_ids))
            for output_id, mask, logprob in zip(output_ids, masks, logprobs):
                self.assertIn(output_id, mask)
                if request_idx == 0:
                    self.assertEqual(mask, [output_id])
                    self.assertEqual(logprob, 0.0)


if __name__ == "__main__":
    unittest.main()
