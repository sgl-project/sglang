import unittest

import requests

from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kl_test_utils import get_input_ids
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)

register_cuda_ci(
    est_time=700,
    stage="nightly",
    runner_config="8-gpu-b200",
)

GLM52_NVFP4_MODEL_PATH = "nvidia/GLM-5.2-NVFP4"


class TestGLM52PDHiSparseSpec(PDDisaggregationServerBase, GSM8KMixin):
    """1P1D GLM-5.2 MTP with HiSparse target KV and dense draft KV transfer."""

    # Use separate TP4 groups for prefill and decode on eight B200s.
    model = GLM52_NVFP4_MODEL_PATH
    prefill_tp_size = decode_tp_size = 4
    decode_base_gpu_id = 4
    common_args = [
        "--page-size",
        "64",
        "--max-running-requests",
        "200",
        "--max-total-tokens",
        "262144",
        "--mem-fraction-static",
        "0.85",
        "--disable-radix-cache",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--speculative-use-rejection-sampling",
        "--model-loader-extra-config",
        '{"enable_multithread_load": true, "num_threads": 6}',
    ]
    # PD transfers KV bytes without converting layouts. Match FlashMLA decode's
    # packed FP8 layout; Blackwell defaults prefill to TRT-LLM's raw FP8 layout.
    extra_prefill_args = common_args + [
        "--dsa-prefill-backend",
        "flashmla_kv",
        # Startup autotune runs a decode forward even on the prefill worker.
        "--dsa-decode-backend",
        "flashmla_kv",
    ]
    extra_decode_args = common_args + [
        "--attn-dp-size",
        "4",
        "--dsa-decode-backend",
        "flashmla_kv",
        "--enable-hisparse",
        "--hisparse-config",
        '{"top_k": 2048, "device_buffer_size": 4096, "host_to_device_ratio": 5}',
    ]

    # Reuse the NVFP4 accuracy floor from test_dsa_glm52_pd_mtp_cp_layersplit.py.
    gsm8k_accuracy_thres = 0.935
    gsm8k_num_questions = 500
    gsm8k_num_threads = 100
    gsm8k_num_shots = 24
    gsm8k_accept_length_thres = 2.5

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.launch_all()

    def test_gsm8k(self):
        GSM8KMixin.test_gsm8k(self)
        # The PD router does not expose the decode server's speculative metrics.
        response = requests.get(self.decode_url + "/server_info", timeout=30)
        self.assertEqual(response.status_code, 200, response.text)
        states = response.json()["internal_states"]
        self.assertTrue(states)
        for rank, state in enumerate(states):
            with self.subTest(rank=rank):
                self.assertGreater(
                    state["avg_spec_accept_length"], self.gsm8k_accept_length_thres
                )

    def test_long_context_spec_swap(self):
        self._test_long_context_spec_swap(temperature=0)

    def test_non_greedy_spec_swap(self):
        self._test_long_context_spec_swap(temperature=0.6)

    def _test_long_context_spec_swap(self, temperature):
        """Retrieve LongBench needles beyond the GPU cache after PD+MTP transfer."""
        tokenizer = get_tokenizer(self.model, trust_remote_code=True)
        contexts = get_input_ids(
            self.model,
            max_prompt_tokens=32768,
            num_samples=3,
            trust_remote_code=True,
        )
        self.assertEqual(len(contexts), 3)
        prompts, answers = [], []
        for context, length, depth, answer in zip(
            contexts,
            (8192, 16384, 16384),
            (0.1, 0.5, 0.9),
            ("739391", "482617", "956083"),
        ):
            self.assertGreaterEqual(len(context), length)
            context = context[:length]
            split = int(length * depth)
            document = (
                tokenizer.decode(context[:split])
                + f"\nImportant record: the verification code is {answer}.\n"
                + tokenizer.decode(context[split:])
                + "\nWhat is the verification code in the important record? "
                "Reply with the digits only."
            )
            prompts.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": document}],
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
            answers.append(answer)

        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": prompts,
                "sampling_params": {
                    "temperature": temperature,
                    "top_p": 0.95,
                    "max_new_tokens": 128,
                    "ignore_eos": True,
                },
            },
            timeout=600,
        )
        self.assertEqual(response.status_code, 200, response.text)
        results = response.json()
        self.assertEqual(len(results), len(prompts))
        for result, prompt, answer in zip(results, prompts, answers):
            with self.subTest(prompt_tokens=len(prompt), answer=answer):
                meta = result["meta_info"]
                self.assertEqual(meta["prompt_tokens"], len(prompt))
                self.assertGreater(meta["prompt_tokens"], 4096)
                self.assertEqual(meta["completion_tokens"], 128)
                self.assertGreater(meta["spec_verify_ct"], 0)
                # ignore_eos can append more text immediately after the answer.
                self.assertRegex(result["text"], rf"^\s*{answer}(?!\d)")


if __name__ == "__main__":
    unittest.main()
