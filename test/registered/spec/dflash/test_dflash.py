import unittest

import openai

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.matched_stop_kit import MatchedStopMixin
from sglang.test.kits.radix_cache_server_kit import (
    gen_radix_tree,
    run_radix_attention_test,
)
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_DFLASH,
    DEFAULT_TARGET_MODEL_DFLASH,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=302, stage="base-b", runner_config="1-gpu-small")
register_cuda_ci(est_time=360, stage="extra-b", runner_config="2-gpu-large")


class TestDFlashServerBase(CustomTestCase, MatchedStopMixin, GSM8KMixin):
    max_running_requests = 64
    attention_backend = "flashinfer"
    page_size = 1
    other_launch_args = []
    spec_v2 = False
    overlap_plan_stream = False
    model = DEFAULT_TARGET_MODEL_DFLASH
    draft_model = DEFAULT_DRAFT_MODEL_DFLASH
    gsm8k_accuracy_thres = 0.75
    gsm8k_accept_length_thres = 2.8

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        launch_args = [
            "--trust-remote-code",
            "--attention-backend",
            cls.attention_backend,
            "--speculative-algorithm",
            "DFLASH",
            "--speculative-draft-model-path",
            cls.draft_model,
            "--page-size",
            str(cls.page_size),
            "--max-running-requests",
            str(cls.max_running_requests),
            "--cuda-graph-bs",
            *[str(i) for i in range(1, cls.max_running_requests + 1)],
        ]
        launch_args.extend(cls.other_launch_args)
        with (
            envs.SGLANG_ENABLE_SPEC_V2.override(cls.spec_v2),
            envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.override(cls.overlap_plan_stream),
            envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(1),
            envs.SGLANG_ENABLE_ASYNC_ASSERT.override(True),
            envs.SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN.override(True),
        ):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=launch_args,
            )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_early_stop(self):
        client = openai.Client(base_url=self.base_url + "/v1", api_key="EMPTY")
        for i in range(8):
            max_tokens = (i % 3) + 1
            response = client.completions.create(
                model=self.model,
                prompt=f"There are {i} apples on the table. How to divide them equally?",
                max_tokens=max_tokens,
                temperature=0,
            )
            text = response.choices[0].text
            print(f"early_stop: max_tokens={max_tokens}, text={text!r}")
        assert self.process.poll() is None

    def test_eos_handling(self):
        client = openai.Client(base_url=self.base_url + "/v1", api_key="EMPTY")
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "Today is a sunny day and I like"}],
            max_tokens=256,
            temperature=0.1,
        )
        text = response.choices[0].message.content
        print(f"eos_handling: text={text!r}")
        self.assertNotIn("<|eot_id|>", text)
        self.assertNotIn("<|end_of_text|>", text)
        assert self.process.poll() is None

    def test_greedy_determinism(self):
        client = openai.Client(base_url=self.base_url + "/v1", api_key="EMPTY")
        prompt = "The capital of France is"
        outputs = []
        for _ in range(2):
            response = client.completions.create(
                model=self.model,
                prompt=prompt,
                max_tokens=32,
                temperature=0,
            )
            outputs.append(response.choices[0].text)
        print(f"determinism: {outputs=}")
        self.assertEqual(outputs[0], outputs[1])
        assert self.process.poll() is None


class TestDFlashServerPage256(TestDFlashServerBase):
    page_size = 256

    def test_radix_attention(self):
        import requests

        nodes = gen_radix_tree(num_nodes=50)
        data = {
            "input_ids": [node["input_ids"] for node in nodes],
            "sampling_params": [
                {"max_new_tokens": node["decode_len"], "temperature": 0}
                for node in nodes
            ],
        }
        res = requests.post(self.base_url + "/generate", json=data)
        assert res.status_code == 200
        assert self.process.poll() is None


class TestDFlashServerChunkedPrefill(TestDFlashServerBase):
    other_launch_args = ["--chunked-prefill-size", "4"]


class TestDFlashServerNoCudaGraph(TestDFlashServerBase):
    other_launch_args = ["--disable-cuda-graph"]


class TestDFlashServerSpecV2(TestDFlashServerBase):
    spec_v2 = True

    def test_radix_attention(self):
        run_radix_attention_test(self.base_url)
        assert self.process.poll() is None


class TestDFlashServerSpecV2PlanStream(TestDFlashServerSpecV2):
    overlap_plan_stream = True


class TestDFlashServerSpecV2DPAttention(TestDFlashServerSpecV2):
    model = "Qwen/Qwen3-8B"
    draft_model = "z-lab/Qwen3-8B-DFlash-b16"
    max_running_requests = 16
    other_launch_args = [
        "--tp-size",
        "2",
        "--dp-size",
        "2",
        "--enable-dp-attention",
        "--mem-fraction-static",
        "0.8",
    ]

    def test_finish_stop_eos(self):
        qwen_format_prompt = """\
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2 + 2?<|im_end|>
<|im_start|>assistant
"""
        qwen_eos_token_ids = [151643, 151645]
        self._run_completions_generation(
            prompt=qwen_format_prompt,
            max_tokens=1000,
            finish_reason="stop",
            matched_stop=qwen_eos_token_ids,
        )
        self._run_chat_completions_generation(
            prompt="What is 2 + 2?",
            max_tokens=1000,
            finish_reason="stop",
            matched_stop=qwen_eos_token_ids,
        )

    @unittest.skip(
        "Qwen DP-attention coverage is a distributed DFLASH smoke; "
        "GSM8K accuracy is covered by the base DFLASH classes."
    )
    def test_gsm8k(self):
        pass

    @unittest.skip(
        "Qwen DP-attention coverage is a distributed DFLASH smoke; "
        "radix stress is covered by the base DFLASH spec-v2 class."
    )
    def test_radix_attention(self):
        pass


if __name__ == "__main__":
    unittest.main()
