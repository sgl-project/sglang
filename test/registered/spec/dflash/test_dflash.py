import unittest

import openai

from sglang.srt.environ import envs
from sglang.srt.utils import is_hip, kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
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

register_cuda_ci(est_time=497, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=302, stage="stage-b", runner_config="1-gpu-small-amd")


class TestDFlashServerBase(CustomTestCase, MatchedStopMixin, GSM8KMixin):
    max_running_requests = 64
    attention_backend = "triton" if is_hip() else "flashinfer"
    page_size = 1
    other_launch_args = []
    # Base classes exercise the non-overlap (synchronous) scheduling path.
    disable_overlap = True
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
            # Keep headroom for the draft KV pool + piecewise cuda graph
            # private pools on 32GB CI cards.
            "--mem-fraction-static",
            "0.7",
            "--cuda-graph-bs",
            *[str(i) for i in range(1, cls.max_running_requests + 1)],
        ]
        if cls.disable_overlap:
            launch_args.append("--disable-overlap-schedule")
        launch_args.extend(cls.other_launch_args)
        with (
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
    disable_overlap = False

    def test_radix_attention(self):
        run_radix_attention_test(self.base_url)
        assert self.process.poll() is None


class TestDFlashServerSpecV2PlanStream(TestDFlashServerSpecV2):
    overlap_plan_stream = True


if __name__ == "__main__":
    unittest.main()
