import multiprocessing as mp
import random
import threading
import time
import unittest
from types import SimpleNamespace

import requests

import sglang as sgl
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
    DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

acc_rate_tolerance = 0.15


class TestEAGLEEngine(unittest.TestCase):
    BASE_CONFIG = {
        "model_path": DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
        "speculative_draft_model_path": DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
        "speculative_algorithm": "EAGLE",
        "speculative_num_steps": 5,
        "speculative_eagle_topk": 8,
        "speculative_num_draft_tokens": 64,
        "mem_fraction_static": 0.7,
        "cuda_graph_max_bs": 32,
    }

    def setUp(self):
        self.prompt = "Today is a sunny day and I like"
        self.sampling_params = {"temperature": 0, "max_new_tokens": 8}

        ref_engine = sgl.Engine(model_path=DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST)
        self.ref_output = ref_engine.generate(self.prompt, self.sampling_params)["text"]
        ref_engine.shutdown()

    def test_correctness(self):
        configs = [
            self.BASE_CONFIG,
            {**self.BASE_CONFIG, "disable_cuda_graph": True},
            {**self.BASE_CONFIG, "chunked_prefill_size": 2},
        ]

        for config in configs:
            with self.subTest(
                cuda_graph=(
                    "enabled" if len(config) == len(self.BASE_CONFIG) else "disabled"
                ),
                chunked_prefill_size=(
                    config["chunked_prefill_size"]
                    if "chunked_prefill_size" in config
                    else "default"
                ),
            ):
                engine = sgl.Engine(**config)
                try:
                    self._test_basic_generation(engine)
                    self._test_eos_token(engine)
                    self._test_batch_generation(engine)
                finally:
                    engine.shutdown()

    def _test_basic_generation(self, engine):
        output = engine.generate(self.prompt, self.sampling_params)["text"]
        print(f"{output=}, {self.ref_output=}")
        self.assertEqual(output, self.ref_output)

    def _test_eos_token(self, engine):
        prompt = "[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\nToday is a sunny day and I like [/INST]"
        params = {
            "temperature": 0,
            "max_new_tokens": 1024,
            "skip_special_tokens": False,
        }

        tokenizer = get_tokenizer(DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST)
        output = engine.generate(prompt, params)["text"]
        print(f"{output=}")

        tokens = tokenizer.encode(output, truncation=False)
        self.assertNotIn(tokenizer.eos_token_id, tokens)

    def _test_batch_generation(self, engine):
        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        params = {"temperature": 0, "max_new_tokens": 30}

        outputs = engine.generate(prompts, params)
        for prompt, output in zip(prompts, outputs):
            print(f"Prompt: {prompt}")
            print(f"Generated: {output['text']}")
            print("-" * 40)


prompts = [
    "[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nToday is a sunny day and I like[/INST]"
    '[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nWhat are the mental triggers in Jeff Walker\'s Product Launch Formula and "Launch" book?[/INST]',
    "[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nSummarize Russell Brunson's Perfect Webinar Script...[/INST]",
    "[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nwho are you?[/INST]",
    "[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nwhere are you from?[/INST]",
]


class TestEAGLEServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-draft-model-path",
                DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
                "--speculative-num-steps",
                "5",
                "--speculative-eagle-topk",
                "8",
                "--speculative-num-draft-tokens",
                "64",
                "--mem-fraction-static",
                "0.7",
                "--chunked-prefill-size",
                "128",
                "--cuda-graph-max-bs",
                "32",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def send_request(self):
        time.sleep(random.uniform(0, 2))
        for prompt in prompts:
            url = self.base_url + "/generate"
            data = {
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 1024,
                },
            }
            response = requests.post(url, json=data)
            assert response.status_code == 200

    def send_requests_abort(self):
        for prompt in prompts:
            try:
                time.sleep(random.uniform(0, 2))
                url = self.base_url + "/generate"
                data = {
                    "model": "base",
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 1024,
                    },
                }
                # set timeout = 1s, mock disconnected
                requests.post(url, json=data, timeout=1)
            except Exception as e:
                print(e)
                pass

    def test_request_abort(self):
        concurrency = 4
        threads = [
            threading.Thread(target=self.send_request) for _ in range(concurrency)
        ] + [
            threading.Thread(target=self.send_requests_abort)
            for _ in range(concurrency)
        ]
        for worker in threads:
            worker.start()
        for p in threads:
            p.join()

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        self.assertGreater(metrics["accuracy"], 0.20)


def measure_acc_rate(engine):
    tic = time.time()
    prompt = [
        "Human: Give me a fully functional FastAPI server. Show the python code.<|separator|>\n\nAssistant:"
    ]
    sampling_params = {"temperature": 0, "max_new_tokens": 512}
    output = engine.generate(prompt, sampling_params)
    output = output[0]
    latency = time.time() - tic

    if "spec_verify_ct" in output["meta_info"]:
        base_acc_length = (
            output["meta_info"]["completion_tokens"]
            / output["meta_info"]["spec_verify_ct"]
        )
    else:
        base_acc_length = 0.0

    base_speed = output["meta_info"]["completion_tokens"] / latency
    return base_acc_length, base_speed


class TestEagleAcceptanceRate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)
        ref_engine = sgl.Engine(
            model_path=DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
            speculative_draft_model_path=DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
            speculative_algorithm="EAGLE",
            speculative_num_steps=5,
            speculative_eagle_topk=8,
            speculative_num_draft_tokens=64,
            mem_fraction_static=0.7,
            disable_radix_cache=True,
        )
        cls.base_acc_length, cls.base_speed = measure_acc_rate(ref_engine)
        ref_engine.shutdown()
        assert cls.base_acc_length > 4.45

    def test_acc_rate(self):
        base_acc_length, base_speed = self.base_acc_length, self.base_speed
        chunk_engine = sgl.Engine(
            model_path=DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
            speculative_draft_model_path=DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
            speculative_algorithm="EAGLE",
            speculative_num_steps=5,
            speculative_eagle_topk=8,
            speculative_num_draft_tokens=64,
            mem_fraction_static=0.7,
            chunked_prefill_size=2,
            disable_radix_cache=True,
        )
        chunked_acc_length, chunked_base_speed = measure_acc_rate(chunk_engine)
        chunk_engine.shutdown()
        print(base_acc_length, base_speed)
        print(chunked_acc_length, chunked_base_speed)
        assert abs(base_acc_length - chunked_acc_length) < acc_rate_tolerance

    def test_acc_rate_prefix_caching(self):
        base_acc_length, base_speed = self.base_acc_length, self.base_speed
        prefix_caching_engine = sgl.Engine(
            model_path=DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
            speculative_draft_model_path=DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
            speculative_algorithm="EAGLE",
            speculative_num_steps=5,
            speculative_eagle_topk=8,
            speculative_num_draft_tokens=64,
            mem_fraction_static=0.7,
            chunked_prefill_size=4,
            schedule_policy="lpm",
        )
        for _ in range(10):
            acc_length, _ = measure_acc_rate(prefix_caching_engine)
            print(f"{acc_length=}")
            assert abs(base_acc_length - acc_length) < acc_rate_tolerance
        # The second one should hit the prefix cache.
        prefix_caching_engine.shutdown()


class TestEAGLERetract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-draft-model-path",
                DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
                "--speculative-num-steps",
                "5",
                "--speculative-eagle-topk",
                "8",
                "--speculative-num-draft-tokens",
                "64",
                "--mem-fraction-static",
                "0.7",
                "--chunked-prefill-size",
                "128",
                "--max-running-requests",
                "64",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        self.assertGreater(metrics["accuracy"], 0.20)
        # Wait a little bit so that the memory check happens.
        time.sleep(5)


class TestEAGLEServerTriton(TestEAGLEServer):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-draft-model-path",
                DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
                "--speculative-num-steps",
                "5",
                "--speculative-eagle-topk",
                "4",
                "--speculative-num-draft-tokens",
                "8",
                "--mem-fraction-static",
                "0.7",
                "--attention-backend",
                "triton",
                "--cuda-graph-max-bs",
                "16",
            ],
        )


class TestEAGLEEngineTokenMap(unittest.TestCase):
    def setUp(self):
        self.prompt = "Today is a sunny day and I like"
        self.sampling_params = {"temperature": 0, "max_new_tokens": 8}

        ref_engine = sgl.Engine(
            model_path="meta-llama/Meta-Llama-3-8B-Instruct", cuda_graph_max_bs=2
        )
        self.ref_output = ref_engine.generate(self.prompt, self.sampling_params)["text"]
        ref_engine.shutdown()

    def test_correctness(self):
        config = {
            "model_path": "meta-llama/Meta-Llama-3-8B-Instruct",
            "speculative_draft_model_path": "lmsys/sglang-EAGLE-LLaMA3-Instruct-8B",
            "speculative_algorithm": "EAGLE",
            "speculative_num_steps": 5,
            "speculative_eagle_topk": 4,
            "speculative_num_draft_tokens": 8,
            "speculative_token_map": "thunlp/LLaMA3-Instruct-8B-FR-Spec/freq_32768.pt",
            "mem_fraction_static": 0.7,
            "cuda_graph_max_bs": 4,
            "dtype": "bfloat16",
        }

        engine = sgl.Engine(**config)
        try:
            self._test_basic_generation(engine)
            self._test_batch_generation(engine)
        finally:
            engine.shutdown()

    def _test_basic_generation(self, engine):
        output = engine.generate(self.prompt, self.sampling_params)["text"]
        print(f"{output=}, {self.ref_output=}")
        self.assertEqual(output, self.ref_output)

    def _test_batch_generation(self, engine):
        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        params = {"temperature": 0, "max_new_tokens": 30}

        outputs = engine.generate(prompts, params)
        for prompt, output in zip(prompts, outputs):
            print(f"Prompt: {prompt}")
            print(f"Generated: {output['text']}")
            print("-" * 40)


if __name__ == "__main__":
    unittest.main()
