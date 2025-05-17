import multiprocessing as mp
import os
import unittest
from types import SimpleNamespace

import requests
import torch

from sglang.srt.utils import get_device_sm, kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.runners import HFRunner, SRTRunner, check_close_model_outputs
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST_EAGLE3,
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_MODEL_NAME_FOR_TEST_MLA_NEXTN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

GSM_DATASET_PATH = None

# In case of some machine lack internet connection, we can set OFFLINE_MODE to True.
OFFLINE_MODE = False

# Change the path below when OFFLINE_MODE is True.
OFFLINE_PATH_DICT = {
    DEFAULT_MODEL_NAME_FOR_TEST: "/shared/public/elr-models/meta-llama/Meta-Llama-3.1-8B-Instruct",
    DEFAULT_MODEL_NAME_FOR_TEST_EAGLE3: "/shared/public/elr-models/jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B",
    DEFAULT_MODEL_NAME_FOR_TEST_MLA: "/shared/public/sharing/deepseek/dsv3-test/snapshots/",
    DEFAULT_MODEL_NAME_FOR_TEST_MLA_NEXTN: "/shared/public/sharing/deepseek/dsv3-test-NextN/snapshots/",
    GSM_DATASET_PATH: "/shared/public/data/gsm8k/test.jsonl",
}


if OFFLINE_MODE:
    DEFAULT_MODEL_NAME_FOR_TEST = OFFLINE_PATH_DICT[DEFAULT_MODEL_NAME_FOR_TEST]
    DEFAULT_MODEL_NAME_FOR_TEST_EAGLE3 = OFFLINE_PATH_DICT[
        DEFAULT_MODEL_NAME_FOR_TEST_EAGLE3
    ]
    DEFAULT_MODEL_NAME_FOR_TEST_MLA = OFFLINE_PATH_DICT[DEFAULT_MODEL_NAME_FOR_TEST_MLA]
    DEFAULT_MODEL_NAME_FOR_TEST_MLA_NEXTN = OFFLINE_PATH_DICT[
        DEFAULT_MODEL_NAME_FOR_TEST_MLA_NEXTN
    ]
    GSM_DATASET_PATH = OFFLINE_PATH_DICT[GSM_DATASET_PATH]


DEFAULT_PROMPTS = [
    "Apple is red. Banana is Yellow. " * 800 + "Apple is",
    "The capital of the United Kingdom is",
    "AI is a field of computer science focused on",
]

# Default server arguments shared across all tests
DEFAULT_SERVER_ARGS = [
    "--trust-remote-code",
    "--cuda-graph-max-bs",
    "8",
    "--prefill-attention-backend",
    "fa3",
    "--decode-attention-backend",
    "flashinfer",
]


@unittest.skipIf(get_device_sm() < 90, "Test requires CUDA SM 90 or higher")
class HybridAttnBackendSGLangRunnerTest(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def test_hybrid_attn_backend(self):
        prefill_tolerance: float = 5e-2
        decode_tolerance: float = 5e-2
        rouge_l_tolerance: float = 1
        model_path = DEFAULT_MODEL_NAME_FOR_TEST
        max_new_tokens = 32
        torch_dtype = torch.float16

        with HFRunner(
            model_path,
            torch_dtype=torch_dtype,
            model_type="generation",
            trust_remote_code=True,
        ) as hf_runner:
            hf_outputs = hf_runner.forward(
                DEFAULT_PROMPTS, max_new_tokens=max_new_tokens
            )

        with SRTRunner(
            model_path,
            torch_dtype=torch_dtype,
            model_type="generation",
            trust_remote_code=True,
            prefill_attention_backend="fa3",
            decode_attention_backend="flashinfer",
        ) as srt_runner:
            srt_outputs = srt_runner.forward(
                DEFAULT_PROMPTS, max_new_tokens=max_new_tokens
            )

        check_close_model_outputs(
            hf_outputs=hf_outputs,
            srt_outputs=srt_outputs,
            prefill_tolerance=prefill_tolerance,
            decode_tolerance=decode_tolerance,
            rouge_l_tolerance=rouge_l_tolerance,
            debug_text=f"model_path={model_path} prompts={DEFAULT_PROMPTS}",
        )


@unittest.skipIf(get_device_sm() < 90, "Test requires CUDA SM 90 or higher")
class TestHybridAttnBackendE2EBase(CustomTestCase):

    model = DEFAULT_MODEL_NAME_FOR_TEST
    base_url = DEFAULT_URL_FOR_TEST
    accuracy_threshold = 0.65  # derived tests need to override this
    speculative_decode = False
    spec_decode_threshold = 1.0  # derived spec decoding tests need to override this

    @classmethod
    def get_server_args(cls):
        """Return the arguments for the server launch. Override in subclasses."""
        return DEFAULT_SERVER_ARGS

    @classmethod
    def setUpClass(cls):
        # disable deep gemm precompile to make launch server faster
        # please don't do this if you want to make your inference workload faster
        os.environ["SGL_JIT_DEEPGEMM_PRECOMPILE"] = "false"
        os.environ["SGL_ENABLE_JIT_DEEPGEMM"] = "false"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.get_server_args(),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            num_shots=4,
            num_questions=100,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
            data_path=GSM_DATASET_PATH,
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        # Use the appropriate metric key based on the test class
        metric_key = "accuracy"
        self.assertGreater(metrics[metric_key], self.accuracy_threshold)

        if self.speculative_decode:
            server_info = requests.get(self.base_url + "/get_server_info")
            avg_spec_accept_length = server_info.json()["internal_states"][0][
                "avg_spec_accept_length"
            ]
            print(f"{avg_spec_accept_length=}")
            self.assertGreater(avg_spec_accept_length, self.spec_decode_threshold)


class TestHybridAttnBackendE2EMLA(TestHybridAttnBackendE2EBase):
    accuracy_threshold = 0.60
    model = DEFAULT_MODEL_NAME_FOR_TEST_MLA

    @classmethod
    def get_server_args(cls):
        return DEFAULT_SERVER_ARGS


class TestHybridAttnBackendE2ETorchCompile(TestHybridAttnBackendE2EBase):
    accuracy_threshold = 0.65

    @classmethod
    def get_server_args(cls):
        return DEFAULT_SERVER_ARGS + ["--enable-torch-compile"]


if __name__ == "__main__":
    unittest.main()
