import json
import os
import tempfile
import unittest

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_TARGET_MODEL_NGRAM,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=230, suite="stage-b-test-1-gpu-large")

GSM_DATASET_PATH = None


# Default server arguments shared across all tests
DEFAULT_SERVER_ARGS = [
    "--trust-remote-code",
    "--cuda-graph-max-bs",
    "8",
    "--speculative-algorithm",
    "NGRAM",
    "--speculative-num-draft-tokens",
    "16",
    "--mem-fraction-static",
    0.8,
]

EXTERNAL_SAM_CORPUS_RECORDS = [
    "The capital of France is Paris.",
    "The answer to life, the universe, and everything is 42.",
]


def _safe_remove(path: str):
    if os.path.exists(path):
        os.remove(path)


def _safe_kill_process(process):
    if process is not None and process.poll() is None:
        kill_process_tree(process.pid)


class TestNgramSpeculativeDecodingBase(GSM8KMixin, CustomTestCase):
    model = DEFAULT_TARGET_MODEL_NGRAM
    base_url = DEFAULT_URL_FOR_TEST
    gsm8k_accuracy_thres = 0.79  # derived tests need to override this
    gsm8k_accept_length_thres = 1.8  # derived spec decoding tests need to override this

    @classmethod
    def get_server_args(cls):
        """Return the arguments for the server launch. Override in subclasses."""
        return DEFAULT_SERVER_ARGS + ["--attention-backend", "fa3"]

    @classmethod
    def setUpClass(cls):
        # disable deep gemm precompile to make launch server faster
        # please don't do this if you want to make your inference workload faster
        envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE.set(False)
        envs.SGLANG_ENABLE_JIT_DEEPGEMM.set(False)
        model = cls.model
        cls.process = popen_launch_server(
            model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.get_server_args(),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestNgramSpeculativeDecodingTriton(TestNgramSpeculativeDecodingBase):

    @classmethod
    def get_server_args(cls):
        return DEFAULT_SERVER_ARGS + ["--attention-backend", "triton"]


class TestNgramSpeculativeDecodingFlashinfer(TestNgramSpeculativeDecodingBase):
    @classmethod
    def get_server_args(cls):
        return DEFAULT_SERVER_ARGS + ["--attention-backend", "flashinfer"]


class TestNgramSpeculativeDecodingPaged(TestNgramSpeculativeDecodingBase):

    @classmethod
    def get_server_args(cls):
        return DEFAULT_SERVER_ARGS + [
            "--attention-backend",
            "flashinfer",
            "--page-size",
            "64",
        ]


class TestNgramExternalSamSmoke(CustomTestCase):
    model = DEFAULT_TARGET_MODEL_NGRAM
    base_url = DEFAULT_URL_FOR_TEST
    attention_backends = ("triton", "flashinfer")

    def get_server_args(self, attention_backend):
        return DEFAULT_SERVER_ARGS + [
            "--attention-backend",
            attention_backend,
            "--speculative-ngram-external-corpus-path",
            self.external_corpus_path,
            "--speculative-ngram-external-sam-budget",
            "4",
        ]

    @classmethod
    def setUpClass(cls):
        envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE.set(False)
        envs.SGLANG_ENABLE_JIT_DEEPGEMM.set(False)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", prefix="ngram_external_sam_", delete=False
        ) as f:
            for record in EXTERNAL_SAM_CORPUS_RECORDS:
                f.write(json.dumps(record))
                f.write("\n")
            cls.external_corpus_path = f.name
        cls.addClassCleanup(_safe_remove, cls.external_corpus_path)

    def _run_external_sam_smoke(self, attention_backend):
        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=self.get_server_args(attention_backend),
        )
        try:
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 8,
                    },
                },
                timeout=120,
            )
            self.assertEqual(response.status_code, 200, response.text)
            response_json = response.json()
            self.assertIn("text", response_json)
            self.assertIn("meta_info", response_json)
            self.assertGreater(response_json["meta_info"]["completion_tokens"], 0)
        finally:
            _safe_kill_process(process)

    def test_generate_with_external_sam(self):
        for attention_backend in self.attention_backends:
            with self.subTest(attention_backend=attention_backend):
                self._run_external_sam_smoke(attention_backend)


if __name__ == "__main__":
    unittest.main()
