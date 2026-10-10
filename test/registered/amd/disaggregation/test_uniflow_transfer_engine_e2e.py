"""Server-level UniFlow PD test: one prefill and one decode server on one host.

Prefill runs on GPU 0 and decode on GPU 1 behind the mini load balancer, with
the UniFlow transfer backend carrying KV between them. A non-disaggregated
reference server with the same model and arguments runs on the next GPU.
Greedy output tokens and their logprobs through the PD path must match the
reference exactly. This compares outputs, not KV bytes: with full attention and
rotary positions, pages swapped within one request leave attention
mathematically unchanged.

A second class repeats the checks with the decode radix cache on, so prefill
sends KV starting after the prefix that decode already holds.
"""

import importlib
import os
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests
import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
    assert_process_healthy,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    is_in_amd_ci,
    popen_launch_pd_server,
    terminate_and_kill_process_tree,
    try_cached_model,
)

register_amd_ci(
    est_time=600,
    suite="stage-b-test-large-8-gpu-mi35x-disaggregation-amd",
    disabled="uniflow._core is not in the CI image",
)

MODEL_ENV_VAR = "SGLANG_UNIFLOW_E2E_TEST_MODEL"
# The model this test was validated with; override with MODEL_ENV_VAR.
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
MAX_NEW_TOKENS = 32
# Deterministic inference with the triton attention backend truncates each
# prefill chunk to a multiple of SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE
# (4096 by default), so at the default a smaller chunk size never schedules a
# prompt longer than one chunk.
CHUNKED_PREFILL_SIZE = 4096
# Arguments shared by prefill, decode, and the reference server, so that the
# reference differs from the PD path only in where the KV comes from. Without
# deterministic inference, logprobs and near-tied greedy tokens can differ
# between the two paths even when the KV is correct.
SERVER_ARGS = [
    "--attention-backend",
    "triton",
    "--enable-deterministic-inference",
    "--mem-fraction-static",
    "0.3",
    "--chunked-prefill-size",
    str(CHUNKED_PREFILL_SIZE),
]


def _prompts() -> list[str]:
    """Short, medium, multi-chunk, and shared-prefix prompts."""
    shared = "You are a careful assistant. " + " ".join(
        f"Fact {i} is {i * 31 % 97}." for i in range(250)
    )
    return [
        "The capital of France is",
        "Write a short story about a lighthouse keeper who finds a message. " * 8,
        " ".join(f"Register r{i % 32} holds {i * 7919 % 10007}." for i in range(300)),
        "Summarize this list. " + " ".join(f"word{i}" for i in range(1800)),
        *(f"{shared} Question {q}: what is fact {q * 3}?" for q in range(4)),
    ]


def _greedy_output(result: dict) -> tuple[list, list]:
    """Output token ids and their logprobs, which must match exactly."""
    return result["output_ids"], result["meta_info"]["output_token_logprobs"]


class TestUniflowTransferEngineE2E(PDDisaggregationServerBase):
    decode_base_gpu_id = 1
    extra_prefill_args = SERVER_ARGS
    extra_decode_args = SERVER_ARGS
    process_reference = None

    @classmethod
    def setUpClass(cls):
        reference_gpu = cls.decode_base_gpu_id + cls.decode_tp_size
        reason = cls._missing_requirement(reference_gpu + 1)
        if reason is not None:
            # In the intended GPU environment a missing requirement is a failure.
            if is_in_amd_ci():
                raise RuntimeError(reason)
            raise unittest.SkipTest(reason)

        super().setUpClass()
        # The shared fixture defaults to Mooncake in CI. It still passes RDMA
        # devices in CI, which UniFlow does not read.
        cls.transfer_backend = ["--disaggregation-transfer-backend", "uniflow"]
        cls.rdma_devices = []
        cls.model = try_cached_model(os.environ.get(MODEL_ENV_VAR, DEFAULT_MODEL))

        reference_port = int(cls.lb_port) + 600
        cls.reference_url = f"http://{cls.base_host}:{reference_port}"
        cls.process_reference = popen_launch_pd_server(
            cls.model,
            cls.reference_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--base-gpu-id",
                str(reference_gpu),
                "--nccl-port",
                str(reference_port + 100),
                *SERVER_ARGS,
            ],
        )
        cls.launch_all()
        cls.wait_server_ready(
            cls.reference_url + "/health", process=cls.process_reference
        )

    @classmethod
    def tearDownClass(cls):
        try:
            if cls.process_reference is not None:
                terminate_and_kill_process_tree(cls.process_reference, wait_timeout=60)
        finally:
            cls.process_reference = None
            # The base teardown assumes its own setUpClass ran.
            if hasattr(cls, "lb_url"):
                super().tearDownClass()

    @staticmethod
    def _missing_requirement(num_gpus: int) -> str | None:
        try:
            importlib.import_module("uniflow._core")
        except ModuleNotFoundError as e:
            # Only a missing binding is a requirement; a load failure is an error.
            if e.name not in ("uniflow", "uniflow._core"):
                raise
            return f"uniflow._core is not installed: {e}"
        if torch.cuda.device_count() < num_gpus:
            return f"needs {num_gpus} GPUs, found {torch.cuda.device_count()}"
        return None

    def _generate(self, url: str, text: str) -> dict:
        response = requests.post(
            url + "/generate",
            json={
                "text": text,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "ignore_eos": True,
                },
                "return_logprob": True,
            },
            timeout=300,
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(len(result["output_ids"]), MAX_NEW_TOKENS)
        return result

    def _assert_all_healthy(self):
        assert_process_healthy(self, "prefill", self.process_prefill, self.prefill_url)
        assert_process_healthy(self, "decode", self.process_decode, self.decode_url)
        assert_process_healthy(self, "load balancer", self.process_lb, self.lb_url)

    def test_uniflow_backend_is_active(self):
        for url in (self.prefill_url, self.decode_url):
            info = requests.get(url + "/server_info", timeout=10).json()
            self.assertEqual(info["disaggregation_transfer_backend"], "uniflow")

    def test_greedy_parity_sequential(self):
        for i, prompt in enumerate(_prompts()):
            with self.subTest(prompt=i):
                self.assertEqual(
                    _greedy_output(self._generate(self.lb_url, prompt)),
                    _greedy_output(self._generate(self.reference_url, prompt)),
                )
        self._assert_all_healthy()

    def test_greedy_parity_concurrent(self):
        prompts = _prompts() * 2
        expected = [self._generate(self.reference_url, p) for p in prompts]
        with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            actual = list(pool.map(lambda p: self._generate(self.lb_url, p), prompts))
        for i, (got, want) in enumerate(zip(actual, expected)):
            with self.subTest(request=i):
                self.assertEqual(_greedy_output(got), _greedy_output(want))
        self._assert_all_healthy()


class TestUniflowTransferEngineE2EDecodeRadixCache(TestUniflowTransferEngineE2E):
    # Without a prefill radix cache, any cached tokens reported for a request
    # come from prefix reuse on decode.
    extra_prefill_args = [*SERVER_ARGS, "--disable-radix-cache"]
    extra_decode_args = [*SERVER_ARGS, "--disaggregation-decode-enable-radix-cache"]

    def test_decode_reuses_prefix(self):
        prompt = _prompts()[-1]
        expected = _greedy_output(self._generate(self.reference_url, prompt))
        self.assertEqual(_greedy_output(self._generate(self.lb_url, prompt)), expected)
        repeated = self._generate(self.lb_url, prompt)
        self.assertGreater(repeated["meta_info"]["cached_tokens"], 0)
        self.assertEqual(_greedy_output(repeated), expected)
        self._assert_all_healthy()


if __name__ == "__main__":
    unittest.main()
