"""Standalone renderer coverage using the existing OpenAI test matrices.

Run this file directly, or use unittest discovery with this directory as the
start directory. Reference suites are imported as modules and are not collected.
"""

import math
import os
import shutil
import subprocess
import time
import unittest
from contextlib import ExitStack, contextmanager
from urllib.parse import urlsplit

import requests
from basic import test_openai_server as openai_cases
from function_call import test_openai_function_calling as tool_cases

from sglang.srt.utils import is_npu, kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.srt.utils.network import get_free_port
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_rust_server_built,
    popen_launch_server,
)

register_cuda_ci(est_time=300, stage="base-b", runner_config="1-gpu-large")


@contextmanager
def launch_rust_renderer(model, base_url, *, timeout, engine_args=(), renderer_args=()):
    binary = shutil.which(os.environ.get("SGLANG_RENDERER_BIN", "sglang-renderer"))
    if binary is None:
        raise FileNotFoundError(
            "build sglang-renderer and add it to PATH or set SGLANG_RENDERER_BIN"
        )
    public = urlsplit(base_url)
    engine_port = get_free_port()
    while engine_port == public.port:
        engine_port = get_free_port()
    engine_url = f"http://127.0.0.1:{engine_port}"
    with ExitStack() as stack:
        engine = popen_launch_server(
            model,
            engine_url,
            timeout=timeout,
            # Python's default warmup sends text to the token-ID-only engine.
            # popen_launch_server waits for its pre-tokenized health probe instead.
            other_args=["--skip-server-warmup", *engine_args],
            env={"SGLANG_RUST_SERVER": "1"},
        )
        stack.callback(kill_process_tree, engine.pid)
        renderer = subprocess.Popen(
            [
                binary,
                model,
                "--engine-url",
                engine_url,
                "--host",
                public.hostname,
                "--port",
                str(public.port),
                "--proxy-unhandled-routes",
                "--sampling-defaults",
                "openai",
                *renderer_args,
            ]
        )
        stack.callback(kill_process_tree, renderer.pid)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if engine.poll() is not None or renderer.poll() is not None:
                raise RuntimeError("renderer or native engine exited during startup")
            try:
                response = requests.get(base_url + "/_sglang_renderer/ready", timeout=1)
                if (
                    response.status_code == 204
                    and response.headers.get("x-sglang-renderer") == "ready"
                ):
                    break
            except requests.RequestException:
                pass
            time.sleep(0.2)
        else:
            raise TimeoutError("standalone renderer did not become ready")
        yield


@unittest.skipUnless(
    is_rust_server_built(),
    "embedded rust server extension not built",
)
class RustRendererFixture:
    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    api_key = "sk-123456"
    engine_args = ()
    renderer_args = ()

    @classmethod
    def setUpClass(cls):
        cls._renderer = ExitStack()
        cls.addClassCleanup(cls._renderer.close)
        cls._renderer.enter_context(
            launch_rust_renderer(
                cls.model,
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                engine_args=cls.engine_args,
                renderer_args=cls.renderer_args,
            )
        )
        cls.base_url = DEFAULT_URL_FOR_TEST + "/v1"
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        cls._renderer.close()


class TestOpenAICompletionWithRust(RustRendererFixture, CustomTestCase):
    run_completion = openai_cases.TestOpenAIServer.run_completion
    run_completion_stream = openai_cases.TestOpenAIServer.run_completion_stream
    test_completion = openai_cases.TestOpenAIServer.test_completion
    test_completion_stream = openai_cases.TestOpenAIServer.test_completion_stream


class TestOpenAIChatWithRust(TestOpenAICompletionWithRust):
    run_chat_completion = openai_cases.TestOpenAIServer.run_chat_completion
    run_chat_completion_stream = (
        openai_cases.TestOpenAIServer.run_chat_completion_stream
    )
    test_chat_completion = openai_cases.TestOpenAIServer.test_chat_completion
    test_chat_completion_stream = (
        openai_cases.TestOpenAIServer.test_chat_completion_stream
    )
    test_completion = None
    test_completion_stream = None


@unittest.skipIf(is_npu(), "the embedded Rust server is not an Ascend path")
class TestOpenAIFunctionCallingWithRust(
    RustRendererFixture, tool_cases.TestOpenAIServerFunctionCalling
):
    renderer_args = ("--tool-call-parser", "llama3")


@unittest.skipIf(is_npu(), "the embedded Rust server is not an Ascend path")
class TestOpenAIPythonicFunctionCallingWithRust(
    RustRendererFixture, tool_cases.TestOpenAIPythonicFunctionCalling
):
    renderer_args = ("--tool-call-parser", "pythonic")


@unittest.skipUnless(
    is_rust_server_built(),
    "embedded rust server extension not built",
)
class TestOpenAICompletionRustParity(CustomTestCase):
    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    api_key = "sk-123456"

    def _get_logprobs(self, *, rust_frontend):
        # compare identical prefill shapes, without graph padding or warmup cache hits
        engine_args = [
            "--random-seed",
            "42",
            "--disable-prefill-cuda-graph",
            "--disable-radix-cache",
        ]
        with ExitStack() as stack:
            if rust_frontend:
                stack.enter_context(
                    launch_rust_renderer(
                        self.model,
                        DEFAULT_URL_FOR_TEST,
                        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                        engine_args=engine_args,
                    )
                )
            else:
                process = popen_launch_server(
                    self.model,
                    DEFAULT_URL_FOR_TEST,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    api_key=self.api_key,
                    env={"SGLANG_RUST_SERVER": "0"},
                    other_args=engine_args,
                )
                stack.callback(kill_process_tree, process.pid)
            response = requests.post(
                DEFAULT_URL_FOR_TEST + "/v1/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "prompt": "The capital of France is",
                    "temperature": 0,
                    "max_tokens": 8,
                    "logprobs": 5,
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["logprobs"]

    @staticmethod
    def _kl_divergence(reference, candidate):
        assert reference.keys() == candidate.keys()
        keys = sorted(reference)
        reference_max = max(reference.values())
        candidate_max = max(candidate.values())
        reference_weights = [math.exp(reference[key] - reference_max) for key in keys]
        candidate_weights = [math.exp(candidate[key] - candidate_max) for key in keys]
        reference_sum = sum(reference_weights)
        candidate_sum = sum(candidate_weights)
        reference_probabilities = [
            weight / reference_sum for weight in reference_weights
        ]
        candidate_probabilities = [
            weight / candidate_sum for weight in candidate_weights
        ]
        return sum(
            reference_probability
            * math.log(reference_probability / candidate_probability)
            for reference_probability, candidate_probability in zip(
                reference_probabilities,
                candidate_probabilities,
                strict=True,
            )
        )

    def test_logprobs_have_zero_kl_against_python_frontend(self):
        python_logprobs = self._get_logprobs(rust_frontend=False)
        rust_logprobs = self._get_logprobs(rust_frontend=True)

        self.assertEqual(rust_logprobs["tokens"], python_logprobs["tokens"])
        self.assertEqual(
            rust_logprobs["token_logprobs"],
            python_logprobs["token_logprobs"],
        )
        self.assertEqual(
            len(rust_logprobs["top_logprobs"]),
            len(python_logprobs["top_logprobs"]),
        )
        for python_top, rust_top in zip(
            python_logprobs["top_logprobs"],
            rust_logprobs["top_logprobs"],
            strict=True,
        ):
            self.assertEqual(rust_top, python_top)
            self.assertEqual(self._kl_divergence(python_top, rust_top), 0.0)


if __name__ == "__main__":
    unittest.main()
