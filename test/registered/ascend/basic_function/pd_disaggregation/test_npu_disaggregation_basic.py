import asyncio
import json
import logging
import os
import threading
import time
import unittest
import uuid
from types import SimpleNamespace
from typing import Any

import aiohttp
import openai
import requests
from transformers import AutoTokenizer

from sglang.test.ascend.test_ascend_utils import (
    EAGLE3_LLAMA3_1_INSTRUCT_8B_WEIGHTS_PATH,
    LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH,
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.json_constrained_kit import JSONConstrainedMixin
from sglang.test.kits.pause_generation_kit import PauseResumeInPlaceMixin
from sglang.test.kits.spec_server_kits import SpecGrammarKit
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)

register_npu_ci(est_time=400, suite="full-16-npu-a3", nightly=True)


class DisaggregationTestBase(PDDisaggregationServerBase):
    """
    Base class for PD-disaggregation tests on Ascend NPU.
    Subclasses only need to define class variables to customize:
      -- model: model path
      -- prefill_extra_args / decode_extra_args: extra args
      -- extra_env_vars: dict of env vars to set during test
      -- gsm8k_score_threshold: if set, call run_gsm8k to test accuracy.
    """

    model = None
    prefill_extra_args = []
    decode_extra_args = []
    extra_env_vars = {}
    gsm8k_score_threshold = 1.0

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.transfer_backend = ["--disaggregation-transfer-backend", "ascend"]
        os.environ["ASCEND_MF_STORE_URL"] = "tcp://127.0.0.1:24667"
        for key, value in cls.extra_env_vars.items():
            os.environ[key] = value

        # Non blocking start servers
        cls.start_prefill()
        cls.start_decode()

        # Block until both
        cls.wait_server_ready(cls.prefill_url + "/health", process=cls.process_prefill)
        cls.wait_server_ready(cls.decode_url + "/health", process=cls.process_decode)

        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp-size",
            1,
        ] + cls.prefill_extra_args
        prefill_args += cls.transfer_backend + cls.rdma_devices
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp-size",
            1,
            "--base-gpu-id",
            1,
        ] + cls.decode_extra_args
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("ASCEND_MF_STORE_URL", None)
        for key, value in cls.extra_env_vars.items():
            os.environ.pop(key, None)
        super().tearDownClass()

    def run_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)

        self.assertGreater(metrics["score"], self.gsm8k_score_threshold)


class TestDisaggregationAccuracy(DisaggregationTestBase, PauseResumeInPlaceMixin):
    """Test the basic functions of PD separation."""

    model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
    gsm8k_score_threshold = 0.62

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.pause_generate_url = cls.lb_url
        cls.pause_target_urls = [cls.prefill_url, cls.decode_url]

    def test_gsm8k(self):
        self.run_gsm8k()

    def test_logprob(self):
        prompt = "The capital of france is "
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {"temperature": 0},
                "return_logprob": True,
                "return_input_logprob": True,
                "logprob_start_len": 0,
            },
        )

        j = response.json()
        completion_tokens = j["meta_info"]["completion_tokens"]
        input_logprobs = j["meta_info"]["input_token_logprobs"]
        output_logprobs = j["meta_info"]["output_token_logprobs"]

        assert (
            len(output_logprobs) == completion_tokens
        ), f"output_logprobs and completion_tokens should have the same length, but got {len(output_logprobs)} and {completion_tokens}"
        assert (
            len(input_logprobs) > 0
        ), f"input_logprobs should have at least one token, but got {len(input_logprobs)}"

    def test_chat_completion_top_logprobs(self):
        client = openai.Client(api_key="empty", base_url=f"{self.lb_url}/v1")
        response = client.chat.completions.create(
            model="dummy",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            temperature=0,
            max_tokens=8,
            logprobs=True,
            top_logprobs=5,
        )

        self.assertIsNotNone(response.choices[0].logprobs)
        content_logprobs = response.choices[0].logprobs.content
        self.assertGreater(len(content_logprobs), 0)

        first_top_logprobs = next(
            (item.top_logprobs for item in content_logprobs if item.top_logprobs),
            None,
        )
        self.assertIsNotNone(first_top_logprobs)
        self.assertGreater(len(first_top_logprobs), 0)
        self.assertIsInstance(first_top_logprobs[0].token, str)

    def test_structured_output(self):
        json_schema = json.dumps(
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "pattern": "^[\\w]+$"},
                    "population": {"type": "integer"},
                },
                "required": ["name", "population"],
            }
        )

        # JSON
        response = requests.post(
            f"{self.lb_url}/generate",
            json={
                "text": "Here is the information of the capital of France in the JSON format.\n",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 64,
                    "json_schema": json_schema,
                },
            },
        )
        output = response.json()["text"]
        # ensure the output is a valid JSON
        json.loads(output)

    def test_first_token_finish(self):
        client = openai.Client(api_key="empty", base_url=f"{self.lb_url}/v1")
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        eos_token = tokenizer.eos_token_id
        prompt = "The best programming language for AI is"

        # First token EOS
        res = client.completions.create(
            model="dummy", prompt=prompt, logit_bias={eos_token: 42}
        ).model_dump()

        assert res["usage"]["completion_tokens"] == 1, (
            "Expected completion_tokens to be 1 when first token is EOS, "
            f"but got {res['usage']['completion_tokens']}"
        )

        # First token EOS with ignore_eos
        res = client.completions.create(
            model="dummy",
            prompt=prompt,
            logit_bias={eos_token: 42},
            extra_body={"ignore_eos": True},
        ).model_dump()

        assert res["usage"]["completion_tokens"] > 1, (
            "Expected completion_tokens to be greater than 1 when ignore_eos is True, "
            f"but got {res['usage']['completion_tokens']}"
        )

        # First token with specified stop token
        stop_token_id = tokenizer.encode(" hello", add_special_tokens=False)[0]
        res = client.completions.create(
            model="dummy",
            prompt=prompt,
            logit_bias={stop_token_id: 42},
            stop=[" hello"],
        ).model_dump()

        assert res["usage"]["completion_tokens"] == 1, (
            "Expected completion_tokens to be 1 when first token is stop token, "
            f"but got {res['usage']['completion_tokens']}"
        )


class TestDisaggregationMooncakeFailure(DisaggregationTestBase):
    """Test fault tolerance against random transport layer failures in a PD separation scenario."""

    model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
    extra_env_vars = {"DISAGGREGATION_TEST_FAILURE_PROB": "0.05"}

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )

        # Expect lots of failure but the server cannot crash
        try:
            metrics = run_eval(args)
            logging.warning(f"Evaluation metrics: {metrics}")
        except Exception as e:
            logging.warning(f"Test encountered expected errors: {e}")
            # Check if servers are still healthy
            try:
                response = requests.get(self.prefill_url + "/health_generate")
                assert response.status_code == 200
                response = requests.get(self.decode_url + "/health_generate")
                assert response.status_code == 200
            except Exception as health_check_error:
                # If health check fails, re-raise the original exception
                raise e from health_check_error


class TestDisaggregationMooncakeSpec(
    JSONConstrainedMixin, SpecGrammarKit, DisaggregationTestBase
):
    """Testing confirmed that EAGLE introduces no loss in accuracy in the PD separation scenario."""

    model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
    gsm8k_score_threshold = 0.74
    extra_env_vars = {"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1"}
    spec_args = [
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-draft-model-path",
        EAGLE3_LLAMA3_1_INSTRUCT_8B_WEIGHTS_PATH,
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "16",
        "--cuda-graph-max-bs",
        "8",
        "--dtype=float16",
    ]
    prefill_extra_args = spec_args
    decode_extra_args = spec_args

    def test_gsm8k(self):
        self.run_gsm8k()


class TestDisaggregationSimulatedRetract(DisaggregationTestBase):
    """Test accuracy results in "retract" mode for the PD separation scenario."""

    model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
    gsm8k_score_threshold = 0.62
    extra_env_vars = {"SGLANG_TEST_RETRACT": "true"}

    def test_gsm8k(self):
        self.run_gsm8k()


class TestDisaggregationPauseResumeDecodeRetract(DisaggregationTestBase):
    """Test the pause/resume functionality of the Decode node in Retract mode within a PD separation scenario."""

    model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH

    def test_retract_pause_decode_running_batch(self):
        """Retract-mode pause on a disagg decode node must preserve in-flight
        requests that are already in running_batch."""
        asyncio.run(self._run_pause_on_decode_running_batch("retract"))

    def test_retract_weight_update_decode_running_batch(self):
        """Retract pause + weight update on a disagg decode node.

        This guards the core reason retract mode exists: while paused, the
        running_batch AND the rebootstrap preallocation queue are empty, so the
        scheduler is fully idle and the post-update cache flush succeeds (a
        regression here trips ``assert ..., "Cache flush failed after updating
        weights"`` and crashes the decode worker). On continue, the retracted
        requests rebootstrap-recompute their prefix KV under the updated weights
        and resume to completion.
        """
        asyncio.run(
            self._run_pause_on_decode_running_batch("retract", weight_update=True)
        )

    async def _get_decode_num_running_reqs(self, session):
        """Query current decode running_batch size from /v1/loads."""
        async with session.get(
            self.decode_url + "/v1/loads?include=core",
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            resp.raise_for_status()
            body = await resp.json()
            return sum(load["num_running_reqs"] for load in body["loads"])

    async def _wait_for_decode_running_batch(self, session, timeout):
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            if await self._get_decode_num_running_reqs(session) > 0:
                return
            await asyncio.sleep(0.2)

        self.fail("Timed out waiting for decode running_batch to become non-empty")

    async def _run_pause_on_decode_running_batch(self, mode, weight_update=False):
        num_requests = 2
        max_new_tokens = 512
        prompt = "Write a detailed numbered explanation of distributed inference. " * 12

        async def _post(session, url, json_data, timeout=30):
            async with session.post(
                url,
                json=json_data,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                resp.raise_for_status()
                return await resp.json()

        async def _generate(session, request_id):
            return await _post(
                session,
                self.lb_url + "/generate",
                {
                    "text": f"Request {request_id}: {prompt}",
                    "background": True,
                    "sampling_params": {
                        "temperature": 0,
                        "ignore_eos": True,
                        "max_new_tokens": max_new_tokens,
                    },
                },
                timeout=180,
            )

        async with aiohttp.ClientSession() as session:
            tasks = [
                asyncio.create_task(_generate(session, i)) for i in range(num_requests)
            ]
            decode_paused = False

            try:
                await self._wait_for_decode_running_batch(session, timeout=30)
                await asyncio.sleep(0.1)

                self.assertTrue(
                    any(not task.done() for task in tasks),
                    "All requests finished before decode retract pause was issued.",
                )

                await _post(
                    session,
                    self.decode_url + "/pause_generation",
                    {"mode": mode},
                )
                decode_paused = True
                await asyncio.sleep(1)

                if weight_update:
                    # Reload the same weights from disk while retract-paused. The
                    # update mechanism (disk/tensor/distributed/ipc) is irrelevant
                    # here: they all share flush_cache_after_weight_update, whose
                    # flush asserts the scheduler is fully idle. This must not
                    # crash, proving retracted reqs are not stuck in the prealloc
                    # queue.
                    wu = await _post(
                        session,
                        self.decode_url + "/update_weights_from_disk",
                        {"model_path": self.model},
                        timeout=180,
                    )
                    self.assertTrue(
                        wu.get("success", False),
                        f"update_weights_from_disk failed during retract pause: {wu}",
                    )

                await _post(session, self.decode_url + "/continue_generation", {})
                decode_paused = False

                responses = await asyncio.wait_for(asyncio.gather(*tasks), timeout=180)
            finally:
                if decode_paused:
                    try:
                        await _post(
                            session, self.decode_url + "/continue_generation", {}
                        )
                    except Exception:
                        pass

                unfinished = [task for task in tasks if not task.done()]
                if unfinished:
                    for url in [self.prefill_url, self.decode_url]:
                        try:
                            await _post(
                                session,
                                url + "/abort_request",
                                {"abort_all": True},
                            )
                        except Exception:
                            pass
                    for task in unfinished:
                        task.cancel()
                    await asyncio.gather(*unfinished, return_exceptions=True)

            for response in responses:
                self.assertIn("text", response)
                self.assertGreater(len(response["text"]), 0)

            self.assertGreater(
                sum(
                    response.get("meta_info", {}).get("num_retractions", 0)
                    for response in responses
                ),
                0,
                "Expected pause_generation(retract) to retract a running decode request.",
            )


class TestDisaggregationPauseResumePrefillLeak(DisaggregationTestBase):
    """Regression test: pause_generation must not leak prefill requests into
    running_batch.  With a small --max-running-requests the leak fills the
    scheduling budget and blocks all subsequent prefills."""

    model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
    MAX_RUNNING = 4
    spec_agrs = [
        "--max-running-requests",
        str(MAX_RUNNING),
        "--enable-metrics",
    ]
    prefill_extra_args = spec_agrs
    decode_extra_args = spec_agrs

    def test_retract_pause_no_leak_on_prefill(self):
        """Retract-mode pause on a disagg prefill node must not leak prefill
        requests into running_batch. Without the fix, each retract pause merges
        last_batch into running_batch, but the prefill event loop never cleans
        them up via update_running_batch. After enough cycles the
        max-running-requests budget is exhausted and all new prefills hang."""
        asyncio.run(self._run_pause_resume_leak_test("retract"))

    def test_retract_pause_empty_running_batch(self):
        """Retract-mode pause must not crash when running_batch is empty.
        Regression test for issue #20272."""
        asyncio.run(self._run_pause_on_idle("retract"))

    async def _run_pause_on_idle(self, mode):
        """Pause/resume on an idle prefill node (no in-flight requests)."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.prefill_url + "/pause_generation",
                json={"mode": mode},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                resp.raise_for_status()
            async with session.post(
                self.prefill_url + "/continue_generation",
                json={},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                resp.raise_for_status()

            # Verify the engine still works after pause/resume
            async with session.post(
                self.lb_url + "/generate",
                json={
                    "text": "What is 1+1?",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                },
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                resp.raise_for_status()
                body = await resp.json()
                self.assertIn("text", body)
                self.assertGreater(len(body["text"]), 0)

    async def _get_num_running_reqs(self, session):
        """Query sglang:num_running_reqs from prefill node's /metrics."""
        async with session.get(
            self.prefill_url + "/metrics",
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            resp.raise_for_status()
            text = await resp.text()
            for line in text.splitlines():
                # Match the gauge line, skip HELP/TYPE comments and
                # per-priority breakdowns (which have priority="<int>")
                if (
                    line.startswith("sglang:num_running_reqs{")
                    and "priority=" not in line
                ):
                    return int(float(line.split()[-1]))
            return 0

    async def _run_pause_resume_leak_test(self, mode):
        NUM_WORKERS = 64
        NUM_PAUSE_RESUME_CYCLES = self.MAX_RUNNING * 4
        MAX_NEW_TOKENS = 1
        LONG_PROMPT = "Tell me a story. " * 200

        async def _background_worker(session, worker_id, cancel_event):
            """Send requests sequentially until cancelled."""
            seq = 0
            while not cancel_event.is_set():
                try:
                    async with session.post(
                        self.lb_url + "/generate",
                        json={
                            "text": f"[w{worker_id}-{seq}] {LONG_PROMPT}",
                            "sampling_params": {
                                "temperature": 0,
                                "max_new_tokens": MAX_NEW_TOKENS,
                            },
                        },
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:
                        await resp.read()
                except Exception:
                    pass
                seq += 1

        async def _post(session, url, json_data):
            async with session.post(
                url,
                json=json_data,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                resp.raise_for_status()

        cancel_event = asyncio.Event()

        async with aiohttp.ClientSession() as session:
            workers = [
                asyncio.create_task(_background_worker(session, i, cancel_event))
                for i in range(NUM_WORKERS)
            ]

            for _ in range(NUM_PAUSE_RESUME_CYCLES):
                await _post(
                    session,
                    self.prefill_url + "/pause_generation",
                    {"mode": mode},
                )
                await _post(
                    session,
                    self.prefill_url + "/continue_generation",
                    {},
                )
                await asyncio.sleep(0.1)

            # Stop workers and abort all in-flight requests
            cancel_event.set()
            await _post(
                session, self.prefill_url + "/abort_request", {"abort_all": True}
            )
            await _post(
                session, self.decode_url + "/abort_request", {"abort_all": True}
            )
            await asyncio.gather(*workers, return_exceptions=True)

            # Wait for abort cleanup, then check for leaked phantom requests.
            # With the bug, running_batch accumulates phantom prefill requests
            # that are never cleaned up.
            await asyncio.sleep(2)
            num_running = await self._get_num_running_reqs(session)
            self.assertEqual(
                num_running,
                0,
                f"Prefill node has {num_running} phantom running requests "
                f"after abort — pause_generation is leaking into running_batch",
            )


_CHUNKED_ABORT_LONG_PROMPT = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "Sphinx of black quartz, judge my vow. "
) * 900


def _decode_response(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text


def _is_abort_result(status_code: int, body: Any) -> bool:
    if status_code == 200:
        reason = (
            body.get("meta_info", {}).get("finish_reason", {})
            if isinstance(body, dict)
            else {}
        )
        return isinstance(reason, dict) and reason.get("type") == "abort"

    if status_code not in (500, 503):
        return False

    text = body if isinstance(body, str) else str(body)
    return "abort" in text.lower()


class TestDisaggChunkedPrefillAbort(DisaggregationTestBase):
    """Test the request ID cancellation function under the chunked prefill mode in a PD-separated scenario."""

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    spec_arges = [
        "--max-running-requests",
        "4",
        "--chunked-prefill-size",
        "128",
    ]

    def _post_abort(self, rid: str):
        for url in (self.prefill_url, self.decode_url):
            requests.post(
                url + "/abort_request",
                json={"rid": rid, "abort_all": False},
                timeout=10,
            )

    def test_abort_mid_chunked_prefill_by_rid(self):
        rid = f"pd-chunked-prefill-abort-{uuid.uuid4().hex}"
        result: dict[str, Any] = {}

        def run_generate():
            try:
                response = requests.post(
                    self.lb_url + "/generate",
                    json={
                        "rid": rid,
                        "text": f"{rid}\n{_CHUNKED_ABORT_LONG_PROMPT}",
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 4096,
                            "ignore_eos": True,
                        },
                    },
                    timeout=180,
                )
                result["status_code"] = response.status_code
                result["body"] = _decode_response(response)
            except requests.RequestException as exc:
                result["exception"] = repr(exc)

        thread = threading.Thread(target=run_generate)
        thread.start()

        time.sleep(1.0)
        abort_deadline = time.monotonic() + 8
        while thread.is_alive() and time.monotonic() < abort_deadline:
            self._post_abort(rid)
            time.sleep(0.2)

        thread.join(timeout=60)
        self.assertFalse(thread.is_alive(), "Chunked-prefill abort request hung")
        self.assertNotIn("exception", result, result.get("exception"))
        self.assertTrue(
            _is_abort_result(result["status_code"], result["body"]),
            f"Expected chunked-prefill request to abort, got {result}",
        )

        for url in (self.lb_url, self.prefill_url, self.decode_url):
            health = requests.get(url + "/health", timeout=10)
            self.assertEqual(health.status_code, 200, health.text)


if __name__ == "__main__":
    unittest.main()
