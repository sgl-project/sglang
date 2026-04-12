import asyncio
import json
import os
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace

import aiohttp
import openai
import requests
from transformers import AutoTokenizer

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE3,
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TARGET_MODEL_EAGLE3,
)

register_cuda_ci(est_time=394, suite="stage-b-test-2-gpu-large")


class TestDisaggregationAccuracy(PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.launch_all()

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=f"http://{self.base_host}:{self.lb_port}",
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"Evaluation metrics: {metrics}")

        self.assertGreater(metrics["score"], 0.62)

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

    def test_pause_resume_in_place(self):
        """Send requests, pause mid-generation, verify no progress during pause, resume."""
        NUM_REQUESTS = 32
        MAX_NEW_TOKENS = 512
        REQUEST_TIMEOUT = 180
        PAUSE_DURATION = 5

        def _generate(prompt_id):
            return requests.post(
                self.lb_url + "/generate",
                json={
                    "text": f"Question {prompt_id}: Write a short essay about the number {prompt_id}.",
                    "sampling_params": {
                        "temperature": 0.8,
                        "max_new_tokens": MAX_NEW_TOKENS,
                    },
                },
                timeout=REQUEST_TIMEOUT,
            )

        with ThreadPoolExecutor(max_workers=NUM_REQUESTS) as executor:
            futures = {executor.submit(_generate, i): i for i in range(NUM_REQUESTS)}

            time.sleep(1)

            requests.post(
                self.prefill_url + "/pause_generation",
                json={"mode": "in_place"},
                timeout=30,
            ).raise_for_status()
            requests.post(
                self.decode_url + "/pause_generation",
                json={"mode": "in_place"},
                timeout=30,
            ).raise_for_status()

            time.sleep(0.5)
            done_before = sum(1 for f in futures if f.done())

            time.sleep(PAUSE_DURATION)
            done_after = sum(1 for f in futures if f.done())

            self.assertLess(
                done_before,
                NUM_REQUESTS,
                "All requests completed before pause took effect — "
                "increase MAX_NEW_TOKENS to make the test meaningful.",
            )

            self.assertEqual(
                done_after - done_before,
                0,
                f"{done_after - done_before} requests completed during pause "
                f"({done_before} before, {done_after} after) — "
                f"pause_generation was not respected by the disagg scheduler.",
            )

            requests.post(
                self.decode_url + "/continue_generation",
                json={},
                timeout=30,
            ).raise_for_status()
            requests.post(
                self.prefill_url + "/continue_generation",
                json={},
                timeout=30,
            ).raise_for_status()

            completed = 0
            errors = []
            for future in as_completed(futures, timeout=REQUEST_TIMEOUT):
                prompt_id = futures[future]
                try:
                    resp = future.result()
                    if resp.status_code == 200:
                        body = resp.json()
                        self.assertIn("text", body)
                        self.assertGreater(len(body["text"]), 0)
                        completed += 1
                    else:
                        errors.append(f"Request {prompt_id}: status={resp.status_code}")
                except Exception as e:
                    errors.append(f"Request {prompt_id}: exception={e}")

        self.assertEqual(
            completed + len(errors),
            NUM_REQUESTS,
            "Some requests did not resolve within the timeout — likely hung during pause.",
        )
        self.assertEqual(
            completed,
            NUM_REQUESTS,
            f"Some requests failed: {completed}/{NUM_REQUESTS} succeeded. Errors: {errors}",
        )

    def test_first_token_finish(self):
        client = openai.Client(api_key="empty", base_url=f"{self.lb_url}/v1")
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        eos_token = tokenizer.eos_token_id
        prompt = "The best programming language for AI is"

        # First token EOS
        res = client.completions.create(
            model="dummy", prompt=prompt, logit_bias={eos_token: 42}
        ).model_dump()
        print(f"{res=}")

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
        print(f"{res=}")

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
        print(f"{res=}")

        assert res["usage"]["completion_tokens"] == 1, (
            "Expected completion_tokens to be 1 when first token is stop token, "
            f"but got {res['usage']['completion_tokens']}"
        )


class TestDisaggregationMooncakeFailure(PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # set DISAGGREGATION_TEST_FAILURE_PROB to simulate failure
        os.environ["DISAGGREGATION_TEST_FAILURE_PROB"] = "0.05"
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.launch_all()

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("DISAGGREGATION_TEST_FAILURE_PROB")
        super().tearDownClass()

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=f"http://{self.base_host}:{self.lb_port}",
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )

        # Expect lots of failure but the server cannot crash
        try:
            metrics = run_eval(args)
            print(f"Evaluation metrics: {metrics}")
        except Exception as e:
            print(f"Test encountered expected errors: {e}")
            # Check if servers are still healthy
            try:
                response = requests.get(self.prefill_url + "/health_generate")
                assert response.status_code == 200
                response = requests.get(self.decode_url + "/health_generate")
                assert response.status_code == 200
            except Exception as health_check_error:
                # If health check fails, re-raise the original exception
                raise e from health_check_error


class TestDisaggregationMooncakeSpec(PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_TARGET_MODEL_EAGLE3
        spec_args = [
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-draft-model-path",
            DEFAULT_DRAFT_MODEL_EAGLE3,
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "4",
            "--speculative-num-draft-tokens",
            "16",
            "--cuda-graph-max-bs",
            "8",
            "--dtype=float16",
        ]
        cls.extra_prefill_args = spec_args
        cls.extra_decode_args = spec_args
        cls.launch_all()

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=f"http://{self.base_host}:{self.lb_port}",
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"Evaluation metrics: {metrics}")

        self.assertGreater(metrics["score"], 0.74)


class TestDisaggregationSimulatedRetract(PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        os.environ["SGLANG_TEST_RETRACT"] = "true"
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.launch_all()

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("SGLANG_TEST_RETRACT")
        super().tearDownClass()

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=f"http://{self.base_host}:{self.lb_port}",
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"Evaluation metrics: {metrics}")

        self.assertGreater(metrics["score"], 0.62)


class TestDisaggregationPauseResumePrefillLeak(PDDisaggregationServerBase):
    """Regression test: pause_generation must not leak prefill requests into
    running_batch.  With a small --max-running-requests the leak fills the
    scheduling budget and blocks all subsequent prefills."""

    MAX_RUNNING = 4

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.extra_prefill_args = [
            "--max-running-requests",
            str(cls.MAX_RUNNING),
            "--enable-metrics",
        ]
        cls.launch_all()

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


if __name__ == "__main__":
    unittest.main()
