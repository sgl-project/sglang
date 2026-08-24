import asyncio
import json
import os
import threading
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import aiohttp
import openai
import requests
from transformers import AutoTokenizer

from sglang.srt.environ import envs
from sglang.srt.mem_cache.kv_cache_builder import BACKUP_ONLY_HICACHE_RATIO
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.json_constrained_kit import JSONConstrainedMixin
from sglang.test.kits.pause_generation_kit import PauseResumeInPlaceMixin
from sglang.test.kits.spec_server_kits import SpecGrammarKit
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
    assert_process_healthy,
)
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE3,
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TARGET_MODEL_EAGLE3,
)

register_cuda_ci(est_time=730, stage="base-b", runner_config="2-gpu-large")


class TestDisaggregationAccuracy(PauseResumeInPlaceMixin, PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.pause_generate_url = cls.lb_url
        cls.pause_target_urls = [cls.prefill_url, cls.decode_url]
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
        self.assertEqual(len(first_top_logprobs), 5)
        self.assertIsInstance(first_top_logprobs[0].token, str)
        self.assertIsInstance(first_top_logprobs[0].logprob, float)

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
        # Inject transfer failures so the retry path is actually exercised.
        # Entered before launch_all() so the server subprocesses inherit it.
        cls._disagg_failure_ctx = envs.SGLANG_TEST_DISAGG_FAILURE_PROB.override(0.05)
        cls._disagg_failure_ctx.__enter__()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.launch_all()

    @classmethod
    def tearDownClass(cls):
        cls._disagg_failure_ctx.__exit__(None, None, None)
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


class TestDisaggregationMooncakeSpec(
    JSONConstrainedMixin, SpecGrammarKit, PDDisaggregationServerBase
):
    min_retraction_accept_length = 1.3

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_TARGET_MODEL_EAGLE3
        spec_args = [
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model-path",
            DEFAULT_DRAFT_MODEL_EAGLE3,
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "4",
            "--speculative-num-draft-tokens",
            "16",
            "--cuda-graph-max-bs-decode",
            "8",
            "--dtype=float16",
        ]
        cls.extra_prefill_args = spec_args
        cls.extra_decode_args = [
            *spec_args,
            "--disaggregation-decode-retraction-backup",
            "host_pool",
        ]
        cls.extra_decode_env = {"SGLANG_TEST_RETRACT": "true"}
        cls.launch_all()

    def test_host_pool_retraction_preserves_spec_acceptance(self):
        prompts = [
            f"Request {i}: explain how speculative decoding works. " * 4
            for i in range(4)
        ]
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": prompts,
                "sampling_params": {
                    "temperature": 0,
                    "ignore_eos": True,
                    "max_new_tokens": 64,
                },
            },
        )
        response.raise_for_status()
        results = response.json()
        retracted_results = [
            result for result in results if result["meta_info"]["num_retractions"] > 0
        ]
        retraction_count = sum(
            result["meta_info"]["num_retractions"] for result in retracted_results
        )
        self.assertGreater(retraction_count, 0)

        completion_tokens = sum(
            result["meta_info"]["completion_tokens"] for result in retracted_results
        )
        verify_count = sum(
            result["meta_info"]["spec_verify_ct"] for result in retracted_results
        )
        self.assertGreater(verify_count, 0)
        accept_length = completion_tokens / verify_count
        print(f"Retraction speculative {accept_length=:.4f}")
        self.assertGreater(accept_length, self.min_retraction_accept_length)

    def test_oversized_backup_aborts_only_its_own_request(self):
        # Backup-only host_pool retraction sizes the host pool at a fraction of the
        # device pool, so a long enough request cannot be backed up. Derive the
        # length from the running server rather than pinning pool sizes, which
        # would change what the other cases in this class exercise.
        info = requests.get(self.decode_url + "/get_server_info", timeout=30).json()
        device_tokens = info["max_total_num_tokens"]
        host_slots = int(device_tokens * BACKUP_ONLY_HICACHE_RATIO)
        # Over the host pool, but still inside both the device pool and the model
        # context — a pool far larger than the context would reject the request
        # before it ever reaches retraction.
        oversized_len = min(int(device_tokens * 0.4), info["max_req_input_len"] - 1024)
        self.assertGreater(
            oversized_len,
            host_slots,
            f"no prompt length both overflows the {host_slots}-slot host pool and "
            f"fits the {info['max_req_input_len']}-token context",
        )

        def oversized_request(seed):
            # Sent on its own: a batched /generate fails as a whole once any member
            # aborts, which would hide the concurrent traffic's own outcome.
            # Generate long enough to still be decoding when a forced retraction
            # lands — a short request finishes first and is never retracted.
            return requests.post(
                self.lb_url + "/generate",
                json={
                    "input_ids": [seed] * oversized_len,
                    "sampling_params": {"max_new_tokens": 512, "ignore_eos": True},
                },
                timeout=900,
            )

        def ordinary_request(seed):
            # Must still be decoding when the oversized prefill lands: retraction
            # keeps one request, so a batch that has drained to a single entry is
            # skipped entirely and nothing is ever picked.
            return requests.post(
                self.lb_url + "/generate",
                json={
                    "input_ids": [seed] * 512,
                    "sampling_params": {"max_new_tokens": 4096, "ignore_eos": True},
                },
                timeout=900,
            )

        # Retraction picks the request with the fewest generated tokens; the prompt
        # length only breaks ties. The oversized request has by far the longest
        # prefill, so in a fixed batch it enters decode last, holds the fewest
        # tokens, and is picked first — but only while nothing newer arrives, which
        # is why the eval below runs after these rather than alongside them.
        with ThreadPoolExecutor(max_workers=4) as pool:
            oversized = pool.submit(oversized_request, 233)
            ordinary = [pool.submit(ordinary_request, 300 + i) for i in range(3)]
            response = oversized.result()
            neighbours = [f.result() for f in ordinary]

        # A 200 here means the request was never retracted, not that the abort path
        # is broken, so surface the retraction count to tell the two apart.
        meta = (
            response.json().get("meta_info", {}) if response.status_code == 200 else {}
        )
        self.assertEqual(
            response.status_code,
            500,
            (
                f"expected an aborted backup; got num_retractions="
                f"{meta.get('num_retractions')} completion_tokens="
                f"{meta.get('completion_tokens')}"
                if meta
                else response.text
            ),
        )
        self.assertIn("Retraction host KV pool exhausted", response.text)
        for neighbour in neighbours:
            self.assertEqual(neighbour.status_code, 200, neighbour.text)

        # The abort must leave the scheduler serving, and ordinary traffic must stay
        # correct afterwards — a leaked host slot or a damaged neighbour shows up as
        # a wrong answer rather than merely a 200.
        assert_process_healthy(self, "decode", self.process_decode, self.decode_url)
        metrics = run_eval(
            SimpleNamespace(
                base_url=f"http://{self.base_host}:{self.lb_port}",
                eval_name="gsm8k",
                api="completion",
                max_tokens=512,
                num_examples=64,
                num_threads=32,
            )
        )
        print(f"Post-abort gsm8k metrics: {metrics}")
        # Looser than the 200-example test_gsm8k bar above: 64 examples is a
        # health check on the post-abort server, not an accuracy measurement.
        self.assertGreater(metrics["score"], 0.62)

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


class TestDisaggregationPauseResumeDecodeRetract(PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.launch_all()

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


PD_CHUNKED_ABORT_EXTRA_ARGS = [
    "--max-running-requests",
    "4",
    "--chunked-prefill-size",
    "64",
]
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


class TestDisaggChunkedPrefillAbort(PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.extra_prefill_args = PD_CHUNKED_ABORT_EXTRA_ARGS
        cls.extra_decode_args = PD_CHUNKED_ABORT_EXTRA_ARGS
        cls.launch_all()

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
