"""Exercise measured cache writes through the real cache, IPC and HTTP APIs.

A cold request must publish reusable input writes. Repeating its exact prompt
must report a cache read without counting a duplicate insertion as a new write.
Terminal SSE usage must retain that observation after the initial request stats
have crossed the scheduler/tokenizer boundary.
"""

import json
import unittest

import requests
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")

_MODEL = "Qwen/Qwen3-0.6B"
_PROMPT = (
    "A radix prefix cache retains reusable input tokens separately from "
    "generated output. A repeated request reads the retained prefix rather "
    "than publishing that prefix twice. "
) * 4


class TestCacheWriteUsage(CustomTestCase):
    process = None

    @classmethod
    def setUpClass(cls):
        # Dummy weights avoid downloading a checkpoint. The ordinary CUDA model,
        # allocator, scheduler, Python radix tree and typed IPC remain active.
        cls.process = popen_launch_server(
            _MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--model-type",
                "llm",
                "--load-format",
                "dummy",
                "--enable-cache-report",
                "--attention-backend",
                "triton",
                "--sampling-backend",
                "pytorch",
                "--cuda-graph-backend-decode",
                "disabled",
                "--cuda-graph-backend-prefill",
                "disabled",
                "--dtype",
                "bfloat16",
                "--context-length",
                "512",
                "--chunked-prefill-size",
                "64",
                "--page-size",
                "1",
                "--max-total-tokens",
                "2048",
                "--max-running-requests",
                "4",
                "--mem-fraction-static",
                "0.2",
                "--random-seed",
                "0",
                "--default-chat-template-kwargs",
                '{"enable_thinking": false}',
            ],
            env={"SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": "python"},
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            terminate_and_kill_process_tree(cls.process)

    def _flush_cache(self):
        response = requests.post(f"{DEFAULT_URL_FOR_TEST}/flush_cache", timeout=30)
        self.assertEqual(response.status_code, 200, response.text)

    def _payload(self, api, *, stream, max_tokens):
        payload = {"model": _MODEL, "temperature": 0, "stream": stream}
        if api == "responses":
            payload.update(input=_PROMPT, max_output_tokens=max_tokens, store=False)
        elif api == "completions":
            payload.update(prompt=_PROMPT, max_tokens=max_tokens, ignore_eos=True)
        else:
            payload.update(
                messages=[{"role": "user", "content": _PROMPT}],
                max_tokens=max_tokens,
            )
            if api == "chat/completions":
                payload["ignore_eos"] = True
        if stream and api in ("chat/completions", "completions"):
            payload["stream_options"] = {
                "include_usage": True,
                "continuous_usage_stats": True,
            }
        return payload

    def _request_usage(self, api, payload):
        with requests.post(
            f"{DEFAULT_URL_FOR_TEST}/v1/{api}",
            json=payload,
            stream=payload["stream"],
            timeout=60,
        ) as response:
            if response.status_code != 200:
                self.fail(f"HTTP {response.status_code}: {response.text}")
            if not payload["stream"]:
                return response.json()["usage"]

            self.assertIn("text/event-stream", response.headers["content-type"])
            events = []
            done = False
            for line in response.iter_lines():
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    done = True
                    continue
                event = json.loads(data)
                self.assertNotIn("error", event, event)
                events.append(event)

        self.assertTrue(events, "The SSE response contained no data events")
        if api == "responses":
            terminal = events[-1]
            self.assertIn(
                terminal["type"], ("response.completed", "response.incomplete")
            )
            return terminal["response"]["usage"]
        if api == "messages":
            self.assertEqual(events[-1]["type"], "message_stop")
            start = next(e for e in events if e["type"] == "message_start")
            self.assertNotIn("cache_creation_input_tokens", start["message"]["usage"])
            terminal = [e for e in events if e["type"] == "message_delta"][-1]
            self.assertIn("cache_creation_input_tokens", terminal["usage"])
            return terminal["usage"]

        self.assertTrue(done, "The OpenAI stream did not terminate with [DONE]")
        self.assertEqual(events[-1]["choices"], [])
        return events[-1]["usage"]

    def _partition(self, api, usage):
        if api == "messages":
            written = usage["cache_creation_input_tokens"]
            cached = usage.get("cache_read_input_tokens", 0)
            prompt = usage["input_tokens"] + cached + written
            output = usage["output_tokens"]
        else:
            is_responses = api == "responses"
            prompt = usage["input_tokens" if is_responses else "prompt_tokens"]
            output = usage["output_tokens" if is_responses else "completion_tokens"]
            details = usage[
                "input_tokens_details" if is_responses else "prompt_tokens_details"
            ]
            if is_responses:
                self.assertIs(details["cache_write_tokens_reported"], True)
            written = details["cache_write_tokens"]
            cached = details["cached_tokens"]
            self.assertEqual(usage["total_tokens"], prompt + output)

        for count in (prompt, output, cached, written):
            self.assertIs(type(count), int)
            self.assertGreaterEqual(count, 0)
        self.assertGreater(prompt, 64, "The request must cross a prefill chunk")
        self.assertGreater(output, 0)
        self.assertLessEqual(cached + written, prompt)
        return prompt, cached, written

    def test_cold_and_warm_usage_across_json_and_sse(self):
        for api in ("chat/completions", "completions", "responses", "messages"):
            cold_writes = []
            for stream in (False, True):
                with self.subTest(api=api, stream=stream):
                    self._flush_cache()
                    payload = self._payload(
                        api, stream=stream, max_tokens=8 if stream else 4
                    )
                    cold = self._partition(api, self._request_usage(api, payload))
                    warm = self._partition(api, self._request_usage(api, payload))
                    self.assertEqual(cold[1], 0)
                    self.assertGreater(cold[2], 0)
                    self.assertEqual(warm[0], cold[0])
                    self.assertGreater(warm[1], 0)
                    self.assertEqual(warm[2], 0, "Duplicate KV is not a new write")
                    cold_writes.append(cold[2])
                    if api == "messages":
                        count = requests.post(
                            f"{DEFAULT_URL_FOR_TEST}/v1/messages/count_tokens",
                            json={"model": _MODEL, "messages": payload["messages"]},
                            timeout=30,
                        )
                        self.assertEqual(count.status_code, 200, count.text)
                        self.assertEqual(cold[0], count.json()["input_tokens"])
            # Different output lengths must not change measured input writes.
            if len(cold_writes) == 2:
                self.assertEqual(cold_writes[0], cold_writes[1], api)

    def test_parallel_choices_keep_writes_unknown(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                self._flush_cache()
                payload = self._payload("chat/completions", stream=stream, max_tokens=4)
                payload["n"] = 2
                usage = self._request_usage("chat/completions", payload)
                self.assertNotIn("cache_write_tokens", usage["prompt_tokens_details"])
                self.assertGreater(usage["completion_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
