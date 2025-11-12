import asyncio
import hashlib
import json
import logging
import os
import time
import unittest
from datetime import datetime
from types import SimpleNamespace
from typing import Optional

import requests
from jsonschema import ValidationError, validate
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)
from sglang.utils import download_and_cache_file

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestKimiK2Thinking(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "moonshotai/Kimi-K2-Thinking"
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--tp",
            "8",
            "--trust-remote-code",
            "--tool-call-parser",
            "kimi_k2",
            "--reasoning-parser",
            "kimi_k2",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=5000,
            other_args=other_args,
            env={"SGLANG_TOOL_STRICT_LEVEL": "0"},
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_a_gsm8k(
        self,
    ):
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (Kimi-K2-Thinking)\n" f'{metrics["accuracy"]=:.3f}\n'
            )
            self.assertGreater(metrics["accuracy"], 0.95)

    def test_tool_call_trigger_similarity_and_schema_accuracy(self):
        """
        Testing on trigger similarity compared against official and schema accuracy.
        Since we are not calling official API to compare, we use a pre-determined
        tool call rate as target.
        Refer to https://github.com/MoonshotAI/K2-Vendor-Verifier/blob/main/README.md
        for more details.
        """
        requests.get(self.base_url + "/flush_cache")
        args = SimpleNamespace(
            num_questions=100,
            concurrency=5,
            data_path="/shared/user/samples.jsonl",
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )

        if args.data_path is None:
            url = "https://raw.githubusercontent.com/MoonshotAI/K2-Vendor-Verifier/refs/heads/main/samples.jsonl"
            file_name = download_and_cache_file(url)
        else:
            file_name = args.data_path

        validator = ToolCallsValidator(
            model=self.model,
            base_url=self.base_url,
            concurrency=args.concurrency,
            output_file="kimi_k2_tool_call_validation_results.jsonl",
            summary_file="kimi_k2_tool_call_validation_summary.json",
            num_questions=args.num_questions,
            timeout=600,
            max_retries=0,
            # Note: testing data provided in K2-Vendor-Verifier until commit 91a154a
            # is for Kimi-K2-Instruct-0905. Thus we need to override the temperature
            # and tool choice in the data as suggested by Kimi-K2-Thinking official
            # doc
            extra_body={
                "temperature": 1.0,
                "tool_choice": "auto",
            },
        )
        asyncio.run(validator.validate_file(file_name))
        summary = validator.summary
        print("Kimi-K2-Thinking Tool Call Validation Summary:")
        print("--------------------------------")
        print(summary)
        print("--------------------------------")
        schema_accuracy = (
            summary["successful_tool_call_count"] / summary["finish_tool_calls"]
        )
        # Expected tool call rate is ~30%.
        tool_call_rate = summary["finish_tool_calls"] / summary["success_count"]
        print(f"Schema Accuracy: {schema_accuracy:.3f}")
        print(f"Tool Call Rate: {tool_call_rate:.3f}")
        if is_in_ci():
            write_github_step_summary(
                f"### test_tool_call_trigger_similarity_and_schema_accuracy (Kimi-K2-Thinking)\n"
                f"{schema_accuracy=:.3f}\n"
                f"{tool_call_rate=:.3f}\n"
            )
            self.assertGreater(schema_accuracy, 0.7)
            self.assertGreater(tool_call_rate, 0.3)


######### Code Below References MoonshotAI/K2-Vendor-Verifier ########
def compute_hash(obj: dict) -> str:
    """Compute a stable hash of the request dict."""
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


class ToolCallsValidator:
    """Validator for tool calls."""

    def __init__(
        self,
        model: str,
        base_url: str,
        concurrency: int = 4,
        output_file: str = "results.jsonl",
        summary_file: str = "summary.json",
        timeout: int = 600,
        max_retries: int = 3,
        extra_body: Optional[dict] = None,
        incremental: bool = False,
        num_questions: Optional[int] = None,
    ):
        self.model = model
        self.base_url = base_url
        self.concurrency = concurrency
        self.semaphore = asyncio.Semaphore(concurrency)
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_body = extra_body or {}
        self.output_file = output_file
        self.summary_file = summary_file
        self.incremental = incremental
        self.num_questions = num_questions
        self.results: list[dict] = []

        self.client = AsyncOpenAI(
            base_url=f"{self.base_url}/v1",
            api_key="sk-123456",
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

        logger.info(f"Results will be saved to {self.output_file}")
        logger.info(f"Summary will be saved to {self.summary_file}")

    def prepare_request(self, request: dict) -> dict:
        """Process request messages and set model."""
        req = request.copy()
        if "messages" in req:
            for message in req["messages"]:
                if message.get("role") == "_input":
                    message["role"] = "system"
        if self.model:
            req["model"] = self.model
        return req

    def read_jsonl(self, file_path: str) -> list[dict]:
        """Load and prepare JSONL requests, compute hash."""
        requests = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    raw_req = json.loads(line.strip())
                    prepared_req = self.prepare_request(raw_req)
                    requests.append(
                        {
                            "data_index": line_num,
                            "raw": raw_req,
                            "prepared": prepared_req,
                            "hash": compute_hash(prepared_req),
                        }
                    )
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing line {line_num}: {e}")
        return requests

    def read_result_jsonl(self, file_path: str) -> list[dict]:
        results = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                results.append(json.loads(line))
        return results

    async def send_request(self, request: dict) -> tuple[str, dict]:
        try:
            if request.get("stream", False):
                return await self._handle_stream_request(request)
            else:
                response = await self.client.chat.completions.create(
                    **request, extra_body=self.extra_body
                )
                return "success", response.model_dump()
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return "failed", {"error": str(e)}

    async def _handle_stream_request(self, request: dict) -> tuple[str, dict]:
        try:
            stream = await self.client.chat.completions.create(
                **request, extra_body=self.extra_body
            )

            request_id = None
            created = None
            full_content = []
            tool_calls: dict[int, dict] = {}
            finish_reason = None
            usage = None

            async for event in stream:
                if hasattr(event, "id") and event.id:
                    request_id = event.id
                if hasattr(event, "created") and event.created:
                    created = event.created

                if not hasattr(event, "choices") or not event.choices:
                    logger.warning("Empty choices in stream event")
                    continue

                choice = event.choices[0]

                if hasattr(choice, "delta") and choice.delta:
                    if hasattr(choice.delta, "content") and choice.delta.content:
                        full_content.append(choice.delta.content)

                    if hasattr(choice.delta, "tool_calls") and choice.delta.tool_calls:
                        for tc in choice.delta.tool_calls:
                            idx = tc.index if tc.index is not None else 0

                            if idx not in tool_calls:
                                tool_calls[idx] = {
                                    "id": tc.id,
                                    "type": tc.type,
                                    "function": {"name": "", "arguments": ""},
                                }

                            if hasattr(tc, "function") and tc.function:
                                if hasattr(tc.function, "name") and tc.function.name:
                                    tool_calls[idx]["function"][
                                        "name"
                                    ] = tc.function.name
                                if (
                                    hasattr(tc.function, "arguments")
                                    and tc.function.arguments
                                ):
                                    tool_calls[idx]["function"][
                                        "arguments"
                                    ] += tc.function.arguments

                if hasattr(choice, "finish_reason") and choice.finish_reason:
                    finish_reason = choice.finish_reason

                if hasattr(choice, "usage") and choice.usage:
                    usage = choice.usage

            response = {
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": request.get("model", ""),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "".join(full_content),
                            "tool_calls": (
                                list(tool_calls.values()) if tool_calls else None
                            ),
                        },
                        "finish_reason": finish_reason or "stop",
                    }
                ],
                "usage": usage,
            }
            return "success", response
        except Exception as e:
            logger.error(f"Stream request failed: {e}")
            return "failed", {"error": str(e)}

    async def process_request(self, prepared_req: dict, data_index: int) -> dict:
        """Process a single request, record duration and status."""
        async with self.semaphore:
            start_time = time.time()
            status, response = await self.send_request(prepared_req["prepared"])
            duration_ms = int((time.time() - start_time) * 1000)

            finish_reason = None
            tool_calls_valid = None

            if response and "choices" in response:
                choice = response["choices"][0] if response["choices"] else {}
                finish_reason = choice.get("finish_reason")
                if finish_reason == "tool_calls":
                    tools = prepared_req["prepared"].get("tools", [])
                    tool_calls = choice.get("message", {}).get("tool_calls", [])
                    tool_calls_valid = all(
                        self.validate_tool_call(tc, tools) for tc in tool_calls
                    )

            result = {
                "data_index": data_index,
                "request": prepared_req["prepared"],
                "response": response,
                "status": status,
                "finish_reason": finish_reason,
                "tool_calls_valid": tool_calls_valid,
                "last_run_at": datetime.now().isoformat(),
                "duration_ms": duration_ms,
                "hash": prepared_req["hash"],
            }
            return result

    def validate_tool_call(self, tool_call: dict, tools: list[dict]) -> bool:
        """Validate tool call arguments against schema."""
        try:
            tool_name = tool_call["function"]["name"]
            schema = next(
                (
                    t["function"]["parameters"]
                    for t in tools
                    if t["function"]["name"] == tool_name
                ),
                None,
            )
            if not schema:
                logger.warning(f"No schema for tool {tool_name}")
                return False
            args = tool_call["function"]["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            validate(instance=args, schema=schema)
            return True
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"Schema validation failed: {e}")
            return False
        except Exception as e:
            logger.warning(f"Unexpected validation error: {e}")
            return False

    async def validate_file(self, file_path: str):
        """Validate all requests from a file, supports incremental mode."""
        all_requests = self.read_jsonl(file_path)
        if self.num_questions is not None:
            all_requests = all_requests[: self.num_questions]
        existing_results = []
        existing_hash_map = {}

        if self.incremental and os.path.exists(self.output_file):
            existing_results = self.read_result_jsonl(self.output_file)
            for r in existing_results:
                existing_hash_map[r["hash"]] = r
            logger.info(f"Loaded {len(existing_results)} existing results")

        tasks = []
        self.results = []

        for req in all_requests:
            h = req["hash"]
            data_index = req["data_index"]
            if self.incremental and h in existing_hash_map:
                r = existing_hash_map[h]
                if r.get("status") == "success":
                    self.results.append(r)
                    continue  # skip successful
            tasks.append(self.process_request(req, data_index))

        with tqdm_asyncio(total=len(tasks), desc="Processing", unit="req") as pbar:
            for task in asyncio.as_completed(tasks):
                try:
                    res = await task
                    self.results.append(res)
                except Exception as e:
                    logger.error(f"Task failed: {e}")
                finally:
                    pbar.update(1)

        self.results.sort(key=lambda r: r["data_index"])
        self.compute_summary()

    def compute_summary(self):
        """Compute summary from all results."""
        summary = {
            "model": self.model,
            "success_count": 0,
            "failure_count": 0,
            "finish_stop": 0,
            "finish_tool_calls": 0,
            "finish_others": 0,
            "finish_others_detail": {},
            "schema_validation_error_count": 0,
            "successful_tool_call_count": 0,
        }
        for r in self.results:
            status = r.get("status")
            finish_reason = r.get("finish_reason")
            tool_calls_valid = r.get("tool_calls_valid")

            if status == "success":
                summary["success_count"] += 1
            else:
                summary["failure_count"] += 1

            if finish_reason == "stop":
                summary["finish_stop"] += 1
            elif finish_reason == "tool_calls":
                summary["finish_tool_calls"] += 1
                if tool_calls_valid:
                    summary["successful_tool_call_count"] += 1
                else:
                    summary["schema_validation_error_count"] += 1
            elif finish_reason:
                summary["finish_others"] += 1
                summary["finish_others_detail"].setdefault(finish_reason, 0)
                summary["finish_others_detail"][finish_reason] += 1
        self.summary = summary


if __name__ == "__main__":
    unittest.main()
