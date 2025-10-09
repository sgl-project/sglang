import asyncio
import json
import random
import sys
import time
import traceback
from argparse import Namespace
from typing import Any, AsyncGenerator, Dict, List

import numpy as np
import requests
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

from sglang.benchmark.backends.base_client import BaseBackendClient
from sglang.benchmark.datasets.common import (
    BaseDatasetLoader,
    DatasetRow,
    RequestFuncInput,
    RequestFuncOutput,
)
from sglang.benchmark.metrics import do_calculate_metrics, print_metrics, save_results
from sglang.benchmark.utils import (
    create_bench_client_session,
    get_auth_headers,
    get_bool_env_var,
)


class BenchmarkRunner:
    def __init__(
        self,
        args: Namespace,
        backend_client: BaseBackendClient,
        dataset_loader: BaseDatasetLoader,
        tokenizer: PreTrainedTokenizerBase,
        input_requests: List[DatasetRow],
        api_url: str,
        base_url: str,
        extra_request_body: Dict[str, Any],
    ):
        self.args = args
        self.backend_client = backend_client
        self.tokenizer = tokenizer
        self.input_requests = input_requests
        self.api_url = api_url
        self.base_url = base_url
        self.extra_request_body = extra_request_body
        self.semaphore = (
            asyncio.Semaphore(args.max_concurrency) if args.max_concurrency else None
        )

        custom_generator_provider = dataset_loader.get_request_generator()
        if custom_generator_provider:
            self.request_generator = custom_generator_provider()
        else:
            self.request_generator = self._default_request_generator()

    async def async_request_profile(self, api_url: str) -> RequestFuncOutput:
        async with create_bench_client_session() as session:
            output = RequestFuncOutput()
            try:
                async with session.post(url=api_url) as response:
                    if response.status == 200:
                        output.success = True
                    else:
                        output.error = response.reason or ""
                        output.success = False
            except Exception:
                output.success = False
                exc_info = sys.exc_info()
                output.error = "".join(traceback.format_exception(*exc_info))

        return output

    async def _default_request_generator(self) -> AsyncGenerator[DatasetRow, None]:
        for request in self.input_requests:
            yield request

            if self.args.request_rate == float("inf"):
                continue

            # Sample the request interval from the exponential distribution.
            interval = np.random.exponential(1.0 / self.args.request_rate)
            # The next request will be sent after the interval.
            await asyncio.sleep(interval)

    async def _warmup(self):
        if self.args.warmup_requests <= 0:
            return

        print(f"Starting warmup with {self.args.warmup_requests} requests...")
        test_request = self.input_requests[0]

        lora_name = None
        if self.args.lora_name:
            lora_name = self.args.lora_name[0]

        test_input = RequestFuncInput(
            model=self.args.model,
            prompt=test_request.prompt,
            api_url=self.api_url,
            prompt_len=test_request.prompt_len,
            output_len=min(test_request.output_len, 32),
            lora_name=lora_name,
            image_data=test_request.image_data,
            extra_request_body=self.extra_request_body,
        )

        warmup_tasks = [
            asyncio.create_task(self.backend_client.make_request(test_input))
            for _ in range(self.args.warmup_requests)
        ]
        warmup_outputs = await asyncio.gather(*warmup_tasks)

        if not any(o.success for o in warmup_outputs):
            raise ValueError(
                "Warmup failed - Please make sure benchmark arguments "
                f"are correctly specified. Error: {warmup_outputs[0].error}"
            )
        print(
            f"Warmup completed with {self.args.warmup_requests} sequences. Starting main benchmark run..."
        )

    async def _dispatch_request(
        self, request_func_input: RequestFuncInput, pbar: tqdm
    ) -> RequestFuncOutput:
        if self.semaphore is None:
            return await self.backend_client.make_request(request_func_input, pbar)
        async with self.semaphore:
            return await self.backend_client.make_request(request_func_input, pbar)

    async def run(self):
        await self._warmup()

        if (
            "sglang" in self.args.backend and get_bool_env_var("SGLANG_IS_IN_CI")
        ) or self.args.flush_cache:
            requests.post(self.base_url + "/flush_cache", headers=get_auth_headers())

        time.sleep(1.0)

        if self.args.profile:
            print("Starting profiler...")
            profile_output = await self.async_request_profile(
                api_url=self.base_url + "/start_profile"
            )
            if profile_output.success:
                print("Profiler started")

        benchmark_start_time = time.perf_counter()
        tasks: List[asyncio.Task] = []
        pbar = tqdm(total=len(self.input_requests), disable=self.args.disable_tqdm)

        async for request_row in self.request_generator:
            lora_name = None
            if self.args.lora_name and len(self.args.lora_name) != 0:
                lora_name = random.choice(self.args.lora_name)

            req_input = RequestFuncInput(
                model=self.args.model,
                prompt=request_row.prompt,
                api_url=self.api_url,
                prompt_len=request_row.prompt_len,
                output_len=request_row.output_len,
                lora_name=lora_name,
                image_data=request_row.image_data,
                extra_request_body=self.extra_request_body,
                timestamp=request_row.timestamp,
            )
            tasks.append(asyncio.create_task(self._dispatch_request(req_input, pbar)))

        outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)
        pbar.close()
        benchmark_duration = time.perf_counter() - benchmark_start_time

        if self.args.profile:
            print("Stopping profiler...")
            profile_output = await self.async_request_profile(
                api_url=self.base_url + "/stop_profile"
            )
            if profile_output.success:
                print("Profiler stopped")

        accept_length = None
        if "sglang" in self.args.backend:
            server_info = requests.get(
                self.base_url + "/get_server_info", headers=get_auth_headers()
            )
            if server_info.status_code == 200:
                server_info_json = server_info.json()
                if "decode" in server_info_json:
                    server_info_json = server_info_json["decode"][0]
                if (
                    "internal_states" in server_info_json
                    and server_info_json["internal_states"]
                ):
                    accept_length = server_info_json["internal_states"][0].get(
                        "avg_spec_accept_length", None
                    )

        metrics, output_lens = do_calculate_metrics(
            self.input_requests, outputs, benchmark_duration, self.tokenizer
        )
        metrics.accept_length = accept_length

        print_metrics(metrics, self.args, benchmark_duration)
        res = save_results(metrics, self.args, benchmark_duration, outputs, output_lens)

        return res
