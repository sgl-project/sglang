import json
import sys
import time
import traceback
from typing import Optional

from tqdm.asyncio import tqdm

from sglang.benchmark.backends.base_client import BaseBackendClient
from sglang.benchmark.datasets.common import RequestFuncInput, RequestFuncOutput
from sglang.benchmark.utils import (
    create_bench_client_session,
    get_auth_headers,
    remove_prefix,
)


class SglangClient(BaseBackendClient):
    async def make_request(
        self,
        request_func_input: RequestFuncInput,
        pbar: Optional[tqdm] = None,
    ) -> RequestFuncOutput:
        api_url = request_func_input.api_url
        prompt = request_func_input.prompt

        async with create_bench_client_session() as session:
            payload = {
                ("text" if isinstance(prompt, str) else "input_ids"): prompt,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": request_func_input.output_len,
                    "ignore_eos": not self.args.disable_ignore_eos,
                },
                "stream": not self.args.disable_stream,
                "lora_path": request_func_input.lora_name,
                "return_logprob": self.args.return_logprob,
                "logprob_start_len": -1,
                **request_func_input.extra_request_body,
            }

            # Add image data if available (list of image urls/base64)
            if request_func_input.image_data:
                payload["image_data"] = request_func_input.image_data

            headers = get_auth_headers()

            output = RequestFuncOutput.from_input(request_func_input)

            generated_text = ""
            output_len = request_func_input.output_len
            ttft = 0.0
            st = time.perf_counter()
            most_recent_timestamp = st
            last_output_len = 0
            try:
                async with session.post(
                    url=api_url, json=payload, headers=headers
                ) as response:
                    if response.status == 200:
                        async for chunk_bytes in response.content:
                            chunk_bytes = chunk_bytes.strip()
                            if not chunk_bytes:
                                continue

                            chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                            latency = time.perf_counter() - st
                            if chunk == "[DONE]":
                                pass
                            else:
                                data = json.loads(chunk)

                                # NOTE: Some completion API might have a last
                                # usage summary response without a token so we
                                # want to check a token was generated
                                if "text" in data and data["text"]:
                                    timestamp = time.perf_counter()
                                    generated_text = data["text"]
                                    output_len = data["meta_info"]["completion_tokens"]

                                    # First token
                                    if ttft == 0.0:
                                        ttft = time.perf_counter() - st
                                        output.ttft = ttft

                                    # Decoding phase
                                    else:
                                        num_new_tokens = output_len - last_output_len
                                        if num_new_tokens == 0:
                                            continue
                                        adjust_itl = (
                                            timestamp - most_recent_timestamp
                                        ) / num_new_tokens
                                        output.itl.extend([adjust_itl] * num_new_tokens)

                                    most_recent_timestamp = timestamp
                                    last_output_len = output_len

                        output.generated_text = generated_text
                        output.success = True
                        output.latency = latency
                        output.output_len = output_len
                    else:
                        output.error = response.reason or ""
                        output.success = False
            except Exception:
                output.success = False
                exc_info = sys.exc_info()
                output.error = "".join(traceback.format_exception(*exc_info))
                print(f"{output.error=}")

        if pbar:
            pbar.update(1)
        return output
