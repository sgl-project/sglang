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


class TrussClient(BaseBackendClient):
    async def make_request(
        self,
        request_func_input: RequestFuncInput,
        pbar: Optional[tqdm] = None,
    ) -> RequestFuncOutput:
        api_url = request_func_input.api_url

        prompt = request_func_input.prompt

        async with create_bench_client_session() as session:
            payload = {
                "model": request_func_input.model,
                "prompt": prompt,
                "temperature": 0.0,
                "best_of": 1,
                "max_tokens": request_func_input.output_len,
                "stream": not self.args.disable_stream,
                "ignore_eos": not self.args.disable_ignore_eos,
                **request_func_input.extra_request_body,
            }
            headers = get_auth_headers()

            output = RequestFuncOutput.init_new(request_func_input)

            generated_text = ""
            ttft = 0.0
            st = time.perf_counter()
            most_recent_timestamp = st
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
                                if data["choices"][0]["text"]:
                                    timestamp = time.perf_counter()
                                    # First token
                                    if ttft == 0.0:
                                        ttft = time.perf_counter() - st
                                        output.ttft = ttft

                                    # Decoding phase
                                    else:
                                        output.itl.append(
                                            timestamp - most_recent_timestamp
                                        )

                                    most_recent_timestamp = timestamp
                                    generated_text += data["choices"][0]["text"]

                        output.generated_text = generated_text
                        output.success = True
                        output.latency = latency
                        output.output_len = request_func_input.output_len
                    else:
                        output.error = response.reason or ""
                        output.success = False
            except Exception:
                output.success = False
                exc_info = sys.exc_info()
                output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output
