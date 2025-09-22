import json
import sys
import time
import traceback
from typing import Optional

from tqdm.asyncio import tqdm

from sglang.benchmark.backends.base_client import BaseBackendClient
from sglang.benchmark.datasets.common import RequestFuncInput, RequestFuncOutput
from sglang.benchmark.utils import create_bench_client_session, remove_prefix


class TrtClient(BaseBackendClient):
    async def make_request(
        self,
        request_func_input: RequestFuncInput,
        pbar: Optional[tqdm] = None,
    ) -> RequestFuncOutput:
        api_url = request_func_input.api_url
        assert api_url.endswith("generate_stream")

        async with create_bench_client_session() as session:
            payload = {
                "accumulate_tokens": True,
                "text_input": request_func_input.prompt,
                "temperature": 0.000001,
                "top_p": 1.0,
                "max_tokens": request_func_input.output_len,
                "stream": True,
                "min_length": request_func_input.output_len,
                "end_id": 1048576,
                **request_func_input.extra_request_body,
            }
            if self.args.disable_ignore_eos:
                del payload["min_length"]
                del payload["end_id"]
            output = RequestFuncOutput.init_new(request_func_input)

            ttft = 0.0
            st = time.perf_counter()
            most_recent_timestamp = st
            try:
                async with session.post(url=api_url, json=payload) as response:
                    if response.status == 200:
                        async for chunk_bytes in response.content:
                            chunk_bytes = chunk_bytes.strip()
                            if not chunk_bytes:
                                continue

                            chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data:")

                            data = json.loads(chunk)
                            output.generated_text += data["text_output"]
                            timestamp = time.perf_counter()
                            # First token
                            if ttft == 0.0:
                                ttft = timestamp - st
                                output.ttft = ttft

                            # Decoding phase
                            else:
                                output.itl.append(timestamp - most_recent_timestamp)

                            most_recent_timestamp = timestamp

                        output.latency = most_recent_timestamp - st
                        output.success = True
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
