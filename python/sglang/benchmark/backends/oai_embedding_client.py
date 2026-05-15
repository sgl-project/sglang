import sys
import time
import traceback
from typing import Optional

from tqdm.asyncio import tqdm

from sglang.benchmark.backends.base_client import (
    BaseBackendClient,
    RequestFuncInput,
    RequestFuncOutput,
)
from sglang.benchmark.backends.common import (
    _ROUTING_KEY_HEADER,
    _create_bench_client_session,
    get_request_headers,
)


class OpenAIEmbeddingBackendClient(BaseBackendClient):
    async def request(
        self,
        request_func_input: RequestFuncInput,
        pbar: Optional[tqdm] = None,
    ) -> RequestFuncOutput:
        api_url = request_func_input.api_url

        async with _create_bench_client_session() as session:
            payload = {
                "input": request_func_input.prompt,
                "model": request_func_input.model,
            }

            if request_func_input.lora_name:
                payload["model"] = request_func_input.lora_name
                payload["lora_path"] = request_func_input.lora_name

            payload.update(request_func_input.extra_request_body)

            headers = get_request_headers(self.args)
            if request_func_input.routing_key:
                headers[_ROUTING_KEY_HEADER] = request_func_input.routing_key

            output = RequestFuncOutput.init_new(request_func_input)

            st = time.perf_counter()
            output.start_time = st
            try:
                async with session.post(
                    url=api_url, json=payload, headers=headers
                ) as response:
                    if response.status == 200:
                        await response.json()
                        output.latency = time.perf_counter() - st
                        output.success = True
                        output.output_len = 0
                    else:
                        output.error = (
                            (response.reason or "") + ": " + (await response.text())
                        )
                        output.success = False
            except Exception:
                output.success = False
                exc_info = sys.exc_info()
                output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output
