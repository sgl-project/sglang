import json
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
from sglang.benchmark.utils import remove_prefix


class OpenAIBackendClient(BaseBackendClient):
    # set ignore_eos True by default
    async def request(
        self,
        request_func_input: RequestFuncInput,
        pbar: Optional[tqdm] = None,
    ) -> RequestFuncOutput:
        api_url = request_func_input.api_url
        assert api_url.endswith(
            "completions"
        ), "OpenAI Completions API URL must end with 'completions'."

        prompt = request_func_input.prompt

        async with _create_bench_client_session() as session:
            # Build payload with defaults that can be overridden by extra_request_body
            payload = {
                "model": request_func_input.model,
                "prompt": prompt,
                "best_of": 1,
                "max_tokens": request_func_input.output_len,
                "stream": not self.args.disable_stream,
            }

            # Add temperature default only if not specified in extra_request_body
            if "temperature" not in request_func_input.extra_request_body:
                payload["temperature"] = 0.0

            # Add ignore_eos default only if not specified in extra_request_body
            if "ignore_eos" not in request_func_input.extra_request_body:
                payload["ignore_eos"] = not self.args.disable_ignore_eos

            if self.args.return_logprob and self.args.top_logprobs_num > 0:
                payload["logprobs"] = self.args.top_logprobs_num

            # Merge in extra parameters - these will override defaults if present
            payload.update(request_func_input.extra_request_body)

            # hack to accommodate different LoRA conventions between SGLang and vLLM.
            if request_func_input.lora_name:
                payload["model"] = request_func_input.lora_name
                payload["lora_path"] = request_func_input.lora_name

            if request_func_input.image_data:
                payload.update({"image_data": request_func_input.image_data})

            headers = get_request_headers(self.args)
            if request_func_input.routing_key:
                headers[_ROUTING_KEY_HEADER] = request_func_input.routing_key

            output = RequestFuncOutput.init_new(request_func_input)

            generated_text = ""
            output_len = request_func_input.output_len
            ttft = 0.0
            st = time.perf_counter()
            output.start_time = st
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
                                        output.text_chunks.append(
                                            data["choices"][0]["text"]
                                        )
                                        output.itl.append(
                                            timestamp - most_recent_timestamp
                                        )

                                    most_recent_timestamp = timestamp
                                    generated_text += data["choices"][0]["text"]
                                    output_len = (data.get("usage") or {}).get(
                                        "completion_tokens", output_len
                                    )

                        output.generated_text = generated_text
                        output.success = True
                        output.latency = latency
                        output.output_len = output_len
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
