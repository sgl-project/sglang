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


class OAIChatClient(BaseBackendClient):
    async def make_request(
        self,
        request_func_input: RequestFuncInput,
        pbar: Optional[tqdm] = None,
    ) -> RequestFuncOutput:
        """Makes a request to the OpenAI Chat Completions API.

        Handles both streaming and non-streaming responses, including support
        for image data in messages. Calculates and returns various performance
        metrics.

        Args:
            request_func_input: Input parameters for the request.
            pbar: Optional tqdm progress bar to update.

        Returns:
            RequestFuncOutput: Output of the request, including generated text,
                            latency, TTFT, ITL, and success status.
        """
        api_url = request_func_input.api_url
        assert api_url.endswith(
            "chat/completions"
        ), "OpenAI Chat Completions API URL must end with 'chat/completions'."

        if request_func_input.image_data:
            # Build multi-image content: a list of image_url entries followed by the text
            content_items = [
                {
                    "type": "image_url",
                    "image_url": {"url": img_url},
                }
                for img_url in request_func_input.image_data
            ]
            content_items.append({"type": "text", "text": request_func_input.prompt})
            messages = [
                {
                    "role": "user",
                    "content": content_items,
                },
            ]
        else:
            messages = [{"role": "user", "content": request_func_input.prompt}]

        async with create_bench_client_session() as session:
            payload = {
                "model": request_func_input.model,
                "messages": messages,
                "temperature": 0.0,
                "max_completion_tokens": request_func_input.output_len,
                "stream": not self.args.disable_stream,
                "ignore_eos": not self.args.disable_ignore_eos,
                **request_func_input.extra_request_body,
            }

            # hack to accommodate different LoRA conventions between SGLang and vLLM.
            if request_func_input.lora_name:
                payload["model"] = request_func_input.lora_name
                payload["lora_path"] = request_func_input.lora_name

            headers = get_auth_headers()

            output = RequestFuncOutput.init_new(request_func_input)

            generated_text = ""
            output_len = request_func_input.output_len
            ttft = 0.0
            st = time.perf_counter()
            most_recent_timestamp = st
            try:
                async with session.post(
                    url=api_url, json=payload, headers=headers
                ) as response:
                    if response.status == 200:
                        if self.args.disable_stream:
                            # Non-streaming response
                            response_json = await response.json()
                            output.generated_text = response_json["choices"][0][
                                "message"
                            ]["content"]
                            output.success = True
                            output.latency = time.perf_counter() - st
                            output.ttft = (
                                output.latency
                            )  # For non-streaming, TTFT = total latency
                            output.output_len = response_json.get("usage", {}).get(
                                "completion_tokens", output_len
                            )
                        else:
                            # Streaming response
                            async for chunk_bytes in response.content:
                                chunk_bytes = chunk_bytes.strip()
                                if not chunk_bytes:
                                    continue

                                chunk = remove_prefix(
                                    chunk_bytes.decode("utf-8"), "data: "
                                )
                                latency = time.perf_counter() - st
                                if chunk == "[DONE]":
                                    pass
                                else:
                                    data = json.loads(chunk)

                                    # Check if this chunk contains content
                                    delta = data.get("choices", [{}])[0].get(
                                        "delta", {}
                                    )
                                    content = delta.get("content", "")

                                    if content:
                                        timestamp = time.perf_counter()
                                        # First token
                                        if ttft == 0.0:
                                            ttft = timestamp - st
                                            output.ttft = ttft

                                        # Decoding phase
                                        else:
                                            output.itl.append(
                                                timestamp - most_recent_timestamp
                                            )

                                        most_recent_timestamp = timestamp
                                        generated_text += content

                                    # Check for usage info in final chunk
                                    output_len = (data.get("usage") or {}).get(
                                        "completion_tokens", output_len
                                    )

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

        if pbar:
            pbar.update(1)
        return output
