import asyncio
import os
import subprocess
import time

import aiohttp
import openai
import requests


class SGlangEngine:
    def __init__(
        self,
        model=os.getenv("MODEL_PATH"),
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 30000)),
    ):
        self.model = model
        self.host = host
        self.port = port
        self.base_url = f"http://{self.host}:{self.port}"
        self.process = None

    def start_server(self):
        command = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]

        # Dictionary of all possible options and their corresponding env var names
        options = {
            "MODEL_PATH": "--model-path",
            "TOKENIZER_PATH": "--tokenizer-path",
            "TOKENIZER_MODE": "--tokenizer-mode",
            "LOAD_FORMAT": "--load-format",
            "DTYPE": "--dtype",
            "CONTEXT_LENGTH": "--context-length",
            "QUANTIZATION": "--quantization",
            "SERVED_MODEL_NAME": "--served-model-name",
            "CHAT_TEMPLATE": "--chat-template",
            "MEM_FRACTION_STATIC": "--mem-fraction-static",
            "MAX_RUNNING_REQUESTS": "--max-running-requests",
            "MAX_TOTAL_TOKENS": "--max-total-tokens",
            "CHUNKED_PREFILL_SIZE": "--chunked-prefill-size",
            "MAX_PREFILL_TOKENS": "--max-prefill-tokens",
            "SCHEDULE_POLICY": "--schedule-policy",
            "SCHEDULE_CONSERVATIVENESS": "--schedule-conservativeness",
            "TENSOR_PARALLEL_SIZE": "--tensor-parallel-size",
            "STREAM_INTERVAL": "--stream-interval",
            "RANDOM_SEED": "--random-seed",
            "LOG_LEVEL": "--log-level",
            "LOG_LEVEL_HTTP": "--log-level-http",
            "API_KEY": "--api-key",
            "FILE_STORAGE_PATH": "--file-storage-path",
            "DATA_PARALLEL_SIZE": "--data-parallel-size",
            "LOAD_BALANCE_METHOD": "--load-balance-method",
            "ATTENTION_BACKEND": "--attention-backend",
            "SAMPLING_BACKEND": "--sampling-backend",
        }

        # Boolean flags
        boolean_flags = [
            "SKIP_TOKENIZER_INIT",
            "TRUST_REMOTE_CODE",
            "LOG_REQUESTS",
            "SHOW_TIME_COST",
            "DISABLE_RADIX_CACHE",
            "DISABLE_CUDA_GRAPH",
            "DISABLE_OUTLINES_DISK_CACHE",
            "ENABLE_TORCH_COMPILE",
            "ENABLE_P2P_CHECK",
            "ENABLE_FLASHINFER_MLA",
            "TRITON_ATTENTION_REDUCE_IN_FP32",
        ]

        # Add options from environment variables only if they are set
        for env_var, option in options.items():
            value = os.getenv(env_var)
            if value is not None and value != "":
                command.extend([option, value])

        # Add boolean flags only if they are set to true
        for flag in boolean_flags:
            if os.getenv(flag, "").lower() in ("true", "1", "yes"):
                command.append(f"--{flag.lower().replace('_', '-')}")

        self.process = subprocess.Popen(command, stdout=None, stderr=None)
        print(f"Server started with PID: {self.process.pid}")

    def wait_for_server(self, timeout=900, interval=5):
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/v1/models")
                if response.status_code == 200:
                    print("Server is ready!")
                    return True
            except requests.RequestException:
                pass
            time.sleep(interval)
        raise TimeoutError("Server failed to start within the timeout period.")

    def shutdown(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            print("Server shut down.")


class OpenAIRequest:
    def __init__(self, base_url="http://0.0.0.0:30000/v1", api_key="EMPTY"):
        self.client = openai.Client(base_url=base_url, api_key=api_key)

    async def request_chat_completions(
        self,
        model="default",
        messages=None,
        max_tokens=100,
        stream=False,
        frequency_penalty=0.0,
        n=1,
        stop=None,
        temperature=1.0,
        top_p=1.0,
    ):
        if messages is None:
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant"},
                {"role": "user", "content": "List 3 countries and their capitals."},
            ]

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stream=stream,
            frequency_penalty=frequency_penalty,
            n=n,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
        )

        if stream:
            async for chunk in response:
                yield chunk.to_dict()
        else:
            yield response.to_dict()

    async def request_completions(
        self,
        model="default",
        prompt="The capital of France is",
        max_tokens=100,
        stream=False,
        frequency_penalty=0.0,
        n=1,
        stop=None,
        temperature=1.0,
        top_p=1.0,
    ):
        response = self.client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            stream=stream,
            frequency_penalty=frequency_penalty,
            n=n,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
        )

        if stream:
            async for chunk in response:
                yield chunk.to_dict()
        else:
            yield response.to_dict()

    async def get_models(self):
        response = await self.client.models.list()
        return response
