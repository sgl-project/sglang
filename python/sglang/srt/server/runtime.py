import atexit
import atexit
import json
import multiprocessing as mp
from typing import Dict, List, Optional, Union

import aiohttp
import requests

from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    is_port_available,
    kill_process_tree,
)


class Runtime:
    """
    A wrapper for the HTTP server.
    This is used for launching the server in a python program without
    using the commond line interface.

    It is mainly used for the frontend language.
    You should use the Engine class above if you want to do normal offline processing.
    """

    def __init__(
        self,
        log_level: str = "error",
        *args,
        **kwargs,
    ):
        """See the arguments in server_args.py::ServerArgs"""
        self.server_args = ServerArgs(*args, log_level=log_level, **kwargs)

        # before python program terminates, call shutdown implicitly. Therefore, users don't have to explicitly call .shutdown()
        atexit.register(self.shutdown)

        # Pre-allocate ports
        for port in range(self.server_args.port, 40000):
            if is_port_available(port):
                break
        self.server_args.port = port

        self.url = self.server_args.url()
        self.generate_url = self.url + "/generate"

        # NOTE: We store pid instead of proc to fix some issues during __delete__
        self.pid = None
        pipe_reader, pipe_writer = mp.Pipe(duplex=False)

        proc = mp.Process(
            target=launch_server,
            args=(self.server_args, pipe_writer),
        )
        proc.start()
        pipe_writer.close()
        self.pid = proc.pid

        try:
            init_state = pipe_reader.recv()
        except EOFError:
            init_state = ""

        if init_state != "ready":
            self.shutdown()
            raise RuntimeError(
                "Initialization failed. Please see the error messages above."
            )

        self.endpoint = RuntimeEndpoint(self.url)

    def shutdown(self):
        if self.pid is not None:
            kill_process_tree(self.pid)
            self.pid = None

    def cache_prefix(self, prefix: str):
        self.endpoint.cache_prefix(prefix)

    def get_tokenizer(self):
        return get_tokenizer(
            self.server_args.tokenizer_path,
            tokenizer_mode=self.server_args.tokenizer_mode,
            trust_remote_code=self.server_args.trust_remote_code,
        )

    async def async_generate(
        self,
        prompt: str,
        sampling_params: Optional[Dict] = None,
    ):
        if self.server_args.skip_tokenizer_init:
            json_data = {
                "input_ids": prompt,
                "sampling_params": sampling_params,
                "stream": True,
            }
        else:
            json_data = {
                "text": prompt,
                "sampling_params": sampling_params,
                "stream": True,
            }
        pos = 0

        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            async with session.post(self.generate_url, json=json_data) as response:
                async for chunk, _ in response.content.iter_chunks():
                    chunk = chunk.decode("utf-8")
                    if chunk and chunk.startswith("data:"):
                        if chunk == "data: [DONE]\n\n":
                            break
                        data = json.loads(chunk[5:].strip("\n"))
                        if "text" in data:
                            cur = data["text"][pos:]
                            if cur:
                                yield cur
                            pos += len(cur)
                        else:
                            yield data

    add_request = async_generate

    def generate(
        self,
        prompt: Union[str, List[str]],
        sampling_params: Optional[Dict] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        lora_path: Optional[List[Optional[str]]] = None,
    ):
        json_data = {
            "text": prompt,
            "sampling_params": sampling_params,
            "return_logprob": return_logprob,
            "logprob_start_len": logprob_start_len,
            "top_logprobs_num": top_logprobs_num,
            "lora_path": lora_path,
        }
        assert not isinstance(lora_path, list) or len(lora_path) == len(prompt)
        response = requests.post(
            self.url + "/generate",
            json=json_data,
            )
        return json.dumps(response.json())

    def encode(
        self,
        prompt: Union[str, List[str], List[Dict], List[List[Dict]]],
    ):
        json_data = {"text": prompt}
        response = requests.post(self.url + "/encode", json=json_data)
        return json.dumps(response.json())

    async def get_server_info(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.url}/get_server_info") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_data = await response.json()
                    raise RuntimeError(
                        f"Failed to get server info. {error_data['error']['message']}"
                    )

    def __del__(self):
        self.shutdown()
