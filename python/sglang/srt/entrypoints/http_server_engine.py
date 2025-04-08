import base64
import copy
import dataclasses
import multiprocessing
import pickle
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
import torch
import torch.distributed as dist

from sglang.srt.entrypoints.base_engine import EngineBase
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    HttpSerializer,
    MultiprocessingSerializer,
    kill_process_tree,
)


def launch_server_worker(server_args: ServerArgs):
    launch_server(server_args)


def launch_server_process(server_args: ServerArgs) -> multiprocessing.Process:

    p = multiprocessing.Process(target=launch_server_worker, args=(server_args,))
    p.start()

    base_url = server_args.url()
    timeout = 180.0
    start_time = time.time()

    with requests.Session() as session:
        while time.time() - start_time < timeout:
            try:
                headers = {
                    "Content-Type": "application/json; charset=utf-8",
                    "Authorization": f"Bearer {server_args.api_key}",
                }
                response = session.get(f"{base_url}/health_generate", headers=headers)
                if response.status_code == 200:
                    return p
            except requests.RequestException:
                pass

            if not p.is_alive():
                raise Exception("Server process terminated unexpectedly.")

            time.sleep(2)

    p.terminate()
    raise TimeoutError("Server failed to start within the timeout period.")


class HttpServerEngineForRL(EngineBase):
    def __init__(self, **kwargs):
        self.server_args = ServerArgs(**kwargs)
        print(f"launch_server_from_verl_engine {self.server_args.port}")
        self.process = launch_server_process(self.server_args)

    def _url(self, path: str) -> str:
        """Construct full URL for server endpoint."""
        return f"http://{self.server_args.host}:{self.server_args.port}/{path}"

    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]],
        load_format: Optional[str] = None,
        flush_cache: bool = False,
    ):
        """
        Update model weights from tensor data. The HTTPS server will only post meta data, and the real weights will be copied directly from GPUs.

        Note: The model should be on GPUs rather than CPU for this functionality to work properly.
        If you encounter issues, ensure your model is loaded on GPU devices rather than CPU.
        """

        print(f"update_weights_from_tensor of HttpServerEngineForRL")
        serialized_named_tensors = HttpSerializer.serialize(
            MultiprocessingSerializer.serialize(named_tensors)
        )

        response = requests.post(
            self._url("update_weights_from_tensor"),
            json={
                "serialized_named_tensors": [
                    serialized_named_tensors for _ in range(self.server_args.tp_size)
                ],
                "load_format": load_format,
                "flush_cache": flush_cache,
            },
        )
        response.raise_for_status()
        return response.json()

    def shutdown(self):
        kill_process_tree(self.process.pid)

    def generate(
        self,
        prompt=None,
        sampling_params=None,
        input_ids=None,
        image_data=None,
        return_logprob=False,
        logprob_start_len=None,
        top_logprobs_num=None,
        token_ids_logprob=None,
        lora_path=None,
        custom_logit_processor=None,
    ):
        payload = {
            "text": prompt,
            "sampling_params": sampling_params,
            "input_ids": input_ids,
            "image_data": image_data,
            "return_logprob": return_logprob,
            "logprob_start_len": logprob_start_len,
            "top_logprobs_num": top_logprobs_num,
            "token_ids_logprob": token_ids_logprob,
            "lora_path": lora_path,
            "custom_logit_processor": custom_logit_processor,
        }
        # Filter out None values
        payload = {k: v for k, v in payload.items() if v is not None}

        response = requests.post(self._url("generate"), json=payload)
        response.raise_for_status()
        return response.json()

    def release_memory_occupation(self):
        response = requests.post(self._url("release_memory_occupation"), json={})
        response.raise_for_status()
        return response.json()

    def resume_memory_occupation(self):
        response = requests.post(self._url("resume_memory_occupation"), json={})
        response.raise_for_status()
        return response.json()
