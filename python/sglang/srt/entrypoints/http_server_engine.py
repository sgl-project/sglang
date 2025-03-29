import base64
import copy
import pickle
import threading
import time
from typing import Dict, List, Optional, Tuple, Union

import requests
import torch
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh, DTensor

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import MultiprocessingSerializer, kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)


def serialize_for_http(data):
    # First pickle the data, then convert to base64 for safe HTTP transmission
    pickled = pickle.dumps(data)
    return base64.b64encode(pickled).decode("utf-8")


import dataclasses


def server_args_to_launch_params(args: ServerArgs, timeout: float = 120.0):
    # 1. model path
    model = args.model_path

    # 2. base url
    base_url = args.url()

    # 3. timeout
    timeout = timeout

    # 4. api key
    api_key = args.api_key

    # 5. other args: convert to CLI style, excluding handled keys
    exclude_keys = {"model_path", "host", "port", "api_key"}
    other_args = []

    for field in dataclasses.fields(ServerArgs):
        key = field.name
        if key in exclude_keys:
            continue
        val = getattr(args, key)
        if isinstance(val, bool):
            if val:
                other_args.append(f"--{key.replace('_', '-')}")
        elif val is not None:
            if isinstance(val, list):
                for v in val:
                    other_args.extend([f"--{key.replace('_', '-')}", str(v)])
            else:
                other_args.extend([f"--{key.replace('_', '-')}", str(val)])

    return model, base_url, timeout, api_key, other_args


class HttpServerEngineAdapter:
    def __init__(self, server_args: ServerArgs):
        self.server_args = copy.deepcopy(server_args)
        print(f"launch_server_from_verl_engine {self.server_args.port}")

        model, base_url, timeout, api_key, other_args = server_args_to_launch_params(
            self.server_args
        )
        self.process = popen_launch_server(
            model=model,
            base_url=base_url,
            timeout=timeout,
            api_key=api_key,
            other_args=other_args + ["--enable-memory-saver"],
        )

    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]],
        load_format: Optional[str] = None,
        flush_cache: bool = False,
    ):

        print(f"update_weights_from_tensor of HttpServerEngineAdapter")
        return requests.post(
            f"http://localhost:{self.server_args.port}/update_weights_from_tensor",
            json={
                "serialized_named_tensors": [
                    serialize_for_http(
                        MultiprocessingSerializer.serialize(named_tensors)
                    )
                    for _ in range(self.server_args.tp_size)
                ],
                "load_format": load_format,
                "flush_cache": flush_cache,
            },
        )

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
        """Implements text generation functionality by forwarding the request to the veRL Engine via HTTP.

        This method packages all generation parameters into a JSON payload, filters out any None values,
        and sends the request to the locally running server. It then returns the parsed response or
        raises an exception if the generation fails.
        """
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

        response = requests.post(
            f"http://localhost:{self.server_args.port}/generate", json=payload
        )

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Generate request failed: {response.text}")

    def release_memory_occupation(self):
        """release memory occupation by HTTP"""
        response = requests.post(
            f"http://localhost:{self.server_args.port}/release_memory_occupation",
            json={},
        )
        if response.status_code != 200:
            raise Exception(f"Failed to release memory: {response.text}")
        return response

    def resume_memory_occupation(self):
        """resume memory occupation by HTTP"""
        response = requests.post(
            f"http://localhost:{self.server_args.port}/resume_memory_occupation",
            json={},
        )
        if response.status_code != 200:
            raise Exception(f"Failed to resume memory: {response.text}")
        return response
