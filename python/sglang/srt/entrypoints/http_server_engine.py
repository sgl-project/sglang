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


def server_args_to_launch_params(args: ServerArgs, timeout: float = 60.0):
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
        self.server_args.port = 2157
        print(f"launch_server_from_verl_engine {self.server_args.port}")

        # server_thread = threading.Thread(
        #     target=launch_server,
        #     args=(self.server_args,),
        #     daemon=True,
        # )
        model, base_url, timeout, api_key, other_args = server_args_to_launch_params(
            self.server_args
        )
        self.process = popen_launch_server(
            model=model,
            base_url=base_url,
            timeout=timeout,
            api_key=api_key,
            other_args=other_args,
        )

    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]],
        load_format: Optional[str] = None,
        flush_cache: bool = False,
    ):

        # obj = UpdateWeightsFromTensorReqInput(
        #     serialized_named_tensors=[
        #         MultiprocessingSerializer.serialize(named_tensors)
        #         for _ in range(self.server_args.tp_size)
        #     ],
        #     load_format=load_format,
        #     flush_cache=flush_cache,
        # )

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
