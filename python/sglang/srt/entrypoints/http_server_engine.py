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

from sglang.srt.entrypoints.EngineBase import EngineBase
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import MultiprocessingSerializer, kill_process_tree


def launch_server_process(server_args: ServerArgs) -> multiprocessing.Process:

    p = multiprocessing.Process(target=launch_server, args=(server_args,))
    p.start()

    base_url = server_args.url()
    timeout = 300.0  # Increased timeout to 5 minutes for downloading large models
    start_time = time.perf_counter()

    with requests.Session() as session:
        while time.perf_counter() - start_time < timeout:
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
