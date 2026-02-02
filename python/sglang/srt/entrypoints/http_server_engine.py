import multiprocessing
import time
from typing import List, Optional, Tuple

import requests
import torch

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


class HttpServerEngineAdapter(EngineBase):
    """
    You can use this class to launch a server from a VerlEngine instance.
    We recommend using this class only you need to use http server.
    Otherwise, you can use Engine directly.
    """

    def __init__(self, **kwargs):
        self.server_args = ServerArgs(**kwargs)
        print(
            f"Launch HttpServerEngineAdapter at: {self.server_args.host}:{self.server_args.port}"
        )
        self.process = launch_server_process(self.server_args)

    def _make_request(self, endpoint: str, payload: Optional[dict] = None):
        """Make a POST request to the specified endpoint with the given payload.
        Args:
            endpoint: The API endpoint to call
            payload: The JSON payload to send (default: empty dict)
        Returns:
            The JSON response from the server
        """
        url = f"http://{self.server_args.host}:{self.server_args.port}/{endpoint}"
        response = requests.post(url, json=payload or {})
        response.raise_for_status()
        return response.json()

    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]],
        load_format: Optional[str] = None,
        flush_cache: bool = False,
    ):
        """
        Update model weights from tensor data. The HTTP server will only post meta data, and the real weights will be copied directly from GPUs.
        Note: The model should be on GPUs rather than CPU for this functionality to work properly.
        If you encounter issues, ensure your model is loaded on GPU devices rather than CPU.
        """

        return self._make_request(
            "update_weights_from_tensor",
            {
                "serialized_named_tensors": [
                    MultiprocessingSerializer.serialize(named_tensors, output_str=True)
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

        return self._make_request("generate", payload)

    def release_memory_occupation(self):
        return self._make_request("release_memory_occupation")

    def resume_memory_occupation(self):
        return self._make_request("resume_memory_occupation")

    def flush_cache(self):
        return self._make_request("flush_cache")
