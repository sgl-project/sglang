# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union

import torch
import torch.distributed as dist
from PIL.Image import Image
from torch.distributed.tensor import DeviceMesh, DTensor

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.entrypoints.http_server_engine import HttpServerEngineAdapter
from sglang.srt.model_executor.model_runner import LocalSerializedTensor
from sglang.srt.patch_torch import monkey_patch_torch_reductions
from sglang.srt.utils import MultiprocessingSerializer, broadcast_pyobj


class VerlEngine:
    def __init__(
        self,
        device_mesh_cpu: DeviceMesh,
        nnodes: int = 1,
        backend: Literal["engine", "server"] = "engine",
        **kwargs,
    ):
        monkey_patch_torch_reductions()
        self._device_mesh_cpu = device_mesh_cpu
        self._tp_rank = device_mesh_cpu.get_local_rank()
        self._tp_size = device_mesh_cpu.size()
        tp_size_per_node = self._tp_size // nnodes
        node_rank = self._tp_rank // tp_size_per_node
        first_rank_in_node = self._tp_rank % tp_size_per_node == 0

        # Common engine keyword arguments
        engine_kwargs = dict(
            **kwargs, tp_size=self._tp_size, node_rank=node_rank, nnodes=nnodes
        )

        if backend == "engine":
            if first_rank_in_node:
                os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
                self._engine = Engine(**engine_kwargs)
            else:
                self._engine = None

        elif backend == "server":
            if self._tp_rank == 0:
                self._engine = HttpServerEngineAdapter(**engine_kwargs)
            else:
                self._engine = None
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        dist.barrier(group=self._device_mesh_cpu.get_group())

    def generate(
        self,
        # The input prompt. It can be a single prompt or a batch of prompts.
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,
        # The token ids for text; one can either specify text or input_ids.
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        # The image input. It can be an image instance, file name, URL, or base64 encoded string.
        # Can be formatted as:
        # - Single image for a single request
        # - List of images (one per request in a batch)
        # - List of lists of images (multiple images per request)
        # See also python/sglang/srt/utils.py:load_image for more details.
        image_data: Optional[
            Union[
                List[List[Union[Image, str]]],
                List[Union[Image, str]],
                Union[Image, str],
            ]
        ] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
        lora_path: Optional[List[Optional[str]]] = None,
        custom_logit_processor: Optional[Union[List[str], str]] = None,
    ) -> Dict:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::GenerateReqInput`.
        Please refer to `GenerateReqInput` for the documentation.
        """
        if self._tp_rank == 0:
            output = self._engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                input_ids=input_ids,
                image_data=image_data,
                return_logprob=return_logprob,
                logprob_start_len=logprob_start_len,
                top_logprobs_num=top_logprobs_num,
                token_ids_logprob=token_ids_logprob,
                lora_path=lora_path,
                custom_logit_processor=custom_logit_processor,
            )
        else:
            output = None

        # Most naive implementation, can extract tensor and send via gloo if too slow
        [output] = broadcast_pyobj(
            data=[output],
            rank=self._tp_rank,
            dist_group=self._device_mesh_cpu.get_group(),
            src=self._device_mesh_cpu.mesh[0].item(),
            force_cpu_device=False,
        )

        return output

    def update_weights_from_tensor(
        self,
        named_tensors: Iterable[Tuple[str, torch.Tensor]],
        load_format: Optional[str] = None,
    ):
        # Most naive implementation, can optimize a lot if it is bottleneck
        for tensor_index, (name, tensor) in enumerate(named_tensors):
            serialized_tensor = MultiprocessingSerializer.serialize(
                _preprocess_tensor_for_update_weights(tensor)
            )

            if self._tp_rank == 0:
                gathered_serialized_tensors = [None for _ in range(self._tp_size)]
            else:
                gathered_serialized_tensors = None
            dist.gather_object(
                obj=serialized_tensor,
                object_gather_list=gathered_serialized_tensors,
                dst=self._device_mesh_cpu.mesh.tolist()[0],
                group=self._device_mesh_cpu.get_group(),
            )

            if self._tp_rank == 0:
                self._engine.update_weights_from_tensor(
                    named_tensors=[
                        (
                            name,
                            LocalSerializedTensor(values=gathered_serialized_tensors),
                        )
                    ],
                    load_format=load_format,
                    flush_cache=False,
                )

        if self._tp_rank == 0:
            self._engine.tokenizer_manager.flush_cache()

    def release_memory_occupation(self):
        if self._tp_rank == 0:
            self._engine.release_memory_occupation()

    def resume_memory_occupation(self):
        if self._tp_rank == 0:
            self._engine.resume_memory_occupation()

    def shutdown(self):
        if self._engine is not None:
            self._engine.shutdown()


def _preprocess_tensor_for_update_weights(tensor: torch.Tensor):
    if isinstance(tensor, DTensor):
        return tensor.full_tensor()
    return tensor
