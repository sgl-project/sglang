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
from typing import Optional, Union, List, Dict, Tuple

import torch
import torch.distributed as dist
from sglang.srt.server import Engine
from sglang.srt.utils import broadcast_pyobj, MultiprocessingSerializer
from torch.distributed.tensor import DeviceMesh


class VerlEngine:
    def __init__(
            self,
            first_rank_in_node: bool,
            device_mesh_cpu: DeviceMesh,
            **kwargs,
    ):
        self._device_mesh_cpu = device_mesh_cpu
        self._tp_rank = device_mesh_cpu.get_local_rank()
        self._tp_size = device_mesh_cpu.size()

        if first_rank_in_node:
            self._engine = Engine(**kwargs, tp_rank=self._tp_rank, tp_size=self._tp_size)
        else:
            self._engine = None

    def generate(
            self,
            # The input prompt. It can be a single prompt or a batch of prompts.
            prompt: Optional[Union[List[str], str]] = None,
            sampling_params: Optional[Union[List[Dict], Dict]] = None,
            # The token ids for text; one can either specify text or input_ids.
            input_ids: Optional[Union[List[List[int]], List[int]]] = None,
            # The image input. It can be a file name, a url, or base64 encoded string.
            # See also python/sglang/srt/utils.py:load_image.
            image_data: Optional[Union[List[str], str]] = None,
            return_logprob: Optional[Union[List[bool], bool]] = False,
            logprob_start_len: Optional[Union[List[int], int]] = None,
            top_logprobs_num: Optional[Union[List[int], int]] = None,
            lora_path: Optional[List[Optional[str]]] = None,
            custom_logit_processor: Optional[Union[List[str], str]] = None,
    ) -> Dict:
        if self._tp_rank == 0:
            output = self._engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                input_ids=input_ids,
                image_data=image_data,
                return_logprob=return_logprob,
                logprob_start_len=logprob_start_len,
                top_logprobs_num=top_logprobs_num,
                lora_path=lora_path,
                custom_logit_processor=custom_logit_processor,
            )
        else:
            output = None

        # Most naive implementation, can extract tensor and send via gloo if too slow
        # TODO improve the args
        ranks = self._device_mesh_cpu.mesh.tolist()
        [output] = broadcast_pyobj(
            data=[output],
            rank=ranks[self._tp_rank],
            dist_group=self._device_mesh_cpu.get_group(),
            src=ranks[0],
        )

        return output

    def update_weights_from_tensor(
            self,
            named_tensors: List[Tuple[str, torch.Tensor]],
            load_format: Optional[str] = None,
    ):
        for name, tensor in named_tensors:
            if self._tp_rank == 0:
                object_gather_list = [None for _ in range(self._tp_size)]
            else:
                object_gather_list = None
            dist.gather_object(
                obj=MultiprocessingSerializer.serialize(tensor),
                object_gather_list=object_gather_list,
                dst=self._device_mesh_cpu.mesh.tolist()[0],
                group=self._device_mesh_cpu.get_group(),
            )

            if self._tp_rank == 0:
                self._engine.update_weights_from_tensor(
                    named_tensors=[(name, TODO)],
                    load_format=load_format,
                )

    def release_memory_occupation(self):
        if self._tp_rank == 0:
            self._engine.release_memory_occupation()

    def resume_memory_occupation(self):
        if self._tp_rank == 0:
            self._engine.resume_memory_occupation()
