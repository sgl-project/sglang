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
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import torch.distributed
import torch.nn.functional as F

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.eplb import eplb_algorithms
from sglang.srt.model_loader import get_model_architecture
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


@dataclass
class ExpertLocationMetadata:
    physical_to_logical_map: torch.Tensor  # (layers, num_physical_experts)
    physical_to_logical_map_cpu: torch.Tensor
    logical_to_all_physical_map: torch.Tensor  # (layers, num_logical_experts, X)
    logical_to_all_physical_map_num_valid: torch.Tensor  # (layers, num_logical_experts)
    # (layers, num_logical_experts)
    logical_to_rank_dispatch_physical_map: Optional[torch.Tensor]

    # -------------------------------- properties ------------------------------------

    @property
    def num_layers(self) -> int:
        return self.physical_to_logical_map.shape[0]

    @property
    def num_physical_experts(self) -> int:
        return self.physical_to_logical_map.shape[1]

    @property
    def num_local_physical_experts(self) -> int:
        ans, remainder = divmod(self.num_physical_experts, self.ep_size)
        assert remainder == 0
        return ans

    @property
    def num_logical_experts(self) -> int:
        return self.logical_to_all_physical_map.shape[1]

    @property
    def ep_size(self):
        # TODO change when EP size != world size
        return torch.distributed.get_world_size()

    def __post_init__(self):
        num_layers_0, num_physical_experts_0 = self.physical_to_logical_map.shape
        num_layers_1, num_logical_experts_0, num_physical_experts_1 = (
            self.logical_to_all_physical_map.shape
        )
        num_layers_2, num_logical_experts_1 = (
            self.logical_to_all_physical_map_num_valid.shape
        )
        assert num_layers_0 == num_layers_1 == num_layers_2
        assert num_logical_experts_0 == num_logical_experts_1
        assert num_physical_experts_0 == num_physical_experts_1

    # -------------------------------- construction ------------------------------------

    @staticmethod
    def init_trivial(server_args: ServerArgs, model_config: ModelConfig):
        """Trivial location - logical expert i corresponds to physical expert i"""
        common = ExpertLocationMetadata._init_common(server_args, model_config)
        num_physical_experts = common["num_physical_experts"]
        model_config_for_expert_location = common["model_config_for_expert_location"]
        num_layers = model_config_for_expert_location.num_layers
        num_logical_experts = model_config_for_expert_location.num_logical_experts

        physical_to_logical_map = (
            torch.arange(0, num_physical_experts).repeat(num_layers, 1)
            % num_logical_experts
        )

        return ExpertLocationMetadata.init_by_mapping(
            server_args,
            model_config,
            physical_to_logical_map=physical_to_logical_map,
        )

    @staticmethod
    def init_by_mapping(
        server_args: ServerArgs,
        model_config: ModelConfig,
        physical_to_logical_map,
    ):
        if not isinstance(physical_to_logical_map, torch.Tensor):
            physical_to_logical_map = torch.tensor(physical_to_logical_map)
        physical_to_logical_map = physical_to_logical_map.to(server_args.device)

        common = ExpertLocationMetadata._init_common(server_args, model_config)
        model_config_for_expert_location = common["model_config_for_expert_location"]
        logical_to_all_physical_map = _compute_logical_to_all_physical_map(
            physical_to_logical_map,
            num_logical_experts=model_config_for_expert_location.num_logical_experts,
        )

        return ExpertLocationMetadata._init_raw(
            server_args=server_args,
            ep_size=common["ep_size"],
            physical_to_logical_map=physical_to_logical_map,
            logical_to_all_physical_map=logical_to_all_physical_map,
        )

    @staticmethod
    def init_by_eplb(
        server_args: ServerArgs, model_config: ModelConfig, logical_count: torch.Tensor
    ):
        if not isinstance(logical_count, torch.Tensor):
            logical_count = torch.tensor(logical_count)
        if len(logical_count.shape) == 2:
            logical_count = logical_count.unsqueeze(0)
        logical_count = logical_count.to(server_args.device)

        common = ExpertLocationMetadata._init_common(server_args, model_config)
        model_config_for_expert_location = common["model_config_for_expert_location"]
        num_physical_experts = common["num_physical_experts"]
        num_groups = model_config_for_expert_location.num_groups
        num_nodes = server_args.nnodes

        physical_to_logical_map, logical_to_all_physical_map, expert_count = (
            eplb_algorithms.rebalance_experts(
                tokens_per_expert=logical_count,
                num_physical_experts=num_physical_experts,
                num_local_physical_experts=num_physical_experts // common["ep_size"],
                num_groups=num_groups,
                num_nodes=num_nodes,
                algorithm=eplb_algorithms.compute_algorithm(
                    raw_algorithm=server_args.eplb_algorithm,
                    num_groups=num_groups,
                    num_nodes=num_nodes,
                ),
            )
        )

        return ExpertLocationMetadata._init_raw(
            server_args=server_args,
            ep_size=common["ep_size"],
            physical_to_logical_map=physical_to_logical_map.to(server_args.device),
            logical_to_all_physical_map=logical_to_all_physical_map.to(
                server_args.device
            ),
        )

    @staticmethod
    def _init_common(server_args: ServerArgs, model_config: ModelConfig):
        model_config_for_expert_location = (
            ModelConfigForExpertLocation.from_model_config(model_config)
        )

        num_physical_experts = (
            model_config_for_expert_location.num_logical_experts
            + server_args.ep_num_redundant_experts
        )
        ep_size = server_args.ep_size
        assert num_physical_experts % ep_size == 0
        num_local_physical_experts = num_physical_experts // ep_size

        return dict(
            model_config_for_expert_location=model_config_for_expert_location,
            num_physical_experts=num_physical_experts,
            num_local_physical_experts=num_local_physical_experts,
            ep_size=ep_size,
        )

    @staticmethod
    def _init_raw(
        server_args: ServerArgs,
        ep_size: int,
        physical_to_logical_map: torch.Tensor,
        logical_to_all_physical_map: torch.Tensor,
    ):
        _, num_physical_experts = physical_to_logical_map.shape

        logical_to_all_physical_map_padded = F.pad(
            logical_to_all_physical_map,
            (0, num_physical_experts - logical_to_all_physical_map.shape[-1]),
            value=-1,
        )

        logical_to_all_physical_map_num_valid = torch.count_nonzero(
            logical_to_all_physical_map != -1, dim=-1
        )

        return ExpertLocationMetadata(
            physical_to_logical_map=physical_to_logical_map,
            physical_to_logical_map_cpu=physical_to_logical_map.cpu(),
            logical_to_all_physical_map=logical_to_all_physical_map_padded,
            logical_to_all_physical_map_num_valid=logical_to_all_physical_map_num_valid,
            logical_to_rank_dispatch_physical_map=(
                compute_logical_to_rank_dispatch_physical_map(
                    logical_to_all_physical_map=logical_to_all_physical_map,
                    num_gpus=ep_size,
                    num_physical_experts=num_physical_experts,
                    # TODO improve when we have real EP rank
                    ep_rank=torch.distributed.get_rank() % ep_size,
                )
                if server_args.ep_dispatch_algorithm == "static"
                else None
            ),
        )

    # -------------------------------- mutation ------------------------------------

    def update(
        self,
        other: "ExpertLocationMetadata",
        update_layer_ids: List[int],
    ):
        for field in [
            "ep_size",
        ]:
            assert getattr(self, field) == getattr(other, field)

        for field in [
            "physical_to_logical_map",
            "physical_to_logical_map_cpu",
            "logical_to_all_physical_map",
            "logical_to_all_physical_map_num_valid",
            "logical_to_rank_dispatch_physical_map",
        ]:
            other_field = getattr(other, field)
            self_field = getattr(self, field)
            assert (other_field is not None) == (self_field is not None)
            if self_field is not None:
                mask_update = torch.tensor(
                    [i in update_layer_ids for i in range(self.num_layers)]
                )
                mask_update = mask_update.view(*([-1] + [1] * (self_field.dim() - 1)))
                mask_update = mask_update.to(self_field.device, non_blocking=True)
                self_field[...] = torch.where(mask_update, other_field, self_field)

    # -------------------------------- usage ------------------------------------

    def logical_to_all_physical(
        self, layer_id: int, logical_expert_id: int
    ) -> List[int]:
        return [
            physical_expert_id
            for physical_expert_id in self.logical_to_all_physical_map[
                layer_id, logical_expert_id
            ].tolist()
            if physical_expert_id != -1
        ]


_global_expert_location_metadata: Optional[ExpertLocationMetadata] = None


def get_global_expert_location_metadata():
    return _global_expert_location_metadata


def set_global_expert_location_metadata(value):
    global _global_expert_location_metadata
    assert _global_expert_location_metadata is None
    _global_expert_location_metadata = value


def _compute_logical_to_all_physical_map(
    physical_to_logical_map: torch.Tensor, num_logical_experts: int
):
    # This is rarely called, so we use for loops for maximum clarity

    num_layers, num_physical_experts = physical_to_logical_map.shape

    logical_to_all_physical_map = [
        [[] for _ in range(num_logical_experts)] for _ in range(num_layers)
    ]
    for layer_id in range(num_layers):
        for physical_expert_id in range(num_physical_experts):
            logical_expert_id = physical_to_logical_map[
                layer_id, physical_expert_id
            ].item()
            logical_to_all_physical_map[layer_id][logical_expert_id].append(
                physical_expert_id
            )

    logical_to_all_physical_map = _pad_nested_array(
        logical_to_all_physical_map, pad_value=-1
    )

    return torch.tensor(
        logical_to_all_physical_map, device=physical_to_logical_map.device
    )


def _pad_nested_array(arr, pad_value):
    max_len = max(len(inner) for outer in arr for inner in outer)
    padded = [
        [inner + [pad_value] * (max_len - len(inner)) for inner in outer]
        for outer in arr
    ]
    return padded


# TODO optimize performance (rewrite and/or run in separate process with overlap)
def compute_logical_to_rank_dispatch_physical_map(
    logical_to_all_physical_map: torch.Tensor,
    num_gpus: int,
    num_physical_experts: int,
    ep_rank: int,
    seed: int = 42,
):
    r = random.Random(seed)

    num_local_physical_experts = num_physical_experts // num_gpus
    num_layers, num_logical_experts, _ = logical_to_all_physical_map.shape
    dtype = logical_to_all_physical_map.dtype

    logical_to_rank_dispatch_physical_map = torch.full(
        size=(num_gpus, num_layers, num_logical_experts),
        fill_value=-1,
        dtype=dtype,
    )

    for layer_id in range(num_layers):
        for logical_expert_id in range(num_logical_experts):
            candidate_physical_expert_ids = _logical_to_all_physical_raw(
                logical_to_all_physical_map, layer_id, logical_expert_id
            )
            output_partial = logical_to_rank_dispatch_physical_map[
                :, layer_id, logical_expert_id
            ]

            for gpu_id in range(num_gpus):
                same_gpu_physical_expert_ids = [
                    physical_expert_id
                    for physical_expert_id in candidate_physical_expert_ids
                    if _compute_gpu_id_of_physical_expert(
                        physical_expert_id, num_local_physical_experts
                    )
                    == gpu_id
                ]
                if len(same_gpu_physical_expert_ids) > 0:
                    output_partial[gpu_id] = same_gpu_physical_expert_ids[0]

            num_remain = torch.sum(output_partial == -1).item()
            output_partial[output_partial == -1] = torch.tensor(
                _fair_choices(candidate_physical_expert_ids, k=num_remain, r=r),
                dtype=dtype,
            )

    assert torch.all(logical_to_rank_dispatch_physical_map != -1)

    device = logical_to_all_physical_map.device
    return logical_to_rank_dispatch_physical_map[ep_rank, :, :].to(device)


def _logical_to_all_physical_raw(
    logical_to_all_physical_map, layer_id: int, logical_expert_id: int
) -> List[int]:
    return [
        physical_expert_id
        for physical_expert_id in logical_to_all_physical_map[
            layer_id, logical_expert_id
        ].tolist()
        if physical_expert_id != -1
    ]


def _compute_gpu_id_of_physical_expert(
    physical_expert_id: int, num_local_physical_experts: int
) -> int:
    return physical_expert_id // num_local_physical_experts


def _fair_choices(arr: List, k: int, r: random.Random) -> List:
    quotient, remainder = divmod(k, len(arr))
    ans = arr * quotient + r.sample(arr, k=remainder)
    r.shuffle(ans)
    return ans


@dataclass
class ModelConfigForExpertLocation:
    num_layers: int
    num_logical_experts: int
    num_groups: Optional[int] = None

    @staticmethod
    def init_dummy():
        return ModelConfigForExpertLocation(num_layers=1, num_logical_experts=1)

    @staticmethod
    def from_model_config(model_config: ModelConfig):
        model_class, _ = get_model_architecture(model_config)
        if hasattr(model_class, "get_model_config_for_expert_location"):
            return model_class.get_model_config_for_expert_location(
                model_config.hf_config
            )
        else:
            return ModelConfigForExpertLocation.init_dummy()


def compute_initial_expert_location_metadata(
    server_args: ServerArgs, model_config: ModelConfig
) -> ExpertLocationMetadata:
    data = server_args.init_expert_location
    if data == "trivial":
        return ExpertLocationMetadata.init_trivial(server_args, model_config)

    # TODO unify with the utils function
    if data.endswith(".pt"):
        data_dict = torch.load(data, weights_only=True)
    elif data.endswith(".json"):
        data_dict = json.loads(Path(data).read_text())
    else:
        data_dict = json.loads(data)

    if "physical_to_logical_map" in data_dict:
        logger.info(
            "init_expert_location from init_by_mapping using ServerArgs.init_expert_location"
        )
        return ExpertLocationMetadata.init_by_mapping(
            server_args, model_config, **data_dict
        )
    elif "logical_count" in data_dict:
        logger.info(
            "init_expert_location from init_by_eplb using ServerArgs.init_expert_location"
        )
        return ExpertLocationMetadata.init_by_eplb(
            server_args, model_config, logical_count=data_dict["logical_count"]
        )
    else:
        raise NotImplementedError(
            f"Unknown init_expert_location format ({list(data_dict.keys())=})"
        )
