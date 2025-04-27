import dataclasses
import json
import random
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers import deepseek_eplb
from sglang.srt.model_loader import get_model_architecture
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_bool_env_var


@dataclass
class ExpertLocationMetadata:
    physical_to_logical_map: torch.Tensor  # (layers, num_physical_experts)
    logical_to_all_physical_map: torch.Tensor  # (layers, num_logical_experts, X)
    logical_to_all_physical_map_num_valid: torch.Tensor  # (layers, num_logical_experts)
    # (num_gpus, layers, num_logical_experts)
    logical_to_rank_dispatch_physical_map: torch.Tensor

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
        return self.logical_to_rank_dispatch_physical_map.shape[0]

    def __post_init__(self):
        num_layers_0, num_physical_experts_0 = self.physical_to_logical_map.shape
        num_layers_1, num_logical_experts_0, num_physical_experts_1 = (
            self.logical_to_all_physical_map.shape
        )
        num_layers_2, num_logical_experts_1 = (
            self.logical_to_all_physical_map_num_valid.shape
        )
        ep_size_0, num_layers_3, num_logical_experts_2 = (
            self.logical_to_rank_dispatch_physical_map.shape
        )
        assert num_layers_0 == num_layers_1 == num_layers_2 == num_layers_3
        assert num_logical_experts_0 == num_logical_experts_1 == num_logical_experts_2
        assert num_physical_experts_0 == num_physical_experts_1

    # -------------------------------- construction ------------------------------------

    @staticmethod
    def init_trivial(server_args: ServerArgs):
        """Trivial location - logical expert i corresponds to physical expert i"""
        common = ExpertLocationMetadata._init_common(server_args)
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
            physical_to_logical_map=physical_to_logical_map,
        )

    # TODO hack
    @staticmethod
    def init_padding(server_args: ServerArgs):
        common = ExpertLocationMetadata._init_common(server_args)
        num_physical_experts = common["num_physical_experts"]
        model_config_for_expert_location = common["model_config_for_expert_location"]
        ep_size = common["ep_size"]
        num_layers = model_config_for_expert_location.num_layers
        num_logical_experts = model_config_for_expert_location.num_logical_experts

        num_redundant_experts = num_physical_experts - num_logical_experts
        num_physical_experts_per_gpu = num_physical_experts // ep_size

        num_padded_gpus = num_redundant_experts
        num_non_padded_gpus = ep_size - num_redundant_experts

        phy2log_non_padded = torch.arange(
            num_physical_experts_per_gpu * num_non_padded_gpus
        )

        phy2log_padded_start = num_physical_experts_per_gpu * num_non_padded_gpus
        phy2log_padded = torch.cat(
            [
                phy2log_padded_start
                + torch.arange(
                    (num_physical_experts_per_gpu - 1) * num_padded_gpus
                ).reshape(-1, (num_physical_experts_per_gpu - 1)),
                torch.arange(num_padded_gpus)[:, None],
            ],
            dim=-1,
        )

        physical_to_logical_map_one_layer = torch.cat(
            [phy2log_non_padded, phy2log_padded.flatten()]
        )
        print(f"{phy2log_non_padded=}")
        print(f"{phy2log_padded=}")
        print(f"{physical_to_logical_map_one_layer=}")

        physical_to_logical_map = torch.tensor(
            physical_to_logical_map_one_layer, dtype=torch.int
        ).repeat(num_layers, 1)
        print(f"hi init_padding {physical_to_logical_map=}")

        return ExpertLocationMetadata.init_by_mapping(
            server_args,
            physical_to_logical_map=physical_to_logical_map,
            hack_logical_to_all_physical_map_pick_first_only=True,
        )

    @staticmethod
    def init_by_mapping(
        server_args: ServerArgs,
        physical_to_logical_map,
        hack_logical_to_all_physical_map_pick_first_only=False,
    ):
        if not isinstance(physical_to_logical_map, torch.Tensor):
            physical_to_logical_map = torch.tensor(physical_to_logical_map)

        common = ExpertLocationMetadata._init_common(server_args)
        model_config_for_expert_location = common["model_config_for_expert_location"]
        logical_to_all_physical_map = _compute_logical_to_all_physical_map(
            physical_to_logical_map,
            num_logical_experts=model_config_for_expert_location.num_logical_experts,
        )

        if hack_logical_to_all_physical_map_pick_first_only:
            logical_to_all_physical_map = logical_to_all_physical_map[:, :, :1]
            print(
                f"hack since hack_logical_to_all_physical_map_pick_first_only! {logical_to_all_physical_map.tolist()=}"
            )

        return ExpertLocationMetadata._init_raw(
            ep_size=common["ep_size"],
            physical_to_logical_map=physical_to_logical_map,
            logical_to_all_physical_map=logical_to_all_physical_map,
        )

    @staticmethod
    def init_by_eplb(server_args: ServerArgs, logical_count: torch.Tensor):
        if not isinstance(logical_count, torch.Tensor):
            logical_count = torch.tensor(logical_count)
        common = ExpertLocationMetadata._init_common(server_args)
        model_config_for_expert_location = common["model_config_for_expert_location"]

        physical_to_logical_map, logical_to_all_physical_map, expert_count = (
            deepseek_eplb.rebalance_experts(
                weight=logical_count,
                num_replicas=common["num_physical_experts"],
                num_groups=model_config_for_expert_location.num_groups,
                num_nodes=server_args.nnodes,
                num_gpus=common["ep_size"],
                hack_shuffle=server_args.deepseek_eplb_hack_shuffle,
            )
        )

        return ExpertLocationMetadata._init_raw(
            ep_size=common["ep_size"],
            physical_to_logical_map=physical_to_logical_map,
            logical_to_all_physical_map=logical_to_all_physical_map,
        )

    @staticmethod
    def _init_common(server_args: ServerArgs):
        model_config = ModelConfig.from_server_args(server_args)
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
            logical_to_all_physical_map=logical_to_all_physical_map_padded,
            logical_to_all_physical_map_num_valid=logical_to_all_physical_map_num_valid,
            logical_to_rank_dispatch_physical_map=compute_logical_to_rank_dispatch_physical_map(
                logical_to_all_physical_map,
                num_gpus=ep_size,
                num_physical_experts=num_physical_experts,
            ),
        )

    # -------------------------------- mutation ------------------------------------

    def update(
        self,
        other: "ExpertLocationMetadata",
        layer_id_start: Optional[int] = None,
        layer_id_len: Optional[int] = None,
    ):
        for field in [
            "ep_size",
        ]:
            assert getattr(self, field) == getattr(other, field)

        for field, layer_id_dim in [
            ("physical_to_logical_map", 0),
            ("logical_to_all_physical_map", 0),
            ("logical_to_all_physical_map_num_valid", 0),
            ("logical_to_rank_dispatch_physical_map", 1),
        ]:

            def _get(obj):
                ans = getattr(obj, field)
                if (layer_id_start is not None) or (layer_id_len is not None):
                    ans = ans.narrow(
                        dim=layer_id_dim, start=layer_id_start, length=layer_id_len
                    )
                return ans

            # Cannot update address to avoid breaking CUDA graph
            dst = _get(self)
            dst[...] = _get(other)

    def to(self, device):
        for field in [
            "logical_to_all_physical_map",
            "logical_to_all_physical_map_num_valid",
            "logical_to_rank_dispatch_physical_map",
        ]:
            setattr(self, field, getattr(self, field).to(device))

    # -------------------------------- usage ------------------------------------

    def local_physical_to_physical(self, rank: int, local_physical_expert_index: int):
        return self.num_local_physical_experts * rank + local_physical_expert_index

    def logical_to_all_physical(
        self, layer_id: int, logical_expert_id: int
    ) -> List[int]:
        return self.logical_to_all_physical_raw(
            self.logical_to_all_physical_map, layer_id, logical_expert_id
        )

    @staticmethod
    def logical_to_all_physical_raw(
        logical_to_all_physical_map, layer_id: int, logical_expert_id: int
    ) -> List[int]:
        return [
            physical_expert_id
            for physical_expert_id in logical_to_all_physical_map[
                layer_id, logical_expert_id
            ].tolist()
            if physical_expert_id != -1
        ]

    def debug_str(self):
        return json.dumps(
            {
                k: v.tolist() if isinstance(v, torch.Tensor) else v
                for k, v in dataclasses.asdict(self).items()
            }
        )


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

    return torch.tensor(logical_to_all_physical_map)


def _pad_nested_array(arr, pad_value):
    max_len = max(len(inner) for outer in arr for inner in outer)
    padded = [
        [inner + [pad_value] * (max_len - len(inner)) for inner in outer]
        for outer in arr
    ]
    return padded


# This is rarely called, so we use for loops for maximum clarity
def compute_logical_to_rank_dispatch_physical_map(
    logical_to_all_physical_map: torch.Tensor,
    num_gpus: int,
    num_physical_experts: int,
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
            candidate_physical_expert_ids = (
                ExpertLocationMetadata.logical_to_all_physical_raw(
                    logical_to_all_physical_map, layer_id, logical_expert_id
                )
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
    return logical_to_rank_dispatch_physical_map


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
