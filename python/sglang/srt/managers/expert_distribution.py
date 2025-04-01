import logging
from abc import ABC
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Type, Any, Optional

import torch
from sglang.srt.configs.deepseekvl2 import DeepseekV2Config
from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import Withable, get_bool_env_var

logger = logging.getLogger(__name__)


# --------------------------------------- Entrypoint -----------------------------------------

class ExpertDistributionRecorder:
    """Global expert distribution recording"""

    def __init__(self, server_args: ServerArgs, metadata: "ModelExpertMetadata", rank: int):
        self._recording = False
        self._current_layer_idx = Withable()
        self._metadata = metadata
        self._accumulator = _Accumulator.init_new(metadata, rank)
        self._single_pass_gatherers = {
            k: _SinglePassGatherer.init_new(server_args, metadata)
            for k in self._accumulator.get_single_pass_gatherer_keys()
        }

    def with_current_layer(self, layer_idx):
        return self._current_layer_idx.with_value(layer_idx)

    @contextmanager
    def with_forward_pass(self, forward_pass_id: int):
        try:
            yield
        finally:
            self._on_forward_pass_end(forward_pass_id)

    def _on_forward_pass_end(self, forward_pass_id: int):
        for gatherer_key, gatherer in self._single_pass_gatherers.items():
            single_pass_physical_count = gatherer.collect()
            self._accumulator.append(forward_pass_id, gatherer_key, single_pass_physical_count)
            gatherer.reset()

    def on_select_experts(self, topk_ids: torch.Tensor):
        if not self._recording:
            return
        gatherer = self._single_pass_gatherers[self._accumulator.get_single_pass_gatherer_key()]
        gatherer.on_select_experts(layer_idx=self._current_layer_idx.value, topk_ids=topk_ids)

    def on_deepep_dispatch_normal(self, num_recv_tokens_per_expert_list: List[int]):
        if not self._recording:
            return
        gatherer = self._single_pass_gatherers[self._accumulator.get_single_pass_gatherer_key()]
        gatherer.on_deepep_dispatch_normal(self._current_layer_idx.value, num_recv_tokens_per_expert_list)

    def _reset(self):
        """Reset the expert distribution recorder."""
        logger.info("Resetting ExpertDistributionRecorder...")
        self._recording = False
        assert self._current_layer_idx.value is None
        for gatherer in self._single_pass_gatherers.values():
            gatherer.reset()
        self._accumulator.reset()

    def start_record(self):
        """Start recording the expert distribution."""
        if self._recording:
            logger.warning(
                "SGLang server is already recording expert ids. Did you forget to dump the expert ids recorded so far by sending requests to the `/stop_expert_distribution_record` and `/dump_expert_distribution_record` endpoints?"
            )
        self._reset()
        self._recording = True

    def stop_record(self):
        """Stop recording the expert distribution."""
        if not self._recording:
            logger.warning(
                "SGLang server has not been recording expert ids. Did you forget to start recording by sending request to the `/start_expert_distribution_record` endpoint?"
            )
        self._recording = False

    def dump_record(self):
        """Dump the expert distribution record and reset the recorder after dumping."""
        output = self._accumulator.dump()
        self._reset()
        return output


# Put global args for easy access, just like `global_server_args_dict`
global_expert_distribution_recorder: Optional[ExpertDistributionRecorder] = None


def postprocess_dumps(physical_dumps: List[Any], physical_to_logical_map: torch.Tensor,
                      metadata: "ModelExpertMetadata"):
    return _Accumulator.get_class().postprocess_dumps(physical_dumps, physical_to_logical_map)


# --------------------------------------- SinglePassGatherer -----------------------------------------


class _SinglePassGatherer(ABC):
    @staticmethod
    def init_new(server_args: ServerArgs, metadata: "ModelExpertMetadata") -> "_SinglePassGatherer":
        if server_args.enable_deepep_moe:
            # TODO DeepEP low latency
            return _DeepepNormalSinglePassGatherer(metadata)
        return _LayerBasedSinglePassGatherer(metadata)

    def __init__(self, metadata: "ModelExpertMetadata"):
        self._metadata = metadata

    def on_select_experts(self, layer_idx: int, topk_ids: torch.Tensor):
        pass

    def on_deepep_dispatch_normal(self, layer_idx: int, num_recv_tokens_per_expert_list: List[int]):
        pass

    def reset(self):
        raise NotImplementedError

    def collect(self) -> torch.Tensor:
        raise NotImplementedError


class _LayerBasedSinglePassGatherer(_SinglePassGatherer):
    def __init__(self, metadata: "ModelExpertMetadata"):
        super().__init__(metadata)
        self._num_recv_tokens_per_expert_list_of_layer = {}

    def _on_layer_data(self, layer_idx: int, num_recv_tokens_per_expert_list: List[int]):
        # TODO for TBO, we may need to relax this restriction
        assert layer_idx not in self._num_recv_tokens_per_expert_list_of_layer
        assert 0 <= layer_idx < self._metadata.num_layers
        self._num_recv_tokens_per_expert_list_of_layer[layer_idx] = num_recv_tokens_per_expert_list

    def reset(self):
        self._num_recv_tokens_per_expert_list_of_layer.clear()

    def collect(self) -> torch.Tensor:
        data = [
            self._num_recv_tokens_per_expert_list_of_layer.get(layer_index) or (
                [0] * self._metadata.num_local_physical_experts)
            for layer_index in range(self._metadata.num_layers)
        ]
        return torch.tensor(data)


class _SelectExpertsSinglePassGatherer(_LayerBasedSinglePassGatherer):
    # pretty slow, but we will use the DeepEP Gatherer in production
    def on_select_experts(self, layer_idx: int, topk_ids: torch.Tensor):
        topk_ids_list = topk_ids.to("cpu", non_blocking=True).numpy().tolist()
        torch.cuda.synchronize()

        num_recv_tokens_per_expert_list = [0] * self._metadata.num_local_physical_experts
        for token_record in topk_ids_list:
            for expert_idx in token_record:
                num_recv_tokens_per_expert_list[expert_idx] += 1

        self._on_layer_data(layer_idx, num_recv_tokens_per_expert_list)


class _DeepepNormalSinglePassGatherer(_LayerBasedSinglePassGatherer):
    def on_deepep_dispatch_normal(self, layer_idx: int, num_recv_tokens_per_expert_list: List[int]):
        assert isinstance(num_recv_tokens_per_expert_list, list)
        self._on_layer_data(layer_idx, num_recv_tokens_per_expert_list)


# TODO Wait for LowLatency DeepEP merging
# e.g. use naive tensor copying
# need to consider CUDA graph, e.g. add initialization and after-end
class _DeepepLowLatencySinglePassGatherer(_SinglePassGatherer):
    pass


# --------------------------------------- Accumulator -----------------------------------------

_SINGLE_PASS_GATHERER_KEY_PRIMARY = "primary"


class _Accumulator(ABC):
    @staticmethod
    def init_new(metadata: "ModelExpertMetadata", rank: int) -> "_Accumulator":
        return _Accumulator.get_class()(metadata, rank)

    @staticmethod
    def get_class() -> Type["_Accumulator"]:
        if get_bool_env_var("SGLANG_EXPERT_DISTRIBUTION_RECORDER_DETAIL"):
            return _DetailAccumulator
        return _StatAccumulator

    def __init__(self, metadata: "ModelExpertMetadata", rank: int):
        self._metadata = metadata
        self._rank = rank

    def get_single_pass_gatherer_keys(self):
        return [_SINGLE_PASS_GATHERER_KEY_PRIMARY]

    def get_single_pass_gatherer_key(self, debug_name: str):
        return _SINGLE_PASS_GATHERER_KEY_PRIMARY

    @classmethod
    def postprocess_dumps(cls, physical_dumps: List[Any], physical_to_logical_map: torch.Tensor,
                          metadata: "ModelExpertMetadata"):
        raise NotImplementedError

    def append(self, forward_pass_id: int, gatherer_key: str, single_pass_physical_count: torch.Tensor):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def dump(self):
        raise NotImplementedError


class _DetailAccumulator(_Accumulator):
    @classmethod
    def postprocess_dumps(cls, physical_dumps: List[Any], physical_to_logical_map: torch.Tensor,
                          metadata: "ModelExpertMetadata"):
        # Do not convert to logical since we want all details
        return [
            record
            for physical_dump in physical_dumps
            for record in physical_dump
        ]

    def __init__(self, metadata: "ModelExpertMetadata"):
        super().__init__(metadata)
        self._records = []

    def get_single_pass_gatherer_keys(self):
        if False:  # TODO `server_args.enable_two_batch_overlap`
            return [_SINGLE_PASS_GATHERER_KEY_PRIMARY, "child_a", "child_b"]
        return super().get_single_pass_gatherer_keys()

    def get_single_pass_gatherer_key(self, debug_name: str):
        if False:  # TODO `server_args.enable_two_batch_overlap`
            return debug_name
        return super().get_single_pass_gatherer_key(debug_name)

    def append(self, forward_pass_id: int, gatherer_key: str, single_pass_physical_count: torch.Tensor):
        self._records.append(dict(
            forward_pass_id=forward_pass_id,
            rank=self._rank,
            gatherer_key=gatherer_key,
            physical_count=single_pass_physical_count.tolist(),
        ))

    def reset(self):
        self._records.clear()

    def dump(self):
        return deepcopy(self._records)


class _StatAccumulator(_Accumulator):
    @classmethod
    def postprocess_dumps(cls, physical_dumps: List[Any], physical_to_logical_map: torch.Tensor,
                          metadata: "ModelExpertMetadata"):
        logical_count = torch.zeros((metadata.num_layers, metadata.num_logical_experts))
        # Most naive implementation, can optimize if it is bottleneck
        for physical_dump in physical_dumps:
            for layer_index in range(metadata.num_layers):
                for local_physical_expert_index in range(metadata.num_local_physical_experts):
                    global_physical_expert_index = metadata.num_local_physical_experts * physical_dump[
                        'rank'] + local_physical_expert_index
                    logical_expert_index = physical_to_logical_map[layer_index, global_physical_expert_index]
                    logical_count[layer_index, logical_expert_index] += physical_dump['physical_count'][
                        layer_index, local_physical_expert_index]
        return dict(logical_count=logical_count)

    def __init__(self, metadata: "ModelExpertMetadata"):
        super().__init__(metadata)
        self._physical_count = torch.zeros((self._metadata.num_layers, self._metadata.num_local_physical_experts))

    def append(self, forward_pass_id: int, gatherer_key: str, single_pass_physical_count: torch.Tensor):
        self._physical_count += single_pass_physical_count

    def reset(self):
        self._physical_count[...] = 0

    def dump(self):
        return dict(
            rank=self._rank,
            physical_count=self._physical_count.tolist(),
        )


# --------------------------------------- Misc -----------------------------------------

@dataclass
class ModelExpertMetadata:
    num_layers: int
    num_local_physical_experts: int
    num_logical_experts: int

    @staticmethod
    def from_model(model):
        return TDO
        return ModelExpertMetadata._init_dummy()

    @staticmethod
    def init_new(
        num_layers: int,
        num_logical_experts: int,
    ):
        return ModelExpertMetadata(
            num_layers=num_layers,
            num_logical_experts=num_logical_experts,
            # TODO handle more complex cases, e.g. duplicate some experts
            num_local_physical_experts=num_logical_experts // get_tensor_model_parallel_world_size(),
        )

    @staticmethod
    def _init_dummy():
        return ModelExpertMetadata(
            num_layers=1,
            num_local_physical_experts=1,
            num_logical_experts=1,
        )
