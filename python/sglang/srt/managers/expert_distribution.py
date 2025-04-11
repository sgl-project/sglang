import logging
from abc import ABC
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, List, Optional, Type

import torch

from sglang.srt.managers.expert_location import ExpertLocationMetadata
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import Withable, get_bool_env_var

logger = logging.getLogger(__name__)


# --------------------------------------- Entrypoint -----------------------------------------


class _ExpertDistributionRecorder:
    """Global expert distribution recording"""

    def __init__(self):
        self._recording = False
        self._current_layer_idx = Withable()
        self._current_debug_name = Withable()

    def initialize(
        self,
        server_args: ServerArgs,
        expert_location_metadata: "ExpertLocationMetadata",
        rank: int,
    ):
        self._expert_location_metadata = expert_location_metadata
        self._accumulator = _Accumulator.init_new(expert_location_metadata, rank)
        self._single_pass_gatherers = {
            k: _SinglePassGatherer.init_new(server_args, expert_location_metadata)
            for k in self._accumulator.get_single_pass_gatherer_keys()
        }

    def with_current_layer(self, layer_idx):
        return self._current_layer_idx.with_value(layer_idx)

    def with_debug_name(self, debug_name):
        return self._current_debug_name.with_value(debug_name)

    @contextmanager
    def with_forward_pass(self, forward_pass_id: int):
        try:
            yield
        finally:
            self._on_forward_pass_end(forward_pass_id)

    def _on_forward_pass_end(self, forward_pass_id: int):
        if not self._recording:
            return
        for gatherer_key, gatherer in self._single_pass_gatherers.items():
            single_pass_physical_count = gatherer.collect()
            self._accumulator.append(
                forward_pass_id, gatherer_key, single_pass_physical_count
            )
            gatherer.reset()

    def on_select_experts(self, topk_ids: torch.Tensor):
        self._on_hook("on_select_experts", topk_ids=topk_ids)

    def on_deepep_dispatch_normal(self, num_recv_tokens_per_expert_list: List[int]):
        self._on_hook(
            "on_deepep_dispatch_normal",
            num_recv_tokens_per_expert_list=num_recv_tokens_per_expert_list,
        )

    def _on_hook(self, hook_name: str, **kwargs):
        if not self._recording:
            return
        gatherer = self._single_pass_gatherers[
            self._accumulator.get_single_pass_gatherer_key(
                self._current_debug_name.value
            )
        ]
        getattr(gatherer, hook_name)(layer_idx=self._current_layer_idx.value, **kwargs)

    def _reset(self):
        """Reset the expert distribution recorder."""
        logger.info("Resetting ExpertDistributionRecorder...")
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


expert_distribution_recorder = _ExpertDistributionRecorder()


def postprocess_dumps(
    physical_dumps: List[Any], expert_location_metadata: "ExpertLocationMetadata"
):
    return _Accumulator.get_class().postprocess_dumps(
        physical_dumps, expert_location_metadata
    )


# --------------------------------------- SinglePassGatherer -----------------------------------------


class _SinglePassGatherer(ABC):
    @staticmethod
    def init_new(
        server_args: ServerArgs, expert_location_metadata: "ExpertLocationMetadata"
    ) -> "_SinglePassGatherer":
        if server_args.enable_deepep_moe:
            # TODO DeepEP low latency
            return _DeepepNormalSinglePassGatherer(expert_location_metadata)
        return _SelectExpertsSinglePassGatherer(expert_location_metadata)

    def __init__(self, expert_location_metadata: "ExpertLocationMetadata"):
        self._expert_location_metadata = expert_location_metadata

    def on_select_experts(self, layer_idx: int, topk_ids: torch.Tensor):
        pass

    def on_deepep_dispatch_normal(
        self, layer_idx: int, num_recv_tokens_per_expert_list: List[int]
    ):
        pass

    def reset(self):
        raise NotImplementedError

    def collect(self) -> torch.Tensor:
        raise NotImplementedError


class _LayerBasedSinglePassGatherer(_SinglePassGatherer):
    def __init__(self, expert_location_metadata: "ExpertLocationMetadata"):
        super().__init__(expert_location_metadata)
        self._num_recv_tokens_per_expert_list_of_layer = {}

    def _on_layer_data(
        self, layer_idx: int, num_recv_tokens_per_expert_list: List[int]
    ):
        # TODO for TBO, we may need to relax this restriction
        assert layer_idx not in self._num_recv_tokens_per_expert_list_of_layer
        assert 0 <= layer_idx < self._expert_location_metadata.num_layers
        self._num_recv_tokens_per_expert_list_of_layer[layer_idx] = (
            num_recv_tokens_per_expert_list
        )

    def reset(self):
        self._num_recv_tokens_per_expert_list_of_layer.clear()

    def collect(self) -> torch.Tensor:
        data = [
            self._num_recv_tokens_per_expert_list_of_layer.get(layer_index)
            or ([0] * self._expert_location_metadata.num_local_physical_experts)
            for layer_index in range(self._expert_location_metadata.num_layers)
        ]
        return torch.tensor(data)


class _SelectExpertsSinglePassGatherer(_LayerBasedSinglePassGatherer):
    # pretty slow, but we will use the DeepEP Gatherer in production
    def on_select_experts(self, layer_idx: int, topk_ids: torch.Tensor):
        topk_ids_list = topk_ids.to("cpu", non_blocking=True).numpy().tolist()
        torch.cuda.synchronize()

        num_recv_tokens_per_expert_list = [
            0
        ] * self._expert_location_metadata.num_local_physical_experts
        for token_record in topk_ids_list:
            for global_physical_expert_idx in token_record:
                local_physical_expert_idx = (
                    self._expert_location_metadata.physical_to_local_physical(
                        global_physical_expert_idx
                    )
                )
                num_recv_tokens_per_expert_list[local_physical_expert_idx] += 1

        self._on_layer_data(layer_idx, num_recv_tokens_per_expert_list)


class _DeepepNormalSinglePassGatherer(_LayerBasedSinglePassGatherer):
    def on_deepep_dispatch_normal(
        self, layer_idx: int, num_recv_tokens_per_expert_list: List[int]
    ):
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
    def init_new(
        expert_location_metadata: "ExpertLocationMetadata", rank: int
    ) -> "_Accumulator":
        return _Accumulator.get_class()(expert_location_metadata, rank)

    @staticmethod
    def get_class() -> Type["_Accumulator"]:
        if get_bool_env_var("SGLANG_EXPERT_DISTRIBUTION_RECORDER_DETAIL"):
            return _DetailAccumulator
        return _StatAccumulator

    def __init__(self, expert_location_metadata: "ExpertLocationMetadata", rank: int):
        self._expert_location_metadata = expert_location_metadata
        self._rank = rank

    def get_single_pass_gatherer_keys(self):
        return [_SINGLE_PASS_GATHERER_KEY_PRIMARY]

    def get_single_pass_gatherer_key(self, debug_name: Optional[str]):
        return _SINGLE_PASS_GATHERER_KEY_PRIMARY

    @classmethod
    def postprocess_dumps(
        cls,
        physical_dumps: List[Any],
        expert_location_metadata: "ExpertLocationMetadata",
    ):
        raise NotImplementedError

    def append(
        self,
        forward_pass_id: int,
        gatherer_key: str,
        single_pass_physical_count: torch.Tensor,
    ):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def dump(self):
        raise NotImplementedError


class _DetailAccumulator(_Accumulator):
    @classmethod
    def postprocess_dumps(
        cls,
        physical_dumps: List[Any],
        expert_location_metadata: "ExpertLocationMetadata",
    ):
        # Do not convert to logical since we want all details
        return [record for physical_dump in physical_dumps for record in physical_dump]

    def __init__(self, expert_location_metadata: "ExpertLocationMetadata", rank: int):
        super().__init__(expert_location_metadata, rank)
        self._records = []

    def get_single_pass_gatherer_keys(self):
        if False:  # TODO `server_args.enable_two_batch_overlap`
            return [_SINGLE_PASS_GATHERER_KEY_PRIMARY, "child_a", "child_b"]
        return super().get_single_pass_gatherer_keys()

    def get_single_pass_gatherer_key(self, debug_name: Optional[str]):
        if False:  # TODO `server_args.enable_two_batch_overlap`
            return debug_name or _SINGLE_PASS_GATHERER_KEY_PRIMARY
        return super().get_single_pass_gatherer_key(debug_name)

    def append(
        self,
        forward_pass_id: int,
        gatherer_key: str,
        single_pass_physical_count: torch.Tensor,
    ):
        self._records.append(
            dict(
                forward_pass_id=forward_pass_id,
                rank=self._rank,
                gatherer_key=gatherer_key,
                physical_count=single_pass_physical_count.tolist(),
            )
        )

    def reset(self):
        self._records.clear()

    def dump(self):
        return deepcopy(self._records)


class _StatAccumulator(_Accumulator):
    @classmethod
    def postprocess_dumps(
        cls,
        physical_dumps: List[Any],
        expert_location_metadata: "ExpertLocationMetadata",
    ):
        logical_count = torch.zeros(
            (
                expert_location_metadata.num_layers,
                expert_location_metadata.num_logical_experts,
            )
        )
        # Most naive implementation, can optimize if it is bottleneck
        for physical_dump in physical_dumps:
            for layer_index in range(expert_location_metadata.num_layers):
                for local_physical_expert_index in range(
                    expert_location_metadata.num_local_physical_experts
                ):
                    global_physical_expert_index = (
                        expert_location_metadata.local_physical_to_physical(
                            rank=physical_dump["rank"],
                            local_physical_expert_index=local_physical_expert_index,
                        )
                    )
                    logical_expert_index = (
                        expert_location_metadata.physical_to_logical_map[
                            layer_index, global_physical_expert_index
                        ]
                    )
                    logical_count[layer_index, logical_expert_index] += physical_dump[
                        "physical_count"
                    ][layer_index][local_physical_expert_index]
        return dict(logical_count=logical_count.tolist())

    def __init__(self, expert_location_metadata: "ExpertLocationMetadata", rank: int):
        super().__init__(expert_location_metadata, rank)
        self._physical_count = torch.zeros(
            (
                self._expert_location_metadata.num_layers,
                self._expert_location_metadata.num_local_physical_experts,
            )
        )

    def append(
        self,
        forward_pass_id: int,
        gatherer_key: str,
        single_pass_physical_count: torch.Tensor,
    ):
        self._physical_count += single_pass_physical_count

    def reset(self):
        self._physical_count[...] = 0

    def dump(self):
        return dict(
            rank=self._rank,
            physical_count=self._physical_count.tolist(),
        )
