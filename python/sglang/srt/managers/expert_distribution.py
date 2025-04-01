import logging
from abc import ABC
from contextlib import contextmanager
from copy import deepcopy
from typing import List, Type, Any, Optional

import torch
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import Withable, get_bool_env_var

logger = logging.getLogger(__name__)


# --------------------------------------- Entrypoint -----------------------------------------

class ExpertDistributionRecorder:
    """Global expert distribution recording"""

    def __init__(self, server_args: ServerArgs):
        self._recording = False
        self._current_layer_idx = Withable()
        self._accumulator = _Accumulator.init_new()
        self._single_pass_gatherers = {
            k: _SinglePassGatherer.init_new(server_args)
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
        single_pass_physical_count = self._single_pass_gatherer.collect()
        self._accumulator.append(forward_pass_id, single_pass_physical_count)
        self._single_pass_gatherer.reset()

    def on_select_experts(self, topk_ids: torch.Tensor):
        if not self._recording:
            return
        self._single_pass_gatherer.on_select_experts(layer_idx=self._current_layer_idx.value, topk_ids=topk_ids)

    def on_deepep_dispatch_normal(self, num_recv_tokens_per_expert_list: List[int]):
        if not self._recording:
            return
        self._single_pass_gatherer.on_deepep_dispatch_normal(self._current_layer_idx.value,
                                                             num_recv_tokens_per_expert_list)

    def _reset(self):
        """Reset the expert distribution recorder."""
        logger.info("Resetting ExpertDistributionRecorder...")
        self._recording = False
        assert self._current_layer_idx.value is None
        self._single_pass_gatherer.reset()
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


def postprocess_dumps(physical_dumps: List[Any], physical_to_logical_map: torch.Tensor):
    return _Accumulator.get_class().postprocess_dumps(physical_dumps, physical_to_logical_map)


# --------------------------------------- SinglePassGatherer -----------------------------------------


class _SinglePassGatherer(ABC):
    @staticmethod
    def init_new(server_args: ServerArgs) -> "_SinglePassGatherer":
        if server_args.enable_deepep_moe:
            # TODO DeepEP low latency
            return _DeepepNormalSinglePassGatherer()
        return _LayerBasedSinglePassGatherer()

    def on_select_experts(self, layer_idx: int, topk_ids: torch.Tensor):
        pass

    def on_deepep_dispatch_normal(self, layer_idx: int, num_recv_tokens_per_expert_list: List[int]):
        pass

    def reset(self):
        raise NotImplementedError

    def collect(self) -> torch.Tensor:
        raise NotImplementedError


class _LayerBasedSinglePassGatherer(_SinglePassGatherer):
    def __init__(self):
        self._num_recv_tokens_per_expert_list_of_layer = {}

    def _on_layer_data(self, layer_idx: int, num_recv_tokens_per_expert_list: List[int]):
        # TODO for TBO, we may need to relax this restriction
        assert layer_idx not in self._num_recv_tokens_per_expert_list_of_layer
        assert 0 <= layer_idx < num_layers
        self._num_recv_tokens_per_expert_list_of_layer[layer_idx] = num_recv_tokens_per_expert_list

    def reset(self):
        self._num_recv_tokens_per_expert_list_of_layer.clear()

    def collect(self) -> torch.Tensor:
        data = [
            self._num_recv_tokens_per_expert_list_of_layer.get(layer_index) or ([0] * num_local_physical_experts)
            for layer_index in range(num_layers)
        ]
        return torch.tensor(data)


class _SelectExpertsSinglePassGatherer(_LayerBasedSinglePassGatherer):
    # pretty slow, but we will use the DeepEP Gatherer in production
    def on_select_experts(self, layer_idx: int, topk_ids: torch.Tensor):
        topk_ids_list = topk_ids.to("cpu", non_blocking=True).numpy().tolist()
        torch.cuda.synchronize()

        num_recv_tokens_per_expert_list = [0] * num_local_physical_experts
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
class _DeepepLowLatencySinglePassGatherer(_SinglePassGatherer):
    pass


# --------------------------------------- Accumulator -----------------------------------------

_SINGLE_PASS_GATHERER_KEY_PRIMARY = "primary"


class _Accumulator(ABC):
    @staticmethod
    def init_new() -> "_Accumulator":
        return _Accumulator.get_class()()

    @staticmethod
    def get_class() -> Type["_Accumulator"]:
        if get_bool_env_var("SGLANG_EXPERT_DISTRIBUTION_RECORDER_DETAIL"):
            return _DetailAccumulator
        return _StatAccumulator

    def get_single_pass_gatherer_keys(self):
        return ["primary"]

    @classmethod
    def postprocess_dumps(cls, physical_dumps: List[Any], physical_to_logical_map: torch.Tensor):
        raise NotImplementedError

    def append(self, forward_pass_id: int, single_pass_physical_count: torch.Tensor):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def dump(self):
        raise NotImplementedError


class _DetailAccumulator(_Accumulator):
    @classmethod
    def postprocess_dumps(cls, physical_dumps: List[Any], physical_to_logical_map: torch.Tensor):
        # Do not convert to logical since we want all details
        return [
            record
            for physical_dump in physical_dumps
            for record in physical_dump
        ]

    def __init__(self):
        self._records = []

    def get_single_pass_gatherer_keys(self):
        if False:  # TODO `server_args.enable_two_batch_overlap`
            return [_SINGLE_PASS_GATHERER_KEY_PRIMARY, "child_a", "child_b"]
        return super().get_single_pass_gatherer_keys()

    def append(self, forward_pass_id: int, single_pass_physical_count: torch.Tensor):
        self._records.append(dict(
            forward_pass_id=forward_pass_id,
            rank=TODO,
            physical_count=single_pass_physical_count.tolist(),
        ))

    def reset(self):
        self._records.clear()

    def dump(self):
        return deepcopy(self._records)


class _StatAccumulator(_Accumulator):
    @classmethod
    def postprocess_dumps(cls, physical_dumps: List[Any], physical_to_logical_map: torch.Tensor):
        logical_count = torch.zeros((num_layers, num_logical_experts))
        # Most naive implementation, can optimize if it is bottleneck
        for physical_dump in physical_dumps:
            for layer_index in range(num_layers):
                for local_physical_expert_index in range(num_local_physical_experts):
                    global_physical_expert_index = num_local_physical_experts * physical_dump[
                        'rank'] + local_physical_expert_index
                    logical_expert_index = physical_to_logical_map[layer_index, global_physical_expert_index]
                    logical_count[layer_index, logical_expert_index] += physical_dump['physical_count'][
                        layer_index, local_physical_expert_index]
        return dict(logical_count=logical_count)

    def __init__(self):
        self._physical_count = torch.zeros((num_layers, num_local_physical_experts))

    def append(self, forward_pass_id: int, single_pass_physical_count: torch.Tensor):
        self._physical_count += single_pass_physical_count

    def reset(self):
        self._physical_count[...] = 0

    def dump(self):
        return dict(
            rank=TODO,
            physical_count=self._physical_count.tolist(),
        )
