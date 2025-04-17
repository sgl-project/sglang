import logging
import os
import time
from abc import ABC
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, List, Optional, Type

import torch
from sglang.srt.managers.expert_location import ExpertLocationMetadata
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import Withable, get_bool_env_var

logger = logging.getLogger(__name__)


# --------------------------------------- Entrypoint -----------------------------------------


class ExpertDistributionRecorder:
    """Global expert distribution recording"""

    @staticmethod
    def init_new(
            server_args: ServerArgs,
            expert_location_metadata: "ExpertLocationMetadata",
            rank: int,
    ):
        if server_args.enable_expert_distribution_recorder:
            return _ExpertDistributionRecorderReal(server_args, expert_location_metadata, rank)
        else:
            return _ExpertDistributionRecorderNoop()

    @contextmanager
    def with_current_layer(self, layer_idx):
        yield

    @contextmanager
    def with_debug_name(self, debug_name):
        yield

    @contextmanager
    def with_forward_pass(self, forward_pass_id: int):
        yield

    def on_select_experts(self, topk_ids: torch.Tensor):
        pass

    def on_deepep_dispatch_normal(self, num_recv_tokens_per_expert_list: List[int]):
        pass

    def on_deepep_dispatch_low_latency(self, recv_count: torch.Tensor):
        pass

    def start_record(self):
        self._on_not_implemented()

    def stop_record(self):
        self._on_not_implemented()

    def dump_record(self):
        self._on_not_implemented()

    def _on_not_implemented(self):
        raise Exception(
            "Please enable ServerArgs.enable_expert_distribution_recorder to use ExpertDistributionRecorder.")


class _ExpertDistributionRecorderNoop(ExpertDistributionRecorder):
    pass


class _ExpertDistributionRecorderReal(ExpertDistributionRecorder):
    def __init__(
            self,
            server_args: ServerArgs,
            expert_location_metadata: "ExpertLocationMetadata",
            rank: int,
    ):
        self._server_args = server_args
        self._expert_location_metadata = expert_location_metadata

        self._recording = False
        self._current_layer_idx = Withable()
        self._current_debug_name = Withable()
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
        self._on_forward_pass_start()
        try:
            yield
        finally:
            self._on_forward_pass_end(forward_pass_id)

    def _on_forward_pass_start(self):
        if not self._recording:
            return
        for gatherer_key, gatherer in self._single_pass_gatherers.items():
            gatherer.reset()

    def _on_forward_pass_end(self, forward_pass_id: int):
        if not self._recording:
            return
        for gatherer_key, gatherer in self._single_pass_gatherers.items():
            single_pass_global_physical_count = gatherer.collect_global_physical_count()
            self._accumulator.append(
                forward_pass_id, gatherer_key, single_pass_global_physical_count
            )

    def flush_buffer_depending_on_expert_location_metadata(self):
        self._accumulator.flush_buffer_depending_on_expert_location_metadata()

    def on_select_experts(self, topk_ids: torch.Tensor):
        self._on_hook("on_select_experts", topk_ids=topk_ids)

    def on_deepep_dispatch_normal(self, num_recv_tokens_per_expert_list: List[int]):
        self._on_hook(
            "on_deepep_dispatch_normal",
            num_recv_tokens_per_expert_list=num_recv_tokens_per_expert_list,
        )

    def on_deepep_dispatch_low_latency(self, recv_count: torch.Tensor):
        self._on_hook("on_deepep_dispatch_low_latency", recv_count=recv_count)

    def _on_hook(self, hook_name: str, **kwargs):
        if not (self._recording or torch.cuda.is_current_stream_capturing()):
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
        assert (
                self._current_layer_idx.value is None
        ), f"{self._current_layer_idx.value=}"
        for gatherer in self._single_pass_gatherers.values():
            gatherer.reset()
        self._accumulator.reset()

    def start_record(self):
        """Start recording the expert distribution."""
        if self._recording:
            logger.warning(
                "SGLang server is already recording expert ids. Did you forget to dump the expert ids recorded so far by sending requests to the `/stop_expert_distribution_record` and `/dump_expert_distribution_record` endpoints?"
            )
        assert (
            self._server_args.disable_overlap_schedule
        ), "ExpertDistributionRecorder needs disable_overlap_schedule currently (will implement this later)"
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


_global_expert_distribution_recorder: Optional[ExpertDistributionRecorder] = None


def get_global_expert_distribution_recorder():
    return _global_expert_distribution_recorder


def set_global_expert_distribution_recorder(value):
    global _global_expert_distribution_recorder
    assert _global_expert_distribution_recorder is None
    _global_expert_distribution_recorder = value


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
            # `auto` has many restrictions now, so we lower the priority to implement low-latency capturing for auto
            if server_args.deepep_mode in ["normal", "auto"]:
                return _DeepepNormalSinglePassGatherer(expert_location_metadata)
            elif server_args.deepep_mode == "low_latency":
                return _DeepepLowLatencySinglePassGatherer(expert_location_metadata)
            else:
                raise NotImplementedError
        return _SelectExpertsSinglePassGatherer(expert_location_metadata)

    def __init__(self, expert_location_metadata: "ExpertLocationMetadata"):
        self._expert_location_metadata = expert_location_metadata

    def on_select_experts(self, layer_idx: int, topk_ids: torch.Tensor):
        pass

    def on_deepep_dispatch_normal(
            self, layer_idx: int, num_recv_tokens_per_expert_list: List[int]
    ):
        pass

    def on_deepep_dispatch_low_latency(self, layer_idx: int, recv_count: torch.Tensor):
        pass

    def reset(self):
        raise NotImplementedError

    def collect_global_physical_count(self) -> torch.Tensor:
        raise NotImplementedError


class _LayerBasedSinglePassGatherer(_SinglePassGatherer):
    def __init__(self, expert_location_metadata: "ExpertLocationMetadata"):
        super().__init__(expert_location_metadata)
        self._objects_of_layer = {}

    def _on_layer_data(
            self, layer_idx: int, objects: List[int]
    ):
        assert layer_idx not in self._objects_of_layer
        assert 0 <= layer_idx < self._expert_location_metadata.num_layers
        self._objects_of_layer[layer_idx] = objects

    def reset(self):
        self._objects_of_layer.clear()

    def _collect_objects(self, pad_len: int) -> torch.Tensor:
        data = [
            self._objects_of_layer.get(layer_index)
            or ([0] * pad_len)
            for layer_index in range(self._expert_location_metadata.num_layers)
        ]
        return torch.tensor(data)


class _SelectExpertsSinglePassGatherer(_LayerBasedSinglePassGatherer):
    # pretty slow, but we will use the DeepEP Gatherer in production
    def on_select_experts(self, layer_idx: int, topk_ids: torch.Tensor):
        topk_ids_list = topk_ids.to("cpu", non_blocking=True).numpy().tolist()
        torch.cuda.synchronize()

        global_physical_count = [0] * self._expert_location_metadata.num_physical_experts
        for token_record in topk_ids_list:
            for global_physical_expert_idx in token_record:
                global_physical_count[global_physical_expert_idx] += 1

        self._on_layer_data(layer_idx, global_physical_count)

    def collect_global_physical_count(self) -> torch.Tensor:
        return TODO


class _DeepepNormalSinglePassGatherer(_LayerBasedSinglePassGatherer):
    def on_deepep_dispatch_normal(
            self, layer_idx: int, num_recv_tokens_per_expert_list: List[int]
    ):
        assert isinstance(num_recv_tokens_per_expert_list, list)
        self._on_layer_data(layer_idx, num_recv_tokens_per_expert_list)

    def collect_global_physical_count(self) -> torch.Tensor:
        return TODO


class _DeepepLowLatencySinglePassGatherer(_SinglePassGatherer):
    def __init__(self, expert_location_metadata: "ExpertLocationMetadata"):
        super().__init__(expert_location_metadata)
        self._data = torch.zeros(
            (expert_location_metadata.num_layers, expert_location_metadata.num_local_physical_experts),
            dtype=torch.int,
            device="cuda",
        )

    def on_deepep_dispatch_low_latency(self, layer_idx: int, recv_count: torch.Tensor):
        # Most naive implementation, can optimize later
        self._data[layer_idx, :] = recv_count

    def reset(self):
        self._data[...] = 0

    def collect_global_physical_count(self) -> torch.Tensor:
        return self._data


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
            single_pass_global_physical_count: torch.Tensor,
    ):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def dump(self):
        raise NotImplementedError

    def flush_buffer_depending_on_expert_location_metadata(self):
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

        self._save_dir = os.environ.get("SGLANG_EXPERT_DISTRIBUTION_RECORDER_SAVE_DIR")
        if self._save_dir is not None and not Path(self._save_dir).exists():
            Path(self._save_dir).mkdir(parents=True, exist_ok=True)

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
            single_pass_global_physical_count: torch.Tensor,
    ):
        single_pass_global_physical_count = single_pass_global_physical_count.to("cpu")
        if self._save_dir is None:
            single_pass_global_physical_count = single_pass_global_physical_count.tolist()

        self._records.append(
            dict(
                forward_pass_id=forward_pass_id,
                rank=self._rank,
                gatherer_key=gatherer_key,
                physical_count=single_pass_global_physical_count,
            )
        )

    def reset(self):
        self._records.clear()

    def dump(self):
        if self._save_dir is None:
            return deepcopy(self._records)
        else:
            path_output = Path(self._save_dir) / f"{time.time()}-{self._rank}.pt"
            logger.info(f"Write expert distribution to {path_output}")
            torch.save(self._records, str(path_output))
            return [dict(path_output=str(path_output))]

    def flush_buffer_depending_on_expert_location_metadata(self):
        pass


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
            single_pass_global_physical_count: torch.Tensor,
    ):
        # Can optimize if overhead here is large
        self._physical_count += single_pass_global_physical_count

    def reset(self):
        self._physical_count[...] = 0

    def dump(self):
        return dict(
            rank=self._rank,
            physical_count=self._physical_count.tolist(),
        )

    def flush_buffer_depending_on_expert_location_metadata(self):
        TODO
