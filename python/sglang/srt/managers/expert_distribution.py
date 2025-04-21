import logging
import os
import time
from abc import ABC
from collections import deque
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import einops
import torch
import torch.distributed

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
        if server_args.expert_distribution_recorder_mode is not None:
            return _ExpertDistributionRecorderReal(
                server_args, expert_location_metadata, rank
            )
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

    def on_deepep_dispatch_normal(self, local_physical_count_of_layer: List[int]):
        pass

    def on_deepep_dispatch_low_latency(
        self, local_physical_count_of_layer: torch.Tensor
    ):
        pass

    def start_record(self):
        self._on_not_implemented()

    def stop_record(self):
        self._on_not_implemented()

    def dump_record(self):
        self._on_not_implemented()

    def _on_not_implemented(self):
        raise Exception(
            "Please set ServerArgs.expert_distribution_recorder_mode to use ExpertDistributionRecorder."
        )


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
        self._accumulator = _Accumulator.init_new(
            server_args, expert_location_metadata, rank
        )
        self._single_pass_gatherers = {
            k: _SinglePassGatherer.init_new(server_args, expert_location_metadata, rank)
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

    def on_deepep_dispatch_normal(self, local_physical_count_of_layer: List[int]):
        self._on_hook(
            "on_deepep_dispatch_normal",
            local_physical_count_of_layer=local_physical_count_of_layer,
        )

    def on_deepep_dispatch_low_latency(
        self, local_physical_count_of_layer: torch.Tensor
    ):
        self._on_hook(
            "on_deepep_dispatch_low_latency",
            local_physical_count_of_layer=local_physical_count_of_layer,
        )

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
    dumps: List[Any],
    server_args: ServerArgs,
    expert_location_metadata: "ExpertLocationMetadata",
):
    return _Accumulator.get_class(server_args).postprocess_dumps(
        dumps, expert_location_metadata
    )


# --------------------------------------- SinglePassGatherer -----------------------------------------


class _SinglePassGatherer(ABC):
    @staticmethod
    def init_new(
        server_args: ServerArgs,
        expert_location_metadata: "ExpertLocationMetadata",
        rank: int,
    ) -> "_SinglePassGatherer":
        if server_args.enable_deepep_moe:
            # `auto` has many restrictions now, so we lower the priority to implement low-latency capturing for auto
            if server_args.deepep_mode in ["normal", "auto"]:
                return _DeepepNormalSinglePassGatherer(expert_location_metadata, rank)
            elif server_args.deepep_mode == "low_latency":
                return _DeepepLowLatencySinglePassGatherer(
                    expert_location_metadata, rank
                )
            else:
                raise NotImplementedError
        return _SelectExpertsSinglePassGatherer(expert_location_metadata, rank)

    def __init__(self, expert_location_metadata: "ExpertLocationMetadata", rank: int):
        self._expert_location_metadata = expert_location_metadata
        self._rank = rank

    def on_select_experts(self, layer_idx: int, topk_ids: torch.Tensor):
        pass

    def on_deepep_dispatch_normal(
        self, layer_idx: int, local_physical_count_of_layer: List[int]
    ):
        pass

    def on_deepep_dispatch_low_latency(
        self, layer_idx: int, local_physical_count_of_layer: torch.Tensor
    ):
        pass

    def reset(self):
        raise NotImplementedError

    def collect_global_physical_count(self) -> torch.Tensor:
        raise NotImplementedError


class _LayerBasedSinglePassGatherer(_SinglePassGatherer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._objects_of_layer = {}

    def _on_layer_data(self, layer_idx: int, objects: List[int]):
        assert 0 <= layer_idx < self._expert_location_metadata.num_layers
        if layer_idx in self._objects_of_layer:
            self._objects_of_layer[layer_idx] = _list_sum(
                self._objects_of_layer[layer_idx], objects
            )
        else:
            self._objects_of_layer[layer_idx] = objects

    def reset(self):
        self._objects_of_layer.clear()

    def _collect_objects(self, pad_len: int) -> torch.Tensor:
        data = [
            self._objects_of_layer.get(layer_index) or ([0] * pad_len)
            for layer_index in range(self._expert_location_metadata.num_layers)
        ]
        return torch.tensor(data)


def _list_sum(a: List, b: List) -> List:
    return [x + y for x, y in zip(a, b, strict=True)]


class _SelectExpertsSinglePassGatherer(_LayerBasedSinglePassGatherer):
    # pretty slow, but we will use the DeepEP Gatherer in production
    def on_select_experts(self, layer_idx: int, topk_ids: torch.Tensor):
        topk_ids_list = topk_ids.to("cpu", non_blocking=True).numpy().tolist()
        torch.cuda.synchronize()

        global_physical_count = [
            0
        ] * self._expert_location_metadata.num_physical_experts
        for token_record in topk_ids_list:
            for global_physical_expert_idx in token_record:
                global_physical_count[global_physical_expert_idx] += 1

        self._on_layer_data(layer_idx, global_physical_count)

    def collect_global_physical_count(self) -> torch.Tensor:
        return super()._collect_objects(
            pad_len=self._expert_location_metadata.num_physical_experts
        )


class _DeepepNormalSinglePassGatherer(_LayerBasedSinglePassGatherer):
    def on_deepep_dispatch_normal(
        self, layer_idx: int, local_physical_count_of_layer: List[int]
    ):
        assert isinstance(local_physical_count_of_layer, list)
        self._on_layer_data(layer_idx, local_physical_count_of_layer)

    def collect_global_physical_count(self) -> torch.Tensor:
        local_physical_count = super()._collect_objects(
            pad_len=self._expert_location_metadata.num_local_physical_experts
        )
        return _convert_local_to_global_physical_count(
            local_physical_count,
            rank=self._rank,
            num_local_physical_experts=self._expert_location_metadata.num_local_physical_experts,
            num_physical_experts=self._expert_location_metadata.num_physical_experts,
        )


class _DeepepLowLatencySinglePassGatherer(_SinglePassGatherer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data = torch.zeros(
            (
                self._expert_location_metadata.num_layers,
                self._expert_location_metadata.num_local_physical_experts,
            ),
            dtype=torch.int,
            device="cuda",
        )

    def on_deepep_dispatch_low_latency(
        self, layer_idx: int, local_physical_count_of_layer: torch.Tensor
    ):
        # Most naive implementation, can optimize later
        self._data[layer_idx, :] = local_physical_count_of_layer

    def reset(self):
        self._data[...] = 0

    def collect_global_physical_count(self) -> torch.Tensor:
        # Can optimize if bottleneck
        return _convert_local_to_global_physical_count(
            self._data,
            rank=self._rank,
            num_local_physical_experts=self._expert_location_metadata.num_local_physical_experts,
            num_physical_experts=self._expert_location_metadata.num_physical_experts,
        )


def _convert_local_to_global_physical_count(
    local_physical_count: torch.Tensor,
    rank: int,
    num_local_physical_experts: int,
    num_physical_experts: int,
) -> torch.Tensor:
    dtype = local_physical_count.dtype
    device = local_physical_count.device
    num_layers, _ = local_physical_count.shape

    ans = torch.zeros((num_layers, num_physical_experts), dtype=dtype, device=device)
    ans[
        :, num_local_physical_experts * rank : num_local_physical_experts * (rank + 1)
    ] = local_physical_count
    return ans


# --------------------------------------- Accumulator -----------------------------------------

_SINGLE_PASS_GATHERER_KEY_PRIMARY = "primary"


class _Accumulator(ABC):
    @staticmethod
    def init_new(
        server_args: ServerArgs,
        expert_location_metadata: "ExpertLocationMetadata",
        rank: int,
    ) -> "_Accumulator":
        return _Accumulator.get_class(server_args)(expert_location_metadata, rank)

    @staticmethod
    def get_class(server_args: ServerArgs) -> Type["_Accumulator"]:
        return {
            "stat": _StatAccumulator,
            "stat_ut": _StatAndUtilizationRateAccumulator,
            "detail": _DetailAccumulator,
        }[server_args.expert_distribution_recorder_mode]

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
        dumps: List[Any],
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
        dumps: List[Any],
        expert_location_metadata: "ExpertLocationMetadata",
    ):
        # Do not convert to logical since we want all details
        return [record for dump in dumps for record in dump]

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
            single_pass_global_physical_count = (
                single_pass_global_physical_count.tolist()
            )

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
        dumps: List[Any],
        expert_location_metadata: "ExpertLocationMetadata",
    ):
        logical_count = torch.tensor([item["logical_count"] for item in dumps]).sum(
            dim=0
        )
        return dict(logical_count=logical_count.tolist())

    def __init__(self, expert_location_metadata: "ExpertLocationMetadata", rank: int):
        super().__init__(expert_location_metadata, rank)
        self._buffer_global_physical_count = torch.zeros(
            (
                self._expert_location_metadata.num_layers,
                self._expert_location_metadata.num_physical_experts,
            )
        )
        self._logical_count = torch.zeros(
            (
                self._expert_location_metadata.num_layers,
                self._expert_location_metadata.num_logical_experts,
            )
        )

    def append(
        self,
        forward_pass_id: int,
        gatherer_key: str,
        single_pass_global_physical_count: torch.Tensor,
    ):
        # Can optimize if overhead here is large
        self._buffer_global_physical_count += single_pass_global_physical_count.cpu()

    def reset(self):
        self._buffer_global_physical_count[...] = 0
        self._logical_count[...] = 0

    def dump(self):
        self.flush_buffer_depending_on_expert_location_metadata()

        return dict(
            rank=self._rank,
            logical_count=self._logical_count.tolist(),
        )

    def flush_buffer_depending_on_expert_location_metadata(self):
        self._logical_count += _convert_global_physical_count_to_logical_count(
            self._buffer_global_physical_count,
            expert_location_metadata=self._expert_location_metadata,
        )
        self._buffer_global_physical_count[...] = 0


def _convert_global_physical_count_to_logical_count(
    global_physical_count: torch.Tensor,
    expert_location_metadata: ExpertLocationMetadata,
):
    num_layers = expert_location_metadata.num_layers
    num_logical_experts = expert_location_metadata.num_logical_experts

    logical_count = torch.zeros((num_layers, num_logical_experts))
    logical_count.scatter_add_(
        dim=1,
        index=expert_location_metadata.physical_to_logical_map,
        src=global_physical_count,
    )
    return logical_count


# TODO use composition instead of inheritance later
class _StatAndUtilizationRateAccumulator(_StatAccumulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        window_sizes = [10, 100, 1000]
        self._history = _DequeCollection(maxlens=window_sizes)
        self._rank = torch.distributed.get_rank()

    def append(
        self,
        forward_pass_id: int,
        gatherer_key: str,
        single_pass_global_physical_count: torch.Tensor,
    ):
        super().append(forward_pass_id, gatherer_key, single_pass_global_physical_count)
        self._append_utilization_rate(
            forward_pass_id, single_pass_global_physical_count
        )

    def reset(self):
        super().reset()
        self._history.clear()

    def _append_utilization_rate(
        self, forward_pass_id: int, single_pass_global_physical_count: torch.Tensor
    ):
        gpu_physical_count = compute_gpu_physical_count(
            single_pass_global_physical_count,
            num_gpu=self._expert_location_metadata.ep_size,
        )
        gpu_physical_count = gpu_physical_count.to("cuda")
        torch.distributed.reduce(
            gpu_physical_count, dst=0, op=torch.distributed.ReduceOp.SUM
        )

        if self._rank == 0:
            utilization_rate_tensor = compute_utilization_rate(gpu_physical_count)
            utilization_rate = torch.mean(utilization_rate_tensor).item()
            self._history.append(utilization_rate)

            gpu_physical_count_sum = gpu_physical_count.sum().item()

            logger.info(
                f"[Expert Utilization Rate] "
                f"forward_pass_id={forward_pass_id} "
                f"current_pass_value={utilization_rate:.03f} "
                f"{''.join(f'last_{size}_value={value:.03f} ' for size, value in self._history.mean().items())} "
                f"gpu_physical_count_sum={gpu_physical_count_sum}"
            )


class _DequeCollection:
    def __init__(self, maxlens: List[int]):
        self._dequeues = [deque(maxlen=maxlen) for maxlen in maxlens]

    def append(self, value):
        for d in self._dequeues:
            d.append(value)

    def clear(self):
        for d in self._dequeues:
            d.clear()

    def mean(self) -> Dict[int, float]:
        return {d.maxlen: sum(d) / len(d) for d in self._dequeues}


def compute_gpu_physical_count(
    physical_count_of_whatever: torch.Tensor,  # (..., num_layer, num_physical_expert)
    num_gpu: int,
):
    """output: gpu_physical_count_of_batch (..., num_layer, num_gpu)"""
    return einops.reduce(
        physical_count_of_whatever,
        "... num_layer (num_gpu num_expert_per_gpu) -> ... num_layer num_gpu",
        "sum",
        num_gpu=num_gpu,
    )


def compute_utilization_rate(
    gpu_physical_count_of_batch: torch.Tensor,  # (..., num_layer, num_gpu)
):
    """output: utilization_rate (..., num_layer)"""
    gpu_physical_count_of_batch = gpu_physical_count_of_batch.float()
    max_gpu_physical_count = einops.reduce(
        gpu_physical_count_of_batch,
        "... num_layer num_gpu -> ... num_layer",
        "max",
    )
    avg_gpu_physical_count = einops.reduce(
        gpu_physical_count_of_batch,
        "... num_layer num_gpu -> ... num_layer",
        "mean",
    )
    return (avg_gpu_physical_count + 1e-5) / (max_gpu_physical_count + 1e-5)
