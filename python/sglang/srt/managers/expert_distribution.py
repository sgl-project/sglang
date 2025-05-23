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
import logging
import os
import time
from abc import ABC
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Type

import einops
import torch
import torch.distributed

from sglang.srt.managers.expert_location import ExpertLocationMetadata
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import Withable, get_bool_env_var

logger = logging.getLogger(__name__)

# --------------------------------------- Entrypoint -----------------------------------------

_OutputMode = Literal["file", "object"]


class ExpertDistributionRecorder(ABC):
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
    def with_forward_pass(self, forward_pass_id: int, forward_batch: ForwardBatch):
        yield

    def on_select_experts(self, topk_ids: torch.Tensor):
        pass

    def on_deepep_dispatch_normal(
        self,
        local_physical_count_of_layer: List[int],
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
    ):
        pass

    def on_deepep_dispatch_low_latency(
        self, local_physical_count_of_layer: torch.Tensor
    ):
        pass

    def start_record(self):
        self._on_not_implemented()

    def stop_record(self):
        self._on_not_implemented()

    def dump_record(self, output_mode: _OutputMode = "file"):
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
        self._current_forward_pass_id = Withable()
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
    def with_forward_pass(self, forward_pass_id: int, forward_batch: ForwardBatch):
        with self._current_forward_pass_id.with_value(forward_pass_id):
            self._on_forward_pass_start(forward_batch)
            try:
                yield
            finally:
                self._on_forward_pass_end(forward_pass_id)

    def _on_forward_pass_start(self, forward_batch: ForwardBatch):
        if not self._recording:
            return
        for gatherer_key, gatherer in self._single_pass_gatherers.items():
            gatherer.reset()
            gatherer.on_forward_pass_start(forward_batch)

    def _on_forward_pass_end(self, forward_pass_id: int):
        if not self._recording:
            return
        for gatherer_key, gatherer in self._single_pass_gatherers.items():
            single_pass_data = gatherer.collect()
            self._accumulator.append(forward_pass_id, gatherer_key, single_pass_data)

    def on_select_experts(self, topk_ids: torch.Tensor):
        self._on_hook("on_select_experts", topk_ids=topk_ids)

    def on_deepep_dispatch_normal(
        self,
        local_physical_count_of_layer: List[int],
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
    ):
        self._on_hook(
            "on_deepep_dispatch_normal",
            local_physical_count_of_layer=local_physical_count_of_layer,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            num_tokens_per_expert=num_tokens_per_expert,
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
        self._reset()
        self._recording = True

    def stop_record(self):
        """Stop recording the expert distribution."""
        if not self._recording:
            logger.warning(
                "SGLang server has not been recording expert ids. Did you forget to start recording by sending request to the `/start_expert_distribution_record` endpoint?"
            )
        self._recording = False

    def dump_record(self, output_mode: _OutputMode = "file"):
        """Dump the expert distribution record and reset the recorder after dumping."""
        output = self._accumulator.dump(output_mode=output_mode)
        self._reset()
        return output


_global_expert_distribution_recorder: Optional[ExpertDistributionRecorder] = (
    _ExpertDistributionRecorderNoop()
)


def get_global_expert_distribution_recorder():
    return _global_expert_distribution_recorder


def set_global_expert_distribution_recorder(value):
    global _global_expert_distribution_recorder
    _global_expert_distribution_recorder = value


# --------------------------------------- SinglePassGatherer -----------------------------------------


class _SinglePassGatherer(ABC):
    @staticmethod
    def init_new(
        server_args: ServerArgs,
        expert_location_metadata: "ExpertLocationMetadata",
        rank: int,
    ) -> "_SinglePassGatherer":
        if server_args.expert_distribution_recorder_mode == "per_token":
            return _DetailSinglePassGatherer(
                server_args, expert_location_metadata, rank
            )
        if server_args.enable_deepep_moe:
            if server_args.deepep_mode == "normal":
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

    def on_forward_pass_start(self, forward_batch: ForwardBatch):
        pass

    def on_select_experts(self, layer_idx: int, topk_ids: torch.Tensor):
        pass

    def on_deepep_dispatch_normal(
        self,
        layer_idx: int,
        local_physical_count_of_layer: List[int],
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
    ):
        pass

    def on_deepep_dispatch_low_latency(
        self, layer_idx: int, local_physical_count_of_layer: torch.Tensor
    ):
        pass

    def reset(self):
        raise NotImplementedError

    def collect(self) -> Dict:
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

    def collect(self) -> Dict:
        global_physical_count = super()._collect_objects(
            pad_len=self._expert_location_metadata.num_physical_experts
        )
        return dict(global_physical_count=global_physical_count)


class _DeepepNormalSinglePassGatherer(_LayerBasedSinglePassGatherer):
    def on_deepep_dispatch_normal(
        self,
        layer_idx: int,
        local_physical_count_of_layer: List[int],
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
    ):
        assert isinstance(local_physical_count_of_layer, list)
        self._on_layer_data(layer_idx, local_physical_count_of_layer)

    def collect(self) -> Dict:
        local_physical_count = super()._collect_objects(
            pad_len=self._expert_location_metadata.num_local_physical_experts
        )
        global_physical_count = _convert_local_to_global_physical_count(
            local_physical_count,
            rank=self._rank,
            num_local_physical_experts=self._expert_location_metadata.num_local_physical_experts,
            num_physical_experts=self._expert_location_metadata.num_physical_experts,
        )
        return dict(global_physical_count=global_physical_count)


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
        self._data[layer_idx, :] += local_physical_count_of_layer

    def reset(self):
        self._data[...] = 0

    def collect(self) -> Dict:
        # Can optimize if bottleneck
        global_physical_count = _convert_local_to_global_physical_count(
            self._data,
            rank=self._rank,
            num_local_physical_experts=self._expert_location_metadata.num_local_physical_experts,
            num_physical_experts=self._expert_location_metadata.num_physical_experts,
        )
        return dict(global_physical_count=global_physical_count)


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
        return _Accumulator.get_class(server_args)(
            server_args, expert_location_metadata, rank
        )

    @staticmethod
    def get_class(server_args: ServerArgs) -> Type["_Accumulator"]:
        return {
            "stat": _StatAccumulator,
            # TODO pr-chain: enable this later
            # "per_pass": _DetailAccumulator,
            # "per_token": _DetailAccumulator,
        }[server_args.expert_distribution_recorder_mode]

    def __init__(
        self,
        server_args: ServerArgs,
        expert_location_metadata: "ExpertLocationMetadata",
        rank: int,
    ):
        self._server_args = server_args
        self._expert_location_metadata = expert_location_metadata
        self._rank = rank

    def get_single_pass_gatherer_keys(self):
        return [_SINGLE_PASS_GATHERER_KEY_PRIMARY]

    def get_single_pass_gatherer_key(self, debug_name: Optional[str]):
        return _SINGLE_PASS_GATHERER_KEY_PRIMARY

    def append(
        self,
        forward_pass_id: int,
        gatherer_key: str,
        single_pass_data: Dict,
    ):
        pass

    def reset(self):
        pass

    def dump(self, output_mode: _OutputMode):
        pass


class _UtilizationRateAccumulatorMixin(_Accumulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._enable = self._server_args.enable_expert_distribution_metrics

        if self._enable:
            window_sizes = [10, 100, 1000]
            self._history = _DequeCollection(maxlens=window_sizes)
            self._rank = torch.distributed.get_rank()

    def append(
        self,
        forward_pass_id: int,
        gatherer_key: str,
        single_pass_data: Dict,
    ):
        super().append(forward_pass_id, gatherer_key, single_pass_data)
        if self._enable:
            self._append_utilization_rate(
                forward_pass_id, single_pass_data["global_physical_count"]
            )

    def reset(self):
        super().reset()
        if self._enable:
            self._history.clear()

    def _append_utilization_rate(
        self, forward_pass_id: int, single_pass_global_physical_count: torch.Tensor
    ):
        gpu_physical_count = compute_gpu_physical_count(
            single_pass_global_physical_count,
            num_gpu=self._expert_location_metadata.ep_size,
        )
        gpu_physical_count = gpu_physical_count.to(self._server_args.device)
        torch.distributed.reduce(
            gpu_physical_count, dst=0, op=torch.distributed.ReduceOp.SUM
        )

        if self._rank == 0:
            utilization_rate_tensor = compute_utilization_rate(gpu_physical_count)
            utilization_rate = torch.mean(utilization_rate_tensor).item()
            self._history.append(utilization_rate)

            gpu_physical_count_sum = gpu_physical_count.sum().item()

            logger.info(
                f"[Expert Balancedness] "
                f"forward_pass_id={forward_pass_id} "
                f"current_pass_balancedness={utilization_rate:.03f} "
                f"{''.join(f'last_{size}_average_balancedness={value:.03f} ' for size, value in self._history.mean().items())} "
                f"gpu_physical_count_sum={gpu_physical_count_sum}"
                # f"current_pass_per_layer={[round(x, 2) for x in utilization_rate_tensor.cpu().tolist()]}"
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


class _StatAccumulator(_UtilizationRateAccumulatorMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._global_physical_count_of_buffered_step = _Buffer.init_new(
            item_shape=(
                self._expert_location_metadata.num_layers,
                # Cannot use local_physical_count to support select_experts
                self._expert_location_metadata.num_physical_experts,
            ),
            buffer_size=self._server_args.expert_distribution_recorder_buffer_size,
            dtype=torch.int32,
            device=self._server_args.device,
        )

    def append(
        self,
        forward_pass_id: int,
        gatherer_key: str,
        single_pass_data: Dict,
    ):
        super().append(forward_pass_id, gatherer_key, single_pass_data)
        # Can optimize if overhead here is large
        self._global_physical_count_of_buffered_step.append(
            single_pass_data["global_physical_count"]
        )

    def reset(self):
        super().reset()
        self._global_physical_count_of_buffered_step.reset()

    def dump(self, output_mode: _OutputMode):
        logical_count_of_buffered_step = _convert_global_physical_count_to_logical_count(
            self._global_physical_count_of_buffered_step.get_all(),
            num_layers=self._expert_location_metadata.num_layers,
            num_logical_experts=self._expert_location_metadata.num_logical_experts,
            physical_to_logical_map=self._expert_location_metadata.physical_to_logical_map,
        )
        torch.distributed.all_reduce(
            logical_count_of_buffered_step, op=torch.distributed.ReduceOp.SUM
        )
        output = dict(
            rank=self._rank,
            logical_count=logical_count_of_buffered_step,
        )

        if output_mode == "file":
            if self._rank == 0:
                _dump_to_file(f"expert_distribution_recorder_{time.time()}.pt", output)
        elif output_mode == "object":
            return output
        else:
            raise NotImplementedError


def _dump_to_file(name, data):
    save_dir = Path(os.environ.get("SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR", "/tmp"))
    path_output = save_dir / name
    logger.info(f"Write expert distribution to {path_output}")
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, str(path_output))


class _Buffer:
    @staticmethod
    def init_new(item_shape: Tuple, buffer_size: int, dtype, device):
        if buffer_size < 0:
            return _InfiniteBuffer(item_shape, dtype=dtype, device=device)
        else:
            return _CircularBuffer(item_shape, buffer_size, dtype=dtype, device=device)

    def append(self, value: torch.Tensor):
        raise NotImplementedError

    def get_all(self) -> torch.Tensor:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class _CircularBuffer(_Buffer):
    def __init__(self, item_shape: Tuple, buffer_size: int, dtype, device):
        self._buffer = torch.zeros(
            (buffer_size, *item_shape), dtype=dtype, device=device
        )
        self._curr_index = 0

    def append(self, value: torch.Tensor):
        self._buffer[self._curr_index] = value
        self._curr_index = (self._curr_index + 1) % len(self._buffer)

    def get_all(self) -> torch.Tensor:
        return self._buffer

    def reset(self):
        self._buffer[...] = 0


class _InfiniteBuffer(_Buffer):
    def __init__(self, item_shape: Tuple, dtype, device):
        self._item_shape = item_shape
        self._buffer = torch.zeros((128, *item_shape), dtype=dtype, device=device)
        self._size = 0

    def append(self, value: torch.Tensor):
        curr_buffer_size = len(self._buffer)
        dtype = self._buffer.dtype
        device = self._buffer.device

        if self._size == curr_buffer_size:
            new_buffer = torch.zeros(
                (2 * curr_buffer_size, *self._item_shape), dtype=dtype, device=device
            )
            new_buffer[:curr_buffer_size] = self._buffer
            self._buffer = new_buffer

        self._buffer[self._size] = value
        self._size += 1

    def get_all(self) -> torch.Tensor:
        return self._buffer[: self._size]

    def reset(self):
        self._buffer[...] = 0
        self._size = 0


def _convert_global_physical_count_to_logical_count(
    # (whatever, num_layers, num_physical_experts)
    global_physical_count: torch.Tensor,
    num_layers: int,
    num_logical_experts: int,
    physical_to_logical_map: torch.Tensor,
):
    dim_extra, _, _ = global_physical_count.shape
    dtype = global_physical_count.dtype
    device = global_physical_count.device
    logical_count = torch.zeros(
        (dim_extra, num_layers, num_logical_experts), dtype=dtype, device=device
    )
    logical_count.scatter_add_(
        dim=2,
        index=physical_to_logical_map.unsqueeze(0).expand(dim_extra, -1, -1),
        src=global_physical_count,
    )
    return logical_count


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
