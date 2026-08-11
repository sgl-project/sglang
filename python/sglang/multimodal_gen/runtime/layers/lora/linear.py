# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Code adapted from SGLang https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py
import os
from abc import abstractmethod
from collections.abc import Iterable
from enum import Enum
from typing import TypeAlias

import torch
from torch import nn
from torch.distributed._composable.fsdp import (
    CPUOffloadPolicy,
    OffloadPolicy,
    fully_shard,
)
from torch.distributed.tensor import DTensor

from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_tp_rank,
    get_tp_world_size,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from sglang.multimodal_gen.utils import get_mixed_precision_state

torch._dynamo.config.recompile_limit = 64
LORA_MERGE_CHUNK_BYTES = 32 * 1024 * 1024

WeightItem: TypeAlias = torch.Tensor | list[torch.Tensor] | None


def _recursive_apply(func, obj, *args, **kwargs):
    """Applies func recursively to tensors in dicts, lists, or tuples, passing extra args."""
    if isinstance(obj, torch.Tensor):
        return func(obj, *args, **kwargs)
    elif isinstance(obj, dict):
        return {k: _recursive_apply(func, v, *args, **kwargs) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_recursive_apply(func, item, *args, **kwargs) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_recursive_apply(func, item, *args, **kwargs) for item in obj)
    else:
        return obj


def _recursive_by_type(data, target_type):
    if isinstance(data, target_type):
        yield data
    if isinstance(data, (str, bytes)):
        return None
    if isinstance(data, dict):
        for value in data.values():
            yield from _recursive_by_type(value, target_type)
    elif isinstance(data, Iterable):
        for item in data:
            yield from _recursive_by_type(item, target_type)


class ParallelLayout(Enum):
    NoneParallelLayout = 0
    RowwiseParallelLayout = 1
    ColwiseParallelLayout = 2


class Adapter:
    def local() -> "Adapter":
        return

    def to_local(weights: dict[str, torch.Tensor]):
        def _to_local(tensor: torch.Tensor):
            return tensor.to_local() if isinstance(tensor, DTensor) else tensor

        return _recursive_apply(_to_local, weights)

    @property
    def dtype(self) -> torch.dtype | None:
        any_item = next(_recursive_by_type(self.weights, torch.Tensor), None)
        return any_item.dtype if any_item is not None else None

    def to(self, *args, **kwargs):
        def _tensor_to(tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            return tensor.to(*args, **kwargs)

        return _recursive_apply(_tensor_to, self.weights, *args, **kwargs)

    def split_weights(
        self, distributed_rank: int, distributed_size: int, layout: ParallelLayout
    ) -> "Adapter":
        return self


class LoRAAdapter(Adapter):

    def __init__(
        self,
        lora_A: WeightItem,
        lora_B: WeightItem,
    ):
        super().__init__()
        self.lora_A = torch.nn.Parameter(lora_A)
        self.lora_B = torch.nn.Parameter(lora_B)

    @property
    def dtype(self) -> torch.dtype | None:
        any_item = next(_recursive_by_type(self.lora_A, torch.Tensor), None)
        return any_item.dtype if any_item is not None else None

    def to(self, *args, **kwargs):
        def _tensor_to(tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            return tensor.to(*args, **kwargs)

        return LoRAAdapter(
            _recursive_apply(_tensor_to, self.lora_A, *args, **kwargs),
            _recursive_apply(_tensor_to, self.lora_B, *args, **kwargs),
        )

    def split_weights(
        self,
        distributed_rank: int,
        distributed_size: int,
        layout: ParallelLayout,
        base_layer,
    ) -> "LoRAAdapter":
        if layout == ParallelLayout.ColwiseParallelLayout:
            shard_size = base_layer.output_partition_sizes[0]
            start_idx = distributed_rank * shard_size
            end_idx = (distributed_rank + 1) * shard_size
            B = self.lora_B[start_idx:end_idx, :]
            return LoRAAdapter(
                self.lora_A,
                B,
            )
        elif layout == ParallelLayout.RowwiseParallelLayout:
            shard_size = base_layer.input_size_per_partition
            start_idx = distributed_rank * shard_size
            end_idx = (distributed_rank + 1) * shard_size
            A = self.lora_A[:, start_idx:end_idx].contiguous()
            return LoRAAdapter(
                A,
                self.lora_B,
            )
        return self

    def delta(self):
        return self.lora_B @ self.lora_A

    def iterate(self, input: torch.Tensor):
        return input @ self.lora_A.T @ self.lora_B.T


class LoKrAdapter(Adapter):
    def __init__(
        self,
        w1: WeightItem,
        w2: WeightItem,
    ):
        super().__init__()
        self.w1 = torch.nn.Parameter(w1)
        self.w2 = torch.nn.Parameter(w2)

    @property
    def dtype(self) -> torch.dtype | None:
        any_item = next(_recursive_by_type(self.w1, torch.Tensor), None)
        return any_item.dtype if any_item is not None else None

    def to(self, *args, **kwargs):
        def _tensor_to(tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            return tensor.to(*args, **kwargs)

        return LoKrAdapter(
            _recursive_apply(_tensor_to, self.w1, *args, **kwargs),
            _recursive_apply(_tensor_to, self.w2, *args, **kwargs),
        )

    def split_weights(
        self,
        distributed_rank: int,
        distributed_size: int,
        layout: ParallelLayout,
        base_layer,
    ) -> "LoRAAdapter":
        if layout == ParallelLayout.RowwiseParallelLayout:
            shard_size = self.w1.shape[0] // distributed_size
            start_idx = distributed_rank * shard_size
            end_idx = (distributed_rank + 1) * shard_size
            w1 = self.w1[start_idx:end_idx, :]
            return LoKrAdapter(
                w1,
                self.w2,
            )
        elif layout == ParallelLayout.ColwiseParallelLayout:
            shard_size = self.w1.shape[1] // distributed_size
            start_idx = distributed_rank * shard_size
            end_idx = (distributed_rank + 1) * shard_size
            w1 = self.w1[:, start_idx:end_idx].contiguous()
            return LoKrAdapter(
                w1,
                self.w2,
            )
        return self

    def delta(self):
        return torch.kron(self.w1, self.w2)

    def iterate(self, input: torch.Tensor):
        return torch.einsum(
            "bpq,pm,qn->bmn",
            input.reshape(input.shape[0], self.w1.shape[0], self.w2.shape[0]),
            self.w1,
            self.w2,
        )


class BaseWeightEntry:
    REGISTRY = {}

    def register(name=None):
        def decorator(cls):
            registry_name = name if name else cls.__name__
            Adapter.REGISTRY[registry_name] = cls
            return cls

        return decorator

    # List of layer fields supported by adapter
    parameter_fields: list[str] = []
    supported_fields: list[str] = [
        *parameter_fields,
    ]
    adapter_class: None

    @classmethod
    @abstractmethod
    def has_adapter(cls, adapter: dict[str, torch.Tensor]) -> bool:
        raise NotImplementedError

    @classmethod
    def has_any_adapter(cls, adapter: dict[str, torch.Tensor]) -> bool:
        return any(
            class_type.has_adapter(adapter) for class_type in cls.REGISTRY.values()
        )

    def __init__(self, lora_path: str | None = None, strength: float = 1.0):
        self.lora_path = lora_path
        self.strength = strength

    @staticmethod
    def create_weight_from_layer(
        adapter: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor] | None:
        for cls in Adapter.REGISTRY.values():
            if cls.Weight.has_adapter(adapter):
                return {
                    supported_field: adapter[supported_field]
                    for supported_field in cls.Weight.supported_fields
                    if supported_field in adapter
                }

        return None


@BaseWeightEntry.register()
class LoRAWeightEntry(BaseWeightEntry):
    parameter_fields: list[str] = ["lora_A", "lora_B"]
    supported_fields: list[str] = [*parameter_fields, "alpha"]

    @classmethod
    def has_adapter(cls, adapter: dict[str, torch.Tensor]) -> bool:
        return all(
            ("lora_A" in adapter),
            ("lora_B" in adapter),
        )

    def __init__(
        self,
        weights,
        lora_path: str | None = None,
        alpha_config: int | None = None,
        strength: float | None = None,
    ):
        super().__init__(lora_path, strength)

        self.weights = LoRAAdapter(weights["lora_A"], weights["lora_B"])

        rank = int(weights["lora_B"].shape[0])
        if "alpha" in weights:
            self.scale = int(weights["alpha"].item()) / rank
        elif alpha_config is not None:
            self.scale = alpha_config / rank
        else:
            self.scale = 1.0


@BaseWeightEntry.register()
class LoKrWeightEntry(BaseWeightEntry):
    parameter_fields: list[str] = [
        "lokr_w1",
        "lokr_w1_a",
        "lokr_w1_b",
        "lokr_w2",
        "lokr_w2_a",
        "lokr_w2_b",
        "lokr_t2",
    ]
    supported_fields: list[str] = [*parameter_fields, "alpha"]

    @classmethod
    def has_adapter(cls, adapter: dict[str, torch.Tensor]) -> bool:
        return all(
            any(
                ("lokr_w1" in adapter),
                all(
                    ("lokr_w1_a" in adapter),
                    ("lokr_w1_b" in adapter),
                ),
            ),
            any(
                ("lokr_w2" in adapter),
                all(
                    ("lokr_w2_a" in adapter),
                    ("lokr_w2_b" in adapter),
                ),
            ),
        )

    def __init__(
        self,
        weights,
        lora_path: str | None = None,
        alpha_config: int | None = None,
        strength: float | None = None,
    ):
        super().__init__(lora_path, strength)

        if "lokr_w1" in weights:
            w1 = weights["lokr_w1"]
        else:
            w1 = weights["lokr_w1_a"] @ weights["lokr_w1_b"]

        if "lokr_w2" in weights:
            w2 = weights["lokr_w2"]
        elif "lokr_t2" in weights:
            w2 = torch.einsum(
                "i j k l, j r, i p -> p r k l",
                weights["lokr_t2"],
                weights["lokr_w2_b"],
                weights["lokr_w2_a"],
            )
        else:
            w2 = weights["lokr_w2_a"] @ weights["lokr_w2_b"]

        self.weights = LoKrAdapter(w1, w2)


def create_weight_from_layer(
    adapter: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor] | None:
    for cls in BaseWeightEntry.REGISTRY.values():
        if cls.has_adapter(adapter):
            return {
                supported_field: adapter[supported_field]
                for supported_field in cls.Weight.supported_fields
                if supported_field in adapter
            }

    return None


def has_any_registered_adapter(adapter: dict[str, torch.Tensor]) -> bool:
    return BaseWeightEntry.has_any_adapter(adapter)


class BaseLayerWithLoRA(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
    ):
        super().__init__()
        self.base_layer: nn.Module = base_layer

        self.merged: bool = False
        # Immutable base-weight snapshot; `to("cpu")` may alias CPU storage.
        # Use `clone()` so merge updates cannot mutate this backup tensor.

        self.cpu_weight = base_layer.weight.detach().to("cpu").clone()
        # indicates adapter weights don't contain this layer
        # (which shouldn't normally happen, but we want to separate it from the case of erroneous merging)
        # Default to True to prevent using uninitialized weights; set to False when weights are loaded
        self.disable_lora: bool = True

        self.global_strength: float = 1.0
        self.lora_weights_list: list[BaseWeightEntry] = []

    @property
    def has_weight(self) -> bool:
        return (
            self.lora_weights_list[-1].has_weight if self.lora_weights_list else False
        )

    @property
    def lora_path(self) -> str | None:
        return self.lora_weights_list[-1].lora_path if self.lora_weights_list else None

    @property
    def strength(self) -> float | None:
        return (
            self.lora_weights_list[-1].strength
            if self.lora_weights_list
            else self.global_strength
        )

    @property
    def weights(self):
        last_strength = (
            self.lora_weights_list[-1].weights if self.lora_weights_list else None
        )

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return getattr(self.base_layer, "bias", None)

    @torch.compile()
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.merged or self.disable_lora:
            return self.base_layer(input_)

        weights = self.weights.to_local()

        input_parallel = self.base_layer.parallel_input(input_)
        output_parallel, output_bias = self.base_layer.apply_quant_method(
            input_parallel, self.base_layer.bias
        )

        lora_dtype = weights.dtype
        input_parallel_lora = input_parallel.to(dtype=lora_dtype)
        weights_sliced = self.slice_weights(
            weights.to(device=input_parallel.device, non_blocking=True)
        )

        delta_parallel = weights_sliced.iterated_delta(input_parallel_lora)
        delta_parallel *= self.strength
        output_parallel += delta_parallel.to(dtype=output_parallel.dtype)

        output = self.base_layer.collect_output(output_parallel)
        return output, output_bias

    def slice_weights(self, weight: Adapter) -> Adapter:
        return weight

    @staticmethod
    def _as_mutable_tensor(tensor: torch.Tensor) -> torch.Tensor:
        # lora can be reconfigured after executor forwards create inference tensors
        if tensor.is_inference():
            with torch.inference_mode(False):
                return tensor.detach().clone()
        return tensor

    def set_adapter_weights(
        self,
        adapter_weights,
        lora_path: str | None = None,
        strength: float = 1.0,
        clear_existing: bool = False,
        merge_weights: bool = True,
    ) -> None:
        """
        Set adapter weights. Supports multiple adapters.

        Args:
            weights: weight tensors for adapter
            lora_path: Path to the LoRA adapter (for logging)
            strength: LoRA strength
            clear_existing: If True, clear existing LoRA weights before adding new one.
                            If False, append to existing list (for multi-LoRA support).
        """
        # share storage with weights in the pipeline
        adapter = BaseWeightEntry.create_weight_from_layer(adapter_weights)

        if clear_existing:
            self.lora_weights_list.clear()

        # Add to list for multi-LoRA support
        self.lora_weights_list.append(adapter)

        self.disable_lora = False
        if merge_weights:
            self.merge_lora_weights()
        elif self.merged:
            self.unmerge_lora_weights()

    @torch.no_grad()
    def _merge_lora_into_data(
        self,
        data: torch.Tensor,
        lora_list: list[BaseWeightEntry],
    ) -> None:
        """
        Merge all LoRA adapters into the data tensor in-place.

        Args:
            data: The base weight tensor to merge LoRA into (modified in-place)
            lora_list: List of (lora_A, lora_B, lora_path, lora_strength, rank, alpha) tuples
        """
        # Merge all LoRA adapters in order
        for adapter in lora_list:
            weight_sliced = self.slice_weights(adapter.to(data))

            # has_adapter_weights
            # adapter_params
            scale = adapter.lora_strength

            lora_delta = adapter.delta()
            if isinstance(lora_delta, torch.Tensor) and lora_delta.dim() > 2:
                lora_delta = lora_delta.reshape(-1, lora_delta.shape[-1])
            data.add_(lora_delta, alpha=scale)

    def _should_merge_in_fp32(
        self,
        lora_list: list[BaseWeightEntry],
    ) -> bool:
        if os.getenv("SGLANG_DIFFUSION_LORA_MERGE_FP32", "1") != "1":
            return False
        for lora in lora_list:
            if lora.lora_path and "distilled-lora" in lora.lora_path.lower():
                return False
        return True

    @torch.no_grad()
    def merge_lora_weights(self, strength: float | None = None) -> None:
        if strength is not None:
            self.strength = strength
            if self.lora_weights_list:
                for lora in self.lora_weights_list:
                    lora.strength = strength

        if self.disable_lora:
            return

        if self.merged:
            self.unmerge_lora_weights()

        # Use lora_weights_list if available, otherwise fall back to single LoRA for backward compatibility
        lora_list = self.lora_weights_list if self.lora_weights_list else []
        if not lora_list:
            raise ValueError("LoRA weights not set. Please set them first.")

        merge_in_fp32 = self._should_merge_in_fp32(lora_list)

        if isinstance(self.base_layer.weight, DTensor):
            mesh = self.base_layer.weight.data.device_mesh
            unsharded_base_layer = ReplicatedLinear(
                input_size=self.base_layer.input_size,
                output_size=self.base_layer.output_size,
                bias=getattr(self.base_layer, "bias", None) is not None,
                skip_bias_add=self.base_layer.skip_bias_add,
                params_dtype=self.base_layer.params_dtype,
                quant_config=self.base_layer.quant_config,
                prefix=self.base_layer.prefix,
            )
            # Using offload param is on CPU, so current_device is for "CPU -> GPU -> merge -> CPU"
            current_device = self.base_layer.weight.data.device
            data = self.base_layer.weight.data.to(
                get_local_torch_device()
            ).full_tensor()
            data = self._as_mutable_tensor(data)
            target_dtype = data.dtype
            if (
                merge_in_fp32
                and data.is_floating_point()
                and data.dtype != torch.float32
            ):
                data = data.to(torch.float32)

            self._merge_lora_into_data(data, lora_list)

            unsharded_base_layer.weight = nn.Parameter(
                self._as_mutable_tensor(data.to(current_device, dtype=target_dtype))
            )
            if isinstance(getattr(self.base_layer, "bias", None), DTensor):
                bias_data = (
                    self.base_layer.bias.to(get_local_torch_device(), non_blocking=True)
                    .full_tensor()
                    .to(current_device)
                )
                unsharded_base_layer.bias = nn.Parameter(
                    self._as_mutable_tensor(bias_data)
                )

            offload_policy = (
                CPUOffloadPolicy() if "cpu" in str(current_device) else OffloadPolicy()
            )
            mp_policy = get_mixed_precision_state().mp_policy

            self.base_layer = fully_shard(
                unsharded_base_layer,
                mesh=mesh,
                mp_policy=mp_policy,
                offload_policy=offload_policy,
            )
        else:
            current_device = self.base_layer.weight.data.device
            data = self.base_layer.weight.data.to(get_local_torch_device())
            data = self._as_mutable_tensor(data)
            target_dtype = data.dtype
            if (
                merge_in_fp32
                and data.is_floating_point()
                and data.dtype != torch.float32
            ):
                data = data.to(torch.float32)

            self._merge_lora_into_data(data, lora_list)

            self.base_layer.weight.data = self._as_mutable_tensor(
                data.to(current_device, dtype=target_dtype, non_blocking=True)
            )

        self.merged = True

    @torch.no_grad()
    # @torch.compile(dynamic=True)
    def unmerge_lora_weights(self) -> None:
        if self.disable_lora:
            return

        if not self.merged:
            raise ValueError(
                "LoRA weights not merged. Please merge them first before unmerging."
            )

        # avoid precision loss
        if isinstance(self.base_layer.weight, DTensor):
            device = self.base_layer.weight.data.device
            old_weight = self.base_layer.weight
            new_weight_data = self._as_mutable_tensor(
                self.cpu_weight.to(device, non_blocking=True)
            )
            self.base_layer.weight = nn.Parameter(new_weight_data)
            del old_weight
        else:
            current_device = self.base_layer.weight.data.device
            cpu_weight_on_device = self.cpu_weight.to(current_device, non_blocking=True)
            if self.base_layer.weight.data.is_inference():
                self.base_layer.weight.data = self._as_mutable_tensor(
                    cpu_weight_on_device
                )
            else:
                self.base_layer.weight.data.copy_(cpu_weight_on_device)
            if (
                cpu_weight_on_device.data_ptr()
                != self.base_layer.weight.data.data_ptr()
            ):
                del cpu_weight_on_device

        self.merged = False

    @torch.no_grad()
    def commit_merged_as_base(self) -> None:
        """Promote the currently merged weights to the permanent base.

        Re-snapshots ``cpu_weight`` so the merged weights become the restore
        target and resets adapter bookkeeping (``merged=False``). A later dynamic
        ``set_adapter_weights`` then adds its delta on top of the merged base instead
        of unmerging it.
        """
        if not self.merged:
            return
        weight = self.base_layer.weight
        if isinstance(weight, DTensor):
            weight = weight.to_local()
        # clone(): to("cpu") may alias storage; we must not mutate this backup.
        self.cpu_weight = weight.detach().to("cpu").clone()
        self.merged = False
        self.disable_lora = True
        self.lora_weights_list = []


class VocabParallelEmbeddingWithLoRA(BaseLayerWithLoRA):
    """
    Vocab parallel embedding layer with support for LoRA (Low-Rank Adaptation).

    Note: The current version does not yet implement the LoRA functionality.
    This class behaves exactly the same as the base VocabParallelEmbedding.
    Future versions will integrate LoRA functionality to support efficient parameter fine-tuning.
    """

    def __init__(
        self,
        base_layer: VocabParallelEmbedding,
    ) -> None:
        super().__init__(base_layer)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "We don't support VocabParallelEmbeddingWithLoRA yet."
        )


class ColumnParallelLinearWithLoRA(BaseLayerWithLoRA):
    def __init__(
        self,
        base_layer: ColumnParallelLinear,
    ) -> None:
        super().__init__(base_layer)

    def slice_weights(self, weights: Adapter) -> Adapter:
        return weights.split_weights(
            get_tp_rank(),
            get_tp_world_size(),
            ParallelLayout.ColwiseParallelLayout,
            self.base_layer,
        )


class MergedColumnParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    def __init__(
        self,
        base_layer: MergedColumnParallelLinear,
    ) -> None:
        super().__init__(base_layer)

    def slice_weights(self, weights: Adapter) -> Adapter:
        return weights.split_weights(
            get_tp_rank(),
            get_tp_world_size(),
            ParallelLayout.ColwiseParallelLayout,
            self.base_layer,
        )


class QKVParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    def __init__(
        self,
        base_layer: QKVParallelLinear,
    ) -> None:
        super().__init__(base_layer)

    def slice_weights(self, weights: Adapter) -> Adapter:
        return weights.split_weights(
            get_tp_rank(),
            get_tp_world_size(),
            ParallelLayout.ColwiseParallelLayout,
            self.base_layer,
        )


class RowParallelLinearWithLoRA(BaseLayerWithLoRA):
    def __init__(
        self,
        base_layer: RowParallelLinear,
    ) -> None:
        super().__init__(base_layer)

    def slice_weights(self, weights: Adapter) -> Adapter:
        return weights.split_weights(
            get_tp_rank(),
            get_tp_world_size(),
            ParallelLayout.RowwiseParallelLayout,
            self.base_layer,
        )


class LinearWithLoRA(BaseLayerWithLoRA):
    """
    Wrapper for standard torch.nn.Linear to support LoRA.
    Unlike custom LinearBase classes, nn.Linear.forward() returns a single tensor,
    not a tuple of (output, bias).
    """

    def __init__(
        self,
        base_layer: nn.Linear,
    ) -> None:
        super().__init__(base_layer)

    @torch.compile()
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.merged or self.disable_lora:
            return self.base_layer(input_)

        weights = self.weights.local()

        output = self.base_layer(input_)

        lora_dtype = weights.dtype
        input_lora = input_.to(dtype=lora_dtype)
        weights_sliced = self.slice_weights(weights)
        delta = weights_sliced.iterate(input_lora)
        delta *= self.strength

        output += delta.to(dtype=output.dtype)
        return output


def wrap_with_lora_layer(
    layer: nn.Module,
) -> BaseLayerWithLoRA | None:
    """
    transform the given layer to its corresponding LoRA layer
    """
    supported_layer_types: dict[
        type[LinearBase] | type[nn.Linear], type[BaseLayerWithLoRA]
    ] = {
        # the order matters
        # VocabParallelEmbedding: VocabParallelEmbeddingWithLoRA,
        QKVParallelLinear: QKVParallelLinearWithLoRA,
        MergedColumnParallelLinear: MergedColumnParallelLinearWithLoRA,
        ColumnParallelLinear: ColumnParallelLinearWithLoRA,
        RowParallelLinear: RowParallelLinearWithLoRA,
        ReplicatedLinear: BaseLayerWithLoRA,
        nn.Linear: LinearWithLoRA,
    }
    for src_layer_type, lora_layer_type in supported_layer_types.items():
        if isinstance(layer, src_layer_type):  # type: ignore[arg-type]
            ret = lora_layer_type(
                layer,
            )
            return ret
    return None


# source: https://github.com/vllm-project/vllm/blob/93b38bea5dd03e1b140ca997dfaadef86f8f1855/vllm/lora/utils.py#L9
def replace_submodule(
    model: nn.Module, module_name: str, new_module: nn.Module
) -> nn.Module:
    """Replace a submodule in a model with a new module."""
    parent = model.get_submodule(".".join(module_name.split(".")[:-1]))
    target_name = module_name.split(".")[-1]
    setattr(parent, target_name, new_module)
    return new_module
