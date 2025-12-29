# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Code adapted from SGLang https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py


from __future__ import annotations

import logging
from collections import defaultdict
from typing import TypedDict

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
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
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

torch._dynamo.config.recompile_limit = 16

logger = logging.getLogger(__name__)


class LoRAAdapterConfig(TypedDict):
    alpha: float
    rank: int


class BaseLayerWithLoRA(nn.Module):

    def __init__(
        self,
        base_layer: nn.Module,
        lora_rank: int | None = None,
        lora_alpha: int | None = None,
    ):
        super().__init__()
        self.base_layer: nn.Module = base_layer

        self.merged: bool = False
        self.cpu_weight = base_layer.weight.to("cpu")
        # indicates adapter weights don't contain this layer
        # (which shouldn't normally happen, but we want to separate it from the case of erroneous merging)
        # Default to True to prevent using uninitialized LoRA weights.
        # This will be set to False once LoRA weights are actually loaded for this layer.
        self.disable_lora: bool = True
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_path: str | None = None
        # Optional: set by the pipeline when wrapping, used for more helpful warnings.
        self.layer_name: str | None = None
        self.strength: float = 1.0

        # Legacy single-LoRA support
        self.lora_A = None
        self.lora_B = None

        # Multi-LoRA support
        self.lora_weights_pool: dict[str, tuple[torch.Tensor, torch.Tensor]] = (
            {}
        )  # nickname -> (lora_A, lora_B)
        self.lora_adapter_configs: dict[str, LoRAAdapterConfig] = (
            {}
        )  # nickname -> {alpha, rank} for per-adapter config
        self.active_lora_indices: torch.Tensor | None = None  # Per-sample LoRA index
        self.lora_nickname_to_index: dict[str, int] = {}  # Mapping nickname to index
        self.max_loras: int = 8
        self.use_multi_lora: bool = False

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return getattr(self.base_layer, "bias", None)

    def prepare_multi_lora_batch(
        self,
        lora_weights_pool: dict[str, tuple[torch.Tensor, torch.Tensor]],
        lora_nickname_to_index: dict[str, int],
        lora_adapter_configs: dict[str, LoRAAdapterConfig] | None = None,
    ):
        """
        Prepare LoRA weights for multi-LoRA batch.

        Args:
            lora_weights_pool: Dict mapping lora_nickname -> (lora_A, lora_B)
            lora_nickname_to_index: Dict mapping lora_nickname -> index
            lora_adapter_configs: Dict mapping lora_nickname -> {alpha, rank} for per-adapter config
        """
        self.lora_weights_pool = lora_weights_pool
        self.lora_nickname_to_index = lora_nickname_to_index
        self.lora_adapter_configs = lora_adapter_configs or {}
        self.use_multi_lora = len(lora_weights_pool) > 0

    def set_multi_lora_indices(self, indices: torch.Tensor | list[int] | None):
        """
        Set LoRA indices for each sample in the batch.

        Args:
            indices: Tensor or list of LoRA indices for each sample (-1 for no LoRA)
        """
        if indices is None:
            self.active_lora_indices = None
            self.use_multi_lora = False
        else:
            if isinstance(indices, list):
                indices = torch.tensor(
                    indices,
                    device=self.base_layer.weight.device,
                    dtype=torch.int32,
                )
            self.active_lora_indices = indices
            self.use_multi_lora = True

    def _index_to_nickname(self, index: int) -> str | None:
        """Convert LoRA index to nickname."""
        for nickname, idx in self.lora_nickname_to_index.items():
            if idx == index:
                return nickname
        return None

    def _apply_multi_lora(
        self, x: torch.Tensor, base_out: torch.Tensor, output_bias
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply LoRA adaptively per sample in batch."""

        if self.active_lora_indices is None or len(self.lora_weights_pool) == 0:
            return base_out, output_bias

        device = x.device

        # Group requests by LoRA to minimize computation
        lora_groups: dict[int, list[int]] = defaultdict(list)
        for i, lora_idx in enumerate(self.active_lora_indices.cpu().tolist()):
            lora_groups[lora_idx].append(i)

        delta = torch.zeros_like(base_out)

        for lora_idx, sample_indices in lora_groups.items():
            if lora_idx < 0:
                continue

            lora_nickname = self._index_to_nickname(lora_idx)
            if lora_nickname is None:
                logger.warning(
                    "LoRA index %d not found in nickname mapping for layer %s",
                    lora_idx,
                    self.layer_name or self.__class__.__name__,
                )
                continue
            if lora_nickname not in self.lora_weights_pool:
                logger.warning(
                    "LoRA adapter '%s' not found in weights pool for layer %s",
                    lora_nickname,
                    self.layer_name or self.__class__.__name__,
                )
                continue

            lora_A, lora_B = self.lora_weights_pool[lora_nickname]
            lora_A = lora_A.to(device, non_blocking=True)
            lora_B = lora_B.to(device, non_blocking=True)

            # Slice for tensor parallelism if needed
            lora_A_sliced = self.slice_lora_a_weights(lora_A)
            lora_B_sliced = self.slice_lora_b_weights(lora_B)

            x_group = x[sample_indices]
            delta_group = x_group @ lora_A_sliced.T @ lora_B_sliced.T

            adapter_config = self.lora_adapter_configs.get(lora_nickname, {})
            alpha = adapter_config.get("alpha", self.lora_alpha or 16.0)
            rank = adapter_config.get("rank", self.lora_rank or 16)
            if alpha != rank:
                delta_group = delta_group * (alpha / rank)

            delta[sample_indices] = delta_group

        return base_out + delta, output_bias

    @torch.compile()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, output_bias = self.base_layer(x)

        if (
            self.use_multi_lora
            and self.active_lora_indices is not None
            and not self.disable_lora
        ):
            out, output_bias = self._apply_multi_lora(x, out, output_bias)
            return out, output_bias

        if self.lora_A is not None and not self.merged and not self.disable_lora:
            lora_A = self.lora_A
            lora_B = self.lora_B
            if isinstance(lora_B, DTensor):
                lora_B = lora_B.to_local()
                lora_A = lora_A.to_local()

            lora_A_sliced = self.slice_lora_a_weights(lora_A.to(x, non_blocking=True))
            lora_B_sliced = self.slice_lora_b_weights(lora_B.to(x, non_blocking=True))
            delta = x @ lora_A_sliced.T @ lora_B_sliced.T
            if (
                self.lora_alpha is not None
                and self.lora_rank is not None
                and self.lora_alpha != self.lora_rank
            ):
                delta = delta * (self.lora_alpha / self.lora_rank)
            out = out + (self.strength * delta)

        return out, output_bias

    def slice_lora_a_weights(self, A: torch.Tensor) -> torch.Tensor:
        return A

    def slice_lora_b_weights(self, B: torch.Tensor) -> torch.Tensor:
        return B

    def set_lora_weights(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        lora_path: str | None = None,
        strength: float = 1.0,
    ) -> None:
        self.lora_A = torch.nn.Parameter(
            A
        )  # share storage with weights in the pipeline
        self.lora_B = torch.nn.Parameter(B)
        self.disable_lora = False
        self.strength = strength
        self.merge_lora_weights()
        self.lora_path = lora_path

    @torch.no_grad()
    def merge_lora_weights(self, strength: float | None = None) -> None:
        if strength is not None:
            self.strength = strength

        if self.disable_lora:
            return

        if self.merged:
            self.unmerge_lora_weights()
        assert (
            self.lora_A is not None and self.lora_B is not None
        ), "LoRA weights not set. Please set them first."
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
            lora_delta = self.slice_lora_b_weights(self.lora_B).to(
                data
            ) @ self.slice_lora_a_weights(self.lora_A).to(data)
            # Apply lora_alpha / lora_rank scaling for consistency with forward()
            if self.lora_alpha is not None and self.lora_rank is not None:
                if self.lora_alpha != self.lora_rank:
                    lora_delta = lora_delta * (self.lora_alpha / self.lora_rank)
            data += self.strength * lora_delta
            unsharded_base_layer.weight = nn.Parameter(data.to(current_device))
            if isinstance(getattr(self.base_layer, "bias", None), DTensor):
                unsharded_base_layer.bias = nn.Parameter(
                    self.base_layer.bias.to(get_local_torch_device(), non_blocking=True)
                    .full_tensor()
                    .to(current_device)
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
            lora_delta = self.slice_lora_b_weights(
                self.lora_B.to(data)
            ) @ self.slice_lora_a_weights(self.lora_A.to(data))
            # Apply lora_alpha / lora_rank scaling for consistency with forward()
            if self.lora_alpha is not None and self.lora_rank is not None:
                if self.lora_alpha != self.lora_rank:
                    lora_delta = lora_delta * (self.lora_alpha / self.lora_rank)
            data += self.strength * lora_delta
            self.base_layer.weight.data = data.to(current_device, non_blocking=True)

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
            self.base_layer.weight = nn.Parameter(
                self.cpu_weight.to(device, non_blocking=True)
            )
        else:
            self.base_layer.weight.data = self.cpu_weight.data.to(
                self.base_layer.weight, non_blocking=True
            )

        self.merged = False


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
        lora_rank: int | None = None,
        lora_alpha: int | None = None,
    ) -> None:
        super().__init__(base_layer, lora_rank, lora_alpha)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # duplicate the logic in ColumnParallelLinear
        bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
        output_parallel = self.base_layer.quant_method.apply(
            self.base_layer, input_, bias
        )
        if self.base_layer.gather_output:
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
        return output, output_bias

    def slice_lora_a_weights(self, A: torch.Tensor) -> torch.Tensor:
        return A

    def slice_lora_b_weights(self, B: torch.Tensor) -> torch.Tensor:
        tp_rank = get_tp_rank()
        shard_size = self.base_layer.output_partition_sizes[0]
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        B = B[start_idx:end_idx, :]
        return B


class MergedColumnParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):

    def __init__(
        self,
        base_layer: MergedColumnParallelLinear,
        lora_rank: int | None = None,
        lora_alpha: int | None = None,
    ) -> None:
        super().__init__(base_layer, lora_rank, lora_alpha)

    def slice_lora_a_weights(self, A: torch.Tensor) -> torch.Tensor:
        return A.to(self.base_layer.weight)

    def slice_lora_b_weights(self, B: torch.Tensor) -> torch.Tensor:
        tp_rank = get_tp_rank()
        # Since the outputs for both gate and up are identical, we use a random one.
        shard_size = self.base_layer.output_partition_sizes[0]
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        return B[:, start_idx:end_idx, :]


class QKVParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):

    def __init__(
        self,
        base_layer: QKVParallelLinear,
        lora_rank: int | None = None,
        lora_alpha: int | None = None,
    ) -> None:
        super().__init__(base_layer, lora_rank, lora_alpha)

    def slice_lora_a_weights(self, A: torch.Tensor) -> torch.Tensor:
        return A

    def slice_lora_b_weights(
        self, B: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tp_rank = get_tp_rank()
        B_q, B_kv = B
        base_layer = self.base_layer
        q_proj_shard_size = base_layer.q_proj_shard_size
        kv_proj_shard_size = base_layer.kv_proj_shard_size
        num_kv_head_replicas = base_layer.num_kv_head_replicas

        q_start_idx = q_proj_shard_size * tp_rank
        q_end_idx = q_start_idx + q_proj_shard_size

        kv_shard_id = tp_rank // num_kv_head_replicas
        kv_start_idx = kv_proj_shard_size * kv_shard_id
        kv_end_idx = kv_start_idx + kv_proj_shard_size

        return B_q[q_start_idx:q_end_idx, :], B_kv[:, kv_start_idx:kv_end_idx, :]


class RowParallelLinearWithLoRA(BaseLayerWithLoRA):

    def __init__(
        self,
        base_layer: RowParallelLinear,
        lora_rank: int | None = None,
        lora_alpha: int | None = None,
    ) -> None:
        super().__init__(base_layer, lora_rank, lora_alpha)

    def forward(self, input_: torch.Tensor):
        # duplicate the logic in RowParallelLinear
        if self.base_layer.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_tp_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.base_layer.tp_size
            )
            input_parallel = splitted_input[tp_rank].contiguous()
        output_parallel = self.base_layer.quant_method.apply(
            self.base_layer, input_parallel
        )

        if self.base_layer.reduce_results and self.base_layer.tp_size > 1:
            output_ = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output_ = output_parallel

        if not self.base_layer.skip_bias_add:
            output = (
                output_ + self.base_layer.bias
                if self.base_layer.bias is not None
                else output_
            )
            output_bias = None
        else:
            output = output_
            output_bias = self.base_layer.bias
        return output, output_bias

    def slice_lora_a_weights(self, A: torch.Tensor) -> torch.Tensor:
        tp_rank = get_tp_rank()
        shard_size = self.base_layer.input_size_per_partition
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        A = A[:, start_idx:end_idx].contiguous()
        return A

    def slice_lora_b_weights(self, B: torch.Tensor) -> torch.Tensor:
        return B


class LinearWithLoRA(BaseLayerWithLoRA):
    """
    Wrapper for standard torch.nn.Linear to support LoRA.
    Unlike custom LinearBase classes, nn.Linear.forward() returns a single tensor,
    not a tuple of (output, bias).
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        lora_rank: int | None = None,
        lora_alpha: int | None = None,
    ) -> None:
        super().__init__(base_layer, lora_rank, lora_alpha)

    @torch.compile()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lora_A = self.lora_A
        lora_B = self.lora_B
        if isinstance(self.lora_B, DTensor):
            lora_B = self.lora_B.to_local()
            lora_A = self.lora_A.to_local()

        if not self.merged and not self.disable_lora:
            lora_A_sliced = self.slice_lora_a_weights(lora_A.to(x, non_blocking=True))
            lora_B_sliced = self.slice_lora_b_weights(lora_B.to(x, non_blocking=True))
            delta = x @ lora_A_sliced.T @ lora_B_sliced.T
            if self.lora_alpha != self.lora_rank:
                delta = delta * (
                    self.lora_alpha / self.lora_rank  # type: ignore
                )  # type: ignore
            delta = delta * self.strength
            # nn.Linear.forward() returns a single tensor, not a tuple
            out = self.base_layer(x)
            return out + delta
        else:
            # nn.Linear.forward() returns a single tensor
            out = self.base_layer(x)
            return out


def wrap_with_lora_layer(
    layer: nn.Module,
    lora_rank: int | None = None,
    lora_alpha: int | None = None,
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
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
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
