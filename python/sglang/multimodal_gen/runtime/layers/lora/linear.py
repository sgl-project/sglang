# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Code adapted from SGLang https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py


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
        # Default to True to prevent using uninitialized weights; set to False when weights are loaded
        self.disable_lora: bool = True
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_weights_list: list[
            tuple[torch.nn.Parameter, torch.nn.Parameter, str | None, float]
        ] = []
        self.lora_path: str | None = None
        self.strength: float = 1.0

        self.lora_A = None
        self.lora_B = None

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return getattr(self.base_layer, "bias", None)

    @torch.compile()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lora_A = self.lora_A
        lora_B = self.lora_B
        if isinstance(self.lora_B, DTensor):
            lora_B = self.lora_B.to_local()
            lora_A = self.lora_A.to_local()

        # TODO: Support multiple LoRA adapters when use not merged mode
        if not self.merged and not self.disable_lora:
            lora_A_sliced = self.slice_lora_a_weights(lora_A.to(x, non_blocking=True))
            lora_B_sliced = self.slice_lora_b_weights(lora_B.to(x, non_blocking=True))
            delta = x @ lora_A_sliced.T @ lora_B_sliced.T
            if self.lora_alpha != self.lora_rank:
                delta = delta * (
                    self.lora_alpha / self.lora_rank  # type: ignore
                )  # type: ignore
            delta = delta * self.strength
            if delta.dim() > 2:
                delta = delta.reshape(-1, delta.shape[-1])
            out, output_bias = self.base_layer(x)
            return out + delta, output_bias
        else:
            out, output_bias = self.base_layer(x)
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
        clear_existing: bool = False,
    ) -> None:
        """
        Set LoRA weights. Supports multiple LoRA adapters.

        Args:
            A: LoRA A weight tensor
            B: LoRA B weight tensor
            lora_path: Path to the LoRA adapter (for logging)
            strength: LoRA strength
            clear_existing: If True, clear existing LoRA weights before adding new one.
                          If False, append to existing list (for multi-LoRA support).
        """
        lora_A_param = torch.nn.Parameter(
            A
        )  # share storage with weights in the pipeline
        lora_B_param = torch.nn.Parameter(B)

        if clear_existing:
            self.lora_weights_list.clear()
            # Also clear backward compatibility attributes
            self.lora_A = None
            self.lora_B = None
            self.lora_path = None
            self.strength = 1.0

        # Add to list for multi-LoRA support
        self.lora_weights_list.append((lora_A_param, lora_B_param, lora_path, strength))

        # Set backward compatibility attributes to point to the last LoRA (for single LoRA case)
        # This ensures backward compatibility while supporting multiple LoRA
        self.lora_A = lora_A_param
        self.lora_B = lora_B_param
        self.lora_path = lora_path
        self.strength = strength

        self.disable_lora = False
        self.merge_lora_weights()

    @torch.no_grad()
    def _merge_lora_into_data(
        self,
        data: torch.Tensor,
        lora_list: list[
            tuple[torch.nn.Parameter, torch.nn.Parameter, str | None, float]
        ],
    ) -> None:
        """
        Merge all LoRA adapters into the data tensor in-place.

        Args:
            data: The base weight tensor to merge LoRA into (modified in-place)
            lora_list: List of (lora_A, lora_B, lora_path, lora_strength) tuples
        """
        # Merge all LoRA adapters in order
        for lora_A, lora_B, _, lora_strength in lora_list:
            lora_delta = self.slice_lora_b_weights(
                lora_B.to(data)
            ) @ self.slice_lora_a_weights(lora_A.to(data))
            # Apply lora_alpha / lora_rank scaling for consistency with forward()
            if self.lora_alpha is not None and self.lora_rank is not None:
                if self.lora_alpha != self.lora_rank:
                    lora_delta = lora_delta * (self.lora_alpha / self.lora_rank)
            if lora_delta.dim() > 2:
                lora_delta = lora_delta.reshape(-1, lora_delta.shape[-1])
            data += lora_strength * lora_delta

    @torch.no_grad()
    def merge_lora_weights(self, strength: float | None = None) -> None:
        if strength is not None:
            self.strength = strength

        if self.disable_lora:
            return

        if self.merged:
            self.unmerge_lora_weights()

        # Use lora_weights_list if available, otherwise fall back to single LoRA for backward compatibility
        lora_list = self.lora_weights_list if self.lora_weights_list else []
        if not lora_list and self.lora_A is not None and self.lora_B is not None:
            lora_list = [(self.lora_A, self.lora_B, self.lora_path, self.strength)]

        if not lora_list:
            raise ValueError("LoRA weights not set. Please set them first.")

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

            self._merge_lora_into_data(data, lora_list)

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

            self._merge_lora_into_data(data, lora_list)

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
            old_weight = self.base_layer.weight
            new_weight_data = self.cpu_weight.to(device, non_blocking=True)
            self.base_layer.weight = nn.Parameter(new_weight_data)
            del old_weight
        else:
            current_device = self.base_layer.weight.data.device
            cpu_weight_on_device = self.cpu_weight.to(current_device, non_blocking=True)
            self.base_layer.weight.data.copy_(cpu_weight_on_device)
            if (
                cpu_weight_on_device.data_ptr()
                != self.base_layer.weight.data.data_ptr()
            ):
                del cpu_weight_on_device

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

        # TODO: Support multiple LoRA adapters when use not merged mode
        if not self.merged and not self.disable_lora:
            lora_A_sliced = self.slice_lora_a_weights(lora_A.to(x, non_blocking=True))
            lora_B_sliced = self.slice_lora_b_weights(lora_B.to(x, non_blocking=True))
            delta = x @ lora_A_sliced.T @ lora_B_sliced.T
            if self.lora_alpha != self.lora_rank:
                delta = delta * (
                    self.lora_alpha / self.lora_rank  # type: ignore
                )  # type: ignore
            delta = delta * self.strength
            if delta.dim() > 2:
                delta = delta.reshape(-1, delta.shape[-1])
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
