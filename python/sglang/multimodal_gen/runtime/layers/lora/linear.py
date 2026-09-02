# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Code adapted from SGLang https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py
import os

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

torch._dynamo.config.recompile_limit = 64


LORA_MERGE_CHUNK_BYTES = 32 * 1024 * 1024
LoRAWeightEntry = tuple[
    torch.nn.Parameter,
    torch.nn.Parameter,
    str | None,
    float,
    int | None,
    int | None,
    torch.nn.Parameter | None,
]


def _compute_lora_delta(
    x: torch.Tensor, lora_A: torch.Tensor, lora_B: torch.Tensor
) -> torch.Tensor:
    """Apply a regular or stacked LoRA projection to the last dimension."""
    if lora_A.dim() == 2 and lora_B.dim() == 2:
        return x @ lora_A.T @ lora_B.T
    if lora_A.dim() == 3 and lora_B.dim() == 3:
        if lora_A.shape[0] != lora_B.shape[0]:
            raise ValueError(
                "Stacked LoRA A/B projections must have the same group count, got "
                f"{lora_A.shape[0]} and {lora_B.shape[0]}"
            )
        hidden = torch.einsum("...i,nri->...nr", x, lora_A)
        delta = torch.einsum("...nr,nor->...no", hidden, lora_B)
        return delta.flatten(start_dim=-2)
    raise ValueError(
        "LoRA A/B projections must both be 2D or both be 3D, got "
        f"{tuple(lora_A.shape)} and {tuple(lora_B.shape)}"
    )


class BaseLayerWithLoRA(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        lora_rank: int | None = None,
        lora_alpha: int | None = None,
        snapshot_base: bool = True,
    ):
        super().__init__()
        self.base_layer: nn.Module = base_layer

        self.merged: bool = False
        # Immutable base-weight snapshot; `to("cpu")` may alias CPU storage.
        # Use `clone()` so in-place merge updates cannot mutate this backup.
        # With snapshot_base=False the snapshot is a zero-copy view instead:
        # valid only while every merge on this layer is a copy-merge (the
        # merged-store path), which never writes the base storage. H3's DiT
        # backup alone is 38 GB of anonymous memory under clone().
        if snapshot_base:
            self.cpu_weight = base_layer.weight.detach().to("cpu").clone()
            self._base_is_view = False
        else:
            self.cpu_weight = base_layer.weight.detach()
            self._base_is_view = True
        # indicates adapter weights don't contain this layer
        # (which shouldn't normally happen, but we want to separate it from the case of erroneous merging)
        # Default to True to prevent using uninitialized weights; set to False when weights are loaded
        self.disable_lora: bool = True
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_weights_list: list[LoRAWeightEntry] = []
        self.lora_path: str | None = None
        self.strength: float = 1.0

        self.lora_A = None
        self.lora_B = None
        self.lora_output_offset = None
        self.has_lora_output_offset = False

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
            lora_dtype = lora_A.dtype
            x_lora = x.to(dtype=lora_dtype)
            lora_A_sliced = self.slice_lora_a_weights(
                lora_A.to(device=x.device, non_blocking=True)
            )
            lora_B_sliced = self.slice_lora_b_weights(
                lora_B.to(device=x.device, non_blocking=True)
            )
            delta = _compute_lora_delta(x_lora, lora_A_sliced, lora_B_sliced)
            if self.lora_alpha != self.lora_rank:
                delta = delta * (
                    self.lora_alpha / self.lora_rank  # type: ignore
                )  # type: ignore
            delta = delta * self.strength
            out, output_bias = self.base_layer(x)
            out = out + delta.to(dtype=out.dtype)
        else:
            out, output_bias = self.base_layer(x)
        return self._add_lora_output_offset(out), output_bias

    def slice_lora_a_weights(self, A: torch.Tensor) -> torch.Tensor:
        return A

    def slice_lora_b_weights(self, B: torch.Tensor) -> torch.Tensor:
        return B

    def _scaled_lora_output_offset(
        self,
        offset: torch.Tensor | None,
        strength: float,
        rank: int | None,
        alpha: int | None,
    ) -> torch.Tensor | None:
        if offset is None:
            return None
        offset = self.slice_lora_b_weights(offset.unsqueeze(-1)).squeeze(-1)
        scale = strength
        if rank is not None and alpha is not None and rank != alpha:
            scale *= alpha / rank
        return offset if scale == 1.0 else offset * scale

    def _active_lora_output_offset(self) -> torch.Tensor | None:
        if self.disable_lora or not self.has_lora_output_offset:
            return None
        if not self.merged:
            return self._scaled_lora_output_offset(
                self.lora_output_offset,
                self.strength,
                self.lora_rank,
                self.lora_alpha,
            )
        combined = None
        for _, _, _, strength, rank, alpha, offset in self.lora_weights_list:
            scaled = self._scaled_lora_output_offset(offset, strength, rank, alpha)
            if scaled is not None:
                combined = scaled if combined is None else combined + scaled
        return combined

    def _add_lora_output_offset(self, output: torch.Tensor) -> torch.Tensor:
        offset = self._active_lora_output_offset()
        if offset is None:
            return output
        return output + offset.to(device=output.device, dtype=output.dtype)

    @staticmethod
    def _as_mutable_tensor(tensor: torch.Tensor) -> torch.Tensor:
        # lora can be reconfigured after executor forwards create inference tensors
        if tensor.is_inference():
            with torch.inference_mode(False):
                return tensor.detach().clone()
        return tensor

    def set_lora_weights(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        lora_path: str | None = None,
        strength: float = 1.0,
        clear_existing: bool = False,
        merge_weights: bool = True,
        output_offset: torch.Tensor | None = None,
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
            output_offset: Optional constant output term paired with this adapter
        """
        lora_A_param = torch.nn.Parameter(
            A
        )  # share storage with weights in the pipeline
        lora_B_param = torch.nn.Parameter(B)
        output_offset_param = (
            torch.nn.Parameter(output_offset, requires_grad=False)
            if output_offset is not None
            else None
        )

        if clear_existing:
            self.lora_weights_list.clear()
            # Also clear backward compatibility attributes
            self.lora_A = None
            self.lora_B = None
            self.lora_output_offset = None
            self.has_lora_output_offset = False
            self.lora_path = None
            self.strength = 1.0

        # Add to list for multi-LoRA support
        self.lora_weights_list.append(
            (
                lora_A_param,
                lora_B_param,
                lora_path,
                strength,
                self.lora_rank,
                self.lora_alpha,
                output_offset_param,
            )
        )

        # Set backward compatibility attributes to point to the last LoRA (for single LoRA case)
        # This ensures backward compatibility while supporting multiple LoRA
        self.lora_A = lora_A_param
        self.lora_B = lora_B_param
        self.lora_output_offset = output_offset_param
        self.has_lora_output_offset |= output_offset_param is not None
        self.lora_path = lora_path
        self.strength = strength

        self.disable_lora = False
        if merge_weights:
            self.merge_lora_weights()
        elif self.merged:
            self.unmerge_lora_weights()

    def _ensure_base_snapshot_owned(self) -> None:
        """An in-place merge is about to write the base storage; if the
        snapshot is a zero-copy view into it, materialize the clone now."""
        if self._base_is_view:
            self.cpu_weight = self.cpu_weight.clone()
            self._base_is_view = False

    @torch.no_grad()
    def _merge_lora_into_data(
        self,
        data: torch.Tensor,
        lora_list: list[LoRAWeightEntry],
    ) -> None:
        """
        Merge all LoRA adapters into the data tensor in-place.

        Args:
            data: The base weight tensor to merge LoRA into (modified in-place)
            lora_list: Adapter factors, path, scale metadata, and output offset
        """
        # Merge all LoRA adapters in order
        for (
            lora_A,
            lora_B,
            _,
            lora_strength,
            lora_rank,
            lora_alpha,
            _,
        ) in lora_list:
            lora_A_sliced = self.slice_lora_a_weights(lora_A.to(data))
            lora_B_sliced = self.slice_lora_b_weights(lora_B.to(data))

            scale = lora_strength
            if (
                lora_alpha is not None
                and lora_rank is not None
                and lora_alpha != lora_rank
            ):
                scale *= lora_alpha / lora_rank

            if not isinstance(lora_B_sliced, torch.Tensor):
                lora_delta = lora_B_sliced @ lora_A_sliced
                if isinstance(lora_delta, torch.Tensor) and lora_delta.dim() > 2:
                    lora_delta = lora_delta.reshape(-1, lora_delta.shape[-1])
                data.add_(lora_delta, alpha=scale)
                continue

            if lora_A_sliced.dim() > 2 or lora_B_sliced.dim() > 2:
                lora_delta = lora_B_sliced @ lora_A_sliced
                if lora_delta.dim() > 2:
                    lora_delta = lora_delta.reshape(-1, lora_delta.shape[-1])
                data_2d = data.reshape(-1, data.shape[-1]) if data.dim() > 2 else data
                data_2d.add_(lora_delta, alpha=scale)
                continue

            data_2d = data.reshape(-1, data.shape[-1]) if data.dim() > 2 else data
            lora_B_2d = (
                lora_B_sliced.reshape(-1, lora_B_sliced.shape[-1])
                if lora_B_sliced.dim() > 2
                else lora_B_sliced
            )

            chunk_rows = max(
                1,
                LORA_MERGE_CHUNK_BYTES
                // (data_2d.shape[-1] * max(1, data_2d.element_size())),
            )
            for start in range(0, lora_B_2d.shape[0], chunk_rows):
                end = min(start + chunk_rows, lora_B_2d.shape[0])
                chunk_delta = lora_B_2d[start:end] @ lora_A_sliced
                data_2d[start:end].add_(chunk_delta, alpha=scale)

    def _should_merge_in_fp32(
        self,
        lora_list: list[LoRAWeightEntry],
    ) -> bool:
        if os.getenv("SGLANG_DIFFUSION_LORA_MERGE_FP32", "1") != "1":
            return False
        for _, _, lora_path, _, _, _, _ in lora_list:
            if lora_path and "distilled-lora" in lora_path.lower():
                return False
        return True

    @torch.no_grad()
    def compute_merged_weight(self) -> torch.Tensor:
        """The merged weight as a new CPU tensor; the base is never written.

        Same math as the in-place merge — computed on the device, in fp32
        when the policy says so, rounded back once — so the bytes are
        identical to what merge_lora_weights would have left in place.
        """
        base = self.weight.data
        target_dtype = base.dtype
        work = base.detach().to(get_local_torch_device())
        if (
            self._should_merge_in_fp32(self.lora_weights_list)
            and work.is_floating_point()
            and work.dtype != torch.float32
        ):
            work = work.to(torch.float32)
        self._merge_lora_into_data(work, self.lora_weights_list)
        return work.to("cpu", dtype=target_dtype)

    def install_merged_weight(
        self, merged: torch.Tensor, base_view: torch.Tensor
    ) -> None:
        """Adopt an externally held merged weight (e.g. a cache mapping).

        The single place the cached-merge state transition happens: the
        parameter points at `merged`, the layer counts as merged, and the
        unmerge snapshot is the untouched base view — zero-copy, because
        nothing wrote the base storage.
        """
        self.weight.data = merged
        self.merged = True
        self.cpu_weight = base_view.detach()
        self._base_is_view = True

    @torch.no_grad()
    def merge_lora_weights(self, strength: float | None = None) -> None:
        if strength is not None:
            self.strength = strength
            if self.lora_weights_list:
                self.lora_weights_list = [
                    (
                        lora_A,
                        lora_B,
                        lora_path,
                        strength,
                        lora_rank,
                        lora_alpha,
                        output_offset,
                    )
                    for (
                        lora_A,
                        lora_B,
                        lora_path,
                        _,
                        lora_rank,
                        lora_alpha,
                        output_offset,
                    ) in self.lora_weights_list
                ]

        if self.disable_lora:
            return

        self._ensure_base_snapshot_owned()
        if self.merged:
            self.unmerge_lora_weights()

        # Use lora_weights_list if available, otherwise fall back to single LoRA for backward compatibility
        lora_list = self.lora_weights_list if self.lora_weights_list else []
        if not lora_list and self.lora_A is not None and self.lora_B is not None:
            lora_list = [
                (
                    self.lora_A,
                    self.lora_B,
                    self.lora_path,
                    self.strength,
                    self.lora_rank,
                    self.lora_alpha,
                    self.lora_output_offset,
                )
            ]

        if not lora_list:
            raise ValueError("LoRA weights not set. Please set them first.")
        if isinstance(self.base_layer.weight, DTensor) and any(
            output_offset is not None for *_, output_offset in lora_list
        ):
            raise ValueError(
                "LoRA output offsets require dynamic mode with FSDP-sharded weights."
            )

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
        ``set_lora_weights`` then adds its delta on top of the merged base instead
        of unmerging it.
        """
        if not self.merged:
            return
        if self._active_lora_output_offset() is not None:
            raise ValueError(
                "A LoRA with a constant output offset cannot be committed as a "
                "weight-only base."
            )
        weight = self.base_layer.weight
        if isinstance(weight, DTensor):
            weight = weight.to_local()
        # clone(): to("cpu") may alias storage; we must not mutate this backup.
        self.cpu_weight = weight.detach().to("cpu").clone()
        self.merged = False
        self.disable_lora = True
        self.lora_weights_list = []
        self.lora_A = None
        self.lora_B = None
        self.lora_output_offset = None
        self.has_lora_output_offset = False
        self.lora_path = None
        self.strength = 1.0


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
        snapshot_base: bool = True,
    ) -> None:
        super().__init__(base_layer, lora_rank, lora_alpha, snapshot_base)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.disable_lora or (self.merged and not self.has_lora_output_offset):
            return self.base_layer(input_)

        lora_A = self.lora_A
        lora_B = self.lora_B
        if isinstance(self.lora_B, DTensor):
            lora_B = self.lora_B.to_local()
            lora_A = self.lora_A.to_local()

        bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
        output_parallel = self.base_layer.quant_method.apply(
            self.base_layer, input_, bias
        )
        if not self.merged and not self.disable_lora:
            lora_dtype = lora_A.dtype
            input_lora = input_.to(dtype=lora_dtype)
            lora_A_sliced = self.slice_lora_a_weights(
                lora_A.to(device=input_.device, non_blocking=True)
            )
            lora_B_sliced = self.slice_lora_b_weights(
                lora_B.to(device=input_.device, non_blocking=True)
            )
            delta_parallel = _compute_lora_delta(
                input_lora, lora_A_sliced, lora_B_sliced
            )
            if self.lora_alpha != self.lora_rank:
                delta_parallel = delta_parallel * (
                    self.lora_alpha / self.lora_rank  # type: ignore
                )  # type: ignore
            delta_parallel = delta_parallel * self.strength
            output_parallel = output_parallel + delta_parallel.to(
                dtype=output_parallel.dtype
            )
        output_parallel = self._add_lora_output_offset(output_parallel)
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
        snapshot_base: bool = True,
    ) -> None:
        super().__init__(base_layer, lora_rank, lora_alpha, snapshot_base)

    def slice_lora_a_weights(self, A: torch.Tensor) -> torch.Tensor:
        return A

    def slice_lora_b_weights(self, B: torch.Tensor) -> torch.Tensor:
        tp_rank = get_tp_rank()
        if B.dim() == 3:
            # Stacked Q/K/V (or gate/up) LoRA weights from diffusers-style adapters.
            shard_size = self.base_layer.output_partition_sizes[0]
            start_idx = tp_rank * shard_size
            end_idx = (tp_rank + 1) * shard_size
            return B[:, start_idx:end_idx, :]

        # Native fused checkpoints (MiniMax H3, etc.) store one concatenated 2D
        # lora_B matrix per logical layer; shard each section independently.
        shards: list[torch.Tensor] = []
        row_offset = 0
        for full_size, part_size in zip(
            self.base_layer.output_sizes,
            self.base_layer.output_partition_sizes,
        ):
            local_start = tp_rank * part_size
            local_end = (tp_rank + 1) * part_size
            shards.append(B[row_offset + local_start : row_offset + local_end, :])
            row_offset += full_size
        return torch.cat(shards, dim=0)


class QKVParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    def __init__(
        self,
        base_layer: QKVParallelLinear,
        lora_rank: int | None = None,
        lora_alpha: int | None = None,
        snapshot_base: bool = True,
    ) -> None:
        super().__init__(base_layer, lora_rank, lora_alpha, snapshot_base)

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
        snapshot_base: bool = True,
    ) -> None:
        super().__init__(base_layer, lora_rank, lora_alpha, snapshot_base)

    def forward(self, input_: torch.Tensor):
        if self.disable_lora or (self.merged and not self.has_lora_output_offset):
            return self.base_layer(input_)

        lora_A = self.lora_A
        lora_B = self.lora_B
        if isinstance(self.lora_B, DTensor):
            lora_B = self.lora_B.to_local()
            lora_A = self.lora_A.to_local()

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
        if not self.merged and not self.disable_lora:
            lora_dtype = lora_A.dtype
            input_parallel_lora = input_parallel.to(dtype=lora_dtype)
            lora_A_sliced = self.slice_lora_a_weights(
                lora_A.to(device=input_parallel.device, non_blocking=True)
            )
            lora_B_sliced = self.slice_lora_b_weights(
                lora_B.to(device=input_parallel.device, non_blocking=True)
            )
            delta_parallel = _compute_lora_delta(
                input_parallel_lora, lora_A_sliced, lora_B_sliced
            )
            if self.lora_alpha != self.lora_rank:
                delta_parallel = delta_parallel * (
                    self.lora_alpha / self.lora_rank  # type: ignore
                )  # type: ignore
            delta_parallel = delta_parallel * self.strength
            output_parallel = output_parallel + delta_parallel.to(
                dtype=output_parallel.dtype
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
        return self._add_lora_output_offset(output), output_bias

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
        snapshot_base: bool = True,
    ) -> None:
        super().__init__(base_layer, lora_rank, lora_alpha, snapshot_base)

    @torch.compile()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lora_A = self.lora_A
        lora_B = self.lora_B
        if isinstance(self.lora_B, DTensor):
            lora_B = self.lora_B.to_local()
            lora_A = self.lora_A.to_local()

        # TODO: Support multiple LoRA adapters when use not merged mode
        if not self.merged and not self.disable_lora:
            lora_dtype = lora_A.dtype
            x_lora = x.to(dtype=lora_dtype)
            lora_A_sliced = self.slice_lora_a_weights(
                lora_A.to(device=x.device, non_blocking=True)
            )
            lora_B_sliced = self.slice_lora_b_weights(
                lora_B.to(device=x.device, non_blocking=True)
            )
            delta = _compute_lora_delta(x_lora, lora_A_sliced, lora_B_sliced)
            if self.lora_alpha != self.lora_rank:
                delta = delta * (
                    self.lora_alpha / self.lora_rank  # type: ignore
                )  # type: ignore
            delta = delta * self.strength
            # nn.Linear.forward() returns a single tensor, not a tuple
            out = self.base_layer(x)
            out = out + delta.to(dtype=out.dtype)
        else:
            # nn.Linear.forward() returns a single tensor
            out = self.base_layer(x)
        return self._add_lora_output_offset(out)


def _use_owned_base_snapshot(snapshot_base: bool, device_type: str) -> bool:
    return snapshot_base or device_type not in ("cpu", "meta")


def wrap_with_lora_layer(
    layer: nn.Module,
    lora_rank: int | None = None,
    lora_alpha: int | None = None,
    snapshot_base: bool = True,
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
            effective_snapshot_base = _use_owned_base_snapshot(
                snapshot_base, layer.weight.device.type
            )
            ret = lora_layer_type(
                layer,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                snapshot_base=effective_snapshot_base,
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
