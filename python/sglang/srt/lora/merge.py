# SPDX-License-Identifier: Apache-2.0
"""Merge a resident dense adapter without changing weight storage addresses."""

from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.lora.layers import (
    ColumnParallelLinearWithLoRA,
    MergedColumnParallelLinearWithLoRA,
    QKVParallelLinearWithLoRA,
    ReplicatedLinearWithLoRA,
    RowParallelLinearWithLoRA,
)
from sglang.srt.lora.utils import get_target_module_name

if TYPE_CHECKING:
    from sglang.srt.lora.lora_manager import LoRAManager


class LoRAMergeUnrecoverableError(RuntimeError):
    """A failed transition left weights unsafe to serve."""


def merge_slices(manager: "LoRAManager", lora_id: str):
    """Use the same TP shards, padded ranks, and fused slices as unmerged LoRA."""
    adapter = manager.loras[lora_id]
    if adapter.embedding_layers or adapter.added_tokens_embeddings:
        raise ValueError("LoRA merging does not support embedding or lm_head adapters")
    pool = manager.memory_pool
    slot = pool.get_buffer_id(lora_id)
    rank = adapter.config.r
    slices = []
    for layer_id, modules in enumerate(manager.lora_modules):
        targets = {
            get_target_module_name(name, pool.target_modules)
            for name in adapter.layers[layer_id].weights
        }
        for name, module in modules.items():
            target = get_target_module_name(name, pool.target_modules)
            if target not in targets:
                continue
            if type(module) not in (
                ColumnParallelLinearWithLoRA,
                MergedColumnParallelLinearWithLoRA,
                QKVParallelLinearWithLoRA,
                RowParallelLinearWithLoRA,
                ReplicatedLinearWithLoRA,
            ) or not isinstance(
                module.base_layer.quant_method, UnquantizedLinearMethod
            ):
                raise ValueError(
                    f"LoRA merging requires an unquantized dense linear: {name}"
                )
            weight = module.base_layer.weight
            if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise ValueError(f"Unsupported merge weight dtype: {weight.dtype}")
            a = pool.A_buffer[target][layer_id][slot]
            b = pool.B_buffer[target][layer_id][slot, :, :rank]
            if isinstance(module, ColumnParallelLinearWithLoRA):
                offsets = module.output_offset_cpu.tolist()
            elif (
                isinstance(module, ReplicatedLinearWithLoRA) and module.first_output_dim
            ):
                offsets = [0, module.first_output_dim, b.shape[0]]
            else:
                offsets = [0, b.shape[0]]
            for i, (start, end) in enumerate(zip(offsets, offsets[1:])):
                a_slice = a[i * rank : (i + 1) * rank]
                w_slice = weight[start:end]
                if a_slice.shape != (rank, w_slice.shape[1]) or end > b.shape[0]:
                    raise ValueError(f"Incompatible LoRA merge shape for {name}")
                slices.append((w_slice, a_slice, b[start:end]))
    if not slices:
        raise ValueError("Adapter has no mergeable linear weights")
    return slices


@torch.no_grad()
def merge_weights(manager: "LoRAManager", lora_id: str):
    if manager.merged_lora_id is not None:
        raise ValueError("Unmerge the current adapter before merging another")
    if lora_id not in manager.loras:
        raise ValueError("The adapter must be loaded before merging")
    for event in manager.pending_lora_load_events.values():
        event.synchronize()
    manager.pending_lora_load_events.clear()
    manager.fetch_new_loras({lora_id})
    slices = merge_slices(manager, lora_id)
    # Allocate every CPU snapshot before mutation. Copies use the compute stream,
    # so each snapshot finishes before its source weight is overwritten.
    originals = []
    if manager.lora_unmerge_mode == "restore":
        for weight, _, _ in slices:
            original = torch.empty_like(weight, device="cpu", pin_memory=weight.is_cuda)
            original.copy_(weight, non_blocking=True)
            originals.append((weight, original))
    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        for weight, a, b in slices:
            merged = torch.addmm(
                weight.float(),
                b.float(),
                a.float(),
                alpha=manager.loras[lora_id].scaling,
            )
            weight.copy_(merged)
    except Exception as e:
        if manager.lora_unmerge_mode == "subtract":
            raise LoRAMergeUnrecoverableError(
                f"LoRA merge failed without a base-weight backup: {e}"
            ) from e
        try:
            for weight, original in originals:
                weight.copy_(original, non_blocking=True)
            if originals[0][0].is_cuda:
                torch.cuda.current_stream(originals[0][0].device).synchronize()
        except Exception as restore_error:
            raise LoRAMergeUnrecoverableError(
                f"LoRA merge rollback failed: {restore_error}"
            ) from restore_error
        raise
    finally:
        torch.set_float32_matmul_precision(precision)
    manager.merged_lora_weights = originals
    manager.merged_lora_id = lora_id
    manager.reset_lora_batch()


@torch.no_grad()
def unmerge_weights(manager: "LoRAManager"):
    if manager.merged_lora_id is None:
        return
    if manager.lora_unmerge_mode == "subtract":
        scaling = manager.loras[manager.merged_lora_id].scaling
        for weight, a, b in merge_slices(manager, manager.merged_lora_id):
            weight.addmm_(b, a, alpha=-scaling)
    else:
        for weight, original in manager.merged_lora_weights:
            weight.copy_(original, non_blocking=True)
    manager.merged_lora_weights.clear()
    manager.merged_lora_id = None
    manager.reset_lora_batch()
