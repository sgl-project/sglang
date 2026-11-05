"""Prepare only incoming BF16 expert rows and rollback copies from fused HF shards."""

from __future__ import annotations

import json
import re
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from sglang.srt.model_loader.weight_utils import _drop_file_cache_after_load
from sglang.srt.utils.common import get_free_cpu_memory


@dataclass
class ExpertRowPatch:
    module: object
    slot: int
    old_w13: torch.Tensor
    old_w2: torch.Tensor
    new_w13: torch.Tensor
    new_w2: torch.Tensor

    def copy_new(self):
        self.module.w13_weight.data[self.slot].copy_(self.new_w13)
        self.module.w2_weight.data[self.slot].copy_(self.new_w2)

    def restore(self):
        self.module.w13_weight.data[self.slot].copy_(self.old_w13)
        self.module.w2_weight.data[self.slot].copy_(self.old_w2)


def node_staging_bytes(plan, modules, tp_size):
    return (
        sum(
            sum(
                weight[change.slot].numel() * weight.element_size()
                for weight in (
                    modules[change.layer].w13_weight,
                    modules[change.layer].w2_weight,
                )
            )
            * 2
            for change in plan.changes
        )
        * tp_size
    )


def prepare_expert_rows(
    plan,
    modules,
    model_path,
    num_experts,
    tp_rank,
    tp_size,
    max_node_staging_bytes,
    memory_reserve_bytes,
    available_cpu_bytes=None,
):
    """No device mutation until every TP rank has prepared and voted successfully."""
    node_bytes = node_staging_bytes(plan, modules, tp_size)
    if max_node_staging_bytes <= 0 or memory_reserve_bytes < 0:
        raise ValueError("Invalid hot-swap node memory budget")
    available = (
        get_free_cpu_memory() if available_cpu_bytes is None else available_cpu_bytes
    )
    if (
        node_bytes > max_node_staging_bytes
        or node_bytes + memory_reserve_bytes > available
    ):
        raise MemoryError(
            f"Hot swap needs {node_bytes} node staging bytes plus {memory_reserve_bytes} reserve"
        )
    path = Path(model_path)
    index = path / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
    else:
        weight_map = {}
        for shard in sorted(path.glob("*.safetensors")):
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                weight_map.update({name: shard.name for name in handle.keys()})
    keys = {}
    for layer in {change.layer for change in plan.changes}:
        for kind in ("gate_up_proj", "down_proj"):
            matches = [
                name
                for name in weight_map
                if re.search(
                    rf"^(?:model\.language_model|model)\.layers\.{layer}\.mlp\.experts\.{kind}(?:\.weight)?$",
                    name,
                )
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one fused {kind} checkpoint tensor for layer {layer}"
                )
            keys[layer, kind] = matches[0]
    prepared = []
    with ExitStack() as stack:
        handles = {}

        def tensor_slice(name):
            shard = weight_map[name]
            if shard not in handles:
                stack.callback(_drop_file_cache_after_load, str(path / shard))
                handles[shard] = stack.enter_context(
                    safe_open(str(path / shard), framework="pt", device="cpu")
                )
            return handles[shard].get_slice(name)

        current_layer = None
        for change in plan.changes:
            if current_layer is not None and change.layer != current_layer:
                # Prepared rows own their storage. Release the previous layer's mmap views.
                gate = up = w13 = w2 = None
                handles.clear()
                stack.close()
            current_layer = change.layer
            module = modules[change.layer]
            if (
                module.w13_weight.dtype != torch.bfloat16
                or module.w2_weight.dtype != torch.bfloat16
            ):
                raise ValueError("Hot expert replacement requires BF16 weights")
            local_intermediate = module.w2_weight.shape[2]
            hidden = module.w2_weight.shape[1]
            intermediate = local_intermediate * tp_size
            start, end = (
                tp_rank * local_intermediate,
                (tp_rank + 1) * local_intermediate,
            )
            w13 = tensor_slice(keys[change.layer, "gate_up_proj"])
            w2 = tensor_slice(keys[change.layer, "down_proj"])
            if w13.get_shape() != [
                num_experts,
                2 * intermediate,
                hidden,
            ] or w2.get_shape() != [num_experts, hidden, intermediate]:
                raise ValueError(
                    "Fused checkpoint dimensions do not match the loaded expert layer"
                )
            gate = w13[change.new_expert, start:end, :]
            up = w13[change.new_expert, intermediate + start : intermediate + end, :]
            new_w13 = torch.cat((gate, up), dim=0)
            if getattr(module.quant_method, "w13_swiglu_interleaved", False):
                new_w13 = torch.stack((gate, up), dim=1).flatten(0, 1)
            new_w2 = w2[change.new_expert, :, start:end].contiguous().clone()
            if new_w13.dtype != torch.bfloat16 or new_w2.dtype != torch.bfloat16:
                raise ValueError("The source expert checkpoint must be BF16")
            if (
                new_w13.shape != module.w13_weight[change.slot].shape
                or new_w2.shape != module.w2_weight[change.slot].shape
            ):
                raise ValueError(
                    "Incoming expert rows do not match the existing physical slot"
                )
            prepared.append(
                ExpertRowPatch(
                    module,
                    change.slot,
                    module.w13_weight[change.slot].detach().to(device="cpu", copy=True),
                    module.w2_weight[change.slot].detach().to(device="cpu", copy=True),
                    new_w13,
                    new_w2,
                )
            )
        gate = up = w13 = w2 = None
        handles.clear()
    return prepared, node_bytes
