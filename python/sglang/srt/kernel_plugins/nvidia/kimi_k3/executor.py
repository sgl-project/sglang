# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""TP8 prepared-width captured-decode executor for Kimi K3."""

from __future__ import annotations

from functools import partial
from typing import Any

import torch

try:
    from sglang.srt.kernel_plugins.nvidia.binding import HybridStepView
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.megakernels.binding import HybridStepView

# Names read from SGLang internals. A contract test checks these before GPU use.
SGLANG_READS = {
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/binding.py": (
        "HybridStepView",
        "block_table",
        "kda",
        "mla",
        "sequence_lengths",
        "state_indices",
        "track_decode_state",
        "write_slots",
        "native_decode",
        "prepare_layer",
        "global_sequence_lengths",
    ),
}


def _require_tensor(
    name: str,
    value: torch.Tensor,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    device: torch.device,
    strides: tuple[int, ...] | None = None,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"Kimi K3 megakernel {name} is not a tensor")
    if value.dtype != dtype:
        raise RuntimeError(
            f"Kimi K3 megakernel {name} has dtype {value.dtype}, expected {dtype}"
        )
    if tuple(value.shape) != shape:
        raise RuntimeError(
            f"Kimi K3 megakernel {name} has shape {tuple(value.shape)}, "
            f"expected {shape}"
        )
    if value.device != device:
        raise RuntimeError(
            f"Kimi K3 megakernel {name} is on {value.device}, expected {device}"
        )
    if strides is not None and tuple(value.stride()) != strides:
        raise RuntimeError(
            f"Kimi K3 megakernel {name} has stride {tuple(value.stride())}, "
            f"expected {strides}"
        )


def _validate_step_inputs(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor | None,
    view: HybridStepView,
    *,
    batch: int,
    device: torch.device,
) -> None:
    if device.type != "cuda":
        raise RuntimeError(
            f"Kimi K3 megakernel execution requires CUDA resources, found {device}"
        )
    _require_tensor(
        "input_ids",
        input_ids,
        dtype=torch.int64,
        shape=(batch,),
        device=device,
        strides=(1,),
    )
    if inputs_embeds is not None:
        _require_tensor(
            "inputs_embeds",
            inputs_embeds,
            dtype=torch.bfloat16,
            shape=(batch, 7168),
            device=device,
            strides=(7168, 1),
        )
    _require_tensor(
        "MLA write slots",
        view.mla.write_slots,
        dtype=torch.int64,
        shape=(batch,),
        device=device,
        strides=(1,),
    )
    _require_tensor(
        "MLA sequence lengths",
        view.mla.sequence_lengths,
        dtype=torch.int32,
        shape=(batch,),
        device=device,
        strides=(1,),
    )
    _require_tensor(
        "KDA state indices",
        view.kda.state_indices,
        dtype=torch.int32,
        shape=(batch,),
        device=device,
        strides=(1,),
    )
    table = view.mla.block_table
    if not isinstance(table, torch.Tensor):
        raise RuntimeError("Kimi K3 megakernel MLA block table is not a tensor")
    if table.dtype != torch.int32 or table.device != device:
        raise RuntimeError(
            "Kimi K3 megakernel MLA block table must be int32 on "
            f"{device}, found {table.dtype} on {table.device}"
        )
    if table.dim() != 2 or int(table.shape[0]) != batch or int(table.shape[1]) <= 0:
        raise RuntimeError(
            "Kimi K3 megakernel MLA block table must be a nonempty "
            f"[{batch}, pages] tensor, found {tuple(table.shape)}"
        )
    rows, columns = int(table.stride(0)), int(table.stride(1))
    if columns != 1 or rows < int(table.shape[1]):
        raise RuntimeError(
            "Kimi K3 megakernel MLA block table has unsupported stride "
            f"{tuple(table.stride())}"
        )


class KimiK3MegakernelExecutor:
    """Run the prepared 93-layer chain against SGLang-owned runtime state."""

    def __init__(self, resources: Any):
        self.resources = resources

    def run(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        view: HybridStepView,
    ) -> torch.Tensor:
        resources = self.resources
        if resources.binding_fingerprint is None:
            raise RuntimeError(
                "Kimi K3 megakernel pools were not bound before graph warmup"
            )
        mla_slots = view.mla.write_slots
        kda_slots = view.kda.state_indices
        prepare_layer = getattr(view, "prepare_layer", None)
        global_lengths = getattr(view.mla, "global_sequence_lengths", None)
        batch = int(input_ids.shape[0])
        if batch not in getattr(
            resources, "batches", (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64)
        ):
            raise RuntimeError(f"unsupported Kimi K3 megakernel batch: {batch}")
        device = resources.scratch["hidden.0"].device
        _validate_step_inputs(
            input_ids,
            inputs_embeds,
            view,
            batch=batch,
            device=device,
        )
        for layer_id, attention in resources.attention.items():
            if attention.kind == "mla" and attention.cache is None:
                raise RuntimeError(f"MLA cache for layer {layer_id} is not bound")
            if attention.kind == "kda" and attention.caches is None:
                raise RuntimeError(f"KDA caches for layer {layer_id} are not bound")
        if getattr(resources, "native_mla", False) and (
            getattr(view.mla, "native_decode", None) is None or prepare_layer is None
        ):
            raise RuntimeError("Kimi native MLA profile requires native runtime hooks")
        if getattr(resources, "full_decode", False) and prepare_layer is None:
            raise RuntimeError(
                "Kimi full decode profile requires restore lifecycle hooks"
            )
        if "dcp" in getattr(resources, "boxes", {}) and global_lengths is None:
            raise RuntimeError("Kimi full DCP requires global sequence lengths")
        embedding = getattr(resources, "embedding", None)
        prefetch = getattr(resources, "prefetch", None)
        x = (
            inputs_embeds
            if inputs_embeds is not None
            else embedding.run(input_ids, resources.scratch["embedding.out"][:batch])
            if embedding is not None
            else resources.model.model.embed_tokens(input_ids)
        )
        if tuple(x.shape) != (batch, 7168):
            raise RuntimeError(f"unexpected Kimi K3 megakernel input shape: {x.shape}")

        prefix = resources.scratch["prefix"][:batch]
        blocks = resources.scratch["blocks"][:batch]
        first = resources.scratch["hidden.0"][:batch]
        second = resources.scratch["hidden.1"][:batch]
        output = resources.scratch["hidden.out"][:batch]
        mla_projected = resources.scratch["mla.projected"][:batch]
        kda_projected = resources.scratch["kda.projected"][:batch]
        local = resources.scratch["attention.local"][:batch]

        hidden = x
        next_input_ready = False
        last_layer_id = -1
        try:
            for layer in resources.layers:
                layer_id = int(layer.layer_idx)
                last_layer_id = layer_id
                if not next_input_ready:
                    resources.before[layer_id].run(hidden, prefix, blocks, first)

                if prepare_layer is not None:
                    prepare_layer(layer_id)

                attention = resources.attention[layer_id]
                projected = mla_projected if attention.kind == "mla" else kda_projected
                projected = attention.projection.run(first, projected)
                if attention.kind == "mla":
                    native_args = (
                        {"native_decode": partial(view.mla.native_decode, layer_id)}
                        if getattr(attention.core, "native_mla", False)
                        else {}
                    )
                    attention.core.run(
                        projected,
                        local,
                        cache=attention.cache,
                        slots=mla_slots,
                        block_table=view.mla.block_table,
                        lengths=(
                            global_lengths
                            if getattr(attention.core, "native_mla", False)
                            and global_lengths is not None
                            else view.mla.sequence_lengths
                        ),
                        totals=global_lengths,
                        **native_args,
                    )
                    if prefetch is not None:
                        prefetch.fork("attention_moe", layer_id, batch)
                else:
                    if prefetch is not None:
                        prefetch.fork("attention_moe", layer_id, batch)
                    attention.core.run(
                        projected, local, caches=attention.caches, slots=kda_slots
                    )
                    view.kda.track_decode_state(layer_id)
                attention.output.run(local, prefix, blocks, second)
                if prefetch is not None:
                    prefetch.fork("oproj_down", layer_id, batch)

                consumer = resources.fused_before.get(layer_id)
                next_input_ready = consumer is not None
                feed_forward = resources.feed_forward[layer_id]
                if consumer is not None:
                    target = output if consumer is resources.final else first
                    consume = partial(consumer.run_ring, prefix, blocks)
                    hidden = feed_forward.run(second, target, consume=consume)
                else:
                    hidden = feed_forward.run(second, first)
                    first, second = second, first

            if not next_input_ready:
                resources.final.run(hidden, prefix, blocks, output)
            if last_layer_id != 92:
                raise RuntimeError(
                    f"Kimi K3 megakernel stopped at layer {last_layer_id}"
                )
            return output
        finally:
            if prefetch is not None:
                prefetch.join()
