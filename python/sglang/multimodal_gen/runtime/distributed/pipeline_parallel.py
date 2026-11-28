# SPDX-License-Identifier: Apache-2.0
"""Split a DiT's repeated blocks across pipeline-parallel stages.

Each stage builds and loads only its own blocks. Every stage runs the model's
prologue (embeddings) and epilogue (output head) on its own ranks, so the shape
of the residual stream entering the blocks is known locally; only that stream
crosses a stage boundary, rank to rank within the same sequence-parallel slot.
The last stage's output replaces the other stages' outputs after the forward.
"""

from contextlib import contextmanager
from typing import Any, Callable

import torch
from torch import nn
from torch.utils._pytree import tree_map

from sglang.multimodal_gen.runtime.distributed import parallel_state
from sglang.multimodal_gen.runtime.distributed.parallel_state import get_pp_group


class PipelineStagePlaceholder(nn.Module):
    """stands in for a block that another pipeline stage owns"""

    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = index

    def forward(self, *args, **kwargs):
        raise RuntimeError(f"Block {self.index} runs on another pipeline stage")


def stage_layer_range(num_layers: int, stage: int, num_stages: int) -> range:
    """contiguous, near-equal block ranges; earlier stages take the remainder"""
    if num_layers < num_stages:
        raise ValueError(
            f"Cannot split {num_layers} blocks across {num_stages} pipeline stages"
        )
    base, extra = divmod(num_layers, num_stages)
    start = stage * base + min(stage, extra)
    return range(start, start + base + (stage < extra))


def prune_pipeline_stage(model: nn.Module) -> None:
    """replace other stages' blocks before weights are materialized"""
    # standalone loaders and tests build models without parallel groups
    if parallel_state._PP is None or parallel_state._PP.world_size == 1:
        return
    pp = get_pp_group()
    name = getattr(model, "_pp_layers", None)
    if name is None:
        raise ValueError(
            f"{type(model).__name__} does not support pipeline parallelism"
        )
    blocks = getattr(model, name)
    owned = stage_layer_range(len(blocks), pp.rank_in_group, pp.world_size)
    for index in range(len(blocks)):
        if index not in owned:
            blocks[index] = PipelineStagePlaceholder(index)
    model._pp_owned = owned
    model.register_forward_hook(_broadcast_last_stage_output)


def owned_by_other_stage(model: nn.Module, param_name: str) -> bool:
    """whether a checkpoint parameter belongs to a block pruned from this stage"""
    owned = getattr(model, "_pp_owned", None)
    if owned is None:
        return False
    prefix = f"{model._pp_layers}."
    if not param_name.startswith(prefix):
        return False
    index = param_name[len(prefix) :].split(".", 1)[0]
    return index.isdigit() and int(index) not in owned


def run_stage_blocks(
    model: nn.Module,
    blocks: nn.ModuleList,
    hidden: torch.Tensor,
    step: Callable[[int, nn.Module, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """run this stage's blocks, receiving and sending the residual stream"""
    owned = getattr(model, "_pp_owned", None)
    if owned is None:
        for index, block in enumerate(blocks):
            hidden = step(index, block, hidden)
        return hidden
    pp = get_pp_group()
    shape, dtype = hidden.shape, hidden.dtype
    if not pp.is_first_rank:
        hidden = pp.recv(shape, dtype)
    for index in owned:
        hidden = step(index, blocks[index], hidden)
    if not pp.is_last_rank:
        if hidden.shape != shape or hidden.dtype != dtype:
            raise RuntimeError(
                "Pipeline parallelism needs blocks that keep the residual stream's "
                f"shape and dtype; got {tuple(hidden.shape)}/{hidden.dtype}, "
                f"expected {tuple(shape)}/{dtype}"
            )
        pp.send(hidden.contiguous())
    return hidden


# outstanding output broadcasts while forwards may overlap; None outside that scope
_pending_outputs: list | None = None


@contextmanager
def overlap_stage_outputs():
    """let back-to-back DiT forwards overlap across stages

    Outputs of forwards inside the block are valid only after it exits: the first
    stage starts the next forward while later stages still finish this one.
    """
    global _pending_outputs
    pp = parallel_state._PP
    if _pending_outputs is not None or pp is None or pp.world_size == 1:
        yield
        return
    _pending_outputs = []
    try:
        yield
    finally:
        pending, _pending_outputs = _pending_outputs, None
        for work in pending:
            work.wait()


def _broadcast_last_stage_output(module: nn.Module, args: Any, output: Any) -> Any:
    pp = get_pp_group()
    src = pp.ranks[pp.world_size - 1]

    def share(value):
        # every stage ran the epilogue, so its output already has the final shapes
        if not isinstance(value, torch.Tensor):
            return value
        value = value.contiguous()
        work = torch.distributed.broadcast(
            value, src=src, group=pp.device_group, async_op=True
        )
        if _pending_outputs is None:
            work.wait()
        else:
            _pending_outputs.append(work)
        return value

    return tree_map(share, output)
