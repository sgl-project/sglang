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
"""Build a model's per-layer module list, sliced to this pipeline stage."""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, Optional, Protocol, Sequence, Tuple

import torch

from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.runtime_context import get_parallel


class LayerFn(Protocol):
    def __call__(self, idx: int, prefix: str) -> torch.nn.Module: ...


def make_layers(
    num_hidden_layers: int,
    layer_fn: LayerFn,
    pp_rank: Optional[int] = None,
    pp_size: Optional[int] = None,
    prefix: str = "",
    return_tuple: bool = False,
    offloader_kwargs: Optional[Dict[str, Any]] = None,
    final_read: Optional[Any] = None,
    stage_facts: Optional[Callable[[int], Sequence[Any]]] = None,
) -> Tuple[torch.nn.Module, int, int]:
    """Make a list of layers with the given layer function.

    The local layers are built inside one layer stack, so layers that declare
    stage boundaries connect in order without naming their neighbours. Across
    a pipeline stage boundary the stack learns the neighbouring stage from
    ``stage_facts``: the model's shared declaration function, which returns
    the stages the layer at a global index declares without building it (an
    empty sequence when it declares none), and which the layer itself uses
    for its own declarations. A model whose layers declare stages must give
    it to run under pipeline parallelism. ``final_read`` is the stack's
    terminal read when it is not a plain final norm (see layer_stack).
    """
    # circular imports
    from sglang.srt.distributed import get_pp_indices
    from sglang.srt.layers.layer_boundary.factories import (
        check_declared_stages,
        layer_stack,
    )
    from sglang.srt.layers.offloader import get_offloader
    from sglang.srt.layers.utils import PPMissingLayer

    assert not pp_size or num_hidden_layers >= pp_size
    start_layer, end_layer = (
        get_pp_indices(
            num_hidden_layers,
            pp_rank,
            pp_size,
        )
        if pp_rank is not None and pp_size is not None
        else (0, num_hidden_layers)
    )

    def neighbour(idx):
        if stage_facts is not None:
            return functools.partial(stage_facts, idx)
        return functools.partial(_no_stage_facts, layer_fn)

    def build(stack, idx):
        appended = len(stack.appends)
        layer = layer_fn(idx=idx, prefix=add_prefix(idx, prefix))
        if stage_facts is not None and (pp_size or 1) > 1:
            # Another rank binds this layer's stages from stage_facts alone.
            check_declared_stages(stack, appended, stage_facts(idx), f"layer {idx}")
        return layer

    with layer_stack(
        previous_layers=[neighbour(idx) for idx in reversed(range(start_layer))],
        next_layers=[neighbour(idx) for idx in range(end_layer, num_hidden_layers)],
        final_read=final_read,
    ) as stack:
        modules = torch.nn.ModuleList(
            [PPMissingLayer(return_tuple=return_tuple) for _ in range(start_layer)]
            + get_offloader().wrap_modules(
                (build(stack, idx) for idx in range(start_layer, end_layer)),
                **(offloader_kwargs or {}),
            )
            + [
                PPMissingLayer(return_tuple=return_tuple)
                for _ in range(end_layer, num_hidden_layers)
            ]
        )
    if pp_rank is None or pp_size is None:
        return modules
    return modules, start_layer, end_layer


def make_pp_layers(
    num_hidden_layers: int,
    layer_fn: LayerFn,
    prefix: str = "",
    return_tuple: bool = False,
    offloader_kwargs: Optional[Dict[str, Any]] = None,
    final_read: Optional[Any] = None,
    stage_facts: Optional[Callable[[int], Sequence[Any]]] = None,
) -> Tuple[torch.nn.Module, int, int]:
    """Make this pipeline stage's layers, and return them with the stage's range.

    Layers outside ``[start_layer, end_layer)`` are ``PPMissingLayer`` stand-ins.
    ``stage_facts`` is the model's shared declaration function (see make_layers).
    """
    parallel = get_parallel()
    return make_layers(
        num_hidden_layers,
        layer_fn,
        pp_rank=parallel.pp_rank,
        pp_size=parallel.pp_size,
        prefix=prefix,
        return_tuple=return_tuple,
        offloader_kwargs=offloader_kwargs,
        final_read=final_read,
        stage_facts=stage_facts,
    )


def _no_stage_facts(layer_fn: LayerFn) -> None:
    """Stands in for the shared declaration function of a model that gives
    none: called only if this pipeline rank's layers declare stages."""
    module = getattr(getattr(layer_fn, "func", layer_fn), "__module__", None)
    raise ValueError(
        f"{module} declares stage boundaries under pipeline parallelism "
        "without a shared declaration function: give make_layers its "
        "stage_facts, the stages a layer declares without building it"
    )
