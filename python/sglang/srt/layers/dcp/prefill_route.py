# Copyright 2023-2026 SGLang Team
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

"""Price the gathered DCP prefill of every attention backend against the owned
route the model takes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass(frozen=True)
class DcpOwnedRoute:
    """What the owned MLA DCP route puts on the wire per extend token and head."""

    v_head_dim: int
    # The a2a carries v_head_dim-wide projected partials, not kv_lora_rank.
    project_before_merge: bool
    # The query is all-gathered rather than built from replicated q_b_proj.
    q_all_gather: bool


def _leaf_backends(backend) -> Iterator:
    children = backend.window_capacity_children()
    if not children:
        yield backend
        return
    for child in children:
        yield from _leaf_backends(child)


def init_dcp_prefill_crossover(
    *, model: torch.nn.Module, attn_backend, partial_dtype: torch.dtype
) -> None:
    """Hand the owned route of ``model``'s first MLA layer to every leaf backend
    under ``attn_backend`` that plans a gathered DCP prefill. Run it once the
    replicated q-proj / w_vc buffers the route depends on exist."""
    inits = [
        init
        for backend in _leaf_backends(attn_backend)
        if (init := getattr(backend, "init_dcp_prefill_crossover", None)) is not None
    ]
    if not inits:
        return
    route = next(
        (
            owned_route()
            for module in model.modules()
            if (owned_route := getattr(module, "dcp_owned_route", None)) is not None
        ),
        None,
    )
    if route is None:
        return
    for init in inits:
        init(route=route, partial_dtype=partial_dtype)
