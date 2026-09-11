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
"""Resolved graph-serialization settings for one process (design section 13).

The plan is a frozen snapshot of the ``exec.graph`` config leaves that select
the feature. It is computed once per process by
``model_runner_components.cuda_graph_serialization.plan_graph_serialization``
and read by ``materializer.resolve_materializer`` when a runner is built.
"""

from __future__ import annotations

import os
from enum import Enum

import msgspec

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_exec


class CacheMode(str, Enum):
    OFF = "off"
    SAVE = "save"
    LOAD = "load"
    AUTO = "auto"


class Placement(str, Enum):
    """How saved device pointers are bound at load (design section 7.2)."""

    RELOCATE = "relocate"
    FIXED_VA = "fixed_va"


class VerifyMode(str, Enum):
    NONE = "none"
    SHADOW_ONE = "shadow-one"
    SHADOW_ALL = "shadow-all"


class GraphSerializationPlan(msgspec.Struct, frozen=True, kw_only=True):
    """Frozen snapshot of the feature's settings for one process.

    ``strict`` (any fallback or shadow diff fails startup, design section 6.11
    step 6) has no configuration surface in this draft: neither a server
    argument nor an ``SGLANG_CUDA_GRAPH_CACHE_STRICT`` env var is declared
    yet, so it is always ``False`` here.
    """

    mode: CacheMode = CacheMode.OFF
    cache_dir: str = ""
    placement: Placement = Placement.RELOCATE
    verify: VerifyMode = VerifyMode.NONE
    strict: bool = False
    disabled_reason: str = ""

    @property
    def enabled(self) -> bool:
        return self.mode is not CacheMode.OFF

    @property
    def saves(self) -> bool:
        return self.mode in (CacheMode.SAVE, CacheMode.AUTO)

    @property
    def loads(self) -> bool:
        return self.mode in (CacheMode.LOAD, CacheMode.AUTO)

    def disabled(self, reason: str) -> GraphSerializationPlan:
        """A copy of this plan with the feature turned off and the reason kept."""
        return msgspec.structs.replace(self, mode=CacheMode.OFF, disabled_reason=reason)


def default_cache_dir() -> str:
    """``$SGLANG_CACHE_DIR/cuda_graphs``; no parallel env var (design section 13)."""
    return os.path.join(os.path.expanduser(envs.SGLANG_CACHE_DIR.get()), "cuda_graphs")


def read_plan_from_config() -> GraphSerializationPlan:
    """Project the ``exec.graph`` leaves into a plan.

    Reads go through the published bag, never the ``ServerArgs`` record. The
    ``getattr`` defaults keep a partial bag in a unit test (one that publishes
    only ``cuda_graph_config``) on the ``off`` path instead of raising.
    """
    graph = get_exec().graph
    mode = CacheMode(getattr(graph, "cuda_graph_cache_mode", None) or "off")
    cache_dir = getattr(graph, "cuda_graph_cache_dir", None) or default_cache_dir()
    placement = Placement(
        getattr(graph, "cuda_graph_cache_placement", None) or "relocate"
    )
    verify = VerifyMode(getattr(graph, "cuda_graph_cache_verify", None) or "none")
    return GraphSerializationPlan(
        mode=mode,
        cache_dir=cache_dir,
        placement=placement,
        verify=verify,
    )
