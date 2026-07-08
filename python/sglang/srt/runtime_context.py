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
"""A single structured accessor for process-static runtime state.

``get_parallel()`` returns a ``ParallelContext`` whose attributes — tp / pp /
moe / attn size and rank, plus the process-group handles — each delegate live to
the canonical getter in ``distributed.parallel_state`` / ``layers.dp_attention``.
Returned values are exactly what those getters return; this is a read-through
wrapper, not a cache. It gives call-sites one import and one naming scheme in
place of a dozen free functions, plus a test-only ``override()`` hook to force a
topology without monkeypatching the underlying getters.

``get_server_args()`` returns the process-wide ``ServerArgs`` (the config
tier). The context owns the storage: publishing goes through
``RuntimeContext.set_server_args`` (the legacy
``set_global_server_args_for_scheduler`` / ``get_global_server_args`` in
``server_args.py`` are thin shims over this slot), and the object is returned
by reference — the same live instance everywhere, never a copy.

``get_flags()`` returns the runtime-flags tier. Resolved configuration lives
on ``server_args`` fields (declarations materialize at the end of
``__post_init__``), so this tier only carries genuine runtime state that is
not a function of the configuration alone — today the capture lifecycle
(``flags.capture``). Flags live in typed dataclass groups; reads and writes
are plain attribute access, and each group offers a transactional, test-only
``override(**kw)``.
"""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


# Imported lazily so this module has no import-time dependencies: any module can
# import get_parallel at module level without risking an import cycle.
def _ps():
    from sglang.srt.distributed import parallel_state

    return parallel_state


def _dp():
    from sglang.srt.layers import dp_attention

    return dp_attention


_PARALLEL_FIELDS = frozenset(
    {
        "world_size",
        "world_rank",
        "tp_size",
        "tp_rank",
        "pp_size",
        "pp_rank",
        "moe_ep_size",
        "moe_ep_rank",
        "moe_dp_size",
        "moe_dp_rank",
        "moe_tp_size",
        "moe_tp_rank",
        "attn_tp_size",
        "attn_tp_rank",
        "attn_cp_size",
        "attn_cp_rank",
        "attn_dp_size",
        "attn_dp_rank",
        "world_group",
        "tp_group",
        "pp_group",
        "moe_ep_group",
        "moe_dp_group",
        "moe_tp_group",
        "attn_tp_group",
        "attn_cp_group",
    }
)


class ParallelContext:
    """Parallel-topology namespace; the only instance state is ``_overrides``."""

    __slots__ = ("_overrides",)

    def __init__(self):
        self._overrides = {}

    def _v(self, name, getter):
        overrides = self._overrides
        return overrides[name] if name in overrides else getter()

    @contextmanager
    def override(self, **kwargs):
        """Temporarily force parallel values, restoring on exit. Validates keys and
        supports nesting."""
        unknown = set(kwargs) - _PARALLEL_FIELDS
        if unknown:
            raise ValueError(f"unknown parallel field(s): {sorted(unknown)}")
        saved = dict(self._overrides)
        self._overrides.update(kwargs)
        try:
            yield self
        finally:
            self._overrides = saved

    @property
    def world_size(self) -> int:
        return self._v("world_size", _ps().get_world_size)

    @property
    def world_rank(self) -> int:
        return self._v("world_rank", _ps().get_world_rank)

    @property
    def tp_size(self) -> int:
        return self._v("tp_size", _ps().get_tensor_model_parallel_world_size)

    @property
    def tp_rank(self) -> int:
        return self._v("tp_rank", _ps().get_tensor_model_parallel_rank)

    @property
    def pp_size(self) -> int:
        return self._v("pp_size", _ps().get_pipeline_model_parallel_world_size)

    @property
    def pp_rank(self) -> int:
        return self._v("pp_rank", _ps().get_pipeline_model_parallel_rank)

    @property
    def moe_ep_size(self) -> int:
        return self._v("moe_ep_size", _ps().get_moe_expert_parallel_world_size)

    @property
    def moe_ep_rank(self) -> int:
        return self._v("moe_ep_rank", _ps().get_moe_expert_parallel_rank)

    @property
    def moe_dp_size(self) -> int:
        return self._v("moe_dp_size", _ps().get_moe_data_parallel_world_size)

    @property
    def moe_dp_rank(self) -> int:
        return self._v("moe_dp_rank", _ps().get_moe_data_parallel_rank)

    @property
    def moe_tp_size(self) -> int:
        return self._v("moe_tp_size", _ps().get_moe_tensor_parallel_world_size)

    @property
    def moe_tp_rank(self) -> int:
        return self._v("moe_tp_rank", _ps().get_moe_tensor_parallel_rank)

    @property
    def attn_tp_size(self) -> int:
        return self._v("attn_tp_size", _ps().get_attn_tensor_model_parallel_world_size)

    @property
    def attn_tp_rank(self) -> int:
        return self._v("attn_tp_rank", _ps().get_attn_tensor_model_parallel_rank)

    @property
    def attn_cp_size(self) -> int:
        return self._v("attn_cp_size", _ps().get_attn_context_model_parallel_world_size)

    @property
    def attn_cp_rank(self) -> int:
        return self._v("attn_cp_rank", _ps().get_attn_context_model_parallel_rank)

    @property
    def attn_dp_size(self) -> int:
        return self._v("attn_dp_size", _dp().get_attention_dp_size)

    @property
    def attn_dp_rank(self) -> int:
        return self._v("attn_dp_rank", _dp().get_attention_dp_rank)

    @property
    def world_group(self) -> Any:
        return self._v("world_group", _ps().get_world_group)

    @property
    def tp_group(self) -> Any:
        return self._v("tp_group", _ps().get_tp_group)

    @property
    def pp_group(self) -> Any:
        return self._v("pp_group", _ps().get_pp_group)

    @property
    def moe_ep_group(self) -> Any:
        return self._v("moe_ep_group", _ps().get_moe_ep_group)

    @property
    def moe_dp_group(self) -> Any:
        return self._v("moe_dp_group", _ps().get_moe_dp_group)

    @property
    def moe_tp_group(self) -> Any:
        return self._v("moe_tp_group", _ps().get_moe_tp_group)

    @property
    def attn_tp_group(self) -> Any:
        return self._v("attn_tp_group", _ps().get_attn_tp_group)

    @property
    def attn_cp_group(self) -> Any:
        return self._v("attn_cp_group", _ps().get_attn_cp_group)


class _FlagGroupBase:
    """Shared flag-group behavior: typo-safe writes + transactional ``override()``.

    Groups are plain dataclasses; ``__dataclass_fields__`` is the single source
    of truth for which leaves exist, so a mistyped name fails loudly instead of
    creating a stray attribute.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        if name not in type(self).__dataclass_fields__:
            raise AttributeError(
                f"{type(self).__name__} has no flag '{name}' (leaves are "
                "declared as dataclass fields; check for typos)"
            )
        object.__setattr__(self, name, value)

    @contextmanager
    def override(self, **kwargs):
        """Temporarily force flag values, restoring on exit. Transactional
        (keys validated before any write) — the test-only injection
        primitive."""
        fields = type(self).__dataclass_fields__
        unknown = set(kwargs) - set(fields)
        if unknown:
            raise ValueError(
                f"unknown flag(s) for {type(self).__name__}: {sorted(unknown)}"
            )
        saved = {name: getattr(self, name) for name in kwargs}
        for name, value in kwargs.items():
            object.__setattr__(self, name, value)
        try:
            yield self
        finally:
            for name, value in saved.items():
                object.__setattr__(self, name, value)


@dataclasses.dataclass
class CaptureFlags(_FlagGroupBase):
    """Capture-time flags; never frozen (written during cuda-graph capture)."""

    # Seeded from server_args at publish; a model whose _can_torch_compile is
    # False clears it during warmup (the only post-publish writer).
    enable_torch_compile: bool = False


@dataclasses.dataclass
class Flags(_FlagGroupBase):
    """Root of the runtime-flags tier.

    Resolved configuration lives on ``server_args`` fields (materialized at
    the end of ``__post_init__``) — this tier only carries genuine runtime
    state whose value is not a function of the configuration alone, grouped
    by lifecycle (today: ``capture``).
    """

    capture: CaptureFlags = dataclasses.field(default_factory=CaptureFlags)


class RuntimeContext:
    """Container for the structured runtime accessors; exposes ``parallel``,
    ``server_args``, and ``flags``."""

    __slots__ = ("parallel", "_server_args", "flags")

    def __init__(self, parallel: ParallelContext):
        self.parallel = parallel
        self._server_args: ServerArgs | None = None
        self.flags = Flags()

    @property
    def server_args(self) -> ServerArgs:
        """The process-wide ``ServerArgs`` (context-owned slot)."""
        server_args = self._server_args
        if server_args is None:
            # Verbatim legacy message: tests and user scripts may match on it.
            raise ValueError("Global server args is not set yet!")
        return server_args

    def set_server_args(self, server_args: ServerArgs) -> None:
        """Publish the process-wide ``ServerArgs`` into the context-owned slot.

        Overwrite-allowed: a re-publish replaces the slot (test kits re-publish
        per test; production ordering discipline lives at the call-sites, e.g.
        the draft-worker guard in ``ModelRunner.__init__``). The published
        object already carries the resolved configuration (declarations
        materialize at the end of ``__post_init__``).
        """
        # Seed the capture tier for the new lifecycle (defaults for sentinel
        # and mock publishes, which carry no config).
        self.flags.capture.enable_torch_compile = getattr(
            server_args, "enable_torch_compile", False
        )
        self._server_args = server_args


_PARALLEL = ParallelContext()
_CONTEXT = RuntimeContext(parallel=_PARALLEL)


def get_context() -> RuntimeContext:
    return _CONTEXT


def get_parallel() -> ParallelContext:
    return _PARALLEL


def get_server_args() -> ServerArgs:
    return _CONTEXT.server_args


def get_flags() -> Flags:
    return _CONTEXT.flags


def reset_context() -> None:
    """Clear the context-owned store (unit-test teardown): drop the published
    ``server_args`` and install a fresh ``Flags``.

    Wrapper subsystems (``parallel``) hold no state and are unaffected.
    """
    _CONTEXT._server_args = None
    _CONTEXT.flags = Flags()
