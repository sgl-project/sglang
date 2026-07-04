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

``get_flags()`` returns the resolved-flags tier: what the system *resolved*
the configuration to (``server_args`` stays the pristine user input). Flags
live in typed dataclass groups (``flags.attn`` / ``flags.moe`` / flat generic
leaves on ``flags`` itself); reads and writes are plain attribute access.
Static groups are writable during resolution and locked by ``freeze()``;
``flags.capture`` stays writable (capture-time state). Each group offers a
transactional, test-only ``override(**kw)`` that also works on frozen groups.
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
        if getattr(self, "_frozen", False):
            raise RuntimeError(
                f"{type(self).__name__} is frozen; cannot write '{name}'. "
                "Test-scoped changes go through override()."
            )
        object.__setattr__(self, name, value)

    @contextmanager
    def override(self, **kwargs):
        """Temporarily force flag values, restoring on exit. Transactional
        (keys validated before any write) and usable on frozen groups — this
        is the test-only injection primitive."""
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


class _StaticFlags(_FlagGroupBase):
    """Static flag-group: writable during resolution, locked by ``freeze()``."""

    def freeze(self) -> None:
        object.__setattr__(self, "_frozen", True)

    @property
    def frozen(self) -> bool:
        return getattr(self, "_frozen", False)


@dataclasses.dataclass
class AttnFlags(_StaticFlags):
    """Attention-family resolved flags (leaves arrive with the V3 sweeps)."""

    # Resolved attention backend; the pristine user request stays on
    # server_args.attention_backend.
    backend: str | None = None


@dataclasses.dataclass
class MoeFlags(_StaticFlags):
    """MoE-family resolved flags (leaves arrive with the V3 sweeps)."""

    # Resolved MoE runner backend; the pristine user request stays on
    # server_args.moe_runner_backend.
    runner_backend: str = "auto"


@dataclasses.dataclass
class CaptureFlags(_FlagGroupBase):
    """Capture-time flags; never frozen (written during cuda-graph capture)."""


@dataclasses.dataclass
class Flags(_StaticFlags):
    """Root of the resolved-flags tier.

    Family groups hang off it (``flags.attn`` / ``flags.moe`` / ``flags.capture``);
    single generic flags live flat on this container, declared as fields here.
    ``freeze()`` locks the container and every static sub-group; ``capture``
    stays writable.
    """

    attn: AttnFlags = dataclasses.field(default_factory=AttnFlags)
    moe: MoeFlags = dataclasses.field(default_factory=MoeFlags)
    capture: CaptureFlags = dataclasses.field(default_factory=CaptureFlags)

    # -- resolved config leaves (flat; materialized at publish) --------------
    # Pristine user requests stay on the matching server_args fields; these
    # leaves carry the model-resolved values.
    dtype: str = "auto"
    enable_tf32_matmul: bool = False
    enable_multi_layer_eagle: bool = False
    swa_full_tokens_ratio: float = 0.8
    disable_hybrid_swa_memory: bool = False
    sampling_backend: str | None = None
    page_size: int | None = None
    quantization: str | None = None
    disable_overlap_schedule: bool = False
    uses_mamba_radix_cache: bool = False
    mamba_radix_cache_strategy: str = "auto"
    speculative_moe_runner_backend: str | None = None
    speculative_moe_a2a_backend: str | None = None
    # Parallel-request fields: flat transitional home, to be re-homed by the
    # Parallel Parameters Clarification module.
    enable_dp_attention: bool = False
    enable_dp_lm_head: bool = False
    moe_a2a_backend: str = "none"
    ep_size: int = 1
    moe_dense_tp_size: int | None = None
    attn_cp_size: int = 1

    def freeze(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, _StaticFlags):
                value.freeze()
        super().freeze()


# Resolved-config field name → dotted flag-leaf path (e.g. a V3 sweep adds
# "use_mla_backend": "attn.use_mla_backend"). Fields not listed default to a
# flat leaf of the same name on the Flags container. Populated per field
# family as readers migrate.
FLAG_LEAF_MAP: dict[str, str] = {
    "attention_backend": "attn.backend",
    "moe_runner_backend": "moe.runner_backend",
}


def resolve_flag_leaf(
    flags: Flags, field: str, *, leaf_map: dict[str, str] | None = None
) -> tuple[Any, str]:
    """Return ``(owning group, leaf attribute name)`` for a resolved-config field."""
    path = (FLAG_LEAF_MAP if leaf_map is None else leaf_map).get(field, field)
    owner: Any = flags
    *groups, leaf = path.split(".")
    for part in groups:
        owner = getattr(owner, part)
    return owner, leaf


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
        the draft-worker guard in ``ModelRunner.__init__``).

        Publishing also resolves the stashed model-override declarations
        into the flags tier (skipped for objects without the stash — dummy /
        "none" fixture ServerArgs and test-kit mocks never compute it).
        Resolution runs first: if it fails, the previous publish stays intact.
        """
        self._resolve_flags(server_args)
        self._server_args = server_args

    def _resolve_flags(self, server_args: ServerArgs) -> None:
        declarations = getattr(server_args, "_resolved_overrides", None)
        if declarations is None:
            return
        from sglang.srt.arg_groups.overrides import (
            apply_model_overrides,
            assert_flag_parity,
        )

        # Resolve into a fresh container and only install it once everything
        # passed: a failed resolution (gate validation or the parity assert)
        # must not leave the process-global flags half-written for callers
        # that catch the error or republish (same install-fresh semantics as
        # reset_context()).
        flags = Flags()
        apply_model_overrides(flags, server_args, declarations)
        # Transition-period drift guard: dual-apply keeps the declared fields
        # on server_args byte-identical to the resolved flag leaves.
        assert_flag_parity(
            flags,
            server_args,
            {field for _source, decl in declarations for field in decl},
        )
        self.flags = flags


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
    ``server_args`` and install a fresh, unfrozen ``Flags``.

    Wrapper subsystems (``parallel``) hold no state and are unaffected.
    """
    _CONTEXT._server_args = None
    _CONTEXT.flags = Flags()
