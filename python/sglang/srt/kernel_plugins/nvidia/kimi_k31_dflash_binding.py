"""Host binding for the Kimi K3.1 megakernel's DFlash runtime.

The DFLASH worker binds the Kimi K3.1 megakernel's DFlash runtime through this module.
A megakernel target build that runs the DFLASH decode round itself names its runtime
class by "module:attribute" (spec.SupportsDFlashRuntime). The worker hands the class
three whole phases; everything else stays SGLang's:

* ``cls(binding, built)``: the end of the worker's ``__init__``, before the KV pools
  are sized, so the runtime's allocations come out of the KV budget;
* ``draft_sampler(lm_head)``: the selector draft sampler folded into the draft cuda
  graph, asked before that graph captures; None keeps SGLang's;
* ``decode_round(batch, on_publish=, grammar_barrier=)``: the decode branch of
  ``forward_batch_generation``. None runs SGLang's round; the runtime decides it on
  every TP rank alike, before it launches anything.

The class pulls worker state on demand, by name, through ``DFlashBinding.pull``:
only the ``EXPORTS`` it lists in ``exports``. It lists in ``invariants`` the value
it expects for each ``GUARD`` key it depends on (a frozenset accepts any of its
members). ``bind_dflash_runtime`` checks both, and the API version, on every TP rank
before it builds the runtime. A name or API SGLang does not have raises; a guard value
the runtime does not serve declines it (one warning naming each key), and SGLang's
stock round runs.
"""

from __future__ import annotations

import logging
import pkgutil
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, Mapping, Optional, Protocol

import msgspec
import torch

from sglang.kernels.plugin_kernels.nvidia.spec import FLAG, SupportsDFlashRuntime
from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.environ import envs
from sglang.srt.kernel_plugins.nvidia.binding import (
    built_megakernel,
    megakernel_agree,
    megakernel_enabled,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.runtime_context import (
    get_exec,
    get_lora,
    get_memory,
    get_parallel,
    get_schedule,
    get_spec,
)
from sglang.srt.speculative.dspark_components.dspark_observability import (
    resolve_enabled_components,
)
from sglang.srt.speculative.ragged_verify import read_ragged_verify_mode
from sglang.srt.utils import is_cuda

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.utils import GenerationBatchResult

# Bumped when an entry, an export's meaning or a guard key's meaning changes.
DFLASH_RUNTIME_API = 1
logger = logging.getLogger(__name__)


class DFlashRuntime(Protocol):
    """What SGLang calls on a bound runtime (see the module docstring)."""

    api_version: ClassVar[int]
    exports: ClassVar[frozenset[str]]
    invariants: ClassVar[Mapping[str, Any]]

    def __init__(self, binding: DFlashBinding, built: Any) -> None: ...

    def draft_sampler(self, lm_head: Any) -> Optional[Any]: ...

    def decode_round(
        self,
        batch: ScheduleBatch,
        *,
        on_publish: Optional[Callable[..., None]],
        grammar_barrier: Optional[Callable[[], Any]],
    ) -> Optional[GenerationBatchResult]: ...


class Export(msgspec.Struct, frozen=True):
    member: str  # the DFlashWorkerV2 attribute or method
    type: Any  # what pull checks the value against


_NONE = type(None)
EXPORTS: dict[str, Export] = {
    # set in __init__
    "block_size": Export("block_size", int),
    "device": Export("device", (str, torch.device)),  # "cuda": the server arg
    "mask_token_id": Export("_mask_token_id", int),
    "noise_embed_scale": Export("_noise_embed_scale", float),
    "batched_small_copy": Export("_batched_small_copy", bool),
    "draft_model": Export("draft_model", torch.nn.Module),
    "draft_model_runner": Export("draft_model_runner", object),
    "draft_parallel_layout": Export("draft_parallel_layout", object),
    "draft_block_spec_info": Export("_draft_block_spec_info", object),
    "target_worker": Export("target_worker", object),
    "target_logits_processor": Export(
        "_target_logits_processor", (LogitsProcessor, _NONE)
    ),
    "sim_acc_rng": Export("_simulated_acc_len_rng", (torch.Generator, _NONE)),
    # set by init_attention_backends
    "need_mamba_verify_commit": Export("_need_mamba_verify_commit", bool),
    # the stock round's helpers
    "compute_compact_draft_seq_lens": Export(
        "_compute_compact_draft_seq_lens", Callable
    ),
    "fill_compact_seq_lens_cpu_bound": Export(
        "_fill_compact_seq_lens_cpu_bound", Callable
    ),
    "rebuild_compact_draft_cache": Export("_rebuild_compact_draft_cache", Callable),
    "accept_block": Export("_accept_block", Callable),
}


def _env_tristate(env) -> Optional[bool]:
    return env.get() if env.is_set() else None


# key -> (where SGLang reads it, how); read at the end of the worker's __init__.
# parallel/schedule/spec/memory/lora name the runtime_context getters.
GUARD: dict[str, tuple[str, Callable[[Any], Any]]] = {
    "platform": ("utils.is_cuda()", lambda w: "cuda" if is_cuda() else "other"),
    "tp_size": ("parallel.tp_size", lambda w: get_parallel().tp_size),
    "pp_size": ("parallel.pp_size", lambda w: get_parallel().pp_size),
    "attn_dp": ("parallel.attn_dp_enabled", lambda w: get_parallel().attn_dp_enabled),
    "dtype": ("model_config.dtype", lambda w: str(w.model_runner.model_config.dtype)),
    "page_size": ("schedule.page_size", lambda w: get_schedule().page_size),
    "recurrent_target": (
        "mambaish_config(model_config)",
        lambda w: mambaish_config(w.model_runner.model_config) is not None,
    ),
    "draft_architectures": (
        "draft hf_config.architectures",
        lambda w: tuple(w.draft_model_runner.model_config.hf_config.architectures),
    ),
    "draft_kv_cache_dtype": (
        "spec.speculative_draft_kv_cache_dtype",
        lambda w: get_spec().speculative_draft_kv_cache_dtype,
    ),
    "block_size": ("worker.block_size", lambda w: int(w.block_size)),
    "compact_draft_cache": (
        "worker.draft_window_size",
        lambda w: w.use_compact_draft_cache,
    ),
    "draft_kv_ring": ("worker.draft_kv_ring", lambda w: w.draft_kv_ring is not None),
    "selector_draft": ("worker.selector", lambda w: w.selector is not None),
    "selector_sampling": ("worker", lambda w: w._selector_sampling_enabled),
    "lilicorr": ("worker.lilicorr", lambda w: w.lilicorr is not None),
    "domino": ("worker._is_domino", lambda w: w._is_domino),
    "block_verification": ("worker", lambda w: w._block_verification),
    "replicated_embed": ("worker", lambda w: w._full_embed_gpu is not None),
    "aux_graph": ("worker._use_aux_graph", lambda w: w._use_aux_graph),
    "decode_cuda_graph_bs": ("exec cuda_graph_config", lambda w: _decode_graph_bs()),
    "ragged_verify_mode": ("env", lambda w: read_ragged_verify_mode().value),
    "eager_draft_sampler": (
        "env",
        lambda w: envs.SGLANG_DFLASH_EAGER_DRAFT_SAMPLER.get(),
    ),
    "defer_plain_gather": (
        "env, None when unset",
        lambda w: _env_tristate(envs.SGLANG_OPT_DFLASH_DEFER_PLAIN_GATHER),
    ),
    "info_dump": ("env", lambda w: bool(resolve_enabled_components())),
    "hierarchical_cache": (
        "memory.enable_hierarchical_cache",
        lambda w: get_memory().enable_hierarchical_cache,
    ),
    "hisparse": ("memory.enable_hisparse", lambda w: get_memory().enable_hisparse),
    "lora": ("lora.enable_lora", lambda w: bool(get_lora().enable_lora)),
}


def _decode_graph_bs() -> Optional[tuple[int, ...]]:
    decode = get_exec().graph.cuda_graph_config.decode
    if decode.backend == Backend.DISABLED:
        return None
    return tuple(sorted(decode.bs))


class DFlashBinding:
    """The runtime's view of its DFLASH worker: the exports it declared, by name."""

    __slots__ = ("_worker", "_names")

    def __init__(self, worker: Any, names: frozenset[str]) -> None:
        self._worker = worker
        self._names = frozenset(names)

    def pull(self, name: str) -> Any:
        """The worker's current value of an export the runtime declared."""
        if name not in self._names:
            raise KeyError(f"the DFLASH runtime pulls {name!r} without declaring it")
        export = EXPORTS[name]
        # Not through the worker's __getattr__, which falls back to the target worker.
        value = object.__getattribute__(self._worker, export.member)
        if not isinstance(value, export.type):
            raise TypeError(
                f"DFLASH export {name!r} is {type(value).__name__}, not {export.type}"
            )
        return value


def _accepts(expected: Any, actual: Any) -> bool:
    return actual in expected if isinstance(expected, frozenset) else actual == expected


def check_dflash_runtime(cls: Any, worker: Any, path: str) -> list[str]:
    """The guard values ``cls`` does not serve here, after raising on an API version,
    export or guard key SGLang lacks. Collective: every TP rank must read the same
    values, so every rank declines alike."""
    if not isinstance(cls, type):
        raise TypeError(f"{FLAG}: {path} is not a class")
    if cls.api_version != DFLASH_RUNTIME_API:
        raise RuntimeError(
            f"{FLAG}: {path} is DFLASH runtime API {cls.api_version}; "
            f"SGLang binds API {DFLASH_RUNTIME_API}"
        )
    unknown = sorted(set(cls.exports) - EXPORTS.keys())
    unknown += sorted(f"guard key {k}" for k in cls.invariants.keys() - GUARD.keys())
    if unknown:
        raise RuntimeError(f"{FLAG}: SGLang does not export {unknown} to {path}")
    actual = {key: GUARD[key][1](worker) for key in sorted(cls.invariants)}
    megakernel_agree(sorted(actual.items()), what="DFLASH runtime guard values")
    return [
        f"{key} is {actual[key]!r} ({GUARD[key][0]}), expected {expected!r}"
        for key, expected in sorted(cls.invariants.items())
        if not _accepts(expected, actual[key])
    ]


def bind_dflash_runtime(worker: Any, target_model: torch.nn.Module) -> Any:
    """The runtime the target's build names, checked and built on ``worker``, or None
    for SGLang's round (flag off, no build, a build naming none, or a setup the runtime
    does not serve). Collective."""
    if not megakernel_enabled():
        return None
    built = built_megakernel(target_model)
    offers = isinstance(built, SupportsDFlashRuntime)
    path = built.dflash_runtime if offers else ""
    megakernel_agree(path, what="DFLASH runtime")
    if built is not None and not offers and get_parallel().tp_rank == 0:
        # E.g. a megakernel package from before SupportsDFlashRuntime.
        logger.warning(
            "%s: the megakernel offers no DFlash runtime; DFlash phases run stock",
            FLAG,
        )
    if not path:
        return None
    if path.count(":") != 1:
        raise ValueError(f"{FLAG}: {path!r} is not a 'module:attribute' path")
    cls = pkgutil.resolve_name(path)
    unserved = check_dflash_runtime(cls, worker, path)
    if unserved:
        if get_parallel().tp_rank == 0:
            logger.warning(
                "%s: %s does not serve this DFLASH setup (%s); DFlash phases run stock",
                FLAG,
                path,
                "; ".join(unserved),
            )
        return None
    return cls(DFlashBinding(worker, cls.exports), built)
