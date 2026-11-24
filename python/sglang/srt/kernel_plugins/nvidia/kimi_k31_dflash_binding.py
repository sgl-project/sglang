"""Host binding for the Kimi K3.1 megakernel's DFlash runtime.

A megakernel target build that runs the DFLASH decode round itself names its runtime
class by "module:attribute" (``spec.SupportsDFlashRuntime``). The DFLASH worker hands
that class three whole phases (``DFlashRuntime``) and keeps everything else.

The class declares two things, each checked against a table here:

* ``bindings``, the worker state it reads, from ``BINDINGS``. It reads them on demand
  through ``DFlashBinding.pull``.
* ``guards``, the value it expects for each setup key it depends on, from ``GUARDS``.
  ``DFlashGuard.unmet`` reads the actual values on every TP rank.

A name or key missing from a table raises at startup: the runtime and SGLang no
longer match. A guard value that differs declines the runtime instead (one warning
naming each key), and the worker runs its own round.
"""

from __future__ import annotations

import logging
import pkgutil
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Protocol

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
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

logger = logging.getLogger(__name__)


class DFlashRuntime(Protocol):
    """The class a megakernel build names; the worker calls the three phases."""

    bindings: ClassVar[frozenset[str]]
    guards: ClassVar[Mapping[str, Any]]

    def __init__(self, binding: DFlashBinding, built: Any) -> None:
        """End of the worker's ``__init__``, before the KV pools are sized."""

    def draft_sampler(self, lm_head: Any) -> Optional[Any]:
        """Before the draft graph captures; None keeps SGLang's sampler."""

    def decode_round(
        self,
        batch: ScheduleBatch,
        *,
        on_publish: Optional[Callable[..., None]],
        grammar_barrier: Optional[Callable[[], Any]],
    ) -> Optional[GenerationBatchResult]:
        """The decode branch of ``forward_batch_generation``. None, decided alike on
        every TP rank before any launch, runs SGLang's round. A runtime that binds
        ``join_aux`` lets SGLang's aux append run on its side stream: it joins the
        pending append before its draft forward, and returns ``aux_done`` as the
        result's ``side_stream_done`` after appending there itself."""


# ---- Binding: the worker state a runtime may read.


class WorkerMember(msgspec.Struct, frozen=True):
    attr: str  # the DFlashWorkerV2 attribute or method
    type: Any  # what pull checks the value against


BINDINGS: dict[str, WorkerMember] = {
    # set in __init__
    "block_size": WorkerMember("block_size", int),
    "device": WorkerMember("device", (str, torch.device)),  # "cuda": the server arg
    "mask_token_id": WorkerMember("_mask_token_id", int),
    "noise_embed_scale": WorkerMember("_noise_embed_scale", float),
    "batched_small_copy": WorkerMember("_batched_small_copy", bool),
    "draft_model": WorkerMember("draft_model", torch.nn.Module),
    "draft_model_runner": WorkerMember("draft_model_runner", object),
    "draft_parallel_layout": WorkerMember("draft_parallel_layout", object),
    "draft_block_spec_info": WorkerMember("_draft_block_spec_info", object),
    "target_worker": WorkerMember("target_worker", object),
    "target_logits_processor": WorkerMember(
        "_target_logits_processor", (LogitsProcessor, type(None))
    ),
    "sim_acc_rng": WorkerMember(
        "_simulated_acc_len_rng", (torch.Generator, type(None))
    ),
    # set by init_attention_backends
    "need_mamba_verify_commit": WorkerMember("_need_mamba_verify_commit", bool),
    # the stock round's helpers
    "compute_compact_draft_seq_lens": WorkerMember(
        "_compute_compact_draft_seq_lens", Callable
    ),
    "fill_compact_seq_lens_cpu_bound": WorkerMember(
        "_fill_compact_seq_lens_cpu_bound", Callable
    ),
    "rebuild_compact_draft_cache": WorkerMember(
        "_rebuild_compact_draft_cache", Callable
    ),
    "accept_block": WorkerMember("_accept_block", Callable),
    # the stock round's phases, for a runtime that composes its round from both
    "draft_sampler": WorkerMember("_draft_sampler", object),  # set by init_cuda_graphs
    "append_target_hidden_decode": WorkerMember(
        "_append_target_hidden_decode", Callable
    ),
    "aux_overlap": WorkerMember("_aux_overlap", bool),
    "join_aux": WorkerMember("_join_aux", Callable),
    "aux_done": WorkerMember("_aux_done", (torch.cuda.Event, type(None))),
    "selector_sampling_accept": WorkerMember("_selector_sampling_accept", Callable),
    "update_target_mamba_state_after_verify": WorkerMember(
        "_update_target_mamba_state_after_verify", Callable
    ),
}


class DFlashBinding:
    """A runtime's view of its worker: the bindings it declared, read by name."""

    __slots__ = ("_worker", "_names")

    def __init__(self, worker: DFlashWorkerV2, names: frozenset[str]) -> None:
        if unknown := sorted(names - BINDINGS.keys()):
            raise RuntimeError(f"{FLAG}: the DFLASH binding has no name {unknown}")
        self._worker = worker
        self._names = names

    def pull(self, name: str) -> Any:
        if name not in self._names:
            raise KeyError(f"the DFLASH runtime pulls {name!r} without declaring it")
        member = BINDINGS[name]
        # Not through the worker's __getattr__, which falls back to the target worker.
        value = object.__getattribute__(self._worker, member.attr)
        if not isinstance(value, member.type):
            raise TypeError(
                f"DFLASH binding {name!r} is {type(value).__name__}, not {member.type}"
            )
        return value


# ---- Guard: the setup values a runtime may depend on, read at the end of __init__.

GUARDS: dict[str, Callable[[DFlashWorkerV2], Any]] = {
    # parallelism and model
    "platform": lambda w: "cuda" if is_cuda() else "other",
    "tp_size": lambda w: get_parallel().tp_size,
    "pp_size": lambda w: get_parallel().pp_size,
    "attn_dp": lambda w: get_parallel().attn_dp_enabled,
    "dtype": lambda w: str(w.model_runner.model_config.dtype),
    "page_size": lambda w: get_schedule().page_size,
    "recurrent_target": lambda w: (
        mambaish_config(w.model_runner.model_config) is not None
    ),
    # the draft and its worker
    "draft_architectures": lambda w: tuple(
        w.draft_model_runner.model_config.hf_config.architectures
    ),
    "draft_kv_cache_dtype": lambda w: get_spec().speculative_draft_kv_cache_dtype,
    "block_size": lambda w: int(w.block_size),
    "compact_draft_cache": lambda w: w.use_compact_draft_cache,
    "draft_kv_ring": lambda w: w.draft_kv_ring is not None,
    "selector_draft": lambda w: w.selector is not None,
    "selector_sampling": lambda w: w._selector_sampling_enabled,
    "lilicorr": lambda w: w.lilicorr is not None,
    "domino": lambda w: w._is_domino,
    "block_verification": lambda w: w._block_verification,
    "replicated_embed": lambda w: w._full_embed_gpu is not None,
    "aux_graph": lambda w: w._use_aux_graph,
    # graphs, verify and envs
    "decode_cuda_graph_bs": lambda w: (
        None
        if (decode := get_exec().graph.cuda_graph_config.decode).backend
        == Backend.DISABLED
        else tuple(sorted(decode.bs))
    ),
    "ragged_verify_mode": lambda w: read_ragged_verify_mode().value,
    "eager_draft_sampler": lambda w: envs.SGLANG_DFLASH_EAGER_DRAFT_SAMPLER.get(),
    "defer_plain_gather_refused": lambda w: (
        envs.SGLANG_OPT_DFLASH_DEFER_PLAIN_GATHER.is_set()
        and not envs.SGLANG_OPT_DFLASH_DEFER_PLAIN_GATHER.get()
    ),
    "info_dump": lambda w: bool(resolve_enabled_components()),
    # cache features
    "hierarchical_cache": lambda w: get_memory().enable_hierarchical_cache,
    "hisparse": lambda w: get_memory().enable_hisparse,
    "lora": lambda w: bool(get_lora().enable_lora),
}


class DFlashGuard:
    """The setup values a runtime expects, checked against the running setup."""

    __slots__ = ("_expected",)

    def __init__(self, expected: Mapping[str, Any]) -> None:
        if unknown := sorted(expected.keys() - GUARDS.keys()):
            raise RuntimeError(f"{FLAG}: the DFLASH guard has no key {unknown}")
        self._expected = dict(sorted(expected.items()))

    def unmet(self, worker: DFlashWorkerV2) -> list[str]:
        """Each key whose value differs; collective, so every rank agrees."""
        actual = {key: GUARDS[key](worker) for key in self._expected}
        megakernel_agree(list(actual.items()), what="DFLASH guard values")
        return [
            f"{key} is {actual[key]!r}, expected {expected!r}"
            for key, expected in self._expected.items()
            if actual[key] != expected
        ]


def bind_dflash_runtime(
    worker: DFlashWorkerV2, target_model: torch.nn.Module
) -> Optional[DFlashRuntime]:
    """The runtime the target's build names, bound to ``worker``, or None for SGLang's
    round (flag off, no build, a build naming none, or a guard value that differs).
    Collective."""
    if not megakernel_enabled():
        return None
    built = built_megakernel(target_model)
    offers = isinstance(built, SupportsDFlashRuntime)
    path = built.dflash_runtime if offers else ""
    megakernel_agree(path, what="DFLASH runtime")
    tp0 = get_parallel().tp_rank == 0
    if built is not None and not offers and tp0:
        logger.warning(
            "%s: the megakernel offers no DFlash runtime; DFlash phases run stock",
            FLAG,
        )
    if not path:
        return None

    runtime_cls = pkgutil.resolve_name(path)
    binding = DFlashBinding(worker, frozenset(runtime_cls.bindings))
    unmet = DFlashGuard(runtime_cls.guards).unmet(worker)
    if unmet:
        if tp0:
            logger.warning(
                "%s: %s does not serve this DFLASH setup (%s); DFlash phases run stock",
                FLAG,
                path,
                "; ".join(unmet),
            )
        return None
    return runtime_cls(binding, built)
