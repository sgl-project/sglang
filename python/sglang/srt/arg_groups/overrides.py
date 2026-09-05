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
"""Declarative model-override registry.

Model-identity adjustments to the server configuration are DECLARED here and
appended to the record's declaration stash (gate order, last writer wins).
Nothing here writes back onto ``ServerArgs``: the record holds the user's raw
input, and a decision is read through ``resolution_result`` or the published
config bags — model code never mutates ``ServerArgs`` fields imperatively. The
one channel that still leaves a field changed is ``declare_direct_writes``,
which does not perform the write: it captures one an out-of-tree plugin already
made, and undoing it would surprise the plugin's own reads.

Two declaration forms, keyed on ``hf_config.architectures[0]``:

- ``MODEL_OVERRIDES``: pure-constant cases — ``arch -> {field: value}``.
- ``@register_model_override(arch)``: derived cases — a callable
  ``fn(server_args, hf_config) -> dict`` that faithfully carries today's
  conditional logic. ``server_args`` is pristine and must be treated
  read-only: the callable returns declarations, it never writes.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from sglang.srt.arg_groups import model_override_base
from sglang.srt.arg_groups.arg_utils import field_names, resolvable_fields

# Re-exported for the callers that already import these names from here; the
# declarations under ``model_overrides/`` import them from the base directly.
from sglang.srt.arg_groups.model_override_base import (  # noqa: F401
    _MODEL_OVERRIDE_FNS,
    _PREDICATE_OVERRIDE_FNS,
    MODEL_OVERRIDES,
    ResolvedView,
    ResolvingConfig,
    _declaration_overlay,
    _invoke_provider,
    _register_for,
    attention_backends_of,
    get_default_attn_backend,
    is_attention_backend_not_set,
    mamba_extra_buffer_of,
    model_config_of,
    record_of,
    register_model_override,
    register_model_override_predicate,
    resolved_view,
    resolving_view,
    use_mla_backend,
)

logger = logging.getLogger(__name__)
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.runtime_context import (
    get_context,
    get_platform,
)
from sglang.srt.utils.common import (
    get_quantization_config,
    is_gfx95_supported,
    xpu_has_xmx_support,
)

# Registered post-process passes. This is a registry, not an execution order:
# each pass is invoked from its own slot via run_post_process_pass.
POST_PROCESS_PASSES: List[Callable[..., dict]] = []


def register_post_process(fn: Callable[..., dict]) -> Callable[..., dict]:
    """Register a post-process pass: ``fn(view) -> {field: resolved_value}``.

    The pass reads a :class:`ResolvedView` (post-model-override state) and
    must not mutate anything; validations may live in a pass (read + raise).
    """
    POST_PROCESS_PASSES.append(fn)
    return fn


def run_post_process_pass(server_args: Any, fn: Callable[..., dict]) -> None:
    """Invoke one pass at its legacy handler slot.

    Evaluates the pass on the resolving state (a read-only view with the
    accumulated declarations overlaid from the stash) and appends its
    declaration to the stash, which is what the config bags are projected from.
    The fields stay untouched.

    A slot that runs after resolution -- ``check_server_args`` hosts one -- lands
    in the same stash, which publish projects from later, so it needs no field
    write either. After *publish* there is no such later projection: the stash
    would grow an entry nothing reads.

    So what is refused is the *declaration*, not the record. A pass that returns
    an empty dict is a validation, and it may run on the published instance --
    it has to, because ``Engine(server_args=sa)`` after ``Engine.shutdown()``
    re-runs ``check_server_args`` on the very instance the context still holds.
    A pass that returns a non-empty dict there is refused, as
    ``declare_late_resolution`` is -- post-publish changes go to the bags through
    ``get_context().override(...)``.
    """

    declared = fn(ResolvedView(server_args, overlay=_declaration_overlay(server_args)))
    if not isinstance(declared, dict):
        raise TypeError(
            f"post-process pass {fn.__qualname__} must return a dict, "
            f"got {type(declared).__name__}"
        )
    if declared:
        # Refused only once there is something to record. A pass that declares
        # nothing is a validation, and `check_server_args` runs those again on
        # a rebuild: `Engine(server_args=sa)` after `Engine.shutdown()` hands
        # back the same instance while the context still holds it, and
        # refusing on identity alone would fail that launch.
        try:
            published = get_context().server_args
        except ValueError:
            published = None
        if published is server_args:
            raise ValueError(
                f"run_post_process_pass({fn.__qualname__!r}) declared "
                f"{sorted(declared)} on the published config; the stash is "
                "projected at publish and never again, so this would be a "
                "silent no-op -- post-publish changes go to the bags via "
                "get_context().override(...)"
            )
        entry = (fn.__qualname__, dict(declared))
        stash = getattr(server_args, "_resolved_overrides", None)
        if stash is None:
            # Handlers hosting pass slots may be invoked directly on fixtures
            # that never ran the monolith dispatch (which owns the stash);
            # create it lazily. Real publishes always pass through the
            # dispatch first — the dispatch ASSIGNS the stash, so pass slots
            # must sit at or after it in __post_init__ order.
            stash = server_args._resolved_overrides = []
        stash.append(entry)
        validate_declarations(server_args, [entry])


def declare_resolution(server_args: Any, source: str, **fields: Any) -> None:
    """Record a resolution write in the declaration stash.

    The stash *is* the resolution result: the bags are projected from it,
    `resolution_result` answers from it, and no field is written. A resolver
    reading a field another resolver may have decided must read `resolving_view`
    (or `resolved_view(server_args)`), which
    `test_resolution_reads_the_declarations` pins.

    For resolvers inside ``__post_init__``; launcher-stage resolution goes
    through ``declare_late_resolution``. A name that is not a field is rejected
    here rather than becoming an attribute nothing reads.
    """
    if dataclasses.is_dataclass(type(server_args)):
        unknown = sorted(set(fields) - field_names(type(server_args)))
        if unknown:
            raise AttributeError(f"{source}: {unknown} are not ServerArgs fields")
    stash = getattr(server_args, "_resolved_overrides", None)
    if stash is None:
        stash = []
        server_args._resolved_overrides = stash
    stash.append((source, dict(fields)))


def declare_late_resolution(server_args: Any, source: str, **fields: Any) -> None:
    """Resolve fields on a config that is **not published yet**.

    A few resolution rules cannot run inside ``__post_init__``: LoRA
    normalization and the auto-parser detection need the launcher's validation
    stage (and, for the parsers, a tokenizer / chat-template load). They still
    belong to the resolution pipeline — they decide what the process will run
    with — so their decision goes to the stash like any other, and the record
    keeps what the caller passed. Every holder of that instance reads the
    decision the same way the rest of the pipeline does: the bags it publishes,
    or ``resolution_result``, both of which survive the pickle to a child.

    Refuses to touch the published instance: after publish the bags exist and a
    field write would desync them, which is what ``get_context().override`` is
    for.
    """

    try:
        published = get_context().server_args
    except ValueError:
        published = None
    if published is server_args:
        raise ValueError(
            f"declare_late_resolution({source!r}) called on the published config; "
            "post-publish changes go to the bags via get_context().override(...)"
        )
    log = getattr(server_args, "_runtime_mutations", None)
    if log is None:
        log = []
        server_args._runtime_mutations = log
    log.append((source, dict(fields)))
    stash = getattr(server_args, "_resolved_overrides", None)
    if stash is None:
        stash = []
        server_args._resolved_overrides = stash
    stash.append((source, dict(fields)))


def declare_direct_writes(
    server_args: Any, source: str, resolve: Callable[[Any], Any]
) -> Any:
    """Run a resolver that writes the fields directly, and declare what it moved.

    Returns whatever the resolver returned, so a provider with a return value
    can go through the same capture.

    Out-of-tree platform plugins are handed the record and set fields on it.
    Their implementations live outside this tree, so they cannot be converted
    by editing the resolver; and the raw snapshot is taken before the pipeline
    starts, so a plugin's default is neither declared nor raw. The write itself
    stays: this captures it into the stash so the projection and the bags carry
    it, but reverting the field would break the plugin's own reads of what it
    just set. It is the only field a record still carries from resolution.

    Rebinding is what the diff sees, and rebinding is all it needs to see: a
    plugin that mutates a value in place reaches the projection anyway, because
    the raw snapshot and the stash entries hold the same object it mutated.

    A stand-in record (tests drive the hooks with a plain namespace) has no
    fields to diff and no projection to feed, so the resolver runs uncaptured.
    """
    if not dataclasses.is_dataclass(server_args):
        return resolve(server_args)
    before = {
        field.name: getattr(server_args, field.name)
        for field in dataclasses.fields(server_args)
    }
    already = len(getattr(server_args, "_resolved_overrides", None) or ())
    result = resolve(server_args)
    stash = getattr(server_args, "_resolved_overrides", None)
    if stash is None:
        stash = []
        server_args._resolved_overrides = stash
    # A resolver reached this way can also declare properly -- the in-tree
    # implementations of these hooks do. Those fields are already explained, and
    # recording them again would attribute them to the wrapper and bury an
    # actual direct write among the echoes.
    declared = {name for _source, fields in stash[already:] for name in fields}
    changed = {
        name: getattr(server_args, name)
        for name, previous in before.items()
        if name not in declared and getattr(server_args, name) is not previous
    }
    if changed:
        stash.append((source, changed))
    return result


def resolution_result(server_args: Any, field: str, default: Any = None) -> Any:
    """What resolution decided for ``field``: the declaration if there is one,
    otherwise what the caller supplied.

    This is what the config projection reads. Reading the field instead would
    work whatever the caller passed onto the record -- and
    the point of declaring is that they will not, so the projection must not
    depend on it. A config that never ran the pipeline (a mock, a partial
    fixture) carries no raw snapshot; its fields are all it has.
    """
    for _source, declared in reversed(
        getattr(server_args, "_resolved_overrides", None) or ()
    ):
        if field in declared:
            return declared[field]
    raw = getattr(server_args, "_raw_input", None)
    if raw is not None and field in raw:
        return raw[field]
    return getattr(server_args, field, default)


def resolution_projection(server_args: Any) -> Dict[str, Any]:
    """Every field's resolved value, nested dataclasses expanded.

    The whole-object shape of ``resolution_result``, for the exits that hand out
    the entire configuration (``/server_info``, the gRPC and engine readbacks).
    They used ``dataclasses.asdict``, which reads the fields -- the operator's
    input, not what resolution decided. Field values only: the private resolution
    bookkeeping and the ``model_config`` memo that a ``vars()`` dump carried into
    the readback are not configuration.
    """
    return {
        field.name: _plain(resolution_result(server_args, field.name))
        for field in dataclasses.fields(server_args)
    }


def _plain(value: Any) -> Any:
    """``dataclasses.asdict``'s conversion, applied to one value: dataclasses
    become dicts, containers recurse, everything else is deep-copied (a caller
    mutating the dump must not reach the live configuration)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _plain(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, tuple) and hasattr(value, "_fields"):  # namedtuple
        return type(value)(*(_plain(item) for item in value))
    if isinstance(value, (list, tuple)):
        return type(value)(_plain(item) for item in value)
    if isinstance(value, dict):
        return type(value)((_plain(k), _plain(v)) for k, v in value.items())
    return copy.deepcopy(value)


def pre_capture_activation_reserve_mb_of(cfg: Any, gpu_mem: Optional[float]) -> float:
    """The activation working-set reserve held back before cuda-graph capture.

    The config-shaped half of the pair; `runtime_context` carries the
    published-bag half, and `TestDerivedPredicatesAgreeAcrossTiers` pins the
    two equal.
    """
    if cfg.disaggregation_mode == "decode":
        running_requests = (
            cfg.max_running_requests or cfg.cuda_graph_config.decode.max_bs or 1
        )
        activation_tokens = max(
            running_requests * (cfg.speculative_num_draft_tokens or 1), 2048
        )
    elif cfg.chunked_prefill_size > 0:
        activation_tokens = max(cfg.chunked_prefill_size, 2048)
    else:
        activation_tokens = max(cfg.max_prefill_tokens, 2048)
    reserved_mem = 512 + activation_tokens * 1.5 + cfg.tp_size * cfg.pp_size / 8 * 1024
    if gpu_mem is not None and gpu_mem > 60 * 1024:
        reserved_mem = max(reserved_mem, 10 * 1024)
    return reserved_mem


def kv_event_block_size_of(cfg: Any) -> int:
    """Width KV events are emitted at.

    Under DCP the radix tree pages at ``page_size * dcp_size``
    (`mem_cache/kv_cache_builder.py`), and subscribers key on this.
    """
    return cfg.page_size * cfg.dcp_size


def modelexpress_config_of(cfg: Any) -> dict:
    """``modelexpress_config`` parsed.

    It is a JSON string (or an already-parsed dict) rather than a leaf of its
    own, so everything that wants a key out of it goes through here -- one parse,
    for every reader.
    """
    raw = cfg.modelexpress_config
    if raw is None:
        return {}
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def modelexpress_url_of(cfg: Any) -> Optional[str]:
    """The modelexpress endpoint a config-shaped object points at."""
    return modelexpress_config_of(cfg).get("url")


def modelexpress_transport_of(cfg: Any) -> str:
    """The modelexpress transport a config-shaped object asks for.

    The shared parse for the transfer-engine gate
    (`remote_instance_transfer_engine_of`) and any bag reader.
    """
    return modelexpress_config_of(cfg).get("transport", "nixl")


def remote_instance_transfer_engine_of(cfg: Any, load_format: Any = None) -> bool:
    """Whether remote-instance weight loading runs over the transfer engine.

    ``load_format`` overrides the config's: a draft runner loading under
    ``--speculative-draft-load-format`` needs its own transfer engine. Every
    input is a ``model`` leaf, so this serves both the pre-publish member and
    the post-publish accessor."""
    if cfg.remote_instance_weight_loader_start_seed_via_transfer_engine:
        return True
    if (load_format or cfg.load_format) != "remote_instance":
        return False
    backend = cfg.remote_instance_weight_loader_backend
    return backend == "transfer_engine" or (
        backend == "modelexpress"
        and modelexpress_transport_of(cfg) == "transfer_engine"
    )


def mamba_extra_buffer_lazy_of(cfg: Any) -> bool:
    """The lazy variant of :func:`mamba_extra_buffer_of`."""
    return (
        cfg.disable_radix_cache is False
        and cfg.mamba_radix_cache_strategy == "extra_buffer_lazy"
    )


def collect_model_override_declarations(
    architecture: str, server_args: Any, hf_config: Any
) -> List[Tuple[str, Dict[str, Any]]]:
    """Collect ``(source, declaration)`` pairs for one architecture.

    Application order (last writer wins downstream in the gate): the constant
    ``MODEL_OVERRIDES`` entry first, then exact-keyed callables in
    registration order, then matching predicate-keyed callables in
    registration order. Empty declarations are dropped.
    """
    # Off the module, not through the imported names: the registrars append to
    # the base's objects, and a copied name here would be a second binding.
    declarations: List[Tuple[str, Dict[str, Any]]] = []
    const = model_override_base.MODEL_OVERRIDES.get(architecture)
    if const:
        declarations.append((f"MODEL_OVERRIDES[{architecture!r}]", dict(const)))
    for fn in model_override_base._MODEL_OVERRIDE_FNS.get(architecture, ()):
        declared = _invoke_provider(fn, server_args, hf_config)
        if declared:
            declarations.append((fn.__qualname__, dict(declared)))
    for predicate, fn in model_override_base._PREDICATE_OVERRIDE_FNS:
        if predicate(architecture):
            declared = _invoke_provider(fn, server_args, hf_config)
            if declared:
                declarations.append((fn.__qualname__, dict(declared)))
    return declarations


# ---------------------------------------------------------------------------
# Derived per-family declarations (faithful ports of legacy arch branches).
# Callables read the PRISTINE server_args, never write; logging is kept
# verbatim from the legacy branch for operator-visible fidelity.
# ---------------------------------------------------------------------------


# Importing the package is what registers the per-model declarations.
import sglang.srt.arg_groups.model_overrides  # noqa: F401


@register_model_override_predicate(
    lambda arch: (
        "Step3p5ForCausalLM" in arch or "Step3p7ForConditionalGeneration" in arch
    )
)
def _step3p_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    overrides: Dict[str, Any] = {}
    if is_attention_backend_not_set(cfg):
        if get_platform().is_blackwell:
            logger.info("Auto-select fa4 attention backend for Step3p7 on Blackwell.")
            overrides["attention_backend"] = "fa4"
        elif get_platform().is_sm90:
            logger.info("Auto-select fa3 attention backend for Step3p7 on Hopper.")
            overrides["attention_backend"] = "fa3"
    if cfg.speculative_algorithm == "EAGLE":
        logger.info(
            "Enable multi-layer EAGLE speculative decoding for Step3p5ForCausalLM model."
        )
        overrides["enable_multi_layer_eagle"] = True
    if cfg.enable_hierarchical_cache:
        logger.warning(
            "Reset swa_full_tokens_ratio to 1.0 for Step3p5ForCausalLM model with hierarchical cache"
        )
        overrides["swa_full_tokens_ratio"] = 1.0
        logger.warning(
            "Disable hybrid SWA memory for Step3p5ForCausalLM model with hierarchical cache"
        )
        overrides["disable_hybrid_swa_memory"] = True
    return overrides


# ---------------------------------------------------------------------------
# Post-process passes (normalization stage).
# ---------------------------------------------------------------------------


# Architectures whose monolith branch routes through the mamba radix cache
# handling (hybrid linear-attention models). Keep in sync with the branch
# guards in _handle_model_specific_adjustments.
_MAMBA_RADIX_CACHE_ARCHS = frozenset(
    {
        "KimiLinearForCausalLM",
        "KimiK3ForConditionalGeneration",
        "BailingMoeV2_5ForCausalLM",
        "BailingMoeV3ForCausalLM",
        "Qwen3NextForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
        "InternS2PreviewForConditionalGeneration",
        "InternS2MobiusForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
        # Text-only entries of the same hybrid stack (models/qwen3_5_text.py);
        # Qwen3.8-2.4T-A95B ships as Qwen3_5MoeForCausalLM.
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5ForCausalLM",
        "MiniCPMV4_6ForConditionalGeneration",
        "NemotronHForCausalLM",
        "NemotronHPuzzleForCausalLM",
        "FalconH1ForCausalLM",
        "JetNemotronForCausalLM",
        "JetVLMForConditionalGeneration",
        "Lfm2ForCausalLM",
        "Lfm2MoeForCausalLM",
        "ZayaForCausalLM",
    }
)

# Architectures that support the extra_buffer mamba radix cache strategy.
# The single source of truth; `supports_mamba_cache_extra_buffer` reads it.
_MAMBA_EXTRA_BUFFER_ARCHS = frozenset(
    {
        "KimiLinearForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        # Text-only entries of the same hybrid stack (models/qwen3_5_text.py);
        # Qwen3.8-2.4T-A95B ships as Qwen3_5MoeForCausalLM.
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5ForCausalLM",
        "Qwen3NextForCausalLM",
        "InternS2PreviewForConditionalGeneration",
        "MiniCPMV4_6ForConditionalGeneration",
        "BailingMoeV2_5ForCausalLM",
        "BailingMoeV3ForCausalLM",
        "FalconH1ForCausalLM",
        "GraniteMoeHybridForCausalLM",
        "NemotronHForCausalLM",
        "NemotronHPuzzleForCausalLM",
        # KDA-based: same MambaPool ping-pong machinery as GDN; requires the
        # KDA backend's track-snapshot writes (decode + extend) so donated
        # slots hold real states for prefix-cache restores.
        "KimiK3ForConditionalGeneration",
    }
)


def supports_mamba_cache_extra_buffer(view: Any, model_arch: str) -> bool:
    """Whether ``model_arch`` supports the extra_buffer strategy on the
    configured linear-attention backend (pure read)."""
    if model_arch in _MAMBA_EXTRA_BUFFER_ARCHS:
        return view.linear_attn_backend == "triton"
    return False


@register_post_process
def _mamba_radix_cache_resolution(view: Any) -> dict:
    """Resolve the hybrid-mamba radix cache fields (pure).

    Slot pass: invoked at each legacy ``_handle_mamba_radix_cache`` slot —
    the hybrid-spec call at the head of the monolith and the per-arch branch
    calls — where it reads the mid-resolution ``page_size`` /
    ``disable_overlap_schedule`` exactly as the legacy helper did. The arch
    guard replicates the union of the legacy call-site guards so the pass is
    self-sufficient in the end-state pass list.
    """
    from sglang.srt.configs.linear_attn_model_registry import (
        get_linear_attn_spec_by_arch,
    )

    hf_config = model_config_of(view).hf_config
    model_arch = hf_config.architectures[0]

    in_branch = model_arch in _MAMBA_RADIX_CACHE_ARCHS
    if model_arch == "GraniteMoeHybridForCausalLM":
        in_branch = any(
            layer_type == "mamba"
            for layer_type in getattr(hf_config, "layer_types", [])
        )
    spec = get_linear_attn_spec_by_arch(model_arch)
    if not ((spec is not None and spec.uses_mamba_radix_cache) or in_branch):
        return {}

    if view.disable_radix_cache:
        return {}

    declared: Dict[str, Any] = {"uses_mamba_radix_cache": True}
    if view.mamba_radix_cache_strategy == "auto":
        wants_overlap = not view.disable_overlap_schedule
        wants_paging = view.page_size is not None and view.page_size > 1
        if (wants_overlap or wants_paging) and supports_mamba_cache_extra_buffer(
            view, model_arch
        ):
            declared["mamba_radix_cache_strategy"] = "extra_buffer"
        else:
            declared["mamba_radix_cache_strategy"] = "no_buffer"
            declared["disable_overlap_schedule"] = True
    return declared


@register_post_process
def _dsa_kv_cache_dtype_default(view: Any) -> dict:
    """Slot pass in the DSA arm, ordered before the split-backend
    resolution: default the kv-cache dtype from the device capability
    (Blackwell FP8, Hopper bf16) and normalize the bf16 alias. Reads the
    PRISTINE dsa split backends (their resolution runs after this pass)."""
    from sglang.srt.configs.model_config import is_deepseek_dsa

    hf_config = model_config_of(view).hf_config
    if hf_config.architectures[0] not in _DEEPSEEK_FAMILY_ARCHS:
        return {}
    if not is_deepseek_dsa(hf_config):
        return {}
    if get_platform().is_npu or get_platform().is_xpu:
        return {}

    import torch

    major, _ = torch.cuda.get_device_capability()

    # If user specified a backend but didn't explicitly set kv_cache_dtype,
    # suggest them to be explicit about kv_cache_dtype to avoid surprises
    if (
        view.dsa_prefill_backend is not None or view.dsa_decode_backend is not None
    ) and view.kv_cache_dtype == "auto":
        logger.warning(
            "When specifying --dsa-prefill-backend or --dsa-decode-backend, "
            "you should also explicitly set --kv-cache-dtype (e.g., 'fp8_e4m3' or 'bfloat16'). "
            "DeepSeek V3.2 defaults to FP8 KV cache which may not be compatible with all backends."
        )

    kv_cache_dtype = view.kv_cache_dtype
    has_attention_sinks = bool(getattr(hf_config, "learnable_sink", False))
    if has_attention_sinks and kv_cache_dtype not in ("auto", "bf16", "bfloat16"):
        raise ValueError(
            "Learnable DSA attention sinks require a bfloat16 KV cache; "
            f"got kv_cache_dtype={kv_cache_dtype}."
        )
    if kv_cache_dtype == "auto":
        kv_cache_dtype = (
            "fp8_e4m3" if major >= 10 and not has_attention_sinks else "bfloat16"
        )
        logger.warning(
            f"Setting KV cache dtype to {kv_cache_dtype} for DeepSeek DSA on SM{major} device."
        )
    if kv_cache_dtype == "bf16":
        kv_cache_dtype = "bfloat16"
    assert kv_cache_dtype in [
        "bfloat16",
        "fp8_e4m3",
    ], "DeepSeek DSA only supports bf16/bfloat16 or fp8_e4m3 kv_cache_dtype"
    if kv_cache_dtype != view.kv_cache_dtype:
        return {"kv_cache_dtype": kv_cache_dtype}
    return {}


def _check_tilelang_dsa_fp8_kv(
    kv_cache_dtype: str,
    prefill_backend: Optional[str],
    decode_backend: Optional[str],
    *,
    hip: bool,
) -> None:
    """tilelang's fp8 KV path is ROCm-only; the CUDA kernel hardcodes bfloat16.
    Reject here instead of crashing at decode CUDA-graph capture."""
    if (
        not hip
        and kv_cache_dtype == "fp8_e4m3"
        and "tilelang" in {prefill_backend, decode_backend}
    ):
        raise ValueError(
            "The tilelang DSA prefill/decode kernels only support an fp8_e4m3 KV "
            "cache on ROCm/HIP; on CUDA they require a bfloat16 KV cache. Use "
            "--kv-cache-dtype bfloat16 with the tilelang backend, or keep "
            "--kv-cache-dtype fp8_e4m3 and pick an fp8-capable DSA backend "
            "(flashmla_kv on Hopper, trtllm on Blackwell)."
        )


@register_post_process
def _dsa_split_backend_resolution(view: Any) -> dict:
    """Slot pass in the DSA arm: default the DSA prefill/decode split
    backends from the mid-resolution kv-cache dtype and the device
    capability. The hisparse arm takes precedence under --enable-hisparse."""
    from sglang.srt.configs.model_config import is_deepseek_dsa

    hf_config = model_config_of(view).hf_config
    if hf_config.architectures[0] not in _DEEPSEEK_FAMILY_ARCHS:
        return {}
    if not is_deepseek_dsa(hf_config):
        return {}
    if get_platform().is_npu or get_platform().is_xpu:
        return {}

    import torch

    major, _ = torch.cuda.get_device_capability()
    kv_cache_dtype = view.kv_cache_dtype
    user_set_prefill = view.dsa_prefill_backend is not None
    user_set_decode = view.dsa_decode_backend is not None
    declared: Dict[str, Any] = {}
    model_arch = hf_config.architectures[0]
    is_glm_sm12_fp8 = (
        model_arch == "GlmMoeDsaForCausalLM"
        and major == 12
        and kv_cache_dtype == "fp8_e4m3"
        and not get_platform().is_hip
    )

    if getattr(hf_config, "learnable_sink", False):
        backend = "flashmla_sparse"
        for field in ("dsa_prefill_backend", "dsa_decode_backend"):
            value = getattr(view, field)
            if value is not None and value != backend:
                option = "--" + field.replace("_", "-")
                raise ValueError(
                    f"{model_arch} uses learnable attention sinks and requires "
                    f"{option} {backend!r}; got {value!r}"
                )
        if not user_set_prefill:
            declared["dsa_prefill_backend"] = backend
        if not user_set_decode:
            declared["dsa_decode_backend"] = backend
        logger.warning(
            "Set DSA backends for learnable attention sinks: "
            f"prefill={declared.get('dsa_prefill_backend', view.dsa_prefill_backend)}, "
            f"decode={declared.get('dsa_decode_backend', view.dsa_decode_backend)}."
        )
        return declared

    if is_glm_sm12_fp8:
        backend = "flashinfer_sparse_mla"
        if not user_set_prefill:
            declared["dsa_prefill_backend"] = backend
        if not user_set_decode:
            declared["dsa_decode_backend"] = backend
        logger.warning(
            "Set DSA backends for GLM FP8 KV Cache on SM120/SM121: "
            f"prefill={backend}, decode={backend}."
        )
        return declared

    if view.enable_hisparse:
        from sglang.srt.arg_groups.hisparse_hook import _hisparse_default_backend

        backend = _hisparse_default_backend(kv_cache_dtype)
        if not user_set_prefill:
            declared["dsa_prefill_backend"] = backend
        if not user_set_decode:
            declared["dsa_decode_backend"] = backend
        prefill = declared.get("dsa_prefill_backend", view.dsa_prefill_backend)
        decode = declared.get("dsa_decode_backend", view.dsa_decode_backend)
        logger.warning(
            f"HiSparse enabled ({kv_cache_dtype}): using DSA backends "
            f"prefill={prefill}, decode={decode}."
        )
        return declared

    if not user_set_prefill and not user_set_decode and get_platform().is_hip:
        declared["dsa_prefill_backend"] = "tilelang"
        declared["dsa_decode_backend"] = "tilelang"
    elif kv_cache_dtype == "fp8_e4m3":
        # Blackwell FP8 defaults to trtllm; Hopper FP8 to flashmla_kv.
        default = "trtllm" if major >= 10 else "flashmla_kv"
        if not user_set_prefill:
            declared["dsa_prefill_backend"] = default
        if not user_set_decode:
            declared["dsa_decode_backend"] = default
    else:
        # Set prefill/decode backends based on hardware architecture.
        if not user_set_prefill:
            declared["dsa_prefill_backend"] = "flashmla_sparse"
        if not user_set_decode:
            declared["dsa_decode_backend"] = "trtllm" if major >= 10 else "fa3"

    prefill = declared.get("dsa_prefill_backend", view.dsa_prefill_backend)
    decode = declared.get("dsa_decode_backend", view.dsa_decode_backend)
    _check_tilelang_dsa_fp8_kv(
        kv_cache_dtype, prefill, decode, hip=get_platform().is_hip
    )
    logger.warning(
        f"Set DSA backends for {kv_cache_dtype} KV Cache: "
        f"prefill={prefill}, decode={decode}."
    )
    return declared


# Keep in sync with the DeepSeek family list on _deepseek_family_overrides.
_DEEPSEEK_FAMILY_ARCHS = frozenset(
    {
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        "KimiK25ForConditionalGeneration",
        "MistralLarge3ForCausalLM",
        "PixtralForConditionalGeneration",
        "GlmMoeDsaForCausalLM",
        "HYV4ForCausalLM",
        "HYV4ForCausalLMNextN",
        "LongcatFlashForCausalLM",
        "LongcatFlashForCausalLMNextN",
        "Dots3NoteForCausalLM",
    }
)


@register_post_process
def _deepseek_moe_quant_resolution(view: Any) -> dict:
    """Slot pass invoked from inside the DeepSeek arch branch ("Set moe
    backend for DeepSeek"), NOT a dispatch-time declaration: the DSA
    kv-cache-dtype default earlier in the branch must read the PRISTINE
    quantization, so this resolution has to stay at its legacy slot."""
    hf_config = model_config_of(view).hf_config
    model_arch = hf_config.architectures[0]
    if model_arch not in _DEEPSEEK_FAMILY_ARCHS:
        return {}
    overrides: Dict[str, Any] = {}
    if get_platform().is_sm100:
        quant_method = get_quantization_config(hf_config)
        quant_cfg = getattr(hf_config, "quantization_config", None) or {}
        config_groups = quant_cfg.get("config_groups", {})
        group0 = config_groups.get("group_0", {})
        weights_cfg = group0.get("weights", {})
        # this also apply to kimi k2.5
        # since it follow the compressed tensor int4 recipe
        # but not kimi k2 instruct or 0905 instruct.
        is_kimi_k2_k25_thinking_int4 = (
            quant_method == "compressed-tensors"
            and weights_cfg.get("num_bits") == 4
            and weights_cfg.get("group_size") == 32
            and weights_cfg.get("strategy") == "group"
            and weights_cfg.get("type") == "int"
        )
        quantization = view.quantization
        if quantization is None and not view._quantization_explicitly_unset:
            # DeepSeek V3/R1 uses native FP8 MoE experts without
            # declaring it in quantization_config.  However, other
            # models that share the same architecture class (e.g.
            # Moonlight-16B-A3B) are purely BF16.  Check the actual
            # safetensors header instead of assuming FP8 by arch name.
            if quant_method is None and model_arch in ["DeepseekV3ForCausalLM"]:
                from sglang.srt.utils.common import has_fp8_weights_in_checkpoint

                if has_fp8_weights_in_checkpoint(view.model_path):
                    overrides["quantization"] = quantization = "fp8"
                    logger.info(
                        "Detected FP8 expert weights in checkpoint, "
                        "default to fp8 for DeepSeek on sm100"
                    )
                else:
                    logger.info(
                        "No FP8 expert weights found in checkpoint, "
                        "keeping bf16 for DeepSeek-arch model on sm100"
                    )
            else:
                overrides["quantization"] = quantization = quant_method
        if (
            view.moe_a2a_backend == "none"
            and view.moe_runner_backend == "auto"
            # LongCat top-k spans the zero-expert logits, which trtllm-gen's
            # fused routing cannot see.
            and not model_arch.startswith("LongcatFlash")
            and (
                quantization
                in ["fp8", "modelopt_fp8", "modelopt_fp4", "modelopt_mixed"]
                or is_kimi_k2_k25_thinking_int4
                or quantization is None
            )
        ):
            overrides["moe_runner_backend"] = "flashinfer_trtllm"
            if is_kimi_k2_k25_thinking_int4:
                logger.info(
                    "Use flashinfer_trtllm as MoE runner backend on Blackwell for Kimi K2 / K2.5 thinking int4"
                )
            else:
                logger.info(
                    "Use flashinfer_trtllm as MoE runner backend on sm100 for DeepseekV3ForCausalLM"
                )
        if (
            model_arch in ["LongcatFlashForCausalLM", "LongcatFlashForCausalLMNextN"]
            and view.fp8_gemm_runner_backend == "auto"
            and quantization in ["fp8", "modelopt_fp8"]
            and quant_cfg.get("scale_fmt", None) != "ue8m0"
        ):
            overrides["fp8_gemm_runner_backend"] = "flashinfer_trtllm"
            logger.info(
                "Use flashinfer_trtllm as FP8 GEMM backend on Blackwell for LongCat FP8 "
                "checkpoint with non-ue8m0 scales"
            )
    return overrides


@register_post_process
def _deepseek_spec_moe_resolution(view: Any) -> dict:
    """Slot pass at the DeepSeek branch's HIP arm: draft (nextn) spec-MoE
    backends for the DeepSeek fp4 checkpoint. Reads the mid-resolution
    quantization (after _deepseek_moe_quant_resolution) and the pre-a2a
    ep_size, exactly like the legacy in-branch writes."""

    hf_config = model_config_of(view).hf_config
    model_arch = hf_config.architectures[0]
    if model_arch not in _DEEPSEEK_FAMILY_ARCHS:
        return {}
    if not get_platform().is_hip:
        return {}
    if not (
        view.quantization == "modelopt_fp4"
        and view.speculative_algorithm == "EAGLE"
        and (
            view.speculative_moe_runner_backend is None
            or view.speculative_moe_a2a_backend is None
        )
    ):
        return {}
    if envs.SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE.get():
        logger.info(
            "Use deep_gemm moe runner and deepep a2a backend for bf16 nextn layer in deepseek fp4 checkpoint."
        )
        # Validate usage of ep
        if view.ep_size == 1:
            raise ValueError(
                "Invalid configuration: 'deep_gemm' speculative MoE runner backend with "
                "'deepep' a2a backend requires expert parallelism (ep_size > 1). "
                f"Current ep_size is {view.ep_size}. "
                "Please set --ep-size > 1 (e.g., --ep-size 8) to use this configuration, "
                "or change --speculative-moe-a2a-backend to 'none' if expert parallelism is not available."
            )
        return {
            "speculative_moe_runner_backend": "deep_gemm",
            "speculative_moe_a2a_backend": "deepep",
        }
    logger.info(
        "Use triton fused moe by default for bf16 nextn layer in deepseek fp4 checkpoint."
    )
    return {
        "speculative_moe_runner_backend": "triton",
        "speculative_moe_a2a_backend": "none",
    }


@register_post_process
def _deepseek_v4_kv_cache_dtype(view: Any) -> dict:
    """Slot pass in the DeepSeek V4 hook: default the kv-cache dtype to FP8
    (bfloat16 on NPU, where the pool geometry differs) and validate the
    result. The NPU split-backend writes stay in the hook."""
    hf_config = model_config_of(view).hf_config
    model_arch = hf_config.architectures[0]
    if model_arch != "DeepseekV4ForCausalLM":
        return {}

    kv_cache_dtype = view.kv_cache_dtype
    if kv_cache_dtype == "auto":
        kv_cache_dtype = "fp8_e4m3"
        logger.warning(f"Setting KV cache dtype to {kv_cache_dtype} for {model_arch}.")
    if view.device == "npu":
        kv_cache_dtype = "bfloat16"
    assert kv_cache_dtype in [
        "fp8_e4m3",
        "bfloat16",
    ], f"{kv_cache_dtype} is not supported for {model_arch}"
    if kv_cache_dtype != view.kv_cache_dtype:
        return {"kv_cache_dtype": kv_cache_dtype}
    return {}


@register_post_process
def _sparse_head_overlap_disable(view: Any) -> dict:

    if envs.SGLANG_EMBEDDINGS_SPARSE_HEAD.is_set():
        logger.warning(
            "Overlap scheduler is disabled when using sparse head for embedding model."
        )
        return {"disable_overlap_schedule": True}
    return {}


# Architectures with explicit FlashInfer AllReduce Fusion support. Keep in
# sync with the model-side fusion implementations.
_FLASHINFER_ALLREDUCE_FUSION_ARCHS = frozenset(
    {
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        "DeepseekV4ForCausalLM",
        "GptOssForCausalLM",
        "GlmMoeDsaForCausalLM",
        "Glm4MoeForCausalLM",
        "Glm4MoeLiteForCausalLM",
        "MistralLarge3ForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3VLMoeForConditionalGeneration",
        "Qwen3NextForCausalLM",
        "KimiK25ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "InternS2PreviewForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
        "NemotronHForCausalLM",
        "NemotronHPuzzleForCausalLM",
    }
)


@register_post_process
def _flashinfer_allreduce_fusion_auto_enable(view: Any) -> dict:
    """Slot pass at the monolith tail: auto-enable FlashInfer AllReduce
    Fusion on SM90/SM100 for models with explicit support. auto resolves to
    mnnvl on Blackwell (single- and multi-node) and trtllm on SM90
    single-node systems. Reads the mid-resolution enable_dp_attention /
    moe_a2a_backend (after the DeepSeek CP and a2a declarations), exactly
    like the legacy tail block."""
    model_arch = model_config_of(view).hf_config.architectures[0]
    if envs.SGLANG_FLASHINFER_MNNVL_CUTEDSL_AR_FUSION.get() and model_arch in {
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
    }:
        # The Qwen backend owns one workspace for ordinary AR and MoE finalize;
        # do not allocate or fall back to the legacy TRTLLM/MNNVL workspace.
        if view.flashinfer_allreduce_fusion_backend is not None:
            logger.warning(
                "SGLANG_FLASHINFER_MNNVL_CUTEDSL_AR_FUSION owns both Qwen3.5 "
                "AllReduce fusion patterns; suppressing the separately configured "
                "--flashinfer-allreduce-fusion-backend=%s",
                view.flashinfer_allreduce_fusion_backend,
            )
        return {"flashinfer_allreduce_fusion_backend": None}
    if (
        view.flashinfer_allreduce_fusion_backend is None
        and model_arch in _FLASHINFER_ALLREDUCE_FUSION_ARCHS
        and (get_platform().is_sm90 or get_platform().is_sm100)
        and view.tp_size > 1
        and not view.enable_dp_attention
        and (view.nnodes == 1 or get_platform().is_sm100)
        and view.moe_a2a_backend == "none"
    ):
        logger.info(
            f"Auto-enabling FlashInfer AllReduce Fusion on SM90/SM10X for {model_arch}"
        )
        return {"flashinfer_allreduce_fusion_backend": "auto"}
    return {}


@register_post_process
def _enforce_disable_allreduce_fusion(view: Any) -> dict:
    """Slot pass right after the auto-enable: the user's enforce-disable
    switch wins over every model-specific adjustment."""
    if view.enforce_disable_flashinfer_allreduce_fusion:
        logger.info(
            "FlashInfer allreduce fusion is forcibly disabled "
            "via --enforce-disable-flashinfer-allreduce-fusion."
        )
        return {"flashinfer_allreduce_fusion_backend": None}
    return {}


@register_post_process
def _sampling_backend_default(view: Any) -> dict:
    if view.sampling_backend is None:
        return {
            "sampling_backend": (
                "flashinfer" if get_platform().has_flashinfer else "pytorch"
            )
        }
    return {}


@register_post_process
def _deterministic_sampling_backend(view: Any) -> dict:
    if view.enable_deterministic_inference and view.sampling_backend != "ascend":
        logger.warning(
            "Sampling backend is set to pytorch for deterministic inference."
        )
        return {"sampling_backend": "pytorch"}
    return {}


def _deterministic_is_deepseek_model(view: Any) -> bool:
    """Faithful copy of the deterministic handler's arch probe (pure read;
    the handler keeps its own copy for the later deepseek validation)."""
    from sglang.srt.connector import ConnectorType
    from sglang.srt.utils.common import parse_connector_type

    if parse_connector_type(view.model_path) == ConnectorType.INSTANCE:
        return False
    try:
        hf_config = model_config_of(view).hf_config
        return hf_config.architectures[0] in [
            "DeepseekV2ForCausalLM",
            "DeepseekV3ForCausalLM",
            "DeepseekV32ForCausalLM",
            "MistralLarge3ForCausalLM",
            "PixtralForConditionalGeneration",
            "GlmMoeDsaForCausalLM",
            "Glm4MoeLiteForCausalLM",
        ]
    except Exception:
        return False


@register_post_process
def _deterministic_allreduce_fusion_disable(view: Any) -> dict:
    if (
        view.enable_deterministic_inference
        and view.flashinfer_allreduce_fusion_backend is not None
    ):
        logger.warning(
            "Disable --flashinfer-allreduce-fusion-backend because deterministic inference is enabled."
        )
        return {"flashinfer_allreduce_fusion_backend": None}
    return {}


@register_post_process
def _deterministic_attention_backend(view: Any) -> dict:
    if not view.enable_deterministic_inference:
        return {}
    from sglang.srt.server_args import DETERMINISTIC_ATTENTION_BACKEND_CHOICES

    if view.attention_backend is None:
        # User didn't specify attention backend, fallback based on GPU architecture
        if get_platform().is_sm100 or get_platform().is_sm120:
            # Blackwell and newer architectures
            if _deterministic_is_deepseek_model(view):
                # fallback to triton for DeepSeek models because flashinfer
                # doesn't support deterministic inference for DeepSeek models yet
                backend = "triton"
            else:
                # fallback to flashinfer on Blackwell for non-DeepSeek models
                backend = "flashinfer"
        else:
            # Hopper (SM90) and older architectures
            backend = "fa3"
        logger.warning(
            f"Attention backend not specified. Falling back to '{backend}' for deterministic inference. "
            f"You can explicitly set --attention-backend to one of {DETERMINISTIC_ATTENTION_BACKEND_CHOICES}."
        )
        return {"attention_backend": backend}
    elif view.attention_backend not in DETERMINISTIC_ATTENTION_BACKEND_CHOICES:
        # User explicitly specified an incompatible attention backend
        raise ValueError(
            f"Currently only {DETERMINISTIC_ATTENTION_BACKEND_CHOICES} attention backends are supported for deterministic inference, "
            f"but you explicitly specified '{view.attention_backend}'."
        )
    return {}


@register_post_process
def _attention_backend_default(view: Any) -> dict:
    if view.prefill_attention_backend is not None and (
        view.prefill_attention_backend == view.decode_attention_backend
    ):  # override the default attention backend
        return {"attention_backend": view.prefill_attention_backend}
    if view.attention_backend is None:
        backend = get_default_attn_backend(
            view, use_mla_backend(view), model_config_of(view)
        )
        logger.info(
            f"Attention backend not specified. Use {backend} backend by default."
        )
        return {"attention_backend": backend}
    return {}


@register_post_process
def _mla_backend_page_constraints(view: Any) -> dict:
    """Page-size constraints of the MLA/TRTLLM backend family (the raises and
    the cutedsl prefill fallback stay in the handler; only the page snaps are
    declared). The snaps chain on a local value exactly as the legacy blocks
    chained on self.page_size."""
    page_size = view.page_size
    if (
        view.attention_backend == "flashmla"
        or view.decode_attention_backend == "flashmla"
    ):
        logger.warning(
            "FlashMLA only supports a page_size of 64, change page_size to 64."
        )
        page_size = 64
    if (
        view.attention_backend == "cutlass_mla"
        or view.decode_attention_backend == "cutlass_mla"
    ):
        logger.warning(
            "Cutlass MLA only supports a page_size of 128, change page_size to 128."
        )
        page_size = 128
    if (
        view.attention_backend == "trtllm_mla"
        or view.decode_attention_backend == "trtllm_mla"
    ):
        if page_size not in [32, 64]:
            logger.warning(
                f"TensorRT-LLM MLA only supports page_size of 32 or 64, changing page_size from {page_size} to 64."
            )
            page_size = 64
    if (
        view.attention_backend == "tokenspeed_mla"
        or view.decode_attention_backend == "tokenspeed_mla"
    ):
        if page_size not in [32, 64]:
            logger.warning(
                f"tokenspeed_mla only supports page_size of 32 or 64, changing page_size from {page_size} to 64."
            )
            page_size = 64
    if (
        view.attention_backend == "cutedsl_mla"
        or view.decode_attention_backend == "cutedsl_mla"
        or view.prefill_attention_backend == "cutedsl_mla"
    ):
        if page_size not in [32, 64]:
            logger.warning(
                f"CuteDSL MLA only supports page_size of 32 or 64, changing page_size from {page_size} to 64."
            )
            page_size = 64
    if (
        view.attention_backend == "trtllm_mha"
        or view.decode_attention_backend == "trtllm_mha"
        or view.prefill_attention_backend == "trtllm_mha"
        or view.speculative_draft_attention_backend == "trtllm_mha"
    ):
        # 128 runs on trtllm-gen's dynamic tokens-per-page kernels (flashinfer
        # >= 0.6.12), which require GQA and equal QK/V head dims — validated at
        # TRTLLMHAAttnBackend init where the model config is known.
        if page_size not in [16, 32, 64, 128]:
            logger.warning(
                f"TensorRT-LLM MHA only supports page_size of 16, 32, 64 or 128, changing page_size from {page_size} to 64."
            )
            page_size = 64
    if (
        view.attention_backend == "hpc_ops"
        or view.decode_attention_backend == "hpc_ops"
        or view.prefill_attention_backend == "hpc_ops"
    ):
        if page_size != 64:
            logger.warning(
                f"HPC-Ops attention only supports a page_size of 64, changing page_size from {page_size} to 64."
            )
            page_size = 64
    if page_size != view.page_size:
        return {"page_size": page_size}
    return {}


@register_post_process
def _mla_kv_cache_dtype_checks(view: Any) -> dict:
    """Read-only validation pass in the attention-backend compatibility
    handler: the TRT-LLM and tokenspeed MLA backends constrain the resolved
    kv-cache dtype (declarations never reach the field, so the checks read
    the view)."""
    if (
        view.attention_backend == "trtllm_mla"
        or view.decode_attention_backend == "trtllm_mla"
    ):
        if not get_platform().is_blackwell:
            raise ValueError(
                "TRTLLM MLA backend is only supported on Blackwell GPUs (SM100/SM12x). Please use a different backend."
            )
        if view.kv_cache_dtype not in ["fp8_e4m3", "fp4_e2m1", "bf16", "auto"]:
            raise ValueError(
                "TensorRT-LLM MLA backend only supports kv-cache-dtype of fp8_e4m3, fp4_e2m1, bf16, or auto."
            )
    if (
        view.attention_backend == "tokenspeed_mla"
        or view.decode_attention_backend == "tokenspeed_mla"
    ):
        if not get_platform().is_blackwell:
            raise ValueError(
                "tokenspeed_mla backend is only supported on Blackwell GPUs (SM100/SM12x)."
            )
        if view.kv_cache_dtype not in ["fp8_e4m3"]:
            raise ValueError(
                "tokenspeed_mla backend requires kv-cache-dtype=fp8_e4m3, "
                f"got {view.kv_cache_dtype}."
            )
    return {}


@register_post_process
def _hisparse_validation(view: Any) -> dict:
    """Read-only validation pass: --enable-hisparse constraints (model class,
    radix cache, kv dtype, DSA backends) read the resolved values through the
    view."""
    from sglang.srt.arg_groups.hisparse_hook import validate_hisparse

    validate_hisparse(view)
    return {}


@register_post_process
def _cutedsl_prefill_backend_fill(view: Any) -> dict:
    """Slot pass in the attention-backend compatibility handler: CuteDSL MLA
    is decode-only, so validate the combination and default the prefill side
    to trtllm_mla. The trtllm_mha check that follows at the legacy slot reads
    the resolved value through the view."""
    if not (
        view.attention_backend == "cutedsl_mla"
        or view.decode_attention_backend == "cutedsl_mla"
        or view.prefill_attention_backend == "cutedsl_mla"
    ):
        return {}
    assert view.prefill_attention_backend != "cutedsl_mla", (
        "CuteDSL MLA only supports decoding for now"
    )
    if not get_platform().is_sm100:
        raise ValueError(
            "CuteDSL MLA backend is only supported on Blackwell GPUs (SM100). Please use a different backend."
        )
    if view.kv_cache_dtype not in [
        "fp8_e4m3",
        "bf16",
        "bfloat16",
        "auto",
    ]:
        raise ValueError(
            "CuteDSL MLA backend only supports kv-cache-dtype of fp8_e4m3, bf16, or auto."
        )
    if view.prefill_attention_backend is None:
        return {"prefill_attention_backend": "trtllm_mla"}
    return {}


@register_post_process
def _attention_backend_fa3_fp8_fallback(view: Any) -> dict:
    if view.attention_backend == "fa3" and view.kv_cache_dtype == "fp8_e5m2":
        logger.warning(
            "FlashAttention3 only supports fp8_e4m3 if using FP8; "
            "Setting attention backend to triton."
        )
        return {"attention_backend": "triton"}
    return {}


@register_post_process
def _fa4_page_constraint(view: Any) -> dict:
    if (
        (
            view.attention_backend == "fa4"
            or view.decode_attention_backend == "fa4"
            or view.prefill_attention_backend == "fa4"
        )
        and not use_mla_backend(view)
        and get_platform().is_sm100
        # EAGLE topk>1 spec runs the two-pass page-tree cascade, which the FA4
        # CUTLASS kernel aborts on at page_size>1. That path only works at
        # page_size==1, so skip the 128 auto-force for it and keep the default.
        and (view.speculative_eagle_topk or 0) <= 1
    ):
        logger.warning(
            f"FA4 backend only supports page size 128 for non-MLA model architectures, changing page_size from {view.page_size} to 128."
        )
        return {"page_size": 128}
    return {}


@register_post_process
def _attention_backend_platform_fallbacks(view: Any) -> dict:
    if (
        view.attention_backend == "intel_amx"
        and view.device == "cpu"
        and not get_platform().has_amx
    ):
        logger.warning(
            "The current platform does not support Intel AMX, will fallback to torch_native backend."
        )
        return {"attention_backend": "torch_native"}
    if (
        view.attention_backend == "intel_xpu"
        and view.device == "xpu"
        and not xpu_has_xmx_support()
    ):
        logger.warning(
            "The current platform does not support Intel XMX, will fallback to triton backend."
        )
        return {"attention_backend": "triton"}
    return {}


@register_post_process
def _intel_xpu_page_constraint(view: Any) -> dict:
    _, decode_backend = attention_backends_of(view)
    if decode_backend == "intel_xpu":
        if use_mla_backend(view):
            supported_page_sizes = [16, 32, 64, 128]
            msg = "Intel XPU attention backend for MLA Decode"
        else:
            supported_page_sizes = [64, 128]
            msg = "Intel XPU attention backend"
        if view.page_size not in supported_page_sizes:
            logger.warning(
                f"{msg} only supports page_sizes of {supported_page_sizes}, changing page_size from {view.page_size} to 128."
            )
            return {"page_size": 128}
    return {}


@register_post_process
def _attention_backend_dual_chunk(view: Any) -> dict:
    if (
        getattr(model_config_of(view).hf_config, "dual_chunk_attention_config", None)
        is not None
    ):
        if view.attention_backend is None:
            logger.info("Dual chunk attention is turned on by default.")
            return {"attention_backend": "dual_chunk_flash_attn"}
        elif view.attention_backend != "dual_chunk_flash_attn":
            raise ValueError(
                "Dual chunk attention is enabled, but attention backend is set to "
                f"{view.attention_backend}. Please set it to 'dual_chunk_flash_attn'."
            )
    return {}


@register_post_process
def _page_size_default(view: Any) -> dict:
    if view.page_size is not None:
        return {}

    # SHUFFLE 5D vectorized KV layout (aiter backend + pa_decode_gluon)
    # is tuned for and prefers page_size=64 — making it the default
    # when the layout flag is set avoids users having to pass
    # --page-size 64 explicitly. The env var is only consumed by the
    # ROCm AITER backend, so the auto-bump is gated on HIP; on other
    # platforms the SHUFFLE 5D pool has no consumer kernels and the
    # env var is silently ignored (see MHATokenToKVPool).
    if (
        get_platform().is_hip
        and envs.SGLANG_AITER_KV_CACHE_LAYOUT.get().lower() == "vectorized_5d"
    ):
        logger.info(
            "Setting page_size=64 as default for "
            "SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d."
        )
        return {"page_size": 64}
    if not get_platform().is_musa:
        return {"page_size": 1}
    return {"page_size": 64}


@register_post_process
def _data_parallelism_defaults(view: Any) -> dict:
    if view.dp_size == 1 and view.ep_join_mode != "scale":
        return {"enable_dp_attention": False, "enable_dp_lm_head": False}
    return {}


@register_post_process
def _tp_lm_head_all_to_all_default(view: Any) -> dict:
    """Enable the TP LM-head all-to-all path only for pure-DP decode nodes.

    Prefill-only and colocated nodes keep the feature disabled by default: the
    LM-head weight layout is fixed at load time, so enabling the TP path would
    also move their long prefills away from the communication-free DP LM head.
    An explicit CLI value always wins.
    """
    if view.enable_tp_lm_head_all_to_all is not None:
        return {}

    enable = (
        view.disaggregation_mode == "decode"
        and view.enable_dp_attention
        and view.dp_size > 1
        and view.tp_size == view.dp_size
        and view.attn_cp_size == 1
        and not view.enable_dp_lm_head
    )
    return {"enable_tp_lm_head_all_to_all": enable}


@register_post_process
def _dp_lm_head_validation(view: Any) -> dict:
    """Read-only validation pass: dp-attention is a prerequisite for the
    dp LM head and the TP LM-head all-to-all path. Reads the mid-resolution
    values through the view."""
    if view.enable_dp_lm_head:
        assert view.enable_dp_attention, (
            "Please enable dp attention when setting enable_dp_lm_head. "
        )
    if view.enable_tp_lm_head_all_to_all:
        assert view.enable_dp_attention, (
            "Please enable dp attention when setting enable_tp_lm_head_all_to_all."
        )
        assert not view.enable_dp_lm_head, (
            "--enable-tp-lm-head-all-to-all uses a TP-sharded LM head and is "
            "incompatible with --enable-dp-lm-head."
        )
        assert view.tp_size == view.dp_size, (
            "--enable-tp-lm-head-all-to-all currently requires tp_size == "
            f"dp_size, got tp_size={view.tp_size}, dp_size={view.dp_size}."
        )
        assert view.attn_cp_size == 1, (
            "--enable-tp-lm-head-all-to-all currently requires "
            f"attn_cp_size == 1, got {view.attn_cp_size}."
        )
    return {}


@register_post_process
def _moe_runner_backend_quant_constraints(view: Any) -> dict:
    """The quantization-driven moe_runner_backend resolutions at the head of
    _handle_moe_kernel_config. The backend-compatibility asserts and the
    disable_shared_experts_fusion writes (post-publish writers exist for that
    field) stay in the handler."""
    moe_runner_backend = view.moe_runner_backend
    if view.quantization == "nvfp4_online":
        if not get_platform().is_sm100:
            raise ValueError(
                "--quantization nvfp4_online is supported only on "
                "NVIDIA Blackwell SM100/SM103 GPUs."
            )
        if moe_runner_backend == "auto":
            moe_runner_backend = "flashinfer_trtllm"
        elif moe_runner_backend not in [
            "flashinfer_trtllm",
            "flashinfer_trtllm_routed",
            "flashinfer_cutedsl",
        ]:
            raise ValueError(
                "--quantization nvfp4_online supports only "
                "--moe-runner-backend flashinfer_trtllm or "
                "flashinfer_trtllm_routed, or flashinfer_cutedsl."
            )
    # Ascend runs MXFP8 MoE on the Ascend runner; every backend selected below is
    # CUDA/ROCm-only. Forcing one here would not merely pick the wrong runner:
    # FusedMoE keys its w1/w3 shard swap ("flashinfer assumes w31") and its
    # 128-alignment round-up off flashinfer_trtllm, so the experts would silently
    # load with gate and up exchanged. Leave the backend at "auto" and let
    # create_moe_runner resolve it to ASCEND.
    if view.quantization == "mxfp8" and not get_platform().is_npu:
        from sglang.srt.server_args import MXFP8_MOE_RUNNER_BACKEND_CHOICES

        is_gfx95_mxfp8 = get_platform().is_hip and is_gfx95_supported()
        allowed = list(MXFP8_MOE_RUNNER_BACKEND_CHOICES)
        if is_gfx95_mxfp8:
            allowed.append("triton")
        mxfp8_default = "triton" if is_gfx95_mxfp8 else "flashinfer_trtllm"
        if moe_runner_backend == "auto":
            moe_runner_backend = mxfp8_default
        elif moe_runner_backend not in allowed:
            logger.warning(
                "mxfp8 quantization supports only %s backends. Overriding %r.",
                ", ".join(allowed),
                moe_runner_backend,
            )
            moe_runner_backend = mxfp8_default
    if (
        moe_runner_backend == "auto"
        and view.quantization == "modelopt_fp4"
        and get_platform().is_sm120
    ):
        moe_runner_backend = "flashinfer_cutlass"
        logger.info(
            "Use flashinfer_cutlass as MoE runner backend on SM120 for "
            "modelopt_fp4 (trtllm-gen MoE kernels are SM100-only)"
        )
    if moe_runner_backend != view.moe_runner_backend:
        return {"moe_runner_backend": moe_runner_backend}
    return {}


@register_post_process
def _moe_runner_fusion_disable(view: Any) -> dict:
    """FlashInfer CuteDSL / TRT-LLM / TRT-LLM-routed MoE runners require the
    shared-experts fusion disabled; declared at the legacy write slots in
    _handle_moe_kernel_config (before the deprecated cutlass env override, so
    the runner value observed is the pre-override one)."""
    runner = view.moe_runner_backend
    if runner == "flashinfer_cutedsl":
        logger.warning(
            "FlashInfer CuteDSL MoE is enabled. --disable-shared-experts-fusion is automatically set."
        )
        return {"disable_shared_experts_fusion": True}
    if runner in ("flashinfer_trtllm", "experimental_sgl_trtllm"):
        logger.warning(
            "FlashInfer TRTLLM MoE is enabled. --disable-shared-experts-fusion is automatically set."
        )
        return {"disable_shared_experts_fusion": True}
    if runner == "flashinfer_trtllm_routed":
        logger.warning(
            "FlashInfer TRTLLM routed MoE is enabled. --disable-shared-experts-fusion is automatically set."
        )
        return {"disable_shared_experts_fusion": True}
    return {}


@register_post_process
def _a2a_fusion_adjustments(view: Any) -> dict:
    """A2A-backend-driven shared-experts fusion adjustments, declared at the
    legacy write slots in _handle_a2a_moe: Waterfill requires the
    fusion enabled; FlashInfer and DeepEP v2 A2A require it disabled."""
    if view.moe_a2a_backend in ("deepep", "megamoe") and view.enable_waterfill:
        if view.disable_shared_experts_fusion:
            logger.warning(
                "disable_shared_experts_fusion is overridden to False because Waterfill requires shared expert fusion."
            )
            return {"disable_shared_experts_fusion": False}
        return {}
    if view.moe_a2a_backend == "flashinfer":
        logger.warning(
            "Flashinfer MoE A2A is enabled. --disable-shared-experts-fusion is automatically set."
        )
        return {"disable_shared_experts_fusion": True}
    if view.moe_a2a_backend == "deepep_v2":
        # Fused shared experts are not validated with DeepEP v2.
        return {"disable_shared_experts_fusion": True}
    return {}


# Every A2A backend that forces expert parallelism to span the TP group.
_A2A_EP_SPANNING_BACKENDS = frozenset(
    {
        "megamoe",
        "deepep",
        "deepep_v2",
        "mooncake",
        "nixl",
        "ascend_fuseep",
        "flashinfer",
        "mori",
        "pplx",
        "deepep_v2",
    }
)


@register_post_process
def _a2a_backend_overrides(view: Any) -> dict:

    moe_a2a_backend = view.moe_a2a_backend
    if view.enable_waterfill and moe_a2a_backend not in ("deepep", "megamoe"):
        logger.warning(
            "moe_a2a_backend is overridden to 'deepep' because Waterfill "
            "requires the DeepEP or MegaMOE backend."
        )
        moe_a2a_backend = "deepep"
    if moe_a2a_backend != view.moe_a2a_backend:
        return {"moe_a2a_backend": moe_a2a_backend}
    return {}


@register_post_process
def _a2a_ep_size(view: Any) -> dict:
    if view.moe_a2a_backend in _A2A_EP_SPANNING_BACKENDS:
        if view.ep_size != view.tp_size:
            logger.info(
                f"{view.moe_a2a_backend} MoE is enabled. The expert parallel size "
                f"is adjusted from {view.ep_size} to the tensor parallel size "
                f"[{view.tp_size}]."
            )
        return {"ep_size": view.tp_size}
    return {}


@register_post_process
def _pipeline_parallel_overlap_disable(view: Any) -> dict:
    if view.pp_size > 1:
        logger.warning("Pipeline parallelism is incompatible with overlap schedule.")
        return {"disable_overlap_schedule": True}
    return {}


@register_post_process
def _speculative_moe_runner_default(view: Any) -> dict:
    """Default the speculative (draft) MoE runner backend to the resolved
    target-model backend. Invoked at the head of the speculative-decoding
    hook, after the MoE kernel chain has resolved."""
    if view.speculative_moe_runner_backend is None:
        return {"speculative_moe_runner_backend": view.moe_runner_backend}
    return {}


@register_post_process
def _gguf_quantization(view: Any) -> dict:
    from sglang.srt.utils.hf_transformers_utils import check_gguf_file

    if (view.load_format == "auto" or view.load_format == "gguf") and check_gguf_file(
        view.model_path
    ):
        return {"quantization": "gguf"}
    return {}


@register_post_process
def _dllm_attention_backend(view: Any) -> dict:
    if view.dllm_algorithm is None:
        return {}
    if get_platform().is_hip:
        if view.attention_backend not in ["triton", "aiter"]:
            logger.warning(
                "Attention backend is set to triton for diffusion LLM inference on AMD GPUs"
            )
            return {"attention_backend": "triton"}
    elif get_platform().is_npu:
        if view.attention_backend != "ascend":
            logger.warning(
                "Attention backend is overridden to 'ascend' when running on NPU for diffusion LLM inference."
            )
            return {"attention_backend": "ascend"}
    elif view.cuda_graph_config.decode.backend != Backend.DISABLED:
        if view.attention_backend != "flashinfer":
            logger.warning(
                "Attention backend is set to flashinfer because of enabling cuda graph in diffusion LLM inference"
            )
            return {"attention_backend": "flashinfer"}
    return {}


@register_post_process
def _dllm_overlap_disable(view: Any) -> dict:
    if view.dllm_algorithm is None:
        return {}
    if view.disable_overlap_schedule:
        return {}
    logger.warning(
        "Overlap schedule is disabled because of using diffusion LLM inference"
    )
    return {"disable_overlap_schedule": True}


@register_post_process
def _dllm_page_size(view: Any) -> dict:
    if view.dllm_algorithm is None:
        return {}
    from sglang.srt.dllm.config import DllmConfig

    config = DllmConfig.from_server_args(view)
    if not view.disable_radix_cache and view.page_size % config.block_size != 0:
        logger.warning(
            f"Setting page size to {config.block_size} for diffusion LLM inference"
        )
        return {"page_size": config.block_size}
    if view.page_size > config.block_size:
        # Legacy scheduler-init fallback, folded into the pass: the page
        # size must not exceed the dllm block size.
        logger.warning(
            "WARNING: "
            f"The page size {view.page_size} should not be larger than dllm block size {config.block_size}."
            f"Page size now falls back to {config.block_size}"
        )
        return {"page_size": config.block_size}
    return {}


def validate_declarations(
    server_args: Any,
    declarations: Sequence[Tuple[str, Dict[str, Any]]],
) -> None:
    """Fail-fast whitelist check at declaration time: a registry typo or a
    not-yet-resolvable field must be rejected at its slot, not only at
    publish time. Declarations never mutate ``server_args``.
    """
    # Non-dataclass fixtures carry no Arg metadata (mirrors the
    # resolvable_fields escape); only real ServerArgs is validated.
    if not dataclasses.is_dataclass(type(server_args)):
        return
    whitelist = resolvable_fields(type(server_args))
    for source, decl in declarations:
        unknown = set(decl) - whitelist
        if unknown:
            raise ValueError(
                f"{source}: {sorted(unknown)} not model-overridable; "
                "declarations are limited to the fields the publish gate "
                "accepts."
            )


@register_post_process
def _hrm_text_attention_force(view: Any) -> dict:
    """HRM-Text's bidirectional prefix attention only works on the Triton
    backend. Invoked as the last attention declaration of the resolution
    (mirroring the legacy runner-side force, which ran after the whole
    pipeline)."""
    if view.attention_backend not in (None, "triton"):
        logger.warning(
            f"Overriding --attention-backend "
            f"{view.attention_backend!r} -> 'triton': only the "
            "Triton backend supports HRM-Text's bidirectional prefix "
            "attention."
        )
    return {"attention_backend": "triton"}


def should_report_expert_balancedness(server_args: Any) -> bool:
    cfg = resolving_view(server_args)
    return cfg.expert_balancedness_report_mode != "off"


def post_capture_kv_sizing_planned(server_args: Any) -> bool:
    """Whether the mem_fraction heuristic may skip the graph reserve; must be
    False for any config the runtime won't post-capture-size, else it gets an
    under-reserved fraction."""
    cfg = resolving_view(server_args)
    mla_enabled = use_mla_backend(server_args)
    if not envs.SGLANG_ENABLE_POST_CAPTURE_KV_SIZING.get():
        return False
    if cfg.device != "cuda":
        return False
    if cfg.dcp_size != 1:
        return False
    if mla_enabled:
        return False
    if cfg.kv_cache_dtype == "fp4_e2m1":
        return False
    if cfg.prefill_only_disable_kv_cache:
        return False
    if cfg.enable_memory_saver:
        return False
    if envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.get() is not None:
        return False

    if (
        cfg.disaggregation_mode != "prefill"
        and cfg.cuda_graph_config.decode.backend == Backend.DISABLED
    ):
        return False

    if cfg.disaggregation_mode != "decode":
        prefill_cfg = cfg.cuda_graph_config.prefill
        # We can only skip eager activation headroom when the largest
        # prefill forward batch size is already graph-captured. Otherwise,
        # an eager forward will need more memory and lead to OOM.
        if (
            prefill_cfg.backend == Backend.DISABLED
            or cfg.chunked_prefill_size <= 0
            or max_prefill_buffer_tokens(server_args) > max(prefill_cfg.bs or (0,))
        ):
            return False

    from sglang.srt.configs.model_config import is_deepseek_v4, is_minimax_sparse

    hf_config = model_config_of(server_args).hf_config
    if is_deepseek_v4(hf_config) or is_minimax_sparse(hf_config):
        return False

    return True


def cutedsl_moe_max_num_tokens(server_args: Any) -> int:
    """Largest number of tokens a single forward routes through a CuteDSL
    MoE layer on one (DP) rank. Single source of truth for both the
    standard-allgather wrapper buffers and the FlashInfer A2A dispatcher
    budget. Max over the prefill (max_prefill_tokens), piecewise-prefill
    capture, and decode/verify bounds; num_tokens_per_req is
    speculative_num_draft_tokens under speculative decoding, else 1.
    """
    cfg = resolving_view(server_args)
    if cfg.speculative_algorithm:
        num_tokens_per_req = cfg.speculative_num_draft_tokens or 1
    else:
        num_tokens_per_req = 1
    prefill_tokens = cfg.max_prefill_tokens
    cg_config = cfg.cuda_graph_config
    if cg_config is not None and cg_config.prefill.backend == Backend.TC_PIECEWISE:
        prefill_tokens = max(prefill_tokens, cg_config.prefill.max_bs or 0)
    decode_max_bs = (cg_config.decode.max_bs if cg_config is not None else 0) or 0
    decode_tokens = decode_max_bs * num_tokens_per_req
    return max(prefill_tokens, decode_tokens)


def max_prefill_buffer_tokens(server_args: Any) -> int:
    """Prefill-buffer ceiling: chunked_prefill_size, except PP dynamic
    chunking can grow chunks toward max_prefill_tokens and probe at 1.25x."""
    cfg = resolving_view(server_args)
    chunked = (
        cfg.chunked_prefill_size
        if cfg.chunked_prefill_size and cfg.chunked_prefill_size > 0
        else 0
    )
    tokens = chunked
    if cfg.enable_dynamic_chunking and cfg.pp_size > 1 and chunked:
        tokens = max(tokens, cfg.max_prefill_tokens or 0, math.ceil(chunked * 1.25))
    return tokens


def mamba_cache_chunk_size(server_args: Any) -> int:
    # For mamba cache with extra buffer, the chunk size is the max of FLA_CHUNK_SIZE
    # (or mamba_chunk_size if it is defined in the model's config) and page_size.
    # It is used to determine the caching point in a sequence during prefill.
    # A pre-seeded `_mamba_cache_chunk_size` (fixtures supply one so a dummy
    # model never loads an HF config) is honored as-is; otherwise the memo
    # is only kept once the record is resolved, because `page_size` below
    # is resolution-written.
    from sglang.srt.arg_groups.overrides import model_config_of

    if not hasattr(server_args, "_mamba_cache_chunk_size"):
        try:
            from sglang.kernels.ops.attention.fla.chunk_delta_h import (
                CHUNK_SIZE as FLA_CHUNK_SIZE,
            )
        except ImportError:
            # Must match sglang.kernels.ops.attention.fla.chunk_delta_h.CHUNK_SIZE
            FLA_CHUNK_SIZE = 64

        hf_config = model_config_of(server_args).hf_config
        chunk_size = getattr(hf_config, "mamba_chunk_size", FLA_CHUNK_SIZE)
        page_size = resolved_view(server_args).page_size
        assert max(chunk_size, page_size) % min(chunk_size, page_size) == 0, (
            f"For SSM models, either chunk_size or page_size must be divisible by the other, got {chunk_size=}, {page_size=}"
        )
        if not getattr(server_args, "_resolution_finished", False):
            return max(chunk_size, page_size)
        server_args._mamba_cache_chunk_size = max(chunk_size, page_size)
    return server_args._mamba_cache_chunk_size


def max_speculative_num_draft_tokens(server_args: Any) -> Optional[int]:
    """Return the maximum draft-token count speculative decoding may use.

    Memoized only once the record is resolved: an answer computed off a raw
    record describes inputs resolution is about to rewrite (auto speculative
    sizing fills `speculative_num_draft_tokens` in), and a cache filled that
    early would keep answering with it.
    """
    cfg = resolving_view(server_args)

    memo = server_args.__dict__.get("_max_speculative_num_draft_tokens")
    if memo is not None:
        return memo
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    result = SpeculativeAlgorithm.from_string(
        cfg.speculative_algorithm
    ).resolve_max_speculative_num_draft_tokens(server_args)
    if (
        result is not None
        and cfg.speculative_num_draft_tokens is not None
        and result < cfg.speculative_num_draft_tokens
    ):
        raise ValueError(
            "The speculative algorithm declared "
            f"max_speculative_num_draft_tokens={result}, below the configured "
            "speculative_num_draft_tokens="
            f"{cfg.speculative_num_draft_tokens}."
        )
    if getattr(server_args, "_resolution_finished", False):
        server_args._max_speculative_num_draft_tokens = result
    return result
