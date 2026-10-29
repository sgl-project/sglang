# SPDX-License-Identifier: Apache-2.0
"""SGLang's binding to the NVIDIA megakernel plugins behind --enable-nvidia-megakernel.

It checks a plugin's spec against the resolved configuration at startup, builds it
once the model is prepared, and installs the whole-step dispatch on the model. It is
also the one place a megakernel reads SGLang's runtime state: the KV pool, the
attention metadata of the running forward, the DSA (sparse MLA) backend's layout and
buffers and the Engram tables. A spec's build gets a ``MegakernelBinding`` as
``BuildContext.binding``; a covered forward gets the running forward's
``ForwardView`` as ``Step.view`` (a draft entry asks ``binding.forward_view()``).
Everything else a megakernel touches is the model it replaces, SGLang's kernel ops,
and the reads listed below.
"""

from __future__ import annotations

import functools
import logging
import time
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Callable, Collection, Mapping, Optional

import msgspec
import torch
import torch.distributed as dist

from sglang.kernels.plugin_kernels.nvidia.spec import (
    ENTRIES,
    ENTRY_POINT,
    FLAG,
    FLAG_SETS_ENV,
    GRAPH_VARIANT_RUNNERS,
    BuildContext,
    BuiltMegakernel,
    Feature,
    MegakernelSpec,
    Step,
    find_spec,
    registered_architectures,
    spec_registered,
)
from sglang.srt.arg_groups.model_override_base import model_config_of, resolving_view
from sglang.srt.environ import envs
from sglang.srt.kernel_plugins.amd.binding import msa_serves
from sglang.srt.kernel_plugins.nvidia.eagle_binding import EagleBinding
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.plugins import GENERAL_PLUGINS_GROUP
from sglang.srt.runtime_context import (
    get_context,
    get_device,
    get_exec,
    get_parallel,
    get_spec,
)

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from sglang.srt.layers.attention.graph_variants import AttentionGraphVariants
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )

logger = logging.getLogger(__name__)

# HF architecture -> the binding its megakernel gets; every other architecture gets
# MegakernelBinding.
_BINDINGS: dict[str, type] = {
    arch: EagleBinding
    for arch in (
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5ForCausalLM",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5ForCausalLMMTP",
    )
}

# Called by a megakernel, tolerated until offered as data (its contract test pins
# both lists).
TOLERATED_CALLABLES = (
    "ForwardView.attention",  # the attention backend's own forward
    "ForwardView.store_cache",  # the KV write
    "ForwardView.swa_out_cache_loc",
    "ForwardView.low_ratio_sources",  # compressor / indexer inputs
    "ForwardView.core_compressor",
    "KvPoolView.set_swa_key_buffer_radix_fused_norm_rope",  # the fused SWA key write
    "KvPoolView.pool",  # passed back to SGLang's own draft KV store only
    "sglang.kernels.ops",  # SGLang's kernel ops
    "the replaced model's modules, weights and quant methods",
)
# Written into SGLang's objects, tolerated until SGLang owns these buffers.
TOLERATED_WRITES = (
    "ForwardView.metadata.q_pad_buffer",
    "ForwardView.metadata.low_ratio_req_indices",
    "ForwardView.metadata.low_ratio_pos_i64",
)
# Read, tolerated: configuration, not runtime state.
TOLERATED_READS = (
    "runtime_context getters (get_parallel, get_exec, get_spec, get_platform)",
    "capture-mode queries (get_is_capture_mode, is_in_breakable_cuda_graph)",
    "MegakernelBinding.engram_host_table",  # the per-layer host table object
)
# Called by a megakernel on a DSA (sparse MLA) backend, tolerated until offered as
# data.
TOLERATED_DSA_CALLABLES = (
    "KvPoolView.set_mla_kv_buffer",  # the MLA key write
    "DsaView.indexer_metadata",  # the indexer's metadata of the running forward
)
# Written into the DSA pool by a megakernel (the step's rows, at its out_cache_loc),
# tolerated until SGLang owns the writes.
TOLERATED_DSA_WRITES = (
    "KvPoolView.get_key_buffer",  # the MLA key rows
    "KvPoolView.get_index_k_with_scale_buffer",  # the index-K rows and their scales
)


class KvPoolView:
    """The running forward's KV pool, under the pool's own names: its layout and
    per-layer key buffers (a pool offers the ones of its kind)."""

    def __init__(self, pool: Any):
        self.pool = pool

    @property
    def uniform_fp8(self) -> bool:
        return self.pool.uniform_fp8

    @property
    def request_window(self) -> Any:
        return self.pool.request_window

    def get_swa_key_buffer_radix(self, layer_id: int) -> torch.Tensor:
        return self.pool.get_swa_key_buffer_radix(layer_id)

    def get_swa_key_page_size(self) -> int:
        return self.pool.get_swa_key_page_size()

    def get_swa_key_bytes_per_token(self) -> int:
        return self.pool.get_swa_key_bytes_per_token()

    def get_extra_key_buffer(self, layer_id: int) -> Optional[torch.Tensor]:
        return self.pool.get_extra_key_buffer(layer_id)

    def get_extra_key_page_size(self, layer_id: int) -> int:
        return self.pool.get_extra_key_page_size(layer_id)

    def get_extra_key_bytes_per_token(self, layer_id: int) -> int:
        return self.pool.get_extra_key_bytes_per_token(layer_id)

    @property
    def set_swa_key_buffer_radix_fused_norm_rope(self) -> Callable[..., None]:
        return self.pool.set_swa_key_buffer_radix_fused_norm_rope

    # MLA / DSA pools
    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.pool.get_key_buffer(layer_id)

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        return self.pool.get_index_k_with_scale_buffer(layer_id)

    @property
    def index_page_size(self) -> Optional[int]:
        """Tokens per page of the index-K buffer; None for a pool without one."""
        return getattr(self.pool, "index_page_size", None)

    @property
    def set_mla_kv_buffer(self) -> Callable[..., None]:
        return self.pool.set_mla_kv_buffer


class DsaView:
    """The running DSA (sparse MLA) backend's layout and the buffers its decode
    uses, under names a megakernel can rely on."""

    def __init__(self, backend: Any):
        self._backend = backend

    @property
    def kv_lora_rank(self) -> int:
        return self._backend.kv_lora_rank

    @property
    def qk_nope_head_dim(self) -> int:
        return self._backend.qk_nope_head_dim

    @property
    def qk_rope_head_dim(self) -> int:
        return self._backend.qk_rope_head_dim

    @property
    def kv_cache_dim(self) -> int:
        return self._backend.kv_cache_dim

    @property
    def physical_page_size(self) -> int:
        return self._backend.physical_page_size

    @property
    def decode_impl(self) -> str:
        return self._backend.dsa_decode_impl

    @property
    def use_mha(self) -> bool:
        """Whether the running forward takes the MHA path instead of MLA."""
        return self._backend.use_mha

    @property
    def kv_cache_dtype(self) -> torch.dtype:
        return self._backend.kv_cache_dtype

    @property
    def use_fused_topk(self) -> bool:
        return self._backend.use_fused_topk

    @property
    def topk_backend(self) -> Any:
        return self._backend.dsa_topk_backend

    @property
    def dcp_size(self) -> int:
        return self._backend.dcp_size

    @property
    def index_kpool(self) -> int:
        return self._backend.dsa_index_kpool

    @property
    def hisparse(self) -> bool:
        return self._backend.hisparse_coordinator is not None

    @property
    def workspace_buffer(self) -> Optional[torch.Tensor]:
        """The decode kernels' workspace, or None when the backend has none."""
        return self._backend.workspace_buffer

    @property
    def multi_ctas_kv_counter_buffer(self) -> Optional[torch.Tensor]:
        """The TRT-LLM decode's multi-CTA KV counter, sized before the decode graphs
        are captured; None when the backend has none."""
        return self._backend._multi_ctas_kv_counter_buffer

    @property
    def indexer_metadata(self) -> Callable[[int, Any], Any]:
        """The backend's indexer metadata of the running forward, by
        (layer_id, forward_batch)."""
        return self._backend.get_indexer_metadata


class ForwardView:
    """The running forward's attention data, read from its attention backend; built
    at capture for a graph step, so its tensors are the ones the graph replays."""

    def __init__(self, backend: Any):
        self._backend = backend
        self.metadata = backend.forward_metadata
        self.kv_pool = KvPoolView(backend.token_to_kv_pool)

    @property
    def trtllm_attn(self) -> bool:
        return self._backend.trtllm_attn

    @property
    def attention(self) -> Callable[..., Any]:
        return self._backend.forward

    @property
    def store_cache(self) -> Callable[..., None]:
        return self._backend.store_cache

    @property
    def swa_out_cache_loc(self) -> Callable[[Any], torch.Tensor]:
        return self._backend.get_swa_out_cache_loc

    @property
    def low_ratio_sources(self) -> Callable[..., Any]:
        return self._backend.forward_low_ratio_sources

    @property
    def core_compressor(self) -> Callable[..., Any]:
        return self._backend.forward_core_compressor

    @property
    def dsa(self) -> Optional[DsaView]:
        """The DSA backend's view, or None when the running backend is not DSA."""
        return _dsa_view(self._backend)


class MegakernelBinding:
    """Built at install, before the KV pools exist; read at build or forward. Each
    source is sealed at the first read that finds it; ``assert_live`` (after the
    decode graphs are captured) raises once one was reallocated."""

    def __init__(self, *, model: torch.nn.Module):
        self._model = model
        self._sealed: dict[str, tuple] = {}

    def forward_view(self) -> ForwardView:
        """The view of the forward running now (a target step or a draft step)."""
        from sglang.srt.model_executor.forward_context import get_attn_backend

        view = ForwardView(get_attn_backend())
        self._seal("kv_pool", view.kv_pool.pool)
        dsa = view.dsa
        if dsa is not None:
            for name, buffer in _dsa_buffers(dsa).items():
                if buffer is not None:
                    self._seal(name, buffer)
        return view

    def engram_history(self) -> Optional[torch.Tensor]:
        """The model's Engram hash history ``[slots + 1, width]`` int32, or None."""
        history = self._history()
        if history is None or history.shape[1] == 0:
            return None
        self._seal("engram_history", history)
        return history

    def engram_host_table(self, engram: torch.nn.Module) -> Any:
        """The host table of one of the model's Engram layers, or None."""
        return engram.embed.host_table

    def assert_live(self) -> None:
        current = {
            "kv_pool": self._current_pool,
            "dsa_workspace": lambda: self._current_dsa_buffer("dsa_workspace"),
            "dsa_kv_counter": lambda: self._current_dsa_buffer("dsa_kv_counter"),
            "engram_history": self._history,
        }
        for name, sealed in self._sealed.items():
            if _identity(current[name]()) != sealed:
                raise RuntimeError(
                    f"megakernel binding: {name} was reallocated after its first read"
                )

    def _seal(self, name: str, value: Any) -> None:
        self._sealed.setdefault(name, _identity(value))

    def _current_pool(self) -> Any:
        from sglang.srt.model_executor.forward_context import get_attn_backend

        return get_attn_backend().token_to_kv_pool

    def _current_dsa_buffer(self, name: str) -> Optional[torch.Tensor]:
        from sglang.srt.model_executor.forward_context import get_attn_backend

        dsa = _dsa_view(get_attn_backend())
        return None if dsa is None else _dsa_buffers(dsa)[name]

    def _history(self) -> Optional[torch.Tensor]:
        # only models with Engram layers have a hasher
        hasher = getattr(self._model, "engram_hasher", None)
        return None if hasher is None else hasher.history


def _dsa_view(backend: Any) -> Optional[DsaView]:
    # only the DSA backend resolves a DSA decode implementation
    if getattr(backend, "dsa_decode_impl", None) is None:
        return None
    return DsaView(backend)


def _dsa_buffers(dsa: DsaView) -> dict[str, Optional[torch.Tensor]]:
    """The DSA backend's buffers a captured step reads by address, sealed by name."""
    return {
        "dsa_workspace": dsa.workspace_buffer,
        "dsa_kv_counter": dsa.multi_ctas_kv_counter_buffer,
    }


def _identity(value: Any) -> tuple:
    if isinstance(value, torch.Tensor):
        return (value.data_ptr(), tuple(value.shape), value.dtype)
    return (id(value),)


PLUGIN_URL = "https://github.com/sgl-project/sgl-nvidia-mega"

# Leave time for peers to abort before the process watchdog fires.
MAX_DEADLINE_S, MIN_DEADLINE_S = 60.0, 5.0


def device_isa() -> Optional[str]:
    """This process's GPU as "sm_<major><minor>", or None without a visible GPU."""
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return f"sm_{props.major}{props.minor}"


def deadline_s(watchdog_timeout: Optional[float]) -> float:
    """The deadline of every cross-GPU wait: half the watchdog, at most MAX_DEADLINE_S."""
    if not watchdog_timeout or watchdog_timeout <= 0:
        return MAX_DEADLINE_S
    return min(MAX_DEADLINE_S, 0.5 * float(watchdog_timeout))


def _ragged_verify(cfg: Any) -> bool:
    mode = envs.SGLANG_RAGGED_VERIFY_MODE.get().lower()
    return cfg.speculative_algorithm is not None and mode != "static"


# One probe a feature, on the resolved settings; a spec accepts features by listing
# them. PARTITIONED_GPU needs the loaded device, so the runtime checks it.
FEATURE_PROBES = {
    Feature.PIPELINE_PARALLEL: lambda c: c.pp_size > 1,
    Feature.MULTI_NODE_TP: lambda c: c.nnodes > 1,
    Feature.DP_ATTENTION: lambda c: (
        bool(c.enable_dp_attention) or (c.attn_dp_size or 1) > 1
    ),
    Feature.EXPERT_PARALLEL: lambda c: (c.ep_size or 1) > 1,
    Feature.EPLB: lambda c: bool(c.enable_eplb),
    Feature.CONTEXT_PARALLEL: lambda c: (
        (c.attn_cp_size or 1) > 1 or (c.dcp_size or 1) > 1 or bool(c.enable_prefill_cp)
    ),
    Feature.SPECULATIVE_DECODING: lambda c: c.speculative_algorithm is not None,
    Feature.RAGGED_VERIFY: _ragged_verify,
    Feature.TWO_BATCH_OVERLAP: lambda c: bool(
        c.enable_two_batch_overlap or c.enable_single_batch_overlap
    ),
    Feature.LORA: lambda c: bool(c.enable_lora or c.lora_paths),
    Feature.PD_DISAGGREGATION: lambda c: (c.disaggregation_mode or "null") != "null",
    Feature.RETURN_ROUTED_EXPERTS: lambda c: bool(c.enable_return_routed_experts),
    Feature.RETURN_INDEXER_TOPK: lambda c: bool(c.enable_return_indexer_topk),
    Feature.KV_OFFLOAD: lambda c: bool(
        c.enable_hierarchical_cache
        or c.disaggregation_decode_enable_offload_kvcache
        or c.enable_lmcache
        or c.enable_flexkv
    ),
    Feature.HISPARSE: lambda c: bool(c.enable_hisparse),
    Feature.MEMORY_RELEASE: lambda c: bool(
        c.enable_memory_saver or c.enable_weights_cpu_backup
    ),
    Feature.CPU_OFFLOAD: lambda c: (
        (c.cpu_offload_gb or 0) > 0 or (c.offload_group_size or 0) > 0
    ),
    Feature.TORCH_COMPILE: lambda c: bool(c.enable_torch_compile),
    Feature.DETERMINISTIC_INFERENCE: lambda c: bool(
        c.enable_deterministic_inference or c.rl_on_policy_target
    ),
}


def features_in_use(cfg: Any) -> set[Feature]:
    return {feature for feature, probe in FEATURE_PROBES.items() if probe(cfg)}


def plugin_reasons() -> list[str]:
    """Why the plugin that registers the specs will not load ([] when it will)."""
    if ENTRY_POINT not in {ep.name for ep in entry_points(group=GENERAL_PLUGINS_GROUP)}:
        return [
            f"the {ENTRY_POINT!r} plugin is not installed; install sgl-nvidia-mega "
            f"({PLUGIN_URL})"
        ]
    allowed = envs.SGLANG_PLUGINS.get()
    if allowed and ENTRY_POINT not in {x.strip() for x in allowed.split(",")}:
        return [f"SGLANG_PLUGINS={allowed!r} does not list the {ENTRY_POINT!r} plugin"]
    return []


def megakernel_disable_reasons(
    *, cfg: Any, hf_config: Any, isa: Optional[str]
) -> tuple[Optional[MegakernelSpec], list[str]]:
    """The spec for this model and every reason it cannot serve this configuration."""
    if isa is None:
        return None, ["no GPU is visible"]
    spec, why = find_spec(architectures=hf_config.architectures, isa=isa)
    if spec is None and not registered_architectures():
        # the plugin loader logs and skips a plugin that fails, so nothing registered
        why = (
            f"the {ENTRY_POINT!r} plugin is installed but registered no spec; see "
            "'Failed to load plugin' or 'Failed to execute general plugin' above"
        )
    if spec is None:
        return None, [why]
    req = spec.requires
    reasons = [
        f"{f.value} is in use and {spec.name} does not accept it"
        for f in sorted(features_in_use(cfg) - spec.features, key=lambda f: f.name)
    ]
    one_of = [
        ("tp_size", cfg.tp_size, req.tp_sizes),
        (
            "decode CUDA graphs",
            cfg.cuda_graph_config.decode.backend,
            req.decode_graph_backends,
        ),
    ] + [(arg, getattr(cfg, arg), ok) for arg, ok in req.server_args.items()]
    exactly = [
        # another model's config may not have the field at all
        (f"config.{key}", getattr(hf_config, key, None), want)
        for key, want in req.hf_config.items()
    ] + [(name, getattr(envs, name).get(), want) for name, want in req.env.items()]
    reasons += [
        f"{what} {value!r} is not one of {sorted(map(str, allowed))}"
        for what, value, allowed in one_of
        if allowed and value not in allowed
    ]
    reasons += [
        f"{what} {value!r} is not {want!r}"
        for what, value, want in exactly
        if value != want
    ]
    watchdog = cfg.watchdog_timeout
    if watchdog and deadline_s(watchdog) < MIN_DEADLINE_S:
        reasons.append(
            f"--watchdog-timeout {watchdog:g} s leaves the waits a deadline under "
            f"{MIN_DEADLINE_S:g} s"
        )
    if spec.config_disable_reasons is not None:
        reasons += spec.config_disable_reasons(cfg, hf_config)
    return spec, reasons


def _dtype_reasons(spec: Optional[MegakernelSpec], model_config: Any) -> list[str]:
    # the resolved dtype, which the HF config alone does not give
    dtype = str(model_config.dtype).replace("torch.", "")
    if spec is None or not spec.requires.dtypes or dtype in spec.requires.dtypes:
        return []
    return [f"dtype {dtype!r} is not one of {sorted(spec.requires.dtypes)}"]


def set_flag_env(spec: Optional[MegakernelSpec]) -> list[str]:
    """Set each variable of FLAG_SETS_ENV that the spec requires and the user left
    unset to the spec's value; return their names."""
    if spec is None:
        return []
    names = [
        name
        for name in spec.requires.env
        if name in FLAG_SETS_ENV and not getattr(envs, name).is_set()
    ]
    for name in names:
        getattr(envs, name).set(spec.requires.env[name])
    return names


def handle_nvidia_megakernel(server_args: Any) -> None:
    """Resolution step: fail before the weights load when the flag cannot be served,
    listing every reason. The model processes, started after it, inherit what it sets
    for the spec (FLAG_SETS_ENV)."""
    cfg = resolving_view(server_args)
    if not cfg.enable_nvidia_megakernel:
        return
    archs = model_config_of(server_args).hf_config.architectures
    if msa_serves(archs) and not cfg.enable_amd_monokernel:
        return  # handle_monokernel checks the MSA plugin's spec
    env_set = []
    reasons = plugin_reasons()
    if not reasons:
        model_config = model_config_of(server_args)
        hf_config, isa = model_config.hf_config, device_isa()
        env_set = set_flag_env(
            find_spec(architectures=hf_config.architectures, isa=isa)[0]
        )
        spec, reasons = megakernel_disable_reasons(
            cfg=cfg, hf_config=hf_config, isa=isa
        )
        reasons += _dtype_reasons(spec, model_config)
    if cfg.enable_amd_monokernel:
        # both replace the model's forward, which would fail only after weights load
        reasons.append("--enable-amd-monokernel is also set; use one of the two")
    if reasons:
        # a refused setup leaves the environment as it was
        for name in env_set:
            getattr(envs, name).clear()
        raise ValueError(
            f"{FLAG} cannot run this setup:\n  - " + "\n  - ".join(reasons)
        )
    logger.warning("%s is experimental. Its kernels and coverage may change.", FLAG)
    logger.info("%s: %s on %s", FLAG, spec.name, device_isa())
    for name in env_set:
        logger.info(
            "%s: set %s=%s for %s", FLAG, name, getattr(envs, name).get(), spec.name
        )


def megakernel_enabled() -> bool:
    """Whether this server runs with --enable-nvidia-megakernel."""
    # the model loader also runs without a server's config (tools, unit tests)
    if not get_context().is_config_namespace_published("exec"):
        return False
    return get_exec().kernel.enable_nvidia_megakernel


def install_megakernel(
    *,
    model: torch.nn.Module,
    hf_config: PretrainedConfig,
    capture_tail_hooks: list[Callable[..., None]],
    is_draft: bool,
) -> None:
    """Build the model's spec and install its forward; runs before KV cache sizing."""
    if not megakernel_enabled() or msa_serves(hf_config.architectures):
        return
    spec = _find_spec_on_every_rank(hf_config=hf_config, is_draft=is_draft)
    if spec is None:
        return
    # Resolution checked the target's spec on the server arguments; a draft's spec
    # is checked only here, on what the runner reads (no server-argument checks).
    reasons = _requirement_reasons(spec=spec, hf_config=hf_config)
    reasons += _model_disable_reasons(spec=spec, model=model, hf_config=hf_config)
    _refuse(_gather_reasons(reasons))

    tokens = _tokens_per_request()
    binding = _binding_class(hf_config.architectures)(model=model)
    context = BuildContext(
        widths=tuple(range(tokens, spec.max_rows + 1, tokens)),
        tokens_per_request=tokens,
        deadline_s=deadline_s(get_device().watchdog_timeout),
        binding=binding,
    )
    start, free_before = time.perf_counter(), torch.cuda.mem_get_info()[0]
    built = spec.build(model, hf_config, context)
    # not setattr: a built object that is an nn.Module must not join the state dict
    model.__dict__["_megakernel"] = built
    # Different widths would leave ranks waiting in different collectives, and
    # different entries would run different code around them.
    megakernel_agree(tuple(built.widths), what=f"widths {spec.name} built")
    entries = _offered_entries(built)
    megakernel_agree(entries, what=f"entries {spec.name} offers")
    variants = _offered_graph_variants(built)
    megakernel_agree(
        sorted((runner, tuple(v.capture_labels)) for runner, v in variants.items()),
        what=f"graph variants {spec.name} offers",
    )
    _refuse(
        [
            f"graph_variants names {runner!r}, not a runner of "
            f"{list(GRAPH_VARIANT_RUNNERS)}"
            for runner in sorted(variants)
            if runner not in GRAPH_VARIANT_RUNNERS
        ]
    )
    if spec.max_rows > 0 and not built.widths:
        _refuse([f"{spec.name} built no width of {list(context.widths)}"])
    # Captured kernels keep state derived from the weights that updates cannot refresh.
    model._derived_weight_cache_error = (
        f"Online weight updates are not supported with {FLAG} ({spec.name})"
    )
    built_s = time.perf_counter() - start
    used_mib = (free_before - torch.cuda.mem_get_info()[0]) / 2**20
    if not built.widths:
        # A spec may cover no step while offering only draft-side entries.
        if built.draft_sampler is not None:
            offers = "draft sampler"
        elif entries:
            offers = "entries " + ", ".join(entries)
        else:
            offers = "no step and no draft sampler"
        logger.info("NVIDIA megakernel: %s: %s", spec.name, offers)
        return
    logger.info(
        "NVIDIA megakernel: active (%s: %s; widths %s; %s numerics; built in %.1f s, "
        "%.1f MiB; wait deadline %.0f s)",
        spec.name,
        built.summary(),
        list(built.widths),
        spec.numerics.kind.value,
        built_s,
        used_mib,
        context.deadline_s,
    )
    dispatch = _Dispatch(built=built, original=model.forward, binding=binding)
    model.forward = dispatch.wrapped()
    if _decode_graphs_enabled():
        capture_tail_hooks.append(_CaptureCheck(spec=spec, dispatch=dispatch))


def built_megakernel(model: torch.nn.Module) -> Optional[BuiltMegakernel]:
    """The object the spec built for this model, or None."""
    return model.__dict__.get("_megakernel")


def megakernel_agree(value: Any, *, what: str, group: Any = None) -> None:
    """Raise on every rank unless every rank of group (default: the TP group) holds
    the same value; ranks that differ would wait in different collectives."""
    values = _gather_tp(value, group=group)
    if any(v != value for v in values):
        raise RuntimeError(
            f"{FLAG}: the ranks' {what} differ: "
            + "; ".join(f"rank {rank}: {v}" for rank, v in enumerate(values))
        )


def megakernel_draft_sampler(draft_model: torch.nn.Module) -> Optional[Callable]:
    """The built draft spec's draft_sampler factory, or None for SGLang's proposal."""
    built = built_megakernel(draft_model)
    return None if built is None else built.draft_sampler


def megakernel_entry(model: torch.nn.Module, name: str) -> Optional[Callable]:
    """The offered entry, or None for SGLang's own path."""
    assert name in ENTRIES, name
    built = built_megakernel(model)
    return None if built is None else _entries(built)[name]


def megakernel_graph_variants(
    model: torch.nn.Module, runner: str, *, batch_sizes: Collection[int]
) -> Optional[AttentionGraphVariants]:
    """Return the variants captured for a runner's megakernel batch sizes."""
    assert runner in GRAPH_VARIANT_RUNNERS, runner
    variants = _offered_graph_variants(built_megakernel(model)).get(runner)
    if variants is None or not batch_sizes:
        return None
    logger.info(
        "NVIDIA megakernel: the %s graphs of batch sizes %s capture the attention "
        "variants %s",
        runner,
        sorted(batch_sizes),
        list(variants.capture_labels),
    )
    return variants


def megakernel_verify_variants(
    model: torch.nn.Module, *, capture_bs: Collection[int], tokens_per_request: int
) -> tuple[Optional[AttentionGraphVariants], Optional[frozenset[int]]]:
    """Return target-verify variants and the captured batch sizes that use them."""
    built = built_megakernel(model)
    if built is None:
        return None, None
    sizes = frozenset(
        bs for bs in capture_bs if bs * tokens_per_request in built.widths
    )
    variants = megakernel_graph_variants(model, "target_verify", batch_sizes=sizes)
    return (None, None) if variants is None else (variants, sizes)


def _entries(built: BuiltMegakernel) -> Mapping[str, Optional[Callable]]:
    return {
        "draft_sampler": built.draft_sampler,
        "draft_loop": built.draft_loop,
        "draft_extend": built.draft_extend,
        "begin_draft_round": built.begin_draft_round,
    }


def _offered_entries(built: BuiltMegakernel) -> tuple[str, ...]:
    return tuple(name for name, entry in _entries(built).items() if entry is not None)


def _offered_graph_variants(
    built: Optional[BuiltMegakernel],
) -> Mapping[str, AttentionGraphVariants]:
    return {} if built is None or built.graph_variants is None else built.graph_variants


def _binding_class(architectures: Optional[list[str]]) -> type:
    """Return the binding for the first registered model architecture."""
    for arch in architectures or ():
        if arch in _BINDINGS:
            return _BINDINGS[arch]
    return MegakernelBinding


def _find_spec_on_every_rank(
    *, hf_config: PretrainedConfig, is_draft: bool
) -> Optional[MegakernelSpec]:
    """The spec every rank found; for a draft model, None when no rank has one
    registered; else raise on every rank."""
    archs = hf_config.architectures
    spec, why = find_spec(architectures=archs, isa=device_isa())
    # Ranks branch together; one skipping the collectives below would hang its peers.
    lookups = _gather_tp(
        (
            spec_registered(archs),
            why,
            None if spec is None else (spec.name, spec.api_version),
        )
    )
    if not any(is_registered for is_registered, _, _ in lookups):
        if is_draft:
            # A draft model may have no registered spec.
            logger.info("NVIDIA megakernel: %s; %s keeps its own forward", why, archs)
            return None
        # resolution found the target's spec, so the plugin failed to load here
        _refuse(
            [
                f"{why} in the model workers, though argument resolution found it; "
                "see 'Failed to load plugin' or 'Failed to execute general plugin' "
                "above"
            ]
        )
    _refuse(
        [
            f"rank {rank}: {why}"
            for rank, (_, why, identity) in enumerate(lookups)
            if identity is None
        ]
    )
    if len({identity for *_, identity in lookups}) > 1:
        raise ValueError(
            f"{FLAG}: the ranks load different specs: "
            + "; ".join(f"rank {r}: {ident}" for r, (*_, ident) in enumerate(lookups))
        )
    return spec


def _requirement_reasons(
    *, spec: MegakernelSpec, hf_config: PretrainedConfig
) -> list[str]:
    """The spec's requirements this runner can check without the server arguments."""
    req, tp_size = spec.requires, get_parallel().tp_size
    reasons = []
    # an empty set accepts any TP size, as at argument resolution
    if req.tp_sizes and tp_size not in req.tp_sizes:
        reasons.append(f"tp_size {tp_size} is not one of {sorted(req.tp_sizes)}")
    graph = get_exec().graph.cuda_graph_config
    backend = Backend.DISABLED if graph is None else graph.decode.backend
    if req.decode_graph_backends and backend not in req.decode_graph_backends:
        reasons.append(
            f"decode CUDA graphs {backend!r} are not one of "
            f"{sorted(req.decode_graph_backends)}"
        )
    exactly = [
        (f"config.{key}", getattr(hf_config, key, None), want)
        for key, want in req.hf_config.items()
    ] + [(name, getattr(envs, name).get(), want) for name, want in req.env.items()]
    reasons += [
        f"{what} {v!r} is not {want!r}" for what, v, want in exactly if v != want
    ]
    return reasons


def _model_disable_reasons(
    *, spec: MegakernelSpec, model: torch.nn.Module, hf_config: PretrainedConfig
) -> list[str]:
    """This rank's reasons the loaded model cannot run the spec."""
    reasons = []
    if spec.model_disable_reasons is not None:
        reasons += spec.model_disable_reasons(model, hf_config)
    if "forward" in model.__dict__:
        reasons.append("the model's forward is already replaced on this instance")
    sms = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    if sms < spec.requires.resident_blocks:
        reasons.append(
            f"{Feature.PARTITIONED_GPU.value}: {sms} SMs, the persistent grids need "
            f"{spec.requires.resident_blocks}"
        )
    return reasons


class _Dispatch:
    """The installed forward: covered steps run the built megakernel."""

    def __init__(self, *, built: BuiltMegakernel, original: Callable, binding: Any):
        # MegakernelBinding, or the model-specific binding in _BINDINGS.
        self.built, self.original, self.binding = built, original, binding
        # batch size -> the step planned while capturing its decode graphs; plan reads
        # only the mode and rows, so every graph variant of a batch size plans alike
        self.captured: dict[int, Optional[Step]] = {}

    def wrapped(self) -> Callable:
        # Graph runners and supports_pp inspect the original forward's signature.
        @functools.wraps(self.original)
        def forward(input_ids, positions, forward_batch, *args, **kwargs):
            step = self.built.plan(input_ids, forward_batch, *args, **kwargs)
            if torch.cuda.is_current_stream_capturing():
                self.captured[int(forward_batch.batch_size)] = step
            if step is None:
                return self.original(
                    input_ids, positions, forward_batch, *args, **kwargs
                )
            return self.built.forward(
                msgspec.structs.replace(step, view=self.binding.forward_view()),
                self.original,
                input_ids,
                positions,
                forward_batch,
                *args,
                **kwargs,
            )

        return forward


class _CaptureCheck:
    """Once every decode graph is captured, log the ones that run the megakernel and
    refuse a runner none of whose graphs do."""

    def __init__(self, *, spec: MegakernelSpec, dispatch: _Dispatch):
        self.spec, self.dispatch = spec, dispatch
        self.is_done = False

    def __call__(
        self,
        runner: DecodeCudaGraphRunner,
        out: Any,
        forward_batch: ForwardBatch,
        num_tokens: int,
    ) -> None:
        captured = self.dispatch.captured
        if self.is_done or not set(runner.capture_bs) <= set(captured):
            return
        self.is_done = True
        self.dispatch.binding.assert_live()
        routed = sorted(bs for bs, step in captured.items() if step is not None)
        if not routed:
            raise ValueError(
                f"{FLAG}: none of the {len(captured)} captured decode graphs "
                f"(batch sizes {sorted(captured)}) runs {self.spec.name}"
            )
        logger.info(
            "NVIDIA megakernel: %s runs the decode graphs of batch sizes %s (%d of "
            "%d); every other step runs the standard forward",
            self.spec.name,
            routed,
            len(routed),
            len(captured),
        )


def _tokens_per_request() -> int:
    """Tokens per request of this server's decode-phase steps."""
    spec_config = get_spec()
    if spec_config.speculative_algorithm is None:
        return 1
    return int(spec_config.speculative_num_draft_tokens or 1)


def _decode_graphs_enabled() -> bool:
    graph = get_exec().graph.cuda_graph_config
    return graph is not None and graph.decode.backend != Backend.DISABLED


def _gather_tp(obj: Any, group: Any = None) -> list[Any]:
    """Gather a value from every rank of group (default: TP) through its CPU group."""
    group = get_parallel().tp_group if group is None else group
    if group.world_size == 1:
        return [obj]
    out: list[Any] = [None] * group.world_size
    dist.all_gather_object(out, obj, group=group.cpu_group)
    return out


def _gather_reasons(reasons: list[str]) -> list[str]:
    return [
        f"rank {rank}: {reason}"
        for rank, rank_reasons in enumerate(_gather_tp(reasons))
        for reason in rank_reasons
    ]


def _refuse(reasons: list[str]) -> None:
    if reasons:
        raise ValueError(
            f"{FLAG} cannot run this setup:\n  - " + "\n  - ".join(reasons)
        )
