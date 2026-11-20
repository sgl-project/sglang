# SPDX-License-Identifier: Apache-2.0
"""SGLang's binding to the NVIDIA megakernel plugins behind --enable-nvidia-megakernel.

It checks a plugin's spec against the resolved configuration at startup, builds it
once the model is prepared, and installs the whole-step dispatch on the model. It is
also the one place a megakernel reads SGLang's runtime state: the KV pool, the
attention metadata of the running forward and the Engram tables. A spec's build gets
a ``MegakernelBinding`` as ``BuildContext.binding``; a covered forward gets the
running forward's ``ForwardView`` as ``Step.view`` (a draft entry asks
``binding.forward_view()``). Everything else a megakernel touches is the model it
replaces, SGLang's kernel ops, and the reads listed below.
"""

from __future__ import annotations

import functools
import importlib
import logging
import time
from dataclasses import dataclass
from importlib.metadata import entry_points
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Collection, Mapping, Optional

import msgspec
import torch
import torch.distributed as dist

from sglang.kernels.plugin_kernels.nvidia.spec import (
    ENTRIES,
    ENTRY_POINT,
    FLAG,
    GRAPH_VARIANT_RUNNERS,
    BuildContext,
    BuiltMegakernel,
    Feature,
    MegakernelSpec,
    PostVerifyStrategy,
    SpeculativeRuntimeHooks,
    Step,
    SupportsDecodeTensorEntry,
    SupportsDraftAccept,
    SupportsDraftAux,
    SupportsDraftEmbed,
    SupportsDraftSelectorHead,
    SupportsKdaCommit,
    SupportsPrefillBody,
    SupportsRank0Broadcast,
    SupportsSampledAccept,
    SupportsStateSlotReset,
    find_spec,
    registered_architectures,
    spec_registered,
)
from sglang.srt.arg_groups.model_override_base import model_config_of, resolving_view
from sglang.srt.environ import envs
from sglang.srt.kernel_plugins.amd.binding import msa_serves
from sglang.srt.kernel_plugins.nvidia.eagle_binding import EagleBinding
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.plugins import GENERAL_PLUGINS_GROUP
from sglang.srt.runtime_context import (
    get_context,
    get_device,
    get_disagg,
    get_exec,
    get_memory,
    get_model,
    get_parallel,
    get_schedule,
    get_spec,
)

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from sglang.srt.layers.attention.graph_variants import AttentionGraphVariants
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner
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
    "KdaStepView.track_decode_state",  # host snapshots after the fused forward
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


_VIEW_DTYPES = {
    0: torch.uint8,
    1: torch.float8_e4m3fn,
    2: torch.bfloat16,
    3: torch.float16,
}


@dataclass(frozen=True)
class MlaPoolView:
    reserved_capacity: int
    serving_capacity: int
    post_capture_active: bool
    page_size: int
    logical_dtype_code: int
    storage_dtype_code: int
    layer_mapping: dict[int, int]
    caches: dict[int, torch.Tensor]

    @property
    def logical_dtype(self) -> torch.dtype:
        return _VIEW_DTYPES[self.logical_dtype_code]

    @property
    def storage_dtype(self) -> torch.dtype:
        return _VIEW_DTYPES[self.storage_dtype_code]

    def cache(self, layer_id: int) -> torch.Tensor:
        return self.caches[layer_id]


@dataclass(frozen=True)
class KdaStateView:
    conv: tuple[torch.Tensor, ...]
    temporal: torch.Tensor


@dataclass(frozen=True)
class KdaPoolView:
    capacity: int
    layer_mapping: dict[int, int]
    states: dict[int, KdaStateView]

    def state(self, layer_id: int) -> KdaStateView:
        return self.states[layer_id]


@dataclass(frozen=True)
class IndexerPoolView:
    page_size: int
    caches: dict[int, torch.Tensor]


@dataclass(frozen=True)
class HybridPoolView:
    mla: MlaPoolView
    kda: KdaPoolView
    indexer: Optional[IndexerPoolView] = None


@dataclass(frozen=True)
class PreparationView:
    hybrid: Optional[HybridPoolView] = None


@dataclass(frozen=True)
class MlaStepView:
    block_table: torch.Tensor
    sequence_lengths: torch.Tensor
    write_slots: torch.Tensor


@dataclass(frozen=True)
class KdaStepView:
    state_indices: torch.Tensor

    def track_decode_state(self, layer_id: int) -> None:
        # SGLang snapshots the completed states once after the fused forward.
        pass


@dataclass(frozen=True)
class HybridStepView:
    mla: MlaStepView
    kda: KdaStepView


@dataclass(frozen=True)
class HybridForwardView:
    hybrid: HybridStepView


class KvPoolView:
    """The running forward's KV pool, under the pool's own names: its layout and
    per-layer key buffers (a pool offers the ones of its kind)."""

    def __init__(self, pool: Any, seal_cache: Optional[Callable] = None):
        self.pool = pool
        self._seal_cache = seal_cache

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        value = self.pool.get_key_buffer(layer_id)
        if self._seal_cache is not None:
            self._seal_cache(layer_id, value)
        return value

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


class ForwardView:
    """The running forward's attention data, read from its attention backend; built
    at capture for a graph step, so its tensors are the ones the graph replays."""

    def __init__(self, backend: Any, seal_cache: Optional[Callable] = None):
        self._backend = backend
        self.kv_pool = KvPoolView(backend.token_to_kv_pool, seal_cache)

    @property
    def metadata(self) -> Any:
        # Read on use: a composite backend keeps it in its children.
        return self._backend.forward_metadata

    @property
    def use_fused_topk(self) -> bool:
        return bool(getattr(self._backend, "use_fused_topk", False))

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


class MegakernelBinding:
    """Built at install, before the KV pools exist; read at build or forward. Each
    source is sealed at the first read that finds it; ``assert_live`` (after the
    decode graphs are captured) raises once one was reallocated."""

    def __init__(self, *, model: torch.nn.Module):
        self.model = model
        self._sealed: dict[str, tuple] = {}
        self._cache_sources: dict[int, tuple] = {}
        # (pool, req_pool) and their tensor identities
        self._hybrid_sources: Optional[tuple] = None
        self._hybrid_identity: Optional[tuple] = None

    def forward_view(
        self, forward_batch: Any = None
    ) -> ForwardView | HybridForwardView:
        """The view of the forward running now (a target step or a draft step)."""
        # imported here: hybrid_linear_attn_backend imports model_runner, which
        # imports this module
        from sglang.srt.layers.attention.dsa_backend import DSAMetadata
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            HybridLinearAttnBackend,
        )

        backend = get_attn_backend()
        self._seal("kv_pool", backend.token_to_kv_pool)
        if forward_batch is not None and isinstance(backend, HybridLinearAttnBackend):
            full = backend.full_attn_backend.forward_metadata
            linear = backend.linear_attn_backend.forward_metadata
            if isinstance(full, DSAMetadata):
                block_table = full.real_page_table
                lengths = full.cache_seqlens_int32
            else:
                block_table = full.block_kv_indices
                lengths = full.seq_lens_k
            return HybridForwardView(
                hybrid=HybridStepView(
                    mla=MlaStepView(
                        block_table=block_table,
                        sequence_lengths=lengths,
                        write_slots=forward_batch.out_cache_loc,
                    ),
                    kda=KdaStepView(state_indices=linear.mamba_cache_indices),
                )
            )
        return ForwardView(backend, self._seal_cache)

    def prefill_view(self) -> ForwardView:
        """The view for an eager prefill step, without graph-static decode metadata."""
        return self.forward_view(forward_batch=None)

    def preparation_view(self, model_runner: ModelRunner) -> PreparationView:
        return self.pool_view(
            model_runner.token_to_kv_pool, model_runner.req_to_token_pool
        )

    def pool_view(self, pool: Any, req_pool: Any) -> PreparationView:
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        if not (isinstance(pool, HybridLinearKVPool) and pool.use_mla):
            return PreparationView()
        full = pool.full_kv_pool
        mapping = dict(pool.full_attention_layer_id_mapping)
        mamba_mapping = dict(req_pool.mamba_map)
        caches = {
            layer: full.kv_buffer[index].view(-1, pool.page_size, full.kv_cache_dim)
            for layer, index in mapping.items()
        }
        states = {}
        for layer in mamba_mapping:
            state = req_pool.mamba_pool.mamba2_layer_cache(req_pool.mamba_map[layer])
            states[layer] = KdaStateView(
                conv=tuple(state.conv), temporal=state.temporal
            )
        indexer = None
        if pool.use_dsa:
            indexer = IndexerPoolView(
                page_size=full.index_page_size,
                caches={
                    layer: full.get_index_k_with_scale_buffer(index)
                    for layer, index in mapping.items()
                },
            )
        hybrid = HybridPoolView(
            mla=MlaPoolView(
                reserved_capacity=full.size,
                serving_capacity=full.size,
                post_capture_active=full.post_capture_active,
                page_size=pool.page_size,
                logical_dtype_code=next(
                    code for code, dtype in _VIEW_DTYPES.items() if dtype == full.dtype
                ),
                storage_dtype_code=next(
                    code
                    for code, dtype in _VIEW_DTYPES.items()
                    if dtype == full.store_dtype
                ),
                layer_mapping=mapping,
                caches=caches,
            ),
            kda=KdaPoolView(
                capacity=req_pool.mamba_pool.size,
                layer_mapping=mamba_mapping,
                states=states,
            ),
            indexer=indexer,
        )
        self._hybrid_sources = pool, req_pool
        self._hybrid_identity = self._pool_identity(pool, req_pool)
        return PreparationView(hybrid=hybrid)

    @staticmethod
    def _pool_identity(pool: Any, req_pool: Any) -> tuple:
        full = pool.full_kv_pool
        tensors = list(full.kv_buffer)
        if pool.use_dsa:
            tensors.extend(full.index_k_with_scale_buffer)
        for layer in req_pool.mamba_map:
            state = req_pool.mamba_pool.mamba2_layer_cache(req_pool.mamba_map[layer])
            tensors.extend((*state.conv, state.temporal))
        return tuple(_identity(tensor) for tensor in tensors)

    def finish_step(self, forward_batch: Any) -> None:
        # Prefix-cache snapshot; runs once every layer's state is final.
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            HybridLinearAttnBackend,
        )

        backend = get_attn_backend()
        if not isinstance(backend, HybridLinearAttnBackend):
            return
        linear = backend.linear_attn_backend
        req_pool = linear.req_to_token_pool
        last_layer = max(req_pool.mamba_map)
        state = linear.state_layer_cache(last_layer)
        linear._track_mamba_state_decode(
            forward_batch,
            state.conv[0],
            state.temporal,
            linear.forward_metadata.mamba_cache_indices,
            layer_id=last_layer,
        )

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
        if (
            self._hybrid_sources is not None
            and self._pool_identity(*self._hybrid_sources) != self._hybrid_identity
        ):
            raise RuntimeError("megakernel hybrid buffers changed after preparation")
        current = {"kv_pool": self._current_pool, "engram_history": self._history}
        for name, sealed in self._sealed.items():
            if _identity(current[name]()) != sealed:
                raise RuntimeError(
                    f"megakernel binding: {name} was reallocated after its first read"
                )

        for layer_id, identity in self._cache_sources.items():
            value = self._current_pool().get_key_buffer(layer_id)
            if _identity(value) != identity:
                raise RuntimeError(
                    f"megakernel binding: kv_cache[{layer_id}] was reallocated "
                    "after its first read"
                )

    def _seal_cache(self, layer_id: int, value: torch.Tensor) -> None:
        identity = _identity(value)
        if self._cache_sources.setdefault(layer_id, identity) != identity:
            raise RuntimeError(
                f"megakernel binding: kv_cache[{layer_id}] was reallocated "
                "after its first read"
            )

    def _seal(self, name: str, value: Any) -> None:
        self._sealed.setdefault(name, _identity(value))

    def _current_pool(self) -> Any:
        return get_attn_backend().token_to_kv_pool

    def _history(self) -> Optional[torch.Tensor]:
        # only models with Engram layers have a hasher
        hasher = getattr(self.model, "engram_hasher", None)
        return None if hasher is None else hasher.history


def _identity(value: Any) -> tuple:
    if isinstance(value, torch.Tensor):
        return (value.data_ptr(), tuple(value.shape), value.dtype, value.stride())
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
    Feature.DP_ATTENTION: lambda c: (c.attn_dp_size or 1) > 1,
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


def handle_nvidia_megakernel(server_args: Any) -> None:
    """Resolution step: fail before the weights load when the flag cannot be served,
    listing every reason."""
    cfg = resolving_view(server_args)
    if not cfg.enable_nvidia_megakernel:
        return
    archs = model_config_of(server_args).hf_config.architectures
    if msa_serves(archs) and not cfg.enable_amd_monokernel:
        return  # handle_monokernel checks the MSA plugin's spec
    reasons = plugin_reasons()
    if not reasons:
        model_config = model_config_of(server_args)
        spec, reasons = megakernel_disable_reasons(
            cfg=cfg, hf_config=model_config.hf_config, isa=device_isa()
        )
        reasons += _dtype_reasons(spec, model_config)
    if cfg.enable_amd_monokernel:
        # both replace the model's forward, which would fail only after weights load
        reasons.append("--enable-amd-monokernel is also set; use one of the two")
    if reasons:
        raise ValueError(
            f"{FLAG} cannot run this setup:\n  - " + "\n  - ".join(reasons)
        )
    logger.warning("%s is experimental. Its kernels and coverage may change.", FLAG)
    logger.info("%s: %s on %s", FLAG, spec.name, device_isa())


class _InstalledMegakernel(msgspec.Struct, frozen=True):
    built: BuiltMegakernel
    binding: Any
    prepare: Optional[Callable[[BuiltMegakernel, PreparationView], None]]
    # the TENSOR_ENTRY_KINDS adapter the build runs through, else None
    tensor_entry: Any = None


def megakernel_enabled() -> bool:
    """Whether this server runs with --enable-nvidia-megakernel."""
    # the model loader also runs without a server's config (tools, unit tests)
    if not get_context().is_config_namespace_published("exec"):
        return False
    return get_exec().kernel.enable_nvidia_megakernel


def _kimi_build_widths(
    max_rows: int, tokens: int, architectures: Any, *, is_draft: bool
) -> tuple[int, ...]:
    """Omit unreachable q6 widths without changing eager or graph dispatch.

    Preparation precedes KV sizing. For ordinary TP8, the resolved request
    limit bounds the eventual pool, and unaligned graph buckets are clamped
    to that pool. Keep every intervening width for eager execution and buckets
    added during sizing. Unknown bounds or other layouts retain full preparation.
    """
    widths = tuple(range(tokens, max_rows + 1, tokens))
    if is_draft or tuple(architectures or ()) != ("KimiK3ForConditionalGeneration",):
        return widths
    from sglang.srt.layers.dp_attention import get_cuda_graph_batch_size_alignment

    config, parallel, disagg = get_spec(), get_parallel(), get_disagg()
    graph = get_exec().graph.cuda_graph_config
    request_limit = get_schedule().max_running_requests
    if (
        tokens == 6
        and config.speculative_algorithm == "DSPARK"
        and not config.speculative_adaptive
        and type(request_limit) is int
        and request_limit > 0
        and parallel.tp_size == 8
        and parallel.pp_size
        == parallel.dp_size
        == parallel.attn_cp_size
        == parallel.dcp_size
        == 1
        and disagg.disaggregation_mode == "null"
        and not disagg.enable_pd_role_switch
        and graph is not None
        and graph.decode.backend == Backend.FULL
        and get_cuda_graph_batch_size_alignment() == 1
    ):
        widths = tuple(range(tokens, min(max_rows, request_limit * tokens) + 1, tokens))
    # Every target rank participates, including ranks retaining full preparation.
    megakernel_agree(widths, what="Kimi preparation widths")
    return widths


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
    tokens = _tokens_per_request()
    tensor_verify = tokens > 1 and spec.tensor_verify_build is not None
    reasons = _requirement_reasons(spec=spec, hf_config=hf_config)
    reasons += _layout_reasons()
    if not tensor_verify:
        reasons += _model_disable_reasons(spec=spec, model=model, hf_config=hf_config)
    _refuse(_gather_reasons(reasons))

    binding = _binding_class(hf_config.architectures, kind=spec.binding)(model=model)
    context = BuildContext(
        widths=_kimi_build_widths(
            spec.max_rows, tokens, hf_config.architectures, is_draft=is_draft
        ),
        tokens_per_request=tokens,
        deadline_s=deadline_s(get_device().watchdog_timeout),
        binding=None if tokens > 1 and spec.tensor_verify_binding_kind else binding,
    )
    start, free_before = time.perf_counter(), torch.cuda.mem_get_info()[0]
    tensor_entry = None
    if tensor_verify:
        tensor_entry = _tensor_entry(spec.tensor_verify_binding_kind, binding)
        weights = tensor_entry.weights_view(
            int(get_model().context_length), query_width=tokens
        )
        built = spec.tensor_verify_build(weights, float(context.deadline_s))
    else:
        built = spec.build(model, hf_config, context)
    if tensor_entry is None and _decode_entry_kind(built):
        tensor_entry = _tensor_entry(built.tensor_binding_kind, binding)
    # Keep plugin resources outside the model's module/state tree.
    model.__dict__["_megakernel"] = _InstalledMegakernel(
        built=built, binding=binding, prepare=spec.prepare, tensor_entry=tensor_entry
    )
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
    dispatch = _Dispatch(
        built=built,
        original=model.forward,
        binding=binding,
        tensor_entry=tensor_entry,
    )
    model.forward = dispatch.wrapped()
    if spec.prefill_graph:
        _install_prefill_body(model, spec=spec, dispatch=dispatch)
    if _decode_graphs_enabled():
        capture_tail_hooks.append(_CaptureCheck(spec=spec, dispatch=dispatch))


def _installed(model: Optional[torch.nn.Module]) -> Optional[_InstalledMegakernel]:
    return None if model is None else model.__dict__.get("_megakernel")


def built_megakernel(model: Optional[torch.nn.Module]) -> Optional[BuiltMegakernel]:
    """The object the spec built for this model, or None."""
    installed = _installed(model)
    return None if installed is None else installed.built


def prepare_megakernel(model_runner: ModelRunner) -> None:
    """Prepare installed plugin resources after model preparation, before capture."""
    installed = _installed(model_runner.model)
    if installed is not None and installed.prepare is not None:
        entry = installed.tensor_entry
        view = (
            installed.binding.preparation_view(model_runner)
            if entry is None
            else entry.pool_view(
                model_runner.token_to_kv_pool, model_runner.req_to_token_pool
            )
        )
        installed.prepare(installed.built, view)
    own_state = megakernel_state(model_runner.model)
    if own_state is not None:
        model_runner.token_to_kv_pool_allocator.state.own_state = own_state


def megakernel_state(model: torch.nn.Module) -> Optional[SupportsStateSlotReset]:
    """The installed build when it keeps recurrent state of its own, else None."""
    built = built_megakernel(model)
    return built if isinstance(built, SupportsStateSlotReset) else None


def bind_speculative_runtime_hooks(
    model: torch.nn.Module,
    defaults: SpeculativeRuntimeHooks,
) -> SpeculativeRuntimeHooks:
    """Resolve target-side speculative strategies once at the binding boundary.

    The caller's stock strategies are returned by identity when the runtime master
    switch is off or the built provider has no factory. Provider presence must agree
    across ranks before any rank can call it. The provider may change the local key
    builder, but the runtime-owned verification runner cannot be replaced.
    """
    if not envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get():
        return defaults

    built = built_megakernel(model)
    factory = None if built is None else built.speculative_runtime_hooks
    megakernel_agree(
        factory is not None,
        what="speculative runtime hooks factory presence",
    )
    if factory is None:
        return defaults

    hooks = factory(defaults)
    if not isinstance(hooks, SpeculativeRuntimeHooks):
        raise TypeError(
            "speculative runtime hooks factory must return SpeculativeRuntimeHooks"
        )
    if not isinstance(hooks.post_verify, PostVerifyStrategy):
        raise TypeError("post_verify must be a PostVerifyStrategy")
    if hooks.post_verify.run is not defaults.post_verify.run:
        raise ValueError("a provider cannot replace the runtime post-verify runner")
    if not callable(hooks.post_verify.local_keys):
        raise TypeError("post_verify.local_keys must be callable")
    return hooks


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


def _offered(model: torch.nn.Module, protocol: type) -> Any:
    if not megakernel_enabled():
        return None
    built = built_megakernel(model)
    return built if isinstance(built, protocol) else None


def _offered_draft(
    draft_model: torch.nn.Module, protocol: type, sizes: str, *, what: str
) -> Any:
    if not megakernel_enabled():
        return None
    built = built_megakernel(draft_model)
    offered = built if isinstance(built, protocol) and getattr(built, sizes) else None
    megakernel_agree(() if offered is None else getattr(offered, sizes), what=what)
    return offered


def megakernel_draft_aux(draft_model: torch.nn.Module) -> Optional[SupportsDraftAux]:
    """Draft spec that writes the DFlash context KV, else None; call on all TP ranks."""
    return _offered_draft(
        draft_model, SupportsDraftAux, "draft_aux_rows", what="draft aux rows"
    )


def megakernel_draft_selector_head(
    draft_model: torch.nn.Module,
) -> Optional[SupportsDraftSelectorHead]:
    """Draft spec whose forward proposes the block, else None; call on all TP ranks."""
    return _offered_draft(
        draft_model,
        SupportsDraftSelectorHead,
        "draft_head_requests",
        what="draft head requests",
    )


def megakernel_draft_accept(
    draft_model: torch.nn.Module,
) -> Optional[SupportsDraftAccept]:
    """Draft spec running the greedy verify accept, else None; call on all TP ranks."""
    return _offered_draft(
        draft_model,
        SupportsDraftAccept,
        "draft_accept_requests",
        what="draft accept requests",
    )


def megakernel_draft_embed(
    draft_model: torch.nn.Module,
) -> Optional[SupportsDraftEmbed]:
    """Draft spec that embeds the draft block, else None; call on all TP ranks."""
    return _offered_draft(
        draft_model,
        SupportsDraftEmbed,
        "draft_embed_requests",
        what="draft embed requests",
    )


def megakernel_kda_commit(target_model: torch.nn.Module) -> Optional[SupportsKdaCommit]:
    """Target spec that commits the KDA state after a verify (rank-local), else None."""
    return _offered(target_model, SupportsKdaCommit)


def megakernel_rank0_broadcast(
    target_model: torch.nn.Module,
) -> Optional[SupportsRank0Broadcast]:
    """Target spec broadcasting rank 0's sampled-accept decision on device, else None."""
    return _offered(target_model, SupportsRank0Broadcast)


def megakernel_sampled_accept(
    target_model: torch.nn.Module,
) -> Optional[SupportsSampledAccept]:
    """Target spec that runs the DFLASH sampled accept on vocab shards, else None."""
    return _offered(target_model, SupportsSampledAccept)


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
        "draft_step_fold": getattr(built, "draft_step_fold", None),  # optional
    }


def _offered_entries(built: BuiltMegakernel) -> tuple[str, ...]:
    return tuple(name for name, entry in _entries(built).items() if entry is not None)


def _offered_graph_variants(
    built: Optional[BuiltMegakernel],
) -> Mapping[str, AttentionGraphVariants]:
    return {} if built is None or built.graph_variants is None else built.graph_variants


# MegakernelSpec.binding -> the binding SGLang builds for that spec.
BINDING_KINDS: Mapping[str, type] = MappingProxyType(
    {"default": MegakernelBinding, "eagle_hybrid": EagleBinding}
)

# Tensor-entry kind (a spec's tensor_verify_binding_kind or a build's
# tensor_binding_kind) -> "module:class" of its host adapter, imported lazily.
TENSOR_ENTRY_KINDS: Mapping[int, str] = MappingProxyType(
    {
        31: "sglang.srt.kernel_plugins.nvidia.kimi_k31_verify_dsa_binding"
        ":K31SparseVerifyHostBinding",
        32: "sglang.srt.kernel_plugins.nvidia.kimi_k31_decode_binding"
        ":K31DecodeHostBinding",
    }
)


def _decode_entry_kind(built: BuiltMegakernel) -> int:
    return (
        built.tensor_binding_kind if isinstance(built, SupportsDecodeTensorEntry) else 0
    )


def _tensor_entry(kind: int, binding: MegakernelBinding) -> Any:
    if kind not in TENSOR_ENTRY_KINDS:
        raise ValueError(f"unknown megakernel tensor binding {kind}")
    module, name = TENSOR_ENTRY_KINDS[kind].split(":")
    return getattr(importlib.import_module(module), name)(binding)


def _binding_class(architectures: Optional[list[str]], kind: str = "") -> type:
    """Return the binding the spec names, else that of the first known architecture."""
    if kind:
        return BINDING_KINDS[kind]
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
    if spec.binding and spec.binding not in BINDING_KINDS:
        reasons.append(
            f"binding {spec.binding!r} is not one of {sorted(BINDING_KINDS)}"
        )
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


def _layout_reasons() -> list[str]:
    """Pool layouts no megakernel can address, whichever steps or entries it
    offers: a target step reads the same pools as the draft entries."""
    if (
        get_memory().enable_unified_memory
        and get_exec().mamba.enable_linear_replayssm_spec
    ):
        # A megakernel addresses the ReplaySSM speculative ring by request slot and
        # its cursors with unit stride; unified memory keeps the ring in units of its
        # request sub-pool and the cursors strided inside each unit.
        return [
            "the ReplaySSM speculative ring of --enable-linear-replayssm-spec is "
            "not addressable under --enable-unified-memory"
        ]
    return []


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

    def __init__(
        self,
        *,
        built: BuiltMegakernel,
        original: Callable,
        binding: Any,
        tensor_entry: Any = None,
    ):
        # MegakernelBinding, or the model-specific binding in _BINDINGS.
        self.built, self.original, self.binding = built, original, binding
        self.tensor_entry = tensor_entry
        # batch size -> the step planned while capturing its decode graphs; plan reads
        # only the mode and rows, so every graph variant of a batch size plans alike
        self.captured: dict[int, Optional[Step]] = {}

    def wrapped(self) -> Callable:
        # Graph runners and supports_pp inspect the original forward's signature.
        @functools.wraps(self.original)
        def forward(input_ids, positions, forward_batch, *args, **kwargs):
            entry = self.tensor_entry
            if entry is None:
                step = self.built.plan(input_ids, forward_batch, *args, **kwargs)
            else:
                step = entry.plan(self.built, input_ids, forward_batch, *args, **kwargs)
            if torch.cuda.is_current_stream_capturing():
                self.captured[int(forward_batch.batch_size)] = step
            if step is None:
                return self.original(
                    input_ids, positions, forward_batch, *args, **kwargs
                )
            if entry is not None:
                return entry.forward(
                    self.built,
                    step,
                    input_ids,
                    positions,
                    forward_batch,
                    *args,
                    **kwargs,
                )
            view = (
                self.binding.prefill_view()
                if step.mode == "prefill"
                else self.binding.forward_view(forward_batch)
            )
            out = self.built.forward(
                msgspec.structs.replace(step, view=view),
                self.original,
                input_ids,
                positions,
                forward_batch,
                *args,
                **kwargs,
            )
            if step.mode == "decode" and isinstance(self.binding, MegakernelBinding):
                self.binding.finish_step(forward_batch)
            return out

        return forward

    def wrapped_body(self, original: Callable) -> Callable:
        # The prefill graph runner inspects the layer module's forward signature.
        @functools.wraps(original)
        def forward(input_ids, positions, forward_batch, *args, **kwargs):
            step = self.built.plan_body(input_ids, forward_batch, *args, **kwargs)
            if step is None:
                return original(input_ids, positions, forward_batch, *args, **kwargs)
            return self.built.forward_body(
                msgspec.structs.replace(step, view=self.binding.prefill_view()),
                original,
                input_ids,
                positions,
                forward_batch,
                *args,
                **kwargs,
            )

        return forward


def _install_prefill_body(
    model: torch.nn.Module, *, spec: MegakernelSpec, dispatch: _Dispatch
) -> None:
    """Route the layer module's forward, as the prefill graph runner resolves it."""
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
        _resolve_transformer_layer_model,
        set_external_prefill_body,
    )

    if not isinstance(dispatch.built, SupportsPrefillBody):
        _refuse([f"{spec.name} sets prefill_graph without SupportsPrefillBody"])
    layer_model = _resolve_transformer_layer_model(model)
    layer_model.forward = dispatch.wrapped_body(layer_model.forward)
    set_external_prefill_body(model, dispatch.built)
    logger.info("NVIDIA megakernel: %s plans the prefill body", spec.name)


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


def megakernel_tensor_graph_variants(
    model: torch.nn.Module, existing: Optional[AttentionGraphVariants]
) -> Optional[AttentionGraphVariants]:
    installed = _installed(model)
    entry = None if installed is None else installed.tensor_entry
    return (
        entry.graph_variants() if existing is None and entry is not None else existing
    )


def megakernel_can_run_graph(
    model: torch.nn.Module, forward_batch: ForwardBatch
) -> bool:
    installed = _installed(model)
    if installed is None or not _decode_entry_kind(installed.built):
        return True
    return installed.tensor_entry.can_run_graph(installed.built, forward_batch)


def configure_megakernel_aux(model: torch.nn.Module, layer_ids: Any) -> None:
    installed = _installed(model)
    if installed is not None and installed.tensor_entry is not None:
        installed.tensor_entry.configure_aux(installed.built, layer_ids)
