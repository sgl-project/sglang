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
import logging
import time
from collections.abc import Mapping
from importlib.metadata import entry_points
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Collection, Optional

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
    Step,
    find_spec,
    registered_architectures,
    spec_registered,
)
from sglang.srt.arg_groups.model_override_base import model_config_of, resolving_view
from sglang.srt.environ import envs
from sglang.srt.kernel_plugins.amd.binding import msa_serves
from sglang.srt.kernel_plugins.nvidia.eagle_binding import EagleBinding
from sglang.srt.mem_cache.layout.paged_view import paged_row_view
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
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridLinearAttnBackend,
    )
    from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend
    from sglang.srt.layers.attention.mamba.mamba2_metadata import (
        ForwardMetadata as KdaForwardMetadata,
    )
    from sglang.srt.layers.attention.trtllm_mla_backend import (
        TRTLLMMLABackend,
        TRTLLMMLADecodeMetadata,
    )
    from sglang.srt.mem_cache.memory_pool import (
        HybridLinearKVPool,
        HybridReqToTokenPool,
        KVCache,
        MambaPool,
    )
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
    "KdaStepView.track_decode_state",  # SGLang's KDA prefix-state tracking
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


class KdaStateView(msgspec.Struct, frozen=True):
    """One KDA layer's SGLang-owned, non-ReplaySSM recurrent state."""

    conv: tuple[torch.Tensor, ...]
    temporal: torch.Tensor


class MlaPoolView:
    """The bounded MLA-cache surface prepared after SGLang allocates its pools."""

    def __init__(
        self,
        *,
        reserved_capacity: int,
        serving_capacity: Callable[[], int],
        post_capture_active: bool,
        page_size: int,
        logical_dtype: torch.dtype,
        storage_dtype: torch.dtype,
        layer_mapping: Mapping[int, int],
        caches: Mapping[int, torch.Tensor],
    ):
        self.reserved_capacity = int(reserved_capacity)
        self._serving_capacity = serving_capacity
        self.post_capture_active = bool(post_capture_active)
        self.page_size = int(page_size)
        self.logical_dtype = logical_dtype
        self.storage_dtype = storage_dtype
        self.layer_mapping = MappingProxyType(dict(layer_mapping))
        self._caches = MappingProxyType(dict(caches))

    @property
    def serving_capacity(self) -> int:
        """The live scheduler capacity (it may shrink after graph capture)."""
        return int(self._serving_capacity())

    def cache(self, layer_id: int) -> torch.Tensor:
        return self._caches[int(layer_id)]


class KdaPoolView:
    """The bounded KDA-state surface prepared after SGLang allocates its pools."""

    def __init__(
        self,
        *,
        capacity: int,
        layer_mapping: Mapping[int, int],
        states: Mapping[int, KdaStateView],
    ):
        self.capacity = int(capacity)
        self.layer_mapping = MappingProxyType(dict(layer_mapping))
        self._states = MappingProxyType(dict(states))

    def state(self, layer_id: int) -> KdaStateView:
        return self._states[int(layer_id)]


class HybridPoolView(msgspec.Struct, frozen=True):
    mla: MlaPoolView
    kda: KdaPoolView


class PreparationView(msgspec.Struct, frozen=True):
    """Runtime resources offered to an optional pre-capture preparation hook."""

    hybrid: Optional[HybridPoolView] = None


class MlaStepView(msgspec.Struct, frozen=True):
    block_table: torch.Tensor
    sequence_lengths: torch.Tensor
    write_slots: torch.Tensor


class KdaStepView:
    """One step's KDA slots and SGLang-owned state-tracking operation."""

    def __init__(
        self,
        *,
        state_indices: torch.Tensor,
        track_decode_state: Callable[[int], None],
    ):
        self.state_indices = state_indices
        self._track_decode_state = track_decode_state

    def track_decode_state(self, layer_id: int) -> None:
        """Track one completed layer; SGLang coalesces this at the last layer."""
        self._track_decode_state(int(layer_id))


class HybridStepView(msgspec.Struct, frozen=True):
    mla: MlaStepView
    kda: KdaStepView


class ForwardView:
    """The running forward's attention data, read from its attention backend; built
    at capture for a graph step, so its tensors are the ones the graph replays."""

    def __init__(
        self,
        backend: Any,
        seal_cache: Optional[Callable] = None,
        *,
        hybrid: Optional[HybridStepView] = None,
        legacy: bool = True,
    ):
        self.hybrid = hybrid
        if legacy:
            # Keep API-v1's nonhybrid behavior without handing the plugin the
            # backend. Hybrid consumers get only the typed surface above.
            self._trtllm_attn = lambda: backend.trtllm_attn
            self._attention = lambda: backend.forward
            self._store_cache = lambda: backend.store_cache
            self._swa_out_cache_loc = lambda: backend.get_swa_out_cache_loc
            self._low_ratio_sources = lambda: backend.forward_low_ratio_sources
            self._core_compressor = lambda: backend.forward_core_compressor
            self._use_fused_topk = lambda: bool(getattr(backend, "use_fused_topk", False))
            self.metadata = backend.forward_metadata
            self.kv_pool = KvPoolView(backend.token_to_kv_pool, seal_cache)

    @property
    def use_fused_topk(self) -> bool:
        return self._use_fused_topk()

    @property
    def trtllm_attn(self) -> bool:
        return self._trtllm_attn()

    @property
    def attention(self) -> Callable[..., Any]:
        return self._attention()

    @property
    def store_cache(self) -> Callable[..., None]:
        return self._store_cache()

    @property
    def swa_out_cache_loc(self) -> Callable[[Any], torch.Tensor]:
        return self._swa_out_cache_loc()

    @property
    def low_ratio_sources(self) -> Callable[..., Any]:
        return self._low_ratio_sources()

    @property
    def core_compressor(self) -> Callable[..., Any]:
        return self._core_compressor()


class _SealedSource(msgspec.Struct, frozen=True):
    identity: tuple
    current: Callable[[], Any]


class _HybridSources(msgspec.Struct, frozen=True):
    model_runner: ModelRunner
    backend: HybridLinearAttnBackend
    full_backend: TRTLLMMLABackend
    linear_backend: KDAAttnBackend
    token_pool: HybridLinearKVPool
    full_pool: KVCache
    request_pool: HybridReqToTokenPool
    mamba_pool: MambaPool


class MegakernelBinding:
    """Built at install, before the KV pools exist; read at build or forward. Each
    source is sealed at the first read that finds it; ``assert_live`` (after the
    decode graphs are captured) raises once one was reallocated."""

    def __init__(self, *, model: torch.nn.Module):
        self._model = model
        self._cache_sources: dict[int, tuple] = {}
        self._sealed: dict[str, _SealedSource] = {}
        self._preparation: Optional[PreparationView] = None
        self._hybrid_sources: Optional[_HybridSources] = None

    def preparation_view(self, model_runner: ModelRunner) -> PreparationView:
        """The post-pool, pre-capture resources for an optional build hook."""
        self._seal("model_runner", model_runner)
        if self._preparation is not None:
            self.assert_live()
            return self._preparation

        hybrid = self._prepare_hybrid(model_runner)
        self._preparation = PreparationView(hybrid=hybrid)
        return self._preparation

    def forward_view(self, forward_batch: Optional[ForwardBatch] = None) -> ForwardView:
        """The view of the forward running now (a target step or a draft step)."""
        from sglang.srt.model_executor.forward_context import get_attn_backend

        backend = get_attn_backend()
        hybrid_family = self._hybrid_sources is not None
        if hybrid_family:
            if backend is not self._hybrid_sources.backend:
                raise RuntimeError("megakernel binding: running hybrid backend changed")
        else:
            self._seal("kv_pool", backend.token_to_kv_pool, self._current_pool)
        hybrid = (
            self._hybrid_step(backend, forward_batch)
            if forward_batch is not None and hybrid_family
            else None
        )
        view = ForwardView(
            backend, self._seal_cache, hybrid=hybrid, legacy=not hybrid_family
        )
        return view

    def seal_capture_view(self, forward_batch: ForwardBatch, view: ForwardView) -> None:
        """Seal graph-static step tensors for this mode and batch-size bucket."""
        if view.hybrid is None:
            return
        self.assert_live()
        mode = str(forward_batch.forward_mode)
        key = f"{mode}:{int(forward_batch.batch_size)}"
        mla, kda = view.hybrid.mla, view.hybrid.kda
        sources = self._current_hybrid_sources()
        mla_metadata = self._mla_metadata(sources)
        kda_metadata = sources.linear_backend.forward_metadata
        # Keep each graph's metadata owners when the backend selects another
        # batch-size bucket, while still detecting replacement of their fields.
        tensors = (
            (
                "mla.block_table",
                mla.block_table,
                lambda: mla_metadata.block_kv_indices,
            ),
            (
                "mla.sequence_lengths",
                mla.sequence_lengths,
                lambda: mla_metadata.seq_lens_k,
            ),
            ("mla.write_slots", mla.write_slots, lambda: forward_batch.out_cache_loc),
            (
                "kda.state_indices",
                kda.state_indices,
                lambda: kda_metadata.mamba_cache_indices,
            ),
        )
        for name, tensor, getter in tensors:
            if _identity(getter()) != _identity(tensor):
                raise RuntimeError(
                    f"megakernel binding: capture[{key}].{name} changed before seal"
                )
            self._seal(f"capture[{key}].{name}", tensor, getter)

    def engram_history(self) -> Optional[torch.Tensor]:
        """The model's Engram hash history ``[slots + 1, width]`` int32, or None."""
        history = self._history()
        if history is None or history.shape[1] == 0:
            return None
        self._seal("engram_history", history, self._history)
        return history

    def engram_host_table(self, engram: torch.nn.Module) -> Any:
        """The host table of one of the model's Engram layers, or None."""
        return engram.embed.host_table

    def assert_live(self) -> None:
        for name, sealed in self._sealed.items():
            if _identity(sealed.current()) != sealed.identity:
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

    def _seal(
        self, name: str, value: Any, current: Optional[Callable[[], Any]] = None
    ) -> None:
        identity = _identity(value)
        sealed = self._sealed.get(name)
        if sealed is not None:
            if sealed.identity != identity:
                raise RuntimeError(
                    f"megakernel binding: {name} was reallocated after its first read"
                )
            return
        self._sealed[name] = _SealedSource(
            identity=identity, current=current or (lambda value=value: value)
        )

    def _prepare_hybrid(self, model_runner: ModelRunner) -> Optional[HybridPoolView]:
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            HybridLinearAttnBackend,
        )

        backend = model_runner.attn_backend
        if not isinstance(backend, HybridLinearAttnBackend):
            return None

        from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend
        from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend
        from sglang.srt.mem_cache.memory_pool import (
            HybridLinearKVPool,
            HybridReqToTokenPool,
        )

        full_backend = backend.full_attn_backend
        linear_backend = backend.linear_attn_backend
        token_pool = backend.token_to_kv_pool
        request_pool = backend.req_to_token_pool
        # This view describes the MLA/KDA layout, not arbitrary hybrid backends.
        if (
            not isinstance(full_backend, TRTLLMMLABackend)
            or not isinstance(linear_backend, KDAAttnBackend)
            or not isinstance(token_pool, HybridLinearKVPool)
            or not isinstance(request_pool, HybridReqToTokenPool)
            or not token_pool.use_mla
        ):
            return None
        full_pool = token_pool.full_kv_pool
        mamba_pool = request_pool.mamba_pool
        full_mapping = token_pool.full_attention_layer_id_mapping
        mamba_mapping = request_pool.mamba_map
        for role, child in (("full", full_backend), ("linear", linear_backend)):
            if child.token_to_kv_pool is not token_pool:
                raise ValueError(f"megakernel binding: {role} backend KV pool differs")
            if child.req_to_token_pool is not request_pool:
                raise ValueError(
                    f"megakernel binding: {role} backend request pool differs"
                )
        if tuple(backend.attn_backend_list) != (full_backend, linear_backend):
            raise ValueError("megakernel binding: hybrid backend children differ")

        sources = _HybridSources(
            model_runner=model_runner,
            backend=backend,
            full_backend=full_backend,
            linear_backend=linear_backend,
            token_pool=token_pool,
            full_pool=full_pool,
            request_pool=request_pool,
            mamba_pool=mamba_pool,
        )
        self._hybrid_sources = sources
        self._seal_hybrid_sources(sources)

        full_map = {int(layer): int(index) for layer, index in full_mapping.items()}
        mamba_map = {int(layer): int(index) for layer, index in mamba_mapping.items()}
        page_size = int(full_pool.page_size)
        caches: dict[int, torch.Tensor] = {}
        reserved_capacity: Optional[int] = None
        for layer_id in full_map:
            cache = _mla_storage_cache(token_pool, layer_id)
            if cache.shape[0] % page_size:
                raise ValueError(
                    f"megakernel binding: MLA layer {layer_id} rows "
                    f"{cache.shape[0]} are not page aligned"
                )
            cache = paged_row_view(cache, page_size)
            capacity = (int(cache.shape[0]) - 1) * page_size
            if reserved_capacity is None:
                reserved_capacity = capacity
            elif capacity != reserved_capacity:
                raise ValueError("megakernel binding: MLA layer capacities differ")
            caches[layer_id] = cache
            self._seal(
                f"mla_cache[{layer_id}]",
                cache,
                lambda layer_id=layer_id: _paged_mla_cache(
                    self._current_hybrid_sources().token_pool, layer_id
                ),
            )

        states: dict[int, KdaStateView] = {}
        for layer_id in mamba_map:
            state = request_pool.mamba2_layer_cache(layer_id)
            if state.replayssm_d is not None:
                raise ValueError(
                    f"megakernel binding: KDA layer {layer_id} has ReplaySSM state; "
                    "the hybrid state view requires --enable-linear-replayssm "
                    "and --enable-linear-replayssm-spec to be disabled"
                )
            conv = tuple(state.conv)
            if len(conv) != 1:
                raise ValueError(
                    f"megakernel binding: KDA layer {layer_id} has {len(conv)} "
                    "convolution states, expected one"
                )
            states[layer_id] = KdaStateView(conv=conv, temporal=state.temporal)
            self._seal(
                f"kda_conv[{layer_id}]",
                conv[0],
                lambda layer_id=layer_id: (
                    self._current_hybrid_sources()
                    .request_pool.mamba2_layer_cache(layer_id)
                    .conv[0]
                ),
            )
            self._seal(
                f"kda_temporal[{layer_id}]",
                state.temporal,
                lambda layer_id=layer_id: (
                    self._current_hybrid_sources()
                    .request_pool.mamba2_layer_cache(layer_id)
                    .temporal
                ),
            )

        if reserved_capacity is None or not states:
            raise ValueError("megakernel binding: hybrid MLA/KDA mappings are empty")
        return HybridPoolView(
            mla=MlaPoolView(
                reserved_capacity=reserved_capacity,
                serving_capacity=lambda: self._current_hybrid_sources().token_pool.size,
                post_capture_active=bool(token_pool.post_capture_active),
                page_size=page_size,
                logical_dtype=full_pool.dtype,
                storage_dtype=full_pool.store_dtype,
                layer_mapping=full_map,
                caches=caches,
            ),
            kda=KdaPoolView(
                capacity=int(mamba_pool.size),
                layer_mapping=mamba_map,
                states=states,
            ),
        )

    def _seal_hybrid_sources(self, sources: _HybridSources) -> None:
        runner = sources.model_runner
        getters = {
            # Ordered root-to-leaf so a replaced minimal parent reports its own
            # named invariant before a deeper path could raise AttributeError.
            "hybrid_backend": lambda: runner.attn_backend,
            "full_backend": lambda: runner.attn_backend.full_attn_backend,
            "linear_backend": lambda: runner.attn_backend.linear_attn_backend,
            "hybrid_kv_pool": lambda: runner.attn_backend.token_to_kv_pool,
            "full_kv_pool": lambda: runner.attn_backend.token_to_kv_pool.full_kv_pool,
            "request_pool": lambda: runner.attn_backend.req_to_token_pool,
            "mamba_pool": lambda: runner.attn_backend.req_to_token_pool.mamba_pool,
            "attention_backend_list": lambda: runner.attn_backend.attn_backend_list,
            "full_backend_kv_pool": lambda: (
                runner.attn_backend.full_attn_backend.token_to_kv_pool
            ),
            "full_backend_request_pool": lambda: (
                runner.attn_backend.full_attn_backend.req_to_token_pool
            ),
            "linear_backend_kv_pool": lambda: (
                runner.attn_backend.linear_attn_backend.token_to_kv_pool
            ),
            "linear_backend_request_pool": lambda: (
                runner.attn_backend.linear_attn_backend.req_to_token_pool
            ),
            "full_layer_mapping": lambda: (
                runner.attn_backend.token_to_kv_pool.full_attention_layer_id_mapping
            ),
            "mamba_layer_mapping": lambda: (
                runner.attn_backend.req_to_token_pool.mamba_map
            ),
        }
        for name, getter in getters.items():
            self._seal(name, getter(), getter)

    def _current_hybrid_sources(self) -> _HybridSources:
        sources = self._hybrid_sources
        if sources is None:
            raise RuntimeError("megakernel binding: hybrid resources were not prepared")
        backend = sources.model_runner.attn_backend
        # Read through the live root objects so assert_live sees replacements.
        return _HybridSources(
            model_runner=sources.model_runner,
            backend=backend,
            full_backend=backend.full_attn_backend,
            linear_backend=backend.linear_attn_backend,
            token_pool=backend.token_to_kv_pool,
            full_pool=backend.token_to_kv_pool.full_kv_pool,
            request_pool=backend.req_to_token_pool,
            mamba_pool=backend.req_to_token_pool.mamba_pool,
        )

    def _hybrid_step(
        self, backend: AttentionBackend, forward_batch: ForwardBatch
    ) -> HybridStepView:
        sources = self._current_hybrid_sources()
        if backend is not sources.backend:
            raise RuntimeError("megakernel binding: running hybrid backend changed")
        mla_metadata = self._mla_metadata(sources)
        kda_metadata: Optional[KdaForwardMetadata] = (
            sources.linear_backend.forward_metadata
        )
        if kda_metadata is None:
            raise RuntimeError("megakernel binding: KDA decode metadata is unavailable")
        block_table = mla_metadata.block_kv_indices
        sequence_lengths = mla_metadata.seq_lens_k
        write_slots = forward_batch.out_cache_loc
        state_indices = kda_metadata.mamba_cache_indices
        if any(
            value is None
            for value in (block_table, sequence_lengths, write_slots, state_indices)
        ):
            raise RuntimeError(
                "megakernel binding: hybrid decode metadata is incomplete"
            )
        if not forward_batch.out_cache_loc_is_physical:
            raise RuntimeError(
                "megakernel binding: hybrid MLA write slots are not physical"
            )

        assert self._preparation is not None
        pool_view = self._preparation.hybrid
        assert pool_view is not None

        def track_decode_state(layer_id: int) -> None:
            state = pool_view.kda.state(layer_id)
            sources.linear_backend._track_mamba_state_decode(
                forward_batch,
                state.conv[0],
                state.temporal,
                state_indices,
                layer_id,
            )

        return HybridStepView(
            mla=MlaStepView(
                block_table=block_table,
                sequence_lengths=sequence_lengths,
                write_slots=write_slots,
            ),
            kda=KdaStepView(
                state_indices=state_indices, track_decode_state=track_decode_state
            ),
        )

    def _mla_metadata(self, sources: _HybridSources) -> TRTLLMMLADecodeMetadata:
        metadata = sources.full_backend.forward_decode_metadata
        if metadata is None:
            raise RuntimeError("megakernel binding: MLA decode metadata is unavailable")
        return metadata

    def _current_pool(self) -> Any:
        from sglang.srt.model_executor.forward_context import get_attn_backend

        return get_attn_backend().token_to_kv_pool

    def _history(self) -> Optional[torch.Tensor]:
        # only models with Engram layers have a hasher
        hasher = getattr(self._model, "engram_hasher", None)
        return None if hasher is None else hasher.history


def _identity(value: Any) -> tuple:
    if isinstance(value, torch.Tensor):
        return (
            value.data_ptr(),
            tuple(value.shape),
            tuple(value.stride()),
            value.storage_offset(),
            value.dtype,
            value.device,
        )
    if isinstance(value, Mapping):
        return (id(value), tuple(sorted(value.items())))
    if isinstance(value, (list, tuple)):
        return (id(value), tuple(id(item) for item in value))
    return (id(value),)


def _paged_mla_cache(pool: HybridLinearKVPool, layer_id: int) -> torch.Tensor:
    cache = _mla_storage_cache(pool, layer_id)
    page_size = int(pool.full_kv_pool.page_size)
    return paged_row_view(cache, page_size)


def _mla_storage_cache(pool: HybridLinearKVPool, layer_id: int) -> torch.Tensor:
    """Return one MLA layer as a zero-copy view of its physical storage dtype.

    Read through the pool's public getter first so its layer mapping and pending
    transfer synchronization run before reinterpreting the equal-width logical
    view as physical storage.
    """
    cache = pool.get_key_buffer(layer_id)
    full_pool = pool.full_kv_pool
    if cache.dtype != full_pool.dtype:
        raise ValueError(
            "megakernel binding: MLA cache logical dtype "
            f"{cache.dtype} differs from pool dtype {full_pool.dtype}"
        )
    storage_dtype = full_pool.store_dtype
    if cache.element_size() != storage_dtype.itemsize:
        raise ValueError(
            "megakernel binding: MLA logical and storage item sizes differ"
        )
    return cache if cache.dtype == storage_dtype else cache.view(storage_dtype)


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
    binding: MegakernelBinding
    prepare: Optional[Callable[[BuiltMegakernel, PreparationView], None]]


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
    # Keep plugin resources outside the model's module/state tree.
    model.__dict__["_megakernel"] = _InstalledMegakernel(
        built=built, binding=binding, prepare=spec.prepare
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
    dispatch = _Dispatch(built=built, original=model.forward, binding=binding)
    model.forward = dispatch.wrapped()
    if _decode_graphs_enabled():
        capture_tail_hooks.append(_CaptureCheck(spec=spec, dispatch=dispatch))


def built_megakernel(model: torch.nn.Module) -> Optional[BuiltMegakernel]:
    """The object the spec built for this model, or None."""
    installed: Optional[_InstalledMegakernel] = model.__dict__.get("_megakernel")
    return None if installed is None else installed.built


def prepare_megakernel(model_runner: ModelRunner) -> None:
    """Prepare installed plugin resources after model preparation, before capture."""
    installed: Optional[_InstalledMegakernel] = model_runner.model.__dict__.get(
        "_megakernel"
    )
    if installed is not None and installed.prepare is not None:
        installed.prepare(
            installed.built, installed.binding.preparation_view(model_runner)
        )


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
            capturing = torch.cuda.is_current_stream_capturing()
            if capturing:
                self.captured[int(forward_batch.batch_size)] = step
            if step is None:
                return self.original(
                    input_ids, positions, forward_batch, *args, **kwargs
                )
            view = self.binding.forward_view(forward_batch)
            if capturing:
                self.binding.seal_capture_view(forward_batch, view)
            return self.built.forward(
                msgspec.structs.replace(step, view=view),
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
