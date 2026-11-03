# SPDX-License-Identifier: Apache-2.0
"""The binding of an EAGLE megakernel whose target mixes GatedDeltaNet and full
attention (Qwen3.5) and of its draft: the data SGLang hands their build, forward and
entries, as binding.py does for the models it serves.

binding._BINDINGS maps these architectures here. SGLang runs every phase of an EAGLE
decode step (a topk-1 chain) and owns every runner, worker and CUDA graph; a
megakernel never gets one, nor a schedule batch.

- A covered target step gets the running forward's ``EagleForwardView`` as
  ``Step.view``: the full-attention and GDN metadata, read from the hybrid attention
  backend, and the KV and mamba pools. The step takes the replaced forward's
  arguments as they are.
- The EAGLE entries (spec.ENTRIES) get the structs below, built by SGLang from its
  own buffers. A draft model's ``draft_loop`` runs the whole draft decode loop of a
  served draft graph (model steps, token selection, any exchange between steps), and
  its ``draft_extend`` the model step and top-1 of a served draft-extend graph, both
  inside the graph's capture; ``begin_draft_round`` runs on the host before a served
  draft graph replays, every round, so it runs no host collective (a TP group may
  span hosts). Draft entries serve graph replays only: an eager draft step runs
  SGLang's loop.

What a megakernel may write (everything else it only reads):
- the target step: the KV at its ``out_cache_loc``, and the GDN intermediate rows and
  ReplaySSM ring entries of its requests (SGLang's commit after the verify advances
  the ring cursors). It returns full-vocabulary logits: SGLang samples, forces
  acceptance and computes logprobs from them;
- the draft entries: the draft KV at their ``out_cache_loc``, and their struct's
  ``*_out`` tensors.
"""

from __future__ import annotations

from typing import Any, Optional

import msgspec
import torch

from sglang.srt.mem_cache.kv.router import STATE_PART
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.runtime_context import get_parallel, get_schedule, get_spec

# Called by a megakernel beyond the data below (its contract test pins the list).
TOLERATED_CALLABLES = (
    # the TP group's process groups: a megakernel builds its exchange buffers over
    # them once, never per step
    "EagleBinding.tp_device_group",
    "EagleBinding.tp_cpu_group",
    "sglang.kernels.ops",  # SGLang's kernel ops
    "the replaced model's modules, weights and quant methods",
)
# Written into SGLang's objects beyond the pools and the entries' outputs.
TOLERATED_WRITES = ()
# Read, tolerated: configuration, the covered forward's own arguments, and the host
# state a graph variant's select (spec.BuiltMegakernel) picks on.
TOLERATED_READS = (
    "runtime_context getters (get_parallel, get_exec, get_spec, get_platform)",
    "capture-mode queries (get_is_capture_mode, is_in_breakable_cuda_graph, "
    "get_capture_attention_variant)",
    "ForwardBatch.batch_size",
    "ForwardBatch.forward_mode",
    "ForwardBatch.out_cache_loc",
    "ForwardBatch.req_pool_indices",
    "ForwardBatch.seq_lens",
    "ForwardBatch.mrope_positions",
    "ForwardBatch.seq_lens_cpu",
    "ForwardBatch.attention_kv_len_lower_bound",
)


class KvPoolBuffers(msgspec.Struct, frozen=True):
    layer_ids: tuple[int, ...]  # the runner's full-attention layers
    k: tuple[torch.Tensor, ...]  # per layer: the pool's key buffer, its layout
    v: tuple[torch.Tensor, ...]
    # per layer: the scale an FP8 cache entry is multiplied by, 1.0 otherwise
    k_scale: tuple[float, ...]
    v_scale: tuple[float, ...]


class MambaPoolBuffers(msgspec.Struct, frozen=True):
    """Three kinds of rows: conv and temporal by mamba slot (state_indices), the
    intermediate buffers by batch row (intermediate_indices), and the ReplaySSM ring
    and its cursors by request slot (req_pool_indices)."""

    layer_ids: tuple[int, ...]  # GDN layers; the leading index of each tensor below
    conv: tuple[torch.Tensor, ...]  # [layers, mamba slots + 1, ...] each
    temporal: torch.Tensor  # [layers, mamba slots + 1, HV, K, V] committed SSM state
    intermediate_conv_window: tuple[torch.Tensor, ...]
    intermediate_ssm: Optional[torch.Tensor]
    # ReplaySSM spec ring [layers, request slots + 1, HV, L, V] / [.., H, L, K] /
    # [.., HV, L]; the ring length L is ring_d.shape[-2]
    ring_d: Optional[torch.Tensor]
    ring_k: Optional[torch.Tensor]
    ring_g: Optional[torch.Tensor]
    ring_rawv: Optional[torch.Tensor]
    ring_rawk: Optional[torch.Tensor]
    ring_beta: Optional[torch.Tensor]
    # cursors [request slots + 1] of the verify ring, shared by every GDN layer; a
    # verify reads them, SGLang's commit after it advances them
    ring_write_pos: Optional[torch.Tensor]
    ring_cache_base: Optional[torch.Tensor]
    ring_is_flush: Optional[torch.Tensor]


class AttnStepMetadata(msgspec.Struct, frozen=True):
    page_table: torch.Tensor  # [bs, pages] int32, physical pages of page_size slots
    # [bs] int32: the KV length including this step's rows (for a verify, seq_lens
    # plus tokens_per_request)
    cache_seqlens: torch.Tensor
    cu_seqlens_q: Optional[torch.Tensor]  # [bs + 1] int32
    cu_seqlens_k: Optional[torch.Tensor]  # [bs + 1] int32
    max_seq_len_q: int


class GdnStepMetadata(msgspec.Struct, frozen=True):
    # [bs] mamba slot per request (conv and temporal rows), pad_slot_id if padded
    state_indices: torch.Tensor
    query_start_loc: torch.Tensor  # [bs + 1] int32
    # [bs] int32: each request's row of the intermediate conv window / SSM scratch
    intermediate_indices: torch.Tensor
    pad_slot_id: int
    track_indices: Optional[torch.Tensor]  # [bs] track slot per request


class EagleForwardView(msgspec.Struct, frozen=True):
    """The running forward's attention data; built at capture for a graph step, so
    its tensors are the ones the graph replays."""

    attn: AttnStepMetadata  # the full-attention layers'
    gdn: Optional[GdnStepMetadata]  # None for a model without GDN layers (the draft)
    kv: KvPoolBuffers
    mamba: Optional[MambaPoolBuffers]  # None for a model without GDN layers


class EagleDraftLoopIO(msgspec.Struct, frozen=True):
    num_requests: int
    num_steps: int
    topk_index: torch.Tensor  # [bs, 1] int64, the round's first draft token
    hidden_states: torch.Tensor  # [bs, H] the round's input hidden state
    # draft step i runs at positions + i (and mrope_positions + i); an entry that
    # advances them in place restores them before it returns
    positions: torch.Tensor  # [bs] int64
    mrope_positions: Optional[torch.Tensor]  # [3, bs] int64
    out_cache_loc: torch.Tensor  # [bs * num_steps] int64, request-major; step i: col i
    req_pool_indices: torch.Tensor  # [bs] int64
    seq_lens: torch.Tensor  # [bs]
    steps: tuple[AttnStepMetadata, ...]  # draft step i's attention metadata
    kv: KvPoolBuffers  # the draft's
    # [bs, num_steps] int64, SGLang-owned: column 0 is topk_index, column i + 1
    # draft step i's token, identical on every rank
    draft_tokens_out: torch.Tensor


class EagleDraftExtendIO(msgspec.Struct, frozen=True):
    num_requests: int
    tokens_per_request: int
    input_ids: torch.Tensor  # [bs * tokens] int64
    positions: torch.Tensor  # [bs * tokens] int64
    mrope_positions: Optional[torch.Tensor]  # [3, bs * tokens] int64
    hidden_states: torch.Tensor  # [bs * tokens, H] the verify's hidden states
    out_cache_loc: torch.Tensor  # [bs * tokens] int64
    req_pool_indices: torch.Tensor  # [bs] int64
    seq_lens: torch.Tensor  # [bs]
    select_index: torch.Tensor  # [bs] int64, the row whose outputs are kept
    num_accept_tokens: torch.Tensor  # [bs] int32
    attn: AttnStepMetadata
    kv: KvPoolBuffers  # the draft's
    # SGLang-owned outputs of the selected rows: top-1 as torch.argmax picks it,
    # 1.0, and the hidden state
    topk_index_out: torch.Tensor  # [bs, 1] int64
    topk_p_out: torch.Tensor  # [bs, 1] float32
    hidden_out: torch.Tensor  # [bs, H]


class EagleDraftRoundIO(msgspec.Struct, frozen=True):
    num_requests: int
    # whether this round's draft input came from a served draft-extend replay of
    # the same requests, with no draft extend since; never on a PD decode
    # instance's first round, whose input comes from the prefill instance
    from_served_extend: bool
    topk_index: torch.Tensor  # [bs, 1] int64
    hidden_states: torch.Tensor  # [bs, H]


class EagleBinding:
    """Built at install, before the pools exist; read at build or forward. The pools
    are sealed at their first read, and ``assert_live`` (after the decode graphs are
    captured) raises once one was reallocated."""

    def __init__(self, *, model: torch.nn.Module):
        self._model = model
        self._pools: dict[str, Any] = {}
        self._sealed: dict[str, tuple] = {}

    @property
    def tp_rank(self) -> int:
        return get_parallel().tp_rank

    @property
    def tp_size(self) -> int:
        return get_parallel().tp_size

    @property
    def tp_device_group(self) -> Any:
        return get_parallel().tp_group.device_group

    @property
    def tp_cpu_group(self) -> Any:
        return get_parallel().tp_group.cpu_group

    @property
    def tokens_per_request(self) -> int:
        """Verify rows per request."""
        return get_spec().speculative_num_draft_tokens

    @property
    def num_steps(self) -> int:
        """A draft round runs num_steps - 1 draft steps."""
        return get_spec().speculative_num_steps

    @property
    def page_size(self) -> int:
        return get_schedule().page_size

    def forward_view(self) -> EagleForwardView:
        """The view of the forward running now."""
        backend = get_attn_backend()
        full, linear = split_backend(backend)
        return EagleForwardView(
            attn=attn_step_metadata(full),
            gdn=None if linear is None else gdn_step_metadata(linear),
            kv=self._read("kv", self._kv, backend),
            mamba=None if linear is None else self._read("mamba", self._mamba, backend),
        )

    def assert_live(self) -> None:
        backend = get_attn_backend()
        readers = {"kv": self._kv, "mamba": self._mamba}
        for name, sealed in self._sealed.items():
            if _identity(readers[name](backend)) != sealed:
                raise RuntimeError(
                    f"megakernel binding: {name} was reallocated after its first read"
                )

    def _read(self, name: str, read: Any, backend: Any) -> Any:
        if name not in self._pools:
            self._pools[name] = read(backend)
            self._sealed[name] = _identity(self._pools[name])
        return self._pools[name]

    def _kv(self, backend: Any) -> KvPoolBuffers:
        return kv_pool_buffers(
            pool=backend.token_to_kv_pool,
            backend=split_backend(backend)[0],
            model=self._model,
        )

    def _mamba(self, backend: Any) -> MambaPoolBuffers:
        return mamba_pool_buffers(backend.token_to_kv_pool)


def split_backend(backend: Any) -> tuple[Any, Optional[Any]]:
    """(full-attention backend, GDN backend or None) of a model's attention backend."""
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridLinearAttnBackend,
    )

    if isinstance(backend, HybridLinearAttnBackend):
        return backend.full_attn_backend, backend.linear_attn_backend
    return backend, None


def attn_step_metadata(backend: Any) -> AttnStepMetadata:
    """The step metadata of a TRT-LLM MHA backend, or of a hybrid's full-attention
    one."""
    metadata = split_backend(backend)[0].forward_metadata
    return AttnStepMetadata(
        page_table=metadata.page_table,
        cache_seqlens=metadata.cache_seqlens_int32,
        cu_seqlens_q=metadata.cu_seqlens_q,
        cu_seqlens_k=metadata.cu_seqlens_k,
        max_seq_len_q=metadata.max_seq_len_q,
    )


def gdn_step_metadata(backend: Any) -> GdnStepMetadata:
    """A GDN backend's metadata of the current target verify step."""
    metadata = backend.forward_metadata
    return GdnStepMetadata(
        state_indices=metadata.mamba_cache_indices,
        query_start_loc=metadata.query_start_loc,
        # one static row per batch row (pipeline parallelism, which may keep them
        # per request, is refused)
        intermediate_indices=backend.verify_intermediate_state_indices,
        pad_slot_id=backend.pad_slot_id,
        track_indices=metadata.mamba_track_indices,
    )


def kv_pool_buffers(
    *, pool: Any, backend: Any, model: torch.nn.Module
) -> KvPoolBuffers:
    """A full-attention backend's KV pool, per layer."""
    from sglang.srt.layers.radix_attention import RadixAttention

    router = pool.layer_router
    layer_ids = router.layer_ids()
    routed_layers = tuple(router.read(i) for i in layer_ids)
    layers = {
        module.layer_id: module
        for module in model.modules()
        if isinstance(module, RadixAttention)
    }
    is_fp8 = backend.data_type == torch.float8_e4m3fn

    def scale(layer_id: int, name: str) -> float:
        # the scale TRT-LLM MHA applies to this layer's FP8 entries
        if not is_fp8:
            return 1.0
        return backend._get_scalar_scale(layers[layer_id], f"{name}_float", name)

    return KvPoolBuffers(
        layer_ids=layer_ids,
        k=tuple(
            layer.view.get_key_buffer(layer.view_layer_id) for layer in routed_layers
        ),
        v=tuple(
            layer.view.get_value_buffer(layer.view_layer_id) for layer in routed_layers
        ),
        k_scale=tuple(scale(i, "k_scale") for i in layer_ids),
        v_scale=tuple(scale(i, "v_scale") for i in layer_ids),
    )


def mamba_pool_buffers(pool: Any) -> MambaPoolBuffers:
    """The GDN state pool and its ReplaySSM ring, in storage layer order."""
    router = pool.layer_router
    mamba_pool = router.part_view(STATE_PART)
    layer_indices = router.part_indices(STATE_PART)
    state = mamba_pool.get_speculative_mamba2_params_all_layers()
    return MambaPoolBuffers(
        layer_ids=tuple(sorted(layer_indices, key=layer_indices.get)),
        conv=tuple(state.conv),
        temporal=state.temporal,
        intermediate_conv_window=tuple(state.intermediate_conv_window),
        intermediate_ssm=state.intermediate_ssm,
        ring_d=state.replayssm_d,
        ring_k=state.replayssm_k,
        ring_g=state.replayssm_g,
        ring_rawv=state.replayssm_rawv,
        ring_rawk=state.replayssm_rawk,
        ring_beta=state.replayssm_beta,
        ring_write_pos=mamba_pool.replayssm_spec_write_pos,
        ring_cache_base=mamba_pool.replayssm_cache_base,
        ring_is_flush=mamba_pool.replayssm_is_flush,
    )


def _identity(value: Any) -> Any:
    """What must not change after a pool's first read: tensors by address, shape and
    dtype, other values as they are."""
    if value is None:
        return None
    if isinstance(value, msgspec.Struct):
        return _identity(msgspec.structs.astuple(value))
    if isinstance(value, tuple):
        return tuple(map(_identity, value))
    if isinstance(value, torch.Tensor):
        return (value.data_ptr(), tuple(value.shape), value.dtype)
    return value
