from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, TypeAlias, Union

import torch
from flash_attn_interface import flash_attn_varlen_func
from flash_attn_interface import flash_attn_with_kvcache as mate_flash_attn_with_kvcache
from flash_attn_interface import get_scheduler_metadata

from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
    FlashAttentionMultiStepBackend,
    merge_state_v2_wrapper,
)
from sglang.srt.layers.attention.local_attention import LocalAttentionMetadata
from sglang.srt.layers.cp.base import CPAttentionBackendKind, get_cp_strategy
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.mem_cache.memory_pool import KVWriteLoc, window_write_locs
from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode
from sglang.srt.runtime_context import get_exec, get_server_args
from sglang.srt.utils import is_musa

_platform_available = is_musa()

if TYPE_CHECKING:
    from sglang.srt.layers.cp.base import BaseContextParallelMetadata
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.forward_mode import ForwardMode
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput


@dataclass
class MusaSchedulerMetadataBatch:
    forward_mode: ForwardMode
    spec_info: Optional[SpecInput]
    batch_size: Optional[int]
    attn_cp_metadata: Optional[BaseContextParallelMetadata] = None


AttentionLayout: TypeAlias = tuple[int, int, int, int]
MusaSchedulerMetadata: TypeAlias = torch.Tensor


@dataclass(frozen=True)
class MusaAttentionLayouts:
    full: Optional[AttentionLayout]
    sliding_window: Optional[AttentionLayout]
    sliding_window_size: Optional[tuple[int, int]]
    has_full_attention: bool
    has_local_attention: bool
    has_cross_attention: bool
    has_sliding_window_local_attention: bool


@dataclass(frozen=True)
class MusaCudaGraphMetadataKey:
    mode: str
    batch_size: int
    topk: int = 0
    speculative_step_id: int = 0
    num_draft_tokens: int = 0


@dataclass
class MusaLocalAttentionMetadata(LocalAttentionMetadata):
    scheduler_metadata: Optional[torch.Tensor] = None


@dataclass
class MusaFlashAttentionMetadata(FlashAttentionMetadata):
    scheduler_metadata: Optional[MusaSchedulerMetadata] = None
    swa_scheduler_metadata: Optional[torch.Tensor] = None
    encoder_scheduler_metadata: Optional[torch.Tensor] = None
    cp_scheduler_metadata_prev: Optional[torch.Tensor] = None
    cp_scheduler_metadata_next: Optional[torch.Tensor] = None
    cp_swa_scheduler_metadata_prev: Optional[torch.Tensor] = None
    cp_swa_scheduler_metadata_next: Optional[torch.Tensor] = None
    cp_local_scheduler_metadata_prev: Optional[torch.Tensor] = None
    cp_local_scheduler_metadata_next: Optional[torch.Tensor] = None
    cp_local_swa_scheduler_metadata_prev: Optional[torch.Tensor] = None
    cp_local_swa_scheduler_metadata_next: Optional[torch.Tensor] = None
    cp_cu_seqlens_k_new_prev: Optional[torch.Tensor] = None
    cp_cu_seqlens_k_new_next: Optional[torch.Tensor] = None


MusaMetadataOwner: TypeAlias = Union[FlashAttentionMetadata, LocalAttentionMetadata]


@dataclass(frozen=True)
class MusaSchedulerMetadataSlot:
    name: str
    owner: MusaMetadataOwner
    attribute: str
    tensor: torch.Tensor


SchedulerMetadataBatch: TypeAlias = Union["ForwardBatch", MusaSchedulerMetadataBatch]

# Canonical scheduler-metadata attributes. Graph capture/replay (`_append_graph_metadata_slots`)
# and `_clear_cp_scheduler_metadata` both consume these lists, so adding or
# clearing a slot cannot drift apart from the refresh path.
_GRAPH_METADATA_ATTRS: tuple[str, ...] = (
    "scheduler_metadata",
    "swa_scheduler_metadata",
    "encoder_scheduler_metadata",
)
_CP_METADATA_SIDES: tuple[str, ...] = ("prev", "next")
_CP_METADATA_PREFIXES: tuple[str, ...] = (
    "cp",
    "cp_swa",
    "cp_local",
    "cp_local_swa",
)
_CP_METADATA_CU_SEQLENS_K_NEW_ATTR = "cp_cu_seqlens_k_new"


def _cp_metadata_attrs(side: str) -> tuple[str, ...]:
    """All CP scheduler-metadata attributes for one context-parallel side."""
    metadata_attrs = tuple(
        f"{prefix}_scheduler_metadata_{side}" for prefix in _CP_METADATA_PREFIXES
    )
    return metadata_attrs + (f"{_CP_METADATA_CU_SEQLENS_K_NEW_ATTR}_{side}",)


def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    qv: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[Union[int, torch.Tensor]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    rotary_seqlens: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    attention_chunk: int = 0,
    softcap: float = 0.0,
    rotary_interleaved: bool = True,
    scheduler_metadata: Optional[Union[torch.Tensor, tuple[torch.Tensor, bool]]] = None,
    num_splits: int = 0,
    pack_gqa=None,
    sm_margin: int = 0,
    return_softmax_lse: bool = False,
    sinks=None,
    score_mod=None,
    aux_tensors=None,
    ver=3,
):
    """MUSA flash_attn_with_kvcache wrapper.

    MUSA FA3 callers pass scheduler metadata explicitly at each call site.
    """
    if ver != 3:
        raise ValueError("Only ver=3 is supported for MUSA FA3.")
    if score_mod is not None or aux_tensors is not None:
        raise NotImplementedError(
            "score_mod and aux_tensors are not supported by the MUSA FA3 backend."
        )

    if scheduler_metadata is None:
        raise RuntimeError(
            "MUSA MATE FA3 flash_attn_with_kvcache requires scheduler_metadata. "
            "init_forward_metadata must precompute it for every kvcache call."
        )

    return mate_flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k=k,
        v=v,
        qv=qv,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=cache_batch_idx,
        cache_leftpad=cache_leftpad,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k_new,
        max_seqlen_q=max_seqlen_q,
        rotary_seqlens=rotary_seqlens,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        attention_chunk=attention_chunk,
        softcap=softcap,
        rotary_interleaved=rotary_interleaved,
        scheduler_metadata=scheduler_metadata,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=sm_margin,
        return_softmax_lse=return_softmax_lse,
        sinks=sinks,
    )


class MusaFlashAttentionBackend(FlashAttentionBackend):
    @property
    def fa3_upgrades_to_fa4(self) -> bool:
        """MUSA FA3 always runs the MATE kernels; the FA4 upgrade is CUDA-only.

        The parent initializer derives this flag from the CUDA device
        capability and rejects configurations where the upgraded path is not
        available. Pinning it to False here keeps the shared FA3 backend free
        of MUSA-specific branches.
        """
        return False

    @fa3_upgrades_to_fa4.setter
    def fa3_upgrades_to_fa4(self, _value: bool) -> None:
        # The parent's CUDA-derived assignment does not apply on MUSA.
        return

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        speculative_step_id=0,
        topk=0,
        speculative_num_steps=0,
        fa_impl_ver=3,
    ):
        if fa_impl_ver != 3:
            raise ValueError("MUSA flash attention backend only supports FA3.")

        super().__init__(
            model_runner,
            skip_prefill=skip_prefill,
            speculative_step_id=speculative_step_id,
            topk=topk,
            speculative_num_steps=speculative_num_steps,
            fa_impl_ver=fa_impl_ver,
        )
        model_config = model_runner.model_config
        self.flash_attn_varlen_func = flash_attn_varlen_func
        self.flash_attn_with_kvcache = flash_attn_with_kvcache
        self._attention_layouts: MusaAttentionLayouts = (
            MusaAttentionLayouts(
                full=None,
                sliding_window=None,
                sliding_window_size=None,
                has_full_attention=False,
                has_local_attention=False,
                has_cross_attention=False,
                has_sliding_window_local_attention=False,
            )
            if self.use_mla
            else self._get_no_mla_layer_attention_layouts(model_runner)
        )
        # Absorbed MLA is called with `qv` (q_nope) and a single-head latent KV
        # cache, i.e. num_heads_kv=1 / headdim=RoPE dim / headdim_v=kv_lora_rank.
        #
        # MATE's public `flash_attn_interface.get_scheduler_metadata` shim does
        # not expose the `has_qv` input, so the metadata plan is built with
        # has_qv=False. On MP31 (S5000) the BF16 qv/non-qv tile configs for the
        # absorbed-MLA (headdim=rope, headdim_v=kv_lora_rank) shape coincide, so
        # the plan still matches the forward kernel; MP32 and FP8 absorbed MLA
        # pick a different tile config and would need MATE's native
        # `mate.get_scheduler_metadata(..., has_qv=True)` instead. Re-check this
        # when either target or KV-cache dtype changes.
        self._mla_attention_layout: Optional[AttentionLayout] = (
            (
                self.num_attention_heads,
                1,
                model_config.qk_rope_head_dim,
                model_config.kv_lora_rank,
            )
            if self.use_mla
            else None
        )
        self._captured_graph_metadata: dict[
            MusaCudaGraphMetadataKey, dict[str, torch.Tensor]
        ] = {}
        self.forward_metadata: Optional[MusaFlashAttentionMetadata] = None
        self.forward_metadata_spec_decode_expand: Optional[
            MusaFlashAttentionMetadata
        ] = None
        # The parent callback uses the FA3 signature (`num_heads`), while the
        # wrapper requires separate `num_heads_q` and `num_heads_kv`. Scheduler
        # metadata is built in the backend lifecycle hooks instead.
        self._get_scheduler_metadata = None

    def _init_flash_attention_functions(self, device_capability):
        """MUSA FA3 kernels come from MATE's ``flash_attn_interface`` wrapper.

        The parent initializer imports the CUDA ``sgl_fa3`` / sgl-kernel FA3
        functions and derives the FA4-upgrade decision from the CUDA device
        capability; neither applies on MUSA. MUSA always runs the MATE FA3
        kernels and never upgrades to FA4.
        """
        self._get_scheduler_metadata = None
        self._get_fa_runtime_policy = None
        self._fa_kernels = {3: (flash_attn_varlen_func, flash_attn_with_kvcache)}

    @staticmethod
    def _as_local_metadata(
        metadata: Optional[LocalAttentionMetadata],
    ) -> Optional[MusaLocalAttentionMetadata]:
        if metadata is None or isinstance(metadata, MusaLocalAttentionMetadata):
            return metadata
        converted = MusaLocalAttentionMetadata()
        converted.__dict__.update(metadata.__dict__)
        return converted

    @classmethod
    def _as_metadata(
        cls, metadata: Optional[FlashAttentionMetadata]
    ) -> Optional[MusaFlashAttentionMetadata]:
        if metadata is None or isinstance(metadata, MusaFlashAttentionMetadata):
            return metadata
        converted = MusaFlashAttentionMetadata()
        converted.__dict__.update(metadata.__dict__)
        converted.local_attn_metadata = cls._as_local_metadata(
            metadata.local_attn_metadata
        )
        converted.swa_spec_metadata = cls._as_metadata(metadata.swa_spec_metadata)
        return converted

    def _ensure_metadata_types(self) -> None:
        self.forward_metadata = self._as_metadata(self.forward_metadata)
        self.forward_metadata_spec_decode_expand = self._as_metadata(
            self.forward_metadata_spec_decode_expand
        )

    @staticmethod
    def _seq_lens_cpu_list(seq_lens_cpu) -> Optional[list]:
        if seq_lens_cpu is None:
            return None
        if isinstance(seq_lens_cpu, torch.Tensor):
            if seq_lens_cpu.device.type != "cpu" and get_is_capture_mode():
                return None
            return seq_lens_cpu.detach().cpu().tolist()
        return list(seq_lens_cpu)

    @staticmethod
    def _layer_attention_layout(module: RadixAttention) -> AttentionLayout:
        return (
            module.tp_q_head_num,
            module.tp_k_head_num,
            module.qk_head_dim,
            module.v_head_dim,
        )

    @classmethod
    def _get_no_mla_layer_attention_layouts(
        cls, model_runner: ModelRunner
    ) -> MusaAttentionLayouts:
        # Wrapper scheduler metadata is built from one attention layout per
        # call. A model may have different full-attention and SWA structures,
        # but each family must be internally uniform:
        # all full-attention layers share one layout, and all SWA layers share
        # one layout.
        #
        # The booleans returned here are model-structure flags, not per-batch
        # state. They let `_init_no_mla_scheduler_metadata` avoid building
        # metadata for layer families that cannot be reached by this model.
        # Cross-attention layers are tracked separately because they use encoder
        # KV lengths and do not participate in the self-attention layout check.
        full_layouts: dict[AttentionLayout, list[int]] = {}
        swa_layouts: dict[AttentionLayout, list[int]] = {}
        swa_window_layers: dict[int, list[int]] = {}
        has_full_attention_layer = False
        has_local_attention_layer = False
        has_cross_attention_layer = False
        has_swa_local_attention_layer = False
        for module in model_runner.model.modules():
            if not isinstance(module, RadixAttention):
                continue

            layer_id = getattr(module, "layer_id", -1)
            if getattr(module, "is_cross_attention", False):
                has_cross_attention_layer = True
                continue

            if (
                module.sliding_window_size is not None
                and module.sliding_window_size > -1
            ):
                swa_layouts.setdefault(cls._layer_attention_layout(module), []).append(
                    layer_id
                )
                swa_window_layers.setdefault(module.sliding_window_size, []).append(
                    layer_id
                )
                has_swa_local_attention_layer |= bool(
                    getattr(module, "use_irope", False)
                )
            else:
                full_layouts.setdefault(cls._layer_attention_layout(module), []).append(
                    layer_id
                )
                has_full_attention_layer = True

            has_local_attention_layer |= bool(getattr(module, "use_irope", False))

        hf_config = model_runner.model_config.hf_config
        model_desc = (
            f"model_type={getattr(hf_config, 'model_type', None)!r} "
            f"architectures={getattr(hf_config, 'architectures', None)!r}"
        )
        if len(full_layouts) > 1 or len(swa_layouts) > 1:
            full_summary = {
                layout: sorted(layers) for layout, layers in full_layouts.items()
            }
            swa_summary = {
                layout: sorted(layers) for layout, layers in swa_layouts.items()
            }
            raise ValueError(
                "MUSA FA3 no-MLA scheduler metadata assumes one uniform "
                "self-attention layout for all full-attention layers and one "
                "uniform self-attention layout for all SWA layers; full and "
                "SWA layouts may differ. This model is served by the CUDA FA3 "
                "backend but would need per-call scheduler metadata on MUSA. "
                f"{model_desc} full={full_summary} swa={swa_summary}."
            )

        if len(swa_window_layers) > 1:
            window_summary = {
                window: sorted(layers) for window, layers in swa_window_layers.items()
            }
            raise ValueError(
                "MUSA FA3 no-MLA scheduler metadata assumes one uniform SWA "
                f"window size. {model_desc} windows={window_summary}."
            )

        full_layout = next(iter(full_layouts), None)
        swa_layout = next(iter(swa_layouts), full_layout)
        full_layout = full_layout if full_layout is not None else swa_layout
        swa_window_size = (
            (next(iter(swa_window_layers)), 0) if len(swa_window_layers) == 1 else None
        )
        return MusaAttentionLayouts(
            full=full_layout,
            sliding_window=swa_layout,
            sliding_window_size=swa_window_size,
            has_full_attention=has_full_attention_layer,
            has_local_attention=has_local_attention_layer,
            has_cross_attention=has_cross_attention_layer,
            has_sliding_window_local_attention=has_swa_local_attention_layer,
        )

    def _needs_no_mla_kvcache_scheduler_metadata(
        self,
        forward_batch: SchedulerMetadataBatch,
        *,
        is_cross_attention: bool = False,
    ) -> bool:
        forward_mode = forward_batch.forward_mode
        extend_prefix_lens_cpu = getattr(forward_batch, "extend_prefix_lens_cpu", None)
        extend_with_prefix = extend_prefix_lens_cpu is not None and any(
            extend_prefix_lens_cpu
        )
        return is_cross_attention or (
            forward_mode.is_decode_or_idle()
            or forward_mode.is_target_verify()
            or forward_mode.is_draft_extend_v2()
            or (
                forward_mode.is_extend_or_draft_extend_or_mixed(
                    include_draft_extend_v2=True
                )
                and extend_with_prefix
            )
        )

    def _is_context_parallel_extend(
        self, forward_batch: SchedulerMetadataBatch
    ) -> bool:
        return (
            forward_batch.forward_mode.is_context_parallel_extend()
            and forward_batch.attn_cp_metadata is not None
            and self.attn_cp_size > 1
        )

    # Scheduler metadata contract for MUSA FA3.
    #
    # `flash_attn_with_kvcache` always requires scheduler metadata. The no-MLA
    # extend/decode call flow intentionally follows `FlashAttentionBackend`;
    # MUSA only adds the precomputed scheduler metadata argument.
    #
    # no-MLA base paths:
    # - EXTEND, DRAFT_EXTEND, TARGET_VERIFY: kvcache extend; uses
    #   `metadata.scheduler_metadata`.
    # - DECODE/IDLE: kvcache decode; uses `metadata.scheduler_metadata`.
    # - Cross attention: kvcache over encoder KV; uses
    #   `metadata.encoder_scheduler_metadata`, with `causal=False`.
    # - Local attention: kvcache over local block table; uses
    #   `metadata.local_attn_metadata.scheduler_metadata`.
    #
    # no-MLA variants:
    # - SWA layers use `metadata.swa_scheduler_metadata`, generated from the
    #   same cache/page/q lengths as the base path but with the uniform SWA
    #   layer window size. Runtime calls pass
    #   `window_size=(layer.sliding_window_size, 0)`.
    # - Speculative cascade has two kvcache calls. The first call uses base/SWA
    #   metadata for the prefix part. The second call uses
    #   `forward_metadata_spec_decode_expand`, and SWA layers must select its
    #   `swa_scheduler_metadata` instead of the full-attention metadata.
    # - Context-parallel extend has prev/next kvcache calls. Full-attention CP
    #   uses `cp_scheduler_metadata_prev/next`; SWA CP uses
    #   `cp_swa_scheduler_metadata_prev/next`; local CP uses
    #   `cp_local_scheduler_metadata_prev/next`.
    #
    # MLA paths use the scheduler-metadata wrapper with the absorbed layout
    # (RoPE head dim plus KV LoRA rank).
    # `get_scheduler_metadata` builds the FA3 scheduler plan consumed by
    # `flash_attn_with_kvcache` (MATE's contract); the plan is specialized on
    # the batch/q/k geometry, causal/window flags and num_splits, so every call
    # site below must pass the same values as the kvcache call it feeds.
    # Dense `flash_attn_varlen_func` calls intentionally do not consume
    # precomputed metadata (they schedule internally), which is why only the
    # kvcache entry points in this backend build it.
    def _build_scheduler_metadata(
        self,
        *,
        max_seqlen_k: int,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k_new: Optional[torch.Tensor],
        max_seqlen_q: int,
        causal: bool,
        window_size: tuple[int, int],
        page_size: int,
        attention_layout: Optional[AttentionLayout] = None,
        num_splits: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        if cache_seqlens is None or cu_seqlens_q is None:
            return None
        num_heads_q, num_heads_kv, head_dim, v_head_dim = (
            attention_layout or self._attention_layouts.full
        )
        return get_scheduler_metadata(
            batch_size=cu_seqlens_q.shape[-1] - 1,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max(max_seqlen_k, 1),
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            headdim=head_dim,
            headdim_v=v_head_dim,
            cache_seqlens=cache_seqlens,
            qkv_dtype=self.kv_cache_dtype,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k_new=cu_seqlens_k_new,
            page_size=page_size,
            causal=causal,
            window_size=window_size,
            has_softcap=self.has_softcap,
            num_splits=self.num_splits if num_splits is None else num_splits,
        )

    def _init_mla_scheduler_metadata(
        self, forward_batch: SchedulerMetadataBatch
    ) -> None:
        if self.forward_metadata is None:
            return
        assert self._mla_attention_layout is not None

        # The FA3 scheduler metadata is specialized on `causal`, and MATE's
        # contract requires it to match the `flash_attn_with_kvcache` call it is
        # used for. MLA kvcache calls pass the decoder-level causal flag (False
        # only for the speculative cascade calls), so the metadata uses the same
        # value here. Decode has q_len == 1, where causal and non-causal plans
        # are identical, but keeping the flags aligned removes the contract
        # mismatch for q_len > 1 prefill / target-verify calls.
        is_decode = forward_batch.forward_mode.is_decode_or_idle()
        use_cascade = self.topk > 1 and (
            (is_decode and forward_batch.spec_info is not None)
            or forward_batch.forward_mode.is_target_verify()
        )
        causal = not use_cascade

        self.forward_metadata.scheduler_metadata = self._build_scheduler_metadata(
            max_seqlen_k=self.forward_metadata.max_seq_len_k,
            cache_seqlens=self.forward_metadata.cache_seqlens_int32,
            cu_seqlens_q=self.forward_metadata.cu_seqlens_q,
            cu_seqlens_k_new=self.forward_metadata.cu_seqlens_k,
            max_seqlen_q=self.forward_metadata.max_seq_len_q,
            causal=causal,
            window_size=(-1, -1),
            page_size=self.page_size,
            attention_layout=self._mla_attention_layout,
            num_splits=self.num_splits,
        )
        metadata_expand = self.forward_metadata_spec_decode_expand
        if metadata_expand is not None:
            metadata_expand.scheduler_metadata = self._build_scheduler_metadata(
                max_seqlen_k=metadata_expand.max_seq_len_k,
                cache_seqlens=metadata_expand.cache_seqlens_int32,
                cu_seqlens_q=metadata_expand.cu_seqlens_q,
                cu_seqlens_k_new=metadata_expand.cu_seqlens_k,
                max_seqlen_q=metadata_expand.max_seq_len_q,
                causal=causal,
                window_size=(-1, -1),
                page_size=self.page_size,
                attention_layout=self._mla_attention_layout,
                num_splits=self.num_splits,
            )

    def _mla_scheduler_metadata_for_decode_slice(
        self,
        metadata: MusaFlashAttentionMetadata,
        cache_seqlens: torch.Tensor,
        batch_index: int,
    ) -> torch.Tensor:
        assert self._mla_attention_layout is not None
        cu_seqlens_q = metadata.cu_seqlens_q[batch_index : batch_index + 2]
        cu_seqlens_q = cu_seqlens_q - cu_seqlens_q[0]
        return self._build_scheduler_metadata(
            max_seqlen_k=metadata.max_seq_len_k,
            cache_seqlens=cache_seqlens[batch_index : batch_index + 1],
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k_new=None,
            max_seqlen_q=1,
            causal=True,
            window_size=(-1, -1),
            page_size=self.page_size,
            attention_layout=self._mla_attention_layout,
            num_splits=1,
        )

    def _graph_metadata_key(
        self,
        bs: int,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ) -> MusaCudaGraphMetadataKey:
        if forward_mode.is_target_verify():
            return MusaCudaGraphMetadataKey(
                mode="target_verify",
                batch_size=bs,
                topk=self.topk,
                num_draft_tokens=self.speculative_num_draft_tokens,
            )
        if forward_mode.is_decode_or_idle() and spec_info is not None:
            return MusaCudaGraphMetadataKey(
                mode="decode_spec",
                batch_size=bs,
                topk=self.topk,
                speculative_step_id=self.speculative_step_id,
            )
        return MusaCudaGraphMetadataKey(mode="decode", batch_size=bs)

    @staticmethod
    def _append_graph_metadata_slots(
        slots: list[MusaSchedulerMetadataSlot],
        owner: Optional[MusaMetadataOwner],
        name_prefix: str = "",
    ) -> None:
        if owner is None:
            return

        for attr in _GRAPH_METADATA_ATTRS:
            tensor = getattr(owner, attr, None)
            if isinstance(tensor, torch.Tensor):
                slots.append(
                    MusaSchedulerMetadataSlot(
                        f"{name_prefix}{attr}", owner, attr, tensor
                    )
                )

        local_metadata = getattr(owner, "local_attn_metadata", None)
        if local_metadata is not None:
            tensor = getattr(local_metadata, "scheduler_metadata", None)
            if isinstance(tensor, torch.Tensor):
                slots.append(
                    MusaSchedulerMetadataSlot(
                        f"{name_prefix}local_attn_metadata.scheduler_metadata",
                        local_metadata,
                        "scheduler_metadata",
                        tensor,
                    )
                )

        swa_spec_metadata = getattr(owner, "swa_spec_metadata", None)
        if swa_spec_metadata is not None:
            tensor = getattr(swa_spec_metadata, "swa_scheduler_metadata", None)
            if isinstance(tensor, torch.Tensor):
                slots.append(
                    MusaSchedulerMetadataSlot(
                        f"{name_prefix}swa_spec_metadata.swa_scheduler_metadata",
                        swa_spec_metadata,
                        "swa_scheduler_metadata",
                        tensor,
                    )
                )

        for side in _CP_METADATA_SIDES:
            for attr in _cp_metadata_attrs(side):
                tensor = getattr(owner, attr, None)
                if isinstance(tensor, torch.Tensor):
                    slots.append(
                        MusaSchedulerMetadataSlot(
                            f"{name_prefix}{attr}", owner, attr, tensor
                        )
                    )

    def _graph_metadata_slots(self) -> list[MusaSchedulerMetadataSlot]:
        slots: list[MusaSchedulerMetadataSlot] = []
        self._append_graph_metadata_slots(slots, self.forward_metadata)
        self._append_graph_metadata_slots(
            slots,
            self.forward_metadata_spec_decode_expand,
            name_prefix="spec_expand.",
        )
        return slots

    def _capture_graph_scheduler_metadata(
        self,
        bs: int,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ) -> None:
        key = self._graph_metadata_key(bs, forward_mode, spec_info)
        self._captured_graph_metadata[key] = {
            slot.name: slot.tensor for slot in self._graph_metadata_slots()
        }

    def _copy_fresh_metadata_to_graph_tensors(
        self,
        bs: int,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ) -> None:
        """Copy replayed scheduler metadata into the graph-captured tensors.

        CUDA graphs bake the captured tensor addresses into the replay, so the
        freshly built metadata has to be copied back into the buffers captured
        for this graph key. Every precondition below is a hard failure: silently
        keeping the captured values would replay stale scheduler metadata and
        produce wrong attention output with no error anywhere.
        """
        key = self._graph_metadata_key(bs, forward_mode, spec_info)
        captured_tensors = self._captured_graph_metadata.get(key)
        if captured_tensors is None:
            raise RuntimeError(
                "MUSA FA3 CUDA graph replay found no captured scheduler metadata "
                f"for key={key}. Replay must only run for a graph that was "
                "captured with the same mode/batch/speculative layout."
            )

        fresh_slots = {slot.name: slot for slot in self._graph_metadata_slots()}
        if set(captured_tensors) != set(fresh_slots):
            raise RuntimeError(
                "MUSA FA3 CUDA graph scheduler metadata slots changed between "
                f"capture and replay for key={key}: "
                f"capture={sorted(captured_tensors)} "
                f"replay={sorted(fresh_slots)}."
            )
        for name, captured_tensor in captured_tensors.items():
            slot = fresh_slots[name]
            fresh_tensor = slot.tensor
            if captured_tensor.shape != fresh_tensor.shape:
                raise RuntimeError(
                    "MUSA FA3 CUDA graph scheduler metadata shape changed for "
                    f"{name}, key={key}: capture={tuple(captured_tensor.shape)} "
                    f"replay={tuple(fresh_tensor.shape)}."
                )
            if captured_tensor.data_ptr() == fresh_tensor.data_ptr():
                raise RuntimeError(
                    "MUSA FA3 CUDA graph replay scheduler metadata refresh did "
                    f"not produce a fresh tensor for {name}, key={key}, "
                    f"shape={tuple(captured_tensor.shape)}. This is a self-copy, "
                    "so the captured graph would keep using stale scheduler "
                    "metadata."
                )
            captured_tensor.copy_(fresh_tensor)
            setattr(slot.owner, slot.attribute, captured_tensor)

    @staticmethod
    def _max_seqlen_from_metadata(
        max_seqlen_k: int, cache_seqlens: torch.Tensor
    ) -> int:
        # Prefer the static max length prepared by the base FA metadata path.
        # The fallback reads `cache_seqlens.max().item()`, which synchronizes a
        # device tensor to host and is not safe while graph capture is active.
        # During capture, this fallback is only a shape placeholder for owners
        # that do not carry a static max length; replay refreshes the captured
        # scheduler tensor contents before graph launch.
        #
        # Eager-path callers that hit the fallback are metadata owners without a
        # static max (e.g. speculative SWA spec metadata). Keep a static
        # `max_seq_len_k` on those owners when adding new call sites so this
        # fallback stays a rare D2H sync rather than a per-layer cost.
        if max_seqlen_k and max_seqlen_k > 0:
            return max_seqlen_k
        if get_is_capture_mode():
            return 1
        return int(cache_seqlens.max().item())

    def _init_swa_scheduler_metadata(
        self,
        owner: MusaMetadataOwner,
        attr: str,
        *,
        max_seqlen_k: int,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k_new: Optional[torch.Tensor],
        max_seqlen_q: int,
        causal: bool,
        page_size: int,
    ) -> None:
        if not (
            self.has_swa and self._attention_layouts.sliding_window_size is not None
        ):
            setattr(owner, attr, None)
            return

        # SWA scheduler metadata must be generated with the same window that the
        # kernel will see at runtime. The model scan above enforces one uniform
        # SWA window, so callers do not need to thread layer objects through
        # metadata initialization.
        setattr(
            owner,
            attr,
            self._build_scheduler_metadata(
                max_seqlen_k=max_seqlen_k,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k_new=cu_seqlens_k_new,
                max_seqlen_q=max_seqlen_q,
                causal=causal,
                window_size=self._attention_layouts.sliding_window_size,
                page_size=page_size,
                attention_layout=self._attention_layouts.sliding_window,
                num_splits=self.num_splits,
            ),
        )

    def _clear_cp_scheduler_metadata(
        self, metadata: MusaFlashAttentionMetadata
    ) -> None:
        for side in _CP_METADATA_SIDES:
            for attr in _cp_metadata_attrs(side):
                setattr(metadata, attr, None)

    @staticmethod
    def _cp_side_lengths(
        cp_meta: BaseContextParallelMetadata, side: str
    ) -> tuple[int, int, torch.Tensor, int]:
        """Return (q_total, k_max, k_lens_tensor, q_max) for one CP side.

        The CPU values published by the CP strategy are used whenever available:
        reading them avoids three device-to-host synchronizations per side.
        The tensor fallbacks only exist for CP metadata implementations that do
        not publish the CPU mirrors.
        """
        if side == "prev":
            q_lens = cp_meta.actual_seq_q_prev_tensor
            k_lens = cp_meta.kv_len_prev_tensor
            k_lens_list = getattr(cp_meta, "kv_len_prev_list", None)
            q_total = getattr(cp_meta, "total_q_prev_tokens", None)
            q_max = getattr(cp_meta, "max_seqlen_q_prev", None)
        elif side == "next":
            q_lens = cp_meta.actual_seq_q_next_tensor
            k_lens = cp_meta.kv_len_next_tensor
            k_lens_list = getattr(cp_meta, "kv_len_next_list", None)
            q_total = getattr(cp_meta, "total_q_next_tokens", None)
            q_max = getattr(cp_meta, "max_seqlen_q_next", None)
        else:
            raise ValueError(f"unknown CP side: {side}")
        if q_lens is None or k_lens is None:
            raise ValueError(f"CP metadata is missing {side} sequence lengths")

        if k_lens_list is not None:
            k_max = max(k_lens_list) if len(k_lens_list) > 0 else 0
        else:
            k_max = int(k_lens.max().item())
        if q_total is None:
            q_total = int(q_lens.sum().item())
        if q_max is None:
            q_max = int(q_lens.max().item())
        return q_total, k_max, k_lens, q_max

    def _init_cp_self_scheduler_metadata(
        self,
        metadata: MusaFlashAttentionMetadata,
        side: str,
        *,
        q_len: int,
        k_len: int,
        k_len_tensor: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k_new: torch.Tensor,
        attention_layout: Optional[AttentionLayout] = None,
    ) -> None:
        full_attr = f"cp_scheduler_metadata_{side}"
        swa_attr = f"cp_swa_scheduler_metadata_{side}"
        if attention_layout is not None or self._attention_layouts.has_full_attention:
            setattr(
                metadata,
                full_attr,
                self._build_scheduler_metadata(
                    max_seqlen_k=k_len,
                    cache_seqlens=k_len_tensor,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k_new=cu_seqlens_k_new,
                    max_seqlen_q=q_len,
                    causal=True,
                    window_size=(-1, -1),
                    page_size=self.page_size,
                    attention_layout=attention_layout,
                ),
            )
        else:
            setattr(metadata, full_attr, None)

        self._init_swa_scheduler_metadata(
            metadata,
            swa_attr,
            max_seqlen_k=k_len,
            cache_seqlens=k_len_tensor,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k_new=cu_seqlens_k_new,
            max_seqlen_q=q_len,
            causal=True,
            page_size=self.page_size,
        )

    def _init_cp_local_scheduler_metadata(
        self,
        metadata: MusaFlashAttentionMetadata,
        local_metadata: Optional[MusaLocalAttentionMetadata],
        side: str,
        *,
        q_len: int,
        k_len: int,
        k_len_tensor: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
    ) -> None:
        local_attr = f"cp_local_scheduler_metadata_{side}"
        local_swa_attr = f"cp_local_swa_scheduler_metadata_{side}"
        # Local attention has separate block tables. Its scheduler metadata uses
        # the same CP q/cache lengths, but no `cu_seqlens_k_new` because the local
        # path does not append newly gathered KV through the normal paged-cache
        # table in the callback.
        if local_metadata is not None and self._attention_layouts.has_local_attention:
            setattr(
                metadata,
                local_attr,
                self._build_scheduler_metadata(
                    max_seqlen_k=k_len,
                    cache_seqlens=k_len_tensor,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k_new=None,
                    max_seqlen_q=q_len,
                    causal=True,
                    window_size=(-1, -1),
                    page_size=self.page_size,
                ),
            )
        else:
            setattr(metadata, local_attr, None)

        if (
            local_metadata is not None
            and self._attention_layouts.has_sliding_window_local_attention
        ):
            self._init_swa_scheduler_metadata(
                metadata,
                local_swa_attr,
                max_seqlen_k=k_len,
                cache_seqlens=k_len_tensor,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k_new=None,
                max_seqlen_q=q_len,
                causal=True,
                page_size=self.page_size,
            )
        else:
            setattr(metadata, local_swa_attr, None)

    def _init_cp_side_scheduler_metadata(
        self,
        metadata: MusaFlashAttentionMetadata,
        local_metadata: Optional[MusaLocalAttentionMetadata],
        cp_meta: BaseContextParallelMetadata,
        side: str,
        device: torch.device,
        attention_layout: Optional[AttentionLayout] = None,
    ) -> None:
        # CP extend is split into two single-sequence kvcache calls by the
        # selected strategy: prev and next. Each side has its own q length
        # Each CP side has a distinct visible KV length, so wrapper metadata must
        # use matching CP-local cu_seqlens instead of the full extend cu_seqlens.
        q_total, k_len, k_len_tensor, q_len = self._cp_side_lengths(cp_meta, side)
        cu_seqlens_q = getattr(cp_meta, f"cu_seqlens_q_{side}_tensor", None)
        cu_seqlens_k_new = getattr(cp_meta, f"cu_seqlens_kv_{side}_tensor", None)
        if cu_seqlens_q is None:
            cu_seqlens_q = torch.tensor([0, q_total], device=device, dtype=torch.int32)
        if cu_seqlens_k_new is None:
            cu_seqlens_k_new = torch.tensor(
                [0, k_len], device=device, dtype=torch.int32
            )
        setattr(metadata, f"cp_cu_seqlens_k_new_{side}", cu_seqlens_k_new)
        self._init_cp_self_scheduler_metadata(
            metadata,
            side,
            q_len=q_len,
            k_len=k_len,
            k_len_tensor=k_len_tensor,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k_new=cu_seqlens_k_new,
            attention_layout=attention_layout,
        )
        self._init_cp_local_scheduler_metadata(
            metadata,
            local_metadata,
            side,
            q_len=q_len,
            k_len=k_len,
            k_len_tensor=k_len_tensor,
            cu_seqlens_q=cu_seqlens_q,
        )

    def _init_cp_scheduler_metadata(
        self,
        forward_batch: ForwardBatch,
        local_metadata: Optional[MusaLocalAttentionMetadata],
        attention_layout: Optional[AttentionLayout] = None,
    ) -> None:
        metadata = self.forward_metadata
        if metadata is None:
            return

        if not self._is_context_parallel_extend(forward_batch):
            self._clear_cp_scheduler_metadata(metadata)
            return

        cp_meta = forward_batch.attn_cp_metadata
        device = metadata.cache_seqlens_int32.device
        self._init_cp_side_scheduler_metadata(
            metadata, local_metadata, cp_meta, "prev", device, attention_layout
        )
        self._init_cp_side_scheduler_metadata(
            metadata, local_metadata, cp_meta, "next", device, attention_layout
        )

    def _init_full_attention_metadata(
        self,
        metadata,
        *,
        needs_kvcache_metadata: bool,
        cu_seqlens_k_new: Optional[torch.Tensor],
        max_seqlen_k: int,
        causal: bool,
        num_splits: Optional[int] = None,
    ):
        # Base self-attention metadata is only useful for full-attention layers.
        # In all-SWA models this would never be consumed, so keep the attr clear
        # and let accidental full-layer use fail at `_scheduler_metadata_for_kvcache`.
        if needs_kvcache_metadata and self._attention_layouts.has_full_attention:
            metadata.scheduler_metadata = self._build_scheduler_metadata(
                max_seqlen_k=max_seqlen_k,
                cache_seqlens=metadata.cache_seqlens_int32,
                cu_seqlens_q=metadata.cu_seqlens_q,
                cu_seqlens_k_new=cu_seqlens_k_new,
                max_seqlen_q=metadata.max_seq_len_q,
                causal=causal,
                window_size=(-1, -1),
                page_size=self.page_size,
                num_splits=num_splits,
            )
        else:
            metadata.scheduler_metadata = None

    def _init_swa_attention_metadata(
        self,
        metadata,
        *,
        needs_kvcache_metadata: bool,
        cu_seqlens_k_new: Optional[torch.Tensor],
        max_seqlen_k: int,
        causal: bool,
    ):
        # SWA metadata mirrors the base cache/page/q shape, but the scheduler is
        # built with the SWA window. `_init_swa_scheduler_metadata` clears the
        # attr when the model has no SWA layers.
        #
        # Check that condition before resolving max_seqlen_k. The fallback in
        # `_max_seqlen_from_metadata` calls `cache_seqlens.max().item()`, which
        # introduces an unnecessary device-to-host synchronization for models
        # without SWA even though the resulting metadata is immediately dropped.
        if not (
            self.has_swa and self._attention_layouts.sliding_window_size is not None
        ):
            metadata.swa_scheduler_metadata = None
            return

        if needs_kvcache_metadata:
            self._init_swa_scheduler_metadata(
                metadata,
                "swa_scheduler_metadata",
                max_seqlen_k=self._max_seqlen_from_metadata(
                    max_seqlen_k,
                    metadata.cache_seqlens_int32,
                ),
                cache_seqlens=metadata.cache_seqlens_int32,
                cu_seqlens_q=metadata.cu_seqlens_q,
                cu_seqlens_k_new=cu_seqlens_k_new,
                max_seqlen_q=metadata.max_seq_len_q,
                causal=causal,
                page_size=self.page_size,
            )
        else:
            metadata.swa_scheduler_metadata = None

    def _init_local_scheduler_metadata(
        self,
        local_metadata,
        *,
        needs_kvcache_metadata: bool,
    ):
        # Local attention owns a different page table and KV length tensor from
        # the normal paged-cache path, so it cannot reuse base/SWA metadata.
        # Only build it when local metadata exists for this batch.
        if local_metadata is None:
            return

        if needs_kvcache_metadata:
            local_metadata.scheduler_metadata = self._build_scheduler_metadata(
                max_seqlen_k=local_metadata.local_max_seq_len,
                cache_seqlens=local_metadata.local_seqused_k,
                cu_seqlens_q=local_metadata.local_query_start_loc,
                cu_seqlens_k_new=None,
                max_seqlen_q=local_metadata.local_max_query_len,
                causal=True,
                window_size=(-1, -1),
                page_size=self.page_size,
            )
        else:
            local_metadata.scheduler_metadata = None

    def _init_encoder_scheduler_metadata(
        self,
        metadata,
        *,
        needs_kvcache_metadata: bool,
        is_decode: bool,
    ):
        # Encoder/cross-attention has separate cache lengths and is always
        # non-causal. Avoid building it for decoder-only models.
        if (
            needs_kvcache_metadata
            and self._attention_layouts.has_cross_attention
            and metadata.encoder_lens_int32 is not None
        ):
            metadata.encoder_scheduler_metadata = self._build_scheduler_metadata(
                max_seqlen_k=metadata.encoder_max_seq_len_k,
                cache_seqlens=metadata.encoder_lens_int32,
                cu_seqlens_q=metadata.cu_seqlens_q,
                cu_seqlens_k_new=metadata.encoder_cu_seqlens_k,
                max_seqlen_q=1 if is_decode else metadata.max_seq_len_q,
                causal=False,
                window_size=(-1, -1),
                page_size=self.page_size,
            )
        else:
            metadata.encoder_scheduler_metadata = None

    def _spec_decode_expand_metadata(self, forward_batch: ForwardBatch, use_cascade):
        if not use_cascade:
            return None
        if (
            forward_batch.forward_mode.is_target_verify()
            and not self._attention_layouts.has_full_attention
        ):
            return None
        return self.forward_metadata_spec_decode_expand

    def _init_full_spec_decode_expand_scheduler_metadata(
        self,
        metadata_expand,
        *,
        page_size: int,
        num_splits: Optional[int] = None,
    ):
        if not self._attention_layouts.has_full_attention:
            metadata_expand.scheduler_metadata = None
            return

        metadata_expand.scheduler_metadata = self._build_scheduler_metadata(
            max_seqlen_k=self._max_seqlen_from_metadata(
                metadata_expand.max_seq_len_k, metadata_expand.cache_seqlens_int32
            ),
            cache_seqlens=metadata_expand.cache_seqlens_int32,
            cu_seqlens_q=metadata_expand.cu_seqlens_q,
            cu_seqlens_k_new=metadata_expand.cu_seqlens_k,
            max_seqlen_q=metadata_expand.max_seq_len_q,
            causal=False,
            window_size=(-1, -1),
            page_size=page_size,
            num_splits=num_splits,
        )

    def _init_swa_spec_decode_expand_scheduler_metadata(
        self,
        metadata,
        metadata_expand,
        *,
        page_size: int,
    ):
        self._init_swa_scheduler_metadata(
            metadata_expand,
            "swa_scheduler_metadata",
            max_seqlen_k=self._max_seqlen_from_metadata(
                metadata_expand.max_seq_len_k,
                metadata_expand.cache_seqlens_int32,
            ),
            cache_seqlens=metadata_expand.cache_seqlens_int32,
            cu_seqlens_q=metadata_expand.cu_seqlens_q,
            cu_seqlens_k_new=metadata_expand.cu_seqlens_k,
            max_seqlen_q=metadata_expand.max_seq_len_q,
            causal=False,
            page_size=page_size,
        )
        if metadata.swa_spec_metadata is None:
            return

        swa_spec_metadata = metadata.swa_spec_metadata
        # Some speculative SWA batches carry a dedicated metadata owner because
        # their page table/length tensors differ from the base expand owner.
        self._init_swa_scheduler_metadata(
            swa_spec_metadata,
            "swa_scheduler_metadata",
            max_seqlen_k=self._max_seqlen_from_metadata(
                0,
                swa_spec_metadata.cache_seqlens_int32,
            ),
            cache_seqlens=swa_spec_metadata.cache_seqlens_int32,
            cu_seqlens_q=swa_spec_metadata.cu_seqlens_q,
            cu_seqlens_k_new=swa_spec_metadata.cu_seqlens_k,
            max_seqlen_q=swa_spec_metadata.max_seq_len_q,
            causal=True,
            page_size=page_size,
        )

    def _init_spec_decode_expand_scheduler_metadata(
        self,
        metadata,
        forward_batch: ForwardBatch,
        *,
        use_cascade: bool,
        num_splits: Optional[int] = None,
    ):
        # Speculative cascade performs a second kvcache call over the expanded
        # draft/verify tokens. Target-verify disables cascade for SWA layers, so
        # a model with no full-attention layers does not need expand metadata in
        # that mode.
        metadata_expand = self._spec_decode_expand_metadata(forward_batch, use_cascade)
        if metadata_expand is None:
            stale_expand_metadata = self.forward_metadata_spec_decode_expand
            if stale_expand_metadata is not None:
                stale_expand_metadata.scheduler_metadata = None
                stale_expand_metadata.swa_scheduler_metadata = None
            swa_spec_metadata = getattr(metadata, "swa_spec_metadata", None)
            if (
                swa_spec_metadata is not None
                and forward_batch.forward_mode.is_target_verify()
                and self.topk > 1
            ):
                self._init_swa_scheduler_metadata(
                    swa_spec_metadata,
                    "swa_scheduler_metadata",
                    max_seqlen_k=self._max_seqlen_from_metadata(
                        0,
                        swa_spec_metadata.cache_seqlens_int32,
                    ),
                    cache_seqlens=swa_spec_metadata.cache_seqlens_int32,
                    cu_seqlens_q=swa_spec_metadata.cu_seqlens_q,
                    cu_seqlens_k_new=swa_spec_metadata.cu_seqlens_k,
                    max_seqlen_q=swa_spec_metadata.max_seq_len_q,
                    causal=True,
                    page_size=1,
                )
            elif swa_spec_metadata is not None:
                swa_spec_metadata.swa_scheduler_metadata = None
            return

        expand_page_size = (
            1 if forward_batch.forward_mode.is_target_verify() else self.page_size
        )
        self._init_full_spec_decode_expand_scheduler_metadata(
            metadata_expand,
            page_size=expand_page_size,
            num_splits=num_splits,
        )
        self._init_swa_spec_decode_expand_scheduler_metadata(
            metadata,
            metadata_expand,
            page_size=expand_page_size,
        )

    def _init_no_mla_scheduler_metadata(self, forward_batch: SchedulerMetadataBatch):
        metadata = self.forward_metadata
        if metadata is None:
            return

        is_decode = forward_batch.forward_mode.is_decode_or_idle()
        use_cascade = self.topk > 1 and (
            (is_decode and forward_batch.spec_info is not None)
            or forward_batch.forward_mode.is_target_verify()
        )
        # Target-verify SWA does not use the cascade path in forward_extend, so
        # its scheduler metadata must keep the normal causal flag.
        use_swa_cascade = (
            self.topk > 1 and is_decode and forward_batch.spec_info is not None
        )
        needs_kvcache_metadata = self._needs_no_mla_kvcache_scheduler_metadata(
            forward_batch
        )
        local_metadata = getattr(metadata, "local_attn_metadata", None)

        if self._is_context_parallel_extend(forward_batch):
            needs_kvcache_metadata = False

        needs_encoder_kvcache_metadata = (
            needs_kvcache_metadata or metadata.encoder_lens_int32 is not None
        )
        cu_seqlens_k_new = None if is_decode else metadata.cu_seqlens_k
        full_num_splits = self.num_splits
        self._init_full_attention_metadata(
            metadata,
            needs_kvcache_metadata=needs_kvcache_metadata,
            cu_seqlens_k_new=cu_seqlens_k_new,
            max_seqlen_k=metadata.max_seq_len_k,
            causal=not use_cascade,
            num_splits=full_num_splits,
        )
        self._init_swa_attention_metadata(
            metadata,
            needs_kvcache_metadata=needs_kvcache_metadata,
            cu_seqlens_k_new=cu_seqlens_k_new,
            max_seqlen_k=metadata.max_seq_len_k,
            causal=not use_swa_cascade,
        )
        self._init_local_scheduler_metadata(
            local_metadata,
            needs_kvcache_metadata=needs_kvcache_metadata,
        )
        self._init_cp_scheduler_metadata(forward_batch, local_metadata)
        self._init_encoder_scheduler_metadata(
            metadata,
            needs_kvcache_metadata=needs_encoder_kvcache_metadata,
            is_decode=is_decode,
        )
        self._init_spec_decode_expand_scheduler_metadata(
            metadata,
            forward_batch,
            use_cascade=use_cascade,
            num_splits=full_num_splits,
        )

    def _init_scheduler_metadata(self, forward_batch: SchedulerMetadataBatch):
        if self.use_mla:
            self._init_mla_scheduler_metadata(forward_batch)
            self._init_cp_scheduler_metadata(
                forward_batch,
                local_metadata=None,
                attention_layout=self._mla_attention_layout,
            )
        else:
            self._init_no_mla_scheduler_metadata(forward_batch)

    @staticmethod
    def _synthetic_forward_batch_for_scheduler_metadata(
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        batch_size: Optional[int] = None,
    ) -> MusaSchedulerMetadataBatch:
        return MusaSchedulerMetadataBatch(
            forward_mode=forward_mode,
            spec_info=spec_info,
            batch_size=batch_size,
            attn_cp_metadata=None,
        )

    def _init_scheduler_metadata_for_batch(self, forward_batch: SchedulerMetadataBatch):
        self._init_scheduler_metadata(forward_batch)

    def _prepare_graph_capture_scheduler_metadata(
        self,
        bs: int,
        forward_mode,
        spec_info,
    ):
        forward_batch_for_init = self._synthetic_forward_batch_for_scheduler_metadata(
            forward_mode, spec_info, batch_size=bs
        )
        self._init_scheduler_metadata_for_batch(forward_batch_for_init)
        self._capture_graph_scheduler_metadata(bs, forward_mode, spec_info)

    def _prepare_graph_replay_scheduler_metadata(
        self,
        bs: int,
        forward_mode,
        spec_info,
    ):
        """Regenerate replay metadata without changing graph-captured addresses."""
        forward_batch_for_init = self._synthetic_forward_batch_for_scheduler_metadata(
            forward_mode, spec_info, batch_size=bs
        )
        self._init_scheduler_metadata(forward_batch_for_init)
        self._copy_fresh_metadata_to_graph_tensors(bs, forward_mode, spec_info)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        self._ensure_metadata_types()
        if (
            getattr(forward_batch, "input_ids", None) is not None
            and forward_batch.input_ids.numel() == 0
        ):
            # Empty DP-attention ranks still traverse target verify so that
            # TBO shifts the first effective attention layer so its MLP
            # collectives match the active rank. The scheduler-metadata JIT
            # cannot compile a zero-query specialization, so metadata stays
            # unset for this batch. Attention forward is a zero-row no-op on
            # these ranks; if one were ever reached with real queries, the
            # kvcache wrapper raises on the missing scheduler_metadata instead
            # of silently reusing another batch's plan.
            return
        self._init_scheduler_metadata_for_batch(forward_batch)

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        """Prepare MUSA scheduler metadata through the v0.5.16 graph hook."""
        super().init_forward_metadata_out_graph(forward_batch, in_capture)
        self._ensure_metadata_types()
        if in_capture:
            self._prepare_graph_capture_scheduler_metadata(
                forward_batch.batch_size,
                forward_batch.forward_mode,
                forward_batch.spec_info,
            )
        else:
            self._prepare_graph_replay_scheduler_metadata(
                forward_batch.batch_size,
                forward_batch.forward_mode,
                forward_batch.spec_info,
            )

    def _scheduler_metadata_for_kvcache(
        self,
        metadata: Optional[MusaFlashAttentionMetadata],
        *,
        is_swa_layer: bool = False,
        is_cross_attention: bool = False,
        local_metadata: Optional[MusaLocalAttentionMetadata] = None,
    ) -> Optional[MusaSchedulerMetadata]:
        # Runtime kvcache call sites should not rebuild scheduler metadata.
        # They select the owner refreshed by init/cuda-graph replay and fail
        # fast if the selected family was not initialized for this batch.
        if local_metadata is not None:
            scheduler_metadata = getattr(local_metadata, "scheduler_metadata", None)
        elif is_cross_attention:
            scheduler_metadata = getattr(metadata, "encoder_scheduler_metadata", None)
        elif is_swa_layer:
            scheduler_metadata = getattr(metadata, "swa_scheduler_metadata", None)
        else:
            scheduler_metadata = getattr(metadata, "scheduler_metadata", None)

        if scheduler_metadata is None:
            return None
        return scheduler_metadata

    def _spec_decode_expand_scheduler_metadata(
        self, *, is_swa_layer: bool
    ) -> Optional[MusaSchedulerMetadata]:
        return self._scheduler_metadata_for_kvcache(
            self.forward_metadata_spec_decode_expand,
            is_swa_layer=is_swa_layer,
        )

    @staticmethod
    def _cp_side_from_kv_lens(
        forward_batch: ForwardBatch, cache_seqlens_cp: torch.Tensor
    ) -> str:
        """Resolve which CP side published ``cache_seqlens_cp``.

        The shared CP strategy calls the attention callback once per side with
        that side's cache-length tensor and does not pass a side label. This
        backend keeps the shared CP stack untouched, so the callback tensor is
        matched against the two per-side tensors published in
        ``attn_cp_metadata``. Anything else raises: the previous fallback
        silently selected ``next`` for non-identical tensors.
        """
        cp_meta = forward_batch.attn_cp_metadata
        if cache_seqlens_cp is cp_meta.kv_len_prev_tensor:
            return "prev"
        if cache_seqlens_cp is cp_meta.kv_len_next_tensor:
            return "next"
        raise RuntimeError(
            "MUSA FA3 CP kvcache attention received a cache-length tensor that "
            "is neither attn_cp_metadata.kv_len_prev_tensor nor "
            "kv_len_next_tensor; cannot select the per-side scheduler metadata."
        )

    def _cp_scheduler_metadata_for_kvcache(
        self,
        metadata: MusaFlashAttentionMetadata,
        forward_batch: ForwardBatch,
        cache_seqlens_cp: torch.Tensor,
        is_swa_layer: bool,
        use_local_attn: bool,
    ) -> torch.Tensor:
        side = self._cp_side_from_kv_lens(forward_batch, cache_seqlens_cp)
        if use_local_attn:
            prefix = "cp_local_swa" if is_swa_layer else "cp_local"
            error_scope = "CP local"
        else:
            prefix = "cp_swa" if is_swa_layer else "cp"
            error_scope = "CP"
        attr = f"{prefix}_scheduler_metadata_{side}"
        scheduler_metadata = getattr(metadata, attr, None)
        if scheduler_metadata is None:
            raise RuntimeError(
                f"MUSA FA3 {error_scope} kvcache attention requires precomputed "
                f"{attr}; init_forward_metadata must refresh it first."
            )
        return scheduler_metadata

    def _cp_cu_seqlens_k_new_for_kvcache(
        self,
        metadata,
        forward_batch: ForwardBatch,
        cache_seqlens_cp: torch.Tensor,
    ) -> torch.Tensor:
        side = self._cp_side_from_kv_lens(forward_batch, cache_seqlens_cp)
        attr = f"{_CP_METADATA_CU_SEQLENS_K_NEW_ATTR}_{side}"
        cu_seqlens_k_new = getattr(metadata, attr, None)
        if cu_seqlens_k_new is None:
            raise RuntimeError(
                "MUSA FA3 CP kvcache attention requires precomputed "
                f"{attr}; init_forward_metadata must refresh it first."
            )
        return cu_seqlens_k_new

    def _gather_paged_kv_for_varlen(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        page_table: torch.Tensor,
        seq_lens_cpu,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize the whole prefix KV as dense varlen tensors.

        Eager-only, deterministic-cached-prefix path: this is O(prefix) device
        memory and needs host sequence lengths, so it must never be wired into a
        CUDA graph. Keep it out of capture/replay call sites.
        """
        if get_is_capture_mode():
            raise RuntimeError(
                "MUSA FA3 varlen KV gather is eager-only and cannot run during "
                "CUDA graph capture; the deterministic cached-prefix path is "
                "not graph-captured."
            )
        seq_lens = self._seq_lens_cpu_list(seq_lens_cpu)
        if seq_lens is None:
            raise RuntimeError(
                "seq_lens_cpu is required for the eager-only MUSA FA3 varlen KV gather."
            )

        key_chunks = []
        value_chunks = []
        for batch_idx, seq_len in enumerate(seq_lens):
            seq_len = int(seq_len)
            num_pages = (seq_len + self.page_size - 1) // self.page_size
            pages = page_table[batch_idx, :num_pages].to(torch.long)
            key_chunks.append(
                key_cache.index_select(0, pages).reshape(
                    -1, key_cache.shape[-2], key_cache.shape[-1]
                )[:seq_len]
            )
            value_chunks.append(
                value_cache.index_select(0, pages).reshape(
                    -1, value_cache.shape[-2], value_cache.shape[-1]
                )[:seq_len]
            )

        return torch.cat(key_chunks, dim=0), torch.cat(value_chunks, dim=0)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
        q_descale: Optional[torch.Tensor] = None,
        k_descale: Optional[torch.Tensor] = None,
        v_descale: Optional[torch.Tensor] = None,
        score_mod=None,
        aux_tensors=None,
        rel_bias=None,
        rel_bias_event=None,
        return_lse: bool = False,
    ):
        if rel_bias is not None or rel_bias_event is not None:
            raise NotImplementedError(
                "Relative position bias is not supported by the MUSA FA3 backend."
            )
        if score_mod is not None or aux_tensors is not None:
            raise NotImplementedError(
                "score_mod and aux_tensors are not supported by the MUSA FA3 backend."
            )

        if q.numel() == 0:
            return q.new_empty((0, layer.tp_q_head_num * layer.v_head_dim))

        if k is not None:
            assert v is not None

            is_cp_mode = (
                forward_batch.forward_mode.is_context_parallel_extend()
                and forward_batch.attn_cp_metadata is not None
                and self.attn_cp_size > 1
            )

            if save_kv_cache and not is_cp_mode and not self.fa_skip_kv_cache:
                write_loc = KVWriteLoc.for_layer(
                    forward_batch,
                    layer,
                    window_locs=window_write_locs(
                        self.runner_ids, self.forward_metadata.swa_out_cache_loc
                    ),
                )
                if not self.use_mla:
                    k_scale = k_descale if self.kv_cache_is_mxfp8 else layer.k_scale
                    v_scale = v_descale if self.kv_cache_is_mxfp8 else layer.v_scale
                    self.token_to_kv_pool.set_kv_buffer(
                        layer,
                        write_loc,
                        k,
                        v,
                        k_scale,
                        v_scale,
                    )
                else:
                    self.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        write_loc,
                        k,
                        k_rope,
                    )
            if is_cp_mode:
                cp_strategy = get_cp_strategy()
                assert cp_strategy is not None
                if self.use_mla:
                    cp_strategy.materialize_full_mla_kv(forward_batch, layer, k, k_rope)
                else:
                    cp_strategy.materialize_full_kv(
                        forward_batch,
                        layer,
                        k,
                        v,
                        window_locs=(
                            window_write_locs(
                                self.runner_ids,
                                self.forward_metadata.swa_out_cache_loc,
                            )
                            if self.use_sliding_window_kv_pool
                            else None
                        ),
                    )

        metadata = self.forward_metadata

        local_metadata = None
        is_swa_layer = (
            layer.sliding_window_size is not None and layer.sliding_window_size > -1
        )
        if layer.attn_type == AttentionType.DECODER_BIDIRECTIONAL:
            window_size = (layer.sliding_window_size, layer.sliding_window_size)
        else:
            window_size = (layer.sliding_window_size, 0) if is_swa_layer else (-1, -1)
        if (
            self.kv_cache_dtype_str != "auto"
            and layer.head_dim <= 256
            and self.fa_impl_ver != 4
        ):
            if k_descale is None and layer.k_scale is not None:
                descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
                k_descale = layer.k_scale.expand(descale_shape)
                if v_descale is None:
                    v_descale = layer.v_scale.expand(descale_shape)
            q = q.to(self.kv_cache_dtype)
            q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None
            k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None
        causal = not (
            layer.is_cross_attention
            or layer.attn_type
            in (AttentionType.ENCODER_ONLY, AttentionType.DECODER_BIDIRECTIONAL)
        )

        use_local_attn = (
            self.has_local_attention
            and self.attention_chunk_size is not None
            and metadata.local_attn_metadata is not None
            and (hasattr(layer, "use_irope") and layer.use_irope)
        )

        use_cascade_attn = (
            forward_batch.forward_mode.is_target_verify()
            and self.topk > 1
            and not is_swa_layer
        )

        kwargs = {}
        if sinks is not None:
            kwargs["sinks"] = sinks
        if q_descale is not None:
            kwargs["q_descale"] = q_descale
        return_lse = return_lse or forward_batch.mha_return_lse

        flash_attn_varlen_func = self.flash_attn_varlen_func
        flash_attn_with_kvcache = self.flash_attn_with_kvcache

        if self.fa_skip_kv_cache:
            # Piecewise CUDA graph pads the token dimension up to a captured
            # bucket size, so q has more rows than cu_seqlens_q covers.
            # flash_attn_varlen_func requires q.shape[0] == cu_seqlens_q[-1];
            # append the padded tail as a dummy self-attending segment (same
            # handling as the parent backend) and cache it on the metadata.
            if metadata.fa_skip_cu_seqlens_q is None:
                num_real_tokens = forward_batch.extend_num_tokens
                num_padded_tokens = q.shape[0]
                if num_real_tokens is not None and num_real_tokens < num_padded_tokens:
                    metadata.fa_skip_cu_seqlens_q = torch.cat(
                        [
                            metadata.cu_seqlens_q,
                            metadata.cu_seqlens_q.new_tensor([num_padded_tokens]),
                        ]
                    )
                    metadata.fa_skip_max_seqlen_q = max(
                        int(metadata.max_seq_len_q),
                        num_padded_tokens - num_real_tokens,
                    )
                else:
                    metadata.fa_skip_cu_seqlens_q = metadata.cu_seqlens_q
                    metadata.fa_skip_max_seqlen_q = metadata.max_seq_len_q
            output = flash_attn_varlen_func(
                q=q.contiguous().view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                k=k.contiguous().view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                v=v.contiguous().view(-1, layer.tp_v_head_num, layer.v_head_dim),
                cu_seqlens_q=metadata.fa_skip_cu_seqlens_q,
                cu_seqlens_k=metadata.fa_skip_cu_seqlens_q,
                max_seqlen_q=metadata.fa_skip_max_seqlen_q,
                max_seqlen_k=metadata.fa_skip_max_seqlen_q,
                softmax_scale=layer.scaling,
                causal=causal,
                window_size=window_size,
                softcap=layer.logit_cap,
                return_softmax_lse=return_lse,
                **kwargs,
            )
            if return_lse:
                return (
                    output[0].view(-1, layer.tp_q_head_num * layer.v_head_dim),
                    output[1],
                )
            return output.view(-1, layer.tp_q_head_num * layer.v_head_dim)

        scheduler_metadata_owner = metadata
        if use_local_attn:
            local_metadata = metadata.local_attn_metadata
            page_table = local_metadata.local_block_table
            cu_seqlens_q = local_metadata.local_query_start_loc
            cache_seqlens = local_metadata.local_seqused_k
            max_seqlen_q = local_metadata.local_max_query_len
        elif is_swa_layer and metadata.swa_spec_metadata is not None:
            swa_spec_metadata = metadata.swa_spec_metadata
            page_table = swa_spec_metadata.page_table
            cu_seqlens_q = swa_spec_metadata.cu_seqlens_q
            cache_seqlens = swa_spec_metadata.cache_seqlens_int32
            max_seqlen_q = swa_spec_metadata.max_seq_len_q
            cu_seqlens_k = swa_spec_metadata.cu_seqlens_k
            scheduler_metadata_owner = swa_spec_metadata
        else:
            page_table = metadata.page_table
            if is_swa_layer and self.use_sliding_window_kv_pool:
                if metadata.swa_page_table is not None:
                    page_table = metadata.swa_page_table
                else:
                    page_table = self.token_to_kv_pool.translate_loc_from_full_to_swa(
                        metadata.page_table
                    )
            cu_seqlens_q = metadata.cu_seqlens_q
            cache_seqlens = metadata.cache_seqlens_int32
            max_seqlen_q = metadata.max_seq_len_q
            cu_seqlens_k = metadata.cu_seqlens_k

        if (
            local_metadata is not None
            and getattr(local_metadata, "scheduler_metadata", None) is not None
        ):
            scheduler_metadata = self._scheduler_metadata_for_kvcache(
                scheduler_metadata_owner,
                is_swa_layer=is_swa_layer,
                is_cross_attention=layer.is_cross_attention,
                local_metadata=local_metadata,
            )
        else:
            scheduler_metadata = self._scheduler_metadata_for_kvcache(
                scheduler_metadata_owner,
                is_swa_layer=is_swa_layer,
                is_cross_attention=layer.is_cross_attention,
            )
        num_splits = self.num_splits

        if not self.use_mla:
            key_cache, value_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)

            key_cache = key_cache.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            )
            value_cache = value_cache.view(
                -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
            )
            if layer.is_cross_attention:
                page_table = metadata.encoder_page_table
                cache_seqlens = metadata.encoder_lens_int32
                cu_seqlens_k = metadata.encoder_cu_seqlens_k
                window_size = (-1, -1)
                scheduler_metadata_owner = metadata

            deterministic_cached_prefix = (
                get_exec().deterministic.enable_deterministic_inference
                and forward_batch.extend_prefix_lens_cpu is not None
                and any(forward_batch.extend_prefix_lens_cpu)
            )
            if deterministic_cached_prefix and (
                not forward_batch.forward_mode.is_extend()
                or forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend_v2()
                or layer.is_cross_attention
                or use_local_attn
                or is_swa_layer
                or use_cascade_attn
                or (
                    forward_batch.forward_mode.is_context_parallel_extend()
                    and forward_batch.attn_cp_metadata is not None
                    and self.attn_cp_size > 1
                )
            ):
                raise NotImplementedError(
                    "MUSA FA3 deterministic cached-prefix attention only supports "
                    "standard self-attention without context parallel, local "
                    "attention, SWA, cascade attention, or speculative decoding."
                )

            if (
                forward_batch.forward_mode.is_context_parallel_extend()
                and forward_batch.attn_cp_metadata is not None
                and self.attn_cp_size > 1
            ):

                def _fa_cp_attn(
                    q_chunk, cu_seqlens_q_cp, cache_seqlens_cp, max_seqlen_q_cp
                ):
                    cu_seqlens_k_cp = None
                    if not use_local_attn:
                        cu_seqlens_k_cp = self._cp_cu_seqlens_k_new_for_kvcache(
                            metadata,
                            forward_batch,
                            cache_seqlens_cp,
                        )
                    scheduler_metadata = self._cp_scheduler_metadata_for_kvcache(
                        metadata,
                        forward_batch,
                        cache_seqlens_cp,
                        is_swa_layer,
                        use_local_attn=use_local_attn,
                    )
                    return flash_attn_with_kvcache(
                        q=q_chunk,
                        k_cache=key_cache,
                        v_cache=value_cache,
                        page_table=page_table,
                        cache_seqlens=cache_seqlens_cp,
                        cu_seqlens_q=cu_seqlens_q_cp,
                        cu_seqlens_k_new=(
                            cu_seqlens_k_cp if not use_local_attn else None
                        ),
                        max_seqlen_q=max_seqlen_q_cp,
                        softmax_scale=layer.scaling,
                        causal=False if use_cascade_attn else causal,
                        window_size=window_size,
                        softcap=layer.logit_cap,
                        k_descale=k_descale,
                        v_descale=v_descale,
                        return_softmax_lse=use_cascade_attn,
                        scheduler_metadata=scheduler_metadata,
                        num_splits=self.num_splits,
                        **kwargs,
                    )

                cp_strategy = get_cp_strategy()
                assert cp_strategy is not None
                result = cp_strategy.run_attention(
                    q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    forward_batch,
                    self.device,
                    _fa_cp_attn,
                    attention_backend=CPAttentionBackendKind.FLASH_ATTENTION,
                )
            elif deterministic_cached_prefix:
                full_k, full_v = self._gather_paged_kv_for_varlen(
                    key_cache,
                    value_cache,
                    page_table,
                    forward_batch.seq_lens_cpu,
                )
                output = flash_attn_varlen_func(
                    q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    k=full_k.to(q.dtype),
                    v=full_v.to(q.dtype),
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=metadata.max_seq_len_k,
                    softmax_scale=layer.scaling,
                    causal=causal,
                    return_softmax_lse=return_lse,
                    **kwargs,
                )
                if return_lse:
                    output, lse, *_rest = output
                    lse = torch.transpose(lse, 0, 1).contiguous()
                    return (
                        output.view(-1, layer.tp_q_head_num * layer.v_head_dim),
                        lse,
                    )
                return output.view(-1, layer.tp_q_head_num * layer.v_head_dim)
            elif self._needs_no_mla_kvcache_scheduler_metadata(
                forward_batch, is_cross_attention=layer.is_cross_attention
            ):
                result = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,
                    max_seqlen_q=max_seqlen_q,
                    softmax_scale=layer.scaling,
                    causal=False if use_cascade_attn else causal,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=use_cascade_attn,
                    scheduler_metadata=scheduler_metadata,
                    num_splits=num_splits,
                    **kwargs,
                )
            else:
                output = flash_attn_varlen_func(
                    q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    k=k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
                    v=v.view(-1, layer.tp_k_head_num, layer.v_head_dim).to(q.dtype),
                    cu_seqlens_q=metadata.cu_seqlens_q,
                    cu_seqlens_k=metadata.cu_seqlens_q,
                    max_seqlen_q=metadata.max_seq_len_q,
                    max_seqlen_k=metadata.max_seq_len_q,
                    softmax_scale=layer.scaling,
                    causal=True,
                    return_softmax_lse=return_lse,
                    **kwargs,
                )
                if return_lse:
                    output, lse, *_rest = output
                    lse = torch.transpose(lse, 0, 1).contiguous()
                    return (
                        output.view(-1, layer.tp_q_head_num * layer.v_head_dim),
                        lse,
                    )
                return output.view(-1, layer.tp_q_head_num * layer.v_head_dim)

            if use_cascade_attn:
                o, softmax_lse, *_rest = result
                o_expand, softmax_lse_expand, *_rest_expand = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache.view(-1, 1, layer.tp_k_head_num, layer.head_dim),
                    v_cache=value_cache.view(
                        -1, 1, layer.tp_v_head_num, layer.head_dim
                    ),
                    page_table=self.forward_metadata_spec_decode_expand.page_table,
                    cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                    cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                    cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                    max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                    softmax_scale=layer.scaling,
                    causal=False,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=True,
                    scheduler_metadata=self._spec_decode_expand_scheduler_metadata(
                        is_swa_layer=is_swa_layer,
                    ),
                    num_splits=self.num_splits,
                    **kwargs,
                )
                o, _ = merge_state_v2_wrapper(
                    o,
                    softmax_lse.T.contiguous(),
                    o_expand,
                    softmax_lse_expand.T.contiguous(),
                )
            else:
                o = result
        else:
            if (
                forward_batch.attn_attend_prefix_cache is not None
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend_v2()
            ):
                if forward_batch.attn_attend_prefix_cache:
                    assert not get_server_args().disable_chunked_prefix_cache
                    assert forward_batch.prefix_chunk_idx is not None
                    assert forward_batch.prefix_chunk_cu_seq_lens is not None
                    assert forward_batch.prefix_chunk_max_seq_lens is not None

                    chunk_idx = forward_batch.prefix_chunk_idx
                    assert chunk_idx >= 0

                    assert return_lse
                    output = flash_attn_varlen_func(
                        q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
                        k=k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
                        v=v.view(-1, layer.tp_k_head_num, layer.v_head_dim).to(q.dtype),
                        cu_seqlens_q=metadata.cu_seqlens_q,
                        cu_seqlens_k=forward_batch.prefix_chunk_cu_seq_lens[chunk_idx],
                        max_seqlen_q=metadata.max_seq_len_q,
                        max_seqlen_k=forward_batch.prefix_chunk_max_seq_lens[chunk_idx],
                        softmax_scale=layer.scaling,
                        causal=False,
                        return_softmax_lse=True,
                        **kwargs,
                    )
                else:
                    cu_seqlens_k = (
                        metadata.cu_seqlens_q
                        if not forward_batch.mha_one_shot
                        else metadata.cu_seqlens_k
                    )
                    max_seqlen_k = (
                        metadata.max_seq_len_q
                        if not forward_batch.mha_one_shot
                        else metadata.max_seq_len_k
                    )
                    output = flash_attn_varlen_func(
                        q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
                        k=k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
                        v=v.view(-1, layer.tp_k_head_num, layer.v_head_dim).to(q.dtype),
                        cu_seqlens_q=metadata.cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_q=metadata.max_seq_len_q,
                        max_seqlen_k=max_seqlen_k,
                        softmax_scale=layer.scaling,
                        causal=True,
                        return_softmax_lse=return_lse,
                        **kwargs,
                    )
                if return_lse:
                    output, lse, *_rest = output
                    lse = torch.transpose(lse, 0, 1).contiguous()
                    return output, lse
                return output
            else:
                kv_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id).to(
                    q.dtype
                )
                k_rope = kv_cache[:, :, layer.v_head_dim :]
                c_kv = kv_cache[:, :, : layer.v_head_dim]
                k_rope_cache = k_rope.view(
                    -1,
                    self.page_size,
                    layer.tp_k_head_num,
                    layer.head_dim - layer.v_head_dim,
                )
                c_kv_cache = c_kv.view(
                    -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                )
                if q_rope is not None:
                    q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                    q_rope = q_rope.view(
                        -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
                    )
                else:
                    q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
                    q_nope = q_all[:, :, : layer.v_head_dim]
                    q_rope = q_all[:, :, layer.v_head_dim :]

                if is_cp_mode:
                    assert not use_cascade_attn, (
                        "Cascade attention under MLA CP is not supported."
                    )
                    q_fused = torch.cat([q_nope, q_rope], dim=-1)

                    def _mla_cp_attn(
                        q_chunk,
                        cu_seqlens_q_cp,
                        cache_seqlens_cp,
                        max_seqlen_q_cp,
                    ):
                        q_nope_chunk = q_chunk[..., : layer.v_head_dim]
                        q_rope_chunk = q_chunk[..., layer.v_head_dim :]
                        use_local_cp = use_local_attn
                        cu_seqlens_k_cp = (
                            None
                            if use_local_cp
                            else self._cp_cu_seqlens_k_new_for_kvcache(
                                metadata, forward_batch, cache_seqlens_cp
                            )
                        )
                        return flash_attn_with_kvcache(
                            q=q_rope_chunk,
                            qv=q_nope_chunk,
                            k_cache=k_rope_cache,
                            v_cache=c_kv_cache,
                            page_table=page_table,
                            cache_seqlens=cache_seqlens_cp,
                            cu_seqlens_q=cu_seqlens_q_cp,
                            cu_seqlens_k_new=cu_seqlens_k_cp,
                            max_seqlen_q=max_seqlen_q_cp,
                            softmax_scale=layer.scaling,
                            causal=causal,
                            softcap=layer.logit_cap,
                            k_descale=k_descale,
                            v_descale=v_descale,
                            scheduler_metadata=self._cp_scheduler_metadata_for_kvcache(
                                metadata,
                                forward_batch,
                                cache_seqlens_cp,
                                is_swa_layer=False,
                                use_local_attn=use_local_cp,
                            ),
                            num_splits=self.num_splits,
                            ver=self.fa_impl_ver,
                        )

                    cp_strategy = get_cp_strategy()
                    assert cp_strategy is not None
                    result = cp_strategy.run_attention(
                        q_fused,
                        forward_batch,
                        self.device,
                        _mla_cp_attn,
                        attention_backend=CPAttentionBackendKind.FLASH_ATTENTION,
                    )
                else:
                    result = flash_attn_with_kvcache(
                        q=q_rope,
                        k_cache=k_rope_cache,
                        v_cache=c_kv_cache,
                        qv=q_nope,
                        page_table=page_table,
                        cache_seqlens=cache_seqlens,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,
                        max_seqlen_q=max_seqlen_q,
                        softmax_scale=layer.scaling,
                        causal=False if use_cascade_attn else causal,
                        softcap=layer.logit_cap,
                        k_descale=k_descale,
                        v_descale=v_descale,
                        return_softmax_lse=use_cascade_attn,
                        scheduler_metadata=self._scheduler_metadata_for_kvcache(
                            metadata
                        ),
                        num_splits=self.num_splits,
                    )
                if use_cascade_attn:
                    o, softmax_lse, *_rest = result
                    o_expand, softmax_lse_expand, *_rest_expand = (
                        flash_attn_with_kvcache(
                            q=q_rope,
                            k_cache=k_rope_cache,
                            v_cache=c_kv_cache,
                            qv=q_nope,
                            page_table=self.forward_metadata_spec_decode_expand.page_table,
                            cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                            cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                            cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                            max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                            softmax_scale=layer.scaling,
                            causal=False,
                            window_size=window_size,
                            softcap=layer.logit_cap,
                            k_descale=k_descale,
                            v_descale=v_descale,
                            return_softmax_lse=True,
                            scheduler_metadata=self._spec_decode_expand_scheduler_metadata(
                                is_swa_layer=False
                            ),
                            num_splits=self.num_splits,
                        )
                    )
                    o, _ = merge_state_v2_wrapper(
                        o,
                        softmax_lse.T.contiguous(),
                        o_expand,
                        softmax_lse_expand.T.contiguous(),
                    )
                else:
                    o = result

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
        q_descale: Optional[torch.Tensor] = None,
        k_descale: Optional[torch.Tensor] = None,
        v_descale: Optional[torch.Tensor] = None,
        score_mod=None,
        aux_tensors=None,
        rel_bias=None,
        rel_bias_event=None,
        return_lse: bool = False,
    ) -> torch.Tensor:
        if rel_bias is not None or rel_bias_event is not None:
            raise NotImplementedError(
                "Relative position bias is not supported by the MUSA FA3 backend."
            )
        if score_mod is not None or aux_tensors is not None:
            raise NotImplementedError(
                "score_mod and aux_tensors are not supported by the MUSA FA3 backend."
            )
        if return_lse:
            raise NotImplementedError(
                "Decode LSE output is not supported by the MUSA FA3 backend."
            )

        if k is not None:
            assert v is not None
            if save_kv_cache:
                if not self.use_mla:
                    k_scale = k_descale if self.kv_cache_is_mxfp8 else layer.k_scale
                    v_scale = v_descale if self.kv_cache_is_mxfp8 else layer.v_scale
                    self.token_to_kv_pool.set_kv_buffer(
                        layer,
                        KVWriteLoc.for_layer(
                            forward_batch,
                            layer,
                            window_locs=window_write_locs(
                                self.runner_ids,
                                self.forward_metadata.swa_out_cache_loc,
                            ),
                        ),
                        k,
                        v,
                        k_scale,
                        v_scale,
                    )
                else:
                    self.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        KVWriteLoc.for_layer(forward_batch, layer),
                        k,
                        k_rope,
                    )

        metadata = self.forward_metadata
        local_attn_metadata = getattr(metadata, "local_attn_metadata", None)
        use_local_attn = (
            self.has_local_attention
            and self.attention_chunk_size is not None
            and local_attn_metadata is not None
            and (hasattr(layer, "use_irope") and layer.use_irope)
        )

        use_cascade_attn = forward_batch.spec_info is not None and self.topk > 1

        is_swa_layer = (
            layer.sliding_window_size is not None and layer.sliding_window_size > -1
        )
        if layer.attn_type == AttentionType.DECODER_BIDIRECTIONAL:
            window_size = (layer.sliding_window_size, layer.sliding_window_size)
        else:
            window_size = (layer.sliding_window_size, 0) if is_swa_layer else (-1, -1)

        causal = not (
            layer.is_cross_attention
            or layer.attn_type
            in (AttentionType.ENCODER_ONLY, AttentionType.DECODER_BIDIRECTIONAL)
        )

        kwargs = {}
        if sinks is not None:
            kwargs["sinks"] = sinks
        if q_descale is not None:
            kwargs["q_descale"] = q_descale
        flash_attn_with_kvcache = self.flash_attn_with_kvcache

        if self.kv_cache_dtype_str != "auto" and layer.head_dim <= 256:
            if k_descale is None and layer.k_scale is not None:
                descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
                k_descale = layer.k_scale.expand(descale_shape)
                if v_descale is None:
                    v_descale = layer.v_scale.expand(descale_shape)
            q = q.to(self.kv_cache_dtype)
            q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None
            k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None

        if not self.use_mla:
            key_cache, value_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            key_cache = key_cache.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            )
            value_cache = value_cache.view(
                -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
            )

            if layer.is_cross_attention:
                o = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=metadata.encoder_page_table,
                    cache_seqlens=metadata.encoder_lens_int32,
                    cu_seqlens_q=metadata.cu_seqlens_q,
                    cu_seqlens_k_new=metadata.encoder_cu_seqlens_k,
                    max_seqlen_q=1,
                    softmax_scale=layer.scaling,
                    causal=False,
                    window_size=(-1, -1),
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    scheduler_metadata=self._scheduler_metadata_for_kvcache(
                        metadata,
                        is_cross_attention=True,
                    ),
                    num_splits=self.num_splits,
                    **kwargs,
                )
            elif use_local_attn:
                o = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=local_attn_metadata.local_block_table,
                    cache_seqlens=local_attn_metadata.local_seqused_k,
                    cu_seqlens_q=local_attn_metadata.local_query_start_loc,
                    cu_seqlens_k_new=None,
                    max_seqlen_q=local_attn_metadata.local_max_query_len,
                    softmax_scale=layer.scaling,
                    causal=True,
                    window_size=(-1, -1),
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    scheduler_metadata=self._scheduler_metadata_for_kvcache(
                        metadata,
                        local_metadata=local_attn_metadata,
                    ),
                    num_splits=self.num_splits,
                    **kwargs,
                )
            else:
                page_table = metadata.page_table
                if is_swa_layer and self.use_sliding_window_kv_pool:
                    if metadata.swa_page_table is not None:
                        page_table = metadata.swa_page_table
                    else:
                        page_table = (
                            self.token_to_kv_pool.translate_loc_from_full_to_swa(
                                metadata.page_table
                            )
                        )
                cache_seqlens = metadata.cache_seqlens_int32
                max_seqlen_q = metadata.max_seq_len_q
                decode_num_splits = self.num_splits
                q_reshaped = q.contiguous().view(
                    -1, layer.tp_q_head_num, layer.head_dim
                )

                result = flash_attn_with_kvcache(
                    q=q_reshaped,
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=metadata.cu_seqlens_q,
                    max_seqlen_q=max_seqlen_q,
                    softmax_scale=layer.scaling,
                    causal=False if use_cascade_attn else causal,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=use_cascade_attn,
                    scheduler_metadata=self._scheduler_metadata_for_kvcache(
                        metadata,
                        is_swa_layer=is_swa_layer,
                    ),
                    num_splits=decode_num_splits,
                    **kwargs,
                )
                if use_cascade_attn:
                    o, softmax_lse, *_rest = result
                    o_expand, softmax_lse_expand, *_rest_expand = (
                        flash_attn_with_kvcache(
                            q=q_reshaped,
                            k_cache=key_cache,
                            v_cache=value_cache,
                            page_table=self.forward_metadata_spec_decode_expand.page_table,
                            cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                            cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                            cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                            max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                            softmax_scale=layer.scaling,
                            causal=False,
                            window_size=window_size,
                            softcap=layer.logit_cap,
                            k_descale=k_descale,
                            v_descale=v_descale,
                            return_softmax_lse=True,
                            scheduler_metadata=self._spec_decode_expand_scheduler_metadata(
                                is_swa_layer=is_swa_layer,
                            ),
                            num_splits=decode_num_splits,
                            **kwargs,
                        )
                    )
                    o, _ = merge_state_v2_wrapper(
                        o,
                        softmax_lse.T.contiguous(),
                        o_expand,
                        softmax_lse_expand.T.contiguous(),
                    )
                else:
                    o = result
        else:
            kv_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id).to(q.dtype)
            k_rope = kv_cache[:, :, layer.v_head_dim :]
            c_kv = kv_cache[:, :, : layer.v_head_dim]
            k_rope_cache = k_rope.view(
                -1,
                self.page_size,
                layer.tp_k_head_num,
                layer.head_dim - layer.v_head_dim,
            )
            c_kv_cache = c_kv.view(
                -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
            )

            if q_rope is not None:
                q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                q_rope = q_rope.view(
                    -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
                )
            else:
                q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
                q_nope = q_all[:, :, : layer.v_head_dim]
                q_rope = q_all[:, :, layer.v_head_dim :]
            max_seqlen_q = metadata.max_seq_len_q

            deterministic_mla_decode = (
                _platform_available
                and get_exec().deterministic.enable_deterministic_inference
                and forward_batch.forward_mode.is_decode()
                and not use_cascade_attn
                and not use_local_attn
            )
            if deterministic_mla_decode:
                assert max_seqlen_q == 1
                single_cu_seqlens_q = metadata.cu_seqlens_q[:2]
                results = []
                for batch_idx in range(forward_batch.batch_size):
                    results.append(
                        flash_attn_with_kvcache(
                            q=q_rope[batch_idx : batch_idx + 1],
                            k_cache=k_rope_cache,
                            v_cache=c_kv_cache,
                            qv=q_nope[batch_idx : batch_idx + 1],
                            page_table=metadata.page_table[batch_idx : batch_idx + 1],
                            cache_seqlens=metadata.cache_seqlens_int32[
                                batch_idx : batch_idx + 1
                            ],
                            cu_seqlens_q=single_cu_seqlens_q,
                            max_seqlen_q=1,
                            softmax_scale=layer.scaling,
                            causal=causal,
                            softcap=layer.logit_cap,
                            k_descale=k_descale,
                            v_descale=v_descale,
                            num_splits=1,
                            scheduler_metadata=self._mla_scheduler_metadata_for_decode_slice(
                                metadata,
                                metadata.cache_seqlens_int32,
                                batch_idx,
                            ),
                        )
                    )
                result = torch.cat(results, dim=0)
            else:
                result = flash_attn_with_kvcache(
                    q=q_rope,
                    k_cache=k_rope_cache,
                    v_cache=c_kv_cache,
                    qv=q_nope,
                    page_table=metadata.page_table,
                    cache_seqlens=metadata.cache_seqlens_int32,
                    cu_seqlens_q=metadata.cu_seqlens_q,
                    cu_seqlens_k_new=metadata.cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    softmax_scale=layer.scaling,
                    causal=False if use_cascade_attn else causal,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=use_cascade_attn,
                    scheduler_metadata=self._scheduler_metadata_for_kvcache(metadata),
                    num_splits=self.num_splits,
                )
            if use_cascade_attn:
                o, softmax_lse, *_rest = result
                o_expand, softmax_lse_expand, *_rest_expand = flash_attn_with_kvcache(
                    q=q_rope,
                    k_cache=k_rope_cache,
                    v_cache=c_kv_cache,
                    qv=q_nope,
                    page_table=self.forward_metadata_spec_decode_expand.page_table,
                    cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                    cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                    cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                    max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                    softmax_scale=layer.scaling,
                    causal=False,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=True,
                    scheduler_metadata=self._spec_decode_expand_scheduler_metadata(
                        is_swa_layer=False
                    ),
                    num_splits=self.num_splits,
                )
                o, _ = merge_state_v2_wrapper(
                    o,
                    softmax_lse.T.contiguous(),
                    o_expand,
                    softmax_lse_expand.T.contiguous(),
                )
            else:
                o = result

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)


class MusaFlashAttentionMultiStepBackend(FlashAttentionMultiStepBackend):
    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
        fa_impl_ver: int = 3,
    ):
        if fa_impl_ver != 3:
            raise ValueError("MUSA flash attention backend only supports FA3.")

        self.model_runner = model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.kv_index_translator = model_runner.kv_index_translator
        self.attn_backends = []
        for i in range(self.speculative_num_steps - 1):
            backend = MusaFlashAttentionBackend(
                model_runner,
                speculative_step_id=i,
                topk=self.topk,
                speculative_num_steps=self.speculative_num_steps,
                fa_impl_ver=fa_impl_ver,
            )
            self.attn_backends.append(backend)
        self.attn_backend_list = self.attn_backends
        self.forward_metadata = None
