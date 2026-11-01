from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
from flash_attn_interface import flash_attn_varlen_func, get_scheduler_metadata
from flash_attn_interface import flash_attn_with_kvcache as mate_flash_attn_with_kvcache
from sglang.srt.distributed.parallel_state import get_pp_group
from sglang.srt.distributed.utils import get_pp_indices
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.musa.layers.utils.cp_utils import (
    musa_cp_attn_forward_extend as cp_attn_forward_extend,
)
from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionBackend,
    FlashAttentionMultiStepBackend,
    merge_state_v2_wrapper,
)
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.utils.cp_utils import cp_allgather_and_save_kv_cache
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode
from sglang.srt.runtime_context import get_exec, get_server_args
from sglang.srt.utils import is_musa

_is_musa = is_musa()

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

# Global workspace buffer for MLA
_MATE_MLA_WORKSPACE_SIZE_BYTES = 128 * 1024 * 1024
MINIMAX_M2_DECODE_NUM_SPLITS = 16
MINIMAX_M2_DECODE_SPLIT_MAX_BS = 4


def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    qv: torch.Tensor | None = None,
    rotary_cos: torch.Tensor | None = None,
    rotary_sin: torch.Tensor | None = None,
    cache_seqlens: int | torch.Tensor | None = None,
    cache_batch_idx: torch.Tensor | None = None,
    cache_leftpad: torch.Tensor | None = None,
    page_table: torch.Tensor | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k_new: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    rotary_seqlens: torch.Tensor | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    attention_chunk: int = 0,
    softcap: float = 0.0,
    rotary_interleaved: bool = True,
    scheduler_metadata: torch.Tensor | tuple[torch.Tensor, bool] | None = None,
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

    assert scheduler_metadata is not None, (
        "MUSA MATE FA3 flash_attn_with_kvcache requires scheduler_metadata."
    )

    if (
        envs.SGLANG_MUSA_FA3_SYNC_BEFORE_KVCACHE_ATTENTION.get()
        and not get_is_capture_mode()
    ):
        torch.get_device_module(q.device).synchronize()

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
    def __init__(self, model_runner: ModelRunner, **kwargs):
        fa_impl_ver = kwargs.get("fa_impl_ver", 3)
        if fa_impl_ver != 3:
            raise ValueError("MUSA flash attention backend only supports FA3.")

        super().__init__(model_runner, **kwargs)
        model_type = getattr(model_runner.model_config.hf_config, "model_type", "")
        architectures = getattr(
            model_runner.model_config.hf_config, "architectures", []
        )
        self._is_gemma4 = model_type in {
            "gemma4",
            "gemma4_text",
            "gemma4_unified",
        } or any("Gemma4" in arch for arch in (architectures or []))
        # Gemma4's short sliding-window attention is faster without split-KV,
        # while full attention benefits from FA3's automatic split heuristic.
        # Keep this model-scoped: forcing split=1 globally regresses other
        # models such as Qwen3-Next.
        if self._is_gemma4:
            self.num_splits = 1
        self.full_attention_num_splits = 0 if self._is_gemma4 else self.num_splits
        self.flash_attn_varlen_func = flash_attn_varlen_func
        self.flash_attn_with_kvcache = flash_attn_with_kvcache
        self.num_hidden_layers = model_runner.model_config.num_hidden_layers
        self.first_k_dense_replace = model_runner.model_config.first_k_dense_replace
        self.full_attention_interval = model_runner.model_config.full_attention_interval
        self._is_minimax_m2 = "MiniMaxM2ForCausalLM" in (architectures or [])
        if not self.use_mla:
            (
                self._full_attention_layout,
                self._swa_attention_layout,
                self._swa_window_size,
                self._has_full_attention_layer,
                self._has_local_attention_layer,
                self._has_cross_attention_layer,
                self._has_swa_local_attention_layer,
            ) = self._get_no_mla_layer_attention_layouts(model_runner)
        _softcapping = getattr(
            model_runner.model_config.hf_text_config, "attn_logit_softcapping", None
        )
        self.has_softcap = _softcapping is not None and _softcapping > 0.0
        self._mate_mla_workspace_buffer = (
            torch.empty(
                _MATE_MLA_WORKSPACE_SIZE_BYTES,
                device=self.device,
                dtype=torch.uint8,
            )
            if self.use_mla
            else None
        )
        self._captured_cuda_graph_metadata: dict[tuple, dict[str, torch.Tensor]] = {}
        # Disable default scheduler metadata for fa3
        self._get_scheduler_metadata = None

    @staticmethod
    def _seq_lens_cpu_list(seq_lens_cpu) -> list | None:
        if seq_lens_cpu is None:
            return None
        if isinstance(seq_lens_cpu, torch.Tensor):
            if seq_lens_cpu.device.type != "cpu" and get_is_capture_mode():
                return None
            return seq_lens_cpu.detach().cpu().tolist()
        return list(seq_lens_cpu)

    @staticmethod
    def _layer_attention_layout(module: RadixAttention):
        return (
            module.tp_q_head_num,
            module.tp_k_head_num,
            module.qk_head_dim,
            module.v_head_dim,
        )

    @classmethod
    def _get_no_mla_layer_attention_layouts(cls, model_runner: ModelRunner):
        # MATE scheduler metadata is built from a single attention layout per
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
        full_layouts = set()
        swa_layouts = set()
        swa_window_sizes = set()
        has_full_attention_layer = False
        has_local_attention_layer = False
        has_cross_attention_layer = False
        has_swa_local_attention_layer = False
        for module in model_runner.model.modules():
            if not isinstance(module, RadixAttention):
                continue

            if getattr(module, "is_cross_attention", False):
                has_cross_attention_layer = True
                continue

            if (
                module.sliding_window_size is not None
                and module.sliding_window_size > -1
            ):
                swa_layouts.add(cls._layer_attention_layout(module))
                swa_window_sizes.add(module.sliding_window_size)
                has_swa_local_attention_layer |= bool(
                    getattr(module, "use_irope", False)
                )
            else:
                full_layouts.add(cls._layer_attention_layout(module))
                has_full_attention_layer = True

            has_local_attention_layer |= bool(getattr(module, "use_irope", False))

        full_layouts = sorted(full_layouts)
        swa_layouts = sorted(swa_layouts)
        if len(full_layouts) > 1 or len(swa_layouts) > 1:
            raise ValueError(
                "MUSA FA3 no-MLA scheduler metadata assumes one uniform "
                "self-attention layout for all full-attention layers and one "
                "uniform self-attention layout for all SWA layers; full and "
                f"SWA layouts may differ. Got full={full_layouts} "
                f"and swa={swa_layouts}."
            )

        swa_window_sizes = sorted(swa_window_sizes)
        if len(swa_window_sizes) > 1:
            raise ValueError(
                "MUSA FA3 no-MLA scheduler metadata assumes one uniform SWA "
                f"window size. Got {swa_window_sizes}."
            )

        full_layout = full_layouts[0] if full_layouts else None
        swa_layout = swa_layouts[0] if swa_layouts else full_layout
        full_layout = full_layout if full_layout is not None else swa_layout
        swa_window_size = (
            (swa_window_sizes[0], 0) if len(swa_window_sizes) == 1 else None
        )
        return (
            full_layout,
            swa_layout,
            swa_window_size,
            has_full_attention_layer,
            has_local_attention_layer,
            has_cross_attention_layer,
            has_swa_local_attention_layer,
        )

    def _needs_no_mla_kvcache_scheduler_metadata(
        self, forward_batch: ForwardBatch
    ) -> bool:
        forward_mode = forward_batch.forward_mode
        extend_prefix_lens_cpu = getattr(forward_batch, "extend_prefix_lens_cpu", None)
        extend_with_prefix = extend_prefix_lens_cpu is not None and any(
            extend_prefix_lens_cpu
        )
        return (
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

    def _num_splits_for_no_mla_full_kvcache(self, forward_batch: ForwardBatch) -> int:
        """Keep MiniMax-M2 decode-like full-attention metadata/calls aligned."""
        if not self._is_minimax_m2:
            return self.full_attention_num_splits

        batch_size = getattr(forward_batch, "batch_size", None)
        if batch_size is None or batch_size > MINIMAX_M2_DECODE_SPLIT_MAX_BS:
            return self.num_splits

        forward_mode = forward_batch.forward_mode
        if (
            forward_mode.is_decode_or_idle()
            or forward_mode.is_target_verify()
            or forward_mode.is_draft_extend_v2()
        ):
            return MINIMAX_M2_DECODE_NUM_SPLITS
        return self.num_splits

    def _num_splits_for_no_mla_swa_kvcache(self, batch_size: int | None) -> int:
        # On Gemma4/S5000, forcing one split only wins at the largest captured
        # decode batch. FA3 auto scheduling is 7-17 us faster per SWA layer at
        # bs=1..7; with 50 SWA layers this materially improves the AIME tail.
        if self._is_gemma4 and batch_size is not None and batch_size < 8:
            return 0
        return self.num_splits

    def _num_splits_for_no_mla_call(
        self,
        forward_batch: ForwardBatch,
        *,
        is_swa_layer: bool = False,
        is_cross_attention: bool = False,
        use_local_attn: bool = False,
    ) -> int:
        if is_swa_layer:
            return self._num_splits_for_no_mla_swa_kvcache(
                getattr(forward_batch, "batch_size", None)
            )
        if is_cross_attention or use_local_attn:
            return self.num_splits
        return self._num_splits_for_no_mla_full_kvcache(forward_batch)

    def _is_context_parallel_extend(self, forward_batch: ForwardBatch) -> bool:
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
    # MLA paths:
    # - All MLA kvcache calls pass `(self._mate_mla_workspace_buffer,
    #   skip_update)` as scheduler metadata.
    # - `_init_mla_scheduler_metadata` only clears stale no-MLA fields. The MLA
    #   tuple is built at each kvcache call because `skip_update` depends on the
    #   current layer. Only the first effective MLA layer updates MATE metadata;
    #   later layers skip repeated updates. PP rank, TBO,
    #   `first_k_dense_replace`, and `full_attention_interval` are included in
    #   that first-layer decision. `SGLANG_MUSA_FA3_FORCE_UPDATE_METADATA`
    #   disables the skip for debugging.
    def _compute_no_mla_scheduler_metadata(
        self,
        *,
        max_seqlen_k: int,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k_new: torch.Tensor | None,
        max_seqlen_q: int,
        causal: bool,
        window_size: tuple[int, int],
        page_size: int,
        attention_layout: tuple[int, int, int, int] | None = None,
        num_splits: int | None = None,
    ) -> torch.Tensor | None:
        assert not self.use_mla
        if cache_seqlens is None or cu_seqlens_q is None:
            return None
        num_heads_q, num_heads_kv, head_dim, v_head_dim = (
            attention_layout or self._full_attention_layout
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

    def _should_skip_mla_scheduler_metadata_update(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        force_update: bool = False,
    ) -> bool:
        assert self.use_mla

        should_update = True
        pp_group = get_pp_group()
        pp_rank = pp_group.rank_in_group
        start_layer_id, _ = get_pp_indices(
            self.num_hidden_layers, pp_rank, pp_group.world_size
        )
        if getattr(forward_batch, "can_run_tbo", False) and pp_rank == 0:
            start_layer_id += (
                self.first_k_dense_replace
                if self.first_k_dense_replace is not None
                else 0
            )

        if self.full_attention_interval is not None:
            start_layer_id += self.full_attention_interval - 1

        if layer.layer_id > start_layer_id:
            should_update = False

        if force_update:
            should_update = True

        if envs.SGLANG_MUSA_FA3_FORCE_UPDATE_METADATA.get():
            should_update = True

        return not should_update

    def _mla_scheduler_metadata_for_kvcache(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        force_update: bool = False,
    ) -> tuple[torch.Tensor, bool]:
        assert self._mate_mla_workspace_buffer is not None
        return (
            self._mate_mla_workspace_buffer,
            self._should_skip_mla_scheduler_metadata_update(
                layer, forward_batch, force_update=force_update
            ),
        )

    def _init_mla_scheduler_metadata(self):
        if self.forward_metadata is None:
            return

        # MLA scheduler metadata is layer-dependent. Clear stale metadata here;
        # kvcache call sites build the actual `(workspace, skip_update)` tuple.
        self.forward_metadata.scheduler_metadata = None
        metadata_expand = self.forward_metadata_spec_decode_expand
        if metadata_expand is not None:
            metadata_expand.scheduler_metadata = None

    def _cuda_graph_metadata_key(self, bs: int, forward_mode, spec_info):
        if forward_mode.is_target_verify():
            return ("target_verify", bs, self.topk, self.speculative_num_draft_tokens)
        if forward_mode.is_decode_or_idle() and spec_info is not None:
            return ("decode_spec", bs, self.topk, self.speculative_step_id)
        return ("decode", bs)

    @staticmethod
    def _append_cuda_graph_metadata_slots(slots, owner, name_prefix: str = ""):
        if owner is None:
            return

        for attr in (
            "scheduler_metadata",
            "swa_scheduler_metadata",
            "encoder_scheduler_metadata",
        ):
            tensor = getattr(owner, attr, None)
            if isinstance(tensor, torch.Tensor):
                slots.append((f"{name_prefix}{attr}", owner, attr, tensor))

        local_metadata = getattr(owner, "local_attn_metadata", None)
        if local_metadata is not None:
            tensor = getattr(local_metadata, "scheduler_metadata", None)
            if isinstance(tensor, torch.Tensor):
                slots.append(
                    (
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
                    (
                        f"{name_prefix}swa_spec_metadata.swa_scheduler_metadata",
                        swa_spec_metadata,
                        "swa_scheduler_metadata",
                        tensor,
                    )
                )

        for side in ("prev", "next"):
            for attr in (
                f"cp_scheduler_metadata_{side}",
                f"cp_swa_scheduler_metadata_{side}",
                f"cp_local_scheduler_metadata_{side}",
                f"cp_local_swa_scheduler_metadata_{side}",
                f"cp_cu_seqlens_k_new_{side}",
            ):
                tensor = getattr(owner, attr, None)
                if isinstance(tensor, torch.Tensor):
                    slots.append((f"{name_prefix}{attr}", owner, attr, tensor))

    def _cuda_graph_metadata_slots(self):
        slots = []
        self._append_cuda_graph_metadata_slots(slots, self.forward_metadata)
        self._append_cuda_graph_metadata_slots(
            slots,
            self.forward_metadata_spec_decode_expand,
            name_prefix="spec_expand.",
        )
        return slots

    def _capture_cuda_graph_scheduler_metadata(self, bs: int, forward_mode, spec_info):
        key = self._cuda_graph_metadata_key(bs, forward_mode, spec_info)
        self._captured_cuda_graph_metadata[key] = {
            name: tensor
            for name, _owner, _attr, tensor in self._cuda_graph_metadata_slots()
        }

    def _copy_fresh_metadata_to_cuda_graph_tensors(
        self, bs: int, forward_mode, spec_info
    ):
        key = self._cuda_graph_metadata_key(bs, forward_mode, spec_info)
        captured_tensors = self._captured_cuda_graph_metadata.get(key)
        if captured_tensors is None:
            return False

        # CUDA graph captured the old tensor addresses. Replay generates fresh
        # metadata, then copies the fresh values back into those old tensors.
        fresh_slots = {
            name: (owner, attr, tensor)
            for name, owner, attr, tensor in self._cuda_graph_metadata_slots()
        }
        assert set(captured_tensors) == set(fresh_slots), (
            "MUSA FA3 CUDA graph scheduler metadata slots changed between "
            f"capture and replay: capture={sorted(captured_tensors)} "
            f"replay={sorted(fresh_slots)}."
        )
        for name, captured_tensor in captured_tensors.items():
            owner, attr, fresh_tensor = fresh_slots[name]
            assert captured_tensor.shape == fresh_tensor.shape, (
                "MUSA FA3 CUDA graph scheduler metadata shape changed for "
                f"{name}: capture={tuple(captured_tensor.shape)} "
                f"replay={tuple(fresh_tensor.shape)}."
            )
            assert captured_tensor.data_ptr() != fresh_tensor.data_ptr(), (
                "MUSA FA3 CUDA graph replay scheduler metadata refresh did not "
                f"produce a fresh tensor for {name}, key={key}, "
                f"shape={tuple(captured_tensor.shape)}. This is a self-copy, "
                "so the captured graph can keep using stale scheduler metadata."
            )
            captured_tensor.copy_(fresh_tensor)
            setattr(owner, attr, captured_tensor)
        return True

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
        if max_seqlen_k and max_seqlen_k > 0:
            return max_seqlen_k
        if get_is_capture_mode():
            return 1
        return int(cache_seqlens.max().item())

    def _init_swa_scheduler_metadata(
        self,
        owner,
        attr: str,
        *,
        max_seqlen_k: int,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k_new: torch.Tensor | None,
        max_seqlen_q: int,
        causal: bool,
        page_size: int,
    ):
        if not (self.has_swa and self._swa_window_size is not None):
            setattr(owner, attr, None)
            return

        # SWA scheduler metadata must be generated with the same window that the
        # kernel will see at runtime. The model scan above enforces one uniform
        # SWA window, so callers do not need to thread layer objects through
        # metadata initialization.
        setattr(
            owner,
            attr,
            self._compute_no_mla_scheduler_metadata(
                max_seqlen_k=max_seqlen_k,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k_new=cu_seqlens_k_new,
                max_seqlen_q=max_seqlen_q,
                causal=causal,
                window_size=self._swa_window_size,
                page_size=page_size,
                attention_layout=self._swa_attention_layout,
                num_splits=self._num_splits_for_no_mla_swa_kvcache(
                    cu_seqlens_q.shape[-1] - 1
                ),
            ),
        )

    def _clear_cp_scheduler_metadata(self, metadata):
        for side in ("prev", "next"):
            for prefix in (
                "cp",
                "cp_swa",
                "cp_local",
                "cp_local_swa",
            ):
                setattr(metadata, f"{prefix}_scheduler_metadata_{side}", None)
            setattr(metadata, f"cp_cu_seqlens_k_new_{side}", None)

    @staticmethod
    def _cp_side_lengths(cp_meta, side: str):
        if side == "prev":
            return (
                cp_meta.actual_seq_q_prev,
                int(cp_meta.kv_len_prev),
                cp_meta.kv_len_prev_tensor,
            )
        if side == "next":
            return (
                cp_meta.actual_seq_q_next,
                int(cp_meta.kv_len_next),
                cp_meta.kv_len_next_tensor,
            )
        raise ValueError(f"unknown CP side: {side}")

    def _init_cp_self_scheduler_metadata(
        self,
        metadata,
        side: str,
        *,
        q_len: int,
        k_len: int,
        k_len_tensor: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k_new: torch.Tensor,
    ):
        full_attr = f"cp_scheduler_metadata_{side}"
        swa_attr = f"cp_swa_scheduler_metadata_{side}"
        if self._has_full_attention_layer:
            setattr(
                metadata,
                full_attr,
                self._compute_no_mla_scheduler_metadata(
                    max_seqlen_k=k_len,
                    cache_seqlens=k_len_tensor,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k_new=cu_seqlens_k_new,
                    max_seqlen_q=q_len,
                    causal=True,
                    window_size=(-1, -1),
                    page_size=self.page_size,
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
        metadata,
        local_metadata,
        side: str,
        *,
        q_len: int,
        k_len: int,
        k_len_tensor: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
    ):
        local_attr = f"cp_local_scheduler_metadata_{side}"
        local_swa_attr = f"cp_local_swa_scheduler_metadata_{side}"
        # Local attention has separate block tables. Its scheduler metadata uses
        # the same CP q/cache lengths, but no `cu_seqlens_k_new` because the local
        # path does not append newly gathered KV through the normal paged-cache
        # table in the callback.
        if local_metadata is not None and self._has_local_attention_layer:
            setattr(
                metadata,
                local_attr,
                self._compute_no_mla_scheduler_metadata(
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

        if local_metadata is not None and self._has_swa_local_attention_layer:
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
        metadata,
        local_metadata,
        cp_meta,
        side: str,
        device: torch.device,
    ):
        # CP extend is split into two single-sequence kvcache calls by
        # `cp_attn_forward_extend`: prev and next. Each side has its own q length
        # and visible KV length, so the MATE scheduler metadata must be generated
        # with matching CP-local cu_seqlens instead of the full extend cu_seqlens.
        q_len, k_len, k_len_tensor = self._cp_side_lengths(cp_meta, side)
        cu_seqlens_q = torch.tensor([0, q_len], device=device, dtype=torch.int32)
        cu_seqlens_k_new = torch.tensor([0, k_len], device=device, dtype=torch.int32)
        setattr(metadata, f"cp_cu_seqlens_k_new_{side}", cu_seqlens_k_new)
        self._init_cp_self_scheduler_metadata(
            metadata,
            side,
            q_len=q_len,
            k_len=k_len,
            k_len_tensor=k_len_tensor,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k_new=cu_seqlens_k_new,
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

    def _init_cp_scheduler_metadata(self, forward_batch: ForwardBatch, local_metadata):
        metadata = self.forward_metadata
        if metadata is None:
            return

        if not self._is_context_parallel_extend(forward_batch):
            self._clear_cp_scheduler_metadata(metadata)
            return

        cp_meta = forward_batch.attn_cp_metadata
        device = metadata.cache_seqlens_int32.device
        self._init_cp_side_scheduler_metadata(
            metadata, local_metadata, cp_meta, "prev", device
        )
        self._init_cp_side_scheduler_metadata(
            metadata, local_metadata, cp_meta, "next", device
        )

    def _init_full_attention_metadata(
        self,
        metadata,
        *,
        needs_kvcache_metadata: bool,
        cu_seqlens_k_new: torch.Tensor | None,
        max_seqlen_k: int,
        causal: bool,
        num_splits: int | None = None,
    ):
        # Base self-attention metadata is only useful for full-attention layers.
        # In all-SWA models this would never be consumed, so keep the attr clear
        # and let accidental full-layer use fail at `_scheduler_metadata_for_kvcache`.
        if needs_kvcache_metadata and self._has_full_attention_layer:
            metadata.scheduler_metadata = self._compute_no_mla_scheduler_metadata(
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
        cu_seqlens_k_new: torch.Tensor | None,
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
        if not (self.has_swa and self._swa_window_size is not None):
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
            local_metadata.scheduler_metadata = self._compute_no_mla_scheduler_metadata(
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
            and self._has_cross_attention_layer
            and metadata.encoder_lens_int32 is not None
        ):
            metadata.encoder_scheduler_metadata = (
                self._compute_no_mla_scheduler_metadata(
                    max_seqlen_k=metadata.encoder_max_seq_len_k,
                    cache_seqlens=metadata.encoder_lens_int32,
                    cu_seqlens_q=metadata.cu_seqlens_q,
                    cu_seqlens_k_new=metadata.encoder_cu_seqlens_k,
                    max_seqlen_q=1 if is_decode else metadata.max_seq_len_q,
                    causal=False,
                    window_size=(-1, -1),
                    page_size=self.page_size,
                )
            )
        else:
            metadata.encoder_scheduler_metadata = None

    def _spec_decode_expand_metadata(self, forward_batch: ForwardBatch, use_cascade):
        if not use_cascade:
            return None
        if (
            forward_batch.forward_mode.is_target_verify()
            and not self._has_full_attention_layer
        ):
            return None
        return self.forward_metadata_spec_decode_expand

    def _init_full_spec_decode_expand_scheduler_metadata(
        self,
        metadata_expand,
        *,
        page_size: int,
        num_splits: int | None = None,
    ):
        if not self._has_full_attention_layer:
            metadata_expand.scheduler_metadata = None
            return

        metadata_expand.scheduler_metadata = self._compute_no_mla_scheduler_metadata(
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
        num_splits: int | None = None,
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

    def _init_no_mla_scheduler_metadata(self, forward_batch: ForwardBatch):
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
        full_num_splits = self._num_splits_for_no_mla_full_kvcache(forward_batch)
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

    def _init_scheduler_metadata(self, forward_batch: ForwardBatch):
        if self.use_mla:
            self._init_mla_scheduler_metadata()
        else:
            self._init_no_mla_scheduler_metadata(forward_batch)

    @staticmethod
    def _synthetic_forward_batch_for_scheduler_metadata(
        forward_mode, spec_info, batch_size: int | None = None
    ):
        return SimpleNamespace(
            forward_mode=forward_mode,
            spec_info=spec_info,
            batch_size=batch_size,
            attn_cp_metadata=None,
        )

    def _init_scheduler_metadata_for_batch(self, forward_batch):
        self._init_scheduler_metadata(forward_batch)

    def _prepare_cuda_graph_capture_scheduler_metadata(
        self,
        bs: int,
        num_tokens: int,
        seq_lens: torch.Tensor,
        forward_mode,
        spec_info,
    ):
        forward_batch_for_init = self._synthetic_forward_batch_for_scheduler_metadata(
            forward_mode, spec_info, batch_size=bs
        )
        self._init_scheduler_metadata_for_batch(forward_batch_for_init)
        self._capture_cuda_graph_scheduler_metadata(bs, forward_mode, spec_info)

    def _prepare_cuda_graph_replay_scheduler_metadata(
        self,
        bs: int,
        seq_lens_sum: int,
        forward_mode,
        spec_info,
        seq_lens_cpu: torch.Tensor | None,
    ):
        """Regenerate replay metadata without changing graph-captured addresses."""
        forward_batch_for_init = self._synthetic_forward_batch_for_scheduler_metadata(
            forward_mode, spec_info, batch_size=bs
        )
        self._init_scheduler_metadata(forward_batch_for_init)
        self._copy_fresh_metadata_to_cuda_graph_tensors(bs, forward_mode, spec_info)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        if (
            getattr(forward_batch, "input_ids", None) is not None
            and forward_batch.input_ids.numel() == 0
        ):
            # Empty DP-attention ranks still traverse target verify so that
            # their MLP collectives match the active rank.  MATE's scheduler
            # metadata JIT cannot compile a zero-query specialization; the
            # corresponding attention forward is a shape-preserving no-op.
            return
        self._init_scheduler_metadata_for_batch(forward_batch)

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        """Prepare MUSA scheduler metadata through the v0.5.16 graph hook."""
        super().init_forward_metadata_out_graph(forward_batch, in_capture)
        if in_capture:
            self._prepare_cuda_graph_capture_scheduler_metadata(
                forward_batch.batch_size,
                forward_batch.positions.numel(),
                forward_batch.seq_lens,
                forward_batch.forward_mode,
                forward_batch.spec_info,
            )
        else:
            seq_lens_sum = getattr(forward_batch, "seq_lens_sum", None)
            if seq_lens_sum is None:
                seq_lens_sum = forward_batch.positions.numel()
            self._prepare_cuda_graph_replay_scheduler_metadata(
                forward_batch.batch_size,
                seq_lens_sum,
                forward_batch.forward_mode,
                forward_batch.spec_info,
                forward_batch.seq_lens_cpu,
            )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: torch.Tensor | None,
        forward_mode,
        spec_info,
    ):
        super().init_forward_metadata_capture_cuda_graph(
            bs,
            num_tokens,
            req_pool_indices,
            seq_lens,
            encoder_lens,
            forward_mode,
            spec_info,
        )
        self._prepare_cuda_graph_capture_scheduler_metadata(
            bs,
            num_tokens,
            seq_lens,
            forward_mode,
            spec_info,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: torch.Tensor | None,
        forward_mode,
        spec_info,
        seq_lens_cpu: torch.Tensor | None,
        out_cache_loc: torch.Tensor | None = None,
    ):
        super().init_forward_metadata_replay_cuda_graph(
            bs,
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            encoder_lens,
            forward_mode,
            spec_info,
            seq_lens_cpu,
            out_cache_loc,
        )
        self._prepare_cuda_graph_replay_scheduler_metadata(
            bs,
            seq_lens_sum,
            forward_mode,
            spec_info,
            seq_lens_cpu,
        )

    def _scheduler_metadata_for_kvcache(
        self,
        metadata,
        *,
        is_swa_layer: bool = False,
        is_cross_attention: bool = False,
        local_metadata=None,
    ):
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

    def _spec_decode_expand_scheduler_metadata(self, *, is_swa_layer: bool):
        return self._scheduler_metadata_for_kvcache(
            self.forward_metadata_spec_decode_expand,
            is_swa_layer=is_swa_layer,
        )

    def _cp_scheduler_metadata_for_kvcache(
        self,
        metadata,
        forward_batch: ForwardBatch,
        cache_seqlens_cp,
        is_swa_layer: bool,
        use_local_attn: bool,
    ):
        assert not self.use_mla
        is_cp_prev = (
            cache_seqlens_cp is forward_batch.attn_cp_metadata.kv_len_prev_tensor
        )
        side = "prev" if is_cp_prev else "next"
        if use_local_attn:
            prefix = "cp_local_swa" if is_swa_layer else "cp_local"
            error_scope = "CP local"
        else:
            prefix = "cp_swa" if is_swa_layer else "cp"
            error_scope = "CP"
        attr = f"{prefix}_scheduler_metadata_{side}"
        scheduler_metadata = getattr(metadata, attr, None)
        assert scheduler_metadata is not None, (
            f"MUSA FA3 {error_scope} kvcache attention requires precomputed "
            f"{attr}; init_forward_metadata must refresh it first."
        )
        return scheduler_metadata

    def _cp_cu_seqlens_k_new_for_kvcache(
        self,
        metadata,
        forward_batch: ForwardBatch,
        cache_seqlens_cp,
    ) -> torch.Tensor:
        is_cp_prev = (
            cache_seqlens_cp is forward_batch.attn_cp_metadata.kv_len_prev_tensor
        )
        attr = "cp_cu_seqlens_k_new_prev" if is_cp_prev else "cp_cu_seqlens_k_new_next"
        cu_seqlens_k_new = getattr(metadata, attr, None)
        assert cu_seqlens_k_new is not None, (
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
        seq_lens = self._seq_lens_cpu_list(seq_lens_cpu)
        assert seq_lens is not None, (
            "seq_lens_cpu is required for MUSA FA3 varlen KV gather."
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
        q_rope: torch.Tensor | None = None,
        k_rope: torch.Tensor | None = None,
        sinks: torch.Tensor | None = None,
    ):
        if q.numel() == 0:
            return q.new_empty((0, layer.tp_q_head_num * layer.v_head_dim))

        if k is not None:
            assert v is not None

            is_cp_mode = (
                forward_batch.forward_mode.is_context_parallel_extend()
                and forward_batch.attn_cp_metadata is not None
                and self.attn_cp_size > 1
            )

            if save_kv_cache and not is_cp_mode:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                if not self.use_mla:
                    self.token_to_kv_pool.set_kv_buffer(
                        layer,
                        KVWriteLoc(cache_loc, self.forward_metadata.swa_out_cache_loc),
                        k,
                        v,
                        layer.k_scale,
                        layer.v_scale,
                    )
                else:
                    self.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        k_rope,
                    )
            if is_cp_mode:
                cp_allgather_and_save_kv_cache(
                    forward_batch,
                    layer,
                    k,
                    v,
                    self.attn_cp_size,
                    swa_loc=(
                        self.forward_metadata.swa_out_cache_loc
                        if self.use_sliding_window_kv_pool
                        else None
                    ),
                )

        metadata = self.forward_metadata

        local_metadata = None
        is_swa_layer = (
            layer.sliding_window_size is not None and layer.sliding_window_size > -1
        )
        window_size = (layer.sliding_window_size, 0) if is_swa_layer else (-1, -1)
        k_descale, v_descale = None, None
        if (
            self.kv_cache_dtype_str != "auto"
            and layer.head_dim <= 256
            and self.fa_impl_ver != 4
        ):
            if layer.k_scale is not None:
                descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
                k_descale = layer.k_scale.expand(descale_shape)
                v_descale = layer.v_scale.expand(descale_shape)
            q = q.to(self.kv_cache_dtype)
            q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None
            k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None
        causal = True
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            causal = False

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

        flash_attn_varlen_func = self.flash_attn_varlen_func
        flash_attn_with_kvcache = self.flash_attn_with_kvcache

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
                local_metadata=local_metadata,
            )
        else:
            scheduler_metadata = self._scheduler_metadata_for_kvcache(
                scheduler_metadata_owner,
                is_swa_layer=is_swa_layer,
            )
        num_splits = self._num_splits_for_no_mla_call(forward_batch)

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

                result = cp_attn_forward_extend(
                    self,
                    forward_batch,
                    q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    self.device,
                    _fa_cp_attn,
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
                    return_softmax_lse=forward_batch.mha_return_lse,
                    **kwargs,
                )
                if forward_batch.mha_return_lse:
                    output, lse, *_rest = output
                    lse = torch.transpose(lse, 0, 1).contiguous()
                    return (
                        output.view(-1, layer.tp_q_head_num * layer.v_head_dim),
                        lse,
                    )
                return output.view(-1, layer.tp_q_head_num * layer.v_head_dim)
            elif self._needs_no_mla_kvcache_scheduler_metadata(forward_batch):
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
                    return_softmax_lse=forward_batch.mha_return_lse,
                    **kwargs,
                )
                if forward_batch.mha_return_lse:
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
                    num_splits=self._num_splits_for_no_mla_full_kvcache(forward_batch),
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

                    assert forward_batch.mha_return_lse
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
                        return_softmax_lse=forward_batch.mha_return_lse,
                        **kwargs,
                    )
                if forward_batch.mha_return_lse:
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
                    scheduler_metadata=self._mla_scheduler_metadata_for_kvcache(
                        layer,
                        forward_batch,
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
                            scheduler_metadata=self._mla_scheduler_metadata_for_kvcache(
                                layer,
                                forward_batch,
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
        q_rope: torch.Tensor | None = None,
        k_rope: torch.Tensor | None = None,
        sinks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                if not self.use_mla:
                    self.token_to_kv_pool.set_kv_buffer(
                        layer,
                        KVWriteLoc(cache_loc, self.forward_metadata.swa_out_cache_loc),
                        k,
                        v,
                        layer.k_scale,
                        layer.v_scale,
                    )
                else:
                    self.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        cache_loc,
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
        window_size = (layer.sliding_window_size, 0) if is_swa_layer else (-1, -1)

        causal = True
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            causal = False

        kwargs = {}
        if sinks is not None:
            kwargs["sinks"] = sinks

        flash_attn_with_kvcache = self.flash_attn_with_kvcache

        k_descale, v_descale = None, None
        if self.kv_cache_dtype_str != "auto" and layer.head_dim <= 256:
            if layer.k_scale is not None:
                descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
                k_descale = layer.k_scale.expand(descale_shape)
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
                decode_num_splits = self._num_splits_for_no_mla_call(
                    forward_batch,
                    is_swa_layer=is_swa_layer,
                )
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
                _is_musa
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
                            scheduler_metadata=self._mla_scheduler_metadata_for_kvcache(
                                layer,
                                forward_batch,
                                force_update=True,
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
                    scheduler_metadata=self._mla_scheduler_metadata_for_kvcache(
                        layer,
                        forward_batch,
                    ),
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
                    scheduler_metadata=self._mla_scheduler_metadata_for_kvcache(
                        layer,
                        forward_batch,
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
        self.attn_backends = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                MusaFlashAttentionBackend(
                    model_runner,
                    speculative_step_id=i,
                    topk=self.topk,
                    speculative_num_steps=self.speculative_num_steps,
                    fa_impl_ver=fa_impl_ver,
                )
            )
