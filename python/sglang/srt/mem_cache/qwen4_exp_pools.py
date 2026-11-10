"""Qwen4-Exp per-request pools: PLE n-gram history and short-conv state.

Qwen4-Exp carries two per-request side states that upstream's hybrid pools know
nothing about: the PLE n-gram context (the host-offloaded n-gram table's
per-request window) and the short-convolution state. Both have exactly the
lifetime of a mamba slot, so they have to be cleared, copied and
host-round-tripped in lockstep with it -- otherwise a recycled slot inherits the
previous request's history, which is silent corruption rather than a crash.

These live here rather than in ``memory_pool.py`` because that file is upstream's
and the cache engine was adopted wholesale in the re-land; the two class-level
hooks upstream provides (``HybridReqToTokenPool.mamba_pool_cls`` and the
``_init_mamba_pool`` override point) are enough to attach everything from
outside.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, MambaPool
from sglang.srt.platforms import current_platform


class Qwen4ExpMambaPool(MambaPool):
    """Mamba pool that drags registered side states through the slot lifecycle."""

    def __init__(self, *args, **kwargs):
        # Registered before super().__init__ because the base constructor can
        # reach clear_slots() before we would otherwise have the attribute.
        self._slot_siblings: List = []
        super().__init__(*args, **kwargs)

    def register_slot_state(self, state) -> None:
        """Attach a state that rides along on clear / copy / host round-trip."""
        self._slot_siblings.append(state)

    def clear_slots(self, indices: torch.Tensor):
        for sibling in self._slot_siblings:
            sibling.reset_slots(indices)
        super().clear_slots(indices)

    def copy_slots(self, src_indices: torch.Tensor, dst_indices: torch.Tensor):
        super().copy_slots(src_indices, dst_indices)
        for sibling in self._slot_siblings:
            sibling.copy_slots(src_indices, dst_indices)

    def get_cpu_copy(self, indices):
        base = super().get_cpu_copy(indices)
        if not self._slot_siblings:
            return base
        siblings_cpu = [s.get_cpu_slots(indices) for s in self._slot_siblings]
        current_platform.synchronize()
        # Upstream already returns a 3-tuple when the ReplaySSM spec-verify ring
        # is active, so the siblings take a fixed 4th slot: a 3-tuple must keep
        # meaning "cursors" on every path, never "siblings".
        cursors_cpu = base[2] if len(base) == 3 else None
        return base[0], base[1], cursors_cpu, siblings_cpu

    def load_cpu_copy(self, mamba_cache_cpu, indices):
        siblings_cpu = None
        if len(mamba_cache_cpu) == 4:
            conv_cpu, temporal_cpu, cursors_cpu, siblings_cpu = mamba_cache_cpu
            mamba_cache_cpu = (
                (conv_cpu, temporal_cpu, cursors_cpu)
                if cursors_cpu is not None
                else (conv_cpu, temporal_cpu)
            )
        super().load_cpu_copy(mamba_cache_cpu, indices)
        if siblings_cpu is not None:
            for sibling, data in zip(self._slot_siblings, siblings_cpu):
                sibling.load_cpu_slots(data, indices)
            current_platform.synchronize()


class Qwen4ExpReqToTokenPool(HybridReqToTokenPool):
    """Hybrid pool plus Qwen4-Exp's PLE n-gram and short-conv side states."""

    mamba_pool_cls = Qwen4ExpMambaPool

    def __init__(
        self,
        *,
        short_conv_layer_ids: Optional[List[int]] = None,
        short_conv_state_shape: Optional[Tuple[int, int]] = None,
        ngram_context_len: int = 0,
        ngram_eos_token_id: int = 0,
        **kwargs,
    ):
        # Stashed for _init_mamba_pool, which the base constructor calls.
        self._short_conv_layer_ids = short_conv_layer_ids or []
        self._short_conv_state_shape = short_conv_state_shape
        self._ngram_context_len = ngram_context_len
        self._ngram_eos_token_id = ngram_eos_token_id
        super().__init__(**kwargs)

    def _init_mamba_pool(self, **kwargs):
        super()._init_mamba_pool(**kwargs)

        # Callers that do not pass their config (e.g. the multi-layer draft
        # clone) get disabled pools, which register as no-ops rather than being
        # absent, so the attribute always exists.
        from sglang.srt.mem_cache.ple_state_pool import NGramPool, ShortConvPool

        self.short_conv_pool = ShortConvPool(
            size=kwargs["mamba_size"],
            spec_state_size=kwargs["mamba_spec_state_size"],
            state_shape=self._short_conv_state_shape,
            layer_ids=self._short_conv_layer_ids,
            dtype=kwargs["cache_params"].dtype.conv,
            device=kwargs["device"],
            enable_memory_saver=self.enable_memory_saver,
            speculative_num_draft_tokens=kwargs.get("speculative_num_draft_tokens"),
        )
        self.ple_window_cache = None
        self.ngram_pool = NGramPool(
            size=kwargs["mamba_size"],
            spec_state_size=kwargs["mamba_spec_state_size"],
            context_len=self._ngram_context_len,
            eos_token_id=self._ngram_eos_token_id,
            device=kwargs["device"],
            enable_memory_saver=self.enable_memory_saver,
            speculative_num_draft_tokens=kwargs.get("speculative_num_draft_tokens"),
        )
        # Disabled pools stay off the sibling list so the host-offload payload
        # keeps its legacy shape for every non-PLE hybrid model.
        if self.short_conv_pool.enabled:
            self.mamba_pool.register_slot_state(self.short_conv_pool)
        if self.ngram_pool.enabled:
            self.mamba_pool.register_slot_state(self.ngram_pool)

    # ===================== short conv =====================

    def get_short_conv_indices(self, req_indices: torch.Tensor) -> torch.Tensor:
        return self.get_mamba_indices(req_indices)

    def short_conv_layer_cache(self, layer_id: int) -> torch.Tensor:
        assert layer_id in self.short_conv_pool.layer_map
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.short_conv_pool.layer_cache(layer_id)

    def short_conv_layer_intermediate_cache(
        self, layer_id: int
    ) -> Optional[torch.Tensor]:
        return self.short_conv_pool.layer_intermediate_cache(layer_id)

    # ===================== PLE n-gram =====================

    def get_ngram_indices(self, req_indices: torch.Tensor) -> torch.Tensor:
        return self.get_mamba_indices(req_indices)

    def get_ngram_context(self, ngram_indices: torch.Tensor) -> torch.Tensor:
        return self.ngram_pool.get_context(ngram_indices)

    def set_ngram_context(
        self, ngram_indices: torch.Tensor, context: torch.Tensor
    ) -> None:
        self.ngram_pool.set_context(ngram_indices, context)

    def set_ngram_intermediate_context(
        self, context: torch.Tensor, indices: Optional[torch.Tensor] = None
    ) -> None:
        self.ngram_pool.set_intermediate_context(context, indices)

    def clear(self):
        super().clear()
        self.short_conv_pool.clear()
        self.ngram_pool.clear()
