"""Per-request side states that live alongside a MambaPool slot.

Qwen4-Exp's PLE keeps two states per request — a short-conv window and an
N-gram token context — addressed by the request's mamba slot index. They are
registered on the owning ``MambaPool`` via ``register_slot_state`` so every
slot lifecycle event (deferred clear, radix COW, host offload round-trip)
carries them along and a slot never changes owner with a stale sibling row.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, List, Optional, Protocol, Tuple

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.utils import maybe_init_custom_mem_pool
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter


class SlotIndexedState(Protocol):
    """Per-request state owned by a MambaPool slot but not layer-major.

    Qwen4-Exp's PLE side states are addressed by the *mamba slot index*
    (`get_ngram_indices` / `get_short_conv_indices` both forward to
    `get_mamba_indices`), so anything that hands out, duplicates or persists a
    slot must carry them too. Implementers own their own shapes, and every
    method must be a no-op when the backing tensor is None so a disabled pool
    can register safely.
    """

    def reset_slots(self, indices: torch.Tensor) -> None: ...

    def copy_slots(self, src_index: torch.Tensor, dst_index: torch.Tensor) -> None: ...

    def get_cpu_slots(self, indices: torch.Tensor) -> Any: ...

    def load_cpu_slots(self, data: Any, indices: torch.Tensor) -> None: ...


class ShortConvPool:
    def __init__(
        self,
        *,
        size: int,
        state_shape: Optional[Tuple[int, int]],
        layer_ids: List[int],
        dtype: torch.dtype,
        device: str,
        spec_state_size: int = 0,
        enable_memory_saver: bool = False,
        speculative_num_draft_tokens: Optional[int] = None,
    ):
        self.size = size
        self.device = device
        self.layer_map = {layer_id: i for i, layer_id in enumerate(layer_ids)}
        self.conv_state = None
        self.intermediate_conv_state = None
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)
        )

        if not layer_ids or state_shape is None:
            return

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE), (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.enable_custom_mem_pool
            else nullcontext()
        ):
            self.conv_state = torch.zeros(
                size=(len(layer_ids), size + 1) + state_shape,
                dtype=dtype,
                device=device,
            )
            if speculative_num_draft_tokens is not None:
                self.intermediate_conv_state = torch.zeros(
                    size=(
                        len(layer_ids),
                        spec_state_size + 1,
                        speculative_num_draft_tokens,
                    )
                    + state_shape,
                    dtype=dtype,
                    device=device,
                )

    @property
    def enabled(self) -> bool:
        return self.conv_state is not None

    def layer_cache(self, layer_id: int) -> torch.Tensor:
        assert self.conv_state is not None
        assert layer_id in self.layer_map
        return self.conv_state[self.layer_map[layer_id]]

    def layer_intermediate_cache(self, layer_id: int) -> Optional[torch.Tensor]:
        if self.intermediate_conv_state is None:
            return None
        assert layer_id in self.layer_map
        return self.intermediate_conv_state[self.layer_map[layer_id]]

    def clear(self):
        if self.conv_state is not None:
            self.conv_state.zero_()

    # SlotIndexedState: slot is dim 1, behind the layer dim.

    def reset_slots(self, indices: torch.Tensor) -> None:
        if self.conv_state is not None and indices.numel() > 0:
            self.conv_state[:, indices] = 0

    def copy_slots(self, src_index: torch.Tensor, dst_index: torch.Tensor) -> None:
        if self.conv_state is not None:
            self.conv_state[:, dst_index] = self.conv_state[:, src_index]

    def get_cpu_slots(self, indices: torch.Tensor) -> Any:
        if self.conv_state is None:
            return None
        return self.conv_state[:, indices].to("cpu", non_blocking=True)

    def load_cpu_slots(self, data: Any, indices: torch.Tensor) -> None:
        if self.conv_state is None or data is None:
            return
        self.conv_state[:, indices] = data.to(self.conv_state.device, non_blocking=True)


class NGramPool:
    def __init__(
        self,
        *,
        size: int,
        context_len: int,
        eos_token_id: int,
        device: str,
        spec_state_size: int = 0,
        enable_memory_saver: bool = False,
        speculative_num_draft_tokens: Optional[int] = None,
    ):
        self.size = size
        self.context_len = context_len
        self.eos_token_id = eos_token_id
        self.device = device
        self.context = None
        self.intermediate_context = None
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)
        )
        if context_len <= 0:
            return

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE), (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.enable_custom_mem_pool
            else nullcontext()
        ):
            self.context = torch.full(
                (size + 1, context_len),
                eos_token_id,
                dtype=torch.long,
                device=device,
            )
            if speculative_num_draft_tokens is not None:
                self.intermediate_context = torch.full(
                    (spec_state_size + 1, speculative_num_draft_tokens, context_len),
                    eos_token_id,
                    dtype=torch.long,
                    device=device,
                )

    @property
    def enabled(self) -> bool:
        return self.context is not None

    def get_context(self, indices: torch.Tensor) -> torch.Tensor:
        assert self.context is not None
        return self.context.index_select(0, indices.to(dtype=torch.long))

    def set_context(self, indices: torch.Tensor, context: torch.Tensor):
        if self.context is not None and indices.numel() > 0:
            self.context[indices.to(dtype=torch.long)] = context.to(
                device=self.context.device, dtype=self.context.dtype
            )

    def set_intermediate_context(self, context: torch.Tensor):
        if self.intermediate_context is not None and context.numel() > 0:
            self.intermediate_context[: context.shape[0], : context.shape[1]].copy_(
                context.to(device=self.context.device, dtype=self.context.dtype)
            )

    def clear(self):
        if self.context is not None:
            self.context.fill_(self.eos_token_id)

    # SlotIndexedState: slot is dim 0, no layer dim.

    def reset_slots(self, indices: torch.Tensor) -> None:
        if self.context is not None and indices.numel() > 0:
            self.context[indices.to(dtype=torch.long)] = self.eos_token_id

    def copy_slots(self, src_index: torch.Tensor, dst_index: torch.Tensor) -> None:
        if self.context is not None:
            src = src_index.to(dtype=torch.long)
            dst = dst_index.to(dtype=torch.long)
            self.context[dst] = self.context[src]

    def get_cpu_slots(self, indices: torch.Tensor) -> Any:
        if self.context is None:
            return None
        return self.context[indices.to(dtype=torch.long)].to("cpu", non_blocking=True)

    def load_cpu_slots(self, data: Any, indices: torch.Tensor) -> None:
        if self.context is None or data is None:
            return
        self.context[indices.to(dtype=torch.long)] = data.to(
            self.context.device, non_blocking=True
        )
