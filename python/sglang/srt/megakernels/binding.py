# SPDX-License-Identifier: Apache-2.0
"""The data SGLang hands a megakernel's build, forward and draft entries.

The one place a megakernel reads SGLang's runtime state: the KV pool, the attention
metadata of the running forward and the Engram tables. A spec's build gets a
``MegakernelBinding`` as ``BuildContext.binding``; a covered forward gets the
running forward's ``ForwardView`` as ``Step.view`` (a draft entry asks
``binding.forward_view()``). Everything else a megakernel touches is the model it
replaces, SGLang's kernel ops, and the reads listed below.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch

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
        current = {"kv_pool": self._current_pool, "engram_history": self._history}
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

    def _history(self) -> Optional[torch.Tensor]:
        # only models with Engram layers have a hasher
        hasher = getattr(self._model, "engram_hasher", None)
        return None if hasher is None else hasher.history


def _identity(value: Any) -> tuple:
    if isinstance(value, torch.Tensor):
        return (value.data_ptr(), tuple(value.shape), value.dtype)
    return (id(value),)
