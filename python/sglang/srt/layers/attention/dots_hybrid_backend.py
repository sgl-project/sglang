"""Layer-wise DSA/SWA attention dispatch for dots.note.omni."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func
from sglang.srt.layers.attention.base_attn_backend import (
    AttentionBackend,
    SharedReadEnds,
)
from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.speculative.spec_info import SpecInput


def _normalize_page_table_rows(
    page_table: torch.Tensor, batch_size: int
) -> torch.Tensor:
    """Match Dots' pre-planned SWA table to the live DP-padded batch."""
    if page_table.shape[0] >= batch_size:
        return page_table[:batch_size]
    return torch.cat(
        [
            page_table,
            page_table.new_zeros(
                (batch_size - page_table.shape[0], page_table.shape[1])
            ),
        ],
        dim=0,
    )


def _normalize_cache_seqlens_rows(
    cache_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Preserve planned rows and fill only newly DP-padded dummy rows."""
    planned_bs = cache_seqlens.shape[0]
    if planned_bs >= batch_size:
        return cache_seqlens[:batch_size]

    dummy_seqlens = seq_lens[planned_bs:batch_size].to(
        device=cache_seqlens.device,
        dtype=cache_seqlens.dtype,
        non_blocking=True,
    )
    return torch.cat([cache_seqlens, dummy_seqlens], dim=0)


def _metadata_mismatches_dp_padded_batch(metadata, forward_batch) -> bool:
    """True when pre-planned attention metadata no longer matches the live batch.

    EAGLE plans draft metadata before ModelRunner runs DP/MLP padding. Dummy
    request rows then change ``batch_size`` / ``out_cache_loc``, leaving
    page tables and SWA write targets short. Rebuilding is required; slicing
    or zero-padding the stale tensors is not enough for DSA + SWA.
    """
    if metadata is None:
        return False
    bs = forward_batch.batch_size
    from sglang.srt.layers.attention.flashattention_backend import (
        FlashAttentionMetadata,
    )

    if isinstance(metadata, FlashAttentionMetadata):
        if metadata.page_table is not None and metadata.page_table.shape[0] != bs:
            return True
        if (
            metadata.swa_page_table is not None
            and metadata.swa_page_table.shape[0] != bs
        ):
            return True
        if (
            metadata.cache_seqlens_int32 is not None
            and metadata.cache_seqlens_int32.shape[0] != bs
        ):
            return True
        swa_loc = metadata.swa_out_cache_loc
        out_loc = forward_batch.out_cache_loc
        return (
            swa_loc is not None
            and out_loc is not None
            and swa_loc.shape[0] != out_loc.shape[0]
        )

    from sglang.srt.layers.attention.dsa_backend import DSAMetadata

    if isinstance(metadata, DSAMetadata):
        return metadata.cache_seqlens_int32.shape[0] != bs
    return False


def _dp_padding_changed_batch_size(forward_batch) -> bool:
    original_bs = forward_batch._original_batch_size
    return original_bs is not None and original_bs != forward_batch.batch_size


def _maybe_rebuild_dots_metadata(backend, forward_batch) -> None:
    """Eager-only rebuild when DP padding invalidated a Dots pre-plan."""
    from sglang.srt.model_executor.runner_utils.capture_mode import (
        get_is_capture_mode,
    )

    if get_is_capture_mode():
        return
    if backend._dp_rebuilt_batch_id == id(forward_batch):
        return
    stale = _metadata_mismatches_dp_padded_batch(
        backend.forward_metadata, forward_batch
    ) or _dp_padding_changed_batch_size(forward_batch)
    if not stale:
        return
    backend.init_forward_metadata(forward_batch)
    backend._dp_rebuilt_batch_id = id(forward_batch)


@dataclass
class DotsSWAMLAPrefillMetadata:
    kv_indices: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seq_len_q: int
    max_seq_len_k: int


class DotsSWAMLAAttnBackend(AttentionBackend):
    """Add Dots latent-cache SWA support around a FlashAttention backend."""

    def __init__(self, backend: AttentionBackend):
        self.backend = backend
        self._active_backend = backend
        self.token_to_kv_pool = backend.token_to_kv_pool
        self.req_to_token_pool = backend.req_to_token_pool
        self.needs_cpu_seq_lens = True
        self._prefill_metadata: DotsSWAMLAPrefillMetadata | None = None
        self._dp_rebuilt_batch_id: int | None = None

    @property
    def forward_metadata(self):
        return self._active_backend.forward_metadata

    @forward_metadata.setter
    def forward_metadata(self, value):
        self._active_backend.forward_metadata = value

    @property
    def verify_mask(self):
        return self.backend.verify_mask

    def shared_read_ends(self, fm: ForwardMode) -> SharedReadEnds:
        return self.backend.shared_read_ends(fm)

    def draft_extend_metadata_captured_in_graph(self) -> bool:
        return self.backend.draft_extend_metadata_captured_in_graph()

    def selected_backend(self, forward_batch: ForwardBatch) -> AttentionBackend:
        return (
            self.backend._select_backend(forward_batch.forward_mode)
            if isinstance(self.backend, HybridAttnBackend)
            else self.backend
        )

    def uses_flash_attention(self, forward_batch: ForwardBatch) -> bool:
        from sglang.srt.layers.attention.flashattention_backend import (
            FlashAttentionBackend,
        )

        return isinstance(self.selected_backend(forward_batch), FlashAttentionBackend)

    def maybe_rebuild_metadata_after_dp_padding(
        self, forward_batch: ForwardBatch
    ) -> None:
        """Rebuild FA + SWA-prefill metadata after eager DP dummy-row padding."""
        self._active_backend = self.selected_backend(forward_batch)
        _maybe_rebuild_dots_metadata(self, forward_batch)

    def select_draft_step_out_cache_loc(self, forward_batch: ForwardBatch):
        """Return this draft step's write locations from a combined SWA buffer."""
        from sglang.srt.layers.attention.flashattention_backend import (
            FlashAttentionBackend,
        )

        out_cache_loc = forward_batch.out_cache_loc
        backend = self._active_backend
        if not isinstance(backend, FlashAttentionBackend):
            return out_cache_loc
        if (
            out_cache_loc is not None
            and forward_batch.forward_mode.is_decode_or_idle()
            and forward_batch.spec_info is not None
            and backend.speculative_num_steps > 0
            and out_cache_loc.numel()
            == forward_batch.batch_size * backend.topk * backend.speculative_num_steps
        ):
            return out_cache_loc.view(
                forward_batch.batch_size,
                backend.topk,
                backend.speculative_num_steps,
            )[:, :, backend.speculative_step_id].reshape(-1)
        return out_cache_loc

    @contextmanager
    def _use_draft_step_out_cache_loc(self, forward_batch: ForwardBatch):
        """Expose only this backend's draft-step write locations to FA."""
        original = forward_batch.out_cache_loc
        forward_batch.out_cache_loc = self.select_draft_step_out_cache_loc(
            forward_batch
        )
        try:
            yield
        finally:
            forward_batch.out_cache_loc = original

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self._active_backend = self.selected_backend(forward_batch)
        with self._use_draft_step_out_cache_loc(forward_batch):
            self.backend.init_forward_metadata(forward_batch)
        self._init_prefill_metadata(forward_batch)

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        self._active_backend = self.selected_backend(forward_batch)
        with self._use_draft_step_out_cache_loc(forward_batch):
            self.backend.init_forward_metadata_out_graph(
                forward_batch, in_capture=in_capture
            )
        self._init_prefill_metadata(forward_batch)

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        self.backend.init_forward_metadata_in_graph(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.backend.init_cuda_graph_state(max_bs, max_num_tokens)

    def get_cuda_graph_seq_len_fill_value(self):
        return self.backend.get_cuda_graph_seq_len_fill_value()

    def on_after_cuda_graph_warmup(self):
        self.backend.on_after_cuda_graph_warmup()

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: int | None
    ):
        return self.backend.update_verify_buffers_to_fill_after_draft(
            spec_info, cuda_graph_bs
        )

    def forward(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        self.maybe_rebuild_metadata_after_dp_padding(forward_batch)
        return self.backend.forward(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def forward_extend(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
    ):
        return self.backend.forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def forward_decode(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
    ):
        return self.backend.forward_decode(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def init_mha_chunk_metadata(self, forward_batch: ForwardBatch):
        self.backend.init_mha_chunk_metadata(forward_batch)

    def _init_prefill_metadata(self, forward_batch: ForwardBatch):
        if not forward_batch.forward_mode.is_extend_without_speculative():
            self._prefill_metadata = None
            return

        metadata = self._active_backend.forward_metadata
        assert forward_batch.seq_lens_cpu is not None
        batch_kv_indices = self._active_backend.req_to_token[
            forward_batch.req_pool_indices, :
        ]
        sliced_indices = []
        kv_lens = []
        for i in range(forward_batch.batch_size):
            q_len = int(forward_batch.extend_seq_lens_cpu[i])
            kv_len = int(forward_batch.seq_lens_cpu[i])
            tail_len = min(q_len + self._active_backend.sliding_window_size, kv_len)
            sliced_indices.append(batch_kv_indices[i, kv_len - tail_len : kv_len])
            kv_lens.append(tail_len)

        full_kv_indices = torch.cat(sliced_indices)
        kv_indices = self.token_to_kv_pool.translate_loc_from_full_to_swa(
            full_kv_indices
        ).to(torch.int32)
        lens_cpu = torch.tensor([0, *kv_lens], dtype=torch.int32, pin_memory=True)
        self._prefill_metadata = DotsSWAMLAPrefillMetadata(
            kv_indices=kv_indices,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k=torch.cumsum(
                lens_cpu.to(device=forward_batch.seq_lens.device, non_blocking=True),
                dim=0,
                dtype=torch.int32,
            ),
            max_seq_len_q=metadata.max_seq_len_q,
            max_seq_len_k=max(kv_lens),
        )

    def get_swa_mla_prefill_latent_cache(
        self, forward_batch: ForwardBatch, layer_id: int
    ):
        assert self._prefill_metadata is not None
        return self.token_to_kv_pool.get_key_buffer(layer_id)[
            self._prefill_metadata.kv_indices
        ]

    def forward_swa_mla_expanded(self, q, k, v, layer, forward_batch=None):
        """Run dense SWA after Dots expands its compact MLA cache."""
        metadata = self._prefill_metadata
        assert metadata is not None
        q = q.view(-1, layer.tp_q_head_num, layer.head_dim)
        k = k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype)
        v = v.view(-1, layer.tp_k_head_num, layer.v_head_dim).to(q.dtype)

        # FA3 requires equal QK/V widths when QK exceeds 192.
        pad_v_to_qk = layer.head_dim > 192 and layer.v_head_dim != layer.head_dim
        if pad_v_to_qk:
            v = torch.nn.functional.pad(v, (0, layer.head_dim - layer.v_head_dim))

        output = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k=metadata.cu_seqlens_k,
            max_seqlen_q=metadata.max_seq_len_q,
            max_seqlen_k=metadata.max_seq_len_k,
            softmax_scale=layer.scaling,
            causal=True,
            window_size=(layer.sliding_window_size, 0),
            ver=self._active_backend.fa_impl_ver,
        )
        if pad_v_to_qk:
            output = output[..., : layer.v_head_dim]
        return output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_swa_mla_absorbed(self, q, layer, forward_batch):
        """Run decode directly against the page64 latent SWA cache."""
        from sglang.srt.layers.attention.swa_mla_fallback.forward import (
            forward_dense_kvlora_swa_torch_fallback,
        )

        backend = self.selected_backend(forward_batch)
        if backend.page_size != 64:
            raise RuntimeError(
                "Dots SWA latent decode requires page_size=64, "
                f"got {backend.page_size}."
            )

        self.maybe_rebuild_metadata_after_dp_padding(forward_batch)
        metadata = backend.forward_metadata
        block_table = metadata.swa_page_table
        if block_table is None:
            raise RuntimeError("Dots SWA latent decode requires an SWA page table.")
        bs = forward_batch.batch_size
        block_table = _normalize_page_table_rows(block_table, bs)
        cache_seqlens = _normalize_cache_seqlens_rows(
            metadata.cache_seqlens_int32,
            forward_batch.seq_lens,
            bs,
        )
        reshape_q = q.view(bs, -1, layer.tp_q_head_num, layer.head_dim)
        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        output = forward_dense_kvlora_swa_torch_fallback(
            reshape_q=reshape_q,
            k_cache=k_cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            layer=layer,
            kv_cache_dim=layer.head_dim,
            head_dim_v=layer.v_head_dim,
            window_size=layer.sliding_window_size + 1,
        )
        return output.view(-1, layer.tp_q_head_num * layer.v_head_dim)


class DotsHybridAttnBackend(AttentionBackend):
    def __init__(
        self,
        dsa_backend: AttentionBackend,
        swa_backend: AttentionBackend,
    ):
        self.dsa_backend = dsa_backend
        # Keep DSA on its radix-aware MLA path.
        self.dsa_backend.supports_mha_one_shot = False
        self.swa_backend = swa_backend
        self.token_to_kv_pool = swa_backend.token_to_kv_pool
        self.req_to_token_pool = swa_backend.req_to_token_pool
        # SWA latent expansion uses host sequence-length mirrors.
        self.needs_cpu_seq_lens = True
        self._dp_rebuilt_batch_id: int | None = None

    @staticmethod
    def _is_swa_layer(layer: RadixAttention) -> bool:
        return layer.sliding_window_size is not None and layer.sliding_window_size > -1

    def backend_for_layer(self, layer: RadixAttention) -> AttentionBackend:
        return self.swa_backend if self._is_swa_layer(layer) else self.dsa_backend

    def selected_swa_backend(self, forward_batch: ForwardBatch) -> AttentionBackend:
        return (
            self.swa_backend._select_backend(forward_batch.forward_mode)
            if isinstance(self.swa_backend, HybridAttnBackend)
            else self.swa_backend
        )

    def maybe_rebuild_metadata_after_dp_padding(
        self, forward_batch: ForwardBatch
    ) -> None:
        """Rebuild both DSA and SWA plans after eager DP dummy-row padding."""
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            get_is_capture_mode,
        )

        if get_is_capture_mode():
            return
        if self._dp_rebuilt_batch_id == id(forward_batch):
            return
        dsa_stale = _metadata_mismatches_dp_padded_batch(
            self.dsa_backend.forward_metadata, forward_batch
        )
        swa_backend = self.selected_swa_backend(forward_batch)
        swa_stale = _metadata_mismatches_dp_padded_batch(
            swa_backend.forward_metadata, forward_batch
        )
        if dsa_stale or swa_stale or _dp_padding_changed_batch_size(forward_batch):
            self.init_forward_metadata(forward_batch)
            self._dp_rebuilt_batch_id = id(forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self.dsa_backend.init_forward_metadata(forward_batch)
        self.swa_backend.init_forward_metadata(forward_batch)

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        self.dsa_backend.init_forward_metadata_out_graph(
            forward_batch, in_capture=in_capture
        )
        self.swa_backend.init_forward_metadata_out_graph(
            forward_batch, in_capture=in_capture
        )

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        self.dsa_backend.init_forward_metadata_in_graph(forward_batch)
        self.swa_backend.init_forward_metadata_in_graph(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.dsa_backend.init_cuda_graph_state(max_bs, max_num_tokens)
        self.swa_backend.init_cuda_graph_state(max_bs, max_num_tokens)

    def get_cuda_graph_seq_len_fill_value(self):
        return self.swa_backend.get_cuda_graph_seq_len_fill_value()

    def on_after_cuda_graph_warmup(self):
        self.dsa_backend.on_after_cuda_graph_warmup()
        self.swa_backend.on_after_cuda_graph_warmup()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        self.maybe_rebuild_metadata_after_dp_padding(forward_batch)
        return self.backend_for_layer(layer).forward(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def forward_extend(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
    ):
        return self.backend_for_layer(layer).forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def forward_decode(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
    ):
        return self.backend_for_layer(layer).forward_decode(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def get_indexer_metadata(self, layer_id: int, forward_batch: ForwardBatch):
        return self.dsa_backend.get_indexer_metadata(layer_id, forward_batch)

    def get_swa_mla_prefill_latent_cache(
        self, forward_batch: ForwardBatch, layer_id: int
    ):
        backend = self.selected_swa_backend(forward_batch)
        return backend.get_swa_mla_prefill_latent_cache(forward_batch, layer_id)

    def forward_swa_mla_expanded(self, q, k, v, layer, forward_batch):
        backend = self.selected_swa_backend(forward_batch)
        return backend.forward_swa_mla_expanded(q, k, v, layer, forward_batch)

    def forward_swa_mla_absorbed(self, q, layer, forward_batch):
        backend = self.selected_swa_backend(forward_batch)
        return backend.forward_swa_mla_absorbed(q, layer, forward_batch)

    def init_mha_chunk_metadata(self, forward_batch: ForwardBatch):
        backend = self.selected_swa_backend(forward_batch)
        backend.init_mha_chunk_metadata(forward_batch)


def _wrap_dots_swa_backend(backend: AttentionBackend) -> AttentionBackend:
    """Add latent-cache SWA behavior when a backend uses FlashAttention."""
    from sglang.srt.layers.attention.flashattention_backend import (
        FlashAttentionBackend,
    )

    if isinstance(backend, FlashAttentionBackend) or (
        isinstance(backend, HybridAttnBackend)
        and (
            isinstance(backend.prefill_backend, FlashAttentionBackend)
            or isinstance(backend.decode_backend, FlashAttentionBackend)
        )
    ):
        return DotsSWAMLAAttnBackend(backend)
    return backend


def wrap_dots_draft_decode_backend(backend: AttentionBackend) -> AttentionBackend:
    """Wrap each per-step backend used by the Dots NextN draft container."""
    backend.attn_backends = [
        _wrap_dots_swa_backend(child) for child in backend.attn_backends
    ]
    return backend


def wrap_dots_attention_backend(runner, full_attn_backend: AttentionBackend):
    """Construct the Dots target or draft attention backend."""
    if runner.model_config.is_draft_model:
        return _wrap_dots_swa_backend(full_attn_backend)

    if runner.model_config.hf_text_config.index_topk is None:
        return DotsSWAMLAAttnBackend(full_attn_backend)

    from sglang.srt.layers.attention.attention_registry import create_dsa_backend

    swa_backend = (
        full_attn_backend.prefill_backend
        if isinstance(full_attn_backend, HybridAttnBackend)
        else full_attn_backend
    )
    return DotsHybridAttnBackend(
        dsa_backend=create_dsa_backend(runner),
        swa_backend=DotsSWAMLAAttnBackend(swa_backend),
    )
