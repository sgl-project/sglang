# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Paged FA4 attention for Nemotron Labs Diffusion."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
)


class NemotronLabsDiffusionAttentionBackend(FlashAttentionBackend):
    def __init__(self, model_runner: "ModelRunner") -> None:
        super().__init__(model_runner, fa_impl_ver=4)
        self.block_size = model_runner.decode_num_tokens_per_req()
        self.denoise_graph_metadata: dict[int, FlashAttentionMetadata] = {}

    def init_forward_metadata(self, forward_batch: "ForwardBatch") -> None:
        if not forward_batch.forward_mode.is_dllm_extend():
            return super().init_forward_metadata(forward_batch)
        self._update_denoise_metadata(
            forward_batch, self._make_denoise_metadata(forward_batch.batch_size)
        )

    def init_forward_metadata_out_graph(
        self, forward_batch: "ForwardBatch", in_capture: bool = False
    ) -> None:
        if not forward_batch.forward_mode.is_dllm_extend():
            return super().init_forward_metadata_out_graph(forward_batch, in_capture)

        bs = forward_batch.batch_size
        # Replays update metadata without replacing captured tensor addresses.
        if in_capture:
            self.denoise_graph_metadata[bs] = self._make_denoise_metadata(bs)
        self._update_denoise_metadata(forward_batch, self.denoise_graph_metadata[bs])

    def _make_denoise_metadata(self, batch_size: int) -> FlashAttentionMetadata:
        return FlashAttentionMetadata(
            cache_seqlens_int32=torch.zeros(
                batch_size, dtype=torch.int32, device=self.device
            ),
            max_seq_len_q=self.block_size,
            max_seq_len_k=self.max_context_len,
            cu_seqlens_q=torch.arange(
                batch_size + 1, dtype=torch.int32, device=self.device
            )
            * self.block_size,
            cu_seqlens_k=torch.zeros(
                batch_size + 1, dtype=torch.int32, device=self.device
            ),
            page_table=torch.zeros(
                (batch_size, self.max_num_pages), dtype=torch.int32, device=self.device
            ),
        )

    def _update_denoise_metadata(
        self, forward_batch: "ForwardBatch", metadata: FlashAttentionMetadata
    ) -> None:
        bs = forward_batch.batch_size
        self._set_decode_page_metadata(
            forward_batch.iteration_ids,
            metadata,
            forward_batch.req_pool_indices[:bs],
            forward_batch.seq_lens[:bs],
            seq_len_delta=0,
        )
        self.forward_metadata = metadata
        self.forward_metadata_spec_decode_expand = None
