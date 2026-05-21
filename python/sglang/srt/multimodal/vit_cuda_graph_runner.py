# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ViT CUDA Graph Runner with bucket-based capture and greedy bin-packing."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple

import numpy as np
import torch

from sglang.srt.layers.attention.vision import VisionAttentionMetadata


@dataclass
class ViTGraphConfig:
    """Model configuration needed for graph setup.

    Provided by the model via get_graph_config().
    """

    hidden_dim: int
    rotary_dim: int
    spatial_merge_unit: int  # spatial_merge_size^2, e.g. 4
    device: torch.device
    dtype: torch.dtype
    attn_backend: str  # "triton_attn", "fa3", "fa4", "flashinfer_cudnn", etc.
    elem_per_token: int = 0  # for FlashInfer element indptr computation


class ViTGraphCapturable(Protocol):
    """Protocol for ViT models supporting CUDA graph capture.

    Models implement these three methods. The runner calls them during
    capture and replay -- it never accesses model internals directly.
    """

    def get_graph_config(self) -> ViTGraphConfig: ...

    def run_blocks(
        self,
        x: torch.Tensor,
        forward_metadata: VisionAttentionMetadata,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run all transformer blocks.

        Returns:
            (block_output, deepstack_outputs)
        """
        ...

    def run_merger(
        self,
        x: torch.Tensor,
        deepstack_outs: List[torch.Tensor],
    ) -> torch.Tensor:
        """Run merger + optional deepstack concatenation."""
        ...


class ViTCudaGraphRunner:
    """Generic ViT CUDA Graph Runner with bucket-based capture.

    Eagerly captures CUDA graphs for a fixed set of bucket sizes at init.
    At runtime, finds the smallest bucket that fits the total token count,
    copies data into pre-allocated buffers, and replays the captured graph.
    Falls back to eager execution when total tokens exceed the max bucket.
    """

    BUCKET_SIZES = [32, 64, 128, 256, 512, 1024, 2048, 4096, 6144, 8192, 16384]
    MAX_IMAGES_PER_BUCKET = 16

    _graph_memory_pool = None

    def __init__(
        self,
        vit: ViTGraphCapturable,
        config: ViTGraphConfig,
    ) -> None:
        self.vit = vit
        self.config = config
        self.device = config.device
        self.dtype = config.dtype

        # Per-bucket-size state
        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self.input_bufs: Dict[int, torch.Tensor] = {}
        self.output_bufs: Dict[int, torch.Tensor] = {}
        self.forward_metadatas: Dict[int, VisionAttentionMetadata] = {}
        self.rotary_cos_bufs: Dict[int, torch.Tensor] = {}
        self.rotary_sin_bufs: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Bucket helpers
    # ------------------------------------------------------------------

    def find_bucket(self, total_tokens: int) -> Optional[int]:
        """Return the smallest bucket size >= total_tokens, or None."""
        idx = bisect.bisect_left(self.BUCKET_SIZES, total_tokens)
        if idx < len(self.BUCKET_SIZES):
            return self.BUCKET_SIZES[idx]
        return None

    # ------------------------------------------------------------------
    # Buffer allocation
    # ------------------------------------------------------------------

    def _alloc_buffers(self, bucket_size: int) -> None:
        """Allocate pre-capture buffers for a given bucket size."""
        B = bucket_size
        M = self.MAX_IMAGES_PER_BUCKET
        cfg = self.config

        # Input: [B, 1, hidden_dim]
        self.input_bufs[B] = torch.zeros(
            B, 1, cfg.hidden_dim, device=self.device, dtype=self.dtype
        )

        # Rotary embeddings: [B, rotary_dim]
        self.rotary_cos_bufs[B] = torch.zeros(
            B, cfg.rotary_dim, device=self.device, dtype=self.dtype
        )
        self.rotary_sin_bufs[B] = torch.zeros(
            B, cfg.rotary_dim, device=self.device, dtype=self.dtype
        )

        # Token-based cu_seqlens [M+1] and seq_lens [M] (all backends need these)
        cu_seqlens = torch.zeros(M + 1, device=self.device, dtype=torch.int32)
        seq_lens = torch.zeros(M, device=self.device, dtype=torch.int32)

        # FlashInfer-specific: separate packed_indptrs and sequence_lengths
        packed_indptrs = None
        sequence_lengths = None
        if cfg.attn_backend == "flashinfer_cudnn":
            packed_indptrs = torch.zeros(
                3 * (M + 1), device=self.device, dtype=torch.int32
            )
            sequence_lengths = torch.zeros(
                M, 1, 1, 1, device=self.device, dtype=torch.int32
            )

        # Fill dummy: single sequence of length B, rest zero-length
        self._fill_single_seq(cu_seqlens, seq_lens, B, packed_indptrs, sequence_lengths)

        self.forward_metadatas[B] = VisionAttentionMetadata(
            cu_seqlens=cu_seqlens,
            seq_lens=seq_lens,
            max_seqlen=B,
            packed_indptrs=packed_indptrs,
            sequence_lengths=sequence_lengths,
            flashinfer_max_seqlen=B if packed_indptrs is not None else None,
        )

    def _fill_single_seq(
        self,
        cu_seqlens: torch.Tensor,
        seq_lens: torch.Tensor,
        total_tokens: int,
        packed_indptrs: Optional[torch.Tensor],
        sequence_lengths: Optional[torch.Tensor],
    ) -> None:
        """Fill buffers for a single sequence of `total_tokens`, rest zero-length."""
        M = self.MAX_IMAGES_PER_BUCKET
        cfg = self.config

        # Token cu_seqlens: [0, total_tokens, total_tokens, ...]
        vals = np.zeros(M + 1, dtype=np.int32)
        vals[1:] = total_tokens
        cu_seqlens.copy_(torch.from_numpy(vals))

        # seq_lens: [total_tokens, 0, 0, ...]
        sl = np.zeros(M, dtype=np.int32)
        sl[0] = total_tokens
        seq_lens.copy_(torch.from_numpy(sl))

        if packed_indptrs is not None:
            elem = cfg.elem_per_token
            indptr = np.zeros(M + 1, dtype=np.int32)
            indptr[1:] = total_tokens * elem
            packed = np.concatenate([indptr, indptr, indptr])
            packed_indptrs.copy_(torch.from_numpy(packed))

        if sequence_lengths is not None:
            sl_4d = np.zeros(M, dtype=np.int32)
            sl_4d[0] = total_tokens
            sequence_lengths.copy_(torch.from_numpy(sl_4d).view(M, 1, 1, 1))

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def _capture(self, bucket_size: int) -> None:
        """Capture a CUDA graph for a single bucket size."""
        B = bucket_size
        self._alloc_buffers(B)

        input_buf = self.input_bufs[B]
        fwd_metadata = self.forward_metadatas[B]
        rotary_cos = self.rotary_cos_bufs[B]
        rotary_sin = self.rotary_sin_bufs[B]

        device_module = torch.get_device_module(self.device)

        if ViTCudaGraphRunner._graph_memory_pool is None:
            ViTCudaGraphRunner._graph_memory_pool = device_module.graph_pool_handle()

        # Warmup (2 runs to stabilize kernels)
        for _ in range(2):
            block_out, ds_outs = self.vit.run_blocks(
                input_buf, fwd_metadata, rotary_cos, rotary_sin
            )
            output = self.vit.run_merger(block_out, ds_outs)
        device_module.synchronize()

        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=ViTCudaGraphRunner._graph_memory_pool):
            block_out, ds_outs = self.vit.run_blocks(
                input_buf, fwd_metadata, rotary_cos, rotary_sin
            )
            output = self.vit.run_merger(block_out, ds_outs)

        self.graphs[B] = graph
        self.output_bufs[B] = output

    def capture_all(self) -> None:
        """Eagerly capture graphs for all bucket sizes. Called at model init."""
        for B in self.BUCKET_SIZES:
            self._capture(B)

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def _copy_metadata_to_buffers(
        self,
        bucket_size: int,
        forward_metadata: VisionAttentionMetadata,
    ) -> None:
        """Copy forward_metadata fields into the pre-allocated graph buffers."""
        B = bucket_size
        M = self.MAX_IMAGES_PER_BUCKET
        buf = self.forward_metadatas[B]

        src_cu = forward_metadata.cu_seqlens
        src_sl = forward_metadata.seq_lens
        N = src_sl.shape[0]  # number of real sequences

        # cu_seqlens: [N+1] -> [M+1], pad tail with last value, set [M]=B
        buf.cu_seqlens[: N + 1].copy_(src_cu[: N + 1])
        if N + 1 < M + 1:
            buf.cu_seqlens[N + 1 :].fill_(int(src_cu[N].item()))
        buf.cu_seqlens[M] = B  # last slot absorbs padding tokens

        # seq_lens: [N] -> [M], pad with 0
        buf.seq_lens[:N].copy_(src_sl[:N])
        if N < M:
            buf.seq_lens[N:].zero_()
            # last slot gets padding tokens
            S_total = int(src_cu[N].item())
            if B > S_total:
                buf.seq_lens[N] = B - S_total

        buf.max_seqlen = B

        # FlashInfer packed_indptrs
        if (
            buf.packed_indptrs is not None
            and forward_metadata.packed_indptrs is not None
        ):
            src_pi = forward_metadata.packed_indptrs
            src_len = src_pi.shape[0]
            buf_len = buf.packed_indptrs.shape[0]
            copy_len = min(src_len, buf_len)
            buf.packed_indptrs[:copy_len].copy_(src_pi[:copy_len])
            if copy_len < buf_len:
                buf.packed_indptrs[copy_len:].fill_(int(src_pi[-1].item()))

        if (
            buf.sequence_lengths is not None
            and forward_metadata.sequence_lengths is not None
        ):
            src_seql = forward_metadata.sequence_lengths.view(-1)
            buf_seql = buf.sequence_lengths.view(-1)
            copy_n = min(src_seql.shape[0], buf_seql.shape[0])
            buf_seql[:copy_n].copy_(src_seql[:copy_n])
            if copy_n < buf_seql.shape[0]:
                buf_seql[copy_n:].zero_()

        if buf.flashinfer_max_seqlen is not None:
            buf.flashinfer_max_seqlen = B

    def replay(
        self,
        bucket_size: int,
        x: torch.Tensor,
        forward_metadata: VisionAttentionMetadata,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
    ) -> torch.Tensor:
        """Replay a captured graph on a packed tensor.

        Args:
            bucket_size: the bucket to replay.
            x: [S, 1, hidden_dim] packed image tokens (already 3D).
            forward_metadata: pre-computed attention metadata.
            rotary_pos_emb_cos: [S, rotary_dim]
            rotary_pos_emb_sin: [S, rotary_dim]

        Returns:
            [S_merged, out_dim] packed merged output (only valid portion).
        """
        B = bucket_size
        S = x.shape[0]
        merge_unit = self.config.spatial_merge_unit

        # Copy metadata into graph buffers
        self._copy_metadata_to_buffers(B, forward_metadata)

        # Copy input (already 3D [S, 1, H])
        self.input_bufs[B][:S].copy_(x[:S])

        # Copy rotary embeddings
        self.rotary_cos_bufs[B][:S].copy_(rotary_pos_emb_cos)
        self.rotary_sin_bufs[B][:S].copy_(rotary_pos_emb_sin)

        # DEBUG: sync all TP ranks before replay to avoid NCCL desync
        torch.cuda.synchronize()
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()

        # Replay
        self.graphs[B].replay()

        # Extract valid output
        S_merged = S // merge_unit
        return self.output_bufs[B][:S_merged]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        x: torch.Tensor,
        forward_metadata: VisionAttentionMetadata,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
    ) -> torch.Tensor:
        """Run ViT forward with CUDA graph.

        If total tokens fit in a bucket, replay one graph directly.
        Otherwise fall back to eager.

        Args:
            x: [total_tokens, 1, hidden_dim] all images packed, 3D.
            forward_metadata: pre-computed VisionAttentionMetadata.
            rotary_pos_emb_cos: [total_tokens, rotary_dim]
            rotary_pos_emb_sin: [total_tokens, rotary_dim]

        Returns:
            [total_merged, out_dim] output.
        """
        total_tokens = x.shape[0]
        bucket = self.find_bucket(total_tokens)

        # Fast path: fits in a bucket
        if bucket is not None:
            return self.replay(
                bucket, x, forward_metadata, rotary_pos_emb_cos, rotary_pos_emb_sin
            )

        # Fallback: total tokens exceed max bucket, run eager
        block_out, ds_outs = self.vit.run_blocks(
            x, forward_metadata, rotary_pos_emb_cos, rotary_pos_emb_sin
        )
        return self.vit.run_merger(block_out, ds_outs)
