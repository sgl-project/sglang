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
from dataclasses import dataclass, field
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


@dataclass
class _Bin:
    """A single bin in the greedy bin-packing algorithm."""

    bucket_size: int
    remaining: int
    image_indices: List[int] = field(default_factory=list)


class ViTCudaGraphRunner:
    """Generic ViT CUDA Graph Runner with bucket-based capture and bin-packing replay.

    Eagerly captures CUDA graphs for a fixed set of bucket sizes at init.
    At runtime, packs multiple images into bins via greedy first-fit-decreasing,
    and replays the corresponding graph for each bin.
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

        M = self.MAX_IMAGES_PER_BUCKET

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
    # Bin-packing
    # ------------------------------------------------------------------

    def bin_pack(self, image_token_counts: List[int]) -> Tuple[List[_Bin], List[int]]:
        """Greedy first-fit-decreasing bin-packing.

        Args:
            image_token_counts: token count per image.

        Returns:
            (bins, eager_indices): bins for graph replay, and indices of
            images too large for any bucket (fall back to eager).
        """
        M = self.MAX_IMAGES_PER_BUCKET
        sorted_indices = sorted(
            range(len(image_token_counts)),
            key=lambda i: -image_token_counts[i],
        )

        bins: List[_Bin] = []
        eager_indices: List[int] = []

        for idx in sorted_indices:
            size = image_token_counts[idx]
            bucket = self.find_bucket(size)
            if bucket is None:
                eager_indices.append(idx)
                continue

            # First-fit
            placed = False
            for b in bins:
                if b.remaining >= size and len(b.image_indices) < M:
                    b.remaining -= size
                    b.image_indices.append(idx)
                    placed = True
                    break

            if not placed:
                bins.append(
                    _Bin(
                        bucket_size=bucket,
                        remaining=bucket - size,
                        image_indices=[idx],
                    )
                )

        return bins, eager_indices

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

    def _prepare_metadata(
        self,
        bucket_size: int,
        image_token_counts: List[int],
    ) -> None:
        """Fill all metadata buffers for a packed bin of images."""
        B = bucket_size
        M = self.MAX_IMAGES_PER_BUCKET
        cfg = self.config
        metadata = self.forward_metadatas[B]
        N = len(image_token_counts)

        # Cumulative token offsets
        offsets = np.zeros(N + 1, dtype=np.int32)
        for i, s in enumerate(image_token_counts):
            offsets[i + 1] = offsets[i] + s
        S_total = int(offsets[N])

        # Token cu_seqlens: [0, s1, s1+s2, ..., S, S, ..., B]
        cu_vals = np.full(M + 1, S_total, dtype=np.int32)
        cu_vals[0] = 0
        for i in range(N):
            cu_vals[i + 1] = offsets[i + 1]
        cu_vals[M] = B  # last sequence gets padding tokens
        metadata.cu_seqlens.copy_(torch.from_numpy(cu_vals))

        # seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        sl_vals = np.zeros(M, dtype=np.int32)
        for i in range(N):
            sl_vals[i] = image_token_counts[i]
        padding = B - S_total
        if padding > 0 and N < M:
            sl_vals[N] = padding  # padding sequence
        elif padding > 0:
            sl_vals[M - 1] = padding  # overflow to last slot
        metadata.seq_lens.copy_(torch.from_numpy(sl_vals))

        metadata.max_seqlen = B

        # FlashInfer packed_indptrs
        if metadata.packed_indptrs is not None:
            elem = cfg.elem_per_token
            indptr = np.full(M + 1, S_total * elem, dtype=np.int32)
            indptr[0] = 0
            for i in range(N):
                indptr[i + 1] = offsets[i + 1] * elem
            indptr[M] = B * elem
            packed = np.concatenate([indptr, indptr, indptr])
            metadata.packed_indptrs.copy_(torch.from_numpy(packed))

        if metadata.sequence_lengths is not None:
            fi_sl = np.zeros(M, dtype=np.int32)
            for i in range(N):
                fi_sl[i] = image_token_counts[i]
            if padding > 0 and N < M:
                fi_sl[N] = padding
            metadata.sequence_lengths.copy_(torch.from_numpy(fi_sl).view(M, 1, 1, 1))

        if metadata.flashinfer_max_seqlen is not None:
            metadata.flashinfer_max_seqlen = B

    def replay(
        self,
        bucket_size: int,
        x: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        token_cu_seqlens: np.ndarray,
    ) -> torch.Tensor:
        """Replay a captured graph on a packed tensor.

        Args:
            bucket_size: the bucket to replay.
            x: [S, 1, hidden_dim] packed image tokens (already 3D).
            rotary_pos_emb_cos: [S, rotary_dim]
            rotary_pos_emb_sin: [S, rotary_dim]
            token_cu_seqlens: numpy [N+1] token-based cumsum.

        Returns:
            [S_merged, out_dim] packed merged output (only valid portion).
        """
        B = bucket_size
        S = x.shape[0]
        merge_unit = self.config.spatial_merge_unit

        # Prepare metadata buffers from cu_seqlens
        image_token_counts = (token_cu_seqlens[1:] - token_cu_seqlens[:-1]).tolist()
        self._prepare_metadata(B, image_token_counts)

        # Copy input (already 3D [S, 1, H])
        self.input_bufs[B][:S].copy_(x[:S])
        if S < B:
            self.input_bufs[B][S:].zero_()

        # Copy rotary embeddings
        self.rotary_cos_bufs[B][:S].copy_(rotary_pos_emb_cos)
        self.rotary_sin_bufs[B][:S].copy_(rotary_pos_emb_sin)
        if S < B:
            self.rotary_cos_bufs[B][S:].zero_()
            self.rotary_sin_bufs[B][S:].zero_()

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
        token_cu_seqlens: np.ndarray,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
    ) -> torch.Tensor:
        """Run ViT forward with CUDA graph.

        If total tokens fit in a bucket, replay one graph directly on the
        packed tensor.  Otherwise bin-pack image groups into multiple replays.

        Args:
            x: [total_tokens, 1, hidden_dim] all images packed, 3D.
            token_cu_seqlens: numpy [N+1] token-based cumsum.
            rotary_pos_emb_cos: [total_tokens, rotary_dim]
            rotary_pos_emb_sin: [total_tokens, rotary_dim]

        Returns:
            [total_merged, out_dim] output.
        """
        total_tokens = x.shape[0]
        bucket = self.find_bucket(total_tokens)

        # Fast path: everything fits in one bucket
        if bucket is not None:
            return self.replay(
                bucket, x, rotary_pos_emb_cos, rotary_pos_emb_sin, token_cu_seqlens
            )

        # Slow path: total tokens exceed max bucket, bin-pack by image groups
        merge_unit = self.config.spatial_merge_unit
        image_token_counts = (token_cu_seqlens[1:] - token_cu_seqlens[:-1]).tolist()
        bins, eager_indices = self.bin_pack(image_token_counts)

        results: List[torch.Tensor] = []

        for b in bins:
            B = b.bucket_size
            indices = b.image_indices
            # Slice the packed tensor for this group
            group_slices = []
            group_cos = []
            group_sin = []
            group_offsets = [0]
            for i in indices:
                start = int(token_cu_seqlens[i])
                end = int(token_cu_seqlens[i + 1])
                group_slices.append(x[start:end])
                group_cos.append(rotary_pos_emb_cos[start:end])
                group_sin.append(rotary_pos_emb_sin[start:end])
                group_offsets.append(group_offsets[-1] + (end - start))
            group_x = torch.cat(group_slices, dim=0)
            group_cu = np.array(group_offsets, dtype=np.int32)
            merged = self.replay(
                B,
                group_x,
                torch.cat(group_cos, dim=0),
                torch.cat(group_sin, dim=0),
                group_cu,
            )
            results.append(merged)

        # Eager fallback for oversized images
        for idx in eager_indices:
            start = int(token_cu_seqlens[idx])
            end = int(token_cu_seqlens[idx + 1])
            img_x = x[start:end]
            S = img_x.shape[0]
            img_cu = np.array([0, S], dtype=np.int32)

            from sglang.srt.layers.attention.vision import (
                prepare_vision_attention_metadata,
            )

            metadata = prepare_vision_attention_metadata(
                torch.from_numpy(img_cu), device=self.device
            )
            cos = rotary_pos_emb_cos[start:end]
            sin = rotary_pos_emb_sin[start:end]
            block_out, ds_outs = self.vit.run_blocks(img_x, metadata, cos, sin)
            merged = self.vit.run_merger(block_out, ds_outs)
            results.append(merged)

        return torch.cat(results, dim=0)
