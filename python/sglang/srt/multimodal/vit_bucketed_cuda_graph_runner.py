# Copyright 2023-2026 SGLang Team
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
"""Token-bucketed CUDA graphs for ViT encoders with varlen attention.

One graph is captured per token bucket. A batch of images is padded up to the
smallest bucket that holds it and replayed once; the padding rows form their
own attention sequence, so they never mix with real tokens. ``cu_seqlens`` has
a fixed length per bucket (unused entries repeat the total, i.e. empty
sequences), so any mix of image sizes and counts that fits the bucket reuses
the same graph. Batches above the largest bucket run eagerly.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from sglang.srt.layers.attention.vision import VisionAttentionMetadata
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

# FA3's varlen kernel schedules tiles from the device-side cu_seqlens, so empty
# padding sequences and an upper-bound max_seqlen cost almost nothing.
SUPPORTED_ATTENTION_BACKENDS = ("fa3",)

# Above ~4K tokens the encoder is GPU-bound on H100, so launch savings no
# longer pay for the padding; those batches run eagerly.
DEFAULT_TOKEN_BUCKETS = (
    256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096,
)  # fmt: skip

_PAD_SEQ_LEN = 64


def parse_token_buckets(spec: str, multiple: int) -> list[int]:
    if not spec.strip():
        buckets = list(DEFAULT_TOKEN_BUCKETS)
    else:
        buckets = [int(s) for s in spec.split(",") if s.strip()]
    if any(b <= 0 or b % multiple for b in buckets):
        raise ValueError(
            f"ViT CUDA graph token buckets must be positive multiples of {multiple}, "
            f"got {buckets}"
        )
    return sorted(set(buckets))


@dataclass
class _BucketGraph:
    graph: torch.cuda.CUDAGraph
    x: torch.Tensor  # [bucket, 1, hidden]
    cos: torch.Tensor  # [bucket, rot_dim]
    sin: torch.Tensor
    # cu_seqlens (max_seqs + 2,) followed by seq_lens (max_seqs + 1,)
    seqlen_buf: torch.Tensor
    output: torch.Tensor  # [bucket // merge_unit, out_hidden]


# (x [S, 1, H], cu_seqlens, cos, sin, metadata) -> [S // merge_unit, out_hidden]
BlocksFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, VisionAttentionMetadata],
    torch.Tensor,
]


class ViTBucketedCudaGraphRunner:
    """Replay the transformer blocks and mergers of a ViT in token buckets."""

    def __init__(
        self,
        vit: nn.Module,
        blocks_fn: BlocksFn,
        merge_unit: int,
        token_buckets: Sequence[int],
    ) -> None:
        self.vit = vit
        self.blocks_fn = blocks_fn
        # Every sequence holds a whole number of merge units, and the merger
        # groups rows by merge unit, so buckets must be multiples of it.
        self.merge_unit = merge_unit
        self.token_buckets = list(token_buckets)
        self.graphs: dict[int, _BucketGraph] = {}
        self.pool = None
        # One capture stream: cuBLAS keeps a workspace per stream.
        self.stream: Optional[torch.cuda.Stream] = None

    @staticmethod
    def supports_backend(backend: Optional[str]) -> bool:
        return backend in SUPPORTED_ATTENTION_BACKENDS

    def _bucket_for(self, num_tokens: int) -> Optional[int]:
        for bucket in self.token_buckets:
            if bucket >= num_tokens:
                return bucket
        return None

    def _capture_context(self):
        # Same rule as ViTCudaGraphRunner: a DP-sharded encoder is rank-local.
        if getattr(self.vit, "use_data_parallel", False):
            return nullcontext()
        ca_comm = get_parallel().tp_group.ca_comm
        return ca_comm.capture() if ca_comm is not None else nullcontext()

    @staticmethod
    def _metadata(seqlen_buf: torch.Tensor, max_seqs: int, bucket: int):
        return VisionAttentionMetadata(
            cu_seqlens=seqlen_buf[: max_seqs + 2],
            seq_lens=seqlen_buf[max_seqs + 2 :],
            # Upper bound; the kernel reads the real lengths from cu_seqlens.
            max_seqlen=bucket,
        )

    @torch.no_grad()
    def _capture(
        self, bucket: int, hidden: torch.Tensor, cos: torch.Tensor
    ) -> _BucketGraph:
        device = hidden.device
        max_seqs = bucket // self.merge_unit
        x = torch.zeros(bucket, 1, hidden.shape[-1], dtype=hidden.dtype, device=device)
        cos_buf = torch.zeros(bucket, cos.shape[-1], dtype=cos.dtype, device=device)
        sin_buf = torch.zeros_like(cos_buf)
        seqlen_buf = torch.from_numpy(self._seqlens(np.array([0, bucket]), bucket)).to(
            device
        )
        metadata = self._metadata(seqlen_buf, max_seqs, bucket)

        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
            self.stream = torch.cuda.Stream(device=device)
        stream = self.stream
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            # Warm up outside the capture: JIT/compiled kernels and cuBLAS
            # workspaces must exist before capture starts.
            for _ in range(2):
                self.blocks_fn(
                    x.clone(), metadata.cu_seqlens, cos_buf, sin_buf, metadata
                )
        torch.cuda.current_stream().wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with (
            self._capture_context(),
            torch.cuda.graph(
                graph, pool=self.pool, stream=stream, capture_error_mode="thread_local"
            ),
        ):
            output = self.blocks_fn(x, metadata.cu_seqlens, cos_buf, sin_buf, metadata)
        return _BucketGraph(graph, x, cos_buf, sin_buf, seqlen_buf, output)

    def _capture_all(self, hidden: torch.Tensor, cos: torch.Tensor) -> None:
        # Capture every bucket once, largest first, so the smaller graphs
        # reuse the shared pool and serving never stalls on a new shape.
        before = torch.cuda.memory_reserved(hidden.device)
        for bucket in reversed(self.token_buckets):
            self.graphs[bucket] = self._capture(bucket, hidden, cos)
        torch.cuda.synchronize(hidden.device)
        # Drop the warmup activations so the log shows what the graphs keep.
        torch.cuda.empty_cache()
        logger.info(
            "Captured %d ViT CUDA graphs for up to %d tokens (%.0f MiB)",
            len(self.graphs),
            self.token_buckets[-1],
            (torch.cuda.memory_reserved(hidden.device) - before) / 2**20,
        )

    def _seqlens(self, cu_seqlens: np.ndarray, bucket: int) -> np.ndarray:
        """Pad cu_seqlens to the bucket layout and append seq_lens.

        Padding rows are split into short sequences so their attention cost
        stays linear in the padding length. Real sequences hold at least one
        merge unit each, so real plus padding sequences never exceed
        bucket // merge_unit; the remaining entries are empty sequences.
        """
        max_seqs = bucket // self.merge_unit
        num_tokens = int(cu_seqlens[-1])
        pad_bounds = np.arange(
            num_tokens + _PAD_SEQ_LEN, bucket, _PAD_SEQ_LEN, dtype=np.int32
        )
        cu = np.full(max_seqs + 2, bucket, dtype=np.int32)
        used = len(cu_seqlens) + len(pad_bounds)
        cu[: len(cu_seqlens)] = cu_seqlens
        cu[len(cu_seqlens) : used] = pad_bounds
        return np.concatenate([cu, np.diff(cu)]).astype(np.int32)

    def run(
        self,
        hidden: torch.Tensor,  # [S, H], patch embeddings + position embeddings
        cu_seqlens: np.ndarray,  # (num_seqs + 1,) int32, host
        rotary_pos_emb_cos: torch.Tensor,  # [S, rot_dim]
        rotary_pos_emb_sin: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Return the encoder output, or None if the batch exceeds every bucket."""
        num_tokens = hidden.shape[0]
        bucket = self._bucket_for(num_tokens)
        if bucket is None:
            return None
        assert num_tokens % self.merge_unit == 0, num_tokens
        assert len(cu_seqlens) - 1 <= bucket // self.merge_unit

        if not self.graphs:
            self._capture_all(hidden, rotary_pos_emb_cos)
        entry = self.graphs[bucket]

        entry.x[:num_tokens, 0].copy_(hidden)
        entry.x[num_tokens:].zero_()
        entry.cos[:num_tokens].copy_(rotary_pos_emb_cos)
        entry.sin[:num_tokens].copy_(rotary_pos_emb_sin)
        entry.seqlen_buf.copy_(
            torch.from_numpy(self._seqlens(cu_seqlens, bucket)), non_blocking=True
        )
        entry.graph.replay()
        # The output buffer is rewritten by the next replay, and the
        # multimodal embedding cache keeps the result, so return a copy.
        return entry.output[: num_tokens // self.merge_unit].clone()
