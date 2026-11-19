"""Bucketed CUDA graphs for the rank-local DeepSeek V4.1 vision encoder."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.attention.vision import (
    VisionAttentionMetadata,
    prepare_vision_attention_metadata,
)
from sglang.srt.model_executor.runner_utils.pool import (
    get_or_create_global_graph_capture_stream,
    get_or_create_global_graph_memory_pool,
    graph_pool_capture_scope,
    graph_pool_replay_scope,
)

if TYPE_CHECKING:
    from sglang.srt.models.deepseek_v41_vit import ViT


@dataclass(frozen=True)
class _CapturedVisionGraph:
    graph: torch.cuda.CUDAGraph
    patches: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    metadata: VisionAttentionMetadata
    output: torch.Tensor


class DeepseekV41ViTCudaGraphRunner:
    @torch.no_grad()
    def __init__(self, vit: "ViT", num_patch_buckets: Sequence[int]) -> None:
        assert not vit.training
        weight = vit.patch_embed.proj.weight
        assert weight.is_cuda
        assert num_patch_buckets and min(num_patch_buckets) > 0
        assert all(
            block.attn.qkv_backend_name in ("fa3", "fa4", "triton_attn")
            for block in vit.blocks
        ), "DeepSeek vision graphs require device-side sequence lengths"
        self.vit = vit
        self.graphs: dict[int, _CapturedVisionGraph] = {}
        pool = get_or_create_global_graph_memory_pool(torch.cuda)
        stream = get_or_create_global_graph_capture_stream()
        with graph_pool_capture_scope():
            for bucket in sorted(set(num_patch_buckets)):
                patches = weight.new_zeros((bucket, weight.shape[1]))
                cos = torch.ones(
                    (bucket, 1, vit.rope_dim),
                    dtype=torch.float32,
                    device=weight.device,
                )
                sin = torch.zeros_like(cos)
                # The second sequence isolates padding from the real image.
                metadata = prepare_vision_attention_metadata(
                    torch.tensor([0, bucket, bucket], dtype=torch.int32),
                    weight.device,
                    max_seqlen=bucket,
                )
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        vit.forward_with_metadata(patches, cos, sin, metadata)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool, stream=stream):
                    output = vit.forward_with_metadata(patches, cos, sin, metadata)
                self.graphs[bucket] = _CapturedVisionGraph(
                    graph, patches, cos, sin, metadata, output
                )

    def run(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor | None:
        from sglang.srt.models.deepseek_v41_vit import get_vision_cos_sin

        assert not self.vit.training
        n = patches.shape[0]
        bucket = next((size for size in self.graphs if n <= size), None)
        if bucket is None:
            return None
        entry = self.graphs[bucket]
        entry.patches[:n].copy_(patches.flatten(1))
        entry.patches[n:].zero_()
        cos, sin = get_vision_cos_sin(n_h, n_w, self.vit.rope_dim, self.vit.rope_theta)
        entry.cos[:n].copy_(cos)
        entry.sin[:n].copy_(sin)
        entry.metadata.cu_seqlens[1].fill_(n)
        entry.metadata.seq_lens[0].fill_(n)
        entry.metadata.seq_lens[1].fill_(bucket - n)
        with graph_pool_replay_scope():
            entry.graph.replay()
            # Image embeddings outlive this call in the multimodal cache.
            return entry.output[:n].clone()
