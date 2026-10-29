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

"""ViT NPU Graph Runner class."""

from __future__ import annotations

from typing import Dict, Hashable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch_npu

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.multimodal.vit_cuda_graph_runner import ViTCudaGraphRunner


class ViTNpuGraphRunner(ViTCudaGraphRunner):
    """Generic ViT NPU Graph Runner.

    This runner captures the "blocks + merger + deepstack merger (optional)" part
    of a vision transformer into a NPU graph and replays it for identical shapes.

    Optional for Qwen3 deepstack:
      - vit.deepstack_vision_indexes: Sequence[int]
      - vit.deepstack_merger_list: nn.ModuleList (same length as deepstack_vision_indexes)
    """

    _graph_memory_pool = None
    # Packed image layouts vary with each scheduling round. Keep capturing
    # repeated layouts useful without retaining a graph for every combination.
    _MAX_CACHED_GRAPHS = 32

    def __init__(
        self,
        vit: nn.Module,
    ) -> None:
        super().__init__(vit)
        self.device_module = torch.get_device_module(self.device)
        self.cu_seq_lens: Dict[Hashable, torch.Tensor] = {}

        # rotary position buffers shared across graphs
        self.sin_cos_ws: Dict[Hashable, Tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    def device(self) -> torch.device:
        return self.vit.device

    @property
    def dtype(self) -> torch.dtype:
        return self.vit.dtype

    def _create_graph(
        self,
        graph_key: Hashable,
    ):

        graph = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(graph, pool=ViTNpuGraphRunner._graph_memory_pool):
            self.block_output[graph_key] = self._forward_blocks(
                self.block_input[graph_key],
                self.cu_seq_lens[graph_key],
                *self.sin_cos_ws[graph_key],
                output_ws=self.block_ws[graph_key],
            )

        self.block_graphs[graph_key] = graph

    def _forward_blocks(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        output_ws: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Share the captured computation with the uncached-layout fallback."""
        if self._attn_backend != "ascend_attn":
            raise RuntimeError(
                f"ViT NPU graph does not support attention backend: {self._attn_backend}"
            )
        if output_ws is None:
            # Ascend attention still uses its graph-mode calling convention
            # while an uncached layout runs eagerly: it needs an output tensor
            # and cumulative seqlens without the leading zero.
            attn = self.vit.blocks[0].attn
            output_ws = torch.empty(
                x.shape[0],
                attn.num_attention_heads_per_partition,
                attn.head_size,
                device=self.device,
                dtype=self.dtype,
            )
        deepstack_outs: List[torch.Tensor] = []
        for layer_num, blk in enumerate(self.vit.blocks):
            x = blk(
                x,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                output_ws=output_ws,
            )
            if layer_num in self._deepstack_visual_indexes:
                if self._deepstack_merger_list is None:
                    raise RuntimeError(
                        "deepstack_visual_indexes exists but deepstack_merger_list is missing."
                    )
                deepstack_outs.append(
                    self._deepstack_merger_list[len(deepstack_outs)](x)
                )
        main_out = self.vit.merger(x)
        if deepstack_outs:
            return torch.cat([main_out] + deepstack_outs, dim=1)
        return main_out

    def create_graph(
        self,
        x_3d: torch.Tensor,  # [S, 1, H]
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: Optional[torch.Tensor] = None,
        rotary_pos_emb_sin: Optional[torch.Tensor] = None,
        attention_layout_key: Optional[Hashable] = None,
    ) -> Hashable:
        vit = self.vit
        graph_key = self._get_graph_key(
            x_3d,
            cu_seqlens,
            None,
            attention_layout_key,
        )
        seq_len = x_3d.shape[0]

        if graph_key in self.block_graphs:
            return graph_key

        if ViTNpuGraphRunner._graph_memory_pool is None:
            ViTNpuGraphRunner._graph_memory_pool = (
                self.device_module.graph_pool_handle()
            )
        # Set graph pool id globally to be able to use symmetric memory
        set_graph_pool_id(ViTNpuGraphRunner._graph_memory_pool)

        # pre-allocate workspace
        attn_module: VisionAttention = vit.blocks[0].attn
        num_heads = attn_module.num_attention_heads_per_partition
        attn_head_dim = attn_module.head_size

        if graph_key not in self.block_output:
            self.block_output[graph_key] = x_3d
            # Vision blocks update their input in place. Capture must not
            # modify the caller's tensor before the first replay copies it.
            self.block_input[graph_key] = x_3d.clone()
            self.block_ws[graph_key] = torch.empty(
                seq_len,
                num_heads,
                attn_head_dim,
                device=self.device,
                dtype=self.dtype,
            )
            if rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
                self.sin_cos_ws[graph_key] = (rotary_pos_emb_cos, rotary_pos_emb_sin)

        if graph_key not in self.cu_seq_lens:
            # TND fused attention expects cumulative seqlens (cu_seqlens[1:]),
            # not per-sequence lengths.
            self.cu_seq_lens[graph_key] = cu_seqlens[1:].to("cpu").to(torch.int32)

        if rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            self._create_graph(
                graph_key=graph_key,
            )

        return graph_key

    def replay(
        self,
        graph_key: Hashable,
        x_3d: torch.Tensor,
        rotary_pos_emb_cos: Optional[torch.Tensor] = None,
        rotary_pos_emb_sin: Optional[torch.Tensor] = None,
        output_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            # update rotary workspace content
            self.sin_cos_ws[graph_key][0].copy_(rotary_pos_emb_cos)
            self.sin_cos_ws[graph_key][1].copy_(rotary_pos_emb_sin)

        # copy input
        self.block_input[graph_key].copy_(x_3d)

        # replay
        self.block_graphs[graph_key].replay()

        out = self.block_output[graph_key]

        # Optional output reordering (Qwen2.5-VL window permutation inverse)
        if output_indices is not None:
            out = out.index_select(0, output_indices)

        # Cache entries and split views outlive this invocation. Graph replay
        # (including another graph using the shared pool) may overwrite out.
        return out.clone() if output_indices is None else out

    def run(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: Optional[torch.Tensor] = None,
        rotary_pos_emb_sin: Optional[torch.Tensor] = None,
        output_indices: Optional[torch.Tensor] = None,
        attention_layout_key: Optional[Hashable] = None,
    ) -> torch.Tensor:
        # x: [seq_len, hidden] -> [S, B=1, H]
        x_3d = x.unsqueeze(1)
        graph_key = self._get_graph_key(
            x_3d,
            cu_seqlens,
            None,
            attention_layout_key,
        )
        if graph_key not in self.block_graphs:
            if len(self.block_graphs) >= self._MAX_CACHED_GRAPHS:
                out = self._forward_blocks(
                    x_3d,
                    cu_seqlens[1:].to("cpu", dtype=torch.int32),
                    rotary_pos_emb_cos,
                    rotary_pos_emb_sin,
                )
                if output_indices is not None:
                    out = out.index_select(0, output_indices)
                return out
            self.create_graph(
                x_3d=x_3d,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                attention_layout_key=attention_layout_key,
            )

        return self.replay(
            graph_key=graph_key,
            x_3d=x_3d,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            output_indices=output_indices,
        )
