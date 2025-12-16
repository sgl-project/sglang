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

"""ViT CUDA Graph Runner class."""
from __future__ import annotations

import inspect
from typing import Dict, Hashable, Optional, Tuple

import torch
import torch.nn as nn

from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.server_args import get_global_server_args


class ViTCudaGraphRunner:
    """ViT CUDA Graph Runner

    Cached with graph_key = seq_len, for each seq_len capture once.
    expose run(), internally call create_graph().
    exceed call invokes replay().
    """

    def __init__(
        self,
        vit: nn.Module,
    ) -> None:
        self.vit = vit

        # graph_key -> buffers / graphs
        self.block_input: Dict[Hashable, torch.Tensor] = {}
        self.block_ws: Dict[Hashable, torch.Tensor] = {}
        self.block_graphs: Dict[Hashable, torch.cuda.CUDAGraph] = {}
        self.block_output: Dict[Hashable, torch.Tensor] = {}

        # captured seqlens buffers (addresses must be stable for cuda-graph replay)
        self.cu_full_len: Dict[Hashable, torch.Tensor] = {}
        self.cu_window_len: Dict[Hashable, torch.Tensor] = {}
        self.cu_full_len_kk: Dict[Hashable, torch.Tensor] = {}
        self.cu_window_len_kk: Dict[Hashable, torch.Tensor] = {}

        # rotary position buffers shared across graphs
        self.sin_cos_ws: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.max_context_len = getattr(vit, "max_context_len", None)

        self._fullatt_block_indexes = set(getattr(vit, "fullatt_block_indexes", ()))

        first_blk = vit.blocks[0]
        self._blk_accepts_output_ws = (
            "output_ws" in inspect.signature(first_blk.forward).parameters
        )

        self._attn: Optional[VisionAttention] = getattr(first_blk, "attn", None)
        self._attn_backend = getattr(self._attn, "qkv_backend", None)

    @property
    def device(self) -> torch.device:
        return self.vit.device

    @property
    def dtype(self) -> torch.dtype:
        return self.vit.dtype

    def _ensure_sin_cos_ws(self, seq_len: int, head_dim: int):
        if self.sin_cos_ws is None:
            max_shape = self.max_context_len or seq_len
            max_shape = max(max_shape, seq_len)
            cos_ws = torch.empty(
                max_shape, head_dim, dtype=self.dtype, device=self.device
            )
            sin_ws = torch.empty(
                max_shape, head_dim, dtype=self.dtype, device=self.device
            )
            self.sin_cos_ws = (cos_ws, sin_ws)
        else:
            if self.sin_cos_ws[0].size(0) < seq_len:
                max_shape = max(self.sin_cos_ws[0].size(0) * 2, seq_len)
                cos_ws = torch.empty(
                    max_shape, head_dim, dtype=self.dtype, device=self.device
                )
                sin_ws = torch.empty(
                    max_shape, head_dim, dtype=self.dtype, device=self.device
                )
                self.sin_cos_ws = (cos_ws, sin_ws)

    def _get_graph_key(self, x_3d: torch.Tensor) -> int:
        # x_3d: [S, B, H], B=1, S as graph_key
        return x_3d.shape[0]

    def _create_graph(self, graph_key: int, temp_cos_sin):
        graph = torch.cuda.CUDAGraph()
        vit = self.vit

        cu_window = self.cu_window_len[graph_key]
        cu_full = self.cu_full_len[graph_key]
        cu_window_kk = self.cu_window_len_kk[graph_key]
        cu_full_kk = self.cu_full_len_kk[graph_key]

        max_full_len = int(cu_full_kk.max().item())
        max_window_len = int(cu_window_kk.max().item())

        override_backend = get_global_server_args().mm_attention_backend

        with torch.cuda.graph(graph):
            y = None
            for layer_num, blk in enumerate(vit.blocks):
                if layer_num in vit.fullatt_block_indexes:
                    cu_seqlens_now = cu_full
                    cu_seqlens_kk_now = cu_full_kk
                    max_len = max_full_len
                else:
                    cu_seqlens_now = cu_window
                    cu_seqlens_kk_now = cu_window_kk
                    max_len = max_window_len

                if override_backend == "triton_attn":
                    cu_seq_len_ws = [cu_seqlens_now, cu_seqlens_kk_now, max_len]
                elif override_backend == "fa3":
                    cu_seq_len_ws = [cu_seqlens_now, max_len]
                else:
                    raise RuntimeError("Not supported ViT attention backend")

                if layer_num == 0:
                    y = blk(
                        self.block_input[graph_key],
                        cu_seqlens=cu_seq_len_ws,
                        position_embeddings=temp_cos_sin,
                        output_ws=self.block_ws[graph_key],
                    )
                else:
                    y = blk(
                        y,
                        cu_seqlens=cu_seq_len_ws,
                        position_embeddings=temp_cos_sin,
                        output_ws=self.block_ws[graph_key],
                    )

            self.block_output[graph_key] = vit.merger(y)

        self.block_graphs[graph_key] = graph

    def create_graph(
        self,
        x_3d: torch.Tensor,  # [S, 1, H]
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # (cos, sin), [S, D]
        cu_seqlens: torch.Tensor,
        cu_window_seqlens: torch.Tensor,
    ) -> int:
        vit = self.vit
        graph_key = self._get_graph_key(x_3d)

        if graph_key in self.block_graphs:
            return graph_key

        # make sure rotary workspace
        head_dim = position_embeddings[0].shape[1]
        self._ensure_sin_cos_ws(graph_key, head_dim)

        used_cos_ws = self.sin_cos_ws[0][:graph_key, :]
        used_sin_ws = self.sin_cos_ws[1][:graph_key, :]
        used_cos_ws.copy_(position_embeddings[0])
        used_sin_ws.copy_(position_embeddings[1])
        temp_cos_sin = (used_cos_ws, used_sin_ws)

        # pre-allocate workspace
        attn_module: VisionAttention = vit.blocks[0].attn
        num_heads = attn_module.num_attention_heads_per_partition
        attn_head_dim = attn_module.head_size

        if graph_key not in self.block_output:
            self.block_output[graph_key] = torch.empty_like(
                x_3d, device=self.device
            ).contiguous()
            self.block_input[graph_key] = torch.empty_like(
                x_3d, device=self.device
            ).contiguous()
            self.block_ws[graph_key] = torch.empty(
                graph_key,
                num_heads,
                attn_head_dim,
                device=self.device,
                dtype=self.dtype,
            )

        if graph_key not in self.cu_window_len:
            self.cu_window_len[graph_key] = cu_window_seqlens
            self.cu_full_len[graph_key] = cu_seqlens
            self.cu_window_len_kk[graph_key] = (
                cu_window_seqlens[1:] - cu_window_seqlens[:-1]
            )
            self.cu_full_len_kk[graph_key] = cu_seqlens[1:] - cu_seqlens[:-1]

        self._create_graph(graph_key, temp_cos_sin)

        return graph_key

    def replay(
        self,
        graph_key: int,
        x_3d: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        output_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # update rotary workspace content
        head_dim = position_embeddings[0].shape[1]
        self._ensure_sin_cos_ws(graph_key, head_dim)
        used_cos_ws = self.sin_cos_ws[0][:graph_key, :]
        used_sin_ws = self.sin_cos_ws[1][:graph_key, :]
        used_cos_ws.copy_(position_embeddings[0])
        used_sin_ws.copy_(position_embeddings[1])

        # copy input
        self.block_input[graph_key].copy_(x_3d)

        # replay
        self.block_graphs[graph_key].replay()

        out = self.block_output[graph_key]

        # Optional output reordering (Qwen2.5-VL window permutation inverse)
        if output_indices is not None:
            out = out.index_select(0, output_indices)

        return out

    def run(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        cu_window_seqlens: torch.Tensor,
        output_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: [seq_len, hidden] -> [S, B=1, H]
        x_3d = x.unsqueeze(1)
        graph_key = self._get_graph_key(x_3d)

        if graph_key not in self.block_graphs:
            self.create_graph(
                x_3d=x_3d,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
                cu_window_seqlens=cu_window_seqlens,
            )

        return self.replay(
            graph_key=graph_key,
            x_3d=x_3d,
            position_embeddings=position_embeddings,
            output_indices=output_indices,
        )
