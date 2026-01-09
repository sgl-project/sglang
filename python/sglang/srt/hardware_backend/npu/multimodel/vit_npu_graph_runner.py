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

import inspect
from typing import Dict, Hashable, List, Optional, Tuple

import torch
import torch.nn as nn

from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.multimodal.vit_cuda_graph_runner import ViTCudaGraphRunner
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.server_args import get_global_server_args


vit_graph_memory_pool = None


def get_vit_graph_memory_pool():
    return vit_graph_memory_pool


def set_vit_graph_memory_pool(val):
    global vit_graph_memory_pool
    vit_graph_memory_pool = val


class ViTNpuGraphRunner(ViTCudaGraphRunner):
    """Generic ViT NPU Graph Runner.

    This runner captures the "blocks + merger + deepstack merger (optional)" part
    of a vision transformer into a NPU graph and replays it for identical shapes.

    Optional for Qwen3 deepstack:
      - vit.deepstack_vision_indexes: Sequence[int]
      - vit.deepstack_merger_list: nn.ModuleList (same length as deepstack_vision_indexes)
    """

    def __init__(
        self,
        vit: nn.Module,
    ) -> None:
        super().__init__(vit)
        self.device_module = torch.get_device_module(self.device)
        self.cu_seq_lens: Dict[Hashable, torch.Tensor] = {}

        # graph_key -> buffers / graphs
        self.block_input: Dict[Hashable, torch.Tensor] = {}
        self.block_ws: Dict[Hashable, torch.Tensor] = {}
        self.block_graphs: Dict[Hashable, torch.cuda.CUDAGraph] = {}
        self.block_output: Dict[Hashable, torch.Tensor] = {}

        # rotary position buffers shared across graphs
        self.sin_cos_ws: Dict[Hashable, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.max_context_len = getattr(vit, "max_context_len", None)

        # Qwen3-VL specific variables.
        self._deepstack_visual_indexes = list(
            getattr(vit, "deepstack_visual_indexes", []) or []
        )
        self._deepstack_merger_list = getattr(vit, "deepstack_merger_list", None)

        first_blk = vit.blocks[0]

        self._attn: Optional[VisionAttention] = getattr(first_blk, "attn", None)
        self._attn_backend = getattr(self._attn, "qkv_backend", None)

    @property
    def device(self) -> torch.device:
        return self.vit.device

    @property
    def dtype(self) -> torch.dtype:
        return self.vit.dtype

    def _create_graph(
        self,
        graph_key: int,
    ):

        graph = torch.cuda.CUDAGraph()
        vit = self.vit

        override_backend = get_global_server_args().mm_attention_backend
        with torch.cuda.graph(graph, pool=get_vit_graph_memory_pool()):
            y = None
            deepstack_outs: List[torch.Tensor] = []
            deepstack_capture_idx = 0

            for layer_num, blk in enumerate(vit.blocks):
                if override_backend == "ascend_attn":
                    cu_seq_lens = self.cu_seq_lens[graph_key]
                else:
                    raise RuntimeError("Not supported ViT attention backend")

                if layer_num == 0:
                    y = blk(
                        self.block_input[graph_key],
                        cu_seqlens=cu_seq_lens,
                        rotary_pos_emb_cos=self.sin_cos_ws[graph_key][0],
                        rotary_pos_emb_sin=self.sin_cos_ws[graph_key][1],
                        output_ws=self.block_ws[graph_key],
                    )
                else:
                    y = blk(
                        y,
                        cu_seqlens=cu_seq_lens,
                        rotary_pos_emb_cos=self.sin_cos_ws[graph_key][0],
                        rotary_pos_emb_sin=self.sin_cos_ws[graph_key][1],
                        output_ws=self.block_ws[graph_key],
                    )

                # Optional deepstack support (Qwen3-VL)
                if (
                    self._deepstack_visual_indexes
                    and layer_num in self._deepstack_visual_indexes
                ):
                    if self._deepstack_merger_list is None:
                        raise RuntimeError(
                            "deepstack_visual_indexes exists but deepstack_merger_list is missing."
                        )
                    deepstack_out = self._deepstack_merger_list[deepstack_capture_idx](
                        y
                    )
                    deepstack_outs.append(deepstack_out)
                    deepstack_capture_idx += 1

            main_out = vit.merger(y)

            if deepstack_outs:
                self.block_output[graph_key] = torch.cat(
                    [main_out] + deepstack_outs, dim=1
                )
            else:
                self.block_output[graph_key] = main_out

        self.block_graphs[graph_key] = graph

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

    def create_graph(
        self,
        x_3d: torch.Tensor,  # [S, 1, H]
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: Optional[torch.Tensor] = None,
        rotary_pos_emb_sin: Optional[torch.Tensor] = None,
    ) -> int:
        vit = self.vit
        graph_key = self._get_graph_key(x_3d)

        if graph_key in self.block_graphs:
            return graph_key

        if get_vit_graph_memory_pool() is None:
            set_vit_graph_memory_pool(self.device_module.graph_pool_handle())
        # Set graph pool id globally to be able to use symmetric memory
        set_graph_pool_id(get_vit_graph_memory_pool())

        # pre-allocate workspace
        attn_module: VisionAttention = vit.blocks[0].attn
        num_heads = attn_module.num_attention_heads_per_partition
        attn_head_dim = attn_module.head_size

        if graph_key not in self.block_output:
            self.block_output[graph_key] = x_3d
            self.block_input[graph_key] = x_3d
            self.block_ws[graph_key] = torch.empty(
                graph_key,
                num_heads,
                attn_head_dim,
                device=self.device,
                dtype=self.dtype,
            )
            if rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
                self.sin_cos_ws[graph_key] = (rotary_pos_emb_cos, rotary_pos_emb_sin)

        if graph_key not in self.cu_seq_lens:
            seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            self.cu_seq_lens[graph_key] = seq_lens.to("cpu").to(torch.int32)

        if rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            self._create_graph(
                graph_key=graph_key,
            )

        return graph_key

    def replay(
        self,
        graph_key: int,
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

        return out

    def run(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: Optional[torch.Tensor] = None,
        rotary_pos_emb_sin: Optional[torch.Tensor] = None,
        output_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: [seq_len, hidden] -> [S, B=1, H]
        x_3d = x.unsqueeze(1)
        graph_key = self._get_graph_key(x_3d)
        if graph_key not in self.block_graphs:
            self.create_graph(
                x_3d=x_3d,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
            )

        return self.replay(
            graph_key=graph_key,
            x_3d=x_3d,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            output_indices=output_indices,
        )
