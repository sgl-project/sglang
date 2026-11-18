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
import logging
from contextlib import nullcontext
from typing import Dict, Hashable, List, Optional, Tuple

import torch
import torch.nn as nn

from sglang.srt.environ import envs
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)


class ViTCudaGraphRunner:
    """Generic ViT CUDA Graph Runner.

    This runner captures the "blocks + merger + deepstack merger (optional)" part
    of a vision transformer into a CUDA graph and replays it for identical shapes.

    Optional for Qwen2.5 windowed attention:
      - vit.fullatt_block_indexes: Sequence[int]
      - run() provides both cu_seqlens and cu_window_seqlens

    Optional for Qwen3 deepstack:
      - vit.deepstack_vision_indexes: Sequence[int]
      - vit.deepstack_merger_list: nn.ModuleList (same length as deepstack_vision_indexes)
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
        # One memory pool for all captured graphs: a private pool per graph grows
        # reserved memory with every new image layout. Sharing relies on replays
        # being serialized on one stream and on replay() returning a copy of the
        # output, since the next replay of any graph can reuse that memory.
        self._graph_pool = None
        # Graphs are keyed on the exact attention layout, so batch compositions can
        # add new keys indefinitely; past this many, layouts run uncaptured.
        self._max_graphs: int = envs.SGLANG_VIT_CUDA_GRAPH_CACHE_CAPACITY.get()
        if self._max_graphs < 0:
            raise ValueError(
                "SGLANG_VIT_CUDA_GRAPH_CACHE_CAPACITY must be >= 0, "
                f"got {self._max_graphs}"
            )
        self._capacity_logged = False

        # captured seqlens buffers (addresses must be stable for cuda-graph replay)
        self.cu_full_len: Dict[Hashable, torch.Tensor] = {}
        self.cu_window_len: Dict[Hashable, torch.Tensor] = {}
        self.cu_full_len_kk: Dict[Hashable, torch.Tensor] = {}
        self.cu_window_len_kk: Dict[Hashable, torch.Tensor] = {}

        # Current rotary workspace plus older allocations retained by graphs
        # captured before the workspace grew.
        self.sin_cos_ws: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._retired_sin_cos_ws: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._sin_cos_ws_by_graph: Dict[
            Hashable, Tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self.max_context_len = getattr(vit, "max_context_len", None)

        # Qwen2.5-VL specific viarable.
        self._fullatt_block_indexes = set(getattr(vit, "fullatt_block_indexes", ()))

        # Qwen3-VL specific variables.
        self._deepstack_visual_indexes = list(
            getattr(vit, "deepstack_visual_indexes", []) or []
        )
        self._deepstack_merger_list = getattr(vit, "deepstack_merger_list", None)

        first_blk = vit.blocks[0]
        self._blk_accepts_output_ws = (
            "output_ws" in inspect.signature(first_blk.forward).parameters
        )

        self._attn: Optional[VisionAttention] = getattr(first_blk, "attn", None)
        self._attn_backend: Optional[str] = getattr(
            self._attn, "qkv_backend_name", None
        )

    @property
    def device(self) -> torch.device:
        return self.vit.device

    @property
    def dtype(self) -> torch.dtype:
        return self.vit.dtype

    def _get_sin_cos_ws(
        self,
        graph_key: Hashable,
        seq_len: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the stable rotary buffers captured by one graph.

        dtype is the dtype of the cos/sin the model passes (fp32 for Qwen3-VL),
        not the ViT dtype, so the graph rotates with the same values as eager.
        """
        graph_ws = self._sin_cos_ws_by_graph.get(graph_key)
        if graph_ws is not None:
            return graph_ws

        needs_new_workspace = self.sin_cos_ws is None or (
            self.sin_cos_ws[0].size(0) < seq_len
            or self.sin_cos_ws[0].size(1) < head_dim
            or self.sin_cos_ws[0].dtype != dtype
        )
        if needs_new_workspace:
            previous = self.sin_cos_ws
            previous_seq_len = previous[0].size(0) if previous is not None else 0
            previous_head_dim = previous[0].size(1) if previous is not None else 0
            max_shape = max(
                self.max_context_len or 0,
                previous_seq_len * 2,
                seq_len,
            )
            max_head_dim = max(previous_head_dim, head_dim)
            cos_ws = torch.empty(
                max_shape, max_head_dim, dtype=dtype, device=self.device
            )
            sin_ws = torch.empty(
                max_shape, max_head_dim, dtype=dtype, device=self.device
            )
            if previous is not None:
                # CUDA graphs retain captured addresses, so an older allocation
                # cannot be freed when a larger request grows the workspace.
                self._retired_sin_cos_ws.append(previous)
            self.sin_cos_ws = (cos_ws, sin_ws)

        graph_ws = (
            self.sin_cos_ws[0][:seq_len, :head_dim],
            self.sin_cos_ws[1][:seq_len, :head_dim],
        )
        self._sin_cos_ws_by_graph[graph_key] = graph_ws
        return graph_ws

    @staticmethod
    def _sequence_layout_key(cu_seqlens: Optional[torch.Tensor]) -> Optional[tuple]:
        if cu_seqlens is None:
            return None
        return tuple(int(value) for value in cu_seqlens.tolist())

    def _get_graph_key(
        self,
        x_3d: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_window_seqlens: Optional[torch.Tensor],
        attention_layout_key: Optional[Hashable] = None,
    ) -> Hashable:
        """Include attention boundaries so equal-length batches stay distinct."""
        if attention_layout_key is None:
            attention_layout_key = (
                self._sequence_layout_key(cu_seqlens),
                self._sequence_layout_key(cu_window_seqlens),
            )
        return (x_3d.shape[0], attention_layout_key)

    def _capture_context(self):
        # A DP-sharded encoder intentionally lets each rank capture only the
        # images it owns (and some ranks can own none). Entering the TP
        # communication capture in that case requires every TP peer to enter
        # the same collective capture sequence, which deadlocks on an uneven
        # image assignment. The encoder's output all-gather is outside this
        # graph, and all layers are local in DP mode, so capture locally.
        if getattr(self.vit, "use_data_parallel", False):
            return nullcontext()
        ca_comm = get_parallel().tp_group.ca_comm
        return ca_comm.capture() if ca_comm is not None else nullcontext()

    def _create_graph(
        self,
        graph_key: int,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # (cos, sin), [S, D]
        rotary_pos_emb_cos: Optional[torch.Tensor] = None,
        rotary_pos_emb_sin: Optional[torch.Tensor] = None,
    ):

        graph = torch.cuda.CUDAGraph()
        vit = self.vit

        cu_window = cu_window_kk = max_window_len = None
        # Qwen2.5-VL
        if self._fullatt_block_indexes:
            cu_window = self.cu_window_len[graph_key]
            cu_window_kk = self.cu_window_len_kk[graph_key]
            max_window_len = int(cu_window_kk.max().item())

        cu_full = self.cu_full_len[graph_key]
        cu_full_kk = self.cu_full_len_kk[graph_key]
        max_full_len = int(cu_full_kk.max().item())

        backend = self._attn_backend

        if self._fullatt_block_indexes and 0 not in vit.fullatt_block_indexes:
            warmup_cu_ws = [cu_window, cu_window_kk, max_window_len]
        else:
            warmup_cu_ws = [cu_full, cu_full_kk, max_full_len]
        warmup_layouts = [warmup_cu_ws]
        if backend in ("fa3", "fa4", "aiter_attn"):
            warmup_layouts = [[warmup_cu_ws[0], warmup_cu_ws[2]]]
        if backend == "fa4" and self._fullatt_block_indexes:
            # FA4 JIT-compiles each kernel variant on first use, which must not
            # happen inside the capture: warm up the windowed and the full layout.
            warmup_layouts = [[cu_window, max_window_len], [cu_full, max_full_len]]

        warmup_kwargs = dict(output_ws=self.block_ws[graph_key])
        if position_embeddings is not None:
            warmup_kwargs["position_embeddings"] = position_embeddings
        elif rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            warmup_kwargs["rotary_pos_emb_cos"] = rotary_pos_emb_cos
            warmup_kwargs["rotary_pos_emb_sin"] = rotary_pos_emb_sin
        with torch.no_grad():
            for layout in warmup_layouts:
                vit.blocks[0](
                    self.block_input[graph_key], cu_seqlens=layout, **warmup_kwargs
                )
        torch.cuda.synchronize()

        if self._graph_pool is None:
            self._graph_pool = torch.cuda.graph_pool_handle()
        # Captured lazily while the LLM serves; other threads (e.g. the c10d NCCL
        # watchdog) keep calling the runtime, which global mode turns into an error.
        with (
            self._capture_context(),
            torch.cuda.graph(
                graph, pool=self._graph_pool, capture_error_mode="thread_local"
            ),
        ):
            self.block_output[graph_key] = self._forward_blocks(
                self.block_input[graph_key],
                cu_full=cu_full,
                cu_full_kk=cu_full_kk,
                max_full_len=max_full_len,
                cu_window=cu_window,
                cu_window_kk=cu_window_kk,
                max_window_len=max_window_len,
                position_embeddings=position_embeddings,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                output_ws=self.block_ws[graph_key],
            )

        self.block_graphs[graph_key] = graph

    def _new_block_ws(self, seq_len: int) -> torch.Tensor:
        attn_module: VisionAttention = self.vit.blocks[0].attn
        return torch.empty(
            seq_len,
            attn_module.num_attention_heads_per_partition,
            attn_module.head_size,
            device=self.device,
            dtype=self.dtype,
        )

    def _forward_blocks(
        self,
        x: torch.Tensor,
        *,
        cu_full: torch.Tensor,
        cu_full_kk: torch.Tensor,
        max_full_len: int,
        cu_window: Optional[torch.Tensor],
        cu_window_kk: Optional[torch.Tensor],
        max_window_len: Optional[int],
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        rotary_pos_emb_cos: Optional[torch.Tensor],
        rotary_pos_emb_sin: Optional[torch.Tensor],
        output_ws: torch.Tensor,
    ) -> torch.Tensor:
        """Blocks + merger (+ deepstack mergers): the work a graph captures."""
        vit = self.vit
        backend = self._attn_backend
        y = None
        deepstack_outs: List[torch.Tensor] = []
        deepstack_capture_idx = 0

        for layer_num, blk in enumerate(vit.blocks):
            if self._fullatt_block_indexes:
                if layer_num in vit.fullatt_block_indexes:
                    cu_seqlens_now = cu_full
                    cu_seqlens_kk_now = cu_full_kk
                    max_len = max_full_len
                else:
                    cu_seqlens_now = cu_window
                    cu_seqlens_kk_now = cu_window_kk
                    max_len = max_window_len
            else:
                cu_seqlens_now = cu_full
                cu_seqlens_kk_now = cu_full_kk
                max_len = max_full_len

            if backend == "triton_attn":
                cu_seq_len_ws = [cu_seqlens_now, cu_seqlens_kk_now, max_len]
            elif backend in ("fa3", "fa4", "aiter_attn"):
                cu_seq_len_ws = [cu_seqlens_now, max_len]
            else:
                raise RuntimeError(
                    f"ViT CUDA graph does not support attention backend: {backend}"
                )

            if position_embeddings is not None:
                if layer_num == 0:
                    y = blk(
                        x,
                        cu_seqlens=cu_seq_len_ws,
                        position_embeddings=position_embeddings,
                        output_ws=output_ws,
                    )
                else:
                    y = blk(
                        y,
                        cu_seqlens=cu_seq_len_ws,
                        position_embeddings=position_embeddings,
                        output_ws=output_ws,
                    )
            elif rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
                if layer_num == 0:
                    y = blk(
                        x,
                        cu_seqlens=cu_seq_len_ws,
                        rotary_pos_emb_cos=rotary_pos_emb_cos,
                        rotary_pos_emb_sin=rotary_pos_emb_sin,
                        output_ws=output_ws,
                    )
                else:
                    y = blk(
                        y,
                        cu_seqlens=cu_seq_len_ws,
                        rotary_pos_emb_cos=rotary_pos_emb_cos,
                        rotary_pos_emb_sin=rotary_pos_emb_sin,
                        output_ws=output_ws,
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
                deepstack_out = self._deepstack_merger_list[deepstack_capture_idx](y)
                deepstack_outs.append(deepstack_out)
                deepstack_capture_idx += 1

        main_out = vit.merger(y)

        if deepstack_outs:
            return torch.cat([main_out] + deepstack_outs, dim=1)
        return main_out

    def create_graph(
        self,
        x_3d: torch.Tensor,  # [S, 1, H]
        cu_seqlens: torch.Tensor,
        cu_window_seqlens: Optional[torch.Tensor],
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ],  # (cos, sin), [S, D]
        rotary_pos_emb_cos: Optional[torch.Tensor] = None,
        rotary_pos_emb_sin: Optional[torch.Tensor] = None,
        attention_layout_key: Optional[Hashable] = None,
    ) -> Hashable:
        graph_key = self._get_graph_key(
            x_3d,
            cu_seqlens,
            cu_window_seqlens,
            attention_layout_key,
        )
        seq_len = x_3d.shape[0]

        if graph_key in self.block_graphs:
            return graph_key

        if graph_key not in self.block_output:
            self.block_output[graph_key] = torch.empty_like(
                x_3d, device=self.device
            ).contiguous()
            self.block_input[graph_key] = torch.empty_like(
                x_3d, device=self.device
            ).contiguous()
            self.block_ws[graph_key] = self._new_block_ws(seq_len)

        # Qwen2.5-VL
        if self._fullatt_block_indexes:
            if graph_key not in self.cu_window_len:
                self.cu_window_len[graph_key] = cu_window_seqlens
                self.cu_full_len[graph_key] = cu_seqlens
                self.cu_window_len_kk[graph_key] = (
                    cu_window_seqlens[1:] - cu_window_seqlens[:-1]
                )
                self.cu_full_len_kk[graph_key] = cu_seqlens[1:] - cu_seqlens[:-1]
        else:
            if graph_key not in self.cu_full_len:
                self.cu_full_len[graph_key] = cu_seqlens
                self.cu_full_len_kk[graph_key] = cu_seqlens[1:] - cu_seqlens[:-1]

        self.block_input[graph_key].copy_(x_3d)

        if position_embeddings is not None:
            head_dim = position_embeddings[0].shape[1]
            used_cos_ws, used_sin_ws = self._get_sin_cos_ws(
                graph_key, seq_len, head_dim, dtype=position_embeddings[0].dtype
            )
            used_cos_ws.copy_(position_embeddings[0])
            used_sin_ws.copy_(position_embeddings[1])
            persist_position_embeddings = (used_cos_ws, used_sin_ws)
            self._create_graph(
                graph_key=graph_key, position_embeddings=persist_position_embeddings
            )
        elif rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            head_dim = rotary_pos_emb_cos.shape[1]
            used_cos_ws, used_sin_ws = self._get_sin_cos_ws(
                graph_key, seq_len, head_dim, dtype=rotary_pos_emb_cos.dtype
            )
            used_cos_ws.copy_(rotary_pos_emb_cos)
            used_sin_ws.copy_(rotary_pos_emb_sin)
            self._create_graph(
                graph_key=graph_key,
                position_embeddings=None,
                rotary_pos_emb_cos=used_cos_ws,
                rotary_pos_emb_sin=used_sin_ws,
            )

        return graph_key

    def replay(
        self,
        graph_key: Hashable,
        x_3d: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rotary_pos_emb_cos: Optional[torch.Tensor] = None,
        rotary_pos_emb_sin: Optional[torch.Tensor] = None,
        output_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        seq_len = x_3d.shape[0]
        if position_embeddings is not None:
            head_dim = position_embeddings[0].shape[1]
            used_cos_ws, used_sin_ws = self._get_sin_cos_ws(
                graph_key, seq_len, head_dim, dtype=position_embeddings[0].dtype
            )
            used_cos_ws.copy_(position_embeddings[0])
            used_sin_ws.copy_(position_embeddings[1])
        elif rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            head_dim = rotary_pos_emb_cos.shape[1]
            used_cos_ws, used_sin_ws = self._get_sin_cos_ws(
                graph_key, seq_len, head_dim, dtype=rotary_pos_emb_cos.dtype
            )
            used_cos_ws.copy_(rotary_pos_emb_cos)
            used_sin_ws.copy_(rotary_pos_emb_sin)

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
        cu_window_seqlens: Optional[torch.Tensor],
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
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
            cu_window_seqlens,
            attention_layout_key,
        )

        if graph_key not in self.block_graphs and self._cache_is_full():
            return self._run_uncaptured(
                x_3d=x_3d,
                cu_seqlens=cu_seqlens,
                cu_window_seqlens=cu_window_seqlens,
                position_embeddings=position_embeddings,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                output_indices=output_indices,
            )

        if graph_key not in self.block_graphs:
            self.create_graph(
                x_3d=x_3d,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
                cu_window_seqlens=cu_window_seqlens,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                attention_layout_key=attention_layout_key,
            )

        return self.replay(
            graph_key=graph_key,
            x_3d=x_3d,
            position_embeddings=position_embeddings,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            output_indices=output_indices,
        )

    def _cache_is_full(self) -> bool:
        if len(self.block_graphs) < self._max_graphs:
            return False
        if not self._capacity_logged:
            logger.warning(
                "ViT CUDA graph cache reached SGLANG_VIT_CUDA_GRAPH_CACHE_CAPACITY=%d; "
                "new attention layouts run without a graph",
                self._max_graphs,
            )
            self._capacity_logged = True
        return True

    def _run_uncaptured(
        self,
        x_3d: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_window_seqlens: Optional[torch.Tensor],
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        rotary_pos_emb_cos: Optional[torch.Tensor],
        rotary_pos_emb_sin: Optional[torch.Tensor],
        output_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run the captured part eagerly, with the same arguments a graph uses."""
        cu_window = cu_window_kk = max_window_len = None
        if self._fullatt_block_indexes:
            cu_window = cu_window_seqlens
            cu_window_kk = cu_window_seqlens[1:] - cu_window_seqlens[:-1]
            max_window_len = int(cu_window_kk.max().item())
        cu_full_kk = cu_seqlens[1:] - cu_seqlens[:-1]

        out = self._forward_blocks(
            x_3d,
            cu_full=cu_seqlens,
            cu_full_kk=cu_full_kk,
            max_full_len=int(cu_full_kk.max().item()),
            cu_window=cu_window,
            cu_window_kk=cu_window_kk,
            max_window_len=max_window_len,
            position_embeddings=position_embeddings,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            output_ws=self._new_block_ws(x_3d.shape[0]),
        )
        if output_indices is not None:
            out = out.index_select(0, output_indices)
        return out
