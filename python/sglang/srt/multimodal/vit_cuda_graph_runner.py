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

import logging
import math
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Tuple

import torch
import torch.nn as nn

from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.server_args import get_global_server_args

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ViTCudaGraphConfig:
    # Budgets are post-merge vision tokens, aligned with LLM placeholders.
    vision_token_budgets: List[int]
    raw_token_budgets: List[int]
    max_batch_size: int


def is_vit_cuda_graph_enabled() -> bool:
    if envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get():
        return True
    try:
        return bool(getattr(get_global_server_args(), "enable_vit_cuda_graph", False))
    except Exception:
        return False


def _get_server_args():
    try:
        return get_global_server_args()
    except Exception:
        return None


def _auto_vision_token_budgets() -> List[int]:
    server_args = _get_server_args()
    max_budget = None
    if server_args is not None:
        max_budget = getattr(server_args, "chunked_prefill_size", None)
        if max_budget is None or max_budget <= 0:
            max_budget = getattr(server_args, "max_prefill_tokens", None)
        max_total_tokens = getattr(server_args, "max_total_tokens", None)
        if max_total_tokens is not None and max_total_tokens > 0:
            max_budget = (
                min(max_budget, max_total_tokens)
                if max_budget is not None and max_budget > 0
                else max_total_tokens
            )

    if max_budget is None or max_budget <= 0:
        max_budget = 8192

    max_budget = max(int(max_budget), 1)
    if max_budget <= 1024:
        return [max_budget]

    budgets = []
    budget = 1024
    while budget < max_budget:
        budgets.append(budget)
        budget *= 2
    budgets.append(max_budget)
    return sorted(set(budgets))


def resolve_vit_cuda_graph_config(spatial_merge_size: int) -> ViTCudaGraphConfig:
    server_args = _get_server_args()
    budgets = None
    max_batch_size = 0
    if server_args is not None:
        budgets = getattr(server_args, "vit_cuda_graph_token_budgets", None)
        max_batch_size = int(
            getattr(server_args, "vit_cuda_graph_max_batch_size", 0) or 0
        )

    vision_token_budgets = sorted(set(int(x) for x in budgets or [] if int(x) > 0))
    if not vision_token_budgets:
        vision_token_budgets = _auto_vision_token_budgets()

    merge_unit = int(spatial_merge_size) ** 2
    raw_token_budgets = [budget * merge_unit for budget in vision_token_budgets]

    if max_batch_size <= 0:
        max_batch_size = max(1, min(64, vision_token_budgets[-1]))

    return ViTCudaGraphConfig(
        vision_token_budgets=vision_token_budgets,
        raw_token_budgets=raw_token_budgets,
        max_batch_size=max_batch_size,
    )


def make_image_grid_for_vision_tokens(
    vision_tokens: int, spatial_merge_size: int
) -> List[int]:
    """Build a single-image grid whose post-merge token count is vision_tokens."""
    h_tokens = int(math.sqrt(vision_tokens))
    while h_tokens > 1 and vision_tokens % h_tokens != 0:
        h_tokens -= 1
    w_tokens = vision_tokens // h_tokens
    return [1, h_tokens * spatial_merge_size, w_tokens * spatial_merge_size]


class ViTCudaGraphRunner:
    """Generic ViT CUDA Graph Runner.

    Captures the "blocks + merger + deepstack merger (optional)" part of a
    vision transformer for fixed post-merge budgets. Runtime batches with fewer
    tokens are padded into the selected budget before replay.
    """

    def __init__(
        self,
        vit: nn.Module,
    ) -> None:
        self.vit = vit
        self.spatial_merge_size = int(getattr(vit, "spatial_merge_size", 1))
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.config = resolve_vit_cuda_graph_config(self.spatial_merge_size)
        self.vision_token_budgets = self.config.vision_token_budgets
        self.raw_token_budgets = self.config.raw_token_budgets
        self.max_batch_size = self.config.max_batch_size
        self.max_raw_budget = self.raw_token_budgets[-1]
        self.max_vision_budget = self.vision_token_budgets[-1]

        # graph_key -> buffers / graphs. graph_key is the raw ViT token budget.
        self.block_input: Dict[Hashable, torch.Tensor] = {}
        self.block_graphs: Dict[Hashable, torch.cuda.CUDAGraph] = {}
        self.block_output: Dict[Hashable, torch.Tensor] = {}
        self.input_buffer: Optional[torch.Tensor] = None

        # Captured seqlens buffers. Addresses are stable; contents are updated
        # before every replay. The *_kk buffers are Triton seq_lens.
        self.cu_full_len: Dict[Hashable, torch.Tensor] = {}
        self.cu_window_len: Dict[Hashable, torch.Tensor] = {}
        self.cu_full_len_kk: Dict[Hashable, torch.Tensor] = {}
        self.cu_window_len_kk: Dict[Hashable, torch.Tensor] = {}
        self.max_full_lens: Dict[Hashable, int] = {}
        self.max_window_lens: Dict[Hashable, int] = {}

        # Rotary position buffers shared across graphs. Allocate to the maximum
        # raw budget on first use so earlier captured graphs never hold stale
        # pointers after a later, larger budget is captured.
        self.sin_cos_ws: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.max_context_len = max(
            int(getattr(vit, "max_context_len", 0) or 0), self.max_raw_budget
        )

        # Qwen2.5-VL specific variables.
        fullatt_block_indexes = getattr(vit, "fullatt_block_indexes", ())
        if isinstance(fullatt_block_indexes, torch.Tensor):
            fullatt_block_indexes = fullatt_block_indexes.tolist()
        self._fullatt_block_indexes = set(int(x) for x in fullatt_block_indexes)

        # Qwen3-VL specific variables.
        self._deepstack_visual_indexes = list(
            getattr(vit, "deepstack_visual_indexes", []) or []
        )
        self._deepstack_merger_list = getattr(vit, "deepstack_merger_list", None)

        self.capture_completed = False
        self.graph_hit_count = 0
        self.graph_hit_by_budget = defaultdict(int)
        self.eager_fallback_count = 0
        self.eager_fallback_reasons = defaultdict(int)

    @property
    def device(self) -> torch.device:
        return self.vit.device

    @property
    def dtype(self) -> torch.dtype:
        return self.vit.dtype

    @property
    def max_full_cu_len(self) -> int:
        # Qwen2.5 has an extra leading empty sequence: [0, 0, ...].
        return self.max_batch_size + (3 if self._fullatt_block_indexes else 2)

    @property
    def max_window_cu_len(self) -> int:
        # Window metadata is per post-merge window and is upper-bounded by the
        # post-merge budget when every window contains one merged token.
        return self.max_vision_budget + 2

    def log_capture_config(self) -> None:
        logger.info(
            "ViT CUDA graph capture config: vision token budgets=%s, raw token budgets=%s, "
            "max batch size=%s",
            self.vision_token_budgets,
            self.raw_token_budgets,
            self.max_batch_size,
        )

    def record_fallback(self, reason: str) -> None:
        self.eager_fallback_count += 1
        self.eager_fallback_reasons[reason] += 1
        if self.eager_fallback_count <= 5 or self.eager_fallback_count % 100 == 0:
            logger.info(
                "ViT CUDA graph fallback stats: fallbacks=%s, reasons=%s, hits=%s",
                self.eager_fallback_count,
                dict(self.eager_fallback_reasons),
                self.graph_hit_count,
            )

    def _record_hit(self, graph_key: int) -> None:
        self.graph_hit_count += 1
        self.graph_hit_by_budget[graph_key] += 1
        if self.graph_hit_count <= 5 or self.graph_hit_count % 100 == 0:
            logger.info(
                "ViT CUDA graph replay stats: hits=%s, raw_budget_hits=%s, fallbacks=%s",
                self.graph_hit_count,
                dict(self.graph_hit_by_budget),
                self.eager_fallback_count,
            )

    def select_budget(self, total_vision_tokens: int) -> Optional[Tuple[int, int]]:
        for vision_budget, raw_budget in zip(
            self.vision_token_budgets, self.raw_token_budgets
        ):
            if total_vision_tokens <= vision_budget:
                return vision_budget, raw_budget
        return None

    def _ensure_sin_cos_ws(
        self, seq_len: int, head_dim: int, sin_cos_dtype: torch.dtype
    ) -> None:
        need_new = self.sin_cos_ws is None
        if not need_new:
            cos_ws, _ = self.sin_cos_ws
            need_new = (
                cos_ws.size(0) < seq_len
                or cos_ws.size(1) != head_dim
                or cos_ws.dtype != sin_cos_dtype
            )

        if need_new:
            current_size = 0 if self.sin_cos_ws is None else self.sin_cos_ws[0].size(0)
            max_shape = max(self.max_context_len or 0, current_size * 2, seq_len)
            cos_ws = torch.empty(
                max_shape, head_dim, dtype=sin_cos_dtype, device=self.device
            )
            sin_ws = torch.empty(
                max_shape, head_dim, dtype=sin_cos_dtype, device=self.device
            )
            self.sin_cos_ws = (cos_ws, sin_ws)

    def _get_graph_key(self, x_3d: torch.Tensor) -> int:
        # x_3d: [S, B=1, H], S is the raw graph budget.
        return x_3d.shape[0]

    def _prepare_input_buffer(self, graph_key: int, x_3d: torch.Tensor) -> None:
        if graph_key in self.block_input:
            return

        if graph_key <= self.max_raw_budget:
            if (
                self.input_buffer is None
                or self.input_buffer.dtype != x_3d.dtype
                or self.input_buffer.device != x_3d.device
                or self.input_buffer.shape[1:] != x_3d.shape[1:]
            ):
                self.input_buffer = torch.empty(
                    self.max_raw_budget,
                    x_3d.shape[1],
                    x_3d.shape[2],
                    dtype=x_3d.dtype,
                    device=x_3d.device,
                ).contiguous()
            self.block_input[graph_key] = self.input_buffer[:graph_key, :, :]
        else:
            self.block_input[graph_key] = torch.empty_like(
                x_3d, device=self.device
            ).contiguous()

    def _store_cu_buffers(
        self,
        graph_key: int,
        cu_seqlens: torch.Tensor,
        cu_window_seqlens: Optional[torch.Tensor],
        max_full_len: Optional[int],
        max_window_len: Optional[int],
    ) -> None:
        cu_full = cu_seqlens.to(device=self.device, dtype=torch.int32).contiguous()
        self.cu_full_len[graph_key] = cu_full.clone()
        self.cu_full_len_kk[graph_key] = (
            self.cu_full_len[graph_key][1:] - self.cu_full_len[graph_key][:-1]
        ).contiguous()
        self.max_full_lens[graph_key] = int(
            max_full_len
            if max_full_len is not None
            else self.cu_full_len_kk[graph_key].max().item()
        )

        if self._fullatt_block_indexes:
            if cu_window_seqlens is None:
                raise RuntimeError(
                    "cu_window_seqlens is required for windowed ViT CUDA graph"
                )
            cu_window = cu_window_seqlens.to(
                device=self.device, dtype=torch.int32
            ).contiguous()
            self.cu_window_len[graph_key] = cu_window.clone()
            self.cu_window_len_kk[graph_key] = (
                self.cu_window_len[graph_key][1:]
                - self.cu_window_len[graph_key][:-1]
            ).contiguous()
            self.max_window_lens[graph_key] = int(
                max_window_len
                if max_window_len is not None
                else self.cu_window_len_kk[graph_key].max().item()
            )

    def _update_cu_buffers(
        self,
        graph_key: int,
        cu_seqlens: Optional[torch.Tensor],
        cu_window_seqlens: Optional[torch.Tensor],
    ) -> None:
        if cu_seqlens is not None:
            cu_full = cu_seqlens.to(device=self.device, dtype=torch.int32)
            if cu_full.numel() != self.cu_full_len[graph_key].numel():
                raise RuntimeError(
                    f"cu_seqlens length mismatch for ViT CUDA graph replay: "
                    f"got {cu_full.numel()}, expected {self.cu_full_len[graph_key].numel()}"
                )
            self.cu_full_len[graph_key].copy_(cu_full)
            self.cu_full_len_kk[graph_key].copy_(cu_full[1:] - cu_full[:-1])

        if self._fullatt_block_indexes and cu_window_seqlens is not None:
            cu_window = cu_window_seqlens.to(device=self.device, dtype=torch.int32)
            if cu_window.numel() != self.cu_window_len[graph_key].numel():
                raise RuntimeError(
                    f"cu_window_seqlens length mismatch for ViT CUDA graph replay: "
                    f"got {cu_window.numel()}, expected {self.cu_window_len[graph_key].numel()}"
                )
            self.cu_window_len[graph_key].copy_(cu_window)
            self.cu_window_len_kk[graph_key].copy_(
                cu_window[1:] - cu_window[:-1]
            )

    def _create_graph(
        self,
        graph_key: int,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # (cos, sin), [S, D]
        rotary_pos_emb_cos: Optional[torch.Tensor] = None,
        rotary_pos_emb_sin: Optional[torch.Tensor] = None,
    ) -> None:
        graph = torch.cuda.CUDAGraph()
        vit = self.vit

        if self._fullatt_block_indexes:
            cu_window = self.cu_window_len[graph_key]
            cu_window_kk = self.cu_window_len_kk[graph_key]
            max_window_len = self.max_window_lens[graph_key]

        cu_full = self.cu_full_len[graph_key]
        cu_full_kk = self.cu_full_len_kk[graph_key]
        max_full_len = self.max_full_lens[graph_key]

        override_backend = get_global_server_args().mm_attention_backend

        tp_group = get_tp_group()
        ca_comm = tp_group.ca_comm
        capture_ctx = ca_comm.capture() if ca_comm is not None else nullcontext()

        with capture_ctx, torch.cuda.graph(graph):
            y = None
            deepstack_outs: List[torch.Tensor] = []
            deepstack_capture_idx = 0

            for layer_num, blk in enumerate(vit.blocks):
                if self._fullatt_block_indexes:
                    if layer_num in self._fullatt_block_indexes:
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

                if override_backend == "triton_attn":
                    cu_seq_len_ws = [cu_seqlens_now, cu_seqlens_kk_now, max_len]
                elif override_backend == "fa3":
                    cu_seq_len_ws = [cu_seqlens_now, max_len]
                else:
                    raise RuntimeError("Not supported ViT attention backend")

                if position_embeddings is not None:
                    if layer_num == 0:
                        y = blk(
                            self.block_input[graph_key],
                            cu_seqlens=cu_seq_len_ws,
                            position_embeddings=position_embeddings,
                        )
                    else:
                        y = blk(
                            y,
                            cu_seqlens=cu_seq_len_ws,
                            position_embeddings=position_embeddings,
                        )
                elif rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
                    if layer_num == 0:
                        y = blk(
                            self.block_input[graph_key],
                            cu_seqlens=cu_seq_len_ws,
                            rotary_pos_emb_cos=rotary_pos_emb_cos,
                            rotary_pos_emb_sin=rotary_pos_emb_sin,
                        )
                    else:
                        y = blk(
                            y,
                            cu_seqlens=cu_seq_len_ws,
                            rotary_pos_emb_cos=rotary_pos_emb_cos,
                            rotary_pos_emb_sin=rotary_pos_emb_sin,
                        )
                else:
                    raise RuntimeError("ViT CUDA graph requires rotary embeddings")

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
        max_full_len: Optional[int] = None,
        max_window_len: Optional[int] = None,
    ) -> int:
        graph_key = self._get_graph_key(x_3d)

        if graph_key in self.block_graphs:
            return graph_key

        self._prepare_input_buffer(graph_key, x_3d)
        self.block_input[graph_key].copy_(x_3d)
        self._store_cu_buffers(
            graph_key,
            cu_seqlens,
            cu_window_seqlens,
            max_full_len,
            max_window_len,
        )

        if position_embeddings is not None:
            head_dim = position_embeddings[0].shape[1]
            sin_cos_dtype = position_embeddings[0].dtype
            self._ensure_sin_cos_ws(graph_key, head_dim, sin_cos_dtype)

            used_cos_ws = self.sin_cos_ws[0][:graph_key, :]
            used_sin_ws = self.sin_cos_ws[1][:graph_key, :]
            used_cos_ws.copy_(position_embeddings[0])
            used_sin_ws.copy_(position_embeddings[1])
            persist_position_embeddings = (used_cos_ws, used_sin_ws)
            torch.cuda.synchronize()
            self._create_graph(
                graph_key=graph_key, position_embeddings=persist_position_embeddings
            )
        elif rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            head_dim = rotary_pos_emb_cos.shape[1]
            sin_cos_dtype = rotary_pos_emb_cos.dtype
            self._ensure_sin_cos_ws(graph_key, head_dim, sin_cos_dtype)

            used_cos_ws = self.sin_cos_ws[0][:graph_key, :]
            used_sin_ws = self.sin_cos_ws[1][:graph_key, :]
            used_cos_ws.copy_(rotary_pos_emb_cos)
            used_sin_ws.copy_(rotary_pos_emb_sin)
            torch.cuda.synchronize()
            self._create_graph(
                graph_key=graph_key,
                position_embeddings=None,
                rotary_pos_emb_cos=used_cos_ws,
                rotary_pos_emb_sin=used_sin_ws,
            )
        else:
            raise RuntimeError("ViT CUDA graph requires rotary embeddings")

        return graph_key

    def replay(
        self,
        graph_key: int,
        x_3d: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rotary_pos_emb_cos: Optional[torch.Tensor] = None,
        rotary_pos_emb_sin: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        cu_window_seqlens: Optional[torch.Tensor] = None,
        output_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if graph_key not in self.block_graphs:
            raise RuntimeError(f"ViT CUDA graph for raw budget {graph_key} is missing")

        self._update_cu_buffers(graph_key, cu_seqlens, cu_window_seqlens)

        if position_embeddings is not None:
            head_dim = position_embeddings[0].shape[1]
            sin_cos_dtype = position_embeddings[0].dtype
            self._ensure_sin_cos_ws(graph_key, head_dim, sin_cos_dtype)
            used_cos_ws = self.sin_cos_ws[0][:graph_key, :]
            used_sin_ws = self.sin_cos_ws[1][:graph_key, :]
            used_cos_ws.copy_(position_embeddings[0])
            used_sin_ws.copy_(position_embeddings[1])
        elif rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            head_dim = rotary_pos_emb_cos.shape[1]
            sin_cos_dtype = rotary_pos_emb_cos.dtype
            self._ensure_sin_cos_ws(graph_key, head_dim, sin_cos_dtype)
            used_cos_ws = self.sin_cos_ws[0][:graph_key, :]
            used_sin_ws = self.sin_cos_ws[1][:graph_key, :]
            used_cos_ws.copy_(rotary_pos_emb_cos)
            used_sin_ws.copy_(rotary_pos_emb_sin)

        self.block_input[graph_key].copy_(x_3d)
        self.block_graphs[graph_key].replay()
        self._record_hit(graph_key)

        out = self.block_output[graph_key]
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
    ) -> torch.Tensor:
        # Compatibility path for direct callers. The Qwen image path pre-captures
        # graphs at launch and calls replay() with padded budget tensors.
        x_3d = x.unsqueeze(1)
        graph_key = self._get_graph_key(x_3d)

        if graph_key not in self.block_graphs:
            self.create_graph(
                x_3d=x_3d,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
                cu_window_seqlens=cu_window_seqlens,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
            )

        return self.replay(
            graph_key=graph_key,
            x_3d=x_3d,
            position_embeddings=position_embeddings,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            cu_seqlens=cu_seqlens,
            cu_window_seqlens=cu_window_seqlens,
            output_indices=output_indices,
        )
