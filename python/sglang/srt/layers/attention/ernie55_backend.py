# Copyright 2026 Baidu ERNIE Team and The SGLang Team. All rights reserved.
#
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
"""Attention backend for ERNIE 5.5 Flash.

The HCA layers (``compress_ratio == 128``) run the compressed-KV backend this
class inherits unchanged. The MLA layers run a DSA sub-backend over
the latent sub-pool of :class:`Ernie5_5TokenToKVPool`; inside
:meth:`Ernie5_5AttnBackend.mla_layer_context` it is the forward-context backend,
so the DSA attention and its lightning indexer resolve to it like in a plain DSA
model.
"""

from __future__ import annotations

import contextlib
import dataclasses
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.attention.dsv4.sparse_prefill_kernels import (
    SPARSE_PREFILL_TOPK_ALIGNMENT,
)
from sglang.srt.layers.attention.base_attn_backend import SharedReadEnds
from sglang.srt.layers.attention.deepseek_v4_backend import (
    DEFAULT_INDEX_TOPK,
    DeepseekV4AttnBackend,
    DeepseekV4MultiStepBackend,
)
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.mem_cache.ernie55_memory_pool import (
    DSA_PAGE_SIZE,
    Ernie5_5TokenToKVPool,
)
from sglang.srt.model_executor.forward_context import (
    forward_context,
    get_forward_context,
)
from sglang.srt.state_capturer.indexer_topk import get_global_indexer_capturer

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.forward_mode import ForwardMode
    from sglang.srt.model_executor.model_runner import ModelRunner


class _MlaSubBackendView:
    """Reads through to the wrapped object except for the replaced fields: for
    the MLA layers, the latent sub-pool, its 64-token page size, and a model
    config carrying the ``hybrid_mla_*`` dims."""

    def __init__(self, base, **replaced):
        self._base = base
        self.__dict__.update(replaced)

    def __getattr__(self, name):
        return getattr(self._base, name)


def _topk_lengths(seq_lens: torch.Tensor, window: int) -> torch.Tensor:
    """Indexer candidates per query: everything before the trailing window."""
    return (seq_lens - window).clamp_min(0)


class Ernie5_5MlaAttnBackend(DeepseekSparseAttnBackend):
    """DSA over ``top-k(prefix before the window) + the last window`` tokens."""

    def __init__(self, model_runner: ModelRunner, *, window: int, **kwargs):
        self.window = window
        super().__init__(model_runner, **kwargs)
        # Learnable sinks are only wired into flashmla_sparse.
        self.dsa_prefill_impl = "flashmla_sparse"
        self.dsa_decode_impl = "flashmla_sparse"
        self.enable_auto_select_prefill_impl = False
        # The dense MHA prefill path takes no learnable sink.
        self.supports_mha_one_shot = False
        if self.dsa_index_topk <= window:
            raise ValueError(
                f"ERNIE 5.5 Flash needs index_topk ({self.dsa_index_topk}) greater "
                f"than the sliding window ({window})"
            )
        self._sched_meta_key = None
        self._eager_sched_meta = None
        # The schedules of the graph being captured, keyed by shape.
        self._graph_sched_meta = {}
        # Every captured graph's schedules, which its replays keep reading.
        self._captured_sched_meta = []
        # Bool mask of the rows whose output is read; set only inside read_rows().
        self._read_rows: Optional[torch.Tensor] = None

    @contextlib.contextmanager
    def read_rows(self, rows: Optional[torch.Tensor], num_rows: int):
        """Compute only ``rows``; the other rows output zero. KV and index-K are
        still written for all."""
        # A captured top-k must stay complete.
        if rows is not None and get_global_indexer_capturer() is None:
            mask = torch.zeros(num_rows, dtype=torch.bool, device=rows.device)
            self._read_rows = mask.index_fill_(0, rows, True)
        yield
        self._read_rows = None

    def _build_topk_v2_plan(self, seqlens_expanded: torch.Tensor):
        return super()._build_topk_v2_plan(_topk_lengths(seqlens_expanded, self.window))

    def _refresh_topk_v2_plan(self, metadata) -> None:
        # Graph replay re-plans in place; it must see the same shortened lengths.
        if metadata.topk_v2_plan is None:
            return
        from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2

        metadata.topk_v2_plan.copy_(
            plan_topk_v2(_topk_lengths(metadata.dsa_seqlens_expanded, self.window))
        )

    def get_indexer_metadata(self, layer_id: int, forward_batch: ForwardBatch):
        metadata = super().get_indexer_metadata(layer_id, forward_batch)
        attn_metadata = metadata.attn_metadata
        # The indexer's view only: attention keeps the full lengths for the window.
        lengths = _topk_lengths(attn_metadata.dsa_seqlens_expanded, self.window)
        k_range = attn_metadata.indexer_k_start_end
        keep = self._read_rows
        if keep is not None:
            # Unread rows get no candidates and an empty key range, so top-k is all -1.
            ks, ke = k_range
            lengths = lengths.masked_fill(~keep, 0)
            k_range = (ks, torch.where(keep, ke, ks))
        return dataclasses.replace(
            metadata,
            attn_metadata=dataclasses.replace(
                attn_metadata,
                dsa_seqlens_expanded=lengths,
                indexer_k_start_end=k_range,
            ),
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch) -> None:
        self._sched_meta_key = None
        super().init_forward_metadata(forward_batch)

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ) -> None:
        self._sched_meta_key = None
        if in_capture:
            # Two graphs with the same shape must not share one schedule: the
            # second would record no build and replay the first one's.
            self._graph_sched_meta = {}
        super().init_forward_metadata_out_graph(forward_batch, in_capture=in_capture)

    def _forward_flashmla_sparse(
        self,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        v_head_dim: int,
        page_table_1: torch.Tensor,
        sm_scale: float,
        topk_length: Optional[torch.Tensor] = None,
        attn_sink: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Attention reads ``top-k plus the last window``; a bf16 cache here is
        a bf16 pool or prefill's dequantized copy, and takes the parent's kernel."""
        from sglang.kernels.ops.attention.dsa.append_window_to_topk import (
            append_window_to_topk,
        )

        metadata = self.forward_metadata
        num_rows = q_all.shape[0]
        page_table_1, topk_length = append_window_to_topk(
            page_table_1[:num_rows].to(torch.int32),
            metadata.dsa_seqlens_expanded,
            metadata.real_page_table,
            window=self.window,
            page_size=metadata.page_size,
            alignment=SPARSE_PREFILL_TOPK_ALIGNMENT,
            page_rows=metadata.token_to_batch_idx,
            ragged_offsets=metadata.topk_indices_offset,
        )
        keep = self._read_rows
        if keep is not None and kv_cache.dtype == torch.bfloat16:
            topk_length = topk_length.masked_fill(~keep, 0)
        if kv_cache.dtype == torch.bfloat16 and q_all.shape[1] == 64:
            # The parent pads q to 128 heads on Blackwell; the 64-head build skips
            # that copy and half the work.
            o, _, _ = self._call_flash_mla_sparse_fwd(
                q=q_all,
                kv=kv_cache,
                indices=page_table_1.unsqueeze(1),
                sm_scale=sm_scale,
                d_v=v_head_dim,
                attn_sink=attn_sink,
                topk_length=topk_length,
            )
            return o
        if kv_cache.dtype == torch.bfloat16:
            return super()._forward_flashmla_sparse(
                q_all=q_all,
                kv_cache=kv_cache,
                v_head_dim=v_head_dim,
                page_table_1=page_table_1,
                sm_scale=sm_scale,
                topk_length=topk_length,
                attn_sink=attn_sink,
            )
        # The sched-meta API is the only FlashMLA entry that reads the packed fp8
        # cache and takes a sink; it runs flashmla_kv_num_q_heads heads.
        from sglang.kernels.ops.attention.flash_mla_interface import (
            flash_mla_with_kvcache,
        )

        if attn_sink is None:
            raise RuntimeError("ERNIE 5.5 Flash MLA layers always carry a sink")
        num_heads, head_dim = q_all.shape[1], q_all.shape[2]
        heads = self.flashmla_kv_num_q_heads
        q, sink = q_all.unsqueeze(1), attn_sink
        if heads != num_heads:
            # The kernel is built for a fixed head count; padded heads get a zero sink.
            q = q_all.new_zeros((num_rows, 1, heads, head_dim))
            q[:, 0, :num_heads] = q_all
            sink = attn_sink.new_zeros(heads)
            sink[:num_heads] = attn_sink
        o, _ = flash_mla_with_kvcache(
            q=q,
            k_cache=kv_cache.view(-1, DSA_PAGE_SIZE, 1, self.kv_cache_dim),
            block_table=None,
            cache_seqlens=None,
            head_dim_v=v_head_dim,
            tile_scheduler_metadata=self._sched_meta(num_rows, page_table_1.shape[-1]),
            softmax_scale=sm_scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=page_table_1.unsqueeze(1),
            attn_sink=sink,
            # The sm90 decode kernel rejects a dynamic topk length; the rows are
            # -1 padded, so None is equivalent there.
            topk_length=None if self.device_sm_major < 10 else topk_length,
        )
        return o[:, 0, :num_heads]

    def _sched_meta(self, num_rows: int, width: int):
        """One FlashMLA schedule per step, shared by the MLA layers.

        The kernel builds only into an empty metadata from per-row lengths,
        so it is reset every step; a captured graph keeps its own objects.
        """
        from sglang.kernels.ops.attention.flash_mla_interface import get_mla_metadata

        key = (num_rows, width)
        if torch.cuda.is_current_stream_capturing():
            # Not setdefault: its default argument would build (and leak into the
            # graph pool) one metadata tensor per call.
            if key not in self._graph_sched_meta:
                meta = get_mla_metadata()[0]
                self._graph_sched_meta[key] = meta
                self._captured_sched_meta.append(meta)
            return self._graph_sched_meta[key]
        if self._sched_meta_key != key:
            self._sched_meta_key = key
            self._eager_sched_meta = get_mla_metadata()[0]
        return self._eager_sched_meta


class Ernie5_5AttnBackend(DeepseekV4AttnBackend):
    def __init__(self, model_runner: ModelRunner, **kwargs):
        super().__init__(model_runner, **kwargs)
        pool = self.token_to_kv_pool
        assert isinstance(pool, Ernie5_5TokenToKVPool), (
            "the model override selects this backend together with the ERNIE pool, "
            f"got {type(pool).__name__}"
        )
        # index_topk sizes the MLA layers' DSA indexer; with no C4 layer (the
        # config rejects it), the HCA metadata only needs a width it accepts.
        self.index_topk = DEFAULT_INDEX_TOPK

        hf_config = model_runner.model_config.hf_config
        mla_model_config = _MlaSubBackendView(
            model_runner.model_config,
            num_attention_heads=hf_config.hybrid_mla_num_attention_heads,
            qk_nope_head_dim=hf_config.hybrid_mla_qk_nope_head_dim,
            qk_rope_head_dim=hf_config.hybrid_mla_qk_rope_head_dim,
            kv_lora_rank=hf_config.hybrid_mla_kv_lora_rank,
        )
        self.mla_attn_backend = Ernie5_5MlaAttnBackend(
            _MlaSubBackendView(
                model_runner,
                model_config=mla_model_config,
                token_to_kv_pool=pool.mla_kv_pool,
                page_size=DSA_PAGE_SIZE,
            ),
            window=hf_config.sliding_window,
            # A multi-step draft step reads its own slice of the shared KV locs.
            speculative_step_id=kwargs.get("speculative_step_id", 0),
            topk=kwargs.get("topk", 0),
            speculative_num_steps=kwargs.get("speculative_num_steps", 0),
        )
        self.mla_attn_backend.prefill_attention_backend_str = "dsa"
        self.mla_attn_backend.decode_attention_backend_str = "dsa"

    @contextlib.contextmanager
    def mla_layer_context(self):
        """Run an MLA layer: publish the MLA sub-backend as the forward-context
        backend, which the DSA attention and indexer read directly."""
        with forward_context(
            dataclasses.replace(
                get_forward_context(), attn_backend=self.mla_attn_backend
            )
        ):
            yield

    def read_rows(self, rows: Optional[torch.Tensor], num_rows: int):
        return self.mla_attn_backend.read_rows(rows, num_rows)

    def init_forward_metadata(self, forward_batch: ForwardBatch) -> None:
        super().init_forward_metadata(forward_batch)
        self.mla_attn_backend.init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:
        super().init_cuda_graph_state(max_bs, max_num_tokens)
        self.mla_attn_backend.init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ) -> None:
        super().init_forward_metadata_out_graph(forward_batch, in_capture=in_capture)
        self.mla_attn_backend.init_forward_metadata_out_graph(
            forward_batch, in_capture=in_capture
        )

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch) -> None:
        super().init_forward_metadata_in_graph(forward_batch)
        self.mla_attn_backend.init_forward_metadata_in_graph(forward_batch)

    def init_forward_metadata_for_breakable_cuda_graph_capture(
        self, forward_batch: ForwardBatch
    ):
        # The MLA layers' metadata would not be refreshed on replay.
        raise NotImplementedError(
            "ERNIE 5.5 Flash does not support breakable prefill CUDA graphs"
        )

    def shared_read_ends(self, fm: ForwardMode) -> SharedReadEnds:
        return SharedReadEnds.max_of(
            (super().shared_read_ends(fm), self.mla_attn_backend.shared_read_ends(fm))
        )


class Ernie5_5MultiStepBackend(DeepseekV4MultiStepBackend):
    """One ``Ernie5_5AttnBackend`` per draft step, so the MLA layer's DSA metadata
    follows each step like the HCA metadata does."""

    def _make_step_backend(
        self, model_runner: ModelRunner, step_id: int
    ) -> Ernie5_5AttnBackend:
        return Ernie5_5AttnBackend(
            model_runner,
            speculative_step_id=step_id,
            topk=self.topk,
            speculative_num_steps=self.speculative_num_steps,
        )

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ) -> None:
        if not in_capture and self.speculative_num_steps == 1:
            return
        from sglang.srt.model_executor.forward_batch_info import build_inner_fb_view
        from sglang.srt.model_executor.forward_mode import ForwardMode

        inner_fb = build_inner_fb_view(
            forward_batch, bs=forward_batch.batch_size, forward_mode=ForwardMode.DECODE
        )
        # Each step builds its own metadata: the parent copies the HCA metadata
        # from step 0, which would leave the MLA sub-backend's stale.
        steps = (
            self.speculative_num_steps if in_capture else self.speculative_num_steps - 1
        )
        for backend in self.attn_backends[:steps]:
            backend.init_forward_metadata_out_graph(inner_fb, in_capture=in_capture)
