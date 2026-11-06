# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 sparse verify host binding: dense/sparse/split graph policy and views."""

import logging
from dataclasses import replace

import torch

from sglang.kernels.plugin_kernels.nvidia.spec import Step
from sglang.srt.kernel_plugins.nvidia.kimi_k31_verify_binding import (
    K31VerifyHostBinding,
)
from sglang.srt.layers.attention.linear.utils import (
    select_verify_intermediate_state_indices,
)
from sglang.srt.layers.dcp.layout import localize_dcp_indices
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_executor.runner_utils.capture_mode import (
    get_capture_attention_variant,
)
from sglang.srt.runtime_context import get_parallel


def _attn_dcp():
    """(size, rank) of the attention DCP group, or (1, 0) when none is published."""
    parallel = get_parallel()
    try:
        return parallel.attn_dcp_size, parallel.attn_dcp_rank
    except (RuntimeError, ValueError):
        return 1, 0


logger = logging.getLogger(__name__)


class K31SparseVerifyGraphVariants:
    def __init__(self, query_width, split_min=0, context_length=None):
        self.query_width = query_width
        dcp = _attn_dcp()[0] > 1
        # Split from this longest length (0: never); not under DCP (replicated index-K)
        # or past the context length.
        short = context_length is not None and context_length < split_min
        self.split_min = 0 if dcp or short else split_min
        if dcp:  # the dense core reads the whole latent cache
            self.capture_labels = ("sparse",)
        else:
            self.capture_labels = ("dense", "sparse") + (
                ("split",) if self.split_min else ()
            )

    def select(self, forward_batch):
        if _attn_dcp()[0] > 1:
            return "sparse"  # no dense variant under DCP
        lengths = forward_batch.seq_lens_cpu
        if lengths is None or lengths.numel() == 0:
            return "sparse"
        # seq_lens_cpu already includes this verify block.
        longest = int(lengths.max())
        if longest <= 2048:
            label = "dense"
        elif self.split_min and longest >= self.split_min:
            label = "split"
        else:
            label = "sparse"
        return label


class K31SparseVerifyHostBinding(K31VerifyHostBinding):
    capture_stock_fallback = False

    def weights_view(self, context_length, *, query_width=8):
        base = super().weights_view(context_length, query_width=query_width)
        self.request_batches = base.request_batches
        # optional plugin: imported only when the megakernel is installed
        from sgl_nvidia_mega.kimi_k31.dsa.split import SPLIT_MIN

        self.graph_policy = K31SparseVerifyGraphVariants(
            query_width, split_min=SPLIT_MIN, context_length=context_length
        )
        layers = self.model.language_model.model.layers
        records = []
        for record in base.layers:
            if record.attention_kind == 0:
                indexer = layers[record.layer_id].self_attn.indexer
                if indexer.num_init_tokens or indexer.num_local_tokens:
                    raise ValueError("sparse verify requires pure score-based top2048")
                record = replace(
                    record,
                    attention=record.attention
                    + (indexer.wq_b.weight, indexer.weights_proj.weight),
                )
            records.append(record)
        return replace(base, layers=tuple(records))

    def pool_view(self, pool, req_pool):
        view = super().pool_view(pool, req_pool)
        # Allocated before capture; every graph shares it (graphs replay serially).
        tokens = req_pool.req_to_token
        pages = (tokens.shape[1] + pool.page_size - 1) // pool.page_size
        self.request_page_table = torch.empty(
            max(self.request_batches) * pages,
            dtype=torch.int32,
            device=tokens.device,
        )
        return view

    def graph_variants(self):
        return self.graph_policy

    def can_run_graph(self, built, forward_batch):
        # optional plugin: imported only when the megakernel is installed
        from sgl_nvidia_mega.kimi_k31.verify_contract import request_capacity

        return (
            forward_batch.forward_mode.is_target_verify()
            and request_capacity(int(forward_batch.batch_size), built.request_batches)
            is not None
        )

    def plan(self, built, input_ids, forward_batch, *args, **kwargs):
        if not self.can_run_graph(built, forward_batch):
            return None
        if kwargs.get("get_embedding", args[0] if args else False):
            return None
        b, r = int(forward_batch.batch_size), int(input_ids.shape[0])
        if not b or r % b or not built.supports(b, r // b, r):
            return None
        if (
            forward_batch.spec_info is None
            or forward_batch.ragged_verify_layout is not None
        ):
            return None
        return Step(mode="target_verify", rows=r)

    def forward(
        self, built, step, input_ids, positions, forward_batch, *args, **kwargs
    ):
        # optional plugin: imported only when the megakernel is installed
        from sgl_nvidia_mega.kimi_k31.dsa.verify import SparseVerifyStep
        from sgl_nvidia_mega.kimi_k31.verify_binding import KdaVerifyStep

        backend = get_attn_backend()
        full = backend.full_attn_backend.forward_metadata
        linear = backend.linear_attn_backend
        metadata = linear.forward_metadata
        b = int(forward_batch.batch_size)
        q = step.rows // b
        slots = metadata.mamba_cache_indices[:b]
        scratch = select_verify_intermediate_state_indices(
            linear.verify_intermediate_state_indices,
            forward_batch.req_pool_indices,
            slots >= 0,
            linear.req_to_token_pool.size,
        )[:b]
        if self.aux is None:
            raise RuntimeError(
                "K31 verify aux capture was not configured before graph warmup"
            )
        kda = KdaVerifyStep(
            slots, scratch, (slots >= 0).to(torch.int32), b, q, step.rows
        )
        label = get_capture_attention_variant() or self.graph_policy.select(
            forward_batch
        )
        if label not in self.graph_policy.capture_labels:
            raise ValueError(f"unexpected sparse verify graph label {label!r}")
        # The stock page table repeats each request q times; keep one row per request.
        pages = full.real_page_table.shape[1]
        table = self.request_page_table[: b * pages].view(b, pages)
        table.copy_(full.real_page_table[: step.rows : q])
        dcp = {}
        size, rank = _attn_dcp()
        if size > 1:
            # The page tables are already rank-local; contexts are this rank's [b, q]
            # share, and a written token's latent row lives on its owning rank only.
            dcp = dict(
                dcp_size=size,
                dcp_rank=rank,
                local_contexts=full.paged_mqa_ctx_lens_2d.reshape(b, q).contiguous(),
                local_slots=localize_dcp_indices(
                    forward_batch.out_cache_loc[: step.rows], size, rank
                ),
            )
        view = SparseVerifyStep(
            table,
            full.cache_seqlens_int32[:b],
            forward_batch.out_cache_loc[: step.rows],
            kda,
            self.aux[: step.rows],
            b,
            q,
            step.rows,
            causal_lengths=full.dsa_seqlens_expanded[: step.rows],
            score_contexts=full.cache_seqlens_int32[:b]
            .view(b, 1)
            .expand(b, q)
            .contiguous(),
            score_schedule=full.paged_mqa_schedule_metadata,
            topk_plan=full.topk_v2_plan,
            sparse=int(label in ("sparse", "split")),
            split=int(label == "split"),
            **dcp,
        )
        lm = self.model.language_model
        embeddings = lm.model.embed_tokens(input_ids)
        hidden, aux = built.run_verify(embeddings, view)
        return lm.logits_processor(input_ids, hidden, lm.lm_head, forward_batch, aux)
