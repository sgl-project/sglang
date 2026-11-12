# SPDX-License-Identifier: Apache-2.0
"""Host binding for the Kimi K3.1 dense/sparse decode megakernel."""

from __future__ import annotations

from sglang.kernels.plugin_kernels.nvidia.spec import Step

# a module import: binding imports this module back
from sglang.srt.kernel_plugins.nvidia import binding as megakernel_binding
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_executor.runner_utils.capture_mode import (
    get_capture_attention_variant,
)


class K31DecodeGraphVariants:
    # Both variants run the plugin; the order matches the stock DSA capture order.
    capture_labels = ("dense", "sparse")

    def select(self, forward_batch):
        lengths = forward_batch.seq_lens_cpu
        if lengths is None or lengths.numel() == 0:
            return "sparse"
        return "dense" if int(lengths.max()) <= 2048 else "sparse"


class K31DecodeHostBinding:
    def __init__(self, model):
        self.model = model
        self.pools = megakernel_binding.MegakernelBinding(model=model)

    def graph_variants(self):
        return K31DecodeGraphVariants()

    def pool_view(self, pool, req_pool):
        return self.pools.preparation_view(pool, req_pool)

    def can_run_graph(self, built, forward_batch):
        # The graph runner pads live requests to a captured width before forward.
        return (
            forward_batch.forward_mode.is_decode()
            and 0 < forward_batch.batch_size <= max(built.widths)
        )

    def plan(self, built, input_ids, forward_batch, *args, **kwargs):
        rows = int(input_ids.shape[0])
        if (
            not self.can_run_graph(built, forward_batch)
            or rows not in built.widths
            or input_ids.shape != (forward_batch.batch_size,)
        ):
            return None
        if kwargs.get("get_embedding", args[0] if args else False):
            return None
        if (
            kwargs.get("pp_proxy_tensors", args[1] if len(args) > 1 else None)
            is not None
        ):
            return None
        return Step(mode="decode", rows=rows)

    def forward(
        self, built, step, input_ids, positions, forward_batch, *args, **kwargs
    ):
        # optional plugin: imported only when the megakernel is installed
        from sgl_nvidia_mega.kimi_k31.dsa_binding import (
            IndexDecodeStep,
            KdaDecodeStep,
            MlaDecodeStep,
            SparseDecodeStep,
        )

        backend = get_attn_backend()
        rows = step.rows
        full = backend.full_attn_backend.forward_metadata
        linear = backend.linear_attn_backend.forward_metadata
        # Captured graphs bake in their label; eager steps select from current lengths.
        label = get_capture_attention_variant()
        if label is None:
            label = K31DecodeGraphVariants().select(forward_batch)
        if label not in ("dense", "sparse"):
            raise ValueError(f"unexpected K3.1 decode capture variant {label!r}")
        schedule, plan = full.paged_mqa_schedule_metadata, full.topk_v2_plan
        if schedule is None or plan is None:
            raise ValueError(
                "K3.1 sparse decode needs current paged-MQA schedule/top-k plan"
            )
        view = SparseDecodeStep(
            MlaDecodeStep(
                full.real_page_table[:rows],
                full.cache_seqlens_int32[:rows],
                forward_batch.out_cache_loc[:rows],
            ),
            KdaDecodeStep(linear.mamba_cache_indices[:rows]),
            IndexDecodeStep(schedule, plan),
            int(label == "sparse"),
        )
        view.validate()
        lm = self.model.language_model
        embeddings = lm.model.embed_tokens(input_ids)
        hidden = built.run_decode(input_ids, embeddings, view)
        self.pools.finish_step(forward_batch)
        return lm.logits_processor(input_ids, hidden, lm.lm_head, forward_batch, None)
