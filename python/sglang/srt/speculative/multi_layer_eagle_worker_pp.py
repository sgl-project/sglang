"""Pipeline-parallel multi-layer EAGLE worker for the Spec V2 scheduler.

The first PP stage builds the MTP tree, every stage runs its target slice, and
the last stage performs sampling and draft-extend.  Tree metadata is carried
inside the normal hidden-state proxy; accepted tokens and the next draft seed
use the existing PP output ring.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.moe.utils import speculative_moe_backend_context
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_worker_common import run_eagle_verify
from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
    MultiLayerEagleWorkerV2,
)


_TREE_FIELDS = (
    "draft_token",
    "custom_mask",
    "positions",
    "retrieve_index",
    "retrieve_next_token",
    "retrieve_next_sibling",
)
_PREFIX = "spec_"


class MultiLayerEagleWorkerPP(MultiLayerEagleWorkerV2):
    """Non-overlap PP worker for non-chain multi-layer MTP (MiMo V2)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        arch = self.draft_worker.draft_runner.model_config.hf_config.architectures[0]
        if self.draft_worker.chain_mtp_hidden_states:
            raise ValueError(
                "PP multi-layer EAGLE currently supports non-chain MiMo MTP only; "
                f"draft architecture {arch!r} uses chain hidden states"
            )

    def init_lm_head(self):
        # The draft model executes only on the final target stage. Earlier
        # stages may expose PPMissingLayer for the target embedding/head, so
        # avoid trying to share those placeholders into an unused draft.
        if not get_pp_group().is_last_rank:
            return
        super().init_lm_head()

    def init_cuda_graphs(self):
        # Draft-extend graphs are only meaningful on the stage that executes
        # the draft model. Target graphs remain initialized by the scheduler.
        if not get_pp_group().is_last_rank:
            self.draft_worker.cuda_graph_runner_for_draft_extend = None
            return
        super().init_cuda_graphs()

    def _serialize_tree(self, spec_info: EagleVerifyInput, proxy: PPProxyTensors):
        for field in _TREE_FIELDS:
            proxy.tensors[_PREFIX + field] = getattr(spec_info, field)

    def _deserialize_tree(
        self, proxy: PPProxyTensors, batch: ScheduleBatch
    ) -> EagleVerifyInput:
        tensors = proxy.tensors
        values = {field: tensors.pop(_PREFIX + field) for field in _TREE_FIELDS}
        seq_lens_cpu = batch.seq_lens_cpu
        return EagleVerifyInput(
            **values,
            retrieve_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.speculative_num_draft_tokens,
            capture_hidden_mode=None,
            seq_lens_sum=(
                int(seq_lens_cpu.sum()) if seq_lens_cpu is not None else None
            ),
            seq_lens_cpu=seq_lens_cpu,
        )

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish=None,
        grammar_barrier=None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> GenerationBatchResult:
        pp = get_pp_group()

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            result = self.target_worker.forward_batch_generation(
                batch,
                pp_proxy_tensors=pp_proxy_tensors,
                capture_hidden_mode=self._target_capture_mode(),
            )
            result.new_seq_lens = batch.seq_lens
            if on_publish is not None:
                on_publish(result.new_seq_lens)
            if not pp.is_last_rank:
                return result

            with (
                self.draft_worker.draft_tp_context(
                    self.draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
            ):
                result.next_draft_input = self.draft_worker._draft_extend_for_prefill(
                    batch,
                    result.logits_output.hidden_states,
                    result.next_token_ids,
                )
            return result

        if pp.is_first_rank:
            with (
                self.draft_worker.draft_tp_context(
                    self.draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
            ):
                verify_input = self.draft_worker.draft(batch)
            incoming = None
        else:
            if pp_proxy_tensors is None:
                raise RuntimeError("PP multi-layer EAGLE requires a verify proxy")
            verify_input = self._deserialize_tree(pp_proxy_tensors, batch)
            incoming = pp_proxy_tensors

        batch.spec_info = verify_input
        if not pp.is_last_rank:
            # Keep the allocation context until the output ring returns the
            # last stage's acceptance decision.
            batch.spec_pp_verify_ctx = (
                batch.out_cache_loc,
                batch.seq_lens,
                batch.seq_lens_cpu,
            )
        result = run_eagle_verify(
            batch,
            target_worker=self.target_worker,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            plan_stream=self.plan_stream,
            plan_stream_ctx=self.plan_stream_ctx,
            topk=self.topk,
            num_draft_tokens=self.speculative_num_draft_tokens,
            device=self.device,
            metadata_ready_pre_pad=False,
            finalize_tree_path=False,
            grammar_barrier=grammar_barrier,
            pp_proxy_tensors=incoming,
        )

        if not pp.is_last_rank:
            if result.pp_hidden_states_proxy_tensors is None:
                raise RuntimeError("target PP stage did not return a hidden proxy")
            self._serialize_tree(verify_input, result.pp_hidden_states_proxy_tensors)
            return result

        if on_publish is not None:
            on_publish(result.new_seq_lens)
        with (
            self.draft_worker.draft_tp_context(
                self.draft_worker.draft_runner.tp_group
            ),
            speculative_moe_backend_context(),
        ):
            self.draft_worker._draft_extend_for_decode(batch, result)
        return result

    def _target_capture_mode(self):
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        return CaptureHiddenMode.FULL

    def reconcile_after_verify(
        self, batch: ScheduleBatch, result: GenerationBatchResult
    ) -> None:
        """Release rejected target-verify slots on non-last PP stages.

        The scheduler processes the last stage's output one loop iteration
        later.  Until then, intermediate stages have allocated the complete
        speculative verify width, so replay the accepted index mask before
        their next decode.  The initial CUDA implementation is deliberately
        restricted to topk=1/page_size=1; wider paged trees need the same page
        shuffle used by the target verify finalizer.
        """
        if get_pp_group().is_last_rank:
            return
        ctx = getattr(batch, "spec_pp_verify_ctx", None)
        indices = result.spec_accept_indices
        if ctx is None or indices is None or result.accept_lens is None:
            return
        batch.spec_pp_verify_ctx = None
        out_cache_loc, seq_lens, seq_lens_cpu = ctx
        page_size = self.page_size
        if page_size != 1 or self.topk != 1:
            raise RuntimeError(
                "PP multi-layer EAGLE reconciliation currently requires "
                "page_size=1 and speculative_eagle_topk=1"
            )
        accepted = indices.reshape(-1)
        accepted = accepted[accepted >= 0].to(torch.long)
        evict = torch.ones_like(out_cache_loc, dtype=torch.bool)
        evict[accepted] = False
        self.token_to_kv_pool_allocator.free(out_cache_loc[evict])

        # Keep scheduler-side device/host sequence mirrors in lockstep with
        # the sampled last stage. The normal V2 output processor commits the
        # corresponding request tokens; these mirrors drive the next KV plan.
        delta = result.accept_lens.to(seq_lens.device)
        seq_lens.add_(delta)
        if seq_lens_cpu is not None:
            seq_lens_cpu.add_(result.accept_lens)
            batch.seq_lens_sum = int(seq_lens_cpu.sum())
