from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional

from sglang.srt.layers.attention.linear.utils import pp_spec_stable_rows_enabled
from sglang.srt.managers.overlap_utils import resolve_forward_inputs
from sglang.srt.runtime_context import get_parallel, get_spec
from sglang.srt.speculative.forward_batch_view import worker_forward_batch
from sglang.srt.speculative.pp_spec_relay import PPSpecRelay
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler_components.bundles import (
        KvResources,
        Streams,
        Workers,
    )
    from sglang.srt.managers.scheduler_components.pending_steps import PendingStep
    from sglang.srt.managers.utils import GenerationBatchResult
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors


class PPSpecForward:
    """The speculative forward of one PP stage, and the PP+spec trees the
    output relay keeps by request slot for it."""

    def __init__(
        self,
        *,
        kv: KvResources,
        streams: Streams,
        future_map,
        workers: Workers,
    ):
        pp_group = get_parallel().pp_group
        spec_algorithm = SpeculativeAlgorithm.from_string(
            get_spec().speculative_algorithm
        )
        self.req_to_token_pool = kv.req_to_token_pool
        self.device = streams.device
        self.future_map = future_map
        self.tp_worker = workers.tp_worker
        self.model_worker = workers.model_worker
        self.pp_group = pp_group
        self.spec_algorithm = spec_algorithm
        # The gated PP+spec relay flow's trees, per request slot; None without it.
        self.pp_spec_relay = (
            PPSpecRelay(self.req_to_token_pool.req_to_token.shape[0], self.device)
            if pp_spec_stable_rows_enabled()
            and get_parallel().pp_size > 1
            and not self.spec_algorithm.is_none()
            else None
        )

    def spec_forward(
        self,
        step: ScheduleBatch,
        *,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        pending_grammar: Optional[Iterable[PendingStep]] = None,
    ) -> GenerationBatchResult:
        """Run a speculative step on this stage, on a view of the step. A decode
        step verifies the trees its rows' slots hold, the same tree on every
        stage: a stage before the last runs only its part of the target on it
        (sampling, accept and draft live on the last stage); the last stage
        verifies and drafts the next round's trees for the relay."""
        if step.forward_mode.is_extend() or step.is_extend_in_batch:
            forward_batch = worker_forward_batch(step)
            resolve_forward_inputs(forward_batch, self.future_map)
            return self.model_worker.forward_batch_generation(
                forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
                pending_grammar=pending_grammar,
            )

        verify_input = self._spec_verify_input(step)
        forward_batch = worker_forward_batch(step, spec_info=verify_input)
        if self.pp_group.is_last_rank:
            return self.model_worker.forward_batch_generation(
                forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
                pending_grammar=pending_grammar,
            )

        from sglang.srt.speculative.eagle_utils import eagle_prepare_for_verify

        verify_forward_batch, can_run_cuda_graph = eagle_prepare_for_verify(
            verify_input, self.req_to_token_pool, forward_batch, self.tp_worker
        )
        result = self.tp_worker.forward_model(
            batch=None,
            forward_batch=verify_forward_batch,
            pp_proxy_tensors=pp_proxy_tensors,
            is_verify=True,
        )
        result.can_run_cuda_graph = can_run_cuda_graph
        # This stage still has to compact its own accepted-path KV once the
        # last stage relays which nodes it kept, so the verify slots have to
        # outlive the forward.
        result.spec_verify_out_cache_loc = verify_forward_batch.out_cache_loc
        return result

    def _spec_verify_input(self, step: ScheduleBatch):
        """The verify input of a decode step, built from the trees the relay
        keeps in its rows' slots, without a draft model.

        Every stage, including the last one, builds it, so all stages verify
        the exact same tree. A request that has not been drafted for yet (its
        first decode after prefill) carries its sampled token over zero
        drafts, which simply get rejected, costing acceptance rate, not
        correctness."""
        from sglang.srt.speculative.eagle_info import EagleVerifyInput
        from sglang.srt.speculative.eagle_utils import (
            TreeMaskMode,
            build_tree_kernel_efficient,
            default_tree_mask_mode,
        )

        spec = get_spec()
        steps = spec.speculative_num_steps
        num_draft_tokens = spec.speculative_num_draft_tokens
        bs = step.batch_size()

        if step.forward_mode.is_idle() or bs == 0:
            return EagleVerifyInput.create_idle_input(
                topk=spec.speculative_eagle_topk,
                spec_steps=steps,
                num_verify_tokens=num_draft_tokens,
                device=self.device,
            )

        tree_rows, parent_list, top_scores_index = self.pp_spec_relay.gather(
            step.req_pool_indices
        )
        bonus_tokens = tree_rows[:, 0].contiguous()
        draft_tokens = tree_rows[:, 1:].contiguous()

        # Mask selection mirrors the last stage's draft() tail
        # (build_eagle_verify_input) so every stage builds the same mask.
        verify_mask = self.tp_worker.model_runner.attn_backend.verify_mask
        if verify_mask is None:
            tree_mask_buf, mask_mode, fill_mask = None, default_tree_mask_mode(), True
        else:
            mask_mode, fill_mask = verify_mask.mode, verify_mask.is_read
            tree_mask_buf = verify_mask.buffer if verify_mask.fits(bs) else None

        seq_lens_sum = step.seq_lens_sum
        if seq_lens_sum is None:
            if tree_mask_buf is not None or mask_mode == TreeMaskMode.QLEN_ONLY:
                seq_lens_sum = 0  # preallocated / bs-sized -> kernel ignores it
            else:
                # Conservative upper bound; backend-agnostic (not every
                # attention backend exposes max_context_len).
                seq_lens_sum = bs * self.tp_worker.model_runner.model_config.context_len

        (
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            flat_draft_tokens,
        ) = build_tree_kernel_efficient(
            bonus_tokens,
            parent_list,
            top_scores_index,
            draft_tokens,
            step.seq_lens,
            seq_lens_sum,
            spec.speculative_eagle_topk,
            steps,
            num_draft_tokens,
            mask_mode,
            tree_mask_buf,
            fill_prefix_mask=fill_mask,
        )
        return EagleVerifyInput(
            draft_token=flat_draft_tokens,
            custom_mask=tree_mask,
            positions=positions,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_cum_len=None,
            spec_steps=steps,
            topk=spec.speculative_eagle_topk,
            draft_token_num=num_draft_tokens,
            capture_hidden_mode=None,
            seq_lens_sum=step.seq_lens_sum,
            seq_lens_cpu=step.seq_lens_cpu,
        )
