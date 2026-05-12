import logging
from copy import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.distributed import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    is_dp_attention_enabled,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.managers.overlap_utils import FutureIndices
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.eagle_info_v2 import (
    EagleDraftInputV2Mixin,
    EagleVerifyInputV2Mixin,
)
from sglang.srt.speculative.eagle_utils import verify_tree_greedy_func
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import (
    SIMULATE_ACC_LEN,
    TREE_SPEC_KERNEL_AVAILABLE,
    align_evict_mask_to_page_size,
    assign_req_to_token_pool_func,
    create_extend_after_decode_spec_info,
    create_num_accept_tokens_filter,
    filter_finished_cache_loc_kernel,
    generate_simulated_accept_index,
    get_src_tgt_cache_loc,
    get_target_cache_loc,
)
from sglang.srt.utils import is_cuda, is_musa, next_power_of_2

if is_cuda() or is_musa():
    from sgl_kernel import (
        top_k_renorm_prob,
        top_p_renorm_prob,
        tree_speculative_sampling_target_only,
    )

logger = logging.getLogger(__name__)


def _draft_runner_of(worker):
    """Draft model_runner accessor that handles v1 / v2 worker naming.

    v1 (`EAGLEWorker` and subclasses) exposes the draft model_runner as
    `model_runner` (the worker itself runs the draft model);
    v2 (`EagleDraftWorker` and subclasses) exposes it as `draft_runner`.
    """
    return (
        worker.draft_runner if hasattr(worker, "draft_runner") else worker.model_runner
    )


@dataclass
class EagleVerifyInput(SpecInput, EagleVerifyInputV2Mixin):
    draft_token: torch.Tensor
    custom_mask: torch.Tensor
    positions: torch.Tensor
    retrieve_index: torch.Tensor
    retrieve_next_token: torch.Tensor
    retrieve_next_sibling: torch.Tensor
    retrieve_cum_len: torch.Tensor
    spec_steps: int
    topk: int
    draft_token_num: int
    capture_hidden_mode: CaptureHiddenMode
    seq_lens_sum: int
    seq_lens_cpu: torch.Tensor
    grammar: BaseGrammarObject = None

    # Shape info for padding
    num_tokens_per_req: int = -1  # -1 auto-fills from draft_token_num.

    def __post_init__(self):
        super().__init__(SpecInputType.EAGLE_VERIFY)
        if self.num_tokens_per_req < 0:
            self.num_tokens_per_req = self.draft_token_num

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.draft_token_num, self.draft_token_num

    @classmethod
    def create_idle_input(cls, topk: int, spec_steps: int, num_verify_tokens: int):
        return cls(
            draft_token=torch.empty((0,), dtype=torch.long, device="cuda"),
            custom_mask=torch.full((0,), True, dtype=torch.bool, device="cuda"),
            positions=torch.empty((0,), dtype=torch.int64, device="cuda"),
            retrieve_index=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device="cuda"
            ),
            retrieve_next_token=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device="cuda"
            ),
            retrieve_next_sibling=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device="cuda"
            ),
            retrieve_cum_len=None,
            topk=topk,
            draft_token_num=num_verify_tokens,
            spec_steps=spec_steps,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=0,
            seq_lens_cpu=torch.empty((0,), dtype=torch.int32),
        )

    def prepare_for_verify(self, batch: ScheduleBatch, page_size: int):

        if batch.forward_mode.is_idle():
            return

        batch.input_ids = self.draft_token

        if page_size == 1:
            batch.out_cache_loc = alloc_token_slots(
                batch.tree_cache,
                len(batch.input_ids),
            )
            end_offset = batch.seq_lens + self.draft_token_num
        else:
            prefix_lens = batch.seq_lens
            prefix_lens_cpu = batch.seq_lens_cpu
            end_offset = prefix_lens + self.draft_token_num
            end_offset_cpu = prefix_lens_cpu + self.draft_token_num
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                prefix_lens,
            )
            batch.out_cache_loc = alloc_paged_token_slots_extend(
                batch.tree_cache,
                prefix_lens,
                prefix_lens_cpu,
                end_offset,
                end_offset_cpu,
                last_loc,
                len(batch.input_ids),
            )

        bs = batch.batch_size()
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            bs,
        )

        if get_global_server_args().enable_mamba_extra_buffer():
            batch.mamba_track_indices = torch.tensor(
                [
                    req.mamba_ping_pong_track_buffer[req.mamba_next_track_idx]
                    for req in batch.reqs
                ],
                dtype=torch.int64,
                device=batch.device,
            )
            batch.mamba_track_mask = None
            batch.mamba_track_seqlens = None

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        device = req_pool_indices.device
        batch_size = len(req_pool_indices)
        qo_indptr = torch.arange(
            0,
            (1 + batch_size) * self.draft_token_num,
            step=self.draft_token_num,
            dtype=torch.int32,
            device=device,
        )
        cum_kv_seq_len = torch.zeros(
            (batch_size + 1,), dtype=torch.int32, device=device
        )

        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        kv_indices = torch.empty(
            paged_kernel_lens_sum + self.draft_token_num * batch_size,
            dtype=torch.int32,
            device=device,
        )
        create_flashinfer_kv_indices_triton[(batch_size,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        mask_numel = (
            paged_kernel_lens_sum * self.draft_token_num
            + (self.draft_token_num**2) * batch_size
        )
        if self.custom_mask.numel() < mask_numel:
            # FIXME(attn): temporary fix for custom mask padding with cuda graph
            self.custom_mask = torch.cat(
                [
                    self.custom_mask,
                    torch.full(
                        (mask_numel - self.custom_mask.numel(),),
                        True,
                        dtype=torch.bool,
                        device=device,
                    ),
                ],
                dim=0,
            )

        return kv_indices, cum_kv_seq_len, qo_indptr, self.custom_mask

    def verify(
        self,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        page_size: int,
        vocab_mask: Optional[torch.Tensor] = None,  # For grammar
    ) -> torch.Tensor:
        """
        Verify and find accepted tokens based on logits output and batch
        (which contains spec decoding information).

        WARNING: This API in-place modifies the states of logits_output

        This API updates values inside logits_output based on the accepted
        tokens. I.e., logits_output.next_token_logits only contains
        accepted token logits.
        """
        if batch.forward_mode.is_idle():
            draft_extend_input = EagleDraftExtendInput.create_idle_input(
                device=batch.device,
                hidden_size=batch.model_config.spec_hidden_size,
                dtype=batch.model_config.dtype,
                capture_hidden_mode=CaptureHiddenMode.LAST,
            )
            return EagleVerifyOutput.create_idle(
                draft_extend_input=draft_extend_input,
                logits_output=logits_output,
                device=batch.device,
                spec_steps=self.spec_steps,
            )

        bs = self.retrieve_index.shape[0]
        candidates = self.draft_token.reshape(bs, self.draft_token_num)
        sampling_info = batch.sampling_info

        predict_shape = list(logits_output.next_token_logits.shape)[:-1]
        predict_shape[-1] += 1
        predict = torch.empty(predict_shape, dtype=torch.int32, device=batch.device)
        accept_index = torch.full(
            (bs, self.spec_steps + 1), -1, dtype=torch.int32, device=batch.device
        )
        num_correct_drafts = torch.empty((bs,), dtype=torch.int32, device=batch.device)

        if bs != len(sampling_info):
            sampling_info = copy.deepcopy(sampling_info)
            # NOTE: retrieve_index are the indices of the requests that are kept.
            sampling_info.filter_batch(
                self.retrieve_index.tolist(), self.retrieve_index
            )

        # Apply the custom logit processors if registered in the sampling info.
        if sampling_info.has_custom_logit_processor:
            apply_custom_logit_processor(
                logits_output.next_token_logits,
                sampling_info,
                num_tokens_in_batch=self.draft_token_num,
            )

        # Apply penalty
        if (
            sampling_info.penalizer_orchestrator.is_required
            or sampling_info.logit_bias is not None
        ):
            # This is a relaxed version of penalties for speculative decoding.
            sampling_info.penalizer_orchestrator.apply(
                logits_output.next_token_logits, repeat=self.draft_token_num
            )
            if sampling_info.logit_bias is not None:
                logits_output.next_token_logits.add_(
                    torch.repeat_interleave(
                        sampling_info.logit_bias, self.draft_token_num, dim=0
                    )
                )

        # Apply grammar mask
        if vocab_mask is not None:
            assert self.grammar is not None
            self.grammar.apply_vocab_mask(
                logits=logits_output.next_token_logits, vocab_mask=vocab_mask
            )

        # Sample tokens. Force greedy sampling on AMD
        is_all_greedy = sampling_info.is_all_greedy
        if (not is_all_greedy) and (not TREE_SPEC_KERNEL_AVAILABLE):
            logger.warning(
                "Tree speculative sampling kernel unavailable (likely AMD/HIP build). "
                "Falling back to greedy verification."
            )

        if is_all_greedy or not TREE_SPEC_KERNEL_AVAILABLE:
            target_predict = torch.argmax(logits_output.next_token_logits, dim=-1)
            target_predict = target_predict.reshape(bs, self.draft_token_num)
            predict, accept_index, num_correct_drafts = verify_tree_greedy_func(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=num_correct_drafts,  # mutable
                candidates=candidates,
                retrieve_index=self.retrieve_index,
                retrieve_next_token=self.retrieve_next_token,
                retrieve_next_sibling=self.retrieve_next_sibling,
                target_predict=target_predict,
                topk=self.topk,
            )

        else:
            # apply temperature and get target probs
            expanded_temperature = torch.repeat_interleave(
                sampling_info.temperatures, self.draft_token_num, dim=0
            )  # (bs * draft_token_num, 1)

            target_probs = F.softmax(
                logits_output.next_token_logits / expanded_temperature, dim=-1
            )  # (bs * draft_token_num, vocab_size)
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ks, self.draft_token_num, dim=0
                ),
            )  # (bs * draft_token_num, vocab_size)
            if sampling_info.need_top_p_sampling:
                target_probs = top_p_renorm_prob(
                    target_probs,
                    torch.repeat_interleave(
                        sampling_info.top_ps, self.draft_token_num, dim=0
                    ),
                )
            target_probs = target_probs.reshape(bs, self.draft_token_num, -1)

            draft_probs = torch.zeros(
                target_probs.shape, dtype=torch.float32, device=batch.device
            )

            # coins for rejection sampling
            coins = torch.rand_like(
                candidates, dtype=torch.float32, device=batch.device
            )
            # coins for final sampling
            coins_for_final_sampling = torch.rand(
                (bs,), dtype=torch.float32, device=batch.device
            )
            tree_speculative_sampling_target_only(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=num_correct_drafts,  # mutable
                candidates=candidates,
                # kwarg LHS retained as `retrive_*` to match sgl_kernel op schema.
                retrive_index=self.retrieve_index,
                retrive_next_token=self.retrieve_next_token,
                retrive_next_sibling=self.retrieve_next_sibling,
                uniform_samples=coins,
                uniform_samples_for_final_sampling=coins_for_final_sampling,
                target_probs=target_probs,
                draft_probs=draft_probs,
                threshold_single=get_global_server_args().speculative_accept_threshold_single,
                threshold_acc=get_global_server_args().speculative_accept_threshold_acc,
                deterministic=True,
            )

            # Sync sampling results across TP ranks: different GPUs may
            # produce slightly different target_probs due to floating-point
            # non-determinism in softmax/top_k/top_p, causing different
            # sampled tokens. Broadcast from rank 0 to ensure consistency.
            tp_group = (
                get_attention_tp_group()
                if is_dp_attention_enabled()
                else get_tp_group()
            )
            if tp_group.world_size > 1:
                tp_group.broadcast(predict, src=0)
                tp_group.broadcast(accept_index, src=0)
                tp_group.broadcast(num_correct_drafts, src=0)

        if SIMULATE_ACC_LEN > 0.0:
            # Do simulation
            accept_index = generate_simulated_accept_index(
                accept_index=accept_index,
                predict=predict,  # mutable
                num_correct_drafts=num_correct_drafts,  # mutable
                bs=bs,
                spec_steps=self.spec_steps,
            )

        unfinished_index = []
        unfinished_accept_index = []
        accept_index_cpu = accept_index.tolist()
        predict_cpu = predict.tolist()
        has_finished = False
        think_end_id = batch.model_config.think_end_id

        # Iterate every accepted token and check if req has finished after append the token
        # should be checked BEFORE free kv cache slots
        for i, (req, accept_index_row) in enumerate(zip(batch.reqs, accept_index_cpu)):
            num_accept_tokens = 0
            for j, idx in enumerate(accept_index_row):
                if idx == -1:
                    break
                num_accept_tokens += 1
                id = predict_cpu[idx]
                req.output_ids.append(id)
                if req.require_reasoning and think_end_id is not None:
                    req.update_reasoning_tokens(id, think_end_id)
                req.check_finished()
                if not req.finished() and req.grammar is not None:
                    try:
                        req.grammar.accept_token(id)
                    except ValueError as e:
                        logger.info(
                            f"{i=}, {req=}\n" f"{accept_index=}\n" f"{predict=}\n"
                        )
                        raise e
                    req.check_finished()
                if req.finished():
                    has_finished = True
                    # set all tokens after finished token to -1 and break
                    accept_index[i, j + 1 :] = -1
                    break
            # Update KV cache tracking for the accepted tokens
            req.kv_committed_len += num_accept_tokens
            req.kv_allocated_len = req.kv_committed_len
            if not req.finished():
                unfinished_index.append(i)
                if idx == -1:
                    unfinished_accept_index.append(accept_index[i, :j])
                else:
                    unfinished_accept_index.append(accept_index[i])
            req.spec_verify_ct += 1
            num_correct_drafts_this_req = (
                sum(1 for idx in accept_index_row if idx != -1) - 1
            )
            req.spec_num_correct_drafts += num_correct_drafts_this_req
            req.update_spec_correct_drafts_histogram(num_correct_drafts_this_req)

        if has_finished:
            num_correct_drafts = (accept_index != -1).sum(dim=1) - 1

        # Free the KV cache for unaccepted tokens
        # TODO: fuse them
        accept_index = accept_index[accept_index != -1]
        accept_tokens = predict[accept_index]
        evict_mask = torch.full_like(self.draft_token, True, dtype=torch.bool)
        evict_mask[accept_index] = False
        num_correct_drafts_cpu = num_correct_drafts.cpu()
        num_accept_tokens_cpu = num_correct_drafts_cpu + 1
        # FIXME: this `tolist()` fixes the numerical calculation consistency
        # try to unify the tensor representation and list representation
        num_correct_drafts_list = num_correct_drafts_cpu.tolist()
        num_accept_tokens_list = num_accept_tokens_cpu.tolist()

        if page_size == 1:
            # TODO: boolean array index leads to a device sync. Remove it.
            token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
        else:
            if self.topk == 1:
                # Only evict full empty page. Do not evict partial empty page
                align_evict_mask_to_page_size[len(batch.seq_lens),](
                    batch.seq_lens,
                    evict_mask,
                    page_size,
                    self.draft_token_num,
                    next_power_of_2(self.draft_token_num),
                )
                token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
            else:
                # Shift the accepted tokens to the beginning.
                # Only evict the last part
                src_cache_loc, tgt_cache_loc, to_free_num_slots = get_src_tgt_cache_loc(
                    batch.seq_lens,
                    batch.out_cache_loc,
                    accept_index,
                    num_correct_drafts,
                    self.draft_token_num,
                    page_size,
                )
                to_free_slots = torch.empty(
                    (to_free_num_slots.sum().item(),),
                    dtype=torch.int64,
                    device=to_free_num_slots.device,
                )

                # out_cache_loc: [0  1  2,  3  4  5,  6  7  8]
                # accept_index:  [0 -1  2,  3  4 -1,  6 -1 -1]
                # tgt_cache_loc: [0  1   ,  3  4   ,  6      ]
                # to_free_slots: [      2,        5,     7  8]
                # to_free_slots also needs to be page-aligned without the first partial page
                #
                # split each row of out_cache_loc into two parts.
                # 1. the first part goes to tgt_cache_loc. length = num_correct_drafts[i] + 1
                # 2. the second part goes to to_free_slots.
                get_target_cache_loc[(bs,)](
                    tgt_cache_loc,
                    to_free_slots,
                    num_correct_drafts,
                    to_free_num_slots,
                    batch.out_cache_loc,
                    self.draft_token_num,
                    next_power_of_2(self.draft_token_num),
                    next_power_of_2(bs),
                )

                # Free the kv cache
                token_to_kv_pool_allocator.free(to_free_slots)

                # Copy the kv cache
                batch.token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
                    tgt_cache_loc, src_cache_loc
                )

        # Construct EagleVerifyOutput
        if not has_finished:
            if page_size == 1 or self.topk == 1:
                batch.out_cache_loc = batch.out_cache_loc[accept_index]
                assign_req_to_token_pool_func(
                    batch.req_pool_indices,
                    batch.req_to_token_pool.req_to_token,
                    batch.seq_lens,
                    batch.seq_lens + num_correct_drafts + 1,
                    batch.out_cache_loc,
                    bs,
                )
            else:
                batch.out_cache_loc = tgt_cache_loc
            batch.seq_lens.add_(num_correct_drafts + 1)
            batch.seq_lens_cpu.add_(num_accept_tokens_cpu)

            draft_extend_input = EagleDraftExtendInput(
                hidden_states=(
                    batch.spec_info.hidden_states[accept_index]
                    if batch.spec_info.hidden_states is not None
                    else None
                ),
                num_correct_drafts=num_correct_drafts,
                num_accept_tokens=num_correct_drafts + 1,
                num_accept_tokens_cpu=num_accept_tokens_list,
                input_ids=accept_tokens,
                seq_lens=batch.seq_lens,
                seq_lens_cpu=batch.seq_lens_cpu,
                req_pool_indices=batch.req_pool_indices,
            )

            return EagleVerifyOutput(
                draft_extend_input=draft_extend_input,
                logits_output=logits_output,
                accept_tokens=accept_tokens,
                num_correct_drafts_per_req_cpu=num_correct_drafts_list,
                accept_indices=accept_index,
            )
        else:
            if page_size == 1 or self.topk == 1:
                assign_req_to_token_pool_func(
                    batch.req_pool_indices,
                    batch.req_to_token_pool.req_to_token,
                    batch.seq_lens,
                    batch.seq_lens + num_correct_drafts + 1,
                    batch.out_cache_loc[accept_index],
                    bs,
                )
                batch.seq_lens.add_(num_correct_drafts + 1)
                batch.seq_lens_cpu.add_(num_accept_tokens_cpu)

            if len(unfinished_accept_index) > 0:
                unfinished_accept_index = torch.cat(unfinished_accept_index)
                unfinished_index_device = torch.tensor(
                    unfinished_index, dtype=torch.int64, device=predict.device
                )
                draft_input_num_correct_drafts_cpu = [
                    num_correct_drafts_list[i] for i in unfinished_index
                ]
                draft_input_num_accept_tokens_cpu = [
                    num_accept_tokens_list[i] for i in unfinished_index
                ]
                if page_size == 1 or self.topk == 1:
                    batch.out_cache_loc = batch.out_cache_loc[unfinished_accept_index]
                else:
                    batch.out_cache_loc = torch.empty(
                        len(unfinished_index) + sum(draft_input_num_correct_drafts_cpu),
                        dtype=torch.int64,
                        device=predict.device,
                    )
                    num_accept_tokens_filter = create_num_accept_tokens_filter(
                        num_correct_drafts,
                        unfinished_index_device,
                        batch.seq_lens,
                    )
                    batch.seq_lens_cpu.add_(num_accept_tokens_cpu)
                    filter_finished_cache_loc_kernel[(bs,)](
                        batch.out_cache_loc,
                        tgt_cache_loc,
                        num_correct_drafts,
                        num_accept_tokens_filter,
                        next_power_of_2(bs),
                        next_power_of_2(self.draft_token_num),
                    )

                unfinished_num_correct_drafts = num_correct_drafts[
                    unfinished_index_device
                ]
                draft_extend_input = EagleDraftExtendInput(
                    hidden_states=(
                        batch.spec_info.hidden_states[unfinished_accept_index]
                        if batch.spec_info.hidden_states is not None
                        else None
                    ),
                    num_accept_tokens_cpu=draft_input_num_accept_tokens_cpu,
                    num_correct_drafts=unfinished_num_correct_drafts,
                    num_accept_tokens=unfinished_num_correct_drafts + 1,
                    input_ids=predict[unfinished_accept_index],
                    seq_lens=batch.seq_lens[unfinished_index_device],
                    seq_lens_cpu=batch.seq_lens_cpu[unfinished_index],
                    req_pool_indices=batch.req_pool_indices[unfinished_index_device],
                )
            else:
                draft_extend_input = EagleDraftExtendInput.create_idle_input(
                    device=batch.device,
                    hidden_size=batch.model_config.spec_hidden_size,
                    dtype=batch.model_config.dtype,
                    capture_hidden_mode=CaptureHiddenMode.LAST,
                )

            return EagleVerifyOutput(
                draft_extend_input=draft_extend_input,
                logits_output=logits_output,
                accept_tokens=accept_tokens,
                num_correct_drafts_per_req_cpu=num_correct_drafts_list,
                accept_indices=accept_index,
            )


@dataclass
class EagleDraftInput(SpecInput, EagleDraftInputV2Mixin):
    # For idle stubs use `create_idle_input`, not the bare ctor: `filter_batch`
    # / `merge_batch` slice / cat `topk_p` / `topk_index` / `hidden_states` /
    # `bonus_tokens` unconditionally.

    # shape: (b, topk)
    topk_p: torch.Tensor = None
    topk_index: torch.Tensor = None
    # shape: (b, hidden_size) - one hidden per req, consumed by `draft` forward.
    # None when the spec algorithm's draft doesn't read hidden_states
    # (e.g., STANDALONE — vanilla LLM draft).
    hidden_states: Optional[torch.Tensor] = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    # Per-req bonus token (the "+1" target prediction at end of each accept
    # chain). Written by `EagleDraftExtendInput.prepare_extend_after_decode`;
    # the worker copies it here for next iter's draft.
    bonus_tokens: torch.Tensor = None

    # shape: (b + 1,)
    kv_indptr: torch.Tensor = None
    kv_indices: torch.Tensor = None

    num_tokens_per_req: int = -1
    num_tokens_for_logprob_per_req: int = -1

    # V2 overlap worker only
    future_indices: Optional[FutureIndices] = None
    new_seq_lens: Optional[torch.Tensor] = None
    verify_done: Optional[torch.cuda.Event] = None
    # V2 reuses `EagleDraftInput` across phases (V1 has a separate
    # `EagleDraftExtendInput` for these). Set during V2's draft-extend.
    num_correct_drafts: Optional[torch.Tensor] = None
    num_accept_tokens: Optional[torch.Tensor] = None

    def __post_init__(self):
        super().__init__(SpecInputType.EAGLE_DRAFT)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.num_tokens_per_req, self.num_tokens_for_logprob_per_req

    def prepare_for_extend(self, batch: ScheduleBatch):

        if batch.forward_mode.is_idle():
            return

        # Prefill only generate 1 token.
        assert len(self.bonus_tokens) == len(batch.seq_lens)

        pt = 0
        for i, extend_len in enumerate(batch.extend_lens):
            input_ids = batch.input_ids[pt : pt + extend_len]
            batch.input_ids[pt : pt + extend_len] = torch.cat(
                (input_ids[1:], self.bonus_tokens[i].reshape(1))
            )
            pt += extend_len

    @classmethod
    def hidden_size_for(cls, worker) -> Optional[int]:
        """Decode-phase `hidden_states` width: draft self-chain output
        (draft model writes its own last hidden back via `capture_for_decode`
        and the draft loop). Returns None when the draft architecture doesn't
        consume the field (e.g., STANDALONE)."""
        if worker.speculative_algorithm.is_standalone():
            return None
        return _draft_runner_of(worker).model_config.spec_hidden_size

    @classmethod
    def dtype_for(cls, worker) -> Optional[torch.dtype]:
        if worker.speculative_algorithm.is_standalone():
            return None
        return _draft_runner_of(worker).model_config.dtype

    @classmethod
    def create_idle_input(
        cls,
        device: torch.device,
        hidden_size: Optional[int],
        dtype: Optional[torch.dtype],
        topk: int,
        capture_hidden_mode: CaptureHiddenMode,
    ):
        return cls(
            bonus_tokens=torch.empty((0,), device=device, dtype=torch.int32),
            hidden_states=(
                torch.empty((0, hidden_size), device=device, dtype=dtype)
                if hidden_size is not None
                else None
            ),
            topk_p=torch.empty((0, topk), device=device, dtype=torch.float32),
            topk_index=torch.empty((0, topk), device=device, dtype=torch.int64),
            capture_hidden_mode=capture_hidden_mode,
            new_seq_lens=torch.empty((0,), device=device, dtype=torch.int32),
        )

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        if self.future_indices is not None:
            self.future_indices.indices = self.future_indices.indices[new_indices]
            return

        strict_check = envs.SGLANG_SPEC_ENABLE_STRICT_FILTER_CHECK.get()
        if has_been_filtered:
            # in eagle_utils.py:verify, we have already filtered the batch by `unfinished_index`
            # therefore, we don't need to filter the batch again in scheduler
            error_msg = f"length of new_indices: {len(new_indices)} != length of topk_p: {len(self.topk_p)}, this should not happen"
            if len(new_indices) != len(self.topk_p):
                if strict_check:
                    raise ValueError(error_msg)
                else:
                    logger.warning(error_msg)

            self.topk_p = self.topk_p[: len(new_indices)]
            self.topk_index = self.topk_index[: len(new_indices)]
            if self.hidden_states is not None:
                self.hidden_states = self.hidden_states[: len(new_indices)]
            self.bonus_tokens = self.bonus_tokens[: len(new_indices)]
        else:
            # in some cases(e.g draft_extend), we have not filtered the batch by `unfinished_index`
            self.topk_p = self.topk_p[new_indices]
            self.topk_index = self.topk_index[new_indices]
            if self.hidden_states is not None:
                self.hidden_states = self.hidden_states[new_indices]
            self.bonus_tokens = self.bonus_tokens[new_indices]

    def merge_batch(self, spec_info: "EagleDraftInput"):
        if self.future_indices is not None:
            assert spec_info.future_indices is not None
            self.future_indices = FutureIndices(
                indices=torch.cat(
                    [self.future_indices.indices, spec_info.future_indices.indices]
                )
            )
            return

        # Detect idle stub by `topk_index` length (idle inputs have
        # shape[0] == 0 across all fields). Don't use `hidden_states is None`:
        # for STANDALONE all non-idle inputs also have None hidden_states.
        if len(self.topk_index) == 0:
            self.hidden_states = spec_info.hidden_states
            self.bonus_tokens = spec_info.bonus_tokens
            self.topk_p = spec_info.topk_p
            self.topk_index = spec_info.topk_index
            return
        if len(spec_info.topk_index) == 0:
            return
        if self.hidden_states is not None and spec_info.hidden_states is not None:
            self.hidden_states = torch.cat(
                [self.hidden_states, spec_info.hidden_states], axis=0
            )
        self.bonus_tokens = torch.cat(
            [self.bonus_tokens, spec_info.bonus_tokens], axis=0
        )
        self.topk_p = torch.cat([self.topk_p, spec_info.topk_p])
        self.topk_index = torch.cat([self.topk_index, spec_info.topk_index])


@dataclass
class EagleDraftExtendInput(SpecInput):
    """Inputs to the draft-extend forward (the per-accepted-token pass after verify).

    Produced by `EagleVerifyInput.verify`, installed on `batch.spec_info` for
    the draft-extend forward, then replaced with a fresh `EagleDraftInput` for
    the next iter's draft.
    """

    # shape: (total_accepted, hidden_size). Sliced from verify-time hidden_states
    # by accept_index; consumed by the draft-extend forward. None when the spec
    # algorithm's draft doesn't read hidden_states (e.g., STANDALONE).
    hidden_states: Optional[torch.Tensor] = None

    # Per-req accept counts. `num_accept_tokens = num_correct_drafts + 1`.
    # Both kept for cuda-graph buffer indexing and the
    # `create_extend_after_decode_spec_info` kernel.
    num_correct_drafts: torch.Tensor = None
    num_accept_tokens: torch.Tensor = None
    # CPU view, read by attention backends during the extend forward.
    num_accept_tokens_cpu: List[int] = None

    # Batch-state slices for the draft-extend forward. Set by verify (sliced to
    # reqs continuing into next iter). `prepare_extend_after_decode` copies
    # these onto `batch.{input_ids, seq_lens, seq_lens_cpu, req_pool_indices}`.
    #   - input_ids:        accept tokens flat over surviving reqs
    #   - seq_lens / _cpu:  per-req sequence length (post-accept)
    #   - req_pool_indices: per-req kv-pool slot
    input_ids: torch.Tensor = None
    seq_lens: torch.Tensor = None
    seq_lens_cpu: torch.Tensor = None
    req_pool_indices: torch.Tensor = None

    # Set by `prepare_extend_after_decode`:
    #   - positions: kernel-written, shape `[total_accepted]`.
    #   - bonus_tokens: kernel-written, shape `[bs]`. The worker reads this
    #     post-extend to populate next iter's `EagleDraftInput.bonus_tokens`.
    positions: Optional[torch.Tensor] = None
    bonus_tokens: Optional[torch.Tensor] = None

    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.LAST
    num_tokens_per_req: int = -1
    num_tokens_for_logprob_per_req: int = 1

    def __post_init__(self):
        super().__init__(SpecInputType.EAGLE_DRAFT_EXTEND)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.num_tokens_per_req, self.num_tokens_for_logprob_per_req

    @classmethod
    def hidden_size_for(cls, worker) -> Optional[int]:
        """Extend-phase `hidden_states` width: target's `spec_hidden_size`,
        widened to `num_aux * target_hidden` for EAGLE-3 aux mode. Returns
        None when the draft architecture doesn't consume the field
        (e.g., STANDALONE)."""
        if worker.speculative_algorithm.is_standalone():
            return None
        target_cfg = worker.target_worker.model_runner.model_config
        if not (
            worker.speculative_algorithm.is_eagle3()
            and worker.eagle_use_aux_hidden_state
        ):
            return target_cfg.spec_hidden_size

        hf_config = target_cfg.hf_config

        # `num_aux` resolution: explicit attr > eagle_config layer_ids > default 3.
        num_aux = getattr(hf_config, "num_aux_hidden_states", None)
        if num_aux is None:
            eagle_config = getattr(hf_config, "eagle_config", None) or {}
            layer_ids = eagle_config.get("eagle_aux_hidden_state_layer_ids")
            num_aux = len(layer_ids) if layer_ids else 3

        target_hidden = getattr(hf_config, "target_hidden_size", target_cfg.hidden_size)
        return target_hidden * num_aux

    @classmethod
    def dtype_for(cls, worker) -> Optional[torch.dtype]:
        if worker.speculative_algorithm.is_standalone():
            return None
        return worker.target_worker.model_runner.model_config.dtype

    @classmethod
    def create_idle_input(
        cls,
        device: torch.device,
        hidden_size: Optional[int],
        dtype: Optional[torch.dtype],
        capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.LAST,
    ) -> "EagleDraftExtendInput":
        return cls(
            hidden_states=(
                torch.empty((0, hidden_size), device=device, dtype=dtype)
                if hidden_size is not None
                else None
            ),
            num_correct_drafts=torch.empty((0,), device=device, dtype=torch.int32),
            num_accept_tokens=torch.empty((0,), device=device, dtype=torch.int32),
            num_accept_tokens_cpu=[],
            input_ids=torch.empty((0,), device=device, dtype=torch.long),
            seq_lens=torch.empty((0,), device=device, dtype=torch.int32),
            seq_lens_cpu=torch.empty((0,), dtype=torch.int32),
            req_pool_indices=torch.empty((0,), device=device, dtype=torch.int64),
            capture_hidden_mode=capture_hidden_mode,
        )

    def prepare_extend_after_decode(
        self,
        batch: ScheduleBatch,
        speculative_num_steps: int,
    ):
        # Caller must have installed `self` as `batch.spec_info` before calling.
        assert batch.spec_info is self
        if batch.forward_mode.is_idle():
            return

        # The kernel below populates `self.positions` and `self.bonus_tokens`;
        # the worker reads `self.bonus_tokens` to construct next iter's
        # `EagleDraftInput`.
        batch.input_ids = self.input_ids
        batch.extend_lens = self.num_accept_tokens_cpu
        batch.extend_num_tokens = sum(batch.extend_lens)
        batch.seq_lens = self.seq_lens
        batch.seq_lens_cpu = self.seq_lens_cpu
        batch.req_pool_indices = self.req_pool_indices
        batch.return_logprob = False
        batch.return_hidden_states = False

        self.capture_hidden_mode = CaptureHiddenMode.LAST
        self.positions = torch.empty_like(batch.input_ids, dtype=torch.long)
        self.bonus_tokens = torch.empty_like(self.num_accept_tokens, dtype=torch.int32)

        create_extend_after_decode_spec_info[(len(batch.seq_lens),)](
            batch.input_ids,
            batch.seq_lens,
            self.num_accept_tokens,
            self.positions,
            self.bonus_tokens,
            next_power_of_2(max(speculative_num_steps + 1, len(batch.seq_lens))),
        )

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: Optional[int],
        req_to_token: torch.Tensor,
    ):
        device = req_pool_indices.device
        bs = self.num_correct_drafts.numel()
        qo_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        qo_indptr[1:] = torch.cumsum(self.num_accept_tokens, dim=0)
        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        if paged_kernel_lens_sum is None:
            paged_kernel_lens_sum = cum_kv_seq_len[-1]

        kv_indices = torch.empty(
            paged_kernel_lens_sum, dtype=torch.int32, device=device
        )

        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        return kv_indices, cum_kv_seq_len, qo_indptr, None


@dataclass
class EagleVerifyOutput:
    # Next iter's draft-extend input, installed as `batch.spec_info` for the
    # draft-extend forward.
    draft_extend_input: EagleDraftExtendInput
    # Logit outputs from target worker.
    logits_output: LogitsProcessorOutput
    # All accepted tokens flat across all reqs incl. those that finished this
    # step. Includes the bonus token. Used for output processing.
    accept_tokens: torch.Tensor
    # Accepted token length per sequence in a batch in CPU (full set).
    num_correct_drafts_per_req_cpu: List[int]
    # Accepted indices from logits_output.next_token_logits
    accept_indices: torch.Tensor

    @classmethod
    def create_idle(
        cls,
        *,
        draft_extend_input: EagleDraftExtendInput,
        logits_output: LogitsProcessorOutput,
        device: torch.device,
        spec_steps: int,
    ) -> "EagleVerifyOutput":
        return cls(
            draft_extend_input=draft_extend_input,
            logits_output=logits_output,
            accept_tokens=torch.empty(0, dtype=torch.long, device=device),
            num_correct_drafts_per_req_cpu=[],
            accept_indices=torch.full(
                (0, spec_steps + 1), -1, dtype=torch.int32, device=device
            ),
        )
