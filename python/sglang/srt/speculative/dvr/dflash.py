from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass

import torch

from sglang.kernels.ops.speculative.cache_locs import (
    assign_extend_cache_locs_func,
)
from sglang.kernels.ops.speculative.dflash import (
    _prepare_dflash_draft_block_unchecked,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    compute_position,
)
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.draft_worker_common import make_draft_input_v2
from sglang.srt.speculative.dvr.cuda_graph_runner import dvr_draft_decode_context
from sglang.srt.speculative.dvr.draft import DVRDraftBackend
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import spec_stage_span

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DVRDFlashDraftBlock:
    candidates: torch.Tensor
    positions: torch.Tensor
    verify_out_cache_loc: torch.Tensor
    verify_out_cache_loc_2d: torch.Tensor


class DVRDFlashDraftWorker(DFlashWorkerV2):
    """DFlash model execution used only as a DVR draft backend."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        capacity = max(
            self.server_args.cuda_graph_config.decode.max_bs or 0,
            self.server_args.max_running_requests or 0,
            1,
        )
        self.reserved_seq_lens_cpu = torch.empty(
            capacity, dtype=torch.int32, device="cpu"
        )
        self.capture_hooks = []

    def init_cuda_graphs(self):
        hooks = self.draft_model_runner.capture_tail_hooks
        if self.capture_hooks:
            hooks[:] = [
                hook
                for hook in hooks
                if all(hook is not old_hook for old_hook in self.capture_hooks)
            ]
        self.capture_hooks = []
        self._draft_sampler = None
        existing_hook_ids = {id(hook) for hook in hooks}
        super().init_cuda_graphs()
        self.capture_hooks = [
            hook for hook in hooks if id(hook) not in existing_hook_ids
        ]

    def make_draft_input(self, batch, bonus_tokens: torch.Tensor) -> DFlashDraftInputV2:
        batch_size = batch.batch_size()
        if batch_size > self.reserved_seq_lens_cpu.numel():
            capacity = max(batch_size, self.reserved_seq_lens_cpu.numel() * 2)
            self.reserved_seq_lens_cpu = torch.empty(
                capacity, dtype=torch.int32, device="cpu"
            )

        reserved_seq_lens = self.reserved_seq_lens_cpu[:batch_size]
        reserved_seq_lens_sum = 0
        for row, req in enumerate(batch.reqs):
            reserved_len = int(req.kv.kv_allocated_len)
            reserved_seq_lens[row] = reserved_len
            reserved_seq_lens_sum += reserved_len

        draft_input = make_draft_input_v2(
            bonus_tokens=bonus_tokens,
            new_seq_lens=batch.seq_lens,
        )
        draft_input.reserved_seq_lens_cpu = reserved_seq_lens
        draft_input.reserved_seq_lens_sum = reserved_seq_lens_sum
        return draft_input

    def commit_prefill(self, batch, batch_output) -> torch.Tensor:
        logits_output = batch_output.logits_output
        if logits_output.hidden_states is None:
            raise RuntimeError(
                "DVR DFlash requires target aux hidden states for prefill."
            )
        if batch.extend_lens is None or batch.prefix_lens is None:
            raise RuntimeError(
                "DVR DFlash expected extend_lens and prefix_lens in prefill."
            )
        if batch.out_cache_loc is None:
            raise RuntimeError("DVR DFlash prefill expected out_cache_loc.")

        device = batch_output.next_token_ids.device
        context_lens = torch.tensor(batch.extend_lens, dtype=torch.int32, device=device)
        draft_seq_lens = torch.tensor(
            batch.prefix_lens, dtype=torch.int32, device=device
        )
        positions, _ = compute_position(
            self.model_runner.server_args.attention_backend,
            draft_seq_lens,
            context_lens,
            int(sum(batch.extend_lens)),
        )
        self._append_target_hidden_to_draft_kv_by_loc(
            target_hidden=logits_output.hidden_states,
            cache_loc=batch.out_cache_loc,
            positions=positions,
        )
        logits_output.hidden_states = None
        return batch_output.next_token_ids

    def draft(self, batch, bonus_tokens: torch.Tensor) -> DVRDFlashDraftBlock:
        draft_input = self.make_draft_input(batch, bonus_tokens)
        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )
        batch_size = batch.batch_size()
        block_size = int(self.block_size)
        device = self.device
        target_model = self.target_worker.model_runner.model
        embed_module = target_model.get_input_embeddings()
        lm_head = getattr(target_model, "lm_head", None)
        if lm_head is None or not hasattr(lm_head, "weight"):
            raise RuntimeError(
                "DVR DFlash requires the target model to expose lm_head.weight."
            )

        self._ensure_draft_block_buffers(batch_size)
        block_ids = self._draft_block_ids_buf[:batch_size]
        positions_2d = self._draft_block_positions_buf[:batch_size]
        candidates = self._draft_block_tokens_buf[:batch_size]
        verify_out_cache_loc_2d = self._draft_verify_out_cache_loc_buf[:batch_size]
        prefix_lens = batch.seq_lens

        if self._use_triton_prepare_block:
            try:
                _prepare_dflash_draft_block_unchecked(
                    bonus_tokens=bonus_tokens.view(-1),
                    prefix_lens=prefix_lens.view(-1),
                    req_pool_indices=batch.req_pool_indices.view(-1),
                    req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                    block_ids_out=block_ids,
                    positions_out=positions_2d,
                    cache_loc_out=verify_out_cache_loc_2d,
                    mask_token_id=int(self._mask_token_id),
                )
            except Exception as error:
                self._use_triton_prepare_block = False
                logger.warning(
                    "DVR DFlash Triton prepare_block failed; using eager setup: %s",
                    error,
                )

        if not self._use_triton_prepare_block:
            block_ids.fill_(int(self._mask_token_id))
            block_ids[:, 0].copy_(bonus_tokens)
            torch.add(
                prefix_lens.unsqueeze(1),
                self._block_pos_offsets,
                out=positions_2d,
            )
            end_offset = prefix_lens + block_size
            verify_out_cache_loc = assign_extend_cache_locs_func(
                req_pool_indices=batch.req_pool_indices,
                req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                start_offset=prefix_lens,
                end_offset=end_offset,
                batch_size=batch_size,
                draft_token_num=block_size,
                device=device,
            )
            verify_out_cache_loc_2d.copy_(
                verify_out_cache_loc.view(batch_size, block_size)
            )

        noise_embedding = embed_module(block_ids)
        input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])
        positions = positions_2d.reshape(-1)
        verify_out_cache_loc = verify_out_cache_loc_2d.reshape(-1)
        seq_lens_cpu = self._draft_seq_lens_cpu_buf[:batch_size]

        if self.use_compact_draft_cache:
            draft_seq_lens = self._compute_compact_draft_seq_lens(prefix_lens)
            self._fill_compact_seq_lens_cpu_bound(
                batch_seq_lens_cpu=batch.seq_lens_cpu,
                reserved_seq_lens_cpu=draft_input.reserved_seq_lens_cpu,
                draft_prefix_lens=draft_seq_lens,
                out=seq_lens_cpu,
            )
            self._rebuild_compact_draft_cache(
                req_pool_indices=batch.req_pool_indices,
                prefix_lens=prefix_lens,
                draft_prefix_lens=draft_seq_lens,
                verify_out_cache_loc_2d=verify_out_cache_loc_2d,
                bs=batch_size,
                block_size=block_size,
            )
            draft_seq_lens_sum = int(seq_lens_cpu.sum().item())
        else:
            draft_seq_lens = prefix_lens
            if batch.seq_lens_cpu is not None:
                seq_lens_cpu.copy_(batch.seq_lens_cpu)
                seq_lens_cpu.add_(block_size)
                draft_seq_lens_sum = int(seq_lens_cpu.sum())
            else:
                seq_lens_cpu.copy_(draft_input.reserved_seq_lens_cpu)
                draft_seq_lens_sum = int(draft_input.reserved_seq_lens_sum)

        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=batch_size,
            input_ids=block_ids.flatten(),
            req_pool_indices=batch.req_pool_indices,
            seq_lens=draft_seq_lens,
            out_cache_loc=verify_out_cache_loc,
            seq_lens_sum=draft_seq_lens_sum,
            seq_lens_cpu=seq_lens_cpu,
            positions=positions,
            input_embeds=input_embeds,
            spec_algorithm=SpeculativeAlgorithm.DFLASH,
            spec_info=self._draft_block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        with torch.inference_mode():
            draft_output = self.draft_model_runner.forward(forward_batch)

        if self._draft_sampler is not None and draft_output.can_run_graph:
            draft_tokens = self._draft_sampler.out[
                : batch_size * (block_size - 1)
            ].view(batch_size, block_size - 1)
        else:
            draft_hidden = draft_output.logits_output.hidden_states
            if draft_hidden is None:
                raise RuntimeError("DVR DFlash draft model returned no hidden states.")
            draft_hidden = draft_hidden.view(batch_size, block_size, -1)
            draft_tokens = self._greedy_sample_from_vocab_parallel_head(
                hidden_states=draft_hidden[:, 1:, :].reshape(
                    -1, draft_hidden.shape[-1]
                ),
                lm_head=lm_head,
            ).view(batch_size, block_size - 1)

        candidates[:, 0].copy_(block_ids[:, 0])
        candidates[:, 1:].copy_(draft_tokens)
        return DVRDFlashDraftBlock(
            candidates=candidates,
            positions=positions,
            verify_out_cache_loc=verify_out_cache_loc,
            verify_out_cache_loc_2d=verify_out_cache_loc_2d,
        )

    def commit_accepted(
        self,
        block: DVRDFlashDraftBlock,
        logits_output,
        commit_lens: torch.Tensor,
    ) -> None:
        hidden_states = logits_output.hidden_states
        if hidden_states is None:
            raise RuntimeError("DVR DFlash verify returned no target hidden states.")
        batch_size = commit_lens.shape[0]
        hidden_states = hidden_states.view(batch_size, int(self.block_size), -1)
        self._append_target_hidden_to_draft_kv_by_loc(
            target_hidden=hidden_states.reshape(-1, hidden_states.shape[-1]),
            cache_loc=block.verify_out_cache_loc,
            cache_loc_2d=block.verify_out_cache_loc_2d,
            positions=block.positions,
            commit_lens=commit_lens,
        )
        logits_output.hidden_states = None


class DFlashDraftBackend(DVRDraftBackend):
    """DFlash block proposals around the common DVR transaction."""

    target_capture_hidden_mode = CaptureHiddenMode.FULL
    uses_point_proposals = True
    requires_short_prompt_verify = False

    def __init__(self, owner, worker):
        super().__init__(owner, worker)
        self.pending_draft_block = None

    @classmethod
    def create(cls, owner, server_args, gpu_id, ps, nccl_port, target_worker):
        draft_args = deepcopy(server_args)
        draft_args.override(
            "dvr_dflash.draft_worker",
            speculative_algorithm="DFLASH",
            speculative_num_steps=1,
            enable_deterministic_inference=False,
        )
        return cls(
            owner,
            DVRDFlashDraftWorker(
                draft_args,
                gpu_id,
                ps,
                nccl_port,
                target_worker,
            ),
        )

    @property
    def draft_worker(self):
        return self.worker.draft_worker

    @property
    def spec_v2_attn_backends(self) -> tuple:
        return (
            self.owner.target_worker.model_runner.attn_backend,
            self.worker.draft_model_runner.attn_backend,
        )

    def iter_runners(self):
        return [("draft", self.worker.draft_model_runner)]

    def context(self):
        return dvr_draft_decode_context(
            self.worker.draft_model_runner,
            self.owner.draft_graph_buffers,
            tune_attention=False,
        )

    def init_cuda_graphs(self):
        with dvr_draft_decode_context(
            self.worker.draft_model_runner,
            self.owner.draft_graph_buffers,
            capture=True,
            tune_attention=False,
        ):
            self.worker.init_cuda_graphs()

    def update_weights_from_disk(self, recv_req):
        return self.update_draft_runner_from_disk(
            self.worker.draft_model_runner, recv_req
        )

    def update_weights_from_ipc(self, recv_req):
        return self.update_draft_runner_from_ipc(
            self.worker.draft_model_runner, recv_req
        )

    def clear_cache_pool(self):
        self.worker.clear_cache_pool()

    @staticmethod
    def make_input(bonus_tokens: torch.Tensor) -> EagleDraftInput:
        return EagleDraftInput(
            bonus_tokens=bonus_tokens,
            topk_p=torch.empty(
                (bonus_tokens.shape[0], 0),
                dtype=torch.float32,
                device=bonus_tokens.device,
            ),
            topk_index=torch.empty(
                (bonus_tokens.shape[0], 0),
                dtype=torch.int64,
                device=bonus_tokens.device,
            ),
            hidden_states=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )

    def idle_input(self):
        return self.make_input(
            torch.empty(0, dtype=torch.int64, device=self.owner.device)
        )

    def finish_prefill(self, batch, batch_result):
        bonus_tokens = self.worker.commit_prefill(batch, batch_result)
        next_input = self.make_input(bonus_tokens)
        batch.spec_info = next_input
        return next_input

    def propose(self, batch):
        if batch.forward_mode.is_idle():
            batch.spec_info = self.idle_input()
            return EagleVerifyInput.create_idle_input(
                1,
                self.owner.num_draft_steps,
                self.owner.num_draft_tokens,
                self.owner.device,
            )

        if not isinstance(batch.spec_info, EagleDraftInput):
            raise RuntimeError("DVR DFlash expected the common DVR draft input.")
        block = self.worker.draft(batch, batch.spec_info.bonus_tokens)
        self.pending_draft_block = block
        batch.out_cache_loc = block.verify_out_cache_loc
        batch_size = batch.batch_size()
        return EagleVerifyInput(
            draft_token=block.candidates.reshape(-1),
            custom_mask=None,
            positions=block.positions,
            retrieve_index=self.owner.chain_retrieve_index[:batch_size],
            retrieve_next_token=self.owner.chain_retrieve_next[:batch_size],
            retrieve_next_sibling=self.owner.chain_retrieve_sibling[:batch_size],
            retrieve_cum_len=None,
            spec_steps=self.owner.num_draft_steps,
            topk=1,
            draft_token_num=self.owner.num_draft_tokens,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=batch.seq_lens_sum,
            seq_lens_cpu=batch.seq_lens_cpu,
            draft_probs=None,
        )

    def commit_draft_state(self, batch, batch_result):
        if batch.forward_mode.is_idle():
            batch_result.next_draft_input = self.make_input(
                batch_result.next_draft_input.bonus_tokens
            )
            return
        if self.pending_draft_block is None:
            raise RuntimeError("DVR DFlash verify completed without a draft block.")
        with self.context(), spec_stage_span("dvr_rollback_draft"):
            self.worker.commit_accepted(
                block=self.pending_draft_block,
                logits_output=batch_result.logits_output,
                commit_lens=batch_result.accept_lens,
            )
        self.pending_draft_block = None
        batch_result.next_draft_input = self.make_input(
            batch_result.next_draft_input.bonus_tokens
        )
