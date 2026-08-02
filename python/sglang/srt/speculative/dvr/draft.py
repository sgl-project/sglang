from __future__ import annotations

import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
)
from sglang.srt.speculative.dvr.cuda_graph_runner import (
    DVRDraftDecodeCudaGraphRunner,
    dvr_draft_decode_context,
)
from sglang.srt.speculative.dvr.sampling import dvr_draft_sample
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.spec_utils import (
    record_stream_each,
    record_stream_for_v2_verify,
)
from sglang.srt.utils.common import get_available_gpu_memory

logger = logging.getLogger(__name__)


class DVRDraftBackend:
    """Draft-side lifecycle consumed by the common DVR transaction."""

    target_capture_hidden_mode = CaptureHiddenMode.NULL
    uses_point_proposals = False
    requires_short_prompt_verify = True

    def __init__(self, owner, worker=None):
        self.owner = owner
        self.worker = worker

    def context(self):
        raise NotImplementedError

    def idle_input(self):
        raise NotImplementedError

    def finish_prefill(self, batch, batch_result):
        raise NotImplementedError

    def propose(self, batch):
        raise NotImplementedError

    def commit_draft_state(self, batch, batch_result):
        raise NotImplementedError

    @property
    def draft_worker(self):
        return self.worker

    @property
    def war_fastpath_runner(self):
        return self.owner.target_worker.model_runner

    @property
    def spec_v2_attn_backends(self) -> tuple:
        return (self.owner.target_worker.model_runner.attn_backend,)

    def iter_runners(self):
        return []

    def create_verify_plan_stream(self):
        if self.worker is not None and envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
            return torch.get_device_module(self.owner.device).Stream()
        return None

    def init_attention_backends(self):
        if self.worker is not None:
            self.worker.init_attention_backends()

    def init_cuda_graphs(self):
        pass

    def alloc_memory_pool(
        self,
        memory_pool_config,
        req_to_token_pool,
        token_to_kv_pool_allocator,
    ):
        if self.worker is None:
            if req_to_token_pool is not None and token_to_kv_pool_allocator is not None:
                return
            raise RuntimeError(
                "DVR self-draft requires Scheduler to pass the target memory pools."
            )
        self.worker.alloc_memory_pool(
            memory_pool_config, req_to_token_pool, token_to_kv_pool_allocator
        )

    def update_weights_from_disk(self, recv_req):
        del recv_req
        return True, "Succeeded to update model weights."

    def update_weights_from_ipc(self, recv_req):
        del recv_req
        return True, "Succeeded to update model weights."

    def update_draft_runner_from_disk(self, runner, recv_req):
        model_path, load_format = self.draft_weight_source(
            recv_req.model_path, recv_req.load_format
        )
        return runner.weight_updater.update_weights_from_disk(
            model_path,
            load_format,
            recapture_cuda_graph=False,
        )

    def update_draft_runner_from_ipc(self, runner, recv_req):
        if self.draft_follows_target:
            return runner.weight_updater.update_weights_from_ipc(recv_req)

        model_path, load_format = self.draft_weight_source(None, None)
        return runner.weight_updater.update_weights_from_disk(
            model_path,
            load_format,
            recapture_cuda_graph=False,
        )

    @property
    def draft_follows_target(self) -> bool:
        server_args = self.owner.server_args
        return (
            server_args.speculative_draft_model_path is not None
            and server_args.speculative_draft_model_path == server_args.model_path
        )

    def draft_weight_source(self, target_model_path, target_load_format):
        server_args = self.owner.server_args
        if self.draft_follows_target:
            return target_model_path, target_load_format
        return (
            server_args.speculative_draft_model_path,
            server_args.speculative_draft_load_format or server_args.load_format,
        )

    def reset_cuda_graphs(self):
        pass

    def clear_cache_pool(self):
        pass

    def prepare_target_verify(self, batch, spec_info):
        if self.worker is None:
            batch.seq_lens_cpu_cache = spec_info.seq_lens_cpu
            return
        record_stream_for_v2_verify(
            batch,
            spec_info,
            torch.get_device_module(batch.device).current_stream(),
        )

    def finish_target_verify_prepare(self, batch, current_stream):
        if self.worker is not None:
            record_stream_each((batch.input_ids, batch.out_cache_loc), current_stream)

    def validate_target_output(self, logits_output):
        if (
            self.target_capture_hidden_mode.need_capture()
            and logits_output.hidden_states is None
        ):
            raise RuntimeError(
                "DVR target verify must return hidden states required by the "
                "draft backend."
            )


class SelfDraftBackend(DVRDraftBackend):
    """Target-model self-draft around the common DVR transaction."""

    def __init__(self, owner):
        super().__init__(owner)
        self.graph_runner = None
        self.proposal_prob_buffer = None

    def context(self):
        return dvr_draft_decode_context(
            self.owner.model_runner,
            self.owner.draft_graph_buffers,
        )

    def idle_input(self):
        return EagleDraftInput.create_idle_input(
            device=self.owner.device,
            hidden_size=None,
            dtype=None,
            topk=1,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )

    @staticmethod
    def make_input(bonus_tokens: torch.Tensor) -> EagleDraftInput:
        return EagleDraftInput(
            hidden_states=None,
            bonus_tokens=bonus_tokens,
            topk_p=torch.ones(
                (bonus_tokens.shape[0], 1),
                dtype=torch.float32,
                device=bonus_tokens.device,
            ),
            topk_index=bonus_tokens.to(torch.long).unsqueeze(-1),
            capture_hidden_mode=CaptureHiddenMode.NULL,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )

    def finish_prefill(self, batch, batch_result):
        draft_input = self.make_input(batch_result.next_token_ids)
        batch.spec_info = draft_input
        return draft_input

    def propose(self, batch: ScheduleBatch) -> EagleVerifyInput:
        owner = self.owner
        if batch.forward_mode.is_idle():
            batch.spec_info = self.idle_input()
            return EagleVerifyInput.create_idle_input(
                1, owner.num_draft_steps, owner.num_draft_tokens, owner.device
            )

        spec_info = batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)

        offsets = batch.seq_lens.to(torch.long).unsqueeze(
            1
        ) + owner.chain_position_offsets.unsqueeze(0)
        rows = batch.req_pool_indices.to(torch.long).unsqueeze(1)
        batch.out_cache_loc = batch.req_to_token_pool.req_to_token[
            rows, offsets
        ].reshape(-1)
        batch.mamba_track_indices = None
        batch.mamba_track_mask = None
        batch.mamba_track_seqlens = None
        spec_info.positions = batch.seq_lens.clone()
        spec_info.num_tokens_per_req = 1
        spec_info.num_tokens_for_logprob_per_req = 1
        spec_info.capture_hidden_mode = CaptureHiddenMode.NULL

        saved_return_logprob = batch.return_logprob
        saved_return_hidden_states = batch.return_hidden_states
        batch.return_logprob = False
        batch.return_hidden_states = False
        try:
            forward_batch = ForwardBatch.init_new(
                batch,
                owner.model_runner,
                return_hidden_states_before_norm=False,
            )
            draft_tokens, draft_probs = self.draft_tokens(forward_batch)
        finally:
            batch.return_logprob = saved_return_logprob
            batch.return_hidden_states = saved_return_hidden_states

        batch_size = draft_tokens.shape[0]
        positions = (
            batch.seq_lens[:, None] + owner.chain_position_offsets[None, :]
        ).flatten()
        return EagleVerifyInput(
            draft_token=draft_tokens.flatten(),
            custom_mask=None,
            positions=positions,
            retrieve_index=owner.chain_retrieve_index[:batch_size],
            retrieve_next_token=owner.chain_retrieve_next[:batch_size],
            retrieve_next_sibling=owner.chain_retrieve_sibling[:batch_size],
            retrieve_cum_len=None,
            spec_steps=owner.num_draft_steps,
            topk=1,
            draft_token_num=owner.num_draft_tokens,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            seq_lens_sum=forward_batch.seq_lens_sum,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
            draft_probs=draft_probs,
        )

    def draft_tokens(self, forward_batch: ForwardBatch):
        owner = self.owner
        spec_info = forward_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)

        out_cache_loc = (
            forward_batch.out_cache_loc.reshape(
                forward_batch.batch_size, owner.num_draft_tokens
            )
            .transpose(0, 1)
            .contiguous()
        )
        draft_tokens = [spec_info.bonus_tokens.to(torch.long)]
        draft_probs = None

        origin_seq_lens = forward_batch.seq_lens
        origin_seq_lens_cpu = forward_batch.seq_lens_cpu
        origin_seq_lens_sum = forward_batch.seq_lens_sum
        origin_spec_info = forward_batch.spec_info
        origin_out_cache_loc = forward_batch.out_cache_loc
        forward_batch.spec_info = None
        position_offset = 0

        try:
            for step in range(owner.num_draft_steps):
                if step:
                    forward_batch.positions.add_(1)
                    position_offset += 1
                forward_batch.input_ids = draft_tokens[-1]
                forward_batch.out_cache_loc = out_cache_loc[step]
                forward_batch.seq_lens = origin_seq_lens + step + 1
                forward_batch.seq_lens_cpu = (
                    None
                    if origin_seq_lens_cpu is None
                    else origin_seq_lens_cpu + step + 1
                )
                forward_batch.seq_lens_sum = (
                    None
                    if origin_seq_lens_sum is None
                    else origin_seq_lens_sum + (step + 1) * forward_batch.batch_size
                )
                logits_output = self.decode_forward(forward_batch)
                next_token_ids, proposal = dvr_draft_sample(
                    logits_output.next_token_logits,
                    forward_batch.sampling_info,
                    forward_batch.positions,
                )
                owner.model_runner.ngram_embedding_manager.update_after_decode(
                    next_token_ids=next_token_ids,
                    forward_batch=forward_batch,
                )
                if proposal is not None:
                    if self.proposal_prob_buffer is None:
                        self.proposal_prob_buffer = torch.empty(
                            (
                                owner.chain_retrieve_index.shape[0],
                                owner.num_draft_steps,
                                proposal.shape[-1],
                            ),
                            dtype=torch.float32,
                            device=proposal.device,
                        )
                    proposal_buffer = self.proposal_prob_buffer
                    proposal_buffer[: forward_batch.batch_size, step].copy_(proposal)
                    draft_probs = proposal_buffer[
                        : forward_batch.batch_size, : owner.num_draft_steps
                    ]
                draft_tokens.append(next_token_ids.to(torch.long))
        finally:
            forward_batch.seq_lens = origin_seq_lens
            forward_batch.seq_lens_cpu = origin_seq_lens_cpu
            forward_batch.seq_lens_sum = origin_seq_lens_sum
            forward_batch.spec_info = origin_spec_info
            if position_offset:
                forward_batch.positions.sub_(position_offset)
            forward_batch.out_cache_loc = origin_out_cache_loc

        return torch.stack(draft_tokens, dim=1), draft_probs

    def decode_forward(self, forward_batch: ForwardBatch) -> LogitsProcessorOutput:
        if self.graph_runner is not None and self.graph_runner.can_run_graph(
            forward_batch
        ):
            return self.graph_runner.execute(forward_batch)

        seq_lens = forward_batch.seq_lens_cpu
        min_seq_len = int(seq_lens.min()) if seq_lens is not None else "GPU-only"
        capture_bs = [] if self.graph_runner is None else self.graph_runner.capture_bs
        raise RuntimeError(
            "DVR self-draft decode requires the dedicated CUDA graph; no eager "
            "fallback is used. The current batch cannot run it: "
            f"batch_size={forward_batch.batch_size}, min_seq_len={min_seq_len}, "
            f"capture_bs={capture_bs}. For batch-size misses, use the default "
            "CUDA graph batch sizes or include the running batch size "
            "in --cuda-graph-bs/--cuda-graph-max-bs."
        )

    def init_cuda_graphs(self):
        owner = self.owner
        if (
            self.graph_runner is None
            and owner.server_args.cuda_graph_config.decode.backend != Backend.DISABLED
            and not owner.server_args.disable_draft_cuda_graph
        ):
            before_mem = get_available_gpu_memory(
                owner.device, owner.model_runner.gpu_id
            )
            logger.info(
                "Capture DVR self-draft CUDA graph begin. avail mem=%.2f GB",
                before_mem,
            )
            with dvr_draft_decode_context(
                owner.model_runner,
                owner.draft_graph_buffers,
                capture=True,
            ):
                self.graph_runner = DVRDraftDecodeCudaGraphRunner(owner.model_runner)
            after_mem = get_available_gpu_memory(
                owner.device, owner.model_runner.gpu_id
            )
            logger.info(
                "Capture DVR self-draft CUDA graph end. mem usage=%.2f GB, "
                "avail mem=%.2f GB",
                before_mem - after_mem,
                after_mem,
            )

    def reset_cuda_graphs(self):
        self.graph_runner = None

    def commit_draft_state(self, batch, batch_result):
        del batch
        batch_result.next_draft_input = self.make_input(
            batch_result.next_draft_input.bonus_tokens
        )
