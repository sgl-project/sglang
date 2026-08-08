from __future__ import annotations

from contextlib import contextmanager

import torch

from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.dvr.cuda_graph_runner import dvr_draft_decode_context
from sglang.srt.speculative.dvr.draft import DVRDraftBackend
from sglang.srt.speculative.dvr.sampling import (
    dvr_sample_from_probs,
    dvr_sampling_probs,
)
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.srt.speculative.spec_utils import spec_stage_span


class DVREagleDraftWorker(EagleDraftWorker):
    """EAGLE draft execution with request-seeded proposal sampling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        max_bs = max(
            self.server_args.cuda_graph_config.decode.max_bs or 0,
            self.server_args.max_running_requests or 0,
            1,
        )
        self.proposal_sampling_seeds = torch.zeros(
            max_bs, dtype=torch.int64, device=self.device
        )

    def set_sampling_seeds(self, seeds: torch.Tensor):
        if seeds is None:
            raise RuntimeError("DVR EAGLE requires one sampling seed per request.")
        seeds = seeds.reshape(-1)
        if seeds.shape[0] > self.proposal_sampling_seeds.shape[0]:
            raise RuntimeError(
                "DVR EAGLE sampling batch exceeds its graph seed buffer: "
                f"batch_size={seeds.shape[0]}, "
                f"capacity={self.proposal_sampling_seeds.shape[0]}."
            )
        self.proposal_sampling_seeds[: seeds.shape[0]].copy_(seeds)

    def _sample_rejection_proposal(
        self,
        logits: torch.Tensor,
        sampling_info,
        positions: torch.Tensor,
        *,
        position_offset: int = 0,
    ):
        probs = torch.softmax(logits / sampling_info.temperatures, dim=-1)
        probs = dvr_sampling_probs(probs, sampling_info)
        if probs.shape[0] == 0:
            return (
                probs,
                probs[:, :1],
                torch.empty((0, 1), dtype=torch.int64, device=logits.device),
            )
        seeds = sampling_info.sampling_seed
        if seeds is None:
            seeds = self.proposal_sampling_seeds[: probs.shape[0]]
        token_ids = dvr_sample_from_probs(
            probs,
            seeds,
            positions,
            position_offset=position_offset,
        ).unsqueeze(1)
        return probs, probs.gather(1, token_ids), token_ids


class EagleDraftBackend(DVRDraftBackend):
    """Upstream EAGLE/MTP draft around the common DVR transaction."""

    target_capture_hidden_mode = CaptureHiddenMode.FULL

    def __init__(self, owner, worker):
        super().__init__(owner, worker)

    @classmethod
    def create(cls, owner, server_args, gpu_id, ps, nccl_port, target_worker):
        server_args.override(
            "spec_worker.match_target_context_length",
            context_length=target_worker.model_runner.model_config.context_len,
        )
        worker = DVREagleDraftWorker(
            server_args,
            gpu_id,
            ps,
            nccl_port,
            target_worker,
        )
        if any(
            isinstance(module, RadixLinearAttention)
            for module in worker.draft_runner.model.modules()
        ):
            raise NotImplementedError(
                "DVR EAGLE does not manage linear-attention draft-model state."
            )
        return cls(owner, worker)

    @property
    def draft_worker(self):
        return self.worker

    @property
    def war_fastpath_runner(self):
        return self.worker.draft_runner

    @property
    def spec_v2_attn_backends(self) -> tuple:
        return (
            self.owner.target_worker.model_runner.attn_backend,
            self.worker.draft_attn_backend,
            self.worker.draft_extend_attn_backend
            or self.worker.draft_runner.attn_backend,
        )

    def iter_runners(self):
        return [("draft", self.worker.draft_runner)]

    def init_cuda_graphs(self):
        with dvr_draft_decode_context(
            self.worker.draft_runner,
            self.owner.draft_graph_buffers,
            capture=True,
            extra_attn_backends=self.spec_v2_attn_backends[1:],
        ):
            self.worker.init_cuda_graphs()

    def update_weights_from_disk(self, recv_req):
        return self.update_draft_runner_from_disk(self.worker.draft_runner, recv_req)

    def update_weights_from_ipc(self, recv_req):
        return self.update_draft_runner_from_ipc(self.worker.draft_runner, recv_req)

    @contextmanager
    def context(self):
        with (
            self.worker.draft_tp_context(self.worker.draft_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
            dvr_draft_decode_context(
                self.worker.draft_runner,
                self.owner.draft_graph_buffers,
                extra_attn_backends=self.owner.spec_v2_attn_backends[1:],
            ),
        ):
            yield

    def idle_input(self):
        return EagleDraftInput.create_idle_input(
            device=self.owner.device,
            hidden_size=EagleDraftInput.hidden_size_for(self.worker),
            dtype=EagleDraftInput.dtype_for(self.worker),
            topk=1,
            capture_hidden_mode=CaptureHiddenMode.LAST,
            vocab_size=self.owner.target_worker.model_config.vocab_size,
        )

    def finish_prefill(self, batch, batch_result):
        with self.context():
            return self.worker._draft_extend_for_prefill(
                batch,
                batch_result.logits_output.hidden_states,
                batch_result.next_token_ids,
                batch_result.logits_output.mm_input_embeds,
            )

    def propose(self, batch):
        self.worker.set_sampling_seeds(batch.sampling_info.sampling_seed)
        return self.worker.draft(batch)

    def commit_draft_state(self, batch, batch_result):
        # Draft extend remains the final shared-pool reader for EAGLE.
        with self.context(), spec_stage_span("dvr_rollback_draft"):
            self.worker._draft_extend_for_decode(batch, batch_result)
