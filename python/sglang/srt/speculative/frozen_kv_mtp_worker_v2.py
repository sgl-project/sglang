# Copyright 2026 SGLang Team
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
"""Spec-v2 worker for Frozen-KV MTP (two layers, like ``eagle_worker_v2``).

The frozen draft reads the target KV cache read-only and owns no KV pool, so
its "draft extend" is not a model forward: it selects the last accepted token +
target hidden state as the next-iter seed, and the seed forward runs at the
start of the next draft.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.cuda_graph_config import cuda_graph_fully_disabled
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import EagleDraftWorkerBase
from sglang.srt.speculative.eagle_utils import (
    build_tree_kernel_efficient,
    organize_draft_results,
)
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2, _get_plan_stream
from sglang.srt.speculative.frozen_kv_mtp_info import (
    FrozenKVMTPContext,
    FrozenKVMTPDraftInput,
    FrozenKVMTPVerifyInput,
)
from sglang.srt.speculative.frozen_kv_mtp_utils import (
    expand_for_topk_draft,
    frozen_kv_target_view,
    position_for_batch,
    select_last_extend_hidden,
    set_frozen_kv_positions,
    target_kv_pool_view,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    draft_tp_context,
    fast_topk,
    select_top_k_tokens,
    spec_stage_span,
)
from sglang.srt.utils import empty_context
from sglang.srt.utils.async_probe import (
    maybe_detect_inf,
    maybe_detect_nan,
    maybe_detect_oob,
)

logger = logging.getLogger(__name__)


class FrozenKVMTPDraftWorker(EagleDraftWorkerBase, TpModelWorker):
    """Frozen-KV MTP draft worker.

    The assistant reads target KV only. It reuses EAGLE's verify input/output
    contract, but owns the seed and recurrent draft loop because there is no
    assistant-side KV extension.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.gpu_id = gpu_id
        self.device = server_args.device
        self.target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        assert self.speculative_algorithm.is_frozen_kv_mtp(), (
            "FrozenKVMTPDraftWorker should only be instantiated for "
            "SpeculativeAlgorithm.FROZEN_KV_MTP, got "
            f"{self.speculative_algorithm.name}."
        )

        # Target pools (read-only) are bound in alloc_memory_pool(), not here, so
        # the worker can be built before the target pool exists (see #29021).
        self.req_to_token_pool = None
        self.token_to_kv_pool_allocator = None
        self.draft_pool_config: Optional[MemoryPoolConfig] = None

        self.hot_token_id = None

        with (
            empty_context()
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            # NOTE: call TpModelWorker.__init__ explicitly -- EagleDraftWorkerBase is
            # an ABC with no __init__, so cooperative super() would be ambiguous.
            TpModelWorker.__init__(
                self,
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
            )

        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
        if hasattr(self.draft_model_runner.model, "set_embed_and_head"):
            self.draft_model_runner.model.set_embed_and_head(embed, head)
        else:
            logger.debug(
                "Draft model %s does not implement set_embed_and_head; "
                "skipping target-embedding bind in Frozen-KV MTP skeleton.",
                type(self.draft_model_runner.model).__name__,
            )

        self.kv_context: Optional[FrozenKVMTPContext] = None

        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )

        self.draft_attn_backend = None
        self.cuda_graph_runner = None
        # Frozen draft has no draft-extend forward (seed-select only); keep these
        # None so inherited probes (spec_v2_attn_backends, adaptive) stay typed.
        self.draft_extend_attn_backend = None
        self.cuda_graph_runner_for_draft_extend = None

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator

        self.draft_pool_config = MemoryPoolConfig(
            max_total_num_tokens=64,  # Dummy value
            max_running_requests=memory_pool_config.max_running_requests,
        )

        # NOTE: call TpModelWorker explicitly -- EagleDraftWorkerBase precedes it in
        # the MRO and its alloc_memory_pool is a no-op stub.
        TpModelWorker.alloc_memory_pool(
            self,
            memory_pool_config=self.draft_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )

        if hasattr(self.draft_model_runner.model, "bind_frozen_kv_context"):
            self._bind_kv_context()

    def init_attention_backends(self):
        with (
            self.draft_tp_context(self.draft_model_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            TpModelWorker.init_attention_backends(self)
            self.draft_attn_backend = self._init_draft_attn_backend()
            self.draft_model_runner.draft_attn_backend = self.draft_attn_backend

    def init_cuda_graphs(self):
        with (
            self.draft_tp_context(self.draft_model_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            TpModelWorker.init_cuda_graphs(self, capture_decode_cuda_graph=False)
            self._capture_cuda_graphs()

    @property
    def draft_model_runner(self):
        return self.model_runner

    @property
    def draft_runner(self):
        # Alias for the inherited EAGLEWorkerV2 forward/verify skeleton, which
        # reads `draft_worker.draft_runner`.
        return self.model_runner

    def get_attn_backend(self):  # pragma: no cover - exposed for adaptive
        return self.draft_attn_backend

    def clear_cache_pool(self):
        pass

    def _resolve_draft_backend_type(self) -> str:
        return (
            self.server_args.speculative_draft_attention_backend
            or self.server_args.decode_attention_backend
            or self.server_args.attention_backend
        )

    def _init_draft_attn_backend(self):
        if self.topk == 1:
            return self.draft_model_runner.attn_backend

        backend_type = self._resolve_draft_backend_type()
        if backend_type != "triton":
            raise ValueError(
                "Frozen-KV MTP topk > 1 currently supports only the triton "
                f"attention backend, got {backend_type}."
            )
        return self._init_triton_draft_attn_backend()

    def _init_triton_draft_attn_backend(self):
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        max_bs = self.req_to_token_pool.size * self.topk
        kv_indptr_buf = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=self.draft_model_runner.device
        )
        return TritonAttnBackend(
            self.draft_model_runner,
            skip_prefill=True,
            kv_indptr_buf=kv_indptr_buf,
        )

    def _bind_kv_context(self) -> None:
        draft_model = self.draft_model_runner.model
        if not hasattr(draft_model, "build_frozen_kv_mtp_context") or not hasattr(
            draft_model, "bind_frozen_kv_context"
        ):
            logger.debug(
                "Draft model %s does not implement Frozen-KV MTP context hooks; "
                "skipping frozen-kv bind.",
                type(draft_model).__name__,
            )
            return

        ctx = draft_model.build_frozen_kv_mtp_context(
            target_model=self.target_worker.model_runner.model,
            target_token_to_kv_pool=self.target_worker.model_runner.token_to_kv_pool,
        )
        draft_model.bind_frozen_kv_context(ctx)
        self.kv_context = ctx

    def _frozen_kv_target_view(self, forward_batch: ForwardBatch):
        return frozen_kv_target_view(
            forward_batch, self.kv_context, self.draft_attn_backend
        )

    def _target_kv_pool_view(self, forward_batch: ForwardBatch):
        return target_kv_pool_view(
            forward_batch, self.kv_context, self.draft_attn_backend
        )

    def _set_positions(self, forward_batch: ForwardBatch) -> None:
        set_frozen_kv_positions(forward_batch, self.topk)

    def _expand_for_topk_draft(self, forward_batch: ForwardBatch) -> None:
        expand_for_topk_draft(forward_batch, self.topk)

    def _position_for_batch(self, batch: ScheduleBatch) -> torch.Tensor:
        return position_for_batch(batch)

    @property
    def _recurrent_hidden_size(self) -> int:
        return int(self.draft_model_runner.model.backbone_hidden_size)

    def _init_frozen_kv_metadata(self, forward_batch: ForwardBatch) -> None:
        if forward_batch.forward_mode.is_idle():
            return
        if forward_batch.seq_lens_cpu is not None:
            forward_batch.seq_lens_sum = forward_batch.seq_lens_cpu.sum().item()
        else:
            forward_batch.seq_lens_sum = torch.sum(forward_batch.seq_lens).item()
        with self._frozen_kv_target_view(forward_batch):
            self.draft_attn_backend.init_forward_metadata(forward_batch)
        forward_batch.mark_forward_metadata_ready()

    def _init_frozen_kv_metadata_capture_cuda_graph(
        self, forward_batch: ForwardBatch
    ) -> None:
        with self._frozen_kv_target_view(forward_batch):
            self.draft_attn_backend.init_forward_metadata_out_graph(
                forward_batch, in_capture=True
            )
        forward_batch.mark_forward_metadata_ready()

    def _init_frozen_kv_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int, seq_lens_sum: int
    ) -> None:
        from types import SimpleNamespace

        fb_view = SimpleNamespace(
            batch_size=bs,
            forward_mode=ForwardMode.DECODE,
            input_ids=getattr(forward_batch, "input_ids", None),
            req_pool_indices=forward_batch.req_pool_indices[:bs],
            seq_lens=forward_batch.seq_lens[:bs],
            seq_lens_sum=seq_lens_sum,
            seq_lens_cpu=(
                forward_batch.seq_lens_cpu[:bs]
                if forward_batch.seq_lens_cpu is not None
                else None
            ),
            encoder_lens=None,
            out_cache_loc=getattr(forward_batch, "out_cache_loc", None),
            spec_info=None,
        )
        with self._frozen_kv_target_view(forward_batch):
            self.draft_attn_backend.init_forward_metadata_out_graph(fb_view)

    def _capture_cuda_graphs(self) -> None:
        if cuda_graph_fully_disabled() or self.speculative_num_steps <= 1:
            return
        if self.target_worker.device != "cuda":
            logger.info(
                "Frozen-KV MTP draft CUDA graph is only supported on CUDA; "
                "running the draft loop eagerly on %s.",
                self.target_worker.device,
            )
            return

        from sglang.srt.speculative.frozen_kv_mtp_cuda_graph_runner import (
            FrozenKVMTPCudaGraphRunner,
        )

        logger.info("Capture Frozen-KV MTP draft cuda graph begin.")
        self.cuda_graph_runner = FrozenKVMTPCudaGraphRunner(self)
        logger.info("Capture Frozen-KV MTP draft cuda graph end.")

    def _select_last_extend_hidden(
        self, batch: ScheduleBatch, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        return select_last_extend_hidden(batch, hidden_states)

    def _idle_seed(self) -> FrozenKVMTPDraftInput:
        return FrozenKVMTPDraftInput.create_idle_input(
            device=self.device,
            hidden_size=self._recurrent_hidden_size,
            dtype=self.model_config.dtype,
            topk=self.topk,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )

    def _build_seed_draft_input(
        self,
        last_token_ids: torch.Tensor,
        last_hidden_states: torch.Tensor,
    ) -> FrozenKVMTPDraftInput:
        """Build the next-iter seed ``FrozenKVMTPDraftInput`` from (bonus token,
        target hidden). No forward here -- the seed forward runs inside the
        captured draft graph (see ``draft_forward``'s seed iter)."""
        if last_token_ids.numel() == 0:
            return self._idle_seed()

        stashed = FrozenKVMTPDraftInput()
        stashed.bonus_tokens = last_token_ids.to(torch.int64)
        stashed.hidden_states = last_hidden_states
        # Real-shaped zeros so inherited `filter_batch`/`merge_batch` can slice
        # them between iters; overwritten by the captured seed iter.
        bs = last_token_ids.shape[0]
        device = last_token_ids.device
        stashed.topk_p = torch.zeros(
            (bs, self.topk), device=device, dtype=torch.float32
        )
        stashed.topk_index = torch.zeros(
            (bs, self.topk), device=device, dtype=torch.int64
        )
        stashed.capture_hidden_mode = CaptureHiddenMode.LAST
        stashed.num_tokens_per_req = 1
        stashed.num_tokens_for_logprob_per_req = 1
        return stashed

    def draft(self, batch: ScheduleBatch):
        if batch.forward_mode.is_idle():
            return FrozenKVMTPVerifyInput.create_idle_input(
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
            )

        spec_info = batch.spec_info
        assert isinstance(spec_info, FrozenKVMTPDraftInput)

        # NOTE: per-iter bookkeeping (penalty cumulation, maybe_evict_swa,
        # decode_batch_idx tick) is done by the scheduler-driven
        # eagle_utils.eagle_prepare_for_decode (see
        # ScheduleBatch.prepare_for_decode), not here -- matching EAGLE v2.
        # Repeating evict/tick here would double-run them: the idx clock
        # gates SWA eviction timing and the SWA prefix-lock release.

        spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        spec_info.num_tokens_per_req = self.topk
        spec_info.num_tokens_for_logprob_per_req = self.topk
        spec_info.positions = self._position_for_batch(batch)
        batch.seq_lens_sum = torch.sum(batch.seq_lens).item()
        batch.return_hidden_states = False

        forward_batch = ForwardBatch.init_new(batch, self.draft_model_runner)
        assert forward_batch.capture_hidden_mode == CaptureHiddenMode.LAST
        self._set_positions(forward_batch)
        self._expand_for_topk_draft(forward_batch)

        # Frozen draft never writes KV; None signals fill_from to skip the slot.
        forward_batch.out_cache_loc = None

        can_run_cuda_graph = (
            self.cuda_graph_runner
            and self.cuda_graph_runner.can_run_graph(forward_batch)
        )
        if can_run_cuda_graph:
            parent_list, top_scores_index, draft_tokens = (
                self.cuda_graph_runner.execute(forward_batch)
            )
        else:
            forward_batch.can_run_dp_cuda_graph = False
            parent_list, top_scores_index, draft_tokens = self.draft_forward(
                forward_batch
            )

        (
            tree_mask,
            position,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            spec_info.bonus_tokens,
            parent_list,
            top_scores_index,
            draft_tokens,
            batch.seq_lens,
            batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
        )

        return FrozenKVMTPVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.speculative_num_draft_tokens,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=batch.seq_lens_sum,
            seq_lens_cpu=batch.seq_lens_cpu,
        )

    def draft_forward(self, forward_batch: ForwardBatch):
        spec_info = forward_batch.spec_info
        assert isinstance(spec_info, FrozenKVMTPDraftInput)

        score_list: list[torch.Tensor] = []
        token_list: list[torch.Tensor] = []
        parents_list: list[torch.Tensor] = []

        # Seed + recurrent iters share the same `seq_lens - 1` rope position,
        # so one init covers the loop. Must run even at num_steps == 1.
        if forward_batch.needs_forward_metadata_init():
            self._init_frozen_kv_metadata(forward_batch)

        # Seed iter: assistant forward on (bonus_token, target_h) to produce
        # iter-0 `(topk_p, topk_index, hidden_states)`. For topk>1, replicate
        # to `bs*topk` to match kernel shapes, then slice back per-req.
        bonus_tokens = spec_info.bonus_tokens
        target_hidden = spec_info.hidden_states
        if self.topk > 1:
            seed_input_ids = bonus_tokens.repeat_interleave(self.topk, dim=0)
            seed_prev_hidden = target_hidden.repeat_interleave(self.topk, dim=0)
        else:
            seed_input_ids = bonus_tokens
            seed_prev_hidden = target_hidden

        forward_batch.input_ids = seed_input_ids
        forward_batch.spec_info.hidden_states = seed_prev_hidden
        self._set_positions(forward_batch)

        with (
            self._target_kv_pool_view(forward_batch),
            forward_context(ForwardContext(attn_backend=self.draft_attn_backend)),
        ):
            seed_output = self.draft_model_runner.forward(forward_batch).logits_output

        maybe_detect_nan(
            seed_output.next_token_logits, "frozen_kv_mtp_draft: seed iter"
        )

        if self.topk > 1:
            seed_next_logits = seed_output.next_token_logits[:: self.topk]
            seed_hidden_per_req = seed_output.hidden_states[:: self.topk]
        else:
            seed_next_logits = seed_output.next_token_logits
            seed_hidden_per_req = seed_output.hidden_states

        probs = torch.softmax(seed_next_logits, dim=-1)
        topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
        maybe_detect_oob(
            topk_index,
            0,
            seed_next_logits.shape[-1],
            "frozen_kv_mtp_draft: seed topk_index OOB",
        )
        hidden_states = seed_hidden_per_req

        scores = None
        for i in range(self.speculative_num_steps):
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

            if i == self.speculative_num_steps - 1:
                break

            forward_batch.input_ids = input_ids
            forward_batch.spec_info.hidden_states = hidden_states
            self._set_positions(forward_batch)

            with (
                self._target_kv_pool_view(forward_batch),
                forward_context(ForwardContext(attn_backend=self.draft_attn_backend)),
            ):
                logits_output = self.draft_model_runner.forward(
                    forward_batch
                ).logits_output

            maybe_detect_nan(
                logits_output.next_token_logits, f"frozen_kv_mtp_draft step {i}"
            )
            maybe_detect_inf(
                logits_output.next_token_logits, f"frozen_kv_mtp_draft step {i}"
            )
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            maybe_detect_oob(
                topk_index,
                0,
                logits_output.next_token_logits.shape[-1],
                "frozen_kv_mtp_draft: topk_index OOB",
            )
            hidden_states = logits_output.hidden_states

        return organize_draft_results(
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )

    def draft_extend(self):
        # EagleDraftWorkerBase contract. Frozen has no draft-KV extend forward; the
        # orchestrator calls `_draft_extend_for_{prefill,decode}` directly.
        pass

    def _draft_extend_for_prefill(
        self,
        batch: ScheduleBatch,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        mm_input_embeds: Optional[torch.Tensor] = None,
    ) -> FrozenKVMTPDraftInput:
        """Seed for the first decode iter after prefill. Frozen draft writes no
        KV (reads target KV), so unlike EAGLE there is no draft-extend forward:
        just select the last prompt hidden + bonus token and stash the seed."""
        del mm_input_embeds  # frozen seed needs no input embeds
        if batch.forward_mode.is_idle():
            return self._idle_seed()
        last_hidden = self._select_last_extend_hidden(batch, target_hidden_states)
        return self._build_seed_draft_input(next_token_ids, last_hidden)

    def _draft_extend_for_decode(self, batch: ScheduleBatch, batch_result) -> None:
        """Frozen 'draft extend': no forward. Pull the last accepted token's
        target hidden from the verify output and stash it as the next-iter seed.

        Replaces verify's `EagleDraftInput` with a `FrozenKVMTPDraftInput` so the
        next draft passes the FROZEN_KV_MTP attn-backend assertions.
        """
        if batch.forward_mode.is_idle():
            batch_result.next_draft_input = self._idle_seed()
            return

        bs = len(batch.seq_lens)
        # Same per-req select_index EAGLE uses on its draft-extend output: the
        # last accepted node (accept_lens - 1) in each per-req block of width
        # num_draft_tokens. Verify already compacted the accepted path to the
        # front (topk > 1) / it is the front chain (topk == 1).
        select_index = (
            torch.arange(bs, device=self.device) * self.speculative_num_draft_tokens
            + batch_result.accept_lens
            - 1
        )
        last_hidden = batch_result.logits_output.hidden_states[select_index]
        bonus_tokens = batch_result.next_draft_input.bonus_tokens
        batch_result.next_draft_input = self._build_seed_draft_input(
            bonus_tokens, last_hidden
        )


class FrozenKVMTPWorkerV2(EAGLEWorkerV2):
    """Spec-v2 (overlap) orchestrator for Frozen-KV MTP.

    Reuses ``EAGLEWorkerV2``'s verify / ``move_accept_tokens`` / forward
    skeleton verbatim; only the draft worker and the seed-based draft-extend
    are frozen-specific.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # NOTE: intentionally does NOT call EAGLEWorkerV2.__init__ -- that builds
        # an EagleDraftWorker (with its own draft KV pool). The frozen draft owns
        # no KV, so we mirror the relevant setup and build a FrozenKVMTPDraftWorker.
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.tp_rank = tp_rank
        self.gpu_id = gpu_id
        self.device = server_args.device
        self._target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        # Match the draft context length to the target (assistant reads target KV).
        server_args.override(
            "spec_worker.match_target_context_length",
            context_length=target_worker.model_runner.model_config.context_len,
        )

        self._draft_worker = FrozenKVMTPDraftWorker(
            server_args,
            gpu_id,
            tp_rank,
            dp_rank,
            moe_ep_rank,
            attn_cp_rank,
            moe_dp_rank,
            nccl_port,
            target_worker,
        )

        # Frozen MTP does not wire the adaptive controller yet.
        assert (
            not server_args.speculative_adaptive
        ), "Frozen-KV MTP does not support adaptive speculative decoding yet."
        self.adaptive_controller = None

        # Some dummy tensors (parity with EAGLEWorkerV2 init).
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

    @property
    def spec_v2_attn_backends(self) -> tuple:
        # Frozen draft touches no draft-extend backend; only target + draft.
        return (
            self._target_worker.model_runner.attn_backend,
            self._draft_worker.draft_attn_backend,
        )

    def forward_batch_generation(self, batch: ScheduleBatch, on_publish=None):
        # Mirrors EAGLEWorkerV2.forward_batch_generation; the only frozen-specific
        # change is the idle draft-input (FrozenKVMTPDraftInput + recurrent hidden
        # size). The draft / seed-based draft-extend hooks are FrozenKVMTPDraftWorker's.
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            # Target prefill (frozen is never standalone -> capture FULL hidden).
            batch_output = self.target_worker.forward_batch_generation(
                batch, capture_hidden_mode=CaptureHiddenMode.FULL
            )

            # Spec_v2 convention: batch.seq_lens = length BEFORE this iter's tokens.
            batch_output.new_seq_lens = batch.seq_lens
            # Publish before draft-extend so the fence is at target-end.
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)

            # Draft prefill seed (no forward).
            with (
                self.draft_worker.draft_tp_context(
                    self.draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
                spec_stage_span("draft_extend"),
            ):
                batch_output.next_draft_input = (
                    self.draft_worker._draft_extend_for_prefill(
                        batch,
                        batch_output.logits_output.hidden_states,
                        batch_output.next_token_ids,
                        batch_output.logits_output.mm_input_embeds,
                    )
                )
                return batch_output
        else:
            self.activate_step_by_batch(batch.seq_lens.shape[0])

            if batch.spec_info is None:
                batch.spec_info = self.draft_worker._idle_seed()
            with (
                self.draft_worker.draft_tp_context(
                    self.draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
                spec_stage_span("draft"),
            ):
                verify_input = self.draft_worker.draft(batch)
            assert verify_input.is_verify_input()
            batch.spec_info = verify_input
            batch_output = self.verify(batch)
            # Publish before draft-extend so the fence is at verify-end.
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)
            with (
                self.draft_worker.draft_tp_context(
                    self.draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
                spec_stage_span("draft_extend"),
            ):
                self.draft_worker._draft_extend_for_decode(batch, batch_output)

            return batch_output
