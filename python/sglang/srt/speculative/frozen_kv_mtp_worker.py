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
"""Gemma 4 frozen-KV MTP draft worker. Assistant reads target KV only; verify
reuses EAGLE's contract. Stubs until draft/verify are wired.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import draft_tp_context
from sglang.srt.utils import empty_context

if TYPE_CHECKING:
    from sglang.srt.models.gemma4_mtp import FrozenKVMTPContext

logger = logging.getLogger(__name__)


class FrozenKVMTPWorker(TpModelWorker):
    """Frozen-KV MTP worker; same constructor shape as EAGLEWorker. Entry:
    :meth:`forward_batch_generation` (stubs for now).
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
            "FrozenKVMTPWorker should only be instantiated for "
            "SpeculativeAlgorithm.FROZEN_KV_MTP, got "
            f"{self.speculative_algorithm.name}. The dispatch happens in "
            "server_args._handle_speculative_decoding -> "
            "_resolve_speculative_algorithm_alias."
        )

        if self.topk != 1:
            raise ValueError(
                "Frozen-KV MTP currently only supports speculative_eagle_topk=1 "
                f"(got {self.topk}); tree fan-out is not yet implemented."
            )

        server_args.context_length = target_worker.model_runner.model_config.context_len

        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True

        # Draft attention uses target req_to_token + KV allocator (read-only).
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        self.hot_token_id = None

        with (
            empty_context()
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            super().__init__(
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
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                memory_pool_config=target_worker.model_runner.memory_pool_config,
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

        self.kv_context: Optional["FrozenKVMTPContext"] = None
        if hasattr(self.draft_model_runner.model, "bind_frozen_kv_context"):
            self._bind_kv_context()

        self.draft_model_runner.server_args.disable_cuda_graph = (
            backup_disable_cuda_graph
        )

        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )

        self.cuda_graph_runner = None

    @property
    def draft_model_runner(self):
        return self.model_runner

    def get_attn_backend(self):  # pragma: no cover - exposed for adaptive
        return self.draft_model_runner.attn_backend

    def clear_cache_pool(self):
        pass

    def _bind_kv_context(self) -> None:
        try:
            from sglang.srt.models.gemma4_mtp import build_frozen_kv_context
        except ImportError:
            logger.debug("gemma4_mtp model not yet available; skipping frozen-kv bind.")
            return

        ctx = build_frozen_kv_context(
            assistant_model=self.draft_model_runner.model,
            target_model=self.target_worker.model_runner.model,
            target_token_to_kv_pool=self.target_worker.model_runner.token_to_kv_pool,
        )
        self.kv_context = ctx
        self.draft_model_runner.model.bind_frozen_kv_context(ctx)

    @contextmanager
    def _frozen_kv_target_view(self, forward_batch: ForwardBatch):
        """Temporarily clear ``spec_info`` and point ``token_to_kv_pool`` at
        the target so attention metadata is built for committed-prefix geometry.
        """
        if self.kv_context is None:
            raise RuntimeError(
                "FrozenKVMTPWorker._frozen_kv_target_view called before "
                "the model was bound; call _bind_kv_context() first."
            )
        saved_spec_info = forward_batch.spec_info
        saved_kv_pool = forward_batch.token_to_kv_pool
        forward_batch.spec_info = None
        forward_batch.token_to_kv_pool = self.kv_context.target_token_to_kv_pool
        try:
            yield
        finally:
            forward_batch.spec_info = saved_spec_info
            forward_batch.token_to_kv_pool = saved_kv_pool

    def _set_positions(self, forward_batch: ForwardBatch) -> None:
        """Rope phase = last written target slot: ``clamp(seq_lens-1)``, not
        advanced per draft step.
        """
        seq_lens = forward_batch.seq_lens
        positions = torch.clamp(seq_lens - 1, min=0).to(torch.int64)
        if forward_batch.positions is None:
            forward_batch.positions = positions
        else:
            if forward_batch.positions.shape == positions.shape:
                forward_batch.positions.copy_(positions)
            else:
                forward_batch.positions = positions

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        raise NotImplementedError(
            "FrozenKVMTPWorker.forward_batch_generation is not yet implemented."
        )

    def forward_target_extend(self, batch: ScheduleBatch):
        raise NotImplementedError(
            "FrozenKVMTPWorker.forward_target_extend is not yet implemented."
        )

    def forward_draft_extend(
        self,
        batch: ScheduleBatch,
        hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        mm_input_embeds: Optional[torch.Tensor] = None,
    ) -> None:
        raise NotImplementedError(
            "FrozenKVMTPWorker.forward_draft_extend is not yet implemented."
        )

    def forward_draft_extend_after_decode(self, batch: ScheduleBatch) -> None:
        raise NotImplementedError(
            "FrozenKVMTPWorker.forward_draft_extend_after_decode is not yet implemented."
        )

    def draft(self, batch: ScheduleBatch):
        raise NotImplementedError("FrozenKVMTPWorker.draft is not yet implemented.")

    def verify(self, batch: ScheduleBatch, spec_info):
        raise NotImplementedError("FrozenKVMTPWorker.verify is not yet implemented.")
