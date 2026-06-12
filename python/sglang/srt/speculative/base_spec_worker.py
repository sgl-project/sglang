from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.speculative.eagle_info import EagleDraftExtendInput


class EagleDraftWorkerBase(ABC):
    @abstractmethod
    def draft():
        pass

    @abstractmethod
    def draft_extend():
        pass

    def alloc_memory_pool(self, **kwargs):
        pass

    def init_backends(self):
        """Initialize standard backends (no cuda graphs) then draft-specific backends.

        Subclasses should wrap this with their context managers (draft_tp_context,
        speculative_moe_backend_context, etc.) rather than reimplementing the logic.
        """
        self.draft_worker.init_backends(disable_cuda_graph=True)
        self.init_attention_backend()
        self.init_cuda_graphs()

    def prepare_for_draft_extend(
        self,
        draft_extend_input: EagleDraftExtendInput,
        batch: ScheduleBatch,
        predict: torch.Tensor,
        num_draft_tokens: int,
        draft_model_runner: Any,
        cuda_graph_runner: Any,
    ):
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
            ForwardMode,
        )
        from sglang.srt.utils.async_probe import maybe_detect_oob
        from sglang.srt.utils.common import is_npu

        bs = len(batch.seq_lens)
        extend_num_tokens = bs * num_draft_tokens
        # When seq_lens_cpu is absent, stay on GPU-only path -- no .tolist()/.cpu().
        gpu_only = batch.seq_lens_cpu is None

        batch.spec_info = draft_extend_input
        batch.input_ids = predict
        maybe_detect_oob(
            batch.input_ids,
            0,
            batch.model_config.vocab_size,
            "v2 prepare_for_draft_extend input_ids",
        )
        # init_new requires both list or both Tensor;
        # gpu_only emits device tensors to skip H2D.
        if gpu_only:
            batch.prefix_lens = batch.seq_lens.to(torch.int32)
            batch.extend_lens = torch.full(
                (bs,), num_draft_tokens, dtype=torch.int32, device=batch.seq_lens.device
            )
        else:
            batch.prefix_lens = batch.seq_lens_cpu.tolist()
            batch.extend_lens = [num_draft_tokens] * bs
        batch.extend_num_tokens = extend_num_tokens
        capture_mode = (
            CaptureHiddenMode.NULL
            if draft_model_runner.spec_algorithm.is_standalone()
            else CaptureHiddenMode.FULL
        )
        batch.forward_mode = (
            ForwardMode.IDLE
            if batch.forward_mode.is_idle()
            else ForwardMode.DRAFT_EXTEND_V2
        )
        batch.capture_hidden_mode = capture_mode
        forward_batch = ForwardBatch.init_new(batch, draft_model_runner)
        # Forward sees post-write length (draft extend writes num_draft_tokens
        # slots); mutation stays on forward_batch to preserve SB.seq_lens.
        forward_batch.seq_lens = forward_batch.seq_lens + num_draft_tokens
        if not gpu_only:
            forward_batch.seq_lens_cpu = forward_batch.seq_lens_cpu + num_draft_tokens
            forward_batch.seq_lens_sum = int(forward_batch.seq_lens_cpu.sum())
        else:
            # Supply CPU mirror (extend_seq_lens are all num_draft_tokens) so
            # backend max() reads from list without a per-iter D2H sync.
            forward_batch.extend_seq_lens_cpu = [num_draft_tokens] * bs
        can_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run(forward_batch)
        if not batch.forward_mode.is_idle() and not can_cuda_graph:
            draft_model_runner.attn_backend.init_forward_metadata(forward_batch)
            # Planned pre-pad; do NOT opt into post-pad re-plan. DSA's indexer
            # cannot rebuild its deep_gemm schedule_meta on a DP-padded batch
            # (the `_batch_size == batch_size` assertion, see #27091); the
            # marked pre-pad metadata is used as-is, matching the proven
            # skip_attn_backend_init=True behavior.
            # On NPU with --disable-cuda-graph, block_table shape won't match
            # after prepare_mlp_sync_batch padding; defer re-init to
            # forward_extend (post-pad) instead.
            if not is_npu() or can_cuda_graph:
                forward_batch.mark_forward_metadata_ready()
        return forward_batch


class BaseSpecWorker(ABC):
    @property
    @abstractmethod
    def target_worker(self) -> TpModelWorker:
        pass

    @property
    @abstractmethod
    def draft_worker(self) -> EagleDraftWorkerBase:
        pass

    @property
    def spec_v2_attn_backends(self) -> tuple:
        """Attn backends touched by spec_v2 forward; OR-ed by decide_needs_cpu_seq_lens.
        Default returns target only; subclasses extend with draft backends."""
        return (self.target_worker.model_runner.attn_backend,)

    @abstractmethod
    def clear_cache_pool(self):
        # TODO: move this abstract method to BaseTpWorker and call through self.model_runner
        pass

    def alloc_memory_pool(self, **kwargs):
        pass

    def init_backends(self):
        pass

    def on_verify_complete_cpu(
        self, num_correct_drafts_per_req: list[int], batch_size: int = 0
    ) -> None:
        """Hook called after verify finishes and accept counts are on CPU.

        Default no-op. Adaptive-aware workers override this to feed the
        controller without forcing a GPU→CPU sync in the worker hot path.
        """
        pass

    def activate_step_by_batch(self, batch_size: int) -> None:
        """Activate the optimal adaptive step for the current batch size.

        Default no-op. Adaptive-aware workers override this to switch
        the runtime state before each draft round.
        """
        pass
