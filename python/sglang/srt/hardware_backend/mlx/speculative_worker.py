"""Synchronous MLX-native Gemma 4 Frozen-KV MTP worker."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.hardware_backend.mlx.gemma4_mtp import (
    Gemma4MTPAssistantLoader,
    Gemma4MTPAssistantRuntime,
)
from sglang.srt.hardware_backend.mlx.spec_decode import (
    MlxVerifySegment,
    build_linear_verify_queries,
    verify_greedy_segment,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.mlx.model_adapter import MlxTargetSeed
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.server_args import ServerArgs


class MlxFrozenKVMTPDraftInput(SpecInput):
    """CPU-only scheduler relay; MLX hidden/KV state remains worker-owned."""

    def __init__(
        self,
        *,
        request_ids: tuple[str, ...],
        draft_token_ids: torch.Tensor,
        valid_draft_counts: torch.Tensor,
        bonus_tokens: torch.Tensor,
        new_seq_lens: torch.Tensor,
        accept_tokens: torch.Tensor,
        accept_lens: torch.Tensor,
    ) -> None:
        super().__init__(SpecInputType.FROZEN_KV_MTP_DRAFT)
        size = len(request_ids)
        for name, value in (
            ("draft_token_ids", draft_token_ids),
            ("valid_draft_counts", valid_draft_counts),
            ("bonus_tokens", bonus_tokens),
            ("new_seq_lens", new_seq_lens),
            ("accept_lens", accept_lens),
        ):
            if value.ndim != 1 or len(value) != size:
                raise ValueError(f"{name} must be a flat tensor of length {size}")
        if accept_tokens.ndim != 1 or len(accept_tokens) != size * 2:
            raise ValueError("accept_tokens must use the fixed MLX result width of two")
        self.request_ids = tuple(request_ids)
        self.draft_token_ids = draft_token_ids.to(dtype=torch.long, device="cpu")
        self.valid_draft_counts = valid_draft_counts.to(dtype=torch.int32, device="cpu")
        self.bonus_tokens = bonus_tokens.to(dtype=torch.long, device="cpu")
        self.new_seq_lens = new_seq_lens
        self.accept_tokens = accept_tokens.to(dtype=torch.long, device="cpu")
        self.accept_lens = accept_lens.to(dtype=torch.int32, device="cpu")
        self.num_tokens_per_req = 2
        self.num_tokens_for_logprob_per_req = 2
        self.topk_p = None
        self.topk_index = None
        self.hidden_states = None
        self.draft_probs = None
        self.dsa_topk_indices = None
        self.future_indices = None

    @property
    def draft_token(self) -> torch.Tensor:
        return self.draft_token_ids

    def filter_batch(
        self,
        new_indices: torch.Tensor,
        has_been_filtered: bool = True,
        new_indices_cpu: Optional[list[int]] = None,
    ) -> None:
        del has_been_filtered
        indices_cpu = (
            list(new_indices_cpu)
            if new_indices_cpu is not None
            else [int(index) for index in new_indices.to("cpu").tolist()]
        )
        indices = torch.tensor(indices_cpu, dtype=torch.long, device="cpu")
        self.request_ids = tuple(self.request_ids[index] for index in indices_cpu)
        self.draft_token_ids = self.draft_token_ids[indices]
        self.valid_draft_counts = self.valid_draft_counts[indices]
        self.bonus_tokens = self.bonus_tokens[indices]
        self.new_seq_lens = self.new_seq_lens[new_indices.to(self.new_seq_lens.device)]
        self.accept_tokens = self.accept_tokens.reshape(-1, 2)[indices].reshape(-1)
        self.accept_lens = self.accept_lens[indices]
        if self.future_indices is not None:
            self.future_indices = self.future_indices[new_indices]

    def merge_batch(self, other: MlxFrozenKVMTPDraftInput) -> None:
        if not isinstance(other, MlxFrozenKVMTPDraftInput):
            raise TypeError("cannot merge a non-MLX Frozen-KV draft input")
        self.request_ids += other.request_ids
        self.draft_token_ids = torch.cat((self.draft_token_ids, other.draft_token_ids))
        self.valid_draft_counts = torch.cat(
            (self.valid_draft_counts, other.valid_draft_counts)
        )
        self.bonus_tokens = torch.cat((self.bonus_tokens, other.bonus_tokens))
        self.new_seq_lens = torch.cat((self.new_seq_lens, other.new_seq_lens))
        self.accept_tokens = torch.cat((self.accept_tokens, other.accept_tokens))
        self.accept_lens = torch.cat((self.accept_lens, other.accept_lens))
        if self.future_indices is not None or other.future_indices is not None:
            if self.future_indices is None or other.future_indices is None:
                raise ValueError("future-index relay state must agree when merging")
            self.future_indices = torch.cat((self.future_indices, other.future_indices))


@dataclass
class MlxGemma4MTPMetrics:
    proposed_tokens: int = 0
    verified_tokens: int = 0
    accepted_draft_tokens: int = 0
    target_verify_seconds: float = 0.0
    assistant_propose_seconds: float = 0.0
    transaction_commit_seconds: float = 0.0


class MlxGemma4MTPProposer:
    def __init__(self, runtime: Gemma4MTPAssistantRuntime):
        self.runtime = runtime

    @staticmethod
    def needs_target_hidden_states() -> bool:
        return True

    def propose_one(self, request_id: str, seed: MlxTargetSeed, cache) -> int:
        # One live view and seed per RID bounds lifecycle state across long runs.
        self.runtime.release_request(request_id)
        bound_seed = self.runtime.bind_seed(request_id, seed)
        view = self.runtime.bind_request(request_id, cache)
        return self.runtime.propose_one(bound_seed, view)


class MlxFrozenKVMTPWorker(BaseSpecWorker):
    """One-proposal, greedy, BS=1 native-cache spec-v2 orchestration."""

    def __init__(self, server_args: ServerArgs, gpu_id, ps, nccl_port, target_worker):
        del gpu_id, ps, nccl_port
        if not hasattr(target_worker, "_mlx_runner"):
            raise TypeError("MlxFrozenKVMTPWorker requires MlxTpModelWorker")
        if not target_worker._mlx_runner.native_cache_fallback:
            raise ValueError("MLX Frozen-KV MTP requires native Gemma 4 target caches")
        self.server_args = server_args
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self._draft_worker = None
        self.speculative_num_draft_tokens = 2
        self._native_runner = target_worker._mlx_runner
        self._target_adapter = self._native_runner._target_adapter
        self._assistant_loader = Gemma4MTPAssistantLoader(self._native_runner.model)
        self._assistant_runtime = self._assistant_loader.load(
            server_args.speculative_draft_model_path,
            revision=getattr(server_args, "speculative_draft_model_revision", None),
        )
        self._proposer = MlxGemma4MTPProposer(self._assistant_runtime)
        self.metrics = MlxGemma4MTPMetrics()
        self._active_rids: set[str] = set()

    @property
    def draft_worker(self):
        return None

    def carries_draft_hidden_states(self) -> bool:
        return False

    def get_draft_kv_pool(self):
        """The assistant reads target KV and deliberately allocates no pool."""

        return None

    def get_speculative_internal_state(self) -> dict[str, object]:
        """Return JSON-safe activation and lifecycle state for ``/server_info``."""

        import mlx.core as mx

        runtime = self._assistant_loader.runtime
        return {
            "implementation": "mlx_gemma4_frozen_kv_mtp",
            "proposed_tokens": self.metrics.proposed_tokens,
            "verified_tokens": self.metrics.verified_tokens,
            "accepted_draft_tokens": self.metrics.accepted_draft_tokens,
            "target_verify_seconds": self.metrics.target_verify_seconds,
            "assistant_propose_seconds": self.metrics.assistant_propose_seconds,
            "transaction_commit_seconds": self.metrics.transaction_commit_seconds,
            "assistant_load_count": self._assistant_loader.load_count,
            "assistant_generation": self._assistant_loader.generation,
            "assistant_fingerprint": None if runtime is None else runtime.fingerprint,
            "active_request_count": len(self._active_rids),
            "native_request_count": len(self._native_runner._req_caches),
            "assistant_request_binding_count": (
                0 if runtime is None else runtime.request_binding_count
            ),
            "mlx_active_memory_bytes": int(mx.get_active_memory()),
            "mlx_cache_memory_bytes": int(mx.get_cache_memory()),
            "mlx_peak_memory_bytes": int(mx.get_peak_memory()),
        }

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ) -> None:
        del memory_pool_config
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator

    def init_attention_backends(self) -> None:
        # The target worker is initialized independently by Scheduler; this
        # worker owns no second attention backend.
        return None

    def init_cuda_graphs(self) -> None:
        # Native MLX verification is synchronous and captures no Torch graphs.
        return None

    def _cleanup_departed(self, current_rids: set[str]) -> None:
        for rid in self._active_rids - current_rids:
            self._assistant_runtime.release_request(rid)
        self._active_rids = set(current_rids)

    @staticmethod
    def _padded_tokens(emitted: tuple[int, ...]) -> torch.Tensor:
        if len(emitted) not in (1, 2):
            raise ValueError("MLX MTP emitted width must be one or two")
        row = list(emitted) + [-1] * (2 - len(emitted))
        return torch.tensor(row, dtype=torch.long, device="cpu")

    def _make_draft_input(
        self,
        *,
        request_id: str,
        proposal: int,
        bonus_token: int,
        new_seq_lens: torch.Tensor,
        accept_tokens: torch.Tensor,
        accept_len: int,
    ) -> MlxFrozenKVMTPDraftInput:
        return MlxFrozenKVMTPDraftInput(
            request_ids=(request_id,),
            draft_token_ids=torch.tensor([proposal], dtype=torch.long),
            valid_draft_counts=torch.tensor(
                [0 if proposal == -1 else 1], dtype=torch.int32
            ),
            bonus_tokens=torch.tensor([bonus_token], dtype=torch.long),
            new_seq_lens=new_seq_lens,
            accept_tokens=accept_tokens,
            accept_lens=torch.tensor([accept_len], dtype=torch.int32),
        )

    def _propose_from_output(
        self,
        request_id: str,
        target_output,
        hidden_row_index: int,
        emitted_token_id: int,
    ) -> int:
        seed = self._target_adapter.make_seed(
            target_output,
            hidden_row_index=hidden_row_index,
            emitted_token_id=emitted_token_id,
        )
        start = time.perf_counter()
        proposal = self._proposer.propose_one(
            request_id, seed, self._native_runner._req_caches[request_id]
        )
        self.metrics.assistant_propose_seconds += time.perf_counter() - start
        self.metrics.proposed_tokens += 1
        return proposal

    def _forward_prefill(self, batch: ScheduleBatch) -> GenerationBatchResult:
        target_result, target_output = (
            self._target_worker.forward_batch_generation_mtp_prefill(batch)
        )
        request = batch.reqs[0]
        token = int(target_result.next_token_ids[0])
        proposal = self._propose_from_output(
            request.rid,
            target_output,
            hidden_row_index=target_output.hidden_states.shape[1] - 1,
            emitted_token_id=token,
        )
        new_seq_lens = batch.seq_lens.clone()
        padded = self._padded_tokens((token,))
        target_result.next_draft_input = self._make_draft_input(
            request_id=request.rid,
            proposal=proposal,
            bonus_token=token,
            new_seq_lens=new_seq_lens,
            accept_tokens=padded,
            accept_len=1,
        )
        target_result.new_seq_lens = new_seq_lens
        target_result.speculative_num_draft_tokens = 2
        return target_result

    def _forward_decode(self, batch: ScheduleBatch) -> GenerationBatchResult:
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        if len(batch.reqs) != 1:
            raise ValueError("MLX Frozen-KV MTP MVP supports batch size one")
        draft_input = batch.spec_info
        if not isinstance(draft_input, MlxFrozenKVMTPDraftInput):
            raise TypeError("MLX Frozen-KV MTP decode requires its native draft input")
        request = batch.reqs[0]
        if draft_input.request_ids != (request.rid,):
            raise ValueError("MLX MTP draft handoff request ordering is stale")
        root = int(self._native_runner._req_token_ids[request.rid][-1])
        if int(draft_input.bonus_tokens[0]) != root:
            raise ValueError("MLX MTP draft handoff bonus token is stale")
        valid = int(draft_input.valid_draft_counts[0])
        draft = int(draft_input.draft_token_ids[0])
        if valid not in (0, 1) or (valid == 0) != (draft == -1):
            raise ValueError("MLX MTP draft sentinel/count metadata is malformed")

        placeholder = MlxVerifySegment(
            request_id=request.rid,
            draft_tokens=(draft,),
            valid_draft_count=valid,
            invalid_draft_count=1 - valid,
            target_token_ids=(0,) * (valid + 1),
            verification_query_count=valid + 1,
        )
        queries = build_linear_verify_queries(root, placeholder)
        pending = None
        verify_start = time.perf_counter()
        try:
            pending = self._native_runner.verify_start(request.rid, queries)
            target_ids = self._native_runner.verify_materialize(pending)
            segment = MlxVerifySegment(
                request_id=request.rid,
                draft_tokens=(draft,),
                valid_draft_count=valid,
                invalid_draft_count=1 - valid,
                target_token_ids=target_ids,
                verification_query_count=len(queries),
            )
            decision = verify_greedy_segment(segment)
            self.metrics.target_verify_seconds += time.perf_counter() - verify_start
            commit_start = time.perf_counter()
            self._native_runner.verify_finalize(pending, decision)
            self.metrics.transaction_commit_seconds += (
                time.perf_counter() - commit_start
            )
        except BaseException:
            if pending is not None:
                self._native_runner.verify_abort(pending)
            self._assistant_runtime.release_request(request.rid)
            raise

        emitted = decision.emitted_token_ids
        padded = self._padded_tokens(emitted)
        accept_lens = torch.tensor([len(emitted)], dtype=torch.int32, device="cpu")
        new_seq_lens = batch.seq_lens + accept_lens.to(batch.seq_lens.device)
        proposal = self._propose_from_output(
            request.rid,
            pending.target_output,
            hidden_row_index=decision.seed_hidden_row_index,
            emitted_token_id=emitted[-1],
        )
        next_input = self._make_draft_input(
            request_id=request.rid,
            proposal=proposal,
            bonus_token=emitted[-1],
            new_seq_lens=new_seq_lens,
            accept_tokens=padded,
            accept_len=len(emitted),
        )
        self.metrics.verified_tokens += valid
        self.metrics.accepted_draft_tokens += decision.accepted_draft_count
        return GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=padded,
            num_correct_drafts=decision.accepted_draft_count,
            num_correct_drafts_per_req_cpu=[decision.accepted_draft_count],
            can_run_cuda_graph=False,
            accept_lens=accept_lens,
            new_seq_lens=new_seq_lens,
            next_draft_input=next_input,
            speculative_num_draft_tokens=2,
        )

    def _forward_idle(self, batch: ScheduleBatch) -> GenerationBatchResult:
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        empty_long = torch.empty((0,), dtype=torch.long, device="cpu")
        empty_int = torch.empty((0,), dtype=torch.int32, device="cpu")
        empty_seq = batch.seq_lens.clone()
        draft_input = MlxFrozenKVMTPDraftInput(
            request_ids=(),
            draft_token_ids=empty_long,
            valid_draft_counts=empty_int,
            bonus_tokens=empty_long,
            new_seq_lens=empty_seq,
            accept_tokens=empty_long,
            accept_lens=empty_int,
        )
        return GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=empty_long,
            accept_lens=empty_int,
            new_seq_lens=empty_seq,
            next_draft_input=draft_input,
            speculative_num_draft_tokens=2,
            can_run_cuda_graph=False,
        )

    def forward_batch_generation(self, batch: ScheduleBatch, on_publish=None, **kwargs):
        del kwargs
        current_rids = {request.rid for request in batch.reqs}
        self._cleanup_departed(current_rids)
        if batch.forward_mode.is_idle():
            result = self._forward_idle(batch)
        elif batch.forward_mode.is_extend():
            if len(batch.reqs) != 1:
                raise ValueError("MLX Frozen-KV MTP MVP supports batch size one")
            result = self._forward_prefill(batch)
        elif batch.forward_mode.is_decode():
            result = self._forward_decode(batch)
        else:
            raise ValueError(f"MLX Frozen-KV MTP does not support {batch.forward_mode}")
        self._active_rids = current_rids
        if on_publish is not None and result.new_seq_lens is not None:
            on_publish(result.new_seq_lens)
        return result

    def note_request_finished(self, *, rid: str, natural_stop: bool) -> None:
        del natural_stop
        self._assistant_runtime.release_request(rid)
        self._active_rids.discard(rid)

    def prepare_for_kv_cache_release(self, req) -> None:
        self._assistant_runtime.release_request(req.rid)
        self._active_rids.discard(req.rid)
        self._target_worker.prepare_for_kv_cache_release(req)

    def clear_cache_pool(self) -> None:
        self._target_worker.clear_cache_pool()
        self._assistant_loader.clear_request_bindings()
        self._active_rids.clear()

    def unload_assistant(self) -> None:
        self._assistant_loader.unload_assistant()

    def replace_assistant(self, checkpoint: str, revision: str | None = None) -> None:
        self._assistant_runtime = self._assistant_loader.replace_assistant(
            checkpoint, revision=revision
        )
        self._proposer = MlxGemma4MTPProposer(self._assistant_runtime)

    @staticmethod
    def _unsupported_weight_update():
        return (
            False,
            "MLX Frozen-KV MTP cannot infer an assistant checkpoint/revision "
            "from generic weight-update requests; restart an idle server with "
            "both revisions pinned.",
        )

    def update_weights_from_disk(self, recv_req):
        return self._unsupported_weight_update()

    def update_weights_from_ipc(self, recv_req):
        return self._unsupported_weight_update()

    def update_weights_from_tensor(self, recv_req):
        return self._unsupported_weight_update()
