"""Spec-v2 worker for HY4 DFly.

Each decode step runs one target verify forward. Before the target embeds its
input, the draft runs over the anchor plus ``N`` mask queries through the
target's attention backend, writing the block's KV into target layers at the
verify slots; its proposals become the verify input, and the target layers
then overwrite every draft write that could be read again. The draft owns no
KV pool. A decode CUDA graph captures that whole forward, draft included;
prefill runs eagerly or replays a breakable prefill graph of the target. Prefill
and verify forwards write per-request target state (final hidden, index top-k)
by request-pool row for the next step.
"""

from __future__ import annotations

import dataclasses
import logging
from contextlib import contextmanager
from typing import Optional

import msgspec
import torch

from sglang.kernels.ops.attention.dfly import build_dfly_sparse_indices_stacked
from sglang.kernels.ops.speculative.cache_locs import assign_extend_cache_locs_func
from sglang.kernels.ops.speculative.dflash import (
    compute_dflash_correct_drafts_and_bonus,
)
from sglang.kernels.ops.speculative.dfly import can_fuse_dfly_inputs, plan_dfly_inputs
from sglang.srt.configs.hy_v4_dfly import hyv4_dfly_cuda_graph_conflicts
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.layers.dp_attention import get_dp_tp_group, require_mlp_sync
from sglang.srt.layers.logits_processor import get_in_autotune_dummy_run
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.kv.alloc.runner_ids import RunnerIds
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import (
    forward_context,
    get_attn_backend,
    get_forward_context,
)
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode, ForwardMode
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
from sglang.srt.models.hunyuan_v4 import HYV4ForCausalLM
from sglang.srt.models.hunyuan_v4_dfly import HYV4ForCausalLMDFly
from sglang.srt.models.hunyuan_v4_vl import HYV4VLForConditionalGeneration
from sglang.srt.runtime_context import attention_backends, get_exec, get_spec
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_utils import commit_accept
from sglang.srt.speculative.dfly_utils import (
    DFLY_DISCARD_SLOT,
    DFlyBlock,
    DFlyProducerCapture,
    VLDFlyRequestState,
    build_dfly_sparse_indices,
    check_dfly_attention_backends,
    dfly_draft_topk_planes,
    dfly_full_width_lengths,
    dfly_mrope_positions,
    dfly_position_ids,
    dfly_producer_layers,
    dfly_source_rows,
    dfly_sparse_attention_metadata,
    dfly_verify_positions,
    dfly_window_positions,
    plan_dfly_block,
    split_dfly_drafts_for_verify,
)
from sglang.srt.speculative.draft_worker_common import (
    build_draft_tp_worker,
    make_draft_block_spec_info,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_tp_sync import SpecTpSync, SpecTpSyncSite
from sglang.srt.utils import async_h2d_stack

logger = logging.getLogger(__name__)


def _hyv4_language_model(model) -> HYV4ForCausalLM:
    if isinstance(model, HYV4VLForConditionalGeneration):
        return model.language_model
    if not isinstance(model, HYV4ForCausalLM):
        raise ValueError(f"DFLY drafts for HYV4 targets only, got {type(model)}")
    return model


def _attach_topk_capture(
    *, model: HYV4ForCausalLM, capture: DFlyProducerCapture, layer_ids: tuple[int, ...]
) -> None:
    for layer_id in layer_ids:
        model.model.layers[layer_id].self_attn.dfly_topk_capture = capture


@contextmanager
def _target_kv_pool(backend, target_pool):
    # The backend routes reads and writes through its pool on every access.
    saved = backend.token_to_kv_pool
    backend.token_to_kv_pool = target_pool
    try:
        yield
    finally:
        backend.token_to_kv_pool = saved


def _make_draft_input(
    *, bonus_tokens: torch.Tensor, new_seq_lens: torch.Tensor
) -> DFlashDraftInputV2:
    # The target state lives in VLDFlyRequestState by pool row, so the batch
    # carries only bonus tokens and lengths through filter and merge.
    empty = (new_seq_lens.numel(), 0)
    return DFlashDraftInputV2(
        topk_p=torch.empty(empty, dtype=torch.float32, device=bonus_tokens.device),
        topk_index=torch.empty(empty, dtype=torch.int32, device=bonus_tokens.device),
        bonus_tokens=bonus_tokens.to(torch.int64),
        new_seq_lens=new_seq_lens.to(torch.int64),
        hidden_states=torch.empty(empty, device=bonus_tokens.device),
    )


class VLDFlyDraftInputs(msgspec.Struct, frozen=True, kw_only=True):
    """One step's draft block and the target state it reads: ``[bs, hidden]``
    target hidden, one sparse-row tensor per draft layer, and ``[3, rows]``
    mRoPE positions when the draft rotates by mRoPE."""

    block: DFlyBlock
    target_hidden: torch.Tensor
    topk_indices: tuple[torch.Tensor, ...]
    mrope_positions: Optional[torch.Tensor]


class VLDFlyExecution:
    """Runs the loaded VL draft inside the target's verify forward.

    The top-level target model calls ``draft_into_verify`` once per
    TARGET_VERIFY forward, before it embeds its input ids. The draft reads and
    writes target KV through the target's attention backend under a draft view
    of the batch, then its proposals replace the block's placeholder ids in
    place. The draft's loader runner keeps owning its weights; this holds a
    plain reference, so module traversal never reaches the draft twice.

    The draft reads only the verify batch's tensors, the request state and the
    fixed buffers here, so a decode CUDA graph captures it with the target and
    every replay drafts from that step's inputs. The buffers hold a block for
    every request the pool can run, which bounds every graph bucket.
    """

    def __init__(
        self,
        *,
        draft_model: HYV4ForCausalLMDFly,
        state: VLDFlyRequestState,
        draft_topk_planes: tuple[int, ...],
        physical_ids: bool,
        max_context_len: int,
        req_to_token: torch.Tensor,
        runner_ids: RunnerIds,
        block_spec_info: DFlashVerifyInput,
        max_requests: int,
        device: torch.device | str,
    ):
        self._draft_model = draft_model
        self._spec = draft_model.spec
        self._state = state
        self._draft_topk_planes = draft_topk_planes
        self._physical_ids = physical_ids
        self._max_context_len = max_context_len
        self._req_to_token = req_to_token
        self._runner_ids = runner_ids
        self._block_spec_info = block_spec_info
        self._max_requests = max_requests
        verify_rows = max_requests * self._spec.query_width
        draft_topk = self._spec.draft_index_topk
        self._live = torch.zeros(max_requests, dtype=torch.bool, device=device)
        self._candidates = torch.full(
            (max_requests, self._spec.query_width), -1, dtype=torch.int64, device=device
        )
        # One sparse-row plane per producer; draft layers sharing one read it.
        self._draft_rows = torch.full(
            (len(set(draft_topk_planes)), verify_rows, draft_topk),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._draft_lengths, self._draft_cu_lengths = dfly_full_width_lengths(
            num_rows=verify_rows, index_topk=draft_topk, device=device
        )
        self._fuse_inputs = can_fuse_dfly_inputs(
            device,
            query_width=self._spec.query_width,
            window=self._spec.context_local_window,
        )
        self._plan_ids = torch.empty(verify_rows, dtype=torch.int64, device=device)
        self._plan_positions = torch.empty_like(self._plan_ids)
        self._plan_verify_positions = torch.empty_like(self._plan_ids)
        self._plan_slots = torch.empty_like(self._plan_ids)
        self._plan_anchors = torch.empty(max_requests, dtype=torch.int64, device=device)
        self._plan_active = torch.empty(max_requests, dtype=torch.bool, device=device)
        self._plan_hidden = torch.empty(
            (max_requests, state.hidden.shape[1]),
            dtype=state.hidden.dtype,
            device=device,
        )
        self._plan_mrope = torch.empty(
            (3, verify_rows), dtype=torch.int64, device=device
        )
        self._sorted_window = torch.empty(
            (max_requests, self._spec.context_local_window),
            dtype=torch.int64,
            device=device,
        )
        self._staged_bs: Optional[int] = None

    def stage(self, *, live: torch.Tensor) -> None:
        """Mark which of the next verify's requests may draft; every row past
        them, a graph bucket's padding included, drafts nothing."""
        bs = live.shape[0]
        self._check_capacity(bs)
        self._live[:bs].copy_(live)
        self._live[bs:].zero_()
        self._staged_bs = bs

    def unstage(self) -> None:
        self._staged_bs = None
        self._live.zero_()

    def stage_capture(self, *, batch_size: int) -> None:
        """Stage a decode graph capture's verify over the runner's dummy rows.
        Outside a step no row is live, so it drafts and commits nothing."""
        if self._staged_bs is not None:
            raise RuntimeError("DFly graph capture overlaps a staged decode step")
        self._check_capacity(batch_size)
        self._staged_bs = batch_size

    def end_capture(self) -> None:
        self._staged_bs = None

    def _check_capacity(self, batch_size: int) -> None:
        if batch_size > self._max_requests:
            raise RuntimeError(
                f"DFly verify buffers hold {self._max_requests} requests, "
                f"a verify has {batch_size}"
            )

    def candidates(self, bs: int) -> torch.Tensor:
        """``[bs, N + 1]`` accept candidates of the last verify: the bonus,
        then the drafts, -1 for a request that did not draft."""
        return self._candidates[:bs]

    def plan(
        self,
        *,
        req_pool_indices: torch.Tensor,
        anchors: torch.Tensor,
        bonus_tokens: torch.Tensor,
        live: torch.Tensor,
        verify_slots: torch.Tensor,
    ) -> VLDFlyDraftInputs:
        self._check_capacity(anchors.numel())
        if self._fuse_inputs:
            return self._plan_fused(
                req_pool_indices=req_pool_indices,
                anchors=anchors,
                bonus_tokens=bonus_tokens,
                live=live,
                verify_slots=verify_slots,
            )
        block = plan_dfly_block(
            bonus_tokens=bonus_tokens,
            anchors=anchors,
            sampled=live,
            verify_slots=verify_slots,
            mask_token_id=self._spec.mask_token_id,
            max_context_len=self._max_context_len,
        )
        mrope_positions = None
        if self._draft_model.is_mrope_enabled:
            mrope_positions = dfly_mrope_positions(
                positions=block.positions,
                active=block.active,
                deltas=self._state.mrope_delta[req_pool_indices],
            )
        return VLDFlyDraftInputs(
            block=block,
            target_hidden=self._state.hidden[req_pool_indices],
            topk_indices=self._sparse_indices(
                req_pool_indices=req_pool_indices, block=block
            ),
            mrope_positions=mrope_positions,
        )

    def _plan_fused(
        self, *, req_pool_indices, anchors, bonus_tokens, live, verify_slots
    ) -> VLDFlyDraftInputs:
        bs = anchors.numel()
        width = self._spec.query_width
        num_rows = bs * width
        if verify_slots.shape != (bs, width):
            raise ValueError(f"DFly verify slots must have shape {(bs, width)}")
        block = DFlyBlock(
            input_ids=self._plan_ids[:num_rows],
            positions=self._plan_positions[:num_rows].view(bs, width),
            verify_positions=self._plan_verify_positions[:num_rows].view(bs, width),
            write_slots=self._plan_slots[:num_rows],
            anchors=self._plan_anchors[:bs],
            active=self._plan_active[:bs],
        )
        hidden = self._plan_hidden[:bs]
        mrope = (
            self._plan_mrope[:, :num_rows]
            if self._draft_model.is_mrope_enabled
            else None
        )
        plan_dfly_inputs(
            req_pool_indices=req_pool_indices,
            anchors=anchors,
            bonus_tokens=bonus_tokens,
            live=live,
            verify_slots=verify_slots,
            hidden=self._state.hidden,
            mrope_delta=self._state.mrope_delta,
            req_to_token=self._req_to_token,
            input_ids=block.input_ids,
            positions=block.positions,
            verify_positions=block.verify_positions,
            write_slots=block.write_slots,
            out_anchors=block.anchors,
            active=block.active,
            target_hidden=hidden,
            mrope_positions=mrope,
            sorted_window=self._sorted_window[:bs],
            mask_token_id=self._spec.mask_token_id,
            max_context_len=self._max_context_len,
            discard_slot=DFLY_DISCARD_SLOT,
            physical=self._physical_ids,
        )
        return VLDFlyDraftInputs(
            block=block,
            target_hidden=hidden,
            topk_indices=self._sparse_indices(
                req_pool_indices=req_pool_indices, block=block
            ),
            mrope_positions=mrope,
        )

    def draft_into_verify(
        self, *, input_ids: torch.Tensor, forward_batch: ForwardBatch
    ) -> None:
        """Replace ``input_ids`` ``[bs * (N + 1)]``, each request's bonus token
        then placeholders, with the bonus token and its drafts."""
        bs = forward_batch.batch_size
        if self._staged_bs is None and get_in_autotune_dummy_run():
            # FlashInfer autotune's warmup verify has no requests and discards
            # its output, so it tunes the target alone.
            return
        if self._staged_bs != bs:
            raise RuntimeError(
                f"DFly verify of {bs} requests has no staged draft step "
                f"(staged {self._staged_bs})"
            )
        query_width = self._spec.query_width
        bonus_tokens = input_ids.view(bs, query_width)[:, 0].clone()
        inputs = self.plan(
            req_pool_indices=forward_batch.req_pool_indices,
            anchors=forward_batch.seq_lens,
            bonus_tokens=bonus_tokens,
            live=self._live[:bs],
            verify_slots=forward_batch.out_cache_loc.view(bs, query_width),
        )
        hidden_states = self._forward_block(forward_batch, inputs)
        drafts = self._draft_model.propose(
            hidden_states, bonus_tokens, inputs.block.active
        )
        candidates, verify_ids = split_dfly_drafts_for_verify(
            bonus_tokens=bonus_tokens, draft_tokens=drafts
        )
        self._candidates[:bs].copy_(candidates)
        input_ids.copy_(verify_ids)

    def _forward_block(
        self, forward_batch: ForwardBatch, inputs: VLDFlyDraftInputs
    ) -> torch.Tensor:
        block = inputs.block
        view = dataclasses.replace(
            forward_batch,
            input_ids=block.input_ids,
            positions=block.positions.flatten(),
            out_cache_loc=block.write_slots,
            origin_out_cache_loc=None,
            mrope_positions=inputs.mrope_positions,
            spec_info=self._block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            residual_stream=None,
        )
        self._runner_ids.bind_own_plan(view)
        num_rows = block.input_ids.shape[0]
        backend = get_attn_backend()
        target_metadata = backend.forward_metadata
        backend.forward_metadata = dfly_sparse_attention_metadata(
            target_metadata,
            lengths=self._draft_lengths[:num_rows],
            cu_lengths=self._draft_cu_lengths[: num_rows + 1],
        )
        try:
            with forward_context(get_forward_context(), forward_batch=view):
                return self._draft_model.forward_block(
                    input_ids=view.input_ids,
                    positions=view.positions,
                    forward_batch=view,
                    target_hidden=inputs.target_hidden,
                    topk_indices=inputs.topk_indices,
                )
        finally:
            backend.forward_metadata = target_metadata

    def _sparse_indices(
        self, *, req_pool_indices: torch.Tensor, block: DFlyBlock
    ) -> tuple[torch.Tensor, ...]:
        if self._fuse_inputs:
            num_rows = block.input_ids.numel()
            build_dfly_sparse_indices_stacked(
                target_topk=self._state.topk,
                req_pool_indices=req_pool_indices,
                anchors=block.anchors,
                active=block.active,
                req_to_token=self._req_to_token,
                sorted_window=self._sorted_window[: block.anchors.numel()],
                query_width=self._spec.query_width,
                physical=self._physical_ids,
                out=self._draft_rows[:, :num_rows],
            )
            return tuple(
                self._draft_rows[plane, :num_rows] for plane in self._draft_topk_planes
            )

        def ids(positions):
            return dfly_position_ids(
                positions=positions,
                req_to_token=self._req_to_token,
                req_pool_indices=req_pool_indices,
                physical=self._physical_ids,
            )

        window_ids = ids(
            dfly_window_positions(
                anchors=block.anchors,
                context_local_window=self._spec.context_local_window,
            )
        )
        block_ids = ids(block.positions)
        target_topk = self._state.topk[:, req_pool_indices]
        num_rows = block.input_ids.shape[0]
        rows = [
            build_dfly_sparse_indices(
                target_rows=target_topk[plane],
                window_ids=window_ids,
                block_ids=block_ids,
                active=block.active,
                index_topk=self._spec.draft_index_topk,
                out=self._draft_rows[plane, :num_rows],
            )
            for plane in range(self._draft_rows.shape[0])
        ]
        return tuple(rows[plane] for plane in self._draft_topk_planes)


def _check_serving_layout() -> None:
    graph = get_exec().graph.cuda_graph_config
    conflicts = hyv4_dfly_cuda_graph_conflicts(
        decode_backend=graph.decode.backend, prefill_backend=graph.prefill.backend
    )
    if require_mlp_sync():
        conflicts.append(
            "padded or attention-TP-sharded verify batches (an MoE all-to-all "
            "backend or --moe-dense-tp-size)"
        )
    if conflicts:
        raise ValueError(f"DFLY does not support {'; '.join(conflicts)}.")


class DFlyWorkerV2(BaseSpecWorker):
    """DFly worker using the spec-v2 relay for overlap scheduling."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        super().__init__()
        _check_serving_layout()
        self.server_args = server_args
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.device = target_worker.device
        self._target_model = _hyv4_language_model(self.model_runner.model)
        # The draft reads and writes target KV, so it must run the target's DSA backend.
        bundle = build_draft_tp_worker(
            server_args=server_args,
            gpu_id=gpu_id,
            nccl_port=nccl_port,
            target_model_config=self.model_runner.model_config,
            algo_label="DFLY",
            attention_backend_override=attention_backends()[0],
        )
        self._draft_worker = bundle.draft_worker
        self.draft_model_runner = bundle.draft_model_runner
        self.draft_model = bundle.draft_model
        if not isinstance(self.draft_model, HYV4ForCausalLMDFly):
            raise ValueError(
                f"DFLY needs the HYV4ForCausalLMDFly draft, got {type(self.draft_model)}"
            )
        self.spec = self.draft_model.spec
        self.block_size = self.spec.query_width
        if get_spec().speculative_num_draft_tokens != self.block_size:
            raise ValueError(
                f"DFLY block width {self.block_size} disagrees with "
                f"speculative_num_draft_tokens={get_spec().speculative_num_draft_tokens}"
            )
        self._share_embed_and_head()
        self._tp_sync = SpecTpSync(get_dp_tp_group())
        self._producer_layers = dfly_producer_layers(self.spec.target_index_layers)
        self._draft_topk_planes = dfly_draft_topk_planes(self.spec.target_index_layers)
        self._capture: Optional[DFlyProducerCapture] = None
        self._state: Optional[VLDFlyRequestState] = None
        self._draft_block_spec_info = make_draft_block_spec_info(
            draft_token_num=self.block_size, device=self.device
        )
        self._max_context_len = self.model_runner.model_config.context_len
        self._physical_ids: Optional[bool] = None
        self._execution: Optional[VLDFlyExecution] = None
        # The separate draft-runner forward is the qualification oracle for
        # drafting inside the target verify; both read VLDFlyExecution.plan.
        self._draft_in_target_forward = True

    def __getattr__(self, name):
        # Delegate the rest of the TpModelWorker surface to the target worker;
        # the guard keeps a lookup before __init__ from recursing.
        if name == "_target_worker":
            raise AttributeError(name)
        return getattr(self.target_worker, name)

    @property
    def draft_worker(self):
        return self._draft_worker

    def weight_update_runners(self):
        return [("draft", self.draft_model_runner)]

    @property
    def spec_v2_attn_backends(self) -> tuple:
        return (self.model_runner.attn_backend, self.draft_model_runner.attn_backend)

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        # The draft keeps its KV in target layers; its own pool is a placeholder.
        self._draft_worker.alloc_memory_pool(
            memory_pool_config=MemoryPoolConfig(
                max_total_num_tokens=64,
                max_running_requests=memory_pool_config.max_running_requests,
            ),
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        num_rows = req_to_token_pool.req_to_token.shape[0]
        topk = self.spec.target_index_topk
        self._state = VLDFlyRequestState(
            num_rows=num_rows,
            hidden_size=self._target_model.config.hidden_size,
            hidden_dtype=self.model_runner.model_config.dtype,
            num_producers=len(self._producer_layers),
            topk=topk,
            device=self.device,
        )
        # Row 0 is the pool's padding row; every other row may be in one verify.
        self._capture = DFlyProducerCapture(
            producer_layers=self._producer_layers,
            max_rows=(num_rows - 1) * self.block_size,
            topk=topk,
            device=self.device,
        )
        _attach_topk_capture(
            model=self._target_model,
            capture=self._capture,
            layer_ids=self._producer_layers,
        )

    def clear_cache_pool(self):
        self._state.invalidate_all()

    def init_attention_backends(self):
        self._draft_worker.init_attention_backends()
        self._physical_ids = self._check_attention_backends()
        self._check_target_kv_layers()
        # One-shot MHA prefill bypasses the MLA core where the top-k capture records.
        self.model_runner.attn_backend.supports_mha_one_shot = False
        self._init_execution()
        self._install_capture_hooks()

    def _init_execution(self) -> None:
        # Bound before the scheduler initializes the target's graphs and runners.
        self._execution = VLDFlyExecution(
            draft_model=self.draft_model,
            state=self._state,
            draft_topk_planes=self._draft_topk_planes,
            physical_ids=self._physical_ids,
            max_context_len=self._max_context_len,
            req_to_token=self.model_runner.req_to_token_pool.req_to_token,
            runner_ids=self.model_runner.runner_ids,
            block_spec_info=self._draft_block_spec_info,
            # A verify holds at most every row but the padding row 0.
            max_requests=self.model_runner.req_to_token_pool.req_to_token.shape[0] - 1,
            device=self.device,
        )
        if self._draft_in_target_forward:
            self.model_runner.model.dfly_execution = self._execution

    def _install_capture_hooks(self) -> None:
        self.model_runner.capture_head_hooks.append(self._stage_graph_capture)
        self.model_runner.capture_tail_hooks.append(self._end_graph_capture)

    def _stage_graph_capture(self, graph_runner, forward_batch: ForwardBatch) -> None:
        # Runs inside the captured region, so it changes host state only.
        self._execution.stage_capture(batch_size=forward_batch.batch_size)
        self._capture.arm(num_rows=forward_batch.input_ids.shape[0])

    def _end_graph_capture(
        self, graph_runner, output, forward_batch: ForwardBatch, num_tokens: int
    ) -> None:
        self._execution.end_capture()
        self._capture.disarm()

    def init_cuda_graphs(self):
        self._draft_worker.init_cuda_graphs(capture_decode_cuda_graph=False)

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish=None,
        grammar_barrier=None,
        pp_proxy_tensors=None,
    ) -> GenerationBatchResult:
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            result = self._prefill(batch)
        else:
            result = self._decode(batch)
        if on_publish is not None:
            on_publish(result.new_seq_lens)
        return result

    def _prefill(self, batch: ScheduleBatch) -> GenerationBatchResult:
        extend_lens = torch.tensor(batch.extend_lens, device=self.device)
        self._capture.begin(
            num_rows=len(batch.extend_lens), keep_rows=extend_lens.cumsum(0) - 1
        )
        batch_output = self.target_worker.forward_batch_generation(
            batch, capture_hidden_mode=CaptureHiddenMode.LAST
        )
        logits_output = batch_output.logits_output
        # Every prefill rewrites its rows, so a resumed or host-restored request
        # never drafts from top-k ids of an earlier KV mapping.
        self._write_state(
            batch,
            hidden=logits_output.hidden_states,
            topk=self._capture.finish(),
            committed_len=batch.seq_lens,
            mrope_delta=(
                self._mrope_deltas(batch) if self.draft_model.is_mrope_enabled else None
            ),
        )
        logits_output.hidden_states = None
        batch_output.new_seq_lens = batch.seq_lens
        batch_output.next_draft_input = _make_draft_input(
            bonus_tokens=batch_output.next_token_ids, new_seq_lens=batch.seq_lens
        )
        return batch_output

    def _decode(self, batch: ScheduleBatch) -> GenerationBatchResult:
        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashDraftInputV2):
            raise RuntimeError("DFLY decode expects the draft state of its prefill.")
        bs = batch.batch_size()
        block_size = self.block_size
        prefix_lens = batch.seq_lens
        verify_slots = assign_extend_cache_locs_func(
            req_pool_indices=batch.req_pool_indices,
            req_to_token=self.model_runner.req_to_token_pool.req_to_token,
            start_offset=prefix_lens,
            end_offset=prefix_lens + block_size,
            batch_size=bs,
            draft_token_num=block_size,
            device=self.device,
        ).view(bs, block_size)
        live = self._live_rows(batch)
        bonus_tokens = draft_input.bonus_tokens
        self._wait_for_target_hicache_load(batch)
        if self._draft_in_target_forward:
            # Placeholders: the verify forward drafts into them before embedding.
            drafts = torch.full(
                (bs, block_size - 1), -1, dtype=torch.int64, device=self.device
            )
        else:
            drafts = self._draft_with_runner(
                batch,
                self._execution.plan(
                    req_pool_indices=batch.req_pool_indices,
                    anchors=prefix_lens,
                    bonus_tokens=bonus_tokens,
                    live=live,
                    verify_slots=verify_slots,
                ),
                bonus_tokens=bonus_tokens,
            )
        candidates, verify_ids = split_dfly_drafts_for_verify(
            bonus_tokens=bonus_tokens, draft_tokens=drafts
        )
        verify_forward_batch = self._prepare_verify(
            batch, verify_ids=verify_ids, verify_slots=verify_slots
        )
        self._execution.stage(live=live)
        self._capture.begin(num_rows=bs * block_size)
        try:
            # Replays the composite decode graph when the batch fits a bucket,
            # else runs the same forward eagerly.
            target_out = self.target_worker.forward_batch_generation(
                batch=None, forward_batch=verify_forward_batch, is_verify=True
            )
        finally:
            self._execution.unstage()
        if self._draft_in_target_forward:
            # The execution's candidate buffer is rewritten by the next verify.
            candidates = self._execution.candidates(bs).clone()
        return self._commit(batch, candidates=candidates, target_out=target_out)

    def _prepare_verify(
        self,
        batch: ScheduleBatch,
        *,
        verify_ids: torch.Tensor,
        verify_slots: torch.Tensor,
    ) -> ForwardBatch:
        block_size = self.block_size
        verify_input = DFlashVerifyInput(
            draft_token=verify_ids,
            positions=dfly_verify_positions(
                anchors=batch.seq_lens,
                query_width=block_size,
                max_context_len=self._max_context_len,
            ).flatten(),
            draft_token_num=block_size,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        batch.out_cache_loc = verify_slots.flatten()
        seq_lens_cpu, seq_lens_sum = batch.seq_lens_cpu, batch.seq_lens_sum
        if seq_lens_cpu is not None:
            batch.seq_lens_cpu = seq_lens_cpu + block_size
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
        elif batch.spec_info.nxt_kv_lens_cpu is not None:
            batch.seq_lens_cpu = batch.spec_info.nxt_kv_lens_cpu
            batch.seq_lens_sum = int(batch.spec_info.nxt_kv_lens_sum)
        # The model runner plans attention once, at the shape it runs.
        verify_forward_batch, _ = verify_input.prepare_for_verify(
            batch, self.target_worker, defer_metadata_init=True
        )
        batch.seq_lens_cpu, batch.seq_lens_sum = seq_lens_cpu, seq_lens_sum
        return verify_forward_batch

    def _wait_for_target_hicache_load(self, batch: ScheduleBatch) -> None:
        # The draft reads late target layers before the target's own per-layer
        # load waits run.
        if batch.hicache_consumer_index >= 0:
            counter = self.target_worker.hicache_layer_transfer_counter
            counter.events[batch.hicache_consumer_index].wait(counter.num_layers - 1)

    def _commit(
        self, batch: ScheduleBatch, *, candidates: torch.Tensor, target_out
    ) -> GenerationBatchResult:
        bs = batch.batch_size()
        block_size = self.block_size
        prefix_lens = batch.seq_lens
        logits_output = target_out.logits_output
        target_predict = torch.argmax(logits_output.next_token_logits, dim=-1)
        target_predict = self._tp_sync.sync(
            SpecTpSyncSite.DFLY_ACCEPT_GREEDY, target_predict.view(bs, block_size)
        )
        accept_len, bonus = compute_dflash_correct_drafts_and_bonus(
            candidates=candidates, target_predict=target_predict
        )
        out_tokens, commit_lens = commit_accept(candidates, accept_len, bonus)
        new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
        source_rows = dfly_source_rows(
            row_starts=torch.arange(bs, device=self.device) * block_size,
            commit_lens=commit_lens,
        )
        # Gathers copy out of the capture planes and the graph's hidden output
        # before the next forward rewrites them.
        target_topk = self._capture.finish()
        self._write_state(
            batch,
            hidden=logits_output.hidden_states[source_rows],
            topk=target_topk[:, source_rows],
            committed_len=new_seq_lens,
        )
        logits_output.hidden_states = None
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=out_tokens.reshape(-1),
            accept_lens=commit_lens,
            can_run_cuda_graph=target_out.can_run_cuda_graph,
            next_draft_input=_make_draft_input(
                bonus_tokens=bonus, new_seq_lens=new_seq_lens
            ),
            speculative_num_draft_tokens=block_size,
            new_seq_lens=new_seq_lens,
        )

    def _live_rows(self, batch: ScheduleBatch) -> torch.Tensor:
        """``[bs]`` whether each request's state was written for its current
        pool row, cache epoch and committed length; others draft nothing."""
        rows_cpu = batch.req_pool_indices_cpu
        return self._state.live(
            rows_cpu=rows_cpu,
            rows=batch.req_pool_indices,
            generations=self.model_runner.req_to_token_pool.req_generation[rows_cpu],
            committed_len=batch.seq_lens,
        )

    def _write_state(self, batch: ScheduleBatch, **values) -> None:
        rows_cpu = batch.req_pool_indices_cpu
        self._state.write(
            rows_cpu=rows_cpu,
            rows=batch.req_pool_indices,
            generations=self.model_runner.req_to_token_pool.req_generation[rows_cpu],
            **values,
        )

    def _draft_with_runner(
        self,
        batch: ScheduleBatch,
        inputs: VLDFlyDraftInputs,
        *,
        bonus_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """``[bs, N]`` greedy drafts from a draft-runner forward; -1 for
        inactive requests."""
        bs = batch.batch_size()
        num_rows = bs * self.block_size
        block = inputs.block
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=block.input_ids,
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            out_cache_loc=block.write_slots,
            seq_lens_sum=int(batch.seq_lens_cpu.sum()) + num_rows,
            seq_lens_cpu=batch.seq_lens_cpu + self.block_size,
            positions=block.positions.flatten(),
            spec_algorithm=SpeculativeAlgorithm.DFLY,
            spec_info=self._draft_block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            global_num_token_non_padded_cpu=num_rows,
            can_run_decode_cuda_graph=False,
        )
        forward_batch.mrope_positions = inputs.mrope_positions
        self.model_runner.runner_ids.bind_own_plan(forward_batch)
        self.draft_model.stage_block(
            target_hidden=inputs.target_hidden, topk_indices=inputs.topk_indices
        )
        with _target_kv_pool(
            self.draft_model_runner.attn_backend, self.model_runner.token_to_kv_pool
        ):
            draft_out = self.draft_model_runner.forward(forward_batch)
        return self.draft_model.propose(
            draft_out.logits_output.hidden_states, bonus_tokens, block.active
        )

    def _mrope_deltas(self, batch: ScheduleBatch) -> torch.Tensor:
        deltas = [
            (
                torch.zeros(1, dtype=torch.int64)
                if mm is None
                else mm.mrope_position_delta.squeeze(0)
            )
            for mm in batch.multimodal_inputs
        ]
        return async_h2d_stack(deltas, device=self.device).view(-1)

    def _share_embed_and_head(self) -> None:
        embed, head = self._target_model.get_embed_and_head()
        draft_embed, draft_head = self.draft_model.get_embed_and_head()
        for name, shared, own in (
            ("embedding", embed, draft_embed),
            ("LM head", head, draft_head),
        ):
            if shared.dtype != torch.bfloat16 or shared.shape != own.shape:
                raise ValueError(
                    f"DFLY shares the target's BF16 {name}; got {shared.dtype} "
                    f"{tuple(shared.shape)} for a draft {name} of {tuple(own.shape)}"
                )
        self.draft_model.set_embed_and_head(embed, head)

    def _check_attention_backends(self) -> bool:
        target = self.model_runner.attn_backend
        draft = self.draft_model_runner.attn_backend
        for role, backend in (("target", target), ("draft", draft)):
            if not isinstance(backend, DeepseekSparseAttnBackend):
                raise ValueError(
                    f"DFLY needs the DSA attention backend for the {role}, "
                    f"got {type(backend).__name__}"
                )
        return check_dfly_attention_backends(
            draft_decode_impl=draft.dsa_decode_impl,
            draft_fused_topk=draft.use_fused_topk,
            target_fused_topk=target.use_fused_topk,
            transform_methods={
                "target prefill": target.get_topk_transform_method(ForwardMode.EXTEND),
                "target verify": target.get_topk_transform_method(
                    ForwardMode.TARGET_VERIFY
                ),
                "draft block": draft.get_topk_transform_method(
                    ForwardMode.TARGET_VERIFY
                ),
            },
            index_kpool=draft.dsa_index_kpool,
            has_hisparse=draft.hisparse_coordinator is not None,
            translates_ids=self.model_runner.runner_ids.is_translating,
        )

    def _check_target_kv_layers(self) -> None:
        target_layers = self._target_model.model.layers
        pool = self.model_runner.token_to_kv_pool
        for layer, target_id in zip(
            self.draft_model.model.layers, self.spec.target_kv_layers, strict=True
        ):
            draft_attn = layer.self_attn.attn_mqa
            target_attn = target_layers[target_id].self_attn.attn_mqa
            geometry = [
                (
                    attn.layer_id,
                    attn.tp_q_head_num,
                    attn.tp_k_head_num,
                    attn.head_dim,
                    attn.v_head_dim,
                    attn.scaling,
                )
                for attn in (draft_attn, target_attn)
            ]
            if geometry[0] != geometry[1]:
                raise ValueError(
                    f"DFLY draft attention must alias target layer {target_id} "
                    f"with the same MLA geometry: draft {geometry[0]}, "
                    f"target {geometry[1]}"
                )
            key_buffer = pool.get_key_buffer(target_id)
            if key_buffer.dtype != torch.bfloat16:
                raise ValueError(
                    f"DFLY writes BF16 MLA KV into target layer {target_id}, "
                    f"whose cache is {key_buffer.dtype}"
                )
