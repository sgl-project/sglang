import logging
import math
from dataclasses import replace
from typing import List, Optional, Tuple

import torch

from sglang.kernels.ops.speculative.cache_locs import (
    assign_extend_cache_locs_func,
    rebuild_compact_draft_req_to_token_func,
)
from sglang.kernels.ops.speculative.dflash import (
    _compute_dflash_accept_bonus_triton_unchecked,
    _prepare_dflash_draft_block_unchecked,
)
from sglang.kernels.ops.speculative.dspark.dspark_accept import (
    accept_sampling,
)
from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method
from sglang.srt.layers.logprob_processor import compute_spec_logprobs
from sglang.srt.lora.layers import unwrap_lora_layer
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    compute_position,
)
from sglang.srt.runtime_context import (
    get_exec,
    get_parallel,
    get_schedule,
    get_spec,
    mamba_track_grid,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_utils import (
    apply_dflash_simulated_acceptance,
    apply_dflash_verify_logits_adjustments,
    can_dflash_use_fused_qkv_proj,
    compute_dflash_correct_drafts_and_bonus,
    compute_dflash_sampling_correct_drafts_and_bonus,
    is_dense_head_weight,
    is_dflash_sampling_verify_available,
    parse_dflash_draft_config,
)
from sglang.srt.speculative.draft_worker_common import (
    build_block_pos_offsets,
    build_draft_tp_worker,
    make_draft_block_spec_info,
    make_draft_input_v2,
    make_draft_sampler_capture_hook,
)
from sglang.srt.speculative.dspark_components.dspark_draft import resolve_greedy_mask
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    SIMULATE_ACC_LEN,
    SIMULATE_ACC_METHOD,
    SIMULATE_ACC_TOKEN_MODE,
    GrammarTree,
    assign_req_to_token_pool_func,
    build_grammar_vocab_mask,
)
from sglang.srt.utils import get_available_gpu_memory, is_cuda, is_hip, is_npu

_is_npu = is_npu()


logger = logging.getLogger(__name__)

# Cross-rank scheduler-agreement assertions for --enable-spec-pp. Forces a sync,
# so it is opt-in.
_PP_DEBUG_CHECK = envs.SGLANG_SPEC_PP_DEBUG_CHECK.get()

_FusedKVMaterializeHelper = None


def _get_fused_kv_materialize_helper():
    global _FusedKVMaterializeHelper
    if _FusedKVMaterializeHelper is None:
        from sglang.kernels.ops.speculative.fused_kv_materialize import (
            FusedKVMaterializeHelper,
        )

        _FusedKVMaterializeHelper = FusedKVMaterializeHelper
    return _FusedKVMaterializeHelper


class _DflashDraftSampler:
    """Capture-safe greedy argmax over the target LM head, run inside the draft
    cuda graph so the draft sampling is captured and counted in fwd_occupancy.
    DFLASH's draft has no head of its own; it borrows the target `lm_head`.

    tp=1: plain argmax over the local (full) vocab shard.
    tp>1: per-rank shard (max, global id) -> all-gather -> first-max select.
    Tie resolution is bit-exact vs a full-vocab argmax: ranks own contiguous
    ascending vocab shards and torch.argmax returns the FIRST max index.
    No added-vocab support (the builder bails to eager in that case).
    """

    def __init__(
        self, *, weight, block_size, num_org, org_vocab_start, max_bs, tp_group=None
    ):
        self.weight = weight
        self.block_size = int(block_size)
        self.num_org = int(num_org)
        self.org_vocab_start = int(org_vocab_start)
        self.tp_group = tp_group
        self.tp_size = int(tp_group.world_size) if tp_group is not None else 1
        max_tokens = int(max_bs) * (self.block_size - 1)
        device = weight.device
        self.out = torch.empty((max_tokens,), dtype=torch.int64, device=device)
        if self.tp_size > 1:
            # Static buffers (fixed addresses) keep the in-graph select replay-safe.
            self.local_max = torch.empty(
                (max_tokens,), dtype=weight.dtype, device=device
            )
            self.local_arg = torch.empty(
                (max_tokens,), dtype=torch.int64, device=device
            )
            self.gathered_max = torch.empty(
                (self.tp_size * max_tokens,), dtype=weight.dtype, device=device
            )
            self.gathered_ids = torch.empty(
                (self.tp_size * max_tokens,), dtype=torch.int64, device=device
            )
            self.best_rank = torch.empty(
                (1, max_tokens), dtype=torch.int64, device=device
            )
            self.selected_ids = torch.empty(
                (1, max_tokens), dtype=torch.int64, device=device
            )

    def __call__(self, hidden_states, input_ids=None):
        # draft tokens are block positions 1: (pos 0 is the seeded bonus token)
        bs = hidden_states.shape[0] // self.block_size
        hs = hidden_states.view(bs, self.block_size, -1)[:, 1:, :].reshape(
            -1, hidden_states.shape[-1]
        )
        if hs.dtype != self.weight.dtype:
            hs = hs.to(self.weight.dtype)
        n = hs.shape[0]
        logits = torch.matmul(hs, self.weight[: self.num_org].T)
        if self.tp_size == 1:
            tokens = torch.argmax(logits, dim=-1).to(torch.long)
            if self.org_vocab_start:
                tokens += self.org_vocab_start
            self.out[:n].copy_(tokens)
            return
        local_max = self.local_max[:n]
        local_arg = self.local_arg[:n]
        torch.max(logits, dim=-1, out=(local_max, local_arg))
        if self.org_vocab_start:
            local_arg.add_(self.org_vocab_start)
        gathered_max = self.gathered_max[: self.tp_size * n]
        gathered_ids = self.gathered_ids[: self.tp_size * n]
        self.tp_group.all_gather_into_tensor(gathered_max, local_max)
        self.tp_group.all_gather_into_tensor(gathered_ids, local_arg)
        best_rank = self.best_rank[:, :n]
        torch.argmax(gathered_max.view(self.tp_size, n), dim=0, out=best_rank[0])
        selected = self.selected_ids[:, :n]
        torch.gather(gathered_ids.view(self.tp_size, n), 0, best_rank, out=selected)
        self.out[:n].copy_(selected.view(-1))


def _commit_accept(candidates, accept_len, bonus_tokens):
    """The committed block: drafted tokens shifted left, the bonus at the accept
    boundary. Returns it with the commit lengths."""
    out_tokens = torch.empty_like(candidates, dtype=torch.int64)
    out_tokens[:, :-1].copy_(candidates[:, 1:])
    out_tokens[:, -1].fill_(0)
    out_tokens.scatter_(1, accept_len.to(torch.int64)[:, None], bonus_tokens[:, None])
    return out_tokens, accept_len.to(torch.int32) + 1


def _is_all_greedy(sampling_info) -> bool:
    return sampling_info is None or sampling_info.is_all_greedy


def _selector_lattice(draft_model, pred_hidden, anchor_token_ids):
    # Flattened to [N, H] and viewed back because the radix top-k kernel is 2D.
    bs, num_pred = pred_hidden.shape[0], pred_hidden.shape[1]
    candidate_ids, unary_logits = draft_model.compute_candidates(
        pred_hidden.reshape(-1, pred_hidden.shape[-1])
    )
    candidate_ids = candidate_ids.view(bs, num_pred, -1)
    return candidate_ids, draft_model.candidate_selector.build_lattice(
        candidate_ids=candidate_ids,
        unary_logits=unary_logits.view(bs, num_pred, -1),
        hidden_states=pred_hidden,
        anchor_token_ids=anchor_token_ids,
    )


class _SelectorDraftSampler:
    """Selector decode folded into the draft cuda graph, greedy and T>0 alike.

    One captured graph serves both: it always walks the sampling path, and a static
    greedy_mask selects the argmax per row.
    """

    def __init__(self, *, draft_model, block_size, max_bs, device):
        self.draft_model = draft_model
        self.selector = draft_model.candidate_selector
        self.block_size = int(block_size)
        max_bs, gamma, top_k = int(max_bs), self.block_size - 1, self.selector.top_k
        self.out = torch.empty((max_bs * gamma,), dtype=torch.int64, device=device)
        # Written by the host before replay, or read after it; the addresses are
        # baked into the captured graph.
        self.temperatures = torch.ones((max_bs,), dtype=torch.float32, device=device)
        self.greedy_mask = torch.ones((max_bs,), dtype=torch.bool, device=device)
        self.uniforms = torch.empty((max_bs, gamma), dtype=torch.float32, device=device)
        self.candidate_out = torch.empty(
            (max_bs, gamma, top_k), dtype=torch.int64, device=device
        )
        self.q_out = torch.empty(
            (max_bs, gamma, top_k), dtype=torch.float32, device=device
        )

    def stage_sampling_params(self, *, bs: int, sampling_info) -> None:
        """Host-side refresh of the static sampling params; must run before the draft
        graph replay that consumes them."""
        if sampling_info is None:
            self.temperatures[:bs].fill_(1.0)
            self.greedy_mask[:bs].fill_(True)
            return
        torch.clamp(
            sampling_info.temperatures.view(-1)[:bs].to(torch.float32),
            min=1e-5,
            out=self.temperatures[:bs],
        )
        self.greedy_mask[:bs].copy_(
            resolve_greedy_mask(
                bs=bs, sampling_info=sampling_info, device=self.greedy_mask.device
            )
        )

    def __call__(self, hidden_states, input_ids):
        bs = hidden_states.shape[0] // self.block_size
        block_ids = input_ids.view(bs, self.block_size)
        hs = hidden_states.view(bs, self.block_size, -1)[:, 1:, :]  # pos 0 = anchor
        candidate_ids, scores = _selector_lattice(self.draft_model, hs, block_ids[:, 0])
        # In-graph philox draw: each replay advances the generator and redraws.
        tokens, q_rows = self.selector.sample_path(
            candidate_ids=candidate_ids,
            scores=scores,
            uniforms=self.uniforms[:bs].uniform_(),
            temperatures=self.temperatures[:bs],
            greedy_mask=self.greedy_mask[:bs],
        )
        self.out[: tokens.numel()].copy_(tokens.reshape(-1))
        self.candidate_out[:bs].copy_(candidate_ids)
        self.q_out[:bs].copy_(q_rows)


class DFlashWorkerV2(BaseSpecWorker):
    """DFLASH speculative decoding worker (spec-v2).

    Drives both overlap and non-overlap scheduling, same as EAGLE: the
    scheduler runs it synchronously when overlap is disabled.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        ps: ParallelState,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        super().__init__()

        self.server_args = server_args
        self.gpu_id = gpu_id
        self.ps = ps
        self.nccl_port = nccl_port
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self._need_mamba_verify_commit = False
        self.page_size = get_schedule().page_size
        # Normalized in arg_groups.speculative_hook.handle_speculative_decoding.
        self.draft_window_size: Optional[int] = get_spec().speculative_draft_window_size
        self.use_compact_draft_cache = self.draft_window_size is not None
        self.device = target_worker.device

        self._warned_sampling_fallback = False
        self._draft_probs_buf = None
        self._logged_first_verify = False

        bundle = build_draft_tp_worker(
            server_args=server_args,
            gpu_id=gpu_id,
            ps=replace(ps, pp_rank=0),
            nccl_port=nccl_port,
            target_model_config=target_worker.model_runner.model_config,
            algo_label="DFLASH",
        )
        self._draft_worker = bundle.draft_worker
        self.draft_model_runner = bundle.draft_model_runner
        self._draft_sampler = None
        self.draft_model = bundle.draft_model
        self.selector = self.draft_model.candidate_selector
        draft_config = parse_dflash_draft_config(
            draft_hf_config=self.draft_model_runner.model_config.hf_config
        )
        if get_spec().speculative_num_draft_tokens is None:
            # Should not happen (ServerArgs should have inferred it), but keep a fallback.
            self.block_size = int(draft_config.resolve_block_size(default=16))
        else:
            self.block_size = int(get_spec().speculative_num_draft_tokens)
            model_block_size = draft_config.block_size
            if model_block_size is None:
                model_block_size = getattr(self.draft_model, "block_size", None)
            if model_block_size is not None and int(model_block_size) != int(
                self.block_size
            ):
                logger.warning(
                    "DFLASH block size mismatch: using speculative_num_draft_tokens=%s but draft config block_size=%s.",
                    self.block_size,
                    model_block_size,
                )
        self.draft_model.set_block_size(self.block_size)
        self.speculative_num_draft_tokens = int(self.block_size)

        self._mask_token = draft_config.mask_token
        self._mask_token_id_override = draft_config.mask_token_id
        self._mask_token_id = self._resolve_mask_token_id(
            mask_token=self._mask_token,
            mask_token_id=self._mask_token_id_override,
        )
        target_model = self._target_worker.model_runner.model
        self._noise_embed_scale = (
            float(target_model.get_dflash_noise_embedding_scale())
            if hasattr(target_model, "get_dflash_noise_embedding_scale")
            else 1.0
        )
        if self.ps.tp_rank == 0:
            logger.info(
                "Initialized DFLASH draft runner. attention_backend=%s, model=%s, block_size=%s, draft_window_size=%s, compact_cache=%s",
                bundle.resolved_attention_backend,
                self.draft_model.__class__.__name__,
                self.block_size,
                self.draft_window_size,
                self.use_compact_draft_cache,
            )
            logger.info(
                "DFLASH draft runner ready. mask_token=%s, mask_token_id=%s, mask_token_id_override=%s, noise_embed_scale=%s",
                self._mask_token,
                self._mask_token_id,
                self._mask_token_id_override,
                self._noise_embed_scale,
            )

        self._block_pos_offsets = build_block_pos_offsets(
            length=self.block_size, device=self.device
        )
        self._draft_block_ids_buf: Optional[torch.Tensor] = None  # [cap_bs, block_size]
        self._draft_block_positions_buf: Optional[torch.Tensor] = (
            None  # [cap_bs, block_size]
        )
        self._draft_block_tokens_buf: Optional[torch.Tensor] = (
            None  # [cap_bs, block_size]
        )
        self._draft_verify_out_cache_loc_buf: Optional[torch.Tensor] = (
            None  # [cap_bs, block_size]
        )
        self._draft_block_end_buf: Optional[torch.Tensor] = None  # [cap_bs]
        self._selector_sample: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._draft_seq_lens_cpu_buf: Optional[torch.Tensor] = None  # [cap_bs] on CPU
        self._draft_block_spec_info = make_draft_block_spec_info(
            draft_token_num=int(self.block_size), device=self.device
        )
        self._draft_greedy_gathered_max_buf: Optional[torch.Tensor] = None
        self._draft_greedy_gathered_ids_buf: Optional[torch.Tensor] = None
        self._draft_greedy_gather_cap: int = 0
        self._draft_greedy_local_max_buf: Optional[torch.Tensor] = None
        self._draft_greedy_local_arg_buf: Optional[torch.Tensor] = None
        self._draft_greedy_local_cap: int = 0
        self._draft_greedy_best_rank_buf: Optional[torch.Tensor] = None
        self._draft_greedy_rank_index_buf: Optional[torch.Tensor] = None
        self._draft_greedy_selected_ids_buf: Optional[torch.Tensor] = None
        self._draft_greedy_index_cap: int = 0
        self._use_fused_kv_materialize = is_cuda() or is_hip()
        self._fused_kv_helper: Optional[object] = None
        if self._use_fused_kv_materialize:
            self._init_fused_kv_helper()

        supports_gpu_triton = is_cuda() or is_hip()
        self._use_triton_prepare_block = supports_gpu_triton
        self._use_triton_accept_bonus = supports_gpu_triton
        # The legacy compact-rebuild path host-syncs twice per step (masked
        # gather's implicit nonzero D2H + lengths.max().item()); keep it only
        # for platforms without GPU triton.
        self._use_triton_compact_rebuild = supports_gpu_triton
        self._accept_bonus_buffer_cap: int = 0
        self._accept_bonus_buffer_slot: int = 0
        self._accept_len_buf: Optional[torch.Tensor] = None
        self._commit_lens_bufs: List[torch.Tensor] = []
        self._bonus_id_bufs: List[torch.Tensor] = []
        self._out_tokens_bufs: List[torch.Tensor] = []
        self._new_seq_lens_bufs: List[torch.Tensor] = []

        # --- Pipeline parallelism (see --enable-spec-pp).
        #
        # The draft runs on PP rank 0. That is forced by ordering, not by
        # preference: the draft block's KV slots come from *this* step's
        # allocation, and only the rank that is about to launch the step has
        # them. Rank 0 also owns `embed_tokens`, which the noise embedding
        # needs.
        #
        # What rank 0 lacks is `lm_head` (last rank) and the auxiliary target
        # hidden states (complete only on the last rank). So:
        #   - `lm_head` is replicated onto rank 0 once, at init;
        #   - the last rank projects the aux hidden through the draft's
        #     `fc`/`hidden_norm` (5x narrower than the packed capture) and ships
        #     `spec_ctx_hidden` back over the existing last->0 output edge;
        #     rank 0 writes it into the draft KV cache when it arrives.
        self._pp_enabled = int(getattr(ps, "pp_size", 1)) > 1
        self._pp_lm_head_replica = None
        self._pp_group = None
        self._pp_is_first = True
        self._pp_is_last = True
        if self._pp_enabled:
            from sglang.srt.distributed import get_pp_group

            self._pp_group = get_pp_group()
            self._pp_is_first = self._pp_group.is_first_rank
            self._pp_is_last = self._pp_group.is_last_rank
            self._pp_lm_head_replica = self._build_pp_lm_head_replica()

    def _build_pp_lm_head_replica(self):
        """Replicate the target `lm_head` onto PP rank 0 for the draft's sampling.

        Sent last->first over the PP group instead of re-read from the
        checkpoint: the PP group pairs ranks that share a tp_rank, so rank 0
        receives exactly the vocab shard its own TP rank would have loaded.
        Middle ranks do not participate and pay nothing.

        This runs before `alloc_memory_pool`, so the extra weight is already
        accounted for when the KV pool is profiled (and PP min-reduces the
        resulting token capacity across ranks).
        """
        pp_group = self._pp_group
        target_model = self.model_runner.model
        device_group = pp_group.device_group
        first_global, last_global = pp_group.ranks[0], pp_group.ranks[-1]

        if pp_group.is_last_rank:
            head = unwrap_lora_layer(getattr(target_model, "lm_head", None))
            weight = getattr(head, "weight", None)
            if weight is None:
                raise RuntimeError(
                    "--enable-spec-pp needs a dense target lm_head weight to "
                    "replicate onto PP rank 0."
                )
            torch.distributed.send(
                weight.data.contiguous(), dst=first_global, group=device_group
            )
            return None

        if not pp_group.is_first_rank:
            return None

        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

        text_config = self.model_runner.model_config.hf_text_config
        replica = ParallelLMHead(
            int(text_config.vocab_size),
            int(text_config.hidden_size),
            quant_config=None,
            use_attn_tp_group=get_parallel().enable_dp_lm_head,
            prefix="pp_spec_lm_head",
        ).to(device=self.device, dtype=self.model_runner.dtype)
        torch.distributed.recv(replica.weight.data, src=last_global, group=device_group)
        return replica

    def _draft_lm_head(self):
        """The head the draft samples through.

        Under PP the draft runs on rank 0, which owns no `lm_head`; use the
        replica built at init instead of the model's `PPMissingLayer`.
        """
        if self._pp_lm_head_replica is not None:
            return self._pp_lm_head_replica
        target_model = self._target_worker.model_runner.model
        return unwrap_lora_layer(getattr(target_model, "lm_head", None))

    @property
    def draft_worker(self):
        # DFLASH drives the draft model through a plain TpModelWorker: the
        # draft KV is materialized from target hidden states, so there is no
        # EagleDraftWorkerBase draft/draft_extend split to wrap it in.
        return self._draft_worker

    @property
    def spec_v2_attn_backends(self) -> tuple:
        # Every attn backend a spec_v2 forward touches; consumed by
        # decide_needs_cpu_seq_lens to gate the seq_lens_cpu D2H.
        return (
            self._target_worker.model_runner.attn_backend,
            self.draft_model_runner.attn_backend,
        )

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        # Without draft windowing, the draft worker aliases the target
        # request->token mapping and allocation state. With draft windowing
        # enabled, the draft worker keeps a private compact req->token table
        # over the same global KV index space, so radix-cache/prefix-hit KV
        # remains reusable while draft attention sees only the recent window.
        self._draft_worker.alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=(
                None if self.use_compact_draft_cache else req_to_token_pool
            ),
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )

    def init_attention_backends(self):
        self._draft_worker.init_attention_backends()
        self._need_mamba_verify_commit = mambaish_config(
            self.model_runner.model_config
        ) is not None and hasattr(
            self.model_runner.attn_backend,
            "update_mamba_state_after_mtp_verify",
        )

    def init_cuda_graphs(self):
        capture_decode_cuda_graph = (
            get_exec().graph.cuda_graph_config.decode.backend != Backend.DISABLED
        )
        if self._pp_enabled and not self._pp_is_first:
            # Only PP rank 0 ever runs the draft model.
            capture_decode_cuda_graph = False
        if is_cuda() and capture_decode_cuda_graph:
            available_mem = get_available_gpu_memory(self.device, self.gpu_id)
            if available_mem < 1.0:
                capture_decode_cuda_graph = False
                logger.warning(
                    "Disable DFLASH draft cuda graph because only %.2f GB GPU "
                    "memory is available after target backend initialization.",
                    available_mem,
                )
        if capture_decode_cuda_graph:
            # Must run before capture so the draft graph folds the head in.
            self._draft_sampler = self._maybe_build_draft_sampler()
            if self._draft_sampler is not None:
                self.draft_model_runner.capture_tail_hooks.append(
                    make_draft_sampler_capture_hook(self._draft_sampler)
                )
        self._draft_worker.init_cuda_graphs(
            capture_decode_cuda_graph=capture_decode_cuda_graph
        )

    def _maybe_build_draft_sampler(self):
        def _eager(reason):
            if self.ps.tp_rank == 0:
                logger.info("DFLASH draft greedy head kept eager (reason=%s).", reason)
            return None

        if envs.SGLANG_DFLASH_EAGER_DRAFT_SAMPLER.get():
            return _eager("SGLANG_DFLASH_EAGER_DRAFT_SAMPLER=1")
        if self.block_size <= 1:
            return _eager("block_size<=1")
        lm_head = self._draft_lm_head()
        if lm_head is None:
            return _eager("no target lm_head")

        if self.selector is not None:
            # compute_candidates needs the target lm_head attached before capture.
            # A gate-admitted quantized head is capture-safe: the target's own
            # logits path already runs the same kernel under CUDA graphs.
            if not is_dense_head_weight(
                getattr(lm_head, "weight", None)
            ) and not should_apply_lm_head_quant_method(
                lm_head, getattr(lm_head, "quant_method", None)
            ):
                return _eager("unsupported quantized lm_head")
            self.draft_model.lm_head = lm_head
            if self.ps.tp_rank == 0:
                logger.info(
                    "DFLASH selector decode (greedy + sampling) folded into the "
                    "draft cuda graph."
                )
            return _SelectorDraftSampler(
                draft_model=self.draft_model,
                block_size=self.block_size,
                max_bs=max(get_exec().graph.cuda_graph_config.decode.bs),
                device=self.device,
            )
        if not hasattr(lm_head, "weight"):
            return _eager("quantized lm_head has no dense weight")
        if not is_dense_head_weight(lm_head.weight):
            # Quantized lm_head (FP8/INT) would break the static matmul.
            return _eager("quantized lm_head")
        tp_group = get_tp_group()
        if not hasattr(lm_head, "shard_indices"):
            if tp_group.world_size != 1:
                # No shard metadata to recover per-rank vocab offsets from.
                return _eager("tp>1 without shard_indices")
            num_org = int(lm_head.weight.shape[0])
            org_vocab_start = 0
        else:
            shard = lm_head.shard_indices
            if int(shard.num_added_elements) != 0:
                return _eager("added vocab")
            num_org = int(shard.num_org_elements)
            org_vocab_start = int(shard.org_vocab_start_index)
        if self.ps.tp_rank == 0:
            logger.info(
                "DFLASH draft greedy head folded into the draft cuda graph (tp=%d).",
                tp_group.world_size,
            )
        return _DflashDraftSampler(
            weight=lm_head.weight,
            block_size=self.block_size,
            num_org=num_org,
            org_vocab_start=org_vocab_start,
            max_bs=max(get_exec().graph.cuda_graph_config.decode.bs),
            tp_group=tp_group if tp_group.world_size > 1 else None,
        )

    def _init_fused_kv_helper(self) -> None:
        """Initialize the fused KV materialization helper with pre-stacked weights."""
        try:
            layers = self.draft_model.layers
            fused_disable_reason: Optional[str] = None

            if len(layers) == 0:
                fused_disable_reason = "no layers found"
            elif not getattr(self.draft_model, "supports_fused_context_kv", True):
                fused_disable_reason = "draft model does not support fused context KV"

            if fused_disable_reason is not None:
                if self.ps.tp_rank == 0:
                    logger.info(
                        "DFLASH fused KV materialization disabled: %s",
                        fused_disable_reason,
                    )
                self._use_fused_kv_materialize = False
                self._fused_kv_helper = None
                return

            for layer_idx, layer in enumerate(layers):
                attn = layer.self_attn
                eligible, reason = can_dflash_use_fused_qkv_proj(attn.qkv_proj)
                if not eligible:
                    fused_disable_reason = f"{reason}: layer={layer_idx}"
                    break

                # Keep semantics aligned with set_kv_buffer scaling behavior.
                k_scale = getattr(attn.attn, "k_scale", None)
                v_scale = getattr(attn.attn, "v_scale", None)
                if k_scale is not None and not math.isclose(float(k_scale), 1.0):
                    fused_disable_reason = (
                        "non-unit k_scale is not supported for fused KV path: "
                        f"layer={layer_idx}, k_scale={k_scale}"
                    )
                    break
                if v_scale is not None and not math.isclose(float(v_scale), 1.0):
                    fused_disable_reason = (
                        "non-unit v_scale is not supported for fused KV path: "
                        f"layer={layer_idx}, v_scale={v_scale}"
                    )
                    break

                rope_is_neox_style = bool(
                    getattr(attn.rotary_emb, "is_neox_style", True)
                )
                if not rope_is_neox_style:
                    fused_disable_reason = (
                        "non-neox RoPE is not supported for fused KV path: "
                        f"layer={layer_idx}, rope_is_neox_style={rope_is_neox_style}"
                    )
                    break

            if fused_disable_reason is not None:
                if self.ps.tp_rank == 0:
                    logger.info(
                        "DFLASH fused KV materialization disabled: %s",
                        fused_disable_reason,
                    )
                self._use_fused_kv_materialize = False
                self._fused_kv_helper = None
                return

            FusedKVMaterializeHelper = _get_fused_kv_materialize_helper()
            first_attn = layers[0].self_attn
            rotary_emb = first_attn.rotary_emb

            self._fused_kv_helper = FusedKVMaterializeHelper(
                layers=layers,
                rotary_emb=rotary_emb,
                num_kv_heads=first_attn.num_kv_heads,
                head_dim=first_attn.head_dim,
                device=self.device,
                max_position_hint=self.target_worker.model_runner.model_config.context_len
                + int(self.block_size),
            )
            if self.ps.tp_rank == 0:
                logger.info(
                    "DFLASH fused KV materialization enabled. "
                    "n_layers=%d, num_kv_heads=%d, head_dim=%d",
                    len(layers),
                    first_attn.num_kv_heads,
                    first_attn.head_dim,
                )
        except Exception as e:
            logger.warning(
                "DFLASH fused KV initialization failed, falling back to sequential path: %s",
                e,
            )
            self._use_fused_kv_materialize = False
            self._fused_kv_helper = None

    def _ensure_draft_block_buffers(self, bs: int) -> None:
        cap = (
            0
            if self._draft_block_ids_buf is None
            else int(self._draft_block_ids_buf.shape[0])
        )
        if cap >= int(bs):
            return

        new_cap = max(int(bs), cap * 2 if cap > 0 else int(bs))
        device = self.device
        block_size = int(self.block_size)
        self._draft_block_ids_buf = torch.empty(
            (new_cap, block_size), dtype=torch.long, device=device
        )
        self._draft_block_positions_buf = torch.empty(
            (new_cap, block_size), dtype=torch.int64, device=device
        )
        self._draft_block_tokens_buf = torch.empty(
            (new_cap, block_size), dtype=torch.long, device=device
        )
        self._draft_verify_out_cache_loc_buf = torch.empty(
            (new_cap, block_size), dtype=torch.int64, device=device
        )
        self._draft_block_end_buf = torch.empty(
            (new_cap,), dtype=torch.int32, device=device
        )
        self._draft_seq_lens_cpu_buf = torch.empty(
            (new_cap,), dtype=torch.int32, device="cpu"
        )

    def __getattr__(self, name):
        # Delegate anything not implemented yet to the target worker. Guard
        # the backing field so a lookup before __init__ sets it raises
        # AttributeError instead of recursing through the property.
        if name == "_target_worker":
            raise AttributeError(name)
        return getattr(self.target_worker, name)

    def clear_cache_pool(self):
        # The target worker owns the shared KV allocator/cache. For the compact
        # sliding-window path, the draft req->token view is rebuilt from committed
        # target state before each draft forward, so there is nothing persistent
        # to flush here.
        pass

    def _gather_req_to_token_masked(
        self,
        *,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        pos2d: torch.Tensor,
        mask: torch.Tensor,
        context: str,
    ) -> torch.Tensor:
        if pos2d.ndim != 2:
            raise RuntimeError(
                f"{context} expected 2D positions, got shape={tuple(pos2d.shape)}."
            )
        if mask.shape != pos2d.shape:
            raise RuntimeError(
                f"{context} mask/position shape mismatch: {tuple(mask.shape)} vs {tuple(pos2d.shape)}."
            )

        if req_pool_indices.dtype != torch.int64:
            req_pool_indices = req_pool_indices.to(torch.int64)
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)

        table_width = int(req_to_token.shape[1])
        if table_width <= 0:
            if bool(mask.any().item()):
                raise RuntimeError(
                    f"{context} req_to_token table is empty but gather mask is non-empty."
                )
            return torch.empty((0,), dtype=torch.int64, device=self.device)

        # Only the masked-off rectangular padding can be out of range in the normal
        # ragged-batch case. Replace those don't-care columns with a valid in-range
        # position before the gather so the kernel only sees real positions.
        safe_pos2d = pos2d.masked_fill(~mask, 0)
        return req_to_token[req_pool_indices[:, None], safe_pos2d][mask].to(torch.int64)

    def _gather_req_to_token_segments(
        self,
        *,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        start: torch.Tensor | None,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        lengths = lengths.to(torch.int64)
        if lengths.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        max_len = int(lengths.max().item())
        if max_len <= 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)

        if req_pool_indices.dtype != torch.int64:
            req_pool_indices = req_pool_indices.to(torch.int64)
        offsets = torch.arange(
            max_len, device=self.device, dtype=torch.int64
        ).unsqueeze(0)
        if start is None:
            pos2d = offsets.expand(req_pool_indices.shape[0], -1)
        else:
            pos2d = start.to(torch.int64).unsqueeze(1) + offsets
        mask = offsets < lengths.unsqueeze(1)
        return self._gather_req_to_token_masked(
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            pos2d=pos2d,
            mask=mask,
            context="DFLASH req_to_token segment gather",
        )

    def _compute_compact_draft_seq_lens(self, seq_lens: torch.Tensor) -> torch.Tensor:
        assert self.draft_window_size is not None
        visible_lens = torch.clamp(
            seq_lens.to(dtype=torch.int32, device=self.device),
            max=int(self.draft_window_size),
        )
        if self.page_size <= 1:
            return visible_lens

        # Paged FA backends derive the page table from local token positions, so the
        # compact suffix must start on a page boundary. Keep up to page_size - 1 extra
        # tokens on the left to preserve valid local page structure.
        seq_lens_i64 = seq_lens.to(torch.int64)
        visible_lens_i64 = visible_lens.to(torch.int64)
        visible_start = seq_lens_i64 - visible_lens_i64
        aligned_start = visible_start - torch.remainder(visible_start, self.page_size)
        return (seq_lens_i64 - aligned_start).to(torch.int32)

    def _compute_compact_draft_seq_lens_host(
        self, host_seq_lens: torch.Tensor, out: torch.Tensor
    ) -> None:
        """Sync-free host upper bound for _compute_compact_draft_seq_lens.

        Deliberately NOT the exact page-align arithmetic: that mapping is a
        non-monotonic sawtooth in [window, window+page), so evaluating it on an
        over-estimated host len (the reserved overlap bound) could UNDER-shoot
        the true device value. min(len, window+page) is its monotonic envelope
        (always >= the exact compact len); consumers only need an upper bound.
        """
        assert self.draft_window_size is not None
        bound = int(self.draft_window_size) + (
            self.page_size if self.page_size > 1 else 0
        )
        lens = host_seq_lens.to(dtype=torch.int64, device="cpu")
        out.copy_(torch.clamp(lens, max=bound).to(torch.int32))

    def _fill_compact_seq_lens_cpu_bound(
        self,
        *,
        batch_seq_lens_cpu: Optional[torch.Tensor],
        nxt_kv_lens_cpu: Optional[torch.Tensor],
        draft_prefix_lens: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """Fill the seq_lens_cpu planning bound, sync-free when a host-side
        length source is available; backends consume it as a safe upper bound
        (same contract as the non-compact path in forward_batch_generation)."""
        if batch_seq_lens_cpu is not None:
            self._compute_compact_draft_seq_lens_host(batch_seq_lens_cpu, out=out)
        elif nxt_kv_lens_cpu is not None:
            self._compute_compact_draft_seq_lens_host(nxt_kv_lens_cpu, out=out)
        else:
            # Last resort: the legacy blocking D2H copy.
            out.copy_(draft_prefix_lens)

    def _rebuild_compact_draft_cache(
        self,
        *,
        req_pool_indices: torch.Tensor,
        prefix_lens: torch.Tensor,
        draft_prefix_lens: torch.Tensor,
        verify_out_cache_loc_2d: torch.Tensor,
        bs: int,
        block_size: int,
    ) -> None:
        """Write the draft-local compact req->token rows: the committed suffix
        window at [0, draft_prefix_len) plus the verify block slots after it."""
        suffix_start = prefix_lens.to(torch.int64) - draft_prefix_lens.to(torch.int64)
        if self._use_triton_compact_rebuild:
            rebuild_compact_draft_req_to_token_func(
                draft_req_to_token=self.draft_model_runner.req_to_token_pool.req_to_token,
                target_req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                req_pool_indices=req_pool_indices,
                suffix_start=suffix_start,
                draft_prefix_lens=draft_prefix_lens,
                verify_out_cache_loc_2d=verify_out_cache_loc_2d,
                batch_size=bs,
                block_size=block_size,
            )
        else:
            suffix_cache_loc = self._gather_req_to_token_segments(
                req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                req_pool_indices=req_pool_indices,
                start=suffix_start,
                lengths=draft_prefix_lens,
            )
            assign_req_to_token_pool_func(
                req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                torch.zeros_like(draft_prefix_lens),
                draft_prefix_lens,
                suffix_cache_loc,
                bs,
            )

            assert self._draft_block_end_buf is not None
            block_end = self._draft_block_end_buf[:bs]
            torch.add(draft_prefix_lens, block_size, out=block_end)
            assign_req_to_token_pool_func(
                req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                draft_prefix_lens,
                block_end,
                verify_out_cache_loc_2d.reshape(-1),
                bs,
            )

    def _resolve_mask_token_id(
        self, *, mask_token: str, mask_token_id: Optional[int] = None
    ) -> int:
        if not isinstance(mask_token, str) or not mask_token:
            raise ValueError(
                f"DFLASH mask_token must be a non-empty string, got {mask_token!r}."
            )

        vocab_size = int(self.target_worker.model_runner.model_config.vocab_size)
        if mask_token_id is not None:
            resolved_id = int(mask_token_id)
            if resolved_id >= vocab_size:
                raise ValueError(
                    "DFLASH mask_token_id is outside the target vocab size. "
                    f"mask_token_id={resolved_id}, vocab_size={vocab_size}. "
                    f"This likely means mask_token={mask_token!r} requires vocab expansion beyond the model's embedding size. "
                    "SGLang does not support resizing target embeddings for DFLASH yet."
                )

            tokenizer = getattr(self.target_worker, "tokenizer", None)
            if tokenizer is not None:
                token_id_from_vocab = tokenizer.get_vocab().get(mask_token, None)
                if (
                    token_id_from_vocab is not None
                    and int(token_id_from_vocab) != resolved_id
                ):
                    raise ValueError(
                        "DFLASH config mismatch: dflash_config.mask_token_id conflicts with tokenizer vocab id "
                        f"for dflash_config.mask_token. mask_token={mask_token!r}, "
                        f"mask_token_id={resolved_id}, tokenizer_vocab_id={int(token_id_from_vocab)}."
                    )
            return resolved_id

        tokenizer = getattr(self.target_worker, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(
                "DFLASH requires tokenizer initialization when dflash_config.mask_token_id is not set "
                "(skip_tokenizer_init is not supported in this mode)."
            )

        resolved_id = None
        if getattr(tokenizer, "mask_token", None) == mask_token:
            resolved_id = getattr(tokenizer, "mask_token_id", None)

        if resolved_id is None:
            # Prefer checking the explicit vocab mapping first.
            vocab = tokenizer.get_vocab()
            resolved_id = vocab.get(mask_token, None)

        if resolved_id is None:
            # Mirror the reference DFlash HF demo by adding the mask token to the tokenizer.
            # This is safe only when the resulting id stays within the target model vocab size.
            added = tokenizer.add_special_tokens({"mask_token": mask_token})
            resolved_id = getattr(tokenizer, "mask_token_id", None)
            if resolved_id is None:
                resolved_id = tokenizer.convert_tokens_to_ids(mask_token)

            if added and self.ps.tp_rank == 0:
                logger.info(
                    "Added DFLASH mask token to tokenizer. token=%s, mask_token_id=%s, tokenizer_len=%s, model_vocab_size=%s",
                    mask_token,
                    resolved_id,
                    len(tokenizer),
                    vocab_size,
                )

        if resolved_id is None or int(resolved_id) < 0:
            raise ValueError(
                "DFLASH requires resolving a mask token id, but it could not be resolved. "
                f"mask_token={mask_token!r}."
            )

        if resolved_id >= vocab_size:
            raise ValueError(
                "DFLASH mask_token_id is outside the target vocab size. "
                f"mask_token_id={resolved_id}, vocab_size={vocab_size}. "
                f"This likely means mask_token={mask_token!r} requires vocab expansion beyond the model's embedding size. "
                "SGLang does not support resizing target embeddings for DFLASH yet."
            )

        return int(resolved_id)

    def _propose_selector_block(
        self,
        *,
        draft_logits_output,
        bs: int,
        lm_head,
        anchor_token_ids: torch.Tensor,
        sampling_info,
    ) -> torch.Tensor:
        """The eager fallback for batches the draft graph cannot take."""
        draft_model = self.draft_model
        if draft_model.lm_head is None:
            draft_model.lm_head = lm_head

        draft_hidden = draft_logits_output.hidden_states
        if draft_hidden is None:
            raise RuntimeError("DFLASH selector draft returned no hidden states.")
        draft_hidden = draft_hidden.view(bs, int(self.block_size), -1)
        pred_hidden = draft_hidden[:, 1:, :]  # [bs, block_size-1, H]
        num_pred = pred_hidden.shape[1]

        candidate_ids, scores = _selector_lattice(
            draft_model, pred_hidden, anchor_token_ids
        )
        device = pred_hidden.device
        # Clamped like DSpark so greedy rows don't divide by zero.
        temperatures = (
            torch.ones(bs, dtype=torch.float32, device=device)
            if sampling_info is None
            else sampling_info.temperatures.view(-1).float().clamp_min(1e-5)
        )
        tokens, q_rows = self.selector.sample_path(
            candidate_ids=candidate_ids,
            scores=scores,
            uniforms=torch.rand(bs, num_pred, dtype=torch.float32, device=device),
            temperatures=temperatures,
            greedy_mask=resolve_greedy_mask(
                bs=bs, sampling_info=sampling_info, device=device
            ),
        )
        if not _is_all_greedy(sampling_info):
            self._selector_sample = (candidate_ids, q_rows)
        return tokens.view(bs, num_pred)

    def _selector_sampling_accept(
        self,
        *,
        candidates: torch.Tensor,
        next_token_logits: torch.Tensor,
        candidate_ids: torch.Tensor,
        q_rows: torch.Tensor,
        sampling_info,
        draft_input,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scatter the selector's sparse q into a dense one for DSpark's kernel."""
        bs, block = candidates.shape
        gamma = block - 1
        vocab = int(next_token_logits.shape[-1])
        # A fresh dense q would zero the whole vocabulary to carry top_k per row.
        buffer = self._draft_probs_buf
        if buffer is None or buffer.shape[0] < bs or buffer.shape[1:] != (gamma, vocab):
            cap = bs if buffer is None else max(bs, buffer.shape[0] * 2)
            buffer = torch.zeros(
                (cap, gamma, vocab), dtype=torch.float32, device=candidates.device
            )
            self._draft_probs_buf = buffer
        draft_probs = buffer[:bs]
        try:
            draft_probs.scatter_(-1, candidate_ids, q_rows.float())
            accept_len, bonus, _ = accept_sampling(
                candidates=candidates,
                target_logits=next_token_logits,
                draft_probs=draft_probs,
                sampling_info=sampling_info,
                draft_input=draft_input,
                gamma=gamma,
                verify_num_draft_tokens=block,
                cutoff_verify_lens=None,
            )
        finally:
            # Here, not before the next write: candidate_ids may be a view of a
            # buffer the next draft step overwrites. In finally because the next
            # call scatters different ids and reads q across the whole vocabulary.
            draft_probs.scatter_(-1, candidate_ids, 0.0)
        return accept_len.to(torch.int32), bonus.to(torch.int64)

    def _greedy_sample_from_quantized_head(
        self,
        *,
        hidden_states: torch.Tensor,
        lm_head,
        chunk_size: int,
    ) -> torch.Tensor:
        """Greedy argmax over a target LM head that has no dense ``weight``.

        A GGUF head stores packed ``qweight`` plus a type tag, so the dense path's
        ``weight[:num_org]`` slicing has nothing to slice. Logits come from the
        layer's own kernel instead -- the same call ``LogitsProcessor._get_logits``
        makes for GGUF models. Padding rows are excluded so argmax cannot return
        an id outside the real vocabulary.
        """
        tp_size = int(get_tp_group().world_size)
        if tp_size != 1:
            raise RuntimeError(
                "DFLASH with a quantized target lm_head is only supported at "
                f"tp=1, got tp_size={tp_size}."
            )

        num_tokens = int(hidden_states.shape[0])
        out_tokens = torch.empty(
            (num_tokens,), dtype=torch.long, device=hidden_states.device
        )
        num_org = int(getattr(lm_head, "org_vocab_size", 0)) or None

        for start in range(0, num_tokens, int(chunk_size)):
            end = min(num_tokens, start + int(chunk_size))
            logits = lm_head.quant_method.apply(lm_head, hidden_states[start:end], None)
            if num_org is not None and logits.shape[-1] > num_org:
                logits = logits[:, :num_org]
            out_tokens[start:end] = torch.argmax(logits, dim=-1).to(torch.long)
        return out_tokens

    def _greedy_sample_from_vocab_parallel_head(
        self,
        *,
        hidden_states: torch.Tensor,
        lm_head,
        chunk_size: int = 256,
    ) -> torch.Tensor:
        """Greedy argmax over the target LM head in a TP-safe way.

        We cannot materialize full logits for large vocabularies efficiently, and with
        TP>1 each rank only owns a shard of the LM head weight. This computes the
        per-rank max, gathers candidates across TP ranks, and selects the global max.
        """

        if hidden_states.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=hidden_states.device)

        if not is_dense_head_weight(getattr(lm_head, "weight", None)):
            return self._greedy_sample_from_quantized_head(
                hidden_states=hidden_states, lm_head=lm_head, chunk_size=chunk_size
            )

        weight = lm_head.weight  # [local_vocab_padded, hidden]
        weight_dtype = weight.dtype
        num_tokens = int(hidden_states.shape[0])
        out_tokens = torch.empty(
            (num_tokens,), dtype=torch.long, device=hidden_states.device
        )

        def _cast_hs(x: torch.Tensor) -> torch.Tensor:
            return x if x.dtype == weight_dtype else x.to(weight_dtype)

        if not hasattr(lm_head, "shard_indices"):
            for start in range(0, num_tokens, int(chunk_size)):
                end = min(num_tokens, start + int(chunk_size))
                hs = _cast_hs(hidden_states[start:end])
                logits = torch.matmul(hs, weight.T)
                out_tokens[start:end] = torch.argmax(logits, dim=-1).to(torch.long)
            return out_tokens

        shard = lm_head.shard_indices
        tp_group = get_tp_group()
        tp_size = int(tp_group.world_size)

        # Valid ranges in the local shard (excluding padding):
        #   base vocab:  [0, num_org)
        #   added vocab: [num_org_padded, num_org_padded + num_added)
        num_org = int(shard.num_org_elements)
        num_org_padded = int(shard.num_org_elements_padded)
        num_added = int(shard.num_added_elements)
        org_vocab_start = int(shard.org_vocab_start_index)
        added_vocab_start = int(shard.added_vocab_start_index)

        def _ensure_local_reduce_buffers(
            chunk_len: int,
            value_dtype: torch.dtype,
            device: torch.device,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if (
                self._draft_greedy_local_cap < chunk_len
                or self._draft_greedy_local_max_buf is None
                or self._draft_greedy_local_arg_buf is None
                or self._draft_greedy_local_max_buf.dtype != value_dtype
                or self._draft_greedy_local_max_buf.device != device
                or self._draft_greedy_local_arg_buf.device != device
            ):
                cap = max(int(chunk_size), chunk_len)
                self._draft_greedy_local_max_buf = torch.empty(
                    (cap,), dtype=value_dtype, device=device
                )
                self._draft_greedy_local_arg_buf = torch.empty(
                    (cap,), dtype=torch.int64, device=device
                )
                self._draft_greedy_local_cap = cap
            return (
                self._draft_greedy_local_max_buf[:chunk_len],
                self._draft_greedy_local_arg_buf[:chunk_len],
            )

        # Fast path (common): single-rank greedy sampling over the base vocab shard.
        # Avoids extra max/id bookkeeping that is only needed for TP sync or added vocab.
        #
        # DFLASH draft sampling only materializes a small fixed block of hidden states
        # each step. On tp=1, splitting those states into many 256-token chunks adds
        # extra matmul/argmax launches without reducing peak memory meaningfully.
        if tp_size == 1 and num_added == 0:
            fast_chunk_size = max(int(chunk_size), 1024)
            for start in range(0, num_tokens, fast_chunk_size):
                end = min(num_tokens, start + fast_chunk_size)
                hs = _cast_hs(hidden_states[start:end])
                if num_org > 0:
                    base_logits = torch.matmul(hs, weight[:num_org].T)
                    local_max, local_arg = _ensure_local_reduce_buffers(
                        end - start, base_logits.dtype, hs.device
                    )
                    torch.max(base_logits, dim=-1, out=(local_max, local_arg))
                    out_tokens[start:end].copy_(local_arg)
                    out_tokens[start:end].add_(org_vocab_start)
                else:
                    out_tokens[start:end] = 0
            return out_tokens

        for start in range(0, num_tokens, int(chunk_size)):
            end = min(num_tokens, start + int(chunk_size))
            hs = _cast_hs(hidden_states[start:end])
            chunk_len = int(hs.shape[0])

            # Base vocab logits.
            if num_org > 0:
                base_logits = torch.matmul(hs, weight[:num_org].T)
                local_max, local_arg = _ensure_local_reduce_buffers(
                    chunk_len, base_logits.dtype, hs.device
                )
                torch.max(base_logits, dim=-1, out=(local_max, local_arg))
            else:
                local_max = torch.full(
                    (chunk_len,),
                    torch.finfo(weight_dtype).min,
                    dtype=weight_dtype,
                    device=hs.device,
                )
                local_arg = torch.zeros(
                    (chunk_len,), dtype=torch.int64, device=hs.device
                )

            # Added vocab logits (e.g., LoRA-added embeddings), if present.
            if num_added > 0:
                added_slice_start = num_org_padded
                added_slice_end = num_org_padded + num_added
                added_logits = torch.matmul(
                    hs, weight[added_slice_start:added_slice_end].T
                )
                added_max, added_arg = torch.max(added_logits, dim=-1)
                use_added = added_max > local_max
                local_max = torch.where(use_added, added_max, local_max)
                # For base/added conversion below, keep local_arg expressed in the full local
                # weight index space (base + padding + added), matching `lm_head.weight`.
                local_arg = torch.where(
                    use_added, added_arg.to(local_arg.dtype) + num_org_padded, local_arg
                )

            # Convert local argmax indices to global token ids.
            if num_added == 0:
                local_arg.add_(org_vocab_start)
                global_ids = local_arg
            else:
                global_ids = torch.empty(
                    (chunk_len,), dtype=torch.int64, device=hs.device
                )
                is_base = local_arg < num_org
                global_ids[is_base] = org_vocab_start + local_arg[is_base]
                global_ids[~is_base] = added_vocab_start + (
                    local_arg[~is_base] - num_org_padded
                )

            if tp_size == 1:
                out_tokens[start:end] = global_ids.to(torch.long)
                continue

            # Gather per-rank maxima and associated global ids, then select the global max.
            needed = tp_size * chunk_len
            chunk_cap = int(chunk_size)
            if (
                self._draft_greedy_gather_cap < needed
                or self._draft_greedy_gathered_max_buf is None
                or self._draft_greedy_gathered_ids_buf is None
                or self._draft_greedy_gathered_max_buf.dtype != local_max.dtype
                or self._draft_greedy_gathered_max_buf.device != hs.device
            ):
                # Allocate enough space for the max chunk size to avoid reallocations.
                cap = tp_size * chunk_cap
                self._draft_greedy_gathered_max_buf = torch.empty(
                    (cap,), dtype=local_max.dtype, device=hs.device
                )
                self._draft_greedy_gathered_ids_buf = torch.empty(
                    (cap,), dtype=global_ids.dtype, device=hs.device
                )
                self._draft_greedy_gather_cap = cap

            if (
                self._draft_greedy_index_cap < chunk_len
                or self._draft_greedy_best_rank_buf is None
                or self._draft_greedy_rank_index_buf is None
                or self._draft_greedy_selected_ids_buf is None
                or self._draft_greedy_best_rank_buf.device != hs.device
                or self._draft_greedy_selected_ids_buf.device != hs.device
            ):
                self._draft_greedy_best_rank_buf = torch.empty(
                    (chunk_cap,), dtype=torch.int64, device=hs.device
                )
                self._draft_greedy_rank_index_buf = torch.empty(
                    (1, chunk_cap), dtype=torch.int64, device=hs.device
                )
                self._draft_greedy_selected_ids_buf = torch.empty(
                    (1, chunk_cap), dtype=torch.int64, device=hs.device
                )
                self._draft_greedy_index_cap = chunk_cap

            gathered_max = self._draft_greedy_gathered_max_buf[:needed]
            gathered_ids = self._draft_greedy_gathered_ids_buf[:needed]

            tp_group.all_gather_into_tensor(gathered_max, local_max.contiguous())
            tp_group.all_gather_into_tensor(gathered_ids, global_ids.contiguous())
            gathered_max = gathered_max.view(tp_size, chunk_len)
            gathered_ids = gathered_ids.view(tp_size, chunk_len)

            best_rank = self._draft_greedy_best_rank_buf[:chunk_len]
            torch.argmax(gathered_max, dim=0, out=best_rank)

            rank_index = self._draft_greedy_rank_index_buf[:, :chunk_len]
            rank_index[0].copy_(best_rank)
            selected_ids = self._draft_greedy_selected_ids_buf[:, :chunk_len]
            torch.gather(gathered_ids, 0, rank_index, out=selected_ids)
            out_tokens[start:end].copy_(selected_ids.view(-1))

        return out_tokens

    def _append_target_hidden_to_draft_kv_by_loc(
        self,
        *,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
        pre_projected: bool = False,
    ) -> None:
        """Materialize target context features into the draft KV cache at explicit slots.

        For the spec-v2 overlap path, callers can pass dense `[bs, block_size]`
        `cache_loc_2d` plus `commit_lens`; the prefix-valid writer then commits
        only the live prefix rows without constructing masked/packed index tensors.

        `pre_projected` says `target_hidden` already went through
        `project_target_hidden`. Under PP the last rank projects before shipping
        the features to rank 0, which shrinks the wire payload from the packed
        capture width down to a single hidden_size.
        """
        if target_hidden is None:
            raise RuntimeError("DFLASH missing target hidden context features.")
        if target_hidden.numel() == 0:
            return
        if target_hidden.ndim != 2:
            raise ValueError(
                "DFLASH target_hidden must be 2D, "
                f"got shape={tuple(target_hidden.shape)}."
            )

        if cache_loc.ndim != 1:
            raise ValueError(
                f"DFLASH cache_loc must be 1D, got shape={tuple(cache_loc.shape)}."
            )
        if positions.ndim != 1:
            raise ValueError(
                f"DFLASH positions must be 1D, got shape={tuple(positions.shape)}."
            )
        num_tokens = int(target_hidden.shape[0])
        if int(cache_loc.numel()) != num_tokens:
            raise ValueError(
                "DFLASH cache_loc length mismatch: "
                f"cache_loc={int(cache_loc.numel())}, target_hidden={num_tokens}."
            )
        if int(positions.numel()) != num_tokens:
            raise ValueError(
                "DFLASH positions length mismatch: "
                f"positions={int(positions.numel())}, target_hidden={num_tokens}."
            )
        if cache_loc_2d is not None:
            if cache_loc_2d.ndim != 2:
                raise ValueError(
                    "DFLASH cache_loc_2d must be 2D, "
                    f"got shape={tuple(cache_loc_2d.shape)}."
                )
            if int(cache_loc_2d.numel()) != num_tokens:
                raise ValueError(
                    "DFLASH cache_loc_2d size mismatch: "
                    f"cache_loc_2d={int(cache_loc_2d.numel())}, target_hidden={num_tokens}."
                )
            if commit_lens is None:
                raise ValueError(
                    "DFLASH cache_loc_2d requires commit_lens for prefix-valid writes."
                )

        device = self.model_runner.device
        if cache_loc.device != device:
            cache_loc = cache_loc.to(device, non_blocking=True)
        if positions.device != device:
            positions = positions.to(device, non_blocking=True)
        if target_hidden.device != device:
            target_hidden = target_hidden.to(device, non_blocking=True)

        if cache_loc.dtype != torch.int64:
            cache_loc = cache_loc.to(torch.int64)
        if positions.dtype != torch.int64:
            positions = positions.to(torch.int64)
        if cache_loc_2d is not None:
            if cache_loc_2d.device != device:
                cache_loc_2d = cache_loc_2d.to(device, non_blocking=True)
            if cache_loc_2d.dtype != torch.int64:
                cache_loc_2d = cache_loc_2d.to(torch.int64)
        if commit_lens is not None:
            if commit_lens.device != device:
                commit_lens = commit_lens.to(device, non_blocking=True)
            if commit_lens.dtype != torch.int32:
                commit_lens = commit_lens.to(torch.int32)

        with torch.inference_mode():
            ctx_hidden = (
                target_hidden
                if pre_projected
                else self.draft_model.project_target_hidden(target_hidden)
            )

            if cache_loc_2d is not None:
                bs = int(commit_lens.shape[0])
                if int(cache_loc_2d.shape[0]) != bs:
                    raise ValueError(
                        "DFLASH cache_loc_2d batch size mismatch: "
                        f"cache_loc_2d={tuple(cache_loc_2d.shape)}, commit_lens={tuple(commit_lens.shape)}."
                    )
                if bs == 0:
                    return
                if self._use_fused_kv_materialize and self._fused_kv_helper is not None:
                    try:
                        self._append_target_hidden_fused(
                            ctx_hidden=ctx_hidden,
                            ctx_positions=positions,
                            ctx_cache_loc=cache_loc,
                            ctx_cache_loc_2d=cache_loc_2d,
                            commit_lens=commit_lens,
                        )
                        return
                    except Exception as e:
                        logger.warning(
                            "DFLASH fused prefix-direct KV append failed; falling back to the per-layer prefix-direct path: %s",
                            e,
                        )
                        self._use_fused_kv_materialize = False
                        self._fused_kv_helper = None

                for layer in self.draft_model.layers:
                    attn = layer.self_attn
                    layer_ctx_hidden = self.draft_model.prepare_context_hidden_for_kv(
                        layer, ctx_hidden
                    )
                    k, v = attn.kv_proj_only(layer_ctx_hidden)
                    k = attn.apply_k_norm(k)
                    k = attn.apply_k_rope(positions, k)
                    k = k.view(-1, attn.num_kv_heads, attn.head_dim)
                    v = v.view(-1, attn.num_kv_heads, attn.head_dim)

                    self.draft_model_runner.token_to_kv_pool.set_kv_buffer_prefix_valid(
                        attn.attn,
                        cache_loc_2d,
                        commit_lens,
                        k,
                        v,
                        attn.attn.k_scale,
                        attn.attn.v_scale,
                    )
                return

            if self._use_fused_kv_materialize and self._fused_kv_helper is not None:
                try:
                    self._append_target_hidden_fused(
                        ctx_hidden=ctx_hidden,
                        ctx_positions=positions,
                        ctx_cache_loc=cache_loc,
                    )
                    return
                except Exception as e:
                    logger.warning(
                        "DFLASH fused KV append-by-loc failed; falling back to sequential path: %s",
                        e,
                    )
                    self._use_fused_kv_materialize = False
                    self._fused_kv_helper = None

            self._append_target_hidden_sequential(
                ctx_hidden=ctx_hidden,
                ctx_positions=positions,
                ctx_cache_loc=cache_loc,
            )

    def _append_target_hidden_sequential(
        self,
        ctx_hidden: torch.Tensor,
        ctx_positions: torch.Tensor,
        ctx_cache_loc: torch.Tensor,
    ) -> None:
        for layer in self.draft_model.layers:
            attn = layer.self_attn
            layer_ctx_hidden = self.draft_model.prepare_context_hidden_for_kv(
                layer, ctx_hidden
            )
            if _is_npu:
                _, k, v = attn.forward_prepare_npu(ctx_positions, layer_ctx_hidden)
            else:
                k, v = attn.kv_proj_only(layer_ctx_hidden)
                k = attn.apply_k_norm(k)
                k = attn.apply_k_rope(ctx_positions, k)
            k = k.view(-1, attn.num_kv_heads, attn.head_dim)
            v = v.view(-1, attn.num_kv_heads, attn.head_dim)
            self.draft_model_runner.token_to_kv_pool.set_kv_buffer(
                attn.attn,
                ctx_cache_loc,
                k,
                v,
                attn.attn.k_scale,
                attn.attn.v_scale,
            )

    def _append_target_hidden_fused(
        self,
        ctx_hidden: torch.Tensor,
        ctx_positions: torch.Tensor,
        ctx_cache_loc: torch.Tensor,
        ctx_cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
    ) -> None:
        """Fused KV materialization using batched projection + Triton kernel."""
        token_to_kv_pool = self.draft_model_runner.token_to_kv_pool
        if self._fused_kv_helper is None:
            raise RuntimeError("DFLASH fused KV helper is not initialized.")

        def _write_layer_kv(
            layer_idx: int,
            cache_k: torch.Tensor,
            cache_v: torch.Tensor,
        ) -> None:
            attn = self.draft_model.layers[layer_idx].self_attn.attn
            if ctx_cache_loc_2d is not None and commit_lens is not None:
                token_to_kv_pool.set_kv_buffer_prefix_valid(
                    attn,
                    ctx_cache_loc_2d,
                    commit_lens,
                    cache_k,
                    cache_v,
                    attn.k_scale,
                    attn.v_scale,
                )
            else:
                token_to_kv_pool.set_kv_buffer(
                    attn,
                    ctx_cache_loc,
                    cache_k,
                    cache_v,
                    attn.k_scale,
                    attn.v_scale,
                )

        self._fused_kv_helper.materialize(
            ctx_hidden=ctx_hidden,
            positions=ctx_positions,
            write_layer_kv=_write_layer_kv,
        )

    def _update_target_mamba_state_after_verify(
        self,
        *,
        batch: ScheduleBatch,
        seq_lens_pre_verify: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> None:
        """Commit Mamba intermediate states for accepted verify steps.

        During TARGET_VERIFY, Mamba kernels run with `disable_state_update=True` and
        cache per-step intermediate states. After acceptance, we need to commit the
        state corresponding to each request's last accepted step.
        """
        if not self._need_mamba_verify_commit:
            return
        attn_backend = self.target_worker.model_runner.attn_backend

        last_correct_step_indices = commit_lens.to(torch.int64) - 1
        mamba_steps_to_track = None

        if batch.mamba_track_indices is not None:
            mamba_track_interval = mamba_track_grid(batch.tree_cache.page_size)
            to_track_mask = (
                seq_lens_pre_verify // mamba_track_interval
                != batch.seq_lens // mamba_track_interval
            )
            tracking_point = (
                batch.seq_lens // mamba_track_interval * mamba_track_interval
            )
            to_track_ith = torch.clamp(tracking_point - seq_lens_pre_verify - 1, min=0)
            can_track_mask = to_track_mask & (
                to_track_ith < commit_lens.to(to_track_ith.dtype)
            )
            mamba_steps_to_track = torch.where(
                can_track_mask,
                to_track_ith.to(torch.int64),
                torch.full_like(to_track_ith, -1, dtype=torch.int64),
            )

        model_runner = self.target_worker.model_runner
        if hasattr(attn_backend, "update_mamba_state_after_mtp_verify"):
            attn_backend.update_mamba_state_after_mtp_verify(
                last_correct_step_indices=last_correct_step_indices,
                mamba_track_indices=batch.mamba_track_indices,
                mamba_steps_to_track=mamba_steps_to_track,
                model=model_runner.model,
                req_pool_indices=batch.req_pool_indices[: commit_lens.shape[0]],
            )

    def _ensure_accept_bonus_buffers(self, bs: int) -> None:
        if self._accept_bonus_buffer_cap >= int(bs):
            return

        new_cap = max(
            int(bs),
            (
                self._accept_bonus_buffer_cap * 2
                if self._accept_bonus_buffer_cap > 0
                else int(bs)
            ),
        )
        device = self.device
        block_size = int(self.block_size)
        self._accept_len_buf = torch.empty((new_cap,), dtype=torch.int32, device=device)
        self._commit_lens_bufs = [
            torch.empty((new_cap,), dtype=torch.int32, device=device) for _ in range(2)
        ]
        # int64 keeps the downstream .to(torch.int64) a no-op.
        self._bonus_id_bufs = [
            torch.empty((new_cap,), dtype=torch.int64, device=device) for _ in range(2)
        ]
        self._out_tokens_bufs = [
            torch.empty((new_cap, block_size), dtype=torch.int64, device=device)
            for _ in range(2)
        ]
        self._new_seq_lens_bufs = [
            torch.empty((new_cap,), dtype=torch.int64, device=device) for _ in range(2)
        ]
        self._accept_bonus_buffer_cap = new_cap

    def _next_accept_bonus_buffers(self, bs: int) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        self._ensure_accept_bonus_buffers(bs)
        assert self._accept_len_buf is not None
        slot = self._accept_bonus_buffer_slot
        self._accept_bonus_buffer_slot = (slot + 1) % 2
        return (
            self._accept_len_buf[:bs],
            self._commit_lens_bufs[slot][:bs],
            self._bonus_id_bufs[slot][:bs],
            self._out_tokens_bufs[slot][:bs],
            self._new_seq_lens_bufs[slot][:bs],
        )

    def _validate_phase1_sampling_support(self, batch: ScheduleBatch) -> None:
        sampling_info = batch.sampling_info
        # A selector draft carries its own q and verifies through accept_sampling, so
        # it never falls back to greedy argmax however this build was compiled.
        if (
            sampling_info is None
            or sampling_info.is_all_greedy
            or self.selector is not None
        ):
            return

        if (
            not is_dflash_sampling_verify_available()
            and not self._warned_sampling_fallback
            and self.ps.tp_rank == 0
        ):
            logger.warning(
                "DFLASH non-greedy verification is unavailable on this build/device; "
                "falling back to greedy argmax verification."
            )
            self._warned_sampling_fallback = True

    def _make_next_draft_input_prefill(
        self,
        *,
        bonus_tokens: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> DFlashDraftInputV2:
        return make_draft_input_v2(bonus_tokens=bonus_tokens, new_seq_lens=seq_lens)

    def _make_next_draft_input_decode(
        self,
        *,
        bonus_tokens: torch.Tensor,
        new_seq_lens: torch.Tensor,
    ) -> DFlashDraftInputV2:
        return make_draft_input_v2(bonus_tokens=bonus_tokens, new_seq_lens=new_seq_lens)

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish=None,
        grammar_barrier=None,
        pp_proxy_tensors=None,
    ) -> GenerationBatchResult:
        self._validate_phase1_sampling_support(batch)

        if self._pp_enabled:
            return self._pp_forward_batch_generation(
                batch,
                pp_proxy_tensors=pp_proxy_tensors,
                on_publish=on_publish,
                grammar_barrier=grammar_barrier,
            )

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            # Target prefill: capture DFlash aux hidden states for prompt tokens.
            batch_output = self.target_worker.forward_batch_generation(
                batch, capture_hidden_mode=CaptureHiddenMode.FULL
            )

            logits_output, next_token_ids = (
                batch_output.logits_output,
                batch_output.next_token_ids,
            )
            batch_output.new_seq_lens = batch.seq_lens
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)

            if logits_output.hidden_states is None:
                raise RuntimeError(
                    "DFLASH requires target aux hidden capture for prefill, but got None. "
                    "Make sure the target model has DFlash layers-to-capture configured."
                )

            if batch.extend_lens is None or batch.prefix_lens is None:
                raise RuntimeError(
                    "DFLASH expected extend_lens / prefix_lens to be populated in extend mode, "
                    "but got None."
                )

            # Materialize prompt tokens into the draft KV cache immediately. This is required
            # for radix cache safety (the scheduler may update radix after prefill returns).
            device = next_token_ids.device
            ctx_lens = torch.tensor(batch.extend_lens, dtype=torch.int32, device=device)
            draft_seq_lens = torch.tensor(
                batch.prefix_lens, dtype=torch.int32, device=device
            )

            if batch.out_cache_loc is None:
                raise RuntimeError(
                    "DFLASH prefill expected out_cache_loc, but got None."
                )
            positions, _ = compute_position(
                self.model_runner.prefill_attention_backend_str,
                draft_seq_lens,
                ctx_lens,
                int(sum(batch.extend_lens)),
            )
            self._append_target_hidden_to_draft_kv_by_loc(
                target_hidden=logits_output.hidden_states,
                cache_loc=batch.out_cache_loc,
                positions=positions,
            )

            # Avoid copying large hidden-state buffers to CPU in overlap scheduling.
            logits_output.hidden_states = None

            batch_output.next_draft_input = self._make_next_draft_input_prefill(
                bonus_tokens=next_token_ids,
                seq_lens=batch.seq_lens,
            )
            return batch_output

        # Decode / target-verify stage.
        if batch.spec_info is None:
            batch.spec_info = DFlashDraftInputV2.create_idle_input(device=self.device)

        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashDraftInputV2):
            raise RuntimeError(
                "DFLASH spec-v2 expected DFlashDraftInputV2 state on the running batch."
            )

        if batch.forward_mode.is_idle():
            empty_ids = torch.empty((0,), dtype=torch.int64, device=self.device)
            empty_lens = torch.empty((0,), dtype=torch.int32, device=self.device)
            next_draft_input = self._make_next_draft_input_decode(
                bonus_tokens=torch.empty((0,), device=self.device, dtype=torch.int64),
                new_seq_lens=torch.empty((0,), device=self.device, dtype=torch.int64),
            )
            if on_publish is not None:
                on_publish(next_draft_input.new_seq_lens)
            return GenerationBatchResult(
                logits_output=None,
                next_token_ids=empty_ids,
                accept_lens=empty_lens,
                next_draft_input=next_draft_input,
                can_run_cuda_graph=False,
                speculative_num_draft_tokens=int(self.block_size),
                new_seq_lens=next_draft_input.new_seq_lens,
            )

        # `seq_lens` is carried over from the previous overlap iteration and may have been
        # produced on another stream.
        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )

        bs = len(batch.seq_lens)
        device = self.device

        # --- 1) Draft a fixed block with the draft model.
        target_model = self.target_worker.model_runner.model
        embed_module = unwrap_lora_layer(target_model.get_input_embeddings())
        lm_head = unwrap_lora_layer(getattr(target_model, "lm_head", None))
        if lm_head is None or not (
            hasattr(lm_head, "weight")
            or callable(getattr(getattr(lm_head, "quant_method", None), "apply", None))
        ):
            raise RuntimeError(
                "DFLASH requires the target model to expose `lm_head` with either "
                "`weight` or a `quant_method` that can produce logits."
            )

        block_size = int(self.block_size)
        self._ensure_draft_block_buffers(bs)
        assert self._draft_block_ids_buf is not None
        assert self._draft_block_positions_buf is not None
        assert self._draft_block_tokens_buf is not None
        assert self._draft_verify_out_cache_loc_buf is not None
        assert self._draft_block_end_buf is not None
        assert self._draft_seq_lens_cpu_buf is not None

        block_ids = self._draft_block_ids_buf[:bs]
        prefix_lens = batch.seq_lens
        positions_2d = self._draft_block_positions_buf[:bs]
        verify_out_cache_loc_2d = self._draft_verify_out_cache_loc_buf[:bs]
        if self._use_triton_prepare_block:
            try:
                _prepare_dflash_draft_block_unchecked(
                    bonus_tokens=draft_input.bonus_tokens.view(-1),
                    prefix_lens=prefix_lens.view(-1),
                    req_pool_indices=batch.req_pool_indices.view(-1),
                    req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                    block_ids_out=block_ids,
                    positions_out=positions_2d,
                    cache_loc_out=verify_out_cache_loc_2d,
                    mask_token_id=int(self._mask_token_id),
                )
            except Exception as e:
                self._use_triton_prepare_block = False
                logger.warning(
                    "DFLASH Triton prepare_block failed; falling back to eager path: %s",
                    e,
                )
                block_ids.fill_(int(self._mask_token_id))
                block_ids[:, 0].copy_(draft_input.bonus_tokens)
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
                    batch_size=bs,
                    draft_token_num=block_size,
                    device=device,
                )
                verify_out_cache_loc_2d.copy_(verify_out_cache_loc.view(bs, block_size))
        else:
            block_ids.fill_(int(self._mask_token_id))
            block_ids[:, 0].copy_(draft_input.bonus_tokens)
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
                batch_size=bs,
                draft_token_num=block_size,
                device=device,
            )
            verify_out_cache_loc_2d.copy_(verify_out_cache_loc.view(bs, block_size))

        noise_embedding = embed_module(block_ids)
        if self._noise_embed_scale != 1.0:
            noise_embedding = noise_embedding * self._noise_embed_scale
        input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])

        positions = positions_2d.reshape(-1)
        verify_out_cache_loc = verify_out_cache_loc_2d.reshape(-1)

        seq_lens_cpu = self._draft_seq_lens_cpu_buf[:bs]
        if self.use_compact_draft_cache:
            # Rebuild the draft-local sliding-window view from committed target state.
            draft_prefix_lens = self._compute_compact_draft_seq_lens(prefix_lens)
            self._fill_compact_seq_lens_cpu_bound(
                batch_seq_lens_cpu=batch.seq_lens_cpu,
                nxt_kv_lens_cpu=draft_input.nxt_kv_lens_cpu,
                draft_prefix_lens=draft_prefix_lens,
                out=seq_lens_cpu,
            )
            self._rebuild_compact_draft_cache(
                req_pool_indices=batch.req_pool_indices,
                prefix_lens=prefix_lens,
                draft_prefix_lens=draft_prefix_lens,
                verify_out_cache_loc_2d=verify_out_cache_loc_2d,
                bs=bs,
                block_size=block_size,
            )
            draft_seq_lens = draft_prefix_lens
            draft_seq_lens_sum = int(seq_lens_cpu.sum().item())
        else:
            # Non-windowed path uses the shared overallocated mapping directly.
            # Backend planning only needs a safe upper bound for the committed
            # prefix lengths, not the full allocator reservation length.
            draft_seq_lens = prefix_lens
            if batch.seq_lens_cpu is not None:
                # Host bound = committed prefix + one verify block.
                seq_lens_cpu.copy_(batch.seq_lens_cpu)
                seq_lens_cpu.add_(block_size)
                draft_seq_lens_sum = int(seq_lens_cpu.sum())
            elif draft_input.nxt_kv_lens_cpu is not None:
                # GPU-only backend: reserved is a safe over-estimate.
                seq_lens_cpu.copy_(draft_input.nxt_kv_lens_cpu)
                draft_seq_lens_sum = int(draft_input.nxt_kv_lens_sum)
            else:
                seq_lens_cpu.copy_(prefix_lens.to("cpu", dtype=torch.int32))
                draft_seq_lens_sum = int(prefix_lens.sum().item())

        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
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

        if self.selector is not None:
            self._selector_sample = None
            if self._draft_sampler is not None:
                # Consumed by the in-graph sample; must be staged before the replay.
                self._draft_sampler.stage_sampling_params(
                    bs=bs, sampling_info=batch.sampling_info
                )

        with torch.inference_mode():
            draft_out = self.draft_model_runner.forward(forward_batch)
        draft_logits_output = draft_out.logits_output

        folded = self._draft_sampler is not None and draft_out.can_run_graph
        if folded:
            draft_next = self._draft_sampler.out[
                : bs * (int(self.block_size) - 1)
            ].view(bs, int(self.block_size) - 1)
            if self.selector is not None and not _is_all_greedy(batch.sampling_info):
                self._selector_sample = (
                    self._draft_sampler.candidate_out[:bs],
                    self._draft_sampler.q_out[:bs],
                )
        elif self.selector is not None:
            draft_next = self._propose_selector_block(
                draft_logits_output=draft_logits_output,
                bs=bs,
                lm_head=lm_head,
                anchor_token_ids=block_ids[:, 0],
                sampling_info=batch.sampling_info,
            )
        else:
            draft_hidden = draft_logits_output.hidden_states
            if draft_hidden is None:
                raise RuntimeError("DFLASH draft model returned no hidden states.")
            draft_hidden = draft_hidden.view(bs, int(self.block_size), -1)
            draft_next = self._greedy_sample_from_vocab_parallel_head(
                hidden_states=draft_hidden[:, 1:, :].reshape(
                    -1, draft_hidden.shape[-1]
                ),
                lm_head=lm_head,
            ).view(bs, int(self.block_size) - 1)

        draft_tokens = self._draft_block_tokens_buf[:bs]
        draft_tokens[:, 0].copy_(block_ids[:, 0])
        draft_tokens[:, 1:].copy_(draft_next)

        # Must stay ahead of the target verify launch below.
        grammar_tree = (
            GrammarTree.from_linear_chain(draft_tokens) if batch.has_grammar else None
        )

        # --- 2) Target verify.
        # TARGET_VERIFY uses standard causal masking; custom masks are unnecessary here.
        custom_mask = None

        verify_input_ids = draft_tokens.reshape(-1)
        verify_input = DFlashVerifyInput(
            draft_token=verify_input_ids,
            positions=positions,
            draft_token_num=int(self.block_size),
            custom_mask=custom_mask,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )

        batch.out_cache_loc = verify_out_cache_loc
        sampling_info = batch.sampling_info

        seq_lens_pre_verify = (
            batch.seq_lens.clone() if self._need_mamba_verify_commit else None
        )
        seq_lens_cpu_backup = batch.seq_lens_cpu
        seq_lens_sum_backup = batch.seq_lens_sum
        if seq_lens_cpu_backup is not None:
            # Verify host bound = committed prefix + one verify block (matches draft).
            verify_host_seq_lens = seq_lens_cpu_backup + block_size
            batch.seq_lens_cpu = verify_host_seq_lens
            batch.seq_lens_sum = int(verify_host_seq_lens.sum())
        elif draft_input.nxt_kv_lens_cpu is not None:
            batch.seq_lens_cpu = draft_input.nxt_kv_lens_cpu
            batch.seq_lens_sum = int(draft_input.nxt_kv_lens_sum)

        verify_forward_batch, _ = verify_input.prepare_for_verify(
            batch, self.target_worker
        )
        batch.seq_lens_cpu = seq_lens_cpu_backup
        batch.seq_lens_sum = seq_lens_sum_backup

        target_out = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        logits_output = target_out.logits_output
        can_run_cuda_graph = target_out.can_run_cuda_graph

        grammar_mask = None
        if batch.has_grammar:
            grammar_mask = build_grammar_vocab_mask(
                reqs=batch.reqs,
                tree=grammar_tree,
                sampling_info=batch.sampling_info,
                device=logits_output.next_token_logits.device,
                barrier=grammar_barrier,
            )

        if sampling_info is not None:
            apply_dflash_verify_logits_adjustments(
                next_token_logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
                draft_token_num=int(self.block_size),
            )

        # Constrain every chain position before accept picks from it.
        if grammar_mask is not None:
            grammar_mask.apply(logits_output.next_token_logits)

        candidates = draft_tokens
        new_seq_lens = None
        target_predict = None
        if self._selector_sample is not None:
            selector_candidate_ids, selector_q_rows = self._selector_sample
            accept_len, bonus = self._selector_sampling_accept(
                candidates=candidates,
                next_token_logits=logits_output.next_token_logits,
                candidate_ids=selector_candidate_ids,
                q_rows=selector_q_rows,
                sampling_info=sampling_info,
                draft_input=draft_input,
            )
            out_tokens, commit_lens = _commit_accept(candidates, accept_len, bonus)
        elif (
            not _is_all_greedy(sampling_info) and is_dflash_sampling_verify_available()
        ):
            accept_len, bonus = compute_dflash_sampling_correct_drafts_and_bonus(
                candidates=candidates,
                next_token_logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
                max_top_k=draft_input.max_top_k,
                uniform_top_k_value=draft_input.uniform_top_k_value,
            )
            out_tokens, commit_lens = _commit_accept(candidates, accept_len, bonus)
        else:
            target_predict = torch.argmax(logits_output.next_token_logits, dim=-1).view(
                bs, int(self.block_size)
            )
            if self._use_triton_accept_bonus:
                try:
                    (
                        accept_len,
                        commit_lens,
                        bonus,
                        out_tokens,
                        new_seq_lens,
                    ) = self._next_accept_bonus_buffers(bs)
                    _compute_dflash_accept_bonus_triton_unchecked(
                        candidates=candidates,
                        target_top1=target_predict,
                        accept_lens_out=accept_len,
                        commit_lens_out=commit_lens,
                        bonus_ids_out=bonus,
                        out_tokens_out=out_tokens,
                        prefix_lens=prefix_lens,
                        new_seq_lens_out=new_seq_lens,
                    )
                except Exception as e:
                    self._use_triton_accept_bonus = False
                    logger.warning(
                        "DFLASH Triton accept/bonus failed; falling back to eager path: %s",
                        e,
                    )
                    accept_len, bonus = compute_dflash_correct_drafts_and_bonus(
                        candidates=candidates,
                        target_predict=target_predict,
                    )
                    out_tokens, commit_lens = _commit_accept(
                        candidates, accept_len, bonus
                    )
            else:
                accept_len, bonus = compute_dflash_correct_drafts_and_bonus(
                    candidates=candidates,
                    target_predict=target_predict,
                )
                out_tokens, commit_lens = _commit_accept(candidates, accept_len, bonus)

        if SIMULATE_ACC_LEN > 0:
            if SIMULATE_ACC_TOKEN_MODE not in ("fixed", "real-draft-token"):
                raise ValueError(
                    "Invalid SGLANG_SIMULATE_ACC_TOKEN_MODE "
                    f"{SIMULATE_ACC_TOKEN_MODE!r}; expected 'fixed' or "
                    "'real-draft-token'."
                )

            if SIMULATE_ACC_TOKEN_MODE == "real-draft-token" and target_predict is None:
                # The sampling-verify branch does not materialize the target argmax.
                target_predict = torch.argmax(
                    logits_output.next_token_logits, dim=-1
                ).view(bs, int(self.block_size))
            apply_dflash_simulated_acceptance(
                candidates=candidates,
                target_predict=target_predict,
                accept_len=accept_len,
                commit_lens=commit_lens,
                bonus=bonus,
                out_tokens=out_tokens,
                simulate_acc_len=SIMULATE_ACC_LEN,
                simulate_acc_method=SIMULATE_ACC_METHOD,
                simulate_acc_token_mode=SIMULATE_ACC_TOKEN_MODE,
            )
            # The Triton path may have written new_seq_lens from the real
            # accept_len; recompute it from the forced commit_lens.
            new_seq_lens = None

        if batch.return_logprob:
            compute_spec_logprobs(
                batch,
                logits_output,
                out_tokens.reshape(-1),
                chain_stride=block_size,
            )

        if self._need_mamba_verify_commit:
            assert seq_lens_pre_verify is not None
            self._update_target_mamba_state_after_verify(
                batch=batch,
                seq_lens_pre_verify=seq_lens_pre_verify,
                commit_lens=commit_lens,
            )

        if new_seq_lens is None:
            new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
        if on_publish is not None:
            on_publish(new_seq_lens)

        # --- 3) Materialize committed verify-input tokens into draft KV cache.
        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError(
                "DFLASH verify requires target hidden states, but got None."
            )
        hidden = hidden.view(bs, int(self.block_size), -1)

        self._append_target_hidden_to_draft_kv_by_loc(
            target_hidden=hidden.reshape(-1, hidden.shape[-1]),
            cache_loc=verify_out_cache_loc,
            cache_loc_2d=verify_out_cache_loc_2d,
            positions=positions,
            commit_lens=commit_lens,
        )

        # Avoid copying large hidden-state buffers to CPU in overlap scheduling.
        logits_output.hidden_states = None

        next_draft_input = self._make_next_draft_input_decode(
            bonus_tokens=bonus,
            new_seq_lens=new_seq_lens,
        )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=out_tokens.reshape(-1),
            accept_lens=commit_lens,
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=int(self.block_size),
            # The non-overlap (sync) scheduler path advances batch.seq_lens
            # from the result; overlap carries it via next_draft_input instead.
            new_seq_lens=new_seq_lens,
            routed_experts_output=target_out.routed_experts_output,
            indexer_topk_output=target_out.indexer_topk_output,
        )

    # ------------------------------------------------------------------
    # Pipeline parallelism (--enable-spec-pp)
    # ------------------------------------------------------------------
    # Per-microbatch message layout:
    #
    #   proxy edge (0 -> 1 -> ... -> last), next to hidden_states/residual:
    #     spec_draft_token       [bs, block_size]      int64
    #
    #   output edge (last -> 0 -> 1 -> ... ), next to next_token_ids:
    #     spec_commit_lens       [bs]                  int32
    #     spec_new_seq_lens      [bs]                  int64
    #     spec_bonus_tokens      [bs]                  int64
    #     spec_ctx_hidden        [num_tokens, hidden]  popped by rank 0
    #
    # Only `draft_token` travels: it is the one thing a rank cannot derive on its
    # own. Positions and KV slots are recomputed per rank from that rank's own
    # req_to_token -- see `_pp_block_layout` for why taking rank 0's copy is
    # actively wrong.
    #
    # Everything that crosses a step boundary is cloned: the draft block
    # scratch buffers are shared by every microbatch in flight and the sends
    # are not ordered against the next microbatch's writes.
    _PP_PROXY_SPEC_KEYS = ("spec_draft_token",)
    # Present only when the selector drafted under temperature > 0, or under
    # SGLANG_SPEC_PP_DEBUG_CHECK.
    _PP_PROXY_SPEC_OPTIONAL_KEYS = (
        "spec_selector_candidates",
        "spec_selector_q",
        "spec_dbg_req_pool_indices",
        "spec_dbg_seq_lens",
        "spec_dbg_verify_cache_loc",
    )

    def _pp_forward_batch_generation(
        self,
        batch: ScheduleBatch,
        *,
        pp_proxy_tensors=None,
        on_publish=None,
        grammar_barrier=None,
    ) -> GenerationBatchResult:
        if batch.has_grammar:
            raise NotImplementedError(
                "--enable-spec-pp does not support grammar-constrained decoding yet."
            )
        if batch.forward_mode.is_idle():
            raise NotImplementedError(
                "--enable-spec-pp does not support idle batches (DP attention) yet."
            )
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return self._pp_forward_prefill(batch, pp_proxy_tensors, on_publish)
        return self._pp_forward_verify(batch, pp_proxy_tensors, on_publish)

    def _pp_relay_proxy_spec_keys(self, result, incoming) -> None:
        """Keep the spec keys travelling forward past a middle rank.

        The target model rebuilds its proxy dict from scratch, so anything the
        spec step added has to be re-attached on the way out.
        """
        proxy = result.pp_hidden_states_proxy_tensors
        if proxy is None or incoming is None:
            return
        for key in self._PP_PROXY_SPEC_KEYS:
            proxy.tensors.setdefault(key, incoming[key])
        for key in self._PP_PROXY_SPEC_OPTIONAL_KEYS:
            if key in incoming.tensors:
                proxy.tensors.setdefault(key, incoming[key])

    def _pp_forward_prefill(
        self, batch: ScheduleBatch, pp_proxy_tensors, on_publish
    ) -> GenerationBatchResult:
        if batch.extend_lens is None or batch.prefix_lens is None:
            raise RuntimeError(
                "DFLASH expected extend_lens / prefix_lens to be populated in extend "
                "mode, but got None."
            )
        if batch.out_cache_loc is None:
            raise RuntimeError("DFLASH prefill expected out_cache_loc, but got None.")

        if self._pp_is_first:
            # Rank 0 performs the draft-KV write once the last rank ships the
            # projected features back, so it is the only rank that needs the
            # slot/position layout of this prefill.
            device = self.device
            ctx_lens = torch.tensor(
                batch.extend_lens, dtype=torch.int32, device=device
            )
            draft_prefix_lens = torch.tensor(
                batch.prefix_lens, dtype=torch.int32, device=device
            )
            positions, _ = compute_position(
                self.model_runner.prefill_attention_backend_str,
                draft_prefix_lens,
                ctx_lens,
                int(sum(batch.extend_lens)),
            )
            batch.pp_spec_ctx_positions = positions
            batch.pp_spec_ctx_cache_loc = batch.out_cache_loc
            batch.pp_spec_ctx_cache_loc_2d = None

        batch_output = self.target_worker.forward_batch_generation(
            batch,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if not self._pp_is_last:
            return batch_output

        logits_output, next_token_ids = (
            batch_output.logits_output,
            batch_output.next_token_ids,
        )
        batch_output.new_seq_lens = batch.seq_lens
        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)

        if logits_output.hidden_states is None:
            raise RuntimeError(
                "DFLASH requires target aux hidden capture for prefill, but got None. "
                "Make sure the target model has DFlash layers-to-capture configured."
            )
        with torch.inference_mode():
            ctx_hidden = self.draft_model.project_target_hidden(
                logits_output.hidden_states
            )
        # Never leaves the GPU; the D2H in copy_to_cpu only handles small tensors.
        logits_output.hidden_states = None

        batch_output.pp_spec_tensors = {"spec_ctx_hidden": ctx_hidden}
        batch_output.next_draft_input = self._make_next_draft_input_prefill(
            bonus_tokens=next_token_ids,
            seq_lens=batch.seq_lens,
        )
        return batch_output

    def _pp_block_layout(self, batch: ScheduleBatch, bs: int):
        """This step's verify-block positions and KV slots, computed locally.

        Every rank derives these from its *own* req_to_token: the slots are the
        ones this rank's allocator handed out, and its attention reads the same
        table back. Taking rank 0's numbers over the wire instead would make a
        rank write its KV at slots its own mapping does not point at, which
        silently cross-contaminates requests.

        Unlike the non-PP path these are freshly allocated rather than carved out
        of the worker's scratch buffers: up to `pp_loop_size` microbatches are in
        flight at once, and their tensors outlive the launch that produced them
        (the accept result lands one PP iteration later), so a shared buffer
        would hand a later microbatch's rows to an earlier one.
        """
        block_size = int(self.block_size)
        prefix_lens = batch.seq_lens
        positions_2d = prefix_lens.unsqueeze(1) + self._block_pos_offsets
        verify_out_cache_loc = assign_extend_cache_locs_func(
            req_pool_indices=batch.req_pool_indices,
            req_to_token=self.model_runner.req_to_token_pool.req_to_token,
            start_offset=prefix_lens,
            end_offset=prefix_lens + block_size,
            batch_size=bs,
            draft_token_num=block_size,
            device=self.device,
        )
        return positions_2d, verify_out_cache_loc.view(bs, block_size)

    def _pp_draft_block(
        self,
        batch: ScheduleBatch,
        draft_input,
        bs: int,
        positions_2d: torch.Tensor,
        verify_out_cache_loc_2d: torch.Tensor,
    ):
        """Rank-0-only: run the draft over the block laid out by `_pp_block_layout`.

        Returns `draft_tokens [bs, block_size]`.
        """
        block_size = int(self.block_size)
        # Only for the scratch consumed inside this launch (compact rebuild).
        self._ensure_draft_block_buffers(bs)
        lm_head = self._draft_lm_head()
        if lm_head is None or not (
            hasattr(lm_head, "weight")
            or callable(getattr(getattr(lm_head, "quant_method", None), "apply", None))
        ):
            raise RuntimeError(
                "DFLASH under PP requires the replicated target `lm_head` to expose "
                "either `weight` or a `quant_method` that can produce logits."
            )
        embed_module = unwrap_lora_layer(
            self._target_worker.model_runner.model.get_input_embeddings()
        )

        block_ids = torch.full(
            (bs, block_size),
            int(self._mask_token_id),
            dtype=torch.long,
            device=self.device,
        )
        prefix_lens = batch.seq_lens
        verify_out_cache_loc = verify_out_cache_loc_2d.reshape(-1)

        block_ids[:, 0].copy_(draft_input.bonus_tokens)

        noise_embedding = embed_module(block_ids)
        if self._noise_embed_scale != 1.0:
            noise_embedding = noise_embedding * self._noise_embed_scale
        input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])
        positions = positions_2d.reshape(-1)

        seq_lens_cpu = torch.empty((bs,), dtype=torch.int32, device="cpu")
        if self.use_compact_draft_cache:
            draft_prefix_lens = self._compute_compact_draft_seq_lens(prefix_lens)
            self._fill_compact_seq_lens_cpu_bound(
                batch_seq_lens_cpu=batch.seq_lens_cpu,
                nxt_kv_lens_cpu=draft_input.nxt_kv_lens_cpu,
                draft_prefix_lens=draft_prefix_lens,
                out=seq_lens_cpu,
            )
            self._rebuild_compact_draft_cache(
                req_pool_indices=batch.req_pool_indices,
                prefix_lens=prefix_lens,
                draft_prefix_lens=draft_prefix_lens,
                verify_out_cache_loc_2d=verify_out_cache_loc_2d,
                bs=bs,
                block_size=block_size,
            )
            draft_seq_lens = draft_prefix_lens
            draft_seq_lens_sum = int(seq_lens_cpu.sum().item())
        else:
            draft_seq_lens = prefix_lens
            if batch.seq_lens_cpu is not None:
                seq_lens_cpu.copy_(batch.seq_lens_cpu)
                seq_lens_cpu.add_(block_size)
                draft_seq_lens_sum = int(seq_lens_cpu.sum())
            elif draft_input.nxt_kv_lens_cpu is not None:
                seq_lens_cpu.copy_(draft_input.nxt_kv_lens_cpu)
                draft_seq_lens_sum = int(draft_input.nxt_kv_lens_sum)
            else:
                seq_lens_cpu.copy_(prefix_lens.to("cpu", dtype=torch.int32))
                draft_seq_lens_sum = int(prefix_lens.sum().item())

        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
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

        if self.selector is not None:
            self._selector_sample = None
            if self._draft_sampler is not None:
                self._draft_sampler.stage_sampling_params(
                    bs=bs, sampling_info=batch.sampling_info
                )

        with torch.inference_mode():
            draft_out = self.draft_model_runner.forward(forward_batch)
        draft_logits_output = draft_out.logits_output

        if self._draft_sampler is not None and draft_out.can_run_graph:
            draft_next = self._draft_sampler.out[: bs * (block_size - 1)].view(
                bs, block_size - 1
            )
            if self.selector is not None and not _is_all_greedy(batch.sampling_info):
                self._selector_sample = (
                    self._draft_sampler.candidate_out[:bs],
                    self._draft_sampler.q_out[:bs],
                )
        elif self.selector is not None:
            draft_next = self._propose_selector_block(
                draft_logits_output=draft_logits_output,
                bs=bs,
                lm_head=lm_head,
                anchor_token_ids=block_ids[:, 0],
                sampling_info=batch.sampling_info,
            )
        else:
            draft_hidden = draft_logits_output.hidden_states
            if draft_hidden is None:
                raise RuntimeError("DFLASH draft model returned no hidden states.")
            draft_hidden = draft_hidden.view(bs, block_size, -1)
            draft_next = self._greedy_sample_from_vocab_parallel_head(
                hidden_states=draft_hidden[:, 1:, :].reshape(
                    -1, draft_hidden.shape[-1]
                ),
                lm_head=lm_head,
            ).view(bs, block_size - 1)

        draft_tokens = torch.empty(
            (bs, block_size), dtype=torch.long, device=self.device
        )
        draft_tokens[:, 0].copy_(block_ids[:, 0])
        draft_tokens[:, 1:].copy_(draft_next)
        return draft_tokens

    def _pp_debug_check_alignment(
        self, batch, pp_proxy_tensors, prefix_lens, verify_out_cache_loc_2d
    ) -> None:
        """Assert this rank's scheduler agrees with PP rank 0 about the batch.

        Every rank runs its own copy of the scheduler and allocator, so the whole
        design rests on them staying bit-identical. If they drift, each rank
        attends to a different request's KV and the outputs cross-contaminate --
        which reads as a model bug, not a scheduling bug. Gated behind
        SGLANG_SPEC_PP_DEBUG_CHECK because the comparisons force a sync.
        """
        tensors = pp_proxy_tensors.tensors
        for name, mine, key in (
            ("req_pool_indices", batch.req_pool_indices, "spec_dbg_req_pool_indices"),
            ("seq_lens", prefix_lens, "spec_dbg_seq_lens"),
            (
                "verify_cache_loc",
                verify_out_cache_loc_2d,
                "spec_dbg_verify_cache_loc",
            ),
        ):
            expected = tensors.get(key)
            if expected is None:
                continue
            if mine.shape != expected.shape or not torch.equal(
                mine, expected.to(mine.dtype)
            ):
                logger.error(
                    "[spec-pp] %s diverged from PP0: mine=%s pp0=%s",
                    name,
                    mine.flatten()[:32].tolist(),
                    expected.flatten()[:32].tolist(),
                )

    def _pp_snapshot_mamba_state_indices(self, bs: int) -> Optional[torch.Tensor]:
        """Pin this microbatch's mamba slot ids for the deferred verify commit.

        `forward_metadata` is transient: by the time the accept result reaches a
        non-last rank, it already describes a later microbatch.
        """
        if not self._need_mamba_verify_commit:
            return None
        backend = getattr(
            self.target_worker.model_runner.attn_backend, "linear_attn_backend", None
        )
        indices = getattr(
            getattr(backend, "forward_metadata", None), "mamba_cache_indices", None
        )
        return None if indices is None else indices[:bs].clone()

    def _pp_forward_verify(
        self, batch: ScheduleBatch, pp_proxy_tensors, on_publish
    ) -> GenerationBatchResult:
        block_size = int(self.block_size)
        if batch.spec_info is None:
            batch.spec_info = DFlashDraftInputV2.create_idle_input(device=self.device)
        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashDraftInputV2):
            raise RuntimeError(
                "DFLASH spec-v2 expected DFlashDraftInputV2 state on the running batch."
            )

        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )
        bs = len(batch.seq_lens)
        prefix_lens = batch.seq_lens

        # Both are derived from state every rank owns a copy of, so each rank
        # computes them itself; only `draft_token` has to travel.
        positions_2d, verify_out_cache_loc_2d = self._pp_block_layout(batch, bs)
        positions = positions_2d.reshape(-1)
        verify_out_cache_loc = verify_out_cache_loc_2d.reshape(-1)

        if self._pp_is_first:
            draft_tokens = self._pp_draft_block(
                batch, draft_input, bs, positions_2d, verify_out_cache_loc_2d
            )
        else:
            draft_tokens = pp_proxy_tensors["spec_draft_token"].view(bs, block_size)
            if _PP_DEBUG_CHECK:
                self._pp_debug_check_alignment(
                    batch, pp_proxy_tensors, prefix_lens, verify_out_cache_loc_2d
                )

        verify_input = DFlashVerifyInput(
            draft_token=draft_tokens.reshape(-1),
            positions=positions,
            draft_token_num=block_size,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        batch.out_cache_loc = verify_out_cache_loc
        sampling_info = batch.sampling_info

        seq_lens_pre_verify = (
            batch.seq_lens.clone() if self._need_mamba_verify_commit else None
        )
        seq_lens_cpu_backup = batch.seq_lens_cpu
        seq_lens_sum_backup = batch.seq_lens_sum
        if seq_lens_cpu_backup is not None:
            verify_host_seq_lens = seq_lens_cpu_backup + block_size
            batch.seq_lens_cpu = verify_host_seq_lens
            batch.seq_lens_sum = int(verify_host_seq_lens.sum())
        elif draft_input.nxt_kv_lens_cpu is not None:
            batch.seq_lens_cpu = draft_input.nxt_kv_lens_cpu
            batch.seq_lens_sum = int(draft_input.nxt_kv_lens_sum)
        verify_forward_batch, _ = verify_input.prepare_for_verify(
            batch, self.target_worker
        )
        batch.seq_lens_cpu = seq_lens_cpu_backup
        batch.seq_lens_sum = seq_lens_sum_backup

        target_out = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        # Post-verify work that needs the accept result: only the last rank has
        # it, so every rank finishes the step later, when the output message
        # comes around. Clone anything living in a shared scratch buffer.
        batch.pp_spec_ctx_cache_loc = verify_out_cache_loc.clone()
        batch.pp_spec_ctx_cache_loc_2d = verify_out_cache_loc_2d.clone()
        batch.pp_spec_ctx_positions = positions.clone()
        batch.pp_spec_seq_lens_pre_verify = seq_lens_pre_verify
        batch.pp_spec_mamba_state_indices = self._pp_snapshot_mamba_state_indices(bs)
        batch.pp_spec_req_pool_indices = batch.req_pool_indices.clone()
        # `prepare_for_verify` rebuilt `batch.mamba_track_indices` *inside* the
        # scheduler's forward isolation, which restores the field on the way out
        # -- so the deferred commit cannot read it off the batch and needs its
        # own copy. Losing it silently skips interval tracking, which leaves
        # tracked mamba pages holding stale state for the next prefix hit.
        batch.pp_spec_mamba_track_indices = (
            None
            if batch.mamba_track_indices is None
            else batch.mamba_track_indices.clone()
        )

        if not self._pp_is_last:
            proxy = target_out.pp_hidden_states_proxy_tensors
            if self._pp_is_first:
                proxy.tensors["spec_draft_token"] = draft_tokens.clone()
                if self._selector_sample is not None:
                    candidates, q_rows = self._selector_sample
                    proxy.tensors["spec_selector_candidates"] = candidates.clone()
                    proxy.tensors["spec_selector_q"] = q_rows.clone()
                if _PP_DEBUG_CHECK:
                    proxy.tensors["spec_dbg_req_pool_indices"] = (
                        batch.req_pool_indices.clone()
                    )
                    proxy.tensors["spec_dbg_seq_lens"] = prefix_lens.clone()
                    proxy.tensors["spec_dbg_verify_cache_loc"] = (
                        batch.pp_spec_ctx_cache_loc_2d
                    )
            else:
                self._pp_relay_proxy_spec_keys(target_out, pp_proxy_tensors)
            return target_out

        return self._pp_verify_accept(
            batch=batch,
            target_out=target_out,
            draft_tokens=draft_tokens,
            draft_input=draft_input,
            sampling_info=sampling_info,
            pp_proxy_tensors=pp_proxy_tensors,
            prefix_lens=prefix_lens,
            bs=bs,
            on_publish=on_publish,
        )
    def _pp_verify_accept(
        self,
        *,
        batch: ScheduleBatch,
        target_out: GenerationBatchResult,
        draft_tokens: torch.Tensor,
        draft_input,
        sampling_info,
        pp_proxy_tensors,
        prefix_lens: torch.Tensor,
        bs: int,
        on_publish,
    ) -> GenerationBatchResult:
        """Last-rank tail of a PP verify step: accept, then publish the result.

        The mamba commit and the draft-KV write are deliberately *not* done here.
        They are deferred to the point where the accept result has reached the
        rank that owns the state -- which for the last rank is the same round
        trip every other rank waits for, so all ranks run one code path.
        """
        block_size = int(self.block_size)
        logits_output = target_out.logits_output
        candidates = draft_tokens

        if sampling_info is not None:
            apply_dflash_verify_logits_adjustments(
                next_token_logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
                draft_token_num=block_size,
            )

        if (
            self.selector is not None
            and not _is_all_greedy(sampling_info)
            and not self._pp_is_first
        ):
            if "spec_selector_candidates" not in pp_proxy_tensors.tensors:
                raise RuntimeError(
                    "--enable-spec-pp: the selector's sampling state did not reach "
                    "the last rank; the draft rank must forward "
                    "spec_selector_candidates / spec_selector_q."
                )
            self._selector_sample = (
                pp_proxy_tensors["spec_selector_candidates"],
                pp_proxy_tensors["spec_selector_q"],
            )
        if self._selector_sample is not None:
            selector_candidate_ids, selector_q_rows = self._selector_sample
            accept_len, bonus = self._selector_sampling_accept(
                candidates=candidates,
                next_token_logits=logits_output.next_token_logits,
                candidate_ids=selector_candidate_ids,
                q_rows=selector_q_rows,
                sampling_info=sampling_info,
                draft_input=draft_input,
            )
            out_tokens, commit_lens = _commit_accept(candidates, accept_len, bonus)
        elif (
            not _is_all_greedy(sampling_info) and is_dflash_sampling_verify_available()
        ):
            accept_len, bonus = compute_dflash_sampling_correct_drafts_and_bonus(
                candidates=candidates,
                next_token_logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
                max_top_k=draft_input.max_top_k,
                uniform_top_k_value=draft_input.uniform_top_k_value,
            )
            out_tokens, commit_lens = _commit_accept(candidates, accept_len, bonus)
        else:
            # The Triton accept/bonus fast path writes into a rotating scratch
            # buffer; under PP those tensors have to outlive the step, so stay
            # on the allocating path here.
            target_predict = torch.argmax(logits_output.next_token_logits, dim=-1).view(
                bs, block_size
            )
            accept_len, bonus = compute_dflash_correct_drafts_and_bonus(
                candidates=candidates,
                target_predict=target_predict,
            )
            out_tokens, commit_lens = _commit_accept(candidates, accept_len, bonus)

        if batch.return_logprob:
            compute_spec_logprobs(
                batch,
                logits_output,
                out_tokens.reshape(-1),
                chain_stride=block_size,
            )

        new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
        if on_publish is not None:
            on_publish(new_seq_lens)
        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError(
                "DFLASH verify requires target hidden states, but got None."
            )
        # Project here rather than on the consuming rank: `fc` narrows the packed
        # multi-layer capture down to one hidden_size, which is what keeps the
        # extra last->0 message the same order as the proxy edge itself.
        with torch.inference_mode():
            ctx_hidden = self.draft_model.project_target_hidden(
                hidden.view(-1, hidden.shape[-1])
            )
        logits_output.hidden_states = None

        commit_lens = commit_lens.clone()
        out_tokens = out_tokens.reshape(-1).clone()
        bonus = bonus.clone()
        new_seq_lens = new_seq_lens.clone()

        target_out.next_token_ids = out_tokens
        target_out.accept_lens = commit_lens
        target_out.new_seq_lens = new_seq_lens
        target_out.speculative_num_draft_tokens = block_size
        target_out.next_draft_input = self._make_next_draft_input_decode(
            bonus_tokens=bonus,
            new_seq_lens=new_seq_lens,
        )
        target_out.pp_spec_tensors = {
            "spec_commit_lens": commit_lens,
            "spec_new_seq_lens": new_seq_lens,
            "spec_bonus_tokens": bonus,
            "spec_ctx_hidden": ctx_hidden,
        }
        if _PP_DEBUG_CHECK:
            # Which microbatch this accept result describes. The ring pairs the
            # output message with a batch by position, so a desync silently
            # applies one microbatch's accept result to another's requests.
            target_out.pp_spec_tensors["spec_dbg_out_req_pool_indices"] = (
                batch.req_pool_indices.clone()
            )
        return target_out
    # --- Deferred post-verify work. The scheduler drives these once the accept
    # --- result has come around the ring to this rank.

    @property
    def pp_spec_enabled(self) -> bool:
        """True when this worker is running the --enable-spec-pp code path."""
        return self._pp_enabled

    def pp_rebuild_next_draft_input(
        self, *, bonus_tokens: torch.Tensor, new_seq_lens: torch.Tensor
    ) -> DFlashDraftInputV2:
        """Rebuild the next step's draft state on a rank that never ran accept."""
        return make_draft_input_v2(
            bonus_tokens=bonus_tokens, new_seq_lens=new_seq_lens
        )

    def pp_commit_mamba_after_verify(
        self, batch: ScheduleBatch, commit_lens: torch.Tensor
    ) -> None:
        """Commit this microbatch's mamba verify states on the local rank.

        Must run after `batch.seq_lens` has been advanced to the post-accept
        lengths: the tracking window is derived from them.
        """
        if not self._need_mamba_verify_commit:
            return
        state_indices = getattr(batch, "pp_spec_mamba_state_indices", None)
        seq_lens_pre_verify = getattr(batch, "pp_spec_seq_lens_pre_verify", None)
        if state_indices is None or seq_lens_pre_verify is None:
            return
        # Snapshot, not batch.mamba_track_indices: forward isolation reverted the
        # field after the verify rebuilt it.
        mamba_track_indices = getattr(batch, "pp_spec_mamba_track_indices", None)

        last_correct_step_indices = commit_lens.to(torch.int64) - 1
        # No interval-track scatter here, matching the non-PP call site: it
        # reaches the commit with `batch.seq_lens` still at the pre-verify prefix
        # (`prepare_for_verify` leaves it alone), so its `to_track_mask` compares
        # a length against itself and every row comes out masked. Tracking the
        # post-accept lengths instead would write track pages the TP path never
        # writes, which shows up later as a prefix hit restoring a state from an
        # offset nothing else agrees on.
        mamba_track_indices = None
        mamba_steps_to_track = None

        model_runner = self.target_worker.model_runner
        # The verify forward keyed its per-step scratch on req_pool_indices (see
        # the spec-pp branch in the GDN backend), so the scatter has to walk that
        # same row space: row r holds request r's snapshots, and rows no request
        # in this microbatch owns are masked off with step -1. The scratch is
        # allocated with one extra padding row (`spec_state_size + 1`), which is
        # where padded/idle rows land, so size the walk to match.
        pool_size = int(model_runner.req_to_token_pool.size) + 1
        rows = batch.pp_spec_req_pool_indices.to(torch.int64).clamp(
            min=0, max=pool_size - 1
        )
        scatter_state_indices = torch.zeros(
            pool_size, dtype=state_indices.dtype, device=state_indices.device
        )
        scatter_state_indices[rows] = state_indices
        scatter_steps = torch.full(
            (pool_size,), -1, dtype=torch.int64, device=state_indices.device
        )
        scatter_steps[rows] = last_correct_step_indices
        if mamba_track_indices is not None:
            scatter_track_indices = torch.zeros(
                pool_size,
                dtype=mamba_track_indices.dtype,
                device=mamba_track_indices.device,
            )
            scatter_track_indices[rows] = mamba_track_indices
            scatter_track_steps = torch.full(
                (pool_size,), -1, dtype=torch.int64, device=state_indices.device
            )
            scatter_track_steps[rows] = mamba_steps_to_track
            mamba_track_indices = scatter_track_indices
            mamba_steps_to_track = scatter_track_steps

        model_runner.attn_backend.update_mamba_state_after_mtp_verify(
            last_correct_step_indices=scatter_steps,
            mamba_track_indices=mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            model=model_runner.model,
            state_indices_tensor=scatter_state_indices,
        )
        batch.pp_spec_mamba_state_indices = None
        batch.pp_spec_seq_lens_pre_verify = None
        batch.pp_spec_mamba_track_indices = None
        batch.pp_spec_req_pool_indices = None

    def pp_write_draft_kv(
        self,
        batch: ScheduleBatch,
        ctx_hidden: torch.Tensor,
        commit_lens: Optional[torch.Tensor],
    ) -> None:
        """Rank-0-only: land the projected target features in the draft KV cache.

        `commit_lens` is None for prefill (every prompt token is committed) and
        the per-request accept length for a verify step.
        """
        cache_loc = getattr(batch, "pp_spec_ctx_cache_loc", None)
        positions = getattr(batch, "pp_spec_ctx_positions", None)
        if cache_loc is None or positions is None:
            raise RuntimeError(
                "--enable-spec-pp: PP rank 0 received spec_ctx_hidden without the "
                "matching draft-KV slot layout."
            )
        self._append_target_hidden_to_draft_kv_by_loc(
            target_hidden=ctx_hidden,
            cache_loc=cache_loc,
            cache_loc_2d=getattr(batch, "pp_spec_ctx_cache_loc_2d", None),
            positions=positions,
            commit_lens=commit_lens,
            pre_projected=True,
        )
        batch.pp_spec_ctx_cache_loc = None
        batch.pp_spec_ctx_cache_loc_2d = None
        batch.pp_spec_ctx_positions = None
