import json
import logging
import math
import os
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from sglang.kernels.ops.memory.small_copy import try_small_copy
from sglang.kernels.ops.speculative.cache_locs import (
    assign_extend_cache_locs_func,
    rebuild_compact_draft_req_to_token_func,
)
from sglang.kernels.ops.speculative.dflash import (
    _compute_dflash_accept_bonus_triton_unchecked,
    _prepare_dflash_draft_block_unchecked,
    _prepare_dflash_mamba_commit,
    compute_dflash_correct_drafts_and_bonus,
)
from sglang.kernels.ops.speculative.dspark.dspark_accept import (
    accept_sampling,
)
from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.environ import envs
from sglang.srt.kernel_plugins.nvidia.binding import (
    built_megakernel,
    megakernel_draft_accept,
    megakernel_draft_aux,
    megakernel_draft_embed,
    megakernel_draft_selector_head,
    megakernel_enabled,
    megakernel_kda_commit,
    megakernel_rank0_broadcast,
    megakernel_sampled_accept,
)
from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.linear.utils import pp_spec_stable_rows_enabled
from sglang.srt.layers.dp_attention import get_dp_tp_group
from sglang.srt.layers.logits_processor import (
    LogitsProcessor,
    LogitsProcessorOutput,
    should_apply_lm_head_quant_method,
)
from sglang.srt.layers.logprob_processor import compute_spec_logprobs
from sglang.srt.lora.layers import unwrap_lora_layer
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.allocation import assign_req_to_token_pool_func
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpaceKind
from sglang.srt.mem_cache.kv.alloc.iteration_ids import IterationIds
from sglang.srt.mem_cache.memory_pool import KVWriteLoc, MLATokenToKVPool
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    compute_position,
    enable_num_token_non_padded,
)
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode, ForwardMode
from sglang.srt.model_executor.model_runner import SamplingPrewarmResult
from sglang.srt.model_executor.runner_utils.pool import (
    disable_graph_pool_borrow,
    graph_pool_borrow_enabled,
)
from sglang.srt.model_loader.weight_utils import get_pp_stage_load_group
from sglang.srt.models.dflash import DFlashDraftModel, DFlashMLAAttention
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import (
    get_exec,
    get_memory,
    get_parallel,
    get_schedule,
    get_serving,
    get_spec,
    mamba_track_grid,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker, HiCacheDraftPlan
from sglang.srt.speculative.dflash_dp import gather_draft_embedding
from sglang.srt.speculative.dflash_draft_l2_prefetch import (
    DraftL2Prefetch,
    draft_l2_prefetch_tensors,
)
from sglang.srt.speculative.dflash_draft_ring import (
    DFlashDraftKVRing,
    draft_kv_ring_enabled,
    draft_kv_ring_rows_per_req,
    draft_kv_ring_window_left,
    set_draft_kv_ring,
)
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_sim_acc import SimAccLenRing
from sglang.srt.speculative.dflash_utils import (
    _get_or_create_chain_verify_buffers,
    apply_dflash_simulated_acceptance,
    apply_dflash_verify_logits_adjustments,
    can_dflash_use_fused_qkv_proj,
    compute_dflash_sampling_correct_drafts_and_bonus,
    is_dense_head_weight,
    is_dflash_sampling_verify_available,
    parse_dflash_draft_config,
)
from sglang.srt.speculative.domino_utils import (
    domino_greedy_rollout,
    validate_domino_runtime,
)
from sglang.srt.speculative.draft_parallel import (
    model_draft_parallel_spec,
    resolve_draft_parallel_layout,
)
from sglang.srt.speculative.draft_worker_common import (
    build_block_pos_offsets,
    build_draft_tp_worker,
    make_draft_block_spec_info,
    make_draft_input_v2,
    make_draft_sampler_capture_hook,
)
from sglang.srt.speculative.dspark_components.dspark_draft import resolve_greedy_mask
from sglang.srt.speculative.lilicorr_utils import (
    build_lilicorr_draft_sampler,
    propose_lilicorr_block,
    publish_anchor,
    resolve_sampling_enabled,
    target_input_embeddings,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_sampling_mask import (
    SpeculativeSamplingMaskCapture,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSync, SpecTpSyncSite
from sglang.srt.speculative.spec_utils import (
    SIMULATE_ACC_LEN,
    SIMULATE_ACC_METHOD,
    SIMULATE_ACC_TOKEN_MODE,
    GrammarTree,
    build_grammar_vocab_mask,
    check_simulate_acc_token_mode,
    draft_pp_context,
    draft_tp_context,
    sample_simulated_acc_len,
    simulated_acc_len_generator,
)
from sglang.srt.utils import async_h2d, is_cuda, is_hip, is_npu, is_xpu

_is_npu = is_npu()


if TYPE_CHECKING:
    from sglang.kernels.ops.speculative.dflash_mla_ctx import MlaContextKV
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

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


def _chain_accept_indices(commit_lens: torch.Tensor, block_size: int) -> torch.Tensor:
    """[bs, block_size] flat offsets into the verify out_cache_loc of each
    committed chain position, -1 past the commit length (EAGLE accept_index
    layout at topk 1)."""
    steps = torch.arange(block_size, device=commit_lens.device)
    row_offsets = torch.arange(commit_lens.numel(), device=commit_lens.device)
    flat = row_offsets[:, None] * block_size + steps
    return torch.where(steps < commit_lens[:, None], flat, -1)


def _resolve_dflash_embedding_module(draft_model, target_model):
    if getattr(draft_model, "is_nemotron_35_draft", False):
        embed_module = draft_model.get_input_embeddings()
        if embed_module is None:
            raise RuntimeError(
                "Nemotron 3.5 DFLASH draft requires its checkpoint embedding."
            )
        return embed_module
    return target_model.get_input_embeddings()


def _hybrid_linear_backend(attn_backend):
    """The HybridLinearAttnBackend that commits Mamba state after a verify, or None."""

    if isinstance(attn_backend, HybridAttnBackend):
        attn_backend = (
            attn_backend.decode_backend
            if attn_backend.spec_attn_is_decode
            else attn_backend.prefill_backend
        )
    return attn_backend if isinstance(attn_backend, HybridLinearAttnBackend) else None


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
    """Selector decode folded into the draft cuda graph.

    On sampling-enabled backends, one captured graph serves both: it always walks
    the sampling path, and a static greedy_mask selects the argmax per row.
    """

    def __init__(
        self, *, draft_model, block_size, max_bs, device, sampling_enabled: bool
    ):
        self.draft_model = draft_model
        self.selector = draft_model.candidate_selector
        self.block_size = int(block_size)
        self.sampling_enabled = sampling_enabled
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
        if sampling_info is None or not self.sampling_enabled:
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


class _MegakernelSelectorDraftSampler:
    """The NVIDIA megakernel head proposes inside the draft forward."""

    def __init__(self, *, head, out, candidate_out, q_out, sampling_enabled):
        self.out, self.candidate_out, self.q_out = out, candidate_out, q_out
        self.sampling_enabled = sampling_enabled
        self._head = head
        self._staged_greedy = False

    def stage_sampling_params(self, *, bs: int, sampling_info) -> None:
        if not self.sampling_enabled:
            return
        if _is_all_greedy(sampling_info):
            if not self._staged_greedy:
                self._head.stage_draft_head_sampling(
                    temperatures=None, greedy_mask=None
                )
                self._staged_greedy = True
            return
        self._head.stage_draft_head_sampling(
            temperatures=sampling_info.temperatures.view(-1)[:bs],
            greedy_mask=resolve_greedy_mask(
                bs=bs, sampling_info=sampling_info, device=self.q_out.device
            )[:bs],
        )
        self._staged_greedy = False

    def __call__(self, hidden_states, input_ids):
        pass


# Cap on captured tails per batch size: one per target graph variant.
_TAIL_VARIANTS_PER_BS = 4


class _MegakernelTail:
    """Post-verify megakernel tail for one batch size: static buffers and graphs."""

    def __init__(self, *, bs: int, block_size: int, device: torch.device):
        rows = bs * block_size
        self.prefix = torch.empty(bs, dtype=torch.int64, device=device)
        self.positions = torch.empty(rows, dtype=torch.int64, device=device)
        self.slots = torch.empty(rows, dtype=torch.int64, device=device)
        self.kda_slots = torch.empty(bs, dtype=torch.int32, device=device)
        self.kda_track = torch.empty(bs, dtype=torch.int32, device=device)
        # target, accept_len, commit_lens, bonus, out_tokens, new_seq_lens per accept
        # slot; consecutive tail rounds take different slots.
        self.outs = tuple(
            (
                torch.empty((bs, block_size), dtype=torch.int64, device=device),
                torch.empty(bs, dtype=torch.int32, device=device),
                torch.empty(bs, dtype=torch.int32, device=device),
                torch.empty(bs, dtype=torch.int64, device=device),
                torch.empty((bs, block_size), dtype=torch.int64, device=device),
                torch.empty(bs, dtype=torch.int64, device=device),
            )
            for _ in range(2)
        )
        self.graphs: dict = {}
        self.pool = None
        self.inputs: Tuple[int, ...] = ()


class _DominoDraftSampler:
    """Capture-safe Domino rollout over a fixed-size draft block."""

    def __init__(
        self,
        *,
        target_embedding,
        lm_head_weight,
        prefix_gru,
        embed_proj,
        vocab_size,
        block_size,
        shift_label,
        max_bs,
        candidate_pool_size,
        tp_group=None,
        lm_head_org_vocab_start=0,
        lm_head_num_org=None,
        lm_head_num_org_padded=None,
    ):
        self.target_embedding = target_embedding
        self.lm_head_weight = lm_head_weight
        self.prefix_gru = prefix_gru
        self.embed_proj = embed_proj
        self.vocab_size = int(vocab_size)
        self.block_size = int(block_size)
        self.shift_label = bool(shift_label)
        self.candidate_pool_size = int(candidate_pool_size)
        self.tp_group = tp_group
        self.lm_head_org_vocab_start = int(lm_head_org_vocab_start)
        self.lm_head_num_org = lm_head_num_org
        self.lm_head_num_org_padded = lm_head_num_org_padded
        max_tokens = int(max_bs) * (self.block_size - 1)
        self.out = torch.empty(
            (max_tokens,), dtype=torch.int64, device=lm_head_weight.device
        )

    def __call__(self, hidden_states, input_ids=None):
        if input_ids is None:
            raise RuntimeError("Domino draft sampler requires block input_ids.")
        bs = hidden_states.shape[0] // self.block_size
        draft_hidden = hidden_states.view(bs, self.block_size, -1)
        bonus_tokens = input_ids.view(bs, self.block_size)[:, 0]
        proposals = domino_greedy_rollout(
            draft_hidden=draft_hidden,
            bonus_tokens=bonus_tokens,
            target_embedding=self.target_embedding,
            lm_head_weight=self.lm_head_weight,
            prefix_gru=self.prefix_gru,
            embed_proj=self.embed_proj,
            vocab_size=self.vocab_size,
            shift_label=self.shift_label,
            candidate_pool_size=self.candidate_pool_size,
            tp_group=self.tp_group,
            lm_head_org_vocab_start=self.lm_head_org_vocab_start,
            lm_head_num_org=self.lm_head_num_org,
            lm_head_num_org_padded=self.lm_head_num_org_padded,
            in_cuda_graph=True,
        )
        self.out[: bs * (self.block_size - 1)].copy_(proposals.reshape(-1))


class DFlashWorkerV2(BaseSpecWorker):
    """DFLASH speculative decoding worker (spec-v2).

    Drives both overlap and non-overlap scheduling, same as EAGLE: the
    scheduler runs it synchronously when overlap is disabled.
    """

    # Without pipeline parallelism every worker hosts the draft; __init__ narrows
    # this to the last PP stage of a pipelined PD prefill.
    _hosts_draft: bool = True

    def weight_update_runners(self) -> List[Tuple[str, "ModelRunner"]]:
        return [("draft", self.draft_model_runner)] if self._hosts_draft else []

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        super().__init__()

        self.server_args = server_args
        self.gpu_id = gpu_id
        self.nccl_port = nccl_port
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self._need_mamba_verify_commit = False
        self.page_size = get_schedule().page_size
        # Normalized in arg_groups.speculative_hook.handle_speculative_decoding.
        self.draft_window_size: Optional[int] = get_spec().speculative_draft_window_size
        self.use_compact_draft_cache = self.draft_window_size is not None
        self.device = target_worker.device
        self.draft_kv_ring: Optional[DFlashDraftKVRing] = None

        self._warned_sampling_fallback = False
        self._warned_lilicorr_eager = False
        self._warned_megakernel_head_eager = False
        self._draft_probs_buf = None
        self._logged_first_verify = False
        self._full_embed_gpu: Optional[torch.Tensor] = None
        # Under dp attention, peer DP ranks run different (idle) paths, so
        # spec broadcasts must stay within the attn-TP group.
        self._tp_sync = SpecTpSync(get_dp_tp_group())
        check_simulate_acc_token_mode()
        # The forced lengths overwrite accept_len after its broadcast, so every rank
        # draws them from the same server-seeded stream.
        self._simulated_acc_len_rng = (
            simulated_acc_len_generator(target_worker.random_seed)
            if SIMULATE_ACC_LEN > 0
            else None
        )

        # DP drafts use per-DP attention TP, independent of idle peers. CP
        # drafts use full TP without CP, including their backend and KV pool.
        self.draft_parallel_layout = resolve_draft_parallel_layout(
            model_draft_parallel_spec(
                target_worker.model_runner.spec_aux_config.draft_model_config,
                spec_algorithm=target_worker.model_runner.spec_algorithm,
            ),
            collapse_cp=True,
        )
        # Inside the draft scope the context answers the draft's narrowed rank.
        self._target_tp_rank = get_parallel().tp_rank
        # On a pipelined PD prefill only the last stage holds the draft: the
        # aux hidden states reach it on the PP proxy, and it materializes and
        # sends the prompt's draft KV. Earlier stages run the target only.
        self._hosts_draft = get_parallel().pp_group.is_last_rank
        if not self._hosts_draft:
            self._draft_worker = None
            self.draft_model_runner = None
            self.draft_model = None
            return
        # The draft worker exists on the last stage only: it reuses the
        # target's seed and loads within the stage, so no all-rank collective
        # waits on the earlier stages (as DSPARK does).
        load_group = get_pp_stage_load_group()
        with draft_pp_context(), draft_tp_context(self.draft_parallel_layout):
            bundle = build_draft_tp_worker(
                server_args=server_args,
                gpu_id=gpu_id,
                nccl_port=nccl_port,
                target_model_config=target_worker.model_runner.model_config,
                algo_label="DFLASH",
                random_seed=target_worker.random_seed,
                load_group=load_group,
            )
        self._draft_worker = bundle.draft_worker
        self.draft_model_runner = bundle.draft_model_runner
        self._draft_sampler = None
        self._megakernel_embed = None
        self.draft_model = bundle.draft_model
        self.selector = self.draft_model.candidate_selector
        # Ascend keeps selector proposal aligned with its greedy-only verify path.
        self._selector_sampling_enabled = not _is_npu
        self.lilicorr = self.draft_model.lilicorr
        # The sampled commit rides the selector's accept path.
        self._lilicorr_sampling_enabled = self.lilicorr is not None and (
            resolve_sampling_enabled(device_supported=self._selector_sampling_enabled)
        )
        self._lilicorr_anchor: Optional[torch.Tensor] = None
        draft_config = parse_dflash_draft_config(
            draft_hf_config=self.draft_model_runner.model_config.hf_config
        )
        self._is_domino = draft_config.is_domino
        if get_parallel().attn_dp_enabled and not _is_npu:
            if self._is_domino or self.lilicorr is not None:
                raise ValueError(
                    "DFLASH with dp attention supports plain DFlash and DFlash2 "
                    "drafts; Domino and LiLiCorr heads are not supported."
                )
        self.domino_candidate_pool_size = int(
            get_spec().speculative_domino_candidate_pool_size
        )
        if self._is_domino:
            if self.domino_candidate_pool_size < 0:
                raise ValueError(
                    "--speculative-domino-candidate-pool-size must be non-negative, "
                    f"got {self.domino_candidate_pool_size}."
                )
            target_model = self.target_worker.model_runner.model
            target_embedding = target_model.get_input_embeddings()
            lm_head = getattr(target_model, "lm_head", None)
            prefix_gru = getattr(self.draft_model, "prefix_gru", None)
            embed_proj = getattr(self.draft_model, "embed_proj", None)
            if lm_head is None or prefix_gru is None or embed_proj is None:
                raise ValueError(
                    "DFLASH Domino requires target lm_head and loaded Domino projector modules."
                )
            validate_domino_runtime(
                device=torch.device(self.device),
                tp_size=int(get_parallel().tp_group.world_size),
                tp_rank=int(get_parallel().tp_rank),
                target_vocab_size=int(self.model_runner.model_config.vocab_size),
                draft_vocab_size=int(self.draft_model_runner.model_config.vocab_size),
                hidden_size=int(self.draft_model.config.hidden_size),
                target_embedding=target_embedding,
                lm_head=lm_head,
                prefix_gru=prefix_gru,
                embed_proj=embed_proj,
            )
        self._block_verification = get_spec().speculative_use_block_verification
        if (
            self._block_verification
            and (
                self._is_domino
                or (self.selector is None and not self._lilicorr_sampling_enabled)
            )
            and get_parallel().tp_rank == 0
        ):
            # A greedy draft is a point mass, where token-wise verification
            # already accepts every prefix with its target probability.
            logger.warning(
                "--speculative-use-block-verification has no effect for this DFLASH "
                "draft: it proposes greedy tokens, and block verification only "
                "differs from token-wise verification for sampled proposals."
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
        if self._is_domino and self.block_size <= 1:
            raise ValueError(
                "DFLASH Domino requires speculative_num_draft_tokens > 1, "
                f"got {self.block_size}."
            )

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
        self._maybe_merge_trained_mask_embedding()
        self._cache_full_embed_weight()
        if get_parallel().tp_rank == 0:
            logger.info(
                "Initialized DFLASH draft runner. attention_backend=%s, model=%s, block_size=%s, draft_window_size=%s, compact_cache=%s",
                bundle.resolved_attention_backend,
                self.draft_model.__class__.__name__,
                self.block_size,
                self.draft_window_size,
                self.use_compact_draft_cache,
            )
            if self._is_domino:
                logger.info(
                    "DFLASH Domino rollout enabled (BF16, TP=%s, block-shared candidate pool size=%s).",
                    int(get_parallel().tp_group.world_size),
                    self.domino_candidate_pool_size,
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
        self._use_fused_kv_materialize = is_cuda() or is_hip() or is_xpu()
        # built on first use: it stacks the draft's kv_a rows, loaded after init
        self._mla_ctx: Optional[MlaContextKV] = None
        self._mla_ctx_resolved = False
        self._fused_kv_helper: Optional[object] = None
        if self._use_fused_kv_materialize:
            self._init_fused_kv_helper()

        supports_gpu_triton = is_cuda() or is_hip() or is_xpu()
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
        self._target_logits_processor = self._defer_target_verify_gather()
        self._aux_graphs = {}
        self._mamba_commit_buffers = None
        self._use_commit_metadata = (
            is_cuda() and envs.SGLANG_DFLASH_COMMIT_METADATA.get()
        )
        self._use_aux_graph = (
            is_cuda()
            and get_exec().graph.cuda_graph_config.decode.backend != Backend.DISABLED
            and self.lilicorr is None
            and envs.SGLANG_DFLASH_AUX_GRAPH.get()
        )
        self._megakernel_aux = self._attach_megakernel_aux()
        self._megakernel_aux_caches: Optional[List[torch.Tensor]] = None
        self._megakernel_kda = megakernel_kda_commit(
            self.target_worker.model_runner.model
        )
        self._megakernel_accept = self._maybe_attach_megakernel_accept()
        self._tail_graph = self._tail_graph_enabled()
        self._warn_megakernel_draft_chain_under_ring()
        self._megakernel_tails: dict = {}
        self._tail_variants_warned = False
        # Megakernel mailbox slots: each alternates on every call, eager or replayed,
        # in the same order on every rank.
        self._accept_parity = 0
        self._aux_parity = 0
        # The KDA commit's switch is only known from its first call (an eager round).
        self._kda_confirmed = False
        self._megakernel_bcast = megakernel_rank0_broadcast(
            self.target_worker.model_runner.model
        )
        self._megakernel_sampled = self._prepare_megakernel_sampled_accept()
        self._kda_commit_buffers: Optional[Tuple[torch.Tensor, ...]] = None
        if self._megakernel_aux is not None and self._target_tp_rank == 0:
            logger.info(
                "DFLASH aux append runs on the NVIDIA megakernel for rows %s, "
                "on the main stream",
                list(self._megakernel_aux.draft_aux_rows),
            )
        # Side stream only where KV frees wait on it (overlap scheduler, no unified
        # memory or HiCache); the megakernel aux's collective keeps it on main.
        self._batched_small_copy = envs.SGLANG_OPT_DFLASH_BATCHED_SMALL_COPY.get()
        self._aux_overlap = (
            self._use_aux_graph
            and self._megakernel_aux is None
            and envs.SGLANG_OPT_DFLASH_AUX_OVERLAP.get()
            and not get_schedule().disable_overlap_schedule
            and not get_memory().enable_unified_memory
            and not get_memory().enable_hierarchical_cache
        )
        self._aux_stream: Optional[torch.cuda.Stream] = None
        self._aux_done: Optional[torch.cuda.Event] = None
        self._aux_pending = False
        if self._aux_overlap:
            logger.info("DFLASH aux graph runs on a side stream beside the draft prep")

    def _prepare_megakernel_sampled_accept(self):
        """Megakernel running the sampled accept on the verify vocab shards, or None."""
        target_lp = self._target_logits_processor
        built = megakernel_sampled_accept(self.target_worker.model_runner.model)
        # Declined before the build: it is collective.
        if (
            built is None
            or target_lp is None
            or self._plain_deferral_refused("sampled accept")
        ):
            return None
        lm_head = unwrap_lora_layer(
            getattr(self.target_worker.model_runner.model, "lm_head", None)
        )
        width = getattr(lm_head, "num_embeddings_per_partition", None)
        if width is None or not built.prepare_sampled_accept(
            vocab_size=int(target_lp.vocab_size),
            softcap=target_lp.final_logit_softcapping or None,
            shard_width=int(width),
            seed=int(torch.initial_seed()),
            topk=int(self.selector.top_k),
        ):
            return None
        self._arm_plain_deferral("sampled accept")
        return built

    def _can_megakernel_sample(
        self, batch: ScheduleBatch, sampling_info, bs: int, shard: torch.Tensor
    ) -> bool:
        """Whether the sampled accept can run on the shard alone (rank-uniform)."""
        if (
            self._megakernel_sampled is None
            or self._selector_sample is None
            or sampling_info is None
            or batch.has_grammar
            or batch.return_logprob
            or self._block_verification
        ):
            return False
        penalizer = sampling_info.penalizer_orchestrator
        return (
            not sampling_info.need_top_k_sampling
            and not sampling_info.need_top_p_sampling
            and not sampling_info.has_custom_logit_processor
            and getattr(sampling_info, "acc_linear_penalties", None) is None
            and (penalizer is None or not penalizer.is_required)
            and getattr(sampling_info, "grammar_mask", None) is None
            and getattr(sampling_info, "logit_bias", None) is None
            and getattr(sampling_info, "sampling_mask_batch_indices", None) is None
            and self._megakernel_sampled.sampled_accept_fits(bs, shard)
        )

    def _defer_target_verify_gather(self) -> Optional[LogitsProcessor]:
        processors = [
            module
            for module in self.target_worker.model_runner.model.modules()
            if isinstance(module, LogitsProcessor)
        ]
        if len(processors) != 1:
            return None
        processors[0].defer_verify_gather = True
        return processors[0]

    def _can_skip_verify_logits(self, batch: ScheduleBatch, sampling_info) -> bool:
        if (
            self._selector_sample is not None
            or batch.has_grammar
            or batch.return_logprob
        ):
            return False
        if sampling_info is None:
            return True
        penalizer = sampling_info.penalizer_orchestrator
        return (
            sampling_info.is_all_greedy
            and not sampling_info.has_custom_logit_processor
            and getattr(sampling_info, "acc_linear_penalties", None) is None
            and (penalizer is None or not penalizer.is_required)
            and getattr(sampling_info, "grammar_mask", None) is None
            and getattr(sampling_info, "logit_bias", None) is None
            and sampling_info.sampling_mask_batch_indices is None
        )

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
        if not self._hosts_draft:
            return (self._target_worker.model_runner.attn_backend,)
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
        if not self._hosts_draft:
            return
        # Without draft windowing, the draft worker aliases the target
        # request->token mapping and allocation state. With draft windowing
        # enabled, the draft worker keeps a private compact req->token table
        # over the same global KV index space, so radix-cache/prefix-hit KV
        # remains reusable while draft attention sees only the recent window.
        with draft_tp_context(self.draft_parallel_layout):
            self._draft_worker.alloc_memory_pool(
                memory_pool_config=memory_pool_config,
                req_to_token_pool=(
                    None if self.use_compact_draft_cache else req_to_token_pool
                ),
                token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            )
        if draft_kv_ring_enabled():
            self._init_draft_kv_ring()

    def _init_draft_kv_ring(self) -> None:
        if self.use_compact_draft_cache:
            raise ValueError(
                "SGLANG_DFLASH_DRAFT_KV_RING replaces the compact draft table; "
                "drop --speculative-draft-window-size."
            )
        parallel = get_parallel()
        dcp_size = parallel.attn_dcp_size
        pool = self.draft_model_runner.token_to_kv_pool
        if dcp_size > 1 and not pool.rows_are_dcp_collapsed:
            raise ValueError(
                "SGLANG_DFLASH_DRAFT_KV_RING under DCP needs DCP-collapsed draft rows."
            )
        window_left = draft_kv_ring_window_left(
            self.draft_model_runner.model_config.hf_config
        )
        req_pool_rows = self.model_runner.req_to_token_pool.req_to_token.shape[0]
        self.draft_kv_ring = DFlashDraftKVRing(
            rows_per_req=draft_kv_ring_rows_per_req(
                window_left=window_left,
                block_size=int(self.block_size),
                dcp_size=dcp_size,
            ),
            dcp_size=dcp_size,
            req_pool_rows=req_pool_rows,
            device=self.device,
            window_left=window_left,
            dcp_rank=parallel.dcp_rank if parallel.dcp_enabled else 0,
            pool=pool,
            mem_pool=pool.maybe_get_custom_mem_pool(),
        )
        self.draft_model_runner.draft_kv_ring = self.draft_kv_ring
        set_draft_kv_ring(self.draft_kv_ring)
        if get_parallel().tp_rank == 0:
            logger.info(
                "DFLASH draft KV ring: %d rows per request per rank (%d positions), "
                "%d req pool rows, draft pool %d rows.",
                self.draft_kv_ring.rows_per_req,
                self.draft_kv_ring.span,
                req_pool_rows,
                pool.size,
            )

    def _build_hicache_draft_plan(self) -> HiCacheDraftPlan:
        # A ring holds no prefix: a cached prefix brings no draft KV back.
        if draft_kv_ring_enabled():
            return HiCacheDraftPlan()
        return super()._build_hicache_draft_plan()

    def init_attention_backends(self):
        if not self._hosts_draft:
            return
        with (
            draft_pp_context(),
            draft_tp_context(self.draft_parallel_layout),
        ):
            self._draft_worker.init_attention_backends()
        self._need_mamba_verify_commit = mambaish_config(
            self.model_runner.model_config
        ) is not None and hasattr(
            self.model_runner.attn_backend,
            "update_mamba_state_after_mtp_verify",
        )

    def init_cuda_graphs(self):
        if not self._hosts_draft:
            return
        target_dp_enabled = get_parallel().attn_dp_enabled
        with (
            draft_pp_context(),
            draft_tp_context(self.draft_parallel_layout),
        ):
            capture_decode_cuda_graph = (
                get_exec().graph.cuda_graph_config.decode.backend != Backend.DISABLED
            )
            if (
                capture_decode_cuda_graph
                and current_platform.is_out_of_tree()
                and not current_platform.capabilities.graph_capture
            ):
                capture_decode_cuda_graph = False
                logger.warning(
                    "Disable DFLASH draft cuda graph because %s does not support "
                    "device graph capture.",
                    type(current_platform).__name__,
                )
            if _is_npu and target_dp_enabled and capture_decode_cuda_graph:
                # Idle DP ranks skip the draft step, so they cannot join a
                # shared graph capture/replay; keep the draft eager under dp
                # attention. On CUDA the draft runs attn-TP-local
                # (draft_tp_context, no cross-DP collective) and keys its
                # graphs by the local batch size, so only busy DP groups
                # replay (see DecodeCudaGraphRunner._forward_is_dp_local).
                # target_dp_enabled is read outside the narrowed draft scope,
                # which hides the target's DP flag.
                capture_decode_cuda_graph = False
                if self._target_tp_rank == 0:
                    logger.warning(
                        "Disable DFLASH draft cuda graph because dp attention "
                        "is enabled (draft runs eager)."
                    )
            if (is_cuda() or is_hip()) and capture_decode_cuda_graph:
                available_mem = self._tp_sync.available_memory_gb(
                    SpecTpSyncSite.DFLASH_MEM,
                    self.device,
                    self.gpu_id,
                    group=get_parallel().tp_group,
                )
                if available_mem < 1.0:
                    capture_decode_cuda_graph = False
                    logger.warning(
                        "Disable DFLASH draft cuda graph because only %.2f GB GPU "
                        "memory is available after target backend initialization.",
                        available_mem,
                    )
            if capture_decode_cuda_graph:
                # Both must run before capture so the draft graph folds them in.
                self._megakernel_embed = self._maybe_attach_megakernel_embed()
                self._draft_sampler = self._maybe_build_draft_sampler()
                if self._draft_sampler is not None:
                    self.draft_model_runner.capture_tail_hooks.append(
                        make_draft_sampler_capture_hook(self._draft_sampler)
                    )
                if is_cuda() and envs.SGLANG_OPT_DFLASH_DRAFT_L2_PREFETCH.get():
                    self._install_draft_l2_prefetch()
            self._draft_worker.init_cuda_graphs(
                capture_decode_cuda_graph=capture_decode_cuda_graph
            )
            if target_dp_enabled and not _is_npu and self.selector is not None:
                self._warm_eager_selector()

    def _warm_eager_selector(self) -> None:
        # Build selector kernels before serving rather than under the watchdog.
        hidden = torch.zeros(
            (int(self.block_size), int(self.draft_model.config.hidden_size)),
            dtype=self.draft_model_runner.dtype,
            device=self.device,
        )
        self._propose_selector_block(
            draft_logits_output=LogitsProcessorOutput(
                next_token_logits=None, hidden_states=hidden
            ),
            bs=1,
            lm_head=unwrap_lora_layer(self._target_worker.model_runner.model.lm_head),
            anchor_token_ids=torch.zeros((1,), dtype=torch.int64, device=self.device),
            sampling_info=None,
        )

    def _install_draft_l2_prefetch(self) -> None:
        prefetch = DraftL2Prefetch(draft_l2_prefetch_tensors(self.draft_model))
        self.draft_model_runner.capture_head_hooks.append(prefetch.capture_head_hook)
        # Appended after the sampler's tail hook: the prefetch joins at the graph's end.
        self.draft_model_runner.capture_tail_hooks.append(prefetch.capture_tail_hook)

    def _prewarm_batch_size(self, block_size: int) -> int:
        """Largest batch the non-greedy verify path can see in one step."""
        bs = self.model_runner.max_decode_logits_rows() // block_size
        max_running_requests = self.model_runner.max_running_requests
        if max_running_requests is not None:
            bs = min(bs, int(max_running_requests))
        return max(1, bs)

    def _measure_sampling_peak(
        self,
        *,
        candidates: torch.Tensor,
        target_logits: torch.Tensor,
        sampling_info: SamplingBatchInfo,
    ) -> int:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base_bytes = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        compute_dflash_sampling_correct_drafts_and_bonus(
            candidates=candidates,
            next_token_logits=target_logits,
            sampling_info=sampling_info,
            max_top_k=None,
            uniform_top_k_value=None,
        )
        torch.cuda.synchronize()
        return torch.cuda.memory_stats()["allocated_bytes.all.peak"] - base_bytes

    def prewarm_sampling(self) -> SamplingPrewarmResult:
        """Rehearse the non-greedy verify path so an undersized graph-pool
        borrow surfaces at startup rather than under traffic."""
        # ROCm prewarms the sampler without graph-pool borrowing; KV is sized before capture.
        if (
            not self._hosts_draft
            or not (is_cuda() or is_hip())
            or not is_dflash_sampling_verify_available()
        ):
            return self.model_runner.prewarm_sampling()

        block_size = int(self.block_size)
        vocab_size = self.model_config.vocab_size
        bs = self._prewarm_batch_size(block_size)
        device = self.device

        sampling_info = SamplingBatchInfo(
            temperatures=torch.ones((bs, 1), dtype=torch.float32, device=device),
            top_ps=torch.full((bs,), 0.95, dtype=torch.float32, device=device),
            top_ks=torch.zeros(bs, dtype=torch.int32, device=device),
            min_ps=torch.zeros(bs, dtype=torch.float32, device=device),
            is_all_greedy=False,
            is_any_greedy=False,
            need_top_p_sampling=True,
            # Top-k off: the sparse path is cheaper, so the dense full-vocab
            # renormalization is the peak this rehearsal has to reproduce.
            need_top_k_sampling=False,
            need_min_p_sampling=False,
            vocab_size=vocab_size,
        )
        candidates = torch.zeros((bs, block_size), dtype=torch.int64, device=device)
        target_logits = torch.zeros(
            (bs * block_size, vocab_size), dtype=torch.float32, device=device
        )

        # Size the process-lifetime chain buffers before anything borrows. The
        # cache is keyed on device index, so pass the resolved device, not "cuda".
        _get_or_create_chain_verify_buffers(
            bs=bs, draft_token_num=block_size, device=target_logits.device
        )
        # One unborrowed pass, so any workspace a kernel caches for the process
        # lifetime lands outside graph storage and off the measured peak.
        with envs.SGLANG_ENABLE_GRAPH_POOL_BORROW.override(False):
            self._measure_sampling_peak(
                candidates=candidates,
                target_logits=target_logits,
                sampling_info=sampling_info,
            )

        borrowing = graph_pool_borrow_enabled()
        try:
            peak_bytes = self._measure_sampling_peak(
                candidates=candidates,
                target_logits=target_logits,
                sampling_info=sampling_info,
            )
        except torch.OutOfMemoryError:
            if not borrowing:
                raise
            reason = (
                "DFLASH sampling rehearsal exhausted graph-pool memory for the "
                f"{bs}x{block_size}x{vocab_size} verify probability matrices"
            )
            logger.warning(
                "Graph pool %s; disabling borrowing and reserving the measured "
                "headroom in the post-capture KV sizing instead",
                reason,
            )
            disable_graph_pool_borrow(reason)
            peak_bytes = self._measure_sampling_peak(
                candidates=candidates,
                target_logits=target_logits,
                sampling_info=sampling_info,
            )

        sampling_input_bytes = target_logits.numel() * target_logits.element_size()
        result = SamplingPrewarmResult(
            sampling_input_bytes=sampling_input_bytes,
            sampling_headroom_bytes=peak_bytes + sampling_input_bytes,
        )
        self.model_runner.sampling_prewarm_result = result
        logger.info(
            "DFLASH sampling pre-warm: borrowing=%s bs=%d peak=%d B headroom=%d B",
            graph_pool_borrow_enabled(),
            bs,
            peak_bytes,
            result.sampling_headroom_bytes,
        )
        return result

    def _maybe_attach_megakernel_head(self, lm_head):
        """Megakernel candidate head if it covers every draft graph bs, else None."""
        head = megakernel_draft_selector_head(self.draft_model)
        graph_bs = set(get_exec().graph.cuda_graph_config.decode.bs)
        if head is None or not graph_bs <= set(head.draft_head_requests):
            return None
        sampling = self._selector_sampling_enabled and head.draft_head_sampling
        if not sampling and not self._default_sampling_is_greedy():
            if self._target_tp_rank == 0:
                logger.warning(
                    "DFLASH megakernel head refused: the server's default sampling is "
                    "not greedy, so requests without sampling params would run the "
                    "eager selector. Serve greedy-only with "
                    "--preferred-sampling-params '{\"temperature\": 0}'."
                )
            return None
        tokens, candidates, q = head.attach_draft_head(
            lm_head=lm_head,
            selector=self.draft_model.candidate_selector,
            draft_config=self.draft_model.draft_config,
        )
        if self._target_tp_rank == 0:
            logger.info(
                "DFLASH selector decode on the NVIDIA megakernel head (%s).",
                "greedy and sampling requests"
                if sampling
                else "greedy; batches with sampling requests take the eager selector",
            )
        return _MegakernelSelectorDraftSampler(
            head=head,
            out=tokens,
            candidate_out=candidates,
            q_out=q,
            sampling_enabled=sampling,
        )

    def _maybe_attach_megakernel_embed(self):
        """Megakernel block embedding if it covers every draft graph bs, else None."""
        embed = megakernel_draft_embed(self.draft_model)
        graph_bs = set(get_exec().graph.cuda_graph_config.decode.bs)
        if embed is None or not graph_bs <= set(embed.draft_embed_requests):
            return None
        if self._full_embed_gpu is not None or self._noise_embed_scale != 1.0:
            # The megakernel embeds the shared target rows unscaled, collective-free.
            if self._target_tp_rank == 0:
                logger.warning(
                    "DFLASH megakernel embed refused: replicated embed=%s, noise scale %s",
                    self._full_embed_gpu is not None,
                    self._noise_embed_scale,
                )
            return None
        embedding = unwrap_lora_layer(
            _resolve_dflash_embedding_module(
                self.draft_model, self.target_worker.model_runner.model
            )
        )
        embed.attach_draft_embed(
            embedding=embedding, mask_token_id=int(self._mask_token_id)
        )
        if self._target_tp_rank == 0:
            logger.info(
                "DFLASH draft block embedding on the NVIDIA megakernel for requests %s",
                list(embed.draft_embed_requests),
            )
        return embed

    def _default_sampling_is_greedy(self) -> bool:
        """Whether a request that sets no sampling params samples greedily."""
        preferred = get_serving().preferred_sampling_params or {}
        if isinstance(preferred, str):
            preferred = json.loads(preferred)
        params = {
            **preferred,
            **self.model_runner.model_config.get_default_sampling_params(),
        }
        return params.get("temperature", 1.0) == 0 or params.get("top_k", -1) == 1

    def _graph_proposal_unusable(self, sampling_info) -> bool:
        """Whether this batch needs the eager selector (greedy-only megakernel head)."""
        if not isinstance(self._draft_sampler, _MegakernelSelectorDraftSampler):
            return False
        if self._draft_sampler.sampling_enabled or _is_all_greedy(sampling_info):
            return False
        if not self._warned_megakernel_head_eager and self._target_tp_rank == 0:
            logger.warning(
                "DFLASH megakernel head: a batch with sampling requests ran the eager "
                "selector; expect a throughput regression on sampling workloads."
            )
            self._warned_megakernel_head_eager = True
        return True

    def _maybe_build_draft_sampler(self):
        def _eager(reason):
            if self._target_tp_rank == 0:
                logger.info("DFLASH draft greedy head kept eager (reason=%s).", reason)
            return None

        if envs.SGLANG_DFLASH_EAGER_DRAFT_SAMPLER.get():
            return _eager("SGLANG_DFLASH_EAGER_DRAFT_SAMPLER=1")
        if self.block_size <= 1:
            return _eager("block_size<=1")
        target_model = self._target_worker.model_runner.model
        lm_head = unwrap_lora_layer(getattr(target_model, "lm_head", None))
        if lm_head is None:
            return _eager("no target lm_head")

        # A gate-admitted quantized head is capture-safe: the target's own
        # logits path already runs the same kernel under CUDA graphs.
        head_supported = is_dense_head_weight(
            getattr(lm_head, "weight", None)
        ) or should_apply_lm_head_quant_method(
            lm_head, getattr(lm_head, "quant_method", None)
        )

        if self.selector is not None:
            # compute_candidates needs the target lm_head attached before capture.
            if not head_supported:
                return _eager("unsupported quantized lm_head")
            self.draft_model.lm_head = lm_head
            megakernel = (
                self._maybe_attach_megakernel_head(lm_head)
                if megakernel_enabled()
                else None
            )
            if megakernel is not None:
                return megakernel
            if self._target_tp_rank == 0:
                logger.info(
                    "DFLASH selector decode folded into the draft cuda graph "
                    "(sampling_enabled=%s).",
                    self._selector_sampling_enabled,
                )
            return _SelectorDraftSampler(
                draft_model=self.draft_model,
                block_size=self.block_size,
                max_bs=max(get_exec().graph.cuda_graph_config.decode.bs),
                device=self.device,
                sampling_enabled=self._selector_sampling_enabled,
            )
        if self.lilicorr is not None:
            if not head_supported:
                return _eager("unsupported quantized lm_head")
            return build_lilicorr_draft_sampler(
                sampling_enabled=self._lilicorr_sampling_enabled,
                head=self.lilicorr,
                draft_model=self.draft_model,
                embed_tokens=target_input_embeddings(target_model),
                lm_head=lm_head,
                block_size=self.block_size,
            )
        if not hasattr(lm_head, "weight"):
            return _eager("quantized lm_head has no dense weight")
        if not is_dense_head_weight(lm_head.weight):
            # Quantized lm_head (FP8/INT) would break the static matmul.
            return _eager("quantized lm_head")
        tp_group = get_parallel().tp_group
        if self._is_domino:
            prefix_gru = self.draft_model.prefix_gru
            embed_proj = self.draft_model.embed_proj
            if prefix_gru is None or embed_proj is None:
                return _eager("Domino projector modules are unavailable")
            if self._target_tp_rank == 0:
                logger.info(
                    "DFLASH Domino rollout folded into the draft cuda graph (tp=%s).",
                    int(tp_group.world_size),
                )
            shard = getattr(lm_head, "shard_indices", None)
            return _DominoDraftSampler(
                target_embedding=target_model.get_input_embeddings(),
                lm_head_weight=lm_head.weight,
                prefix_gru=prefix_gru,
                embed_proj=embed_proj,
                vocab_size=int(self.model_runner.model_config.vocab_size),
                block_size=self.block_size,
                shift_label=self.draft_model.shift_label,
                max_bs=max(get_exec().graph.cuda_graph_config.decode.bs),
                candidate_pool_size=self.domino_candidate_pool_size,
                tp_group=tp_group,
                lm_head_org_vocab_start=(
                    int(shard.org_vocab_start_index) if shard is not None else 0
                ),
                lm_head_num_org=(
                    int(shard.num_org_elements) if shard is not None else None
                ),
                lm_head_num_org_padded=(
                    int(shard.num_org_elements_padded) if shard is not None else None
                ),
            )
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
        if self._target_tp_rank == 0:
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
                if get_parallel().tp_rank == 0:
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
                if get_parallel().tp_rank == 0:
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
            if get_parallel().tp_rank == 0:
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
            # Compact device lengths never exceed the window plus one page.
            out.fill_(
                int(self.draft_window_size)
                + (self.page_size if self.page_size > 1 else 0)
            )

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

    def _maybe_merge_trained_mask_embedding(self) -> None:
        """Merge a trained mask embedding (mask_embedding.pt) into the target table.

        The target's own mask-token row is typically an untrained added-vocab
        slot; overwrite it with the learned vector. VocabParallelEmbedding-aware:
        each rank only updates the row if the token falls within its shard.
        """
        from sglang.srt.arg_groups.model_override_base import resolving_view

        draft_model_path = resolving_view(self.server_args).speculative_draft_model_path
        if draft_model_path is None:
            return

        mask_emb_path = os.path.join(draft_model_path, "mask_embedding.pt")
        if not os.path.exists(mask_emb_path):
            return

        saved = torch.load(mask_emb_path, map_location=self.device, weights_only=True)
        embedding_tensor = saved["embedding"]
        saved_token_id = int(saved["mask_token_id"])

        if saved_token_id != self._mask_token_id:
            raise ValueError(
                f"DFLASH mask_embedding.pt was trained with mask_token_id={saved_token_id}, "
                f"but the current resolved mask_token_id={self._mask_token_id}. "
                "These must match."
            )

        target_model = self._target_worker.model_runner.model
        embed_module = target_model.get_input_embeddings()

        if saved.get("per_position"):
            raise NotImplementedError(
                "DFLASH per-position mask embeddings are not supported in "
                "inference yet (mask_embedding.pt per_position=True)."
            )

        token_id = self._mask_token_id
        shard_indices = getattr(embed_module, "shard_indices", None)
        with torch.no_grad():
            if shard_indices is not None:
                # VocabParallelEmbedding: only update if this token is in our shard.
                start = shard_indices.org_vocab_start_index
                end = shard_indices.org_vocab_end_index
                if start <= token_id < end:
                    local_idx = token_id - start
                    embed_module.weight[local_idx].copy_(
                        embedding_tensor.to(embed_module.weight.dtype)
                    )
            else:
                embed_module.weight[token_id].copy_(
                    embedding_tensor.to(embed_module.weight.dtype)
                )

        if get_parallel().tp_rank == 0:
            logger.info(
                "Merged trained mask embedding into target model "
                "(mask_token_id=%s, source=%s)",
                self._mask_token_id,
                mask_emb_path,
            )

    def _cache_full_embed_weight(self) -> None:
        if not get_parallel().attn_dp_enabled:
            return
        embedding = unwrap_lora_layer(
            _resolve_dflash_embedding_module(
                self.draft_model, self._target_worker.model_runner.model
            )
        )
        # Reconstruct on the table's own group before DP ranks can go idle.
        self._full_embed_gpu = gather_draft_embedding(embedding)

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

            if added and get_parallel().tp_rank == 0:
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
            if sampling_info is None or not self._selector_sampling_enabled
            else sampling_info.temperatures.view(-1).float().clamp_min(1e-5)
        )
        greedy_mask = (
            torch.ones(bs, dtype=torch.bool, device=device)
            if not self._selector_sampling_enabled
            else resolve_greedy_mask(bs=bs, sampling_info=sampling_info, device=device)
        )
        tokens, q_rows = self.selector.sample_path(
            candidate_ids=candidate_ids,
            scores=scores,
            uniforms=torch.rand(bs, num_pred, dtype=torch.float32, device=device),
            temperatures=temperatures,
            greedy_mask=greedy_mask,
        )
        if self._selector_sampling_enabled and not _is_all_greedy(sampling_info):
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
                block_verification=self._block_verification,
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
        tp_size = int(get_parallel().tp_group.world_size)
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
        tp_group = get_parallel().tp_group
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

    def _draft_write_ids(
        self, plan: IterationIds, kind=IdSpaceKind.FULL
    ) -> Optional[torch.Tensor]:
        """The plan's write ids as the draft pool indexes them: the target's
        physical ids for a fused draft, the virtual ids for a private one."""
        return plan.write_ids(self.draft_model_runner.runner_ids, kind=kind)

    def _mla_context_kv(self) -> Optional["MlaContextKV"]:
        """Return the ROCm monokernel's stacked context-KV writer, or None."""
        if self._mla_ctx_resolved:
            return self._mla_ctx
        self._mla_ctx_resolved = True
        if not (is_hip() and get_exec().kernel.enable_amd_monokernel):
            return None
        from sglang.kernels.ops.speculative.dflash_mla_ctx import MlaContextKV

        attns = [layer.self_attn for layer in self.draft_model.layers]
        # one GEMM serves every layer only when each reads the same projected row
        if (
            type(self.draft_model).prepare_context_hidden_for_kv
            is not DFlashDraftModel.prepare_context_hidden_for_kv
            or not all(
                isinstance(a, DFlashMLAAttention) and a.q_lora_rank is not None
                for a in attns
            )
        ):
            blocker = "not low-rank-q MLA layers reading one projected context row"
        else:
            blocker = MlaContextKV.blocker(attns)
        if blocker is None:
            self._mla_ctx = MlaContextKV(attns)
            # the stacked kv_a copy would keep serving the old weights
            self.draft_model._derived_weight_cache_error = (
                "Online weight updates are not supported with the stacked DFLASH MLA "
                "context KV of --enable-amd-monokernel."
            )
        if get_parallel().tp_rank == 0:
            status = (
                f"{len(attns)} layers" if blocker is None else f"disabled: {blocker}"
            )
            logger.info("DFLASH MLA fused context KV: %s", status)
        return self._mla_ctx

    def _append_target_hidden_to_draft_kv_by_loc(
        self,
        *,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
        extend_lens: Optional[torch.Tensor] = None,
        swa_cache_loc: Optional[torch.Tensor] = None,
        projected: bool = False,
    ) -> None:
        """Materialize target context features into the draft KV cache at explicit slots.

        `projected=True`: `target_hidden` already went through `project_target_hidden`.

        For the spec-v2 overlap path, callers can pass dense `[bs, block_size]`
        `cache_loc_2d` plus `commit_lens`; the prefix-valid writer then commits
        only the live prefix rows without constructing masked/packed index tensors.
        `cache_loc_2d` holds the same ids as `cache_loc`, row-major.
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
        # Drop trailing alignment-padding rows (from TP/EP-size rounding)
        # before materializing into the draft KV; they would corrupt other
        # requests' cache slots.
        expected_tokens = int(cache_loc.numel())
        if num_tokens > expected_tokens:
            if not getattr(self, "_logged_padding_trim", False):
                logger.warning(
                    "DFLASH target_hidden has %d trailing padding row(s); trimming "
                    "to cache_loc length=%d (target_hidden=%d). Logged once per worker.",
                    num_tokens - expected_tokens,
                    expected_tokens,
                    num_tokens,
                )
                self._logged_padding_trim = True
            target_hidden = target_hidden[:expected_tokens]
            num_tokens = expected_tokens
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

        if swa_cache_loc is not None and swa_cache_loc.shape != cache_loc.shape:
            raise ValueError(
                "DFLASH full and window cache locations must have matching shapes."
            )

        device = self.model_runner.device
        if cache_loc.device != device:
            cache_loc = cache_loc.to(device, non_blocking=True)
        if swa_cache_loc is not None:
            swa_cache_loc = swa_cache_loc.to(
                device=device, dtype=torch.int64, non_blocking=True
            )
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

        # `cache_loc` arrives in the draft pool's id space (`_draft_write_ids`);
        # the callers' window buffers stay virtual for the compact
        # req_to_token rebuild.
        with (
            torch.inference_mode(),
            draft_tp_context(self.draft_parallel_layout),
        ):
            ctx_hidden = (
                target_hidden
                if projected
                else self.draft_model.project_target_hidden(target_hidden)
            )
            if self.lilicorr is not None:
                self._lilicorr_anchor = publish_anchor(
                    draft_sampler=self._draft_sampler,
                    ctx_hidden=ctx_hidden,
                    extend_lens=extend_lens,
                    commit_lens=commit_lens,
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
                            ctx_swa_cache_loc=swa_cache_loc,
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

                mla_ctx = self._mla_context_kv()
                if mla_ctx is not None:
                    k_nope, k_pe = mla_ctx.project_all(positions, ctx_hidden)
                    # set_mla_kv_buffer_prefix_valid's masking, built once for every layer
                    offsets = torch.arange(cache_loc_2d.shape[1], device=device)
                    loc = KVWriteLoc(
                        torch.where(
                            offsets < commit_lens[:, None], cache_loc_2d, 0
                        ).reshape(-1),
                        physical=True,
                    )
                    pool = self.draft_model_runner.token_to_kv_pool
                    for i, attn in enumerate(mla_ctx.attns):
                        pool.set_mla_kv_buffer(attn.attn_mqa, loc, k_nope[i], k_pe[i])
                    return
                for layer in self.draft_model.layers:
                    attn = layer.self_attn
                    layer_ctx_hidden = self.draft_model.prepare_context_hidden_for_kv(
                        layer, ctx_hidden
                    )
                    if isinstance(attn, DFlashMLAAttention):
                        attn.write_context_kv(
                            pool=self.draft_model_runner.token_to_kv_pool,
                            positions=positions,
                            hidden_states=layer_ctx_hidden,
                            cache_loc=cache_loc,
                            swa_cache_loc=swa_cache_loc,
                            cache_loc_2d=cache_loc_2d,
                            commit_lens=commit_lens,
                        )
                        continue
                    k, v = attn.kv_proj_only(layer_ctx_hidden)
                    k = attn.apply_k_norm(k)
                    k = attn.apply_k_rope(positions, k)
                    k = k.view(-1, attn.num_kv_heads, attn.head_dim)
                    v = v.view(-1, attn.num_kv_heads, attn.head_dim)

                    self.draft_model_runner.token_to_kv_pool.layer_router.layer(
                        attn.attn.layer_id
                    ).store_prefix(
                        attn.attn,
                        KVWriteLoc(
                            cache_loc_2d,
                            swa_loc=swa_cache_loc.view_as(cache_loc_2d)
                            if swa_cache_loc is not None
                            else None,
                            physical=True,
                        ),
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
                        ctx_swa_cache_loc=swa_cache_loc,
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
                ctx_swa_cache_loc=swa_cache_loc,
            )

    def _append_target_hidden_decode(
        self,
        *,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        cache_loc_2d: torch.Tensor,
        commit_lens: torch.Tensor,
        swa_cache_loc: Optional[torch.Tensor] = None,
    ) -> None:
        # The aux graphs hold no sliding-window locations: such drafts append eagerly.
        if not self._use_aux_graph or swa_cache_loc is not None:
            self._append_target_hidden_to_draft_kv_by_loc(
                target_hidden=target_hidden,
                cache_loc=cache_loc,
                positions=positions,
                cache_loc_2d=cache_loc_2d,
                commit_lens=commit_lens,
                swa_cache_loc=swa_cache_loc,
            )
            return
        if (
            self._megakernel_aux is not None
            and int(target_hidden.shape[0]) in self._megakernel_aux.draft_aux_rows
        ):
            self._append_target_hidden_megakernel(
                target_hidden=target_hidden,
                positions=positions,
                cache_loc=cache_loc,
                cache_loc_2d=cache_loc_2d,
                commit_lens=commit_lens,
            )
            return

        # A TP-split fc all-reduces, so project on the main stream in program order;
        # the aux graph may replay on a side stream beside other collectives.
        projected = self.draft_model.fc_tp_columns is not None
        if projected:
            with torch.inference_mode(), draft_tp_context(self.draft_parallel_layout):
                target_hidden = self.draft_model.project_target_hidden(target_hidden)

        key = (tuple(target_hidden.shape), tuple(cache_loc_2d.shape))
        entry = self._aux_graphs.get(key)
        inputs = (target_hidden, cache_loc, positions, commit_lens)
        if entry is None:
            static = tuple(torch.empty_like(t) for t in inputs)
            for dst, src in zip(static, inputs):
                dst.copy_(src)
            hidden_buf, loc_buf, pos_buf, commit_buf = static

            def append():
                self._append_target_hidden_to_draft_kv_by_loc(
                    target_hidden=hidden_buf,
                    cache_loc=loc_buf,
                    positions=pos_buf,
                    cache_loc_2d=loc_buf.view(cache_loc_2d.shape),
                    commit_lens=commit_buf,
                    projected=projected,
                )

            stream = torch.cuda.Stream(device=target_hidden.device)
            current = torch.cuda.current_stream(target_hidden.device)
            stream.wait_stream(current)
            # Warm CUBLAS/custom operators before capture. Repeating these
            # prefix writes is idempotent; uncommitted cache rows stay untouched.
            with torch.cuda.stream(stream):
                append()
            current.wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                append()
            current.wait_stream(stream)
            entry = (static, graph)
            self._aux_graphs[key] = entry
            logger.info(
                "DFLASH aux CUDA graph captured rows=%d", target_hidden.shape[0]
            )
        elif self._batched_small_copy and try_small_copy(
            list(entry[0][1:]), list(inputs[1:])
        ):
            entry[0][0].copy_(inputs[0])
        else:
            for dst, src in zip(entry[0], inputs):
                dst.copy_(src)
        if self._aux_overlap:
            self._replay_aux_overlapped(entry[1])
        else:
            entry[1].replay()

    def _append_target_hidden_megakernel(
        self,
        *,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        cache_loc_2d: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> None:
        """Megakernel decode aux append, one graph per shape, on the main stream."""
        key = ("megakernel", tuple(target_hidden.shape), tuple(cache_loc_2d.shape))
        entry = self._aux_graphs.get(key)
        inputs = (cache_loc, positions, commit_lens)
        parity = self._take_aux_parity()
        if entry is not None:
            for dst, src in zip(entry[0], inputs):
                dst.copy_(src)
            if target_hidden.data_ptr() != entry[2]:
                raise RuntimeError(
                    "DFLASH megakernel aux: the target's capture buffer moved after capture"
                )
            entry[1][parity].replay()
            return
        static = (
            torch.empty_like(cache_loc, dtype=torch.int64),
            torch.empty_like(positions, dtype=torch.int64),
            torch.empty_like(commit_lens, dtype=torch.int32),
        )
        for dst, src in zip(static, inputs):
            dst.copy_(src)
        loc_buf, pos_buf, commit_buf = static
        caches = self._draft_mla_caches()

        # target_hidden is the target graph's capture buffer, read in place.
        def append(slot: int):
            self._megakernel_aux.draft_aux(
                target_hidden=target_hidden,
                positions=pos_buf,
                slots_2d=loc_buf.view(cache_loc_2d.shape),
                commit_lens=commit_buf,
                kv_caches=caches,
                parity=slot,
            )

        # This round's append runs eagerly (the capture warm-up); then both slots.
        stream = torch.cuda.Stream(device=target_hidden.device)
        current = torch.cuda.current_stream(target_hidden.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            append(parity)
        current.wait_stream(stream)
        graphs, pool = {}, None
        for slot in (0, 1):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                append(slot)
            pool, graphs[slot] = graph.pool(), graph
        current.wait_stream(stream)
        self._aux_graphs[key] = (static, graphs, target_hidden.data_ptr())
        logger.info(
            "DFLASH aux CUDA graph captured on the NVIDIA megakernel rows=%d",
            target_hidden.shape[0],
        )

    def _draft_mla_caches(self) -> List[torch.Tensor]:
        """Each draft layer's MLA cache buffer, in draft layer order."""
        if self._megakernel_aux_caches is not None:
            return self._megakernel_aux_caches
        pool = self.draft_model_runner.token_to_kv_pool
        # The megakernel writes whole FP8 rows at slot * 576 bytes: the plain layout.
        plain = (
            type(pool) is MLATokenToKVPool
            and pool.dtype == torch.float8_e4m3fn
            and not pool.dsa_kv_cache_store_fp8
            and (
                pool.dcp_replicated
                or pool.write_loc_is_dcp_resolved
                or get_parallel().attn_dcp_size == 1
            )
        )
        if not plain:
            raise RuntimeError(
                "DFLASH megakernel aux needs a plain FP8 MLA draft KV pool, "
                f"got {type(pool).__name__}"
            )
        caches = []
        for layer in self.draft_model.layers:
            if not isinstance(layer.self_attn, DFlashMLAAttention):
                raise RuntimeError("DFLASH megakernel aux needs MLA draft layers")
            layer_id = layer.self_attn.attn_mqa.layer_id
            caches.append(pool.kv_buffer[layer_id - pool.start_layer])
        self._megakernel_aux_caches = caches
        return caches

    def _replay_aux_overlapped(self, graph: torch.cuda.CUDAGraph) -> None:
        """Replay the aux graph on the side stream; ``_join_aux`` joins it back."""
        current = torch.cuda.current_stream(self.device)
        if self._aux_stream is None:
            self._aux_stream = torch.cuda.Stream(device=self.device)
            self._aux_done = torch.cuda.Event()
        self._aux_stream.wait_stream(current)
        with torch.cuda.stream(self._aux_stream):
            graph.replay()
        self._aux_done.record(self._aux_stream)
        self._aux_pending = True

    def _join_aux(self) -> None:
        if self._aux_pending:
            torch.cuda.current_stream(self.device).wait_event(self._aux_done)
            self._aux_pending = False

    def _append_target_hidden_sequential(
        self,
        ctx_hidden: torch.Tensor,
        ctx_positions: torch.Tensor,
        ctx_cache_loc: torch.Tensor,
        ctx_swa_cache_loc: Optional[torch.Tensor] = None,
    ) -> None:
        for layer in self.draft_model.layers:
            attn = layer.self_attn
            layer_ctx_hidden = self.draft_model.prepare_context_hidden_for_kv(
                layer, ctx_hidden
            )
            if isinstance(attn, DFlashMLAAttention):
                attn.write_context_kv(
                    pool=self.draft_model_runner.token_to_kv_pool,
                    positions=ctx_positions,
                    hidden_states=layer_ctx_hidden,
                    cache_loc=ctx_cache_loc,
                    swa_cache_loc=ctx_swa_cache_loc,
                )
                continue
            if _is_npu:
                _, k, v = attn.forward_prepare_npu(ctx_positions, layer_ctx_hidden)
                # Keep V scaling consistent with forward().
                if attn.v_scale is not None:
                    v = v * attn.v_scale
            else:
                k, v = attn.kv_proj_only(layer_ctx_hidden)
                k = attn.apply_k_norm(k)
                k = attn.apply_k_rope(ctx_positions, k)
            k = k.view(-1, attn.num_kv_heads, attn.head_dim)
            v = v.view(-1, attn.num_kv_heads, attn.head_dim)
            self.draft_model_runner.token_to_kv_pool.layer_router.layer(
                attn.attn.layer_id
            ).store(
                attn.attn,
                KVWriteLoc(ctx_cache_loc, swa_loc=ctx_swa_cache_loc, physical=True),
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
        ctx_swa_cache_loc: Optional[torch.Tensor] = None,
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
                token_to_kv_pool.layer_router.layer(attn.layer_id).store_prefix(
                    attn,
                    KVWriteLoc(
                        ctx_cache_loc_2d,
                        swa_loc=ctx_swa_cache_loc.view_as(ctx_cache_loc_2d)
                        if ctx_swa_cache_loc is not None
                        else None,
                        physical=True,
                    ),
                    commit_lens,
                    cache_k,
                    cache_v,
                    attn.k_scale,
                    attn.v_scale,
                )
            else:
                token_to_kv_pool.layer_router.layer(attn.layer_id).store(
                    attn,
                    KVWriteLoc(ctx_cache_loc, swa_loc=ctx_swa_cache_loc, physical=True),
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
        seq_lens_post_verify: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> None:
        """Commit Mamba intermediate states for accepted verify steps.

        During TARGET_VERIFY, Mamba kernels run with `disable_state_update=True` and
        cache per-step intermediate states. After acceptance, we need to commit the
        state corresponding to each request's last accepted step.
        """
        if not self._need_mamba_verify_commit:
            return
        if self._megakernel_kda_commit(
            batch=batch,
            seq_lens_pre_verify=seq_lens_pre_verify,
            commit_lens=commit_lens,
        ):
            return
        attn_backend = self.target_worker.model_runner.attn_backend

        mamba_steps_to_track = None
        if self._use_commit_metadata:
            bs = commit_lens.numel()
            if (
                self._mamba_commit_buffers is None
                or self._mamba_commit_buffers[0].numel() < bs
            ):
                self._mamba_commit_buffers = tuple(
                    torch.empty_like(commit_lens, dtype=torch.int64) for _ in range(2)
                )
            last_buf, track_buf = self._mamba_commit_buffers
            last_correct_step_indices = last_buf[:bs]
            has_track = batch.mamba_track_indices is not None
            _prepare_dflash_mamba_commit(
                seq_lens_pre_verify,
                commit_lens,
                last_correct_step_indices,
                track_buf[:bs],
                mamba_track_grid(batch.tree_cache.page_size) if has_track else 1,
                has_track,
            )
            if has_track:
                mamba_steps_to_track = track_buf[:bs]
        else:
            last_correct_step_indices = commit_lens.to(torch.int64) - 1

        if not self._use_commit_metadata and batch.mamba_track_indices is not None:
            mamba_track_interval = mamba_track_grid(batch.tree_cache.page_size)
            to_track_mask = (
                seq_lens_pre_verify // mamba_track_interval
                != seq_lens_post_verify // mamba_track_interval
            )
            tracking_point = (
                seq_lens_post_verify // mamba_track_interval * mamba_track_interval
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

    def _draft_block_embeds(
        self, *, block_ids: torch.Tensor, embed_module, bs: int
    ) -> Optional[torch.Tensor]:
        """Block input embeds [bs * block, hidden]; None if the megakernel embeds."""
        embed = self._megakernel_embed
        if embed is not None and bs in embed.draft_embed_requests:
            return None
        if self._full_embed_gpu is not None:
            # A borrowed embedding may use target-wide collectives, while
            # other DP replicas are idle or looking up different token IDs.
            noise_embedding = torch.nn.functional.embedding(
                block_ids, self._full_embed_gpu
            )
        else:
            noise_embedding = embed_module(block_ids)
        if self._noise_embed_scale != 1.0:
            noise_embedding = noise_embedding * self._noise_embed_scale
        return noise_embedding.view(-1, noise_embedding.shape[-1])

    def _maybe_attach_megakernel_accept(self):
        """The megakernel's one-launch greedy verify accept, or None for SGLang's."""
        if self.draft_model is None:
            return None
        accept = megakernel_draft_accept(self.draft_model)
        if accept is None or self._plain_deferral_refused("megakernel accept"):
            return None
        accept.attach_draft_accept(
            sim_real_token=SIMULATE_ACC_TOKEN_MODE == "real-draft-token"
        )
        self._arm_plain_deferral("megakernel accept")
        self._sim_ring = SimAccLenRing(self.device)
        self._sim_off = torch.zeros(1, dtype=torch.int32, device=self.device)
        self._accept_target_buf: Optional[torch.Tensor] = None
        if self._target_tp_rank == 0:
            logger.info(
                "DFLASH greedy verify accept runs on the NVIDIA megakernel for "
                "requests %s",
                list(accept.draft_accept_requests),
            )
        return accept

    def _ring_declines(self, feature: str) -> bool:
        """Off under the draft KV ring, which replaces the per-slot draft table."""
        if not draft_kv_ring_enabled():
            return False
        if self._target_tp_rank == 0:
            logger.info(
                "DFLASH megakernel %s off: the draft KV ring keeps no per-slot draft "
                "table (pass --speculative-draft-window-size)",
                feature,
            )
        return True

    def _attach_megakernel_aux(self):
        """The megakernel's draft aux append, or None (also under the draft KV ring)."""
        if not self._use_aux_graph or self.draft_model is None:
            return None
        aux = megakernel_draft_aux(self.draft_model)
        if aux is not None and self._ring_declines("draft aux"):
            return None
        return aux

    def _tail_graph_enabled(self) -> bool:
        """SGLANG_OPT_DFLASH_MEGAKERNEL_TAIL_GRAPH, off under the draft KV ring."""
        if not envs.SGLANG_OPT_DFLASH_MEGAKERNEL_TAIL_GRAPH.get():
            return False
        ring = envs.SGLANG_DFLASH_DRAFT_KV_RING
        if draft_kv_ring_enabled() and ring.is_set():
            raise ValueError(
                "SGLANG_OPT_DFLASH_MEGAKERNEL_TAIL_GRAPH=1 with SGLANG_DFLASH_DRAFT_KV_RING=1: "
                "the megakernel tail writes the per-slot draft table the ring replaces; "
                "pass --speculative-draft-window-size or unset one of them."
            )
        return not self._ring_declines("post-verify tail")

    def _warn_megakernel_draft_chain_under_ring(self) -> None:
        if (
            draft_kv_ring_enabled()
            and self.draft_model is not None
            and built_megakernel(self.draft_model) is not None
            and self._target_tp_rank == 0
        ):
            logger.warning(
                "DFLASH megakernel draft chain (M1a / M1b) with the draft KV ring has not "
                "been validated; it writes at the ring's out_cache_loc rows"
            )

    def _plain_deferral_refused(self, consumer: str) -> bool:
        """Whether SGLANG_DFLASH_DEFER_PLAIN_GATHER=0 keeps the full verify gather."""
        target_lp = self._target_logits_processor
        plain = envs.SGLANG_DFLASH_DEFER_PLAIN_GATHER
        refused = (
            target_lp is not None
            and not target_lp.final_logit_softcapping
            and plain.is_set()
            and not plain.get()
        )
        if refused and self._target_tp_rank == 0:
            logger.info(
                "DFLASH %s off: SGLANG_DFLASH_DEFER_PLAIN_GATHER=0 keeps the full "
                "verify gather",
                consumer,
            )
        return refused

    def _arm_plain_deferral(self, consumer: str) -> None:
        """Turn on the plain verify deferral; must run before the graphs capture."""
        target_lp = self._target_logits_processor
        if (
            target_lp is None
            or target_lp.final_logit_softcapping
            or target_lp.defer_plain_gather
        ):
            return
        target_lp.defer_plain_gather = True
        if self._target_tp_rank == 0:
            logger.info(
                "DFLASH plain verify deferral on (the %s reads the shard)", consumer
            )

    def _tail_covered(self, bs: int) -> bool:
        """Whether the captured megakernel tail can replace this round's steps."""
        accept = self._megakernel_accept
        return (
            self._tail_graph
            and accept is not None
            and bs in accept.draft_accept_requests
            and self._megakernel_kda is not None
            and self._kda_confirmed
            and self._need_mamba_verify_commit
        )

    def _tail_appends_aux(self, bs: int) -> bool:
        """Whether the tail also does the context-KV append (aux covers these rows)."""
        aux = self._megakernel_aux
        return aux is not None and bs * int(self.block_size) in aux.draft_aux_rows

    def _take_accept_parity(self) -> int:
        """Next ``draft_accept`` parity slot (alternates 0 / 1)."""
        slot, self._accept_parity = self._accept_parity, self._accept_parity ^ 1
        return slot

    def _take_aux_parity(self) -> int:
        """Next ``draft_aux`` parity slot (alternates 0 / 1)."""
        slot, self._aux_parity = self._aux_parity, self._aux_parity ^ 1
        return slot

    def _tail_on_megakernel(
        self,
        *,
        target_lp,
        logits: torch.Tensor,
        hidden: torch.Tensor,
        candidates: torch.Tensor,
        prefix_lens: torch.Tensor,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        batch: ScheduleBatch,
        bs: int,
    ):
        """Run or replay the megakernel post-verify tail; None keeps SGLang's steps."""
        with_aux = self._tail_appends_aux(bs)
        caches = self._draft_mla_caches() if with_aux else []
        sim_buf = self._sim_ring.device_len if SIMULATE_ACC_LEN > 0 else self._sim_off
        # Keyed by input addresses: each target graph variant has its own buffers.
        key = (
            bs,
            tuple(t.data_ptr() for t in (logits, hidden, candidates, sim_buf, *caches)),
        )
        tail = self._megakernel_tails.get(key)
        first = tail is None
        if first:
            if sum(k[0] == bs for k in self._megakernel_tails) >= _TAIL_VARIANTS_PER_BS:
                if not self._tail_variants_warned and self._target_tp_rank == 0:
                    logger.warning(
                        "DFLASH megakernel tail: bs=%d inputs keep moving; this round "
                        "keeps the steps",
                        bs,
                    )
                self._tail_variants_warned = True
                return None
            tail = _MegakernelTail(
                bs=bs, block_size=int(self.block_size), device=self.device
            )
        kda = self._kda_commit_kwargs(
            batch=batch, bs=bs, slots=tail.kda_slots, track=tail.kda_track
        )
        if kda is None:
            return None
        state = tuple(t.data_ptr() for t in kda.values() if isinstance(t, torch.Tensor))
        if not first and state != tail.inputs:
            raise RuntimeError("DFLASH megakernel tail: the captured KDA state moved")
        tail.prefix.copy_(prefix_lens)
        tail.positions.copy_(positions)
        tail.slots.copy_(cache_loc)
        block_size = int(self.block_size)
        sim_commit = (
            self._sim_ring.stage(
                sample_simulated_acc_len(
                    SIMULATE_ACC_LEN,
                    SIMULATE_ACC_METHOD,
                    block_size,
                    generator=self._simulated_acc_len_rng,
                )
            )
            if SIMULATE_ACC_LEN > 0
            else self._sim_off
        )

        def body(accept_slot: int, aux_slot: Optional[int]) -> bool:
            out = tail.outs[accept_slot]
            self._megakernel_accept.draft_accept(
                local=target_lp.deferred_local_argmax(logits),
                candidates=candidates,
                prefix_lens=tail.prefix,
                sim_commit=sim_commit,
                out=out,
                parity=accept_slot,
            )
            used = self._megakernel_kda.kda_commit(
                **kda, commit_lens=out[2], prefix_lens=tail.prefix
            )
            if aux_slot is not None:
                self._megakernel_aux.draft_aux(
                    target_hidden=hidden,
                    positions=tail.positions,
                    slots_2d=tail.slots.view(bs, block_size),
                    commit_lens=out[2],
                    kv_caches=caches,
                    parity=aux_slot,
                )
            return used

        slots = (
            self._take_accept_parity(),
            self._take_aux_parity() if with_aux else None,
        )
        if first:
            # First round eager, doubling as warm-up: the KDA commit must not run twice.
            if not body(*slots):
                raise RuntimeError("DFLASH megakernel tail: the KDA commit turned off")
            tail.inputs = state
            self._megakernel_tails[key] = tail
        else:
            graph = tail.graphs.get(slots)
            if graph is None:
                graph = self._capture_tail(tail, body, slots)
            graph.replay()
        out = tail.outs[slots[0]]
        self._tp_sync.sync(SpecTpSyncSite.DFLASH_ACCEPT_GREEDY, out[0])
        return out

    def _capture_tail(self, tail: _MegakernelTail, body, slots) -> torch.cuda.CUDAGraph:
        """Capture the tail for one (accept slot, aux slot) pair in the tail's pool."""
        stream = torch.cuda.Stream(device=self.device)
        current = torch.cuda.current_stream(self.device)
        stream.wait_stream(current)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=tail.pool, stream=stream):
            body(*slots)
        current.wait_stream(stream)
        tail.pool, tail.graphs[slots] = graph.pool(), graph
        if self._target_tp_rank == 0:
            logger.info(
                "DFLASH post-verify tail captured on the NVIDIA megakernel "
                "(accept + KDA commit%s; bs=%d, accept slot %d, aux slot %s)",
                " + aux" if slots[1] is not None else "",
                tail.prefix.shape[0],
                *slots,
            )
        return graph

    def _accept_on_megakernel(
        self,
        *,
        local: torch.Tensor,
        candidates: torch.Tensor,
        prefix_lens: torch.Tensor,
        bs: int,
    ):
        """Greedy accept on the megakernel from this rank's row argmax."""
        accept_len, commit_lens, bonus, out_tokens, new_seq_lens = (
            self._next_accept_bonus_buffers(bs)
        )
        block_size = int(self.block_size)
        if self._accept_target_buf is None or self._accept_target_buf.shape[0] < bs:
            self._accept_target_buf = torch.empty(
                (bs, block_size), dtype=torch.int64, device=self.device
            )
        target_predict = self._accept_target_buf[:bs]
        # One host draw per round, same generator as apply_dflash_simulated_acceptance.
        sim_commit = (
            self._sim_ring.stage(
                sample_simulated_acc_len(
                    SIMULATE_ACC_LEN,
                    SIMULATE_ACC_METHOD,
                    block_size,
                    generator=self._simulated_acc_len_rng,
                )
            )
            if SIMULATE_ACC_LEN > 0
            else self._sim_off
        )
        self._megakernel_accept.draft_accept(
            local=local,
            candidates=candidates,
            prefix_lens=prefix_lens,
            sim_commit=sim_commit,
            out=(
                target_predict,
                accept_len,
                commit_lens,
                bonus,
                out_tokens,
                new_seq_lens,
            ),
            parity=self._take_accept_parity(),
        )
        self._tp_sync.sync(SpecTpSyncSite.DFLASH_ACCEPT_GREEDY, target_predict)
        return target_predict, accept_len, commit_lens, bonus, out_tokens, new_seq_lens

    def _megakernel_kda_commit(
        self,
        *,
        batch: ScheduleBatch,
        seq_lens_pre_verify: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> bool:
        """Run the KDA commit on the target megakernel; False keeps SGLang's commit."""
        if self._megakernel_kda is None:
            return False
        bs = int(commit_lens.shape[0])
        slots, track, commit = self._kda_commit_inputs(bs)
        kwargs = self._kda_commit_kwargs(batch=batch, bs=bs, slots=slots, track=track)
        if kwargs is None:
            return False
        commit.copy_(commit_lens)
        used = self._megakernel_kda.kda_commit(
            **kwargs,
            commit_lens=commit,
            prefix_lens=seq_lens_pre_verify.to(torch.int64),
        )
        if not used:
            # The switch is fixed at build: skip these copies from now on.
            self._megakernel_kda = None
        self._kda_confirmed = used
        return used

    def _kda_commit_kwargs(
        self,
        *,
        batch: ScheduleBatch,
        bs: int,
        slots: torch.Tensor,
        track: torch.Tensor,
    ) -> Optional[dict]:
        """The KDA commit kwargs, or None where SGLang's commit takes another branch."""
        backend = _hybrid_linear_backend(self.target_worker.model_runner.attn_backend)
        if backend is None or pp_spec_stable_rows_enabled():
            return None
        linear = backend.linear_attn_backend
        req_pool = linear.req_to_token_pool
        # Host-only checkpoints index the KDA ring by req_pool_idx; this commit
        # writes it through the mamba slot, so SGLang's commit takes over.
        if (
            not req_pool.mamba_pool.replayssm_is_kda
            or req_pool.mamba_pool.replayssm_request_ring
        ):
            return None
        spec_state = linear.state_pool.get_speculative_mamba2_params_all_layers()
        if len(spec_state.conv) != 1:
            return None
        slots.copy_(linear.forward_metadata.mamba_cache_indices[:bs])
        if batch.mamba_track_indices is None:
            track.fill_(-1)
        else:
            track.copy_(linear._translate_mamba_indices(batch.mamba_track_indices))
        return dict(
            temporal=spec_state.temporal,
            rawk=spec_state.replayssm_rawk,
            rawv=spec_state.replayssm_rawv,
            gk=spec_state.replayssm_g,
            beta=spec_state.replayssm_beta,
            conv=spec_state.conv[0],
            window=spec_state.intermediate_conv_window[0],
            slots=slots,
            track_slots=track,
            track_interval=mamba_track_grid(batch.tree_cache.page_size),
        )

    def _kda_commit_inputs(self, bs: int) -> Tuple[torch.Tensor, ...]:
        """Static int32 slots / track slots / commit lengths for the KDA commit."""
        if self._kda_commit_buffers is None or self._kda_commit_buffers[0].numel() < bs:
            self._kda_commit_buffers = tuple(
                torch.empty(bs, dtype=torch.int32, device=self.device) for _ in range(3)
            )
        return tuple(buf[:bs] for buf in self._kda_commit_buffers)

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

    def _next_accept_bonus_buffers(
        self, bs: int
    ) -> tuple[
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

    def _accept_block(
        self,
        *,
        candidates: torch.Tensor,
        next_token_logits: torch.Tensor,
        sampling_info,
        draft_input,
        prefix_lens: torch.Tensor,
        bs: int,
        target_predict: Optional[torch.Tensor] = None,
        verify_shard: Optional[torch.Tensor] = None,
    ):
        new_seq_lens = None
        if verify_shard is not None:
            # Samples on the vocab shards; every rank gets rank 0's decision.
            selector_candidate_ids, selector_q_rows = self._selector_sample
            accept_len, bonus, target_predict = self._megakernel_sampled.sampled_accept(
                logits=verify_shard,
                temperatures=sampling_info.temperatures,
                candidates=candidates,
                candidate_ids=selector_candidate_ids,
                q_rows=selector_q_rows,
            )
            out_tokens, commit_lens = _commit_accept(candidates, accept_len, bonus)
        elif self._selector_sample is not None:
            selector_candidate_ids, selector_q_rows = self._selector_sample
            accept_len, bonus = self._selector_sampling_accept(
                candidates=candidates,
                next_token_logits=next_token_logits,
                candidate_ids=selector_candidate_ids,
                q_rows=selector_q_rows,
                sampling_info=sampling_info,
                draft_input=draft_input,
            )
            # Rank 0's decision to all ranks: one megakernel hop (rank-uniform branch),
            # else the two SGLANG_SPEC_TP_SYNC broadcasts.
            bcast = self._megakernel_bcast if megakernel_enabled() else None
            if bcast is None or not bcast.rank0_broadcast(accept_len, bonus):
                self._tp_sync.sync(SpecTpSyncSite.DFLASH_SELECTOR, accept_len)
                self._tp_sync.sync(SpecTpSyncSite.DFLASH_SELECTOR, bonus)
            out_tokens, commit_lens = _commit_accept(candidates, accept_len, bonus)
        elif target_predict is None and (
            not _is_all_greedy(sampling_info) and is_dflash_sampling_verify_available()
        ):
            accept_len, bonus = compute_dflash_sampling_correct_drafts_and_bonus(
                candidates=candidates,
                next_token_logits=next_token_logits,
                sampling_info=sampling_info,
                max_top_k=draft_input.max_top_k,
                uniform_top_k_value=draft_input.uniform_top_k_value,
            )
            self._tp_sync.sync(SpecTpSyncSite.DFLASH_ACCEPT_SAMPLE, accept_len)
            self._tp_sync.sync(SpecTpSyncSite.DFLASH_ACCEPT_SAMPLE, bonus)
            out_tokens, commit_lens = _commit_accept(candidates, accept_len, bonus)
        else:
            if target_predict is None:
                target_predict = torch.argmax(next_token_logits, dim=-1)
            target_predict = target_predict.view(bs, int(self.block_size))
            self._tp_sync.sync(SpecTpSyncSite.DFLASH_ACCEPT_GREEDY, target_predict)
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
                next_token_logits=next_token_logits,
                generator=self._simulated_acc_len_rng,
            )
            # The Triton path may have written new_seq_lens from the real
            # accept_len; recompute it from the forced commit_lens.
            new_seq_lens = None

        return accept_len, commit_lens, bonus, out_tokens, new_seq_lens

    def _validate_phase1_sampling_support(self, batch: ScheduleBatch) -> None:
        sampling_info = batch.sampling_info
        if sampling_info is None or sampling_info.is_all_greedy:
            return

        if self.selector is not None:
            if self._selector_sampling_enabled:
                return
            if not self._warned_sampling_fallback and get_parallel().tp_rank == 0:
                logger.warning(
                    "DFLASH non-greedy verification is unavailable on this "
                    "build/device; falling back to greedy argmax verification. "
                    "The requested sampling distribution will not be preserved; "
                    "use temperature=0 and top_k=1 for lossless greedy decoding."
                )
                self._warned_sampling_fallback = True
            return

        # The sampled head hands verify the q it drew from, so no fallback is needed.
        if self.lilicorr is not None and self._lilicorr_sampling_enabled:
            return

        if (
            not is_dflash_sampling_verify_available()
            and not self._warned_sampling_fallback
            and get_parallel().tp_rank == 0
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
        if not self._hosts_draft:
            # An earlier pipeline stage of a PD prefill: the target forward,
            # whose captured aux hidden states ride the PP proxy onward.
            batch_output = self.target_worker.forward_batch_generation(
                batch,
                pp_proxy_tensors=pp_proxy_tensors,
                capture_hidden_mode=CaptureHiddenMode.FULL,
            )
            batch_output.new_seq_lens = batch.seq_lens
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)
            return batch_output
        self._validate_phase1_sampling_support(batch)
        if (
            batch.forward_mode.is_extend()
            or batch.is_extend_in_batch
            or batch.forward_mode.is_idle()
        ):
            # A pending aux append may own draft KV rows this forward reuses; only the
            # decode draft prep (no KV writes) runs beside it, up to the draft forward.
            self._join_aux()

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            self._wait_for_fused_hicache_load(batch)
            # Target prefill: capture DFlash aux hidden states for prompt tokens.
            batch_output = self.target_worker.forward_batch_generation(
                batch,
                pp_proxy_tensors=pp_proxy_tensors,
                capture_hidden_mode=CaptureHiddenMode.FULL,
                return_iteration_ids=True,
            )
            iteration_ids, batch_output.iteration_ids = batch_output.iteration_ids, None

            logits_output, next_token_ids = (
                batch_output.logits_output,
                batch_output.next_token_ids,
            )
            self._tp_sync.sync(SpecTpSyncSite.DFLASH_TARGET, next_token_ids)
            new_seq_lens = batch.seq_lens
            batch_output.new_seq_lens = new_seq_lens
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)

            # An idle DP rank runs the empty target prefill above to stay in
            # the DP collective, but must skip the draft KV materialization,
            # which needs per-request extend info.
            if batch.forward_mode.is_idle():
                batch_output.next_draft_input = DFlashDraftInputV2.create_idle_input(
                    device=self.device
                )
                return batch_output

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
            # Enqueue metadata copies without waiting for the target prefill's
            # outstanding GPU work; the same stream orders their consumers.
            ctx_lens = async_h2d(batch.extend_lens, dtype=torch.int32, device=device)
            draft_seq_lens = async_h2d(
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
            if self.draft_kv_ring is not None:
                self.draft_kv_ring.record_floors(batch.reqs, batch.prefix_lens)
                draft_cache_loc = self.draft_kv_ring.extend_locs(
                    req_pool_indices=batch.req_pool_indices,
                    positions=positions,
                    extend_lens=ctx_lens,
                    seq_ends=draft_seq_lens + ctx_lens,
                )
            else:
                # The draft KV goes to the slots the target prefill just wrote.
                draft_cache_loc = self._draft_write_ids(iteration_ids)
            self._append_target_hidden_to_draft_kv_by_loc(
                target_hidden=logits_output.hidden_states,
                cache_loc=draft_cache_loc,
                swa_cache_loc=self._draft_write_ids(
                    iteration_ids, IdSpaceKind.SLIDING_WINDOW
                ),
                positions=positions,
                extend_lens=ctx_lens,
            )

            # Avoid copying large hidden-state buffers to CPU in overlap scheduling.
            logits_output.hidden_states = None

            batch_output.next_draft_input = self._make_next_draft_input_prefill(
                bonus_tokens=next_token_ids,
                seq_lens=new_seq_lens,
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
            # Under dp attention an idle DP rank must still run the target
            # verify forward (IDLE mode) so its cross-DP collectives stay in
            # lockstep with the active DP group; the draft block's
            # collectives are within-rank and skipped.
            idle_can_run_cuda_graph = False
            if get_parallel().attn_dp_enabled:
                idle_verify_input = DFlashVerifyInput(
                    draft_token=torch.empty((0,), dtype=torch.long, device=self.device),
                    positions=torch.empty((0,), dtype=torch.int64, device=self.device),
                    draft_token_num=int(self.block_size),
                    custom_mask=None,
                    capture_hidden_mode=CaptureHiddenMode.FULL,
                )
                # Deferred like the active ranks' verify under DP attention:
                # ModelRunner plans after DP padding has fixed the shapes.
                idle_verify_forward_batch, idle_can_run_cuda_graph = (
                    idle_verify_input.prepare_for_verify(
                        batch, self._target_worker, defer_metadata_init=True
                    )
                )
                if _is_npu:
                    # Force eager: NPU active ranks run eager verify when any
                    # DP rank is idle (see the idle-guard below); a
                    # graph-replaying idle rank would disagree with them on
                    # DP-gather segment offsets (padded bucket vs raw counts).
                    idle_verify_forward_batch.can_run_decode_cuda_graph = False
                    idle_can_run_cuda_graph = False
                # On CUDA the idle rank keeps prepare_for_verify's graph
                # decision: the bucket is max(original_global_num_tokens_cpu)
                # on every rank, so it replays the active ranks' graph with
                # zero real rows (raw_bs=0) and the same [bucket*w]*ndp
                # DP-gather layout, like plain decode and EAGLE idle ranks.
                self._target_worker.forward_batch_generation(
                    batch=None,
                    forward_batch=idle_verify_forward_batch,
                    is_verify=True,
                    skip_attn_backend_init=True if _is_npu else None,
                )
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
                can_run_cuda_graph=idle_can_run_cuda_graph,
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
        embed_module = unwrap_lora_layer(
            _resolve_dflash_embedding_module(self.draft_model, target_model)
        )
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

        input_embeds = self._draft_block_embeds(
            block_ids=block_ids, embed_module=embed_module, bs=bs
        )

        positions = positions_2d.reshape(-1)
        verify_out_cache_loc = verify_out_cache_loc_2d.reshape(-1)
        # The iteration's window -- the draft block, the verify and the
        # target-hidden KV writes all write it -- planned once.
        iteration_ids = self.target_worker.model_runner.runner_ids.plan(
            req_pool_indices=batch.req_pool_indices,
            seq_lens=prefix_lens,
            seq_lens_cpu=(
                batch.seq_lens_cpu
                if batch.seq_lens_cpu is not None
                else draft_input.nxt_kv_lens_cpu
            ),
            write_virtual=verify_out_cache_loc,
            read_extent=block_size,
        )

        seq_lens_cpu = self._draft_seq_lens_cpu_buf[:bs]
        if self.use_compact_draft_cache:
            # Rebuild the draft-local sliding-window view from committed target state.
            draft_prefix_lens = self._compute_compact_draft_seq_lens(prefix_lens)
            self._fill_compact_seq_lens_cpu_bound(
                batch_seq_lens_cpu=batch.seq_lens_cpu,
                nxt_kv_lens_cpu=draft_input.nxt_kv_lens_cpu,
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
                # Allocated lengths include the in-flight verify reservation.
                for i, req in enumerate(batch.reqs):
                    seq_lens_cpu[i] = req.kv.kv_allocated_len
                draft_seq_lens_sum = int(seq_lens_cpu.sum())

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
            num_token_non_padded=(
                torch.full((), bs * block_size, dtype=torch.int32, device=device)
                if enable_num_token_non_padded()
                else None
            ),
            global_num_token_non_padded_cpu=bs * block_size,
            can_run_decode_cuda_graph=batch.can_run_decode_cuda_graph,
        )
        # Graph eligibility follows the target batch (the scheduler's DP vote),
        # as for the DSpark draft; the draft keys its graph by local batch size.
        forward_batch.can_run_decode_cuda_graph = batch.can_run_decode_cuda_graph
        # The draft block writes the whole window. A compact draft reads its
        # own `req_to_token` rows over its own lengths.
        draft_translator = self.draft_model_runner.runner_ids
        draft_plan = (
            iteration_ids.reads_from(
                draft_translator,
                seq_lens=draft_seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                read_extent=block_size,
            )
            if self.use_compact_draft_cache
            else iteration_ids
        )
        draft_plan.bind(forward_batch, draft_translator)
        ring_block_locs = None
        if self.draft_kv_ring is not None:
            ring_block_locs = self.draft_kv_ring.block_locs(
                batch.req_pool_indices, positions, block_size
            )
            forward_batch.out_cache_loc = ring_block_locs
            forward_batch.out_cache_loc_virtual = None

        if self.selector is not None or self.lilicorr is not None:
            self._selector_sample = None
            if self._draft_sampler is not None:
                # Consumed by the in-graph sample; must be staged before the replay.
                self._draft_sampler.stage_sampling_params(
                    bs=bs, sampling_info=batch.sampling_info
                )

        self._join_aux()
        with (
            torch.inference_mode(),
            draft_tp_context(self.draft_parallel_layout),
        ):
            draft_out = self.draft_model_runner.forward(forward_batch)
        draft_logits_output = draft_out.logits_output

        if (
            self._is_domino
            and self._draft_sampler is not None
            and draft_out.can_run_graph
        ):
            draft_next = self._draft_sampler.out[
                : bs * (int(self.block_size) - 1)
            ].view(bs, int(self.block_size) - 1)
        elif self._is_domino:
            draft_hidden = draft_logits_output.hidden_states
            if draft_hidden is None:
                raise RuntimeError("DFLASH draft model returned no hidden states.")
            draft_hidden = draft_hidden.view(bs, int(self.block_size), -1)
            prefix_gru = self.draft_model.prefix_gru
            embed_proj = self.draft_model.embed_proj
            if prefix_gru is None or embed_proj is None:
                raise RuntimeError("DFLASH Domino projector modules are unavailable.")
            tp_group = get_parallel().tp_group
            shard = getattr(lm_head, "shard_indices", None)
            draft_next = domino_greedy_rollout(
                draft_hidden=draft_hidden,
                bonus_tokens=block_ids[:, 0],
                target_embedding=embed_module,
                lm_head_weight=lm_head.weight,
                prefix_gru=prefix_gru,
                embed_proj=embed_proj,
                vocab_size=int(self.model_runner.model_config.vocab_size),
                shift_label=bool(self.draft_model.shift_label),
                candidate_pool_size=self.domino_candidate_pool_size,
                tp_group=tp_group,
                lm_head_org_vocab_start=(
                    int(shard.org_vocab_start_index) if shard is not None else 0
                ),
                lm_head_num_org=(
                    int(shard.num_org_elements) if shard is not None else None
                ),
                lm_head_num_org_padded=(
                    int(shard.num_org_elements_padded) if shard is not None else None
                ),
                in_cuda_graph=False,
            )
        elif (
            self._draft_sampler is not None
            and draft_out.can_run_graph
            and not self._graph_proposal_unusable(batch.sampling_info)
        ):
            draft_next = self._draft_sampler.out[
                : bs * (int(self.block_size) - 1)
            ].view(bs, int(self.block_size) - 1)
            if (
                self.selector is not None
                and not _is_all_greedy(batch.sampling_info)
                and self._selector_sampling_enabled
            ):
                self._selector_sample = (
                    self._draft_sampler.candidate_out[:bs],
                    self._draft_sampler.q_out[:bs],
                )
            elif (
                self.lilicorr is not None
                and self._lilicorr_sampling_enabled
                and not _is_all_greedy(batch.sampling_info)
            ):
                self._selector_sample = (
                    self._draft_sampler.candidate_out[:bs],
                    self._draft_sampler.q_out[:bs],
                )
        elif self.selector is not None:
            with draft_tp_context(self.draft_parallel_layout):
                draft_next = self._propose_selector_block(
                    draft_logits_output=draft_logits_output,
                    bs=bs,
                    lm_head=lm_head,
                    anchor_token_ids=block_ids[:, 0],
                    sampling_info=batch.sampling_info,
                )
        elif self.lilicorr is not None:
            if (
                self._lilicorr_sampling_enabled
                and not self._warned_lilicorr_eager
                and get_parallel().tp_rank == 0
            ):
                logger.warning(
                    "LiLiCorr sampled draft ran the eager head on a decode step "
                    "(draft cuda graph unavailable for batch size %d; captured "
                    "buckets do not cover it, or graphs are disabled). Acceptance is "
                    "unaffected but expect a large throughput regression, and do not "
                    "compare tokens/s from this run against a folded one.",
                    bs,
                )
                self._warned_lilicorr_eager = True
            draft_hidden = draft_logits_output.hidden_states
            if draft_hidden is None:
                raise RuntimeError("DFLASH draft model returned no hidden states.")
            with draft_tp_context(self.draft_parallel_layout):
                draft_next, lilicorr_candidate_ids, lilicorr_q_rows = (
                    propose_lilicorr_block(
                        head=self.lilicorr,
                        draft_hidden=draft_hidden.view(bs, int(self.block_size), -1),
                        lm_head=lm_head,
                        embed_tokens=target_input_embeddings(
                            self.target_worker.model_runner.model
                        ),
                        anchor=self._lilicorr_anchor,
                        sampling_info=batch.sampling_info,
                        sampling_enabled=self._lilicorr_sampling_enabled,
                    )
                )
            if lilicorr_q_rows is not None and not _is_all_greedy(batch.sampling_info):
                self._selector_sample = (lilicorr_candidate_ids, lilicorr_q_rows)
        else:
            draft_hidden = draft_logits_output.hidden_states
            if draft_hidden is None:
                raise RuntimeError("DFLASH draft model returned no hidden states.")
            draft_hidden = draft_hidden.view(bs, int(self.block_size), -1)
            with draft_tp_context(self.draft_parallel_layout):
                draft_next = self._greedy_sample_from_vocab_parallel_head(
                    hidden_states=draft_hidden[:, 1:, :].reshape(
                        -1, draft_hidden.shape[-1]
                    ),
                    lm_head=lm_head,
                ).view(bs, int(self.block_size) - 1)

        draft_tokens = self._draft_block_tokens_buf[:bs]
        if not (
            bs == 1
            and self._batched_small_copy
            and try_small_copy(
                [draft_tokens[:, 0], draft_tokens[:, 1:]], [block_ids[:, 0], draft_next]
            )
        ):
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
            iteration_ids=iteration_ids,
        )

        batch.out_cache_loc = verify_out_cache_loc
        if batch.hisparse_coordinator is not None:
            batch.hisparse_coordinator.prepare_spec_verify(batch)
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
            batch,
            self.target_worker,
            defer_metadata_init=get_parallel().attn_dp_enabled,
        )
        batch.seq_lens_cpu = seq_lens_cpu_backup
        batch.seq_lens_sum = seq_lens_sum_backup

        # Idle-DP guard (NPU): an idle rank's eager verify contributes raw
        # scheduler counts while a graph-replaying rank assumes the padded
        # bucket, so their DP-gather segment offsets disagree. Fall back to
        # eager verify whenever any DP rank is idle this round. CUDA idle
        # ranks replay the same bucket instead (see the idle branch above).
        if (
            _is_npu
            and get_parallel().attn_dp_enabled
            and verify_forward_batch.original_global_num_tokens_cpu is not None
            and min(verify_forward_batch.original_global_num_tokens_cpu) == 0
        ):
            verify_forward_batch.can_run_decode_cuda_graph = False

        # Mixed-round guard: an extend rank's raw token counts disagree with
        # the verify batch's spec-scaled counts on the DP-gather layout; run
        # eager, symmetric with the idle guard above.
        if get_parallel().attn_dp_enabled and batch.is_extend_in_batch:
            verify_forward_batch.can_run_decode_cuda_graph = False

        target_out = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=(
                None if _is_npu or get_parallel().attn_dp_enabled else True
            ),
        )
        logits_output = target_out.logits_output
        can_run_cuda_graph = target_out.can_run_cuda_graph
        target_predict = None
        accepted = None
        tail_done = False
        verify_shard = None
        target_lp = self._target_logits_processor
        if target_lp is not None and target_lp.is_deferred_shard(
            logits_output.next_token_logits
        ):
            if self._can_skip_verify_logits(batch, sampling_info):
                if self._tail_covered(bs):
                    accepted = self._tail_on_megakernel(
                        target_lp=target_lp,
                        logits=logits_output.next_token_logits,
                        hidden=logits_output.hidden_states,
                        candidates=draft_tokens,
                        prefix_lens=prefix_lens,
                        positions=positions,
                        cache_loc=ring_block_locs
                        if ring_block_locs is not None
                        else self._draft_write_ids(iteration_ids),
                        batch=batch,
                        bs=bs,
                    )
                    tail_done = accepted is not None
                if tail_done:
                    target_predict = accepted[0]
                elif (
                    self._megakernel_accept is not None
                    and bs in self._megakernel_accept.draft_accept_requests
                ):
                    accepted = self._accept_on_megakernel(
                        local=target_lp.deferred_local_argmax(
                            logits_output.next_token_logits
                        ),
                        candidates=draft_tokens,
                        prefix_lens=prefix_lens,
                        bs=bs,
                    )
                    target_predict = accepted[0]
                else:
                    target_predict = target_lp.deferred_argmax(
                        logits_output.next_token_logits
                    )
                logits_output.next_token_logits = None
            elif self._can_megakernel_sample(
                batch, sampling_info, bs, logits_output.next_token_logits
            ):
                # The megakernel samples from the shard itself (_accept_block).
                verify_shard = logits_output.next_token_logits
                logits_output.next_token_logits = None
            else:
                logits_output.next_token_logits = target_lp.finish_deferred_gather(
                    logits_output.next_token_logits
                )

        grammar_mask = None
        if batch.has_grammar:
            grammar_mask = build_grammar_vocab_mask(
                reqs=batch.reqs,
                tree=grammar_tree,
                sampling_info=batch.sampling_info,
                device=logits_output.next_token_logits.device,
                barrier=grammar_barrier,
            )

        if (
            sampling_info is not None
            and target_predict is None
            and verify_shard is None
        ):
            apply_dflash_verify_logits_adjustments(
                next_token_logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
                draft_token_num=int(self.block_size),
            )

        # Constrain every chain position before accept picks from it.
        if grammar_mask is not None:
            grammar_mask.apply(logits_output.next_token_logits)

        candidates = draft_tokens
        if accepted is not None:
            accept_len, commit_lens, bonus, out_tokens, new_seq_lens = accepted[1:]
        else:
            (
                accept_len,
                commit_lens,
                bonus,
                out_tokens,
                new_seq_lens,
            ) = self._accept_block(
                candidates=candidates,
                next_token_logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
                draft_input=draft_input,
                prefix_lens=prefix_lens,
                bs=bs,
                target_predict=target_predict,
                verify_shard=verify_shard,
            )
        if batch.hisparse_coordinator is not None:
            batch.hisparse_coordinator.commit_spec_accept_tokens(
                batch=batch,
                accept_indices=_chain_accept_indices(
                    commit_lens=commit_lens, block_size=block_size
                ),
            )

        sampling_mask_capture = SpeculativeSamplingMaskCapture.from_logits(
            sampling_info,
            next_token_logits=logits_output.next_token_logits,
            draft_input=draft_input,
            draft_token_num=int(self.block_size),
            bs=bs,
        )
        if sampling_mask_capture is not None:
            logits_output.sampling_mask_output = sampling_mask_capture.build_output(
                out_tokens=out_tokens,
                commit_lens=commit_lens,
            )

        if batch.return_logprob:
            compute_spec_logprobs(
                batch,
                logits_output,
                out_tokens.reshape(-1),
                chain_stride=block_size,
            )

        if self._need_mamba_verify_commit and not tail_done:
            assert seq_lens_pre_verify is not None
            if new_seq_lens is None:
                new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
            self._update_target_mamba_state_after_verify(
                batch=batch,
                seq_lens_pre_verify=seq_lens_pre_verify,
                seq_lens_post_verify=new_seq_lens,
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

        # Consume in this step: every decode graph size shares one aux output,
        # which the next target forward overwrites (resolve_aux_hidden_states_width).
        if not (tail_done and self._tail_appends_aux(bs)):
            cache_loc = (
                ring_block_locs
                if ring_block_locs is not None
                else self._draft_write_ids(iteration_ids)
            )
            self._append_target_hidden_decode(
                target_hidden=hidden.reshape(-1, hidden.shape[-1]),
                cache_loc=cache_loc,
                cache_loc_2d=cache_loc.view(verify_out_cache_loc_2d.shape),
                positions=positions,
                commit_lens=commit_lens,
                swa_cache_loc=self._draft_write_ids(
                    iteration_ids, IdSpaceKind.SLIDING_WINDOW
                ),
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
            side_stream_done=self._aux_done if self._aux_pending else None,
            speculative_num_draft_tokens=int(self.block_size),
            # The non-overlap (sync) scheduler path advances batch.seq_lens
            # from the result; overlap carries it via next_draft_input instead.
            new_seq_lens=new_seq_lens,
            routed_experts_output=target_out.routed_experts_output,
            indexer_topk_output=target_out.indexer_topk_output,
            kv_weight_version_record=target_out.kv_weight_version_record,
        )
