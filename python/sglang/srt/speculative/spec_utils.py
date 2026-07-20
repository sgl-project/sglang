from __future__ import annotations

import contextlib
import logging
import os
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Tuple

import torch
from huggingface_hub import snapshot_download

from sglang.kernels.ops.speculative.cache_locs import (
    align_evict_mask_to_page_size as align_evict_mask_to_page_size,
)
from sglang.kernels.ops.speculative.cache_locs import (
    assign_extend_cache_locs as assign_extend_cache_locs,
)
from sglang.kernels.ops.speculative.cache_locs import (
    filter_finished_cache_loc_kernel as filter_finished_cache_loc_kernel,
)
from sglang.kernels.ops.speculative.cache_locs import (
    generate_draft_decode_kv_indices as generate_draft_decode_kv_indices,
)
from sglang.kernels.ops.speculative.cache_locs import (
    get_src_tgt_cache_loc as get_src_tgt_cache_loc,
)
from sglang.kernels.ops.speculative.cache_locs import (
    get_target_cache_loc as get_target_cache_loc,
)
from sglang.kernels.ops.speculative.eagle import (
    fill_accept_out_cache_loc_func as fill_accept_out_cache_loc_func,
)
from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.distributed.parallel_state import (
    GroupCoordinator,
    patch_tensor_parallel_group,
)
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import set_mamba_track_indices_from_reqs
from sglang.srt.mem_cache.allocation import (
    assign_req_to_token_pool as assign_req_to_token_pool,
)
from sglang.srt.mem_cache.allocation import (
    assign_req_to_token_pool_func as assign_req_to_token_pool_func,
)
from sglang.srt.runtime_context import get_server_args
from sglang.srt.utils import (
    is_cpu,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
    is_xpu,
    next_power_of_2,
)
from sglang.srt.utils.async_probe import maybe_detect_oob
from sglang.srt.utils.nvtx_utils import profile_range

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_musa = is_musa()
_is_xpu = is_xpu()
_is_cpu = is_cpu()

if TYPE_CHECKING:
    from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.eagle_info import EagleVerifyInput


if _is_cuda:
    from sgl_kernel import fast_topk
elif _is_hip:
    from sgl_kernel import fast_topk
else:
    from sglang.srt.utils.common import fast_topk

if _is_cpu:
    from sgl_kernel import assign_extend_cache_locs_cpu


logger = logging.getLogger(__name__)


def resolve_num_tokens_per_req(
    *,
    phase: Literal["draft_decode", "draft_extend", "target_verify"],
    server_args: ServerArgs,
    spec_algorithm=None,
    is_draft_worker: bool = False,
    num_draft_tokens: Optional[int] = None,
) -> int:
    """Single static derivation point for a spec phase's per-request token
    width (sizes capture shapes / buffers); the per-forward dynamic width
    lives on ``SpecInput.num_tokens_per_req``. Draft phases are
    EAGLE-family-only; "target_verify" is algorithm-generic via the hook.
    """
    if phase == "draft_decode":
        return server_args.speculative_eagle_topk
    if phase == "draft_extend":
        return server_args.speculative_num_draft_tokens
    if phase == "target_verify":
        if num_draft_tokens is None:
            num_draft_tokens = server_args.speculative_num_draft_tokens
        return spec_algorithm.get_num_tokens_per_req_for_target_verify(
            num_draft_tokens, is_draft_worker
        )
    raise ValueError(f"Unknown speculative phase: {phase}")


def fast_sample(probs: torch.Tensor, num_samples: int = 1):
    """Draw from `probs` via the Gumbel-max trick: argmax(probs / Exp(1)).

    Distributionally equivalent to torch.multinomial, but avoids multinomial's
    device-side distribution-validity assert, which the draft CUDA graph would
    otherwise capture and replay every step. q is clamped off zero so a zero
    draw can't yield inf/NaN scores that argmax would wrongly select; fp32
    avoids bf16 argmax ties biasing the draw. Set SGLANG_OPT_USE_GUMBEL_SAMPLE=0
    to fall back to torch.multinomial.
    """
    if not envs.SGLANG_OPT_USE_GUMBEL_SAMPLE.get():
        sample_index = torch.multinomial(probs, num_samples=num_samples)
        return probs.gather(1, sample_index), sample_index
    q = torch.empty_like(probs, dtype=torch.float32).exponential_(1.0)
    q.clamp_min_(torch.finfo(torch.float32).tiny)
    scores = probs.float() / q
    if num_samples == 1:
        sample_index = scores.argmax(dim=-1, keepdim=True)
    else:
        sample_index = scores.topk(num_samples, dim=-1).indices
    sample_p = probs.gather(1, sample_index)
    return sample_p, sample_index


def renorm_draft_probs(
    next_token_logits: torch.Tensor,
    sampling_info,
    use_rejection_sampling: bool,
) -> torch.Tensor:
    """Draft-side next-token distribution.

    Plain softmax, except under rejection sampling where logits are
    temperature-scaled so the draft proposal q tracks the target sampling
    temperature (higher acceptance; correctness holds for any q).
    """
    if not use_rejection_sampling or not next_token_logits.size(0):
        return torch.softmax(next_token_logits, dim=-1)
    return torch.softmax(next_token_logits / sampling_info.temperatures, dim=-1)


def sample_draft_proposal(next_token_logits: torch.Tensor, temperatures: torch.Tensor):
    """Leviathan draft proposal: q = softmax(logits / T), X ~ q.

    Returns (q, q(X), X). The verify's accept test coin*q(X) < p(X) is unbiased
    only if q is exactly the distribution X was drawn from, so callers must hand
    the returned q (not a recomputed one) to the verify.
    """
    probs = torch.softmax(next_token_logits / temperatures, dim=-1)
    topk_p, topk_index = fast_sample(probs, num_samples=1)
    return probs, topk_p, topk_index


# Simulate acceptance length for benchmarking purposes
SIMULATE_ACC_LEN = envs.SGLANG_SIMULATE_ACC_LEN.get()  # turn off if < 0
SIMULATE_ACC_METHOD = envs.SGLANG_SIMULATE_ACC_METHOD.get()
SIMULATE_ACC_TOKEN_MODE = envs.SGLANG_SIMULATE_ACC_TOKEN_MODE.get()

TREE_TRAVERSE_TIME_THRESHOLD = 1  # TODO: set this properly
TREE_SPEC_KERNEL_AVAILABLE = (
    _is_cuda or _is_musa
)  # This kernel is only available for CUDA and MUSA now


def draft_kv_indices_buffer_width(
    num_seqs: int, topk: int, max_context_len: int
) -> int:
    """Per-step row width of the EAGLE draft-decode kv_indices buffer.

    num_seqs * topk branches each attend up to max_context_len KV slots; the topk
    factor is mandatory -- dropping it under-allocates and overflows the row (#27338, #27460).
    """
    assert (
        num_seqs * topk * max_context_len < 2**31
    ), "kv_indices flat offset would overflow int32; reduce batch/topk/context"
    return num_seqs * topk * max_context_len


def draft_kv_indices_used_len(
    seq_lens_sum: int, topk: int, bs: int, num_steps: int
) -> int:
    """kv_indices length used through num_steps draft-decode steps.

    bs = topk * num_seqs branches, one index appended per branch per step. Called with
    num_steps = i + 1 (per-step slice) and speculative_num_steps (capacity assert).
    """
    return seq_lens_sum * topk + bs * num_steps


def record_stream_each(tensors, stream):
    """Call record_stream(stream) on each cuda tensor in `tensors`, skipping
    non-tensor / non-cuda entries. Tells the caching allocator that the
    tensors are also used on `stream`, so memory is not recycled while
    queued work is still in flight after Python refs drop.
    """
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.is_cuda:
            t.record_stream(stream)


def record_stream_for_v2_verify(batch, verify_input, fwd_stream):
    """Mark pre-prepare SB / verify_input GPU tensors as used on `fwd_stream`.

    Spec V2 mutates SB mid-forward (`prepare_for_verify` rebinds
    `batch.input_ids` / `out_cache_loc`; `_draft_extend_for_decode` later
    replaces `batch.input_ids` again). Each rebind drops the only SB Python
    ref to the old tensor while the verify forward kernel may still be
    reading its memory on `fwd_stream`; `record_stream` tells the caching
    allocator to wait for `fwd_stream` before recycling the block.

    Covers pre-prepare tensors only; caller must also `record_stream_each`
    the post-prepare rebinds (new `batch.input_ids` / `out_cache_loc`).
    """
    candidates = [
        batch.seq_lens,
        batch.req_pool_indices,
        batch.input_ids,
        batch.out_cache_loc,
    ]
    if verify_input is not None:
        candidates.extend(
            [
                getattr(verify_input, attr, None)
                for attr in (
                    "draft_token",
                    "custom_mask",
                    "positions",
                    "retrieve_index",
                    "retrieve_next_token",
                    "retrieve_next_sibling",
                )
            ]
        )
    record_stream_each(candidates, fwd_stream)


def spec_need_hidden_states(server_args: Optional[ServerArgs] = None) -> bool:
    if server_args is None:
        server_args = get_server_args()

    # STANDALONE drafts don't consume `spec_info.hidden_states` (vanilla LLM).
    # multi_layer_eagle, DFLASH, and DSPARK don't relay hidden_states through FutureMap.
    # TODO(lsyin): also skip when step == 1.
    if server_args.speculative_algorithm in ("STANDALONE", "DFLASH", "DSPARK"):
        return False
    return not server_args.enable_multi_layer_eagle


@torch.compile(dynamic=True, disable=_is_npu or _is_xpu)
def create_num_accept_tokens_filter(
    num_correct_drafts: torch.Tensor,
    unfinished_index_device: torch.Tensor,
    seq_lens: torch.Tensor,
):
    num_accept_tokens_filter = torch.zeros_like(num_correct_drafts)
    num_accept_tokens_filter[unfinished_index_device] = (
        num_correct_drafts[unfinished_index_device] + 1
    )
    seq_lens.add_(num_correct_drafts + 1)
    return num_accept_tokens_filter


def _select_top_k_tokens_first(
    topk_p: torch.Tensor,
    topk_index: torch.Tensor,
    hidden_states: Optional[torch.Tensor],
    topk: int,
):
    input_ids = topk_index.flatten()
    if hidden_states is not None:
        hidden_states = hidden_states.repeat_interleave(topk, dim=0)

    tree_info = (
        topk_p.unsqueeze(1),  # (b, 1, topk)
        topk_index,  # (b, topk)
        torch.arange(-1, topk, dtype=torch.long, device=input_ids.device).expand(
            topk_p.shape[0], -1
        ),  # (b, topk + 1) — expand avoids the allocation of repeat
    )
    return input_ids, hidden_states, topk_p, tree_info


@torch.compile(dynamic=True, disable=_is_npu or _is_xpu)
def _select_top_k_tokens_later(
    i: int,
    topk_p: torch.Tensor,
    topk_index: torch.Tensor,
    hidden_states: torch.Tensor,
    scores: torch.Tensor,
    topk: int,
):
    topk_sq = topk * topk

    expand_scores = scores.unsqueeze(2) * topk_p.view(-1, topk, topk)
    # (b, topk, 1) * (b, topk, topk) -> (b, topk, topk)

    topk_cs_p, topk_cs_index = fast_topk(
        expand_scores.flatten(start_dim=1), topk, dim=-1
    )  # (b, topk)

    topk_index = topk_index.view(-1, topk_sq)
    input_ids = torch.gather(topk_index, 1, topk_cs_index).flatten()

    if hidden_states is not None and hidden_states.shape[0] > 0:
        flat_cs = topk_cs_index.flatten()
        batch_offsets = torch.arange(
            0, hidden_states.shape[0], step=topk, device=flat_cs.device
        )
        selected_input_index = flat_cs // topk + batch_offsets.repeat_interleave(topk)
        hidden_states = hidden_states[selected_input_index]

    tree_info = (
        expand_scores,  # (b, topk, topk)
        topk_index,  # (b, topk * topk)
        topk_cs_index + (topk_sq * (i - 1) + topk),  # (b, topk)
    )
    return input_ids, hidden_states, topk_cs_p, tree_info


def select_top_k_tokens(
    i: int,
    topk_p: torch.Tensor,
    topk_index: torch.Tensor,
    hidden_states: torch.Tensor,
    scores: torch.Tensor,
    topk: int,
):
    if i == 0:
        return _select_top_k_tokens_first(topk_p, topk_index, hidden_states, topk)
    return _select_top_k_tokens_later(
        i, topk_p, topk_index, hidden_states, scores, topk
    )


def _sample_simulated_acc_len(
    simulate_acc_len: float,
    simulate_acc_method: str,
    max_len: int,
) -> int:
    """Sample a simulated acceptance length in [1, max_len]."""
    if simulate_acc_method == "multinomial":
        simulated_values = torch.normal(
            mean=simulate_acc_len,
            std=1.0,
            size=(1,),
            device="cpu",
        )
        # clamp simulated values to be between 1 and max_len
        simulated_values = torch.clamp(simulated_values, min=1.0, max=max_len)
        simulate_acc_len = int(simulated_values.round().item())
    elif simulate_acc_method == "match-expected":
        # multinomial sampling does not match the expected length
        # we keep it for the sake of compatibility of existing tests
        # but it's better to use "match-expected" for the cases that need to
        # match the expected length, One caveat is that this will only sample
        # either round down or round up of the expected length
        simulate_acc_len = max(1.0, min(max_len, simulate_acc_len))
        lower = int(simulate_acc_len // 1)
        upper = lower + 1 if lower < max_len else lower
        if lower == upper:
            simulate_acc_len = lower
        else:
            weight_upper = simulate_acc_len - lower
            weight_lower = 1.0 - weight_upper
            probs = torch.tensor([weight_lower, weight_upper], device="cpu")
            sampled_index = torch.multinomial(probs, num_samples=1)
            simulate_acc_len = lower if sampled_index == 0 else upper
    else:
        raise ValueError(f"Invalid simulate_acc_method: {simulate_acc_method}")
    return int(simulate_acc_len)


def generate_simulated_accept_index(
    accept_index,
    predict,
    num_correct_drafts,
    candidates,
    target_predict,
    bs,
    spec_steps,
    simulate_acc_len: float = SIMULATE_ACC_LEN,
    simulate_acc_method: str = SIMULATE_ACC_METHOD,
    simulate_acc_token_mode: str = SIMULATE_ACC_TOKEN_MODE,
):
    use_real_draft_tokens = simulate_acc_token_mode == "real-draft-token"

    assert simulate_acc_len > 0.0
    simulate_acc_len = _sample_simulated_acc_len(
        simulate_acc_len, simulate_acc_method, spec_steps + 1
    )

    accept_indx_first_col = accept_index[:, 0].view(-1, 1)
    sim_accept_index = torch.full(
        (bs, spec_steps + 1), -1, dtype=torch.int32, device=accept_index.device
    )
    sim_accept_index[:, :simulate_acc_len] = accept_indx_first_col + torch.arange(
        simulate_acc_len, device=accept_index.device
    )
    num_correct_drafts.fill_(simulate_acc_len - 1)

    if not use_real_draft_tokens:
        predict.fill_(100)  # some legit token id
        return sim_accept_index

    # Use the topk=1 draft chain for forced acceptance, then a target-derived bonus.
    if simulate_acc_len > 1:
        draft_node_indices = sim_accept_index[:, : simulate_acc_len - 1].long()
        predict[draft_node_indices] = candidates[:, 1:simulate_acc_len].to(
            dtype=predict.dtype
        )
    bonus_node_indices = sim_accept_index[:, simulate_acc_len - 1].long()
    predict[bonus_node_indices] = target_predict[:, simulate_acc_len - 1].to(
        dtype=predict.dtype
    )
    return sim_accept_index


def traverse_tree(
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    draft_tokens: torch.Tensor,
    grammar: BaseGrammarObject,
    allocate_token_bitmask: torch.Tensor,
    vocab_size: Optional[int] = None,
):
    """
    Traverse the tree constructed by the draft model to generate the logits mask.
    """
    assert (
        retrieve_next_token.shape == retrieve_next_sibling.shape == draft_tokens.shape
    )

    def dfs(
        curr: int,
        retrieve_next_token: torch.Tensor,
        retrieve_next_sibling: torch.Tensor,
        parent_pos: int,
    ):
        if curr == 0:
            # the first token generated by the target model, and thus it is always
            # accepted from the previous iteration
            is_accepted = True
        else:
            parent_bitmask = allocate_token_bitmask[parent_pos]
            current_token = draft_tokens[curr]
            if vocab_size and current_token >= vocab_size:
                is_accepted = False
            else:
                # 32 boolean bitmask values are packed into 32-bit integers
                is_accepted = (
                    parent_bitmask[current_token // 32] & (1 << (current_token % 32))
                ) != 0

        if is_accepted:
            if curr != 0:
                # Accept the current token
                grammar.accept_token(int(draft_tokens[curr]))
            if not grammar.is_terminated():
                # Generate the bitmask for the current token
                grammar.fill_vocab_mask(allocate_token_bitmask, curr)
                if retrieve_next_token[curr] != -1:
                    # Visit the child node
                    dfs(
                        int(retrieve_next_token[curr]),
                        retrieve_next_token,
                        retrieve_next_sibling,
                        curr,
                    )

            if curr != 0:
                # Rollback the current token
                grammar.rollback(1)

        if retrieve_next_sibling[curr] != -1:
            # Visit the sibling node
            dfs(
                int(retrieve_next_sibling[curr]),
                retrieve_next_token,
                retrieve_next_sibling,
                parent_pos,
            )

    dfs(0, retrieve_next_token, retrieve_next_sibling, -1)


def generate_token_bitmask(
    reqs: List[Req],
    verify_input: EagleVerifyInput,
    retrieve_next_token_cpu: torch.Tensor,
    retrieve_next_sibling_cpu: torch.Tensor,
    draft_tokens_cpu: torch.Tensor,
    vocab_size: int,
):
    """
    Generate the logit mask for structured output.
    Draft model's token can be either valid or invalid with respect to the grammar.
    We need to perform DFS to
    1. figure out which tokens are accepted by the grammar.
    2. if so, what is the corresponding logit mask.
    """

    num_draft_tokens = draft_tokens_cpu.shape[-1]

    allocate_token_bitmask = None
    assert len(reqs) == retrieve_next_token_cpu.shape[0]
    grammar = None
    for i, req in enumerate(reqs):
        if req.grammar is not None:
            if allocate_token_bitmask is None:
                allocate_token_bitmask = req.grammar.allocate_vocab_mask(
                    vocab_size=vocab_size,
                    batch_size=draft_tokens_cpu.numel(),
                    device="cpu",
                )
            grammar = req.grammar
            s = time.perf_counter()
            traverse_tree(
                retrieve_next_token_cpu[i],
                retrieve_next_sibling_cpu[i],
                draft_tokens_cpu[i],
                req.grammar,
                allocate_token_bitmask[
                    i * num_draft_tokens : (i + 1) * num_draft_tokens
                ],
                vocab_size=vocab_size,
            )
            tree_traverse_time = time.perf_counter() - s
            if tree_traverse_time > TREE_TRAVERSE_TIME_THRESHOLD:
                logger.warning(
                    f"Bit mask generation took {tree_traverse_time} seconds with "
                    f"grammar: {req.grammar}"
                )

    verify_input.grammar = grammar
    return allocate_token_bitmask


def load_token_map(token_map_path: str) -> List[int]:
    if not os.path.exists(token_map_path):
        repo_id = os.path.dirname(token_map_path)
        file_name = os.path.basename(token_map_path)

        cache_dir = None
        if envs.SGLANG_USE_MODELSCOPE.get():
            from modelscope.utils.file_utils import get_model_cache_root

            cached_repo_path = os.path.join(get_model_cache_root(), repo_id)
            if os.path.exists(cached_repo_path):
                cache_dir = cached_repo_path

        if cache_dir is None:
            if envs.SGLANG_USE_MODELSCOPE.get():
                from modelscope.hub.snapshot_download import (
                    snapshot_download as download_func,
                )
            else:
                download_func = snapshot_download
            cache_dir = download_func(
                repo_id,
                ignore_patterns=["*.bin", "*.safetensors"],
            )

        token_map_path = os.path.join(cache_dir, file_name)
    hot_token_id = torch.load(token_map_path, weights_only=True)
    return torch.tensor(hot_token_id, dtype=torch.int64)


@contextmanager
def draft_tp_context(tp_group: GroupCoordinator):
    # Draft model doesn't use dp and has its own tp group.
    # We disable mscclpp now because it doesn't support 2 comm groups.
    with patch_tensor_parallel_group(tp_group):
        yield


def spec_stage_span(name: str):
    """Profiler span for a coarse speculative-decoding stage (``draft`` /
    ``draft_extend`` / ``verify``).
    """
    return profile_range(name)


def move_accept_tokens_to_target_kvcache(
    batch: ScheduleBatch,
    accept_index: torch.Tensor,
    num_correct_drafts: torch.Tensor,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
):
    """
    Move accepted tokens (drafts + bonus) to the target KV cache.

    Args:
        batch: The batch to run.
        accept_index: The index of the accepted tokens (incl. bonus).
        num_correct_drafts: Per-req count of correct drafts (excludes bonus);
            seq_lens is advanced by ``num_correct_drafts + 1`` to cover the bonus slot.
    """
    bs = len(batch.seq_lens)
    device = batch.seq_lens.device
    # accept_index element count, NOT bs * num_draft_tokens: for topk > 1 the
    # tree exceeds the accepted chain, over-reading accept_index (illegal memory).
    size = bs * accept_index.shape[1]

    # fill_accept_out_cache_loc reads out_cache_loc[accept_index]; -1 sentinel ok.
    maybe_detect_oob(
        accept_index,
        -1,
        batch.out_cache_loc.size(0),
        "spec v2 move_accept_tokens accept_index",
    )

    tgt_cache_loc = torch.zeros(
        size,
        dtype=torch.int64,
        device=device,
    )
    accept_out_cache_loc = torch.zeros(size, dtype=torch.int64, device=device)
    if _is_cpu:
        assign_extend_cache_locs_cpu(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + num_correct_drafts + 1,
            tgt_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
        )
    else:
        assign_extend_cache_locs[(bs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + num_correct_drafts + 1,
            tgt_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            next_power_of_2(bs),
        )
    fill_accept_out_cache_loc_func(
        accept_index,
        batch.out_cache_loc,
        accept_out_cache_loc,
        size,
    )
    token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
        tgt_cache_loc, accept_out_cache_loc
    )


def prepare_mamba_track_for_verify(batch: ScheduleBatch) -> None:
    """Rebuild mamba track indices from reqs before a TARGET_VERIFY forward.

    Spec batches skip the refresh in prepare_for_decode, and filter/merge
    null these fields, so they must be rebuilt right before verify. Clearing
    the mask also keeps a stale extend-time mask from triggering in-forward
    tracking during TARGET_VERIFY; tracking is done in
    commit_mamba_states_after_verify instead.
    """
    if not get_server_args().enable_mamba_extra_buffer():
        return
    set_mamba_track_indices_from_reqs(batch)
    batch.mamba_track_mask = None
    batch.mamba_track_seqlens = None


def commit_mamba_states_after_verify(
    target_worker: TpModelWorker,
    batch: ScheduleBatch,
    accept_lens: torch.Tensor,
    accept_index: torch.Tensor,
    draft_token_num: int,
) -> None:
    """Commit accepted per-step mamba states into the persistent caches.

    During TARGET_VERIFY, hybrid linear attention backends keep per-step
    states in intermediate caches instead of advancing the persistent
    conv/ssm caches. After acceptance, the state of each request's last
    accepted step is committed back, plus the interval-crossing state used
    for prefix-cache tracking (mamba extra_buffer mode).

    No-op for models without mamba-style state or backends without the
    commit hook.
    """
    model_runner = target_worker.model_runner
    if mambaish_config(model_runner.model_config) is None:
        return
    attn_backend = model_runner.attn_backend

    bs = accept_lens.shape[0]
    # `accept_lens` already includes the bonus token (drafts + 1 per req).
    if not batch.forward_mode.is_idle() and accept_index.numel() > 0:
        accept_indices_offset = torch.arange(
            0,
            bs * draft_token_num,
            step=draft_token_num,
            dtype=accept_lens.dtype,
            device=accept_lens.device,
        )
        req_idx = torch.arange(bs, dtype=torch.int64, device=accept_lens.device)
        # Per-req tree step of the last accepted node, i.e. the step whose
        # mamba state to commit; reduces to accept_lens - 1 for topk == 1.
        last_correct_step_indices = (
            accept_index[req_idx, (accept_lens - 1).to(torch.int64)]
            - accept_indices_offset
        )

        if batch.mamba_track_indices is not None:
            # If after verify, the request's seq_lens has crossed a mamba track interval,
            # we need to update the mamba state for the request at the crossing point.
            seq_lens_pre_verify = batch.seq_lens
            seq_lens_post_verify = batch.seq_lens + accept_lens
            mamba_track_interval = get_server_args().mamba_track_interval
            to_track_mask = (
                seq_lens_pre_verify // mamba_track_interval
                != seq_lens_post_verify // mamba_track_interval
            )
            tracking_point = (
                seq_lens_post_verify // mamba_track_interval * mamba_track_interval
            )
            to_track_ith = torch.clamp(
                tracking_point - seq_lens_pre_verify - 1, min=0
            ).to(torch.int64)
            candidate_track_steps = (
                accept_index[req_idx, to_track_ith] - accept_indices_offset
            )
            mamba_steps_to_track = torch.where(
                to_track_mask,
                candidate_track_steps,
                torch.full_like(candidate_track_steps, -1),
            )
        else:
            mamba_steps_to_track = None

        if hasattr(attn_backend, "update_mamba_state_after_mtp_verify"):
            attn_backend.update_mamba_state_after_mtp_verify(
                last_correct_step_indices=last_correct_step_indices,
                mamba_track_indices=batch.mamba_track_indices,
                mamba_steps_to_track=mamba_steps_to_track,
                model=model_runner.model,
            )
        elif hasattr(model_runner.model, "update_conv_state_after_mtp_verify"):
            # Models whose conv layers bypass the attention-backend wrapper
            # (Inkling) own the commit themselves.
            model_runner.model.update_conv_state_after_mtp_verify(
                req_to_token_pool=model_runner.req_to_token_pool,
                req_pool_indices=batch.req_pool_indices[:bs],
                last_correct_step_indices=last_correct_step_indices,
                mamba_track_indices=batch.mamba_track_indices,
                mamba_steps_to_track=mamba_steps_to_track,
            )


def spec_prepare_for_decode(batch: ScheduleBatch) -> None:
    """eagle/ngram share a stateless free function; dflash keeps stateful
    prep on its draft input -- the dispatcher routes.
    """
    if batch.spec_algorithm.is_dflash_family():
        batch.spec_info.prepare_for_decode(batch)
    else:
        from sglang.srt.speculative.eagle_utils import eagle_prepare_for_decode

        eagle_prepare_for_decode(batch)


def get_plan_stream(
    device: str,
) -> Tuple[Any, contextlib.AbstractContextManager]:
    if envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        plan_stream = torch.get_device_module(device).Stream()
        plan_stream_ctx = torch.get_device_module(device).stream(plan_stream)
        return plan_stream, plan_stream_ctx
    else:
        return None, contextlib.nullcontext()
