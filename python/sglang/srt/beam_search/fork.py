"""Fork primitive: spawn decode-ready members and reparent KV (copy-on-fork).

- spawn: a plain member Req that enters decode directly -- no member prefill;
  its first decode step computes the selected token's KV like normal decode.
- reparent: all beams in a group share the same length, so reparenting is a
  pure KV **data** copy onto the destination's own slots: no allocator
  traffic, no req_to_token remapping, stream-safe and capturable.

Prompt ownership on existing fields: cache_protected_len = prompt_len marks
the (leader- or tree-owned) prompt as not-mine-to-free;
skip_radix_cache_insert keeps member suffixes out of the tree.

FORK_STATE_PLAN classifies every Req attribute for forking; the drift-guard
test parses Req.__init__ and fails on any unclassified new attribute.
"""

from __future__ import annotations

import ast
import inspect
from typing import List, Sequence

import torch

# ==================== fork-state enumeration ====================

# Freshly initialized by Req.__init__ for the member; nothing to carry over.
REBUILD = "rebuild"
# Sourced from the leader by value (ctor argument or post-init copy).
COPY = "copy"
# Same object as the leader, read-only by convention.
SHARE = "share"
# Set explicitly by the fork primitive (KV mapping, bookkeeping, first token).
SPAWN = "spawn"
# Unsupported-with-beam path (validated away) or legacy state pending removal.
EXCLUDE = "exclude"

FORK_STATE_PLAN = {
    # --- identity / inputs ---
    "rid": COPY,  # derived: f"{leader.rid}#beam{i}"
    "origin_input_ids": SHARE,
    "origin_input_ids_unpadded": SHARE,
    "sampling_params": COPY,  # neutralized clone, see neutral_member_sampling_params
    "custom_logit_processor": COPY,
    "extra_key": COPY,
    "lora_id": COPY,
    "routing_key": COPY,
    "priority": COPY,
    "routed_dp_rank": COPY,  # members co-locate with the leader's DP rank
    "tokenizer": SHARE,
    "eos_token_ids": SHARE,
    "vocab_size": SHARE,
    # --- set by the fork primitive ---
    "beam_group": SPAWN,  # shared BeamGroup overlay (leader's group);
    #                  is_beam_leader derives from it (leader identity)
    "output_ids": SPAWN,  # [first_token]: the next decode input
    "req_pool_idx": SPAWN,
    "kv": SPAWN,  # ReqKvInfo(prompt_len, 0)
    "kv_committed_len": SPAWN,  # == prompt_len at birth
    "prefix_indices": SPAWN,
    "last_node": SPAWN,  # radix-on: locked tree handle; alias mode: None
    "cache_protected_len": SPAWN,  # == prompt_len: prompt is not mine to free
    "skip_radix_cache_insert": SPAWN,  # True: member suffixes never enter the tree
    # --- fresh member state (Req.__init__ defaults are correct) ---
    "full_untruncated_fill_ids": REBUILD,
    "extend_range": REBUILD,
    "session": REBUILD,
    "session_id": REBUILD,
    "input_embeds": REBUILD,
    "positional_embed_overrides": REBUILD,
    "multi_item_delimiter_indices": REBUILD,
    "token_type_ids": REBUILD,
    "swa_evict_floor": REBUILD,
    "extend_batch_idx": REBUILD,
    "decode_batch_idx": REBUILD,
    "http_worker_ipc": REBUILD,  # members are internal, never emit to a worker
    "require_reasoning": REBUILD,
    "_is_reasoning_over": REBUILD,
    "reasoning_tokens": REBUILD,
    "return_hidden_states": REBUILD,
    "finished_reason": REBUILD,  # coordinator-owned: members never self-finish
    "finished_len": REBUILD,
    "finished_output": REBUILD,
    "to_finish": REBUILD,
    "stream": REBUILD,  # beam emits once at group finish
    "surr_offset": REBUILD,
    "read_offset": REBUILD,
    "decoded_text": REBUILD,
    "multimodal_inputs": REBUILD,  # prefill-only; members never prefill
    "mm_image_tokens": REBUILD,
    "mm_audio_tokens": REBUILD,
    "mm_video_tokens": REBUILD,
    "last_host_node": REBUILD,
    "best_match_node": REBUILD,
    "host_hit_length": REBUILD,
    "swa_host_hit_length": REBUILD,
    "mamba_host_hit_length": REBUILD,
    "num_matched_prefix_tokens": REBUILD,
    "storage_hit_length": REBUILD,
    "inflight_middle_chunks": REBUILD,
    "is_retracted": REBUILD,
    "retracted_stain": REBUILD,
    "send_token_offset": REBUILD,
    "send_decode_id_offset": REBUILD,
    "send_output_token_logprobs_offset": REBUILD,
    "send_output_sampling_mask_offset": REBUILD,
    "return_logprob": SPAWN,  # True: internal top-2k channel (never user-facing)
    "logprob_start_len": REBUILD,
    "logprob": SPAWN,  # fresh ReqLogprob, top_logprobs_num set to 2k
    "temp_scaled_logprobs": REBUILD,
    "top_p_normalized_logprobs": REBUILD,
    "return_sampling_mask": REBUILD,
    "input_logprob_sent": REBUILD,
    "input_token_logprobs": REBUILD,
    "temp_input_top_logprobs_val": REBUILD,
    "temp_input_top_logprobs_idx": REBUILD,
    "temp_input_token_ids_logprobs_val": REBUILD,
    "temp_input_token_ids_logprobs_idx": REBUILD,
    "hidden_states": REBUILD,
    "hidden_states_tensor": REBUILD,
    "output_topk_p": REBUILD,
    "output_topk_index": REBUILD,
    "output_dsa_topk_indices": REBUILD,
    "return_routed_experts": REBUILD,
    "routed_experts_start_len": REBUILD,
    "routed_experts": REBUILD,
    "return_indexer_topk": REBUILD,
    "indexer_topk": REBUILD,
    "customized_info": REBUILD,
    "embedding": REBUILD,
    "cached_tokens": REBUILD,
    "already_computed": REBUILD,
    "cached_tokens_device": REBUILD,
    "cached_tokens_host": REBUILD,
    "cached_tokens_storage": REBUILD,
    "_cache_breakdown_computed": REBUILD,
    "retraction_count": REBUILD,
    "retraction_mb_id": REBUILD,
    "metrics_collector": REBUILD,
    "has_log_time_stats": REBUILD,
    "dimensions": REBUILD,
    "return_pooled_hidden_states": REBUILD,
    "pooled_hidden_state": REBUILD,
    "hisparse_staging": REBUILD,
    "output_token_sampling_mask": REBUILD,
    "output_token_sampling_logprobs": REBUILD,
    "time_stats": REBUILD,
    # --- paths beam requests validate away (rejected or long-term excluded) ---
    "mamba_pool_idx": EXCLUDE,
    "mamba_ping_pong_track_buffer": EXCLUDE,
    "mamba_next_track_idx": EXCLUDE,
    "mamba_last_track_seqlen": EXCLUDE,
    "mamba_branching_seqlen": EXCLUDE,
    "mamba_cow_src_index": EXCLUDE,
    "mamba_needs_clear": EXCLUDE,
    "mamba_lazy_is_insert": EXCLUDE,
    "swa_uuid_for_lock": EXCLUDE,
    "swa_prefix_lock_released": EXCLUDE,
    "grammar_key": EXCLUDE,
    "grammar": EXCLUDE,
    "grammar_wait_ct": EXCLUDE,
    "spec_verify_ct": EXCLUDE,
    "spec_num_correct_drafts": EXCLUDE,
    "spec_num_block_accept_tokens": EXCLUDE,
    "spec_num_cap_tokens": EXCLUDE,
    "spec_correct_drafts_histogram": EXCLUDE,
    "spec_cap_lens_histogram": EXCLUDE,
    "bootstrap_host": EXCLUDE,
    "bootstrap_port": EXCLUDE,
    "bootstrap_room": EXCLUDE,
    "pd_rebootstrap_forced_output_id": EXCLUDE,
    "disagg_kv_sender": EXCLUDE,
    "disagg_prefill_dp_rank": EXCLUDE,
    "start_send_idx": EXCLUDE,
    "tmp_end_idx": EXCLUDE,
    "metadata_buffer_index": EXCLUDE,
    "pending_bootstrap": EXCLUDE,
    "prefill_attempt_count": EXCLUDE,
    # --- dllm mixin (conditional init; beam requests never enable dllm) ---
    "dllm_initialized": EXCLUDE,
    "dllm_phase": EXCLUDE,
    "dllm_incomplete_ids": EXCLUDE,
    "dllm_algo_state": EXCLUDE,
    "dllm_block_offset": EXCLUDE,
    "dllm_config": EXCLUDE,
}


def collect_req_state_fields() -> List[str]:
    """AST-parse the Req initializers for every `self.X = ...` target."""

    def collect(module, cls_name, fn_name):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == cls_name:
                for fn in node.body:
                    if isinstance(fn, ast.FunctionDef) and fn.name == fn_name:
                        fields = []
                        for sub in ast.walk(fn):
                            if isinstance(sub, (ast.Assign, ast.AnnAssign)):
                                targets = (
                                    sub.targets
                                    if isinstance(sub, ast.Assign)
                                    else [sub.target]
                                )
                                for t in targets:
                                    if (
                                        isinstance(t, ast.Attribute)
                                        and isinstance(t.value, ast.Name)
                                        and t.value.id == "self"
                                        and t.attr not in fields
                                    ):
                                        fields.append(t.attr)
                        return fields
        raise AssertionError(f"{cls_name}.{fn_name} not found")

    from sglang.srt.dllm.mixin import req as dllm_req_mixin
    from sglang.srt.managers import schedule_batch

    fields = collect(schedule_batch, "Req", "__init__")
    for extra in collect(dllm_req_mixin, "ReqDllmMixin", "init_diffusion_llm"):
        if extra not in fields:
            fields.append(extra)
    return fields


# ==================== spawn ====================

# Members outlive the group's own length checks by a margin so that member-side
# length finish can never race the coordinator's deterministic advance_final.
MEMBER_LENGTH_MARGIN = 4


def neutral_member_sampling_params(leader_params):
    """Neutral params: raw logprob scoring, and no member-side finish path.

    Members never self-finish (the coordinator owns all stop/length
    semantics), so stop conditions are stripped and ignore_eos is forced.
    """
    from sglang.srt.sampling.sampling_params import SamplingParams

    return SamplingParams(
        max_new_tokens=(leader_params.max_new_tokens or 0) + MEMBER_LENGTH_MARGIN,
        temperature=1.0,
        top_p=1.0,
        min_p=0.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        min_new_tokens=0,
        n=1,
        ignore_eos=True,
        skip_special_tokens=leader_params.skip_special_tokens,
        spaces_between_special_tokens=leader_params.spaces_between_special_tokens,
    )


def spawn_member(leader, first_token: int, member_index: int):
    """Build a decode-ready member Req off the leader (no member prefill).

    KV/pool state is not touched here; call init_member_kv_state once the
    member's req_to_token row is known. Attribute handling follows
    FORK_STATE_PLAN.
    """
    from sglang.srt.managers.schedule_batch import Req

    member = Req(
        rid=f"{leader.rid}#beam{member_index}",
        origin_input_text="",
        origin_input_ids=leader.origin_input_ids,
        sampling_params=neutral_member_sampling_params(leader.sampling_params),
        origin_input_ids_unpadded=leader.origin_input_ids_unpadded,
        lora_id=leader.lora_id,
        eos_token_ids=leader.eos_token_ids,
    )

    member.tokenizer = leader.tokenizer
    member.vocab_size = leader.vocab_size
    member.custom_logit_processor = leader.custom_logit_processor
    member.extra_key = leader.extra_key
    member.routing_key = leader.routing_key
    member.priority = leader.priority
    member.routed_dp_rank = leader.routed_dp_rank

    member.beam_group = leader.beam_group
    # Internal top-2k logprob channel: members ride the standard per-row
    # top_logprobs_num machinery; the payload never reaches the user.
    member.return_logprob = True
    member.logprob.top_logprobs_num = leader.logprob.top_logprobs_num

    member.output_ids.append(first_token)
    return member


def init_member_kv_state(
    member,
    req_to_token: torch.Tensor,
    leader_row: int,
    prompt_len: int,
):
    """Alias-mode prompt mapping + born-correct linear KV bookkeeping, for a
    member whose row (req_pool_idx) was assigned by the standard pool alloc.

    The prompt KV indices are aliased read-only from the leader's row; the
    member owns only its decode suffix, which standard alloc_for_decode
    extends from here. Any tree lock on a matched prompt prefix is held by
    the leader for the whole group's lifetime; members never touch the tree.
    """
    from sglang.srt.managers.schedule_batch import ReqKvInfo

    req_to_token[member.req_pool_idx, :prompt_len] = req_to_token[
        leader_row, :prompt_len
    ]
    member.kv = ReqKvInfo(kv_allocated_len=prompt_len, swa_evicted_seqlen=0)
    member.kv_committed_len = prompt_len
    member.cache_protected_len = prompt_len
    member.skip_radix_cache_insert = True


# ==================== reparent ====================


def reparent_kv(
    req_to_token: torch.Tensor,
    kv_buffers: Sequence[torch.Tensor],
    dst_rows: torch.Tensor,
    src_rows: torch.Tensor,
    prefix_len: int,
    seq_len: int,
) -> None:
    """Copy the decode-suffix KV data of src rows onto dst rows.

    Synchronized lengths make this a pure data move: dst keeps its own KV
    slots and its req_to_token mapping; only buffer contents change. All
    index math is tensor-side (no host sync), so the whole call can be
    enqueued in-stream and captured. Rows with parent == self must simply be
    omitted from dst_rows/src_rows.
    """
    src_slots = req_to_token[src_rows, prefix_len:seq_len].reshape(-1)
    dst_slots = req_to_token[dst_rows, prefix_len:seq_len].reshape(-1)
    for buf in kv_buffers:
        buf[dst_slots] = buf[src_slots]
