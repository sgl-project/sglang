"""PD draft-state handoff validation and decode-side initialization."""

from __future__ import annotations


def validate_draft_handoff(
    *,
    prefill_has_draft,
    prefill_rs,
    decode_has_draft,
    decode_rs,
    decode_bootstrap=False,
):
    """Return the source of draft state; never infer capability from an RS flag.

    Decode bootstrap uses a target-only metadata layout and must be configured
    before RDMA buffers are registered. Mixed seeded/seedless peers on that
    listener are intentionally rejected, not interpreted using the wrong stride.
    """
    if type(prefill_has_draft) is not bool:
        raise RuntimeError(
            "PD prefill draft capability missing or invalid; upgrade both peers"
        )
    if decode_bootstrap:
        if prefill_has_draft or not decode_has_draft:
            raise RuntimeError(
                "PD decode draft bootstrap requires target-only prefill and EAGLE decode"
            )
        return "decode"
    if prefill_has_draft != decode_has_draft:
        raise RuntimeError(
            "PD draft-state layout mismatch: target-only prefill with EAGLE decode "
            "requires --disaggregation-decode-draft-bootstrap; other asymmetric "
            "draft layouts are not supported"
        )
    if not decode_has_draft:
        return "none"
    if type(prefill_rs) is not bool or prefill_rs != decode_rs:
        raise RuntimeError(
            "PD --speculative-use-rejection-sampling mismatch or unknown setting: "
            f"prefill={prefill_rs}, decode={decode_rs}. Both drafting workers "
            "must agree when transferring draft probabilities."
        )
    return "prefill"


def bootstrap_prompt(req, max_tokens):
    """Replay the committed prefix, excluding the already sampled handoff token."""
    if not req.output_ids:
        raise RuntimeError("PD draft bootstrap requires a committed handoff token")
    if req.multimodal_inputs is not None or req.input_embeds is not None:
        raise RuntimeError(
            "PD decode draft bootstrap currently supports text tokens only"
        )
    if req.grammar is not None:
        raise RuntimeError(
            "PD decode draft bootstrap does not yet support constrained decoding"
        )
    tokens = list(req.origin_input_ids) + list(req.output_ids[:-1])
    if not tokens or len(tokens) > max_tokens:
        raise RuntimeError(
            f"PD draft bootstrap prefix length {len(tokens)} exceeds budget {max_tokens}"
        )
    return tokens


def bootstrap_decode_draft(scheduler, req):
    """Initialize real draft KV/state after a target-only PD transfer.

    This conservative, opt-in fallback replays the complete committed prefix.
    One target-only decode step cannot initialize NEXTN's missing prompt KV.
    The scheduler has drained overlap; no existing forward may use runner
    scratch buffers while this synchronous one-request extend runs.
    """
    import torch

    from sglang.srt.layers.moe.utils import (
        speculative_moe_a2a_backend_context,
        speculative_moe_backend_context,
    )
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardMode,
    )
    from sglang.srt.runtime_context import get_disagg
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

    tokens = bootstrap_prompt(
        req, get_disagg().disaggregation_decode_draft_bootstrap_max_tokens
    )
    length = len(tokens)
    batch = ScheduleBatch.init_new(
        [req],
        scheduler.req_to_token_pool,
        scheduler.token_to_kv_pool_allocator,
        scheduler.tree_cache,
        scheduler.model_config,
        False,
        scheduler.spec_algorithm,
    )
    device = batch.device
    idx = req.kv.req_pool_idx
    # Reuse already allocated prompt slots; never allocate a duplicate KV chain.
    slots = scheduler.req_to_token_pool.req_to_token[idx, :length].to(torch.int64)
    if slots.numel() != length or torch.any(slots <= 0).item():
        raise RuntimeError("PD draft bootstrap has an incomplete committed KV mapping")
    batch.forward_mode = ForwardMode.EXTEND
    batch.input_ids = torch.tensor(tokens, dtype=torch.int64, device=device)
    batch.req_pool_indices = torch.tensor([idx], dtype=torch.int64, device=device)
    batch.req_pool_indices_cpu = torch.tensor([idx], dtype=torch.int64)
    batch.seq_lens = torch.tensor([length], dtype=torch.int64, device=device)
    batch.seq_lens_cpu = torch.tensor([length], dtype=torch.int64)
    batch.orig_seq_lens = batch.seq_lens.to(torch.int32)
    batch.seq_lens_sum = length
    batch.out_cache_loc = slots
    batch.prefix_lens = [0]
    batch.extend_lens = [length]
    batch.extend_num_tokens = length
    batch.extend_logprob_start_lens = [length]
    batch.multimodal_inputs = [None]
    batch.return_logprob = False
    batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
        batch, scheduler.model_config.vocab_size
    )
    handoff = torch.tensor([req.output_ids[-1]], dtype=torch.int64, device=device)
    worker = scheduler.model_worker
    # is_verify means skip the target sampler, not TARGET_VERIFY forward mode.
    # The CTX handoff token and its original logprob must remain unchanged.
    result = worker.target_worker.forward_batch_generation(
        batch, is_verify=True, capture_hidden_mode=CaptureHiddenMode.FULL
    )
    draft = worker.draft_worker
    with (
        draft.draft_tp_context(draft.draft_runner.tp_group),
        speculative_moe_backend_context(),
        speculative_moe_a2a_backend_context(),
    ):
        seed = draft._draft_extend_for_prefill(
            batch, result.logits_output.hidden_states, handoff
        )
    # Commit only after the complete initialization succeeded. Synchronous CPU
    # copies fence completion before process_prebuilt publishes the relay row.
    req.output_topk_p = seed.topk_p[0].detach().cpu().clone()
    req.output_topk_index = seed.topk_index[0].detach().cpu().clone()
    req.hidden_states_tensor = seed.hidden_states[0].detach().cpu().clone()
    req.output_draft_probs = (
        seed.draft_probs[0].detach().cpu().clone()
        if seed.draft_probs is not None
        else None
    )
    # GEN-local DSA indices must not go through the CTX request-relative remap.
    # No seed is conservative: the first draft iteration recomputes the indexer.
    req.output_dsa_topk_indices = None
    req.pd_draft_bootstrap_tokens += length
    req.pd_draft_bootstrap_pending = False
