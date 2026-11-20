"""Adapt a single greedy NextN chain to the EAGLE v2 verify input."""

import torch
from sglang.srt.hardware_backend.npu.megakernel.plugin import (
    all_admit,
    candidate_present,
    greedy,
    window_fits,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.speculative.eagle_utils import per_step_draft_out_cache_loc


def can_fuse(worker, binding, batch, result):
    # The upstream allocator already double-buffers the verify window and
    # rounds to pages. Stay native when the last suffix write would overrun
    # the request's actual allocation, including at a page boundary.
    span = worker.speculative_num_draft_tokens + binding.steps - 1
    hidden = result.logits_output.hidden_states
    return all_admit(
        len(batch.reqs) == 1
        and greedy(batch)
        and worker.hot_token_id is None
        and window_fits(batch, span, positive=True)
        and int(batch.seq_lens_cpu[0]) + span <= batch.reqs[0].kv.kv_allocated_len
        and hidden is not None
        and hidden.dtype == torch.bfloat16
        and hidden.shape == (binding.steps + 1, binding.bridge.model.config.hidden_size)
        and result.accept_lens.shape == (1,)
        and result.next_token_ids.numel() == binding.steps + 1
    )


def publish(binding, batch, forward_batch, select, next_input):
    result = binding.fused(forward_batch, select)
    hidden = result.first_hidden_states
    if (
        result.token_ids.shape != (1, binding.steps)
        or hidden is None
        or hidden.shape != (1, binding.bridge.model.config.hidden_size)
    ):
        raise RuntimeError("Invalid NextN graph output")
    tokens = result.token_ids.to(torch.int64, copy=True)
    next_input.topk_index = tokens[:, :1]
    next_input.topk_p = torch.ones_like(next_input.topk_index, dtype=torch.float32)
    next_input.hidden_states = hidden.clone()
    next_input.dsa_topk_indices = None
    # Keep the lease on the request across the overlap scheduler's FutureMap.
    # The next draft removes it before handing the chain to verify.
    expected = forward_batch.positions.index_select(0, select.long()) + 1
    batch.reqs[0].ascend_candidate = (tokens, expected, batch.reqs[0].kv.req_pool_idx)


def draft_chain(worker, binding, batch, forward_batch):
    req = batch.reqs[0] if len(batch.reqs) == 1 else None
    if binding.mode == "fused_extend":
        lease = getattr(req, "ascend_candidate", None)
        valid = lease is None or (
            lease[0].shape == (1, binding.steps)
            and lease[2] == req.kv.req_pool_idx
            and torch.equal(lease[1], forward_batch.positions)
        )
        if candidate_present(lease is not None, valid):
            req.ascend_candidate = None
            binding.hit("consume_fused_chain")
            return lease[0]
        return None
    seed = forward_batch.spec_info
    count = binding.steps - 1
    if not all_admit(
        req is not None
        and greedy(batch)
        and worker.hot_token_id is None
        and (not binding.pd_decode or getattr(req, "ascend_seed_ready", False))
        and seed.topk_index is not None
        and seed.topk_index.shape == (1, 1)
        and seed.hidden_states is not None
        and seed.hidden_states.dtype == torch.bfloat16
        and seed.hidden_states.shape == (1, binding.bridge.model.config.hidden_size)
        and window_fits(batch, count, positive=True)
        and forward_batch.out_cache_loc is not None
        and forward_batch.out_cache_loc.numel() == binding.steps
    ):
        return None
    slots = per_step_draft_out_cache_loc(
        forward_batch.out_cache_loc, 1, 1, binding.steps
    )
    backend = worker.draft_attn_backend.attn_backends[0]
    with forward_context(
        ForwardContext(attn_backend=backend), forward_batch=forward_batch
    ):
        result = binding.suffix(
            forward_batch=forward_batch,
            input_ids=seed.topk_index.reshape(-1),
            positions=forward_batch.positions,
            previous_hidden_states=seed.hidden_states,
            per_step_out_cache_loc=slots[:count],
            num_forward_steps=count,
        )
    if result.token_ids.shape != (1, count):
        raise RuntimeError("Invalid NextN suffix width")
    return torch.cat(
        (seed.topk_index, result.token_ids.to(seed.topk_index.dtype)), dim=1
    )
