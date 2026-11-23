"""GLM noaux routing in FP32, as the reference model computes it; serving and the trainer share it."""

import torch
import triton
import triton.language as tl

from sglang.srt.layers.moe.topk import StandardTopKOutput, TopK
from sglang.srt.true_on_policy.glm_topk import (
    _prepare_scores_fused,
    _select_ids,
    _validate_scores,
)
from sglang.srt.true_on_policy.moe import refuse_routed_experts_capture


def route_glm(router_logits, correction_bias, topk):
    """Select with FP32 sigmoid scores plus the FP32 correction bias; renormalize the unbiased FP32 scores.

    Selection must stay FP32: BF16 biased scores tie many of the 256 experts, and the ID tie-break then routes those
    tokens to the lowest expert IDs, overloading the EP rank that owns them.
    """
    _validate_scores(router_logits, topk)
    if torch.is_grad_enabled() and correction_bias.requires_grad:
        raise NotImplementedError("TOP GLM top-k is forward-only")
    scores = torch.sigmoid(router_logits.float())
    _, adjusted = _prepare_scores_fused(scores + correction_bias.float())
    if topk == _FUSED_TOPK and scores.shape[0]:
        return _select_and_normalize(adjusted, scores)
    ids = _select_ids(adjusted, topk)
    weights = torch.gather(scores, 1, ids)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return weights, ids.to(torch.int32)


# The fused path reproduces the unfused one bit for bit: the same selection (adjusted score descending, expert ID
# ascending on ties, as FlashInfer's deterministic top-k orders it), the unbiased scores gathered unchanged, and
# torch's FP32 row sum of 8, ((w0 + w4) + (w2 + w6)) + ((w1 + w5) + (w3 + w7)), followed by a correctly rounded divide.
_FUSED_TOPK = 8


@triton.jit
def _select_and_normalize_kernel(
    adjusted_ptr,
    scores_ptr,
    weights_ptr,
    ids_ptr,
    n_cols,
    adjusted_stride,
    scores_stride,
    BLOCK: tl.constexpr,
):
    """Stands in for the stock router's FlashInfer fused_topk_deepseek (biased_grouped_topk_gpu), which refuses
    GLM-5.2's single-group top-8 (topk_group * n_group must be >= topk): the top-8 select and FP32 renormalize."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    taken = cols >= n_cols
    adjusted = tl.load(
        adjusted_ptr + row * adjusted_stride + cols, mask=~taken, other=0.0
    )
    scores = tl.load(scores_ptr + row * scores_stride + cols, mask=~taken, other=0.0)
    slots = tl.arange(0, 8)
    ids = tl.zeros([8], dtype=tl.int32)
    weights = tl.zeros([8], dtype=tl.float32)
    for k in tl.static_range(8):
        best = tl.max(tl.where(taken, -float("inf"), adjusted), axis=0)
        pick = tl.min(tl.where(~taken & (adjusted == best), cols, BLOCK), axis=0)
        taken = taken | (cols == pick)
        ids = tl.where(slots == k, pick, ids)
        weights = tl.where(
            slots == k,
            tl.max(tl.where(cols == pick, scores, -float("inf")), axis=0),
            weights,
        )
    pairs = tl.sum(tl.reshape(weights, [2, 4]), axis=0)
    total = tl.sum(tl.sum(tl.reshape(pairs, [2, 2]), axis=0), axis=0)
    tl.store(weights_ptr + row * 8 + slots, tl.math.div_rn(weights, total + 1e-20))
    tl.store(ids_ptr + row * 8 + slots, ids)


def _select_and_normalize(adjusted, scores):
    rows, n_cols = scores.shape
    weights = torch.empty((rows, 8), device=scores.device, dtype=torch.float32)
    ids = torch.empty((rows, 8), device=scores.device, dtype=torch.int32)
    _select_and_normalize_kernel[(rows,)](
        adjusted,
        scores,
        weights,
        ids,
        n_cols,
        adjusted.stride(0),
        scores.stride(0),
        BLOCK=triton.next_power_of_2(n_cols),
    )
    return weights, ids


class TopGLMRouter(TopK):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        refuse_routed_experts_capture("GLM")
        cfg = self.topk_config
        if (
            cfg.scoring_func != "sigmoid"
            or cfg.correction_bias is None
            or not cfg.renormalize
            or cfg.top_k <= 1
            or cfg.num_expert_group not in (None, 1)
            or cfg.topk_group not in (None, 1)
            or cfg.num_shared_topk_slots
            or cfg.custom_routing_function is not None
        ):
            raise NotImplementedError(
                "[top] GLM requires single-group sigmoid/bias routing without fused shared slots"
            )

    def forward(
        self,
        hidden_states,
        router_logits,
        *,
        num_token_non_padded=None,
        expert_location_dispatch_info=None,
    ):
        # When a caller omits these, TopK resolves them from the forward context and the placement metadata.
        num_token_non_padded, expert_location_dispatch_info = (
            self._resolve_routing_inputs(
                num_token_non_padded, expert_location_dispatch_info
            )
        )
        if expert_location_dispatch_info is not None:
            raise NotImplementedError(
                "[top] GLM routing requires fixed expert ownership"
            )
        weights, ids = route_glm(
            router_logits, self.topk_config.correction_bias, self.topk_config.top_k
        )
        if num_token_non_padded is not None:
            from sglang.srt.layers.moe.topk import _mask_topk_ids_padded_region

            _mask_topk_ids_padded_region(ids, num_token_non_padded)
        return StandardTopKOutput(
            topk_weights=weights, topk_ids=ids, router_logits=router_logits
        )
