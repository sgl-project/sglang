from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def is_npu_mtp_sampling_supported(
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    accept_index: torch.Tensor,
    tree_topk: int,
    use_rejection_sampling: bool,
) -> bool:
    """Check whether the NPU kernels can preserve the supplied tree semantics."""
    if tree_topk < 1 or candidates.ndim != 2:
        return False
    batch_size, num_draft_tokens = candidates.shape
    if (
        retrieve_index.shape != candidates.shape
        or retrieve_next_token.shape != candidates.shape
        or retrieve_next_sibling.shape != candidates.shape
        or accept_index.ndim != 2
        or accept_index.shape[0] != batch_size
        or not 1 <= accept_index.shape[1] <= num_draft_tokens
    ):
        return False

    if use_rejection_sampling:
        return tree_topk == 1 and accept_index.shape == candidates.shape
    return True


def npu_mtp_non_greedy_sample(
    next_token_logits: torch.Tensor,
    sampling_info,
    verify_input,
    candidates: torch.Tensor,
    predict: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the NPU-only non-greedy MTP verify path."""
    from sglang.srt.runtime_context import get_parallel, get_server_args

    server_args = get_server_args()
    use_rejection_sampling = server_args.speculative_use_rejection_sampling
    if not is_npu_mtp_sampling_supported(
        candidates=candidates,
        retrieve_index=verify_input.retrieve_index,
        retrieve_next_token=verify_input.retrieve_next_token,
        retrieve_next_sibling=verify_input.retrieve_next_sibling,
        accept_index=accept_index,
        tree_topk=verify_input.tree_topk,
        use_rejection_sampling=use_rejection_sampling,
    ):
        raise ValueError(
            "NPU non-greedy speculative sampling cannot consume this draft "
            "topology. "
            "Target-only sampling supports regular EAGLE trees with "
            "tree_topk >= 1, while classic rejection sampling requires a "
            "tree_topk=1 linear chain."
        )

    try:
        from sgl_kernel_npu.sample import (
            chain_speculative_sampling_rejection,
            top_k_top_p_renorm_probs,
            tree_speculative_sampling_target_only,
        )
    except ImportError as exc:
        raise RuntimeError(
            "NPU non-greedy speculative sampling requires the matching "
            "sgl-kernel-npu sampling kernels."
        ) from exc

    batch_size, num_draft_tokens = candidates.shape
    expanded_temperature = torch.repeat_interleave(
        sampling_info.temperatures, num_draft_tokens, dim=0
    )
    target_probs = F.softmax(next_token_logits.float() / expanded_temperature, dim=-1)

    from sglang.srt.utils.async_probe import maybe_detect_nan

    maybe_detect_nan(target_probs, "NPU v2 verify: target_probs after softmax")
    target_probs = top_k_top_p_renorm_probs(
        target_probs,
        torch.repeat_interleave(sampling_info.top_ks, num_draft_tokens, dim=0),
        torch.repeat_interleave(sampling_info.top_ps, num_draft_tokens, dim=0),
        sampling_info.need_top_k_sampling,
        sampling_info.need_top_p_sampling,
    )
    maybe_detect_nan(target_probs, "NPU v2 verify: target_probs after renorm")
    target_probs = target_probs.reshape(batch_size, num_draft_tokens, -1).contiguous()

    if use_rejection_sampling:
        draft_probs = verify_input.draft_probs
        if draft_probs is None or draft_probs.shape[-1] != target_probs.shape[-1]:
            raise ValueError(
                "NPU rejection sampling requires a target-vocab draft proposal "
                "distribution; draft_probs is missing or vocab-mismatched."
            )
        draft_probs = draft_probs.float().contiguous()
        sampling_fn = chain_speculative_sampling_rejection
    else:
        # The target-only NPU wrapper initializes this scratch buffer itself.
        draft_probs = torch.empty_like(target_probs)
        sampling_fn = tree_speculative_sampling_target_only

    coins = torch.rand_like(candidates, dtype=torch.float32)
    coins_for_final_sampling = torch.rand(
        (batch_size,), dtype=torch.float32, device=candidates.device
    )
    sampling_fn(
        predicts=predict,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=verify_input.retrieve_index,
        retrive_next_token=verify_input.retrieve_next_token,
        retrive_next_sibling=verify_input.retrieve_next_sibling,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=coins_for_final_sampling,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=server_args.speculative_accept_threshold_single,
        threshold_acc=server_args.speculative_accept_threshold_acc,
        deterministic=True,
    )

    from sglang.srt.distributed import get_tp_group
    from sglang.srt.layers.dp_attention import is_dp_attention_enabled

    tp_group = (
        get_parallel().attn_tp_group if is_dp_attention_enabled() else get_tp_group()
    )
    if tp_group.world_size > 1:
        tp_group.broadcast(predict, src=0)
        tp_group.broadcast(accept_index, src=0)
        tp_group.broadcast(accept_token_num, src=0)

    return predict, accept_index, accept_token_num
