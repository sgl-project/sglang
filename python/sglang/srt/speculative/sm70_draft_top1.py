"""Exact global top-1 for a sharded native MTP head on Volta."""

import torch

from sglang.srt.environ import envs


def configure_draft_top1(runner, topk, hot_token_id, server_args):
    """Keep the full draft vocabulary, but exchange only each shard's winner.

    The draft runner's logits stay shard-local (no all-gather), and the winner
    is resolved with a two-float all-gather, so a full-vocabulary draft head
    costs one small collective per step instead of a full-vocabulary gather.
    """
    if not envs.SGLANG_SM70_MTP_LOCAL_TOP1.get():
        return False

    # Cheap gates first: they never need the model to be importable.
    if (
        topk != 1
        or hot_token_id is not None
        or server_args.enable_dp_attention
        or server_args.enable_lora
        or getattr(server_args, "pp_size", 1) != 1
    ):
        raise ValueError(
            "SM70 draft top-1 requires ordinary TP, topk=1, no LoRA/DP attention and the full vocabulary"
        )

    from sglang.srt.models.qwen4_exp_mtp import Qwen4ExpForCausalLMMTP

    model = runner.model
    if not isinstance(model, Qwen4ExpForCausalLMMTP):
        raise ValueError("SM70 draft top-1 requires native Qwen4 MTP")
    head, processor = model.lm_head, model.logits_processor
    shard = head.shard_indices
    if (
        head.weight.dtype != torch.float16
        or head.weight.device.type != "cuda"
        or torch.cuda.get_device_capability(head.weight.device) != (7, 0)
        or not processor.do_tensor_parallel_all_gather
        or processor.do_tensor_parallel_all_gather_dp_attn
        or processor.use_attn_tp_group
        or processor.return_full_logits
        or processor.use_fp32_lm_head
        or processor.logit_scale is not None
        or processor.final_logit_softcapping is not None
        or getattr(head, "num_added_embeddings", 0) != 0
        or not (0 < shard.num_org_elements <= head.weight.shape[0])
        or not (0 < processor.vocab_size < 2**24)
    ):
        raise ValueError("Unsupported head layout for SM70 draft top-1")

    runner.sm70_draft_top1_spec = (shard.org_vocab_start_index, shard.num_org_elements)
    runner.sm70_draft_logits_width = head.weight.shape[0]
    # This processor belongs only to the draft runner; target verification and
    # its full-vocabulary sampling/logprob path remain unchanged.
    processor.do_tensor_parallel_all_gather = False
    return True


def exchange_top1(logits, shard_start, valid_columns, group):
    """Match argmax of concatenated TP shards, including the lowest-ID tie rule."""
    values, indices = logits[:, :valid_columns].max(dim=-1)
    candidates = torch.stack((values.float(), (indices + shard_start).float()), dim=-1)
    gathered = group.all_gather(candidates, dim=-1).reshape(
        logits.shape[0], group.world_size, 2
    )
    rank = gathered[..., 0].argmax(dim=-1, keepdim=True)
    return gathered[..., 1].gather(1, rank).long()


def draft_top1(logits, runner):
    spec = getattr(runner, "sm70_draft_top1_spec", None)
    if spec is None:
        return torch.argmax(logits, dim=-1, keepdim=True)
    from sglang.srt.distributed.parallel_state import get_tp_group

    return exchange_top1(logits, *spec, get_tp_group())
