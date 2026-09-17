"""CPU-side eligibility for the optional compact verifier."""

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_device, get_parallel, get_spec


def configured():
    if not envs.SGLANG_ENABLE_COMPACT_SPEC_VERIFY.get():
        return False
    p, s = get_parallel(), get_spec()
    return (
        get_device().device == "cuda"
        and p.tp_size == 4
        and p.pp_size == 1
        and p.dp_size == 1
        and p.dcp_size == 1
        and p.attn_cp_size == 1
        and p.nnodes == 1
        and p.attn_dp_size in (None, 1)
        and s.speculative_use_rejection_sampling
        and s.speculative_eagle_topk == 1
        and s.speculative_algorithm in ("EAGLE", "EAGLE3", "NEXTN")
    )


def sharded_graph_output(model_runner):
    import torch

    return (
        configured()
        and not model_runner.is_draft_worker
        and model_runner.model_config.vocab_size == 154880
        and model_runner.model_config.dtype == torch.bfloat16
        and not getattr(
            model_runner.model_config.hf_config, "final_logit_softcapping", None
        )
    )


def sampling_supported(verify, batch, grammar_mask):
    s = batch.sampling_info
    return (
        grammar_mask is None
        and not batch.return_logprob
        and verify.tree_topk == 1
        and verify.max_tree_depth == verify.draft_token_num
        and verify.draft_token_num >= 2
        and not s.is_any_greedy
        and not s.need_top_k_sampling
        and not s.need_top_p_sampling
        and not s.need_min_p_sampling
        and not s.has_custom_logit_processor
        and s.acc_additive_penalties is None
        and s.acc_scaling_penalties is None
        and s.logit_bias is None
        and all(req.sampling_params.temperature == 1.0 for req in batch.reqs)
        and not any(s.return_sampling_masks or [])
    )
