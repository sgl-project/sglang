"""Host ABI for opt-in whole-model providers. No device imports at discovery."""

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import entry_points
from types import SimpleNamespace

log = logging.getLogger(__name__)
GROUP = "sglang.ascend_megakernel"


@lru_cache(None)
def draft_mode():
    explicit = os.getenv("SGLANG_ASCEND_DRAFT_MODE")
    legacy = os.getenv("MK_FUSED_DRAFT")
    alias = None
    if legacy is not None:
        if legacy.lower() in ("0", "false", "no", "n"):
            alias = "decode_loop"
        elif legacy.lower() in ("1", "true", "yes", "y"):
            alias = "fused_extend"
        else:
            raise ValueError("Invalid MK_FUSED_DRAFT")
    if explicit is not None and alias is not None and explicit != alias:
        raise ValueError("Conflicting SGLANG_ASCEND_DRAFT_MODE and MK_FUSED_DRAFT")
    mode = explicit or alias or "native"
    if mode not in ("native", "decode_loop", "fused_extend"):
        raise ValueError(f"Unknown Ascend draft mode: {mode}")
    return mode


def linear_config(args):
    return (
        args.device == "npu"
        and args.max_running_requests == 1
        and args.page_size == 128
        and args.kv_cache_dtype == "fp8_e4m3"
        and args.cuda_graph_config.decode.backend
        == args.cuda_graph_config.prefill.backend
        == "disabled"
        and args.disaggregation_mode in ("null", "decode")
        and not args.disable_overlap_schedule
        and args.attn_dp_size == 1
        and args.tp_size == 8
        and args.dwdp_size == 1
        and args.pp_size == 1
        and args.dp_size == 1
        and args.ep_size == 1
        and args.attn_cp_size == 1
        and args.moe_dp_size == 1
        and args.dcp_size == 1
        and not args.enable_return_hidden_states
        and not args.enable_return_routed_experts
        and not args.enable_return_indexer_topk
        and args.return_hidden_states_mode is None
        and not args.enable_hisparse
        and not args.enable_hierarchical_cache
        and not args.enable_watermark
        and not args.enable_lora
        and not args.enable_two_batch_overlap
        and not args.speculative_adaptive
        and not args.enable_multi_layer_eagle
        and not args.speculative_use_rejection_sampling
        and (
            args.speculative_algorithm is None
            or (
                args.speculative_algorithm in ("NEXTN", "EAGLE")
                and args.speculative_eagle_topk == 1
                and (args.speculative_num_steps or 0) > 1
                and args.speculative_num_draft_tokens == args.speculative_num_steps + 1
            )
        )
    )


def greedy(batch):
    sampling = batch.sampling_info
    return (
        sampling is not None
        and sampling.is_all_greedy
        and getattr(batch, "mm_input_embeds", None) is None
        and getattr(batch, "replace_embeds", None) is None
        and not any(getattr(sampling, "return_sampling_masks", None) or ())
        and not batch.return_logprob
        and not getattr(batch, "has_grammar", False)
        and not getattr(sampling, "grammars", None)
        and not getattr(sampling, "has_custom_logit_processor", False)
        and getattr(sampling, "logit_bias", None) is None
        and getattr(sampling, "acc_additive_penalties", None) is None
        and getattr(sampling, "acc_scaling_penalties", None) is None
        and not getattr(
            getattr(sampling, "penalizer_orchestrator", None), "is_required", False
        )
    )


def window_fits(batch, ahead, *, positive=False):
    lengths = batch.seq_lens_cpu
    if lengths is None or len(lengths) != 1:
        return False
    length = int(lengths[0])
    return int(positive) <= length and length + ahead <= 128 * 128


def all_admit(local):
    import torch
    import torch.distributed as dist
    from sglang.srt.runtime_context import get_parallel

    vote = torch.tensor(int(local), dtype=torch.int32, device="cpu")
    dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=get_parallel().tp_group.cpu_group)
    return bool(vote.item())


def candidate_present(local, valid=True):
    import torch
    import torch.distributed as dist
    from sglang.srt.runtime_context import get_parallel

    group = get_parallel().tp_group.cpu_group
    vote = torch.tensor(
        int(local) if valid else dist.get_world_size(group) + 1, dtype=torch.int32
    )
    dist.all_reduce(vote, group=group)
    count = int(vote.item())
    if count not in (0, dist.get_world_size(group)):
        raise RuntimeError("Ascend candidate ownership diverged across TP ranks")
    return count != 0


@dataclass(frozen=True)
class ModelSpec:
    version: int
    roles: dict
    prepare: object


def bind_runner(runner):
    import torch.distributed as dist
    from sglang.srt.arg_groups.model_override_base import resolved_view
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_parallel

    args, model = resolved_view(runner.server_args), runner.model
    group = get_parallel().tp_group.cpu_group
    error = role = binding = mode = None
    try:
        mode = draft_mode()
        if (
            linear_config(args)
            and not runner.runner_ids.is_translating
            and not runner.token_to_kv_pool_allocator.hands_out_virtual_ids
            and not envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get()
        ):
            choices = tuple(entry_points(group=GROUP, name="ascend_megakernel"))
            if len(choices) != 1:
                raise RuntimeError(
                    f"Expected one installed {GROUP} provider, got {len(choices)}"
                )
            spec = choices[0].load()
            if not isinstance(spec, ModelSpec) or spec.version != 2:
                raise RuntimeError("Incompatible Ascend model provider")
            role = spec.roles.get(type(model).__name__)
            if role == "nextn" and (
                mode == "native"
                or (mode == "fused_extend" and args.disaggregation_mode != "null")
            ):
                role = None
            if role:
                resources = SimpleNamespace(
                    model=model,
                    token_to_kv_pool=runner.token_to_kv_pool,
                    server_args=args,
                )
                binding = spec.prepare(resources, role, mode)
                if binding is None:
                    role = None
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    decision = (mode, role, type(model).__name__, args.speculative_num_steps, error)
    ranks = [None] * dist.get_world_size(group)
    dist.all_gather_object(ranks, decision, group=group)
    if error or any(other != decision for other in ranks):
        raise RuntimeError(f"Ascend provider preflight failed: {ranks}")
    if binding is not None:
        binding.build(model)
    log.info("Ascend MegaKernel role=%s mode=%s", role or "host", mode)
    return binding


def run_target(binding, batch):
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

    verify = batch.forward_mode.is_target_verify()
    build = binding.bridge.build_target_verify if verify else binding.bridge.build
    info, ids, positions = build(batch, batch.input_ids, batch.positions)
    result = binding.target(info, ids, positions, None, None)
    logits = getattr(result, "next_token_logits", result)
    # BlockRT reuses graph output buffers. The overlap worker owns these copies.
    hidden = binding.runtime.last_hidden_states.clone() if verify else None
    return LogitsProcessorOutput(next_token_logits=logits.clone(), hidden_states=hidden)
