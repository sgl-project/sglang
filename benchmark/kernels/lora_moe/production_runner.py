"""Drive the PRODUCTION MoE-LoRA runner as an observed backend of the lab.

Until this existed, the guardrail matrix validated `serial_materialized_control`
— the lab's own reference backend — while the shipping path
(`SglMoeLoraRunner.run`) was covered only by a 4-case pipeline test at one
small shape. Plan section 31.6: every case whose geometry the runner admits
now ALSO runs the production runner, gated by the same signal engine against
the same FP32 reference. A divergence between control and production is
exactly the class of bug nothing else can catch, because each is internally
consistent.

Admission mirrors `SglMoeLoraRunner._admit` semantics, minus the parts that
validate a live `FusedMoE` (quant method, dispatcher wiring) which the lab
constructs directly:

* canonical gated SiLU only — `relu2` cases are control-only;
* any expert-ID domain: `global`-domain cases are localized here with the
  case's own provider map, which is precisely what the production dispatcher
  (``skip_local_expert_mapping == False``) does before the runner sees ids.
"""

from __future__ import annotations

import torch

from benchmark.kernels.lora_moe.cases import CaseTensors, MoeLoraBenchCase

_OUTPUT_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}

_distributed_ready = False


def ensure_single_rank_distributed() -> None:
    """One-time tp=1/ep=1 process-group init for the production runner.

    The runner's finalize allocates through `get_tp_group()` (symmetric-memory
    eligibility) and the dispatcher asserts the expert-parallel group, so even
    a single-GPU lab forward needs the groups to exist — exactly as the
    registered pipeline test sets them up.
    """
    global _distributed_ready
    if _distributed_ready:
        return
    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.utils.network import get_open_port

    init_distributed_environment(
        backend="nccl",
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
    )
    _distributed_ready = True


def production_runner_skip_reason(case: MoeLoraBenchCase) -> str | None:
    """None if the production runner can execute this case; else the reason."""
    if case.activation != "silu_glu":
        return "production runner supports canonical gated SiLU only"
    if "fp32" in (
        case.bridge_gate_a_out,
        case.bridge_gate_up_delta,
        case.bridge_activation_lora_input,
        case.bridge_down_a_out,
        case.bridge_down_delta,
    ):
        return "production runner materializes BF16 bridges only"
    return None


def _localized_topk_ids(tensors: CaseTensors) -> torch.Tensor:
    """Map declared-domain expert ids to EP-local ids with -1 sentinels.

    The production dispatcher performs this localization before the runner
    sees ids (admission requires ``skip_local_expert_mapping == False``), so
    the lab does the same rather than teaching the runner a second domain.
    """
    ids = tensors.topk_ids
    table = tensors.lora_expert_map
    if table is None:
        return ids
    in_map = (ids >= 0) & (ids < table.numel())
    safe = ids.clamp(min=0, max=table.numel() - 1)
    return torch.where(in_map, table[safe.long()].to(ids.dtype), ids.new_full((), -1))


def prepare_production_forward(
    case: MoeLoraBenchCase,
    tensors: CaseTensors,
    *,
    device: torch.device,
    disable_lora: bool = False,
    provider_kwargs: dict | None = None,
):
    """Construct everything one production forward needs; run nothing.

    Returns ``(runner, dispatcher, batch, dispatch_output)``. Split from
    `run_production_runner` so the provider benchmark can hoist construction
    (runner attach compiles the CuTeDSL provider) and time only the
    run+combine forward it wraps in a thunk. ``provider_kwargs`` forwards the
    providers' LAB hooks (expected_m_hint / force_token_width) so a candidate
    arm can be measured at the complete-pipeline boundary; production passes
    nothing here.

    Fails CLOSED at the executor entry (first S3 review): a case this runner
    cannot honor raises here rather than silently executing the grouped/BF16
    path under a different declaration. Candidate executors that intend a
    different semantics run their own entry points, not this one.
    """
    skip_reason = production_runner_skip_reason(case)
    if skip_reason is not None:
        raise ValueError(
            f"case {case.case_id} declares semantics the production runner "
            f"does not execute: {skip_reason}"
        )
    from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatcher
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.lora.sgl_lora.moe_lora_runner import (
        SglMoeLoraBatch,
        SglMoeLoraRunner,
    )
    from sglang.srt.lora.sgl_lora.quant_info import SglLoraBf16QuantInfo

    ensure_single_rank_distributed()
    moved = {
        name: (
            value.to(device)
            if isinstance(value := getattr(tensors, name), torch.Tensor)
            else value
        )
        for name in tensors.__struct_fields__
    }
    dt = CaseTensors(**moved)
    top_k = int(dt.topk_ids.shape[1])

    quant_info = SglLoraBf16QuantInfo(
        w13_weight=dt.w13,
        w2_weight=dt.w2,
        num_local_experts=case.num_experts_local,
        intermediate_size=int(dt.w2.shape[2]),
        hidden_size=int(dt.w2.shape[1]),
    )
    # Through the production selector, so SGLANG_LORA_MOE_BASE_PROVIDER picks
    # the provider the matrix observes (a named class here silently pinned
    # every "provider" run to DeepGEMM).
    runner = SglMoeLoraRunner(
        provider=SglMoeLoraRunner.select_provider_cls()(
            quant_info, **(provider_kwargs or {})
        ),
        top_k=top_k,
        routed_scaling_factor=case.routed_scaling_factor,
    )
    token_slots = dt.token_lora_mapping.to(torch.int32)
    if disable_lora:
        token_slots = torch.full_like(token_slots, -1)
    batch = SglMoeLoraBatch(
        gate_up_lora_a=dt.lora_a_gate_up,
        gate_up_lora_b=dt.lora_b_gate_up,
        down_lora_a=dt.lora_a_down,
        down_lora_b=dt.lora_b_down,
        token_slots=token_slots,
        adapter_enabled=None,
        physical_rank=case.physical_rank,
        shared_outer=case.shared_factor_signature != "per_expert",
    )
    config = MoeRunnerConfig(
        num_experts=case.num_experts_local,
        num_local_experts=case.num_experts_local,
        hidden_size=int(dt.w2.shape[1]),
        intermediate_size_per_partition=int(dt.w2.shape[2]),
        top_k=top_k,
        num_fused_shared_experts=0,
        params_dtype=torch.bfloat16,
        activation="silu",
        is_gated=True,
        apply_router_weight_on_input=False,
        no_combine=False,
        routed_scaling_factor=case.routed_scaling_factor,
    )
    dispatcher = StandardDispatcher(config)
    topk_output = StandardTopKOutput(
        topk_weights=dt.topk_weights,
        topk_ids=_localized_topk_ids(dt),
        router_logits=None,
    )
    dispatch_output = dispatcher.dispatch(dt.hidden_states.clone(), topk_output)
    return runner, dispatcher, batch, dispatch_output


def run_production_runner(
    case: MoeLoraBenchCase,
    tensors: CaseTensors,
    *,
    device: torch.device,
    disable_lora: bool = False,
) -> torch.Tensor:
    """One forward of `SglMoeLoraRunner` on this case; returns ``[T, H]``.

    ``disable_lora`` runs the SAME topology with every token as a base row —
    the matched-base denominator for signal-relative gating. Imports are local
    so the CPU-only suite can import the module without serving dependencies.
    """
    runner, dispatcher, batch, dispatch_output = prepare_production_forward(
        case, tensors, device=device, disable_lora=disable_lora
    )
    combine_input = runner.run(
        dispatch_output, batch, output_dtype=_OUTPUT_DTYPES[case.output_dtype]
    )
    return dispatcher.combine(combine_input)
