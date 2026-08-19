"""Static correctness policy for the experimental Marlin MoE-LoRA path.

Keep these checks free of CUDA imports so they can be covered by CPU-only unit
tests.  The backend deliberately supports a narrow configuration: widening it
requires implementing the corresponding Marlin activation/EP semantics first.
"""

from __future__ import annotations

from typing import Any


def validate_experimental_sgl_marlin_server_args(
    server_args: Any, resolved_args: Any
) -> None:
    """Validate startup options before the experimental runner is constructed."""

    if resolved_args.ep_size > 1 and resolved_args.moe_a2a_backend != "none":
        raise ValueError("experimental_sgl_marlin EP requires --moe-a2a-backend none")

    # A provided adapter path implicitly enables LoRA later unless it was
    # explicitly disabled. No-LoRA delegates to the stock Marlin fused path.
    lora_enabled = bool(server_args.enable_lora) or (
        server_args.enable_lora is None and bool(server_args.lora_paths)
    )
    if not lora_enabled:
        return

    if not server_args.lora_use_virtual_experts:
        raise ValueError(
            "experimental_sgl_marlin LoRA requires --lora-use-virtual-experts"
        )
    if server_args.lora_backend != "triton":
        # The temporary dense/sink kernels consume Triton SGEMM batch metadata
        # directly; other global backends are not adapted in this tree.
        raise ValueError("experimental_sgl_marlin LoRA requires --lora-backend triton")
    if resolved_args.ep_size <= 1:
        return

    if (
        server_args.init_expert_location != "trivial"
        or server_args.ep_num_redundant_experts != 0
        or server_args.enable_eplb
        or server_args.elastic_ep_backend is not None
        or server_args.enable_elastic_expert_backup
        or server_args.elastic_ep_rejoin
    ):
        raise ValueError(
            "experimental_sgl_marlin EP requires trivial expert placement "
            "without redundancy, EPLB, or elastic EP"
        )


def validate_experimental_sgl_marlin_contract(
    runner_config: Any,
    *,
    moe_ep_size: int,
    device_capability: tuple[int, int],
) -> None:
    """Fail before capture when the specialized pipeline would change semantics."""

    errors: list[str] = []

    if runner_config.activation != "silu":
        errors.append(f"activation must be 'silu', got {runner_config.activation!r}")
    if not runner_config.is_gated:
        errors.append("only gated SwiGLU MoE is supported")
    if runner_config.gemm1_alpha is not None:
        errors.append("gemm1_alpha is not supported")
    if runner_config.gemm1_clamp_limit is not None:
        errors.append("gemm1_clamp_limit is not supported")
    if runner_config.swiglu_limit is not None:
        errors.append("swiglu_limit is not supported")
    if runner_config.apply_router_weight_on_input:
        errors.append("apply_router_weight_on_input must be false")
    if runner_config.no_combine:
        errors.append("no_combine must be false")

    num_experts = runner_config.num_experts
    num_local_experts = runner_config.num_local_experts
    if moe_ep_size < 1:
        errors.append(f"moe_ep_size must be positive, got {moe_ep_size}")
    elif num_experts is not None and num_local_experts is not None:
        if num_experts % moe_ep_size != 0 or num_local_experts != (
            num_experts // moe_ep_size
        ):
            errors.append(
                "num_local_experts must equal num_experts / moe_ep_size, got "
                f"{num_local_experts}, {num_experts}, and {moe_ep_size}"
            )

    if device_capability[0] < 9:
        errors.append(
            "CUDA compute capability 9.0 or newer is required, "
            f"got {device_capability[0]}.{device_capability[1]}"
        )

    if errors:
        raise ValueError(
            "experimental_sgl_marlin configuration is unsupported: " + "; ".join(errors)
        )


def use_post_reduce_down_delta(
    *, run_lora: bool, routed_scaling_factor: float, num_tokens: int
) -> bool:
    """Whether the down delta may be accumulated after the base top-k reduce."""

    return run_lora and routed_scaling_factor == 1.0 and num_tokens <= 2048
