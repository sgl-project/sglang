"""Translate gated MoE activations into Humming CUDA C++ expressions."""

import math

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig


def get_humming_activation(
    config: MoeRunnerConfig, dtype: torch.dtype
) -> dict[str, str]:
    """Build a gated CUDA expression with materialized FP16/BF16 rounding."""
    if not config.is_gated:
        raise ValueError("Humming MoE input processing requires a gated activation")
    if dtype == torch.bfloat16:
        cast = "__bfloat162float(__float2bfloat16_rn({}))"
    elif dtype == torch.float16:
        cast = "__half2float(__float2half_rn({}))"
    else:
        raise ValueError(f"Unsupported Humming activation dtype: {dtype}")

    activation_type = "binary_split"
    if config.activation_impl is not None:
        if config.activation_type not in ("binary_split", "binary_interleaved"):
            raise ValueError(
                "A custom Humming activation requires binary_split or "
                "binary_interleaved activation_type"
            )
        activation_type = config.activation_type
        expression = config.activation_impl
    else:
        if config.activation_type is not None:
            raise ValueError("activation_type requires a custom activation_impl")
        if config.activation == "silu":
            gate, up = "a", "b"
            if config.swiglu_limit is not None:
                if not math.isfinite(config.swiglu_limit):
                    raise ValueError("Humming swiglu_limit must be finite")
                # Triton rounds a same-kind scalar to the input dtype before clamp.
                limit = cast.format(f"{float(config.swiglu_limit)!r}f")
                gate = f"fminf(a, {limit})"
                up = f"fmaxf(fminf(b, {limit}), -({limit}))"
            activated = f"({gate}) / (1.0f + expf(-({gate})))"
            if config.swiglu_limit is not None:
                # The clamped Triton path also rounds SiLU before multiplying by up.
                activated = cast.format(activated)
            expression = f"({activated}) * ({up})"
        elif config.activation == "gelu":
            expression = "(a * (0.5f * (1.0f + erff(a * 0.7071067811865476f)))) * b"
        elif config.activation == "situ":
            beta = config.gemm1_alpha
            if beta is None:
                beta = 4.0
            if not math.isfinite(beta) or beta <= 0:
                raise ValueError("Humming SiTU requires a finite positive gemm1_alpha")
            linear_beta = config.gemm1_clamp_limit
            up = "b"
            if linear_beta is not None:
                if not math.isfinite(linear_beta) or linear_beta <= 0:
                    raise ValueError(
                        "Humming SiTU requires a finite positive gemm1_clamp_limit"
                    )
                up = f"({float(linear_beta)!r}f * tanhf(b / {float(linear_beta)!r}f))"
            expression = (
                f"({float(beta)!r}f * tanhf(a / {float(beta)!r}f)"
                f" / (1.0f + expf(-a))) * ({up})"
            )
        else:
            raise ValueError(f"Unsupported Humming activation: {config.activation}")

    return {
        "activation_type": activation_type,
        "activation_impl": cast.format(expression),
    }
