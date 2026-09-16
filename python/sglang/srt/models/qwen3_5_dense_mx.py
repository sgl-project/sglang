"""Qwen3.5's dense-MX policy: which bf16 layers quark may take to MXFP6, on ROCm.

An MXFP4 Qwen3.5 checkpoint quantizes only its routed experts, so every attention,
GDN and shared-expert projection arrives bf16. Only these four are worth converting
-- they carry most of that work and measure ~1.9x on aiter's MXFP6 GEMM once a
forward pass is wide enough to amortize quantizing the activation. The rest lose or
barely break even: the narrow GDN ``b``/``a`` projections, the routing gates,
``conv1d``, embeddings, and ``shared_expert.down_proj`` (whose own win comes from
fusing silu+mul into the quant, not from the GEMM).

The names and the threshold are aiter-tuned, so they live with the model rather
than in quark.

A note on why the format matters more than the layer list here. Measured on gsm8k
against a 0.936 bf16 base, MXFP4 on these same four projections scored 0.851 with
8.9% of outputs unparseable, and a per-projection ablation put nearly all of that
on ``qkv_proj``: -3.0 points alone, -5.3 marginal on top of the other three, and
the sole cause of the invalid-output spike. Its doubled Q block is half attention
output gate under ``attn_output_gate``, so the error lands on a gate rather than a
plain projection -- while the GDN ``z`` gate inside ``in_proj_qkvz`` turned out
almost free. MXFP6's ~4% relative error against MXFP4's ~16% is what makes all
four safe (0.937, i.e. no regression), which is why ``--dense-mx-format`` defaults
to mxfp6 and mxfp4 is kept only so the trade stays measurable.

:func:`register` is a no-op off aiter or on a non-quark checkpoint, and the policy
it hands over does nothing unless ``--enable-dense-mx`` is set, which is not the
default.
"""

from __future__ import annotations

from typing import Optional

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.utils import get_bool_env_var, is_hip

_USE_AITER = get_bool_env_var("SGLANG_USE_AITER") and is_hip()

_INCLUDE = (
    ".in_proj_qkvz",
    ".out_proj",
    ".qkv_proj",
    ".o_proj",
)
_EXCLUDE = (
    "conv1d",
    "shared_expert_gate",
    "shared_expert.down_proj",
    "mlp.gate",
    ".in_proj_a",
    ".in_proj_b",
    ".in_proj_ba",
    "lm_head",
    "embed",
)
# Backstop behind the include list: every projection above is >= 4096 wide per rank
# at TP2, so anything narrower is a name collision rather than a target.
_MIN_OUTPUT_SIZE = 2048
# Below this many tokens in a forward pass the GEMM is not compute-bound, the extra
# activation quantization dominates, and decode stays bit-identical to bf16.
_MIN_TOKENS = 1024


def register(quant_config: Optional[QuantizationConfig]) -> None:
    """Hand this model's policy to quark, which decides layer by layer from there."""
    if not _USE_AITER or quant_config is None:
        return

    from sglang.srt.layers.quantization.quark.quark import QuarkConfig

    if not isinstance(quant_config, QuarkConfig):
        return

    from sglang.srt.layers.quantization.quark import dense_mx

    dense_mx.register(
        quant_config,
        include=_INCLUDE,
        exclude=_EXCLUDE,
        min_output_size=_MIN_OUTPUT_SIZE,
        min_tokens=_MIN_TOKENS,
    )
