"""MiniMax-M3 mono spec (gfx950, TP4) -- skeleton: shapes, widths, layer modes and refusals are declared; the kernels
(ROCm/ATOM atom/models/minimax_m3/mono, built on the same L0 as sglang.srt.layers.mono.common) are not vendored yet,
so enabling refuses at model load with that reason. Its contract differs from DeepSeek-V4.1's on purpose: whole-layer
launches carrying the plain (hidden, residual) pair, at most 16 rows, TP4 only.
"""

from __future__ import annotations

from sglang.srt.layers.mono.spec import Feature, LayerMode, MonoSpec

MAX_ROWS = 16  # tokens one step serves (the MFMA B operand holds 16)
HIDDEN = 6144
DENSE_LAYERS = (
    3  # layers 0..2: dense MLP, split around the model's own attention kernel
)


def _text_config(hf_config):
    return getattr(hf_config, "text_config", None) or hf_config


def layers(hf_config) -> dict[int, LayerMode]:
    return {
        i: LayerMode.WHOLE for i in range(_text_config(hf_config).num_hidden_layers)
    }


def refuse_config(cfg, hf_config) -> list[str]:
    t = _text_config(hf_config)
    want = dict(
        hidden_size=HIDDEN,
        num_hidden_layers=60,
        num_local_experts=128,
        num_experts_per_tok=4,
        num_attention_heads=64,
        num_key_value_heads=4,
        head_dim=128,
        scoring_func="sigmoid",
    )
    return [
        f"config {k}={getattr(t, k, None)!r}, the kernels are built for {v!r}"
        for k, v in want.items()
        if getattr(t, k, None) != v
    ]


def refuse_model(model) -> list[str]:
    return [
        "the MiniMax-M3 mono kernels are not vendored into sglang yet (planned: P2)"
    ]


MONO_SPEC = MonoSpec(
    name="minimax_m3",
    architectures=(
        "MiniMaxM3SparseForCausalLM",
        "MiniMaxM3SparseForConditionalGeneration",
    ),
    layers=layers,
    widths=tuple(range(1, MAX_ROWS + 1)),
    supports=frozenset({Feature.SPEC_VERIFY}),
    supported_tp=(4,),
    refuse_config=refuse_config,
    refuse_model=refuse_model,
)
