"""Unit tests for ParameterMapper."""

from types import SimpleNamespace

import pytest

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.model_loader.parameter_mapper import ParameterMapper

_DEEPSEEK_N_ROUTED = 4
_DEEPSEEK_N_LOCAL = _DEEPSEEK_N_ROUTED + 1  # +1 fused shared expert
_QWEN3MOE_N = 4
_GLM4LITE_N_ROUTED = 4
_GLM4LITE_N_LOCAL = _GLM4LITE_N_ROUTED + 1  # +1 fused shared expert


def _make_model(**kwargs):
    """Create a stub model object for ParameterMapper.from_model()."""
    return SimpleNamespace(**kwargs)


def _deepseek_mutate(name):
    if "mlp.shared_experts" in name:
        return name.replace("mlp.shared_experts", f"mlp.experts.{_DEEPSEEK_N_ROUTED}")
    return name


def _deepseek_scale_remap(name):
    for s in ["k_scale", "v_scale"]:
        if s in name:
            return name.replace(f"{s[0]}_proj", "attn_mqa")
    return name


def _glm4lite_mutate(name):
    if "mlp.shared_experts" in name:
        return name.replace("mlp.shared_experts", f"mlp.experts.{_GLM4LITE_N_ROUTED}")
    return name


_LLAMA_SCALE_PATTERNS = [
    (".activation_scale", ".activation_scale", ".input_scale"),
    (".weight_scale_inv", ".weight_scale_inv", ".weight_scale"),
]


def _llama_scale_remap(name):
    for suffix, pattern, replacement in _LLAMA_SCALE_PATTERNS:
        if name.endswith(suffix) and pattern in name:
            return name.replace(pattern, replacement)
    return name


@pytest.fixture
def qwen_mapper():
    """Qwen2/Qwen3 (dense): QKV fusion, gate/up fusion, no experts."""
    return ParameterMapper.from_model(
        _make_model(
            stacked_params_mapping=[
                ("qkv_proj", "q_proj", "q"),
                ("qkv_proj", "k_proj", "k"),
                ("qkv_proj", "v_proj", "v"),
                ("gate_up_proj", "gate_proj", 0),
                ("gate_up_proj", "up_proj", 1),
            ],
        )
    )


@pytest.fixture
def llama_mapper():
    """Llama/GLM4 (dense): dot-prefixed stacked params, custom scale remap."""
    return ParameterMapper.from_model(
        _make_model(
            stacked_params_mapping=[
                (".qkv_proj", ".q_proj", "q"),
                (".qkv_proj", ".k_proj", "k"),
                (".qkv_proj", ".v_proj", "v"),
                (".gate_up_proj", ".gate_proj", 0),
                (".gate_up_proj", ".up_proj", 1),
            ],
            custom_scale_remap=_llama_scale_remap,
        )
    )


@pytest.fixture
def qwen3moe_mapper():
    """Qwen3-MoE: QKV fusion + experts, no shared expert fusion."""
    return ParameterMapper.from_model(
        _make_model(
            stacked_params_mapping=[
                ("qkv_proj", "q_proj", "q"),
                ("qkv_proj", "k_proj", "k"),
                ("qkv_proj", "v_proj", "v"),
                ("gate_up_proj", "gate_proj", 0),
                ("gate_up_proj", "up_proj", 1),
            ],
            expert_params_mapping=FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=_QWEN3MOE_N,
            ),
        )
    )


@pytest.fixture
def deepseek_mapper():
    """DeepSeek V2/V3: MLA A-proj fusion, shared expert fusion, custom scale remap."""
    return ParameterMapper.from_model(
        _make_model(
            stacked_params_mapping=[
                ("gate_up_proj", "gate_proj", 0),
                ("gate_up_proj", "up_proj", 1),
                ("fused_qkv_a_proj_with_mqa", "q_a_proj", 0),
                ("fused_qkv_a_proj_with_mqa", "kv_a_proj_with_mqa", 1),
            ],
            expert_params_mapping=FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=_DEEPSEEK_N_LOCAL,
            ),
            mutate_weight_preload=_deepseek_mutate,
            custom_scale_remap=_deepseek_scale_remap,
        )
    )


@pytest.fixture
def glm4lite_mapper():
    """GLM4-MoE-Lite (GLM-4.7): QKV fusion, MLA A-proj fusion, shared expert fusion, custom scale remap."""
    return ParameterMapper.from_model(
        _make_model(
            stacked_params_mapping=[
                ("qkv_proj", "q_proj", "q"),
                ("qkv_proj", "k_proj", "k"),
                ("qkv_proj", "v_proj", "v"),
                ("gate_up_proj", "gate_proj", 0),
                ("gate_up_proj", "up_proj", 1),
                ("fused_qkv_a_proj_with_mqa", "q_a_proj", 0),
                ("fused_qkv_a_proj_with_mqa", "kv_a_proj_with_mqa", 1),
            ],
            expert_params_mapping=FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=_GLM4LITE_N_LOCAL,
            ),
            mutate_weight_preload=_glm4lite_mutate,
            custom_scale_remap=_deepseek_scale_remap,
        )
    )


# ── Helpers ──────────────────────────────────────────────────────────────────


def to_expect(name, shard=None, n=1, expert=None, n_exp=None):
    """Shorthand for expected MappingResult fields."""
    return (name, shard, n, expert, n_exp)


def _assert(mapper, ckpt, expected):
    r = mapper.map(ckpt)
    name, shard, n, expert, n_exp = expected
    assert (
        r.sglang_name,
        r.shard_id,
        r.num_shards,
        r.expert_id,
        r.num_local_experts,
    ) == (name, shard, n, expert, n_exp), f"map({ckpt!r}) = {r}"


# ── Tests ────────────────────────────────────────────────────────────────────

# fmt: off
_QWEN_CASES = [
    # QKV fusion (Qwen2, Qwen3, GLM4-MoE)
    ("layers.0.attn.q_proj.weight",     to_expect("layers.0.attn.qkv_proj.weight", "q", 3)),
    ("layers.0.attn.k_proj.weight",     to_expect("layers.0.attn.qkv_proj.weight", "k", 3)),
    ("layers.0.attn.v_proj.weight",     to_expect("layers.0.attn.qkv_proj.weight", "v", 3)),
    # Gate/Up fusion
    ("layers.0.mlp.gate_proj.weight",   to_expect("layers.0.mlp.gate_up_proj.weight", 0, 2)),
    ("layers.0.mlp.up_proj.weight",     to_expect("layers.0.mlp.gate_up_proj.weight", 1, 2)),
    # Pass-through
    ("layers.0.mlp.down_proj.weight",   to_expect("layers.0.mlp.down_proj.weight")),
    ("embed_tokens.weight",             to_expect("embed_tokens.weight")),
    # Standard scale remap (no custom_scale_remap)
    ("model.layers.0.self_attn.k_scale", to_expect("model.layers.0.self_attn.attn.k_scale")),
    ("model.layers.0.self_attn.v_scale", to_expect("model.layers.0.self_attn.attn.v_scale")),
]

_LLAMA_CASES = [
    # Dot-prefixed QKV fusion (Llama, GLM4)
    ("model.layers.0.self_attn.q_proj.weight", to_expect("model.layers.0.self_attn.qkv_proj.weight", "q", 3)),
    ("model.layers.0.self_attn.k_proj.weight", to_expect("model.layers.0.self_attn.qkv_proj.weight", "k", 3)),
    # Dot-prefixed gate/up
    ("model.layers.0.mlp.gate_proj.weight",    to_expect("model.layers.0.mlp.gate_up_proj.weight", 0, 2)),
    # Llama-specific scale remap + stacked (scales follow their weights)
    ("model.layers.0.mlp.gate_proj.activation_scale", to_expect("model.layers.0.mlp.gate_up_proj.input_scale", 0, 2)),
    ("model.layers.0.mlp.gate_proj.weight_scale_inv", to_expect("model.layers.0.mlp.gate_up_proj.weight_scale", 0, 2)),
    # Pass-through
    ("model.layers.0.mlp.down_proj.weight",    to_expect("model.layers.0.mlp.down_proj.weight")),
]

_QWEN3MOE_CASES = [
    # QKV fusion
    ("model.layers.0.self_attn.q_proj.weight",          to_expect("model.layers.0.self_attn.qkv_proj.weight", "q", 3)),
    # Expert mapping (no shared expert fusion)
    ("model.layers.0.mlp.experts.0.gate_proj.weight",   to_expect("model.layers.0.mlp.experts.w13_weight", "w1", 2, 0, _QWEN3MOE_N)),
    ("model.layers.0.mlp.experts.3.down_proj.weight",   to_expect("model.layers.0.mlp.experts.w2_weight", "w2", 1, 3, _QWEN3MOE_N)),
    # shared_experts falls through to stacked mapping (no mutate_weight_preload)
    ("model.layers.0.mlp.shared_experts.gate_proj.weight", to_expect("model.layers.0.mlp.shared_experts.gate_up_proj.weight", 0, 2)),
]

_DEEPSEEK_CASES = [
    # MLA A-proj fusion
    ("model.layers.0.self_attn.q_a_proj.weight",            to_expect("model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight", 0, 2)),
    ("model.layers.0.self_attn.kv_a_proj_with_mqa.weight",  to_expect("model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight", 1, 2)),
    # Shared expert fusion via mutate_weight_preload
    ("model.layers.0.mlp.shared_experts.gate_proj.weight",   to_expect("model.layers.0.mlp.experts.w13_weight", "w1", 2, _DEEPSEEK_N_ROUTED, _DEEPSEEK_N_LOCAL)),
    ("model.layers.0.mlp.shared_experts.down_proj.weight",   to_expect("model.layers.0.mlp.experts.w2_weight", "w2", 1, _DEEPSEEK_N_ROUTED, _DEEPSEEK_N_LOCAL)),
    # Custom scale remap (k_proj/v_proj -> attn_mqa, NOT double-remapped)
    ("model.layers.0.self_attn.k_proj.k_scale",             to_expect("model.layers.0.self_attn.attn_mqa.k_scale")),
    ("model.layers.0.self_attn.v_proj.v_scale",             to_expect("model.layers.0.self_attn.attn_mqa.v_scale")),
    # kv_b_proj pass-through (decomposed in post_load_weights)
    ("model.layers.0.self_attn.kv_b_proj.weight",           to_expect("model.layers.0.self_attn.kv_b_proj.weight")),
]

_GLM4LITE_CASES = [
    # QKV fusion (GLM-4.7 uses standard QKV unlike DeepSeek which uses MLA-only)
    ("model.layers.0.self_attn.q_proj.weight",              to_expect("model.layers.0.self_attn.qkv_proj.weight", "q", 3)),
    ("model.layers.0.self_attn.k_proj.weight",              to_expect("model.layers.0.self_attn.qkv_proj.weight", "k", 3)),
    ("model.layers.0.self_attn.v_proj.weight",              to_expect("model.layers.0.self_attn.qkv_proj.weight", "v", 3)),
    # MLA A-proj fusion (GLM-4.7 also uses MLA with q_lora_rank)
    ("model.layers.0.self_attn.q_a_proj.weight",            to_expect("model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight", 0, 2)),
    ("model.layers.0.self_attn.kv_a_proj_with_mqa.weight",  to_expect("model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight", 1, 2)),
    # Gate/Up fusion (non-expert layers)
    ("model.layers.0.mlp.gate_proj.weight",                 to_expect("model.layers.0.mlp.gate_up_proj.weight", 0, 2)),
    ("model.layers.0.mlp.up_proj.weight",                   to_expect("model.layers.0.mlp.gate_up_proj.weight", 1, 2)),
    # Expert mapping
    ("model.layers.0.mlp.experts.0.gate_proj.weight",       to_expect("model.layers.0.mlp.experts.w13_weight", "w1", 2, 0, _GLM4LITE_N_LOCAL)),
    ("model.layers.0.mlp.experts.0.up_proj.weight",         to_expect("model.layers.0.mlp.experts.w13_weight", "w3", 2, 0, _GLM4LITE_N_LOCAL)),
    ("model.layers.0.mlp.experts.3.down_proj.weight",       to_expect("model.layers.0.mlp.experts.w2_weight", "w2", 1, 3, _GLM4LITE_N_LOCAL)),
    # Shared expert fusion via mutate_weight_preload
    ("model.layers.0.mlp.shared_experts.gate_proj.weight",  to_expect("model.layers.0.mlp.experts.w13_weight", "w1", 2, _GLM4LITE_N_ROUTED, _GLM4LITE_N_LOCAL)),
    ("model.layers.0.mlp.shared_experts.down_proj.weight",  to_expect("model.layers.0.mlp.experts.w2_weight", "w2", 1, _GLM4LITE_N_ROUTED, _GLM4LITE_N_LOCAL)),
    # Custom scale remap (same as DeepSeek: k_proj/v_proj -> attn_mqa)
    ("model.layers.0.self_attn.k_proj.k_scale",             to_expect("model.layers.0.self_attn.attn_mqa.k_scale")),
    ("model.layers.0.self_attn.v_proj.v_scale",             to_expect("model.layers.0.self_attn.attn_mqa.v_scale")),
    # Pass-through
    ("model.layers.0.mlp.down_proj.weight",                 to_expect("model.layers.0.mlp.down_proj.weight")),
    ("model.layers.0.self_attn.kv_b_proj.weight",           to_expect("model.layers.0.self_attn.kv_b_proj.weight")),
]
# fmt: on


@pytest.mark.parametrize("ckpt,expected", _QWEN_CASES, ids=[c[0] for c in _QWEN_CASES])
def test_qwen(qwen_mapper, ckpt, expected):
    _assert(qwen_mapper, ckpt, expected)


@pytest.mark.parametrize(
    "ckpt,expected", _LLAMA_CASES, ids=[c[0] for c in _LLAMA_CASES]
)
def test_llama(llama_mapper, ckpt, expected):
    _assert(llama_mapper, ckpt, expected)


@pytest.mark.parametrize(
    "ckpt,expected", _QWEN3MOE_CASES, ids=[c[0] for c in _QWEN3MOE_CASES]
)
def test_qwen3moe(qwen3moe_mapper, ckpt, expected):
    _assert(qwen3moe_mapper, ckpt, expected)


@pytest.mark.parametrize(
    "ckpt,expected", _DEEPSEEK_CASES, ids=[c[0] for c in _DEEPSEEK_CASES]
)
def test_deepseek(deepseek_mapper, ckpt, expected):
    _assert(deepseek_mapper, ckpt, expected)


@pytest.mark.parametrize(
    "ckpt,expected", _GLM4LITE_CASES, ids=[c[0] for c in _GLM4LITE_CASES]
)
def test_glm4lite(glm4lite_mapper, ckpt, expected):
    _assert(glm4lite_mapper, ckpt, expected)
