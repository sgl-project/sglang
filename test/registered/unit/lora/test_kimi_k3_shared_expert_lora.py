from types import SimpleNamespace

import torch

from sglang.srt.lora.mem_pool import _has_shared_experts, _set_expert_weight
from sglang.srt.lora.utils import get_hidden_dim
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _config():
    return SimpleNamespace(
        hidden_size=7168,
        intermediate_size=33792,
        num_attention_heads=96,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        moe_intermediate_size=3072,
        num_shared_experts=2,
        routed_expert_hidden_size=3584,
    )


def test_kimi_lora_dimensions_follow_the_k3_moe_config():
    """K3 names its shared-expert count ``num_shared_experts`` (not DeepSeek's
    ``n_shared_experts``) and feeds its routed experts from a 3584-wide latent
    (``routed_expert_hidden_size``) rather than the 7168 hidden size. Reading
    either from the DeepSeek field sizes the adapter buffers wrong, which shows
    up as a shape mismatch deep inside the memory pool at adapter-load time.
    """
    config = _config()
    model = torch.nn.Linear(1, 1)

    assert _has_shared_experts(config)

    # Shared experts: 2 experts x 3072 intermediate, gate+up fused, over hidden.
    assert get_hidden_dim("gate_up_proj", config, model, layer_idx=1) == (
        7168,
        2 * 3072 * 2,
    )
    assert get_hidden_dim("down_proj", config, model, layer_idx=1) == (
        3072 * 2,
        7168,
    )

    # Routed experts: same intermediate, but against the 3584 latent.
    assert get_hidden_dim("gate_up_proj_moe", config, model, layer_idx=1) == (
        3584,
        2 * 3072,
    )
    assert get_hidden_dim("down_proj_moe", config, model, layer_idx=1) == (
        3072,
        3584,
    )


def test_shared_gate_up_a_does_not_block_per_expert_b():
    """K3 mixes layouts within one target module: gate_up lora_A is a single
    shared-outer 3D tensor while lora_B is per-expert. The per-expert branch
    used to initialize the A and B dicts together, so loading the first
    per-expert B wiped the already-loaded shared A.
    """
    shared_a = torch.ones(1, 16, 7168)
    expert_b = torch.ones(6144, 16)
    a_buffers = {"gate_up_proj_moe": shared_a}
    b_buffers = {"gate_up_proj_moe": None}
    a_cache_keys = {"gate_up_proj_moe": "shared_a"}
    b_cache_keys = {"gate_up_proj_moe": None}

    _set_expert_weight(
        b_buffers,
        b_cache_keys,
        "gate_up_proj_moe",
        7,
        expert_b,
        "experts.7.gate_up_proj.lora_B",
    )

    assert a_buffers["gate_up_proj_moe"] is shared_a
    assert a_cache_keys["gate_up_proj_moe"] == "shared_a"
    loaded_b = b_buffers["gate_up_proj_moe"]
    loaded_b_cache_keys = b_cache_keys["gate_up_proj_moe"]
    assert isinstance(loaded_b, dict)
    assert isinstance(loaded_b_cache_keys, dict)
    assert loaded_b[7] is expert_b
    assert loaded_b_cache_keys == {7: "experts.7.gate_up_proj.lora_B"}
