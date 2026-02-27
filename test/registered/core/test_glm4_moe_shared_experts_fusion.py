from types import SimpleNamespace

import sglang.srt.models.glm4_moe as glm4_moe_mod


def test_disable_shared_experts_fusion_for_ignored_shared_experts(monkeypatch):
    monkeypatch.setattr(glm4_moe_mod, "_is_cuda", True)
    monkeypatch.setattr(glm4_moe_mod, "_device_sm", 90)
    monkeypatch.setattr(glm4_moe_mod, "get_moe_expert_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        glm4_moe_mod,
        "get_moe_a2a_backend",
        lambda: SimpleNamespace(is_deepep=lambda: False),
    )

    server_args = SimpleNamespace(disable_shared_experts_fusion=False)
    monkeypatch.setattr(glm4_moe_mod, "get_global_server_args", lambda: server_args)

    model = object.__new__(glm4_moe_mod.Glm4MoeForCausalLM)
    model.config = SimpleNamespace(n_shared_experts=1)
    model.quant_config = SimpleNamespace(
        ignore=["model.layers.1.mlp.shared_experts.gate_proj"]
    )
    model.num_fused_shared_experts = 0

    model.determine_num_fused_shared_experts()

    assert model.num_fused_shared_experts == 0
    assert server_args.disable_shared_experts_fusion is True
