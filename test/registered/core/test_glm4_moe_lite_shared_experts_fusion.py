from types import SimpleNamespace

import sglang.srt.models.glm4_moe_lite as glm4_moe_lite_mod


def test_disable_shared_experts_fusion_for_ignored_shared_experts(monkeypatch):
    monkeypatch.setattr(glm4_moe_lite_mod, "_is_cuda", True)
    monkeypatch.setattr(
        glm4_moe_lite_mod.torch.cuda, "get_device_capability", lambda *_: (8, 6)
    )
    monkeypatch.setattr(
        glm4_moe_lite_mod, "get_moe_expert_parallel_world_size", lambda: 1
    )

    server_args = SimpleNamespace(disable_shared_experts_fusion=False)
    monkeypatch.setattr(
        glm4_moe_lite_mod, "get_global_server_args", lambda: server_args
    )

    model = object.__new__(glm4_moe_lite_mod.Glm4MoeLiteForCausalLM)
    model.config = SimpleNamespace(
        architectures=["Glm4MoeLiteForCausalLM"],
        n_shared_experts=1,
    )
    model.quant_config = SimpleNamespace(
        ignore=["model.layers.1.mlp.shared_experts.gate_proj"]
    )

    model.determine_num_fused_shared_experts()

    assert model.num_fused_shared_experts == 0
    assert server_args.disable_shared_experts_fusion is True
