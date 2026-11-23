"""The GLM-5.3-Flash target and draft may share a process with other MoE layers."""

import pytest
import torch

from sglang.srt.configs.glm5_next import Glm5NextTextConfig
from sglang.srt.configs.shared_experts import SharedExpertsConfig
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.models import deepseek_v2, glm5_next
from sglang.srt.runtime_context import get_context, get_flags, get_model, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def test_only_glm5_next_target_experts_opt_in(monkeypatch):
    # Build real expert layers. Attention and its stage boundary aren't needed.
    class Attention(torch.nn.Identity):
        prepare_qkv_latent = None

    monkeypatch.setattr(glm5_next, "Glm5NextLinearAttention", Attention)
    monkeypatch.setattr(glm5_next, "DeepseekV2AttentionMLA", Attention)
    monkeypatch.setattr(glm5_next, "append_stages", lambda *args: (None, None))
    config = Glm5NextTextConfig(
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=128,
        n_routed_experts=4,
        n_shared_experts=0,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=0,
        num_hidden_layers=3,
        mhc=False,
    )
    with (
        get_context().override_server_args(model_path="dummy", device="cpu"),
        get_flags().moe.override(
            a2a_backend=MoeA2ABackend.NONE, runner_backend=MoeRunnerBackend.AUTO
        ),
        get_model().override(
            shared_experts=SharedExpertsConfig(num_fused_shared_experts=0)
        ),
        get_parallel().override(
            tp_size=1,
            tp_rank=0,
            attn_tp_size=1,
            attn_tp_rank=0,
            moe_tp_size=1,
            moe_tp_rank=0,
            moe_ep_size=1,
            moe_ep_rank=0,
        ),
    ):

        def enabled(**kwargs):
            layer = glm5_next.Glm5NextDecoderLayer(config, 0, **kwargs)
            return layer.mlp.experts.moe_runner_config.enable_glm5_next_moe_kernels

        assert enabled()
        assert not enabled(is_nextn=True)
        with get_flags().moe.override(in_speculative_scope=True):
            assert not enabled()
        # Same underlying MoE and even the same config: the caller determines
        # the route, not the HF config, expert dimensions, or a global flag.
        other = deepseek_v2.DeepseekV2MoE(config, 0)
        assert not other.experts.moe_runner_config.enable_glm5_next_moe_kernels
        assert enabled()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
