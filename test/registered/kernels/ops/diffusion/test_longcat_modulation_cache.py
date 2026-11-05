"""CFG timestep modulation reuse and request/capture fallbacks."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import sglang.multimodal_gen.runtime.layers.linear as linear
import sglang.multimodal_gen.runtime.models.dits.longcat_image as longcat
import torch
import torch.nn as nn
import torch.nn.functional as F
from sglang.multimodal_gen.configs.pipeline_configs.longcat_image import (
    LongCatImagePipelineConfig,
)
from sglang.multimodal_gen.runtime.managers.forward_context import (
    DENOISING_LOOP_STATE,
    set_forward_context,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@torch.inference_mode()
def test_wrapped_projection_does_not_reuse_cached_modulation():
    norm = longcat._LongCatAdaLayerNormZeroSingle(16)
    emb = torch.randn(1, 16)
    cache = {}
    first = longcat._longcat_modulation(norm, emb, cache)
    assert torch.equal(first, norm.linear(norm.silu(emb)))
    norm.linear = nn.Sequential(norm.linear, nn.Identity())
    emb.add_(1)
    assert torch.equal(
        longcat._longcat_modulation(norm, emb, cache), norm.linear(norm.silu(emb))
    )


class _SDPA(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, q, k, v, **kwargs):
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_model_reuses_only_current_cfg_step_and_falls_back_for_capture():
    torch.manual_seed(42)
    config = LongCatImagePipelineConfig()
    arch = config.dit_config.arch_config
    arch.num_layers = arch.num_single_layers = 1
    arch.num_attention_heads = 2
    arch.joint_attention_dim = 32
    with (
        patch.object(longcat, "get_tp_world_size", return_value=1),
        patch.object(
            linear,
            "get_tp_group",
            return_value=SimpleNamespace(all_gather=lambda value, dim: value),
        ),
        patch.object(linear, "get_group_size", return_value=1),
        patch.object(linear, "get_group_rank", return_value=0),
        patch.object(longcat, "USPAttention", _SDPA),
    ):
        model = (
            longcat.LongCatImageTransformer2DModel(config.dit_config).cuda().bfloat16()
        )
    for parameter in model.parameters():
        parameter.normal_(std=0.02)
    norms = [
        model.transformer_blocks[0].norm1,
        model.transformer_blocks[0].norm1_context,
        model.single_transformer_blocks[0].norm,
    ]
    calls = []
    handles = [
        norm.linear.register_forward_hook(lambda *args: calls.append(1))
        for norm in norms
    ]
    x = torch.randn(1, 8, 64, device="cuda", dtype=torch.bfloat16)
    texts = [
        torch.randn(1, 4, 32, device="cuda", dtype=torch.bfloat16) for _ in range(2)
    ]
    img_ids = torch.arange(24, device="cuda").reshape(8, 3).float()
    txt_ids = torch.zeros(4, 3, device="cuda")
    timestep = torch.tensor([500], device="cuda", dtype=torch.bfloat16)
    server = SimpleNamespace(
        enable_breakable_cuda_graph=False, enable_cfg_parallel=False
    )
    batch = SimpleNamespace(do_classifier_free_guidance=True, extra={})

    def run(text):
        return model(
            x,
            encoder_hidden_states=text,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
        )

    with patch.object(longcat, "get_global_server_args", return_value=server):
        for step, value in [(0, 500), (1, 480), (0, 600)]:
            if value == 600:
                batch = SimpleNamespace(do_classifier_free_guidance=True, extra={})
            timestep.fill_(value)
            expected = [run(text) for text in texts]
            calls.clear()
            with set_forward_context(step, None, batch):
                actual = [run(text) for text in texts]
            assert len(calls) == 3
            assert all(torch.equal(a, b) for a, b in zip(actual, expected))
            state = batch.extra[DENOISING_LOOP_STATE]["longcat_modulations"][id(model)]
            assert state[0][0] == step and len(state[1]) == 3

        for flag in ["enable_breakable_cuda_graph", "enable_cfg_parallel"]:
            setattr(server, flag, True)
            calls.clear()
            with set_forward_context(0, None, batch):
                assert torch.equal(run(texts[0]), expected[0])
            assert len(calls) == 3
            setattr(server, flag, False)
        model.layerwise_offload_managers = [object()]
        with set_forward_context(0, None, batch):
            assert model._modulation_cache(torch.zeros(1, 256, device="cuda")) is None
        model.layerwise_offload_managers = []
        graph = torch.cuda.CUDAGraph()
        with set_forward_context(0, None, batch), torch.cuda.graph(graph):
            captured = run(texts[0])
        timestep.fill_(400)
        graph.replay()
        assert torch.equal(captured, run(texts[0]))
    for handle in handles:
        handle.remove()
