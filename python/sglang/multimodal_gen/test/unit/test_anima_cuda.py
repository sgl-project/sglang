# SPDX-License-Identifier: Apache-2.0
"""Small, checkpoint-free numerical contracts against the Cosmos reference."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from diffusers.models.transformers.transformer_cosmos import CosmosTransformer3DModel

import sglang.multimodal_gen.runtime.models.dits.anima as anima
from sglang.multimodal_gen.configs.models.adapter.anima import (
    AnimaTextConditionerArchConfig,
    AnimaTextConditionerConfig,
)
from sglang.multimodal_gen.configs.models.dits.anima import AnimaDiTConfig
from sglang.multimodal_gen.configs.pipeline_configs.anima import AnimaPipelineConfig
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.managers.forward_context import (
    DENOISING_LOOP_STATE,
    set_forward_context,
)
from sglang.multimodal_gen.runtime.models.adapter.anima import AnimaTextConditioner
from sglang.multimodal_gen.runtime.models.dits.anima import AnimaTransformer3DModel
from sglang.multimodal_gen.runtime.server_args import ServerArgs, server_args
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    enable_breakable_cuda_graph,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module", autouse=True)
def cleanup_owned_process_group():
    initialized = torch.distributed.is_initialized()
    yield
    if not initialized:
        cleanup_dist_env_and_memory()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 2])
@torch.no_grad()
def test_transformer_matches_cosmos(dtype, batch_size, monkeypatch):
    kwargs = dict(
        in_channels=4,
        out_channels=4,
        num_attention_heads=4,
        attention_head_dim=32,
        num_layers=2,
        mlp_ratio=2,
        text_embed_dim=32,
        adaln_lora_dim=16,
        patch_size=(1, 2, 2),
        max_size=(128, 240, 240),
        rope_scale=(1.0, 4.0, 4.0),
        concat_padding_mask=True,
        extra_pos_embed_type=None,
        use_crossattn_projection=False,
    )
    config = AnimaDiTConfig()
    config.update_model_arch(kwargs)
    args = ServerArgs(
        model_path="circlestone-labs/Anima-Base-v1.0-Diffusers",
        num_gpus=1,
        pipeline_config=AnimaPipelineConfig(dit_config=config),
        attention_backend="torch_sdpa",
    )
    monkeypatch.setattr(server_args, "_global_server_args", args)
    ensure_distributed_env_defaults()
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    torch.manual_seed(42)
    reference = CosmosTransformer3DModel(**kwargs).cuda().to(dtype).eval()
    model = AnimaTransformer3DModel(config, kwargs).cuda().to(dtype).eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(batch_size, 4, 1, 8, 12, device="cuda", dtype=dtype)
    context = torch.randn(batch_size, 7, 32, device="cuda", dtype=dtype)
    t = torch.linspace(600, 900, batch_size, device="cuda")
    padding = torch.zeros(1, 1, 8, 12, device="cuda", dtype=dtype)
    expected = reference(x, t.to(dtype) / 1000, context, padding_mask=padding).sample
    with (
        set_forward_context(current_timestep=0, attn_metadata=None),
        patch.object(
            anima, "residual_gate_add", wraps=anima.residual_gate_add
        ) as fused,
    ):
        actual = model(x, context, t)
        assert fused.call_count == 0
        with enable_breakable_cuda_graph():
            graph_path = model(x, context, t)
        assert fused.call_count == 3 * kwargs["num_layers"]
        after_graph = model(x, context, t)
        assert fused.call_count == 3 * kwargs["num_layers"]
    torch.testing.assert_close(graph_path, actual, atol=0, rtol=0)
    torch.testing.assert_close(after_graph, actual, atol=0, rtol=0)
    tolerance = 2e-5 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)

    project = anima.AnimaAdaLayerNorm.project_modulation
    calls = []

    def counted(norm, embedded_timestep, temb):
        calls.append(norm)
        return project(norm, embedded_timestep, temb)

    req = SimpleNamespace(extra={}, do_classifier_free_guidance=True)
    next_t = t / 2
    norms_per_forward = 3 * kwargs["num_layers"] + 1
    with patch.object(anima.AnimaAdaLayerNorm, "project_modulation", counted):
        for i, (conditioning, timestep) in enumerate(
            ((context, t), (-context, t), (context, next_t), (-context, next_t))
        ):
            expected = reference(
                x, timestep.to(dtype) / 1000, conditioning, padding_mask=padding
            ).sample
            with set_forward_context(i // 2, None, forward_batch=req):
                actual = model(x, conditioning, timestep)
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
            assert len(calls) == norms_per_forward * (i // 2 + 1)

        with (
            set_forward_context(1, None, forward_batch=req),
            enable_breakable_cuda_graph(),
        ):
            graph_path = model(x, conditioning, timestep)
        torch.testing.assert_close(graph_path, actual, atol=0, rtol=0)
        assert len(calls) == 3 * norms_per_forward

        # A new loop, a new request, or a different step cannot reuse old values.
        req.extra.pop(DENOISING_LOOP_STATE)
        other_req = SimpleNamespace(extra={}, do_classifier_free_guidance=True)
        for count, (request, step) in enumerate(
            ((req, 1), (other_req, 1), (other_req, 2)), start=4
        ):
            with set_forward_context(step, None, forward_batch=request):
                fresh = model(x, conditioning, timestep)
            torch.testing.assert_close(fresh, actual, atol=0, rtol=0)
            assert len(calls) == count * norms_per_forward

        no_cfg = SimpleNamespace(extra={}, do_classifier_free_guidance=False)
        with set_forward_context(1, None, forward_batch=no_cfg):
            unguided = model(x, conditioning, timestep)
        torch.testing.assert_close(unguided, actual, atol=0, rtol=0)
        assert len(calls) == 7 * norms_per_forward
        assert DENOISING_LOOP_STATE not in no_cfg.extra

        with (
            patch.object(model, "layerwise_offload_managers", [object()]),
            set_forward_context(2, None, forward_batch=other_req),
        ):
            offloaded = model(x, conditioning, timestep)
        torch.testing.assert_close(offloaded, actual, atol=0, rtol=0)
        assert len(calls) == 8 * norms_per_forward

        # Existing wrappers need neither the new keyword nor cached time values.
        norm_forward = model.norm_out.forward
        time_forward = model.time_embed.forward
        wrappers = (
            (model.norm_out, lambda x, emb, temb: norm_forward(x, emb, temb)),
            (model.time_embed, lambda t, dtype: time_forward(t, dtype)),
        )
        for count, (module, forward) in enumerate(wrappers, start=9):
            with (
                patch.object(module, "forward", side_effect=forward),
                set_forward_context(2, None, forward_batch=other_req),
            ):
                wrapped = model(x, conditioning, timestep)
            torch.testing.assert_close(wrapped, actual, atol=0, rtol=0)
            assert len(calls) == count * norms_per_forward

        with (
            patch.object(
                anima, "get_classifier_free_guidance_world_size", return_value=2
            ),
            set_forward_context(2, None, forward_batch=other_req),
        ):
            cfg_parallel = model(x, conditioning, timestep)
        torch.testing.assert_close(cfg_parallel, actual, atol=0, rtol=0)
        assert len(calls) == 11 * norms_per_forward


@torch.no_grad()
def test_conditioner_masks_source_and_zero_pads_output(monkeypatch):
    args = ServerArgs(
        model_path="circlestone-labs/Anima-Base-v1.0-Diffusers",
        num_gpus=1,
        pipeline_config=AnimaPipelineConfig(),
        attention_backend="torch_sdpa",
    )
    monkeypatch.setattr(server_args, "_global_server_args", args)
    config = AnimaTextConditionerConfig(
        arch_config=AnimaTextConditionerArchConfig(
            source_dim=32,
            target_dim=32,
            model_dim=32,
            num_attention_heads=2,
            num_layers=2,
            target_vocab_size=16,
        )
    )
    model = AnimaTextConditioner(config).cuda().bfloat16().eval()
    source = torch.randn(2, 4, 32, device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor([[1, 2, 0], [1, 0, 0]], device="cuda")
    target_mask = ids != 0
    source_mask = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0]], device="cuda")
    changed = source.clone()
    changed[source_mask == 0] += 10
    with set_forward_context(current_timestep=0, attn_metadata=None):
        actual = model(source, ids, target_mask, source_mask)
        expected = model(changed, ids, target_mask, source_mask)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.shape == (2, 512, 32)
    assert torch.isfinite(actual).all()
    assert not actual[:, 3:].any()
    assert not actual[:, :3][~target_mask].any()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
