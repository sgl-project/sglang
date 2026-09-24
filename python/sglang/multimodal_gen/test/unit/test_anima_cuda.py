# SPDX-License-Identifier: Apache-2.0
"""Small, checkpoint-free numerical contracts against the Cosmos reference."""

import sys

import pytest
import torch
from diffusers.models.transformers.transformer_cosmos import CosmosTransformer3DModel

from sglang.multimodal_gen.configs.models.dits.anima import AnimaDiTConfig
from sglang.multimodal_gen.configs.pipeline_configs.anima import AnimaPipelineConfig
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.anima import AnimaTransformer3DModel
from sglang.multimodal_gen.runtime.server_args import ServerArgs, server_args

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


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
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1, sp_size=1, distributed_init_method="tcp://127.0.0.1:29638"
    )
    torch.manual_seed(42)
    reference = CosmosTransformer3DModel(**kwargs).cuda().to(dtype).eval()
    model = AnimaTransformer3DModel(config, kwargs).cuda().to(dtype).eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(batch_size, 4, 1, 8, 12, device="cuda", dtype=dtype)
    context = torch.randn(batch_size, 7, 32, device="cuda", dtype=dtype)
    t = torch.linspace(600, 900, batch_size, device="cuda")
    padding = torch.zeros(1, 1, 8, 12, device="cuda", dtype=dtype)
    expected = reference(x, t.to(dtype) / 1000, context, padding_mask=padding).sample
    with set_forward_context(current_timestep=0, attn_metadata=None):
        actual = model(x, context, t)
    tolerance = 2e-5 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
