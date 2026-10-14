# SPDX-License-Identifier: Apache-2.0
"""Run with torchrun --standalone --nproc-per-node=2 -m pytest -q <this file>."""

import os
from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig as HFQwen3VLConfig,
)

from sglang.multimodal_gen.configs.models.vaes.qwenimage21 import (
    QwenImage21VAEArchConfig,
    QwenImage21VAEConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_group,
    get_tp_group,
    maybe_init_distributed_environment_and_model_parallel,
    use_tensor_parallel_group,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.encoders.qwen3vl import (
    Qwen3VLForConditionalGeneration,
)
from sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_qwenimage21 import (
    AutoencoderKLQwenImage21,
)
from sglang.multimodal_gen.runtime.pipelines_core import Req
from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 2,
    reason="requires two CUDA ranks launched by torchrun",
)


@pytest.fixture(scope="module", autouse=True)
def distributed():
    args = ServerArgs(
        model_path="Qwen/Qwen-Image-2.1",
        num_gpus=2,
        tp_size=2,
        sp_degree=1,
        attention_backend="torch_sdpa",
    )
    set_global_server_args(args)
    maybe_init_distributed_environment_and_model_parallel(tp_size=2, sp_size=1)
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
    torch.backends.cudnn.allow_tf32 = cudnn_tf32


@pytest.mark.parametrize("edit", [False, True])
@torch.no_grad()
def test_encoder_tp_shards_weights_and_preserves_conditioning(edit):
    arch = HFQwen3VLConfig(
        text_config=dict(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=32,
            pad_token_id=0,
            rope_scaling=dict(rope_type="default", mrope_section=[2, 3, 3]),
        ),
        vision_config=dict(
            hidden_size=32,
            intermediate_size=64,
            depth=2,
            num_heads=4,
            patch_size=2,
            temporal_patch_size=1,
            in_channels=3,
            num_position_embeddings=16,
            spatial_merge_size=2,
            out_hidden_size=64,
            deepstack_visual_indexes=[],
        ),
        image_token_id=8,
        video_token_id=9,
        vision_start_token_id=7,
        vision_end_token_id=6,
    )
    arch._fsdp_shard_conditions = []
    arch.stacked_params_mapping = []
    config = SimpleNamespace(arch_config=arch, quant_config=None)
    torch.manual_seed(42)
    with use_tensor_parallel_group(get_sp_group()):
        reference = Qwen3VLForConditionalGeneration(config).cuda().eval()
        for param in reference.parameters():
            torch.nn.init.normal_(param, std=0.02)
    reference.bind_encoder_tp_group(get_sp_group())
    with use_tensor_parallel_group(get_tp_group()):
        model = Qwen3VLForConditionalGeneration(config).cuda().eval()
    model.bind_encoder_tp_group(get_tp_group())
    model.load_weights(reference.state_dict().items())
    for layer in model.model.language_model.layers:
        assert layer.self_attn.q_proj.weight.shape == (32, 64)
        assert layer.mlp.gate_proj.weight.shape == (64, 64)
    tokens = [1, 7, 8, 8, 8, 8, 6, 3] if edit else [1, 2, 3, 4]
    inputs = dict(
        input_ids=torch.tensor([tokens], device="cuda"),
        attention_mask=torch.ones(1, len(tokens), device="cuda", dtype=torch.long),
        output_hidden_states=True,
        use_cache=False,
        logits_to_keep=1,
    )
    if edit:
        inputs.update(
            pixel_values=torch.randn(16, 12, device="cuda"),
            image_grid_thw=torch.tensor([[1, 4, 4]], device="cuda"),
        )
    with set_forward_context(
        current_timestep=None, attn_metadata=None, forward_batch=Req(prompt="test")
    ):
        expected = reference(**inputs).hidden_states[-1]
        actual = model(**inputs).hidden_states[-1]
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("height", [4, 5])
@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@torch.no_grad()
def test_vae_spatial_shard_matches_full_decode(height, residual, dtype):
    arch = QwenImage21VAEArchConfig(
        base_dim=4,
        decoder_base_dim=4,
        z_dim=4,
        dim_mult=(1, 2, 4, 4, 4),
        num_res_blocks=1,
        temperal_downsample=(False, True, True, True),
        is_residual=residual,
    )
    torch.manual_seed(42)
    reference = (
        AutoencoderKLQwenImage21(
            QwenImage21VAEConfig(arch_config=arch, load_encoder=False)
        )
        .cuda()
        .eval()
    )
    parallel = (
        AutoencoderKLQwenImage21(
            QwenImage21VAEConfig(
                arch_config=arch,
                load_encoder=False,
                parallel_decode_mode="spatial_shard",
            )
        )
        .cuda()
        .eval()
    )
    parallel.load_state_dict(reference.state_dict())
    assert parallel.spatial_parallel
    z = torch.randn(1, 4, 1, height, 4, device="cuda")
    torch.distributed.broadcast(z, src=0)
    expected = reference.to(dtype).decode(z.to(dtype))
    actual = parallel.to(dtype).decode(z.to(dtype))
    assert actual.shape == expected.shape == (1, 4, 1, height * 16, 64)
    # full and sharded convolutions select different FP32 reduction kernels
    tolerance = 1e-10 if dtype == torch.float64 else 1e-4
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
