import pytest
import torch
import torch.nn as nn

from sglang.srt.configs.qwen3_vl import Qwen3VLConfig
from sglang.srt.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.srt.layers.quantization.unquant import (
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, suite="stage-b-test-large-1-gpu")


@pytest.fixture(scope="module", autouse=True)
def _set_default_device_npu():
    prev_device = torch.get_default_device()
    torch.set_default_device("npu")
    try:
        yield
    finally:
        torch.set_default_device(prev_device)


def _build_model():
    server_args = ServerArgs(model_path="dummy", device="npu")
    model_config = Qwen3VLConfig(
        hidden_size=64,
        num_heads=1,
        num_position_embeddings=2304,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        deepstack_visual_indexes=[5, 11, 17],
        in_channels=3,
        depth=24,
        intermediate_size=256,
        hidden_act="gelu_pytorch_tanh",
        out_hidden_size=2560,
    )
    set_global_server_args_for_scheduler(server_args)
    init_distributed_environment(
        backend="gloo",
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="tcp://127.0.0.1:2646",
    )
    initialize_model_parallel()
    initialize_dp_attention(
        server_args=server_args,
        model_config=model_config,
    )
    model = Qwen3VLMoeVisionModel(
        model_config,
        quant_config=None,
        norm_eps=1e-6,
        prefix="visual",
    )
    return model, model_config


def test_conv3d_to_linear():
    assert issubclass(UnquantizedLinearMethod, LinearMethodBase)

    model, model_config = _build_model()

    batch_size = 8
    in_channels = model_config.in_channels
    temporal_patch_size = model_config.temporal_patch_size
    patch_size = model_config.patch_size
    embed_dim = model_config.hidden_size
    kernel_size = [temporal_patch_size, patch_size, patch_size]
    flat_patch_dim = in_channels * temporal_patch_size * patch_size**2

    input_data = torch.randn(
        batch_size,
        flat_patch_dim,
        device=model.patch_embed.linear.weight.device,
        dtype=model.patch_embed.linear.weight.dtype,
    )

    with torch.inference_mode():
        linear_output = model.patch_embed(input_data)

        conv3d = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
            device=model.patch_embed.linear.weight.device,
            dtype=model.patch_embed.linear.weight.dtype,
        )
        conv3d.weight.copy_(
            model.patch_embed.linear.weight.view(
                embed_dim,
                in_channels,
                temporal_patch_size,
                patch_size,
                patch_size,
            )
        )
        conv3d.bias.copy_(model.patch_embed.linear.bias)

        input_5d = input_data.view(
            batch_size,
            in_channels,
            temporal_patch_size,
            patch_size,
            patch_size,
        )
        conv3d_output = conv3d(input_5d).view(batch_size, embed_dim)

    torch.testing.assert_close(
        linear_output,
        conv3d_output,
        atol=1e-4,
        rtol=1e-4,
    )
