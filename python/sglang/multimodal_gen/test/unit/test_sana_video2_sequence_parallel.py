# SPDX-License-Identifier: Apache-2.0
"""Compare sequence-parallel SANA-Video 2.0 forwards with a single GPU."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.multimodal_gen.configs.models.dits.sana_video2 import (
    SanaVideo2ArchConfig,
    SanaVideo2Config,
)
from sglang.multimodal_gen.configs.pipeline_configs.sana_video2 import (
    SanaVideo2PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.sana_video2 import SanaVideo2SamplingParams
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.distributed.sp_shard_utils import SpShard
from sglang.multimodal_gen.runtime.models.dits.sana_video2 import (
    GatedLinearAttention,
    GatedSoftmaxAttention,
    SanaVideo2Transformer3DModel,
)
from sglang.multimodal_gen.runtime.server_args import set_global_server_args
from sglang.srt.utils.network import get_free_port_below_ephemeral


@pytest.mark.usefixtures("single_process_model_parallel")
@pytest.mark.parametrize("attention_cls", [GatedLinearAttention, GatedSoftmaxAttention])
def test_attention_ignores_tail_padding(attention_cls):
    torch.manual_seed(21)
    module = attention_cls(24, 12, fp32_attention=True).eval()
    x = torch.randn(2, 5, 24)
    shard = SpShard(orig_len=3, local_len=5, num_pad=2, sp_size=1, sp_rank=0)
    with torch.no_grad():
        expected = module(x[:, :3])
        actual = module(x, shard=shard)
    torch.testing.assert_close(actual[:, :3], expected, atol=2e-6, rtol=2e-5)


def test_linear_attention_ignores_padding_spanning_ranks(monkeypatch):
    from sglang.multimodal_gen.runtime.models.dits import sana_video2

    torch.manual_seed(22)
    module = GatedLinearAttention(24, 12, fp32_attention=True).eval()
    x = torch.randn(2, 8, 24)
    # 8 tokens over 10 ranks: ranks 8 and 9 hold nothing but padding
    padded = torch.cat([x, torch.randn(2, 2, 24)], dim=1)
    shards = [
        SpShard(orig_len=8, local_len=1, num_pad=2, sp_size=10, sp_rank=rank)
        for rank in range(10)
    ]

    def forward_ranks(all_reduce):
        monkeypatch.setattr(
            sana_video2, "sequence_model_parallel_all_reduce", all_reduce
        )
        with torch.no_grad():
            return [
                module(padded[:, rank : rank + 1], shard=shard)
                for rank, shard in enumerate(shards)
            ]

    partial_sums = []
    forward_ranks(lambda kv: partial_sums.append(kv) or kv)
    total = torch.stack(partial_sums).sum(0)
    actual = torch.cat(forward_ranks(lambda kv: total)[:8], dim=1)
    with torch.no_grad():
        expected = module(x)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_unsupported_parallel_layouts_are_rejected():
    config = SanaVideo2PipelineConfig()
    config.validate_server_args(SimpleNamespace(ring_degree=1, sp_degree=2))
    with pytest.raises(ValueError, match="ring"):
        config.validate_server_args(SimpleNamespace(ring_degree=2, sp_degree=2))
    # the softmax layers have 10 heads to spread over the ranks
    with pytest.raises(ValueError, match="10 softmax attention heads"):
        config.validate_server_args(SimpleNamespace(ring_degree=1, sp_degree=4))


def test_extra_gpus_go_to_sequence_parallelism():
    deployment = SanaVideo2PipelineConfig().get_model_deployment_config()
    assert not deployment.auto_enable_cfg_parallel
    assert not deployment.supports_cfg_parallel


def test_frame_count_is_not_aligned_to_gpus():
    assert not SanaVideo2SamplingParams().adjust_frames


def _forward(rank, world_size, port, output_dir):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    torch.cuda.set_device(rank)
    set_global_server_args(
        SimpleNamespace(
            attention_backend=None,
            attention_backend_config=None,
            kv_gather_degree=1,
            sp_split_auto=False,
            pipeline_config=SanaVideo2PipelineConfig(),
        )
    )
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1, sp_size=world_size, ulysses_degree=world_size, ring_degree=1
    )
    model = SanaVideo2Transformer3DModel(
        SanaVideo2Config(
            arch_config=SanaVideo2ArchConfig(
                hidden_size=48,
                depth=4,
                num_heads=2,
                linear_head_dim=12,
                softmax_head_dim=24,
                in_channels=4,
                caption_channels=8,
                model_max_length=5,
                mlp_ratio=2,
                attn_res_block_size=2,
            )
        )
    ).eval()
    generator = torch.Generator().manual_seed(7)
    weights_path = Path(output_dir) / "weights.pt"
    with torch.no_grad():
        if world_size == 1:
            for parameter in model.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.2)
            torch.save(model.state_dict(), weights_path)
        else:
            model.load_state_dict(torch.load(weights_path, weights_only=True))
    model = model.to(device=rank)
    block_tokens = []
    model.blocks[0].attn.register_forward_pre_hook(
        lambda _, args: block_tokens.append(args[0].shape[1])
    )

    # 3 frames of 3x3 tokens leave an odd sequence that needs tail padding
    for frames, per_frame_timestep in ((2, False), (3, False), (3, True)):
        generator.manual_seed(frames)

        def tensor(*shape):
            return torch.randn(*shape, generator=generator).to(device=rank)

        timestep = torch.tensor([500.0, 20.0], device=rank)
        if per_frame_timestep:
            timestep = timestep[:, None, None, None, None].repeat(1, 1, frames, 1, 1)
            timestep[:, :, 0] = 0
        with torch.no_grad():
            output = model(
                hidden_states=tensor(2, 4, frames, 3, 3),
                timestep=timestep,
                encoder_hidden_states=tensor(2, 5, 8),
                encoder_attention_mask=torch.tensor(
                    [[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], device=rank
                ),
            )
        assert output.shape == (2, 4, frames, 3, 3)
        assert block_tokens.pop() == -(-frames * 9 // world_size)
        path = Path(output_dir) / f"{frames}-{per_frame_timestep}.pt"
        if world_size == 1:
            torch.save(output.cpu(), path)
        else:
            expected = torch.load(path, weights_only=True, map_location=f"cuda:{rank}")
            torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)
    dist.destroy_process_group()


def _launch(world_size, output_dir):
    port = get_free_port_below_ephemeral()
    mp.spawn(_forward, args=(world_size, port, output_dir), nprocs=world_size)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
def test_sequence_parallel_matches_single_gpu(tmp_path):
    _launch(1, str(tmp_path))
    _launch(2, str(tmp_path))
