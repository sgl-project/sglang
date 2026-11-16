# SPDX-License-Identifier: Apache-2.0
"""Verify SR token sharding, TP weight loading, and repeated distributed forwards."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.multimodal_gen.configs.models.dits.kandinsky6_sr import (
    Kandinsky6SRDitConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.kandinsky6_sr import (
    Kandinsky6SRPipelineConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.layers.attention.selector import (
    global_force_attn_backend,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.kandinsky6_sr import (
    Kandinsky6SRTransformer3DModel,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import set_global_server_args
from sglang.multimodal_gen.runtime.utils.precision import set_mixed_precision_policy
from sglang.multimodal_gen.test.unit.kandinsky6_sr_tiny_components import TINY_DIT
from sglang.srt.utils.network import get_free_port_below_ephemeral


def _forward(rank, world_size, tp_size, ring, port, output_dir):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    torch.cuda.set_device(rank)
    set_mixed_precision_policy(torch.bfloat16, torch.float32)
    global_force_attn_backend(AttentionBackendEnum.FA)
    set_global_server_args(
        SimpleNamespace(
            attention_backend="fa",
            attention_backend_config=None,
            kv_gather_degree=1,
            sp_split_auto=False,
            pipeline_config=Kandinsky6SRPipelineConfig(),
        )
    )
    sp_size = world_size // tp_size
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=tp_size,
        sp_size=sp_size,
        ulysses_degree=1 if ring else sp_size,
        ring_degree=sp_size if ring else 1,
    )
    config = Kandinsky6SRDitConfig()
    config.update_model_arch(
        TINY_DIT | dict(model_dim=512, time_dim=32, ff_dim=1024, axes_dims=[32, 48, 48])
    )
    model = Kandinsky6SRTransformer3DModel(config, {}).eval()
    generator = torch.Generator().manual_seed(7)
    directory = Path(output_dir)
    with torch.no_grad():
        if world_size == 1:
            for parameter in model.parameters():
                parameter.copy_(
                    torch.randn(parameter.shape, generator=generator) * 0.02
                )
            torch.save(model.state_dict(), directory / "weights.pt")
        else:
            weights = torch.load(directory / "weights.pt", weights_only=True)
            for name, parameter in model.named_parameters():
                loader = parameter.__dict__.get("weight_loader")
                if loader is None:
                    parameter.copy_(weights[name])
                else:
                    loader(parameter, weights[name])
    model.to(device=rank, dtype=torch.bfloat16)
    assert model.visual_transformer_blocks[0].self_attention.local_num_heads == (
        4 // tp_size
    )

    observed = []

    def record_shard(module, args):
        visual, _, rope, _ = args
        observed.append((visual.shape[0], visual.shape[1], rope.shape[1]))

    handle = model.visual_transformer_blocks[0].register_forward_pre_hook(record_shard)
    first = None
    for batch_size, frames in ((1, 2), (2, 3), (1, 2)):
        generator.manual_seed(frames)
        inputs = torch.randn(batch_size, frames, 6, 6, 4, generator=generator).to(
            device=rank, dtype=torch.bfloat16
        )
        with (
            torch.no_grad(),
            set_forward_context(current_timestep=0, attn_metadata=None),
        ):
            actual = model(
                inputs,
                torch.full((batch_size,), 500.0, device=rank),
                [torch.arange(n, device=rank) for n in (frames, 3, 3)],
            )
        local_len = (frames * 3 * 3 + sp_size - 1) // sp_size
        assert observed[-1] == (batch_size, local_len, local_len)
        assert actual.shape == (batch_size, frames, 6, 6, 12)
        assert torch.isfinite(actual).all()
        if frames == 2:
            if first is None:
                first = actual.clone()
            else:
                torch.testing.assert_close(actual, first, rtol=0, atol=0)
        path = directory / f"{batch_size}-{frames}.pt"
        if world_size == 1:
            torch.save(actual.cpu(), path)
        else:
            expected = torch.load(path, weights_only=True, map_location=f"cuda:{rank}")
            torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)
    handle.remove()
    dist.destroy_process_group()


def _launch(world_size, tp_size, ring, output_dir):
    mp.spawn(
        _forward,
        args=(world_size, tp_size, ring, get_free_port_below_ephemeral(), output_dir),
        nprocs=world_size,
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
def test_sr_two_gpus(tmp_path):
    _launch(1, 1, False, str(tmp_path))
    _launch(2, 1, False, str(tmp_path))
    _launch(2, 1, True, str(tmp_path))
    _launch(2, 2, False, str(tmp_path))


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four CUDA GPUs")
def test_sr_four_gpus(tmp_path):
    _launch(1, 1, False, str(tmp_path))
    for tp_size in (1, 2, 4):
        _launch(4, tp_size, False, str(tmp_path))
    _launch(4, 2, True, str(tmp_path))
