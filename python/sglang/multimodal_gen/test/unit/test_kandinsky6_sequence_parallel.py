# SPDX-License-Identifier: Apache-2.0
"""Compare native Ulysses/Ring joint video-audio forwards with a single GPU."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.multimodal_gen.configs.models.dits.kandinsky6 import (
    Kandinsky6VideoAudioConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.kandinsky6 import (
    Kandinsky6TI2VAPipelineConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.layers.attention.selector import (
    global_force_attn_backend,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.kandinsky6 import (
    Kandinsky6Transformer3DModel,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import set_global_server_args
from sglang.multimodal_gen.runtime.utils.precision import set_mixed_precision_policy
from sglang.srt.utils.network import get_free_port_below_ephemeral


def _forward(rank, world_size, ring, tp_size, port, output_dir, model_dim):
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
            pipeline_config=Kandinsky6TI2VAPipelineConfig(),
        )
    )
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=tp_size,
        sp_size=world_size // tp_size,
        ulysses_degree=1 if ring else world_size // tp_size,
        ring_degree=world_size // tp_size if ring else 1,
    )
    config = Kandinsky6VideoAudioConfig()
    config.update_model_arch(
        dict(
            model_dim=model_dim,
            time_dim=32,
            ff_dim=512,
            axes_dims=(32, 48, 48),
            in_visual_dim=4,
            out_visual_dim=4,
            in_text_dim=16,
            in_text_dim2=8,
            in_audio_dim=4,
            out_audio_dim=4,
            num_text_blocks=1,
            num_visual_blocks=2,
            patch_size=(1, 1, 1),
            visual_cond=False,
            visual_token_type_num_embeddings=0,
            is_multimodal=True,
            model_dim_a=model_dim,
            time_dim_a=32,
            ff_dim_a=256,
            axes_dims_a=(32, 48, 48),
            ca_rope=True,
            cross_gates=True,
            fix_modulation=True,
        )
    )
    model = Kandinsky6Transformer3DModel(config, {}).eval()
    generator = torch.Generator().manual_seed(7)
    weights_path = Path(output_dir) / "weights.pt"
    with torch.no_grad():
        if world_size == 1:
            for parameter in model.parameters():
                parameter.copy_(
                    torch.randn(parameter.shape, generator=generator) * 0.02
                )
            torch.save(model.state_dict(), weights_path)
        else:
            weights = torch.load(weights_path, weights_only=True)
            for name, parameter in model.named_parameters():
                weight_loader = parameter.__dict__.get("weight_loader")
                if weight_loader is None:
                    parameter.copy_(weights[name])
                else:
                    weight_loader(parameter, weights[name])
    model = model.to(device=rank, dtype=torch.bfloat16)

    for frames in (2, 3):
        # odd visual sequence length exercises masked tail padding; audio stays odd
        generator.manual_seed(frames)

        def tensor(*shape):
            return torch.randn(*shape, generator=generator).to(
                device=rank, dtype=torch.bfloat16
            )

        with (
            torch.no_grad(),
            set_forward_context(current_timestep=0, attn_metadata=None),
        ):
            video, audio = model(
                hidden_states=tensor(1, frames, 3, 3, 4),
                hidden_states_audio=tensor(1, 5, 4),
                encoder_hidden_states=tensor(1, 7, 16),
                pooled_projections=tensor(1, 8),
                timestep=torch.tensor([500.0], device=rank),
                visual_rope_pos=[torch.arange(n, device=rank) for n in (frames, 3, 3)],
                text_rope_pos=torch.arange(7, device=rank),
            )
        assert video.shape == (1, frames, 3, 3, 4)
        assert audio.shape == (1, 5, 4)
        assert torch.isfinite(video).all() and torch.isfinite(audio).all()
        path = Path(output_dir) / f"{frames}.pt"
        if world_size == 1:
            torch.save((video.cpu(), audio.cpu()), path)
        else:
            reference = torch.load(path, weights_only=True, map_location=f"cuda:{rank}")
            for actual, expected in zip((video, audio), reference, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)
    dist.destroy_process_group()


def _launch(world_size, ring, output_dir, tp_size=1, model_dim=256):
    port = get_free_port_below_ephemeral()
    mp.spawn(
        _forward,
        args=(world_size, ring, tp_size, port, output_dir, model_dim),
        nprocs=world_size,
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
def test_joint_video_audio_sequence_parallel(tmp_path):
    _launch(1, False, str(tmp_path))
    _launch(2, False, str(tmp_path))
    _launch(2, True, str(tmp_path))
    _launch(2, False, str(tmp_path), tp_size=2)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four CUDA GPUs")
def test_joint_video_audio_tensor_sequence_parallel(tmp_path):
    _launch(1, False, str(tmp_path), model_dim=512)
    for tp_size in (1, 2, 4):
        _launch(4, False, str(tmp_path), tp_size=tp_size, model_dim=512)
    _launch(4, True, str(tmp_path), tp_size=2, model_dim=512)
