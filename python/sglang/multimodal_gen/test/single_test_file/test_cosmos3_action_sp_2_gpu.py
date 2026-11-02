# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 action tokens under Ulysses sequence parallelism must match one rank.

Action latents join the GEN stream behind the video tokens, so under SP they are
padded, sharded and all-gathered with everything else, and the action head runs
on the reassembled sequence. A tiny randomly initialised transformer runs on one
rank (reference) and on two Ulysses ranks; the token count is chosen so the
two-rank split needs a pad token, which lands behind the action tokens.

    pytest -v python/sglang/multimodal_gen/test/single_test_file/test_cosmos3_action_sp_2_gpu.py
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.multimodal_gen.configs.models.dits.cosmos3video import (
    Cosmos3VideoArchConfig,
    Cosmos3VideoConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.layers.attention.selector import (
    global_force_attn_backend,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.cosmos3video import (
    Cosmos3OmniTransformer,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import set_global_server_args
from sglang.multimodal_gen.runtime.utils.precision import set_mixed_precision_policy
from sglang.srt.utils.network import get_free_port_below_ephemeral

# 2 latent frames x (4x4 latent / patch 2 = 4 tokens) = 8 video tokens plus 3
# action tokens = 11 tokens: two ranks need one pad token behind the actions.
_LATENT_CHANNELS, _LATENT_FRAMES, _LATENT_HEIGHT, _LATENT_WIDTH = 4, 2, 4, 4
_ACTION_TOKENS, _ACTION_DIM = 3, 6
_TEXT_TOKENS, _VOCAB = 5, 512


def _tiny_config() -> Cosmos3VideoConfig:
    return Cosmos3VideoConfig(
        arch_config=Cosmos3VideoArchConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            intermediate_size=128,
            latent_channel=_LATENT_CHANNELS,
            out_channels=_LATENT_CHANNELS,
            mrope_section=(8, 4, 4),
            vocab_size=_VOCAB,
            action_gen=True,
            action_dim=_ACTION_DIM,
            num_embodiment_domains=3,
            frequency_embedding_size=32,
        )
    )


def _forward(rank: int, world_size: int, port: int, output_dir: str) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    torch.cuda.set_device(rank)
    set_mixed_precision_policy(torch.float32, torch.float32)
    # Dense SDPA keeps the comparison about the sequence bookkeeping, not kernels.
    global_force_attn_backend(AttentionBackendEnum.TORCH_SDPA)
    set_global_server_args(
        SimpleNamespace(
            attention_backend="torch_sdpa",
            attention_backend_config=None,
            enable_attention_backend_autotune=False,
            kv_gather_degree=1,
            sp_split_auto=False,
        )
    )
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1, sp_size=world_size, ulysses_degree=world_size, ring_degree=1
    )

    model = Cosmos3OmniTransformer(_tiny_config(), {}).eval()
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
            model.load_state_dict(torch.load(weights_path, weights_only=True))
    model = model.to(device=rank, dtype=torch.float32)

    generator.manual_seed(11)

    def tensor(*shape):
        return torch.randn(*shape, generator=generator).to(
            device=rank, dtype=torch.float32
        )

    latents = tensor(1, _LATENT_CHANNELS, _LATENT_FRAMES, _LATENT_HEIGHT, _LATENT_WIDTH)
    action_latents = tensor(1, _ACTION_TOKENS, _ACTION_DIM)
    text_ids = torch.randint(0, _VOCAB, (1, _TEXT_TOKENS), generator=generator).to(rank)
    text_mask = torch.ones(1, _TEXT_TOKENS, dtype=torch.long, device=rank)
    with (
        torch.no_grad(),
        set_forward_context(current_timestep=0, attn_metadata=None),
    ):
        video, action = model(
            hidden_states=latents,
            encoder_hidden_states=None,
            timestep=torch.tensor([500.0], device=rank),
            text_ids=text_ids,
            text_mask=text_mask,
            fps=24.0,
            action_latents=action_latents,
            action_domain_ids=torch.tensor([1], device=rank),
            cache_key="cond",
        )
    assert video.shape == latents.shape
    assert action.shape == action_latents.shape
    assert torch.isfinite(video).all() and torch.isfinite(action).all()

    path = Path(output_dir) / "outputs.pt"
    if world_size == 1:
        torch.save((video.cpu(), action.cpu()), path)
    else:
        reference = torch.load(path, weights_only=True, map_location=f"cuda:{rank}")
        for actual, expected in zip((video, action), reference, strict=True):
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    dist.destroy_process_group()


def _launch(world_size: int, output_dir: str) -> None:
    port = get_free_port_below_ephemeral()
    mp.spawn(_forward, args=(world_size, port, output_dir), nprocs=world_size)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
def test_action_tokens_match_single_rank_under_ulysses(tmp_path):
    _launch(1, str(tmp_path))
    _launch(2, str(tmp_path))
