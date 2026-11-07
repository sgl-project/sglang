# SPDX-License-Identifier: Apache-2.0
"""Nomas TP=2 and Ulysses SP=2 reproduce the single-rank segment outputs (2-GPU lane)."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.multimodal_gen.configs.models.dits.nomas import NomasArchConfig, NomasConfig
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.models.dits.nomas import NomasTransformer3DModel
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.nomas.denoise_loop import (
    NomasLatentSession,
)
from sglang.multimodal_gen.runtime.server_args import set_global_server_args
from sglang.srt.utils.network import get_free_port_below_ephemeral

SHAPE = (2, 2, 4)  # 16 video tokens: divisible by sp 2, camera chunks straddle shards


def _condition(device, dtype):
    from msgspec.structs import replace

    from sglang.multimodal_gen.test.unit.test_nomas import condition

    return replace(condition(device, dtype), camera_has_mask=None, action_start=2)


def _run(rank, world_size, tp_size, backend, port, output_dir):
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
            pipeline_config=SimpleNamespace(
                dit_config=SimpleNamespace(hidden_size=512)
            ),
        )
    )
    sp_size = world_size // tp_size
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=tp_size, sp_size=sp_size, ulysses_degree=sp_size, ring_degree=1
    )
    # FA3 and Flex run in BF16; SDPA in FP32 is the tight reference.
    dtype = torch.float32 if backend == "sdpa" else torch.bfloat16
    config = NomasConfig(
        arch_config=NomasArchConfig(
            hidden_size=512, num_attention_heads=4, num_layers=2
        )
    )
    model = NomasTransformer3DModel(config, attention_backend=backend).eval()
    weights_path = Path(output_dir) / "weights.pt"
    generator = torch.Generator().manual_seed(123)
    with torch.no_grad():
        if world_size == 1:
            for name, p in model.named_parameters():
                if p.ndim == 1 and "norm" in name and name.endswith("weight"):
                    p.fill_(1)
                else:
                    p.copy_(torch.randn(p.shape, generator=generator) * 0.02)
            torch.save(model.state_dict(), weights_path)
        else:
            weights = torch.load(weights_path, weights_only=True)
            for name, p in model.named_parameters():
                loader = p.__dict__.get("weight_loader")
                if loader is None:
                    p.copy_(weights[name])
                else:
                    loader(p, weights[name])
    model = model.to(device=rank, dtype=dtype)
    model.post_load_weights()
    assert model.local_attention_heads == 4 // world_size
    session = NomasLatentSession(model, cache_capacity=64, seed=7, steps=2)
    cond = _condition(rank, dtype)
    with torch.inference_mode():
        outputs = [session.generate_segment(SHAPE, cond, evict=True) for _ in range(2)]
    assert session.cache.valid_length == 2 * (16 + 8)
    assert all(torch.isfinite(o).all() for o in outputs)
    path = Path(output_dir) / f"{backend}.pt"
    if world_size == 1:
        torch.save([o.cpu() for o in outputs], path)
    else:
        reference = torch.load(path, weights_only=True, map_location=f"cuda:{rank}")
        # fp32 differs only by all-reduce summation order; bf16 by rounding.
        tol = (
            dict(atol=1e-5, rtol=1e-4)
            if backend == "sdpa"
            else dict(atol=0.05, rtol=0.05)
        )
        for actual, expected in zip(outputs, reference, strict=True):
            torch.testing.assert_close(actual, expected, **tol)
    dist.destroy_process_group()


def _launch(world_size, tp_size, backend, output_dir):
    mp.spawn(
        _run,
        args=(
            world_size,
            tp_size,
            backend,
            get_free_port_below_ephemeral(),
            output_dir,
        ),
        nprocs=world_size,
    )


# SDPA in FP32 is the tight reference; the other backend is the one this GPU
# serves with (FA3 on Hopper, FlexAttention elsewhere).
_HOPPER = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
_BACKENDS = ["sdpa", "fa3" if _HOPPER else "flex"]


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
@pytest.mark.parametrize("backend", _BACKENDS)
def test_tensor_and_sequence_parallel_match_single_rank(tmp_path, backend):
    _launch(1, 1, backend, str(tmp_path))
    _launch(2, 2, backend, str(tmp_path))  # TP=2
    _launch(2, 1, backend, str(tmp_path))  # Ulysses SP=2


def _run_window_parallel_decode(rank, port):
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_decode_parallel_world_size,
    )
    from sglang.multimodal_gen.runtime.models.vaes.nomas import (
        merge_windows,
        window_starts,
    )
    from sglang.multimodal_gen.test.unit.test_nomas import tiny_decoder

    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="2",
    )
    torch.cuda.set_device(rank)
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1, sp_size=2, ulysses_degree=2, ring_degree=1
    )
    assert get_decode_parallel_world_size() == 2
    decoder = tiny_decoder().cuda()
    latent = torch.randn(
        1, 32, 21, 2, 2, generator=torch.Generator().manual_seed(11)
    ).cuda()
    with torch.inference_mode():
        starts = window_starts(21, 10, 5)
        serial = merge_windows(
            [
                decoder.decoder(
                    latent[:, :, s : s + 10], decoder._noise(latent, s, s + 10)
                )
                for s in starts
            ],
            [s * 4 for s in starts],
        )
        # Four windows over two ranks: each rank decodes two, then all gather.
        assert torch.equal(decoder.decode(latent).sample, serial)
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
def test_window_parallel_decode_matches_serial_decode():
    mp.spawn(
        _run_window_parallel_decode,
        args=(get_free_port_below_ephemeral(),),
        nprocs=2,
    )
