# SPDX-License-Identifier: Apache-2.0
"""SR decode parallelism must preserve causal state and request isolation."""

import os
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.multimodal_gen.configs.models.vaes.kandinsky6_sr import (
    Kandinsky6SRVAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.kandinsky6_sr import (
    Kandinsky6SRPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.kandinsky6_sr import (
    Kandinsky6SRSamplingParams,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_decode_parallel_group_coordinator,
    get_dp_rank,
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.layers.parallel_conv import (
    SpatialParallelConv3d,
    disable_spatial_parallel_decode,
    gather_and_trim_height,
    split_height_for_parallel_decode,
)
from sglang.multimodal_gen.runtime.models.vaes.kandinsky6_sr_vae import (
    Kandinsky6SRVAE,
    _ChunkedConv3d,
    _SpatialChunkedConv3d,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.run_spec import (
    SR_DENOISED_KEY,
    SR_TILES_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.stages import (
    Kandinsky6SRDecodeStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.tiled import (
    decode_chunks,
)
from sglang.multimodal_gen.runtime.server_args import set_global_server_args
from sglang.multimodal_gen.test.unit.kandinsky6_sr_tiny_components import TINY_KVAE
from sglang.srt.utils.network import get_free_port_below_ephemeral


def _decode(rank, world_size, tp_size, dp_size, port, spatial):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    torch.cuda.set_device(rank)
    sp_size = world_size // tp_size // dp_size
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=tp_size, sp_size=sp_size, ulysses_degree=sp_size, dp_size=dp_size
    )
    group = get_decode_parallel_group_coordinator()
    assert group.world_size == world_size // dp_size
    spatial_active = spatial and group.world_size > 1
    if spatial_active:
        torch.manual_seed(3 + get_dp_rank())
        conv = _ChunkedConv3d(2, 3, 3, padding=(0, 1, 1)).to(rank)
        parallel_conv = _SpatialChunkedConv3d.from_conv(conv)
        assert parallel_conv.weight is conv.weight
        x = torch.randn(1, 2, 9, 9, 5, device=rank)
        with torch.inference_mode():
            expected_conv = conv(x)
            shard, height = split_height_for_parallel_decode(
                x, x.shape[-2], group.world_size, group.rank_in_group
            )
            # force different temporal chunk counts on uneven height shards
            with patch(
                "sglang.multimodal_gen.runtime.models.vaes.kandinsky6_sr_vae._MAX_CONV_NUMEL",
                200,
            ):
                actual_conv = gather_and_trim_height(parallel_conv(shard), height)
                torch.testing.assert_close(actual_conv, expected_conv)
                with disable_spatial_parallel_decode():
                    torch.testing.assert_close(parallel_conv(x), expected_conv)
    config = Kandinsky6SRVAEConfig()
    config.update_model_arch(
        dict(
            encoder_config=dict(TINY_KVAE, in_channels=3),
            decoder_config=dict(TINY_KVAE, out_ch=3),
            scaling_factor=0.5,
            spatial_factor=2 ** (len(TINY_KVAE["ch_mult"]) - 1),
        )
    )
    if spatial_active:
        # fp32 sharding must be bit-exact; bf16 may only differ by the rare
        # shape-dependent flips calibrated in _assert_bf16_tile_close.
        passes = ((torch.float32, (False, True)), (torch.bfloat16, (False,)))
    else:
        passes = ((torch.bfloat16, (False, True)),)
    for vae_dtype, offloads in passes:
        torch.manual_seed(17)
        reference_vae = Kandinsky6SRVAE(config).eval().to(device=rank, dtype=vae_dtype)
        vae_config = replace(
            config, use_parallel_decode=spatial, parallel_decode_mode="spatial_shard"
        )
        vae = Kandinsky6SRVAE(vae_config).eval().to(device=rank, dtype=vae_dtype)
        vae.load_state_dict(reference_vae.state_dict())
        assert (
            any(isinstance(m, SpatialParallelConv3d) for m in vae.decoder.modules())
            == spatial_active
        )
        bit_exact = vae_dtype == torch.float32 or not spatial_active
        exact_fp32 = _exact_fp32() if vae_dtype == torch.float32 else nullcontext()
        with exact_fp32, torch.inference_mode():
            for offload in offloads:
                _run_requests(
                    rank=rank,
                    group=group,
                    vae=vae,
                    reference_vae=reference_vae,
                    vae_config=vae_config,
                    spatial_active=spatial_active,
                    offload=offload,
                    bit_exact=bit_exact,
                )
    dist.destroy_process_group()


@contextmanager
def _exact_fp32():
    # Keep the decode in fp32: skip the stage's bf16 autocast, and disable cuDNN
    # TF32, whose conv rounding is shape dependent on B200 (1/255 on ~2% of pixels).
    allow_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.tiled.bf16_autocast",
            lambda device: nullcontext(),
        ):
            yield
    finally:
        torch.backends.cudnn.allow_tf32 = allow_tf32


# Calibrated on H100 and B200 (2 and 4 ranks, ~1400 random tiles each): bf16 conv
# rounding is shape dependent, and this tiny random VAE amplifies a rare 1-ulp
# flip so that up to 0.035% of pixels exceed 4; a one-row halo or trim bug
# moves at least 0.55%, so 0.1% leaves margin on both sides.
_BF16_TILE_ATOL = 4
_BF16_TILE_MAX_OUTLIER_FRACTION = 1e-3


def _assert_bf16_tile_close(tile: torch.Tensor, reference: torch.Tensor) -> None:
    outliers = (tile.int() - reference.int()).abs() > _BF16_TILE_ATOL
    fraction = outliers.float().mean().item()
    assert fraction <= _BF16_TILE_MAX_OUTLIER_FRACTION, (
        f"{fraction:.4%} of pixels differ by more than {_BF16_TILE_ATOL} "
        f"(allowed {_BF16_TILE_MAX_OUTLIER_FRACTION:.2%})"
    )


def _run_requests(
    *,
    rank,
    group,
    vae,
    reference_vae,
    vae_config,
    spatial_active,
    offload,
    bit_exact,
):
    pipeline_config = Kandinsky6SRPipelineConfig(vae_config=vae_config)
    args = SimpleNamespace(pipeline_config=pipeline_config, component_precisions={})
    set_global_server_args(args)
    stage = Kandinsky6SRDecodeStage(vae)
    stage.server_args = args
    if offload:
        vae.configure_layerwise_offload(
            SimpleNamespace(
                performance_mode="speed",
                pin_cpu_memory=True,
                layerwise_tuning_for=lambda *args, **kwargs: (
                    1,
                    0,
                    "leading",
                    "forward",
                ),
            )
        )
    first = None
    # Uneven batches, idle ranks, two temporal segments, then a repeat.
    for sizes, frames, height in (
        ((2, 2, 1), 9, 5),
        ((1,), 1, 1),
        ((2, 1), 5, 4),
        ((2, 2, 1), 9, 5),
    ):
        generator = torch.Generator().manual_seed(7 + get_dp_rank())
        chunks = [
            torch.randn(size, frames, height, 3, 4, generator=generator)
            for size in sizes
        ]
        reference_vae.prepare_for_next_req()
        expected = decode_chunks(
            chunks,
            reference_vae,
            scaling_factor=reference_vae.scaling_factor,
            device=torch.device("cuda", rank),
        )
        for parallel in (False, True):
            vae_config.use_parallel_tiling = parallel
            vae.prepare_for_next_req()
            request = Req(sampling_params=Kandinsky6SRSamplingParams())
            request.extra[SR_DENOISED_KEY] = chunks
            with (
                patch.object(vae, "decode", wraps=vae.decode) as decode,
                patch.object(
                    vae.decoder, "forward", wraps=vae.decoder.forward
                ) as decoder,
            ):
                actual = stage.forward(request, args).extra[SR_TILES_KEY]
            count = len(range(group.rank_in_group, sum(sizes), group.world_size))
            assert decode.call_count == (
                count
                if parallel and not spatial_active and sum(sizes) > 1
                else sum(sizes)
            )
            local_height = height
            if spatial_active and height >= group.world_size:
                local_height = height // group.world_size + (
                    group.rank_in_group < height % group.world_size
                )
            assert all(
                call.args[0].shape[-2] == local_height
                for call in decoder.call_args_list
            )
            assert len(actual) == len(expected)
            for tile, reference in zip(actual, expected):
                assert tile.device.type == "cpu" and tile.dtype == torch.uint8
                if bit_exact:
                    torch.testing.assert_close(tile, reference, rtol=0, atol=0)
                else:
                    _assert_bf16_tile_close(tile, reference)
        if frames == 9:
            if first is None:
                first = actual
            else:
                for tile, reference in zip(actual, first):
                    torch.testing.assert_close(tile, reference, rtol=0, atol=0)
    if offload:
        vae.disable_offload()


@pytest.mark.parametrize(
    "world_size,tp_size,dp_size",
    [(1, 1, 1), (2, 2, 1), (2, 1, 1), (4, 2, 1), (4, 1, 2)],
)
@pytest.mark.parametrize("spatial", [False, True], ids=["tiles", "spatial"])
def test_sr_parallel_decode(world_size, tp_size, dp_size, spatial):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} CUDA GPUs")
    mp.spawn(
        _decode,
        args=(world_size, tp_size, dp_size, get_free_port_below_ephemeral(), spatial),
        nprocs=world_size,
    )
