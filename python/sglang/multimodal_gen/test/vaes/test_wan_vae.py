import os
import tempfile

import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.testing import assert_close
from sglang.multimodal_gen.configs.models.vaes import WanVAEConfig
from sglang.multimodal_gen.runtime.models.vaes.wanvae import (
    WanResample,
    WanDistResample,
    WanDecoder3d,
    WanCausalConv3d,
    WanDistCausalConv3d,
    WanDistConv2d,
    WanResidualBlock,
    WanDistResidualBlock,
    WanAttentionBlock,
    WanDistAttentionBlock,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
    destroy_distributed_environment,
)

backend = "nccl" if torch.cuda.is_available() else "gloo"
device = "cuda" if torch.cuda.is_available() else "cpu"

def set_envar(envar_map: dict):
    for k, v in envar_map.items():
        os.environ[k] = str(v)

@pytest.mark.parametrize("dim", [192, 384])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("height", [80])
@pytest.mark.parametrize("width", [40])
@pytest.mark.parametrize("ulysses_degree", [2, 4])
@pytest.mark.parametrize("ring_degree", [1])
@pytest.mark.parametrize(
    "kernel_size, padding",
    [
        (3, 1),
    ]
)
def test_wan_dist_conv_2d(
    dim: int,
    batch_size: int,
    height: int,
    width: int,
    ulysses_degree: int,
    ring_degree: int,
    kernel_size: int,
    padding: int
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        actual_output_file = f.name

    conv2d = nn.Conv2d(dim, dim * 3, kernel_size, padding=padding)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        saved_model_file = f.name
        torch.save(conv2d.state_dict(), f)

    conv2d = conv2d.eval().requires_grad_(False).to(device)

    # latent shape [B, C, H, W]
    x = torch.randn(batch_size, dim, height, width)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        input_data_file = f.name
        torch.save(x, f)
    x = x.to(device)

    expected = conv2d(x)

    sequence_parallel_size = ulysses_degree * ring_degree

    mp.spawn(
        _test_dist_conv,
        args=(
            sequence_parallel_size,
            input_data_file,
            saved_model_file,
            WanDistConv2d,
            dim,
            kernel_size,
            padding,
            ulysses_degree,
            ring_degree,
            sequence_parallel_size,
            actual_output_file,
        ),
        nprocs=sequence_parallel_size,
        join=True,
    )

    # verify the result
    with open(actual_output_file, "rb") as f:
        actual = torch.load(f)

    assert_close(actual, expected, atol=1e-3, rtol=1e-3)

    if os.path.exists(saved_model_file):
        os.remove(saved_model_file)
    if os.path.exists(input_data_file):
        os.remove(input_data_file)
    if os.path.exists(actual_output_file):
        os.remove(actual_output_file)

@pytest.mark.parametrize("dim", [192, 384])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("height", [80])
@pytest.mark.parametrize("width", [40])
@pytest.mark.parametrize("ulysses_degree", [2, 4])
@pytest.mark.parametrize("ring_degree", [1])
@pytest.mark.parametrize(
    "kernel_size, padding",
    [
        (3, 1),
        ((3, 1, 1), (1, 0, 0)), # time_conv
    ]
)
def test_wan_dist_conv_3d(
    dim: int,
    batch_size: int,
    height: int,
    width: int,
    ulysses_degree: int,
    ring_degree: int,
    kernel_size: int | tuple,
    padding: int | tuple,
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        actual_output_file = f.name

    conv3d = WanCausalConv3d(dim, dim * 3, kernel_size, padding=padding)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        saved_model_file = f.name
        torch.save(conv3d.state_dict(), f)

    conv3d = conv3d.eval().requires_grad_(False).to(device)

    # latent shape [B, C, H, W]
    x = torch.randn(batch_size, dim, 30, height, width)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        input_data_file = f.name
        torch.save(x, f)
    x = x.to(device)

    expected = conv3d(x)

    sequence_parallel_size = ulysses_degree * ring_degree

    mp.spawn(
        _test_dist_conv,
        args=(
            sequence_parallel_size,
            input_data_file,
            saved_model_file,
            WanDistCausalConv3d,
            dim,
            kernel_size,
            padding,
            ulysses_degree,
            ring_degree,
            sequence_parallel_size,
            actual_output_file,
        ),
        nprocs=sequence_parallel_size,
        join=True,
    )

    # verify the result
    with open(actual_output_file, "rb") as f:
        actual = torch.load(f)

    assert_close(actual, expected, atol=1e-3, rtol=1e-3)

    if os.path.exists(saved_model_file):
        os.remove(saved_model_file)
    if os.path.exists(input_data_file):
        os.remove(input_data_file)
    if os.path.exists(actual_output_file):
        os.remove(actual_output_file)


def _test_dist_conv(
    local_rank: int,
    world_size: int,
    input_data_file: str,
    saved_model_file: str,
    conv_cls: type,
    dim: int,
    kernel_size: int | tuple,
    padding: int | tuple,
    ulysses_degree: int,
    ring_degree: int,
    sequence_parallel_size: int,
    output_file: str
):
    set_envar({
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "43200",
        "RANK": str(local_rank),
        "LOCAL_RANK": str(local_rank),
        "WORLD_SIZE": str(world_size)
    })

    torch.cuda.set_device(local_rank)
    init_distributed_environment(
        rank=local_rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=backend
    )

    initialize_model_parallel(
        data_parallel_size=1,
        classifier_free_guidance_degree=1,
        tensor_parallel_degree=1,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        sequence_parallel_degree=sequence_parallel_size,
    )

    global device
    current_device = torch.device(f"{device}:{local_rank}")

    dist_conv = conv_cls(dim, dim * 3, kernel_size, padding=padding)
    dist_conv.load_state_dict(torch.load(saved_model_file))
    dist_conv = dist_conv.eval().requires_grad_(False).to(current_device)

    x = torch.load(input_data_file).to(current_device)
    x_local = torch.chunk(x, world_size, dim=-2)[local_rank].contiguous()

    out_local = dist_conv(x_local)

    tensor_list = [torch.empty_like(out_local) for _ in range(world_size)]
    dist.all_gather(tensor_list, out_local.contiguous())
    actual = torch.concat(tensor_list, dim=-2)

    if local_rank == 0:
        with open(output_file, "wb") as f:
            torch.save(actual, f)

    destroy_distributed_environment()


@pytest.mark.parametrize("dim", [192, 384])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("height", [80])
@pytest.mark.parametrize("width", [40])
@pytest.mark.parametrize("ulysses_degree", [2, 4])
@pytest.mark.parametrize("ring_degree", [1])
@pytest.mark.parametrize("mode", ["upsample2d", "upsample3d"])
def test_wan_dist_resample(
    dim: int,
    batch_size: int,
    height: int,
    width: int,
    ulysses_degree: int,
    ring_degree: int,
    mode: str,
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        actual_output_file = f.name

    resample = WanResample(dim, mode=mode)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        saved_model_file = f.name
        torch.save(resample.state_dict(), f)

    resample = resample.eval().requires_grad_(False).to(device)

    # latent shape [B, C, H, W]
    x = torch.randn(batch_size, dim, 1, height, width)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        input_data_file = f.name
        torch.save(x, f)
    x = x.to(device)

    expected = resample(x)

    sequence_parallel_size = ulysses_degree * ring_degree

    mp.spawn(
        _test_wan_dist_resample,
        args=(
            sequence_parallel_size,
            saved_model_file,
            dim,
            mode,
            input_data_file,
            ulysses_degree,
            ring_degree,
            sequence_parallel_size,
            actual_output_file,
        ),
        nprocs=sequence_parallel_size,
        join=True,
    )

    # verify the result
    with open(actual_output_file, "rb") as f:
        actual = torch.load(f)

    assert_close(actual, expected, atol=1e-3, rtol=1e-3)

    if os.path.exists(actual_output_file):
        os.remove(actual_output_file)
    if os.path.exists(saved_model_file):
        os.remove(saved_model_file)
    if os.path.exists(input_data_file):
        os.remove(input_data_file)

def _test_wan_dist_resample(
    local_rank: int,
    world_size: int,
    saved_model_file: str,
    dim: int,
    mode: str,
    input_data_file: str,
    ulysses_degree: int,
    ring_degree: int,
    sequence_parallel_size: int,
    output_file: str
):
    set_envar({
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "43200",
        "RANK": str(local_rank),
        "LOCAL_RANK": str(local_rank),
        "WORLD_SIZE": str(world_size)
    })

    torch.cuda.set_device(local_rank)
    init_distributed_environment(
        rank=local_rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=backend
    )

    initialize_model_parallel(
        data_parallel_size=1,
        classifier_free_guidance_degree=1,
        tensor_parallel_degree=1,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        sequence_parallel_degree=sequence_parallel_size,
    )

    global device
    current_device = torch.device(f"{device}:{local_rank}")

    resample = WanDistResample(dim, mode=mode)
    resample.load_state_dict(torch.load(saved_model_file))
    resample = resample.eval().requires_grad_(False).to(current_device)

    x = torch.load(input_data_file).to(current_device)
    x_local = torch.chunk(x, world_size, dim=-2)[local_rank]

    out_local = resample(x_local).contiguous()

    tensor_list = [torch.empty_like(out_local) for _ in range(world_size)]
    dist.all_gather(tensor_list, out_local)
    actual = torch.concat(tensor_list, dim=-2)

    if local_rank == 0:
        with open(output_file, "wb") as f:
            torch.save(actual, f)

    destroy_distributed_environment()


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("height", [80])
@pytest.mark.parametrize("width", [40])
@pytest.mark.parametrize("ulysses_degree", [2, 4])
@pytest.mark.parametrize("ring_degree", [1])
@pytest.mark.parametrize("in_dim", [192, 384])
@pytest.mark.parametrize("out_dim", [192, 384])
def test_wan_dist_residual_block(
    batch_size: int,
    height: int,
    width: int,
    ulysses_degree: int,
    ring_degree: int,
    in_dim: int,
    out_dim: int,
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        actual_output_file = f.name

    residual_block = WanResidualBlock(in_dim, out_dim)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        saved_model_file = f.name
        torch.save(residual_block.state_dict(), f)
    residual_block = residual_block.eval().requires_grad_(False).to(device)

    # latent shape [B, C, H, W]
    x = torch.randn(batch_size, in_dim, 1, height, width)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        input_data_file = f.name
        torch.save(x, f)
    x = x.to(device)

    expected = residual_block(x)

    sequence_parallel_size = ulysses_degree * ring_degree

    mp.spawn(
        _test_wan_dist_residual_block,
        args=(
            sequence_parallel_size,
            saved_model_file,
            input_data_file,
            ulysses_degree,
            ring_degree,
            in_dim,
            out_dim,
            sequence_parallel_size,
            actual_output_file,
        ),
        nprocs=sequence_parallel_size,
        join=True,
    )

    # verify the result
    with open(actual_output_file, "rb") as f:
        actual = torch.load(f)

    assert_close(actual, expected, atol=1e-3, rtol=1e-3)

    if os.path.exists(saved_model_file):
        os.remove(saved_model_file)
    if os.path.exists(input_data_file):
        os.remove(input_data_file)
    if os.path.exists(actual_output_file):
        os.remove(actual_output_file)

def _test_wan_dist_residual_block(
    local_rank: int,
    world_size: int,
    saved_model_file: str,
    input_data_file: str,
    ulysses_degree: int,
    ring_degree: int,
    in_dim: int,
    out_dim: int,
    sequence_parallel_size: int,
    output_file: str
):
    set_envar({
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "43200",
        "RANK": str(local_rank),
        "LOCAL_RANK": str(local_rank),
        "WORLD_SIZE": str(world_size)
    })

    torch.cuda.set_device(local_rank)
    init_distributed_environment(
        rank=local_rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=backend
    )

    initialize_model_parallel(
        data_parallel_size=1,
        classifier_free_guidance_degree=1,
        tensor_parallel_degree=1,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        sequence_parallel_degree=sequence_parallel_size,
        backend=backend
    )

    global device
    current_device = torch.device(f"{device}:{local_rank}")

    residual_block = WanDistResidualBlock(in_dim, out_dim)
    residual_block.load_state_dict(torch.load(saved_model_file))
    residual_block = residual_block.eval().requires_grad_(False).to(current_device)

    x = torch.load(input_data_file).to(current_device)
    x_local = torch.chunk(x, world_size, dim=-2)[local_rank]

    out_local = residual_block(x_local)

    tensor_list = [torch.empty_like(out_local) for _ in range(world_size)]
    dist.all_gather(tensor_list, out_local)
    actual = torch.concat(tensor_list, dim=-2)

    if local_rank == 0:
        with open(output_file, "wb") as f:
            torch.save(actual, f)

    destroy_distributed_environment()


@pytest.mark.parametrize("dim", [1, 192, 384])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("height", [80])
@pytest.mark.parametrize("width", [40])
@pytest.mark.parametrize("ulysses_degree", [2, 4])
@pytest.mark.parametrize("ring_degree", [1])
def test_wan_dist_attention_block(
    dim: int,
    batch_size: int,
    height: int,
    width: int,
    ulysses_degree: int,
    ring_degree: int,
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        actual_output_file = f.name

    block = WanAttentionBlock(dim)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        saved_model_file = f.name
        torch.save(block.state_dict(), f)
    block = block.eval().requires_grad_(False).to(device)

    # latent shape [B, C, H, W]
    x = torch.randn(batch_size, dim, 1, height, width)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        input_data_file = f.name
        torch.save(x, f)
    x = x.to(device)

    expected = block(x)

    sequence_parallel_size = ulysses_degree * ring_degree

    mp.spawn(
        _test_wan_dist_attention_block,
        args=(
            sequence_parallel_size,
            dim,
            saved_model_file,
            input_data_file,
            ulysses_degree,
            ring_degree,
            sequence_parallel_size,
            actual_output_file,
        ),
        nprocs=sequence_parallel_size,
        join=True,
    )

    # verify the result
    with open(actual_output_file, "rb") as f:
        actual = torch.load(f)

    assert_close(actual, expected, atol=1e-3, rtol=1e-3)

    if os.path.exists(saved_model_file):
        os.remove(saved_model_file)
    if os.path.exists(input_data_file):
        os.remove(input_data_file)
    if os.path.exists(actual_output_file):
        os.remove(actual_output_file)

def _test_wan_dist_attention_block(
    local_rank: int,
    world_size: int,
    dim: int,
    saved_model_file: str,
    input_data_file: str,
    ulysses_degree: int,
    ring_degree: int,
    sequence_parallel_size: int,
    output_file: str
):
    set_envar({
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "43200",
        "RANK": str(local_rank),
        "LOCAL_RANK": str(local_rank),
        "WORLD_SIZE": str(world_size)
    })

    torch.cuda.set_device(local_rank)
    init_distributed_environment(
        rank=local_rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=backend
    )

    initialize_model_parallel(
        data_parallel_size=1,
        classifier_free_guidance_degree=1,
        tensor_parallel_degree=1,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        sequence_parallel_degree=sequence_parallel_size,
        backend=backend
    )

    global device
    current_device = torch.device(f"{device}:{local_rank}")

    block = WanDistAttentionBlock(dim)
    block.load_state_dict(torch.load(saved_model_file))
    block = block.eval().requires_grad_(False).to(current_device)

    x = torch.load(input_data_file).to(current_device)
    x_local = torch.chunk(x, world_size, dim=-2)[local_rank].contiguous()

    out_local = block(x_local).contiguous()

    tensor_list = [torch.empty_like(out_local) for _ in range(world_size)]
    dist.all_gather(tensor_list, out_local)
    actual = torch.concat(tensor_list, dim=-2)

    if local_rank == 0:
        with open(output_file, "wb") as f:
            torch.save(actual, f)

    destroy_distributed_environment()


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("height", [90])
@pytest.mark.parametrize("width", [40])
@pytest.mark.parametrize("ulysses_degree", [2, 4])
@pytest.mark.parametrize("ring_degree", [1])
def test_wan_parallel_decoder(
    batch_size: int,
    height: int,
    width: int,
    ulysses_degree: int,
    ring_degree: int,
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        actual_output_file = f.name

    vae_config = WanVAEConfig()
    decoder = WanDecoder3d(
        dim=vae_config.base_dim,
        z_dim=vae_config.z_dim,
        dim_mult=vae_config.dim_mult,
        num_res_blocks=vae_config.num_res_blocks,
        attn_scales=vae_config.attn_scales,
        temperal_upsample=list(vae_config.temperal_downsample)[::-1],
        dropout=vae_config.dropout,
        out_channels=vae_config.out_channels,
        is_residual=vae_config.is_residual,
    )
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        saved_model_file = f.name
        torch.save(decoder.state_dict(), saved_model_file)
    decoder = decoder.eval().requires_grad_(False).to(device)

    # generate latent
    z = torch.randn(batch_size, vae_config.z_dim, 1, height, width)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        input_data_file = f.name
        torch.save(z, f)
    z = z.to(device)

    expected = decoder(z)

    sequence_parallel_size = ulysses_degree * ring_degree

    mp.spawn(
        _test_wan_parallel_decoder,
        args=(
            sequence_parallel_size,
            saved_model_file,
            input_data_file,
            ulysses_degree,
            ring_degree,
            sequence_parallel_size,
            actual_output_file,
        ),
        nprocs=sequence_parallel_size,
        join=True,
    )

    with open(actual_output_file, "rb") as f:
        actual = torch.load(f)

    assert actual.shape == expected.shape
    assert_close(actual, expected, atol=5e-2, rtol=5e-2)

    if os.path.exists(saved_model_file):
        os.remove(saved_model_file)
    if os.path.exists(input_data_file):
        os.remove(input_data_file)
    if os.path.exists(actual_output_file):
        os.remove(actual_output_file)


def _test_wan_parallel_decoder(
    local_rank: int,
    world_size: int,
    saved_model_file: str,
    input_data_file: str,
    ulysses_degree: int,
    ring_degree: int,
    sequence_parallel_size: int,
    output_file: str
):
    set_envar({
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "43210",
        "RANK": str(local_rank),
        "LOCAL_RANK": str(local_rank),
        "WORLD_SIZE": str(world_size)
    })

    torch.cuda.set_device(local_rank)
    init_distributed_environment(
        rank=local_rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=backend
    )

    initialize_model_parallel(
        data_parallel_size=1,
        classifier_free_guidance_degree=1,
        tensor_parallel_degree=1,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        sequence_parallel_degree=sequence_parallel_size,
        backend=backend
    )

    global device
    current_device = torch.device(f"{device}:{local_rank}")

    vae_config = WanVAEConfig()

    parallel_decoder = WanDecoder3d(
        dim=vae_config.base_dim,
        z_dim=vae_config.z_dim,
        dim_mult=vae_config.dim_mult,
        num_res_blocks=vae_config.num_res_blocks,
        attn_scales=vae_config.attn_scales,
        temperal_upsample=list(vae_config.temperal_downsample)[::-1],
        dropout=vae_config.dropout,
        out_channels=vae_config.out_channels,
        is_residual=vae_config.is_residual,
        use_parallel_decode=True,
    )
    parallel_decoder.load_state_dict(torch.load(saved_model_file))
    parallel_decoder = parallel_decoder.eval().requires_grad_(False).to(current_device)

    z = torch.load(input_data_file).to(current_device)

    actual = parallel_decoder(z)

    if local_rank == 0:
        with open(output_file, "wb") as f:
            torch.save(actual, f)

    destroy_distributed_environment()