"""Wan VAE communication must follow the group that shards its activations."""

import argparse
import os
import signal
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    destroy_model_parallel,
    get_decode_parallel_group_coordinator,
    get_sp_group,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.multimodal_gen.runtime.layers.parallel_conv import (
    SpatialParallelConv2d,
    chunk_height_for_parallel_decode,
    gather_variable_height,
)
from sglang.multimodal_gen.runtime.models.vaes.wanvae import (
    WanDecoder3d,
    WanEncoder3d,
)


def _worker(sp_size, tp_size):
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    init_distributed_environment(world_size=4, rank=rank, local_rank=rank)
    initialize_model_parallel(
        tensor_parallel_degree=tp_size,
        sequence_parallel_degree=sp_size,
        ulysses_degree=sp_size,
        ring_degree=1,
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sp_group = get_sp_group()
    decode_group = get_decode_parallel_group_coordinator()
    assert sp_group.world_size == sp_size
    assert decode_group.world_size == 4

    # distinct TP replicas catch collectives that accidentally use the decode group
    torch.manual_seed(42)
    full = torch.randn(1, 2, 9, 8, device="cuda") + get_tp_group().rank_in_group
    local = chunk_height_for_parallel_decode(full, parallel_group=sp_group)
    gathered, heights = gather_variable_height(local, parallel_group=sp_group)
    torch.testing.assert_close(gathered, full, rtol=0, atol=0)
    assert heights == [part.shape[-2] for part in torch.tensor_split(full, sp_size, -2)]
    conv = SpatialParallelConv2d(
        2, 3, 3, stride=2, padding=1, parallel_group=sp_group
    ).cuda()
    expected = F.conv2d(full, conv.weight, conv.bias, stride=2, padding=1)
    actual, _ = gather_variable_height(conv(local), parallel_group=sp_group)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    # exercise real Wan block construction, attention, downsampling and final gather
    for residual in (False, True):
        torch.manual_seed(123)
        kwargs = dict(
            dim=4,
            z_dim=2,
            dim_mult=(1, 2, 2),
            num_res_blocks=1,
            temperal_downsample=(False, False),
            is_residual=residual,
        )
        reference = WanEncoder3d(**kwargs).cuda().eval()
        parallel = WanEncoder3d(**kwargs, use_parallel_encode=True).cuda().eval()
        parallel.load_state_dict(reference.state_dict())
        if sp_size > 1:
            assert parallel.parallel_group is sp_group
            assert parallel.conv_in.parallel_group is sp_group
            assert parallel.conv_out.parallel_group is sp_group
        x = torch.randn(1, 3, 3, 32, 16, device="cuda") + get_tp_group().rank_in_group
        with torch.inference_mode():
            torch.testing.assert_close(parallel(x), reference(x), rtol=1e-4, atol=1e-4)

    # default decoder callers must still use the full decode group, not SP
    torch.manual_seed(321)
    kwargs = dict(
        dim=4, z_dim=2, dim_mult=(1, 1), num_res_blocks=1, temperal_upsample=(False,)
    )
    reference = WanDecoder3d(**kwargs).cuda().eval()
    parallel = WanDecoder3d(**kwargs, use_parallel_decode=True).cuda().eval()
    parallel.load_state_dict(reference.state_dict())
    assert parallel.conv_in.parallel_group is None
    assert parallel.conv_in.world_size == decode_group.world_size
    z = torch.randn(1, 2, 1, 8, 4, device="cuda")
    with torch.inference_mode():
        torch.testing.assert_close(parallel(z), reference(z), rtol=1e-4, atol=1e-4)
    destroy_model_parallel()
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("sp_size,tp_size", [(1, 4), (2, 2), (4, 1)])
def test_wan_vae_parallel_groups(sp_size, tp_size):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        pytest.skip("requires four CUDA GPUs")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=4",
            __file__,
            "--worker",
            "--sp-size",
            str(sp_size),
            "--tp-size",
            str(tp_size),
        ],
        start_new_session=True,
    )
    try:
        returncode = process.wait(timeout=180)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, process.args)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--worker", action="store_true")
        parser.add_argument("--sp-size", type=int, required=True)
        parser.add_argument("--tp-size", type=int, required=True)
        args = parser.parse_args()
        _worker(args.sp_size, args.tp_size)
    else:
        sys.exit(pytest.main([__file__, *sys.argv[1:]]))
