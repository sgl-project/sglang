"""Four-GPU checks for shared replicated-Q projection weight storage.

Usage::

    python \
      test/registered/kernels/ops/communication/test_dcp_replicated_query_weights.py \
      --num-gpu 4
"""

from __future__ import annotations

import atexit
import functools
import gc
import json
import logging
import os

import pytest
import sglang.srt.distributed.parallel_state as ps
import torch
import torch.distributed as dist
from sglang.srt.layers.dcp.query_weights import (
    bind_parameter_to_replicated_rank_slice_,
    refresh_replicated_weight_,
    replicated_rank_slice,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=60, stage="extra-b", runner_config="4-gpu-h200")

WORLD_SIZE = 4
LOCAL_HEADS = 24
QK_HEAD_DIM = 192
Q_LORA_RANK = 1536
LOCAL_ROWS = LOCAL_HEADS * QK_HEAD_DIM
LOCAL_WEIGHT_BYTES = LOCAL_ROWS * Q_LORA_RANK * torch.bfloat16.itemsize
LOCAL_W_KC_BYTES = LOCAL_HEADS * 128 * 512 * torch.bfloat16.itemsize


@functools.cache
def _init_world() -> ps.GroupCoordinator:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    atexit.register(dist.destroy_process_group)
    logging.disable(logging.INFO)
    assert ps._WORLD is not None
    return ps._WORLD


@torch.inference_mode()
def test_dcp_replicated_query_storage_alias_refresh_and_memory() -> None:
    world = _init_world()
    if world.world_size != WORLD_SIZE:
        pytest.skip("The Kimi-K3 TP4/DCP4 storage test requires four ranks.")

    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    local = torch.nn.Parameter(
        torch.full(
            (LOCAL_ROWS, Q_LORA_RANK),
            world.rank_in_group,
            dtype=torch.bfloat16,
            device=device,
        ),
        requires_grad=False,
    )
    after_local = torch.cuda.memory_allocated(device)
    replicated = world.all_gather(local.data.contiguous(), dim=0)
    torch.cuda.synchronize(device)
    after_gather = torch.cuda.memory_allocated(device)
    replicated_ptr = replicated.data_ptr()

    bind_parameter_to_replicated_rank_slice_(
        local,
        replicated,
        rank=world.rank_in_group,
        world_size=world.world_size,
    )
    gc.collect()
    torch.cuda.synchronize(device)
    after_alias = torch.cuda.memory_allocated(device)

    assert local.is_contiguous()
    assert local.untyped_storage().data_ptr() == replicated.untyped_storage().data_ptr()
    assert local.storage_offset() == world.rank_in_group * local.numel()
    # The local 13.5 MiB q_b allocation is released; tolerate one allocator page.
    assert after_gather - after_alias >= LOCAL_WEIGHT_BYTES - (2 << 20)
    assert after_gather - after_local >= replicated.nbytes
    if world.rank_in_group == 0:
        print(
            "QREP_STORAGE_RESULT "
            + json.dumps(
                {
                    "local_weight_bytes": LOCAL_WEIGHT_BYTES,
                    "allocated_after_local": after_local,
                    "allocated_after_gather": after_gather,
                    "allocated_after_alias": after_alias,
                    "released_bytes": after_gather - after_alias,
                },
                sort_keys=True,
            )
        )

    expected = torch.arange(
        WORLD_SIZE, dtype=torch.bfloat16, device=device
    ).repeat_interleave(LOCAL_ROWS * Q_LORA_RANK)
    torch.testing.assert_close(replicated.flatten(), expected, atol=0, rtol=0)

    local_w_kc = (
        torch.full(
            (LOCAL_HEADS, 128, 512),
            world.rank_in_group,
            dtype=torch.bfloat16,
            device=device,
        )
        .transpose(1, 2)
        .contiguous()
        .transpose(1, 2)
    )
    assert local_w_kc.stride() == (65536, 1, 128)
    w_kc_after_local = torch.cuda.memory_allocated(device)
    full_w_kc = world.all_gather(local_w_kc.contiguous(), dim=0)
    torch.cuda.synchronize(device)
    w_kc_after_gather = torch.cuda.memory_allocated(device)
    full_w_kc_ptr = full_w_kc.data_ptr()
    local_w_kc = replicated_rank_slice(
        full_w_kc,
        local_shape=local_w_kc.shape,
        rank=world.rank_in_group,
        world_size=world.world_size,
    )
    gc.collect()
    torch.cuda.synchronize(device)
    w_kc_after_alias = torch.cuda.memory_allocated(device)

    assert full_w_kc.stride() == (65536, 512, 1)
    assert local_w_kc.stride() == (65536, 512, 1)
    assert (
        local_w_kc.untyped_storage().data_ptr()
        == full_w_kc.untyped_storage().data_ptr()
    )
    assert w_kc_after_gather - w_kc_after_alias >= LOCAL_W_KC_BYTES

    # Simulate Kimi-K3 post_load_weights reconstructing an independent
    # K-contiguous local tensor during an online reload.
    reloaded_local_w_kc = (
        torch.full_like(local_w_kc, 20 + world.rank_in_group)
        .transpose(1, 2)
        .contiguous()
        .transpose(1, 2)
    )
    assert reloaded_local_w_kc.stride() == (65536, 1, 128)
    refresh_replicated_weight_(reloaded_local_w_kc, full_w_kc, group=world)
    local_w_kc = replicated_rank_slice(
        full_w_kc,
        local_shape=reloaded_local_w_kc.shape,
        rank=world.rank_in_group,
        world_size=world.world_size,
    )
    torch.cuda.synchronize(device)
    assert full_w_kc.data_ptr() == full_w_kc_ptr
    assert (
        local_w_kc.untyped_storage().data_ptr()
        == full_w_kc.untyped_storage().data_ptr()
    )
    for rank in range(WORLD_SIZE):
        torch.testing.assert_close(
            full_w_kc[rank * LOCAL_HEADS : (rank + 1) * LOCAL_HEADS],
            torch.full_like(local_w_kc, 20 + rank),
            atol=0,
            rtol=0,
        )

    if world.rank_in_group == 0:
        print(
            "QREP_W_KC_STORAGE_RESULT "
            + json.dumps(
                {
                    "local_weight_bytes": LOCAL_W_KC_BYTES,
                    "allocated_after_local": w_kc_after_local,
                    "allocated_after_gather": w_kc_after_gather,
                    "allocated_after_alias": w_kc_after_alias,
                    "released_bytes": w_kc_after_gather - w_kc_after_alias,
                },
                sort_keys=True,
            )
        )

    local.fill_(10 + world.rank_in_group)
    refresh_replicated_weight_(local, replicated, group=world)
    torch.cuda.synchronize(device)
    assert replicated.data_ptr() == replicated_ptr
    expected = torch.arange(
        10, 10 + WORLD_SIZE, dtype=torch.bfloat16, device=device
    ).repeat_interleave(LOCAL_ROWS * Q_LORA_RANK)
    torch.testing.assert_close(replicated.flatten(), expected, atol=0, rtol=0)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,))
