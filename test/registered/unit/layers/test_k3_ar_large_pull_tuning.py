# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Multi-GPU test of SGLANG_K3_AR_LARGE_PULL_TUNING (K3 AR fusion, NVLS pull kernels with more CTAs above 4 MB).

Checks, with tolerances fixed before the first run:
  * wiring: above 4 MB the flag launches the large (num_blocks, unroll) for the o_proj AR (pull_res), the MoE latent
    AR + RMSNorm (pull_norm) and the overlapped shared AR (low-SM pull_res); at or below 4 MB, and with the flag off,
    the stock tables are used (num_blocks/unroll None or the caller's own);
  * numerics: flag-on output is bit-identical to flag-off output (the CTA count only partitions rows: one
    multimem.ld_reduce per 16 B vector either way), and both match an independent fp32 reference of the same bf16
    inputs within 1 bf16 ulp-scale (max|o-ref|/max|ref| <= 1e-2).
Needs multicast (NVLS) between the ranks, i.e. an NVLink-switch system (SM100/103/107).

Usage::

    python test/registered/unit/layers/test_k3_ar_large_pull_tuning.py
"""

from __future__ import annotations

import atexit
import os
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.utils import cache_once
from sglang.srt.environ import envs
from sglang.srt.runtime_context import SpawnRanks, get_parallel
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main
from sglang.test.parallel_groups import publish

register_cuda_ci(est_time=60, stage="base-b", runner_config="4-gpu-b200")

H, LAT = 7168, 3584
# The expected table (from a TP4 NVLS sweep): independent of the module's constants so a changed table fails here.
LARGE_BYTES = 4 * 1024 * 1024
EXPECTED_LARGE = {"oproj": (32, 8), "latent_norm": (128, 2), "shared_low_sm": (16, 8)}


@cache_once
def _fusion():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    os.environ["SGLANG_K3_AR_FUSION"] = "1"
    dist.init_process_group(backend="gloo")
    publish(
        ServerArgs(model_path="dummy", tp_size=world_size, chunked_prefill_size=4096),
        role="test",
        ranks=SpawnRanks(world_rank=local_rank),
    )
    ps._WORLD = ps.init_world_group(
        ranks=list(range(world_size)), local_rank=local_rank, backend="nccl"
    )
    get_parallel().override_permanently(world_group=ps._WORLD, tp_group=ps._WORLD)
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )
    from sglang.srt.layers.communication import k3_ar_fusion

    ps._WORLD.ca_comm = CustomAllReduceV2(
        ps._WORLD.cpu_group, torch.device("cuda", local_rank)
    )
    atexit.register(dist.destroy_process_group)
    if not k3_ar_fusion.enabled():
        pytest.skip("K3 AR fusion needs CustomAllReduceV2 with multicast")
    # Deliberate breaks (K3_LPT_MUTATE); each must make this test FAIL.
    mutate = os.environ.get("K3_LPT_MUTATE", "")
    if mutate == "flag_ignored":
        k3_ar_fusion._LARGE_PULL_RES = (None, None)
    elif mutate == "threshold_zero":
        k3_ar_fusion._LARGE_PULL_BYTES = 0
    elif mutate == "norm_gets_res_tuning":
        k3_ar_fusion._LARGE_PULL_NORM = k3_ar_fusion._LARGE_PULL_RES
    elif mutate:
        raise ValueError(mutate)
    return k3_ar_fusion


def _record(mod, name):
    calls = []
    real = getattr(mod, name)

    def wrapped(*args, **kwargs):
        calls.append((kwargs.get("num_blocks"), kwargs.get("unroll")))
        return real(*args, **kwargs)

    return calls, patch.object(mod, name, wrapped)


def _ref_sum(local: torch.Tensor) -> torch.Tensor:
    full = local.float().clone()
    dist.all_reduce(full, group=ps._WORLD.device_group)
    return full


@pytest.mark.parametrize("rows", [64, 1024, 4096])
@pytest.mark.parametrize("kind", ["oproj", "latent_norm", "shared_low_sm"])
def test_large_pull_tuning(kind, rows):
    fusion = _fusion()
    from sglang.kernels.ops.communication import all_reduce_residual as mod

    g = torch.Generator(device="cuda").manual_seed(1000 * rows + dist.get_rank())
    width = LAT if kind == "latent_norm" else H
    src = (torch.randn(rows, width, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    resid = torch.randn(
        rows, width, generator=torch.Generator(device="cuda").manual_seed(rows), device="cuda"
    ).to(torch.bfloat16)
    weight = (
        1 + 0.1 * torch.randn(LAT, generator=torch.Generator(device="cuda").manual_seed(5), device="cuda")
    ).to(torch.bfloat16)
    if kind == "latent_norm":
        buf = fusion.symm_buffer(fusion.MOE_LATENT_SHARED, rows, LAT + H, torch.bfloat16).view(-1)
        x = buf[: rows * LAT].view(rows, LAT)
    else:
        x = fusion.symm_buffer(fusion.ATTN_O_PROJ, rows, H, torch.bfloat16)
    nbytes = src.numel() * 2
    large = nbytes > LARGE_BYTES and nbytes > fusion._PUSH_MAX_BYTES

    def run(flag):
        x.copy_(src)
        op = "all_reduce_pull_norm" if kind == "latent_norm" else "all_reduce_pull_res"
        calls, ctx = _record(mod, op)
        with envs.SGLANG_K3_AR_LARGE_PULL_TUNING.override(flag), ctx:
            if kind == "oproj":
                fusion.all_reduce(x, resid)
            elif kind == "latent_norm":
                fusion.all_reduce_norm(x, weight, 1e-5, num_tokens=rows)
            else:
                fusion.all_reduce_low_sm(x, num_blocks=4, unroll=8)
        torch.cuda.synchronize()
        return x.clone(), calls

    off, calls_off = run(False)
    on, calls_on = run(True)
    assert torch.equal(on, off), "the CTA count must not change any bit"

    # Wiring: the large config only above the threshold and only with the flag.
    if nbytes <= fusion._PUSH_MAX_BYTES and kind != "shared_low_sm":
        assert calls_on == [] and calls_off == [], "push path: no pull call expected"
    else:
        stock = {"oproj": (None, None), "latent_norm": (None, None), "shared_low_sm": (4, 8)}[kind]
        assert calls_off == [stock], calls_off
        if large or (kind == "shared_low_sm" and nbytes > LARGE_BYTES):
            assert calls_on == [EXPECTED_LARGE[kind]], calls_on
        else:
            assert calls_on == [stock], calls_on

    # Independent reference: fp32 sum of the bf16 inputs (+ residual / RMSNorm).
    ref = _ref_sum(src)
    if kind == "oproj":
        ref = ref.to(torch.bfloat16).float() + resid.float()
    elif kind == "latent_norm":
        ref = ref.to(torch.bfloat16).float()
        ref = ref * torch.rsqrt(ref.pow(2).mean(-1, keepdim=True) + 1e-5) * weight.float()
    err = ((on.float() - ref).abs().max() / ref.abs().max()).item()
    assert err <= 1e-2, err


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=[4])
