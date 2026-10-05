"""Cake all-gather matmul through sglang.kernels.

Checks three things for the Cake adapter distributed by FlashInfer: the
registry resolves the explicit FlashInfer backend (no GPU); the in-process
``supports_*`` admission admits the engine's operands (any row count, the
``[N, K]`` parameter through its ``.t()`` view, any ``N % 256 == 0``), rejects
wrong world sizes, shapes, strides and dtypes and returns False when the
installed FlashInfer lacks the Cake module; and, when launched under
``torchrun`` with 2, 4 or 8 ranks on sm_100a / sm_103a with the NVSHMEM
symmetric-memory backend, the fused result matches NCCL all-gather +
``torch.matmul`` for the one-shot kernel and the capacity-bound prepared
launcher. The multi-rank tests skip with the reason otherwise.

Usage::

    python test/registered/kernels/ops/communication/test_cake_all_gather_matmul.py
    python test/registered/kernels/ops/communication/test_cake_all_gather_matmul.py --num-gpu 4
"""

from __future__ import annotations

import os
from typing import Optional

import pytest
import torch
import torch.distributed as dist

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import communication as cake_comm
from sglang.kernels.ops.communication.cake import (
    cake_all_gather_matmul,
    cake_prepare_all_gather_matmul,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
register_cuda_ci(est_time=180, stage="nightly", runner_config="8-gpu-b200")

OPS = ("communication.all_gather_matmul", "communication.prepare_all_gather_matmul")


@pytest.mark.parametrize("op", OPS)
def test_registry_resolves_flashinfer_backend(op):
    spec = select_kernel(op, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.communication:")


def _cuda_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda", torch.cuda.current_device())


def _module_available() -> bool:
    return cake_comm.flashinfer_module_available(
        cake_comm.FI_AG_MODULE, cake_comm.FI_AG_DISPATCH_MODULE
    )


def test_supports_admits_engine_operands_and_rejects_bad_inputs(monkeypatch):
    device = _cuda_or_skip()
    if not _module_available():
        pytest.skip("installed FlashInfer lacks the Cake all-gather matmul module")
    if torch.cuda.get_device_capability(device) not in cake_comm.ARCHS:
        pytest.skip("Cake all-gather matmul is built for sm_100a / sm_103a")
    inp = torch.empty(256, cake_comm.AG_K, device=device, dtype=torch.bfloat16)
    w = torch.empty(cake_comm.AG_K, 2048, device=device, dtype=torch.bfloat16)
    # The engine's [N, K] parameter through its transposed view (no copy).
    w_param = torch.empty(1280, cake_comm.AG_K, device=device, dtype=torch.bfloat16)
    assert cake_comm.supports_all_gather_matmul(inp, w, world_size=8)
    assert cake_comm.supports_all_gather_matmul(inp, w_param.t(), world_size=8)
    assert cake_comm.supports_all_gather_matmul(inp.half(), w.half(), world_size=2)
    # Any row count (tail rows are masked in the kernel); any N % 256 == 0.
    assert cake_comm.supports_all_gather_matmul(inp[:125], w_param.t(), world_size=8)
    assert cake_comm.supports_all_gather_matmul(inp[:1], w, world_size=4)
    for n in (256, 1280, 2560, 7168, 14336):
        wide = torch.empty(n, cake_comm.AG_K, device=device, dtype=torch.bfloat16)
        assert cake_comm.supports_all_gather_matmul(inp, wide.t(), world_size=8)
    assert not cake_comm.supports_all_gather_matmul(inp, w, world_size=3)
    assert not cake_comm.supports_all_gather_matmul(inp, w, world_size=16)
    assert not cake_comm.supports_all_gather_matmul(inp[:0], w, world_size=8)
    assert not cake_comm.supports_all_gather_matmul(inp, w[:, :1024], world_size=8)
    assert not cake_comm.supports_all_gather_matmul(inp, w[:, :1000], world_size=8)
    assert not cake_comm.supports_all_gather_matmul(inp, w[:4096], world_size=8)
    # A column slice of the parameter view is still a layout (the first rows of
    # the [N, K] parameter); a K-strided view is not.
    assert cake_comm.supports_all_gather_matmul(
        inp, w_param.t()[:, :1024], world_size=8
    )
    wide_param = torch.empty(
        1280, 2 * cake_comm.AG_K, device=device, dtype=torch.bfloat16
    )
    assert not cake_comm.supports_all_gather_matmul(
        inp,
        wide_param.t()[: cake_comm.AG_K],
        world_size=8,  # strides (1, 2K)
    )
    assert not cake_comm.supports_all_gather_matmul(
        inp.t().contiguous().t(),
        w,
        world_size=8,  # strided input
    )
    assert not cake_comm.supports_all_gather_matmul(
        inp.float(), w.float(), world_size=8
    )
    assert not cake_comm.supports_all_gather_matmul(inp, w.half(), world_size=8)
    assert not cake_comm.supports_all_gather_matmul(inp.cpu(), w.cpu(), world_size=8)

    # The prepared launcher admits the same operands plus a capacity >= rows.
    assert cake_comm.supports_prepare_all_gather_matmul(inp, w_param.t(), world_size=8)
    assert cake_comm.supports_prepare_all_gather_matmul(
        inp, w_param.t(), world_size=8, max_rows=2048
    )
    assert cake_comm.supports_prepare_all_gather_matmul(
        inp[:125].half(), w_param.t().half(), world_size=4, max_rows=125
    )
    assert not cake_comm.supports_prepare_all_gather_matmul(
        inp, w_param.t(), world_size=8, max_rows=255
    )
    assert not cake_comm.supports_prepare_all_gather_matmul(inp, w, world_size=3)
    assert not cake_comm.supports_prepare_all_gather_matmul(
        inp, w[:, :1000], world_size=8
    )

    monkeypatch.setattr(cake_comm, "flashinfer_module_available", lambda *a: False)
    assert not cake_comm.supports_all_gather_matmul(inp, w, world_size=8)
    assert not cake_comm.supports_prepare_all_gather_matmul(
        inp, w_param.t(), world_size=8
    )


# ---------------------------------------------------------------------------
# Multi-rank parity (torchrun workers only)
# ---------------------------------------------------------------------------


def _dist_group() -> Optional[dist.ProcessGroup]:
    """WORLD group when running inside a torchrun worker; ``None`` otherwise."""
    if dist.is_available() and dist.is_initialized():
        return dist.group.WORLD
    if "WORLD_SIZE" not in os.environ or "RANK" not in os.environ:
        return None
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group(backend="nccl")
    return dist.group.WORLD


@pytest.fixture(scope="session", autouse=True)
def _destroy_process_group_at_session_end():
    """Tear the NCCL group down while the interpreter is alive: destroying it
    from an ``atexit`` hook segfaults under the NVSHMEM symmetric-memory
    backend (torch 2.13), which torchrun then reports as a failed worker."""
    yield
    if dist.is_available() and dist.is_initialized():
        torch.cuda.synchronize()
        dist.barrier()
        dist.destroy_process_group()


def _multi_rank_setup(world_sizes):
    device = _cuda_or_skip()
    group = _dist_group()
    if group is None or dist.get_world_size(group) not in world_sizes:
        pytest.skip(f"needs {'/'.join(map(str, world_sizes))} ranks with NVSHMEM")
    if not _module_available():
        pytest.skip("installed FlashInfer lacks the Cake all-gather matmul module")
    if torch.cuda.get_device_capability(device) not in cake_comm.ARCHS:
        pytest.skip("Cake all-gather matmul is built for sm_100a / sm_103a")
    try:
        import torch.distributed._symmetric_memory as symm_mem

        symm_mem.set_backend("NVSHMEM")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"NVSHMEM symmetric-memory backend unavailable: {exc}")
    return device, group


def _reference(inp: torch.Tensor, w: torch.Tensor, group: dist.ProcessGroup):
    world = dist.get_world_size(group)
    gathered = torch.empty(
        world * inp.shape[0], inp.shape[1], dtype=inp.dtype, device=inp.device
    )
    dist.all_gather_into_tensor(gathered, inp.contiguous(), group=group)
    return gathered.float() @ w.float()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows", [256, 125])
def test_all_gather_matmul_matches_nccl_reference(dtype, rows):
    device, group = _multi_rank_setup(cake_comm.AG_WORLD_SIZES)
    rank = dist.get_rank(group)
    world = dist.get_world_size(group)
    torch.manual_seed(1000 + rank)
    inp = torch.randn(rows, cake_comm.AG_K, device=device, dtype=dtype)
    torch.manual_seed(7)  # replicated weight
    w = torch.randn(cake_comm.AG_K, 2048, device=device, dtype=dtype) * 0.02
    assert cake_comm.supports_all_gather_matmul(inp, w, world_size=world)
    out = cake_all_gather_matmul(inp, w, group)
    torch.cuda.synchronize()
    assert tuple(out.shape) == (world * rows, 2048)
    assert out.dtype == dtype
    # FP32 accumulation, one 16-bit rounding of the output.
    torch.testing.assert_close(
        out.float(), _reference(inp, w, group), atol=1e-2, rtol=1e-2
    )
    dist.barrier(group=group)


def test_all_gather_matmul_consumes_the_engine_parameter_view():
    """The [N, K] parameter's ``.t()`` view is a supported weight layout."""
    device, group = _multi_rank_setup(cake_comm.AG_WORLD_SIZES)
    rank = dist.get_rank(group)
    world = dist.get_world_size(group)
    torch.manual_seed(3000 + rank)
    inp = torch.randn(1025, cake_comm.AG_K, device=device, dtype=torch.bfloat16)
    torch.manual_seed(13)
    param = (
        torch.randn(2048, cake_comm.AG_K, device=device, dtype=torch.bfloat16) * 0.02
    )
    assert cake_comm.supports_all_gather_matmul(inp, param.t(), world_size=world)
    out = cake_all_gather_matmul(inp, param.t(), group)
    torch.cuda.synchronize()
    assert tuple(out.shape) == (world * 1025, 2048)
    torch.testing.assert_close(
        out.float(), _reference(inp, param.t(), group), atol=1e-2, rtol=1e-2
    )
    dist.barrier(group=group)


def test_prepared_launcher_serves_every_row_count_up_to_its_capacity():
    """Llama-3.1-70B column-parallel widths of this TP degree, [N, K] parameters."""
    device, group = _multi_rank_setup((4, 8))
    rank = dist.get_rank(group)
    world = dist.get_world_size(group)
    for n in {8: (1280, 7168), 4: (2560, 14336)}[world]:
        torch.manual_seed(2000 + rank)
        sample = torch.randn(512, cake_comm.AG_K, device=device, dtype=torch.bfloat16)
        torch.manual_seed(11)
        param = (
            torch.randn(n, cake_comm.AG_K, device=device, dtype=torch.bfloat16) * 0.02
        )
        assert cake_comm.supports_prepare_all_gather_matmul(
            sample, param.t(), world_size=world, max_rows=2048
        )
        launcher = cake_prepare_all_gather_matmul(
            sample, param.t(), group, max_rows=2048
        )
        for rows in (512, 125, 1025, 2048):
            inp = torch.randn(rows, cake_comm.AG_K, device=device, dtype=torch.bfloat16)
            out = launcher(inp)
            torch.cuda.synchronize()
            assert tuple(out.shape) == (world * rows, n)
            torch.testing.assert_close(
                out.float(), _reference(inp, param.t(), group), atol=1e-2, rtol=1e-2
            )
        with pytest.raises(ValueError):
            launcher(
                torch.randn(2049, cake_comm.AG_K, device=device, dtype=torch.bfloat16)
            )
        dist.barrier(group=group)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(2, 4, 8))
