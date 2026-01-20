from __future__ import annotations

import socket
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.fla.layernorm_gated import (
    _layer_norm_fwd as layer_norm_fwd,
)
from sglang.srt.layers.attention.fla.layernorm_gated import layernorm_fn, rms_norm_ref

# Optional dependency in sglang repo; skip collection cleanly if absent.
custom_all_reduce_utils = pytest.importorskip(
    "sglang.srt.distributed.device_communicators.custom_all_reduce_utils"
)
parallel_state = pytest.importorskip("sglang.srt.distributed.parallel_state")

update_environment_variables = custom_all_reduce_utils.update_environment_variables
init_distributed_environment = parallel_state.init_distributed_environment
initialize_model_parallel = parallel_state.initialize_model_parallel

NUM_GPUS = 2


def _find_free_port() -> int:
    # Avoid hard-coded port collisions when pytest runs tests in parallel.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        s.listen(1)
        return int(s.getsockname()[1])


def _skip_if_no_cuda_or_not_enough_gpus(required_gpus: int = NUM_GPUS) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")
    if torch.cuda.device_count() < required_gpus:
        pytest.skip(f"Need >= {required_gpus} GPUs, got {torch.cuda.device_count()}")


def _skip_if_dtype_unsupported(dtype: torch.dtype) -> None:
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 not supported on this CUDA device")


def _setup_sglang_distributed(
    local_rank: int,
    world_size: int,
    master_port: int,
    dtype: torch.dtype,
) -> torch.device:
    # Match sglang test style: set per-rank CUDA device + default dtype/device.
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if hasattr(torch, "set_default_device"):
        torch.set_default_device(device)
    if hasattr(torch, "set_default_dtype"):
        torch.set_default_dtype(dtype)

    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(master_port),
        }
    )

    init_distributed_environment(
        world_size=world_size, rank=local_rank, local_rank=local_rank
    )
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    return device


def layer_norm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    z: torch.Tensor | None = None,
    eps: float = 1e-6,
    group_size: int | None = None,
    norm_before_gate: bool = True,
    is_rms_norm: bool = False,
) -> torch.Tensor:
    """Reference implementation for both LayerNorm and RMSNorm (supports optional gate + group norm)."""
    if is_rms_norm:
        return rms_norm_ref(
            x,
            weight,
            bias,
            z=z,
            eps=eps,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            upcast=True,
        )

    dtype = x.dtype
    x_f = x.float()
    w_f = weight.float()
    b_f = bias.float() if bias is not None else None
    z_f = z.float() if z is not None else None

    if z_f is not None and not norm_before_gate:
        x_f = x_f * F.silu(z_f)

    if group_size is None:
        mean = x_f.mean(dim=-1, keepdim=True)
        var = (x_f - mean).square().mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + eps)
        out = (x_f - mean) * rstd * w_f
        if b_f is not None:
            out = out + b_f
    else:
        hidden = x_f.shape[-1]
        assert hidden % group_size == 0
        ng = hidden // group_size
        xg = x_f.view(*x_f.shape[:-1], ng, group_size)
        mean = xg.mean(dim=-1, keepdim=True)
        var = (xg - mean).square().mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + eps)
        xg = (xg - mean) * rstd
        out = xg.reshape(*x_f.shape[:-1], hidden) * w_f
        if b_f is not None:
            out = out + b_f

    if z_f is not None and norm_before_gate:
        out = out * F.silu(z_f)

    return out.to(dtype)


@dataclass(frozen=True)
class FwdCase:
    name: str
    with_gate: bool
    norm_before_gate: bool
    group_size: int | None
    is_rms_norm: bool


CASES: list[FwdCase] = [
    FwdCase(
        "layernorm",
        with_gate=False,
        norm_before_gate=True,
        group_size=None,
        is_rms_norm=False,
    ),
    FwdCase(
        "rmsnorm",
        with_gate=False,
        norm_before_gate=True,
        group_size=None,
        is_rms_norm=True,
    ),
    FwdCase(
        "layernorm_gate_pre",
        with_gate=True,
        norm_before_gate=True,
        group_size=None,
        is_rms_norm=False,
    ),
    FwdCase(
        "layernorm_gate_post",
        with_gate=True,
        norm_before_gate=False,
        group_size=None,
        is_rms_norm=False,
    ),
    FwdCase(
        "rmsnorm_gate_pre",
        with_gate=True,
        norm_before_gate=True,
        group_size=None,
        is_rms_norm=True,
    ),
    FwdCase(
        "group_layernorm",
        with_gate=False,
        norm_before_gate=True,
        group_size=128,
        is_rms_norm=False,
    ),
    FwdCase(
        "group_rmsnorm",
        with_gate=False,
        norm_before_gate=True,
        group_size=128,
        is_rms_norm=True,
    ),
]


@pytest.mark.parametrize("num_tokens", [128])
@pytest.mark.parametrize("hidden_size", [256])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_layernorm_guard_fwd_spawn(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    case: FwdCase,
    device: str = "cuda",
):
    _skip_if_no_cuda_or_not_enough_gpus(NUM_GPUS)
    _skip_if_dtype_unsupported(dtype)

    if case.group_size is not None and hidden_size % case.group_size != 0:
        pytest.skip(
            f"hidden_size {hidden_size} not divisible by group_size {case.group_size}"
        )

    master_port = _find_free_port()
    world_size = NUM_GPUS

    torch.multiprocessing.spawn(
        _layernorm_guard_fwd_worker,
        args=(
            world_size,
            master_port,
            num_tokens,
            hidden_size,
            dtype,
            case,
            device,
        ),
        nprocs=world_size,
        join=True,
    )


def _layernorm_guard_fwd_worker(
    local_rank: int,
    world_size: int,
    master_port: int,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    case: FwdCase,
    device: str,
):
    device = _setup_sglang_distributed(local_rank, world_size, master_port, dtype)

    with torch.inference_mode():
        torch.manual_seed(42 + local_rank)
        torch.cuda.manual_seed_all(42 + local_rank)

        x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
        z = (
            torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
            if case.with_gate
            else None
        )
        weight = torch.randn(hidden_size, dtype=dtype, device=device)
        bias = (
            None
            if case.is_rms_norm
            else torch.randn(hidden_size, dtype=dtype, device=device)
        )
        eps = 1e-6

        out, mean, rstd = layer_norm_fwd(
            x,
            weight,
            bias,
            eps,
            z=z,
            group_size=case.group_size,
            norm_before_gate=case.norm_before_gate,
            is_rms_norm=case.is_rms_norm,
        )

        ref_out = layer_norm_ref(
            x,
            weight,
            bias,
            z=z,
            eps=eps,
            group_size=case.group_size,
            norm_before_gate=case.norm_before_gate,
            is_rms_norm=case.is_rms_norm,
        )

        assert out.shape == x.shape
        assert out.dtype == x.dtype
        torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)

        # mean/rstd shape checks (same spirit as original vLLM tests)
        if case.group_size is None:
            if not case.is_rms_norm:
                assert mean.shape == (num_tokens,)
            assert rstd.shape == (num_tokens,)
        else:
            ngroups = hidden_size // case.group_size
            if not case.is_rms_norm:
                assert mean.shape == (ngroups * num_tokens,)
            assert rstd.shape == (ngroups * num_tokens,)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_layernorm_guard_misc_spawn(dtype: torch.dtype, device: str = "cuda"):
    _skip_if_no_cuda_or_not_enough_gpus(NUM_GPUS)
    _skip_if_dtype_unsupported(dtype)

    master_port = _find_free_port()
    world_size = NUM_GPUS

    torch.multiprocessing.spawn(
        _layernorm_guard_misc_worker,
        args=(world_size, master_port, dtype, device),
        nprocs=world_size,
        join=True,
    )


def _layernorm_guard_misc_worker(
    local_rank: int,
    world_size: int,
    master_port: int,
    dtype: torch.dtype,
    device: str,
):
    device = _setup_sglang_distributed(local_rank, world_size, master_port, dtype)

    with torch.inference_mode():
        torch.manual_seed(123 + local_rank)
        torch.cuda.manual_seed_all(123 + local_rank)

        # 1) rows_per_block-like sizes
        hidden_size = 1024
        weight = torch.randn(hidden_size, dtype=dtype, device=device)
        bias = torch.randn(hidden_size, dtype=dtype, device=device)
        eps = 1e-6
        for num_tokens in [513]:
            x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
            out, _, _ = layer_norm_fwd(x, weight, bias, eps, z=None, is_rms_norm=False)
            ref = layer_norm_ref(x, weight, bias, z=None, eps=eps, is_rms_norm=False)
            torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

        # 2) strided input (slice then contiguous)
        num_tokens = 128
        x_large = torch.randn(num_tokens, hidden_size * 2, dtype=dtype, device=device)
        x = x_large[:, :hidden_size]
        x_contig = x.contiguous()
        out, _, _ = layer_norm_fwd(
            x_contig, weight, bias, eps, z=None, is_rms_norm=False
        )
        ref = layer_norm_ref(x_contig, weight, bias, z=None, eps=eps, is_rms_norm=False)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

        # 3) provided output buffer
        num_tokens = 256
        x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
        out_buf = torch.empty_like(x)
        out, _, _ = layer_norm_fwd(
            x, weight, bias, eps, z=None, out=out_buf, is_rms_norm=False
        )
        assert out.data_ptr() == out_buf.data_ptr()
        ref = layer_norm_ref(x, weight, bias, z=None, eps=eps, is_rms_norm=False)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

        # 4) multidimensional input via autograd fn
        for shape in [(4, 16, 1024)]:
            hs = shape[-1]
            x = torch.randn(*shape, dtype=dtype, device=device)
            w = torch.randn(hs, dtype=dtype, device=device)
            b = torch.randn(hs, dtype=dtype, device=device)
            out = layernorm_fn(x, w, b, z=None, eps=eps)
            ref = layer_norm_ref(x, w, b, z=None, eps=eps, is_rms_norm=False)
            torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
