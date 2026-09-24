"""TP4 tiny-row all-reduce/post using AITER's registered peer buffers."""

import hashlib
from functools import lru_cache
from pathlib import Path

import torch
from aiter.jit.core import AITER_CSRC_DIR, compile_ops, get_args_of_build


@lru_cache(maxsize=1)
def _build_args():
    source = (
        Path(__file__).resolve().parents[2]
        / "jit/csrc/distributed/all_reduce_mhc_hip.cu"
    )
    header = Path(AITER_CSRC_DIR) / "include/custom_all_reduce.cuh"
    revision = hashlib.sha256(source.read_bytes() + header.read_bytes()).hexdigest()[
        :16
    ]
    args = get_args_of_build("module_fused_ar_mhc")
    args.update(
        md_name=f"module_sglang_dsv41_ar_mhc_{revision}",
        srcs=[str(source)],
        # The following native HIP boundary expects separate FP32 multiplies/adds.
        flags_extra_hip=[*args["flags_extra_hip"], "-ffp-contract=off"],
    )
    return args


@compile_ops(
    "module_fused_ar_mhc",
    fc_name="run",
    gen_func=lambda *args, **kwargs: _build_args(),
    develop=True,
)
def _dsv41_all_reduce_mhc_post(
    ptr: int,
    input: torch.Tensor,
    output: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    registered_ptr: int,
    registered_bytes: int,
) -> None: ...


def all_reduce_mhc_post(
    input: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    communicator,
) -> torch.Tensor:
    """TP4 all-reduce of input fused with hc_post onto residual; the kernel checks the
    shapes (up to 8 rows of DeepSeek-V4.1's hidden size)."""
    assert communicator.world_size == 4 and not communicator.disabled
    capturing = torch.cuda.is_current_stream_capturing()
    # capture() registers the peer addresses when the enclosing graph scope exits.
    assert not capturing or (
        communicator._IS_CAPTURING and communicator.enable_register_for_capturing
    )
    pool = communicator._pool["input"]
    output = torch.empty_like(residual)
    _dsv41_all_reduce_mhc_post(
        communicator._ptr,
        input,
        output,
        residual,
        post,
        comb,
        0 if capturing else pool.data_ptr,
        0 if capturing else pool.max_size,
    )
    return output
