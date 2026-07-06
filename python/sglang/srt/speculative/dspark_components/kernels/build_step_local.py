from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.srt.environ import envs

_KERNEL_IMPL = envs.SGLANG_DSPARK_KERNEL_BUILD_STEP_LOCAL.get()

_BLOCK = 1024


class BuildStepLocal:
    @classmethod
    def execute(cls, *args, **kwargs) -> torch.Tensor:
        if _KERNEL_IMPL == "torch":
            return cls.torch(*args, **kwargs)
        return cls.triton(*args, **kwargs)

    @classmethod
    def torch(cls, *, bias: torch.Tensor, base_local: torch.Tensor) -> torch.Tensor:
        return build_step_local(bias=bias, base_local=base_local)

    @classmethod
    def triton(cls, *, bias: torch.Tensor, base_local: torch.Tensor) -> torch.Tensor:
        return build_step_local_triton(bias=bias, base_local=base_local)


def build_step_local(*, bias: torch.Tensor, base_local: torch.Tensor) -> torch.Tensor:
    per_partition = base_local.shape[-1]
    pad = per_partition - bias.shape[-1]
    padded = (
        F.pad(bias.to(torch.float32), (0, pad)) if pad > 0 else bias.to(torch.float32)
    )
    return base_local + padded


@triton.jit
def _build_step_local_kernel(
    bias_ptr,
    base_ptr,
    out_ptr,
    org_width,
    per_partition,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK + tl.arange(0, BLOCK)
    mask = offs < per_partition
    base = tl.load(base_ptr + row * per_partition + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    bias = tl.load(
        bias_ptr + row * org_width + offs, mask=offs < org_width, other=0.0
    ).to(tl.float32)
    tl.store(out_ptr + row * per_partition + offs, base + bias, mask=mask)


def build_step_local_triton(
    *, bias: torch.Tensor, base_local: torch.Tensor
) -> torch.Tensor:
    bs, per_partition = base_local.shape
    org_width = bias.shape[-1]
    base_local = base_local.contiguous()
    bias = bias.contiguous()
    out = torch.empty(
        (bs, per_partition), dtype=torch.float32, device=base_local.device
    )
    grid = (bs, triton.cdiv(per_partition, _BLOCK))
    _build_step_local_kernel[grid](
        bias, base_local, out, org_width, per_partition, BLOCK=_BLOCK
    )
    return out
