from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.speculative.dspark_components.kernels.dispatch import (
    inputs_on_cuda,
)

try:
    from flashinfer.sampling import softmax as _flashinfer_softmax
except ImportError:
    _flashinfer_softmax = None


class SoftmaxTemp:
    @classmethod
    def execute(cls, *args, **kwargs) -> torch.Tensor:
        if not inputs_on_cuda(*args, **kwargs):
            return cls.torch(*args, **kwargs)
        if _flashinfer_softmax is not None:
            return cls.flashinfer(*args, **kwargs)
        return cls.triton(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        rows_per_request: int,
    ) -> torch.Tensor:
        return softmax_temp(
            logits=logits,
            temperatures=temperatures,
            rows_per_request=rows_per_request,
        )

    @classmethod
    def triton(
        cls,
        *,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        rows_per_request: int,
    ) -> torch.Tensor:
        return softmax_temp_triton(
            logits=logits,
            temperatures=temperatures,
            rows_per_request=rows_per_request,
        )

    @classmethod
    def flashinfer(
        cls,
        *,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        rows_per_request: int,
    ) -> torch.Tensor:
        return softmax_temp_flashinfer(
            logits=logits,
            temperatures=temperatures,
            rows_per_request=rows_per_request,
        )


def softmax_temp(
    *,
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    rows_per_request: int,
) -> torch.Tensor:
    num_rows = logits.shape[0]
    bs = num_rows // rows_per_request
    assert (
        bs * rows_per_request == num_rows
    ), f"num_rows {num_rows} not divisible by rows_per_request {rows_per_request}"
    temp_per_row = torch.repeat_interleave(
        temperatures.reshape(bs).to(torch.float32), rows_per_request, dim=0
    )
    scaled = logits.to(torch.float32) / temp_per_row[:, None]
    return torch.softmax(scaled, dim=-1)


@triton.jit
def _softmax_temp_kernel(
    logits_ptr,
    temp_ptr,
    out_ptr,
    vocab,
    rows_per_request,
    logits_row_stride,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    temp = tl.load(temp_ptr + row // rows_per_request).to(tl.float32)
    base = logits_ptr + row.to(tl.int64) * logits_row_stride
    out_base = out_ptr + row.to(tl.int64) * vocab

    row_max = -float("inf")
    for v0 in range(0, vocab, BLOCK_V):
        offs = v0 + tl.arange(0, BLOCK_V)
        vmask = offs < vocab
        x = tl.load(base + offs, mask=vmask, other=-float("inf")).to(tl.float32)
        x = x / temp
        row_max = tl.maximum(row_max, tl.max(x, axis=0))

    sum_exp = 0.0
    for v0 in range(0, vocab, BLOCK_V):
        offs = v0 + tl.arange(0, BLOCK_V)
        vmask = offs < vocab
        x = tl.load(base + offs, mask=vmask, other=-float("inf")).to(tl.float32)
        x = x / temp
        e = tl.exp(x - row_max)
        e = tl.where(vmask, e, 0.0)
        sum_exp += tl.sum(e, axis=0)

    for v0 in range(0, vocab, BLOCK_V):
        offs = v0 + tl.arange(0, BLOCK_V)
        vmask = offs < vocab
        x = tl.load(base + offs, mask=vmask, other=-float("inf")).to(tl.float32)
        x = x / temp
        e = tl.exp(x - row_max)
        tl.store(out_base + offs, e / sum_exp, mask=vmask)


def softmax_temp_triton(
    *,
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    rows_per_request: int,
) -> torch.Tensor:
    num_rows, vocab = logits.shape[0], logits.shape[-1]
    bs = num_rows // rows_per_request
    assert (
        bs * rows_per_request == num_rows
    ), f"num_rows {num_rows} not divisible by rows_per_request {rows_per_request}"
    temperatures = temperatures.reshape(bs).to(torch.float32).contiguous()
    out = torch.empty((num_rows, vocab), dtype=torch.float32, device=logits.device)
    BLOCK_V = 4096
    _softmax_temp_kernel[(num_rows,)](
        logits,
        temperatures,
        out,
        vocab,
        rows_per_request,
        logits.stride(0),
        BLOCK_V=BLOCK_V,
    )
    return out


def softmax_temp_flashinfer(
    *,
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    rows_per_request: int,
) -> torch.Tensor:
    if _flashinfer_softmax is None:
        raise RuntimeError(
            "softmax_temp_flashinfer requires flashinfer.sampling.softmax, "
            "which is unavailable in this environment"
        )
    num_rows, vocab = logits.shape[0], logits.shape[-1]
    bs = num_rows // rows_per_request
    assert (
        bs * rows_per_request == num_rows
    ), f"num_rows {num_rows} not divisible by rows_per_request {rows_per_request}"
    temp_per_row = torch.repeat_interleave(
        temperatures.reshape(bs).to(torch.float32), rows_per_request, dim=0
    ).contiguous()
    logits_2d = logits.to(torch.float32).contiguous()
    return _flashinfer_softmax(logits=logits_2d, temperature=temp_per_row)
