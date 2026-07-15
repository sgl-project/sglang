from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.utils import (
    is_cpu,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
    is_xpu,
    next_power_of_2,
)
from sglang.srt.utils.common import is_pin_memory_available

_is_cpu = is_cpu()
_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_musa = is_musa()
_is_xpu = is_xpu()

if _is_cpu:
    from sgl_kernel import assign_extend_cache_locs_cpu, assign_req_to_token_pool_cpu

_VANILLA_NEEDS_MIRRORS = (
    "this gather has no kernel on this platform and falls back to a host loop, "
    "which needs req_pool_indices_cpu / start_offset_cpu / end_offset_cpu. On NPU "
    "this means a ragged gather: torch.ops.npu.cache_loc_update serves the "
    "equal-length entry only, so pass the host mirrors rather than route ragged "
    "ranges through it."
)


class WriteReqToTokenPool:

    @classmethod
    def execute(
        cls,
        req_to_token: torch.Tensor,
        *,
        req_pool_indices: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        alloc_starts: torch.Tensor,
        alloc_starts_cpu: torch.Tensor,
        alloc_ends: torch.Tensor,
        alloc_ends_cpu: torch.Tensor,
        prefix_tensors: list[torch.Tensor],
        out_cache_loc: torch.Tensor,
        use_triton: bool,
    ) -> None:
        implementation = cls.triton if use_triton else cls.vanilla
        implementation(
            req_to_token,
            req_pool_indices=req_pool_indices,
            req_pool_indices_cpu=req_pool_indices_cpu,
            prefix_lens=prefix_lens,
            prefix_lens_cpu=prefix_lens_cpu,
            alloc_starts=alloc_starts,
            alloc_starts_cpu=alloc_starts_cpu,
            alloc_ends=alloc_ends,
            alloc_ends_cpu=alloc_ends_cpu,
            prefix_tensors=prefix_tensors,
            out_cache_loc=out_cache_loc,
        )

    @classmethod
    def vanilla(
        cls,
        req_to_token: torch.Tensor,
        *,
        req_pool_indices: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        alloc_starts: torch.Tensor,
        alloc_starts_cpu: torch.Tensor,
        alloc_ends: torch.Tensor,
        alloc_ends_cpu: torch.Tensor,
        prefix_tensors: list[torch.Tensor],
        out_cache_loc: torch.Tensor,
    ) -> None:
        out_cache_offset = 0
        for index in range(req_pool_indices_cpu.shape[0]):
            req_pool_index = int(req_pool_indices_cpu[index].item())
            prefix_len = int(prefix_lens_cpu[index].item())
            alloc_start = int(alloc_starts_cpu[index].item())
            alloc_end = int(alloc_ends_cpu[index].item())
            alloc_len = alloc_end - alloc_start
            req_to_token[req_pool_index, :prefix_len] = prefix_tensors[index]
            req_to_token[req_pool_index, alloc_start:alloc_end] = out_cache_loc[
                out_cache_offset : out_cache_offset + alloc_len
            ]
            out_cache_offset += alloc_len

    @classmethod
    def triton(
        cls,
        req_to_token: torch.Tensor,
        *,
        req_pool_indices: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        alloc_starts: torch.Tensor,
        alloc_starts_cpu: torch.Tensor,
        alloc_ends: torch.Tensor,
        alloc_ends_cpu: torch.Tensor,
        prefix_tensors: list[torch.Tensor],
        out_cache_loc: torch.Tensor,
    ) -> None:
        prefix_pointers = torch.tensor(
            [tensor.data_ptr() for tensor in prefix_tensors],
            dtype=torch.uint64,
            pin_memory=is_pin_memory_available(req_to_token.device),
        ).to(req_to_token.device, non_blocking=True)
        _write_req_to_token_pool_kernel[(req_pool_indices.shape[0],)](
            req_to_token,
            req_pool_indices,
            prefix_pointers,
            prefix_lens,
            alloc_starts,
            alloc_ends,
            out_cache_loc,
            req_to_token.stride(0),
        )

    @staticmethod
    def _validate_inputs(
        req_to_token: torch.Tensor,
        *,
        req_pool_indices: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        alloc_starts: torch.Tensor,
        alloc_starts_cpu: torch.Tensor,
        alloc_ends: torch.Tensor,
        alloc_ends_cpu: torch.Tensor,
        prefix_tensors: list[torch.Tensor],
        out_cache_loc: torch.Tensor,
    ) -> int:
        assert req_to_token.ndim == 2, f"{req_to_token.shape=}"
        assert req_to_token.dtype == torch.int32, f"{req_to_token.dtype=}"
        assert req_to_token.is_contiguous(), f"{req_to_token.stride()=}"
        assert req_to_token.stride(1) == 1, f"{req_to_token.stride()=}"
        assert (
            req_to_token.shape[0] > 0 and req_to_token.shape[1] > 0
        ), f"{req_to_token.shape=}"

        assert req_pool_indices.ndim == 1, f"{req_pool_indices.shape=}"
        num_reqs = req_pool_indices.shape[0]
        device_tensors = (
            req_pool_indices,
            prefix_lens,
            alloc_starts,
            alloc_ends,
        )
        cpu_tensors = (
            req_pool_indices_cpu,
            prefix_lens_cpu,
            alloc_starts_cpu,
            alloc_ends_cpu,
        )
        input_tensors = device_tensors + cpu_tensors
        assert all(
            tensor.ndim == 1 for tensor in input_tensors
        ), f"shapes={[tensor.shape for tensor in input_tensors]}"
        assert all(
            tensor.shape == (num_reqs,) for tensor in input_tensors
        ), f"{num_reqs=}, shapes={[tensor.shape for tensor in input_tensors]}"
        assert all(
            tensor.dtype == torch.int64 for tensor in input_tensors
        ), f"dtypes={[tensor.dtype for tensor in input_tensors]}"
        assert all(
            tensor.is_contiguous() for tensor in input_tensors
        ), f"strides={[tensor.stride() for tensor in input_tensors]}"
        assert all(
            tensor.device == req_to_token.device for tensor in device_tensors
        ), f"{req_to_token.device=}, devices={[tensor.device for tensor in device_tensors]}"
        assert all(
            tensor.device.type == "cpu" for tensor in cpu_tensors
        ), f"devices={[tensor.device for tensor in cpu_tensors]}"

        assert len(prefix_tensors) == num_reqs, f"{len(prefix_tensors)=}, {num_reqs=}"
        assert all(
            tensor.ndim == 1 for tensor in prefix_tensors
        ), f"shapes={[tensor.shape for tensor in prefix_tensors]}"
        assert all(
            tensor.dtype == torch.int64 for tensor in prefix_tensors
        ), f"dtypes={[tensor.dtype for tensor in prefix_tensors]}"
        assert all(
            tensor.is_contiguous() for tensor in prefix_tensors
        ), f"strides={[tensor.stride() for tensor in prefix_tensors]}"
        assert all(
            tensor.device == req_to_token.device or tensor.numel() == 0
            for tensor in prefix_tensors
        ), f"{req_to_token.device=}, devices={[tensor.device for tensor in prefix_tensors]}"

        assert out_cache_loc.ndim == 1, f"{out_cache_loc.shape=}"
        assert out_cache_loc.dtype in (
            torch.int32,
            torch.int64,
        ), f"{out_cache_loc.dtype=}"
        assert out_cache_loc.is_contiguous(), f"{out_cache_loc.stride()=}"
        assert (
            out_cache_loc.device == req_to_token.device
        ), f"{out_cache_loc.device=}, {req_to_token.device=}"

        assert bool(torch.all(req_pool_indices_cpu >= 0)), f"{req_pool_indices_cpu=}"
        assert bool(
            torch.all(req_pool_indices_cpu < req_to_token.shape[0])
        ), f"{req_pool_indices_cpu=}, rows={req_to_token.shape[0]}"
        assert bool(torch.all(prefix_lens_cpu >= 0)), f"{prefix_lens_cpu=}"
        assert bool(
            torch.all(alloc_starts_cpu >= prefix_lens_cpu)
        ), f"{prefix_lens_cpu=}, {alloc_starts_cpu=}"
        assert bool(
            torch.all(alloc_ends_cpu >= alloc_starts_cpu)
        ), f"{alloc_starts_cpu=}, {alloc_ends_cpu=}"
        assert bool(
            torch.all(alloc_ends_cpu <= req_to_token.shape[1])
        ), f"{alloc_ends_cpu=}, row_width={req_to_token.shape[1]}"
        assert all(
            tensor.numel() == int(prefix_len)
            for tensor, prefix_len in zip(prefix_tensors, prefix_lens_cpu.tolist())
        ), f"prefix_shapes={[tensor.shape for tensor in prefix_tensors]}, {prefix_lens_cpu=}"
        assert (
            int((alloc_ends_cpu - alloc_starts_cpu).sum().item())
            == out_cache_loc.numel()
        ), (
            f"alloc_sum={int((alloc_ends_cpu - alloc_starts_cpu).sum().item())}, "
            f"out_cache_tokens={out_cache_loc.numel()}"
        )
        return num_reqs


class AssignExtendCacheLocs:

    @classmethod
    def execute_equal_length(
        cls,
        req_to_token: torch.Tensor,
        *,
        req_pool_indices: torch.Tensor,
        start_offset: torch.Tensor,
        batch_size: int,
        draft_token_num: int,
        device: torch.device,
        req_pool_indices_cpu: Optional[torch.Tensor] = None,
        start_offset_cpu: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Gather ``draft_token_num`` slots per request, starting at ``start_offset``.

        The end of each range is derived here rather than taken from the caller,
        so every range is the same length by construction. Backends that can only
        be trusted with equal-length ranges are reachable from this entry alone.
        """
        end_offset_cpu = (
            None if start_offset_cpu is None else start_offset_cpu + draft_token_num
        )
        return cls._execute(
            req_to_token,
            req_pool_indices=req_pool_indices,
            req_pool_indices_cpu=req_pool_indices_cpu,
            start_offset=start_offset,
            start_offset_cpu=start_offset_cpu,
            end_offset=start_offset + draft_token_num,
            end_offset_cpu=end_offset_cpu,
            batch_size=batch_size,
            out_tokens=batch_size * draft_token_num,
            device=device,
            ranges_are_equal_length=True,
        )

    @classmethod
    def execute_ragged(
        cls,
        req_to_token: torch.Tensor,
        *,
        req_pool_indices: torch.Tensor,
        start_offset: torch.Tensor,
        end_offset: torch.Tensor,
        batch_size: int,
        out_tokens: int,
        device: torch.device,
        req_pool_indices_cpu: Optional[torch.Tensor] = None,
        start_offset_cpu: Optional[torch.Tensor] = None,
        end_offset_cpu: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Gather a separately sized range per request into an ``out_tokens`` buffer.

        ``out_tokens`` is the output length, which callers that need a static
        shape may size above the total the ranges actually cover; the slots past
        that total are left untouched.
        """
        return cls._execute(
            req_to_token,
            req_pool_indices=req_pool_indices,
            req_pool_indices_cpu=req_pool_indices_cpu,
            start_offset=start_offset,
            start_offset_cpu=start_offset_cpu,
            end_offset=end_offset,
            end_offset_cpu=end_offset_cpu,
            batch_size=batch_size,
            out_tokens=out_tokens,
            device=device,
            ranges_are_equal_length=False,
        )

    @classmethod
    def _execute(
        cls,
        req_to_token: torch.Tensor,
        *,
        req_pool_indices: torch.Tensor,
        req_pool_indices_cpu: Optional[torch.Tensor],
        start_offset: torch.Tensor,
        start_offset_cpu: Optional[torch.Tensor],
        end_offset: torch.Tensor,
        end_offset_cpu: Optional[torch.Tensor],
        batch_size: int,
        out_tokens: int,
        device: torch.device,
        ranges_are_equal_length: bool,
    ) -> torch.Tensor:
        if start_offset_cpu is not None and end_offset_cpu is not None:
            covered = int((end_offset_cpu - start_offset_cpu).sum())
            assert covered <= out_tokens, (
                f"gather ranges cover {covered} slots but the output holds only "
                f"{out_tokens}"
            )

        if _is_cuda or _is_hip or _is_musa or _is_xpu:
            out_cache_loc = torch.empty(
                (out_tokens,),
                dtype=torch.int64,
                device=device,
            )
            cls.triton(
                req_to_token,
                req_pool_indices=req_pool_indices,
                start_offset=start_offset,
                end_offset=end_offset,
                out_cache_loc=out_cache_loc,
                batch_size=batch_size,
            )

            return out_cache_loc

        elif _is_npu and ranges_are_equal_length:
            out_cache_loc = torch.empty(
                (out_tokens,),
                dtype=torch.int32,
                device=device,
            )
            cls.npu(
                req_to_token,
                req_pool_indices=req_pool_indices,
                start_offset=start_offset,
                end_offset=end_offset,
                out_cache_loc=out_cache_loc,
            )

            return out_cache_loc

        elif _is_cpu:
            out_cache_loc = torch.empty(
                (out_tokens,),
                dtype=torch.int64,
                device=device,
            )
            cls.cpu(
                req_to_token,
                req_pool_indices=req_pool_indices,
                start_offset=start_offset,
                end_offset=end_offset,
                out_cache_loc=out_cache_loc,
            )

            return out_cache_loc

        assert req_pool_indices_cpu is not None, _VANILLA_NEEDS_MIRRORS
        assert start_offset_cpu is not None, _VANILLA_NEEDS_MIRRORS
        assert end_offset_cpu is not None, _VANILLA_NEEDS_MIRRORS

        out_cache_loc = torch.empty(
            (out_tokens,),
            dtype=torch.int32,
            device=device,
        )
        cls.vanilla(
            req_to_token,
            req_pool_indices_cpu=req_pool_indices_cpu,
            start_offset_cpu=start_offset_cpu,
            end_offset_cpu=end_offset_cpu,
            out_cache_loc=out_cache_loc,
        )

        return out_cache_loc.to(torch.int64)

    @staticmethod
    def vanilla(
        req_to_token: torch.Tensor,
        *,
        req_pool_indices_cpu: torch.Tensor,
        start_offset_cpu: torch.Tensor,
        end_offset_cpu: torch.Tensor,
        out_cache_loc: torch.Tensor,
    ) -> None:
        out_cache_offset = 0
        for index in range(req_pool_indices_cpu.shape[0]):
            req_pool_index = int(req_pool_indices_cpu[index].item())
            start_index = int(start_offset_cpu[index].item())
            end_index = int(end_offset_cpu[index].item())
            gather_len = end_index - start_index
            out_cache_loc[out_cache_offset : out_cache_offset + gather_len] = (
                req_to_token[req_pool_index, start_index:end_index]
            )
            out_cache_offset += gather_len

    @staticmethod
    def triton(
        req_to_token: torch.Tensor,
        *,
        req_pool_indices: torch.Tensor,
        start_offset: torch.Tensor,
        end_offset: torch.Tensor,
        out_cache_loc: torch.Tensor,
        batch_size: int,
    ) -> None:
        _assign_extend_cache_locs_kernel[(batch_size,)](
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
            req_to_token.shape[1],
            next_power_of_2(batch_size),
        )

    @staticmethod
    def npu(
        req_to_token: torch.Tensor,
        *,
        req_pool_indices: torch.Tensor,
        start_offset: torch.Tensor,
        end_offset: torch.Tensor,
        out_cache_loc: torch.Tensor,
    ) -> None:
        torch.ops.npu.cache_loc_update(
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
        )

    @staticmethod
    def cpu(
        req_to_token: torch.Tensor,
        *,
        req_pool_indices: torch.Tensor,
        start_offset: torch.Tensor,
        end_offset: torch.Tensor,
        out_cache_loc: torch.Tensor,
    ) -> None:
        assign_extend_cache_locs_cpu(
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
            req_to_token.shape[1],
        )


class AssignReqToTokenPool:

    @classmethod
    def execute(
        cls,
        req_to_token: torch.Tensor,
        *,
        req_pool_indices: torch.Tensor,
        start_offset: torch.Tensor,
        end_offset: torch.Tensor,
        out_cache_loc: torch.Tensor,
        batch_size: int,
    ) -> None:
        if _is_cpu:
            cls.cpu(
                req_to_token,
                req_pool_indices=req_pool_indices,
                start_offset=start_offset,
                end_offset=end_offset,
                out_cache_loc=out_cache_loc,
            )
            return
        cls.triton(
            req_to_token,
            req_pool_indices=req_pool_indices,
            start_offset=start_offset,
            end_offset=end_offset,
            out_cache_loc=out_cache_loc,
            batch_size=batch_size,
        )

    @staticmethod
    def triton(
        req_to_token: torch.Tensor,
        *,
        req_pool_indices: torch.Tensor,
        start_offset: torch.Tensor,
        end_offset: torch.Tensor,
        out_cache_loc: torch.Tensor,
        batch_size: int,
    ) -> None:
        _assign_req_to_token_pool_kernel[(batch_size,)](
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
            req_to_token.shape[1],
            next_power_of_2(batch_size),
        )

    @staticmethod
    def cpu(
        req_to_token: torch.Tensor,
        *,
        req_pool_indices: torch.Tensor,
        start_offset: torch.Tensor,
        end_offset: torch.Tensor,
        out_cache_loc: torch.Tensor,
    ) -> None:
        assign_req_to_token_pool_cpu(
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
            req_to_token.shape[1],
        )


@triton.jit
def _write_req_to_token_pool_kernel(
    req_to_token_ptr,
    req_pool_indices,
    prefix_pointers,
    prefix_lens,
    alloc_starts,
    alloc_ends,
    out_cache_loc,
    req_to_token_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    request_index = tl.program_id(0)

    req_pool_index = tl.load(req_pool_indices + request_index)
    prefix_len = tl.load(prefix_lens + request_index)
    alloc_start = tl.load(alloc_starts + request_index)
    alloc_end = tl.load(alloc_ends + request_index)
    prefix_pointer = tl.load(prefix_pointers + request_index).to(
        tl.pointer_type(tl.int64)
    )

    prefix_num_blocks = tl.cdiv(prefix_len, BLOCK_SIZE)
    for block_index in range(prefix_num_blocks):
        offset = tl.arange(0, BLOCK_SIZE) + block_index * BLOCK_SIZE
        mask = offset < prefix_len
        value = tl.load(prefix_pointer + offset, mask=mask)
        tl.store(
            req_to_token_ptr + req_pool_index * req_to_token_stride + offset,
            value,
            mask=mask,
        )

    output_start = tl.cast(0, tl.int64)
    for index in range(request_index):
        output_start += tl.load(alloc_ends + index) - tl.load(alloc_starts + index)

    alloc_len = alloc_end - alloc_start
    alloc_num_blocks = tl.cdiv(alloc_len, BLOCK_SIZE)
    for block_index in range(alloc_num_blocks):
        offset = tl.arange(0, BLOCK_SIZE) + block_index * BLOCK_SIZE
        mask = offset < alloc_len
        value = tl.load(out_cache_loc + output_start + offset, mask=mask)
        tl.store(
            req_to_token_ptr
            + req_pool_index * req_to_token_stride
            + alloc_start
            + offset,
            value,
            mask=mask,
        )


@triton.jit
def _assign_extend_cache_locs_kernel(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid, other=0)
    end = tl.load(end_offset + length_offset, mask=length_offset < pid, other=0)
    out_offset = tl.sum(end - start, axis=0)

    out_cache_ptr = out_cache_loc + out_offset

    load_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    save_offset = tl.arange(0, BLOCK_SIZE)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = load_offset < kv_end
        data = tl.load(token_pool + load_offset, mask=mask)
        tl.store(out_cache_ptr + save_offset, data, mask=mask)
        load_offset += BLOCK_SIZE
        save_offset += BLOCK_SIZE


@triton.jit
def _assign_req_to_token_pool_kernel(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid, other=0)
    end = tl.load(end_offset + length_offset, mask=length_offset < pid, other=0)
    out_offset = tl.sum(end - start, axis=0)

    out_cache_ptr = out_cache_loc + out_offset

    save_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    load_offset = tl.arange(0, BLOCK_SIZE)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = save_offset < kv_end
        data = tl.load(out_cache_ptr + load_offset, mask=mask)
        tl.store(token_pool + save_offset, data, mask=mask)
        save_offset += BLOCK_SIZE
        load_offset += BLOCK_SIZE
