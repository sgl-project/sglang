from __future__ import annotations

import torch
import triton
import triton.language as tl


class GatherReqToTokenPool:

    @classmethod
    def execute(
        cls,
        req_to_token: torch.Tensor,
        *,
        req_pool_indices: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        extend_lens: torch.Tensor,
        extend_lens_cpu: torch.Tensor,
        extend_num_tokens: int,
        out_dtype: torch.dtype,
        use_triton: bool,
    ) -> torch.Tensor:
        implementation = cls.triton if use_triton else cls.vanilla
        return implementation(
            req_to_token,
            req_pool_indices=req_pool_indices,
            req_pool_indices_cpu=req_pool_indices_cpu,
            prefix_lens=prefix_lens,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            extend_lens=extend_lens,
            extend_lens_cpu=extend_lens_cpu,
            extend_num_tokens=extend_num_tokens,
            out_dtype=out_dtype,
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
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        extend_lens: torch.Tensor,
        extend_lens_cpu: torch.Tensor,
        extend_num_tokens: int,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        num_reqs = cls._validate_inputs(
            req_to_token,
            req_pool_indices=req_pool_indices,
            req_pool_indices_cpu=req_pool_indices_cpu,
            prefix_lens=prefix_lens,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            extend_lens=extend_lens,
            extend_lens_cpu=extend_lens_cpu,
            extend_num_tokens=extend_num_tokens,
            out_dtype=out_dtype,
        )
        if num_reqs == 0 or extend_num_tokens == 0:
            return torch.empty((0,), dtype=out_dtype, device=req_to_token.device)

        chunks: list[torch.Tensor] = []
        for index in range(num_reqs):
            req_pool_index = int(req_pool_indices_cpu[index].item())
            prefix_len = int(prefix_lens_cpu[index].item())
            seq_len = int(seq_lens_cpu[index].item())
            chunks.append(req_to_token[req_pool_index, prefix_len:seq_len])
        return torch.cat(chunks).to(out_dtype)

    @classmethod
    def triton(
        cls,
        req_to_token: torch.Tensor,
        *,
        req_pool_indices: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        extend_lens: torch.Tensor,
        extend_lens_cpu: torch.Tensor,
        extend_num_tokens: int,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        num_reqs = cls._validate_inputs(
            req_to_token,
            req_pool_indices=req_pool_indices,
            req_pool_indices_cpu=req_pool_indices_cpu,
            prefix_lens=prefix_lens,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            extend_lens=extend_lens,
            extend_lens_cpu=extend_lens_cpu,
            extend_num_tokens=extend_num_tokens,
            out_dtype=out_dtype,
        )
        if num_reqs == 0 or extend_num_tokens == 0:
            return torch.empty((0,), dtype=out_dtype, device=req_to_token.device)

        supported_accelerator_types = ("cuda", "npu", "xpu", "musa")
        assert req_to_token.device.type in supported_accelerator_types, (
            f"{req_to_token.device=}"
        )

        output = torch.empty(
            (extend_num_tokens,), dtype=torch.int32, device=req_to_token.device
        )
        _gather_req_to_token_pool_kernel[(num_reqs,)](
            req_to_token,
            req_pool_indices,
            prefix_lens,
            seq_lens,
            extend_lens,
            output,
            req_to_token.shape[1],
        )
        return output.to(out_dtype)

    @staticmethod
    def _validate_inputs(
        req_to_token: torch.Tensor,
        *,
        req_pool_indices: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        extend_lens: torch.Tensor,
        extend_lens_cpu: torch.Tensor,
        extend_num_tokens: int,
        out_dtype: torch.dtype,
    ) -> int:
        assert req_to_token.ndim == 2, f"{req_to_token.shape=}"
        assert req_to_token.dtype == torch.int32, f"{req_to_token.dtype=}"
        assert req_to_token.is_contiguous(), f"{req_to_token.stride()=}"
        assert req_to_token.stride(1) == 1, f"{req_to_token.stride()=}"
        assert req_to_token.shape[0] > 0 and req_to_token.shape[1] > 0, (
            f"{req_to_token.shape=}"
        )

        num_reqs = req_pool_indices.shape[0]
        device_tensors = (
            req_pool_indices,
            prefix_lens,
            seq_lens,
            extend_lens,
        )
        cpu_tensors = (
            req_pool_indices_cpu,
            prefix_lens_cpu,
            seq_lens_cpu,
            extend_lens_cpu,
        )
        input_tensors = device_tensors + cpu_tensors
        assert all(tensor.ndim == 1 for tensor in input_tensors), (
            f"shapes={[tensor.shape for tensor in input_tensors]}"
        )
        assert all(tensor.shape == (num_reqs,) for tensor in input_tensors), (
            f"{num_reqs=}, shapes={[tensor.shape for tensor in input_tensors]}"
        )
        assert all(tensor.dtype == torch.int64 for tensor in input_tensors), (
            f"dtypes={[tensor.dtype for tensor in input_tensors]}"
        )
        assert all(tensor.is_contiguous() for tensor in input_tensors), (
            f"strides={[tensor.stride() for tensor in input_tensors]}"
        )
        assert all(tensor.device == req_to_token.device for tensor in device_tensors), (
            f"{req_to_token.device=}, devices={[tensor.device for tensor in device_tensors]}"
        )
        assert all(tensor.device.type == "cpu" for tensor in cpu_tensors), (
            f"devices={[tensor.device for tensor in cpu_tensors]}"
        )

        assert type(extend_num_tokens) is int, f"{type(extend_num_tokens)=}"
        assert extend_num_tokens >= 0, f"{extend_num_tokens=}"
        assert out_dtype in (torch.int32, torch.int64), f"{out_dtype=}"
        assert bool(torch.all(req_pool_indices_cpu >= 0)), (
            f"{req_pool_indices_cpu=}"
        )
        assert bool(torch.all(req_pool_indices_cpu < req_to_token.shape[0])), (
            f"{req_pool_indices_cpu=}, rows={req_to_token.shape[0]}"
        )
        assert bool(torch.all(prefix_lens_cpu >= 0)), f"{prefix_lens_cpu=}"
        assert bool(torch.all(seq_lens_cpu >= prefix_lens_cpu)), (
            f"{prefix_lens_cpu=}, {seq_lens_cpu=}"
        )
        assert bool(torch.all(seq_lens_cpu <= req_to_token.shape[1])), (
            f"{seq_lens_cpu=}, row_width={req_to_token.shape[1]}"
        )
        assert bool(torch.all(extend_lens_cpu >= 0)), f"{extend_lens_cpu=}"
        assert torch.equal(seq_lens_cpu - prefix_lens_cpu, extend_lens_cpu), (
            f"{prefix_lens_cpu=}, {seq_lens_cpu=}, {extend_lens_cpu=}"
        )
        assert int(extend_lens_cpu.sum().item()) == extend_num_tokens, (
            f"extend_sum={int(extend_lens_cpu.sum().item())}, {extend_num_tokens=}"
        )
        assert num_reqs > 0 or extend_num_tokens == 0, (
            f"{num_reqs=}, {extend_num_tokens=}"
        )
        return num_reqs


@triton.jit
def _gather_req_to_token_pool_kernel(
    req_to_token_ptr,
    req_pool_indices,
    prefix_lens,
    seq_lens,
    extend_lens,
    output,
    req_to_token_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    request_index = tl.program_id(0)

    req_pool_index = tl.load(req_pool_indices + request_index)
    prefix_len = tl.load(prefix_lens + request_index)
    seq_len = tl.load(seq_lens + request_index)

    output_start = tl.cast(0, tl.int64)
    for index in range(request_index):
        output_start += tl.load(extend_lens + index)

    num_blocks = tl.cdiv(seq_len - prefix_len, BLOCK_SIZE)
    for block_index in range(num_blocks):
        offset = tl.arange(0, BLOCK_SIZE) + block_index * BLOCK_SIZE
        mask = offset < seq_len - prefix_len
        value = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_stride
            + prefix_len
            + offset,
            mask=mask,
        )
        tl.store(output + output_start + offset, value, mask=mask)
