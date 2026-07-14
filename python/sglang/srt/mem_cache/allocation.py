from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
    maybe_write_dsv4_decode,
    maybe_write_dsv4_extend,
)
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, EvictParams
from sglang.srt.mem_cache.common import (
    MAMBA_STATE_PER_REQ_NO_CACHE,
    MAMBA_STATE_PER_REQ_PREFIX_CACHE,
    MAMBA_STATE_PER_REQ_PREFIX_CACHE_LAZY,
    available_and_evictable_str,
    evict_from_tree_cache,
)
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.mem_cache.triton_ops.common import (
    gather_req_to_token_pool_triton,
    get_last_loc_triton,
    get_last_loc_triton_safe,
    write_req_to_token_pool_triton,
)
from sglang.srt.runtime_context import get_server_args
from sglang.srt.utils import is_cuda, is_hip, is_npu, next_power_of_2, support_triton
from sglang.srt.utils.common import ceil_align, is_pin_memory_available

_is_hip = is_hip()
_is_npu = is_npu()
_is_cuda = is_cuda()

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import DSV4StateLens

logger = logging.getLogger(__name__)


def write_cache_indices(
    out_cache_loc: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    req_pool_indices_cpu: torch.Tensor,
    prefix_write_lens_tensor: torch.Tensor,
    prefix_write_lens_cpu: torch.Tensor,
    alloc_start_lens_tensor: torch.Tensor,
    alloc_start_lens_cpu: torch.Tensor,
    alloc_end_lens_tensor: torch.Tensor,
    alloc_end_lens_cpu: torch.Tensor,
    alloc_extend_lens_tensor: torch.Tensor,
    alloc_extend_lens_cpu: torch.Tensor,
    prefix_tensors: list[torch.Tensor],
    req_to_token_pool: ReqToTokenPool,
) -> None:
    req_to_token: torch.Tensor = req_to_token_pool.req_to_token
    num_reqs: int = req_pool_indices_cpu.shape[0]
    device_lens: tuple[torch.Tensor, ...] = (
        prefix_write_lens_tensor,
        alloc_start_lens_tensor,
        alloc_end_lens_tensor,
        alloc_extend_lens_tensor,
    )
    cpu_lens: tuple[torch.Tensor, ...] = (
        prefix_write_lens_cpu,
        alloc_start_lens_cpu,
        alloc_end_lens_cpu,
        alloc_extend_lens_cpu,
    )
    device_inputs: tuple[torch.Tensor, ...] = (
        req_pool_indices_tensor,
        *device_lens,
    )
    cpu_inputs: tuple[torch.Tensor, ...] = (
        req_pool_indices_cpu,
        *cpu_lens,
    )

    assert req_to_token.ndim == 2, f"{req_to_token.shape=}"
    assert req_to_token.dtype == torch.int32, f"{req_to_token.dtype=}"
    assert req_to_token.is_contiguous(), f"{req_to_token.stride()=}"
    assert (
        req_to_token.shape[0] > 0 and req_to_token.shape[1] > 0
    ), f"{req_to_token.shape=}"
    assert out_cache_loc.ndim == 1, f"{out_cache_loc.shape=}"
    assert out_cache_loc.dtype in (torch.int32, torch.int64), f"{out_cache_loc.dtype=}"
    assert out_cache_loc.is_contiguous(), f"{out_cache_loc.stride()=}"
    assert (
        out_cache_loc.device == req_to_token.device
    ), f"{out_cache_loc.device=}, {req_to_token.device=}"
    assert all(
        tensor.ndim == 1 for tensor in device_inputs + cpu_inputs
    ), f"shapes={[tensor.shape for tensor in device_inputs + cpu_inputs]}"
    assert all(tensor.shape == (num_reqs,) for tensor in device_inputs + cpu_inputs), (
        f"{num_reqs=}, "
        f"shapes={[tensor.shape for tensor in device_inputs + cpu_inputs]}"
    )
    assert all(
        tensor.dtype == torch.int64 for tensor in device_inputs + cpu_inputs
    ), f"dtypes={[tensor.dtype for tensor in device_inputs + cpu_inputs]}"
    assert all(
        tensor.is_contiguous() for tensor in device_inputs + cpu_inputs
    ), f"strides={[tensor.stride() for tensor in device_inputs + cpu_inputs]}"
    assert all(
        tensor.device == req_to_token.device for tensor in device_inputs
    ), f"{req_to_token.device=}, devices={[tensor.device for tensor in device_inputs]}"
    assert all(
        tensor.device.type == "cpu" for tensor in cpu_inputs
    ), f"devices={[tensor.device for tensor in cpu_inputs]}"
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

    assert bool(torch.all(req_pool_indices_cpu >= 0)), f"{req_pool_indices_cpu=}"
    assert bool(
        torch.all(req_pool_indices_cpu < req_to_token.shape[0])
    ), f"{req_pool_indices_cpu=}, rows={req_to_token.shape[0]}"
    assert bool(torch.all(prefix_write_lens_cpu >= 0)), f"{prefix_write_lens_cpu=}"
    assert bool(
        torch.all(alloc_start_lens_cpu >= prefix_write_lens_cpu)
    ), f"{prefix_write_lens_cpu=}, {alloc_start_lens_cpu=}"
    assert bool(
        torch.all(alloc_end_lens_cpu >= alloc_start_lens_cpu)
    ), f"{alloc_start_lens_cpu=}, {alloc_end_lens_cpu=}"
    assert bool(torch.all(alloc_extend_lens_cpu >= 0)), f"{alloc_extend_lens_cpu=}"
    assert bool(
        torch.all(alloc_end_lens_cpu <= req_to_token.shape[1])
    ), f"{alloc_end_lens_cpu=}, row_width={req_to_token.shape[1]}"
    assert torch.equal(
        alloc_end_lens_cpu - alloc_start_lens_cpu,
        alloc_extend_lens_cpu,
    ), (
        f"{alloc_start_lens_cpu=}, {alloc_end_lens_cpu=}, " f"{alloc_extend_lens_cpu=}"
    )
    alloc_extend_num_tokens: int = int(alloc_extend_lens_cpu.sum().item())
    assert alloc_extend_num_tokens == out_cache_loc.numel(), (
        f"{alloc_extend_num_tokens=}, " f"out_numel={out_cache_loc.numel()}"
    )
    prefix_write_lens: list[int] = prefix_write_lens_cpu.tolist()
    assert all(
        tensor.numel() == prefix_write_len
        for tensor, prefix_write_len in zip(prefix_tensors, prefix_write_lens)
    ), (
        f"prefix_numels={[tensor.numel() for tensor in prefix_tensors]}, "
        f"{prefix_write_lens=}"
    )
    assert all(
        prefix_write_len == 0 or tensor.device == req_to_token.device
        for tensor, prefix_write_len in zip(prefix_tensors, prefix_write_lens)
    ), (
        f"{req_to_token.device=}, "
        f"prefix_devices={[tensor.device for tensor in prefix_tensors]}"
    )

    if num_reqs == 0:
        return

    if support_triton(get_server_args().attention_backend):
        supported_accelerator_types: tuple[str, ...] = (
            "cuda",
            "npu",
            "xpu",
            "musa",
        )
        assert (
            req_to_token.device.type in supported_accelerator_types
        ), f"{req_to_token.device=}"
        assert all(
            tensor.device.type in supported_accelerator_types
            for tensor in device_inputs
        ), f"devices={[tensor.device for tensor in device_inputs]}"
        prefix_pointers: torch.Tensor = torch.tensor(
            [t.data_ptr() for t in prefix_tensors],
            dtype=torch.uint64,
            pin_memory=is_pin_memory_available(req_to_token.device),
        ).to(req_to_token.device, non_blocking=True)
        # TODO: some tensors can be reused for ForwardBatchInfo (e.g., extend_lens, cumsum_start)
        write_req_to_token_pool_triton[(req_pool_indices_tensor.shape[0],)](
            req_to_token_pool.req_to_token,
            req_pool_indices_tensor,
            prefix_pointers,
            prefix_write_lens_tensor,
            alloc_start_lens_tensor,
            alloc_end_lens_tensor,
            alloc_extend_lens_tensor,
            out_cache_loc,
            req_to_token_pool.req_to_token.shape[1],
        )
    else:
        pt = 0
        for i in range(req_pool_indices_cpu.shape[0]):
            req_idx = req_pool_indices_cpu[i].item()
            prefix_write_len = prefix_write_lens_cpu[i].item()
            alloc_start = alloc_start_lens_cpu[i].item()
            alloc_end = alloc_end_lens_cpu[i].item()
            alloc_extend_len = alloc_extend_lens_cpu[i].item()

            req_to_token_pool.write(
                (req_idx, slice(0, prefix_write_len)),
                prefix_tensors[i],
            )
            req_to_token_pool.write(
                (req_idx, slice(alloc_start, alloc_end)),
                out_cache_loc[pt : pt + alloc_extend_len],
            )
            pt += alloc_extend_len


def gather_out_cache_loc_extend(
    req_to_token_pool: ReqToTokenPool,
    *,
    req_pool_indices_tensor: torch.Tensor,
    req_pool_indices_cpu: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens_tensor: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    extend_lens_tensor: torch.Tensor,
    extend_lens_cpu: torch.Tensor,
    extend_num_tokens: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    req_to_token: torch.Tensor = req_to_token_pool.req_to_token

    assert req_to_token.ndim == 2, f"{req_to_token.shape=}"
    assert req_to_token.dtype == torch.int32, f"{req_to_token.dtype=}"
    assert req_to_token.is_contiguous(), f"{req_to_token.stride()=}"
    assert req_to_token.stride(1) == 1, f"{req_to_token.stride()=}"
    assert (
        req_to_token.shape[0] > 0 and req_to_token.shape[1] > 0
    ), f"{req_to_token.shape=}"
    assert req_pool_indices_tensor.ndim == 1, f"{req_pool_indices_tensor.shape=}"
    num_reqs: int = req_pool_indices_tensor.shape[0]

    input_tensors: tuple[torch.Tensor, ...] = (
        req_pool_indices_tensor,
        req_pool_indices_cpu,
        prefix_lens_tensor,
        prefix_lens_cpu,
        seq_lens_tensor,
        seq_lens_cpu,
        extend_lens_tensor,
        extend_lens_cpu,
    )
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
        tensor.stride(0) == 1 for tensor in input_tensors
    ), f"strides={[tensor.stride() for tensor in input_tensors]}"

    device_tensors: tuple[torch.Tensor, ...] = (
        req_pool_indices_tensor,
        prefix_lens_tensor,
        seq_lens_tensor,
        extend_lens_tensor,
    )
    cpu_tensors: tuple[torch.Tensor, ...] = (
        req_pool_indices_cpu,
        prefix_lens_cpu,
        seq_lens_cpu,
        extend_lens_cpu,
    )
    assert all(
        tensor.device == req_to_token.device for tensor in device_tensors
    ), f"{req_to_token.device=}, devices={[tensor.device for tensor in device_tensors]}"
    assert all(
        tensor.device.type == "cpu" for tensor in cpu_tensors
    ), f"devices={[tensor.device for tensor in cpu_tensors]}"

    assert type(extend_num_tokens) is int, f"{type(extend_num_tokens)=}"
    assert extend_num_tokens >= 0, f"{extend_num_tokens=}"
    assert out_dtype in (torch.int32, torch.int64), f"{out_dtype=}"
    assert bool(torch.all(req_pool_indices_cpu >= 0)), f"{req_pool_indices_cpu=}"
    assert bool(
        torch.all(req_pool_indices_cpu < req_to_token.shape[0])
    ), f"{req_pool_indices_cpu=}, rows={req_to_token.shape[0]}"
    assert bool(torch.all(prefix_lens_cpu >= 0)), f"{prefix_lens_cpu=}"
    assert bool(
        torch.all(seq_lens_cpu >= prefix_lens_cpu)
    ), f"{prefix_lens_cpu=}, {seq_lens_cpu=}"
    assert bool(
        torch.all(seq_lens_cpu <= req_to_token.shape[1])
    ), f"{seq_lens_cpu=}, row_width={req_to_token.shape[1]}"
    assert bool(torch.all(extend_lens_cpu >= 0)), f"{extend_lens_cpu=}"
    assert torch.equal(
        seq_lens_cpu - prefix_lens_cpu, extend_lens_cpu
    ), f"{prefix_lens_cpu=}, {seq_lens_cpu=}, {extend_lens_cpu=}"
    extend_num_tokens_cpu: int = int(extend_lens_cpu.sum().item())
    assert (
        extend_num_tokens_cpu == extend_num_tokens
    ), f"extend_sum={extend_num_tokens_cpu}, {extend_num_tokens=}"
    assert num_reqs > 0 or extend_num_tokens == 0, f"{num_reqs=}, {extend_num_tokens=}"

    if num_reqs == 0 or extend_num_tokens == 0:
        return torch.empty((0,), dtype=out_dtype, device=req_to_token.device)

    if support_triton(get_server_args().attention_backend):
        supported_accelerator_types: tuple[str, ...] = (
            "cuda",
            "npu",
            "xpu",
            "musa",
        )
        assert (
            req_to_token.device.type in supported_accelerator_types
        ), f"{req_to_token.device=}"
        assert all(
            tensor.device.type in supported_accelerator_types
            for tensor in device_tensors
        ), f"devices={[tensor.device for tensor in device_tensors]}"
        assert all(
            tensor.device == req_to_token.device for tensor in device_tensors
        ), f"{req_to_token.device=}, devices={[tensor.device for tensor in device_tensors]}"
        out_gather: torch.Tensor = torch.empty(
            (extend_num_tokens,), dtype=torch.int32, device=req_to_token.device
        )
        gather_req_to_token_pool_triton[(num_reqs,)](
            req_to_token,
            req_pool_indices_tensor,
            prefix_lens_tensor,
            seq_lens_tensor,
            extend_lens_tensor,
            out_gather,
            req_to_token.shape[1],
        )
        return out_gather.to(out_dtype)

    chunks: list[torch.Tensor] = []
    for index in range(num_reqs):
        req_pool_index: int = req_pool_indices_cpu[index].item()
        prefix_len: int = prefix_lens_cpu[index].item()
        seq_len: int = seq_lens_cpu[index].item()
        chunks.append(req_to_token[req_pool_index, prefix_len:seq_len])
    return torch.cat(chunks).to(out_dtype)


def get_last_loc(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    attn_backend = get_server_args().attention_backend
    uses_triton_dispatch = attn_backend not in ("ascend", "torch_native")

    if _is_hip and uses_triton_dispatch:
        # HIP-only: the legacy get_last_loc_triton kernel emits a
        # mixed-width int32->int64 store that Triton mis-compiles on HIP,
        # producing out-of-range last_loc values under EAGLE +
        # page_size>1 (e.g. with aiter unified attention or the triton
        # attention backend). The bug is in the Triton HIP codegen, not
        # in any particular attention backend, so route every HIP path
        # that would otherwise use get_last_loc_triton through the
        # int32-safe variant. Non-HIP hardware keeps the original
        # dispatcher below.
        return get_last_loc_triton_safe(
            req_to_token, req_pool_indices_tensor, prefix_lens_tensor
        )

    if uses_triton_dispatch:
        impl = get_last_loc_triton
    else:
        impl = get_last_loc_torch

    return impl(req_to_token, req_pool_indices_tensor, prefix_lens_tensor)


def get_last_loc_torch(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    return torch.where(
        prefix_lens_tensor > 0,
        req_to_token[req_pool_indices_tensor, prefix_lens_tensor - 1],
        torch.full_like(prefix_lens_tensor, -1),
    )


def alloc_token_slots(
    tree_cache: BasePrefixCache,
    num_tokens: int,
    backup_state: bool = False,
):
    allocator = tree_cache.token_to_kv_pool_allocator
    evict_from_tree_cache(tree_cache, num_tokens)

    state = None
    if backup_state:
        state = allocator.backup_state()

    out_cache_loc = allocator.alloc(num_tokens)

    if out_cache_loc is None:
        error_msg = (
            f"Out of memory. Try to lower your batch size.\n"
            f"Try to allocate {num_tokens} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    return (out_cache_loc, state) if backup_state else out_cache_loc


def _compute_dsv4_state_lens(batch, *, is_decode: bool):
    """Per-req c{4,128}_state pool alloc lens (``DSV4StateLens``) for this step.
    None on CUDA / non-V4 paths (allocator has no ``compute_dsv4_state_lens_*``).
    """
    allocator = batch.token_to_kv_pool_allocator
    if not hasattr(allocator, "compute_dsv4_state_lens_extend"):
        return None
    if is_decode:
        return allocator.compute_dsv4_state_lens_decode(batch.reqs)
    return allocator.compute_dsv4_state_lens_extend(
        batch.reqs, batch.seq_lens_cpu.tolist()
    )


def assert_alloc_extend_lens_page_aligned(
    prefix_lens_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    extend_num_tokens: int,
    page_size: int,
) -> None:
    if _is_npu or page_size == 1:
        return

    assert bool(torch.all(prefix_lens_cpu % page_size == 0)), (
        f"alloc_extend prefix lens must be page-aligned: "
        f"{prefix_lens_cpu=}, {page_size=}"
    )
    assert bool(
        torch.all(seq_lens_cpu % page_size == 0)
    ), f"alloc_extend seq lens must be page-aligned: {seq_lens_cpu=}, {page_size=}"
    assert extend_num_tokens % page_size == 0, (
        f"alloc_extend token count must be page-aligned: "
        f"{extend_num_tokens=}, {page_size=}"
    )


def alloc_paged_token_slots_extend(
    tree_cache: BasePrefixCache,
    prefix_lens: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    extend_num_tokens: int,
    backup_state: bool = False,
    req_pool_indices: Optional[torch.Tensor] = None,
    dsv4_state_lens: Optional[DSV4StateLens] = None,
    batch=None,
):
    # Over estimate the number of tokens: assume each request needs a new page.
    allocator = tree_cache.token_to_kv_pool_allocator
    assert_alloc_extend_lens_page_aligned(
        prefix_lens_cpu=prefix_lens_cpu,
        seq_lens_cpu=seq_lens_cpu,
        extend_num_tokens=extend_num_tokens,
        page_size=allocator.page_size,
    )
    num_tokens = extend_num_tokens + len(seq_lens_cpu) * allocator.page_size
    evict_from_tree_cache(tree_cache, num_tokens)

    state = None
    if backup_state:
        state = allocator.backup_state()

    is_dsv4 = req_pool_indices is not None and hasattr(allocator, "c4_attn_allocator")
    extra_alloc_kwargs = {}
    if is_dsv4:
        extra_alloc_kwargs["req_pool_indices"] = req_pool_indices
        # Per-call per-req tables for the c-pool / state last_loc lookup.
        if batch is not None:
            extra_alloc_kwargs["req_to_token_pool"] = batch.req_to_token_pool
        if dsv4_state_lens is not None:
            extra_alloc_kwargs["dsv4_state_lens"] = dsv4_state_lens

    out = allocator.alloc_extend(
        prefix_lens,
        prefix_lens_cpu,
        seq_lens,
        seq_lens_cpu,
        last_loc,
        extend_num_tokens,
        **extra_alloc_kwargs,
    )

    if is_dsv4:
        bundle = out
        out_cache_loc = None if bundle is None else bundle.out_full_loc
        if batch is not None:
            batch.out_cache_loc_dsv4 = bundle
    else:
        out_cache_loc = out

    if out_cache_loc is None:
        error_msg = (
            f"Prefill out of memory. Try to lower your batch size.\n"
            f"Try to allocate {extend_num_tokens} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    return (out_cache_loc, state) if backup_state else out_cache_loc


def alloc_req_slots(
    req_to_token_pool: ReqToTokenPool,
    reqs: list[Req],
    tree_cache: BasePrefixCache | None,
) -> list[int]:
    """Allocate request slots from the pool.

    Fail-loud: raises ``RuntimeError`` if the pool can't satisfy the batch. An
    alloc failure here means the admission budget (``PrefillAdder``) was wrong
    and should surface rather than be masked.
    """
    num_reqs = len(reqs)
    if isinstance(req_to_token_pool, HybridReqToTokenPool):
        # Byte-coordinated for the shared allocator (accounts for the peer full
        # sub-pool's bytes); plain slot free count for the non-shared one.
        mamba_available_size = (
            req_to_token_pool.mamba_allocator.schedulable_available_size()
        )
        # Eviction headroom factor: 3x (or lazy variant) for radix COW, 1x for chunk.
        if tree_cache.supports_mamba():
            factor = (
                MAMBA_STATE_PER_REQ_PREFIX_CACHE_LAZY
                if req_to_token_pool.enable_mamba_extra_buffer_lazy
                else MAMBA_STATE_PER_REQ_PREFIX_CACHE
            )
        else:
            factor = MAMBA_STATE_PER_REQ_NO_CACHE
        mamba_state_needed = num_reqs * factor
        if mamba_available_size < mamba_state_needed:
            if tree_cache is not None and tree_cache.supports_mamba():
                mamba_num = max(0, mamba_state_needed - mamba_available_size)
                tree_cache.evict(EvictParams(num_tokens=0, mamba_num=mamba_num))
    req_pool_indices = req_to_token_pool.alloc(reqs)
    if req_pool_indices is None:
        raise RuntimeError(
            "alloc_req_slots runs out of memory. "
            "Please set a smaller number for `--max-running-requests`. "
            f"{req_to_token_pool.available_size()=}, {num_reqs=}, "
        )
    return req_pool_indices


def _alloc_page_size(batch: ScheduleBatch) -> int:
    # DCP swaps in an allocator whose page_size is server_args.page_size *
    # dcp_size, so it can be > 1 even when tree_cache.page_size is 1; branch on
    # the real allocator's page_size there. Elsewhere the two are equal.
    if (_is_hip or _is_cuda) and get_server_args().dcp_size > 1:
        return batch.tree_cache.token_to_kv_pool_allocator.page_size
    return batch.tree_cache.page_size


def alloc_for_extend(
    batch: ScheduleBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Allocate KV cache for extend batch and write to req_to_token_pool.

    Returns ``(out_cache_loc, req_pool_indices_device, req_pool_indices_cpu)``
    (the last is the host/CPU mirror). ``alloc_req_slots`` raises ``RuntimeError``
    if the pool can't satisfy the batch (fail-loud — see its docstring).
    """
    # free out-of-window swa tokens
    batch.maybe_evict_swa()

    prefix_tensors: list[torch.Tensor] = [r.prefix_indices for r in batch.reqs]

    # Create tensors for allocation
    prefix_lens_cpu: torch.Tensor = torch.tensor(batch.prefix_lens, dtype=torch.int64)
    extend_lens_cpu: torch.Tensor = torch.tensor(batch.extend_lens, dtype=torch.int64)
    prefix_lens_device: torch.Tensor = prefix_lens_cpu.to(
        batch.device, non_blocking=True
    )
    extend_lens_device: torch.Tensor = extend_lens_cpu.to(
        batch.device, non_blocking=True
    )
    allocator_page: int = _alloc_page_size(batch)
    uses_aligned_lens: bool = not _is_npu and allocator_page > 1
    alloc_start_lens_cpu: torch.Tensor
    alloc_end_lens_cpu: torch.Tensor
    alloc_extend_lens_cpu: torch.Tensor
    alloc_start_lens_device: torch.Tensor
    alloc_end_lens_device: torch.Tensor
    alloc_extend_lens_device: torch.Tensor
    if uses_aligned_lens:
        alloc_start_lens: list[int] = [
            max(
                prefix_len,
                req.kv.kv_allocated_len if req.kv is not None else 0,
            )
            for req, prefix_len in zip(batch.reqs, batch.prefix_lens)
        ]
        alloc_end_lens: list[int] = [
            ceil_align(seq_len, allocator_page)
            for seq_len in batch.seq_lens_cpu.tolist()
        ]
        alloc_extend_lens: list[int] = [
            alloc_end - alloc_start
            for alloc_start, alloc_end in zip(alloc_start_lens, alloc_end_lens)
        ]
        assert all(extend_len >= 0 for extend_len in alloc_extend_lens)
        alloc_start_lens_cpu = torch.tensor(alloc_start_lens, dtype=torch.int64)
        alloc_end_lens_cpu = torch.tensor(alloc_end_lens, dtype=torch.int64)
        alloc_extend_lens_cpu = torch.tensor(alloc_extend_lens, dtype=torch.int64)
        alloc_start_lens_device = alloc_start_lens_cpu.to(
            batch.device, non_blocking=True
        )
        alloc_end_lens_device = alloc_end_lens_cpu.to(batch.device, non_blocking=True)
        alloc_extend_lens_device = alloc_extend_lens_cpu.to(
            batch.device, non_blocking=True
        )
    else:
        alloc_start_lens_cpu = prefix_lens_cpu
        alloc_end_lens_cpu = batch.seq_lens_cpu
        alloc_extend_lens_cpu = extend_lens_cpu
        alloc_start_lens_device = prefix_lens_device
        alloc_end_lens_device = batch.seq_lens
        alloc_extend_lens_device = extend_lens_device
    alloc_extend_num_tokens: int = int(alloc_extend_lens_cpu.sum().item())

    # Allocate req slots (raises RuntimeError if the pool is exhausted)
    req_pool_indices = alloc_req_slots(
        batch.req_to_token_pool, batch.reqs, batch.tree_cache
    )
    req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
    req_pool_indices_device = req_pool_indices_cpu.to(batch.device, non_blocking=True)

    # Allocate KV cache (throws exception on failure)
    if allocator_page == 1:
        out_cache_loc = alloc_token_slots(batch.tree_cache, batch.extend_num_tokens)
    else:
        # Paged allocation - build last_loc
        last_loc: list[torch.Tensor] = []
        for index, (req, prefix_tensor, alloc_start) in enumerate(
            zip(batch.reqs, prefix_tensors, alloc_start_lens_cpu.tolist())
        ):
            if uses_aligned_lens and req.kv is not None and alloc_start > 0:
                req_pool_index = req_pool_indices[index]
                last_loc.append(
                    batch.req_to_token_pool.req_to_token[
                        req_pool_index, alloc_start - 1 : alloc_start
                    ]
                )
            elif len(prefix_tensor) > 0:
                last_loc.append(prefix_tensor[-1:])
            else:
                last_loc.append(torch.tensor([-1], device=batch.device))
        out_cache_loc = alloc_paged_token_slots_extend(
            tree_cache=batch.tree_cache,
            prefix_lens=alloc_start_lens_device,
            prefix_lens_cpu=alloc_start_lens_cpu,
            seq_lens=alloc_end_lens_device,
            seq_lens_cpu=alloc_end_lens_cpu,
            last_loc=torch.cat(last_loc),
            extend_num_tokens=alloc_extend_num_tokens,
            req_pool_indices=req_pool_indices_device,
            dsv4_state_lens=_compute_dsv4_state_lens(batch, is_decode=False),
            batch=batch,
        )

    # Write to req_to_token_pool
    write_cache_indices(
        out_cache_loc=out_cache_loc,
        req_pool_indices_tensor=req_pool_indices_device,
        req_pool_indices_cpu=req_pool_indices_cpu,
        prefix_write_lens_tensor=prefix_lens_device,
        prefix_write_lens_cpu=prefix_lens_cpu,
        alloc_start_lens_tensor=alloc_start_lens_device,
        alloc_start_lens_cpu=alloc_start_lens_cpu,
        alloc_end_lens_tensor=alloc_end_lens_device,
        alloc_end_lens_cpu=alloc_end_lens_cpu,
        alloc_extend_lens_tensor=alloc_extend_lens_device,
        alloc_extend_lens_cpu=alloc_extend_lens_cpu,
        prefix_tensors=prefix_tensors,
        req_to_token_pool=batch.req_to_token_pool,
    )

    gathered: torch.Tensor = gather_out_cache_loc_extend(
        batch.req_to_token_pool,
        req_pool_indices_tensor=req_pool_indices_device,
        req_pool_indices_cpu=req_pool_indices_cpu,
        prefix_lens_tensor=prefix_lens_device,
        prefix_lens_cpu=prefix_lens_cpu,
        seq_lens_tensor=batch.seq_lens,
        seq_lens_cpu=batch.seq_lens_cpu,
        extend_lens_tensor=extend_lens_device,
        extend_lens_cpu=extend_lens_cpu,
        extend_num_tokens=batch.extend_num_tokens,
        out_dtype=out_cache_loc.dtype,
    )
    if envs.SGLANG_DEBUG_MEMORY_POOL.get():
        assert gathered.numel() == batch.extend_num_tokens
        assert bool(torch.all(gathered != 0))
        if uses_aligned_lens:
            for req_pool_index, alloc_start, alloc_end in zip(
                req_pool_indices,
                alloc_start_lens_cpu.tolist(),
                alloc_end_lens_cpu.tolist(),
            ):
                allocated_slots = batch.req_to_token_pool.req_to_token[
                    req_pool_index, alloc_start:alloc_end
                ]
                assert bool(torch.all(allocated_slots != 0))
    out_cache_loc = gathered

    # DSV4-NPU hook: no-op on non-DSV4 paths.
    if _is_npu:
        maybe_write_dsv4_extend(
            batch,
            req_pool_indices_cpu,
            prefix_lens_cpu,
            batch.seq_lens_cpu,
        )

    from sglang.srt.managers.schedule_batch import ReqKvInfo

    for req, allocated_len in zip(batch.reqs, alloc_end_lens_cpu.tolist()):
        if req.kv is None:
            req.kv = ReqKvInfo(
                kv_allocated_len=allocated_len,
                swa_evicted_seqlen=0,
            )
        else:
            req.kv.kv_allocated_len = allocated_len

    return out_cache_loc, req_pool_indices_device, req_pool_indices_cpu


def alloc_paged_token_slots_decode(
    tree_cache: BasePrefixCache,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    token_per_req: int = 1,
    req_pool_indices: Optional[torch.Tensor] = None,
    dsv4_state_lens: Optional[DSV4StateLens] = None,
    batch=None,
) -> torch.Tensor:
    """Allocate paged KV cache for decode batch."""
    allocator = tree_cache.token_to_kv_pool_allocator
    # Over estimate the number of tokens: assume each request needs a new page.
    num_tokens = len(seq_lens) * allocator.page_size
    evict_from_tree_cache(tree_cache, num_tokens)

    # DSV4-NPU allocator also needs req_pool_indices + per-req state lens and
    # returns a DSV4OutCacheLoc bundle; hasattr-gated so others stay unchanged.
    is_dsv4 = req_pool_indices is not None and hasattr(allocator, "c4_attn_allocator")
    extra_alloc_kwargs = {}
    if is_dsv4:
        extra_alloc_kwargs["req_pool_indices"] = req_pool_indices
        # Per-call per-req tables for the last_loc lookup.
        if batch is not None:
            extra_alloc_kwargs["req_to_token_pool"] = batch.req_to_token_pool
        if dsv4_state_lens is not None:
            extra_alloc_kwargs["dsv4_state_lens"] = dsv4_state_lens

    out = allocator.alloc_decode(seq_lens, seq_lens_cpu, last_loc, **extra_alloc_kwargs)

    if is_dsv4:
        bundle = out
        out_cache_loc = None if bundle is None else bundle.out_full_loc
        if batch is not None:
            batch.out_cache_loc_dsv4 = bundle
    else:
        out_cache_loc = out

    if out_cache_loc is None:
        error_msg = (
            f"Decode out of memory. Try to lower your batch size.\n"
            f"Try to allocate {len(seq_lens) * token_per_req} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    return out_cache_loc


def alloc_for_decode(batch: ScheduleBatch, token_per_req: int) -> torch.Tensor:
    """
    Allocate KV cache for decode batch and write to req_to_token_pool.

    Returns:
        out_cache_loc: allocated cache locations
    """

    batch.maybe_evict_swa()

    seq_lens_gpu: torch.Tensor = batch.seq_lens
    bs: int = seq_lens_gpu.shape[0]
    allocator_page: int = _alloc_page_size(batch)
    locs: torch.Tensor
    locs_cpu: torch.Tensor
    if batch.model_config.is_encoder_decoder:
        assert batch.encoder_lens is not None
        assert batch.encoder_lens_cpu is not None
        encoder_lens_cpu: torch.Tensor = torch.tensor(
            batch.encoder_lens_cpu,
            dtype=batch.seq_lens_cpu.dtype,
            device=batch.seq_lens_cpu.device,
        )
        locs = batch.encoder_lens + seq_lens_gpu
        locs_cpu = encoder_lens_cpu + batch.seq_lens_cpu
    else:
        locs = seq_lens_gpu.clone()
        locs_cpu = batch.seq_lens_cpu.clone()

    if allocator_page == 1:
        # Non-paged allocation
        out_cache_loc = alloc_token_slots(batch.tree_cache, bs * token_per_req)
    # Paged allocation
    elif _is_npu:
        last_loc = batch.req_to_token_pool.req_to_token[
            batch.req_pool_indices, seq_lens_gpu - 1
        ]
        seq_lens_next = seq_lens_gpu + token_per_req
        out_cache_loc = alloc_paged_token_slots_decode(
            tree_cache=batch.tree_cache,
            seq_lens=seq_lens_next,
            seq_lens_cpu=batch.seq_lens_cpu + token_per_req,
            last_loc=last_loc,
            token_per_req=token_per_req,
            req_pool_indices=batch.req_pool_indices,
            dsv4_state_lens=_compute_dsv4_state_lens(batch, is_decode=True),
            batch=batch,
        )
    else:
        assert token_per_req == 1
        last_loc = batch.req_to_token_pool.req_to_token[
            batch.req_pool_indices, locs - 1
        ]
        seq_lens_next = locs + 1
        out_cache_loc = alloc_paged_token_slots_decode(
            tree_cache=batch.tree_cache,
            seq_lens=seq_lens_next,
            seq_lens_cpu=locs_cpu + 1,
            last_loc=last_loc,
            token_per_req=token_per_req,
            req_pool_indices=batch.req_pool_indices,
            dsv4_state_lens=_compute_dsv4_state_lens(batch, is_decode=True),
            batch=batch,
        )

    # Write to req_to_token_pool
    if allocator_page == 1 or _is_npu:
        batch.req_to_token_pool.write(
            (batch.req_pool_indices, locs), out_cache_loc.to(torch.int32)
        )
        for req in batch.reqs:
            req.kv.kv_allocated_len += token_per_req
    else:
        allocated_old_cpu: torch.Tensor = torch.tensor(
            [req.kv.kv_allocated_len for req in batch.reqs],
            dtype=locs_cpu.dtype,
            device=locs_cpu.device,
        )
        aligned_mask_cpu: torch.Tensor = allocated_old_cpu % allocator_page == 0
        aligned_end_cpu: torch.Tensor = (
            (locs_cpu + allocator_page) // allocator_page * allocator_page
        )
        crossing_mask_cpu: torch.Tensor = aligned_mask_cpu & (
            aligned_end_cpu > allocated_old_cpu
        )
        transitional_mask_cpu: torch.Tensor = ~aligned_mask_cpu
        crossing_indices_cpu: torch.Tensor = torch.nonzero(
            crossing_mask_cpu, as_tuple=False
        ).flatten()
        transitional_indices_cpu: torch.Tensor = torch.nonzero(
            transitional_mask_cpu, as_tuple=False
        ).flatten()
        crossing_indices: torch.Tensor = crossing_indices_cpu.to(
            device=batch.device, non_blocking=True
        )
        transitional_indices: torch.Tensor = transitional_indices_cpu.to(
            device=batch.device, non_blocking=True
        )

        if crossing_indices.numel() > 0:
            position_offsets: torch.Tensor = torch.arange(
                allocator_page,
                dtype=locs.dtype,
                device=batch.device,
            )
            value_offsets: torch.Tensor = torch.arange(
                allocator_page,
                dtype=out_cache_loc.dtype,
                device=batch.device,
            )
            crossing_positions: torch.Tensor = (
                locs[crossing_indices].unsqueeze(1) + position_offsets
            )
            crossing_values: torch.Tensor = (
                out_cache_loc[crossing_indices].unsqueeze(1) + value_offsets
            )
            crossing_req_indices: torch.Tensor = batch.req_pool_indices[
                crossing_indices
            ].unsqueeze(1)
            batch.req_to_token_pool.write(
                (crossing_req_indices, crossing_positions),
                crossing_values.to(torch.int32),
            )

        if transitional_indices.numel() > 0:
            batch.req_to_token_pool.write(
                (
                    batch.req_pool_indices[transitional_indices],
                    locs[transitional_indices],
                ),
                out_cache_loc[transitional_indices].to(torch.int32),
            )

        aligned_allocated_cpu: torch.Tensor = torch.maximum(
            allocated_old_cpu,
            aligned_end_cpu,
        )
        transitional_allocated_cpu: torch.Tensor = torch.maximum(
            allocated_old_cpu,
            locs_cpu + 1,
        )
        allocated_next_cpu: torch.Tensor = torch.where(
            aligned_mask_cpu,
            aligned_allocated_cpu,
            transitional_allocated_cpu,
        )
        for req, allocated_len in zip(batch.reqs, allocated_next_cpu.tolist()):
            req.kv.kv_allocated_len = allocated_len

    gathered: torch.Tensor = batch.req_to_token_pool.req_to_token[
        batch.req_pool_indices, locs
    ].to(out_cache_loc.dtype)
    if envs.SGLANG_DEBUG_MEMORY_POOL.get():
        assert bool(torch.all(gathered != 0))
        if allocator_page == 1 or _is_npu:
            torch.testing.assert_close(gathered, out_cache_loc, rtol=0, atol=0)
        else:
            if transitional_indices.numel() > 0:
                torch.testing.assert_close(
                    gathered[transitional_indices],
                    out_cache_loc[transitional_indices],
                    rtol=0,
                    atol=0,
                )
            if crossing_indices.numel() > 0:
                assert bool(
                    torch.all(out_cache_loc[crossing_indices] % allocator_page == 0)
                )
            for index, req in enumerate(batch.reqs):
                if bool(aligned_mask_cpu[index]):
                    assert req.kv.kv_allocated_len % allocator_page == 0
                    assert req.kv.kv_allocated_len >= ceil_align(
                        req.kv_committed_len,
                        allocator_page,
                    )
    out_cache_loc = gathered

    # DSV4-NPU hook: no-op on non-DSV4 paths.
    if _is_npu:
        maybe_write_dsv4_decode(
            batch,
            batch.seq_lens_cpu + token_per_req,
            token_per_req,
        )

    return out_cache_loc


@triton.jit
def assign_req_to_token_pool(
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


def assign_req_to_token_pool_func(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
    batch_size: int,
):
    assign_req_to_token_pool[(batch_size,)](
        req_pool_indices,
        req_to_token,
        start_offset,
        end_offset,
        out_cache_loc,
        req_to_token.shape[1],
        next_power_of_2(batch_size),
    )


def _alloc_paged_token_slots_extend_npu(*args, **kwargs):
    from sglang.srt.hardware_backend.npu.dsv4.dsv4_allocator import (
        alloc_paged_token_slots_extend_npu,
    )

    return alloc_paged_token_slots_extend_npu(*args, **kwargs)


ALLOC_EXTEND_FUNCS = defaultdict(
    lambda: alloc_paged_token_slots_extend,
    {"npu": _alloc_paged_token_slots_extend_npu},
)


def alloc_for_spec_decode(
    tree_cache: BasePrefixCache,
    req_to_token_pool: ReqToTokenPool,
    *,
    reqs: list[Req],
    req_pool_indices: torch.Tensor,
    cur_kv_lens: torch.Tensor,
    cur_kv_lens_cpu: torch.Tensor,
    nxt_kv_lens: torch.Tensor,
    nxt_kv_lens_cpu: torch.Tensor,
    num_needed_tokens: int,
    batch: Optional[ScheduleBatch] = None,
) -> None:
    allocator_page: int = tree_cache.token_to_kv_pool_allocator.page_size
    allocation_nxt_kv_lens_cpu: torch.Tensor
    allocation_nxt_kv_lens: torch.Tensor
    allocation_num_needed_tokens: int
    if not _is_npu and allocator_page > 1:
        allocation_nxt_kv_lens_cpu = (
            (nxt_kv_lens_cpu + allocator_page - 1) // allocator_page * allocator_page
        )
        allocation_nxt_kv_lens = (
            (nxt_kv_lens + allocator_page - 1) // allocator_page * allocator_page
        )
        allocation_num_needed_tokens = int(
            (allocation_nxt_kv_lens_cpu - cur_kv_lens_cpu).sum().item()
        )
    else:
        allocation_nxt_kv_lens_cpu = nxt_kv_lens_cpu
        allocation_nxt_kv_lens = nxt_kv_lens
        allocation_num_needed_tokens = num_needed_tokens

    if allocation_nxt_kv_lens_cpu.numel() > 0:
        max_allocated_len: int = int(allocation_nxt_kv_lens_cpu.max().item())
        row_width: int = req_to_token_pool.req_to_token.shape[1]
        assert max_allocated_len <= row_width, (
            f"spec decode allocation endpoint ({max_allocated_len}) exceeds "
            f"req_to_token row width ({row_width}); page_size={allocator_page}"
        )

    if allocation_num_needed_tokens > 0:
        if allocator_page == 1:
            out_cache_loc = alloc_token_slots(
                tree_cache=tree_cache,
                num_tokens=allocation_num_needed_tokens,
            )
        else:
            last_loc = get_last_loc(
                req_to_token_pool.req_to_token, req_pool_indices, cur_kv_lens
            )
            device_type = getattr(
                batch.device, "type", str(batch.device).split(":", 1)[0]
            )
            out_cache_loc = ALLOC_EXTEND_FUNCS[device_type](
                tree_cache,
                cur_kv_lens,
                cur_kv_lens_cpu,
                allocation_nxt_kv_lens,
                allocation_nxt_kv_lens_cpu,
                last_loc,
                allocation_num_needed_tokens,
                req_pool_indices=req_pool_indices,
                batch=batch,
            )
        # Updating req_to_token is a write to a shared tensor: it must not overlap
        # with the previous batch's forward, which also reads req_to_token.
        assign_req_to_token_pool_func(
            req_pool_indices,
            req_to_token_pool.req_to_token,
            cur_kv_lens,
            allocation_nxt_kv_lens,
            out_cache_loc,
            len(reqs),
        )

    for i, req in enumerate(reqs):
        req.kv.kv_allocated_len = max(
            req.kv.kv_allocated_len,
            int(allocation_nxt_kv_lens_cpu[i]),
        )
