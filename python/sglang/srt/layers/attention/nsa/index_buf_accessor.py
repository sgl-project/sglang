from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool

"""
k: data, 128 item per token, fp8
s: scale, 1 item per token, fp32
"""


class GetK:
    @classmethod
    def execute(cls, *args, **kwargs):
        return cls.triton(*args, **kwargs)

    @classmethod
    def slow(
        cls, pool: "NSATokenToKVPool", buf, seq_len: int, page_indices: torch.Tensor
    ):
        num_pages = (seq_len + pool.page_size - 1) // pool.page_size
        seq_len_ = num_pages * pool.page_size
        index_k_fp8 = torch.empty(
            (seq_len_, pool.index_head_dim),
            dtype=torch.uint8,
            device=pool.device,
        )
        for i in range(num_pages):
            page_index = page_indices[i]
            index_k_fp8[i * pool.page_size : (i + 1) * pool.page_size] = buf[
                page_index
            ][: pool.page_size * pool.index_head_dim].view(-1, pool.index_head_dim)

        return index_k_fp8[:seq_len]

    @classmethod
    def torch_fast(
        cls, pool: "NSATokenToKVPool", buf, seq_len: int, page_indices: torch.Tensor
    ):
        """
        :param page_indices: (num_pages,), int32
        :return: (seq_len, index_head_dim), uint8
        """

        # can handle per 128B instead of per element

        # page_indices: (num_pages,), element := a page index
        buf_numel_per_page = buf.shape[1]

        num_k_bytes_per_page = pool.page_size * pool.index_head_dim
        num_k_bytes_per_token = pool.index_head_dim

        # buf: (num_pages, page_size 64 * head_dim 128 + page_size 64 * fp32_nbytes 4), uint8
        # flat_buf: (whatever,), uint8
        flat_buf = buf.flatten()

        # flat_indices: (num_pages, num_k_bytes_per_page), int32, element := an index into flat_buf that we want to access
        flat_indices = (page_indices * buf_numel_per_page)[:, None] + torch.arange(
            num_k_bytes_per_page, dtype=torch.int32, device="cuda"
        )[None, :]
        flat_indices = flat_indices.flatten()[: seq_len * num_k_bytes_per_token]

        out = flat_buf[flat_indices]
        return out.view(-1, 128)

    @classmethod
    def triton(
        cls, pool: "NSATokenToKVPool", buf, seq_len: int, page_indices: torch.Tensor
    ):
        """
        Triton implementation for gathering K data from paged buffer.
        :param page_indices: (num_pages,), int32/int64
        :return: (seq_len, index_head_dim), uint8
        """
        return _get_k_triton(
            buf=buf,
            page_indices=page_indices,
            seq_len=seq_len,
            page_size=pool.page_size,
            index_head_dim=pool.index_head_dim,
        )


class GetS:
    @classmethod
    def execute(cls, *args, **kwargs):
        return cls.triton(*args, **kwargs)

    @classmethod
    def slow(
        cls, pool: "NSATokenToKVPool", buf, seq_len: int, page_indices: torch.Tensor
    ):
        num_pages = (seq_len + pool.page_size - 1) // pool.page_size
        seq_len_ = num_pages * pool.page_size
        assert pool.index_head_dim // pool.quant_block_size == 1
        index_k_scale_fp8 = torch.empty(
            (seq_len_, 4),
            dtype=torch.uint8,
            device=pool.device,
        )
        for i in range(num_pages):
            page_index = page_indices[i]
            index_k_scale_fp8[i * pool.page_size : (i + 1) * pool.page_size] = buf[
                page_index
            ][pool.page_size * pool.index_head_dim :].view(-1, 4)
        return index_k_scale_fp8[:seq_len]

    @classmethod
    def torch_fast(
        cls, pool: "NSATokenToKVPool", buf, seq_len: int, page_indices: torch.Tensor
    ):
        """
        :param page_indices: (num_pages,), int32
        :return: (seq_len, index_head_dim // quant_block_size), uint8
        """
        buf_numel_per_page = buf.shape[1]

        num_s_bytes_per_page = buf.shape[1] - pool.page_size * pool.index_head_dim
        num_s_bytes_per_token = pool.index_head_dim // pool.quant_block_size * 4
        s_offset_in_page = pool.page_size * pool.index_head_dim

        flat_buf = buf.flatten()
        flat_indices = (
            (page_indices * buf_numel_per_page)[:, None]
            + torch.arange(num_s_bytes_per_page, dtype=torch.int32, device="cuda")[
                None, :
            ]
            + s_offset_in_page
        )
        flat_indices = flat_indices.flatten()[: seq_len * num_s_bytes_per_token]

        out = flat_buf[flat_indices]
        return out.view(-1, 4)

    @classmethod
    def triton(
        cls, pool: "NSATokenToKVPool", buf, seq_len: int, page_indices: torch.Tensor
    ):
        """
        Triton implementation for gathering S (scale) data from paged buffer.
        :param page_indices: (num_pages,), int32/int64
        :return: (seq_len, 4), uint8
        """
        return _get_s_triton(
            buf=buf,
            page_indices=page_indices,
            seq_len=seq_len,
            page_size=pool.page_size,
            index_head_dim=pool.index_head_dim,
        )


class GetKAndS:
    @classmethod
    def execute(cls, *args, **kwargs):
        return cls.triton(*args, **kwargs)

    @classmethod
    def triton(
        cls, pool: "NSATokenToKVPool", buf, seq_len: int, page_indices: torch.Tensor
    ):
        """
        Triton implementation for gathering both K and S data from paged buffer in a single call.
        :param page_indices: (num_pages,), int32/int64
        :return: tuple of (k_fp8, k_scale) where
                 k_fp8: (seq_len, index_head_dim), uint8
                 k_scale: (seq_len, 4), uint8
        """
        return _get_k_and_s_triton(
            buf=buf,
            page_indices=page_indices,
            seq_len=seq_len,
            page_size=pool.page_size,
            index_head_dim=pool.index_head_dim,
        )


class SetK:
    @classmethod
    def execute(cls, *args, buf, **kwargs):
        return cls.torch_fast(*args, **kwargs, buf=buf)

    @classmethod
    def slow(
        cls,
        pool: "NSATokenToKVPool",
        buf: torch.Tensor,
        loc: torch.Tensor,
        index_k: torch.Tensor,
    ):
        for i in range(len(loc)):
            page_index = loc[i] // pool.page_size
            offset = loc[i] % pool.page_size
            buf[
                page_index,
                offset * pool.index_head_dim : (offset + 1) * pool.index_head_dim,
            ] = index_k[i].view(torch.uint8)

    @classmethod
    def torch_fast(
        cls,
        pool: "NSATokenToKVPool",
        buf: torch.Tensor,
        loc: torch.Tensor,
        index_k: torch.Tensor,
    ):
        (num_tokens_to_write,) = loc.shape
        buf_numel_per_page = buf.shape[1]
        num_k_bytes_per_token = pool.index_head_dim

        # loc: (num_tokens_to_write,), int32, element := the token index to write to
        loc_page_index = loc // pool.page_size
        loc_token_offset_in_page = loc % pool.page_size

        flat_buf = buf.flatten()
        flat_indices = (
            (loc_page_index * buf_numel_per_page)[:, None]
            + (loc_token_offset_in_page * num_k_bytes_per_token)[:, None]
            + torch.arange(num_k_bytes_per_token, dtype=torch.int32, device="cuda")[
                None, :
            ]
        )
        num_k_bytes_total = num_tokens_to_write * num_k_bytes_per_token
        flat_indices = flat_indices.flatten()[:num_k_bytes_total]
        flat_buf[flat_indices] = index_k.view(torch.uint8).flatten()


class SetS:
    @classmethod
    def execute(cls, *args, buf, **kwargs):
        return cls.torch_fast(*args, **kwargs, buf=buf)

    @classmethod
    def slow(
        cls,
        pool: "NSATokenToKVPool",
        buf: torch.Tensor,
        loc: torch.Tensor,
        index_k_scale: torch.Tensor,
    ):
        for i in range(len(loc)):
            page_index = loc[i] // pool.page_size
            offset = loc[i] % pool.page_size
            start = pool.page_size * pool.index_head_dim
            buf[page_index, start + offset * 4 : start + (offset + 1) * 4] = (
                index_k_scale[i].view(torch.uint8)
            )

    @classmethod
    def torch_fast(
        cls,
        pool: "NSATokenToKVPool",
        buf: torch.Tensor,
        loc: torch.Tensor,
        index_k_scale: torch.Tensor,
    ):
        (num_tokens_to_write,) = loc.shape
        buf_numel_per_page = buf.shape[1]
        num_s_bytes_per_token = 4
        s_offset_in_page = pool.page_size * pool.index_head_dim

        # loc: (num_tokens_to_write,), int32, element := the token index to write to
        loc_page_index = loc // pool.page_size
        loc_token_offset_in_page = loc % pool.page_size

        flat_buf = buf.flatten()
        flat_indices = (
            (loc_page_index * buf_numel_per_page)[:, None]
            + s_offset_in_page
            + (loc_token_offset_in_page * num_s_bytes_per_token)[:, None]
            + torch.arange(num_s_bytes_per_token, dtype=torch.int32, device="cuda")[
                None, :
            ]
        )
        number_s_bytes_total = num_tokens_to_write * num_s_bytes_per_token
        flat_indices = flat_indices.flatten()[:number_s_bytes_total]
        flat_buf[flat_indices] = index_k_scale.view(torch.uint8).flatten()


class SetKAndS:
    @classmethod
    def execute(cls, *args, buf, **kwargs):
        if 0:
            # print("SetK, SetS comparison test")
            buf_cloned = buf.clone()
            cls.vanilla(*args, **kwargs, buf=buf)
            cls.triton(*args, **kwargs, buf=buf_cloned)

            def _clear_token_0(target):
                target[0, :128] = target[0, 64 * 128 : 64 * 128 + 4] = 0

            _clear_token_0(buf)
            _clear_token_0(buf_cloned)

            assert torch.all(
                buf == buf_cloned
            ), f"{buf=} {buf_cloned=} {kwargs['loc'].to_list()=}"
            return

        cls.triton(*args, **kwargs, buf=buf)

    @classmethod
    def vanilla(cls, pool, buf, loc, index_k, index_k_scale):
        SetK.execute(pool=pool, buf=buf, loc=loc, index_k=index_k)
        SetS.execute(pool=pool, buf=buf, loc=loc, index_k_scale=index_k_scale)

    @classmethod
    def triton(cls, pool, buf, loc, index_k, index_k_scale):
        _set_k_and_s_triton(
            buf=buf,
            loc=loc,
            index_k=index_k,
            index_k_scale=index_k_scale,
            page_size=pool.page_size,
        )


def _set_k_and_s_triton(
    buf: torch.Tensor,
    loc: torch.Tensor,
    index_k: torch.Tensor,
    index_k_scale: torch.Tensor,
    page_size: int,
):
    """
    :param buf: (num_pages, page_size 64 * (128B data + 4B scale)), uint8
    :param loc: (num_tokens_to_write,), int, element := the token index to write to
    :param index_k: (num_tokens_to_write, 128 elem), fp8
    :param index_k_scale: (num_tokens_to_write, 1 elem), fp32
    :return:
    """
    num_pages, buf_numel_per_page = buf.shape
    (num_tokens_to_write,) = loc.shape
    num_tokens_to_write_, index_head_dim = index_k.shape

    # Handle both 1D (num_tokens,) and 2D (num_tokens, 1) shapes for index_k_scale
    if index_k_scale.ndim == 1:
        num_tokens_to_write__ = index_k_scale.shape[0]
        scale_dim = 1
    elif index_k_scale.ndim == 2:
        num_tokens_to_write__, scale_dim = index_k_scale.shape
    else:
        raise ValueError(
            f"index_k_scale must be 1D or 2D, got shape {index_k_scale.shape}"
        )

    assert buf_numel_per_page == 64 * (128 + 4)
    assert num_tokens_to_write == num_tokens_to_write_ == num_tokens_to_write__
    assert index_head_dim == 128
    assert scale_dim == 1
    assert page_size == 64

    assert buf.dtype == torch.uint8
    assert loc.dtype == torch.int64, f"{loc.dtype=}"  # can be int32
    assert index_k.dtype == torch.float8_e4m3fn
    assert index_k_scale.dtype == torch.float32

    assert buf.is_contiguous()
    assert loc.is_contiguous()
    assert index_k.is_contiguous()
    assert index_k_scale.is_contiguous()

    buf_fp8 = buf.view(torch.float8_e4m3fn)
    buf_fp32 = buf.view(torch.float32)

    _set_k_and_s_triton_kernel[(num_tokens_to_write,)](
        buf_fp8,
        buf_fp32,
        loc,
        index_k,
        index_k_scale,
        index_k.stride(0),
        PAGE_SIZE=page_size,
        BUF_NUMEL_PER_PAGE=buf_numel_per_page,
        NUM_K_ELEMS_PER_TOKEN=index_head_dim,
        S_OFFSET_NBYTES_IN_PAGE=page_size * index_head_dim,
    )


@triton.jit
def _set_k_and_s_triton_kernel(
    buf_fp8_ptr,
    buf_fp32_ptr,
    loc_ptr,
    index_k_ptr,
    index_k_scale_ptr,
    index_k_ptr_stride_0,
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    NUM_K_ELEMS_PER_TOKEN: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
):
    token_id = tl.program_id(0)

    loc = tl.load(loc_ptr + token_id)

    in_k_offsets = token_id * index_k_ptr_stride_0 + tl.arange(0, NUM_K_ELEMS_PER_TOKEN)

    # no need for `mask`, since we read 128B for k and 4B for scale, both pow of 2
    k = tl.load(index_k_ptr + in_k_offsets)
    k_scale = tl.load(index_k_scale_ptr + token_id)

    loc_page_index = loc // PAGE_SIZE
    loc_token_offset_in_page = loc % PAGE_SIZE

    out_k_offsets = (
        loc_page_index * BUF_NUMEL_PER_PAGE
        + loc_token_offset_in_page * NUM_K_ELEMS_PER_TOKEN
        + tl.arange(0, NUM_K_ELEMS_PER_TOKEN)
    )

    # "//4" b/c it is fp32 instead of uint8
    out_s_offset = (
        loc_page_index * BUF_NUMEL_PER_PAGE // 4
        + S_OFFSET_NBYTES_IN_PAGE // 4
        + loc_token_offset_in_page
    )

    tl.store(buf_fp8_ptr + out_k_offsets, k)
    tl.store(buf_fp32_ptr + out_s_offset, k_scale)


def _get_k_triton(
    buf: torch.Tensor,
    page_indices: torch.Tensor,
    seq_len: int,
    page_size: int,
    index_head_dim: int,
):
    """
    Gather K (key) data from paged buffer using Triton.

    :param buf: (num_pages, page_size * 128 + page_size * 4), uint8
    :param page_indices: (num_pages,), int32/int64
    :param seq_len: int, number of tokens to gather
    :param page_size: int, typically 64
    :param index_head_dim: int, typically 128
    :return: (seq_len, index_head_dim), uint8
    """
    num_pages, buf_numel_per_page = buf.shape

    # Allocate output
    out = torch.empty((seq_len, index_head_dim), dtype=torch.uint8, device=buf.device)

    # Launch kernel with one thread per token
    grid = (seq_len,)
    _get_k_triton_kernel[grid](
        buf,
        page_indices,
        out,
        seq_len,
        page_size,
        buf_numel_per_page,
        index_head_dim,
        BLOCK_SIZE=128,
    )

    return out


@triton.jit
def _get_k_triton_kernel(
    buf_ptr,
    page_indices_ptr,
    out_ptr,
    seq_len: tl.constexpr,
    page_size: tl.constexpr,
    buf_numel_per_page: tl.constexpr,
    index_head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each program handles one token (seq_len tokens total).
    Loads 128 bytes from the appropriate page.
    """
    token_id = tl.program_id(0)

    # Calculate which page and offset within page
    page_idx = token_id // page_size
    token_offset_in_page = token_id % page_size

    # Load the page index from page_indices
    page_index = tl.load(page_indices_ptr + page_idx)

    # Calculate source offset in buf
    # buf[page_index, token_offset_in_page * index_head_dim : ...]
    src_base_offset = (
        page_index * buf_numel_per_page + token_offset_in_page * index_head_dim
    )

    # Load 128 bytes (index_head_dim elements)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < index_head_dim
    data = tl.load(buf_ptr + src_base_offset + offsets, mask=mask)

    # Store to output
    dst_offset = token_id * index_head_dim
    tl.store(out_ptr + dst_offset + offsets, data, mask=mask)


def _get_s_triton(
    buf: torch.Tensor,
    page_indices: torch.Tensor,
    seq_len: int,
    page_size: int,
    index_head_dim: int,
):
    """
    Gather S (scale) data from paged buffer using Triton.

    :param buf: (num_pages, page_size * 128 + page_size * 4), uint8
    :param page_indices: (num_pages,), int32/int64
    :param seq_len: int, number of tokens to gather
    :param page_size: int, typically 64
    :param index_head_dim: int, typically 128
    :return: (seq_len, 4), uint8 (representing fp32 scale)
    """
    num_pages, buf_numel_per_page = buf.shape
    s_offset_in_page = page_size * index_head_dim  # Scales start after K data

    # Allocate output
    out = torch.empty((seq_len, 4), dtype=torch.uint8, device=buf.device)

    # Launch kernel with one thread per token
    grid = (seq_len,)
    _get_s_triton_kernel[grid](
        buf,
        page_indices,
        out,
        seq_len,
        page_size,
        buf_numel_per_page,
        s_offset_in_page,
    )

    return out


@triton.jit
def _get_s_triton_kernel(
    buf_ptr,
    page_indices_ptr,
    out_ptr,
    seq_len: tl.constexpr,
    page_size: tl.constexpr,
    buf_numel_per_page: tl.constexpr,
    s_offset_in_page: tl.constexpr,
):
    """
    Each program handles one token (seq_len tokens total).
    Loads 4 bytes (fp32 scale) from the appropriate page.
    """
    token_id = tl.program_id(0)

    # Calculate which page and offset within page
    page_idx = token_id // page_size
    token_offset_in_page = token_id % page_size

    # Load the page index from page_indices
    page_index = tl.load(page_indices_ptr + page_idx)

    # Calculate source offset in buf
    # Scales are stored after K data: page_size * index_head_dim offset
    # buf[page_index, s_offset_in_page + token_offset_in_page * 4 : ...]
    src_base_offset = (
        page_index * buf_numel_per_page + s_offset_in_page + token_offset_in_page * 4
    )

    # Load 4 bytes (fp32 scale)
    offsets = tl.arange(0, 4)
    data = tl.load(buf_ptr + src_base_offset + offsets)

    # Store to output
    dst_offset = token_id * 4
    tl.store(out_ptr + dst_offset + offsets, data)


def _get_k_and_s_triton(
    buf: torch.Tensor,
    page_indices: torch.Tensor,
    seq_len: int,
    page_size: int,
    index_head_dim: int,
):
    """
    Fused gather of both K (key) and S (scale) data from paged buffer using Triton.
    This is more efficient than calling GetK and GetS separately.

    :param buf: (num_pages, page_size * 128 + page_size * 4), uint8
    :param page_indices: (num_pages,), int32/int64
    :param seq_len: int, number of tokens to gather
    :param page_size: int, typically 64
    :param index_head_dim: int, typically 128
    :return: tuple of (k_out, s_out) where
             k_out: (seq_len, index_head_dim), uint8
             s_out: (seq_len, 4), uint8
    """
    num_pages, buf_numel_per_page = buf.shape
    s_offset_in_page = page_size * index_head_dim  # Scales start after K data

    # Allocate outputs
    k_out = torch.empty((seq_len, index_head_dim), dtype=torch.uint8, device=buf.device)
    s_out = torch.empty((seq_len, 4), dtype=torch.uint8, device=buf.device)

    # Launch kernel with one thread per token
    grid = (seq_len,)
    _get_k_and_s_triton_kernel[grid](
        buf,
        page_indices,
        k_out,
        s_out,
        seq_len,
        page_size,
        buf_numel_per_page,
        index_head_dim,
        s_offset_in_page,
        BLOCK_SIZE_K=128,
    )

    return k_out, s_out


@triton.jit
def _get_k_and_s_triton_kernel(
    buf_ptr,
    page_indices_ptr,
    k_out_ptr,
    s_out_ptr,
    seq_len: tl.constexpr,
    page_size: tl.constexpr,
    buf_numel_per_page: tl.constexpr,
    index_head_dim: tl.constexpr,
    s_offset_in_page: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused kernel that gathers both K and S data in a single pass.
    Each program handles one token (seq_len tokens total).
    Loads 128 bytes (K) + 4 bytes (S) from the appropriate page.
    """
    token_id = tl.program_id(0)

    # Calculate which page and offset within page
    page_idx = token_id // page_size
    token_offset_in_page = token_id % page_size

    # Load the page index from page_indices
    page_index = tl.load(page_indices_ptr + page_idx)

    # ===== Load K data (128 bytes) =====
    # Calculate source offset for K in buf
    k_src_base_offset = (
        page_index * buf_numel_per_page + token_offset_in_page * index_head_dim
    )

    # Load 128 bytes (index_head_dim elements)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    k_mask = k_offsets < index_head_dim
    k_data = tl.load(buf_ptr + k_src_base_offset + k_offsets, mask=k_mask)

    # Store K to output
    k_dst_offset = token_id * index_head_dim
    tl.store(k_out_ptr + k_dst_offset + k_offsets, k_data, mask=k_mask)

    # ===== Load S data (4 bytes) =====
    # Calculate source offset for S in buf
    s_src_base_offset = (
        page_index * buf_numel_per_page + s_offset_in_page + token_offset_in_page * 4
    )

    # Load 4 bytes (fp32 scale)
    s_offsets = tl.arange(0, 4)
    s_data = tl.load(buf_ptr + s_src_base_offset + s_offsets)

    # Store S to output
    s_dst_offset = token_id * 4
    tl.store(s_out_ptr + s_dst_offset + s_offsets, s_data)
