import torch
import triton
from sglang.third_party.triton_kernels.topk_details._topk_forward import _topk_forward
from sglang.third_party.triton_kernels.topk_details._topk_backward import _topk_backward
from sglang.third_party.triton_kernels.tensor import SparseMatrix, Tensor
from sglang.third_party.triton_kernels.tensor import Bitmatrix, BIT
from typing import Optional, Union
from sglang.third_party.triton_kernels.distributed import symm_mem_pool
import torch.distributed as dist


def make_empty(offset, shape, dtype, device, all_gather):
    if all_gather:
        rank_id = dist.get_rank()
        ret_bufs = symm_mem_pool.make_empty(shape=shape, dtype=dtype, region="topk", region_offset=offset)
        ret = ret_bufs[rank_id]
        offset = symm_mem_pool.align_up(offset + ret.numel() * ret.element_size(),
                                        symm_mem_pool.regions["topk"].alignment)
        return ret_bufs, ret, offset
    ret = torch.empty(shape, dtype=dtype, device=device)
    return (ret, ), ret, 0


def topk_forward(x, k, apply_softmax=True, dim=1, y_indx=None, n_rows=None, all_gather=False):
    if not isinstance(x, Tensor):
        x_shape = [x.shape[0] if n_rows is None else n_rows, x.shape[1]]
        x_shape_max = [x.shape[0], x.shape[1]]
        x = Tensor(x, shape=x_shape, shape_max=x_shape_max)
    cdiv = lambda a, b: (a + b - 1) // b
    BLOCK_M = 32
    BLOCK_N = 32
    use_provided_indx = y_indx is not None
    assert len(x.shape) == 2
    assert x.shape_max[-1] < 32768
    assert dim == 1
    n_rows, n_cols = x.shape
    n_rows_max, _ = x.shape_max
    dev = x.device
    n_rows_out_max = n_rows_max * dist.get_world_size() if all_gather else n_rows_max
    # scratchpad tensors
    # NOTE: these are not returned
    y_vals_bufs, y_vals, offset = make_empty(0, (n_rows_out_max, k), x.dtype, dev, all_gather=all_gather)
    if y_indx is None:
        y_indx_bufs, y_indx, offset = make_empty(offset, (n_rows_out_max, k), torch.int16, dev, all_gather=all_gather)
    else:
        y_indx_bufs = (y_indx, )
    # create bitmatrix in transposed memory layout:
    n_cols_pad = cdiv(n_cols, BLOCK_N) * BLOCK_N
    n_cols_words = n_cols_pad // 32
    bitmatrix_bufs, bitmatrix_data, offset = make_empty(offset, (n_cols_words, cdiv(n_rows_out_max, 32) * 32),
                                                        torch.uint32, dev, all_gather=all_gather)
    bitmatrix_data = torch.transpose(bitmatrix_data, 0, 1)[:n_rows_max]
    pids = cdiv(n_rows_max, BLOCK_M)
    _topk_forward[(pids, )](
        x, x.stride(0),  # inputs
        y_vals_bufs, y_indx_bufs, y_vals.stride(0), use_provided_indx,  # output [topk]
        bitmatrix_bufs, bitmatrix_data.stride(0), bitmatrix_data.stride(1),  # output [bitmatrix]
        n_rows, n_cols,  # shapes
        dist.get_rank() * n_rows_max if all_gather else 0, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,  # tunable parameter
        APPLY_SOFTMAX=apply_softmax, N_EXPTS_PAD=n_cols_pad, N_EXPTS_ACT=k,  # constants
    )
    if all_gather:
        symm_mem_pool.hdl.barrier(channel=0)
    bitmatrix_shape = [n_rows * dist.get_world_size() if all_gather else n_rows, n_cols]
    bitmatrix_shape_max = [n_rows_out_max, None]
    bitmatrix = Bitmatrix(bitmatrix_data, dtype=BIT, shape=bitmatrix_shape, shape_max=bitmatrix_shape_max)
    return y_vals, y_indx, bitmatrix


def topk_backward(x, y_indx, dy_vals, k, n_rows, apply_softmax):
    assert dy_vals.shape[-1] == k
    n_expts_pad = triton.next_power_of_2(x.shape[-1])
    dx = torch.empty_like(x)
    _topk_backward[(dy_vals.shape[0], )](
        y_indx, y_indx.stride(0), dy_vals, dy_vals.stride(0), x, x.stride(0),  # inputs
        dx,  # outputs
        dx.stride(0), x.shape[0], n_rows, x.shape[-1], APPLY_SOFTMAX=apply_softmax, N_EXPTS_ACT=k,
        N_EXPTS_PAD=n_expts_pad)
    return dx


class TopK(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, k, apply_softmax, dim, y_indx, n_rows, all_gather):
        y_vals, y_indx, bitmatrix = topk_forward(x, k, apply_softmax, dim, y_indx, n_rows, all_gather)
        ctx.save_for_backward(x, y_indx)
        ctx.apply_softmax = apply_softmax
        ctx.k = k
        ctx.n_rows = n_rows
        return y_vals, y_indx, bitmatrix

    @staticmethod
    def backward(ctx, dy_vals, _0, _1):
        x, y_indx = ctx.saved_tensors
        dx = topk_backward(x, y_indx, dy_vals, ctx.k, ctx.n_rows, ctx.apply_softmax)
        return dx, None, None, None, None, None, None, None


def topk(
    x: Union[Tensor, torch.Tensor],
    k: int,
    apply_softmax: bool = True,
    dim: int = 1,
    y_indx: Optional[torch.Tensor] = None,
    n_rows: Optional[int] = None,
    all_gather: bool = False,
):
    """
    Computes the top-k values and indices along a specified dimension of a tensor.
    Note that the input can be either a `Tensor` or a `torch.Tensor`, but the output will always be a `torch.Tensor`.

    Parameters
    ----------
    x : Union[triton_kernels.Tensor, torch.Tensor]
        Input tensor of shape (n_tokens, n_expts).
    k : int
        Number of top elements to retrieve.
    apply_softmax : bool, default True
        Whether to apply softmax to the input tensor before computing top-k.
    dim : int, default 1
        Dimension along which to compute top-k.
    y_indx : torch.Tensor, optional
        Pre-allocated tensor for storing indices of top-k elements with shape (n_tokens, k).
        If provided, we skip the computation of top-k indices and use this tensor instead.
    n_rows : int, optional
        Number of rows to apply top-k on. If None, we consider all rows in `x`.

    Returns
    -------
    SparseMatrix: sparse matrix equal to `x` with non-selected entries set to 0
    """
    y_vals, y_indx, bitmatrix = TopK.apply(x, k, apply_softmax, dim, y_indx, n_rows, all_gather)
    return SparseMatrix(vals=y_vals, indx=y_indx, mask=bitmatrix)


def topk_torch(
    x,
    k,
    apply_softmax: bool = True,
    dim: int = 1,
    y_indx: Optional[torch.Tensor] = None,
    n_rows: Optional[int] = None,
) -> SparseMatrix:
    if n_rows is None:
        n_rows = x.shape[0]
    has_user_provided_indx = y_indx is not None
    cdiv = lambda a, b: (a + b - 1) // b
    device = x.device
    assert dim == 1
    assert not isinstance(x, Tensor)
    if not has_user_provided_indx:
        y_indx = torch.argsort(-x, dim=1, stable=True)[:, :k]
    y_indx = y_indx.long()
    y_vals = torch.take_along_dim(x[:n_rows, :], y_indx[:n_rows, :], dim=1)
    y_vals = torch.cat([y_vals, x[n_rows:, :k]], dim=0)
    y_indx = y_indx.int()
    # compute bitmatrix
    _, n_cols = x.shape
    bitmatrix_data = torch.zeros((cdiv(n_cols, 32), cdiv(x.shape[0], 32) * 32), dtype=torch.int32, device=device)
    bitmatrix_data = torch.transpose(bitmatrix_data, 0, 1)[:x.shape[0]]
    # fill bitmatrix
    if apply_softmax:
        y_vals = torch.softmax(y_vals.float(), dim=-1).to(x.dtype)
    if not has_user_provided_indx:
        y_indx, sort_indices = torch.sort(y_indx, dim=1)
        y_vals = torch.gather(y_vals, 1, sort_indices)
    y_indx[n_rows:, :] = -1
    rows = torch.arange(x.shape[0], device=device).unsqueeze(1).expand(-1, y_indx.shape[1]).reshape(-1)
    cols = y_indx.reshape(-1)  # 64-bit safe for div/mod
    word_idx = torch.div(cols, 32, rounding_mode='floor')
    bit_idx = cols % 32
    masks = torch.ones_like(bit_idx) << bit_idx
    bitmatrix_data.index_put_((rows, word_idx), masks, accumulate=True)
    bitmatrix_data = bitmatrix_data.view(torch.uint32)
    bitmatrix = Bitmatrix(bitmatrix_data, shape=x.shape, dtype=BIT)
    return SparseMatrix(vals=y_vals, indx=y_indx, mask=bitmatrix)
