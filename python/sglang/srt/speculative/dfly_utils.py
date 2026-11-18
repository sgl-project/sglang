import torch
import triton
import triton.language as tl


@triton.jit
def _build_draft_topk_kernel(
    state_topk,
    req_pool_indices,
    verify_slots,
    out_rows,
    out_counts,
    state_layer_stride,
    state_req_stride,
    out_layer_stride,
    out_row_stride,
    WIDTH: tl.constexpr,
    TARGET_TOPK: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
):
    req_row = tl.program_id(0)
    layer = tl.program_id(1)
    req = tl.load(req_pool_indices + req_row).to(tl.int64)
    cols = tl.arange(0, TARGET_TOPK)
    history = tl.load(
        state_topk + layer * state_layer_stride + req * state_req_stride + cols
    )
    valid = history >= 0
    packed = tl.cumsum(valid.to(tl.int32), axis=0) - 1
    num_valid = tl.sum(valid.to(tl.int32), axis=0)
    block = tl.arange(0, BLOCK_WIDTH)
    slots = tl.load(
        verify_slots + req_row * WIDTH + block, mask=block < WIDTH, other=-1
    ).to(tl.int32)
    for j in tl.static_range(WIDTH):
        row = req_row * WIDTH + j
        base = out_rows + layer * out_layer_stride + row * out_row_stride
        tl.store(base + packed, history, mask=valid)
        tl.store(base + num_valid + block, slots, mask=block <= j)
        if layer == 0:
            tl.store(out_counts + row, num_valid + j + 1)


def build_draft_topk(
    *,
    state_topk: torch.Tensor,
    req_pool_indices: torch.Tensor,
    verify_slots: torch.Tensor,
    out_rows: torch.Tensor,
    out_counts: torch.Tensor,
) -> None:
    bs, width = verify_slots.shape
    layers, _, target_topk = state_topk.shape
    rows = out_rows[:, : bs * width]
    rows.fill_(-1)
    _build_draft_topk_kernel[(bs, layers)](
        state_topk,
        req_pool_indices,
        verify_slots,
        out_rows,
        out_counts,
        state_topk.stride(0),
        state_topk.stride(1),
        out_rows.stride(0),
        out_rows.stride(1),
        WIDTH=width,
        TARGET_TOPK=target_topk,
        BLOCK_WIDTH=triton.next_power_of_2(width),
    )


class DFlyRequestState:
    def __init__(
        self,
        *,
        pool_size: int,
        hidden_size: int,
        num_layers: int,
        target_topk: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.hidden = torch.zeros((pool_size, hidden_size), dtype=dtype, device=device)
        self.topk = torch.full(
            (num_layers, pool_size, target_topk), -1, dtype=torch.int32, device=device
        )

    def write(
        self,
        *,
        req_pool_indices: torch.Tensor,
        hidden: torch.Tensor,
        topk: torch.Tensor,
    ) -> None:
        self.hidden[req_pool_indices] = hidden
        self.topk[:, req_pool_indices] = topk
