# SPDX-License-Identifier: Apache-2.0
"""VA 24487f4f48 mask kernels; SG retains512 complete groups plus tail."""

import torch
import triton
import triton.language as tl

QSA_FIA_MASK_WORD_TILE = 4096
_QSA_FIA_MASK_WORKSPACES: dict[str, "QSAFIAMaskWorkspace"] = {}


class QSAFIAMaskWorkspace:
    """Process-local FIA mask storage shared by every QSA layer and step."""

    def __init__(self, device: torch.device, max_rows: int, max_bytes: int, max_update_width: int) -> None:
        self.device = device
        self.max_rows = max_rows
        self.max_update_width = max_update_width
        self.mask_word_capacity = max_bytes // torch.int32.itemsize
        self.word_capacity = self.mask_word_capacity + max_rows * max_update_width
        self.storage = torch.empty((self.word_capacity,), dtype=torch.int32, device=device)
        self.words = torch.empty((self.mask_word_capacity,), dtype=torch.int32, device=device)
        self.indices = torch.empty((max_rows, max_update_width), dtype=torch.int64, device=device)
        self.updates = torch.empty((max_rows, max_update_width), dtype=torch.int32, device=device)
        self.selected = torch.empty((max_rows,), dtype=torch.bool, device=device)

    def views(
        self, rows: int, mask_words: int, update_width: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_size = rows * mask_words
        word_stride = mask_words + update_width
        storage_size = rows * word_stride
        if (
            rows > self.max_rows
            or mask_size > self.mask_word_capacity
            or storage_size > self.word_capacity
            or update_width > self.max_update_width
        ):
            raise ValueError(
                f"QSA FIA mask workspace cannot cover rows={rows}, words={mask_words}, updates={update_width}"
            )
        storage = self.storage[:storage_size].view(rows, word_stride)
        words = self.words[:mask_size].view(rows, mask_words)
        return (
            storage,
            words,
            self.indices[:rows, :update_width],
            self.updates[:rows, :update_width],
            self.selected[:rows],
        )


def get_qsa_fia_mask_workspace(
    device: torch.device, max_rows: int, max_bytes: int, max_update_width: int
) -> QSAFIAMaskWorkspace:
    """Allocate the process-local dense mask buffer once, then reuse it."""
    key = str(device)
    workspace = _QSA_FIA_MASK_WORKSPACES.get(key)
    if workspace is None:
        workspace = QSAFIAMaskWorkspace(device, max_rows, max_bytes, max_update_width)
        _QSA_FIA_MASK_WORKSPACES[key] = workspace
    elif (
        workspace.max_rows < max_rows
        or workspace.mask_word_capacity * torch.int32.itemsize < max_bytes
        or workspace.max_update_width < max_update_width
    ):
        raise ValueError("QSA FIA mask workspace was initialized with insufficient capacity")
    return workspace


@triton.jit(do_not_specialize=["mask_words"])
def _initialize_qsa_mask_words_kernel(
    words,
    mask_words,
    word_stride,
    WORD_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    for word_start in range(0, mask_words, WORD_BLOCK):
        word = word_start + tl.arange(0, WORD_BLOCK)
        tl.store(words + row * word_stride + word, 0x01010101, word < mask_words)


@triton.jit(do_not_specialize=["kv_length", "mask_words"])
def _prepare_qsa_mask_words_kernel(
    compact_indices,
    visible_lens,
    scatter_indices,
    updates,
    has_selection,
    stride_row,
    stride_column,
    scatter_stride_row,
    updates_stride_row,
    kv_length,
    mask_words,
    update_width,
    GROUPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    visible = tl.minimum(tl.maximum(tl.load(visible_lens + row), 0), kv_length)
    group = tl.load(compact_indices + row * stride_row + columns * stride_column, columns < GROUPS, other=-1)
    valid = (columns < GROUPS) & (group >= 0) & (group < visible // 4)
    residual = visible % 4
    valid_count = tl.sum(valid.to(tl.int32), 0)
    # SG retains all 512 complete groups plus the incomplete tail.
    selected = (valid_count > 0) | (residual != 0)
    # Each complete four-token group is one aligned int32 mask word. Keep
    # row-local targets so ScatterElementsV2 can use its efficient dim=1 path.
    target = tl.where(valid, group, mask_words + columns)
    tail = (columns == GROUPS) & (residual != 0)
    tail_value = (0x01010101 << (residual * 8)).to(tl.int32)
    target = tl.where(tail, visible // 4, target)
    value = tl.where(tail, tail_value, 0)
    # An empty row attends token zero and is explicitly zeroed after FIA.
    dummy = (columns == 0) & (~selected)
    target = tl.where(dummy, 0, target)
    value = tl.where(dummy, 0x01010100, value)
    store_mask = columns < update_width
    tl.store(scatter_indices + row * scatter_stride_row + columns, target.to(tl.int64), store_mask)
    tl.store(updates + row * updates_stride_row + columns, value, store_mask)
    tl.store(has_selection + row, selected)


@triton.jit
def _copy_qsa_fia_output_kernel(
    result,
    selected,
    output,
    stride_row,
    stride_head,
    stride_dim,
    HEAD_DIM: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    CONTIGUOUS: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(result + row * WIDTH + columns, columns < WIDTH, other=0)
    valid = tl.load(selected + row)
    if CONTIGUOUS:
        offset = row * WIDTH + columns
    else:
        offset = row * stride_row + columns // HEAD_DIM * stride_head + columns % HEAD_DIM * stride_dim
    tl.store(output + offset, tl.where(valid, values, 0), columns < WIDTH)


def build_qsa_fia_mask(
    compact_indices: torch.Tensor,
    visible_lens: torch.Tensor,
    kv_length: int,
    mask_width: int,
    workspace: QSAFIAMaskWorkspace,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fill and return a reusable [T, KV] mask plus nonempty-row flags."""
    rows = compact_indices.shape[0]
    # Dense DMA stores must not share a 32-byte block with another row.
    mask_words = triton.cdiv(mask_width, 32) * 8
    groups = compact_indices.shape[1]
    update_width = triton.cdiv(groups + 1, 8) * 8
    storage, words, indices, updates, selected = workspace.views(rows, mask_words, update_width)
    word_stride = mask_words + update_width
    word_block = min(QSA_FIA_MASK_WORD_TILE, triton.next_power_of_2(mask_words))
    _initialize_qsa_mask_words_kernel[(rows,)](storage, mask_words, word_stride, WORD_BLOCK=word_block)
    _prepare_qsa_mask_words_kernel[(rows,)](
        compact_indices,
        visible_lens,
        indices,
        updates,
        selected,
        compact_indices.stride(0),
        compact_indices.stride(1),
        indices.stride(0),
        updates.stride(0),
        kv_length=kv_length,
        mask_words=mask_words,
        update_width=update_width,
        GROUPS=groups,
        BLOCK=triton.next_power_of_2(update_width),
    )
    storage.scatter_(1, indices, updates)
    words.copy_(storage[:, :mask_words])
    # The NPU's little-endian int32 representation is four bool bytes. This is
    # a dtype view, not a numerical conversion (which would collapse the bytes).
    mask = words.view(torch.bool)
    return mask[:, :mask_width], selected


def copy_qsa_fia_output(result: torch.Tensor, selected: torch.Tensor, output: torch.Tensor) -> None:
    width = output.shape[1] * output.shape[2]
    _copy_qsa_fia_output_kernel[(output.shape[0], triton.cdiv(width, 1024))](
        result,
        selected,
        output,
        *output.stride(),
        HEAD_DIM=output.shape[2],
        WIDTH=width,
        BLOCK=1024,
        CONTIGUOUS=output.is_contiguous(),
    )
