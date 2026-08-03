from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout


@dataclass(frozen=True)
class EagleRaggedVerifyWindow:
    input_ids: torch.Tensor
    positions: torch.Tensor
    out_cache_loc: torch.Tensor
    query_layout: RaggedVerifyLayout


def _pad_eagle_query_lens_torch(
    *,
    verify_lens: torch.Tensor,
    graph_num_tokens: int,
    padded_bs: int,
    draft_token_num: int,
) -> torch.Tensor:
    real_bs = verify_lens.numel()
    padded = torch.cat(
        [
            verify_lens.to(torch.int32),
            torch.zeros(
                padded_bs - real_bs,
                dtype=torch.int32,
                device=verify_lens.device,
            ),
        ]
    )
    leftover = (
        torch.as_tensor(
            graph_num_tokens,
            dtype=torch.int64,
            device=verify_lens.device,
        )
        - padded.to(torch.int64).sum()
    )

    # Fill one row per available request slot at a time. The fixed EAGLE tree
    # width bounds both the loop and every padded query length.
    for _ in range(draft_token_num):
        has_capacity = padded < draft_token_num
        capacity_rank = torch.cumsum(has_capacity.to(torch.int64), dim=0) - 1
        capacity_count = has_capacity.to(torch.int64).sum()
        rows_to_add = torch.minimum(leftover, capacity_count)
        padded.add_((has_capacity & (capacity_rank < rows_to_add)).to(torch.int32))
        leftover.sub_(rows_to_add)
    return padded


@triton.jit
def _pad_eagle_query_lens_kernel(
    verify_lens_ptr,
    padded_lens_ptr,
    real_bs,
    padded_bs,
    graph_num_tokens,
    DRAFT_TOKEN_NUM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.arange(0, BLOCK)
    is_slot = request < padded_bs
    is_real = request < real_bs
    value = tl.load(verify_lens_ptr + request, mask=is_real, other=0).to(tl.int64)
    leftover = graph_num_tokens - tl.sum(value)

    for _ in range(DRAFT_TOKEN_NUM):
        has_capacity = is_slot & (value < DRAFT_TOKEN_NUM)
        capacity_rank = tl.cumsum(has_capacity.to(tl.int32), axis=0) - 1
        capacity_count = tl.sum(has_capacity.to(tl.int32))
        rows_to_add = tl.minimum(leftover, capacity_count)
        value += (has_capacity & (capacity_rank < rows_to_add)).to(tl.int64)
        leftover -= rows_to_add

    tl.store(padded_lens_ptr + request, value.to(tl.int32), mask=is_slot)


def _build_eagle_query_layout(
    *,
    layout: RaggedVerifyLayout,
    padded_bs: int,
    draft_token_num: int,
) -> RaggedVerifyLayout:
    if draft_token_num < 1:
        raise ValueError(f"draft_token_num must be positive, got {draft_token_num}")
    if layout.verify_lens.ndim != 1:
        raise ValueError(
            "EAGLE query lengths must be a rank-1 tensor, got "
            f"shape {tuple(layout.verify_lens.shape)}"
        )
    if padded_bs < layout.bs:
        raise ValueError(
            f"padded_bs {padded_bs} is smaller than real batch size {layout.bs}"
        )
    if layout.graph_num_tokens < 0:
        raise ValueError(
            f"graph token bucket must be non-negative, got {layout.graph_num_tokens}"
        )
    if layout.graph_num_tokens > padded_bs * draft_token_num:
        raise ValueError(
            f"graph token bucket {layout.graph_num_tokens} exceeds EAGLE "
            f"capacity {padded_bs} * {draft_token_num}"
        )

    # CUDA callers validate these inputs asynchronously. Repeat the complete
    # check for host-side helpers without synchronizing the serving hot path.
    if not layout.verify_lens.is_cuda:
        verify_lens_cpu = [int(value) for value in layout.verify_lens.tolist()]
        if any(value < 1 or value > draft_token_num for value in verify_lens_cpu):
            raise ValueError(
                "EAGLE query lengths must be in "
                f"[1, {draft_token_num}], got {verify_lens_cpu}"
            )
        if sum(verify_lens_cpu) > layout.graph_num_tokens:
            raise ValueError(
                f"real EAGLE query rows {sum(verify_lens_cpu)} exceed graph "
                f"token bucket {layout.graph_num_tokens}"
            )

    if layout.verify_lens.is_cuda:
        verify_lens = layout.verify_lens.to(torch.int32).contiguous()
        padded_lens = torch.empty(
            padded_bs,
            dtype=torch.int32,
            device=verify_lens.device,
        )
        block = triton.next_power_of_2(max(padded_bs, 1))
        _pad_eagle_query_lens_kernel[(1,)](
            verify_lens,
            padded_lens,
            layout.bs,
            padded_bs,
            layout.graph_num_tokens,
            DRAFT_TOKEN_NUM=draft_token_num,
            BLOCK=block,
        )
    else:
        padded_lens = _pad_eagle_query_lens_torch(
            verify_lens=layout.verify_lens,
            graph_num_tokens=layout.graph_num_tokens,
            padded_bs=padded_bs,
            draft_token_num=draft_token_num,
        )

    return RaggedVerifyLayout.from_verify_lens_device(
        verify_lens=padded_lens,
        graph_num_tokens=layout.graph_num_tokens,
    )


@triton.jit
def _compact_eagle_verify_inputs_kernel(
    draft_tokens_ptr,
    positions_ptr,
    cache_locs_ptr,
    query_lens_ptr,
    query_starts_ptr,
    compact_tokens_ptr,
    compact_positions_ptr,
    compact_cache_locs_ptr,
    real_bs,
    DRAFT_TOKEN_NUM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    within = tl.arange(0, BLOCK)
    query_len = tl.load(query_lens_ptr + request)
    query_start = tl.load(query_starts_ptr + request).to(tl.int64)
    in_query = within < query_len
    is_real = request < real_bs
    source = request * DRAFT_TOKEN_NUM + within
    source_mask = in_query & is_real & (within < DRAFT_TOKEN_NUM)

    token = tl.load(draft_tokens_ptr + source, mask=source_mask, other=0)
    position = tl.load(positions_ptr + source, mask=source_mask, other=0)
    cache_loc = tl.load(cache_locs_ptr + source, mask=source_mask, other=0)
    destination = query_start + within

    tl.store(compact_tokens_ptr + destination, token, mask=in_query)
    tl.store(compact_positions_ptr + destination, position, mask=in_query)
    tl.store(compact_cache_locs_ptr + destination, cache_loc, mask=in_query)


def _compact_eagle_verify_inputs_torch(
    *,
    draft_tokens: torch.Tensor,
    positions: torch.Tensor,
    out_cache_loc: torch.Tensor,
    query_layout: RaggedVerifyLayout,
    real_bs: int,
    draft_token_num: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = tuple(
        torch.zeros(
            query_layout.graph_num_tokens,
            dtype=dense.dtype,
            device=dense.device,
        )
        for dense in (draft_tokens, positions, out_cache_loc)
    )
    for request in range(real_bs):
        query_len = int(query_layout.verify_lens[request])
        query_start = int(query_layout.extend_start_loc[request])
        dense_start = request * draft_token_num
        for dense, compact in zip(
            (draft_tokens, positions, out_cache_loc), outputs, strict=True
        ):
            compact[query_start : query_start + query_len].copy_(
                dense[dense_start : dense_start + query_len]
            )
    return outputs


def _compact_eagle_verify_inputs(
    *,
    draft_tokens: torch.Tensor,
    positions: torch.Tensor,
    out_cache_loc: torch.Tensor,
    query_layout: RaggedVerifyLayout,
    real_bs: int,
    draft_token_num: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dense_inputs = (draft_tokens, positions, out_cache_loc)
    if draft_token_num < 1:
        raise ValueError(f"draft_token_num must be positive, got {draft_token_num}")
    if not 0 <= real_bs <= query_layout.bs:
        raise ValueError(
            f"real_bs {real_bs} must be within query batch size {query_layout.bs}"
        )
    if any(tensor.ndim != 1 for tensor in dense_inputs):
        raise ValueError("EAGLE verify input compaction expects rank-1 tensors")
    if any(tensor.device != draft_tokens.device for tensor in dense_inputs):
        raise ValueError("EAGLE verify inputs must be on the same device")
    if (
        query_layout.verify_lens.device != draft_tokens.device
        or query_layout.extend_start_loc.device != draft_tokens.device
    ):
        raise ValueError("EAGLE verify inputs and query layout must share a device")
    required_dense_rows = real_bs * draft_token_num
    if any(tensor.numel() < required_dense_rows for tensor in dense_inputs):
        raise ValueError(
            "EAGLE verify inputs are shorter than the real dense tree: "
            f"need {required_dense_rows} rows"
        )

    if not draft_tokens.is_cuda:
        return _compact_eagle_verify_inputs_torch(
            draft_tokens=draft_tokens,
            positions=positions,
            out_cache_loc=out_cache_loc,
            query_layout=query_layout,
            real_bs=real_bs,
            draft_token_num=draft_token_num,
        )

    draft_tokens, positions, out_cache_loc = (
        tensor.contiguous() for tensor in dense_inputs
    )
    compact_tokens = torch.empty(
        query_layout.graph_num_tokens,
        dtype=draft_tokens.dtype,
        device=draft_tokens.device,
    )
    compact_positions = torch.empty(
        query_layout.graph_num_tokens,
        dtype=positions.dtype,
        device=positions.device,
    )
    compact_cache_locs = torch.empty(
        query_layout.graph_num_tokens,
        dtype=out_cache_loc.dtype,
        device=out_cache_loc.device,
    )
    block = triton.next_power_of_2(draft_token_num)
    _compact_eagle_verify_inputs_kernel[(query_layout.bs,)](
        draft_tokens,
        positions,
        out_cache_loc,
        query_layout.verify_lens,
        query_layout.extend_start_loc,
        compact_tokens,
        compact_positions,
        compact_cache_locs,
        real_bs,
        DRAFT_TOKEN_NUM=draft_token_num,
        BLOCK=block,
    )
    return compact_tokens, compact_positions, compact_cache_locs


@triton.jit
def _restore_eagle_verify_output_kernel(
    compact_ptr,
    compact_starts_ptr,
    verify_lens_ptr,
    dense_ptr,
    dense_width,
    feature_size,
    BLOCK_FEATURE: tl.constexpr,
):
    dense_row = tl.program_id(0).to(tl.int64)
    feature_block = tl.program_id(1)
    request = dense_row // dense_width
    within = dense_row - request * dense_width
    verify_len = tl.load(verify_lens_ptr + request)
    compact_start = tl.load(compact_starts_ptr + request).to(tl.int64)

    feature = feature_block * BLOCK_FEATURE + tl.arange(0, BLOCK_FEATURE)
    valid_feature = feature < feature_size
    retained = within < verify_len
    compact_row = compact_start + within
    value = tl.load(
        compact_ptr + compact_row * feature_size + feature,
        mask=retained & valid_feature,
        other=0.0,
    )
    tl.store(
        dense_ptr + dense_row * feature_size + feature,
        value,
        mask=valid_feature,
    )


def _restore_eagle_verify_output_torch(
    *,
    compact: torch.Tensor,
    compact_starts: torch.Tensor,
    verify_lens: torch.Tensor,
    draft_token_num: int,
) -> torch.Tensor:
    batch_size = verify_lens.numel()
    dense = torch.zeros(
        (batch_size * draft_token_num, compact.shape[1]),
        dtype=compact.dtype,
        device=compact.device,
    )
    for request in range(batch_size):
        verify_len = int(verify_lens[request])
        compact_start = int(compact_starts[request])
        dense_start = request * draft_token_num
        dense[dense_start : dense_start + verify_len].copy_(
            compact[compact_start : compact_start + verify_len]
        )
    return dense


def _restore_eagle_verify_output(
    *,
    compact: torch.Tensor,
    compact_starts: torch.Tensor,
    verify_lens: torch.Tensor,
    draft_token_num: int,
) -> torch.Tensor:
    if draft_token_num < 1:
        raise ValueError(f"draft_token_num must be positive, got {draft_token_num}")
    if compact.ndim != 2:
        raise ValueError("EAGLE verify output restoration expects a rank-2 tensor")
    if compact_starts.ndim != 1 or verify_lens.ndim != 1:
        raise ValueError("EAGLE restore starts and query lengths must be rank-1")
    if compact_starts.numel() < verify_lens.numel():
        raise ValueError("EAGLE restore needs one compact start for every real request")
    if not verify_lens.is_cuda:
        verify_lens_cpu = [int(value) for value in verify_lens.tolist()]
        compact_starts_cpu = [
            int(value) for value in compact_starts[: verify_lens.numel()].tolist()
        ]
        if any(value < 1 or value > draft_token_num for value in verify_lens_cpu):
            raise ValueError(
                "EAGLE restore lengths must be in "
                f"[1, {draft_token_num}], got {verify_lens_cpu}"
            )
        if any(
            start < 0 or start + length > compact.shape[0]
            for start, length in zip(compact_starts_cpu, verify_lens_cpu, strict=True)
        ):
            raise ValueError("EAGLE restore range exceeds the compact output")
    if not compact.is_cuda:
        return _restore_eagle_verify_output_torch(
            compact=compact,
            compact_starts=compact_starts,
            verify_lens=verify_lens,
            draft_token_num=draft_token_num,
        )

    compact = compact.contiguous()
    compact_starts = compact_starts.to(
        device=compact.device, dtype=torch.int64
    ).contiguous()
    verify_lens = verify_lens.to(device=compact.device, dtype=torch.int32).contiguous()
    batch_size = verify_lens.numel()
    feature_size = compact.shape[1]
    dense = torch.empty(
        (batch_size * draft_token_num, feature_size),
        dtype=compact.dtype,
        device=compact.device,
    )
    block_feature = 1024
    grid = (
        batch_size * draft_token_num,
        triton.cdiv(feature_size, block_feature),
    )
    _restore_eagle_verify_output_kernel[grid](
        compact,
        compact_starts,
        verify_lens,
        dense,
        draft_token_num,
        feature_size,
        BLOCK_FEATURE=block_feature,
    )
    return dense


def build_eagle_ragged_verify_window(
    *,
    draft_tokens: torch.Tensor,
    positions: torch.Tensor,
    out_cache_loc: torch.Tensor,
    layout: RaggedVerifyLayout,
    draft_token_num: int,
    padded_bs: int,
) -> EagleRaggedVerifyWindow:
    """Build EAGLE inputs for the selected target-verify token tier."""
    real_bs = layout.bs
    if real_bs < 1:
        raise ValueError("EAGLE ragged verify requires a non-empty batch")
    if padded_bs < real_bs:
        raise ValueError(
            f"padded_bs {padded_bs} is smaller than real batch size {real_bs}"
        )

    padded_layout = _build_eagle_query_layout(
        layout=layout,
        padded_bs=padded_bs,
        draft_token_num=draft_token_num,
    )
    compact_tokens, compact_positions, compact_cache_locs = (
        _compact_eagle_verify_inputs(
            draft_tokens=draft_tokens,
            positions=positions,
            out_cache_loc=out_cache_loc,
            query_layout=padded_layout,
            real_bs=real_bs,
            draft_token_num=draft_token_num,
        )
    )

    return EagleRaggedVerifyWindow(
        input_ids=compact_tokens,
        positions=compact_positions,
        out_cache_loc=compact_cache_locs,
        query_layout=padded_layout,
    )


def apply_eagle_retrieval_layout(
    *,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    verify_lens: torch.Tensor,
) -> None:
    """Update EAGLE retrieval metadata for the selected layout."""
    if retrieve_index.ndim != 2:
        raise ValueError(f"retrieve tensors must be rank 2, got {retrieve_index.ndim}")
    if verify_lens.shape != (retrieve_index.shape[0],):
        raise ValueError(
            f"verify_lens shape {tuple(verify_lens.shape)} does not match "
            f"retrieve batch size {retrieve_index.shape[0]}"
        )

    width = retrieve_index.shape[1]
    lens = verify_lens.to(
        device=retrieve_index.device, dtype=retrieve_next_token.dtype
    ).unsqueeze(1)
    slots = torch.arange(width, device=retrieve_index.device).unsqueeze(0)
    retained_source = slots < lens
    retrieve_index.masked_fill_(~retained_source, -1)

    for links in (retrieve_next_token, retrieve_next_sibling):
        retained_edge = retained_source & (links >= 0) & (links < lens)
        links.copy_(torch.where(retained_edge, links, torch.full_like(links, -1)))


def scatter_eagle_verify_output(
    *,
    compact: torch.Tensor,
    layout: RaggedVerifyLayout,
    query_layout: RaggedVerifyLayout | None = None,
    draft_token_num: int,
) -> torch.Tensor:
    compact_starts = (
        query_layout.extend_start_loc
        if query_layout is not None
        else layout.extend_start_loc
    )
    return _restore_eagle_verify_output(
        compact=compact,
        compact_starts=compact_starts,
        verify_lens=layout.verify_lens,
        draft_token_num=draft_token_num,
    )
