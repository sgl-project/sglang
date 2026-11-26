from types import SimpleNamespace

import torch

from sglang.srt.speculative.dfly_utils import VLDFlyRequestState
from sglang.srt.speculative.dfly_worker_v2 import VLDFlyExecution


def dfly_plan_fixture(
    *,
    device,
    physical,
    window,
    width,
    mrope=True,
    producers=2,
    hidden_dtype=torch.float32,
):
    capacity, context, topk = 5, 160, 7
    state = VLDFlyRequestState(
        num_rows=6,
        hidden_size=13,
        hidden_dtype=hidden_dtype,
        num_producers=producers,
        topk=topk,
        device=device,
    )
    state.hidden.copy_(torch.arange(78, device=device).view(6, 13) / 8)
    state.mrope_delta.copy_(torch.tensor([0, 7, -2, 11, 4, 3], device=device))
    positions = torch.arange(context, device=device)
    # A permutation prevents logical comparisons from passing as physical membership.
    table = (
        ((positions * 53 + 17) % context)[None, :]
        + torch.arange(6, device=device)[:, None] * context
        + 1
    )
    table = table.to(torch.int32)
    history = torch.tensor([0, 1, 2, 3, 5, 6, -1], device=device)
    for plane in range(producers):
        logical = history.roll(plane)
        state.topk[plane].copy_(
            torch.where(logical[None, :] >= 0, table[:, logical.clamp(min=0)], -1)
            if physical
            else logical.expand(6, -1)
        )
    spec = SimpleNamespace(
        query_width=width,
        mask_token_id=99,
        context_local_window=window,
        draft_index_topk=((topk + window + width + 127) // 128) * 128,
    )
    execution = VLDFlyExecution(
        draft_model=SimpleNamespace(spec=spec, is_mrope_enabled=mrope),
        state=state,
        draft_topk_planes=(0, 0) if producers == 1 else (1, 0, 1),
        physical_ids=physical,
        max_context_len=context,
        req_to_token=table,
        runner_ids=None,
        block_spec_info=None,
        max_requests=capacity,
        device=device,
    )
    inputs = dict(
        req_pool_indices=torch.tensor(
            [3, 1, 2, 0, 4], dtype=torch.int64, device=device
        ),
        anchors=torch.tensor(
            [7, 1, context - width + 1, 0, context - width],
            dtype=torch.int32,
            device=device,
        ),
        bonus_tokens=torch.arange(10, 20, device=device)[::2],
        live=torch.tensor([True, True, True, False, True], device=device),
        verify_slots=torch.arange(501, 501 + capacity * width, device=device).view(
            capacity, width
        ),
    )
    return execution, inputs


def assert_dfly_plan_equal(actual, expected):
    for field in (
        "input_ids",
        "positions",
        "verify_positions",
        "write_slots",
        "anchors",
        "active",
    ):
        got, want = getattr(actual.block, field), getattr(expected.block, field)
        torch.testing.assert_close(got, want, rtol=0, atol=0)
        assert got.stride() == want.stride()
    torch.testing.assert_close(
        actual.target_hidden, expected.target_hidden, rtol=0, atol=0
    )
    assert actual.target_hidden.stride() == expected.target_hidden.stride()
    for got, want in zip(actual.topk_indices, expected.topk_indices, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
        assert got.stride() == want.stride()
    if expected.mrope_positions is None:
        assert actual.mrope_positions is None
    else:
        torch.testing.assert_close(
            actual.mrope_positions, expected.mrope_positions, rtol=0, atol=0
        )
        assert actual.mrope_positions.stride(1) == 1
