"""Verify scratch rows must match the rows consumed by speculative commit."""

import os
import sys
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
    fused_mamba_state_scatter_with_mask,
)
from sglang.kernels.ops.mamba.qwen4_short_conv import (
    can_fuse_qwen4_verify_conv,
    fused_qwen4_verify_conv,
)
from sglang.srt.layers.attention.linear.utils import (
    select_verify_intermediate_state_indices,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
_STABLE_ROWS = "sglang.srt.layers.attention.linear.utils.pp_spec_stable_rows_enabled"


def _inputs(requests=2, history=2, channels=8):
    steps = 3
    x = torch.arange(
        1, requests * steps * channels + 1, device="cuda", dtype=torch.bfloat16
    ).reshape(requests * steps, channels)
    residual = torch.full_like(x, 0.5)
    weight = torch.ones(channels, 1, history + 1, device="cuda", dtype=x.dtype)
    state = torch.zeros(8, channels, history, device="cuda", dtype=x.dtype)
    slots = torch.tensor([3, 5] + [0] * (requests - 2), device="cuda")[:requests]
    valid = torch.ones(requests * steps, device="cuda", dtype=torch.bool)
    windows = torch.full(
        (7, steps, channels, history), -7, device="cuda", dtype=x.dtype
    )
    return [x, residual, weight, state, slots, valid, steps, 1, windows]


def _reference(args):
    x, residual, weight, state, slots, valid, steps, dilation, _ = args
    channels, history = state.shape[1:]
    joined = torch.cat(
        [state[slots].to(x.dtype), x.reshape(-1, steps, channels).transpose(1, 2)], -1
    )
    conv = F.conv1d(joined, weight, dilation=dilation, groups=channels).float()
    activated = (conv * torch.sigmoid(conv)).to(x.dtype)
    output = (activated.transpose(1, 2).reshape_as(x).float() + residual.float()).to(
        x.dtype
    )
    output = torch.where(valid[:, None], output, 0)
    windows = torch.stack(
        [joined[:, :, step + 1 : step + 1 + history] for step in range(steps)], 1
    )
    windows = torch.where(valid.reshape(-1, steps, 1, 1), windows, 0)
    return output, windows


def _assert_output(actual, expected):
    # Interpreter BF16 stores can differ from CUDA RNE by one ULP. The unchanged
    # finish kernel has the same difference; state/window checks remain exact.
    tolerance = 0.01 if os.environ.get("TRITON_INTERPRET") == "1" else 0
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("case", ["A", "B", "C", "D", "E", "F"])
@pytest.mark.parametrize("row_dtype", [torch.int32, torch.int64])
def test_verify_commit_rows(case, row_dtype):
    # E deliberately has more padded batch rows than scratch rows (pool size + 1).
    args = _inputs(requests=9 if case == "E" else 2)
    x, residual, weight, state, slots, valid, steps, _, scratch = args
    request_rows = torch.tensor([0, 1] if case == "B" else [4, 2], device="cuda")
    accepted = torch.tensor([1, 2], device="cuda")
    if case in ("D", "F"):
        state.copy_(torch.arange(1, 9, device="cuda")[:, None, None] * 10)
        scratch.copy_(-torch.arange(1, 8, device="cuda")[:, None, None, None] * 10)
        accepted.zero_()
    if case == "C":
        order = torch.tensor([1, 0], device="cuda")
        request_rows = request_rows[order]
        accepted = accepted[order]
        args[0] = x.reshape(2, steps, -1)[order].reshape_as(x).contiguous()
        args[4] = slots[order]
    if case == "E":
        request_rows = torch.cat([request_rows, request_rows.new_full((7,), -1)])
        valid[5:] = False  # Partial final live request, followed by padding.
        accepted = torch.tensor([1, 1] + [-1] * 7, device="cuda")
    lengths = valid.reshape(-1, steps).sum(1)
    with patch(_STABLE_ROWS, return_value=case != "B"):
        rows = select_verify_intermediate_state_indices(
            None, request_rows, lambda: lengths.ne(0), pool_size=6
        )
    if rows is not None:
        # Both integer dtypes and strided index tensors are valid metadata.
        storage = torch.empty(rows.numel() * 2, device="cuda", dtype=row_dtype)
        storage[::2] = rows
        rows = storage[::2]
    if case == "F":
        args[-1] = None
    initial_state, initial_scratch = state.clone(), scratch.clone()
    expected_output, expected_windows = _reference(args)
    output = fused_qwen4_verify_conv(*args, intermediate_state_indices=rows)
    _assert_output(output, expected_output)
    torch.testing.assert_close(state, initial_state, rtol=0, atol=0)
    if case == "F":
        torch.testing.assert_close(
            output, fused_qwen4_verify_conv(*args), rtol=0, atol=0
        )
        torch.testing.assert_close(scratch, initial_scratch, rtol=0, atol=0)
        return
    write_rows = (
        torch.arange(len(request_rows), device="cuda") if rows is None else rows.long()
    )
    expected_scratch = initial_scratch.clone()
    # Explicit row mapping must not let padding overwrite a live request.
    for req, row in enumerate(write_rows.tolist()):
        if rows is None:
            expected_scratch[row] = expected_windows[req]
        else:
            live = valid.reshape(-1, steps)[req]
            expected_scratch[row, live] = expected_windows[req, live]
    torch.testing.assert_close(scratch, expected_scratch, rtol=0, atol=0)
    committed = initial_state.clone()[None]
    fused_mamba_state_scatter_with_mask(
        committed, scratch[None], args[4], accepted, request_rows
    )
    expected_state = initial_state.clone()
    for req, step in enumerate(accepted.tolist()):
        if step >= 0:
            expected_state[args[4][req]] = expected_windows[req, step]
    torch.testing.assert_close(committed[0], expected_state, rtol=0, atol=0)
    print(
        f"case={case} rows={write_rows.tolist()} "
        f"observed={committed[0, args[4][:2], 0].tolist()} "
        f"expected={expected_state[args[4][:2], 0].tolist()}"
    )


def test_invalid_scratch_rows_are_masked():
    args = _inputs(requests=4)
    rows = torch.tensor([6, -1, 7, 2**32], device="cuda", dtype=torch.int64)
    expected_output, windows = _reference(args)
    expected_scratch = args[-1].clone()
    expected_scratch[6] = windows[0]
    output = fused_qwen4_verify_conv(*args, intermediate_state_indices=rows)
    _assert_output(output, expected_output)
    torch.testing.assert_close(args[-1], expected_scratch, rtol=0, atol=0)


@pytest.mark.parametrize("intermediate", [True, False])
def test_no_history(intermediate):
    args = _inputs(history=0)
    if not intermediate:
        args[-1] = None
    rows = torch.tensor([4, 2], device="cuda", dtype=torch.int32)
    expected, _ = _reference(args)
    output = fused_qwen4_verify_conv(*args, intermediate_state_indices=rows)
    _assert_output(output, expected)
    assert args[3].numel() == 0


def test_empty_batch():
    args = _inputs(requests=0)
    state, scratch = args[3].clone(), args[-1].clone()
    output = fused_qwen4_verify_conv(
        *args,
        intermediate_state_indices=torch.empty(0, device="cuda", dtype=torch.int32),
    )
    assert output.shape == args[0].shape
    torch.testing.assert_close(args[3], state, rtol=0, atol=0)
    torch.testing.assert_close(args[-1], scratch, rtol=0, atol=0)


def test_row_metadata_validation():
    args = _inputs()
    rows = torch.tensor([4, 2], device="cuda", dtype=torch.int32)
    for invalid in (rows.float(), rows.bool(), rows[:, None], rows[:1], rows.cpu()):
        assert not can_fuse_qwen4_verify_conv(*args, invalid)
        with pytest.raises(ValueError, match="unsupported input"):
            fused_qwen4_verify_conv(*args, invalid)
    # Positional writes still require one scratch row per batch request.
    args[-1] = args[-1][:1]
    assert not can_fuse_qwen4_verify_conv(*args)
    assert can_fuse_qwen4_verify_conv(*args, torch.zeros_like(rows))


def test_stable_rows_disabled_does_not_build_mask():
    def unexpected_mask():
        raise AssertionError("ordinary verify must not launch a row-mask kernel")

    with patch(_STABLE_ROWS, return_value=False):
        assert (
            select_verify_intermediate_state_indices(None, None, unexpected_mask, 6)
            is None
        )


@pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") == "1", reason="requires CUDA JIT"
)
def test_cuda_graph_replay_uses_current_rows():
    args = _inputs()
    request_rows = torch.tensor([4, 2], device="cuda")
    lengths = torch.tensor([3, 3], device="cuda")
    expected_output, expected_windows = _reference(args)

    def run():
        rows = select_verify_intermediate_state_indices(
            None, request_rows, lambda: lengths.ne(0), pool_size=6
        )
        return fused_qwen4_verify_conv(*args, intermediate_state_indices=rows)

    with patch(_STABLE_ROWS, return_value=True):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run()
        for values in ([4, 2], [2, 4]):
            request_rows.copy_(torch.tensor(values, device="cuda"))
            args[-1].fill_(-7)
            graph.replay()
            _assert_output(output, expected_output)
            expected_scratch = torch.full_like(args[-1], -7)
            for req, row in enumerate(values):
                expected_scratch[row] = expected_windows[req]
            torch.testing.assert_close(args[-1], expected_scratch, rtol=0, atol=0)


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_strided_scratch_and_dilated_conv(state_dtype):
    args = _inputs(history=4, channels=257)
    args[2] = torch.ones(257, 1, 3, device="cuda", dtype=args[0].dtype)
    args[3] = args[3].to(state_dtype).fill_(1.00390625)
    args[7] = 2
    backing = torch.full((7, 3, 257, 8), -7, device="cuda", dtype=state_dtype)
    args[-1] = backing[..., ::2]
    rows = torch.tensor([4, 2], device="cuda", dtype=torch.int32)
    expected_output, expected_windows = _reference(args)
    expected_backing = backing.clone()
    expected_backing[4, ..., ::2] = expected_windows[0]
    expected_backing[2, ..., ::2] = expected_windows[1]
    initial_state = args[3].clone()
    output = fused_qwen4_verify_conv(*args, intermediate_state_indices=rows)
    _assert_output(output, expected_output)
    torch.testing.assert_close(backing, expected_backing, rtol=0, atol=0)
    torch.testing.assert_close(args[3], initial_state, rtol=0, atol=0)


def test_last_live_request_row_with_padding():
    # ReqToTokenPool reserves row 0 and allocates live rows 1..size. The helper
    # also maps padding to size, so invalid tokens must not overwrite live data.
    args = _inputs(requests=9)
    args[5][6:] = False
    request_rows = torch.tensor([6, 2] + [0] * 7, device="cuda")
    lengths = args[5].reshape(-1, args[6]).sum(1)
    with patch(_STABLE_ROWS, return_value=True):
        rows = select_verify_intermediate_state_indices(
            None, request_rows, lambda: lengths.ne(0), pool_size=6
        )
    expected_output, expected_windows = _reference(args)
    expected_scratch = args[-1].clone()
    expected_scratch[6] = expected_windows[0]
    expected_scratch[2] = expected_windows[1]
    output = fused_qwen4_verify_conv(*args, intermediate_state_indices=rows)
    _assert_output(output, expected_output)
    torch.testing.assert_close(args[-1], expected_scratch, rtol=0, atol=0)
    committed = args[3].clone()[None]
    accepted = torch.tensor([1, 2] + [-1] * 7, device="cuda")
    fused_mamba_state_scatter_with_mask(
        committed, args[-1][None], args[4], accepted, request_rows
    )
    for req, step in enumerate([1, 2]):
        torch.testing.assert_close(
            committed[0, args[4][req]], expected_windows[req, step], rtol=0, atol=0
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
