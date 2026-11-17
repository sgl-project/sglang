"""NPU PLE convolution output and per-request state updates."""

import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models import qwen4_exp as model
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=30, suite="base-b-test-1-npu-a3")
pytestmark = pytest.mark.skipif(not model._is_npu, reason="NPU is required")


def _make_case(mode, fast, monkeypatch):
    torch.manual_seed(123)
    width = 1 if mode.is_decode() else (4 if mode.is_target_verify() else 16)
    lengths = [width, 0 if mode.is_target_verify() else (7 if width == 16 else width)]
    # Verify keeps padded rows; prefill packs only the real tokens.
    packed_lengths = [width, width] if mode.is_target_verify() else lengths
    req = torch.repeat_interleave(torch.arange(2), torch.tensor(packed_lengths))
    offsets = torch.cat([torch.arange(n) for n in packed_lengths])
    x = torch.randn(sum(packed_lengths), 10240, dtype=torch.bfloat16, device="npu")
    if mode.is_target_verify():
        x[width:] = 0
    state = torch.randn(5, 10240, 9, dtype=x.dtype, device=x.device)
    scratch = torch.zeros(3, width, 10240, 9, dtype=x.dtype, device=x.device)
    layer = model.Qwen4ExpPLELayer.__new__(model.Qwen4ExpPLELayer)
    nn.Module.__init__(layer)
    layer.layer_id, layer.conv_channels = 0, 10240
    layer.conv_kernel_size, layer.short_conv_dilation, layer.short_conv_state_len = (
        4,
        3,
        9,
    )
    layer.conv1d = nn.Module()
    # Model weights may be stored in FP32; the kernel takes the input dtype.
    layer.conv1d.weight = nn.Parameter(torch.randn(10240, 1, 4, device="npu"))
    pool = SimpleNamespace(
        size=2,
        layer_router=SimpleNamespace(
            read=lambda *_: SimpleNamespace(
                view_layer_id=0,
                view=SimpleNamespace(
                    layer_cache=lambda _: state,
                    layer_intermediate_cache=lambda _: scratch,
                ),
            )
        ),
    )
    monkeypatch.setattr(model, "get_token_to_kv_pool", lambda: pool)
    monkeypatch.setattr(model, "get_req_to_token_pool", lambda: pool)
    track_indices = torch.tensor([3, 4], device="npu")
    backend = SimpleNamespace(
        linear_attn_backend=SimpleNamespace(state_tracking_indices=track_indices)
    )
    monkeypatch.setattr(model, "get_attn_backend", lambda: backend)
    batch = model._PLEBatch(
        mode=mode,
        use_decode_fast_path=fast,
        physical_tokens=x.shape[0],
        processed_tokens=x.shape[0],
        lengths=torch.tensor(lengths, device="npu"),
        row_width=width,
        req_indices=req.to("npu"),
        token_offsets=offsets.to("npu"),
        valid_tokens=(req == 0).to("npu")
        if mode.is_target_verify()
        else torch.ones(x.shape[0], dtype=torch.bool, device="npu"),
        state_indices=torch.tensor([2, 1], device="npu"),
        query_start_loc=torch.tensor([0, lengths[0], sum(lengths)], device="npu")
        if mode.is_extend()
        else None,
        ngram_context=None,
        ngram_eos_token_id=None,
    )
    forward_batch = SimpleNamespace(
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int32, device="npu"),
        mamba_track_indices=track_indices,
        mamba_track_mask=torch.tensor([True, False], device="npu"),
        mamba_track_aligned_lens=lambda: torch.tensor([8, 0], device="npu"),
    )
    return layer, x, state, scratch, batch, forward_batch


def _reference(layer, x, state, batch):
    x, state = x.cpu(), state.cpu()
    expected, next_state = torch.empty_like(x), state.clone()
    scratch = torch.zeros(3, batch.row_width, 10240, 9, dtype=x.dtype)
    weight = layer.conv1d.weight.detach().cpu().to(x.dtype).float()
    for req, slot in enumerate([2, 1]):
        rows = batch.req_indices.cpu() == req
        tokens = x[rows].T
        sequence = torch.cat([state[slot], tokens], dim=1)
        conv = F.conv1d(sequence[None].float(), weight, dilation=3, groups=10240)
        expected[rows] = F.silu(conv.to(x.dtype)).squeeze(0).T
        length = int(batch.lengths[req].cpu())
        if batch.mode.is_target_verify():
            # Verify writes candidate histories without committing the working state.
            for step in range(length):
                scratch[req, step] = sequence[:, step + 1 : step + 10]
        else:
            next_state[slot] = sequence[:, length : length + 9]
            track_slot = 3 if req == 0 else 0
            offset = length if batch.mode.is_decode() else (8 if req == 0 else 0)
            next_state[track_slot] = sequence[:, offset : offset + 9]
    return expected, next_state, scratch


@pytest.mark.parametrize(
    "mode,fast,capture",
    [
        pytest.param(
            ForwardMode.DECODE, True, True, id="qwen3.8-flash-next-decode-b2-graph"
        ),
        pytest.param(
            ForwardMode.DECODE, False, False, id="qwen3.8-flash-next-decode-b2-padded"
        ),
        pytest.param(
            ForwardMode.TARGET_VERIFY,
            False,
            True,
            id="qwen3.8-flash-next-verify-w4-inactive-request-graph",
        ),
        pytest.param(
            ForwardMode.EXTEND, False, False, id="qwen3.8-flash-next-prefill-t16-t7"
        ),
    ],
)
@torch.no_grad()
def test_ple_short_conv(monkeypatch, mode, fast, capture):
    layer, x, state, scratch, batch, forward_batch = _make_case(mode, fast, monkeypatch)
    initial_state = state.clone()
    for _ in range(2):
        layer._short_conv(x, forward_batch, batch)
        state.copy_(initial_state)
    graph = None
    if capture:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = layer._short_conv(x, forward_batch, batch)

    for step in range(2 if capture else 1):
        if step:
            x.mul_(-0.5)
            initial_state.mul_(0.5)
        state.copy_(initial_state)
        scratch.zero_()
        expected, expected_state, expected_scratch = _reference(layer, x, state, batch)
        if graph is None:
            actual = layer._short_conv(x, forward_batch, batch)
        else:
            graph.replay()
            torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(state.cpu(), expected_state, atol=0, rtol=0)
        torch.testing.assert_close(scratch.cpu(), expected_scratch, atol=0, rtol=0)


@torch.no_grad()
def test_ple_grouped_norm():
    torch.manual_seed(7)
    norm = model.Qwen4ExpPLEGroupedNorm(10240, group_size=2560).to(
        device="npu", dtype=torch.bfloat16
    )
    norm.weight.normal_(std=0.1)
    x = torch.randn(4, 10240, device="npu", dtype=torch.bfloat16)
    grouped = x.cpu().float().reshape(4, 4, 2560)
    expected = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + 1e-6)
    expected = expected.flatten(1) * (1 + norm.weight.cpu().float())
    torch.testing.assert_close(norm(x).cpu().float(), expected, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
