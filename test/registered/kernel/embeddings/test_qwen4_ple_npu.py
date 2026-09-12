from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.kernels.ops import qwen4_ple
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import qwen4_exp as model
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=20, suite="base-b-test-1-npu-a3")


def _make_case(device, dtype, mode, kernel_size, dilation, *, reference=False):
    generator = torch.Generator().manual_seed(123)

    def random_tensor(*shape):
        # Quantize both paths identically, then compute the CPU reference in FP32.
        return (
            torch.randn(*shape, generator=generator)
            .to(dtype)
            .to(device=device, dtype=torch.float32 if reference else dtype)
        )

    channels = 16
    width = 4 if mode in ("verify", "prefill") else 1
    state_len = (kernel_size - 1) * dilation
    layer = model.Qwen4ExpPLELayer.__new__(model.Qwen4ExpPLELayer)
    nn.Module.__init__(layer)
    layer.layer_id = 0
    layer.conv_channels = channels
    layer.conv_kernel_size = kernel_size
    layer.short_conv_dilation = dilation
    layer.short_conv_state_len = state_len
    layer.conv1d = nn.Module()
    layer.conv1d.weight = nn.Parameter(
        random_tensor(channels, 1, kernel_size), requires_grad=False
    )
    state = random_tensor(3, channels, state_len)
    initial_state = state.clone()
    intermediate = state.new_zeros(2, width, channels, state_len)
    pool = SimpleNamespace(
        short_conv_layer_cache=lambda _: state,
        short_conv_layer_intermediate_cache=lambda _: intermediate,
    )
    x = random_tensor(2 * width, channels)
    lengths = torch.tensor([width, width], device=device)
    valid = torch.ones(2 * width, device=device, dtype=torch.bool)
    if mode == "verify":
        lengths[1] = 0
        valid[width:] = False
        x[width:] = 0
    batch = model._PLEBatch(
        mode={
            "decode_fast": ForwardMode.DECODE,
            "decode": ForwardMode.DECODE,
            "verify": ForwardMode.TARGET_VERIFY,
            "prefill": ForwardMode.EXTEND,
        }[mode],
        use_decode_fast_path=mode == "decode_fast",
        physical_tokens=x.shape[0],
        processed_tokens=x.shape[0],
        lengths=lengths,
        row_width=width,
        req_indices=torch.arange(2, device=device).repeat_interleave(width),
        token_offsets=torch.arange(width, device=device).repeat(2),
        valid_tokens=valid,
        state_indices=torch.tensor([2, 1], device=device),
        ngram_context=None,
        ngram_eos_token_id=None,
    )
    forward_batch = SimpleNamespace(mamba_track_indices=None, mamba_track_mask=None)
    return SimpleNamespace(
        layer=layer,
        x=x,
        state=state,
        initial_state=initial_state,
        intermediate=intermediate,
        pool=pool,
        batch=batch,
        forward_batch=forward_batch,
    )


def _run(case, monkeypatch, *, npu):
    monkeypatch.setattr(model, "_is_npu", npu)
    monkeypatch.setattr(model, "get_req_to_token_pool", lambda: case.pool)
    return case.layer._short_conv(case.x, case.forward_batch, case.batch)


@pytest.mark.parametrize("execution", ["cpu", "npu", "npu_graph"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kernel_size,dilation", [(1, 1), (4, 3)])
@pytest.mark.parametrize("mode", ["decode_fast", "decode", "verify", "prefill"])
def test_ple_short_conv_npu(monkeypatch, execution, dtype, kernel_size, dilation, mode):
    if execution != "cpu" and not model._is_npu:
        pytest.skip("NPU is not available")
    if execution == "npu_graph" and mode == "prefill":
        pytest.skip("Prefill retains native convolution outside graph capture")
    # Isolate convolution from the optional CUDA fused state-movement kernel.
    monkeypatch.setattr(qwen4_ple, "can_fuse_qwen4_short_conv_state", lambda *args: False)
    device = "cpu" if execution == "cpu" else "npu"
    case = _make_case(device, dtype, mode, kernel_size, dilation)
    reference = _make_case("cpu", dtype, mode, kernel_size, dilation, reference=True)
    graph = None
    if execution == "npu_graph":
        for _ in range(2):
            case.state.copy_(case.initial_state)
            _run(case, monkeypatch, npu=True)
        case.state.copy_(case.initial_state)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        monkeypatch.setattr(model, "_is_npu", True)
        monkeypatch.setattr(model, "get_req_to_token_pool", lambda: case.pool)
        with torch.npu.graph(graph):
            actual = case.layer._short_conv(case.x, case.forward_batch, case.batch)

    original_conv1d = torch.nn.functional.conv1d
    for step in range(2):
        if step:
            # Replay must consume changed inputs and persistent state contents.
            case.x.mul_(-0.5)
            reference.x.mul_(-0.5)
            case.initial_state.mul_(0.5)
            reference.initial_state.mul_(0.5)
        case.state.copy_(case.initial_state)
        reference.state.copy_(reference.initial_state)
        monkeypatch.setattr(torch.nn.functional, "conv1d", original_conv1d)
        expected = _run(reference, monkeypatch, npu=False)
        calls = []

        def record_conv1d(*args, **kwargs):
            calls.append(True)
            return original_conv1d(*args, **kwargs)

        monkeypatch.setattr(torch.nn.functional, "conv1d", record_conv1d)
        if graph is None:
            actual = _run(case, monkeypatch, npu=True)
            assert len(calls) == (1 if mode == "prefill" else 0)
        else:
            graph.replay()
            torch.npu.synchronize()
        # The decomposition rounds individual products in the input dtype;
        # FP32 native convolution is a numerical reference, not bitwise identical.
        tolerance = 2e-2 if dtype == torch.bfloat16 else 3e-3
        torch.testing.assert_close(
            actual.float().cpu(), expected, rtol=tolerance, atol=tolerance
        )
        torch.testing.assert_close(
            case.state.float().cpu(), reference.state, rtol=0, atol=0
        )
        torch.testing.assert_close(
            case.intermediate.float().cpu(), reference.intermediate, rtol=0, atol=0
        )
