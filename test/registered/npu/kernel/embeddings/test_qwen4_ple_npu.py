from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.kernels.ops import qwen4_ple
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import qwen4_exp as model
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=20, suite="base-b-test-1-npu-a3")


def _make_case(
    device, dtype, mode, kernel_size, dilation, *, width, channels, reference=False
):
    generator = torch.Generator().manual_seed(123)

    def random_tensor(*shape):
        # Quantize both paths identically, then compute the CPU reference in FP32.
        return (
            torch.randn(*shape, generator=generator)
            .to(dtype)
            .to(device=device, dtype=torch.float32 if reference else dtype)
        )

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
    state = random_tensor(5, channels, state_len)
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
    forward_batch = SimpleNamespace(
        mamba_track_indices=torch.tensor([3, 4], device=device),
        mamba_track_mask=torch.tensor([True, False], device=device),
        mamba_track_aligned_lens=lambda: torch.tensor([max(0, width - 1), 0], device=device),
    )
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


@pytest.mark.parametrize(
    "execution,kernel_size,dilation,channels",
    [("cpu", 1, 1, 16), ("cpu", 4, 3, 16),
     ("npu", 4, 3, 10240), ("npu_graph", 4, 3, 10240)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "mode,width", [("decode_fast", 1), ("decode", 1), ("verify", 2),
                   ("verify", 3), ("verify", 4), ("prefill", 4)],
)
def test_ple_short_conv_npu(
    monkeypatch, execution, dtype, kernel_size, dilation, channels, mode, width
):
    if execution != "cpu" and not model._is_npu:
        pytest.skip("NPU is not available")
    if execution == "npu_graph" and mode == "prefill":
        pytest.skip("Prefill retains native convolution outside graph capture")
    # Isolate convolution from the optional CUDA fused state-movement kernel.
    monkeypatch.setattr(
        qwen4_ple, "can_fuse_qwen4_short_conv_state", lambda *args: False
    )
    device = "cpu" if execution == "cpu" else "npu"
    case = _make_case(device, dtype, mode, kernel_size, dilation,
                      width=width, channels=channels)
    # On NPU compare the whole model operation against actual native F.conv1d,
    # including its output rounding before SiLU. CPU keeps generic coverage.
    reference = _make_case(device, dtype, mode, kernel_size, dilation,
                           width=width, channels=channels, reference=execution == "cpu")
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
    for step in range(3):
        if step:
            # Replay must consume changed inputs and persistent state contents.
            case.x.mul_(-0.5)
            reference.x.mul_(-0.5)
            case.initial_state.mul_(0.5)
            reference.initial_state.mul_(0.5)
        if step == 2:
            case.layer.conv1d.weight.mul_(-0.5)
            reference.layer.conv1d.weight.mul_(-0.5)
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
            actual = _run(case, monkeypatch, npu=execution != "cpu")
            assert len(calls) == (1 if execution == "cpu" or mode == "prefill" else 0)
        else:
            graph.replay()
            torch.npu.synchronize()
        # Keep the original dtype tolerances; graph must see updated state,
        # inputs and weights without running native convolution inside capture.
        tolerance = 2e-2 if dtype == torch.bfloat16 else 3e-3
        torch.testing.assert_close(
            actual.float().cpu(), expected.float().cpu(), rtol=tolerance, atol=tolerance
        )
        torch.testing.assert_close(
            case.state.float().cpu(), reference.state.float().cpu(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            case.intermediate.float().cpu(), reference.intermediate.float().cpu(), rtol=0, atol=0
        )


def test_ple_short_conv_empty(monkeypatch):
    layer = model.Qwen4ExpPLELayer.__new__(model.Qwen4ExpPLELayer)
    nn.Module.__init__(layer)

    def unexpected_pool():
        raise AssertionError("Empty input must not access state or launch convolution")

    monkeypatch.setattr(model, "get_req_to_token_pool", unexpected_pool)
    x = torch.empty(0, 10240)
    assert layer._short_conv(x, None, None) is x


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("capture", [False, True])
def test_ple_short_conv_state_preparation(monkeypatch, record_property, dtype, capture):
    """Exercise the real state-helper predicate and the caller's weight cast.

    Service NPU initialization may redirect Tensor.is_cuda. Unlike the isolated
    convolution tests, keep the production state-preparation choice here.
    """
    if not model._is_npu:
        pytest.skip("NPU is not available")
    case = _make_case("npu", dtype, "decode_fast", 4, 3, width=1, channels=10240)
    reference = _make_case("npu", dtype, "decode_fast", 4, 3, width=1, channels=10240)
    record_property(
        "fused_state",
        qwen4_ple.can_fuse_qwen4_short_conv_state(
            case.state, case.batch.state_indices, case.x
        ),
    )
    # Model parameters can have a different storage dtype. The framework must
    # convert them before calling the same-dtype kernel, preserving rank three.
    for item in (case, reference):
        item.layer.conv1d.weight = nn.Parameter(
            item.layer.conv1d.weight.float(), requires_grad=False
        )
    for _ in range(2):
        case.state.copy_(case.initial_state)
        _run(case, monkeypatch, npu=True)
    graph = None
    if capture:
        case.state.copy_(case.initial_state)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = _run(case, monkeypatch, npu=True)
    for step in range(3):
        if step:
            for item in (case, reference):
                item.x.mul_(-0.5)
                item.initial_state.mul_(0.5)
                item.layer.conv1d.weight.mul_(-0.5)
        for item in (case, reference):
            item.state.copy_(item.initial_state)
        with monkeypatch.context() as reference_patch:
            reference_patch.setattr(
                qwen4_ple, "can_fuse_qwen4_short_conv_state", lambda *args: False
            )
            expected = _run(reference, reference_patch, npu=False)
        if graph is None:
            actual = _run(case, monkeypatch, npu=True)
        else:
            graph.replay()
            torch.npu.synchronize()
        tolerance = 2e-2 if dtype == torch.bfloat16 else 3e-3
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(case.state, reference.state, atol=0, rtol=0)
