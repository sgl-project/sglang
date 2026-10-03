"""Prepared-row reuse through BF16/FP8 CuTe GEMMs and graph replay."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.lora.workspace import LoraWorkspace
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_E, _H, _I, _K = 4, 128, 128, 2


@pytest.mark.parametrize("family", ("bf16",))
@pytest.mark.parametrize("row_mode", ("expert_major", "route_major"))
@pytest.mark.parametrize("input_buffer_reuse", (False, True))
def test_down_output_reuses_workspace_rows(family, row_mode, input_buffer_reuse):
    from sglang.srt.lora.moe.base_gemm_provider import select_provider_cls

    cls = select_provider_cls(row_mode, family, "cutedsl")
    provider = cls.__new__(cls)
    provider.quant_info = SimpleNamespace(num_local_experts=_E, hidden_size=_H)
    shape = (_E, 8, _H) if row_mode == "expert_major" else (128, _H)
    dtype = torch.bfloat16 if family == "bf16" else torch.float8_e4m3fn
    workspace = LoraWorkspace()
    workspace.begin_forward(graph_mode=True)
    rows = (
        provider._prepare_input_rows(
            workspace if input_buffer_reuse else None, shape, "cpu"
        )
        if family == "fp8"
        else torch.empty(shape, dtype=dtype)
    )
    state = SimpleNamespace(
        hidden_permuted=rows,
        hidden_compact=rows,
        input_buffer_reuse=input_buffer_reuse,
        pair_to_row=torch.empty(2, dtype=torch.int32),
        m_max=8,
        m_pad_ceiling=128,
    )
    output = provider.down_output(state, workspace)
    assert output.shape == shape
    assert output.dtype == torch.bfloat16
    assert (output.data_ptr() == rows.data_ptr()) == input_buffer_reuse
    assert rows.is_contiguous()
    assert rows.shape == output.shape
    if input_buffer_reuse:
        assert rows.untyped_storage().nbytes() == output.numel() * 2
    assert any(key[0] == "base:down" for key in workspace._graph_storage) == (
        family != "bf16" or not input_buffer_reuse
    )


@pytest.fixture(
    scope="module",
    params=(
        ("bf16", "expert_major", False),
        ("bf16", "route_major", False),
    ),
    ids=(
        "bf16-masked",
        "bf16-contiguous",
    ),
)
def provider_case(request):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability()[0] not in (9, 10):
        pytest.skip("CuTe provider requires SM90 or SM100")
    pytest.importorskip("cutlass")
    pytest.importorskip("cuda.bindings.driver")
    from sglang.srt.lora.moe.base_gemm_provider import select_provider_cls
    from sglang.srt.lora.moe.quant_info import MoeLoraBf16QuantInfo, MoeLoraFp8QuantInfo
    from sglang.srt.runtime_context import get_context

    generator = torch.Generator().manual_seed(1631)
    w13 = (torch.randn(_E, 2 * _I, _H, generator=generator) * 0.08).to(
        device="cuda", dtype=torch.bfloat16
    )
    w2 = (torch.randn(_E, _H, _I, generator=generator) * 0.08).to(
        device="cuda", dtype=torch.bfloat16
    )
    family, row_mode, separate_quant = request.param
    weights = dict(w13_weight=w13, w2_weight=w2)
    info_cls = MoeLoraBf16QuantInfo
    if family == "fp8":
        q13, s13, w13 = _block_quant(w13)
        q2, s2, w2 = _block_quant(w2)
        weights = dict(
            w13_weight=q13,
            w13_scale=s13,
            w2_weight=q2,
            w2_scale=s2,
            block_shape=(128, 128),
        )
        info_cls = MoeLoraFp8QuantInfo
    info = info_cls(
        **weights,
        num_local_experts=_E,
        intermediate_size=_I,
        hidden_size=_H,
    )
    with get_context().override_server_args():
        provider = select_provider_cls(row_mode, family, "cutedsl")(info)
        yield provider, w13, w2, family, separate_quant


def _block_quant(weight):
    experts, rows, cols = weight.shape
    blocks = weight.float().view(experts, rows // 128, 128, cols // 128, 128)
    scales = blocks.abs().amax(dim=(2, 4)).clamp(min=1e-6) / 448
    expanded = scales.repeat_interleave(128, dim=1).repeat_interleave(128, dim=2)
    quantized = (weight.float() / expanded).to(torch.float8_e4m3fn)
    return quantized, scales, quantized.float() * expanded


def _traffic(ids, pattern):
    ids[:, 0] = _E - 2
    ids[:, 1] = _E - 1
    if pattern == "padded":
        ids[::2, 0] = -1
        ids[1::3, 1] = 0
        ids[-max(1, ids.shape[0] // 4) :] = -1
    elif pattern == "empty":
        ids.fill_(-1)


def _reference(hidden, ids, weights, w13, w2):
    output = torch.zeros_like(hidden, dtype=torch.float32)
    for expert in range(_E):
        coefficients = (weights * (ids == expert)).sum(dim=1)
        gateup = (hidden.float() @ w13[expert].float().T).to(torch.bfloat16).float()
        act = (F.silu(gateup[:, :_I]) * gateup[:, _I:]).to(torch.bfloat16)
        down = (act.float() @ w2[expert].float().T).to(torch.bfloat16).float()
        output.add_(down * coefficients[:, None])
    return output * 0.75


def _case(provider, workspace, n, is_prefill, *, graph_mode=True, separate_quant=False):
    hidden = torch.randn(n, _H, device="cuda", dtype=torch.bfloat16) * 0.2
    ids = torch.empty(n, _K, device="cuda", dtype=torch.int32)
    _traffic(ids, "skewed")
    weights = torch.tensor([0.65, 0.35], device="cuda").expand(n, _K).contiguous()

    def run():
        workspace.begin_forward(graph_mode=graph_mode, is_prefill_graph=is_prefill)
        state = provider.prepare(hidden, ids, _K, workspace)

        def buffer(name, shape):
            return workspace.tensor(name, shape, dtype=torch.bfloat16, device="cuda")

        gateup = buffer("test:gateup", provider.gateup_out_shape(state))
        act = buffer("test:act", provider.act_out_shape(state))
        pairs = buffer("test:pairs", (n, _K, _I))
        provider.gateup(state, gateup)
        provider.release_prepared_inputs(state)
        provider.act_with_delta(state, gateup, None, ids, act, pairs)
        if separate_quant:
            state.act_quant_ready = False
        down = provider.down_output(state, workspace)
        provider.down(state, act, down)
        output = torch.empty_like(hidden)
        provider.finalize(state, down, ids, weights, 0.75, output)
        return state, down, output

    return run, hidden, ids, weights


def _check(output, hidden, ids, weights, w13, w2, family):
    expected = _reference(hidden, ids, weights, w13, w2)
    actual = output.float()
    assert bool(torch.isfinite(actual).all())
    error = (actual - expected).abs()
    relative_l2 = error.norm() / expected.norm().clamp(min=1e-12)
    assert relative_l2.item() < (0.08 if family == "fp8" else 0.02)
    assert bool((actual[(ids < 0).all(dim=1)] == 0).all())
    return (
        relative_l2.item(),
        error.max().item() if error.numel() else 0.0,
        error.mean().item() if error.numel() else 0.0,
    )


def _record_errors(record_property, errors):
    for name, values in zip(
        ("relative_l2", "max_abs_error", "mean_abs_error"), zip(*errors)
    ):
        record_property(name, max(values))


def _capture(run):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = run()
    torch.cuda.current_stream().wait_stream(stream)
    return graph, result


@pytest.mark.parametrize("n", (0, 7, 8, 9, 63, 64, 65, 127, 128, 129, 255, 256, 257))
def test_prepared_rows_reused_in_eager(provider_case, n, record_property):
    provider, w13, w2, family, separate_quant = provider_case
    workspace = LoraWorkspace()
    run, hidden, ids, weights = _case(
        provider, workspace, n, False, graph_mode=False, separate_quant=separate_quant
    )
    errors = []
    for pattern in ("skewed", "padded", "empty", "skewed"):
        hidden.mul_(-0.9).add_(0.01)
        _traffic(ids, pattern)
        state, down, output = run()
        assert down.data_ptr() == provider._input_rows(state).data_ptr()
        assert down.data_ptr() != hidden.data_ptr()
        errors.append(_check(output, hidden, ids, weights, w13, w2, family))
    assert ("base:down" in {key[0] for key in workspace._eager_buffers}) == (
        family == "fp8"
    )
    _record_errors(record_property, errors)


def test_reused_rows_old_graphs_survive_growth_and_other_phase(
    provider_case, record_property
):
    provider, w13, w2, family, separate_quant = provider_case
    workspace = LoraWorkspace()
    captured = []
    # The non-monotonic order explicitly grows storage after capture.
    buckets = (
        (False, 9),
        (True, 129),
        (False, 7),
        (False, 257),
        (True, 4096),
        (False, 8),
        (False, 0),
    )
    for is_prefill, n in buckets:
        run, hidden, ids, weights = _case(
            provider, workspace, n, is_prefill, separate_quant=separate_quant
        )
        graph, result = _capture(run)
        state, down, output = result
        assert down.data_ptr() == provider._input_rows(state).data_ptr()
        assert down.data_ptr() != hidden.data_ptr()
        assert down.shape == provider.down_out_shape(state)
        captured.append((graph, hidden, ids, weights, down, output))

    addresses = [item[4].data_ptr() for item in captured]
    assert addresses[0] == addresses[2]
    assert addresses[3] == addresses[5]
    assert addresses[0] != addresses[3]
    assert addresses[1] != addresses[4]
    assert not {addresses[0], addresses[3]} & {addresses[1], addresses[4]}
    retired = {tensor.data_ptr() for tensor in workspace._retired}
    assert {addresses[0], addresses[1]} <= retired
    assert ("base:down" in {key[0] for key in workspace._graph_storage}) == (
        family == "fp8"
    )

    errors = []
    for pattern in ("skewed", "padded", "empty", "skewed"):
        for graph, hidden, ids, weights, down, output in reversed(captured):
            hidden.mul_(-0.9).add_(0.01)
            _traffic(ids, pattern)
            # Poison all rows; dispatch must restore every subsequently read row.
            down.fill_(float("nan"))
            graph.replay()
            errors.append(_check(output, hidden, ids, weights, w13, w2, family))
    _record_errors(record_property, errors)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
