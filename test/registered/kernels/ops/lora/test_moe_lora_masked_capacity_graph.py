"""Real CuTe provider replay at slab boundaries, with one compilation per tile."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.lora.workspace import LoraWorkspace
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_EXPERTS, _HIDDEN, _INTERMEDIATE, _TOP_K = 4, 128, 128, 2
# Include the measured server maximum (84), a larger cap, and request width > 1.
_BUCKETS = (0, 1, 7, 8, 9, 63, 64, 65, 84, 127, 128, 129, 255, 256, 257, 64 * 5, 512)


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


@pytest.fixture(
    scope="module",
    params=(("bf16", False), ("fp8", False), ("fp8", True)),
    ids=("bf16", "fp8", "fp8-separate-quant"),
)
def masked_provider(request):
    _require_cuda()
    if torch.cuda.get_device_capability()[0] not in (9, 10):
        pytest.skip("CuTe providers require SM90 or SM100")
    pytest.importorskip("cutlass")
    pytest.importorskip("cuda.bindings.driver")

    from sglang.srt.lora.moe.base_gemm_provider import select_provider_cls
    from sglang.srt.lora.moe.quant_info import (
        MoeLoraBf16QuantInfo,
        MoeLoraFp8QuantInfo,
    )
    from sglang.srt.runtime_context import get_context

    generator = torch.Generator().manual_seed(719)

    def weight(rows, cols):
        return (torch.randn(_EXPERTS, rows, cols, generator=generator) * 0.05).to(
            device="cuda", dtype=torch.bfloat16
        )

    w13, w2 = weight(2 * _INTERMEDIATE, _HIDDEN), weight(_HIDDEN, _INTERMEDIATE)
    geometry = dict(
        num_local_experts=_EXPERTS,
        intermediate_size=_INTERMEDIATE,
        hidden_size=_HIDDEN,
    )
    family, separate_quant = request.param
    if family == "fp8":
        # Legal block-128 checkpoint scales; reference the effective weights,
        # leaving only activation quantization error in the provider comparison.
        def quantize(w):
            scale = torch.full(
                (_EXPERTS, w.shape[1] // 128, w.shape[2] // 128),
                1 / 1024,
                dtype=torch.float32,
                device="cuda",
            )
            q = (w.float() * 1024).to(torch.float8_e4m3fn)
            return q, scale, q.float() / 1024

        q13, s13, w13 = quantize(w13)
        q2, s2, w2 = quantize(w2)
        info = MoeLoraFp8QuantInfo(
            w13_weight=q13,
            w13_scale=s13,
            w2_weight=q2,
            w2_scale=s2,
            block_shape=(128, 128),
            **geometry,
        )
    else:
        info = MoeLoraBf16QuantInfo(w13_weight=w13, w2_weight=w2, **geometry)
    with get_context().override_server_args():
        provider = select_provider_cls("expert_major", family, "cutedsl")(info)
    return family, provider, w13.float(), w2.float(), separate_quant


def _capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = fn()
    torch.cuda.current_stream().wait_stream(stream)
    return graph, result


def _set_routing(ids, traffic):
    # Every token chooses distinct experts; the last expert receives N rows,
    # testing the physical end of the slab rather than a balanced mean load.
    ids[:, 0] = _EXPERTS - 2
    ids[:, 1] = _EXPERTS - 1
    if traffic == "padded":
        ids[::2, 0] = -1
        ids[1::3, 1] = 0
        ids[-max(1, ids.shape[0] // 4) :] = -1
    elif traffic == "empty":
        ids.fill_(-1)


def _check_pairs(hidden, ids, result, w13, w2, family):
    state, gateup, act, down = result
    flat_ids = ids.flatten().long()
    valid = flat_ids >= 0
    pairs = torch.nonzero(valid, as_tuple=True)[0]
    experts = flat_ids[pairs]
    rows = state.pair_to_row[pairs].long()
    torch.testing.assert_close(
        state.masked_m.long(), torch.bincount(experts, minlength=_EXPERTS)
    )
    assert bool((state.pair_to_row[~valid] == -1).all())
    assert bool(
        ((rows >= experts * state.m_max) & (rows < (experts + 1) * state.m_max)).all()
    )
    assert rows.unique().numel() == pairs.numel()
    if not pairs.numel():
        assert state.gemm1_tiles.item() == state.gemm2_tiles.item() == 0
        return

    def gather(tensor):
        return tensor.reshape(-1, tensor.shape[-1])[rows].float()

    ref_gateup = torch.einsum("th,enh->ten", hidden.float(), w13)[
        pairs // _TOP_K, experts
    ]
    got_gateup, got_act, got_down = map(gather, (gateup, act, down))
    ref_act = F.silu(got_gateup[:, :_INTERMEDIATE]) * got_gateup[:, _INTERMEDIATE:]
    torch.testing.assert_close(got_act, ref_act, rtol=1e-2, atol=1e-5)
    ref_down = torch.einsum("pi,ehi->peh", got_act, w2)[
        torch.arange(pairs.numel(), device=hidden.device), experts
    ]
    tolerance = 4e-2 if family == "fp8" else 1e-2
    for actual, expected in ((got_gateup, ref_gateup), (got_down, ref_down)):
        assert bool(torch.isfinite(actual).all())
        relative_l2 = (actual - expected).norm() / expected.norm().clamp(min=1e-12)
        assert relative_l2.item() < tolerance


@pytest.mark.parametrize("token_width", (8, 64, 128))
def test_masked_capacity_boundary_graphs(masked_provider, monkeypatch, token_width):
    from sglang.srt.runtime_context import get_context

    family, provider, w13, w2, separate_quant = masked_provider
    # Exercise each already-compiled kernel without modifying dispatch policy.
    monkeypatch.setattr(provider, "_token_width_for", lambda *_: token_width)
    workspace = LoraWorkspace()
    workspace.begin_forward(graph_mode=True)
    captured = []
    generator = torch.Generator().manual_seed(720)
    with get_context().override_server_args():
        for n in reversed(_BUCKETS):
            hidden = (torch.randn(n, _HIDDEN, generator=generator) * 0.1).to(
                device="cuda", dtype=torch.bfloat16
            )
            ids = torch.empty((n, _TOP_K), dtype=torch.int32, device="cuda")
            _set_routing(ids, "last")

            def run():
                state = provider.prepare(hidden, ids, _TOP_K, workspace)

                def buffer(name, shape):
                    return workspace.tensor(
                        name, shape, dtype=torch.bfloat16, device=hidden.device
                    )

                gateup = buffer("test:gateup", provider.gateup_out_shape(state))
                act = buffer("test:act", provider.act_out_shape(state))
                down = buffer("test:down", provider.down_out_shape(state))
                act_pairs = buffer("test:act_pairs", (n, _TOP_K, _INTERMEDIATE))
                provider.gateup(state, gateup)
                provider.act_with_delta(state, gateup, None, ids, act, act_pairs)
                if separate_quant:
                    # Exercise the down-quantization path used by fused-B plans.
                    state.act_quant_ready = False
                provider.down(state, act, down)
                return state, gateup, act, down

            graph, result = _capture(run)
            alignment = token_width if family == "fp8" else 8
            expected_capacity = max(
                alignment, (n + alignment - 1) // alignment * alignment
            )
            assert result[0].m_max == expected_capacity
            assert result[0].token_width == token_width
            captured.append((graph, hidden, ids, result))

        # Replay old graphs after every smaller bucket has reused their storage.
        # Restoring a populated route after all -1 catches stale counters/maps.
        for graph, hidden, ids, result in captured:
            for traffic in ("last", "padded", "empty", "last"):
                _set_routing(ids, traffic)
                graph.replay()
                _check_pairs(hidden, ids, result, w13, w2, family)


def test_graph_workspace_growth_preserves_old_graphs_and_phase_isolation():
    _require_cuda()
    workspace = LoraWorkspace()
    captured = []
    # Grow decode after its first capture; a larger prefill allocation must not
    # replace its buffers either. No Python reference retains the old scratch.
    for is_prefill, n in ((False, 8), (True, 512), (False, 257)):
        workspace.begin_forward(graph_mode=True, is_prefill_graph=is_prefill)
        value = torch.tensor(1.0, device="cuda")
        out = torch.empty(n, device="cuda")

        def run():
            scratch = workspace.tensor(
                "scratch", (n,), dtype=torch.float32, device="cuda"
            )
            indices = workspace.iota(n, "cuda")
            scratch.copy_(value.expand_as(scratch))
            out.copy_(scratch + indices)
            return scratch.data_ptr(), indices.data_ptr()

        graph, addresses = _capture(run)
        captured.append((graph, value, out, addresses))

    assert len({addresses[0] for _, _, _, addresses in captured}) == 3
    assert len({addresses[1] for _, _, _, addresses in captured}) == 3
    old_addresses = captured[0][3]
    assert all(
        any(t.data_ptr() == address for t in workspace._retired)
        for address in old_addresses
    )
    for new_value in (3.0, -2.0):
        for graph, value, out, _ in captured:
            value.fill_(new_value)
            graph.replay()
            torch.testing.assert_close(
                out, torch.arange(out.numel(), device="cuda") + new_value
            )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
