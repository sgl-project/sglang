"""CPU numerical oracle for A-side routing and direct result ownership."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.afd.expert_exchange import (
    combine_partials,
    direct_payload,
)
from sglang.srt.afd.pipeline import AFDAttentionPipeline
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("num_f", [2, 4, 8])
@pytest.mark.parametrize("rows", [0, 1, 32])
def test_direct_routes_match_independent_weighted_expert_oracle(num_f, rows):
    generator = torch.Generator().manual_seed(123)
    hidden = torch.randn(rows, 12, generator=generator)
    ids = torch.randint(0, 128, (rows, 8), generator=generator, dtype=torch.int32)
    weights = torch.rand(rows, 8, generator=generator)
    weights /= weights.sum(dim=1, keepdim=True)
    if rows:
        # Several experts on one F, a dropped route, and zero-weight routes.
        ids[0] = torch.tensor([0, 1, 2, 3, 4, 5, 6, -1])
        weights[0, -2:] = 0
    experts = torch.randn(128, 12, 12, generator=generator)
    expected = torch.zeros_like(hidden)
    for token in range(rows):
        for route in range(8):
            expert = int(ids[token, route])
            if expert >= 0:
                expected[token] += weights[token, route] * (
                    hidden[token] @ experts[expert]
                )

    x, idx, w = direct_payload(hidden, ids, weights)
    assert x.data_ptr() == hidden.data_ptr()  # no per-F hidden packing
    assert idx.dtype == torch.int64
    torch.testing.assert_close(idx, ids.to(torch.int64))
    torch.testing.assert_close(w, weights)
    partials = []
    seen = torch.zeros_like(ids, dtype=torch.int32)
    for f in range(num_f):
        # Match the native StandardDispatcher's global-to-local drop behavior.
        local = (idx >= f * (128 // num_f)) & (idx < (f + 1) * (128 // num_f))
        seen += local.int()
        selected_weights = experts[idx.clamp_min(0)]
        y = torch.einsum("rh,rkhj->rkj", x, selected_weights)
        partials.append((y * torch.where(local, w, 0)[..., None]).sum(dim=1))
    torch.testing.assert_close(seen, (ids >= 0).int())
    actual = torch.empty_like(hidden)
    before_ptr = actual.data_ptr()
    combine_partials(torch.stack(partials), actual)
    assert actual.data_ptr() == before_ptr
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_native_exchange_waits_for_send_before_reusing_receive_buffer(monkeypatch):
    from contextlib import nullcontext

    import sglang.srt.afd.model_adapters.base as base
    from sglang.srt.afd.contracts import AFDRole
    from sglang.srt.afd.expert_exchange import AFDDirectExchange

    trace = []
    plane = object.__new__(AFDDirectExchange)
    plane.role = AFDRole.FFN
    plane.num_a, plane.rows, plane.hidden = 1, 1, 2
    plane.buffers = [
        (torch.zeros(1, 1, 2), torch.zeros(1, 1, 8), torch.zeros(1, 1, 8))
        for _ in range(2)
    ]
    plane.transport = SimpleNamespace(wait=lambda event: trace.append(("wait", event)))
    count = [0, 0]

    def exchange(stage, direction, tensors, *, send):
        if not send:
            event = ("recv", stage, count[stage])
            count[stage] += 1
        else:
            event = ("send", stage, count[stage] - 1)
        trace.append(event)
        return event

    monkeypatch.setattr(base, "stage_forward_context", lambda _: nullcontext())

    def compute(hidden, topk):
        # Native in-place runners may return the receive storage itself.
        stage = next(
            i
            for i, fields in enumerate(plane.buffers)
            if fields[0].data_ptr() == hidden.data_ptr()
        )
        trace.append(("compute", stage, count[stage] - 1))
        return hidden

    plane.adapter = SimpleNamespace(
        inner=SimpleNamespace(
            layers=[SimpleNamespace(mlp=SimpleNamespace(experts=compute))] * 3
        )
    )
    plane._exchange = exchange
    for layer in range(3):
        for stage in range(2):
            plane.run_layer(layer=layer, stage=stage)
    assert trace[0] == ("recv", 0, 0)
    assert count == [3, 3]  # no unmatched receive past the final layer
    for stage in range(2):
        for layer in range(3):
            assert trace.index(("wait", ("recv", stage, layer))) < trace.index(
                ("compute", stage, layer)
            )
            if layer:
                assert trace.index(("compute", stage, layer - 1)) < trace.index(
                    ("recv", stage, layer)
                )
                assert trace.index(("wait", ("send", stage, layer - 1))) < trace.index(
                    ("recv", stage, layer)
                )
            assert trace.index(("send", stage, layer)) < trace.index(
                ("wait", ("send", stage, layer))
            )
    # All ranks retain dispatch0/combine0/dispatch1/combine1 launch order.
    launches = [event[:2] for event in trace if event[0] in ("send", "recv")]
    assert launches == [("recv", 0), ("send", 0), ("recv", 1), ("send", 1)] * 3


@pytest.mark.parametrize("shared_value", [0, 3])
def test_backend_attention_preserves_layer_order_and_shared_output(shared_value):
    trace = []

    def route(hidden):
        return SimpleNamespace(
            topk_ids=torch.zeros((16, 8), dtype=torch.int64),
            topk_weights=torch.full((16, 8), 0.125),
        )

    def local_compute(*, layer, stage, hidden_states, residual, **kwargs):
        trace.append(("attention", layer, stage.index))
        return hidden_states + 1, residual

    def finish_layer(*, layer, stage, ffn_output, residual):
        trace.append(("finish", layer, stage.index))
        return ffn_output.clone(), residual

    def dispatch(*, hidden, ids, weights, stage, output):
        trace.append(("backend", stage))
        assert ids.dtype == torch.int64 and weights.dtype == torch.float32
        output.copy_(hidden * 2)
        return ("ready", stage)

    adapter = SimpleNamespace(
        num_layers=2,
        inner=SimpleNamespace(
            layers=[
                SimpleNamespace(
                    mlp=SimpleNamespace(
                        route=route,
                        shared_experts=(lambda h: torch.full_like(h, shared_value))
                        if shared_value
                        else None,
                    )
                )
                for _ in range(2)
            ]
        ),
        prepare_stage=lambda **kw: None,
        local_compute=local_compute,
        finish_layer=finish_layer,
    )
    transport = SimpleNamespace(
        ffn_backend=SimpleNamespace(first_layer=0, dispatch=dispatch),
        wait=lambda event: trace.append(("wait", event[1])),
    )
    stages = [
        SimpleNamespace(
            index=i,
            rows=16,
            forward_batch=None,
            hidden_states=torch.ones((16, 4)),
            residual=None,
            positions=None,
        )
        for i in range(2)
    ]
    pipeline = AFDAttentionPipeline.__new__(AFDAttentionPipeline)
    pipeline._adapter = adapter
    pipeline._connector = SimpleNamespace(transport=transport)
    pipeline._run_layers(
        stages=stages,
        recv_buffers=[torch.empty((16, 4)) for _ in stages],
        participates=(True, True),
        uses_backend=True,
    )
    assert [(x[1], x[2]) for x in trace if x[0] == "finish"] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    assert sum(x[0] == "backend" for x in trace) == 4
    for stage in stages:
        torch.testing.assert_close(
            stage.hidden_states, torch.full((16, 4), 10.0 + 3 * shared_value)
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))
