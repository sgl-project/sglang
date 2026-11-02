"""Grouped target SWA: self-contained reference, transport and graph tests."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention import aiter_backend, aiter_utils
from sglang.srt.layers.attention.grouped_pa_swa_eval import forward_grouped_pa_swa
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")
requires_rocm = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm"
)


def _layer(layer_id):
    return SimpleNamespace(
        layer_id=layer_id,
        tp_q_head_num=16,
        tp_k_head_num=2,
        qk_head_dim=192,
        v_head_dim=128,
        sliding_window_size=128,
        scaling=192**-0.5,
    )


@pytest.mark.parametrize(
    "enabled,draft", [(False, False), (False, True), (True, True), (True, False)]
)
def test_grouped_swa_configuration_is_opt_in_and_target_only(
    monkeypatch, enabled, draft
):
    monkeypatch.delenv("SGLANG_MIMO_GROUPED_PA_SWA", raising=False)
    if enabled:
        monkeypatch.setenv("SGLANG_MIMO_GROUPED_PA_SWA", "1")
    backend = object.__new__(aiter_backend.AiterAttnBackend)
    backend.is_draft_worker = draft
    runner = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(hybrid_layer_pattern=[0, 1, 1])
        ),
        server_args=SimpleNamespace(pp_size=1, speculative_algorithm="DFLASH"),
    )
    backend._configure_grouped_pa_swa(runner)
    assert backend._grouped_pa_swa_first_layer == (1 if enabled and not draft else None)


@pytest.mark.parametrize(
    "pp,spec,pattern",
    [
        (2, "DFLASH", [0, 1]),
        (1, "EAGLE", [0, 1]),
        (1, "DFLASH", [0, 0]),
        (1, "DFLASH", None),
    ],
)
def test_grouped_swa_rejects_unsupported_model_contract(monkeypatch, pp, spec, pattern):
    monkeypatch.setenv("SGLANG_MIMO_GROUPED_PA_SWA", "1")
    backend = object.__new__(aiter_backend.AiterAttnBackend)
    backend.is_draft_worker = False
    runner = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(hybrid_layer_pattern=pattern)
        ),
        server_args=SimpleNamespace(pp_size=pp, speculative_algorithm=spec),
    )
    with pytest.raises(ValueError, match="single-stage DFlash"):
        backend._configure_grouped_pa_swa(runner)


def test_grouped_swa_empty_batch_does_not_allocate_or_launch(monkeypatch):
    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")

    def unexpected(*args, **kwargs):
        pytest.fail("An empty batch must not plan or launch")

    monkeypatch.setattr(pa, "pa_decode", unexpected)
    monkeypatch.setattr(pa, "plan_pa_decode", unexpected)
    backend = SimpleNamespace(
        page_size=64, forward_metadata=SimpleNamespace(max_q_len=4)
    )
    q = torch.empty((0, 16 * 192), dtype=torch.bfloat16)
    out = torch.empty((0, 16 * 128), dtype=torch.bfloat16)
    forward_grouped_pa_swa(
        backend, q, _layer(1), SimpleNamespace(batch_size=0), q, q, out, torch.zeros(16)
    )
    assert not hasattr(backend, "_grouped_pa_swa_workspace")


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("qlen", 3, "query length1/4/8"),
        ("page_size", 32, "page64"),
        ("tp_k_head_num", 1, "Flash TP4"),
        ("sliding_window_size", 1024, "window128"),
    ],
)
def test_grouped_swa_geometry_guards(field, value, match):
    backend = SimpleNamespace(
        page_size=64, forward_metadata=SimpleNamespace(max_q_len=4)
    )
    layer = _layer(1)
    if field == "qlen":
        backend.forward_metadata.max_q_len = value
    elif field == "page_size":
        backend.page_size = value
    else:
        setattr(layer, field, value)
    q = torch.empty((0, 16 * 192), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=match):
        forward_grouped_pa_swa(
            backend, q, layer, SimpleNamespace(batch_size=0), q, q, q, torch.zeros(16)
        )


def _case(bs, qlen, backend=None):
    torch.manual_seed(58872 + bs + qlen)
    backend = backend or SimpleNamespace(
        page_size=64,
        is_draft_worker=False,
        _grouped_pa_swa_first_layer=1,
        _flydsl_dflash_swa_max_pages=4,
    )
    metadata = SimpleNamespace(
        max_q_len=qlen,
        flydsl_swa_context_lens=torch.empty(bs, device="cuda", dtype=torch.int32),
        flydsl_swa_kv_indptr=torch.empty(bs + 1, device="cuda", dtype=torch.int32),
        flydsl_swa_kv_indices=torch.zeros(bs * 4, device="cuda", dtype=torch.int32),
    )
    table = torch.randperm(bs * 4, device="cuda", dtype=torch.int32).view(bs, 4)
    inputs = []
    for layer_id in (1, 2):
        key = torch.empty(
            bs * 4, 2, 64, 192, device="cuda", dtype=torch.bfloat16
        ).uniform_(-0.5, 0.5)
        value = torch.empty(
            bs * 4, 2, 64, 128, device="cuda", dtype=torch.bfloat16
        ).uniform_(-0.5, 0.5)
        k = key.view(bs * 4, 2, 64, 24, 8).permute(0, 1, 3, 2, 4).contiguous()
        v = value.view(bs * 4, 2, 8, 8, 128).permute(0, 1, 2, 4, 3).contiguous()
        q = torch.empty(
            bs * qlen * 2, 16 * 192, device="cuda", dtype=torch.bfloat16
        ).uniform_(-0.5, 0.5)[::2]
        sink = torch.linspace(
            -4 + layer_id, 4 + layer_id, 16, device="cuda", dtype=torch.float32
        )
        guarded = torch.full(
            (bs * qlen + 2, 16 * 128), 17.0, device="cuda", dtype=torch.bfloat16
        )
        inputs.append(
            SimpleNamespace(
                layer=_layer(layer_id),
                key=key,
                value=value,
                k=k,
                v=v,
                q=q,
                sink=sink,
                guarded=guarded,
                out=guarded[1:-1],
            )
        )
    batch = SimpleNamespace(
        batch_size=bs, forward_mode=SimpleNamespace(is_target_verify=lambda: True)
    )
    case = SimpleNamespace(
        backend=backend,
        metadata=metadata,
        batch=batch,
        table=table,
        inputs=inputs,
        qlen=qlen,
    )
    return case


def _set_metadata(case, lengths, table):
    counts = [(n + 63) // 64 for n in lengths]
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    csr = torch.cat([table[i, :n] for i, n in enumerate(counts)])
    meta = case.metadata
    meta.flydsl_swa_context_lens.copy_(
        torch.tensor(lengths, device="cuda", dtype=torch.int32)
    )
    meta.flydsl_swa_kv_indptr.copy_(
        torch.tensor(offsets, device="cuda", dtype=torch.int32)
    )
    meta.flydsl_swa_kv_indices.zero_()
    meta.flydsl_swa_kv_indices[: csr.numel()].copy_(csr)
    case.lengths = list(lengths)
    case.active_table = table


def _forward(case):
    case.backend.forward_metadata = case.metadata
    for data in case.inputs:
        aiter_utils.forward_target_verify_flydsl_swa_5d(
            case.backend,
            data.q,
            data.layer,
            case.batch,
            data.k,
            data.v,
            data.out,
            data.sink,
        )
    return [x.out for x in case.inputs]


def _reference(case, data):
    expected = torch.empty_like(data.out).view(
        case.batch.batch_size, case.qlen, 16, 128
    )
    for row, length in enumerate(case.lengths):
        page_ids = case.active_table[row].long()
        k = (
            data.key[page_ids]
            .permute(0, 2, 1, 3)
            .flatten(0, 1)[:length]
            .float()
            .repeat_interleave(8, dim=1)
        )
        v = (
            data.value[page_ids]
            .permute(0, 2, 1, 3)
            .flatten(0, 1)[:length]
            .float()
            .repeat_interleave(8, dim=1)
        )
        q = data.q.view(case.batch.batch_size, case.qlen, 16, 192)[row].float()
        scores = torch.einsum("qhd,khd->hqk", q, k) * 192**-0.5
        pos = torch.arange(length - case.qlen, length, device="cuda")
        tokens = torch.arange(length, device="cuda")
        visible = (tokens[None, :] <= pos[:, None]) & (
            tokens[None, :] > pos[:, None] - 128
        )
        scores.masked_fill_(~visible[None], float("-inf"))
        probs = torch.softmax(
            torch.cat(
                (scores, data.sink[:, None, None].expand(-1, case.qlen, 1)), dim=-1
            ),
            dim=-1,
        )[..., :length]
        expected[row] = torch.einsum("hqk,khd->qhd", probs, v).to(torch.bfloat16)
    return expected.view_as(data.out)


def _check(case):
    for data in case.inputs:
        torch.testing.assert_close(
            data.out, _reference(case, data), atol=0.005, rtol=0.025
        )
        assert torch.all(data.guarded[[0, -1]] == 17)


@requires_rocm
@pytest.mark.parametrize("qlen", [1, 4, 8])
def test_grouped_swa_reuses_plan_across_layers_and_dynamic_graph(qlen, monkeypatch):
    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    original = pa.plan_pa_decode
    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs.get("plan"))
        return original(*args, **kwargs)

    monkeypatch.setattr(pa, "plan_pa_decode", record)
    case = _case(8, qlen)
    _set_metadata(case, [0, 1, 3, 4, 127, 128, 191, 256], case.table)
    _forward(case)
    _check(case)
    assert len(calls) == 2  # Allocate once, then refresh only at the first layer.
    cache = case.backend._grouped_pa_swa_workspace
    workspace = next(iter(cache.values()))
    plan = workspace[2]
    assert plan.max_partitions == 1 and plan.capacity == 8
    pointers = [
        x.data_ptr()
        for x in (
            workspace[0],
            workspace[1],
            plan.work_info,
            plan.reduce_info,
            *workspace[3:],
        )
    ]
    _forward(case)
    assert len(calls) == 3
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        _forward(case)
    torch.cuda.current_stream().wait_stream(stream)
    for lengths, table in (
        ([4, 3, 1, 0, 128, 127, 256, 191], case.table.flip(0)),
        ([128, 129, 63, 64, 65, 192, 0, 255], case.table.roll(1, dims=1)),
    ):
        _set_metadata(case, lengths, table)
        graph.replay()
        _check(case)
    assert len(calls) == 4  # Replay executes the recorded GPU plan refresh.
    assert len(cache) == 1 and next(iter(cache.values())) is workspace
    assert pointers == [
        x.data_ptr()
        for x in (
            workspace[0],
            workspace[1],
            plan.work_info,
            plan.reduce_info,
            *workspace[3:],
        )
    ]


@requires_rocm
def test_grouped_swa_cold_capture_and_first_layer_guards(monkeypatch):
    case = _case(2, 4)
    _set_metadata(case, [128, 191], case.table)
    case.backend.forward_metadata = case.metadata
    data = case.inputs[1]
    with pytest.raises(RuntimeError, match="first SWA layer"):
        forward_grouped_pa_swa(
            case.backend,
            data.q,
            data.layer,
            case.batch,
            data.k,
            data.v,
            data.out,
            data.sink,
        )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="warmed before graph capture"):
        _forward(case)


@requires_rocm
def test_grouped_swa_alternates_sglang_graph_batches():
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
        FullCudaGraphBackend,
    )

    cases = [_case(8, 4)]
    backend = cases[0].backend
    cases.extend(_case(bs, 4, backend) for bs in (2, 1))
    runner = SimpleNamespace(
        device_module=torch.cuda,
        model_runner=SimpleNamespace(tp_group=SimpleNamespace(barrier=lambda: None)),
    )
    graphs = FullCudaGraphBackend(runner)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with graphs.capture_session(stream):
            for case in cases:
                _set_metadata(case, [128] * case.batch.batch_size, case.table)
                graphs.capture_one(
                    case.batch.batch_size, lambda case=case: _forward(case)
                )
        torch.cuda.current_stream().wait_stream(stream)
        assert len(backend._grouped_pa_swa_workspace) == 3
        for case in (cases[2], cases[0], cases[1], cases[0]):
            bs = case.batch.batch_size
            _set_metadata(
                case, [0, 1, 63, 64, 65, 127, 128, 256][:bs], case.table.flip(0)
            )
            graphs.replay(bs, case.batch)
            _check(case)
        assert len(backend._grouped_pa_swa_workspace) == 3
    finally:
        torch.cuda.synchronize()
        graphs.cleanup()
